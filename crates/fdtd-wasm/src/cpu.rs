//! The CPU fallback: the f64 reference kernel, stepped in a Web Worker
//! when the browser has no WebGPU, with the same live probe, intensity
//! mean and pixels (`render::CpuPainter`) as the GPU path.

use fdtd_core::kernel::{CpuKernel, Kernel};
use fdtd_core::scene::Scene;
use fdtd_core::{FdtdError, Result};
use fdtd_gpu::buffers::Layout;
use fdtd_gpu::render::{CpuPainter, Frame};

pub struct CpuBackend {
    sim: CpuKernel,
    painter: CpuPainter,
    /// Material id per node with the wall entry on the ring, for drawing.
    ids: Vec<u32>,
    /// Running mean of the node field squared (Ez^2, or Hz^2 in TEz) and its
    /// 1 / tau, while the intensity view is on.
    mean: Vec<f64>,
    alpha: Option<f64>,
    probe: Option<usize>,
    samples: Vec<f32>,
}

/// Most probe samples kept between collections, as on the GPU.
const MAX_PROBE_SAMPLES: usize = 1 << 16;

impl CpuBackend {
    pub fn new(scene: &Scene) -> Result<CpuBackend> {
        Ok(CpuBackend {
            sim: CpuKernel::from_scene(scene)?,
            painter: CpuPainter::new(scene),
            ids: Layout::new(scene).ids,
            mean: vec![0.0; scene.grid.nx * scene.grid.ny],
            alpha: None,
            probe: None,
            samples: Vec::new(),
        })
    }

    pub fn time_step(&self) -> u64 {
        self.sim.time_step()
    }

    /// `n` steps, one at a time so the probe and the mean see every one.
    pub fn step(&mut self, n: u64) {
        for _ in 0..n {
            self.sim.step(1);
            let ez = self.sim.node_field();
            if let Some(a) = self.alpha {
                for (m, e) in self.mean.iter_mut().zip(ez) {
                    *m += a * (e * e - *m);
                }
            }
            if let Some(k) = self.probe {
                self.samples.push(ez[k] as f32);
            }
        }
        let excess = self.samples.len().saturating_sub(MAX_PROBE_SAMPLES);
        self.samples.drain(..excess);
    }

    pub fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        self.sim.set_materials(scene, bounds)?;
        self.ids = Layout::new(scene).ids;
        Ok(())
    }

    pub fn set_sources(&mut self, scene: &Scene) -> Result<()> {
        self.sim.set_sources(scene)
    }

    /// As `GpuSim::set_intensity`: a fresh mean with time constant `tau`
    /// steps (at least 1), or off.
    pub fn set_intensity(&mut self, tau: Option<f64>) -> Result<()> {
        match tau {
            None => self.alpha = None,
            Some(t) => {
                if !(t.is_finite() && t >= 1.0) {
                    return Err(FdtdError::InvalidParameter {
                        name: "intensity tau",
                        value: t,
                    });
                }
                self.mean.iter_mut().for_each(|m| *m = 0.0);
                self.alpha = Some(1.0 / t);
            }
        }
        Ok(())
    }

    /// As `GpuSim::set_probe`.
    pub fn set_probe(&mut self, node: Option<(usize, usize)>) -> Result<()> {
        let (nx, ny) = (self.sim.nx(), self.sim.ny());
        self.samples.clear();
        self.probe = match node {
            None => None,
            Some((i, j)) if i < nx && j < ny => Some(j * nx + i),
            Some((i, j)) => {
                return Err(FdtdError::OutOfDomain {
                    what: "probe",
                    i,
                    j,
                });
            }
        };
        Ok(())
    }

    pub fn take_probe(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }

    /// RGBA8 pixels of `frame`, row-major from the top left.
    pub fn pixels(&self, frame: &Frame) -> Vec<u8> {
        self.painter
            .draw(&self.ids, self.sim.node_field(), &self.mean, frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Session;
    use fdtd_core::scene::ViewField;
    use fdtd_gpu::render::Viewport;

    const SCENE: &str = r#"{"version": 1,
        "grid": {"nx": 60, "ny": 50, "dx_phys_m": 0.001, "courant": 0.5},
        "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0}],
        "sources": [{"i": 20, "j": 25, "amplitude": 1,
            "waveform": {"type": "ramped_cw", "f0": 0.05, "ramp_periods": 3, "phase": 0}}],
        "pml": {"thickness": 5, "m": 3, "sigma_scale": 1.25, "kappa_max": 2, "alpha_max": 0.15}}"#;

    #[test]
    fn probe_and_mean_follow_the_reference() {
        let s = Session::load(SCENE).unwrap();
        let mut cpu = CpuBackend::new(&s.scene).unwrap();
        let mut reference = fdtd_core::tm::SimTm::from_scene(&s.scene).unwrap();
        cpu.set_probe(Some((40, 25))).unwrap();
        cpu.set_intensity(Some(20.0)).unwrap();
        let mut mean = vec![0.0; 60 * 50];
        let mut trace = Vec::new();
        for _ in 0..120 {
            reference.step_n(1);
            let ez = reference.ez();
            trace.push(ez[25 * 60 + 40] as f32);
            for (m, e) in mean.iter_mut().zip(ez) {
                *m += (1.0 / 20.0) * (e * e - *m);
            }
        }
        cpu.step(50);
        cpu.step(70);
        assert_eq!(cpu.take_probe(), trace);
        assert!(cpu.take_probe().is_empty());
        assert_eq!(cpu.mean, mean);
        assert!(cpu.set_probe(Some((60, 0))).is_err());
        assert!(cpu.set_intensity(Some(0.0)).is_err());
    }

    #[test]
    fn pixels_cover_the_canvas_and_follow_edits() {
        let mut s = Session::load(SCENE).unwrap();
        let mut cpu = CpuBackend::new(&s.scene).unwrap();
        let frame = Frame {
            viewport: Viewport::fit(60, 50, 120.0, 100.0),
            gain: 1.0,
            field: ViewField::Ez,
        };
        let before = cpu.pixels(&frame);
        assert_eq!(before.len(), 120 * 100 * 4);
        let out = s.paint((30.0, 10.0), (30.0, 40.0), 1.0, 3).unwrap();
        cpu.set_materials(&s.scene, out.bounds.unwrap()).unwrap();
        let after = cpu.pixels(&frame);
        assert_ne!(before, after, "a painted conductor changes the picture");
    }
}
