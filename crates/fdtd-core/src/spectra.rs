//! Reflection and transmission spectra from a TF/SF plane wave and two flux
//! lines, with a normalization run. Shared by the tests, `fdtd validate` and
//! `fdtd report`, so there is one definition of the measurement.
//!
//! The scene has an x-travelling plane wave. R is read on a flux line in the
//! scattered-field region behind the entry face: only reflected field exists
//! there, so nothing has to be subtracted. T is read on a flux line in the
//! total-field region past the structure. Both are divided by the incident
//! flux, which is the T line's flux in the same scene with every shape
//! removed. Both runs go on until the fields have died down (see
//! `run_until_quiet`), or the spectra would ring.
//!
//! Normalized units throughout; frequencies in cycles per unit time.

use crate::error::{FdtdError, Result};
use crate::kernel::{CpuKernel, Kernel, Spectrum};
use crate::monitor::Normal;
use crate::scene::{Monitor, Scene};
use crate::tfsf::Direction;

/// The decay level a spectrum run waits for, relative to the peak |Ez| seen.
pub const QUIET: f64 = 1e-6;

/// Steps `sim` until max |node field| (Ez, or Hz in TEz) over the grid has
/// stayed below `QUIET` times the largest value seen for `window`
/// consecutive steps. Returns the step count, or `NotDecayed` if
/// `max_steps` pass first.
pub fn run_until_quiet(sim: &mut CpuKernel, window: u64, max_steps: u64) -> Result<u64> {
    let (mut peak, mut quiet) = (0.0_f64, 0u64);
    let mut level = 1.0;
    while sim.time_step() < max_steps {
        sim.step(1);
        let m = sim.node_field().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        peak = peak.max(m);
        level = if peak > 0.0 { m / peak } else { 1.0 };
        quiet = if peak > 0.0 && level < QUIET {
            quiet + 1
        } else {
            0
        };
        if quiet >= window {
            return Ok(sim.time_step());
        }
    }
    Err(FdtdError::NotDecayed {
        steps: sim.time_step(),
        level,
    })
}

/// R(f) and T(f) of one structure, and how long each run took.
#[derive(Debug, Clone, PartialEq)]
pub struct RtSpectrum {
    pub freqs: Vec<f64>,
    pub r: Vec<f64>,
    pub t: Vec<f64>,
    /// Steps of the structure run and of the normalization run.
    pub steps: (u64, u64),
}

/// Measures R and T for `scene`, which must have a +x or -x plane wave
/// spanning every row with periodic y. `r_at` and `t_at` are the columns of
/// the two flux lines (full height). `window` and `max_steps` go to
/// `run_until_quiet`.
///
/// `R = -flux_R / flux_inc` and `T = flux_T / flux_inc` hold for either
/// direction: reversing the incidence flips the sign of every flux.
pub fn rt_spectrum(
    scene: &Scene,
    r_at: usize,
    t_at: usize,
    freqs: &[f64],
    window: u64,
    max_steps: u64,
) -> Result<RtSpectrum> {
    let along_x = scene
        .plane_wave
        .is_some_and(|pw| matches!(pw.direction, Direction::PlusX | Direction::MinusX));
    if !along_x {
        return Err(FdtdError::InvalidParameter {
            name: "plane wave along x",
            value: 0.0,
        });
    }
    let line = |at| Monitor::Flux {
        normal: Normal::X,
        at,
        from: 0,
        to: scene.grid.ny - 1,
        freqs: freqs.to_vec(),
    };
    let run = |s: &Scene| -> Result<(Vec<f64>, Vec<f64>, u64)> {
        let mut s = s.clone();
        s.monitors = vec![line(r_at), line(t_at)];
        let mut sim = CpuKernel::from_scene(&s)?;
        let steps = run_until_quiet(&mut sim, window, max_steps)?;
        let mut fluxes = Kernel::monitor_spectra(&mut sim)?
            .into_iter()
            .filter_map(|m| match m {
                Spectrum::Flux(f) => Some(f),
                Spectrum::Field(_) | Spectrum::FarField(_) => None,
            });
        match (fluxes.next(), fluxes.next()) {
            (Some(r), Some(t)) => Ok((r, t, steps)),
            _ => Err(FdtdError::InvalidParameter {
                name: "flux lines",
                value: 0.0,
            }),
        }
    };
    let (fr, ft, steps) = run(scene)?;
    let mut empty = scene.clone();
    empty.geometry.clear();
    let (_, inc, norm_steps) = run(&empty)?;
    Ok(RtSpectrum {
        freqs: freqs.to_vec(),
        r: fr.iter().zip(&inc).map(|(a, b)| -a / b).collect(),
        t: ft.iter().zip(&inc).map(|(a, b)| a / b).collect(),
        steps: (steps, norm_steps),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpml::PmlSpec;
    use crate::material::Material;
    use crate::scene::{GridSpec, SCENE_VERSION};
    use crate::source::Waveform;
    use crate::tfsf::PlaneWave;

    fn periodic_line(nx: usize, dir: Direction) -> Scene {
        Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx,
                ny: 3,
                dx_phys_m: 1e-3,
                courant: 0.5,
            },
            materials: vec![Material::VACUUM],
            geometry: vec![],
            sources: vec![],
            monitors: vec![],
            pml: Some(PmlSpec::recommended(16)),
            periodic_y: true,
            plane_wave: Some(PlaneWave {
                x0: 30,
                y0: 0,
                x1: nx - 31,
                y1: 2,
                direction: dir,
                waveform: Waveform::Ricker {
                    t0: 60.0,
                    f_peak: 0.03,
                },
                amplitude: 1.0,
            }),
            paint: None,
            view: None,
            mode: crate::scene::Mode::Tm,
        }
    }

    /// With nothing in the box the structure run is the normalization run:
    /// T is exactly 1 and R is the TF/SF leakage, round-off squared.
    #[test]
    fn empty_box_is_transparent() {
        let modes = [crate::scene::Mode::Tm, crate::scene::Mode::Te];
        for (dir, mode) in [Direction::PlusX, Direction::MinusX]
            .into_iter()
            .flat_map(|d| modes.map(|m| (d, m)))
        {
            let mut scene = periodic_line(160, dir);
            scene.mode = mode;
            let (r_at, t_at) = match dir {
                Direction::PlusX => (24, 120),
                _ => (135, 40),
            };
            let rt = rt_spectrum(&scene, r_at, t_at, &[0.01, 0.02, 0.03], 300, 20_000).unwrap();
            assert!(rt.t.iter().all(|&t| t == 1.0), "{dir:?}: {:?}", rt.t);
            assert!(rt.r.iter().all(|r| r.abs() < 1e-20), "{dir:?}: {:?}", rt.r);
            assert_eq!(rt.steps.0, rt.steps.1);
            assert!(rt.steps.0 > 300 && rt.steps.0 < 20_000);
        }
    }

    /// A lossless closed PEC box with a scatterer never goes quiet (the
    /// reflected wave bounces between the walls forever), and the run says
    /// so instead of returning a ringing spectrum. The scatterer matters: in
    /// an empty TF/SF box the scattered field is zero and nothing reaches
    /// the walls, so even a closed box goes quiet.
    #[test]
    fn closed_box_reports_no_decay() {
        let mut scene = periodic_line(160, Direction::PlusX);
        scene.pml = None;
        scene.materials.push(Material::Dielectric {
            eps_inf: 4.0,
            sigma: 0.0,
        });
        scene.geometry.push(crate::scene::Shape::Rect {
            x0: 70.0,
            y0: -1.0,
            x1: 90.0,
            y1: 3.0,
            material: 1,
        });
        let mut sim = CpuKernel::from_scene(&scene).unwrap();
        assert!(matches!(
            run_until_quiet(&mut sim, 100, 3000),
            Err(FdtdError::NotDecayed { steps: 3000, .. })
        ));
    }

    #[test]
    fn rt_needs_an_x_plane_wave() {
        let mut scene = periodic_line(160, Direction::PlusX);
        scene.plane_wave = None;
        assert!(rt_spectrum(&scene, 24, 120, &[0.02], 10, 100).is_err());
    }
}
