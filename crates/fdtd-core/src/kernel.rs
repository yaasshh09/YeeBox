//! The interface every kernel implements, so one harness can drive the f64
//! CPU reference and the f32 GPU kernel from the same `Scene` and diff them.

use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::scene::Scene;

pub trait Kernel: Sized {
    /// Validates the scene and builds the kernel at time step 0, fields zero.
    fn from_scene(scene: &Scene) -> Result<Self>;

    /// Advances `n` full steps.
    fn step(&mut self, n: u64);

    /// Full steps taken so far. E is at time n S, H at (n - 1/2) S.
    fn time_step(&self) -> u64;

    /// A copy of one field, normalized, `nx * ny` padded, index `j * nx + i`.
    /// Ez on nodes, Hx on y-edges, Hy on x-edges (see `tm.rs`).
    fn read_field(&mut self, which: Component) -> Result<Vec<f64>>;

    /// Ez history of every point monitor, in scene order. Each has
    /// `time_step() + 1` entries.
    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>>;

    /// The accumulated spectrum of every DFT and flux monitor, in scene
    /// order (point monitors are skipped), one entry per scene frequency.
    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>>;
}

/// What a spectral monitor has accumulated so far, per frequency of its list.
#[derive(Debug, Clone, PartialEq)]
pub enum Spectrum {
    /// The Ez transform at a node, `(re, im)`, normalized.
    Field(Vec<(f64, f64)>),
    /// Poynting flux through a line, positive along its normal, normalized.
    Flux(Vec<f64>),
}

/// How far kernel `b` is from the reference kernel `a` after `step` steps.
/// Every entry is `max |b - a| / max |a|` over the whole array (0 when both
/// are all zero, infinite when only `a` is), so 1e-4 means "within 1e-4 of
/// the reference peak".
#[derive(Debug, Clone, PartialEq)]
pub struct Diff {
    pub step: u64,
    /// Ez, Hx, Hy, padded arrays included.
    pub fields: [f64; 3],
    /// All point probe histories together.
    pub probes: f64,
    /// All spectral monitors together, real and imaginary parts.
    pub spectra: f64,
}

impl Diff {
    /// The largest entry.
    pub fn worst(&self) -> f64 {
        self.fields
            .iter()
            .chain([&self.probes, &self.spectra])
            .fold(0.0, |m, &d| m.max(d))
    }
}

/// `max |b - a| / max |a|`, with the conventions of `Diff`. Mismatched
/// lengths are an error.
pub fn relative_diff(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(FdtdError::Case(format!(
            "compared arrays of length {} and {}",
            a.len(),
            b.len()
        )));
    }
    let peak = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((y - x).abs()));
    Ok(if worst == 0.0 { 0.0 } else { worst / peak })
}

fn flatten(spectra: &[Spectrum]) -> Vec<f64> {
    let mut out = Vec::new();
    for s in spectra {
        match s {
            Spectrum::Field(v) => out.extend(v.iter().flat_map(|&(re, im)| [re, im])),
            Spectrum::Flux(v) => out.extend(v),
        }
    }
    out
}

/// Steps both kernels, built from the same scene, to each step count in
/// `checkpoints` (non-decreasing, none below where they already are) and
/// diffs every field, the probe histories and the spectra there. `a` is
/// the reference.
pub fn compare<A: Kernel, B: Kernel>(
    a: &mut A,
    b: &mut B,
    checkpoints: &[u64],
) -> Result<Vec<Diff>> {
    let mut out = Vec::with_capacity(checkpoints.len());
    for &step in checkpoints {
        for k in [a.time_step(), b.time_step()] {
            if step < k {
                return Err(FdtdError::Case(format!(
                    "checkpoint {step} is behind step {k}"
                )));
            }
        }
        a.step(step - a.time_step());
        b.step(step - b.time_step());
        let mut fields = [0.0; 3];
        for (d, c) in fields
            .iter_mut()
            .zip([Component::Ez, Component::Hx, Component::Hy])
        {
            *d = relative_diff(&a.read_field(c)?, &b.read_field(c)?)?;
        }
        let pa: Vec<f64> = a.probe_history()?.concat();
        let pb: Vec<f64> = b.probe_history()?.concat();
        let sa = flatten(&a.monitor_spectra()?);
        let sb = flatten(&b.monitor_spectra()?);
        out.push(Diff {
            step,
            fields,
            probes: relative_diff(&pa, &pb)?,
            spectra: relative_diff(&sa, &sb)?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tm::SimTm;
    use std::path::Path;

    fn scene(name: &str) -> Scene {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../validation/scenes")
            .join(name);
        Scene::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// The reference against itself is exactly zero everywhere, on a scene
    /// with TF/SF, flux lines and probes.
    #[test]
    fn same_kernel_is_exactly_zero() {
        let s = scene("tfsf_rod_minus_x.json");
        let (mut a, mut b) = (
            SimTm::from_scene(&s).unwrap(),
            SimTm::from_scene(&s).unwrap(),
        );
        for d in compare(&mut a, &mut b, &[0, 50, 300]).unwrap() {
            assert_eq!(d.worst(), 0.0, "{d:?}");
        }
    }

    /// A kernel whose Ez reads 1e-3 high, and whose spectra are off by the
    /// same factor, is caught in exactly those entries.
    struct Nudged(SimTm);

    impl Kernel for Nudged {
        fn from_scene(scene: &Scene) -> Result<Self> {
            SimTm::from_scene(scene).map(Nudged)
        }
        fn step(&mut self, n: u64) {
            self.0.step_n(n)
        }
        fn time_step(&self) -> u64 {
            self.0.time_step()
        }
        fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
            let v = self.0.read_field(which)?;
            let g = if which == Component::Ez { 1.001 } else { 1.0 };
            Ok(v.into_iter().map(|x| x * g).collect())
        }
        fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
            self.0.probe_history()
        }
        fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
            Ok(self
                .0
                .monitor_spectra()?
                .into_iter()
                .map(|s| match s {
                    Spectrum::Flux(v) => Spectrum::Flux(v.into_iter().map(|x| x * 1.001).collect()),
                    other => other,
                })
                .collect())
        }
    }

    #[test]
    fn perturbed_kernel_is_caught() {
        let s = scene("tfsf_rod_minus_x.json");
        let mut a = SimTm::from_scene(&s).unwrap();
        let mut b = Nudged::from_scene(&s).unwrap();
        let d = compare(&mut a, &mut b, &[300]).unwrap().remove(0);
        assert!((d.fields[0] - 1e-3).abs() < 1e-12, "{d:?}");
        assert!((d.spectra - 1e-3).abs() < 1e-12, "{d:?}");
        assert_eq!((d.fields[1], d.fields[2], d.probes), (0.0, 0.0, 0.0));
    }

    #[test]
    fn relative_diff_conventions() {
        assert_eq!(relative_diff(&[0.0, 0.0], &[0.0, 0.0]).unwrap(), 0.0);
        assert_eq!(relative_diff(&[0.0], &[1.0]).unwrap(), f64::INFINITY);
        assert_eq!(relative_diff(&[2.0, -4.0], &[2.0, -3.0]).unwrap(), 0.25);
        assert!(relative_diff(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn checkpoints_cannot_go_back() {
        let s = scene("hard_sources.json");
        let (mut a, mut b) = (
            SimTm::from_scene(&s).unwrap(),
            SimTm::from_scene(&s).unwrap(),
        );
        assert!(compare(&mut a, &mut b, &[20, 10]).is_err());
    }
}
