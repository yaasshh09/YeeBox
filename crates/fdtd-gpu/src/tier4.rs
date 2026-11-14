//! Tier 4 runs, shared by the tests and `fdtd report`: one scene, both
//! kernels, every field diffed against the CPU reference (see
//! `fdtd_core::kernel::compare` for the measure).

use crate::{Gpu, GpuSim};
use fdtd_core::kernel::{Diff, compare};
use fdtd_core::scene::Scene;
use fdtd_core::tm::SimTm;
use fdtd_core::{FdtdError, Result};
use std::path::{Path, PathBuf};

/// Every field, probe record and spectrum within this fraction of the
/// reference peak, at every checkpoint (Phase 4 acceptance).
pub const GATE: f64 = 1e-4;

/// The steps of the standard run.
pub const STEPS: u64 = 2000;

/// Step 1, then every `every` steps up to `last`.
pub fn checkpoints(last: u64, every: u64) -> Vec<u64> {
    std::iter::once(1)
        .chain((1..=last / every).map(|k| k * every))
        .collect()
}

/// `validation/scenes` of this repository.
pub fn scenes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/scenes")
}

fn io(e: std::io::Error) -> FdtdError {
    FdtdError::Case(e.to_string())
}

/// Every scene file there, by file name, sorted.
pub fn scenes() -> Result<Vec<(String, Scene)>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(scenes_dir()).map_err(io)? {
        let path = entry.map_err(io)?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let text = std::fs::read_to_string(&path).map_err(io)?;
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            out.push((name, Scene::from_json(&text)?));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Both kernels from `scene`, compared at `checkpoints`.
pub fn run(gpu: &'static Gpu, scene: &Scene, checkpoints: &[u64]) -> Result<Vec<Diff>> {
    let mut cpu = SimTm::from_scene(scene)?;
    let mut gpu_sim = GpuSim::new(gpu, scene)?;
    compare(&mut cpu, &mut gpu_sim, checkpoints)
}

/// The long run: no faster than linear growth of the difference.
pub const LONG_STEPS: u64 = 50_000;
pub const LONG_EVERY: u64 = 5_000;

/// Upper bound on the fitted growth exponent, chosen from theory before the
/// first run: f32 phase drift grows linearly (exponent 1), round-off that
/// does not compound grows like a random walk (1/2), and an instability
/// grows faster than any power. 1.2 leaves room for fit noise only.
pub const MAX_GROWTH: f64 = 1.2;

/// The scene of the long run: a closed lossless cavity, so the field never
/// leaves and the difference has the whole run to grow.
pub const LONG_SCENE: &str = "cavity_tm.json";

/// Least-squares slope of ln(worst difference) against ln(step) over the
/// checkpoints from `from` on. Zero differences are skipped (they carry no
/// growth information).
pub fn growth_exponent(diffs: &[Diff], from: u64) -> f64 {
    let pts: Vec<(f64, f64)> = diffs
        .iter()
        .filter(|d| d.step >= from && d.worst() > 0.0)
        .map(|d| ((d.step as f64).ln(), d.worst().ln()))
        .collect();
    let n = pts.len() as f64;
    let (mx, my) = pts
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x / n, b + y / n));
    let (sxy, sxx) = pts.iter().fold((0.0, 0.0), |(a, b), (x, y)| {
        (a + (x - mx) * (y - my), b + (x - mx) * (x - mx))
    });
    sxy / sxx
}

/// The long run on `LONG_SCENE`, checkpoints every `LONG_EVERY` steps.
pub fn long_run(gpu: &'static Gpu) -> Result<Vec<Diff>> {
    let path = scenes_dir().join(LONG_SCENE);
    let text = std::fs::read_to_string(path).map_err(io)?;
    run(
        gpu,
        &Scene::from_json(&text)?,
        &checkpoints(LONG_STEPS, LONG_EVERY),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diffs(f: impl Fn(f64) -> f64) -> Vec<Diff> {
        checkpoints(LONG_STEPS, LONG_EVERY)
            .into_iter()
            .map(|step| Diff {
                step,
                fields: [f(step as f64), 0.0, 0.0],
                probes: 0.0,
                spectra: 0.0,
            })
            .collect()
    }

    /// The fit returns the exponent of an exact power law, whatever the
    /// prefactor, and skips step 1.
    #[test]
    fn exponent_of_a_power_law() {
        for (p, c) in [(0.5, 3e-7), (1.0, 1e-9), (2.0, 4e-12)] {
            let e = growth_exponent(&diffs(|n| c * n.powf(p)), LONG_EVERY);
            assert!((e - p).abs() < 1e-12, "{p}: {e}");
        }
        let mut d = diffs(|n| 1e-9 * n);
        d[0].fields[0] = 1.0;
        assert!((growth_exponent(&d, LONG_EVERY) - 1.0).abs() < 1e-12);
    }

    /// An exponential instability fits far above the bound.
    #[test]
    fn exponential_growth_fails_the_bound() {
        let e = growth_exponent(&diffs(|n| 1e-9 * (n / 5000.0).exp()), LONG_EVERY);
        assert!(e > 3.0 * MAX_GROWTH, "{e}");
    }
}
