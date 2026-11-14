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
