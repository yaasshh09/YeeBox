//! The in-browser self-test: the browser's GPU kernel against the native
//! CPU kernel's committed fields (`fdtd_core::selftest`), with the Tier 4
//! measure and gate. Browser only.

use crate::{ErrorInfo, JsResult, js_err};
use fdtd_core::grid::Component;
use fdtd_core::kernel::{Comparator, GPU_GATE, Snapshot};
use fdtd_core::scene::Scene;
use fdtd_core::selftest::{CHECKPOINTS, read_golden};
use fdtd_gpu::{Gpu, GpuSim};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct Row {
    step: u64,
    fields: [f64; 3],
    worst: f64,
}

#[derive(Serialize)]
struct Report {
    scene: String,
    adapter: String,
    gate: f64,
    diffs: Vec<Row>,
    worst: f64,
    pass: bool,
}

/// Steps the browser GPU kernel on `scene_json` (exactly as given, no
/// palette) and compares Ez, Hx and Hy at each checkpoint with `golden`.
/// Resolves with a JSON report: per checkpoint the three field
/// differences as fractions of the reference peak, the worst, the gate
/// and pass or fail. Rejects with `no_webgpu` if there is no adapter.
#[wasm_bindgen(js_name = runSelfTest)]
pub async fn run_self_test(name: String, scene_json: String, golden: Vec<u8>) -> JsResult<String> {
    let scene = Scene::from_json(&scene_json).map_err(js_err)?;
    let reference = read_golden(&golden, scene.grid.nx * scene.grid.ny).map_err(js_err)?;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let gpu = Gpu::request(&instance, None)
        .await
        .map_err(|e| ErrorInfo::no_webgpu(e.to_string()))?;
    // ponytail: one small leaked device per self-test run (three per page).
    let gpu: &'static Gpu = Box::leak(Box::new(gpu));
    let mut sim = GpuSim::new(gpu, &scene).map_err(js_err)?;
    let mut cmp = Comparator::new();
    let mut diffs = Vec::new();
    for (k, &n) in CHECKPOINTS.iter().enumerate() {
        sim.step_n(n - sim.time_step());
        let mut snap = Snapshot::default();
        for (f, c) in [Component::Ez, Component::Hx, Component::Hy]
            .into_iter()
            .enumerate()
        {
            snap.fields[f] = sim.read_async(c).await.map_err(js_err)?;
        }
        // The reference peak over the run so far, as Tier 4 measures it.
        for (_, p) in reference.peaks.iter().filter(|(s, _)| *s <= n) {
            cmp.raise_field_peaks(*p);
        }
        let d = cmp
            .checkpoint(n, &reference.snapshots[k], &snap)
            .map_err(js_err)?;
        diffs.push(Row {
            step: n,
            fields: d.fields,
            worst: d.worst(),
        });
    }
    let worst = diffs.iter().fold(0.0f64, |m, r| m.max(r.worst));
    let report = Report {
        scene: name,
        adapter: gpu.describe(),
        gate: GPU_GATE,
        diffs,
        worst,
        pass: worst.is_finite() && worst < GPU_GATE,
    };
    serde_json::to_string(&report).map_err(|e| js_err(fdtd_core::FdtdError::Json(e.to_string())))
}
