//! Tier 4: the GPU kernel against the CPU reference.
//!
//! Its own harness (libtest-mimic) so that a machine without a GPU adapter
//! reports every trial as ignored, with a banner, instead of as passed.
//! With `FDTD_REQUIRE_GPU=1` (set in CI) a missing adapter is a failure.

use fdtd_core::grid::Component;
use fdtd_core::kernel::compare;
use fdtd_core::scene::Scene;
use fdtd_core::tm::SimTm;
use fdtd_gpu::{Gpu, GpuSim};
use libtest_mimic::{Arguments, Failed, Trial};
use std::path::Path;
use wgpu::util::DeviceExt;

type Check = fn(&'static Gpu) -> Result<(), Failed>;

const FIELDS: [Component; 3] = [Component::Ez, Component::Hx, Component::Hy];

fn scene(name: &str) -> Scene {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../validation/scenes")
        .join(name);
    Scene::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The scene with only its grid, materials, geometry, layer and periodicity.
fn fields_only(name: &str) -> Scene {
    let mut s = scene(name);
    s.sources.clear();
    s.monitors.clear();
    s.plane_wave = None;
    s
}

/// Field-only scenes covering closed boxes, CPML, periodic y and PEC,
/// lossy and dielectric shapes.
const FIELD_SCENES: [&str; 5] = [
    "cavity_tm.json",
    "line_source_cpml.json",
    "waveguide_tm.json",
    "materials_clutter.json",
    "hard_sources.json",
];

/// A smooth bump of Ez a third of the way across, the same on both kernels.
fn bump(scene: &Scene) -> Vec<f64> {
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let (cx, cy) = (nx as f64 / 3.0, ny as f64 / 2.0);
    (0..nx * ny)
        .map(|k| {
            let (x, y) = ((k % nx) as f64 - cx, (k / nx) as f64 - cy);
            (-(x * x + y * y) / 18.0).exp()
        })
        .collect()
}

/// A buffer written on the CPU reads back bit for bit.
fn readback_round_trip(gpu: &'static Gpu) -> Result<(), Failed> {
    let values: Vec<f32> = (0..1000).map(|k| (k as f32 * 0.37).sin()).collect();
    let buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let back = gpu.read_f32(&buf)?;
    let want: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    if back != want {
        return Err("readback differs from the upload".into());
    }
    Ok(())
}

fn fields_zero_at_step_0(gpu: &'static Gpu) -> Result<(), Failed> {
    for name in FIELD_SCENES {
        let mut sim = GpuSim::new(gpu, &fields_only(name))?;
        for c in FIELDS {
            if sim.read(c)?.iter().any(|&v| v != 0.0) {
                return Err(format!("{name}: {c:?} not zero at step 0").into());
            }
        }
    }
    Ok(())
}

/// Checkpoints every 250 steps up to 2000, so the bound also holds while
/// the field is large, not only after it has left through the layer.
fn checkpoints() -> Vec<u64> {
    std::iter::once(1).chain((1..=8).map(|k| 250 * k)).collect()
}

/// The H and E updates alone, from the same initial Ez, on every kind of
/// grid: within 1e-4 of the reference peak field at every checkpoint.
fn initial_value_matches_cpu(gpu: &'static Gpu) -> Result<(), Failed> {
    for name in FIELD_SCENES {
        let s = fields_only(name);
        let mut cpu = SimTm::from_scene(&s)?;
        let mut gpu_sim = GpuSim::new(gpu, &s)?;
        cpu.set_ez(&bump(&s))?;
        gpu_sim.set_ez(&bump(&s))?;
        for d in compare(&mut cpu, &mut gpu_sim, &checkpoints())? {
            eprintln!(
                "  {name} step {}: worst {:.2e} {:?}",
                d.step,
                d.worst(),
                d.fields
            );
            if d.worst().is_nan() || d.worst() >= 1e-4 {
                return Err(format!("{name}: {d:?}").into());
            }
        }
    }
    Ok(())
}

/// Hx row ny - 1 (unless y is periodic), Hy column nx - 1 and the wall ring
/// of Ez stay exactly zero.
fn padding_stays_zero(gpu: &'static Gpu) -> Result<(), Failed> {
    for name in FIELD_SCENES {
        let s = fields_only(name);
        let (nx, ny) = (s.grid.nx, s.grid.ny);
        let mut sim = GpuSim::new(gpu, &s)?;
        sim.set_ez(&bump(&s))?;
        sim.step_n(300);
        let (ez, hx, hy) = (
            sim.read(Component::Ez)?,
            sim.read(Component::Hx)?,
            sim.read(Component::Hy)?,
        );
        let live = s.live_rows();
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                let wall = i == 0 || i == nx - 1 || !live.contains(&j);
                if wall && ez[k] != 0.0 {
                    return Err(format!("{name}: wall Ez ({i}, {j}) = {}", ez[k]).into());
                }
                if j == ny - 1 && !s.periodic_y && hx[k] != 0.0 {
                    return Err(format!("{name}: padding Hx ({i}, {j}) = {}", hx[k]).into());
                }
                if i == nx - 1 && hy[k] != 0.0 {
                    return Err(format!("{name}: padding Hy ({i}, {j}) = {}", hy[k]).into());
                }
            }
        }
        if ez.iter().all(|&v| v == 0.0) {
            return Err(format!("{name}: nothing ran").into());
        }
    }
    Ok(())
}

/// Workgroup edges 4, 8 and 16 give the same bits. An out-of-bounds write
/// at the dispatch overhang would break this.
fn workgroup_size_independent(gpu: &'static Gpu) -> Result<(), Failed> {
    for name in FIELD_SCENES {
        let s = fields_only(name);
        let mut runs = Vec::new();
        for wg in [4, 8, 16] {
            let mut sim = GpuSim::with_workgroup(gpu, &s, wg)?;
            sim.set_ez(&bump(&s))?;
            sim.step_n(300);
            let mut bits = Vec::new();
            for c in FIELDS {
                bits.extend(sim.read(c)?.iter().map(|v| v.to_bits()));
            }
            runs.push((wg, bits));
        }
        for (wg, bits) in &runs[1..] {
            if *bits != runs[0].1 {
                return Err(format!("{name}: workgroup {wg} differs from {}", runs[0].0).into());
            }
        }
    }
    Ok(())
}

fn main() {
    let args = Arguments::from_args();
    let gpu = match Gpu::shared() {
        Ok(gpu) => {
            eprintln!("tier 4 on {}", gpu.describe());
            Some(gpu)
        }
        Err(e) => {
            if std::env::var("FDTD_REQUIRE_GPU").is_ok_and(|v| v == "1") {
                eprintln!("FDTD_REQUIRE_GPU=1 but no GPU: {e}");
                std::process::exit(1);
            }
            eprintln!(
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            );
            eprintln!("!!! TIER 4 SKIPPED: no GPU adapter ({e})");
            eprintln!("!!! every GPU trial below is IGNORED, not passed");
            eprintln!(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            );
            None
        }
    };
    let checks: Vec<(&str, Check)> = vec![
        ("readback_round_trip", readback_round_trip),
        ("fields_zero_at_step_0", fields_zero_at_step_0),
        ("initial_value_matches_cpu", initial_value_matches_cpu),
        ("padding_stays_zero", padding_stays_zero),
        ("workgroup_size_independent", workgroup_size_independent),
    ];
    let trials = checks
        .into_iter()
        .map(|(name, check)| {
            Trial::test(name, move || check(gpu.expect("ignored without a gpu")))
                .with_ignored_flag(gpu.is_none())
        })
        .collect();
    libtest_mimic::run(&args, trials).exit();
}
