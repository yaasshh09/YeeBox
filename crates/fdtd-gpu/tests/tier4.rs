//! Tier 4: the GPU kernel against the CPU reference.
//!
//! Its own harness (libtest-mimic) so that a machine without a GPU adapter
//! reports every trial as ignored, with a banner, instead of as passed.
//! With `FDTD_REQUIRE_GPU=1` (set in CI) a missing adapter is a failure.
//!
//! Every scene under `validation/scenes/` is a trial of its own: 2000
//! steps, every field, probe record and spectrum within 1e-4 of the
//! reference peak at every 250-step checkpoint, padding exactly zero.

use fdtd_core::grid::Component;
use fdtd_core::kernel::{Diff, Kernel, compare};
use fdtd_core::scene::Scene;
use fdtd_core::scene::ViewField;
use fdtd_core::tm::SimTm;
use fdtd_gpu::render::{Frame, Renderer, Viewport};
use fdtd_gpu::tier4::{self, GATE, STEPS};
use fdtd_gpu::{Gpu, GpuSim};
use libtest_mimic::{Arguments, Failed, Trial};
use wgpu::util::DeviceExt;

type Check = fn(&'static Gpu) -> Result<(), Failed>;

const FIELDS: [Component; 3] = [Component::Ez, Component::Hx, Component::Hy];

fn scene(name: &str) -> Scene {
    let path = tier4::scenes_dir().join(name);
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

/// Every checkpoint within the gate, the last one printed.
fn within_gate(name: &str, diffs: &[Diff]) -> Result<(), Failed> {
    for d in diffs {
        if d.worst().is_nan() || d.worst() >= GATE {
            return Err(format!("{name}: {d:?}").into());
        }
    }
    if let Some(d) = diffs.last() {
        eprintln!(
            "  {name} step {}: worst {:.2e} fields {:.2e} {:.2e} {:.2e} probes {:.2e} spectra {:.2e}",
            d.step,
            d.worst(),
            d.fields[0],
            d.fields[1],
            d.fields[2],
            d.probes,
            d.spectra
        );
    }
    Ok(())
}

/// Hx row ny - 1 (unless y is periodic), Hy column nx - 1 and the wall ring
/// of Ez are exactly zero.
fn padding_is_zero(name: &str, s: &Scene, sim: &mut GpuSim) -> Result<(), Failed> {
    let (nx, ny) = (s.grid.nx, s.grid.ny);
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
    Ok(())
}

/// One committed scene, as it is: the Tier 4 acceptance case.
fn scene_matches_cpu(gpu: &'static Gpu, name: &str, s: &Scene) -> Result<(), Failed> {
    let mut cpu = SimTm::from_scene(s)?;
    let mut gpu_sim = GpuSim::new(gpu, s)?;
    let diffs = compare(&mut cpu, &mut gpu_sim, &tier4::checkpoints(STEPS, 250))?;
    within_gate(name, &diffs)?;
    padding_is_zero(name, s, &mut gpu_sim)
}

/// An edit applied at a step: it changes the scene at normalized time `t`
/// and returns the bounds of a material edit, or `None` for a source edit.
type Edit = (u64, fn(&mut Scene, f64) -> Option<[usize; 4]>);

/// Runs both kernels to each edit's step, applies the edit to the scene and
/// to both kernels, and carries on to `STEPS`. Checkpoints every 250 steps
/// and at every edit, all within the gate.
fn edited_run(gpu: &'static Gpu, name: &str, mut s: Scene, edits: &[Edit]) -> Result<(), Failed> {
    let mut cpu = SimTm::from_scene(&s)?;
    let mut gpu_sim = GpuSim::new(gpu, &s)?;
    let mut diffs = Vec::new();
    let mut prev = 0;
    let marks = edits.iter().map(|e| e.0).chain([STEPS]);
    for (n, mark) in marks.enumerate() {
        let mut cps: Vec<u64> = (prev + 1..=mark).filter(|c| c % 250 == 0).collect();
        if cps.last() != Some(&mark) {
            cps.push(mark);
        }
        diffs.extend(compare(&mut cpu, &mut gpu_sim, &cps)?);
        if let Some((_, edit)) = edits.get(n) {
            let t = mark as f64 * s.grid.courant;
            match edit(&mut s, t) {
                Some(b) => {
                    cpu.set_materials(&s, b)?;
                    gpu_sim.set_materials(&s, b)?;
                }
                None => {
                    cpu.set_sources(&s)?;
                    gpu_sim.set_sources(&s)?;
                }
            }
        }
        prev = mark;
    }
    within_gate(name, &diffs)?;
    padding_is_zero(name, &s, &mut gpu_sim)
}

fn cw(f0: f64) -> fdtd_core::source::Waveform {
    fdtd_core::source::Waveform::RampedCw {
        f0,
        ramp_periods: 3.0,
        phase: 0.0,
    }
}

/// Painting a wall through the layer, erasing part of it, dragging and
/// retuning the source and adding a second one, mid-run.
fn live_edits_match_cpu(gpu: &'static Gpu) -> Result<(), Failed> {
    let mut s = scene("painted_walls.json");
    s.sources[0].waveform = cw(0.05);
    let edits: [Edit; 5] = [
        (500, |s, _| {
            s.paint_stroke((65.0, 0.0), (65.0, 40.0), 1.0, 2)
                .unwrap()
                .bounds
        }),
        (900, |s, _| {
            s.paint_stroke((65.0, 15.0), (65.0, 25.0), 1.5, 0)
                .unwrap()
                .bounds
        }),
        (1200, |s, t| {
            let patch = fdtd_core::scene::SourcePatch {
                i: Some(25),
                j: Some(35),
                wavelength_cells: Some(15.0),
                ..Default::default()
            };
            s.patch_source(0, &patch, t).unwrap();
            None
        }),
        (1500, |s, _| {
            let mut extra = s.sources[0];
            extra.j = 15;
            s.sources.push(extra);
            None
        }),
        (1750, |s, _| {
            s.paint_stroke((20.0, 50.0), (40.0, 50.0), 2.0, 1)
                .unwrap()
                .bounds
        }),
    ];
    edited_run(gpu, "live edits", s, &edits)
}

/// A scene built without sources gets its first one mid-run.
fn first_source_added_live(gpu: &'static Gpu) -> Result<(), Failed> {
    let s = fields_only("cavity_tm.json");
    let edits: [Edit; 1] = [(300, |s, _| {
        s.sources.push(fdtd_core::scene::Source {
            // In vacuum, clear of the PEC bar at x 20 to 22.
            i: 12,
            j: 20,
            waveform: cw(0.04),
            amplitude: 1.0,
            kind: Default::default(),
        });
        None
    })];
    edited_run(gpu, "first source live", s, &edits)
}

/// Renders `sim` onto a `w` x `h` canvas with the grid fitted, and reads
/// the pixels back, row-major from the top left.
fn render_pixels(
    gpu: &'static Gpu,
    s: &Scene,
    sim: &GpuSim,
    w: u32,
    h: u32,
) -> Result<Vec<[u8; 4]>, Failed> {
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    let renderer = Renderer::new(gpu, sim, s, format);
    let frame = Frame {
        viewport: Viewport::fit(s.grid.nx, s.grid.ny, w as f64, h as f64),
        gain: 1.0,
        field: ViewField::Ez,
    };
    let row = (w * 4).div_ceil(256) * 256;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame copy"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    renderer.draw(gpu, &mut enc, &view, &frame);
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        size,
    );
    gpu.queue.submit([enc.finish()]);
    let bytes = gpu.read(&buf, 0, (row * h) as u64)?;
    Ok((0..h as usize)
        .flat_map(|y| {
            let at = y * row as usize;
            bytes[at..at + w as usize * 4]
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect::<Vec<_>>()
        })
        .collect())
}

/// At 1 px per cell, a node shows its exact table color, the CPML is
/// tinted, a conductor is opaque, and outside the grid is the stage.
fn render_pixels_match_tables(gpu: &'static Gpu) -> Result<(), Failed> {
    let s = scene("painted_walls.json");
    let (nx, ny) = (s.grid.nx, s.grid.ny);
    let mut sim = GpuSim::new(gpu, &s)?;
    let mut ez = vec![0.0; nx * ny];
    ez[40 * nx + 30] = 0.5;
    sim.set_ez(&ez)?;
    let px = render_pixels(gpu, &s, &sim, nx as u32, ny as u32)?;
    let at = |i: usize, j: usize| px[(ny - 1 - j) * nx + i];
    let lut = fdtd_gpu::colormap::diverging();
    let close = |a: [u8; 4], b: [f32; 3]| (0..3).all(|c| (a[c] as f32 - b[c] * 255.0).abs() <= 1.0);
    if at(30, 40) != lut[192] {
        return Err(format!(
            "node (30, 40) at 0.5: {:?}, want {:?}",
            at(30, 40),
            lut[192]
        )
        .into());
    }
    if at(35, 30) != lut[128] {
        return Err(format!("empty node: {:?}, want {:?}", at(35, 30), lut[128]).into());
    }
    let zero = lut[128].map(|v| v as f32 / 255.0);
    let tinted: [f32; 3] = std::array::from_fn(|c| zero[c] * 0.65 + [0.20, 0.22, 0.30][c] * 0.35);
    if !close(at(3, 30), tinted) {
        return Err(format!("cpml node: {:?}, want about {tinted:?}", at(3, 30)).into());
    }
    let pec = fdtd_gpu::render::overlay(&fdtd_core::material::Material::Pec);
    if !close(at(45, 5), [pec[0], pec[1], pec[2]]) {
        return Err(format!("painted conductor: {:?}", at(45, 5)).into());
    }
    // A canvas half again as wide: 20 px of stage on each side.
    let wide = render_pixels(gpu, &s, &sim, nx as u32 + 40, ny as u32)?;
    let (left, inside) = (wide[30 * (nx + 40) + 5], wide[30 * (nx + 40) + 20 + 35]);
    if left != [0, 0, 0, 255] || inside != at(35, 29) {
        return Err(format!("letterbox: {left:?}, inside {inside:?}").into());
    }
    Ok(())
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

/// The H and E updates alone, from the same initial Ez, on every kind of
/// grid, with nothing else running.
fn initial_value_matches_cpu(gpu: &'static Gpu) -> Result<(), Failed> {
    for name in FIELD_SCENES {
        let s = fields_only(name);
        let mut cpu = SimTm::from_scene(&s)?;
        let mut gpu_sim = GpuSim::new(gpu, &s)?;
        cpu.set_ez(&bump(&s))?;
        gpu_sim.set_ez(&bump(&s))?;
        let diffs = compare(&mut cpu, &mut gpu_sim, &tier4::checkpoints(STEPS, 250))?;
        within_gate(&format!("{name} (initial value)"), &diffs)?;
        padding_is_zero(name, &s, &mut gpu_sim)?;
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

/// 50000 steps on a lossless cavity: the difference grows no faster than
/// linearly, which rules out a hidden instability in the f32 kernel.
fn long_run_growth(gpu: &'static Gpu) -> Result<(), Failed> {
    let diffs = tier4::long_run(gpu)?;
    for d in &diffs {
        eprintln!("  long run step {:6}: worst {:.3e}", d.step, d.worst());
    }
    let e = tier4::growth_exponent(&diffs, tier4::LONG_EVERY);
    eprintln!(
        "  long run growth exponent {e:.3} (bound {})",
        tier4::MAX_GROWTH
    );
    if e.is_nan() || e > tier4::MAX_GROWTH {
        return Err(format!("difference grows like n^{e:.3}").into());
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
        ("workgroup_size_independent", workgroup_size_independent),
        ("long_run_growth", long_run_growth),
        ("live_edits_match_cpu", live_edits_match_cpu),
        ("first_source_added_live", first_source_added_live),
        ("render_pixels_match_tables", render_pixels_match_tables),
    ];
    let mut trials: Vec<Trial> = checks
        .into_iter()
        .map(|(name, check)| Trial::test(name, move || check(gpu.expect("ignored without a gpu"))))
        .collect();
    let scenes = tier4::scenes().expect("validation/scenes is readable");
    assert!(scenes.len() >= 8, "only {} tier 4 scenes", scenes.len());
    for (name, s) in scenes {
        let label = format!("scene {}", name.trim_end_matches(".json"));
        trials.push(Trial::test(label, move || {
            scene_matches_cpu(gpu.expect("ignored without a gpu"), &name, &s)
        }));
    }
    let trials = trials
        .into_iter()
        .map(|t| t.with_ignored_flag(gpu.is_none()))
        .collect();
    libtest_mimic::run(&args, trials).exit();
}
