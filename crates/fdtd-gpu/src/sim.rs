//! `GpuSim`: the TMz scene on the GPU, stepped in batches.
//!
//! A batch of up to `BATCH` steps is one command buffer and one compute
//! pass, with the passes of every step in the order of `SimTm::step`. Each
//! step reads its slot of the `Params` uniform through a dynamic offset.

use crate::buffers::{Layout, PARAMS_STRIDE, Params, field_id, pack_corrections};
use crate::device::{Gpu, Mapped};
use crate::pipeline::{self, Access, Pass};
use fdtd_core::grid::Component;
use fdtd_core::kernel::{BuildKey, material_edit};
#[cfg(not(target_arch = "wasm32"))]
use fdtd_core::kernel::{Kernel, Spectrum};
use fdtd_core::monitor::{Dft, FluxLine};
use fdtd_core::ntff::Ntff;
use fdtd_core::scene::{Mode, Monitor, Scene, Source};
use fdtd_core::source::SourceKind;
use fdtd_core::tfsf::Tfsf;

/// The medium update every E shader calls, prepended to its source.
const MEDIUM: &str = include_str!("../shaders/medium.wgsl");
use fdtd_core::{FdtdError, Result};
use std::collections::VecDeque;
use wgpu::util::DeviceExt;

/// Most steps encoded into one submission.
pub const BATCH: u64 = 64;

/// Field and psi buffers, in this order, named for TMz. TEz uses the same
/// slots by layout: Hz, Ex, Ey, then psi_exy, psi_eyx, psi_hzx, psi_hzy.
const EZ: usize = 0;
const HX: usize = 1;
const HY: usize = 2;
const PSI_HXY: usize = 3;
const PSI_HYX: usize = 4;
const PSI_EZX: usize = 5;
const PSI_EZY: usize = 6;

/// One compute dispatch of a step: its pass, storage bind group and
/// workgroup counts.
struct Stage {
    pass: Pass,
    group: wgpu::BindGroup,
    count: (u32, u32),
}

/// A monitor as the GPU kernel runs it: its taps start at `at` in each
/// step's gather row, and the accumulation is the `fdtd-core` code in f64.
enum Mon {
    Point(Vec<f64>),
    Dft(Dft),
    Flux(FluxLine),
    Ntff(Ntff),
}

/// The f32 GPU kernel for one scene.
pub struct GpuSim {
    gpu: &'static Gpu,
    n: u64,
    /// The Courant number in f64, for source times; `base.s` is its f32 copy.
    courant: f64,
    base: Params,
    mode: Mode,
    /// The uploaded layout in f64, kept current through live edits: node
    /// material ids, the medium table, the TEz edge ids.
    layout: Layout,
    /// Nodes whose field `set_ez` forces to zero: walls and PEC in TMz, the
    /// ring in TEz (as `SimTm::set_ez` and `SimTe::set_hz`).
    node_zero: Vec<bool>,
    /// Material id per node: the TMz E update and the render pass read it.
    ids: wgpu::Buffer,
    /// TEz: the medium row of every Ex and every Ey edge.
    edge_ids: Option<[wgpu::Buffer; 2]>,
    /// J, P and P^(n-1) of each E array: one in TMz, Ex and Ey in TEz.
    aux: Vec<wgpu::Buffer>,
    /// What live edits must leave alone.
    key: BuildKey,
    /// Index in `stages` of the sources stage, which always exists.
    src_stage: usize,
    /// Running mean of Ez^2 for the intensity view, and its 1 / tau.
    mean: wgpu::Buffer,
    alpha: wgpu::Buffer,
    /// The mean's stage, dispatched last in a step while the view is on.
    intensity: Stage,
    intensity_on: bool,
    fields: Vec<wgpu::Buffer>,
    params: wgpu::Buffer,
    params_group: wgpu::BindGroup,
    /// The dispatches of one step, in the order of `SimTm::step`.
    stages: Vec<Stage>,
    sources: Vec<Source>,
    /// Per-step source values of the current batch, `[b * n_src + s]`.
    src_values: wgpu::Buffer,
    /// The TF/SF auxiliary grid, stepped on the CPU in lockstep, f64.
    tfsf: Option<Tfsf>,
    /// Incident Ez and H of the current batch, `[b * n_inc + q]`, `[2 b + i]`.
    inc_e: wgpu::Buffer,
    inc_h: wgpu::Buffer,
    /// Monitors in scene order, each with its first tap.
    monitors: Vec<(usize, Mon)>,
    n_taps: usize,
    /// Gathered taps of the current batch, `[b * n_taps + t]`.
    gathered: wgpu::Buffer,
    /// The first error seen while stepping, returned by the next read.
    pending: Option<FdtdError>,
    /// Gathered taps of submitted batches, oldest first, on their way to
    /// the CPU. Drained strictly in order.
    tap_queue: VecDeque<QueuedTaps>,
    /// The tap list; its last entry is the live probe's slot.
    tap_buf: wgpu::Buffer,
    /// The live probe's node, if one is set, and the Ez samples it has
    /// taken since they were last collected.
    probe: Option<usize>,
    probe_samples: Vec<f32>,
}

/// Most live probe samples kept between collections; older ones drop.
const MAX_PROBE_SAMPLES: usize = 1 << 16;

/// The taps of one batch, copied to a staging buffer and being mapped.
struct QueuedTaps {
    first: u64,
    buf: wgpu::Buffer,
    mapped: Mapped,
}

/// A storage buffer with `contents`, at least 16 bytes: a zero-sized
/// binding is invalid, so an empty list still gets a buffer.
fn storage_buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    let mut bytes = contents.to_vec();
    bytes.resize(bytes.len().max(16), 0);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

impl GpuSim {
    /// Validates the scene and uploads it, workgroup edge 8.
    pub fn new(gpu: &'static Gpu, scene: &Scene) -> Result<Self> {
        Self::with_workgroup(gpu, scene, 8)
    }

    /// The same with a `wg` x `wg` workgroup, `wg` in 1 to 16. The result
    /// must not depend on it; Tier 4 checks 4, 8 and 16.
    pub fn with_workgroup(gpu: &'static Gpu, scene: &Scene, wg: u32) -> Result<Self> {
        scene.validate()?;
        if !(1..=16).contains(&wg) {
            return Err(FdtdError::InvalidParameter {
                name: "workgroup edge",
                value: wg as f64,
            });
        }
        let (nx, ny) = (scene.grid.nx, scene.grid.ny);
        let layout = Layout::new(scene);
        let limits = gpu.device.limits();
        let bytes = (nx * ny * 4) as u64;
        let aux_bytes = if layout.dispersive() { 3 * bytes } else { 16 };
        let groups = nx.max(ny).div_ceil(wg as usize) as u32;
        let too_big =
            |b: u64| b > limits.max_storage_buffer_binding_size || b > limits.max_buffer_size;
        if too_big(bytes)
            || too_big(aux_bytes)
            || groups > limits.max_compute_workgroups_per_dimension
        {
            return Err(FdtdError::GridSize { nx, ny });
        }
        let device = &gpu.device;
        // Storage buffers are never empty: a zero-sized binding is invalid,
        // so a scene without sources still gets a 16-byte buffer.
        let storage = |label: &str, contents: &[u8]| storage_buffer(device, label, contents);
        let zeros = vec![0f32; nx * ny];
        let fields: Vec<_> = [
            "field 0", "field 1", "field 2", "psi 3", "psi 4", "psi 5", "psi 6",
        ]
        .iter()
        .map(|name| storage(name, bytemuck::cast_slice(&zeros)))
        .collect();
        let ids = storage("ids", bytemuck::cast_slice(&layout.ids));
        let coef = storage("coef", bytemuck::cast_slice(&layout.coef()));
        let aux_len = if layout.dispersive() { 3 * nx * ny } else { 0 };
        let n_aux = if scene.mode == Mode::Te { 2 } else { 1 };
        let aux: Vec<_> = (0..n_aux)
            .map(|_| storage("aux", bytemuck::cast_slice(&vec![0f32; aux_len])))
            .collect();
        let edge_ids = layout.edge_ids.as_ref().map(|[x, y]| {
            [
                storage("edge ids x", bytemuck::cast_slice(x)),
                storage("edge ids y", bytemuck::cast_slice(y)),
            ]
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: BATCH * PARAMS_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_layout = pipeline::params_layout(device);
        let params_group = pipeline::params_group(device, &params_layout, &params);
        use Access::{Read, ReadWrite};
        let shader = |label: &str, src: &str, access: &[Access]| {
            pipeline::pass(device, &params_layout, label, src, wg, access)
        };
        let stage = |pass: Pass, buffers: &[&wgpu::Buffer], count: (u32, u32)| Stage {
            group: pipeline::storage_group(device, &pass, buffers),
            pass,
            count,
        };
        let with_medium = |src: &str| format!("{MEDIUM}\n{src}");
        let grid = (
            nx.div_ceil(wg as usize) as u32,
            ny.div_ceil(wg as usize) as u32,
        );
        // TF/SF, from the same lists the CPU applies. TEz runs the TMz
        // auxiliary grid through duality (`SimTe`): waveform delayed by S/2,
        // vacuum, and the edge corrections with the sign flipped.
        let s = scene.grid.courant;
        let tfsf = match &scene.plane_wave {
            Some(pw) => {
                let mut launch = *pw;
                let eps = match scene.mode {
                    Mode::Tm => scene.materials[0].eps(),
                    Mode::Te => {
                        launch.waveform = pw.waveform.delayed(0.5 * s);
                        1.0
                    }
                };
                Some(Tfsf::new(&launch, eps, s)?)
            }
            None => None,
        };
        let (mut n_inc, mut n_hgroups, mut n_egroups) = (0, 0, 0);
        let (mut edge_fix, mut node_fix) = (None, None);
        let (inc_e, inc_h) = match &tfsf {
            Some(t) => {
                n_inc = t.incident_e().len();
                let mut hc = t.h_corrections(nx, s, scene.periodic_y);
                if scene.mode == Mode::Te {
                    for c in hc.iter_mut().flatten() {
                        c.factor = -c.factor;
                    }
                }
                let ec = t.e_corrections(nx, &layout.cb_nodes());
                (n_hgroups, n_egroups) = (hc.len(), ec.len());
                let hc = storage("tfsf edges", bytemuck::cast_slice(&pack_corrections(&hc)));
                let ec = storage("tfsf nodes", bytemuck::cast_slice(&pack_corrections(&ec)));
                let inc_e = storage("inc e", &vec![0u8; BATCH as usize * n_inc * 4]);
                let inc_h = storage("inc h", &vec![0u8; BATCH as usize * 2 * 4]);
                let pass = shader(
                    "tfsf_h",
                    include_str!("../shaders/tfsf_h.wgsl"),
                    &[ReadWrite, ReadWrite, Read, Read],
                );
                edge_fix = Some(stage(
                    pass,
                    &[&fields[HX], &fields[HY], &hc, &inc_e],
                    ((n_hgroups as u32).div_ceil(64), 1),
                ));
                let pass = shader(
                    "tfsf_e",
                    include_str!("../shaders/tfsf_e.wgsl"),
                    &[ReadWrite, Read, Read],
                );
                node_fix = Some(stage(
                    pass,
                    &[&fields[EZ], &ec, &inc_h],
                    ((n_egroups as u32).div_ceil(64), 1),
                ));
                (inc_e, inc_h)
            }
            None => (storage("inc e", &[]), storage("inc h", &[])),
        };
        let n_src = scene.sources.len();
        let nodes: Vec<[u32; 2]> = scene
            .sources
            .iter()
            .map(|src| {
                let hard = src.kind == SourceKind::Hard;
                [(src.j * nx + src.i) as u32, hard as u32]
            })
            .collect();
        let src_nodes = storage("source nodes", bytemuck::cast_slice(&nodes));
        let src_values = storage("source values", &vec![0u8; BATCH as usize * n_src * 4]);
        // Always there, so a live edit can add the first source; with none
        // the shader loop is empty. It writes the node field in both modes.
        let pass = shader(
            "sources",
            include_str!("../shaders/sources.wgsl"),
            &[ReadWrite, Read, Read],
        );
        let sources_stage = stage(pass, &[&fields[EZ], &src_nodes, &src_values], (1, 1));
        // The dispatches of one step, in the order of `SimTm::step` or
        // `SimTe::step`.
        let mut stages = Vec::new();
        let src_stage;
        match scene.mode {
            Mode::Tm => {
                let h = shader(
                    "update_h",
                    include_str!("../shaders/update_h.wgsl"),
                    &[Read, ReadWrite, ReadWrite, ReadWrite, ReadWrite, Read],
                );
                let h_buffers = [
                    &fields[EZ],
                    &fields[HX],
                    &fields[HY],
                    &fields[PSI_HXY],
                    &fields[PSI_HYX],
                    &coef,
                ];
                stages.push(stage(h, &h_buffers, grid));
                stages.extend(edge_fix);
                let e = shader(
                    "update_e",
                    &with_medium(include_str!("../shaders/update_e.wgsl")),
                    &[
                        ReadWrite, Read, Read, ReadWrite, ReadWrite, Read, Read, ReadWrite,
                    ],
                );
                let e_buffers = [
                    &fields[EZ],
                    &fields[HX],
                    &fields[HY],
                    &fields[PSI_EZX],
                    &fields[PSI_EZY],
                    &ids,
                    &coef,
                    &aux[0],
                ];
                stages.push(stage(e, &e_buffers, grid));
                stages.extend(node_fix);
                src_stage = stages.len();
                stages.push(sources_stage);
            }
            Mode::Te => {
                let Some([ids_x, ids_y]) = edge_ids.as_ref() else {
                    return Err(FdtdError::Gpu("a TEz layout without edge ids".into()));
                };
                let hz = shader(
                    "update_hz",
                    include_str!("../shaders/update_hz.wgsl"),
                    &[ReadWrite, Read, Read, ReadWrite, ReadWrite, Read],
                );
                let hz_buffers = [
                    &fields[EZ],
                    &fields[HX],
                    &fields[HY],
                    &fields[PSI_EZX],
                    &fields[PSI_EZY],
                    &coef,
                ];
                stages.push(stage(hz, &hz_buffers, grid));
                stages.extend(node_fix);
                src_stage = stages.len();
                stages.push(sources_stage);
                let edge_access = [Read, ReadWrite, ReadWrite, Read, Read, ReadWrite];
                let ex = shader(
                    "update_ex",
                    &with_medium(include_str!("../shaders/update_ex.wgsl")),
                    &edge_access,
                );
                let ex_buffers = [
                    &fields[EZ],
                    &fields[HX],
                    &fields[PSI_HXY],
                    &coef,
                    ids_x,
                    &aux[0],
                ];
                stages.push(stage(ex, &ex_buffers, grid));
                let ey = shader(
                    "update_ey",
                    &with_medium(include_str!("../shaders/update_ey.wgsl")),
                    &edge_access,
                );
                let ey_buffers = [
                    &fields[EZ],
                    &fields[HY],
                    &fields[PSI_HYX],
                    &coef,
                    ids_y,
                    &aux[1],
                ];
                stages.push(stage(ey, &ey_buffers, grid));
                stages.extend(edge_fix);
            }
        }
        let mut taps: Vec<[u32; 2]> = Vec::new();
        let mut monitors = Vec::new();
        for m in &scene.monitors {
            let at = taps.len();
            taps.extend(
                m.taps(scene.mode, nx, ny)?
                    .iter()
                    .map(|&(c, k)| [field_id(c), k as u32]),
            );
            // Entry 0 of a point history is Ez at n = 0, which is zero.
            monitors.push((
                at,
                match m {
                    Monitor::Point { .. } => Mon::Point(vec![0.0]),
                    Monitor::Dft { freqs, .. } => Mon::Dft(Dft::new(freqs, 1)?),
                    &Monitor::Flux {
                        normal,
                        at,
                        from,
                        to,
                        ref freqs,
                    } => Mon::Flux(FluxLine::new(normal, at, from, to, freqs)?.in_mode(scene.mode)),
                    Monitor::Ntff { .. } => m
                        .ntff(scene.mode, scene.materials[0].eps())?
                        .map_or(Mon::Point(vec![]), Mon::Ntff),
                },
            ));
        }
        // One more slot at the end for the live probe (`set_probe`), so it
        // can move without changing the row layout of batches in flight.
        // It starts on wall node 0, which is always zero.
        taps.push([field_id(Component::Ez), 0]);
        let n_taps = taps.len();
        let tap_buf = storage("taps", bytemuck::cast_slice(&taps));
        let gathered = storage("gathered", &vec![0u8; BATCH as usize * n_taps * 4]);
        if n_taps > 0 {
            let pass = pipeline::pass(
                device,
                &params_layout,
                "gather",
                include_str!("../shaders/gather.wgsl"),
                wg,
                &[Read, Read, Read, Read, ReadWrite],
            );
            let group = pipeline::storage_group(
                device,
                &pass,
                &[&fields[EZ], &fields[HX], &fields[HY], &tap_buf, &gathered],
            );
            stages.push(Stage {
                pass,
                group,
                count: ((n_taps as u32).div_ceil(64), 1),
            });
        }
        let mean = storage("intensity mean", bytemuck::cast_slice(&zeros));
        let alpha = storage("intensity alpha", &[0u8; 4]);
        let pass = pipeline::pass(
            device,
            &params_layout,
            "intensity",
            include_str!("../shaders/intensity.wgsl"),
            wg,
            &[Read, ReadWrite, Read],
        );
        let group = pipeline::storage_group(device, &pass, &[&fields[EZ], &mean, &alpha]);
        let intensity = Stage {
            pass,
            group,
            count: grid,
        };
        gpu.check()?;
        let node_zero = match scene.mode {
            Mode::Tm => layout.cb_nodes().iter().map(|&cb| cb == 0.0).collect(),
            Mode::Te => (0..nx * ny)
                .map(|k| {
                    let (i, j) = (k % nx, k / nx);
                    i == 0 || i == nx - 1 || !scene.live_rows().contains(&j)
                })
                .collect(),
        };
        Ok(GpuSim {
            gpu,
            n: 0,
            courant: scene.grid.courant,
            base: Params {
                nx: nx as u32,
                ny: ny as u32,
                periodic_y: scene.periodic_y as u32,
                step: 0,
                s: scene.grid.courant as f32,
                n_src: n_src as u32,
                n_taps: n_taps as u32,
                n_hgroups: n_hgroups as u32,
                n_egroups: n_egroups as u32,
                n_inc: n_inc as u32,
                pad: [0; 2],
            },
            mode: scene.mode,
            node_zero,
            layout,
            ids,
            edge_ids,
            aux,
            key: BuildKey::of(scene),
            src_stage,
            mean,
            alpha,
            intensity,
            intensity_on: false,
            fields,
            params,
            params_group,
            stages,
            sources: scene.sources.clone(),
            src_values,
            tfsf,
            inc_e,
            inc_h,
            monitors,
            n_taps,
            gathered,
            pending: None,
            tap_queue: VecDeque::new(),
            tap_buf,
            probe: None,
            probe_samples: Vec::new(),
        })
    }

    /// Replaces the node field (Ez, or Hz in TEz; one value per node,
    /// `j * nx + i`, normalized), forcing the nodes `SimTm::set_ez` or
    /// `SimTe::set_hz` force to zero. The edge fields are left alone.
    pub fn set_ez(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.node_zero.len() {
            return Err(FdtdError::InvalidParameter {
                name: "ez length",
                value: values.len() as f64,
            });
        }
        if let Some(&bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(FdtdError::InvalidParameter {
                name: "ez value",
                value: bad,
            });
        }
        let ez: Vec<f32> = values
            .iter()
            .zip(&self.node_zero)
            .map(|(&v, &zero)| if zero { 0.0 } else { v as f32 })
            .collect();
        self.gpu
            .queue
            .write_buffer(&self.fields[EZ], 0, bytemuck::cast_slice(&ez));
        Ok(())
    }

    /// Rewrites the ids an edit changes (node material ids, and in TEz every
    /// edge id) and restarts J and P where a pole changed, by the rule of the
    /// CPU kernel of the mode: the material id in TMz, the row value in TEz.
    /// The writes land between the batches already submitted and the next.
    pub fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        self.key.check(scene)?;
        let Some(edit) = material_edit(scene, bounds)? else {
            return Ok(());
        };
        let nx = self.base.nx as usize;
        let n = nx * self.base.ny as usize;
        let w = edit.i1 - edit.i0 + 1;
        let dispersive = self.layout.dispersive();
        let mut restart: Vec<Vec<usize>> = vec![Vec::new(); self.aux.len()];
        let queue = &self.gpu.queue;
        for (r, j) in edit.rows.clone().enumerate() {
            let k0 = j * nx + edit.i0;
            for (c, &id) in edit.ids[r * w..(r + 1) * w].iter().enumerate() {
                let k = k0 + c;
                let table = &self.layout.table;
                if self.mode == Mode::Tm {
                    let pole = |m: usize| table[m].is_dispersive().then_some(m);
                    if dispersive && pole(self.layout.ids[k] as usize) != pole(id) {
                        restart[0].push(k);
                    }
                    self.node_zero[k] = table[id].cb == 0.0;
                }
                self.layout.ids[k] = id as u32;
            }
            let row = &self.layout.ids[k0..k0 + w];
            queue.write_buffer(&self.ids, (k0 * 4) as u64, bytemuck::cast_slice(row));
        }
        if let Some(bufs) = &self.edge_ids {
            let new = self.layout.edges(scene, &scene.rasterize());
            if let Some(old) = &self.layout.edge_ids {
                let table = &self.layout.table;
                let pole = |m: u32| {
                    let r = table[m as usize];
                    r.is_dispersive().then_some(r)
                };
                for a in 0..2 {
                    for k in 0..n {
                        if dispersive && pole(old[a][k]) != pole(new[a][k]) {
                            restart[a].push(k);
                        }
                    }
                }
            }
            for a in 0..2 {
                queue.write_buffer(&bufs[a], 0, bytemuck::cast_slice(&new[a]));
            }
            self.layout.edge_ids = Some(new);
        }
        for (a, ks) in restart.iter().enumerate() {
            for &k in ks {
                for part in 0..3 {
                    let at = ((part * n + k) * 4) as u64;
                    queue.write_buffer(&self.aux[a], at, bytemuck::bytes_of(&0f32));
                }
            }
        }
        Ok(())
    }

    /// Rebuilds the source node and value buffers and the sources stage's
    /// bind group for the new list.
    pub fn set_sources(&mut self, scene: &Scene) -> Result<()> {
        self.key.check(scene)?;
        for src in &scene.sources {
            scene.check_source(src)?;
        }
        let nx = self.base.nx as usize;
        let nodes: Vec<[u32; 2]> = scene
            .sources
            .iter()
            .map(|src| {
                [
                    (src.j * nx + src.i) as u32,
                    (src.kind == SourceKind::Hard) as u32,
                ]
            })
            .collect();
        let device = &self.gpu.device;
        let src_nodes = storage_buffer(device, "source nodes", bytemuck::cast_slice(&nodes));
        self.src_values = storage_buffer(
            device,
            "source values",
            &vec![0u8; BATCH as usize * nodes.len() * 4],
        );
        let stage = &mut self.stages[self.src_stage];
        stage.group = pipeline::storage_group(
            device,
            &stage.pass,
            &[&self.fields[EZ], &src_nodes, &self.src_values],
        );
        self.sources = scene.sources.clone();
        self.base.n_src = nodes.len() as u32;
        Ok(())
    }

    /// Starts a fresh running mean of Ez^2 for the intensity view, with a
    /// time constant of `tau` steps (at least 1): after each step,
    /// `mean += (ez^2 - mean) / tau`. `None` stops updating it. A display
    /// quantity only; the fields are untouched.
    pub fn set_intensity(&mut self, tau: Option<f64>) -> Result<()> {
        match tau {
            None => self.intensity_on = false,
            Some(t) => {
                if !(t.is_finite() && t >= 1.0) {
                    return Err(FdtdError::InvalidParameter {
                        name: "intensity tau",
                        value: t,
                    });
                }
                let q = &self.gpu.queue;
                q.write_buffer(&self.mean, 0, &vec![0u8; self.node_zero.len() * 4]);
                q.write_buffer(&self.alpha, 0, bytemuck::bytes_of(&((1.0 / t) as f32)));
                self.intensity_on = true;
            }
        }
        Ok(())
    }

    /// The running mean of Ez^2, `nx * ny`, index `j * nx + i`.
    pub async fn read_intensity_async(&mut self) -> Result<Vec<f64>> {
        let bytes = self.gpu.read_async(&self.mean, 0, self.mean.size()).await?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| v as f64)
            .collect())
    }

    /// The intensity mean buffer, for the render pass.
    pub(crate) fn mean_buffer(&self) -> &wgpu::Buffer {
        &self.mean
    }

    /// Puts the live probe on Ez node (i, j), or removes it. Its samples,
    /// Ez after every step, collect until `take_probe_samples`; moving it
    /// drops the old ones. Batches already in flight finish at the old node.
    pub fn set_probe(&mut self, node: Option<(usize, usize)>) -> Result<()> {
        let (nx, ny) = (self.base.nx as usize, self.base.ny as usize);
        self.probe_samples.clear();
        self.probe = match node {
            None => None,
            Some((i, j)) => {
                if i >= nx || j >= ny {
                    return Err(FdtdError::OutOfDomain {
                        what: "probe",
                        i,
                        j,
                    });
                }
                let k = j * nx + i;
                let tap = [field_id(Component::Ez), k as u32];
                let at = ((self.n_taps - 1) * 8) as u64;
                self.gpu
                    .queue
                    .write_buffer(&self.tap_buf, at, bytemuck::cast_slice(&tap));
                Some(k)
            }
        };
        Ok(())
    }

    /// The live probe's samples since the last call, oldest first, at most
    /// the latest 65536.
    pub fn take_probe_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.probe_samples)
    }

    /// Zeroes every DFT, flux and far-field accumulator, as
    /// `SimTm::restart_spectra`. Samples of batches already in flight still
    /// land afterwards, a few steps from before the edit; the browser calls
    /// this after live edits, where the field itself takes far longer than
    /// that to settle.
    pub fn restart_spectra(&mut self) {
        for (_, m) in &mut self.monitors {
            match m {
                Mon::Point(_) => {}
                Mon::Dft(d) => d.restart(),
                Mon::Flux(line) => line.restart(),
                Mon::Ntff(ntff) => ntff.restart(),
            }
        }
    }

    /// The far fields of every contour as accumulated so far, in scene
    /// order (`Ntff::far_field`), without waiting for batches in flight.
    pub fn far_fields(&self) -> Vec<Vec<(f64, f64)>> {
        self.monitors
            .iter()
            .filter_map(|(_, m)| match m {
                Mon::Ntff(ntff) => Some(ntff.far_field()),
                _ => None,
            })
            .collect()
    }

    /// The Ez buffer, for the render pass.
    pub(crate) fn ez_buffer(&self) -> &wgpu::Buffer {
        &self.fields[EZ]
    }

    /// The material id buffer, for the render pass.
    pub(crate) fn ids_buffer(&self) -> &wgpu::Buffer {
        &self.ids
    }

    /// Encodes and submits `steps` (at most `BATCH`) steps.
    fn run_batch(&mut self, steps: u64) {
        let mut slots = vec![0u8; (steps * PARAMS_STRIDE) as usize];
        for b in 0..steps {
            let p = Params {
                step: b as u32,
                ..self.base
            };
            let at = (b * PARAMS_STRIDE) as usize;
            slots[at..at + std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        let gpu = self.gpu;
        gpu.queue.write_buffer(&self.params, 0, &slots);
        // The auxiliary grid runs ahead of the GPU by one batch: for each
        // step, its E^n for the H corrections, then its step, then its
        // H^(n+1/2) for the E corrections, as in `SimTm::step`.
        // TEz advances the auxiliary grid first and reads both from it then,
        // as `SimTe::step` does.
        let mode = self.mode;
        if let Some(t) = &mut self.tfsf {
            let mut inc_e = Vec::with_capacity(steps as usize * self.base.n_inc as usize);
            let mut inc_h = Vec::with_capacity(steps as usize * 2);
            for _ in 0..steps {
                if mode == Mode::Te {
                    t.advance();
                }
                inc_e.extend(t.incident_e().iter().map(|&v| v as f32));
                if mode == Mode::Tm {
                    t.advance();
                }
                inc_h.extend(t.incident_h().iter().map(|&v| v as f32));
            }
            gpu.queue
                .write_buffer(&self.inc_e, 0, bytemuck::cast_slice(&inc_e));
            gpu.queue
                .write_buffer(&self.inc_h, 0, bytemuck::cast_slice(&inc_h));
        }
        // Step b of the batch takes E to n + b + 1, with TMz sources at
        // t = (n + b + 1) S and TEz sources on Hz at (n + b + 1/2) S.
        if !self.sources.is_empty() {
            let s = self.courant;
            let mut values = Vec::with_capacity(steps as usize * self.sources.len());
            for b in 0..steps {
                let t = match mode {
                    Mode::Tm => (self.n + b + 1) as f64,
                    Mode::Te => (self.n + b) as f64 + 0.5,
                } * s;
                values.extend(self.sources.iter().map(|src| src.value_at(t) as f32));
            }
            gpu.queue
                .write_buffer(&self.src_values, 0, bytemuck::cast_slice(&values));
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("steps"),
                timestamp_writes: None,
            });
            for b in 0..steps {
                let offset = [(b * PARAMS_STRIDE) as u32];
                let extra = self.intensity_on.then_some(&self.intensity);
                for stage in self.stages.iter().chain(extra) {
                    pass.set_pipeline(&stage.pass.pipeline);
                    pass.set_bind_group(0, &self.params_group, &offset);
                    pass.set_bind_group(1, &stage.group, &[]);
                    pass.dispatch_workgroups(stage.count.0, stage.count.1, 1);
                }
            }
        }
        let taps = (self.n_taps > 0).then(|| {
            let size = steps * self.n_taps as u64 * 4;
            let buf = gpu.staging(size);
            encoder.copy_buffer_to_buffer(&self.gathered, 0, &buf, 0, size);
            buf
        });
        gpu.queue.submit([encoder.finish()]);
        if let Some(buf) = taps {
            let mapped = Mapped::read(&buf);
            self.tap_queue.push_back(QueuedTaps {
                first: self.n,
                buf,
                mapped,
            });
        }
        self.n += steps;
        if let Err(e) = self.drain_taps() {
            self.pending.get_or_insert(e);
        }
    }

    /// Feeds every batch whose taps have reached the CPU to the monitors,
    /// oldest first, stopping at the first still in flight. Never blocks.
    /// The browser calls it each frame; stepping calls it too.
    pub fn drain_taps(&mut self) -> Result<()> {
        // Drives the map callbacks natively; a no-op on WebGPU.
        self.gpu
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(|e| FdtdError::Gpu(e.to_string()))?;
        while let Some(state) = self.tap_queue.front().map(|q| q.mapped.state()) {
            match state {
                None => break,
                Some(Err(e)) => {
                    self.tap_queue.clear();
                    return Err(FdtdError::Gpu(e));
                }
                Some(Ok(())) => {
                    if let Some(q) = self.tap_queue.pop_front() {
                        let bytes = q
                            .buf
                            .get_mapped_range(..)
                            .map_err(|e| FdtdError::Gpu(e.to_string()))?
                            .to_vec();
                        q.buf.unmap();
                        self.take_taps(q.first, &bytes);
                    }
                }
            }
        }
        Ok(())
    }

    /// Waits for every queued batch and drains it. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    fn settle_taps(&mut self) -> Result<()> {
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| FdtdError::Gpu(e.to_string()))?;
        self.drain_taps()?;
        if self.tap_queue.is_empty() {
            Ok(())
        } else {
            Err(FdtdError::Gpu("taps still in flight after a wait".into()))
        }
    }

    /// Reads the gathered taps of a batch that started at step `first` and
    /// feeds every monitor, step by step, as `SimTm::step` does: E samples
    /// at t = n S, flux H at (n - 1/2) S (inside `record_taps`).
    fn take_taps(&mut self, first: u64, bytes: &[u8]) {
        let values: Vec<f64> = bytemuck::cast_slice::<u8, f32>(bytes)
            .iter()
            .map(|&v| v as f64)
            .collect();
        let s = self.courant;
        // The node field is E at n S in TMz and Hz at (n - 1/2) S in TEz.
        let lag = if self.mode == Mode::Te { 0.5 } else { 0.0 };
        for (b, row) in values.chunks_exact(self.n_taps).enumerate() {
            let n = first + b as u64 + 1;
            for (at, m) in &mut self.monitors {
                match m {
                    Mon::Point(h) => h.push(row[*at]),
                    Mon::Dft(d) => d.add(&row[*at..*at + 1], (n as f64 - lag) * s, s),
                    Mon::Flux(line) => {
                        let len = 3 * (line.to - line.from + 1);
                        line.record_taps(&row[*at..*at + len], n, s);
                    }
                    Mon::Ntff(ntff) => {
                        let len = ntff.taps_len();
                        ntff.record_taps(&row[*at..*at + len], n, s);
                    }
                }
            }
            if self.probe.is_some() {
                self.probe_samples.push(row[self.n_taps - 1] as f32);
            }
        }
        let excess = self.probe_samples.len().saturating_sub(MAX_PROBE_SAMPLES);
        self.probe_samples.drain(..excess);
    }

    /// Advances `steps` full steps.
    pub fn step_n(&mut self, steps: u64) {
        let mut left = steps;
        while left > 0 {
            let b = left.min(BATCH);
            self.run_batch(b);
            left -= b;
        }
    }

    /// One field read back, widened to f64, `nx * ny`, index `j * nx + i`.
    pub async fn read_async(&mut self, which: Component) -> Result<Vec<f64>> {
        if let Some(e) = self.pending.take() {
            return Err(e);
        }
        let buf = &self.fields[match (self.mode, which) {
            (Mode::Tm, Component::Ez) | (Mode::Te, Component::Hz) => EZ,
            (Mode::Tm, Component::Hx) | (Mode::Te, Component::Ex) => HX,
            (Mode::Tm, Component::Hy) | (Mode::Te, Component::Ey) => HY,
            (_, component) => return Err(FdtdError::FieldNotInMode { component }),
        }];
        let bytes = self.gpu.read_async(buf, 0, buf.size()).await?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| v as f64)
            .collect())
    }

    /// `read_async`, blocking. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&mut self, which: Component) -> Result<Vec<f64>> {
        pollster::block_on(self.read_async(which))
    }

    /// Full steps taken so far. E is at time n S, H at (n - 1/2) S.
    pub fn time_step(&self) -> u64 {
        self.n
    }

    /// Blocks until every submitted step has finished on the device, then
    /// reports any device error. For timing. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sync(&mut self) -> Result<()> {
        if let Some(e) = self.pending.take() {
            return Err(e);
        }
        self.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| FdtdError::Gpu(e.to_string()))?;
        self.gpu.check()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Kernel for GpuSim {
    /// Uses the shared device, `Gpu::shared()`.
    fn from_scene(scene: &Scene) -> Result<Self> {
        GpuSim::new(Gpu::shared()?, scene)
    }

    fn step(&mut self, n: u64) {
        self.step_n(n);
    }

    fn time_step(&self) -> u64 {
        self.n
    }

    fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
        self.read(which)
    }

    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
        if let Some(e) = self.pending.take() {
            return Err(e);
        }
        self.settle_taps()?;
        Ok(self
            .monitors
            .iter()
            .filter_map(|(_, m)| match m {
                Mon::Point(h) => Some(h.clone()),
                _ => None,
            })
            .collect())
    }

    fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        GpuSim::set_materials(self, scene, bounds)
    }

    fn set_sources(&mut self, scene: &Scene) -> Result<()> {
        GpuSim::set_sources(self, scene)
    }

    fn mode(&self) -> Mode {
        self.mode
    }

    /// The same shapes as `SimTm`: point monitors skipped, one entry per
    /// frequency.
    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
        if let Some(e) = self.pending.take() {
            return Err(e);
        }
        self.settle_taps()?;
        Ok(self
            .monitors
            .iter()
            .filter_map(|(_, m)| match m {
                Mon::Point(_) => None,
                Mon::Dft(d) => Some(Spectrum::Field(
                    (0..d.freqs().len()).map(|f| d.get(0, f)).collect(),
                )),
                Mon::Flux(line) => Some(Spectrum::Flux(line.flux())),
                Mon::Ntff(ntff) => Some(Spectrum::FarField(ntff.far_field())),
            })
            .collect())
    }
}
