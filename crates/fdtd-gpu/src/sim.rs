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
use fdtd_core::scene::{Monitor, Scene, Source};
use fdtd_core::source::SourceKind;
use fdtd_core::tfsf::Tfsf;
use fdtd_core::{FdtdError, Result};
use std::collections::VecDeque;
use wgpu::util::DeviceExt;

/// Most steps encoded into one submission.
pub const BATCH: u64 = 64;

/// Field and psi buffers, in this order.
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
}

/// The f32 GPU kernel for one scene.
pub struct GpuSim {
    gpu: &'static Gpu,
    n: u64,
    /// The Courant number in f64, for source times; `base.s` is its f32 copy.
    courant: f64,
    base: Params,
    /// Cb per node in f64, zero on walls and PEC nodes.
    cb: Vec<f64>,
    /// Material id per node, as the E update reads it.
    ids: wgpu::Buffer,
    /// (Ca, Cb) per material in f64, the wall entry last.
    table: Vec<(f64, f64)>,
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
}

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
        let limits = gpu.device.limits();
        let bytes = (nx * ny * 4) as u64;
        let groups = nx.max(ny).div_ceil(wg as usize) as u32;
        if bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || groups > limits.max_compute_workgroups_per_dimension
        {
            return Err(FdtdError::GridSize { nx, ny });
        }
        let layout = Layout::new(scene);
        let device = &gpu.device;
        // Storage buffers are never empty: a zero-sized binding is invalid,
        // so a scene without sources still gets a 16-byte buffer.
        let storage = |label: &str, contents: &[u8]| storage_buffer(device, label, contents);
        let zeros = vec![0f32; nx * ny];
        let fields: Vec<_> = ["ez", "hx", "hy", "psi_hxy", "psi_hyx", "psi_ezx", "psi_ezy"]
            .iter()
            .map(|name| storage(name, bytemuck::cast_slice(&zeros)))
            .collect();
        let ids = storage("ids", bytemuck::cast_slice(&layout.ids));
        let coef = storage("coef", bytemuck::cast_slice(&layout.coef()));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: BATCH * PARAMS_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_layout = pipeline::params_layout(device);
        let params_group = pipeline::params_group(device, &params_layout, &params);
        use Access::{Read, ReadWrite};
        let h = pipeline::pass(
            device,
            &params_layout,
            "update_h",
            include_str!("../shaders/update_h.wgsl"),
            wg,
            &[Read, ReadWrite, ReadWrite, ReadWrite, ReadWrite, Read],
        );
        let h_group = pipeline::storage_group(
            device,
            &h,
            &[
                &fields[EZ],
                &fields[HX],
                &fields[HY],
                &fields[PSI_HXY],
                &fields[PSI_HYX],
                &coef,
            ],
        );
        let e = pipeline::pass(
            device,
            &params_layout,
            "update_e",
            include_str!("../shaders/update_e.wgsl"),
            wg,
            &[ReadWrite, Read, Read, ReadWrite, ReadWrite, Read, Read],
        );
        let e_group = pipeline::storage_group(
            device,
            &e,
            &[
                &fields[EZ],
                &fields[HX],
                &fields[HY],
                &fields[PSI_EZX],
                &fields[PSI_EZY],
                &ids,
                &coef,
            ],
        );
        let grid = (
            nx.div_ceil(wg as usize) as u32,
            ny.div_ceil(wg as usize) as u32,
        );
        let mut incident = None;
        let mut stages = vec![
            Stage {
                pass: h,
                group: h_group,
                count: grid,
            },
            Stage {
                pass: e,
                group: e_group,
                count: grid,
            },
        ];
        // TF/SF: H corrections right after H, E corrections right after E,
        // from the same lists the CPU applies.
        let tfsf = match &scene.plane_wave {
            Some(pw) => Some(Tfsf::new(pw, scene.materials[0].eps(), scene.grid.courant)?),
            None => None,
        };
        let (mut n_inc, mut n_hgroups, mut n_egroups) = (0, 0, 0);
        if let Some(t) = &tfsf {
            n_inc = t.incident_e().len();
            let hc = t.h_corrections(nx, scene.grid.courant, scene.periodic_y);
            let ec = t.e_corrections(nx, &layout.cb_nodes());
            (n_hgroups, n_egroups) = (hc.len(), ec.len());
            let hc = storage("tfsf h", bytemuck::cast_slice(&pack_corrections(&hc)));
            let ec = storage("tfsf e", bytemuck::cast_slice(&pack_corrections(&ec)));
            let inc_e = storage("inc e", &vec![0u8; BATCH as usize * n_inc * 4]);
            let inc_h = storage("inc h", &vec![0u8; BATCH as usize * 2 * 4]);
            let pass = pipeline::pass(
                device,
                &params_layout,
                "tfsf_h",
                include_str!("../shaders/tfsf_h.wgsl"),
                wg,
                &[ReadWrite, ReadWrite, Read, Read],
            );
            let group =
                pipeline::storage_group(device, &pass, &[&fields[HX], &fields[HY], &hc, &inc_e]);
            stages.insert(
                1,
                Stage {
                    pass,
                    group,
                    count: ((n_hgroups as u32).div_ceil(64), 1),
                },
            );
            let pass = pipeline::pass(
                device,
                &params_layout,
                "tfsf_e",
                include_str!("../shaders/tfsf_e.wgsl"),
                wg,
                &[ReadWrite, Read, Read],
            );
            let group = pipeline::storage_group(device, &pass, &[&fields[EZ], &ec, &inc_h]);
            stages.insert(
                3,
                Stage {
                    pass,
                    group,
                    count: ((n_egroups as u32).div_ceil(64), 1),
                },
            );
            incident = Some((inc_e, inc_h));
        }
        let (inc_e, inc_h) =
            incident.unwrap_or_else(|| (storage("inc e", &[]), storage("inc h", &[])));
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
        // the shader loop is empty.
        let pass = pipeline::pass(
            device,
            &params_layout,
            "sources",
            include_str!("../shaders/sources.wgsl"),
            wg,
            &[ReadWrite, Read, Read],
        );
        let group = pipeline::storage_group(device, &pass, &[&fields[EZ], &src_nodes, &src_values]);
        let src_stage = stages.len();
        stages.push(Stage {
            pass,
            group,
            count: (1, 1),
        });
        let mut taps: Vec<[u32; 2]> = Vec::new();
        let mut monitors = Vec::new();
        for m in &scene.monitors {
            let at = taps.len();
            taps.extend(
                m.taps(nx, ny)?
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
                    } => Mon::Flux(FluxLine::new(normal, at, from, to, freqs)?),
                },
            ));
        }
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
            cb: layout.cb_nodes(),
            ids,
            table: layout.table.clone(),
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
        })
    }

    /// Replaces Ez (one value per node, `j * nx + i`, normalized), forcing
    /// wall and PEC nodes to zero, like `SimTm::set_ez`. H is left alone.
    pub fn set_ez(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.cb.len() {
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
            .zip(&self.cb)
            .map(|(&v, &cb)| if cb == 0.0 { 0.0 } else { v as f32 })
            .collect();
        self.gpu
            .queue
            .write_buffer(&self.fields[EZ], 0, bytemuck::cast_slice(&ez));
        Ok(())
    }

    /// Rewrites the edited rows of the id buffer and the f64 Cb copy. The
    /// write lands between the batches already submitted and the next.
    pub fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        self.key.check(scene)?;
        let Some(edit) = material_edit(scene, bounds)? else {
            return Ok(());
        };
        let nx = self.base.nx as usize;
        let w = edit.i1 - edit.i0 + 1;
        for (r, j) in edit.rows.clone().enumerate() {
            let row = &edit.ids[r * w..(r + 1) * w];
            let k0 = j * nx + edit.i0;
            for (c, &id) in row.iter().enumerate() {
                self.cb[k0 + c] = self.table[id].1;
            }
            let ids: Vec<u32> = row.iter().map(|&id| id as u32).collect();
            self.gpu
                .queue
                .write_buffer(&self.ids, (k0 * 4) as u64, bytemuck::cast_slice(&ids));
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
                q.write_buffer(&self.mean, 0, &vec![0u8; self.cb.len() * 4]);
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
        if let Some(t) = &mut self.tfsf {
            let mut inc_e = Vec::with_capacity(steps as usize * self.base.n_inc as usize);
            let mut inc_h = Vec::with_capacity(steps as usize * 2);
            for _ in 0..steps {
                inc_e.extend(t.incident_e().iter().map(|&v| v as f32));
                t.advance();
                inc_h.extend(t.incident_h().iter().map(|&v| v as f32));
            }
            gpu.queue
                .write_buffer(&self.inc_e, 0, bytemuck::cast_slice(&inc_e));
            gpu.queue
                .write_buffer(&self.inc_h, 0, bytemuck::cast_slice(&inc_h));
        }
        // Step b of the batch takes E to n + b + 1, at t = (n + b + 1) S.
        if !self.sources.is_empty() {
            let s = self.courant;
            let mut values = Vec::with_capacity(steps as usize * self.sources.len());
            for b in 0..steps {
                let t = (self.n + b + 1) as f64 * s;
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
        for (b, row) in values.chunks_exact(self.n_taps).enumerate() {
            let n = first + b as u64 + 1;
            for (at, m) in &mut self.monitors {
                match m {
                    Mon::Point(h) => h.push(row[*at]),
                    Mon::Dft(d) => d.add(&row[*at..*at + 1], n as f64 * s, s),
                    Mon::Flux(line) => {
                        let len = 3 * (line.to - line.from + 1);
                        line.record_taps(&row[*at..*at + len], n, s);
                    }
                }
            }
        }
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
        let buf = &self.fields[match which {
            Component::Ez => EZ,
            Component::Hx => HX,
            Component::Hy => HY,
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
            })
            .collect())
    }
}
