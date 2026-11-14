//! `GpuSim`: the TMz scene on the GPU, stepped in batches.
//!
//! A batch of up to `BATCH` steps is one command buffer and one compute
//! pass, with the passes of every step in the order of `SimTm::step`. Each
//! step reads its slot of the `Params` uniform through a dynamic offset.

use crate::buffers::{Layout, PARAMS_STRIDE, Params};
use crate::device::Gpu;
use crate::pipeline::{self, Access, Pass};
use fdtd_core::grid::Component;
use fdtd_core::kernel::{Kernel, Spectrum};
use fdtd_core::scene::Scene;
use fdtd_core::{FdtdError, Result};
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

/// The f32 GPU kernel for one scene.
pub struct GpuSim {
    gpu: &'static Gpu,
    nx: usize,
    ny: usize,
    n: u64,
    wg: u32,
    base: Params,
    /// Cb per node in f64, zero on walls and PEC nodes.
    cb: Vec<f64>,
    fields: Vec<wgpu::Buffer>,
    params: wgpu::Buffer,
    params_group: wgpu::BindGroup,
    h: (Pass, wgpu::BindGroup),
    e: (Pass, wgpu::BindGroup),
    /// The first error seen while stepping, returned by the next read.
    pending: Option<FdtdError>,
}

fn not_yet(what: &str) -> FdtdError {
    FdtdError::Gpu(format!("{what} not supported by the gpu kernel yet"))
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
        if !scene.sources.is_empty() {
            return Err(not_yet("sources"));
        }
        if !scene.monitors.is_empty() {
            return Err(not_yet("monitors"));
        }
        if scene.plane_wave.is_some() {
            return Err(not_yet("plane waves"));
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
        let storage = |label: &str, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
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
        gpu.check()?;
        Ok(GpuSim {
            gpu,
            nx,
            ny,
            n: 0,
            wg,
            base: Params {
                nx: nx as u32,
                ny: ny as u32,
                periodic_y: scene.periodic_y as u32,
                step: 0,
                s: scene.grid.courant as f32,
                n_src: 0,
                n_taps: 0,
                n_hgroups: 0,
                n_egroups: 0,
                n_inc: 0,
                pad: [0; 2],
            },
            cb: layout.cb_nodes(),
            fields,
            params,
            params_group,
            h: (h, h_group),
            e: (e, e_group),
            pending: None,
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
        let (gx, gy) = (
            self.nx.div_ceil(self.wg as usize) as u32,
            self.ny.div_ceil(self.wg as usize) as u32,
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("steps"),
                timestamp_writes: None,
            });
            for b in 0..steps {
                let offset = [(b * PARAMS_STRIDE) as u32];
                for (stage, group) in [&self.h, &self.e] {
                    pass.set_pipeline(&stage.pipeline);
                    pass.set_bind_group(0, &self.params_group, &offset);
                    pass.set_bind_group(1, group, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        }
        gpu.queue.submit([encoder.finish()]);
        self.n += steps;
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
    pub fn read(&mut self, which: Component) -> Result<Vec<f64>> {
        if let Some(e) = self.pending.take() {
            return Err(e);
        }
        let k = match which {
            Component::Ez => EZ,
            Component::Hx => HX,
            Component::Hy => HY,
        };
        self.gpu.read_f32(&self.fields[k])
    }

    /// Full steps taken so far. E is at time n S, H at (n - 1/2) S.
    pub fn time_step(&self) -> u64 {
        self.n
    }
}

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
        Ok(Vec::new())
    }

    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
        Ok(Vec::new())
    }
}
