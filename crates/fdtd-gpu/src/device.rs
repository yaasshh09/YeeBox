//! Adapter and device setup, and blocking buffer readback for native use.

use fdtd_core::{FdtdError, Result};
use std::sync::{Arc, Mutex, OnceLock};

/// A device and its queue, plus the first error the device reported since
/// the last check. Errors from `wgpu` arrive through a callback, not as
/// return values, so they are stored here and surfaced by `check`.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    error: Arc<Mutex<Option<String>>>,
}

fn gpu_err(e: impl std::fmt::Display) -> FdtdError {
    FdtdError::Gpu(e.to_string())
}

impl Gpu {
    /// Picks the high-performance adapter, falling back to a software one,
    /// and opens a device with the default WebGPU limits, so whatever runs
    /// here also fits a browser.
    pub fn new() -> Result<Gpu> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = [false, true]
            .into_iter()
            .find_map(|force_fallback_adapter| {
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter,
                    compatible_surface: None,
                    ..Default::default()
                }))
                .ok()
            })
            .ok_or_else(|| gpu_err("no adapter"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fdtd"),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(gpu_err)?;
        let error = Arc::new(Mutex::new(None));
        let sink = error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut slot = sink.lock().unwrap_or_else(|p| p.into_inner());
            slot.get_or_insert_with(|| e.to_string());
        }));
        Ok(Gpu {
            device,
            queue,
            adapter: adapter.get_info(),
            error,
        })
    }

    /// One device for the whole process, made on first use. Every caller
    /// gets the same error if no adapter exists.
    pub fn shared() -> Result<&'static Gpu> {
        static GPU: OnceLock<std::result::Result<Gpu, String>> = OnceLock::new();
        GPU.get_or_init(|| Gpu::new().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| FdtdError::Gpu(e.clone()))
    }

    /// The adapter in one line, for logs: name and backend.
    pub fn describe(&self) -> String {
        format!("{} ({:?})", self.adapter.name, self.adapter.backend)
    }

    /// Returns, and clears, the first error the device reported.
    pub fn check(&self) -> Result<()> {
        let mut slot = self.error.lock().unwrap_or_else(|p| p.into_inner());
        match slot.take() {
            Some(e) => Err(FdtdError::Gpu(e)),
            None => Ok(()),
        }
    }

    /// Copies `size` bytes of `src` from `offset` into a staging buffer and
    /// blocks until they are on the CPU. Native only; the browser phase
    /// needs an async version. Offsets and sizes are multiples of 4.
    pub fn read(&self, src: &wgpu::Buffer, offset: u64, size: u64) -> Result<Vec<u8>> {
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(gpu_err)?;
        rx.recv().map_err(gpu_err)?.map_err(gpu_err)?;
        let bytes = staging.get_mapped_range(..).map_err(gpu_err)?.to_vec();
        staging.unmap();
        self.check()?;
        Ok(bytes)
    }

    /// `read` for a whole f32 buffer, widened to f64.
    pub fn read_f32(&self, src: &wgpu::Buffer) -> Result<Vec<f64>> {
        let bytes = self.read(src, 0, src.size())?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| v as f64)
            .collect())
    }
}
