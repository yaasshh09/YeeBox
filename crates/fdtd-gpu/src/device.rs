//! Adapter and device setup, and blocking buffer readback for native use.

use fdtd_core::{FdtdError, Result};
use std::future::Future;
use std::pin::Pin;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// A device and its queue, plus the first error the device reported since
/// the last check. Errors from `wgpu` arrive through a callback, not as
/// return values, so they are stored here and surfaced by `check`.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    /// The adapter itself, for surface capabilities in the browser.
    pub handle: wgpu::Adapter,
    error: Arc<Mutex<Option<String>>>,
}

fn gpu_err(e: impl std::fmt::Display) -> FdtdError {
    FdtdError::Gpu(e.to_string())
}

impl Gpu {
    /// Picks the high-performance adapter of `instance` that can present to
    /// `surface` (if given), falling back to a software one, and opens a
    /// device with the default WebGPU limits, so whatever runs natively also
    /// fits a browser. The browser awaits this; native code calls `new`.
    pub async fn request(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Gpu> {
        let mut adapter = None;
        for force_fallback_adapter in [false, true] {
            let options = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter,
                compatible_surface: surface,
                ..Default::default()
            };
            if let Ok(a) = instance.request_adapter(&options).await {
                adapter = Some(a);
                break;
            }
        }
        let adapter = adapter.ok_or_else(|| gpu_err("no adapter"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fdtd"),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
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
            handle: adapter,
            error,
        })
    }

    /// `request` without a surface, blocking. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Gpu> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        pollster::block_on(Gpu::request(&instance, None))
    }

    /// One device for the whole process, made on first use. Every caller
    /// gets the same error if no adapter exists. Native only: a browser
    /// device is neither `Send` nor `Sync`, and the page makes its own.
    #[cfg(not(target_arch = "wasm32"))]
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

    /// A buffer the CPU can map, `size` bytes, filled by copies.
    pub fn staging(&self, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Copies `size` bytes of `src` from `offset` into a staging buffer and
    /// awaits them on the CPU. On WebGPU the map resolves from the event
    /// loop; natively the poll below drives it. Offsets and sizes are
    /// multiples of 4.
    pub async fn read_async(&self, src: &wgpu::Buffer, offset: u64, size: u64) -> Result<Vec<u8>> {
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = self.staging(size);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let mapped = Mapped::read(&staging);
        // A no-op on WebGPU.
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(gpu_err)?;
        mapped.await.map_err(FdtdError::Gpu)?;
        let bytes = staging.get_mapped_range(..).map_err(gpu_err)?.to_vec();
        staging.unmap();
        self.check()?;
        Ok(bytes)
    }

    /// `read_async`, blocking. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read(&self, src: &wgpu::Buffer, offset: u64, size: u64) -> Result<Vec<u8>> {
        pollster::block_on(self.read_async(src, offset, size))
    }

    /// `read` for a whole f32 buffer, widened to f64. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_f32(&self, src: &wgpu::Buffer) -> Result<Vec<f64>> {
        let bytes = self.read(src, 0, src.size())?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes)
            .iter()
            .map(|&v| v as f64)
            .collect())
    }
}

/// A `map_async` in flight. Await it, or check `state` without waiting (the
/// tap queue does that each frame). The callback fills the slot and wakes
/// whoever awaits.
pub struct Mapped {
    slot: Arc<Mutex<MapSlot>>,
}

#[derive(Default)]
struct MapSlot {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

impl Mapped {
    /// Starts mapping all of `buf` for reading.
    pub fn read(buf: &wgpu::Buffer) -> Mapped {
        let slot = Arc::new(Mutex::new(MapSlot::default()));
        let sink = slot.clone();
        buf.map_async(wgpu::MapMode::Read, .., move |r| {
            let mut s = sink.lock().unwrap_or_else(|p| p.into_inner());
            s.result = Some(r.map_err(|e| e.to_string()));
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        });
        Mapped { slot }
    }

    /// `None` while the map is pending, then its outcome.
    pub fn state(&self) -> Option<std::result::Result<(), String>> {
        self.slot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .result
            .clone()
    }
}

impl Future for Mapped {
    type Output = std::result::Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        match s.result.clone() {
            Some(r) => Poll::Ready(r),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
