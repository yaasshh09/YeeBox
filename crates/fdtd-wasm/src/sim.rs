//! The simulation as JavaScript sees it: one scene session, one viewport,
//! and a backend. `createSim` builds the GPU backend on a canvas (main
//! thread); `createCpuSim` builds the f64 CPU fallback for a Web Worker,
//! which draws by `pixels()` into its own canvas. Every other method is the
//! same for both, so the page has one code path. Browser only.

use crate::cpu::CpuBackend;
use crate::session::Session;
use crate::{ErrorInfo, JsResult, js_err};
use fdtd_core::FdtdError;
use fdtd_core::scene::{Scene, Shape, SlitPair, SourcePatch, View, ViewField};
use fdtd_gpu::render::{Frame, Renderer, Viewport, intensity_tau};
use fdtd_gpu::{Gpu, GpuSim};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

// One Sim per page, so the size gap between the variants costs nothing.
#[allow(clippy::large_enum_variant)]
enum Backend {
    Gpu {
        gpu: &'static Gpu,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        sim: GpuSim,
        renderer: Renderer,
    },
    Cpu(CpuBackend),
}

#[wasm_bindgen]
pub struct Sim {
    session: Session,
    viewport: Viewport,
    backend: Backend,
}

/// Parses JSON into `T`, as a coded `json` error.
fn parse<T: serde::de::DeserializeOwned>(json: &str) -> JsResult<T> {
    serde_json::from_str(json).map_err(|e| js_err(FdtdError::Json(e.to_string())))
}

/// The scene's view, or the default: Ez at gain 1.
fn view_of(scene: &Scene) -> View {
    scene.view.unwrap_or(View {
        field: ViewField::Ez,
        gain: 1.0,
        auto_gain: false,
    })
}

/// Builds the GPU simulation on `canvas`. Rejects with code `no_webgpu`
/// when the browser has no WebGPU or no adapter that can draw to the
/// canvas; the page then starts the CPU fallback in a worker.
#[wasm_bindgen(js_name = createSim)]
pub async fn create_sim(canvas: HtmlCanvasElement, scene_json: String) -> JsResult<Sim> {
    let session = Session::load(&scene_json).map_err(js_err)?;
    let (width, height) = (canvas.width().max(1), canvas.height().max(1));
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::BROWSER_WEBGPU;
    let instance = wgpu::Instance::new(desc);
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| ErrorInfo::no_webgpu(e.to_string()))?;
    let gpu = Gpu::request(&instance, Some(&surface))
        .await
        .map_err(|e| ErrorInfo::no_webgpu(e.to_string()))?;
    // ponytail: one device per page, leaked so the kernel can hold it for
    // 'static; a device loss leaks one more. Upgrade to Arc if that matters.
    let gpu: &'static Gpu = Box::leak(Box::new(gpu));
    let caps = surface.get_capabilities(&gpu.handle);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .ok_or_else(|| ErrorInfo::no_webgpu("the canvas offers no non-sRGB format"))?;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        color_space: Default::default(),
        width,
        height,
        desired_maximum_frame_latency: 2,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };
    surface.configure(&gpu.device, &config);
    let scene = &session.scene;
    let sim = GpuSim::new(gpu, scene).map_err(js_err)?;
    let renderer = Renderer::new(gpu, &sim, scene, format);
    let viewport = Viewport::fit(scene.grid.nx, scene.grid.ny, width as f64, height as f64);
    let mut out = Sim {
        session,
        viewport,
        backend: Backend::Gpu {
            gpu,
            surface,
            config,
            sim,
            renderer,
        },
    };
    out.apply_view()?;
    Ok(out)
}

/// Builds the CPU fallback for a `width` x `height` pixel canvas.
#[wasm_bindgen(js_name = createCpuSim)]
pub fn create_cpu_sim(scene_json: &str, width: u32, height: u32) -> JsResult<Sim> {
    let session = Session::load(scene_json).map_err(js_err)?;
    let scene = &session.scene;
    let cpu = CpuBackend::new(scene).map_err(js_err)?;
    let viewport = Viewport::fit(
        scene.grid.nx,
        scene.grid.ny,
        width.max(1) as f64,
        height.max(1) as f64,
    );
    let mut out = Sim {
        session,
        viewport,
        backend: Backend::Cpu(cpu),
    };
    out.apply_view()?;
    Ok(out)
}

impl Sim {
    fn time_step(&self) -> u64 {
        match &self.backend {
            Backend::Gpu { sim, .. } => sim.time_step(),
            Backend::Cpu(cpu) => cpu.time_step(),
        }
    }

    /// Normalized time of the E field now, for edits that start or retune.
    fn t_now(&self) -> f64 {
        self.time_step() as f64 * self.session.scene.grid.courant
    }

    fn sources_changed(&mut self) -> JsResult<()> {
        let scene = &self.session.scene;
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.set_sources(scene),
            Backend::Cpu(cpu) => cpu.set_sources(scene),
        }
        .map_err(js_err)
    }

    fn materials_changed(&mut self, bounds: [usize; 4]) -> JsResult<()> {
        let scene = &self.session.scene;
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.set_materials(scene, bounds),
            Backend::Cpu(cpu) => cpu.set_materials(scene, bounds),
        }
        .map_err(js_err)
    }

    /// Turns the intensity mean on or off to match the scene's view.
    fn apply_view(&mut self) -> JsResult<()> {
        let scene = &self.session.scene;
        let tau = (view_of(scene).field == ViewField::Intensity).then(|| intensity_tau(scene));
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.set_intensity(tau),
            Backend::Cpu(cpu) => cpu.set_intensity(tau),
        }
        .map_err(js_err)
    }

    fn frame(&self) -> Frame {
        let view = view_of(&self.session.scene);
        Frame {
            viewport: self.viewport,
            gain: view.gain,
            field: view.field,
        }
    }
}

#[wasm_bindgen]
impl Sim {
    /// `"gpu"` or `"cpu"`.
    pub fn kernel(&self) -> String {
        match self.backend {
            Backend::Gpu { .. } => "gpu",
            Backend::Cpu(_) => "cpu",
        }
        .into()
    }

    /// The adapter in one line, for the status strip.
    pub fn adapter(&self) -> String {
        match &self.backend {
            Backend::Gpu { gpu, .. } => gpu.describe(),
            Backend::Cpu(_) => "f64 reference kernel, one thread".into(),
        }
    }

    /// Advances `n` steps (on the GPU submitted, not waited on).
    pub fn step(&mut self, n: u32) {
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.step_n(n as u64),
            Backend::Cpu(cpu) => cpu.step(n as u64),
        }
    }

    /// Feeds finished GPU batches to the probe and reports the first device
    /// error since the last call. Call once per frame.
    pub fn check(&mut self) -> JsResult<()> {
        if let Backend::Gpu { gpu, sim, .. } = &mut self.backend {
            sim.drain_taps().map_err(js_err)?;
            gpu.check().map_err(js_err)?;
        }
        Ok(())
    }

    /// Draws the current field to the canvas (GPU backend; the CPU backend
    /// draws through `pixels`).
    pub fn render(&mut self) -> JsResult<()> {
        let frame_desc = self.frame();
        let Backend::Gpu {
            gpu,
            surface,
            config,
            renderer,
            ..
        } = &mut self.backend
        else {
            return Ok(());
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&gpu.device, config);
                return Ok(());
            }
            _ => return Ok(()),
        };
        let target = frame.texture.create_view(&Default::default());
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        renderer.draw(gpu, &mut enc, &target, &frame_desc);
        gpu.queue.submit([enc.finish()]);
        gpu.queue.present(frame);
        Ok(())
    }

    /// RGBA8 pixels of the current frame for the CPU backend, row-major
    /// from the top left; empty for the GPU backend.
    pub fn pixels(&self) -> Vec<u8> {
        match &self.backend {
            Backend::Cpu(cpu) => cpu.pixels(&self.frame()),
            Backend::Gpu { .. } => Vec::new(),
        }
    }

    /// The canvas changed size in device pixels: reconfigure and refit.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if let Backend::Gpu {
            gpu,
            surface,
            config,
            ..
        } = &mut self.backend
        {
            config.width = w;
            config.height = h;
            surface.configure(&gpu.device, config);
        }
        let g = self.session.scene.grid;
        self.viewport = Viewport::fit(g.nx, g.ny, w as f64, h as f64);
    }

    /// Sets the view (`{"field": "ez" | "intensity", "gain": g,
    /// "auto_gain": b}`); it travels with the scene in shared links.
    #[wasm_bindgen(js_name = setView)]
    pub fn set_view(&mut self, json: &str) -> JsResult<()> {
        let view: View = parse(json)?;
        let mut next = self.session.scene.clone();
        next.view = Some(view);
        next.validate().map_err(js_err)?;
        let changed = view_of(&self.session.scene).field != view.field;
        self.session.scene.view = Some(view);
        if changed {
            self.apply_view()?;
        }
        Ok(())
    }

    /// The status strip numbers as JSON (`fdtd_core::scene::Status`).
    pub fn status(&self) -> JsResult<String> {
        let st = self
            .session
            .scene
            .status(self.time_step())
            .map_err(js_err)?;
        serde_json::to_string(&st).map_err(|e| js_err(FdtdError::Json(e.to_string())))
    }

    /// The current scene, edits included, as JSON.
    pub fn scene(&self) -> JsResult<String> {
        self.session.scene.to_json().map_err(js_err)
    }

    /// Switches the polarization, `"tm"` or `"te"`, keeping the scene, and
    /// rebuilds the kernel: the fields start again from zero. A scene the
    /// other mode cannot run is refused and nothing changes.
    #[wasm_bindgen(js_name = setMode)]
    pub fn set_mode(&mut self, mode: &str) -> JsResult<()> {
        let mode = match mode {
            "tm" => fdtd_core::scene::Mode::Tm,
            "te" => fdtd_core::scene::Mode::Te,
            _ => {
                return Err(js_err(FdtdError::InvalidParameter {
                    name: "mode (tm or te)",
                    value: f64::NAN,
                }));
            }
        };
        self.session.set_mode(mode).map_err(js_err)?;
        self.reset_fields()
    }

    /// Zeroes the fields and time, keeping the scene.
    #[wasm_bindgen(js_name = resetFields)]
    pub fn reset_fields(&mut self) -> JsResult<()> {
        let scene = &self.session.scene;
        match &mut self.backend {
            Backend::Gpu {
                gpu,
                config,
                sim,
                renderer,
                ..
            } => {
                *sim = GpuSim::new(gpu, scene).map_err(js_err)?;
                *renderer = Renderer::new(gpu, sim, scene, config.format);
            }
            Backend::Cpu(cpu) => *cpu = CpuBackend::new(scene).map_err(js_err)?,
        }
        self.apply_view()
    }

    /// The cell position `[x, y]` under canvas pixel (px, py).
    #[wasm_bindgen(js_name = cellAt)]
    pub fn cell_at(&self, px: f64, py: f64) -> Vec<f64> {
        let (x, y) = self.viewport.cell_at(px, py);
        vec![x, y]
    }

    /// Zooms by `factor` about canvas pixel (px, py).
    #[wasm_bindgen(js_name = zoomAt)]
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        self.viewport.zoom_at(px, py, factor);
    }

    /// Moves the picture by (dx, dy) canvas pixels.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        self.viewport.pan(dx, dy);
    }

    /// One brush stroke from (x0, y0) to (x1, y1), cells, with palette
    /// entry `palette` (0 erases). Returns `{"painted": n, "skipped": m}`.
    #[wasm_bindgen(js_name = paintStroke)]
    pub fn paint_stroke(
        &mut self,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        radius: f64,
        palette: usize,
    ) -> JsResult<String> {
        let out = self
            .session
            .paint((x0, y0), (x1, y1), radius, palette)
            .map_err(js_err)?;
        if let Some(b) = out.bounds {
            self.materials_changed(b)?;
        }
        Ok(format!(
            r#"{{"painted":{},"skipped":{}}}"#,
            out.painted, out.skipped
        ))
    }

    /// Adds a rect, circle or polygon (scene JSON form) whose `material`
    /// is a palette index.
    #[wasm_bindgen(js_name = addShape)]
    pub fn add_shape(&mut self, json: &str) -> JsResult<()> {
        let shape: Shape = parse(json)?;
        let b = self.session.add_shape(shape).map_err(js_err)?;
        self.materials_changed(b)
    }

    /// Adds a slit pair (`fdtd_core::scene::SlitPair` JSON) whose
    /// `material` is a palette index.
    #[wasm_bindgen(js_name = addSlitPair)]
    pub fn add_slit_pair(&mut self, json: &str) -> JsResult<()> {
        let spec: SlitPair = parse(json)?;
        let b = self.session.add_slit_pair(&spec).map_err(js_err)?;
        self.materials_changed(b)
    }

    /// A shape drawn by dragging from (x0, y0) to (x1, y1), cells: `kind`
    /// is `"rect"` (two corners) or `"circle"` (center, then a rim point).
    #[wasm_bindgen(js_name = addDragShape)]
    pub fn add_drag_shape(
        &mut self,
        kind: &str,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        palette: usize,
    ) -> JsResult<()> {
        let b = self
            .session
            .drag_shape(kind, (x0, y0), (x1, y1), palette)
            .map_err(js_err)?;
        self.materials_changed(b)
    }

    /// Sets every source's vacuum wavelength in cells, smoothly. Returns
    /// how many were retuned (a plane wave is not; it is part of the build).
    #[wasm_bindgen(js_name = retuneSources)]
    pub fn retune_sources(&mut self, wavelength_cells: f64) -> JsResult<u32> {
        let t = self.t_now();
        let n = self
            .session
            .retune_sources(wavelength_cells, t)
            .map_err(js_err)?;
        self.sources_changed()?;
        Ok(n as u32)
    }

    /// Removes every shape and all paint; sources stay.
    #[wasm_bindgen(js_name = clearScene)]
    pub fn clear_scene(&mut self) -> JsResult<()> {
        let b = self.session.clear();
        self.materials_changed(b)
    }

    /// Adds a CW point source at the node nearest cell (x, y), turning on
    /// now over 3 periods.
    #[wasm_bindgen(js_name = addSource)]
    pub fn add_source(&mut self, x: f64, y: f64, wavelength_cells: f64) -> JsResult<()> {
        let t = self.t_now();
        self.session
            .add_source(x, y, wavelength_cells, t)
            .map_err(js_err)?;
        self.sources_changed()
    }

    #[wasm_bindgen(js_name = removeSource)]
    pub fn remove_source(&mut self, index: usize) -> JsResult<()> {
        self.session.remove_source(index).map_err(js_err)?;
        self.sources_changed()
    }

    /// Edits a source with `fdtd_core::scene::SourcePatch` JSON (position,
    /// wavelength in cells, amplitude, phase), keeping a CW drive smooth.
    #[wasm_bindgen(js_name = editSource)]
    pub fn edit_source(&mut self, index: usize, patch_json: &str) -> JsResult<()> {
        let patch: SourcePatch = parse(patch_json)?;
        let t = self.t_now();
        self.session.edit_source(index, &patch, t).map_err(js_err)?;
        self.sources_changed()
    }

    /// The index of the source nearest cell (x, y) within `within` cells.
    #[wasm_bindgen(js_name = sourceNear)]
    pub fn source_near(&self, x: f64, y: f64, within: f64) -> Option<u32> {
        self.session.source_near(x, y, within).map(|k| k as u32)
    }

    /// Puts the oscilloscope probe on the node nearest cell (x, y).
    #[wasm_bindgen(js_name = setProbe)]
    pub fn set_probe(&mut self, x: f64, y: f64) -> JsResult<()> {
        if !(x.is_finite() && y.is_finite() && x >= -0.5 && y >= -0.5) {
            return Err(js_err(FdtdError::OutOfDomain {
                what: "probe",
                i: 0,
                j: 0,
            }));
        }
        let node = Some((x.round() as usize, y.round() as usize));
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.set_probe(node),
            Backend::Cpu(cpu) => cpu.set_probe(node),
        }
        .map_err(js_err)
    }

    #[wasm_bindgen(js_name = clearProbe)]
    pub fn clear_probe(&mut self) -> JsResult<()> {
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.set_probe(None),
            Backend::Cpu(cpu) => cpu.set_probe(None),
        }
        .map_err(js_err)
    }

    /// The probe's Ez samples, one per step, since the last call.
    #[wasm_bindgen(js_name = takeProbe)]
    pub fn take_probe(&mut self) -> Vec<f32> {
        match &mut self.backend {
            Backend::Gpu { sim, .. } => sim.take_probe_samples(),
            Backend::Cpu(cpu) => cpu.take_probe(),
        }
    }
}
