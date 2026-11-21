//! The GPU simulation as JavaScript sees it: one canvas, one scene, the
//! `fdtd-gpu` kernel and renderer, and the viewport. Browser only.

use crate::{ErrorInfo, JsResult, js_err};
use fdtd_core::scene::{Scene, View, ViewField};
use fdtd_gpu::render::{Frame, Renderer, Viewport, intensity_tau};
use fdtd_gpu::{Gpu, GpuSim};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen]
pub struct Sim {
    gpu: &'static Gpu,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    scene: Scene,
    sim: GpuSim,
    renderer: Renderer,
    viewport: Viewport,
}

/// Parses a scene, merges the paint palette into its table, and checks it.
pub(crate) fn load_scene(json: &str) -> JsResult<Scene> {
    let mut scene = Scene::from_json(json).map_err(js_err)?;
    scene.with_palette();
    scene.validate().map_err(js_err)?;
    Ok(scene)
}

/// The scene's view, or the default: Ez at gain 1.
pub(crate) fn view_of(scene: &Scene) -> View {
    scene.view.unwrap_or(View {
        field: ViewField::Ez,
        gain: 1.0,
        auto_gain: false,
    })
}

/// Builds the simulation on `canvas`. Rejects with code `no_webgpu` when
/// the browser has no WebGPU or no adapter that can draw to the canvas;
/// the page then starts the CPU fallback.
#[wasm_bindgen(js_name = createSim)]
pub async fn create_sim(canvas: HtmlCanvasElement, scene_json: String) -> JsResult<Sim> {
    let scene = load_scene(&scene_json)?;
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
    let sim = GpuSim::new(gpu, &scene).map_err(js_err)?;
    let renderer = Renderer::new(gpu, &sim, &scene, format);
    let viewport = Viewport::fit(scene.grid.nx, scene.grid.ny, width as f64, height as f64);
    let mut out = Sim {
        gpu,
        surface,
        config,
        scene,
        sim,
        renderer,
        viewport,
    };
    out.apply_view()?;
    Ok(out)
}

impl Sim {
    /// Turns the intensity mean on or off to match the scene's view.
    fn apply_view(&mut self) -> JsResult<()> {
        let tau = (view_of(&self.scene).field == ViewField::Intensity)
            .then(|| intensity_tau(&self.scene));
        self.sim.set_intensity(tau).map_err(js_err)
    }
}

#[wasm_bindgen]
impl Sim {
    /// Advances `n` steps (submitted in batches, not waited on).
    pub fn step(&mut self, n: u32) {
        self.sim.step_n(n as u64);
    }

    /// Feeds finished monitor batches to the probes and reports the first
    /// device error since the last call. Call once per frame.
    pub fn check(&mut self) -> JsResult<()> {
        self.sim.drain_taps().map_err(js_err)?;
        self.gpu.check().map_err(js_err)
    }

    /// Draws the current field to the canvas.
    pub fn render(&mut self) -> JsResult<()> {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(());
            }
            _ => return Ok(()),
        };
        let target = frame.texture.create_view(&Default::default());
        let view = view_of(&self.scene);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        self.renderer.draw(
            self.gpu,
            &mut enc,
            &target,
            &Frame {
                viewport: self.viewport,
                gain: view.gain,
                field: view.field,
            },
        );
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.queue.present(frame);
        Ok(())
    }

    /// The canvas changed size in device pixels: reconfigure and refit.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.gpu.device, &self.config);
        self.viewport = Viewport::fit(
            self.scene.grid.nx,
            self.scene.grid.ny,
            self.config.width as f64,
            self.config.height as f64,
        );
    }

    /// Sets the view (`{"field": "ez" | "intensity", "gain": g,
    /// "auto_gain": b}`); it travels with the scene in shared links.
    #[wasm_bindgen(js_name = setView)]
    pub fn set_view(&mut self, json: &str) -> JsResult<()> {
        let view: View = serde_json::from_str(json)
            .map_err(|e| js_err(fdtd_core::FdtdError::Json(e.to_string())))?;
        let mut next = self.scene.clone();
        next.view = Some(view);
        next.validate().map_err(js_err)?;
        let changed = view_of(&self.scene).field != view.field;
        self.scene.view = Some(view);
        if changed {
            self.apply_view()?;
        }
        Ok(())
    }

    /// The status strip numbers as JSON (`fdtd_core::scene::Status`).
    pub fn status(&self) -> JsResult<String> {
        let st = self.scene.status(self.sim.time_step()).map_err(js_err)?;
        serde_json::to_string(&st).map_err(|e| js_err(fdtd_core::FdtdError::Json(e.to_string())))
    }

    /// The current scene, edits included, as JSON.
    pub fn scene(&self) -> JsResult<String> {
        self.scene.to_json().map_err(js_err)
    }

    /// Zeroes the fields and time, keeping the scene.
    #[wasm_bindgen(js_name = resetFields)]
    pub fn reset_fields(&mut self) -> JsResult<()> {
        self.sim = GpuSim::new(self.gpu, &self.scene).map_err(js_err)?;
        self.renderer = Renderer::new(self.gpu, &self.sim, &self.scene, self.config.format);
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

    /// The adapter in one line, for the status strip.
    pub fn adapter(&self) -> String {
        self.gpu.describe()
    }
}
