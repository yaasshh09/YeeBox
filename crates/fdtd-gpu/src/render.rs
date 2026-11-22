//! Drawing the field: the view transform, the material overlay colors, and
//! the render pass that reads the simulation's own storage buffers, so a
//! frame costs no readback. Display only; nothing here feeds a kernel.
//!
//! Screen pixels have y down, cells have y up: node (0, 0) is drawn at the
//! bottom left. A node covers the cells `[i - 1/2, i + 1/2)` around it.

use crate::colormap;
use crate::device::Gpu;
use crate::sim::GpuSim;
use fdtd_core::material::Material;
use fdtd_core::scene::{Scene, ViewField};
use wgpu::util::DeviceExt;

/// Which part of the grid lands where on a canvas of `width` x `height`
/// pixels. `center` is the cell position at the canvas center, and
/// `px_per_cell` the zoom. Shared by the GPU render, the CPU fallback and
/// the pointer mapping, so a click lands on the node drawn under it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub nx: usize,
    pub ny: usize,
    pub width: f64,
    pub height: f64,
    pub center: (f64, f64),
    pub px_per_cell: f64,
}

/// Furthest zoom in, pixels per cell.
pub const MAX_PX_PER_CELL: f64 = 64.0;

impl Viewport {
    /// The whole grid, centered, as large as fits.
    pub fn fit(nx: usize, ny: usize, width: f64, height: f64) -> Viewport {
        Viewport {
            nx,
            ny,
            width,
            height,
            center: ((nx as f64 - 1.0) / 2.0, (ny as f64 - 1.0) / 2.0),
            px_per_cell: Self::fit_scale(nx, ny, width, height),
        }
    }

    fn fit_scale(nx: usize, ny: usize, width: f64, height: f64) -> f64 {
        (width / nx as f64).min(height / ny as f64)
    }

    /// The cell position under pixel (px, py), pixel centers at + 1/2.
    pub fn cell_at(&self, px: f64, py: f64) -> (f64, f64) {
        (
            self.center.0 + (px - 0.5 * self.width) / self.px_per_cell,
            self.center.1 - (py - 0.5 * self.height) / self.px_per_cell,
        )
    }

    /// The pixel position of cell position (x, y).
    pub fn px_of(&self, x: f64, y: f64) -> (f64, f64) {
        (
            0.5 * self.width + (x - self.center.0) * self.px_per_cell,
            0.5 * self.height - (y - self.center.1) * self.px_per_cell,
        )
    }

    /// Zooms by `factor` about pixel (px, py), which keeps the same cell
    /// under it. Never zooms out past the fit or in past `MAX_PX_PER_CELL`.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        if !(factor.is_finite() && factor > 0.0) {
            return;
        }
        let anchor = self.cell_at(px, py);
        let lo = Self::fit_scale(self.nx, self.ny, self.width, self.height);
        self.px_per_cell = (self.px_per_cell * factor).clamp(lo, MAX_PX_PER_CELL.max(lo));
        let moved = self.cell_at(px, py);
        self.center.0 += anchor.0 - moved.0;
        self.center.1 += anchor.1 - moved.1;
    }

    /// Moves the picture by (dx, dy) pixels.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        if dx.is_finite() && dy.is_finite() {
            self.center.0 -= dx / self.px_per_cell;
            self.center.1 += dy / self.px_per_cell;
        }
    }
}

/// The overlay drawn over a node of material `m`: color and opacity, in
/// encoded sRGB. Vacuum is clear; dielectrics a white veil that thickens
/// with eps; lossy media a dark brown; conductors an opaque light gray;
/// Drude metals a lighter, nearly opaque silver; Lorentz resonators a violet
/// veil.
pub fn overlay(m: &Material) -> [f32; 4] {
    match *m {
        Material::Pec => [0.66, 0.68, 0.72, 1.0],
        Material::Drude { .. } => [0.82, 0.84, 0.88, 0.9],
        Material::Lorentz { .. } => [0.55, 0.38, 0.78, 0.45],
        Material::Dielectric { sigma, .. } if sigma > 0.0 => [0.24, 0.16, 0.10, 0.55],
        Material::Dielectric { eps_inf, .. } if eps_inf > 1.0 => {
            let a = 0.10 + 0.25 * ((eps_inf - 1.0) / 11.0).min(1.0);
            [1.0, 1.0, 1.0, a as f32]
        }
        Material::Dielectric { .. } => [0.0; 4],
    }
}

/// Time constant of the intensity mean, in steps: four periods of the
/// slowest drive (point sources and the plane wave), or 200 steps when no
/// drive has a frequency. Long enough to average a period out, short
/// enough to follow an edit within a second or two.
pub fn intensity_tau(scene: &Scene) -> f64 {
    let s = scene.grid.courant;
    scene
        .sources
        .iter()
        .map(|src| src.waveform)
        .chain(scene.plane_wave.map(|pw| pw.waveform))
        .filter_map(|w| w.frequency())
        .map(|f| 4.0 / (f * s))
        .fold(None, |m: Option<f64>, t| Some(m.map_or(t, |m| m.max(t))))
        .unwrap_or(200.0)
}

/// The render pass done on the CPU, for the fallback kernel: per pixel the
/// same steps as `shaders/render.wgsl`, in f64, so it matches the GPU pass
/// up to rounding (one colormap step at most, checked in Tier 4).
pub struct CpuPainter {
    diverging: Vec<[u8; 4]>,
    sequential: Vec<[u8; 4]>,
    /// Overlay per material id, the wall entry last.
    mats: Vec<[f32; 4]>,
    nx: usize,
    ny: usize,
    pml: usize,
    periodic_y: bool,
}

const PML_TINT: [f64; 3] = [0.20, 0.22, 0.30];

impl CpuPainter {
    pub fn new(scene: &Scene) -> CpuPainter {
        CpuPainter {
            diverging: colormap::diverging(),
            sequential: colormap::sequential(),
            mats: scene
                .materials
                .iter()
                .map(overlay)
                .chain([overlay(&Material::Pec)])
                .collect(),
            nx: scene.grid.nx,
            ny: scene.grid.ny,
            pml: scene.pml_thickness(),
            periodic_y: scene.periodic_y,
        }
    }

    /// RGBA8 pixels, row-major from the top left, of `frame`. `ids` is the
    /// material id per node with the wall entry on the ring (`Layout::ids`),
    /// `ez` the field and `mean` the intensity mean (read only in that view).
    pub fn draw(&self, ids: &[u32], ez: &[f64], mean: &[f64], frame: &Frame) -> Vec<u8> {
        let v = &frame.viewport;
        let (w, h) = (v.width as usize, v.height as usize);
        let (nx, ny, d) = (self.nx, self.ny, self.pml);
        let mut out = vec![0u8; w * h * 4];
        for py in 0..h {
            for px in 0..w {
                let at = (py * w + px) * 4;
                let (x, y) = v.cell_at(px as f64 + 0.5, py as f64 + 0.5);
                let (fi, fj) = (x.round(), y.round());
                if fi < 0.0 || fj < 0.0 || fi > (nx - 1) as f64 || fj > (ny - 1) as f64 {
                    out[at..at + 4].copy_from_slice(&[0, 0, 0, 255]);
                    continue;
                }
                let (i, j) = (fi as usize, fj as usize);
                let k = j * nx + i;
                let entry = match frame.field {
                    ViewField::Intensity => {
                        let u = (2.0 * mean[k] * frame.gain * frame.gain).clamp(0.0, 1.0);
                        self.sequential[(u * 256.0).round() as usize]
                    }
                    ViewField::Ez => {
                        let val = (ez[k] * frame.gain).clamp(-1.0, 1.0);
                        self.diverging[((val + 1.0) * 128.0).round() as usize]
                    }
                };
                let mut c = [0.0f64; 3];
                for (ch, e) in c.iter_mut().zip(entry) {
                    *ch = e as f64 / 255.0;
                }
                let in_y = !self.periodic_y && (j < d || j > ny - 1 - d);
                if d > 0 && (i < d || i > nx - 1 - d || in_y) {
                    for (ch, tint) in c.iter_mut().zip(PML_TINT) {
                        *ch = *ch * 0.65 + tint * 0.35;
                    }
                }
                let id = ids[k];
                let edge = (i > 0 && ids[k - 1] != id)
                    || (i + 1 < nx && ids[k + 1] != id)
                    || (j > 0 && ids[k - nx] != id)
                    || (j + 1 < ny && ids[k + nx] != id);
                let m = self.mats[id as usize];
                let a = if edge && m[3] > 0.0 {
                    m[3].max(0.8) as f64
                } else {
                    m[3] as f64
                };
                for (n, ch) in c.iter_mut().enumerate() {
                    let mixed = *ch * (1.0 - a) + m[n] as f64 * a;
                    out[at + n] = (mixed * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                out[at + 3] = 255;
            }
        }
        out
    }
}

/// The color outside the grid: the black stage.
pub const BACKGROUND: [f32; 3] = [0.0, 0.0, 0.0];

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    nx: u32,
    ny: u32,
    pml: u32,
    periodic_y: u32,
    cx: f32,
    cy: f32,
    px_per_cell: f32,
    gain: f32,
    width: f32,
    height: f32,
    mode: u32,
    pad: u32,
}

/// What one frame shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub viewport: Viewport,
    /// Field value drawn at full color is `1 / gain`.
    pub gain: f64,
    pub field: ViewField,
}

/// The render pass for one simulation and one target format. Bind it once;
/// live paint writes the same id buffer, so frames stay current.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    uniforms: wgpu::Buffer,
    base: Uniforms,
}

impl Renderer {
    /// `format` must be a non-sRGB format (the tables are already encoded).
    pub fn new(gpu: &Gpu, sim: &GpuSim, scene: &Scene, format: wgpu::TextureFormat) -> Renderer {
        let device = &gpu.device;
        let luts: Vec<u32> = colormap::diverging()
            .into_iter()
            .chain(colormap::sequential())
            .map(u32::from_le_bytes)
            .collect();
        // Overlay per material id, then the wall entry, drawn like PEC.
        let mats: Vec<[f32; 4]> = scene
            .materials
            .iter()
            .map(overlay)
            .chain([overlay(&Material::Pec)])
            .collect();
        let init = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let base = Uniforms {
            nx: scene.grid.nx as u32,
            ny: scene.grid.ny as u32,
            pml: scene.pml_thickness() as u32,
            periodic_y: scene.periodic_y as u32,
            cx: 0.0,
            cy: 0.0,
            px_per_cell: 1.0,
            gain: 1.0,
            width: 1.0,
            height: 1.0,
            mode: 0,
            pad: 0,
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let uniforms = init(
            "render uniforms",
            bytemuck::bytes_of(&base),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let luts = init("luts", bytemuck::cast_slice(&luts), storage);
        let mats = init("overlays", bytemuck::cast_slice(&mats), storage);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, read),
            ],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &layout,
            entries: &[
                &uniforms,
                sim.ez_buffer(),
                sim.ids_buffer(),
                &luts,
                &mats,
                sim.mean_buffer(),
            ]
            .iter()
            .enumerate()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: b as u32,
                resource: buf.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Renderer {
            pipeline,
            group,
            uniforms,
            base,
        }
    }

    /// Encodes one frame into `target`, which must match the viewport size.
    pub fn draw(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        frame: &Frame,
    ) {
        let v = &frame.viewport;
        let u = Uniforms {
            cx: v.center.0 as f32,
            cy: v.center.1 as f32,
            px_per_cell: v.px_per_cell as f32,
            gain: frame.gain as f32,
            width: v.width as f32,
            height: v.height as f32,
            mode: match frame.field {
                ViewField::Ez => 0,
                ViewField::Intensity => 1,
            },
            ..self.base
        };
        gpu.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_centers_and_maps_pixel_centers_to_nodes() {
        // 1 px per cell on a canvas the size of the grid: pixel (i, ny-1-j)
        // is node (i, j) exactly.
        let v = Viewport::fit(40, 30, 40.0, 30.0);
        assert_eq!(v.px_per_cell, 1.0);
        for (i, j) in [(0, 0), (39, 29), (12, 7)] {
            let (x, y) = v.cell_at(i as f64 + 0.5, (29 - j) as f64 + 0.5);
            assert_eq!((x, y), (i as f64, j as f64));
        }
        // A wide canvas letterboxes left and right.
        let v = Viewport::fit(40, 30, 200.0, 60.0);
        assert_eq!(v.px_per_cell, 2.0);
        assert_eq!(v.px_of(19.5, 14.5), (100.0, 30.0));
    }

    #[test]
    fn zoom_keeps_the_cell_under_the_cursor() {
        let mut v = Viewport::fit(100, 80, 500.0, 400.0);
        let before = v.cell_at(123.0, 77.0);
        v.zoom_at(123.0, 77.0, 3.0);
        assert_eq!(v.px_per_cell, 15.0);
        let after = v.cell_at(123.0, 77.0);
        assert!((before.0 - after.0).abs() < 1e-12 && (before.1 - after.1).abs() < 1e-12);
        // Out past the fit clamps to the fit; in past the limit clamps too.
        v.zoom_at(0.0, 0.0, 1e-6);
        assert_eq!(v.px_per_cell, 5.0);
        v.zoom_at(0.0, 0.0, 1e6);
        assert_eq!(v.px_per_cell, MAX_PX_PER_CELL);
        v.zoom_at(0.0, 0.0, f64::NAN);
        assert_eq!(v.px_per_cell, MAX_PX_PER_CELL);
    }

    #[test]
    fn pan_moves_with_the_pointer() {
        let mut v = Viewport::fit(100, 80, 500.0, 400.0);
        let cell = v.cell_at(200.0, 200.0);
        v.pan(10.0, -20.0);
        let moved = v.cell_at(210.0, 180.0);
        assert!((cell.0 - moved.0).abs() < 1e-12 && (cell.1 - moved.1).abs() < 1e-12);
    }

    #[test]
    fn intensity_tau_is_four_slowest_periods() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../validation/scenes/painted_walls.json"
        );
        let mut s = Scene::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
        // Ricker at 0.05, S = 0.5: a period is 40 steps.
        assert_eq!(intensity_tau(&s), 160.0);
        let mut slow = s.sources[0];
        slow.waveform = fdtd_core::source::Waveform::RampedCw {
            f0: 0.025,
            ramp_periods: 3.0,
            phase: 0.0,
            start: 0.0,
        };
        s.sources.push(slow);
        assert_eq!(intensity_tau(&s), 320.0);
        s.sources.clear();
        assert_eq!(intensity_tau(&s), 200.0);
    }

    #[test]
    fn overlays() {
        assert_eq!(overlay(&Material::VACUUM)[3], 0.0);
        assert_eq!(overlay(&Material::Pec)[3], 1.0);
        let glass = overlay(&Material::Dielectric {
            eps_inf: 2.25,
            sigma: 0.0,
        });
        let si = overlay(&Material::Dielectric {
            eps_inf: 12.0,
            sigma: 0.0,
        });
        assert!(0.0 < glass[3] && glass[3] < si[3] && si[3] < 1.0);
    }
}
