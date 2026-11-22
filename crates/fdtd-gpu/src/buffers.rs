//! What the GPU kernel uploads, built on the CPU from `fdtd-core` and
//! nothing else. Every number here is a value `fdtd-core` computed in f64,
//! cast to f32 at the end; no formula is repeated.
//!
//! Field and psi buffers are f32 `nx * ny`, index `j * nx + i`, padded
//! exactly like the CPU arrays (see `fdtd_core::tm`). The coefficient buffer
//! is one `array<vec4<f32>>`:
//!
//!   [0, nx)                    x_int  (b, c, 1/kappa, 0) at node x = i
//!   [nx, 2 nx)                 x_half at x = i + 1/2
//!   [2 nx, 2 nx + ny)          y_int  at node y = j
//!   [2 nx + ny, 2 (nx + ny))   y_half at y = j + 1/2
//!   then the material table    (Ca, Cb, 0, 0), one entry per scene
//!                              material, and a last (0, 0) for the walls.
//!
//! One packed buffer, not one per profile: WebGPU allows 8 storage buffers
//! per shader stage by default, and the E update would need 11.

use fdtd_core::cpml::{Cpml, Profile};
use fdtd_core::grid::Component;
use fdtd_core::scene::Scene;
use fdtd_core::tfsf::Correction;

/// Uniform parameters of one step of a batch. Every step of a batch gets its
/// own 256-byte slot (the default `min_uniform_buffer_offset_alignment`),
/// and only `step` differs between slots.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub nx: u32,
    pub ny: u32,
    pub periodic_y: u32,
    /// Index of this step within the batch, for the per-step data arrays.
    pub step: u32,
    /// The Courant number S, which is dt in normalized units.
    pub s: f32,
    pub n_src: u32,
    pub n_taps: u32,
    pub n_hgroups: u32,
    pub n_egroups: u32,
    /// Entries of incident Ez per step (the TF/SF span plus one).
    pub n_inc: u32,
    pub pad: [u32; 2],
}

/// Byte stride between step slots in the uniform buffer.
pub const PARAMS_STRIDE: u64 = 256;

/// One TF/SF correction as the shaders read it: `field[k] += factor *
/// inc[src]`, field numbered like the gather taps (0 Ez, 1 Hx, 2 Hy).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCorrection {
    pub k: u32,
    pub src: u32,
    pub field: u32,
    pub factor: f32,
}

/// The shader number of a field: 0 the node field (Ez or Hz), 1 the y-edge
/// field (Hx or Ex), 2 the x-edge field (Hy or Ey). TEz shares the TMz slots.
pub fn field_id(c: Component) -> u32 {
    match c {
        Component::Ez | Component::Hz => 0,
        Component::Hx | Component::Ex => 1,
        Component::Hy | Component::Ey => 2,
    }
}

/// Correction groups flattened in order, entry e of group g at `2 g + e`.
pub fn pack_corrections(groups: &[[Correction; 2]]) -> Vec<GpuCorrection> {
    groups
        .iter()
        .flatten()
        .map(|c| GpuCorrection {
            k: c.k as u32,
            src: c.src as u32,
            field: field_id(c.field),
            factor: c.factor as f32,
        })
        .collect()
}

/// Material ids, the coefficient table and the CPML profiles of a
/// validated scene.
#[derive(Debug, Clone)]
pub struct Layout {
    pub nx: usize,
    pub ny: usize,
    /// Material id per node; `wall_id()` on the PEC wall ring, which is
    /// every node the E update skips.
    pub ids: Vec<u32>,
    /// (Ca, Cb) per material, then (0, 0) for the wall. f64, as computed.
    pub table: Vec<(f64, f64)>,
    pub cpml: Cpml,
}

impl Layout {
    /// For a scene that has passed `Scene::validate`.
    pub fn new(scene: &Scene) -> Self {
        let (nx, ny, s) = (scene.grid.nx, scene.grid.ny, scene.grid.courant);
        let mut table: Vec<(f64, f64)> =
            scene.materials.iter().map(|m| m.coefficients(s)).collect();
        table.push((0.0, 0.0));
        let wall = scene.materials.len() as u32;
        let raster = scene.rasterize();
        let mut ids = vec![wall; nx * ny];
        for j in scene.live_rows() {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                ids[k] = raster[k] as u32;
            }
        }
        Layout {
            nx,
            ny,
            ids,
            table,
            cpml: Cpml::for_scene(scene),
        }
    }

    /// The id of the PEC wall entry, the last in the table.
    pub fn wall_id(&self) -> u32 {
        (self.table.len() - 1) as u32
    }

    /// Cb per node in f64, the array the TF/SF E corrections scale by.
    pub fn cb_nodes(&self) -> Vec<f64> {
        self.ids.iter().map(|&m| self.table[m as usize].1).collect()
    }

    /// The packed coefficient buffer described in the module docs.
    pub fn coef(&self) -> Vec<[f32; 4]> {
        let Cpml {
            x_int,
            x_half,
            y_int,
            y_half,
        } = &self.cpml;
        let rows = |p: &Profile| {
            (0..p.b.len())
                .map(|i| [p.b[i] as f32, p.c[i] as f32, p.inv_kappa[i] as f32, 0.0])
                .collect::<Vec<_>>()
        };
        let mut out = Vec::with_capacity(self.table_offset() + self.table.len());
        for p in [x_int, x_half, y_int, y_half] {
            out.extend(rows(p));
        }
        out.extend(
            self.table
                .iter()
                .map(|&(ca, cb)| [ca as f32, cb as f32, 0.0, 0.0]),
        );
        out
    }

    /// Where the material table starts in `coef`.
    pub fn table_offset(&self) -> usize {
        2 * (self.nx + self.ny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdtd_core::tm::SimTm;
    use std::path::Path;

    fn scenes() -> Vec<(String, Scene)> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/scenes");
        let mut out: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .map(|p| {
                let text = std::fs::read_to_string(&p).unwrap();
                (p.display().to_string(), Scene::from_json(&text).unwrap())
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The uploaded coefficients are the CPU kernel arrays cast to f32,
    /// node by node and profile entry by profile entry, on every scene.
    #[test]
    fn upload_matches_the_cpu_arrays() {
        for (name, scene) in scenes() {
            let cpu = SimTm::from_scene(&scene).unwrap();
            let layout = Layout::new(&scene);
            let coef = layout.coef();
            let (ca, cb) = cpu.coefficients();
            let base = layout.table_offset();
            for k in 0..ca.len() {
                let m = coef[base + layout.ids[k] as usize];
                assert_eq!(
                    (m[0], m[1]),
                    (ca[k] as f32, cb[k] as f32),
                    "{name} node {k}"
                );
            }
            assert_eq!(layout.cb_nodes(), cb, "{name}");
            let (nx, ny) = (layout.nx, layout.ny);
            let want = Cpml::for_scene(&scene);
            for (off, p) in [
                (0, &want.x_int),
                (nx, &want.x_half),
                (2 * nx, &want.y_int),
                (2 * nx + ny, &want.y_half),
            ] {
                for (i, c) in coef[off..off + p.b.len()].iter().enumerate() {
                    let e = [p.b[i] as f32, p.c[i] as f32, p.inv_kappa[i] as f32, 0.0];
                    assert_eq!(*c, e, "{name} profile at {off} entry {i}");
                }
            }
            assert_eq!(coef.len(), base + scene.materials.len() + 1);
        }
    }

    /// Wall ids sit exactly on the nodes the CPU E loop never updates.
    #[test]
    fn wall_ids_on_the_wall_ring() {
        for (name, scene) in scenes() {
            let l = Layout::new(&scene);
            for j in 0..l.ny {
                for i in 0..l.nx {
                    let wall = i == 0 || i == l.nx - 1 || !scene.live_rows().contains(&j);
                    assert_eq!(
                        l.ids[j * l.nx + i] == l.wall_id(),
                        wall,
                        "{name} ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn params_fit_a_slot() {
        assert_eq!(std::mem::size_of::<Params>(), 48);
        assert!(std::mem::size_of::<Params>() as u64 <= PARAMS_STRIDE);
    }
}
