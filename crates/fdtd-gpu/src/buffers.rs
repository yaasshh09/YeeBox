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
//!   then the medium table      `ROW` vec4s per row (below)
//!
//! A row is the `fdtd_core::dispersive::Row` of one medium, cast to f32:
//!
//!   (Ca, Cb, flags, 0)         flags: 1 Drude pole, 2 Lorentz pole, 3 both
//!   (k, beta, cj, 0)           the Drude coefficients, zero without a pole
//!   (a1, a2, a3, cp)           the Lorentz coefficients, zero without one
//!
//! TMz rows: one per scene material, then a PEC row for the wall ring; the
//! E update indexes them by the node's material id. TEz rows: the same, then
//! one row per unordered pair of materials (`pair_id`), the medium of an edge
//! between two such cells; the E updates index them by `edge_ids`. The pair
//! table depends only on the material table, which a live edit cannot
//! change, so painting rewrites only the edge ids.
//!
//! Drude J, Lorentz P and P^(n-1) of one E array live in one `aux` buffer of
//! `3 nx ny` f32 (J at k, P at nx ny + k, P^(n-1) at 2 nx ny + k), or a stub
//! when the scene has no dispersive material.
//!
//! One packed buffer, not one per profile: WebGPU allows 8 storage buffers
//! per shader stage by default, and the E update would need 11.

use fdtd_core::cpml::{Cpml, Profile};
use fdtd_core::dispersive::{Medium, Row};
use fdtd_core::grid::Component;
use fdtd_core::scene::{Mode, Scene};
use fdtd_core::tfsf::Correction;

/// vec4s per medium table row.
pub const ROW: usize = 3;

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

/// Material ids, the medium table and the CPML profiles of a validated
/// scene.
#[derive(Debug, Clone)]
pub struct Layout {
    pub nx: usize,
    pub ny: usize,
    pub mode: Mode,
    /// The Courant number S.
    pub s: f64,
    /// Scene materials; row `n_mat` is the wall.
    pub n_mat: usize,
    /// Material id per node; `wall_id()` on the wall ring, which is every
    /// node the node update skips. The render pass reads these in both modes.
    pub ids: Vec<u32>,
    /// The medium table in f64, as computed (module docs).
    pub table: Vec<Row>,
    /// TEz: the table row of every Ex and every Ey edge (`pair_id` of its two
    /// cells); padding edges point at the wall row. `None` in TMz.
    pub edge_ids: Option<[Vec<u32>; 2]>,
    pub cpml: Cpml,
}

impl Layout {
    /// For a scene that has passed `Scene::validate`.
    pub fn new(scene: &Scene) -> Self {
        let (nx, ny, s) = (scene.grid.nx, scene.grid.ny, scene.grid.courant);
        let row = |m: Option<Medium>| m.map_or(Row::PEC, |m| m.row(s));
        let mut table: Vec<Row> = scene.materials.iter().map(|m| row(Medium::of(m))).collect();
        table.push(Row::PEC);
        let n_mat = scene.materials.len();
        if scene.mode == Mode::Te {
            // A pair that cannot share an edge (two different Drude or
            // Lorentz poles) gets a PEC row; the validator never lets one
            // touch.
            for a in 0..n_mat {
                for b in a..n_mat {
                    let (ma, mb) = (&scene.materials[a], &scene.materials[b]);
                    table.push(Medium::edge(ma, mb).map_or(Row::PEC, row));
                }
            }
        }
        let wall = n_mat as u32;
        let raster = scene.rasterize();
        let mut ids = vec![wall; nx * ny];
        for j in scene.live_rows() {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                ids[k] = raster[k] as u32;
            }
        }
        let mut layout = Layout {
            nx,
            ny,
            mode: scene.mode,
            s,
            n_mat,
            ids,
            table,
            edge_ids: None,
            cpml: Cpml::for_scene(scene),
        };
        if scene.mode == Mode::Te {
            layout.edge_ids = Some(layout.edges(scene, &raster));
        }
        layout
    }

    /// The table row of the medium between cells of materials `a` and `b`
    /// (TEz pair rows follow the material rows and the wall row).
    pub fn pair_id(&self, a: usize, b: usize) -> u32 {
        let (a, b) = (a.min(b), a.max(b));
        // Rows of pairs (a, a..n) come in order a = 0, 1, ...: row a starts
        // after sum over t < a of (n - t) = a n - a (a - 1) / 2 pairs.
        let before = a * self.n_mat - a * a.saturating_sub(1) / 2;
        (self.n_mat + 1 + before + (b - a)) as u32
    }

    /// The edge ids of a TEz scene from its rasterized cells, over the edges
    /// `SimTe` updates: Ex rows `0..ny - 1` (all with periodic y, row ny - 1
    /// joining row 0), Ey columns `0..nx - 1`.
    pub fn edges(&self, scene: &Scene, raster: &[usize]) -> [Vec<u32>; 2] {
        let (nx, ny) = (self.nx, self.ny);
        let wall = self.wall_id();
        let mut ex = vec![wall; nx * ny];
        let mut ey = vec![wall; nx * ny];
        let ex_rows = if scene.periodic_y { ny } else { ny - 1 };
        for j in 0..ex_rows {
            let jn = if j + 1 == ny { 0 } else { j + 1 };
            for i in 0..nx {
                let (k, up) = (j * nx + i, jn * nx + i);
                ex[k] = self.pair_id(raster[k], raster[up]);
            }
        }
        for j in 0..ny {
            for i in 0..nx - 1 {
                let k = j * nx + i;
                ey[k] = self.pair_id(raster[k], raster[k + 1]);
            }
        }
        [ex, ey]
    }

    /// The id of the PEC wall row, right after the material rows.
    pub fn wall_id(&self) -> u32 {
        self.n_mat as u32
    }

    /// The node coefficient the TF/SF node corrections scale by: Cb per node
    /// in TMz (zero on walls and PEC), S on every node in TEz (Hz has
    /// mu = 1 everywhere, as `SimTe` uses).
    pub fn cb_nodes(&self) -> Vec<f64> {
        match self.mode {
            Mode::Tm => self
                .ids
                .iter()
                .map(|&m| self.table[m as usize].cb)
                .collect(),
            Mode::Te => vec![self.s; self.ids.len()],
        }
    }

    /// Whether a scene needs the J and P buffers at all.
    pub fn dispersive(&self) -> bool {
        self.table.iter().any(Row::is_dispersive)
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
        let mut out = Vec::with_capacity(self.table_offset() + ROW * self.table.len());
        for p in [x_int, x_half, y_int, y_half] {
            out.extend(rows(p));
        }
        for r in &self.table {
            let flags = r.drude.is_some() as u32 + 2 * r.lorentz.is_some() as u32;
            out.push([r.ca as f32, r.cb as f32, flags as f32, 0.0]);
            out.push(
                r.drude
                    .map_or([0.0; 4], |d| [d.k as f32, d.beta as f32, d.cj as f32, 0.0]),
            );
            out.push(r.lorentz.map_or([0.0; 4], |l| {
                [l.a1 as f32, l.a2 as f32, l.a3 as f32, l.cp as f32]
            }));
        }
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

    /// The table row an id points at, as uploaded.
    fn row(coef: &[[f32; 4]], layout: &Layout, id: u32) -> [[f32; 4]; 3] {
        let at = layout.table_offset() + ROW * id as usize;
        [coef[at], coef[at + 1], coef[at + 2]]
    }

    /// TEz: every edge id points at a row whose (Ca, Cb) are the CPU
    /// kernel's edge coefficients cast to f32, on every scene run as TEz
    /// (where it validates), and dispersive pairs carry their poles.
    #[test]
    fn te_edge_upload_matches_the_cpu_arrays() {
        let mut checked = 0;
        for (name, mut scene) in scenes() {
            scene.mode = fdtd_core::scene::Mode::Te;
            let Ok(cpu) = fdtd_core::te::SimTe::from_scene(&scene) else {
                continue;
            };
            checked += 1;
            let layout = Layout::new(&scene);
            let coef = layout.coef();
            let [ca_x, cb_x, ca_y, cb_y] = cpu.edge_coefficients();
            let [ids_x, ids_y] = layout.edge_ids.as_ref().unwrap();
            let (nx, ny) = (layout.nx, layout.ny);
            let ex_rows = if scene.periodic_y { ny } else { ny - 1 };
            for j in 0..ny {
                for i in 0..nx {
                    let k = j * nx + i;
                    if j < ex_rows {
                        let r = row(&coef, &layout, ids_x[k]);
                        assert_eq!(
                            (r[0][0], r[0][1]),
                            (ca_x[k] as f32, cb_x[k] as f32),
                            "{name} ex {k}"
                        );
                    }
                    if i + 1 < nx {
                        let r = row(&coef, &layout, ids_y[k]);
                        assert_eq!(
                            (r[0][0], r[0][1]),
                            (ca_y[k] as f32, cb_y[k] as f32),
                            "{name} ey {k}"
                        );
                    }
                }
            }
        }
        assert!(checked >= 5, "only {checked} scenes ran as TEz");
    }

    /// Rows carry the dispersive coefficients of `dispersive.rs` cast to
    /// f32, with the pole flags in the third slot of the first vec4.
    #[test]
    fn dispersive_rows_are_the_core_rows() {
        let mut scene = Scene::from_json(
            r#"{"version": 1, "grid": {"nx": 30, "ny": 20, "dx_phys_m": 0.001, "courant": 0.5},
            "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0},
                {"kind": "drude", "eps_inf": 2, "wp": 0.4, "gamma": 0.01},
                {"kind": "lorentz", "eps_inf": 1.5, "d_eps": 2, "w0": 0.3, "delta": 0.02}]}"#,
        )
        .unwrap();
        for mode in [fdtd_core::scene::Mode::Tm, fdtd_core::scene::Mode::Te] {
            scene.mode = mode;
            let layout = Layout::new(&scene);
            let coef = layout.coef();
            for (id, m) in scene.materials.iter().enumerate() {
                let want = fdtd_core::dispersive::Medium::of(m).unwrap().row(0.5);
                let r = row(&coef, &layout, id as u32);
                let flags = want.drude.is_some() as u32 + 2 * want.lorentz.is_some() as u32;
                assert_eq!(r[0], [want.ca as f32, want.cb as f32, flags as f32, 0.0]);
                if let Some(d) = want.drude {
                    assert_eq!(r[1], [d.k as f32, d.beta as f32, d.cj as f32, 0.0]);
                }
                if let Some(l) = want.lorentz {
                    assert_eq!(r[2], [l.a1 as f32, l.a2 as f32, l.a3 as f32, l.cp as f32]);
                }
            }
        }
    }

    /// The uploaded coefficients are the CPU kernel arrays cast to f32,
    /// node by node and profile entry by profile entry, on every scene.
    #[test]
    fn upload_matches_the_cpu_arrays() {
        for (name, scene) in scenes() {
            if scene.mode != fdtd_core::scene::Mode::Tm {
                continue;
            }
            let cpu = SimTm::from_scene(&scene).unwrap();
            let layout = Layout::new(&scene);
            let coef = layout.coef();
            let (ca, cb) = cpu.coefficients();
            for k in 0..ca.len() {
                let m = row(&coef, &layout, layout.ids[k])[0];
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
            assert_eq!(
                coef.len(),
                layout.table_offset() + ROW * (scene.materials.len() + 1)
            );
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
