//! 2D TEz solver on the TMz layout with the roles swapped (`grid.rs`).
//! Fields are normalized (eps0 = mu0 = 1, dx = 1, dt = S), every array is
//! `nx * ny` with index `j * nx + i`:
//!
//! - `hz[i, j]` on node (i, j), the cell of material `ids[i, j]`
//! - `ex[i, j]` on the y-edge (i, j + 1/2), between cells (i, j) and (i, j + 1)
//! - `ey[i, j]` on the x-edge (i + 1/2, j), between cells (i, j) and (i + 1, j)
//!
//! Update, in this order, the dual of `tm.rs` (Ez -> Hz, Hx -> -Ex, Hy -> -Ey):
//!
//!   Hz^(n+1/2) = Hz^(n-1/2) + S (dEx/dy - dEy/dx)          interior nodes
//!   Ex^(n+1)   = Ca Ex^n + Cb dHz/dy                       y-edges
//!   Ey^(n+1)   = Ca Ey^n - Cb dHz/dx                       x-edges
//!
//! After n steps the solver holds E^n and Hz^(n - 1/2), the same time levels
//! as TMz holds E and H. Each edge takes `Material::edge_mean` of its two
//! cells, so an edge touching PEC is PEC (Ca = Cb = 0).
//!
//! The outer wall: the padded ring of Hz nodes is never updated and stays
//! zero, exactly as the TMz ring of Ez nodes does. With the index ranges of
//! `tm.rs` that makes a TEz box the exact dual of a TMz PEC box, which is a
//! magnetic wall (PMC, Hz = 0) for TEz. Painted PEC inside the domain is a
//! real electric conductor. Behind a CPML the wall type does not matter.
//!
//! CPML runs through the same loops, as in `tm.rs`: each difference term is
//! scaled by the 1/kappa of its axis and the psi of its derivative is added.
//! Hz uses the profiles at integer positions, Ex the y profile at j + 1/2,
//! Ey the x profile at i + 1/2. psi is kept on the plain difference:
//! `psi_exy` (Ex, d/dy), `psi_eyx` (Ey, d/dx), `psi_hzx`, `psi_hzy`.
//!
//! Point sources drive Hz (a magnetic line current) right after the Hz
//! update, with the waveform at t = (n + 1/2) S, so the E update of the same
//! step sees them. Probes record Hz.

use crate::cpml::Cpml;
use crate::error::{FdtdError, Result};
use crate::material::Material;
use crate::monitor::PointProbe;
use crate::scene::{Mode, Monitor, Scene, Source};
use crate::source::SourceKind;

/// TEz grid state built from a validated scene.
#[derive(Debug, Clone)]
pub struct SimTe {
    nx: usize,
    ny: usize,
    s: f64,
    periodic_y: bool,
    n: u64,
    hz: Vec<f64>,
    ex: Vec<f64>,
    ey: Vec<f64>,
    /// Edge coefficients and permittivity (0 on PEC edges).
    ca_x: Vec<f64>,
    cb_x: Vec<f64>,
    eps_x: Vec<f64>,
    ca_y: Vec<f64>,
    cb_y: Vec<f64>,
    eps_y: Vec<f64>,
    pml: Cpml,
    psi_exy: Vec<f64>,
    psi_eyx: Vec<f64>,
    psi_hzx: Vec<f64>,
    psi_hzy: Vec<f64>,
    sources: Vec<Source>,
    probes: Vec<PointProbe>,
}

impl SimTe {
    /// Validates the scene, rasterizes its geometry, and precomputes the
    /// per edge Ca, Cb. All fields start at zero, at time step 0.
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        scene.validate()?;
        if scene.mode != Mode::Te {
            return Err(FdtdError::WrongMode { mode: scene.mode });
        }
        // ponytail: plane waves and spectral monitors arrive in the next
        // Phase 6 commits; refused until then rather than ignored.
        if scene.plane_wave.is_some() {
            return Err(FdtdError::InvalidParameter {
                name: "plane wave in TEz (not yet)",
                value: f64::NAN,
            });
        }
        let (nx, ny, s) = (scene.grid.nx, scene.grid.ny, scene.grid.courant);
        let mut probes = Vec::new();
        for m in &scene.monitors {
            match m {
                &Monitor::Point { i, j } => probes.push(PointProbe::new(i, j)),
                _ => {
                    return Err(FdtdError::InvalidParameter {
                        name: "spectral monitor in TEz (not yet)",
                        value: f64::NAN,
                    });
                }
            }
        }
        let hz = vec![0.0; nx * ny];
        for p in &mut probes {
            p.record(&hz, nx);
        }
        let mut sim = Self {
            nx,
            ny,
            s,
            periodic_y: scene.periodic_y,
            n: 0,
            hz,
            ex: vec![0.0; nx * ny],
            ey: vec![0.0; nx * ny],
            ca_x: vec![0.0; nx * ny],
            cb_x: vec![0.0; nx * ny],
            eps_x: vec![0.0; nx * ny],
            ca_y: vec![0.0; nx * ny],
            cb_y: vec![0.0; nx * ny],
            eps_y: vec![0.0; nx * ny],
            pml: Cpml::for_scene(scene),
            psi_exy: vec![0.0; nx * ny],
            psi_eyx: vec![0.0; nx * ny],
            psi_hzx: vec![0.0; nx * ny],
            psi_hzy: vec![0.0; nx * ny],
            sources: scene.sources.clone(),
            probes,
        };
        sim.set_edges(scene, &scene.rasterize());
        Ok(sim)
    }

    /// Edge coefficients from the rasterized cell ids, over the edges the
    /// loops update: Ex rows `0..ny - 1` (every row with periodic y, row
    /// ny - 1 then joins row 0), Ey columns `0..nx - 1`.
    fn set_edges(&mut self, scene: &Scene, ids: &[usize]) {
        let (nx, ny, s) = (self.nx, self.ny, self.s);
        let mat = |k: usize| &scene.materials[ids[k]];
        let ex_rows = if self.periodic_y { ny } else { ny - 1 };
        for j in 0..ex_rows {
            let jn = if j + 1 == ny { 0 } else { j + 1 };
            for i in 0..nx {
                let (k, up) = (j * nx + i, jn * nx + i);
                let m = Material::edge_mean(mat(k), mat(up));
                (self.ca_x[k], self.cb_x[k]) = m.coefficients(s);
                self.eps_x[k] = m.eps();
            }
        }
        for j in 0..ny {
            for i in 0..nx - 1 {
                let k = j * nx + i;
                let m = Material::edge_mean(mat(k), mat(k + 1));
                (self.ca_y[k], self.cb_y[k]) = m.coefficients(s);
                self.eps_y[k] = m.eps();
            }
        }
    }

    /// Advances one full step: Hz to n + 1/2, the point sources on Hz with
    /// the waveform at (n + 1/2) S, then E to n + 1, then the probes.
    pub fn step(&mut self) {
        let (nx, ny, s) = (self.nx, self.ny, self.s);
        let (hz, ex, ey) = (&mut self.hz, &mut self.ex, &mut self.ey);
        let p = &self.pml;
        let (ex_rows, hz_rows) = if self.periodic_y {
            (0..ny, 0..ny)
        } else {
            (0..ny - 1, 1..ny - 1)
        };

        // Hz[i, j] at node (i, j): dEx/dy - dEy/dx, profiles at the integer
        // positions, summed in the order of the stencil table.
        for j in hz_rows {
            let (by, cy, iky) = (p.y_int.b[j], p.y_int.c[j], p.y_int.inv_kappa[j]);
            let jp = if j == 0 { ny - 1 } else { j - 1 };
            for i in 1..nx - 1 {
                let (bx, cx, ikx) = (p.x_int.b[i], p.x_int.c[i], p.x_int.inv_kappa[i]);
                let (k, down) = (j * nx + i, jp * nx + i);
                let px = &mut self.psi_hzx[k];
                *px = bx * *px + cx * (ey[k] - ey[k - 1]);
                let py = &mut self.psi_hzy[k];
                *py = by * *py + cy * (ex[k] - ex[down]);
                let curl =
                    -ey[k] * ikx + ey[k - 1] * ikx + ex[k] * iky - ex[down] * iky - self.psi_hzx[k]
                        + self.psi_hzy[k];
                hz[k] += s * curl;
            }
        }

        let t = (self.n as f64 + 0.5) * s;
        for src in &self.sources {
            let k = src.j * nx + src.i;
            let v = src.value_at(t);
            match src.kind {
                SourceKind::Soft => hz[k] += v,
                SourceKind::Hard => hz[k] = v,
            }
        }

        // Ex[i, j] at (i, j + 1/2): dHz/dy, y profile at j + 1/2.
        for j in ex_rows {
            let (b, c, ik) = (p.y_half.b[j], p.y_half.c[j], p.y_half.inv_kappa[j]);
            let jn = if j + 1 == ny { 0 } else { j + 1 };
            for i in 0..nx {
                let (k, up) = (j * nx + i, jn * nx + i);
                let psi = &mut self.psi_exy[k];
                *psi = b * *psi + c * (hz[up] - hz[k]);
                ex[k] = self.ca_x[k] * ex[k] + self.cb_x[k] * (hz[up] * ik - hz[k] * ik + *psi);
            }
        }
        // Ey[i, j] at (i + 1/2, j): -dHz/dx, x profile at i + 1/2.
        for j in 0..ny {
            for i in 0..nx - 1 {
                let (b, c, ik) = (p.x_half.b[i], p.x_half.c[i], p.x_half.inv_kappa[i]);
                let k = j * nx + i;
                let psi = &mut self.psi_eyx[k];
                *psi = b * *psi + c * (hz[k + 1] - hz[k]);
                ey[k] = self.ca_y[k] * ey[k] + self.cb_y[k] * (-hz[k + 1] * ik + hz[k] * ik - *psi);
            }
        }

        self.n += 1;
        for p in &mut self.probes {
            p.record(hz, nx);
        }
    }

    /// Advances `steps` full steps.
    pub fn step_n(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Replaces Hz with `values` (one per node, `j * nx + i`), forcing the
    /// ring nodes to zero. For initial conditions; E is left as it is.
    pub fn set_hz(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.hz.len() {
            return Err(FdtdError::InvalidParameter {
                name: "hz length",
                value: values.len() as f64,
            });
        }
        if let Some(&bad) = values.iter().find(|v| !v.is_finite()) {
            return Err(FdtdError::InvalidParameter {
                name: "hz value",
                value: bad,
            });
        }
        let (nx, ny) = (self.nx, self.ny);
        for j in 0..ny {
            for i in 0..nx {
                let ring = i == 0 || i == nx - 1 || (!self.periodic_y && (j == 0 || j == ny - 1));
                self.hz[j * nx + i] = if ring { 0.0 } else { values[j * nx + i] };
            }
        }
        Ok(())
    }

    /// Point probes in scene monitor order (Hz histories).
    pub fn probes(&self) -> &[PointProbe] {
        &self.probes
    }

    /// Hz on nodes, normalized, `nx * ny`, index `j * nx + i`.
    pub fn hz(&self) -> &[f64] {
        &self.hz
    }

    /// Ex on y-edges (i, j + 1/2), normalized, `nx * ny`. Row ny - 1 is
    /// padding, except with periodic y.
    pub fn ex(&self) -> &[f64] {
        &self.ex
    }

    /// Ey on x-edges (i + 1/2, j), normalized, `nx * ny` with column nx - 1 padding.
    pub fn ey(&self) -> &[f64] {
        &self.ey
    }

    /// Relative permittivity per Ex and per Ey edge (0 on PEC edges), for the energy.
    pub fn eps(&self) -> (&[f64], &[f64]) {
        (&self.eps_x, &self.eps_y)
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    pub fn periodic_y(&self) -> bool {
        self.periodic_y
    }

    /// The Courant number S, which is dt in normalized units.
    pub fn courant(&self) -> f64 {
        self.s
    }

    /// Number of full steps taken. E is at time n S, Hz at (n - 1/2) S.
    pub fn time_step(&self) -> u64 {
        self.n
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::grid::Component;
    use crate::scene::Shape;
    use crate::tm::tests::{CpmlFields, Fields, random_fields, random_profile, vacuum_box};

    pub(crate) fn vacuum_box_te(nx: usize, ny: usize, s: f64) -> Scene {
        let mut scene = vacuum_box(nx, ny, s);
        scene.mode = Mode::Te;
        scene
    }

    /// Oracle state: `f` holds Hz in the node slot, Ex and Ey in the edge
    /// slots (its ca, cb are unused); the edge coefficients sit here.
    struct TeOracle {
        f: CpmlFields,
        ca_x: Vec<f64>,
        cb_x: Vec<f64>,
        ca_y: Vec<f64>,
        cb_y: Vec<f64>,
    }

    impl TeOracle {
        /// One step from the stencil table, over the same index ranges as TMz.
        fn step(&mut self, s: f64, periodic: bool) {
            let (nx, ny) = (self.f.f.nx, self.f.f.ny);
            let (ex_rows, hz_rows) = if periodic {
                (0..ny, 0..ny)
            } else {
                (0..ny - 1, 1..ny - 1)
            };
            for j in hz_rows {
                for i in 1..nx - 1 {
                    let d = self.f.apply(Component::Hz, i, j);
                    self.f.f.ez[j * nx + i] += s * d;
                }
            }
            for j in ex_rows {
                for i in 0..nx {
                    let k = j * nx + i;
                    let d = self.f.apply(Component::Ex, i, j);
                    self.f.f.hx[k] = self.ca_x[k] * self.f.f.hx[k] + self.cb_x[k] * d;
                }
            }
            for j in 0..ny {
                for i in 0..nx - 1 {
                    let k = j * nx + i;
                    let d = self.f.apply(Component::Ey, i, j);
                    self.f.f.hy[k] = self.ca_y[k] * self.f.f.hy[k] + self.cb_y[k] * d;
                }
            }
        }
    }

    /// The hand-written loops equal the TEz stencil table bit for bit, with
    /// random fields, random per edge coefficients and, with `cpml`, random
    /// b, c, 1/kappa and psi everywhere. Non-square, so x and y cannot swap.
    fn oracle_case(seed: u64, cpml: bool, periodic: bool) {
        let (nx, ny, s) = (23, 17, 0.5);
        let r = |k: u64| random_fields(seed * 100 + k, nx * ny);
        let pos = |v: Vec<f64>| v.iter().map(|x| 0.5 * (x + 1.0)).collect::<Vec<_>>();
        let mut f = Fields {
            nx,
            ny,
            ez: r(1),
            hx: r(2),
            hy: r(3),
            ca: vec![0.0; nx * ny],
            cb: vec![0.0; nx * ny],
        };
        if !periodic {
            for i in 0..nx {
                f.hx[(ny - 1) * nx + i] = 0.0;
            }
        }
        for j in 0..ny {
            f.hy[j * nx + nx - 1] = 0.0;
        }
        let pml = if cpml {
            Cpml {
                x_int: random_profile(seed * 100 + 10, nx),
                x_half: random_profile(seed * 100 + 20, nx),
                y_int: random_profile(seed * 100 + 30, ny),
                y_half: random_profile(seed * 100 + 40, ny),
            }
        } else {
            Cpml::none(nx, ny)
        };
        let psi = if cpml {
            [r(6), r(7), r(8), r(9)]
        } else {
            std::array::from_fn(|_| vec![0.0; nx * ny])
        };
        let mut scene = vacuum_box_te(nx, ny, s);
        scene.periodic_y = periodic;
        let mut sim = SimTe::from_scene(&scene).unwrap();
        (sim.hz, sim.ex, sim.ey) = (f.ez.clone(), f.hx.clone(), f.hy.clone());
        (sim.ca_x, sim.cb_x) = (r(11), pos(r(12)));
        (sim.ca_y, sim.cb_y) = (r(13), pos(r(14)));
        sim.pml = pml.clone();
        // The oracle keeps psi in the curl-sign convention; the solver keeps
        // it on the plain difference, which is minus that for the two terms
        // with a minus sign in the table (Ey and the Ey part of Hz).
        let neg = |v: &[f64]| v.iter().map(|x| -x).collect::<Vec<_>>();
        sim.psi_exy = psi[0].clone();
        sim.psi_eyx = neg(&psi[1]);
        sim.psi_hzx = neg(&psi[2]);
        sim.psi_hzy = psi[3].clone();
        let mut o = TeOracle {
            ca_x: sim.ca_x.clone(),
            cb_x: sim.cb_x.clone(),
            ca_y: sim.ca_y.clone(),
            cb_y: sim.cb_y.clone(),
            f: CpmlFields { f, pml, psi },
        };
        for _ in 0..3 {
            sim.step();
            o.step(s, periodic);
        }
        let tag = format!("seed {seed} cpml {cpml} periodic {periodic}");
        assert_eq!(sim.hz, o.f.f.ez, "hz, {tag}");
        assert_eq!(sim.ex, o.f.f.hx, "ex, {tag}");
        assert_eq!(sim.ey, o.f.f.hy, "ey, {tag}");
        assert_eq!(sim.psi_exy, o.f.psi[0], "psi_exy, {tag}");
        assert_eq!(sim.psi_eyx, neg(&o.f.psi[1]), "psi_eyx, {tag}");
        assert_eq!(sim.psi_hzx, neg(&o.f.psi[2]), "psi_hzx, {tag}");
        assert_eq!(sim.psi_hzy, o.f.psi[3], "psi_hzy, {tag}");
    }

    #[test]
    fn stencil_oracle_matches_step() {
        for seed in 1..=4 {
            oracle_case(seed, false, false);
        }
    }

    #[test]
    fn stencil_oracle_with_cpml() {
        for seed in 1..=4 {
            oracle_case(seed, true, false);
        }
    }

    #[test]
    fn stencil_oracle_periodic_y() {
        for seed in 1..=4 {
            oracle_case(seed, false, true);
        }
    }

    #[test]
    fn refuses_a_tm_scene() {
        assert_eq!(
            SimTe::from_scene(&vacuum_box(10, 8, 0.5)).unwrap_err(),
            FdtdError::WrongMode { mode: Mode::Tm }
        );
    }

    /// Edges take the mean of their two cells, and an edge next to PEC is PEC.
    #[test]
    fn edge_materials() {
        let mut scene = vacuum_box_te(12, 10, 0.5);
        scene.materials.push(Material::Dielectric {
            eps_inf: 3.0,
            sigma: 0.0,
        });
        scene.materials.push(Material::Pec);
        // Cell (5, 4) is eps 3, cell (8, 4) is PEC.
        for (x, m) in [(5.0, 1), (8.0, 2)] {
            scene.geometry.push(Shape::Rect {
                x0: x - 0.25,
                y0: 3.75,
                x1: x + 0.25,
                y1: 4.25,
                material: m,
            });
        }
        let sim = SimTe::from_scene(&scene).unwrap();
        let nx = 12;
        let (ex, ey) = sim.eps();
        let k = |i: usize, j: usize| j * nx + i;
        assert_eq!(ey[k(4, 4)], 2.0, "x-edge between vacuum and eps 3");
        assert_eq!(ey[k(5, 4)], 2.0);
        assert_eq!(ex[k(5, 4)], 2.0, "y-edge above the eps 3 cell");
        assert_eq!(ex[k(5, 3)], 2.0, "y-edge below it");
        assert_eq!(ey[k(6, 4)], 1.0);
        for (i, j, e) in [(7, 4, ey), (8, 4, ey), (8, 4, ex), (8, 3, ex)] {
            assert_eq!(e[k(i, j)], 0.0, "PEC edge ({i}, {j})");
        }
        assert_eq!(sim.cb_y[k(7, 4)], 0.0);
        assert_eq!(sim.cb_y[k(6, 4)], 0.5);
    }
}
