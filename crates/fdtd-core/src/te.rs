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
//! step sees them. Probes, DFT monitors and flux lines read Hz on the nodes,
//! entered at its true time (n - 1/2) S.
//!
//! TF/SF runs the TMz plane wave (`tfsf.rs`) through duality, in a vacuum
//! background only (the scene validator enforces it). The auxiliary grid
//! gets the waveform delayed by S/2, because its Ez^m stands for the
//! incident Hz^(m - 1/2), and it is advanced at the start of each step. The
//! TMz E corrections, with Cb = S, then correct the Hz face nodes from
//! `incident_h`, and the TMz H corrections, with the sign flipped (E is
//! minus H under duality), correct the Ex and Ey edges from `incident_e`.

use crate::cpml::Cpml;
use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::kernel::{BuildKey, Kernel, Spectrum};
use crate::material::Material;
use crate::monitor::{Dft, FluxLine, PointProbe};
use crate::scene::{Mode, Monitor, Scene, Source};
use crate::source::SourceKind;
use crate::tfsf::Tfsf;
use crate::tm::Spectral;

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
    tfsf: Option<Tfsf>,
    /// S on every node: the Hz coefficient, for the TF/SF Hz corrections.
    node_s: Vec<f64>,
    spectral: Vec<Spectral>,
    key: BuildKey,
}

impl SimTe {
    /// Validates the scene, rasterizes its geometry, and precomputes the
    /// per edge Ca, Cb. All fields start at zero, at time step 0.
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        scene.validate()?;
        if scene.mode != Mode::Te {
            return Err(FdtdError::WrongMode { mode: scene.mode });
        }
        let (nx, ny, s) = (scene.grid.nx, scene.grid.ny, scene.grid.courant);
        let mut probes = Vec::new();
        let mut spectral = Vec::new();
        for m in &scene.monitors {
            match m {
                &Monitor::Point { i, j } => probes.push(PointProbe::new(i, j)),
                Monitor::Dft { i, j, freqs } => {
                    spectral.push(Spectral::Field(j * nx + i, Dft::new(freqs, 1)?))
                }
                Monitor::Flux {
                    normal,
                    at,
                    from,
                    to,
                    freqs,
                } => spectral.push(Spectral::Flux(
                    FluxLine::new(*normal, *at, *from, *to, freqs)?.in_mode(Mode::Te),
                )),
            }
        }
        // Validated: material 0 is vacuum when there is a plane wave.
        let tfsf = match &scene.plane_wave {
            Some(pw) => {
                let mut aux = *pw;
                aux.waveform = pw.waveform.delayed(0.5 * s);
                Some(Tfsf::new(&aux, 1.0, s)?)
            }
            None => None,
        };
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
            tfsf,
            node_s: vec![s; nx * ny],
            spectral,
            key: BuildKey::of(scene),
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

    /// Advances one full step: the TF/SF auxiliary grid, Hz to n + 1/2 and
    /// its TF/SF corrections, the point sources on Hz with the waveform at
    /// (n + 1/2) S, then E to n + 1 and its TF/SF corrections, then the
    /// monitors.
    pub fn step(&mut self) {
        let (nx, ny, s) = (self.nx, self.ny, self.s);
        if let Some(t) = &mut self.tfsf {
            t.advance();
        }
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
        if let Some(t) = &self.tfsf {
            t.correct_e(hz, &self.node_s, nx);
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
        if let Some(t) = &self.tfsf {
            let inc = t.incident_e();
            for c in t.h_corrections(nx, s, self.periodic_y).iter().flatten() {
                let field = if c.field == Component::Hx {
                    &mut *ex
                } else {
                    &mut *ey
                };
                field[c.k] -= c.factor * inc[c.src];
            }
        }

        self.n += 1;
        for p in &mut self.probes {
            p.record(hz, nx);
        }
        let mut spectral = std::mem::take(&mut self.spectral);
        let n = self.n;
        for m in &mut spectral {
            match m {
                Spectral::Field(k, dft) => dft.add(&[self.hz[*k]], (n as f64 - 0.5) * s, s),
                Spectral::Flux(line) => {
                    let values: Vec<f64> = line
                        .taps(nx, ny)
                        .iter()
                        .map(|&(c, k)| self.field(c).map_or(0.0, |f| f[k]))
                        .collect();
                    line.record_taps(&values, n, s);
                }
            }
        }
        self.spectral = spectral;
    }

    /// A TEz field by name, `None` for a TMz one.
    fn field(&self, c: Component) -> Option<&[f64]> {
        match c {
            Component::Hz => Some(&self.hz),
            Component::Ex => Some(&self.ex),
            Component::Ey => Some(&self.ey),
            Component::Ez | Component::Hx | Component::Hy => None,
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

impl Kernel for SimTe {
    fn from_scene(scene: &Scene) -> Result<Self> {
        SimTe::from_scene(scene)
    }

    fn step(&mut self, n: u64) {
        self.step_n(n);
    }

    fn time_step(&self) -> u64 {
        self.n
    }

    fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
        self.field(which)
            .map(<[f64]>::to_vec)
            .ok_or(FdtdError::FieldNotInMode { component: which })
    }

    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
        Ok(self.probes.iter().map(|p| p.history().to_vec()).collect())
    }

    /// Every edge is rebuilt from the edited scene: an edge reads the cells
    /// on both sides, so one just outside `bounds` can change too.
    /// ponytail: O(nx ny) per edit, fine at CPU fallback sizes; walk the
    /// bounds plus one cell if a large CPU grid ever paints live.
    fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        self.key.check(scene)?;
        if crate::kernel::material_edit(scene, bounds)?.is_none() {
            return Ok(());
        }
        self.set_edges(scene, &scene.rasterize());
        Ok(())
    }

    fn set_sources(&mut self, scene: &Scene) -> Result<()> {
        self.key.check(scene)?;
        for src in &scene.sources {
            scene.check_source(src)?;
        }
        self.sources = scene.sources.clone();
        Ok(())
    }

    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
        Ok(self.spectral.iter().map(Spectral::spectrum).collect())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::grid::Component;
    use crate::kernel::{Kernel, Spectrum};
    use crate::scene::Shape;
    use crate::source::Waveform;
    use crate::tfsf::{Direction, PlaneWave};
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

    /// A TMz scene with a plane wave, a probe inside and one outside the
    /// box, a DFT and a flux line, travelling `dir`.
    fn plane_wave_scene(dir: Direction, periodic: bool) -> Scene {
        let mut s = vacuum_box(81, 61, 0.5);
        s.pml = Some(crate::cpml::PmlSpec::recommended(10));
        s.periodic_y = periodic;
        let (y0, y1) = if periodic { (0, 60) } else { (16, 44) };
        s.plane_wave = Some(PlaneWave {
            x0: 20,
            y0,
            x1: 60,
            y1,
            direction: dir,
            waveform: Waveform::Ricker {
                t0: 40.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
        });
        s.monitors = vec![
            Monitor::Point { i: 40, j: 30 },
            Monitor::Point { i: 14, j: 30 },
            Monitor::Dft {
                i: 45,
                j: 33,
                freqs: vec![0.04, 0.05, 0.06],
            },
            Monitor::Flux {
                normal: crate::monitor::Normal::X,
                at: 50,
                from: 20,
                to: 40,
                freqs: vec![0.04, 0.05, 0.06],
            },
        ];
        s
    }

    /// TF/SF in TEz is the TMz plane wave under duality: Hz equals Ez bit for
    /// bit on every node for all four directions and with periodic y (so the
    /// scattered region really stays at round-off, as in TMz), with the TEz
    /// waveform moved S/2 earlier like A18. The flux spectra agree, and the
    /// DFT magnitudes too (the DFT phases differ by the half step).
    #[test]
    fn plane_wave_is_the_dual_of_tm() {
        let cases = [
            (Direction::PlusX, false),
            (Direction::MinusX, false),
            (Direction::PlusY, false),
            (Direction::MinusY, false),
            (Direction::PlusX, true),
        ];
        for (dir, periodic) in cases {
            let tm_scene = plane_wave_scene(dir, periodic);
            let mut te_scene = tm_scene.clone();
            te_scene.mode = Mode::Te;
            let pw = te_scene.plane_wave.as_mut().unwrap();
            pw.waveform = pw.waveform.delayed(-0.25);
            let mut tm = crate::tm::SimTm::from_scene(&tm_scene).unwrap();
            let mut te = SimTe::from_scene(&te_scene).unwrap();
            for _ in 0..8 {
                tm.step_n(100);
                te.step_n(100);
                assert_eq!(tm.ez(), te.hz(), "{dir:?} periodic {periodic}");
            }
            assert_eq!(
                tm.probes()[1].history(),
                te.probes()[1].history(),
                "{dir:?} outside probe"
            );
            let (a, b) = (tm.monitor_spectra().unwrap(), te.monitor_spectra().unwrap());
            let (Spectrum::Field(fa), Spectrum::Field(fb)) = (&a[0], &b[0]) else {
                panic!("dft order")
            };
            for (x, y) in fa.iter().zip(fb) {
                let (ma, mb) = (x.0.hypot(x.1), y.0.hypot(y.1));
                assert!((ma - mb).abs() <= 1e-12 * ma, "{dir:?} dft {ma} vs {mb}");
            }
            let (Spectrum::Flux(xa), Spectrum::Flux(xb)) = (&a[1], &b[1]) else {
                panic!("flux order")
            };
            for (x, y) in xa.iter().zip(xb) {
                let scale = xa.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                // y travel puts no net flux through this x line: round-off only.
                assert!(
                    (x - y).abs() <= 1e-6 * scale + 1e-12,
                    "{dir:?} flux {x} vs {y}"
                );
            }
        }
    }

    /// Painting live equals building the painted scene, in the closed box
    /// and with strokes running into a CPML; a mode change needs a rebuild.
    #[test]
    fn live_paint_equals_painted_scene() {
        let mut a = crate::scene::tests::sample();
        a.mode = Mode::Te;
        let mut b = a.clone();
        b.geometry.clear();
        b.pml = Some(crate::cpml::tests::spec(5));
        for base in [a, b] {
            let mut painted = base.clone();
            let out = painted
                .paint_stroke((0.0, 9.0), (20.0, 9.0), 1.5, 2)
                .unwrap();
            let mut live = SimTe::from_scene(&base).unwrap();
            live.step_n(40);
            live.set_materials(&painted, out.bounds.unwrap()).unwrap();
            let mut fresh = SimTe::from_scene(&painted).unwrap();
            fresh.step_n(40);
            // Same fields after the edit: restart both from the live state.
            (fresh.hz, fresh.ex, fresh.ey) = (live.hz.clone(), live.ex.clone(), live.ey.clone());
            (fresh.psi_exy, fresh.psi_eyx) = (live.psi_exy.clone(), live.psi_eyx.clone());
            (fresh.psi_hzx, fresh.psi_hzy) = (live.psi_hzx.clone(), live.psi_hzy.clone());
            live.step_n(200);
            fresh.step_n(200);
            assert_eq!(live.hz, fresh.hz);
            assert_eq!(live.eps(), fresh.eps());
            let mut tm = base.clone();
            tm.mode = Mode::Tm;
            assert_eq!(live.set_sources(&tm), Err(FdtdError::Rebuild));
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
