//! 2D TMz reference solver (Ez, Hx, Hy). Plain f64 loops, normalized units:
//! dx = dy = 1, c0 = eps0 = mu0 = 1, dt = S.
//!
//! Layout: all three fields are `nx * ny` arrays indexed `j * nx + i`, padded
//! like the GPU buffers. Ez[i, j] is node (i, j); Hx[i, j] is the y-edge at
//! (i, j + 1/2) and its row j = ny - 1 is padding; Hy[i, j] is the x-edge at
//! (i + 1/2, j) and its column i = nx - 1 is padding. Padding is never
//! written and stays zero.
//!
//! Stability (von Neumann): a plane wave `exp(j(w t - kx x - ky y))` on this
//! grid obeys `(sin(w S / 2) / S)^2 = sin^2(kx / 2) + sin^2(ky / 2)`. The right
//! side reaches 2 at kx = ky = pi, so a real w exists for every grid mode only
//! if `S^2 * 2 <= 1`, that is `S <= 1/sqrt(2)`. Above it the checkerboard mode
//! has a complex w and grows every step.
//!
//! The outer ring of Ez nodes is a PEC wall: the E loop covers interior nodes
//! only, so those values never leave zero. PEC material nodes have Ca = Cb = 0.
//!
//! CPML (`cpml.rs`) runs through the same loops. Each difference term is
//! scaled by the 1/kappa of its axis, and the psi of each derivative is added
//! on. Outside the layer 1/kappa = 1 and b = c = 0, so psi stays zero and every
//! operation reduces exactly to the plain Yee update, in the same order. With
//! no layer the result is bit for bit the closed-box solver. psi arrays are
//! full `nx * ny` and named for the field and the derivative axis:
//! `psi_hxy` (Hx, d/dy), `psi_hyx` (Hy, d/dx), `psi_ezx`, `psi_ezy`.

use crate::cpml::Cpml;
use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::kernel::Kernel;
use crate::monitor::PointProbe;
use crate::scene::{Monitor, Scene, Source};
use crate::source::SourceKind;

/// TMz grid state built from a validated scene.
#[derive(Debug, Clone)]
pub struct SimTm {
    nx: usize,
    ny: usize,
    s: f64,
    n: u64,
    ez: Vec<f64>,
    hx: Vec<f64>,
    hy: Vec<f64>,
    ca: Vec<f64>,
    cb: Vec<f64>,
    eps: Vec<f64>,
    pml: Cpml,
    psi_hxy: Vec<f64>,
    psi_hyx: Vec<f64>,
    psi_ezx: Vec<f64>,
    psi_ezy: Vec<f64>,
    sources: Vec<Source>,
    probes: Vec<PointProbe>,
}

impl SimTm {
    /// Validates the scene, rasterizes its geometry, and precomputes the per
    /// node Ca, Cb. All fields start at zero, at time step 0.
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        scene.validate()?;
        Ok(Self::build(scene))
    }

    /// Same as `from_scene` but accepts a Courant number above the 2D limit.
    /// Only the stability test uses it.
    #[cfg(test)]
    pub(crate) fn from_scene_unchecked(scene: &Scene) -> Result<Self> {
        scene.validate_inner(false)?;
        Ok(Self::build(scene))
    }

    fn build(scene: &Scene) -> Self {
        let (nx, ny, s) = (scene.grid.nx, scene.grid.ny, scene.grid.courant);
        let ids = scene.rasterize();
        let mut ca = vec![0.0; nx * ny];
        let mut cb = vec![0.0; nx * ny];
        let mut eps = vec![0.0; nx * ny];
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let k = j * nx + i;
                let m = scene.materials[ids[k]];
                (ca[k], cb[k]) = m.coefficients(s);
                eps[k] = m.eps();
            }
        }
        let mut probes: Vec<PointProbe> = scene
            .monitors
            .iter()
            .map(|&Monitor::Point { i, j }| PointProbe::new(i, j))
            .collect();
        let ez = vec![0.0; nx * ny];
        for p in &mut probes {
            p.record(&ez, nx);
        }
        // sigma_opt is set for the background material; a PEC background
        // has no wave impedance, so it falls back to vacuum.
        let eps_r = scene.materials[0].eps().max(1.0);
        let pml = match &scene.pml {
            Some(spec) => Cpml::new(spec, nx, ny, s, eps_r),
            None => Cpml::none(nx, ny),
        };
        Self {
            nx,
            ny,
            s,
            n: 0,
            ez,
            hx: vec![0.0; nx * ny],
            hy: vec![0.0; nx * ny],
            ca,
            cb,
            eps,
            pml,
            psi_hxy: vec![0.0; nx * ny],
            psi_hyx: vec![0.0; nx * ny],
            psi_ezx: vec![0.0; nx * ny],
            psi_ezy: vec![0.0; nx * ny],
            sources: scene.sources.clone(),
            probes,
        }
    }

    /// Advances one full step: H to n + 1/2, E to n + 1, then the sources,
    /// which use the waveform at t = (n + 1) * S, then the probes. Soft
    /// sources add to Ez, hard sources overwrite it.
    pub fn step(&mut self) {
        let (nx, ny, s) = (self.nx, self.ny, self.s);
        let (ez, hx, hy) = (&mut self.ez, &mut self.hx, &mut self.hy);
        let p = &self.pml;

        // Hx[i, j] at (i, j + 1/2): minus the y-difference of Ez. y profile
        // at the half position j + 1/2.
        for j in 0..ny - 1 {
            let (b, c, ik) = (p.y_half.b[j], p.y_half.c[j], p.y_half.inv_kappa[j]);
            for i in 0..nx {
                let k = j * nx + i;
                let psi = &mut self.psi_hxy[k];
                *psi = b * *psi + c * (ez[k + nx] - ez[k]);
                hx[k] -= s * (ez[k + nx] * ik - ez[k] * ik + *psi);
            }
        }
        // Hy[i, j] at (i + 1/2, j): the x-difference of Ez. x profile at the
        // half position i + 1/2.
        for j in 0..ny {
            for i in 0..nx - 1 {
                let (b, c, ik) = (p.x_half.b[i], p.x_half.c[i], p.x_half.inv_kappa[i]);
                let k = j * nx + i;
                let psi = &mut self.psi_hyx[k];
                *psi = b * *psi + c * (ez[k + 1] - ez[k]);
                hy[k] += s * (ez[k + 1] * ik - ez[k] * ik + *psi);
            }
        }
        // Ez[i, j] at (i, j), interior only, profiles at the integer
        // positions. Summed in the order of the stencil table so the oracle
        // tests can compare bit for bit.
        for j in 1..ny - 1 {
            let (by, cy, iky) = (p.y_int.b[j], p.y_int.c[j], p.y_int.inv_kappa[j]);
            for i in 1..nx - 1 {
                let (bx, cx, ikx) = (p.x_int.b[i], p.x_int.c[i], p.x_int.inv_kappa[i]);
                let k = j * nx + i;
                let px = &mut self.psi_ezx[k];
                *px = bx * *px + cx * (hy[k] - hy[k - 1]);
                let py = &mut self.psi_ezy[k];
                *py = by * *py + cy * (hx[k] - hx[k - nx]);
                let curl = hy[k] * ikx - hy[k - 1] * ikx - hx[k] * iky
                    + hx[k - nx] * iky
                    + self.psi_ezx[k]
                    - self.psi_ezy[k];
                ez[k] = self.ca[k] * ez[k] + self.cb[k] * curl;
            }
        }

        self.n += 1;
        let t = self.n as f64 * s;
        for src in &self.sources {
            let k = src.j * nx + src.i;
            let v = src.amplitude * src.waveform.value(t);
            match src.kind {
                SourceKind::Soft => ez[k] += v,
                SourceKind::Hard => ez[k] = v,
            }
        }
        for p in &mut self.probes {
            p.record(ez, nx);
        }
    }

    /// Advances `steps` full steps.
    pub fn step_n(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Point probes in scene monitor order.
    pub fn probes(&self) -> &[PointProbe] {
        &self.probes
    }

    /// Replaces Ez with `values` (one per node, `j * nx + i`), forcing wall and
    /// PEC nodes to zero. For initial conditions; H is left as it is.
    pub fn set_ez(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.ez.len() {
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
        for (k, (e, &v)) in self.ez.iter_mut().zip(values).enumerate() {
            // Only walls and PEC nodes have cb = 0; every dielectric has cb > 0.
            *e = if self.cb[k] == 0.0 { 0.0 } else { v };
        }
        Ok(())
    }

    /// Ez at nodes, normalized, `nx * ny`, index `j * nx + i`.
    pub fn ez(&self) -> &[f64] {
        &self.ez
    }

    /// Hx on y-edges (i, j + 1/2), normalized, `nx * ny` with row ny - 1 padding.
    pub fn hx(&self) -> &[f64] {
        &self.hx
    }

    /// Hy on x-edges (i + 1/2, j), normalized, `nx * ny` with column nx - 1 padding.
    pub fn hy(&self) -> &[f64] {
        &self.hy
    }

    /// Relative permittivity per node (0 on walls and PEC), for the energy.
    pub fn eps(&self) -> &[f64] {
        &self.eps
    }

    pub fn nx(&self) -> usize {
        self.nx
    }

    pub fn ny(&self) -> usize {
        self.ny
    }

    /// The Courant number S, which is dt in normalized units.
    pub fn courant(&self) -> f64 {
        self.s
    }

    /// Number of full steps taken. E is at time n S, H at (n - 1/2) S.
    pub fn time_step(&self) -> u64 {
        self.n
    }

    /// True on wall nodes and PEC material nodes of node (i, j).
    pub fn is_pec(&self, i: usize, j: usize) -> bool {
        self.cb[j * self.nx + i] == 0.0
    }
}

impl Kernel for SimTm {
    fn from_scene(scene: &Scene) -> Result<Self> {
        SimTm::from_scene(scene)
    }

    fn step(&mut self, n: u64) {
        self.step_n(n);
    }

    fn time_step(&self) -> u64 {
        self.n
    }

    fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
        Ok(match which {
            Component::Ez => self.ez.clone(),
            Component::Hx => self.hx.clone(),
            Component::Hy => self.hy.clone(),
        })
    }

    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
        Ok(self.probes.iter().map(|p| p.history().to_vec()).collect())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::cpml::{Cpml, Profile};
    use crate::grid::{Axis, Component, curl_stencil_tm};
    use crate::material::Material;
    use crate::scene::{GridSpec, SCENE_VERSION, Shape};
    use crate::source::Waveform;

    /// An empty vacuum box with PEC walls.
    pub(crate) fn vacuum_box(nx: usize, ny: usize, s: f64) -> Scene {
        Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx,
                ny,
                dx_phys_m: 1e-3,
                courant: s,
            },
            materials: vec![Material::VACUUM],
            geometry: vec![],
            sources: vec![],
            monitors: vec![],
            pml: None,
        }
    }

    /// Deterministic values in [-1, 1) from a 64-bit LCG.
    pub(crate) fn random_fields(seed: u64, n: usize) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    /// Fields and coefficients for the oracle step, padded like `SimTm`.
    struct Fields {
        nx: usize,
        ny: usize,
        ez: Vec<f64>,
        hx: Vec<f64>,
        hy: Vec<f64>,
        ca: Vec<f64>,
        cb: Vec<f64>,
    }

    impl Fields {
        fn apply(&self, target: Component, i: usize, j: usize) -> f64 {
            curl_stencil_tm(target)
                .iter()
                .map(|t| {
                    let src = match t.src {
                        Component::Ez => &self.ez,
                        Component::Hx => &self.hx,
                        Component::Hy => &self.hy,
                    };
                    let (si, sj) = ((i as i64 + t.di) as usize, (j as i64 + t.dj) as usize);
                    t.sign * src[sj * self.nx + si]
                })
                .sum()
        }

        /// One step driven by the stencil table in `grid.rs` over the
        /// meaningful entries of each padded array, Ez on interior nodes only.
        fn stencil_step(&mut self, s: f64) {
            let (nx, ny) = (self.nx, self.ny);
            for j in 0..ny - 1 {
                for i in 0..nx {
                    let d = self.apply(Component::Hx, i, j);
                    self.hx[j * nx + i] += s * d;
                }
            }
            for j in 0..ny {
                for i in 0..nx - 1 {
                    let d = self.apply(Component::Hy, i, j);
                    self.hy[j * nx + i] += s * d;
                }
            }
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let k = j * nx + i;
                    let d = self.apply(Component::Ez, i, j);
                    self.ez[k] = self.ca[k] * self.ez[k] + self.cb[k] * d;
                }
            }
        }
    }

    /// Random b in [0, 1), c in (-1, 0], 1/kappa in (0.1, 1], `n` entries.
    fn random_profile(seed: u64, n: usize) -> Profile {
        let r = |k| random_fields(seed + k, n);
        Profile {
            b: r(1).iter().map(|v| 0.5 * (v + 1.0)).collect(),
            c: r(2).iter().map(|v| -0.5 * (v + 1.0)).collect(),
            inv_kappa: r(3).iter().map(|v| 1.0 - 0.45 * (v + 1.0)).collect(),
        }
    }

    /// CPML oracle, written from doc 02 and the stencil table. Every term
    /// `sign * src` is scaled by the 1/kappa of its own axis. psi is kept in
    /// the curl-sign convention: with `D` the signed stencil sum along an
    /// axis, `psi = b psi + c D` and `update = coef * (sum(term / kappa) + psi)`.
    /// The solver keeps psi on the plain difference, which for the two
    /// y-derivatives (Hx and the Hx part of Ez) is exactly minus this one.
    struct CpmlFields {
        f: Fields,
        pml: Cpml,
        /// Hx along y, Hy along x, Ez along x, Ez along y.
        psi: [Vec<f64>; 4],
    }

    impl CpmlFields {
        fn slot(target: Component, axis: Axis) -> usize {
            match (target, axis) {
                (Component::Hx, _) => 0,
                (Component::Hy, _) => 1,
                (Component::Ez, Axis::X) => 2,
                (Component::Ez, Axis::Y) => 3,
            }
        }

        /// The profile entry a (target, axis) pair at (i, j) uses. H sits half
        /// a cell off the node along its derivative axis, Ez sits on the node.
        /// x is indexed by i, y by j.
        fn coef(&self, target: Component, axis: Axis, i: usize, j: usize) -> (f64, f64, f64) {
            let (p, k) = match (target, axis) {
                (Component::Ez, Axis::X) => (&self.pml.x_int, i),
                (Component::Ez, Axis::Y) => (&self.pml.y_int, j),
                (_, Axis::X) => (&self.pml.x_half, i),
                (_, Axis::Y) => (&self.pml.y_half, j),
            };
            (p.b[k], p.c[k], p.inv_kappa[k])
        }

        fn apply(&mut self, target: Component, i: usize, j: usize) -> f64 {
            let nx = self.f.nx;
            let (mut d, mut used, mut sum) = ([0.0; 2], [false; 2], 0.0);
            for t in curl_stencil_tm(target) {
                let src = match t.src {
                    Component::Ez => &self.f.ez,
                    Component::Hx => &self.f.hx,
                    Component::Hy => &self.f.hy,
                };
                let (si, sj) = ((i as i64 + t.di) as usize, (j as i64 + t.dj) as usize);
                let v = t.sign * src[sj * nx + si];
                sum += v * self.coef(target, t.axis, i, j).2;
                d[t.axis as usize] += v;
                used[t.axis as usize] = true;
            }
            for axis in [Axis::X, Axis::Y] {
                if used[axis as usize] {
                    let (b, c, _) = self.coef(target, axis, i, j);
                    let psi = &mut self.psi[Self::slot(target, axis)][j * nx + i];
                    *psi = b * *psi + c * d[axis as usize];
                    sum += *psi;
                }
            }
            sum
        }

        fn step(&mut self, s: f64) {
            let (nx, ny) = (self.f.nx, self.f.ny);
            for j in 0..ny - 1 {
                for i in 0..nx {
                    let d = self.apply(Component::Hx, i, j);
                    self.f.hx[j * nx + i] += s * d;
                }
            }
            for j in 0..ny {
                for i in 0..nx - 1 {
                    let d = self.apply(Component::Hy, i, j);
                    self.f.hy[j * nx + i] += s * d;
                }
            }
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let k = j * nx + i;
                    let d = self.apply(Component::Ez, i, j);
                    self.f.ez[k] = self.f.ca[k] * self.f.ez[k] + self.f.cb[k] * d;
                }
            }
        }
    }

    /// The CPML update in `step` equals the oracle bit for bit, with random
    /// fields, random psi, random Ca and Cb, and random b, c, 1/kappa at
    /// every position, on a non-square grid so x and y cannot be swapped.
    #[test]
    fn stencil_oracle_with_cpml() {
        let (nx, ny, s) = (23, 17, 0.5);
        for seed in 1..=4 {
            let r = |k: u64| random_fields(seed * 100 + k, nx * ny);
            let mut f = Fields {
                nx,
                ny,
                ez: r(1),
                hx: r(2),
                hy: r(3),
                ca: r(4),
                cb: r(5).iter().map(|v| 0.5 * (v + 1.0)).collect(),
            };
            for i in 0..nx {
                f.hx[(ny - 1) * nx + i] = 0.0;
            }
            for j in 0..ny {
                f.hy[j * nx + nx - 1] = 0.0;
            }
            let pml = Cpml {
                x_int: random_profile(seed * 100 + 10, nx),
                x_half: random_profile(seed * 100 + 20, nx),
                y_int: random_profile(seed * 100 + 30, ny),
                y_half: random_profile(seed * 100 + 40, ny),
            };
            let psi = [r(6), r(7), r(8), r(9)];

            let mut sim = SimTm::from_scene(&vacuum_box(nx, ny, s)).unwrap();
            (sim.ez, sim.hx, sim.hy) = (f.ez.clone(), f.hx.clone(), f.hy.clone());
            (sim.ca, sim.cb) = (f.ca.clone(), f.cb.clone());
            sim.pml = pml.clone();
            let neg = |v: &[f64]| v.iter().map(|x| -x).collect::<Vec<_>>();
            sim.psi_hxy = neg(&psi[0]);
            sim.psi_hyx = psi[1].clone();
            sim.psi_ezx = psi[2].clone();
            sim.psi_ezy = neg(&psi[3]);
            let mut o = CpmlFields { f, pml, psi };

            for _ in 0..3 {
                sim.step();
                o.step(s);
            }
            assert_eq!(sim.hx, o.f.hx, "hx differs, seed {seed}");
            assert_eq!(sim.hy, o.f.hy, "hy differs, seed {seed}");
            assert_eq!(sim.ez, o.f.ez, "ez differs, seed {seed}");
            assert_eq!(sim.psi_hxy, neg(&o.psi[0]), "psi_hxy, seed {seed}");
            assert_eq!(sim.psi_hyx, o.psi[1], "psi_hyx, seed {seed}");
            assert_eq!(sim.psi_ezx, o.psi[2], "psi_ezx, seed {seed}");
            assert_eq!(sim.psi_ezy, neg(&o.psi[3]), "psi_ezy, seed {seed}");
        }
    }

    /// The hand-written loops must equal the stencil table exactly, with
    /// random fields and random per node coefficients. This proves the loop
    /// indices match the documented staggering. Built with no PML, so this
    /// also proves the CPML update is bit for bit the Phase 1 update when the
    /// profiles are the identity.
    #[test]
    fn stencil_oracle_matches_step() {
        let (nx, ny, s) = (23, 17, 0.5);
        for seed in 1..=4 {
            let mut f = Fields {
                nx,
                ny,
                ez: random_fields(seed, nx * ny),
                hx: random_fields(seed + 10, nx * ny),
                hy: random_fields(seed + 20, nx * ny),
                ca: random_fields(seed + 30, nx * ny),
                cb: random_fields(seed + 40, nx * ny)
                    .iter()
                    .map(|v| 0.5 * (v + 1.0))
                    .collect(),
            };
            for i in 0..nx {
                f.hx[(ny - 1) * nx + i] = 0.0;
            }
            for j in 0..ny {
                f.hy[j * nx + nx - 1] = 0.0;
            }
            let mut sim = SimTm::from_scene(&vacuum_box(nx, ny, s)).unwrap();
            (sim.ez, sim.hx, sim.hy) = (f.ez.clone(), f.hx.clone(), f.hy.clone());
            (sim.ca, sim.cb) = (f.ca.clone(), f.cb.clone());

            for _ in 0..3 {
                sim.step();
                f.stencil_step(s);
            }
            assert_eq!(sim.hx, f.hx, "hx differs, seed {seed}");
            assert_eq!(sim.hy, f.hy, "hy differs, seed {seed}");
            assert_eq!(sim.ez, f.ez, "ez differs, seed {seed}");
        }
    }

    #[test]
    fn courant_guard() {
        for bad in [0.7072, 1.0] {
            assert!(matches!(
                SimTm::from_scene(&vacuum_box(10, 10, bad)),
                Err(FdtdError::CourantAboveLimit { .. })
            ));
        }
        assert!(SimTm::from_scene(&vacuum_box(10, 10, std::f64::consts::FRAC_1_SQRT_2)).is_ok());
        assert!(SimTm::from_scene_unchecked(&vacuum_box(10, 10, 0.72)).is_ok());
    }

    #[test]
    fn coefficients_per_node() {
        let mut scene = vacuum_box(12, 10, 0.5);
        scene.materials.push(Material::Dielectric {
            eps_inf: 4.0,
            sigma: 0.0,
        });
        scene.materials.push(Material::Pec);
        scene.geometry = vec![
            Shape::Rect {
                x0: 2.0,
                y0: 2.0,
                x1: 3.0,
                y1: 3.0,
                material: 1,
            },
            Shape::Rect {
                x0: 7.0,
                y0: 7.0,
                x1: 7.0,
                y1: 7.0,
                material: 2,
            },
        ];
        let sim = SimTm::from_scene(&scene).unwrap();
        let at = |i: usize, j: usize| {
            let k = j * 12 + i;
            (sim.ca[k], sim.cb[k], sim.eps[k])
        };
        assert_eq!(at(5, 5), (1.0, 0.5, 1.0));
        assert_eq!(at(2, 3), (1.0, 0.125, 4.0));
        assert_eq!(at(7, 7), (0.0, 0.0, 0.0));
        assert_eq!(at(0, 5), (0.0, 0.0, 0.0));
        assert_eq!(at(11, 9), (0.0, 0.0, 0.0));
        assert!(sim.is_pec(7, 7) && sim.is_pec(0, 0) && !sim.is_pec(2, 3));
    }

    #[test]
    fn hard_source_sets_value() {
        let mut scene = vacuum_box(20, 20, 0.5);
        let w = Waveform::Gaussian { t0: 3.0, tau: 1.0 };
        scene.sources.push(Source {
            i: 7,
            j: 9,
            waveform: w,
            amplitude: 2.0,
            kind: SourceKind::Hard,
        });
        let mut sim = SimTm::from_scene(&scene).unwrap();
        for n in 1..=12 {
            sim.step();
            assert_eq!(sim.ez()[9 * 20 + 7], 2.0 * w.value(n as f64 * 0.5));
        }
    }

    /// Walls, PEC nodes, and padding never leave zero while a source rings
    /// right next to a PEC node.
    #[test]
    fn pec_and_padding_stay_zero() {
        let (nx, ny) = (16, 14);
        let mut scene = vacuum_box(nx, ny, 0.7);
        scene.materials.push(Material::Pec);
        scene.geometry.push(Shape::Rect {
            x0: 5.0,
            y0: 5.0,
            x1: 6.0,
            y1: 8.0,
            material: 1,
        });
        scene.sources.push(Source {
            i: 4,
            j: 6,
            waveform: Waveform::Ricker {
                t0: 20.0,
                f_peak: 0.1,
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        });
        let mut sim = SimTm::from_scene(&scene).unwrap();
        let mut seen = 0.0_f64;
        for _ in 0..400 {
            sim.step();
            seen = seen.max(sim.ez().iter().fold(0.0, |m, v| m.max(v.abs())));
            for j in 0..ny {
                for i in 0..nx {
                    let k = j * nx + i;
                    if sim.is_pec(i, j) {
                        assert_eq!(sim.ez()[k], 0.0, "ez at ({i}, {j})");
                    }
                    if j == ny - 1 {
                        assert_eq!(sim.hx()[k], 0.0, "hx padding at ({i}, {j})");
                    }
                    if i == nx - 1 {
                        assert_eq!(sim.hy()[k], 0.0, "hy padding at ({i}, {j})");
                    }
                }
            }
        }
        assert!(seen > 0.1, "the source must actually drive the grid");
    }

    #[test]
    fn set_ez_masks_pec_and_checks_input() {
        let mut sim = SimTm::from_scene(&vacuum_box(6, 5, 0.5)).unwrap();
        sim.set_ez(&[1.0; 30]).unwrap();
        assert_eq!(sim.ez()[0], 0.0);
        assert_eq!(sim.ez()[2 * 6 + 2], 1.0);
        assert!(sim.set_ez(&[1.0; 29]).is_err());
        let mut bad = [1.0; 30];
        bad[3] = f64::NAN;
        assert!(sim.set_ez(&bad).is_err());
    }

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    /// A4. A random field in a 32 x 32 PEC box at S = 0.70 stays bounded for
    /// 20000 steps. At S = 0.72, above 1/sqrt(2), the highest box mode has
    /// `2 S^2 (sin^2(kx/2) + sin^2(ky/2)) > 2`, so it grows by about 1.44 per
    /// step, and the field passes a million times its start within 2000 steps.
    #[test]
    fn a4_stability_bounded_at_070_unbounded_at_072() {
        let n = 32;
        let start = random_fields(7, n * n);
        let peak0 = max_abs(&start);

        let mut sim = SimTm::from_scene(&vacuum_box(n, n, 0.70)).unwrap();
        sim.set_ez(&start).unwrap();
        let mut worst = 0.0_f64;
        for _ in 0..20_000 {
            sim.step();
            worst = worst.max(max_abs(sim.ez()));
        }
        assert!(worst < 10.0 * peak0, "S = 0.70: max |ez| {worst}");

        let mut sim = SimTm::from_scene_unchecked(&vacuum_box(n, n, 0.72)).unwrap();
        sim.set_ez(&start).unwrap();
        let mut steps = 0;
        while max_abs(sim.ez()) <= 1e6 * peak0 {
            assert!(steps < 2000, "S = 0.72 did not blow up in 2000 steps");
            sim.step();
            steps += 1;
        }
        eprintln!(
            "A4: S = 0.70 max {worst:.3} (start {peak0:.3}), S = 0.72 passed 1e6x at step {steps}"
        );
    }

    /// Drives any kernel through the trait only.
    fn run_through_trait<K: Kernel>(scene: &Scene, steps: u64) -> (u64, Vec<f64>, Vec<Vec<f64>>) {
        let mut k = K::from_scene(scene).unwrap();
        k.step(steps);
        (
            k.time_step(),
            k.read_field(Component::Ez).unwrap(),
            k.probe_history().unwrap(),
        )
    }

    /// The trait path equals the direct path, and probe entry n is Ez at the
    /// probe node after n steps.
    #[test]
    fn kernel_trait_and_probes() {
        let mut scene = vacuum_box(20, 16, 0.5);
        scene.sources.push(Source {
            i: 5,
            j: 6,
            waveform: Waveform::Ricker {
                t0: 15.0,
                f_peak: 0.1,
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        });
        scene.monitors = vec![
            Monitor::Point { i: 12, j: 9 },
            Monitor::Point { i: 5, j: 6 },
        ];

        let mut sim = SimTm::from_scene(&scene).unwrap();
        let mut expect = vec![vec![0.0], vec![0.0]];
        for _ in 0..60 {
            sim.step();
            expect[0].push(sim.ez()[9 * 20 + 12]);
            expect[1].push(sim.ez()[6 * 20 + 5]);
        }
        let (n, ez, probes) = run_through_trait::<SimTm>(&scene, 60);
        assert_eq!(n, 60);
        assert_eq!(ez, sim.ez());
        assert_eq!(probes, expect);
        assert_eq!(sim.probes()[0].history(), &expect[0][..]);
        assert!(expect[0].iter().any(|v| *v != 0.0));
    }
}
