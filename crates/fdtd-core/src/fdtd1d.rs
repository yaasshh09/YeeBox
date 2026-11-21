//! 1D reference solver: TMz restricted to d/dy = 0, so the fields are `ez` on
//! nodes (x = i) and `hy` on x-edges (x = i + 1/2). Normalized units, a
//! uniform lossless background of relative permittivity eps (vacuum unless
//! `set_eps` says otherwise). It is also the TF/SF auxiliary grid.
//!
//! The update is written in the same operation order as the 2D TMz loops
//! (`tm.rs`), CPML terms included, so a y-uniform 2D field reproduces it bit
//! for bit. Outside a layer every extra term is an exact 1 or 0, so a layer
//! free vacuum grid is bit for bit the plain `ez += S dH`, `hy += S dE` scheme.

use crate::cpml::{Cpml, PmlSpec, Profile};
use crate::error::{FdtdError, Result};
use crate::source::{SourceKind, Waveform};

/// Treatment of one end node of the 1D grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundary1d {
    /// Perfect electric conductor: `ez` at the end node is held at zero.
    Pec,
    /// First-order Mur absorbing end:
    /// `E0^{n+1} = E1^n + (S v - 1)/(S v + 1) * (E1^{n+1} - E0^n)`, with
    /// `v = 1 / sqrt(eps)`. Exact at S v = 1, reflects a little below it.
    Mur1,
    /// A CPML layer of `thickness` cells in front of a PEC end node, graded
    /// like the 2D layer (`cpml.rs`) for the grid's background eps.
    Cpml(PmlSpec),
}

/// A point source on an `ez` node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Source1d {
    /// Node index of the driven `ez` sample (x = node).
    pub node: usize,
    pub waveform: Waveform,
    /// Multiplies the waveform. Normalized field units.
    pub amplitude: f64,
    pub kind: SourceKind,
}

/// 1D Yee grid state.
#[derive(Debug, Clone)]
pub struct Sim1d {
    ez: Vec<f64>,
    hy: Vec<f64>,
    s: f64,
    n: u64,
    left: Boundary1d,
    right: Boundary1d,
    sources: Vec<Source1d>,
    /// Background relative permittivity and its `Cb = S / eps`.
    eps: f64,
    cb: f64,
    /// CPML profiles at nodes (x = i, for `ez`) and at x = i + 1/2 (`hy`),
    /// the identity outside the layers, and the psi memories.
    x_int: Profile,
    x_half: Profile,
    psi_e: Vec<f64>,
    psi_h: Vec<f64>,
}

impl Sim1d {
    /// Builds a vacuum grid of `nx` `ez` nodes and `nx - 1` `hy` x-edges, all
    /// fields zero. `s` is the Courant number, which must satisfy 0 < S <= 1.
    /// A CPML end must be valid and leave at least 3 nodes outside the
    /// layers (`nx >= 2 d + 3`, the same fit rule as the 2D grid).
    pub fn new(nx: usize, s: f64, left: Boundary1d, right: Boundary1d) -> Result<Self> {
        if nx < 3 {
            return Err(FdtdError::InvalidParameter {
                name: "nx",
                value: nx as f64,
            });
        }
        if !(s.is_finite() && s > 0.0) {
            return Err(FdtdError::InvalidParameter {
                name: "courant",
                value: s,
            });
        }
        if s > 1.0 {
            return Err(FdtdError::CourantAboveLimit { s, limit: 1.0 });
        }
        for end in [left, right] {
            if let Boundary1d::Cpml(spec) = end {
                spec.validate()?;
                if nx < 2 * spec.thickness + 3 {
                    return Err(FdtdError::PmlTooThick {
                        thickness: spec.thickness,
                        nx,
                        ny: 1,
                    });
                }
            }
        }
        let id = Cpml::none(nx, 1);
        let mut sim = Self {
            ez: vec![0.0; nx],
            hy: vec![0.0; nx - 1],
            s,
            n: 0,
            left,
            right,
            sources: Vec::new(),
            eps: 1.0,
            cb: s,
            x_int: id.x_int,
            x_half: id.x_half,
            psi_e: vec![0.0; nx],
            psi_h: vec![0.0; nx - 1],
        };
        sim.build_profiles();
        Ok(sim)
    }

    /// Sets the background relative permittivity (finite, at least 1) and
    /// regrades any CPML end for it. Call before stepping.
    pub fn set_eps(&mut self, eps: f64) -> Result<()> {
        if !(eps.is_finite() && eps >= 1.0) {
            return Err(FdtdError::InvalidParameter {
                name: "eps",
                value: eps,
            });
        }
        self.eps = eps;
        // The lossless case of `Material::coefficients`, so it equals the 2D Cb.
        self.cb = self.s / eps;
        self.build_profiles();
        Ok(())
    }

    /// Background relative permittivity.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Takes the low half of the profiles from the left layer and the high
    /// half from the right one, each graded by `Cpml::new` exactly as the 2D
    /// x profiles are. A layer is at most (nx - 3) / 2 cells, so neither one
    /// reaches the midpoint, and two equal layers give the 2D profile exactly.
    fn build_profiles(&mut self) {
        let nx = self.ez.len();
        let id = Cpml::none(nx, 1);
        let (mut xi, mut xh) = (id.x_int, id.x_half);
        let mut splice = |spec: &PmlSpec, low: bool| {
            let p = Cpml::new(spec, nx, 1, self.s, self.eps);
            for k in 0..nx {
                if (k < nx / 2) == low {
                    for (dst, src) in [(&mut xi, &p.x_int), (&mut xh, &p.x_half)] {
                        dst.b[k] = src.b[k];
                        dst.c[k] = src.c[k];
                        dst.inv_kappa[k] = src.inv_kappa[k];
                    }
                }
            }
        };
        if let Boundary1d::Cpml(spec) = self.left {
            splice(&spec, true);
        }
        if let Boundary1d::Cpml(spec) = self.right {
            splice(&spec, false);
        }
        (self.x_int, self.x_half) = (xi, xh);
    }

    /// Adds a source. The node must be an existing `ez` node and the waveform
    /// and amplitude must be finite.
    pub fn add_source(&mut self, src: Source1d) -> Result<()> {
        if src.node >= self.ez.len() {
            return Err(FdtdError::InvalidParameter {
                name: "node",
                value: src.node as f64,
            });
        }
        if !src.amplitude.is_finite() {
            return Err(FdtdError::InvalidParameter {
                name: "amplitude",
                value: src.amplitude,
            });
        }
        src.waveform.validate()?;
        self.sources.push(src);
        Ok(())
    }

    /// Advances one full step: H to n + 1/2, E to n + 1, then the boundaries,
    /// then the sources, which use the waveform at t = (n + 1) * S.
    pub fn step(&mut self) {
        let (s, cb) = (self.s, self.cb);
        let nx = self.ez.len();
        let (ez, hy) = (&mut self.ez, &mut self.hy);
        let (xi, xh) = (&self.x_int, &self.x_half);

        // Same expressions, in the same order, as the 2D Hy and Ez loops.
        for i in 0..nx - 1 {
            let (b, c, ik) = (xh.b[i], xh.c[i], xh.inv_kappa[i]);
            let psi = &mut self.psi_h[i];
            *psi = b * *psi + c * (ez[i + 1] - ez[i]);
            hy[i] += s * (ez[i + 1] * ik - ez[i] * ik + *psi);
        }

        let (l0, l1) = (ez[0], ez[1]);
        let (r0, r1) = (ez[nx - 1], ez[nx - 2]);
        for i in 1..nx - 1 {
            let (b, c, ik) = (xi.b[i], xi.c[i], xi.inv_kappa[i]);
            let psi = &mut self.psi_e[i];
            *psi = b * *psi + c * (hy[i] - hy[i - 1]);
            let curl = hy[i] * ik - hy[i - 1] * ik + *psi;
            ez[i] += cb * curl;
        }

        let sv = s / self.eps.sqrt();
        let k = (sv - 1.0) / (sv + 1.0);
        ez[0] = match self.left {
            Boundary1d::Pec | Boundary1d::Cpml(_) => 0.0,
            Boundary1d::Mur1 => l1 + k * (ez[1] - l0),
        };
        ez[nx - 1] = match self.right {
            Boundary1d::Pec | Boundary1d::Cpml(_) => 0.0,
            Boundary1d::Mur1 => r1 + k * (ez[nx - 2] - r0),
        };

        self.n += 1;
        let t = self.n as f64 * s;
        for src in &self.sources {
            let v = src.amplitude * src.waveform.value(t);
            match src.kind {
                SourceKind::Soft => ez[src.node] += v,
                SourceKind::Hard => ez[src.node] = v,
            }
        }
    }

    /// Advances `steps` full steps.
    pub fn step_n(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// `ez` on nodes, x = i, time level n (the current step).
    pub fn ez(&self) -> &[f64] {
        &self.ez
    }

    /// `hy` on x-edges, x = i + 1/2, time level n - 1/2.
    pub fn hy(&self) -> &[f64] {
        &self.hy
    }

    /// Number of completed steps n.
    pub fn time_step(&self) -> u64 {
        self.n
    }

    /// The Courant number S, which is also dt in normalized units.
    pub fn courant(&self) -> f64 {
        self.s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::{Axis, Component, curl_stencil_tm};

    /// Deterministic values in [-1, 1) from a 64-bit LCG. Enough for test data.
    fn random_fields(seed: u64, n: usize) -> Vec<f64> {
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

    /// One step driven by the stencil table in `grid.rs`, restricted to 1D by
    /// dropping every y-difference. Interior nodes only.
    fn stencil_step(ez: &mut [f64], hy: &mut [f64], s: f64) {
        let apply = |target: Component, src: &[f64], i: usize| -> f64 {
            curl_stencil_tm(target)
                .iter()
                .filter(|t| t.axis == Axis::X)
                .map(|t| t.sign * src[(i as i64 + t.di) as usize])
                .sum()
        };
        for (i, h) in hy.iter_mut().enumerate() {
            *h += s * apply(Component::Hy, ez, i);
        }
        for i in 1..ez.len() - 1 {
            ez[i] += s * apply(Component::Ez, hy, i);
        }
    }

    /// The hand-written loop in `step` must equal the stencil table exactly,
    /// which proves the loop indices match the documented staggering.
    #[test]
    fn stencil_oracle_matches_step() {
        let (nx, s) = (64, 0.5);
        for seed in 1..=5 {
            let mut sim = Sim1d::new(nx, s, Boundary1d::Pec, Boundary1d::Pec).unwrap();
            sim.ez = random_fields(seed, nx);
            sim.hy = random_fields(seed + 100, nx - 1);
            let (mut ez, mut hy) = (sim.ez.clone(), sim.hy.clone());

            for _ in 0..3 {
                sim.step();
                stencil_step(&mut ez, &mut hy, s);
                ez[0] = 0.0;
                ez[nx - 1] = 0.0;
            }
            assert_eq!(sim.hy, hy, "hy differs, seed {seed}");
            assert_eq!(sim.ez, ez, "ez differs, seed {seed}");
        }
    }

    fn pulse(node: usize, kind: SourceKind) -> Source1d {
        Source1d {
            node,
            waveform: Waveform::Gaussian { t0: 40.0, tau: 6.0 },
            amplitude: 1.0,
            kind,
        }
    }

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
    }

    #[test]
    fn courant_above_limit_errors() {
        let new = |s| Sim1d::new(10, s, Boundary1d::Pec, Boundary1d::Pec);
        assert_eq!(
            new(1.0001).unwrap_err(),
            FdtdError::CourantAboveLimit {
                s: 1.0001,
                limit: 1.0
            }
        );
        for s in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(new(s).is_err(), "accepted S = {s}");
        }
        assert!(new(1.0).is_ok());
        assert!(Sim1d::new(2, 0.5, Boundary1d::Pec, Boundary1d::Pec).is_err());
    }

    #[test]
    fn add_source_rejects_bad_input() {
        let mut sim = Sim1d::new(10, 0.5, Boundary1d::Pec, Boundary1d::Pec).unwrap();
        assert!(sim.add_source(pulse(10, SourceKind::Soft)).is_err());
        let mut bad_amp = pulse(5, SourceKind::Soft);
        bad_amp.amplitude = f64::NAN;
        assert!(sim.add_source(bad_amp).is_err());
        let mut bad_wave = pulse(5, SourceKind::Soft);
        bad_wave.waveform = Waveform::Gaussian { t0: 1.0, tau: -1.0 };
        assert!(sim.add_source(bad_wave).is_err());
    }

    /// A hard source overwrites its node with the waveform at t = n * S.
    #[test]
    fn hard_source_sets_value() {
        let s = 0.5;
        let mut sim = Sim1d::new(100, s, Boundary1d::Mur1, Boundary1d::Mur1).unwrap();
        let mut src = pulse(40, SourceKind::Hard);
        src.amplitude = 2.5;
        sim.add_source(src).unwrap();
        for n in 1..=80u64 {
            sim.step();
            assert_eq!(sim.time_step(), n);
            assert_eq!(sim.ez()[40], 2.5 * src.waveform.value(n as f64 * s));
        }
    }

    /// A dielectric background with a CPML at both ends is the y-uniform
    /// slice of the 2D periodic grid with the same background and layer: a
    /// soft source on every 2D row gives the 1D fields bit for bit, inside the
    /// layer too. The 2D layer and the 2D update are validated (A5, A9, the
    /// oracles), so this pins the 1D layer and eps without a second derivation.
    #[test]
    fn eps_and_cpml_equal_2d_periodic_slice() {
        use crate::cpml::PmlSpec;
        use crate::material::Material;
        use crate::scene::{GridSpec, SCENE_VERSION, Scene, Source};
        use crate::tm::SimTm;

        let (nx, ny, s, eps) = (90, 4, 0.5, 2.25);
        let spec = PmlSpec::recommended(10);
        let w = Waveform::Ricker {
            t0: 30.0,
            f_peak: 0.06,
        };
        let scene = Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx,
                ny,
                dx_phys_m: 1e-3,
                courant: s,
            },
            materials: vec![Material::Dielectric {
                eps_inf: eps,
                sigma: 0.0,
            }],
            geometry: vec![],
            sources: (0..ny)
                .map(|j| Source {
                    i: 33,
                    j,
                    waveform: w,
                    amplitude: 1.0,
                    kind: SourceKind::Soft,
                })
                .collect(),
            monitors: vec![],
            pml: Some(spec),
            periodic_y: true,
            plane_wave: None,
            paint: None,
        };
        let mut two = SimTm::from_scene(&scene).unwrap();
        let mut one = Sim1d::new(nx, s, Boundary1d::Cpml(spec), Boundary1d::Cpml(spec)).unwrap();
        one.set_eps(eps).unwrap();
        one.add_source(Source1d {
            node: 33,
            waveform: w,
            amplitude: 1.0,
            kind: SourceKind::Soft,
        })
        .unwrap();
        let mut seen = 0.0_f64;
        for n in 0..600 {
            two.step();
            one.step();
            assert_eq!(&two.ez()[..nx], one.ez(), "ez, step {n}");
            assert_eq!(&two.hy()[..nx - 1], one.hy(), "hy, step {n}");
            seen = seen.max(max_abs(&one.ez()[..10]));
        }
        assert!(seen > 1e-3, "the pulse must reach the left layer");
    }

    /// The far-end CPML reflects at or below -100 dB at normal incidence, in
    /// vacuum and in eps = 4, for the 16-cell preset and for the 32-cell
    /// layer the TF/SF auxiliary grid uses. Launch: a hard source on node 0
    /// (the left end), so all the energy goes right. Reference-domain
    /// method: the same grid 800 nodes longer, so its layer's echo cannot
    /// reach the probe in time. Pulse: Ricker at 20 cells per wavelength in
    /// the medium.
    #[test]
    fn far_end_cpml_reflection() {
        use crate::cpml::PmlSpec;
        for (d, eps) in [(16, 1.0), (16, 4.0), (32, 1.0), (32, 4.0)] {
            let f_peak = 1.0 / (20.0 * f64::sqrt(eps));
            let w = Waveform::Ricker {
                t0: 2.0 / f_peak,
                f_peak,
            };
            let (nx, probe, steps) = (220, 100, 1600);
            let run = |n: usize| {
                let spec = PmlSpec::recommended(d);
                let mut sim = Sim1d::new(n, 0.5, Boundary1d::Pec, Boundary1d::Cpml(spec)).unwrap();
                sim.set_eps(eps).unwrap();
                sim.add_source(Source1d {
                    node: 0,
                    waveform: w,
                    amplitude: 1.0,
                    kind: SourceKind::Hard,
                })
                .unwrap();
                (0..steps)
                    .map(|_| {
                        sim.step();
                        sim.ez()[probe]
                    })
                    .collect::<Vec<f64>>()
            };
            let (test, reference) = (run(nx), run(nx + 800));
            let diff: Vec<f64> = test.iter().zip(&reference).map(|(a, b)| a - b).collect();
            let db = 20.0 * (max_abs(&diff) / max_abs(&reference)).log10();
            eprintln!("1D far-end CPML, {d} cells, eps {eps}: {db:.2} dB");
            assert!(db <= -100.0, "{d} cells, eps {eps}: {db} dB");
        }
    }

    /// Layer reflection per frequency, |DFT(test - reference) / DFT(reference)|
    /// at the probe, for the 16-cell preset with its CFS alpha and with alpha
    /// = 0. The CFS term sigma / (alpha + j w) turns real below w of about
    /// alpha, so the layer stretches but barely absorbs: the preset is
    /// excellent at the 20 cells per wavelength it was tuned for and poor at
    /// long wavelengths. alpha exists for evanescent and grazing waves in 2D;
    /// a 1D line has neither, so alpha = 0 is the right layer there.
    #[test]
    fn layer_reflection_against_frequency() {
        use crate::cpml::PmlSpec;
        use crate::monitor::Dft;
        // The Ricker (peak 0.01) carries energy over this band; above about
        // 0.035 it has none and the ratio would be noise over noise.
        let freqs = [0.0025, 0.005, 0.01, 0.025];
        let w = Waveform::Ricker {
            t0: 200.0,
            f_peak: 0.01,
        };
        let spectrum = |spec: PmlSpec, n: usize| {
            let mut sim = Sim1d::new(n, 0.5, Boundary1d::Pec, Boundary1d::Cpml(spec)).unwrap();
            sim.add_source(Source1d {
                node: 0,
                waveform: w,
                amplitude: 1.0,
                kind: SourceKind::Hard,
            })
            .unwrap();
            let mut d = Dft::new(&freqs, 1).unwrap();
            for _ in 0..8000 {
                sim.step();
                let t = sim.time_step() as f64 * 0.5;
                d.add(&[sim.ez()[100]], t, 0.5);
            }
            d
        };
        let preset = PmlSpec::recommended(16);
        let no_alpha = PmlSpec {
            alpha_max: 0.0,
            ..preset
        };
        let reference = spectrum(preset, 6000);
        let db = |spec| {
            let test = spectrum(spec, 220);
            (0..freqs.len())
                .map(|f| {
                    let (a, b) = (test.get(0, f), reference.get(0, f));
                    let (dr, di) = (a.0 - b.0, a.1 - b.1);
                    10.0 * ((dr * dr + di * di) / (b.0 * b.0 + b.1 * b.1)).log10()
                })
                .collect::<Vec<f64>>()
        };
        let (with, without) = (db(preset), db(no_alpha));
        for (k, f) in freqs.iter().enumerate() {
            eprintln!(
                "layer reflection at f {f}: alpha 0.1 {:7.1} dB, alpha 0 {:7.1} dB",
                with[k], without[k]
            );
        }
        assert!(without.iter().all(|&v| v <= -80.0), "alpha 0: {without:?}");
        assert!(
            with[0] > -60.0,
            "the preset was expected to be weak at f = 0.0025"
        );
    }

    /// The Mur end uses the wave speed of the medium, v = 1 / sqrt(eps). In
    /// eps = 4 at S = 0.5 it is not exact (S v = 0.25), but what it reflects
    /// comes from grid dispersion only, far below -30 dB at 20 cells per
    /// wavelength. With the vacuum speed in the formula instead, the end
    /// would see a wave twice too slow and reflect about 1/3, near -10 dB.
    #[test]
    fn mur_end_uses_the_medium_speed() {
        use crate::cpml::PmlSpec;
        let eps = 4.0;
        let f_peak = 1.0 / (20.0 * f64::sqrt(eps));
        let w = Waveform::Ricker {
            t0: 2.0 / f_peak,
            f_peak,
        };
        let run = |n: usize, right: Boundary1d| {
            let mut sim = Sim1d::new(n, 0.5, Boundary1d::Pec, right).unwrap();
            sim.set_eps(eps).unwrap();
            sim.add_source(Source1d {
                node: 0,
                waveform: w,
                amplitude: 1.0,
                kind: SourceKind::Hard,
            })
            .unwrap();
            (0..1600)
                .map(|_| {
                    sim.step();
                    sim.ez()[100]
                })
                .collect::<Vec<f64>>()
        };
        let test = run(200, Boundary1d::Mur1);
        let reference = run(1000, Boundary1d::Cpml(PmlSpec::recommended(16)));
        let diff: Vec<f64> = test.iter().zip(&reference).map(|(a, b)| a - b).collect();
        let db = 20.0 * (max_abs(&diff) / max_abs(&reference)).log10();
        eprintln!("1D Mur end, eps 4, S 0.5: {db:.2} dB");
        assert!(db <= -30.0, "{db} dB");
    }

    #[test]
    fn eps_and_cpml_input_checked() {
        use crate::cpml::PmlSpec;
        let mut sim = Sim1d::new(40, 0.5, Boundary1d::Pec, Boundary1d::Pec).unwrap();
        for bad in [0.5, 0.0, f64::NAN, f64::INFINITY] {
            assert!(sim.set_eps(bad).is_err(), "accepted eps {bad}");
        }
        assert!(sim.set_eps(1.0).is_ok());
        let mut spec = PmlSpec::recommended(10);
        let new = |l, r| Sim1d::new(40, 0.5, l, r);
        assert!(new(Boundary1d::Pec, Boundary1d::Cpml(spec)).is_ok());
        // Two 19-cell layers leave one node between them; 2 d + 3 = 41 > 40.
        spec.thickness = 19;
        assert!(matches!(
            new(Boundary1d::Cpml(spec), Boundary1d::Cpml(spec)),
            Err(FdtdError::PmlTooThick { thickness: 19, .. })
        ));
        spec.thickness = 10;
        spec.kappa_max = 0.5;
        assert!(new(Boundary1d::Pec, Boundary1d::Cpml(spec)).is_err());
    }

    /// At S = 1 the first-order Mur end is exact, so once both halves of the
    /// pulse have left, nothing at all remains on the grid. This needs t0 >= 6 tau:
    /// a turn-on step leaves a checkerboard (-1)^(i+n) box that Mur cannot see.
    #[test]
    fn mur_exact_at_s1() {
        let mut sim = Sim1d::new(200, 1.0, Boundary1d::Mur1, Boundary1d::Mur1).unwrap();
        sim.add_source(pulse(100, SourceKind::Soft)).unwrap();
        sim.step_n(60);
        assert!(max_abs(sim.ez()) > 0.1, "pulse should be on the grid");
        sim.step_n(340);
        assert!(max_abs(sim.ez()) < 1e-14, "ez left {}", max_abs(sim.ez()));
        assert!(max_abs(sim.hy()) < 1e-14, "hy left {}", max_abs(sim.hy()));
    }
}
