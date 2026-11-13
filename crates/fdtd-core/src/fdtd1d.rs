//! 1D reference solver: TMz restricted to d/dy = 0, so the fields are `ez` on
//! nodes (x = i) and `hy` on x-edges (x = i + 1/2). Vacuum, normalized units.
//! Later phases reuse this as the auxiliary grid for TF/SF.

use crate::error::{FdtdError, Result};
use crate::source::{SourceKind, Waveform};

/// Treatment of one end node of the 1D grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary1d {
    /// Perfect electric conductor: `ez` at the end node is held at zero.
    Pec,
    /// First-order Mur absorbing end:
    /// `E0^{n+1} = E1^n + (S - 1)/(S + 1) * (E1^{n+1} - E0^n)`.
    /// Exact at S = 1, reflects a little below it.
    Mur1,
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
}

impl Sim1d {
    /// Builds a vacuum grid of `nx` `ez` nodes and `nx - 1` `hy` x-edges, all
    /// fields zero. `s` is the Courant number, which must satisfy 0 < S <= 1.
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
        Ok(Self {
            ez: vec![0.0; nx],
            hy: vec![0.0; nx - 1],
            s,
            n: 0,
            left,
            right,
            sources: Vec::new(),
        })
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
        let s = self.s;
        let nx = self.ez.len();
        let (ez, hy) = (&mut self.ez, &mut self.hy);

        for i in 0..nx - 1 {
            hy[i] += s * (ez[i + 1] - ez[i]);
        }

        let (l0, l1) = (ez[0], ez[1]);
        let (r0, r1) = (ez[nx - 1], ez[nx - 2]);
        for i in 1..nx - 1 {
            ez[i] += s * (hy[i] - hy[i - 1]);
        }

        let k = (s - 1.0) / (s + 1.0);
        ez[0] = match self.left {
            Boundary1d::Pec => 0.0,
            Boundary1d::Mur1 => l1 + k * (ez[1] - l0),
        };
        ez[nx - 1] = match self.right {
            Boundary1d::Pec => 0.0,
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
