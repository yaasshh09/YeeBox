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
