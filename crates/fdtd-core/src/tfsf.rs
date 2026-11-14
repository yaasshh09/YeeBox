//! Total-field / scattered-field plane wave source, axis-aligned only. See
//! `docs/02-numerics.md`. The box `[x0, x1] x [y0, y1]` (Ez node indices,
//! inclusive) holds the total field; outside it holds the scattered field.
//! The incident wave comes from a 1D auxiliary grid running in lockstep.
//!
//! Only axis-aligned incidence: along an axis the 2D dispersion relation is
//! exactly the 1D one at the same dx and dt, so the auxiliary grid matches the
//! 2D grid and the leakage is at round-off. Oblique incidence would leak at
//! about -40 dB without a dispersion-matched auxiliary grid, so it is not offered.

use crate::cpml::PmlSpec;
use crate::error::Result;
use crate::fdtd1d::{Boundary1d, Sim1d, Source1d};
use crate::source::{SourceKind, Waveform};
use serde::{Deserialize, Serialize};

/// CPML on the far end of the auxiliary grid. Measured at -120 dB in vacuum
/// at normal incidence (`fdtd1d` tests), so the incident wave it reflects
/// back into the box is far below anything the TF/SF tests resolve.
const AUX_PML: usize = 16;

/// The running TF/SF source: the 1D auxiliary grid and the map from 2D
/// indices along the travel axis to its nodes.
///
/// Along the travel axis q (x for x travel, y for y travel), with the box
/// spanning q in [lo, hi]:
///
/// - forward travel (+x, +y): aux node p = q - (lo - 2)
/// - backward travel (-x, -y): the axis is mirrored, p = (hi + 2) - q
///
/// so aux node 2 is the entry face either way and node 0, two cells before
/// it, is the hard-source launch node. The aux `hy` edge p + 1/2 is the 2D
/// edge q + 1/2 going forward and q - 1/2 going backward. The incident
/// transverse H follows from the symmetry of the update equations: mirroring
/// x flips Hy, and y travel has Hx = -h (h obeys the 1D `hy` update), so
///
///   +x: Hy = +h,   -x: Hy = -h,   +y: Hx = -h,   -y: Hx = +h.
#[derive(Debug, Clone)]
pub struct Tfsf {
    pw: PlaneWave,
    aux: Sim1d,
}

impl Tfsf {
    /// Builds the auxiliary grid for a validated plane wave in a background
    /// of relative permittivity `eps`, at Courant number `s`.
    pub fn new(pw: &PlaneWave, eps: f64, s: f64) -> Result<Self> {
        let span = match pw.direction {
            Direction::PlusX | Direction::MinusX => pw.x1 - pw.x0,
            Direction::PlusY | Direction::MinusY => pw.y1 - pw.y0,
        };
        // Nodes 0..=span + 2 are used; 4 spare nodes, then the layer.
        let n = (span + 7 + AUX_PML).max(2 * AUX_PML + 3);
        let mut aux = Sim1d::new(
            n,
            s,
            Boundary1d::Pec,
            Boundary1d::Cpml(PmlSpec::recommended(AUX_PML)),
        )?;
        aux.set_eps(eps)?;
        aux.add_source(Source1d {
            node: 0,
            waveform: pw.waveform,
            amplitude: pw.amplitude,
            kind: SourceKind::Hard,
        })?;
        Ok(Self { pw: *pw, aux })
    }

    fn forward(&self) -> bool {
        matches!(self.pw.direction, Direction::PlusX | Direction::PlusY)
    }

    fn along_x(&self) -> bool {
        matches!(self.pw.direction, Direction::PlusX | Direction::MinusX)
    }

    /// (lo, hi) of the box along the travel axis.
    fn span(&self) -> (usize, usize) {
        if self.along_x() {
            (self.pw.x0, self.pw.x1)
        } else {
            (self.pw.y0, self.pw.y1)
        }
    }

    /// Incident Ez at travel-axis node q, time level n (the current step).
    fn ez_inc(&self, q: usize) -> f64 {
        let (lo, hi) = self.span();
        let p = if self.forward() {
            q + 2 - lo
        } else {
            hi + 2 - q
        };
        self.aux.ez()[p]
    }

    /// Incident transverse H (Hy for x travel, Hx for y travel) on the edge
    /// q + 1/2 along the travel axis, time level n - 1/2 (after `advance`,
    /// n + 1/2 of the step being taken).
    fn h_inc(&self, q: usize) -> f64 {
        let (lo, hi) = self.span();
        let h = &self.aux.hy();
        match self.pw.direction {
            Direction::PlusX => h[q + 2 - lo],
            Direction::MinusX => -h[hi + 1 - q],
            Direction::PlusY => -h[q + 2 - lo],
            Direction::MinusY => h[hi + 1 - q],
        }
    }

    /// Incident Ez at 2D node (i, j).
    fn ez_at(&self, i: usize, j: usize) -> f64 {
        self.ez_inc(if self.along_x() { i } else { j })
    }

    /// H corrections, applied right after the 2D H update, with the
    /// auxiliary grid still at E^n. An edge on the scattered side of a face
    /// read a total Ez across it, so the incident Ez there is taken back out:
    ///
    ///   hy[x0 - 1, j] -= S Ez_inc(x0),  hy[x1, j] += S Ez_inc(x1)
    ///   hx[i, y0 - 1] += S Ez_inc(y0),  hx[i, y1] -= S Ez_inc(y1)
    ///
    /// Arrays are the padded 2D ones, index `j * nx + i`. With `periodic_y`
    /// there are no y faces.
    pub fn correct_h(&self, hx: &mut [f64], hy: &mut [f64], nx: usize, s: f64, periodic_y: bool) {
        let PlaneWave { x0, y0, x1, y1, .. } = self.pw;
        for j in y0..=y1 {
            hy[j * nx + x0 - 1] -= s * self.ez_at(x0, j);
            hy[j * nx + x1] += s * self.ez_at(x1, j);
        }
        if !periodic_y {
            for i in x0..=x1 {
                hx[(y0 - 1) * nx + i] += s * self.ez_at(i, y0);
                hx[y1 * nx + i] -= s * self.ez_at(i, y1);
            }
        }
    }

    /// Steps the auxiliary grid: its H to n + 1/2 and its E to n + 1.
    pub fn advance(&mut self) {
        self.aux.step();
    }

    /// E corrections, applied right after the 2D E update, with the
    /// auxiliary H at n + 1/2. A total-field node on a face read a
    /// scattered H across it, so the incident H there is added back, with
    /// the sign of that term in the curl:
    ///
    ///   ez[x0, j] -= cb Hy_inc(x0 - 1/2),  ez[x1, j] += cb Hy_inc(x1 + 1/2)
    ///   ez[i, y0] += cb Hx_inc(y0 - 1/2),  ez[i, y1] -= cb Hx_inc(y1 + 1/2)
    ///
    /// Only the transverse H of the travel axis is nonzero, so x travel
    /// corrects the x faces and y travel the y faces. `cb` is the per-node
    /// 2D coefficient array.
    pub fn correct_e(&self, ez: &mut [f64], cb: &[f64], nx: usize) {
        let PlaneWave { x0, y0, x1, y1, .. } = self.pw;
        if self.along_x() {
            for j in y0..=y1 {
                let (a, b) = (j * nx + x0, j * nx + x1);
                ez[a] -= cb[a] * self.h_inc(x0 - 1);
                ez[b] += cb[b] * self.h_inc(x1);
            }
        } else {
            for i in x0..=x1 {
                let (a, b) = (y0 * nx + i, y1 * nx + i);
                ez[a] += cb[a] * self.h_inc(y0 - 1);
                ez[b] -= cb[b] * self.h_inc(y1);
            }
        }
    }
}

/// Travel direction of the incident plane wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    #[serde(rename = "+x")]
    PlusX,
    #[serde(rename = "-x")]
    MinusX,
    #[serde(rename = "+y")]
    PlusY,
    #[serde(rename = "-y")]
    MinusY,
}

/// A TF/SF plane wave. The box corners are Ez node indices; the nodes on the
/// box are total field. `waveform` times `amplitude` is the incident Ez where
/// the wave enters the box, at the 1D grid's launch node (normalized units).
///
/// JSON form: `{"x0": 20, "y0": 0, "x1": 180, "y1": 9, "direction": "+x",
/// "waveform": {"type": "ricker", "t0": 60, "f_peak": 0.05}, "amplitude": 1}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaneWave {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
    pub direction: Direction,
    pub waveform: Waveform,
    pub amplitude: f64,
}

/// Why a plane wave box was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaneWaveFault {
    /// `x0 > x1` or `y0 > y1`.
    EmptyBox,
    /// A box node, or a scattered-field node just across a face, is on a
    /// wall, outside the grid, or inside the CPML layer.
    FaceTooClose,
    /// A box face node or the node across it is not material 0. The
    /// corrections assume the background medium on both sides of each face.
    NotBackground,
    /// Material 0 is lossy or PEC. The auxiliary grid is lossless.
    LossyBackground,
    /// With periodic y the wave cannot travel along y; it would wrap.
    WrapsAlongY,
    /// With periodic y the box must span every row (y0 = 0, y1 = ny - 1),
    /// because there are no y faces.
    NotFullHeight,
}

impl std::fmt::Display for PlaneWaveFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::EmptyBox => "the box is empty",
            Self::FaceTooClose => "a face is on a wall, in the pml, or one node from either",
            Self::NotBackground => "a face is not in material 0",
            Self::LossyBackground => "material 0 must be a lossless dielectric",
            Self::WrapsAlongY => "a y-travelling wave cannot be used with periodic y",
            Self::NotFullHeight => "with periodic y the box must span every row",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_json_names() {
        for (d, name) in [
            (Direction::PlusX, "\"+x\""),
            (Direction::MinusX, "\"-x\""),
            (Direction::PlusY, "\"+y\""),
            (Direction::MinusY, "\"-y\""),
        ] {
            assert_eq!(serde_json::to_string(&d).unwrap(), name);
            assert_eq!(serde_json::from_str::<Direction>(name).unwrap(), d);
        }
        assert!(serde_json::from_str::<Direction>("\"x\"").is_err());
    }
}
