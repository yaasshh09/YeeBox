//! Total-field / scattered-field plane wave source, axis-aligned only. See
//! `docs/02-numerics.md`. The box `[x0, x1] x [y0, y1]` (Ez node indices,
//! inclusive) holds the total field; outside it holds the scattered field.
//! The incident wave comes from a 1D auxiliary grid running in lockstep.
//!
//! Only axis-aligned incidence: along an axis the 2D dispersion relation is
//! exactly the 1D one at the same dx and dt, so the auxiliary grid matches the
//! 2D grid and the leakage is at round-off. Oblique incidence would leak at
//! about -40 dB without a dispersion-matched auxiliary grid, so it is not offered.

use crate::source::Waveform;
use serde::{Deserialize, Serialize};

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
