//! Yee grid bookkeeping for TMz, TEz and the 1D restriction, following
//! `docs/01-theory.md` exactly. TEz uses the TMz layout with the roles
//! swapped: Hz on the nodes, Ex on the y-edges, Ey on the x-edges.
//!
//! Positions and time levels are returned DOUBLED, as i64, so half-integer
//! coordinates are exact integers: "half a cell" is 1 and "one cell" is 2.
//! Staggering checks then compare integers instead of floats.

use crate::error::{FdtdError, Result};

/// A field component. Names match `docs/01-theory.md`: `Ez, Hx, Hy` are
/// TMz, `Hz, Ex, Ey` are TEz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Ez,
    Hx,
    Hy,
    Hz,
    Ex,
    Ey,
}

/// The axis a difference is taken along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
}

/// Doubled position (2x, 2y) of sample `[i, j]` of component `c`:
/// Ez and Hz at (i, j) are nodes, Hx and Ex at (i, j + 1/2) are y-edges, and
/// Hy and Ey at (i + 1/2, j) are x-edges.
pub fn pos(c: Component, i: usize, j: usize) -> (i64, i64) {
    let (x, y) = (2 * i as i64, 2 * j as i64);
    match c {
        Component::Ez | Component::Hz => (x, y),
        Component::Hx | Component::Ex => (x, y + 1),
        Component::Hy | Component::Ey => (x + 1, y),
    }
}

/// Doubled x position of sample `i` in the 1D grid: Ez on node i, Hy on x-edge
/// i + 1/2. The 1D grid is TMz restricted to a line, so only Ez and Hy exist.
pub fn pos_1d(c: Component, i: usize) -> Result<i64> {
    match c {
        Component::Ez | Component::Hy => Ok(pos(c, i, 0).0),
        _ => Err(not_in_1d()),
    }
}

/// Doubled time level of component `c` at step index `n`: E at 2n, H at
/// 2n + 1 (H with index n lives at time n + 1/2), in both modes.
pub fn time_level(c: Component, n: u64) -> i64 {
    let t = 2 * n as i64;
    match c {
        Component::Ez | Component::Ex | Component::Ey => t,
        Component::Hx | Component::Hy | Component::Hz => t + 1,
    }
}

/// Array shape (columns along x, rows along y) of component `c` on a grid of
/// `nx` by `ny` nodes.
pub fn shape(c: Component, nx: usize, ny: usize) -> (usize, usize) {
    match c {
        Component::Ez | Component::Hz => (nx, ny),
        Component::Hx | Component::Ex => (nx, ny.saturating_sub(1)),
        Component::Hy | Component::Ey => (nx.saturating_sub(1), ny),
    }
}

/// Array length of component `c` in a 1D grid of `nx` Ez nodes.
pub fn len_1d(c: Component, nx: usize) -> Result<usize> {
    match c {
        Component::Ez | Component::Hy => Ok(shape(c, nx, 1).0),
        _ => Err(not_in_1d()),
    }
}

/// Flat index of `[i, j]` in a row-major array `nx` wide, x fastest. CPU and GPU use the same layout.
pub fn idx(nx: usize, i: usize, j: usize) -> usize {
    j * nx + i
}

fn not_in_1d() -> FdtdError {
    FdtdError::InvalidParameter {
        name: "component outside the 1D grid (only Ez and Hy)",
        value: f64::NAN,
    }
}

/// One operand of a curl update: `sign * src[i + di, j + dj]`, part of the
/// difference along `axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Term {
    pub src: Component,
    pub di: i64,
    pub dj: i64,
    pub sign: f64,
    pub axis: Axis,
}

const fn term(src: Component, di: i64, dj: i64, sign: f64, axis: Axis) -> Term {
    Term {
        src,
        di,
        dj,
        sign,
        axis,
    }
}

use Axis::{X, Y};
use Component::{Ex, Ey, Ez, Hx, Hy, Hz};

const HX_TERMS: [Term; 2] = [term(Ez, 0, 1, -1.0, Y), term(Ez, 0, 0, 1.0, Y)];
const HY_TERMS: [Term; 2] = [term(Ez, 1, 0, 1.0, X), term(Ez, 0, 0, -1.0, X)];
const EZ_TERMS: [Term; 4] = [
    term(Hy, 0, 0, 1.0, X),
    term(Hy, -1, 0, -1.0, X),
    term(Hx, 0, 0, -1.0, Y),
    term(Hx, 0, -1, 1.0, Y),
];
// TEz is the dual of TMz on the same layout (Ez -> Hz, Hx -> -Ex, Hy -> -Ey),
// in the same operand order so the TE oracle can compare bit for bit too.
const EX_TERMS: [Term; 2] = [term(Hz, 0, 1, 1.0, Y), term(Hz, 0, 0, -1.0, Y)];
const EY_TERMS: [Term; 2] = [term(Hz, 1, 0, -1.0, X), term(Hz, 0, 0, 1.0, X)];
const HZ_TERMS: [Term; 4] = [
    term(Ey, 0, 0, -1.0, X),
    term(Ey, -1, 0, 1.0, X),
    term(Ex, 0, 0, 1.0, Y),
    term(Ex, 0, -1, -1.0, Y),
];

/// The lossless TMz and TEz curl stencils from `docs/01-theory.md`, written as data.
/// The update of `target[i, j]` is `coef * sum(sign * src[i + di, j + dj])`,
/// where `coef` is `dt / mu` for H and `dt / eps` for E (dx = dy = 1, normalized).
/// The hand-written solver loops are tested against this table.
pub fn curl_stencil(target: Component) -> &'static [Term] {
    match target {
        Component::Hx => &HX_TERMS,
        Component::Hy => &HY_TERMS,
        Component::Ez => &EZ_TERMS,
        Component::Ex => &EX_TERMS,
        Component::Ey => &EY_TERMS,
        Component::Hz => &HZ_TERMS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Component; 6] = [Ez, Hx, Hy, Hz, Ex, Ey];

    /// Doubled position of the operand `t` for a target at `[i, j]`.
    fn operand_pos(t: &Term, i: usize, j: usize) -> (i64, i64) {
        let (oi, oj) = (i as i64 + t.di, j as i64 + t.dj);
        pos(t.src, oi as usize, oj as usize)
    }

    #[test]
    fn array_sizes_tm_and_1d() {
        assert_eq!(shape(Ez, 7, 5), (7, 5));
        assert_eq!(shape(Hx, 7, 5), (7, 4));
        assert_eq!(shape(Hy, 7, 5), (6, 5));
        assert_eq!(shape(Hz, 7, 5), (7, 5));
        assert_eq!(shape(Ex, 7, 5), (7, 4));
        assert_eq!(shape(Ey, 7, 5), (6, 5));
        assert!(len_1d(Hz, 9).is_err());
        assert_eq!(len_1d(Ez, 9).unwrap(), 9);
        assert_eq!(len_1d(Hy, 9).unwrap(), 8);
        assert!(len_1d(Hx, 9).is_err());
    }

    #[test]
    fn component_positions_half_integer() {
        assert_eq!(pos(Ez, 3, 4), (6, 8));
        assert_eq!(pos(Hx, 3, 4), (6, 9));
        assert_eq!(pos(Hy, 3, 4), (7, 8));
        assert_eq!(pos_1d(Ez, 3).unwrap(), 6);
        assert_eq!(pos_1d(Hy, 3).unwrap(), 7);
        assert!(pos_1d(Hx, 3).is_err());
        assert_eq!(time_level(Ez, 5), 10);
        assert_eq!(time_level(Hx, 5), 11);
        assert_eq!(time_level(Hy, 5), 11);
        // TEz swaps the roles on the same layout: Hz on the nodes, Ex on the
        // y-edges, Ey on the x-edges, E still at integer time.
        assert_eq!(pos(Hz, 3, 4), (6, 8));
        assert_eq!(pos(Ex, 3, 4), (6, 9));
        assert_eq!(pos(Ey, 3, 4), (7, 8));
        assert!(pos_1d(Hz, 3).is_err());
        assert_eq!(time_level(Hz, 5), 11);
        assert_eq!(time_level(Ex, 5), 10);
        assert_eq!(time_level(Ey, 5), 10);
    }

    /// Every centered difference uses two samples of one component exactly one
    /// cell apart along its axis, level on the other axis, with their midpoint
    /// on the target. A half-cell indexing error breaks this.
    #[test]
    fn stencil_pairs_one_cell_apart_midpoint_on_target() {
        let (i, j) = (5, 7);
        for target in ALL {
            let t = pos(target, i, j);
            for axis in [X, Y] {
                let pair: Vec<&Term> = curl_stencil(target)
                    .iter()
                    .filter(|t| t.axis == axis)
                    .collect();
                if pair.is_empty() {
                    continue;
                }
                assert_eq!(pair.len(), 2, "{target:?} {axis:?}");
                assert_eq!(pair[0].src, pair[1].src);
                assert_eq!(pair[0].sign, -pair[1].sign);
                let (a, b) = (operand_pos(pair[0], i, j), operand_pos(pair[1], i, j));
                let (along, across) = match axis {
                    X => (a.0 - b.0, a.1 - b.1),
                    Y => (a.1 - b.1, a.0 - b.0),
                };
                assert_eq!(along.abs(), 2, "{target:?} {axis:?} not one cell apart");
                assert_eq!(across, 0, "{target:?} {axis:?} not level");
                assert_eq!(
                    (a.0 + b.0, a.1 + b.1),
                    (2 * t.0, 2 * t.1),
                    "{target:?} {axis:?} midpoint off target"
                );
            }
        }
    }

    /// For linear fields the centered difference is exact, so applying the
    /// stencil must give the analytic Maxwell curl term bit for bit:
    /// dHx/dt = -dEz/dy, dHy/dt = dEz/dx, dEz/dt = dHy/dx - dHx/dy, and in
    /// TEz dHz/dt = dEx/dy - dEy/dx, dEx/dt = dHz/dy, dEy/dt = -dHz/dx.
    /// This pins the signs that the geometric test cannot see.
    #[test]
    fn stencil_exact_on_linear_fields() {
        // Coefficients are small dyadic rationals so every operation is exact.
        let field = |c: Component, p: (i64, i64)| -> f64 {
            let (x, y) = (p.0 as f64 / 2.0, p.1 as f64 / 2.0);
            match c {
                Ez => 0.75 * x - 1.25 * y + 3.0,
                Hx => -0.5 * x + 2.0 * y - 1.0,
                Hy => 1.5 * x + 0.25 * y + 0.5,
                Hz => -0.25 * x + 1.75 * y - 2.0,
                Ex => 0.5 * x - 0.75 * y + 1.0,
                Ey => 1.25 * x + 0.5 * y - 0.5,
            }
        };
        let analytic = |target: Component| -> f64 {
            match target {
                Hx => 1.25,         // -dEz/dy
                Hy => 0.75,         // dEz/dx
                Ez => 1.5 - 2.0,    // dHy/dx - dHx/dy
                Hz => -0.75 - 1.25, // dEx/dy - dEy/dx
                Ex => 1.75,         // dHz/dy
                Ey => 0.25,         // -dHz/dx
            }
        };
        for (i, j) in [(1, 1), (4, 9), (20, 3)] {
            for target in ALL {
                let got: f64 = curl_stencil(target)
                    .iter()
                    .map(|t| t.sign * field(t.src, operand_pos(t, i, j)))
                    .sum();
                assert_eq!(got, analytic(target), "{target:?} at ({i}, {j})");
            }
        }
    }

    #[test]
    fn idx_row_major_roundtrip() {
        let nx = 13;
        for j in 0..5 {
            for i in 0..nx {
                let k = idx(nx, i, j);
                assert_eq!((k % nx, k / nx), (i, j));
            }
        }
        assert_eq!(idx(nx, 1, 0) - idx(nx, 0, 0), 1, "x must be fastest");
    }
}
