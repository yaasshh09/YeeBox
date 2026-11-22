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
use crate::grid::Component;
use crate::source::{SourceKind, Waveform};
use serde::{Deserialize, Serialize};

/// CPML on the far end of the auxiliary grid. Its echo travels back through
/// the box as a weak backward incident wave. With 16 cells (-127 dB in the
/// `fdtd1d` reflection test) the plane wave E/H identity was off by 4e-6 at
/// low frequency; 32 cells (-158 dB) leave 3e-8, and a 1D layer costs
/// nothing, so 32.
const AUX_PML: usize = 32;

/// The auxiliary layer: the graded preset without the CFS alpha. alpha is
/// for evanescent and grazing waves in 2D, and a 1D line has neither; with
/// it the layer stops absorbing below w of about alpha (-41 dB at f = 0.0025
/// against -157 dB without, `fdtd1d` layer_reflection_against_frequency),
/// which put a 3e-3 error into the A13 slab at its lowest frequencies.
fn aux_layer() -> PmlSpec {
    PmlSpec {
        alpha_max: 0.0,
        ..PmlSpec::recommended(AUX_PML)
    }
}

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
        let mut aux = Sim1d::new(n, s, Boundary1d::Pec, Boundary1d::Cpml(aux_layer()))?;
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

    /// The incident Ez on every travel-axis node of the box, `lo..=hi`, at
    /// the auxiliary grid's current level (E^n during the H corrections).
    /// Entry `q - lo` is node q. This is what `Correction::src` indexes for
    /// the H corrections.
    pub fn incident_e(&self) -> Vec<f64> {
        let (lo, hi) = self.span();
        (lo..=hi).map(|q| self.ez_inc(q)).collect()
    }

    /// The incident transverse H on the edge just outside each face, `[lo -
    /// 1/2, hi + 1/2]` along the travel axis, at the auxiliary level (n + 1/2
    /// after `advance`). This is what `Correction::src` indexes for the E
    /// corrections.
    pub fn incident_h(&self) -> [f64; 2] {
        let (lo, hi) = self.span();
        [self.h_inc(lo - 1), self.h_inc(hi)]
    }

    /// Where `incident_e` holds the incident Ez of 2D node (i, j).
    fn src_e(&self, i: usize, j: usize) -> usize {
        (if self.along_x() { i } else { j }) - self.span().0
    }

    /// The H corrections, applied right after the 2D H update with the
    /// auxiliary grid still at E^n. An edge on the scattered side of a face
    /// read a total Ez across it, so the incident Ez there is taken back out:
    ///
    ///   hy[x0 - 1, j] -= S Ez_inc(x0),  hy[x1, j] += S Ez_inc(x1)
    ///   hx[i, y0 - 1] += S Ez_inc(y0),  hx[i, y1] -= S Ez_inc(y1)
    ///
    /// One group per line index (j for the x faces, then i for the y faces),
    /// its two entries in this order; `src` indexes `incident_e`. Indices are
    /// into the padded 2D arrays, `j * nx + i`. With `periodic_y` there are
    /// no y faces. Different groups never touch the same edge, so a kernel
    /// may apply groups in parallel and the entries of a group in order.
    pub fn h_corrections(&self, nx: usize, s: f64, periodic_y: bool) -> Vec<[Correction; 2]> {
        let PlaneWave { x0, y0, x1, y1, .. } = self.pw;
        let c = |field, k, (i, j), factor| Correction {
            field,
            k,
            src: self.src_e(i, j),
            factor,
        };
        let mut out = Vec::new();
        for j in y0..=y1 {
            out.push([
                c(Component::Hy, j * nx + x0 - 1, (x0, j), -s),
                c(Component::Hy, j * nx + x1, (x1, j), s),
            ]);
        }
        if !periodic_y {
            for i in x0..=x1 {
                out.push([
                    c(Component::Hx, (y0 - 1) * nx + i, (i, y0), s),
                    c(Component::Hx, y1 * nx + i, (i, y1), -s),
                ]);
            }
        }
        out
    }

    /// The E corrections, applied right after the 2D E update with the
    /// auxiliary H at n + 1/2. A total-field node on a face read a
    /// scattered H across it, so the incident H there is added back, with
    /// the sign of that term in the curl:
    ///
    ///   ez[x0, j] -= cb Hy_inc(x0 - 1/2),  ez[x1, j] += cb Hy_inc(x1 + 1/2)
    ///   ez[i, y0] += cb Hx_inc(y0 - 1/2),  ez[i, y1] -= cb Hx_inc(y1 + 1/2)
    ///
    /// Only the transverse H of the travel axis is nonzero, so x travel
    /// corrects the x faces and y travel the y faces. `cb` is the per-node
    /// 2D coefficient array; `src` indexes `incident_h`. One group per line
    /// index, entries in order. A box one node thick puts both entries of a
    /// group on the same node, which is why they stay together.
    pub fn e_corrections(&self, nx: usize, cb: &[f64]) -> Vec<[Correction; 2]> {
        let PlaneWave { x0, y0, x1, y1, .. } = self.pw;
        let c = |k: usize, src, sign: f64| Correction {
            field: Component::Ez,
            k,
            src,
            factor: sign * cb[k],
        };
        if self.along_x() {
            (y0..=y1)
                .map(|j| [c(j * nx + x0, 0, -1.0), c(j * nx + x1, 1, 1.0)])
                .collect()
        } else {
            (x0..=x1)
                .map(|i| [c(y0 * nx + i, 0, 1.0), c(y1 * nx + i, 1, -1.0)])
                .collect()
        }
    }

    /// Applies `h_corrections` from the current `incident_e`.
    pub fn correct_h(&self, hx: &mut [f64], hy: &mut [f64], nx: usize, s: f64, periodic_y: bool) {
        let inc = self.incident_e();
        for c in self.h_corrections(nx, s, periodic_y).iter().flatten() {
            let field = if c.field == Component::Hx {
                &mut *hx
            } else {
                &mut *hy
            };
            field[c.k] += c.factor * inc[c.src];
        }
    }

    /// Steps the auxiliary grid: its H to n + 1/2 and its E to n + 1.
    pub fn advance(&mut self) {
        self.aux.step();
    }

    /// Applies `e_corrections` from the current `incident_h`.
    pub fn correct_e(&self, ez: &mut [f64], cb: &[f64], nx: usize) {
        let inc = self.incident_h();
        for c in self.e_corrections(nx, cb).iter().flatten() {
            ez[c.k] += c.factor * inc[c.src];
        }
    }
}

/// One TF/SF correction: `field[k] += factor * incident[src]`, with
/// `incident` the `Tfsf::incident_e` list for H corrections and
/// `incident_h` for E corrections. Both kernels apply the same lists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correction {
    pub field: Component,
    /// Index into the padded `j * nx + i` array of `field`.
    pub k: usize,
    pub src: usize,
    /// Normalized: plus or minus S for H, plus or minus the node's Cb for E.
    pub factor: f64,
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
    /// A TEz plane wave runs the TMz auxiliary grid through duality, which
    /// is exact in vacuum only. ponytail: a dual grid with mu would lift it.
    TeNeedsVacuum,
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
            Self::TeNeedsVacuum => "a TE plane wave needs vacuum as material 0",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The correction loops as they were written before they became data.
    /// The lists must reproduce them bit for bit.
    fn ez_at(t: &Tfsf, i: usize, j: usize) -> f64 {
        t.ez_inc(if t.along_x() { i } else { j })
    }

    fn loops_h(t: &Tfsf, hx: &mut [f64], hy: &mut [f64], nx: usize, s: f64, periodic_y: bool) {
        let PlaneWave { x0, y0, x1, y1, .. } = t.pw;
        for j in y0..=y1 {
            hy[j * nx + x0 - 1] -= s * ez_at(t, x0, j);
            hy[j * nx + x1] += s * ez_at(t, x1, j);
        }
        if !periodic_y {
            for i in x0..=x1 {
                hx[(y0 - 1) * nx + i] += s * ez_at(t, i, y0);
                hx[y1 * nx + i] -= s * ez_at(t, i, y1);
            }
        }
    }

    fn loops_e(t: &Tfsf, ez: &mut [f64], cb: &[f64], nx: usize) {
        let PlaneWave { x0, y0, x1, y1, .. } = t.pw;
        if t.along_x() {
            for j in y0..=y1 {
                let (a, b) = (j * nx + x0, j * nx + x1);
                ez[a] -= cb[a] * t.h_inc(x0 - 1);
                ez[b] += cb[b] * t.h_inc(x1);
            }
        } else {
            for i in x0..=x1 {
                let (a, b) = (y0 * nx + i, y1 * nx + i);
                ez[a] += cb[a] * t.h_inc(y0 - 1);
                ez[b] -= cb[b] * t.h_inc(y1);
            }
        }
    }

    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Every direction, closed and periodic, a normal box and boxes one node
    /// thick along each axis, at several auxiliary time levels.
    #[test]
    fn correction_lists_equal_the_loops() {
        let (nx, ny, s) = (40, 30, 0.5);
        let boxes = [(10, 8, 25, 20), (17, 8, 17, 20), (10, 13, 25, 13)];
        let dirs = [
            Direction::PlusX,
            Direction::MinusX,
            Direction::PlusY,
            Direction::MinusY,
        ];
        for (b, &(x0, y0, x1, y1)) in boxes.iter().enumerate() {
            for (d, &direction) in dirs.iter().enumerate() {
                for periodic_y in [false, true] {
                    let pw = PlaneWave {
                        x0,
                        y0,
                        x1,
                        y1,
                        direction,
                        waveform: Waveform::Ricker {
                            t0: 20.0,
                            f_peak: 0.08,
                        },
                        amplitude: 1.3,
                    };
                    let mut t = Tfsf::new(&pw, 2.0, s).unwrap();
                    let seed = (b * 8 + d * 2 + periodic_y as usize) as u64;
                    let cb = noise(seed + 100, nx * ny);
                    for _ in 0..40 {
                        let (mut hx, mut hy) = (noise(seed, nx * ny), noise(seed + 1, nx * ny));
                        let (mut hx2, mut hy2) = (hx.clone(), hy.clone());
                        loops_h(&t, &mut hx, &mut hy, nx, s, periodic_y);
                        t.correct_h(&mut hx2, &mut hy2, nx, s, periodic_y);
                        t.advance();
                        let mut ez = noise(seed + 2, nx * ny);
                        let mut ez2 = ez.clone();
                        loops_e(&t, &mut ez, &cb, nx);
                        t.correct_e(&mut ez2, &cb, nx);
                        assert_eq!(bits(&hx), bits(&hx2));
                        assert_eq!(bits(&hy), bits(&hy2));
                        assert_eq!(bits(&ez), bits(&ez2));
                    }
                }
            }
        }
    }

    /// No two groups touch the same target, so a kernel may run groups in
    /// parallel. Checked on a box one node thick, the tightest case.
    #[test]
    fn groups_never_share_a_target() {
        let pw = PlaneWave {
            x0: 17,
            y0: 8,
            x1: 17,
            y1: 20,
            direction: Direction::PlusX,
            waveform: Waveform::Gaussian {
                t0: 60.0,
                tau: 10.0,
            },
            amplitude: 1.0,
        };
        let t = Tfsf::new(&pw, 1.0, 0.5).unwrap();
        let cb = vec![0.5; 40 * 30];
        for groups in [t.h_corrections(40, 0.5, false), t.e_corrections(40, &cb)] {
            let mut seen = std::collections::BTreeMap::new();
            for (g, group) in groups.iter().enumerate() {
                for c in group {
                    let owner = *seen.entry((c.field as u8, c.k)).or_insert(g);
                    assert_eq!(owner, g, "{c:?} is in two groups");
                }
            }
        }
    }

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
