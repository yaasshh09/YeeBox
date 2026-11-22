//! Near-to-far-field transform in 2D (`docs/02-numerics.md`).
//!
//! A contour of Ez nodes (a rectangle, or some of its sides) records the
//! running DFT of the tangential fields with `monitor::FluxLine`, one line
//! per side, so the samples are co-located on the nodes and enter at their
//! true times. The equivalent currents are `J = n x H` and `M = -n x E`, with
//! `n` the outward normal. With the `exp(+j w t)` convention of the DFTs and
//! the large-argument form of the 2D Green's function
//! `H0^(2)(k |r - r'|) ~ sqrt(2j / (pi k rho)) exp(-j k rho) exp(j k rhat . r')`
//! the far field is
//!
//! ```text
//! TMz: Ez ~ F(phi) exp(-j k rho) / sqrt(rho),
//!      F = -(k/4) sqrt(2j / (pi k)) sum w [eta Jz - (cos phi My - sin phi Mx)] exp(j k rhat . r')
//! TEz: Hz ~ F(phi) exp(-j k rho) / sqrt(rho),
//!      F = -(k/4) sqrt(2j / (pi k)) sum w [Mz / eta + (cos phi Jy - sin phi Jx)] exp(j k rhat . r')
//! ```
//!
//! where `w` is the trapezoid weight of a node along its side (1, and 1/2 at
//! each end, so a closed rectangle counts each corner once in total), `r'` is
//! measured from the contour center, and `k`, `eta` are the continuum values
//! of the background (material 0): `k = 2 pi f sqrt(eps)`, `eta = 1 / sqrt(eps)`.
//! The TEz line is the dual of the TMz one (E to H, H to -E, J to M, M to -J,
//! eta to 1 / eta). Normalized units, angles in radians from +x,
//! counterclockwise.
//!
//! The phase term uses the continuum k. I tried the numerical k(phi) of
//! `dispersion` and it did not help: what the line source tests see is the
//! co-location instead. The node average of H is low by cos(k_n / 2) (1.2
//! percent at 20 cells per wavelength), which tips the J and M balance, so
//! a little of the source radiates as if it were travelling inward and its
//! phase cycles with the contour size. It is second order in the cell size
//! (`tests/acceptance_ntff.rs`).
// ponytail: one contour on Ez nodes with H averaged; a second contour on
// the H edges with E averaged would cancel the J and M imbalance to first
// order, add it if a gate ever needs better than about 0.1 dB at 20 cells
// per wavelength.

use crate::error::{FdtdError, Result};
use crate::monitor::{FluxLine, Normal, Tap};
use crate::scene::Mode;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// One side of a contour rectangle. JSON form: `"left"`, `"right"`,
/// `"bottom"`, `"top"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Left,
    Right,
    Bottom,
    Top,
}

/// The four sides, a closed contour.
pub const ALL_SIDES: [Side; 4] = [Side::Left, Side::Right, Side::Bottom, Side::Top];

/// The most far-field angles a monitor may ask for.
pub const MAX_ANGLES: usize = 3600;

impl Side {
    /// The flux line along this side of the node box `[x0, x1] x [y0, y1]`.
    fn line(self, x0: usize, y0: usize, x1: usize, y1: usize, freqs: &[f64]) -> Result<FluxLine> {
        match self {
            Side::Left => FluxLine::new(Normal::X, x0, y0, y1, freqs),
            Side::Right => FluxLine::new(Normal::X, x1, y0, y1, freqs),
            Side::Bottom => FluxLine::new(Normal::Y, y0, x0, x1, freqs),
            Side::Top => FluxLine::new(Normal::Y, y1, x0, x1, freqs),
        }
    }

    /// +1 when the outward normal points along +x or +y.
    fn outward(self) -> f64 {
        match self {
            Side::Left | Side::Bottom => -1.0,
            Side::Right | Side::Top => 1.0,
        }
    }
}

type C = (f64, f64);

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn scale(a: C, s: f64) -> C {
    (a.0 * s, a.1 * s)
}

/// One contour node as the integral sees it: its position (cell units,
/// relative to the contour center), trapezoid weight, side, and the
/// phasors of the node field (Ez or Hz) and the averaged edge field (Hy or
/// Hx in TMz, Ey or Ex in TEz) at one frequency.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sample {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub side: Side,
    pub node: C,
    pub edge: C,
}

/// The far-field amplitude `F(phi)` of the module doc from `samples` at
/// wavenumber `k` and impedance `eta` of the background.
pub(crate) fn integrate(mode: Mode, k: f64, eta: f64, samples: &[Sample], phi: f64) -> C {
    let (c, s) = (phi.cos(), phi.sin());
    let mut sum = (0.0, 0.0);
    for p in samples {
        let o = p.side.outward();
        let x_normal = matches!(p.side, Side::Left | Side::Right);
        // The bracket of the module doc for this node.
        let term = match (mode, x_normal) {
            // Jz = o Hy, My = o Ez.
            (Mode::Tm, true) => {
                let (jz, my) = (scale(p.edge, o), scale(p.node, o));
                (eta * jz.0 - c * my.0, eta * jz.1 - c * my.1)
            }
            // Jz = -o Hx, Mx = -o Ez.
            (Mode::Tm, false) => {
                let (jz, mx) = (scale(p.edge, -o), scale(p.node, -o));
                (eta * jz.0 + s * mx.0, eta * jz.1 + s * mx.1)
            }
            // Jy = -o Hz, Mz = -o Ey.
            (Mode::Te, true) => {
                let (jy, mz) = (scale(p.node, -o), scale(p.edge, -o));
                (mz.0 / eta + c * jy.0, mz.1 / eta + c * jy.1)
            }
            // Jx = o Hz, Mz = o Ex.
            (Mode::Te, false) => {
                let (jx, mz) = (scale(p.node, o), scale(p.edge, o));
                (mz.0 / eta - s * jx.0, mz.1 / eta - s * jx.1)
            }
        };
        let ph = k * (c * p.x + s * p.y);
        let t = cmul(term, (ph.cos(), ph.sin()));
        sum.0 += p.w * t.0;
        sum.1 += p.w * t.1;
    }
    // -(k/4) sqrt(2 / (pi k)) exp(j pi / 4)
    let a = -(k / 4.0) * (2.0 / (PI * k)).sqrt();
    let q = PI / 4.0;
    cmul(sum, (a * q.cos(), a * q.sin()))
}

/// Scattering width `2 pi |F|^2 / |inc|^2` (cells) from a far-field
/// amplitude and the incident field phasor at the same frequency.
pub fn scattering_width(f: C, inc: C) -> f64 {
    2.0 * PI * (f.0 * f.0 + f.1 * f.1) / (inc.0 * inc.0 + inc.1 * inc.1)
}

/// A far-field monitor as the solver runs it: one `FluxLine` per side of
/// the node box `[x0, x1] x [y0, y1]`.
#[derive(Debug, Clone)]
pub struct Ntff {
    mode: Mode,
    eps: f64,
    center: (f64, f64),
    angles: usize,
    lines: Vec<(Side, FluxLine)>,
}

impl Ntff {
    /// A contour over the listed sides of the node box, recording `freqs`,
    /// with `angles` far-field directions `2 pi m / angles`, in a
    /// background of permittivity `eps`. The scene checks the placement
    /// against the grid; this checks the box, the side list, the angle
    /// count and the frequencies.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
        sides: &[Side],
        freqs: &[f64],
        angles: usize,
        mode: Mode,
        eps: f64,
    ) -> Result<Self> {
        let bad = |name, value: f64| Err(FdtdError::InvalidParameter { name, value });
        if x0 >= x1 || y0 >= y1 {
            let size = (x1 as f64 - x0 as f64).min(y1 as f64 - y0 as f64);
            return bad("far-field contour size", size);
        }
        if sides.is_empty() || (1..sides.len()).any(|i| sides[..i].contains(&sides[i])) {
            return bad("far-field contour sides", sides.len() as f64);
        }
        if !(1..=MAX_ANGLES).contains(&angles) {
            return bad("far-field angles", angles as f64);
        }
        if !(eps.is_finite() && eps > 0.0) {
            return bad("far-field background eps", eps);
        }
        let lines = sides
            .iter()
            .map(|&side| Ok((side, side.line(x0, y0, x1, y1, freqs)?.in_mode(mode))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            mode,
            eps,
            center: ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0),
            angles,
            lines,
        })
    }

    pub fn freqs(&self) -> &[f64] {
        self.lines[0].1.freqs()
    }

    pub fn angles(&self) -> usize {
        self.angles
    }

    /// The contour center in cell units, the phase origin of `far_field`.
    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    /// The taps of every side in order, three per node (see
    /// `FluxLine::taps`).
    pub fn taps(&self, nx: usize, ny: usize) -> Vec<Tap> {
        self.lines
            .iter()
            .flat_map(|(_, l)| l.taps(nx, ny))
            .collect()
    }

    /// How many taps `taps` returns.
    pub fn taps_len(&self) -> usize {
        self.lines.iter().map(|(_, l)| 3 * l.len()).sum()
    }

    /// Adds one step from the values of `taps`, taken after step `n`.
    pub fn record_taps(&mut self, values: &[f64], n: u64, s: f64) {
        let mut at = 0;
        for (_, line) in &mut self.lines {
            let len = 3 * line.len();
            line.record_taps(&values[at..at + len], n, s);
            at += len;
        }
    }

    /// Zeroes every transform (after a live edit).
    pub fn restart(&mut self) {
        for (_, line) in &mut self.lines {
            line.restart();
        }
    }

    /// The far-field amplitude `F(phi_m, f)` at every frequency and angle,
    /// entry `f * angles + m`, `phi_m = 2 pi m / angles`. Normalized.
    pub fn far_field(&self) -> Vec<C> {
        let n = self.eps.sqrt();
        let mut out = Vec::with_capacity(self.freqs().len() * self.angles);
        for (f, &freq) in self.freqs().iter().enumerate() {
            let samples = self.samples(f);
            let k = 2.0 * PI * freq * n;
            for m in 0..self.angles {
                let phi = 2.0 * PI * m as f64 / self.angles as f64;
                out.push(integrate(self.mode, k, 1.0 / n, &samples, phi));
            }
        }
        out
    }

    fn samples(&self, f: usize) -> Vec<Sample> {
        let (cx, cy) = self.center;
        let mut out = Vec::new();
        for (side, line) in &self.lines {
            let last = line.len() - 1;
            for p in 0..=last {
                let q = (line.from + p) as f64;
                let at = line.at as f64;
                let (x, y) = match line.normal {
                    Normal::X => (at, q),
                    Normal::Y => (q, at),
                };
                out.push(Sample {
                    x: x - cx,
                    y: y - cy,
                    w: if p == 0 || p == last { 0.5 } else { 1.0 },
                    side: *side,
                    node: line.node_dft(p, f),
                    edge: line.edge_dft(p, f),
                });
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Composite Simpson rule over [a, b] with `m` (even) intervals.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, m: usize) -> f64 {
        let h = (b - a) / m as f64;
        let inner: f64 = (1..m)
            .map(|i| f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
            .sum();
        (f(a) + f(b) + inner) * h / 3.0
    }

    /// Bessel J_n and Y_n for x > 0 from their integral forms, for test
    /// fields only:
    /// `J_n = (1/pi) int_0^pi cos(n t - x sin t) dt`,
    /// `Y_n = (1/pi) int_0^pi sin(x sin t - n t) dt
    ///      - (1/pi) int_0^inf (e^(n t) + (-1)^n e^(-n t)) e^(-x sinh t) dt`.
    fn bessel(n: i32, x: f64) -> (f64, f64) {
        let n = n as f64;
        let j = simpson(|t| (n * t - x * t.sin()).cos(), 0.0, PI, 2000) / PI;
        let y1 = simpson(|t| (x * t.sin() - n * t).sin(), 0.0, PI, 2000) / PI;
        let sign = if n as i32 % 2 == 0 { 1.0 } else { -1.0 };
        let tail = |t: f64| ((n * t).exp() + sign * (-n * t).exp()) * (-x * t.sinh()).exp();
        (j, y1 - simpson(tail, 0.0, 8.0, 8000) / PI)
    }

    /// H_n^(2)(x) = J_n - j Y_n.
    fn hankel2(n: i32, x: f64) -> C {
        let (j, y) = bessel(n, x);
        (j, -y)
    }

    #[test]
    fn bessel_values() {
        // Abramowitz and Stegun table 9.1.
        let (j0, y0) = bessel(0, 1.0);
        let (j1, y1) = bessel(1, 1.0);
        assert!((j0 - 0.765_197_686_6).abs() < 1e-9, "{j0}");
        assert!((y0 - 0.088_256_964_2).abs() < 1e-9, "{y0}");
        assert!((j1 - 0.440_050_585_7).abs() < 1e-9, "{j1}");
        assert!((y1 + 0.781_212_821_3).abs() < 1e-9, "{y1}");
    }

    /// Exact phasors (Ez, Hx, Hy) of a line current `I = 1` at `r0` in a
    /// medium with wavenumber `k` and impedance `eta`:
    /// `Ez = -(k eta / 4) H0^(2)(k R)` and `H = (j / (w mu)) (dEz/dy, -dEz/dx)`,
    /// with `d/dR H0^(2)(k R) = -k H1^(2)(k R)` and `w mu = k eta`.
    fn line_source(k: f64, eta: f64, r0: (f64, f64), x: f64, y: f64) -> (C, C, C) {
        let (dx, dy) = (x - r0.0, y - r0.1);
        let r = dx.hypot(dy);
        let a = -k * eta / 4.0;
        let ez = scale(hankel2(0, k * r), a);
        let dr = scale(hankel2(1, k * r), -a * k);
        let (dex, dey) = (scale(dr, dx / r), scale(dr, dy / r));
        let j_over = |v: C| (-v.1 / (k * eta), v.0 / (k * eta));
        (ez, j_over(dey), j_over(scale(dex, -1.0)))
    }

    /// The analytic far field of that line source, referred to the origin.
    fn line_far(k: f64, eta: f64, r0: (f64, f64), phi: f64) -> C {
        let a = -(k * eta / 4.0) * (2.0 / (PI * k)).sqrt();
        let ph = PI / 4.0 + k * (phi.cos() * r0.0 + phi.sin() * r0.1);
        (a * ph.cos(), a * ph.sin())
    }

    /// The samples of a 2L x 2L contour centered on the origin, from a field
    /// function giving (Ez, Hx, Hy) at a point.
    fn contour(l: usize, field: impl Fn(f64, f64) -> (C, C, C)) -> Vec<Sample> {
        let mut out = Vec::new();
        let l = l as i64;
        for side in ALL_SIDES {
            for q in -l..=l {
                let (x, y) = match side {
                    Side::Left => (-l, q),
                    Side::Right => (l, q),
                    Side::Bottom => (q, -l),
                    Side::Top => (q, l),
                };
                let (ez, hx, hy) = field(x as f64, y as f64);
                let edge = match side {
                    Side::Left | Side::Right => hy,
                    Side::Bottom | Side::Top => hx,
                };
                out.push(Sample {
                    x: x as f64,
                    y: y as f64,
                    w: if q.abs() == l { 0.5 } else { 1.0 },
                    side,
                    node: ez,
                    edge,
                });
            }
        }
        out
    }

    /// Worst relative error of the integral against the analytic far field
    /// of a line source at `r0` on a 2L x 2L contour, over 24 angles.
    fn line_source_error(eps: f64, f: f64, l: usize, r0: (f64, f64)) -> f64 {
        let (k, eta) = (2.0 * PI * f * eps.sqrt(), 1.0 / eps.sqrt());
        let s = contour(l, |x, y| line_source(k, eta, r0, x, y));
        (0..24)
            .map(|m| {
                let phi = 2.0 * PI * m as f64 / 24.0;
                let got = integrate(Mode::Tm, k, eta, &s, phi);
                let want = line_far(k, eta, r0, phi);
                (got.0 - want.0).hypot(got.1 - want.1) / want.0.hypot(want.1)
            })
            .fold(0.0, f64::max)
    }

    /// The integral against the exact field of a line source, independent of
    /// the solver: with the source inside the contour it gives the analytic
    /// far field, level and phase. What is left is the trapezoid rule's end
    /// term at the corners: 1.6e-3 at 20 cells per wavelength and 2.9e-3 at
    /// 13 (eps 2.25), falling 2.4 to 2.8 times when the wavelength in cells
    /// doubles. A formula error would not fall with resolution.
    #[test]
    fn exact_line_source_field() {
        for eps in [1.0, 2.25] {
            for r0 in [(0.0, 0.0), (7.0, -4.0)] {
                let (fine, coarse) = (
                    line_source_error(eps, 0.025, 40, r0),
                    line_source_error(eps, 0.05, 40, r0),
                );
                assert!(coarse < 4e-3, "eps {eps} r0 {r0:?}: {coarse}");
                assert!(
                    fine < coarse / 2.0,
                    "eps {eps} r0 {r0:?}: {fine} vs {coarse}"
                );
            }
        }
    }

    /// With the source outside the contour the equivalent currents radiate
    /// nothing (extinction). A sign slip between the J and M terms turns
    /// this into a field of the source's own size.
    #[test]
    fn source_outside_the_contour_gives_no_far_field() {
        for eps in [1.0_f64, 2.25] {
            let (k, eta) = (2.0 * PI * 0.05 * eps.sqrt(), 1.0 / eps.sqrt());
            let outside = (60.0, 15.0);
            let s = contour(40, |x, y| line_source(k, eta, outside, x, y));
            let level = line_far(k, eta, outside, 0.0);
            for m in 0..24 {
                let got = integrate(Mode::Tm, k, eta, &s, 2.0 * PI * m as f64 / 24.0);
                let ratio = got.0.hypot(got.1) / level.0.hypot(level.1);
                assert!(ratio < 4e-3, "eps {eps}: {ratio}");
            }
        }
    }

    /// TEz is the dual of TMz: with eta = 1, the TEz fields Hz = Ez and
    /// E = -H of a magnetic line source give, term by term, the TMz integral
    /// of the electric one, so the two far fields are equal.
    #[test]
    fn te_is_the_dual_of_tm() {
        let (k, eta) = (2.0 * PI * 0.05, 1.0);
        let tm = contour(30, |x, y| line_source(k, eta, (3.0, 5.0), x, y));
        let te: Vec<Sample> = tm
            .iter()
            .map(|p| Sample {
                edge: scale(p.edge, -1.0),
                ..*p
            })
            .collect();
        for m in 0..12 {
            let phi = 2.0 * PI * m as f64 / 12.0;
            let a = integrate(Mode::Tm, k, eta, &tm, phi);
            let b = integrate(Mode::Te, k, eta, &te, phi);
            assert!(
                (a.0 - b.0).abs() < 1e-14 && (a.1 - b.1).abs() < 1e-14,
                "{a:?} {b:?}"
            );
        }
    }

    #[test]
    fn scattering_width_definition() {
        let want = 2.0 * PI * 25.0 / 4.0;
        assert!((scattering_width((3.0, 4.0), (0.0, 2.0)) - want).abs() < 1e-12);
    }

    #[test]
    fn constructor_checks() {
        let f = [0.05];
        let new = |x1, sides: &[Side], freqs: &[f64], angles| {
            Ntff::new(5, 5, x1, 30, sides, freqs, angles, Mode::Tm, 1.0)
        };
        let ok = new(20, &ALL_SIDES, &f, 360).unwrap();
        assert_eq!(ok.taps(50, 50).len(), 3 * (2 * 26 + 2 * 16));
        assert_eq!(ok.far_field().len(), 360);
        assert_eq!(ok.center(), (12.5, 17.5));
        assert!(new(5, &ALL_SIDES, &f, 360).is_err());
        assert!(new(20, &[], &f, 360).is_err());
        assert!(new(20, &[Side::Top, Side::Top], &f, 360).is_err());
        assert!(new(20, &ALL_SIDES, &f, 0).is_err());
        assert!(new(20, &ALL_SIDES, &f, MAX_ANGLES + 1).is_err());
        assert!(new(20, &ALL_SIDES, &[], 360).is_err());
    }
}
