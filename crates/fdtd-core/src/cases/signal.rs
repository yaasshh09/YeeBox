//! Signal helpers for the cases (and the acceptance tests). Normalized
//! units: sample n of a probe record is at time n S. Complex values are
//! `(re, im)`.

use std::f64::consts::PI;

pub fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// DTFT `sum_n w[n] p[n] exp(-j omega n S)` of a record, with an optional
/// window `w` of the same length.
pub fn dtft(p: &[f64], window: Option<&[f64]>, omega: f64, s: f64) -> (f64, f64) {
    p.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let v = window.map_or(v, |w| v * w[n]);
        let ph = omega * n as f64 * s;
        (re + v * ph.cos(), im - v * ph.sin())
    })
}

/// Complex quotient a / b.
pub fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Wavenumber between two probes `len` cells apart on the propagation axis.
///
/// When everything both probes record is a right-going wave of one guided
/// mode, the spectra satisfy F2(omega) = F1(omega) exp(-j k L) exactly, with
/// k the discrete wavenumber. So -arg(F2 / F1) / L measures k. The 2 pi branch
/// is the one nearest `k_guess`, valid while |k - k_guess| L < pi.
pub fn wavenumber_from_probes(
    p1: &[f64],
    p2: &[f64],
    omega: f64,
    s: f64,
    len: f64,
    k_guess: f64,
) -> f64 {
    let (a, b) = dtft(p1, None, omega, s);
    let (c, d) = dtft(p2, None, omega, s);
    // arg(F2 / F1) = arg(F2 * conj(F1))
    let phase = (d * a - c * b).atan2(c * a + d * b);
    let m = ((k_guess * len + phase) / (2.0 * PI)).round();
    (2.0 * PI * m - phase) / len
}

/// 4-term Blackman-Harris window, sidelobes below -92 dB, main lobe 8 bins wide.
pub fn blackman_harris(n: usize) -> Vec<f64> {
    let a = [0.35875, 0.48829, 0.14128, 0.01168];
    (0..n)
        .map(|k| {
            let x = 2.0 * PI * k as f64 / (n - 1) as f64;
            a[0] - a[1] * x.cos() + a[2] * (2.0 * x).cos() - a[3] * (3.0 * x).cos()
        })
        .collect()
}

/// Windowed spectrum magnitude at omega.
pub fn magnitude(p: &[f64], window: &[f64], omega: f64, s: f64) -> f64 {
    let (re, im) = dtft(p, Some(window), omega, s);
    re.hypot(im)
}

/// Golden-section search for the maximum of `f` on [lo, hi]. Returns the
/// argmax; the caller checks it is not stuck on an end of the bracket.
pub fn golden_max(f: impl Fn(f64) -> f64, mut lo: f64, mut hi: f64, tol: f64) -> f64 {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut x1 = hi - g * (hi - lo);
    let mut x2 = lo + g * (hi - lo);
    let (mut f1, mut f2) = (f(x1), f(x2));
    while hi - lo > tol {
        if f1 < f2 {
            lo = x1;
            (x1, f1) = (x2, f2);
            x2 = lo + g * (hi - lo);
            f2 = f(x2);
        } else {
            hi = x2;
            (x2, f2) = (x1, f1);
            x1 = hi - g * (hi - lo);
            f1 = f(x1);
        }
    }
    0.5 * (lo + hi)
}

/// Least-squares slope of y against x.
pub fn slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>()
}

/// Deterministic values in [-1, 1) from a 64-bit LCG, for random initial
/// fields (A4 and the oracle tests).
pub fn random_fields(seed: u64, n: usize) -> Vec<f64> {
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
