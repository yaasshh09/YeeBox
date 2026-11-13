//! Numerical dispersion of the Yee scheme, normalized (dx = 1, c0 = 1, dt = S).
//!
//! The discrete plane wave `exp(j(omega t - k.x))` satisfies
//! `(sin(omega S / 2) / S)^2 = sin^2(k cos(theta) / 2) + sin^2(k sin(theta) / 2)`
//! in 2D, and `sin(omega S / 2) / S = sin(k / 2)` in 1D. `omega` is radians per
//! normalized time unit, `k` is radians per cell.

use crate::error::{FdtdError, Result};
use std::f64::consts::PI;

/// Courant stability limit `1 / sqrt(dims)` for 1, 2, or 3 dimensions.
pub fn courant_limit(dims: u32) -> Result<f64> {
    match dims {
        1..=3 => Ok(1.0 / f64::from(dims).sqrt()),
        _ => Err(FdtdError::InvalidParameter {
            name: "dims",
            value: f64::from(dims),
        }),
    }
}

/// Checks omega and S, returns `sin(omega S / 2) / S`.
fn check_and_rhs(omega: f64, s: f64, dims: u32) -> Result<f64> {
    if !(omega.is_finite() && omega > 0.0) {
        return Err(FdtdError::InvalidParameter {
            name: "omega",
            value: omega,
        });
    }
    if !(s.is_finite() && s > 0.0) {
        return Err(FdtdError::InvalidParameter {
            name: "courant",
            value: s,
        });
    }
    let limit = courant_limit(dims)?;
    // A few ulps of slack so `1.0 / 2.0.sqrt()` and FRAC_1_SQRT_2 both count as the limit.
    if s > limit * (1.0 + 4.0 * f64::EPSILON) {
        return Err(FdtdError::CourantAboveLimit { s, limit });
    }
    // Above the temporal Nyquist frequency omega * S = pi the sampling aliases.
    if omega * s > PI {
        return Err(FdtdError::AboveGridCutoff { omega, s });
    }
    Ok((omega * s / 2.0).sin() / s)
}

/// 1D numerical wavenumber, closed form `k = 2 asin(sin(omega S / 2) / S)`.
/// Requires 0 < S <= 1 and omega below the grid cutoff.
pub fn numerical_wavenumber_1d(omega: f64, s: f64) -> Result<f64> {
    let r = check_and_rhs(omega, s, 1)?;
    if r > 1.0 {
        return Err(FdtdError::AboveGridCutoff { omega, s });
    }
    Ok(2.0 * r.asin())
}

/// 2D numerical wavenumber magnitude for propagation angle `theta` (radians
/// from the x axis). Requires 0 < S <= 1/sqrt(2).
///
/// Solves `f(k) = sin^2(k c / 2) + sin^2(k s / 2) - r^2 = 0` with
/// `c = |cos theta|`, `s = |sin theta|`. On `[0, pi / max(c, s)]`, f is monotonic
/// increasing, so that interval is a bracket. Newton steps that leave the
/// bracket fall back to bisection.
pub fn numerical_wavenumber(omega: f64, theta: f64, s: f64) -> Result<f64> {
    if !theta.is_finite() {
        return Err(FdtdError::InvalidParameter {
            name: "theta",
            value: theta,
        });
    }
    let r = check_and_rhs(omega, s, 2)?;
    let target = r * r;
    let (c, sn) = (theta.cos().abs(), theta.sin().abs());
    let f = |k: f64| (k * c / 2.0).sin().powi(2) + (k * sn / 2.0).sin().powi(2) - target;
    let df = |k: f64| (c * (k * c).sin() + sn * (k * sn).sin()) / 2.0;

    let (mut lo, mut hi) = (0.0, PI / c.max(sn));
    if f(hi) < 0.0 {
        return Err(FdtdError::AboveGridCutoff { omega, s });
    }
    let mut k = omega.clamp(lo, hi);
    for _ in 0..200 {
        let fk = f(k);
        if fk == 0.0 {
            return Ok(k);
        }
        if fk < 0.0 {
            lo = k;
        } else {
            hi = k;
        }
        let d = df(k);
        let newton = k - fk / d;
        let next = if d > 0.0 && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if (next - k).abs() <= 1e-15 * k {
            return Ok(next);
        }
        k = next;
    }
    Ok(k)
}

/// Numerical phase velocity `omega / k` in units of c0, for angle `theta` in 2D.
pub fn phase_velocity(omega: f64, theta: f64, s: f64) -> Result<f64> {
    Ok(omega / numerical_wavenumber(omega, theta, s)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::f64::consts::FRAC_PI_4;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn courant_limits() {
        assert_eq!(courant_limit(1).unwrap(), 1.0);
        assert!(rel(courant_limit(2).unwrap(), std::f64::consts::FRAC_1_SQRT_2) < 1e-15);
        assert!(rel(courant_limit(3).unwrap(), 1.0 / 3f64.sqrt()) < 1e-15);
        assert!(courant_limit(0).is_err());
        assert!(courant_limit(4).is_err());
    }

    /// At S = 1 the 1D scheme has no dispersion at all: k = omega.
    #[test]
    fn magic_step_1d_exact() {
        for omega in [0.01, 0.3, 1.0, 2.0, 3.0] {
            let k = numerical_wavenumber_1d(omega, 1.0).unwrap();
            assert!(rel(k, omega) < 1e-15, "omega {omega}: k {k}");
        }
    }

    /// At the 2D limit S = 1/sqrt(2), diagonal propagation is exact.
    #[test]
    fn diagonal_exact_at_2d_limit() {
        let s = courant_limit(2).unwrap();
        for omega in [0.05, 0.5, 1.5, 2.5] {
            let k = numerical_wavenumber(omega, FRAC_PI_4, s).unwrap();
            assert!(rel(k, omega) < 1e-14, "omega {omega}: k {k}");
        }
    }

    #[test]
    fn axis_2d_equals_1d() {
        for (omega, s) in [(0.1, 0.5), (1.0, 0.5), (2.0, 0.7), (0.3, 0.1)] {
            let k2 = numerical_wavenumber(omega, 0.0, s).unwrap();
            let k1 = numerical_wavenumber_1d(omega, s).unwrap();
            assert!(rel(k2, k1) < 1e-14, "omega {omega} S {s}: {k2} vs {k1}");
        }
    }

    proptest! {
        #[test]
        fn residual_zero(omega in 0.01..1.5_f64, theta in 0.0..(2.0 * PI), s in 0.05..0.7_f64) {
            let k = numerical_wavenumber(omega, theta, s).unwrap();
            let r = (omega * s / 2.0).sin() / s;
            let f = (k * theta.cos() / 2.0).sin().powi(2) + (k * theta.sin() / 2.0).sin().powi(2) - r * r;
            prop_assert!(f.abs() < 1e-14, "residual {}", f);
        }

        #[test]
        fn symmetries(omega in 0.01..1.5_f64, theta in 0.0..(PI / 2.0), s in 0.05..0.7_f64) {
            let k = numerical_wavenumber(omega, theta, s).unwrap();
            for other in [-theta, PI / 2.0 - theta, PI - theta, theta + PI] {
                let k2 = numerical_wavenumber(omega, other, s).unwrap();
                prop_assert!(rel(k2, k) < 1e-14, "theta {} vs {}: {} vs {}", theta, other, k, k2);
            }
        }
    }

    /// The axis is the slowest direction for S below the 2D limit.
    #[test]
    fn axis_slower_than_diagonal() {
        for s in [0.1, 0.5, 0.7] {
            let k_axis = numerical_wavenumber(0.5, 0.0, s).unwrap();
            let k_diag = numerical_wavenumber(0.5, FRAC_PI_4, s).unwrap();
            assert!(k_axis > k_diag, "S {s}: axis {k_axis}, diagonal {k_diag}");
        }
    }

    /// Series expansion of the relation: (k - omega) / omega -> (c^4 + s^4 - S^2) omega^2 / 24.
    /// On the axis that is (1 - S^2) omega^2 / 24, the 1D result.
    #[test]
    fn leading_error_coefficient() {
        let omega = 0.02;
        for (theta, s) in [(0.0, 0.5), (0.0, 0.2), (0.3, 0.5), (FRAC_PI_4, 0.4)] {
            let (c, sn) = (f64::cos(theta), f64::sin(theta));
            let predicted = (c.powi(4) + sn.powi(4) - s * s) * omega * omega / 24.0;
            let k = numerical_wavenumber(omega, theta, s).unwrap();
            let measured = (k - omega) / omega;
            assert!(
                rel(measured, predicted) < 1e-3,
                "theta {theta} S {s}: {measured:e} vs {predicted:e}"
            );
        }
        let k1 = numerical_wavenumber_1d(omega, 0.5).unwrap();
        let predicted = (1.0 - 0.25) * omega * omega / 24.0;
        assert!(rel((k1 - omega) / omega, predicted) < 1e-3);
    }

    /// Log-log slope of the relative wavenumber error against cells per
    /// wavelength N, with omega = 2 pi / N. Second order means slope -2.
    #[test]
    fn second_order_convergence() {
        for theta in [0.0, PI / 8.0] {
            let ns = [20.0, 40.0, 80.0, 160.0];
            let pts: Vec<(f64, f64)> = ns
                .iter()
                .map(|&n: &f64| {
                    let omega = 2.0 * PI / n;
                    let k = numerical_wavenumber(omega, theta, 0.5).unwrap();
                    (n.ln(), ((k - omega) / omega).abs().ln())
                })
                .collect();
            let m = pts.len() as f64;
            let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
            let (mx, my) = (sx / m, sy / m);
            let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
            let den: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
            let order = -num / den;
            assert!(
                (1.95..=2.05).contains(&order),
                "theta {theta}: order {order}"
            );
        }
    }

    #[test]
    fn above_cutoff_errors() {
        // 1D at S = 0.5: sin(omega / 4) / 0.5 > 1 once omega > 2 pi / 3.
        assert!(matches!(
            numerical_wavenumber_1d(2.5, 0.5),
            Err(FdtdError::AboveGridCutoff { .. })
        ));
        // 2D on the axis the right side can reach at most 1.
        assert!(matches!(
            numerical_wavenumber(2.9, 0.0, 0.5),
            Err(FdtdError::AboveGridCutoff { .. })
        ));
        assert!(matches!(
            numerical_wavenumber(0.5, 0.0, 0.71),
            Err(FdtdError::CourantAboveLimit { .. })
        ));
        assert!(numerical_wavenumber(0.5, 0.0, std::f64::consts::FRAC_1_SQRT_2).is_ok());
        assert!(numerical_wavenumber_1d(-1.0, 0.5).is_err());
        assert!(numerical_wavenumber(0.5, f64::NAN, 0.5).is_err());
    }
}
