//! Source time waveforms. Time is normalized: one unit is dx / c0, so step n
//! happens at t = n * S. Frequencies are cycles per normalized time unit
//! (cycles per step is f * S).

use crate::error::{FdtdError, Result};
use std::f64::consts::PI;

/// A source time function. All times and frequencies are normalized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    /// `exp(-((t - t0) / tau)^2)`. This waveform HAS a DC component: its
    /// integral is `sqrt(pi) * tau`. Use it for pulses, not for anything that
    /// must leave no static field behind. Use t0 >= 6 tau so the value at t = 0 is
    /// below 1e-15; a visible turn-on step injects a grid-scale checkerboard.
    Gaussian { t0: f64, tau: f64 },
    /// `sin(2 pi f0 (t - t0)) * exp(-((t - t0) / tau)^2)`. Center frequency
    /// `f0`, bandwidth set by `tau`. Zero DC exactly, by odd symmetry about t0.
    ModulatedGaussian { t0: f64, tau: f64, f0: f64 },
    /// Ricker wavelet `(1 - 2 u^2) exp(-u^2)` with `u = pi f_peak (t - t0)`.
    /// Spectral peak at `f_peak`, zero DC exactly.
    Ricker { t0: f64, f_peak: f64 },
    /// `sin(2 pi f0 t + phase)` turned on over `ramp_periods` periods by the
    /// raised-cosine ramp `(1 - cos(pi t / T)) / 2`, with `T = ramp_periods / f0`.
    /// Zero for t < 0, value and slope continuous everywhere. At least 3
    /// periods, because a hard turn-on injects broadband junk and DC.
    RampedCw {
        f0: f64,
        ramp_periods: f64,
        phase: f64,
    },
}

/// How a source couples into its node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceKind {
    /// Adds the waveform to the field. Transparent to passing waves. The default.
    #[default]
    Soft,
    /// Overwrites the field with the waveform. Reflects passing waves like a wall.
    Hard,
}

fn check(name: &'static str, value: f64, ok: bool) -> Result<()> {
    if ok && value.is_finite() {
        Ok(())
    } else {
        Err(FdtdError::InvalidParameter { name, value })
    }
}

impl Waveform {
    /// Checks that every parameter is finite and in range.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Waveform::Gaussian { t0, tau } => {
                check("t0", t0, true)?;
                check("tau", tau, tau > 0.0)
            }
            Waveform::ModulatedGaussian { t0, tau, f0 } => {
                check("t0", t0, true)?;
                check("tau", tau, tau > 0.0)?;
                check("f0", f0, f0 > 0.0)
            }
            Waveform::Ricker { t0, f_peak } => {
                check("t0", t0, true)?;
                check("f_peak", f_peak, f_peak > 0.0)
            }
            Waveform::RampedCw {
                f0,
                ramp_periods,
                phase,
            } => {
                check("f0", f0, f0 > 0.0)?;
                check("ramp_periods", ramp_periods, ramp_periods >= 3.0)?;
                check("phase", phase, true)
            }
        }
    }

    /// Waveform value at normalized time `t`.
    pub fn value(&self, t: f64) -> f64 {
        match *self {
            Waveform::Gaussian { t0, tau } => {
                let u = (t - t0) / tau;
                (-u * u).exp()
            }
            Waveform::ModulatedGaussian { t0, tau, f0 } => {
                let u = (t - t0) / tau;
                (2.0 * PI * f0 * (t - t0)).sin() * (-u * u).exp()
            }
            Waveform::Ricker { t0, f_peak } => {
                let u2 = (PI * f_peak * (t - t0)).powi(2);
                (1.0 - 2.0 * u2) * (-u2).exp()
            }
            Waveform::RampedCw {
                f0,
                ramp_periods,
                phase,
            } => {
                if t <= 0.0 {
                    return 0.0;
                }
                let t_ramp = ramp_periods / f0;
                let ramp = if t < t_ramp {
                    0.5 * (1.0 - (PI * t / t_ramp).cos())
                } else {
                    1.0
                };
                ramp * (2.0 * PI * f0 * t + phase).sin()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Riemann sum of the waveform over [a, b] with step dt. For a smooth
    /// function that has decayed to zero at both ends this is spectrally accurate.
    fn integral(w: &Waveform, a: f64, b: f64, dt: f64) -> f64 {
        let n = ((b - a) / dt).round() as usize;
        (0..=n).map(|k| w.value(a + k as f64 * dt)).sum::<f64>() * dt
    }

    #[test]
    fn gaussian_peak_width_and_dc() {
        let (t0, tau) = (40.0, 8.0);
        let w = Waveform::Gaussian { t0, tau };
        assert_eq!(w.value(t0), 1.0);
        for t in [t0 - tau, t0 + tau] {
            assert!((w.value(t) - (-1.0_f64).exp()).abs() < 1e-15);
        }
        // Window of +-10 tau so truncation is below 1e-40. The start offset and dt
        // are not commensurate with t0, so this is not a symmetric-sample accident.
        let dc = integral(&w, t0 - 10.0 * tau + 0.123, t0 + 10.0 * tau, 0.37);
        assert!((dc - PI.sqrt() * tau).abs() < 1e-12, "dc {dc}");
    }

    #[test]
    fn modulated_gaussian_zero_dc() {
        let w = Waveform::ModulatedGaussian {
            t0: 50.0,
            tau: 12.0,
            f0: 0.05,
        };
        let dc = integral(&w, -69.877, 170.0, 0.37);
        assert!(dc.abs() < 1e-12, "dc {dc}");
    }

    #[test]
    fn ricker_zero_dc() {
        let w = Waveform::Ricker {
            t0: 50.0,
            f_peak: 0.04,
        };
        let dc = integral(&w, 0.0, 150.0, 0.37);
        assert!(dc.abs() < 1e-12, "dc {dc}");
        assert_eq!(w.value(50.0), 1.0);
    }

    #[test]
    fn cw_ramp_c1_continuous() {
        let (f0, periods) = (0.05, 3.0);
        let w = Waveform::RampedCw {
            f0,
            ramp_periods: periods,
            phase: 0.7,
        };
        let t_ramp = periods / f0;
        let h = 1e-6;
        for t in [0.0, t_ramp] {
            let (l, c, r) = (w.value(t - h), w.value(t), w.value(t + h));
            assert!((r - l).abs() < 1e-5, "value jump at {t}");
            let (sl, sr) = ((c - l) / h, (r - c) / h);
            assert!((sr - sl).abs() < 1e-4, "slope jump at {t}: {sl} vs {sr}");
        }
        // After the ramp it is the plain sinusoid.
        let t = t_ramp + 3.3;
        assert!((w.value(t) - (2.0 * PI * f0 * t + 0.7).sin()).abs() < 1e-15);
    }

    #[test]
    fn cw_ramp_rejects_short_ramp() {
        let w = Waveform::RampedCw {
            f0: 0.05,
            ramp_periods: 2.9,
            phase: 0.0,
        };
        assert!(w.validate().is_err());
    }

    #[test]
    fn rejects_nonpositive_and_nonfinite() {
        let bad = [
            Waveform::Gaussian { t0: 1.0, tau: 0.0 },
            Waveform::Gaussian {
                t0: f64::NAN,
                tau: 1.0,
            },
            Waveform::ModulatedGaussian {
                t0: 1.0,
                tau: 1.0,
                f0: -0.1,
            },
            Waveform::Ricker {
                t0: 1.0,
                f_peak: f64::INFINITY,
            },
        ];
        for w in bad {
            assert!(w.validate().is_err(), "{w:?}");
        }
    }
}
