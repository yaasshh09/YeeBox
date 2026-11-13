//! Source time waveforms. Time is normalized (one unit is dx / c0, so step n
//! happens at t = n * S).

use crate::error::{FdtdError, Result};

/// A source time function. All times and frequencies are normalized.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    /// `exp(-((t - t0) / tau)^2)`. Note that this waveform HAS a DC component,
    /// equal to `sqrt(pi) * tau`. Use it for pulses, not for anything that must
    /// leave no static charge behind.
    Gaussian { t0: f64, tau: f64 },
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
        }
    }

    /// Waveform value at normalized time `t`.
    pub fn value(&self, t: f64) -> f64 {
        match *self {
            Waveform::Gaussian { t0, tau } => {
                let u = (t - t0) / tau;
                (-u * u).exp()
            }
        }
    }
}
