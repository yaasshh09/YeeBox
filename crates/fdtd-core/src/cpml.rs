//! Convolutional PML (Roden and Gedney 2000) with the complex frequency
//! shifted stretch `s = kappa + sigma / (alpha + j w)`, normalized
//! (eps0 = 1, dx = 1, dt = S). See `docs/02-numerics.md`.
//!
//! The layer has the same thickness `d` on all four sides. The PEC wall stays
//! at node 0 and node n - 1, and the layer runs from the wall (depth d) to the
//! interface node d (depth 0).

use crate::error::{FdtdError, Result};
use serde::{Deserialize, Serialize};

/// CPML layer parameters, the same on all four sides. `sigma_scale`
/// multiplies `sigma_opt`; `alpha_max` is normalized like a conductivity
/// (`alpha_n = alpha_SI * eta0 * dx`).
///
/// JSON form: `{"thickness": 10, "m": 3, "sigma_scale": 1, "kappa_max": 5, "alpha_max": 0.05}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmlSpec {
    /// Cells between the PEC wall and the interface node. 0 means no layer.
    pub thickness: usize,
    /// Polynomial grading order of sigma and kappa.
    pub m: f64,
    /// sigma_max as a multiple of `sigma_opt = 0.8 (m + 1) / sqrt(eps_r)`.
    pub sigma_scale: f64,
    /// kappa at the wall; 1 at the interface.
    pub kappa_max: f64,
    /// alpha at the interface; 0 at the wall (linear grading).
    pub alpha_max: f64,
}

impl PmlSpec {
    /// Rejects non-finite values, `m < 1`, `sigma_scale < 0`, `kappa_max < 1`
    /// and `alpha_max < 0`. Whether the layer fits the grid is the scene's check.
    pub fn validate(&self) -> Result<()> {
        let checks = [
            ("m", self.m, self.m >= 1.0),
            ("sigma_scale", self.sigma_scale, self.sigma_scale >= 0.0),
            ("kappa_max", self.kappa_max, self.kappa_max >= 1.0),
            ("alpha_max", self.alpha_max, self.alpha_max >= 0.0),
        ];
        for (name, value, ok) in checks {
            if !(ok && value.is_finite()) {
                return Err(FdtdError::InvalidParameter { name, value });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn spec(thickness: usize) -> PmlSpec {
        PmlSpec {
            thickness,
            m: 3.0,
            sigma_scale: 1.0,
            kappa_max: 5.0,
            alpha_max: 0.05,
        }
    }

    #[test]
    fn validate_ranges() {
        assert!(spec(10).validate().is_ok());
        let bad = |f: fn(&mut PmlSpec)| {
            let mut p = spec(10);
            f(&mut p);
            p.validate().unwrap_err()
        };
        for (err, name) in [
            (bad(|p| p.m = 0.5), "m"),
            (bad(|p| p.sigma_scale = -1.0), "sigma_scale"),
            (bad(|p| p.kappa_max = 0.9), "kappa_max"),
            (bad(|p| p.alpha_max = -0.1), "alpha_max"),
            (bad(|p| p.alpha_max = f64::INFINITY), "alpha_max"),
            (bad(|p| p.m = f64::NAN), "m"),
        ] {
            assert!(
                matches!(err, FdtdError::InvalidParameter { name: n, .. } if n == name),
                "{err:?}"
            );
        }
    }
}
