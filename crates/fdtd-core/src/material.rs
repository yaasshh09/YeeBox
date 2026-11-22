//! Material table entries and the Ez update coefficients. Everything is
//! normalized (eps0 = mu0 = 1, dx = 1, dt = S). `mu_r` is 1 everywhere in v1, so
//! the H coefficient is just S and has no table entry.

use crate::error::{FdtdError, Result};
use serde::{Deserialize, Serialize};

/// One entry of a scene's material table. Each Ez node takes its material
/// from its own cell (staircased, no averaging in TMz).
///
/// JSON form is tagged: `{"kind": "dielectric", "eps_inf": 4, "sigma": 0}`,
/// `{"kind": "pec"}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Material {
    /// Relative permittivity `eps_inf >= 1` and electric conductivity
    /// `sigma >= 0` (normalized, `sigma_n = sigma_SI * eta0 * dx`). Vacuum is
    /// `eps_inf = 1, sigma = 0`; a lossy medium is any `sigma > 0`.
    Dielectric { eps_inf: f64, sigma: f64 },
    /// Perfect electric conductor: Ez is held at zero.
    Pec,
}

/// The materials the paint tools offer, by name, in picker order. The
/// absorber's normalized sigma of 0.1 takes a wave down by e in about 20
/// cells; it is a display material for soaking up light, not a matched
/// layer (that is the CPML's job).
pub const PALETTE: [(&str, Material); 5] = [
    ("vacuum", Material::VACUUM),
    (
        "glass",
        Material::Dielectric {
            eps_inf: 2.25,
            sigma: 0.0,
        },
    ),
    (
        "high index",
        Material::Dielectric {
            eps_inf: 12.0,
            sigma: 0.0,
        },
    ),
    ("conductor", Material::Pec),
    (
        "absorber",
        Material::Dielectric {
            eps_inf: 1.0,
            sigma: 0.1,
        },
    ),
];

impl Material {
    pub const VACUUM: Material = Material::Dielectric {
        eps_inf: 1.0,
        sigma: 0.0,
    };

    /// Rejects `eps_inf < 1` (a phase velocity above c0 would break the
    /// vacuum Courant limit), negative conductivity, and non-finite values.
    pub fn validate(&self) -> Result<()> {
        if let Material::Dielectric { eps_inf, sigma } = *self {
            if !(eps_inf.is_finite() && eps_inf >= 1.0) {
                return Err(FdtdError::InvalidParameter {
                    name: "eps_inf",
                    value: eps_inf,
                });
            }
            if !(sigma.is_finite() && sigma >= 0.0) {
                return Err(FdtdError::InvalidParameter {
                    name: "sigma",
                    value: sigma,
                });
            }
        }
        Ok(())
    }

    /// Relative permittivity used in the discrete energy. PEC returns 0, since
    /// its Ez is always zero and contributes nothing either way.
    pub fn eps(&self) -> f64 {
        match *self {
            Material::Dielectric { eps_inf, .. } => eps_inf,
            Material::Pec => 0.0,
        }
    }

    /// The material of a TEz E edge between cells `a` and `b`: the
    /// arithmetic mean of eps_inf and of sigma (`docs/01-theory.md`,
    /// Materials), and PEC if either cell is PEC. Staircased: no subpixel
    /// smoothing, so curved surfaces converge at first order.
    pub fn edge_mean(a: &Material, b: &Material) -> Material {
        match (*a, *b) {
            (
                Material::Dielectric {
                    eps_inf: ea,
                    sigma: sa,
                },
                Material::Dielectric {
                    eps_inf: eb,
                    sigma: sb,
                },
            ) => Material::Dielectric {
                eps_inf: 0.5 * (ea + eb),
                sigma: 0.5 * (sa + sb),
            },
            _ => Material::Pec,
        }
    }

    /// Ez update coefficients `(Ca, Cb)` for Courant number `s`, so that
    /// `Ez = Ca * Ez + Cb * curl_H` at a node of this material:
    ///
    /// `Ca = (1 - l) / (1 + l)`, `Cb = (s / eps) / (1 + l)`, `l = sigma s / (2 eps)`.
    ///
    /// Lossless gives exactly `(1, s / eps)`. PEC gives `(0, 0)`. Call
    /// `validate` first; this does not re-check.
    pub fn coefficients(&self, s: f64) -> (f64, f64) {
        match *self {
            Material::Dielectric { eps_inf, sigma } => {
                let l = sigma * s / (2.0 * eps_inf);
                ((1.0 - l) / (1.0 + l), (s / eps_inf) / (1.0 + l))
            }
            Material::Pec => (0.0, 0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_coefficients_exact() {
        for eps_inf in [1.0, 2.25, 4.0, 11.7] {
            let m = Material::Dielectric {
                eps_inf,
                sigma: 0.0,
            };
            assert_eq!(m.coefficients(0.5), (1.0, 0.5 / eps_inf));
        }
        assert_eq!(Material::VACUUM.coefficients(0.7), (1.0, 0.7));
    }

    /// As sigma grows, Ca falls toward -1 from above and never passes it.
    #[test]
    fn large_sigma_ca_toward_minus_one() {
        let mut last = 1.0;
        for sigma in [0.01, 0.1, 1.0, 10.0, 1e3, 1e6, 1e12] {
            let (ca, cb) = Material::Dielectric {
                eps_inf: 2.0,
                sigma,
            }
            .coefficients(0.5);
            assert!(ca < last && ca > -1.0, "sigma {sigma}: ca {ca}");
            assert!(cb > 0.0, "sigma {sigma}: cb {cb}");
            last = ca;
        }
        assert!(last < -0.999_999);
    }

    /// The TEz edge rule: the mean of the two cells, and PEC wins.
    #[test]
    fn edge_mean() {
        let glass = Material::Dielectric {
            eps_inf: 3.0,
            sigma: 0.25,
        };
        assert_eq!(
            Material::edge_mean(&Material::VACUUM, &glass),
            Material::Dielectric {
                eps_inf: 2.0,
                sigma: 0.125
            }
        );
        assert_eq!(Material::edge_mean(&glass, &glass), glass);
        assert_eq!(Material::edge_mean(&glass, &Material::Pec), Material::Pec);
        assert_eq!(Material::edge_mean(&Material::Pec, &glass), Material::Pec);
    }

    #[test]
    fn pec_zero() {
        assert_eq!(Material::Pec.coefficients(0.5), (0.0, 0.0));
        assert_eq!(Material::Pec.eps(), 0.0);
        assert!(Material::Pec.validate().is_ok());
    }

    #[test]
    fn rejects_invalid() {
        for (eps_inf, sigma) in [
            (0.99, 0.0),
            (f64::NAN, 0.0),
            (f64::INFINITY, 0.0),
            (1.0, -1e-9),
            (1.0, f64::NAN),
        ] {
            assert!(
                Material::Dielectric { eps_inf, sigma }.validate().is_err(),
                "{eps_inf} {sigma}"
            );
        }
        assert!(Material::VACUUM.validate().is_ok());
    }

    #[test]
    fn json_form() {
        let m: Material =
            serde_json::from_str(r#"{"kind": "dielectric", "eps_inf": 4, "sigma": 0.1}"#).unwrap();
        assert_eq!(
            m,
            Material::Dielectric {
                eps_inf: 4.0,
                sigma: 0.1
            }
        );
        let p: Material = serde_json::from_str(r#"{"kind": "pec"}"#).unwrap();
        assert_eq!(p, Material::Pec);
        let text = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<Material>(&text).unwrap(), m);
    }
}
