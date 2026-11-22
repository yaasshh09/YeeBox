//! Material table entries and the Ez update coefficients. Everything is
//! normalized (eps0 = mu0 = 1, dx = 1, dt = S). `mu_r` is 1 everywhere in v1, so
//! the H coefficient is just S and has no table entry.

use crate::error::{FdtdError, Result};
use serde::{Deserialize, Serialize};

/// One entry of a scene's material table. Each Ez node takes its material
/// from its own cell (staircased, no averaging in TMz).
///
/// JSON form is tagged: `{"kind": "dielectric", "eps_inf": 4, "sigma": 0}`,
/// `{"kind": "pec"}`, `{"kind": "drude", "eps_inf": 4, "wp": 0.9, "gamma":
/// 0.01}`, `{"kind": "lorentz", "eps_inf": 2, "d_eps": 3, "w0": 0.3,
/// "delta": 0.02}`. Frequencies are normalized angular frequencies (radians
/// per dx / c0); see `dispersive.rs` for the models and the update.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Material {
    /// Relative permittivity `eps_inf >= 1` and electric conductivity
    /// `sigma >= 0` (normalized, `sigma_n = sigma_SI * eta0 * dx`). Vacuum is
    /// `eps_inf = 1, sigma = 0`; a lossy medium is any `sigma > 0`.
    Dielectric { eps_inf: f64, sigma: f64 },
    /// Perfect electric conductor: Ez is held at zero.
    Pec,
    /// Drude metal, `eps(w) = eps_inf - wp^2 / (w^2 - j gamma w)`.
    Drude { eps_inf: f64, wp: f64, gamma: f64 },
    /// Lorentz resonance,
    /// `eps(w) = eps_inf + d_eps w0^2 / (w0^2 - w^2 + 2 j delta w)`.
    Lorentz {
        eps_inf: f64,
        d_eps: f64,
        w0: f64,
        delta: f64,
    },
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
    /// Dispersive poles need a positive strength (`wp`, `d_eps`) and
    /// resonance (`w0`), and non-negative damping. The Courant limit on `w0`
    /// depends on S, so `Scene::validate` checks it.
    pub fn validate(&self) -> Result<()> {
        let check = |name, value: f64, ok: bool| {
            if value.is_finite() && ok {
                Ok(())
            } else {
                Err(FdtdError::InvalidParameter { name, value })
            }
        };
        match *self {
            Material::Dielectric { eps_inf, sigma } => {
                check("eps_inf", eps_inf, eps_inf >= 1.0)?;
                check("sigma", sigma, sigma >= 0.0)
            }
            Material::Pec => Ok(()),
            Material::Drude { eps_inf, wp, gamma } => {
                check("eps_inf", eps_inf, eps_inf >= 1.0)?;
                check("wp", wp, wp > 0.0)?;
                check("gamma", gamma, gamma >= 0.0)
            }
            Material::Lorentz {
                eps_inf,
                d_eps,
                w0,
                delta,
            } => {
                check("eps_inf", eps_inf, eps_inf >= 1.0)?;
                check("d_eps", d_eps, d_eps > 0.0)?;
                check("w0", w0, w0 > 0.0)?;
                check("delta", delta, delta >= 0.0)
            }
        }
    }

    /// Drude or Lorentz: the update needs auxiliary fields, and the material
    /// must stay out of the CPML.
    pub fn is_dispersive(&self) -> bool {
        matches!(self, Material::Drude { .. } | Material::Lorentz { .. })
    }

    /// Relative permittivity used in the discrete energy (eps_inf for the
    /// dispersive kinds). PEC returns 0, since its Ez is always zero and
    /// contributes nothing either way.
    pub fn eps(&self) -> f64 {
        match *self {
            Material::Dielectric { eps_inf, .. }
            | Material::Drude { eps_inf, .. }
            | Material::Lorentz { eps_inf, .. } => eps_inf,
            Material::Pec => 0.0,
        }
    }

    /// Ez update coefficients `(Ca, Cb)` for Courant number `s`, so that
    /// `Ez = Ca * Ez + Cb * curl_H` at a node of this material:
    ///
    /// `Ca = (1 - l) / (1 + l)`, `Cb = (s / eps) / (1 + l)`, `l = sigma s / (2 eps)`.
    ///
    /// Lossless gives exactly `(1, s / eps)`. PEC gives `(0, 0)`. For a
    /// dispersive material these are the `Row` coefficients of
    /// `dispersive.rs`, to which the J and P terms add. Call `validate`
    /// first; this does not re-check.
    pub fn coefficients(&self, s: f64) -> (f64, f64) {
        match *self {
            Material::Dielectric { eps_inf, sigma } => {
                let l = sigma * s / (2.0 * eps_inf);
                ((1.0 - l) / (1.0 + l), (s / eps_inf) / (1.0 + l))
            }
            Material::Pec => (0.0, 0.0),
            Material::Drude { .. } | Material::Lorentz { .. } => {
                let row = crate::dispersive::Medium::of(self)
                    .map_or(crate::dispersive::Row::PEC, |m| m.row(s));
                (row.ca, row.cb)
            }
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
    fn dispersive_validation_and_json() {
        let d: Material =
            serde_json::from_str(r#"{"kind": "drude", "eps_inf": 4, "wp": 0.9, "gamma": 0.01}"#)
                .unwrap();
        assert!(d.validate().is_ok() && d.is_dispersive());
        let l: Material = serde_json::from_str(
            r#"{"kind": "lorentz", "eps_inf": 2, "d_eps": 3, "w0": 0.3, "delta": 0}"#,
        )
        .unwrap();
        assert!(l.validate().is_ok() && l.is_dispersive());
        assert_eq!(
            serde_json::from_str::<Material>(&serde_json::to_string(&l).unwrap()).unwrap(),
            l
        );
        assert!(!Material::VACUUM.is_dispersive());
        for bad in [
            Material::Drude {
                eps_inf: 0.5,
                wp: 0.9,
                gamma: 0.0,
            },
            Material::Drude {
                eps_inf: 1.0,
                wp: 0.0,
                gamma: 0.0,
            },
            Material::Drude {
                eps_inf: 1.0,
                wp: 0.9,
                gamma: -1e-9,
            },
            Material::Lorentz {
                eps_inf: 1.0,
                d_eps: 0.0,
                w0: 0.3,
                delta: 0.0,
            },
            Material::Lorentz {
                eps_inf: 1.0,
                d_eps: 1.0,
                w0: f64::NAN,
                delta: 0.0,
            },
            Material::Lorentz {
                eps_inf: 1.0,
                d_eps: 1.0,
                w0: 0.3,
                delta: -0.1,
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
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
