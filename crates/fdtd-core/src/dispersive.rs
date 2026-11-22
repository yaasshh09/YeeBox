//! Drude and Lorentz media by auxiliary differential equation (ADE),
//! `docs/02-numerics.md`. Normalized throughout: eps0 = mu0 = 1, dx = 1,
//! dt = S, frequencies in radians per unit time (one unit is dx / c0).
//!
//! **Sign convention: exp(+j w t).** A field is Re(F exp(+j w t)), so d/dt is
//! +j w and a lossy medium has Im(eps) < 0. `validation/gen/slab_dispersive.py`
//! uses the same convention. Mixing conventions flips the loss into gain.
//!
//!   Drude:   eps(w) = eps_inf - wp^2 / (w^2 - j gamma w)
//!   Lorentz: eps(w) = eps_inf + d_eps w0^2 / (w0^2 - w^2 + 2 j delta w)
//!
//! # The update, derived
//!
//! Ampere with a conductivity, a Drude current J and a Lorentz polarization P:
//!
//!   eps_inf dE/dt + sigma E + J + dP/dt = curl H
//!   dJ/dt + gamma J = wp^2 E
//!   d2P/dt2 + 2 delta dP/dt + w0^2 P = d_eps w0^2 E
//!
//! E, J, P live at integer steps n, H at n + 1/2, all on the same E location.
//!
//! Drude, centered at n + 1/2 (trapezoid on the right):
//!
//!   (J+ - J)/S + gamma (J+ + J)/2 = wp^2 (E+ + E)/2
//!   J+ = k J + beta (E+ + E),   k = (1 - gamma S/2) / (1 + gamma S/2),
//!                               beta = (wp^2 S/2) / (1 + gamma S/2)
//!
//! Lorentz, centered at n (explicit in E):
//!
//!   (P+ - 2P + P-)/S^2 + 2 delta (P+ - P-)/(2S) + w0^2 P = d_eps w0^2 E
//!   P+ = a1 P + a2 P- + a3 E,   a1 = (2 - w0^2 S^2) / (1 + delta S),
//!                               a2 = -(1 - delta S) / (1 + delta S),
//!                               a3 = d_eps w0^2 S^2 / (1 + delta S)
//!
//! Ampere centered at n + 1/2, with (J+ + J)/2 from the Drude line and
//! dP/dt as (P+ - P)/S, P+ known before E+:
//!
//!   eps_inf (E+ - E)/S + sigma (E+ + E)/2 + (k J + beta (E+ + E) + J)/2
//!       + (P+ - P)/S = curl H
//!
//! Collect E+ with L = (sigma + beta) S / 2 and d = eps_inf + L:
//!
//!   E+ = Ca E + Cb (curl H - cj J) - cp (P+ - P)
//!   Ca = (eps_inf - L) / d,   Cb = S / d,   cj = (1 + k) / 2,   cp = 1 / d
//!
//! With no Lorentz pole this is doc 02's Drude update (multiply through by
//! 2/S); with no Drude pole and sigma = 0 it is doc 02's Lorentz update; with
//! neither it is `Material::coefficients`, and then `Row` uses exactly that
//! function so every non-dispersive scene keeps its bits.
//!
//! A TEz edge between two cells takes the mean of their eps(w) (`Medium::edge`):
//! eps_inf and sigma are averaged, and each cell's pole enters with its
//! strength halved (wp^2 / 2, d_eps / 2), or whole when both cells share it.

use crate::error::{FdtdError, Result};
use crate::material::Material;

/// A Drude pole with its strength folded in: `wp2` is the (weighted) wp^2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrudePole {
    pub wp2: f64,
    pub gamma: f64,
}

/// A Lorentz pole with its strength folded into `d_eps`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorentzPole {
    pub d_eps: f64,
    pub w0: f64,
    pub delta: f64,
}

/// What one E location holds: everything but PEC (which is `None` where a
/// `Medium` is expected).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Medium {
    pub eps_inf: f64,
    pub sigma: f64,
    pub drude: Option<DrudePole>,
    pub lorentz: Option<LorentzPole>,
}

/// The precomputed update of one medium (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Row {
    pub ca: f64,
    pub cb: f64,
    /// Relative permittivity for the energy (eps_inf).
    pub eps: f64,
    pub drude: Option<DrudeRow>,
    pub lorentz: Option<LorentzRow>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrudeRow {
    pub k: f64,
    pub beta: f64,
    /// (1 + k) / 2, the weight of J^n in the E update.
    pub cj: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorentzRow {
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    /// 1 / d, the weight of P^(n+1) - P^n in the E update.
    pub cp: f64,
}

impl Row {
    /// The PEC row: E held at zero.
    pub const PEC: Row = Row {
        ca: 0.0,
        cb: 0.0,
        eps: 0.0,
        drude: None,
        lorentz: None,
    };

    pub fn is_dispersive(&self) -> bool {
        self.drude.is_some() || self.lorentz.is_some()
    }
}

impl Medium {
    /// The medium of one cell, `None` for PEC.
    pub fn of(m: &Material) -> Option<Medium> {
        match *m {
            Material::Dielectric { eps_inf, sigma } => Some(Medium {
                eps_inf,
                sigma,
                drude: None,
                lorentz: None,
            }),
            Material::Drude { eps_inf, wp, gamma } => Some(Medium {
                eps_inf,
                sigma: 0.0,
                drude: Some(DrudePole {
                    wp2: wp * wp,
                    gamma,
                }),
                lorentz: None,
            }),
            Material::Lorentz {
                eps_inf,
                d_eps,
                w0,
                delta,
            } => Some(Medium {
                eps_inf,
                sigma: 0.0,
                drude: None,
                lorentz: Some(LorentzPole { d_eps, w0, delta }),
            }),
            Material::Pec => None,
        }
    }

    /// The medium of a TEz edge between cells `a` and `b`: the mean of their
    /// eps(w), `None` (PEC) if either is PEC. Two different Drude poles (or
    /// two different Lorentz poles) on one edge are refused: one slot each.
    pub fn edge(a: &Material, b: &Material) -> Result<Option<Medium>> {
        let (Some(a), Some(b)) = (Medium::of(a), Medium::of(b)) else {
            return Ok(None);
        };
        let drude = match (a.drude, b.drude) {
            (Some(p), Some(q)) if p == q => Some(p),
            (Some(_), Some(_)) => return Err(two_poles("drude")),
            (p, q) => p.or(q).map(|p| DrudePole {
                wp2: 0.5 * p.wp2,
                ..p
            }),
        };
        let lorentz = match (a.lorentz, b.lorentz) {
            (Some(p), Some(q)) if p == q => Some(p),
            (Some(_), Some(_)) => return Err(two_poles("lorentz")),
            (p, q) => p.or(q).map(|p| LorentzPole {
                d_eps: 0.5 * p.d_eps,
                ..p
            }),
        };
        Ok(Some(Medium {
            eps_inf: 0.5 * (a.eps_inf + b.eps_inf),
            sigma: 0.5 * (a.sigma + b.sigma),
            drude,
            lorentz,
        }))
    }

    /// The update row at Courant number `s` (module docs). Without poles it
    /// is `Material::coefficients` itself, bit for bit.
    pub fn row(&self, s: f64) -> Row {
        if self.drude.is_none() && self.lorentz.is_none() {
            let (ca, cb) = Material::Dielectric {
                eps_inf: self.eps_inf,
                sigma: self.sigma,
            }
            .coefficients(s);
            return Row {
                ca,
                cb,
                eps: self.eps_inf,
                drude: None,
                lorentz: None,
            };
        }
        let drude = self.drude.map(|p| {
            let g = 1.0 + p.gamma * s / 2.0;
            let k = (1.0 - p.gamma * s / 2.0) / g;
            DrudeRow {
                k,
                beta: (p.wp2 * s / 2.0) / g,
                cj: (1.0 + k) / 2.0,
            }
        });
        let beta = drude.map_or(0.0, |d| d.beta);
        let l = (self.sigma + beta) * s / 2.0;
        let d = self.eps_inf + l;
        let lorentz = self.lorentz.map(|p| {
            let g = 1.0 + p.delta * s;
            LorentzRow {
                a1: (2.0 - p.w0 * p.w0 * s * s) / g,
                a2: -(1.0 - p.delta * s) / g,
                a3: p.d_eps * p.w0 * p.w0 * s * s / g,
                cp: 1.0 / d,
            }
        });
        Row {
            ca: (self.eps_inf - l) / d,
            cb: s / d,
            eps: self.eps_inf,
            drude,
            lorentz,
        }
    }

    /// The continuum relative permittivity at angular frequency `w`,
    /// `(re, im)`, exp(+j w t) convention.
    pub fn eps_at(&self, w: f64) -> (f64, f64) {
        let (mut re, mut im) = (self.eps_inf, -self.sigma / w);
        if let Some(p) = self.drude {
            // -wp2 / (w^2 - j gamma w) = -wp2 (w^2 + j gamma w) / (w^4 + gamma^2 w^2)
            let den = w.powi(4) + (p.gamma * w).powi(2);
            re -= p.wp2 * w * w / den;
            im -= p.wp2 * p.gamma * w / den;
        }
        if let Some(p) = self.lorentz {
            // d_eps w0^2 / (a + j b), a = w0^2 - w^2, b = 2 delta w
            let (a, b) = (p.w0 * p.w0 - w * w, 2.0 * p.delta * w);
            let num = p.d_eps * p.w0 * p.w0;
            re += num * a / (a * a + b * b);
            im -= num * b / (a * a + b * b);
        }
        (re, im)
    }
}

/// No dispersive pole at this location.
pub const NO_POLE: u32 = u32::MAX;

/// The id of `row` in `rows`, appended if new; `NO_POLE` if it is plain.
pub fn pole_id(rows: &mut Vec<Row>, row: &Row) -> u32 {
    if !row.is_dispersive() {
        return NO_POLE;
    }
    match rows.iter().position(|r| r == row) {
        Some(i) => i as u32,
        None => {
            rows.push(*row);
            (rows.len() - 1) as u32
        }
    }
}

/// The auxiliary fields of one E array: Drude J, Lorentz P and P^(n-1),
/// `nx * ny` each, or all empty when the scene has no dispersive material.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Aux {
    pub j: Vec<f64>,
    pub p: Vec<f64>,
    pub p_old: Vec<f64>,
}

impl Aux {
    /// Allocated for `n` locations if `needed`, else empty.
    pub fn new(n: usize, needed: bool) -> Aux {
        let n = if needed { n } else { 0 };
        Aux {
            j: vec![0.0; n],
            p: vec![0.0; n],
            p_old: vec![0.0; n],
        }
    }

    /// A location whose pole changed starts from rest.
    pub fn clear(&mut self, k: usize) {
        if !self.j.is_empty() {
            (self.j[k], self.p[k], self.p_old[k]) = (0.0, 0.0, 0.0);
        }
    }

    /// The dispersive E update at `k` from E^n = `e` and the curl term:
    /// P^(n+1) first, then E^(n+1), then J^(n+1) (module docs).
    pub fn update(&mut self, r: &Row, k: usize, e: f64, curl: f64) -> f64 {
        let j = r.drude.map_or(0.0, |d| d.cj * self.j[k]);
        let dp = match r.lorentz {
            Some(l) => {
                let p_new = l.a1 * self.p[k] + l.a2 * self.p_old[k] + l.a3 * e;
                let dp = l.cp * (p_new - self.p[k]);
                (self.p_old[k], self.p[k]) = (self.p[k], p_new);
                dp
            }
            None => 0.0,
        };
        let e_new = r.ca * e + r.cb * (curl - j) - dp;
        if let Some(d) = r.drude {
            self.j[k] = d.k * self.j[k] + d.beta * (e_new + e);
        }
        e_new
    }
}

fn two_poles(kind: &'static str) -> FdtdError {
    FdtdError::InvalidParameter {
        name: match kind {
            "drude" => "two different drude materials on one edge",
            _ => "two different lorentz materials on one edge",
        },
        value: f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SILVERISH: Material = Material::Drude {
        eps_inf: 4.0,
        wp: 0.9,
        gamma: 0.01,
    };
    const RESONATOR: Material = Material::Lorentz {
        eps_inf: 2.0,
        d_eps: 3.0,
        w0: 0.3,
        delta: 0.02,
    };

    /// No poles: the row is `Material::coefficients` exactly, so every
    /// Phase 5 scene keeps its bits.
    #[test]
    fn plain_row_is_the_dielectric_row() {
        for (eps_inf, sigma) in [(1.0, 0.0), (2.25, 0.0), (4.0, 0.1)] {
            let m = Material::Dielectric { eps_inf, sigma };
            let row = Medium::of(&m).unwrap().row(0.5);
            assert_eq!((row.ca, row.cb), m.coefficients(0.5));
            assert!(!row.is_dispersive());
        }
        assert_eq!(Medium::of(&Material::Pec), None);
    }

    /// The Drude row against doc 02 written out by hand: the E update there
    /// is E+ = (2 eps - beta S)/(2 eps + beta S) E + 2 S/(2 eps + beta S)
    /// (curl H - (1 + k)/2 J).
    #[test]
    fn drude_row_is_doc_02() {
        let (eps, wp, gamma, s) = (4.0, 0.9, 0.01, 0.5);
        let row = Medium::of(&SILVERISH).unwrap().row(s);
        let k = (1.0 - gamma * s / 2.0) / (1.0 + gamma * s / 2.0);
        let beta = (wp * wp * s / 2.0) / (1.0 + gamma * s / 2.0);
        let d = row.drude.unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs().max(1.0);
        assert!(close(d.k, k) && close(d.beta, beta) && close(d.cj, (1.0 + k) / 2.0));
        assert!(close(
            row.ca,
            (2.0 * eps - beta * s) / (2.0 * eps + beta * s)
        ));
        assert!(close(row.cb, 2.0 * s / (2.0 * eps + beta * s)));
        assert!(row.lorentz.is_none());
    }

    /// The Lorentz row against doc 02: Ca = 1, Cb = S/eps, and the
    /// polarization recurrence coefficients a1, a2, a3.
    #[test]
    fn lorentz_row_is_doc_02() {
        let (eps, d_eps, w0, delta, s) = (2.0, 3.0, 0.3, 0.02, 0.5);
        let row = Medium::of(&RESONATOR).unwrap().row(s);
        let l = row.lorentz.unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15 * b.abs().max(1.0);
        assert!(close(l.a1, (2.0 - w0 * w0 * s * s) / (1.0 + delta * s)));
        assert!(close(l.a2, -(1.0 - delta * s) / (1.0 + delta * s)));
        assert!(close(l.a3, d_eps * w0 * w0 * s * s / (1.0 + delta * s)));
        assert!(close(l.cp, 1.0 / eps));
        assert_eq!((row.ca, row.cb), (1.0, s / eps));
        assert!(row.drude.is_none());
    }

    /// Lossless limits: gamma = 0 gives k = 1, delta = 0 gives a2 = -1.
    #[test]
    fn lossless_limits() {
        let drude = Medium::of(&Material::Drude {
            eps_inf: 1.0,
            wp: 0.5,
            gamma: 0.0,
        })
        .unwrap()
        .row(0.5);
        assert_eq!(drude.drude.unwrap().k, 1.0);
        let lorentz = Medium::of(&Material::Lorentz {
            eps_inf: 1.0,
            d_eps: 1.0,
            w0: 0.2,
            delta: 0.0,
        })
        .unwrap()
        .row(0.5);
        assert_eq!(lorentz.lorentz.unwrap().a2, -1.0);
    }

    /// Edges: half strength against a plain cell, whole when both cells
    /// share the pole, PEC wins, and two different poles are refused.
    #[test]
    fn edge_mean_of_poles() {
        let vac = Material::VACUUM;
        let e = Medium::edge(&SILVERISH, &vac).unwrap().unwrap();
        assert_eq!(e.eps_inf, 2.5);
        assert_eq!(e.drude.unwrap().wp2, 0.5 * (0.9 * 0.9));
        let same = Medium::edge(&SILVERISH, &SILVERISH).unwrap().unwrap();
        assert_eq!(same, Medium::of(&SILVERISH).unwrap());
        let mixed = Medium::edge(&SILVERISH, &RESONATOR).unwrap().unwrap();
        assert_eq!(mixed.drude.unwrap().wp2, 0.5 * (0.9 * 0.9));
        assert_eq!(mixed.lorentz.unwrap().d_eps, 1.5);
        assert_eq!(Medium::edge(&SILVERISH, &Material::Pec).unwrap(), None);
        let other = Material::Drude {
            eps_inf: 4.0,
            wp: 0.8,
            gamma: 0.01,
        };
        assert!(Medium::edge(&SILVERISH, &other).is_err());
        // Two plain cells: the plain mean, as before dispersion existed.
        let glass = Material::Dielectric {
            eps_inf: 3.0,
            sigma: 0.25,
        };
        let plain = Medium::edge(&vac, &glass).unwrap().unwrap();
        assert_eq!((plain.eps_inf, plain.sigma), (2.0, 0.125));
    }

    /// A uniform field (curl H = 0) in a lossless Drude medium: the update is
    /// Crank-Nicolson on the plasma oscillator of frequency wp / sqrt(eps_inf),
    /// so from J = 0 it gives E_n = E_0 cos(n theta) exactly, with
    /// tan(theta / 2) = (wp / sqrt(eps_inf)) S / 2.
    #[test]
    fn drude_plasma_oscillation_is_exact() {
        let (eps, wp, s) = (2.0_f64, 0.6_f64, 0.5);
        let m = Material::Drude {
            eps_inf: eps,
            wp,
            gamma: 0.0,
        };
        let row = Medium::of(&m).unwrap().row(s);
        let mut aux = Aux::new(1, true);
        let theta = 2.0 * (wp / eps.sqrt() * s / 2.0).atan();
        let mut e = 1.0;
        for n in 1..=2000 {
            e = aux.update(&row, 0, e, 0.0);
            let want = (n as f64 * theta).cos();
            assert!((e - want).abs() < 1e-11, "step {n}: {e} vs {want}");
        }
    }

    /// A uniform field in a Lorentz medium: E + P / eps_inf never changes
    /// (charge conservation of the discrete Ampere law with curl H = 0), and
    /// without damping P oscillates at cos(theta) = 1 - w0^2 S^2 (1 +
    /// d_eps / eps_inf) / 2.
    #[test]
    fn lorentz_uniform_field_identities() {
        let (eps, d_eps, w0, s) = (2.0_f64, 3.0, 0.3_f64, 0.5);
        let m = Material::Lorentz {
            eps_inf: eps,
            d_eps,
            w0,
            delta: 0.0,
        };
        let row = Medium::of(&m).unwrap().row(s);
        let mut aux = Aux::new(1, true);
        let mut e = 1.0;
        let mut ps = Vec::new();
        for _ in 0..4000 {
            e = aux.update(&row, 0, e, 0.0);
            assert!((e + aux.p[0] / eps - 1.0).abs() < 1e-12);
            ps.push(aux.p[0]);
        }
        // P_n = A + B cos(n theta + phi): its second difference about the
        // mean is (2 cos theta - 2) times the deviation, exactly.
        let mean = d_eps / (eps + d_eps) * eps; // P where E + P/eps = 1 balances
        let cos_t = 1.0 - w0 * w0 * s * s * (1.0 + d_eps / eps) / 2.0;
        for n in 1..ps.len() - 1 {
            let lhs = ps[n + 1] + ps[n - 1] - 2.0 * mean;
            let rhs = 2.0 * cos_t * (ps[n] - mean);
            assert!((lhs - rhs).abs() < 1e-10, "step {n}: {lhs} vs {rhs}");
        }
    }

    /// The continuum eps(w) in the exp(+jwt) convention: a Drude metal is
    /// negative below its plasma frequency, loss is Im(eps) < 0, and a
    /// Lorentz pole goes from eps_inf + d_eps at DC to eps_inf far above.
    #[test]
    fn continuum_permittivity() {
        let d = Medium::of(&SILVERISH).unwrap();
        let (re, im) = d.eps_at(0.3);
        assert!(re < 0.0 && im < 0.0, "{re} {im}");
        let w: f64 = 0.3;
        let want = 4.0 - 0.81 / (w * w + 0.01 * 0.01);
        assert!((re - want).abs() < 1e-12);
        let l = Medium::of(&RESONATOR).unwrap();
        assert!((l.eps_at(1e-6).0 - 5.0).abs() < 1e-6);
        assert!((l.eps_at(1e3).0 - 2.0).abs() < 1e-6);
        assert!(l.eps_at(0.3).1 < 0.0);
    }
}
