//! Convolutional PML (Roden and Gedney 2000) with the complex frequency
//! shifted stretch `s = kappa + sigma / (alpha + j w)`, normalized
//! (eps0 = 1, dx = 1, dt = S). See `docs/02-numerics.md`.
//!
//! The layer has the same thickness `d` on all four sides. The PEC wall stays
//! at node 0 and node n - 1, and the layer runs from the wall (depth d) to the
//! interface node d (depth 0).
//!
//! Each spatial derivative taken normal to a face is replaced by
//!
//!   psi = b psi + c (d field / d normal)
//!   derivative used = (d field / d normal) / kappa + psi
//!
//! with `b = exp(-(sigma / kappa + alpha) S)` and
//! `c = sigma (b - 1) / (sigma kappa + kappa^2 alpha)`. Where sigma is 0 the
//! recursion is switched off (`b = c = 0`), and at depth 0 kappa is exactly 1,
//! so interior nodes see the plain Yee update with no branch.
//!
//! Every profile is evaluated at the true coordinate of the component that
//! uses it: Ez at node x = i, Hy at x = i + 1/2 (and the same along y for Ez
//! and Hx). Using the integer positions for H costs about 20 dB.

use crate::error::{FdtdError, Result};
use serde::{Deserialize, Serialize};

/// `b`, `c` and `1/kappa` at one set of positions along one axis, one entry
/// per array index. Interior entries are exactly `(0, 0, 1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub inv_kappa: Vec<f64>,
}

/// The six 1D coefficient profiles per axis that both kernels consume.
/// `x_int[i]` is at node x = i (Ez), `x_half[i]` at x = i + 1/2 (Hy), and
/// likewise `y_int[j]` at y = j (Ez), `y_half[j]` at y = j + 1/2 (Hx). The last
/// half entry is padding, like the padded H arrays, and is the identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Cpml {
    pub x_int: Profile,
    pub x_half: Profile,
    pub y_int: Profile,
    pub y_half: Profile,
}

/// `sigma_opt = 0.8 (m + 1) / (eta dx)` normalized, with `eta = 1 / sqrt(eps_r)`
/// for the medium next to the layer (Taflove eq. 7.67 with the stretch in
/// eps0, see the known deviation in `PROGRESS.md`).
pub fn sigma_opt(m: f64, eps_r: f64) -> f64 {
    0.8 * (m + 1.0) / eps_r.sqrt()
}

/// `(sigma, kappa, alpha)` at fractional depth `u = rho / d` in [0, 1]:
/// `sigma = sigma_max u^m`, `kappa = 1 + (kappa_max - 1) u^m`,
/// `alpha = alpha_max (1 - u)`.
pub fn graded(spec: &PmlSpec, sigma_max: f64, u: f64) -> (f64, f64, f64) {
    let p = u.powf(spec.m);
    (
        sigma_max * p,
        1.0 + (spec.kappa_max - 1.0) * p,
        spec.alpha_max * (1.0 - u),
    )
}

/// `(b, c, 1/kappa)` for one point, Courant number `s` (normalized dt).
/// With sigma = 0 the recursion is off: `b = c = 0`.
pub fn coefficients(sigma: f64, kappa: f64, alpha: f64, s: f64) -> (f64, f64, f64) {
    if sigma == 0.0 {
        return (0.0, 0.0, 1.0 / kappa);
    }
    let b = (-(sigma / kappa + alpha) * s).exp();
    let c = sigma * (b - 1.0) / (sigma * kappa + kappa * kappa * alpha);
    (b, c, 1.0 / kappa)
}

impl Profile {
    /// Profile over `len` entries along an axis of `n` nodes, entry `k` at
    /// doubled position `2 k + half` (half is 0 or 1).
    fn new(spec: &PmlSpec, sigma_max: f64, s: f64, n: usize, len: usize, half: i64) -> Self {
        let d2 = 2 * spec.thickness as i64;
        let wall2 = 2 * (n as i64 - 1);
        let mut p = Profile {
            b: vec![0.0; len],
            c: vec![0.0; len],
            inv_kappa: vec![1.0; len],
        };
        if d2 == 0 {
            return p;
        }
        for k in 0..len {
            let x2 = 2 * k as i64 + half;
            if x2 > wall2 {
                continue; // padding
            }
            // Doubled depth into the layer, from the nearer face.
            let depth2 = (d2 - x2).max(x2 - (wall2 - d2)).max(0);
            if depth2 == 0 {
                continue;
            }
            let (sigma, kappa, alpha) = graded(spec, sigma_max, depth2 as f64 / d2 as f64);
            (p.b[k], p.c[k], p.inv_kappa[k]) = coefficients(sigma, kappa, alpha, s);
        }
        p
    }
}

impl Cpml {
    /// Profiles for an `nx` by `ny` node grid at Courant number `s`, next to a
    /// medium of relative permittivity `eps_r`. Thickness 0 gives the identity
    /// everywhere. Call `PmlSpec::validate` and check the fit first.
    pub fn new(spec: &PmlSpec, nx: usize, ny: usize, s: f64, eps_r: f64) -> Self {
        let sigma_max = spec.sigma_scale * sigma_opt(spec.m, eps_r);
        Cpml {
            x_int: Profile::new(spec, sigma_max, s, nx, nx, 0),
            x_half: Profile::new(spec, sigma_max, s, nx, nx, 1),
            y_int: Profile::new(spec, sigma_max, s, ny, ny, 0),
            y_half: Profile::new(spec, sigma_max, s, ny, ny, 1),
        }
    }

    /// The same layer on the x faces only, for periodic y: the y profiles are
    /// replaced by the identity, so every row runs the plain y update.
    pub fn x_only(self) -> Self {
        let ny = self.y_int.b.len();
        let id = Cpml::none(1, ny);
        Cpml {
            y_int: id.y_int,
            y_half: id.y_half,
            ..self
        }
    }

    /// No layer: every entry is the identity.
    pub fn none(nx: usize, ny: usize) -> Self {
        let spec = PmlSpec {
            thickness: 0,
            m: 1.0,
            sigma_scale: 0.0,
            kappa_max: 1.0,
            alpha_max: 0.0,
        };
        Cpml::new(&spec, nx, ny, 0.5, 1.0)
    }
}

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
    /// Layers tuned with `fdtd pml-sweep` on the standard reflection sweep
    /// (`ReflectionSetup::standard`: Ricker pulse peaking at 20 cells per
    /// wavelength, S = 0.5, probes from normal to 80 degrees plus a corner),
    /// ranked by the worst reflection over every probe. The optimum is flat to
    /// about 0.5 dB, so these are round values from its middle. Below 16 cells
    /// this is the 10-cell tuning, from 16 cells up the 16-cell one. Measured
    /// numbers are in `PROGRESS.md`.
    pub fn recommended(thickness: usize) -> PmlSpec {
        if thickness < 16 {
            PmlSpec {
                thickness,
                m: 3.0,
                sigma_scale: 1.25,
                kappa_max: 2.0,
                alpha_max: 0.15,
            }
        } else {
            PmlSpec {
                thickness,
                m: 4.0,
                sigma_scale: 1.0,
                kappa_max: 2.0,
                alpha_max: 0.1,
            }
        }
    }

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

    const S: f64 = 0.5;

    fn all(p: &Profile) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        (0..p.b.len()).map(|k| (p.b[k], p.c[k], p.inv_kappa[k]))
    }

    /// Interior entries are exactly the identity, so the interior update is
    /// the plain Yee update bit for bit.
    #[test]
    fn interior_is_exact_identity() {
        let (nx, d) = (40, 10);
        let p = Cpml::new(&spec(d), nx, 30, S, 1.0);
        for i in d..=nx - 1 - d {
            let t = (p.x_int.b[i], p.x_int.c[i], p.x_int.inv_kappa[i]);
            assert_eq!(t, (0.0, 0.0, 1.0));
        }
        // Half positions i + 1/2 strictly between the interfaces.
        for i in d..nx - 1 - d {
            let t = (p.x_half.b[i], p.x_half.c[i], p.x_half.inv_kappa[i]);
            assert_eq!(t, (0.0, 0.0, 1.0));
        }
        // Padding entry and a layer-free grid.
        assert_eq!(p.x_half.inv_kappa[nx - 1], 1.0);
        let none = Cpml::none(40, 30);
        for prof in [&none.x_int, &none.x_half, &none.y_int, &none.y_half] {
            assert!(all(prof).all(|t| t == (0.0, 0.0, 1.0)));
        }
    }

    /// `x_only` keeps the x profiles bit for bit and makes both y profiles
    /// the identity at full length, so periodic rows see no layer.
    #[test]
    fn x_only_drops_the_y_layer() {
        let full = Cpml::new(&spec(8), 37, 29, S, 1.0);
        let x = full.clone().x_only();
        assert_eq!((&x.x_int, &x.x_half), (&full.x_int, &full.x_half));
        assert!(all(&full.y_int).any(|t| t != (0.0, 0.0, 1.0)));
        for prof in [&x.y_int, &x.y_half] {
            assert_eq!(prof.b.len(), 29);
            assert!(all(prof).all(|t| t == (0.0, 0.0, 1.0)));
        }
    }

    /// Low and high sides mirror exactly: node i and node n - 1 - i, half
    /// position i + 1/2 and (n - 2 - i) + 1/2.
    #[test]
    fn profiles_mirror() {
        let (nx, ny) = (37, 29);
        let p = Cpml::new(&spec(8), nx, ny, S, 1.0);
        for (prof, n) in [(&p.x_int, nx), (&p.y_int, ny)] {
            for i in 0..n {
                assert_eq!(prof.b[i], prof.b[n - 1 - i]);
                assert_eq!(prof.c[i], prof.c[n - 1 - i]);
                assert_eq!(prof.inv_kappa[i], prof.inv_kappa[n - 1 - i]);
            }
        }
        for (prof, n) in [(&p.x_half, nx), (&p.y_half, ny)] {
            for i in 0..n - 1 {
                assert_eq!(prof.b[i], prof.b[n - 2 - i]);
                assert_eq!(prof.c[i], prof.c[n - 2 - i]);
                assert_eq!(prof.inv_kappa[i], prof.inv_kappa[n - 2 - i]);
            }
        }
    }

    /// Going from the interface to the wall, the half-cell entries sit
    /// strictly between their integer neighbours: deeper means smaller
    /// 1/kappa. This fails if the H profile is evaluated at the integer
    /// positions. (c is not monotone: near the wall kappa grows and c shrinks
    /// like (b - 1) / kappa, so only its sign is checked.)
    #[test]
    fn half_cell_profiles_interleave() {
        let d = 10;
        let p = Cpml::new(&spec(d), 40, 40, S, 1.0);
        let (ki, kh) = (&p.x_int.inv_kappa, &p.x_half.inv_kappa);
        for i in 0..d {
            // Low side: node i is deeper than i + 1/2, which is deeper than i + 1.
            assert!(ki[i] < kh[i] && kh[i] < ki[i + 1], "1/kappa at {i}");
            // Every position with depth > 0 runs the recursion.
            assert!(p.x_int.c[i] < 0.0 && p.x_half.c[i] < 0.0, "c at {i}");
        }
        // The graded values rise strictly with depth (alpha falls).
        let sm = sigma_opt(3.0, 1.0);
        let mut last = graded(&spec(d), sm, 0.0);
        for k in 1..=20 {
            let g = graded(&spec(d), sm, k as f64 / 20.0);
            assert!(g.0 > last.0 && g.1 > last.1 && g.2 < last.2);
            last = g;
        }
        assert_eq!(last, (sm, 5.0, 0.0));
    }

    #[test]
    fn coefficient_ranges() {
        let p = Cpml::new(&spec(12), 50, 40, S, 2.25);
        for prof in [&p.x_int, &p.x_half, &p.y_int, &p.y_half] {
            for (b, c, ik) in all(prof) {
                assert!((0.0..1.0).contains(&b), "b {b}");
                assert!(c <= 0.0, "c {c}");
                assert!(ik > 0.0 && ik <= 1.0, "1/kappa {ik}");
            }
        }
    }

    /// At DC the recursion settles to psi = c d / (1 - b), so the derivative
    /// used is `d (1/kappa + c / (1 - b))`. That must equal `d / s(0)` with
    /// the continuous stretch `s(0) = kappa + sigma / alpha`, an exact
    /// identity of the recursive convolution.
    #[test]
    fn dc_limit_matches_stretch() {
        for (sigma, kappa, alpha) in [(1.2, 5.0, 0.05), (0.3, 1.0, 0.2), (3.2, 11.0, 0.01)] {
            let (b, c, ik) = coefficients(sigma, kappa, alpha, S);
            let got = ik + c / (1.0 - b);
            let want = 1.0 / (kappa + sigma / alpha);
            assert!((got - want).abs() < 1e-13 * want, "{got} vs {want}");
        }
    }

    #[test]
    fn sigma_zero_and_alpha_zero() {
        assert_eq!(coefficients(0.0, 1.0, 0.1, S), (0.0, 0.0, 1.0));
        assert_eq!(coefficients(0.0, 4.0, 0.1, S), (0.0, 0.0, 0.25));
        let (b, c, ik) = coefficients(2.0, 4.0, 0.0, S);
        assert_eq!(b, (-0.5 * S).exp());
        assert!((c - (b - 1.0) / 4.0).abs() < 1e-16 && ik == 0.25);
    }

    #[test]
    fn sigma_opt_value() {
        assert_eq!(sigma_opt(3.0, 1.0), 3.2);
        assert_eq!(sigma_opt(3.0, 4.0), 1.6);
    }

    #[test]
    fn recommended_layers_are_valid() {
        for t in [8, 10, 15, 16, 20] {
            let p = PmlSpec::recommended(t);
            assert!(p.validate().is_ok());
            assert_eq!(p.thickness, t);
        }
        assert_eq!(PmlSpec::recommended(10).m, 3.0);
        assert_eq!(PmlSpec::recommended(16).m, 4.0);
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
