//! The only place physical units meet the solver. Inside the solver,
//! c0 = eps0 = mu0 = 1, dx = 1, and the time unit is dx / c0, so dt = S.
//! Every function here converts between those normalized quantities and SI.

use crate::error::{FdtdError, Result};

/// Speed of light in vacuum, m/s (exact by definition of the metre).
pub const C0: f64 = 299_792_458.0;

/// Maps normalized grid quantities to SI for a chosen cell size and Courant number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Units {
    dx_m: f64,
    courant: f64,
}

fn positive(name: &'static str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(FdtdError::InvalidParameter { name, value })
    }
}

impl Units {
    /// `dx_m` is the physical cell size in metres. `courant` is S. Both must be
    /// positive and finite. The stability limit is checked by the solvers, not here.
    pub fn new(dx_m: f64, courant: f64) -> Result<Self> {
        Ok(Self {
            dx_m: positive("dx_m", dx_m)?,
            courant: positive("courant", courant)?,
        })
    }

    /// Physical cell size, metres.
    pub fn dx_m(&self) -> f64 {
        self.dx_m
    }

    /// The Courant number S, which is also dt in normalized units.
    pub fn courant(&self) -> f64 {
        self.courant
    }

    /// Physical time step, seconds: `S * dx / c0`.
    pub fn dt_s(&self) -> f64 {
        self.courant * self.dx_m / C0
    }

    /// Normalized frequency (cycles per step) to hertz.
    pub fn hz_from_cycles_per_step(&self, f_n: f64) -> f64 {
        f_n / self.dt_s()
    }

    /// Hertz to normalized frequency (cycles per step).
    pub fn cycles_per_step_from_hz(&self, hz: f64) -> f64 {
        hz * self.dt_s()
    }

    /// Hertz to normalized angular frequency (radians per normalized time unit dx / c0).
    pub fn omega_n_from_hz(&self, hz: f64) -> f64 {
        2.0 * std::f64::consts::PI * hz * self.dx_m / C0
    }

    /// Normalized angular frequency (radians per dx / c0) to hertz.
    pub fn hz_from_omega_n(&self, omega_n: f64) -> f64 {
        omega_n * C0 / (2.0 * std::f64::consts::PI * self.dx_m)
    }

    /// Length in cells to metres.
    pub fn m_from_cells(&self, cells: f64) -> f64 {
        cells * self.dx_m
    }

    /// Length in metres to cells.
    pub fn cells_from_m(&self, m: f64) -> f64 {
        m / self.dx_m
    }

    /// Number of steps to seconds.
    pub fn s_from_steps(&self, steps: f64) -> f64 {
        steps * self.dt_s()
    }

    /// Seconds to number of steps.
    pub fn steps_from_s(&self, s: f64) -> f64 {
        s / self.dt_s()
    }

    /// Cells per wavelength at `hz` in a medium of relative permittivity `eps_r`
    /// (mu_r = 1): `c0 / (hz * sqrt(eps_r) * dx)`. This is the resolution number
    /// that controls numerical dispersion.
    pub fn cells_per_wavelength(&self, hz: f64, eps_r: f64) -> f64 {
        C0 / (hz * eps_r.sqrt() * self.dx_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    fn units() -> impl Strategy<Value = Units> {
        (1e-9..1.0_f64, 0.01..1.0_f64).prop_map(|(dx, s)| Units::new(dx, s).unwrap())
    }

    proptest! {
        #[test]
        fn roundtrip_hz_cycles_per_step(u in units(), f in 1e-4..0.5_f64) {
            prop_assert!(rel(u.cycles_per_step_from_hz(u.hz_from_cycles_per_step(f)), f) < 1e-14);
        }

        #[test]
        fn roundtrip_omega(u in units(), w in 1e-4..3.0_f64) {
            prop_assert!(rel(u.omega_n_from_hz(u.hz_from_omega_n(w)), w) < 1e-14);
        }

        #[test]
        fn roundtrip_cells_m(u in units(), c in 1e-3..1e5_f64) {
            prop_assert!(rel(u.cells_from_m(u.m_from_cells(c)), c) < 1e-14);
        }

        #[test]
        fn roundtrip_steps_s(u in units(), n in 1.0..1e7_f64) {
            prop_assert!(rel(u.steps_from_s(u.s_from_steps(n)), n) < 1e-14);
        }

        /// omega_n is 2 pi f_n / S, since one normalized time unit is 1/S steps.
        #[test]
        fn omega_matches_cycles_per_step(u in units(), hz in 1e6..1e15_f64) {
            let w = u.omega_n_from_hz(hz);
            let f_n = u.cycles_per_step_from_hz(hz);
            prop_assert!(rel(w, 2.0 * std::f64::consts::PI * f_n / u.courant()) < 1e-14);
        }
    }

    #[test]
    fn dt_known_value() {
        let u = Units::new(1e-3, 0.5).unwrap();
        assert!(rel(u.dt_s(), 0.5e-3 / 299_792_458.0) < 1e-15);
    }

    #[test]
    fn cells_per_wavelength_vacuum_and_dielectric() {
        let u = Units::new(1e-3, 0.5).unwrap();
        let hz = C0 / 20e-3; // 20 mm wavelength in vacuum
        assert!(rel(u.cells_per_wavelength(hz, 1.0), 20.0) < 1e-14);
        assert!(rel(u.cells_per_wavelength(hz, 4.0), 10.0) < 1e-14);
    }

    #[test]
    fn rejects_invalid() {
        for (dx, s) in [
            (0.0, 0.5),
            (-1e-3, 0.5),
            (f64::NAN, 0.5),
            (f64::INFINITY, 0.5),
            (1e-3, 0.0),
            (1e-3, -0.5),
            (1e-3, f64::NAN),
        ] {
            assert!(Units::new(dx, s).is_err(), "accepted dx={dx}, S={s}");
        }
    }
}
