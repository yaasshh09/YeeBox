//! Phase 3 acceptance for spectra: A11 to A14, measured in
//! `fdtd_core::cases::plane_wave` (each case documents its setup, its gate
//! against the continuum, and its guard against the exact discrete answer).

mod common;

use common::assert_cases;
use fdtd_core::cases::plane_wave;

#[test]
fn a13_dielectric_slab_and_r_plus_t() {
    assert_cases(plane_wave::a13());
}

#[test]
fn a14_lossy_attenuation() {
    assert_cases(plane_wave::a14());
}

#[test]
fn a11_parallel_plate_cutoff_and_guided_wavelength() {
    assert_cases(plane_wave::a11());
}

#[test]
fn a12_parallel_plate_evanescent_decay() {
    assert_cases(plane_wave::a12());
}
