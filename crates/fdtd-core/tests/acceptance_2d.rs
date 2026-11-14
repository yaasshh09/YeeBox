//! Phase 1 acceptance for the 2D TMz reference solver: A4 to A7, measured
//! in `fdtd_core::cases::tier2d` (each case documents its setup).

mod common;

use common::assert_cases;
use fdtd_core::cases::tier2d;

#[test]
fn a4_stability_bounded_at_070_unbounded_at_072() {
    assert_cases(tier2d::a4());
}

#[test]
fn a5_dispersion_axis_and_diagonal() {
    assert_cases(tier2d::a5());
}

#[test]
fn a6_cavity_first_six_resonances() {
    assert_cases(tier2d::a6());
}

#[test]
fn a7_cavity_convergence_order() {
    assert_cases(tier2d::a7());
}
