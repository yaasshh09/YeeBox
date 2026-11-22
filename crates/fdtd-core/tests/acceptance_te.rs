//! Phase 6 acceptance: A16, A17 and A21 (`fdtd_core::cases::dispersive`)
//! and A18 (`fdtd_core::cases::duality`).

mod common;

use common::assert_cases;
use fdtd_core::cases::{dispersive, duality};

#[test]
fn a16_drude_slab_against_complex_eps() {
    assert_cases(dispersive::a16());
}

#[test]
fn a17_lorentz_slab_against_complex_eps() {
    assert_cases(dispersive::a17());
}

#[test]
fn a21_drude_cylinder_resonance() {
    assert_cases(dispersive::a21());
}

#[test]
fn a18_te_and_tm_agree_under_duality() {
    assert_cases(duality::a18());
}
