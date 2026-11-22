//! Phase 6 acceptance for TEz: A18, measured in `fdtd_core::cases::duality`.

mod common;

use common::assert_cases;
use fdtd_core::cases::duality;

#[test]
fn a18_te_and_tm_agree_under_duality() {
    assert_cases(duality::a18());
}
