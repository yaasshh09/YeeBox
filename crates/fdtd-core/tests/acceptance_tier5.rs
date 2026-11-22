//! Tier 5, the Meep cross-checks (`cases::tier5`). Meep is a second
//! implementation, not the reference, so these are recorded rather than
//! gated; the bounds here are regression guards at about five times what
//! was measured when the Meep data was committed (bend 6.2e-3, Drude slab
//! 1.3e-3; the cylinder, 0.06 dB, runs in `fdtd report` only, as it repeats
//! the A20 run).

use fdtd_core::cases::tier5::{bend_vs_meep, drude_vs_meep};

#[test]
fn bend_matches_meep() {
    let d = bend_vs_meep().unwrap();
    println!("bend |T_yeebox - T_meep| {d:.2e}");
    assert!(d < 0.03, "{d}");
}

#[test]
fn drude_slab_matches_meep() {
    let (d, meep) = drude_vs_meep().unwrap();
    println!("Drude slab YeeBox minus Meep {d:.2e}, Meep minus analytic {meep:.2e}");
    assert!(d < 0.007, "{d}");
}
