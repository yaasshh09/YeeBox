//! A19 and A20: PEC and eps 4 cylinders against the 2D Mie series
//! (`validation/gen/cylinder_series.py`), gated in TMz, with the
//! convergence over radius 10, 20, 40 cells. See `cases::scattering`.

use fdtd_core::cases::scattering::{a19, a20};

#[test]
fn a19_pec_cylinder() {
    let rows = a19().unwrap();
    println!("{rows:#?}");
    assert!(rows.iter().all(|r| r.pass), "{rows:#?}");
}

#[test]
fn a20_dielectric_cylinder() {
    let rows = a20().unwrap();
    println!("{rows:#?}");
    assert!(rows.iter().all(|r| r.pass), "{rows:#?}");
}
