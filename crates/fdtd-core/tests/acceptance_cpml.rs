//! Phase 2 acceptance for the CPML, through the public API only. All
//! quantities are normalized (dx = dy = 1, c0 = 1, dt = S). Reflection is
//! measured by the reference-domain method in `fdtd_core::reflection`.

use fdtd_core::cpml::PmlSpec;
use fdtd_core::reflection::{ReflectionSetup, reflection_db};
use std::sync::OnceLock;

/// The standard sweep and its reference records, computed once per test
/// binary. The reference is the expensive run (a PEC box about 700 x 600).
fn reference() -> &'static (ReflectionSetup, Vec<Vec<f64>>) {
    static REF: OnceLock<(ReflectionSetup, Vec<Vec<f64>>)> = OnceLock::new();
    REF.get_or_init(|| {
        let setup = ReflectionSetup::standard();
        let records = setup.reference().unwrap();
        (setup, records)
    })
}

/// Reflection in dB per probe for one layer, printed with each angle.
fn measure(spec: PmlSpec) -> (f64, Vec<f64>) {
    let (setup, reference) = reference();
    let db = reflection_db(&setup.test_domain(Some(spec)).unwrap(), reference);
    for (k, v) in db.iter().enumerate() {
        eprintln!(
            "  {:>2} cells, {:5.1} deg: {v:8.2} dB",
            spec.thickness,
            setup.angle_deg(k)
        );
    }
    (setup.worst_db(&db, 70.0), db)
}

/// A9: a 10-cell layer reflects at most -60 dB from normal incidence to 70
/// degrees, the interface corner included.
#[test]
fn a9_ten_cell_layer() {
    let (worst, _) = measure(PmlSpec::recommended(10));
    eprintln!("A9: worst 0 to 70 deg, 10 cells: {worst:.2} dB (gate -60)");
    assert!(worst <= -60.0, "{worst} dB");
}

/// A10: the tuned 16-cell layer. The spec asks for the best value to be
/// recorded; it also clears the -80 dB target, so that is asserted too.
#[test]
fn a10_tuned_sixteen_cell_layer() {
    let (worst, _) = measure(PmlSpec::recommended(16));
    eprintln!("A10: worst 0 to 70 deg, 16 cells: {worst:.2} dB (target -80)");
    assert!(worst <= -80.0, "{worst} dB");
}
