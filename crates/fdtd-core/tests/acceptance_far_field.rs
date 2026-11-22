//! A22 (double slit against Fraunhofer), A23 (phased array beam direction)
//! and A24 (photonic crystal stop band, recorded). See `cases::scattering`.

use fdtd_core::cases::scattering::{a22, a23, a24, stop_band};

#[test]
fn a22_double_slit() {
    let rows = a22().unwrap();
    println!("{rows:?}");
    assert!(rows.iter().all(|r| r.pass), "{rows:?}");
}

#[test]
fn a23_phased_array() {
    let rows = a23().unwrap();
    println!("{rows:?}");
    assert!(rows.iter().all(|r| r.pass), "{rows:?}");
}

/// Recorded, not gated: the crystal must show one stop band inside the
/// scan, and the run must be long enough that its edges no longer move by
/// more than one scan step (0.005 c / a).
#[test]
fn a24_photonic_crystal_recorded() {
    let rows = a24().unwrap();
    println!("{rows:?}");
    let r = &rows[0];
    assert!(r.pass && r.measured > 0.1 && r.measured < 0.6, "{r:?}");
    let moved: f64 = r
        .note
        .split("moved ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse().ok())
        .unwrap();
    assert!(moved < 0.005, "{r:?}");
}

#[test]
fn stop_band_edges_interpolate() {
    let t = [
        (0.1, 0.9),
        (0.2, 0.5),
        (0.3, 0.05),
        (0.4, 0.01),
        (0.5, 0.15),
        (0.6, 0.9),
    ];
    let (lo, hi) = stop_band(&t).unwrap();
    assert!((lo - (0.2 + 0.1 * 0.4 / 0.45)).abs() < 1e-12, "{lo}");
    assert!((hi - (0.4 + 0.1 * 0.09 / 0.14)).abs() < 1e-12, "{hi}");
    assert_eq!(stop_band(&[(0.1, 0.9), (0.2, 0.8)]), None);
}
