//! A22 (double slit against Fraunhofer), A23 (phased array beam direction)
//! and A24 (photonic crystal gap edges against MPB). See `cases::scattering`.

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

/// A24: the stop band edges within 3 percent of the MPB Gamma-X edges.
#[test]
fn a24_photonic_crystal() {
    let rows = a24().unwrap();
    println!("{rows:?}");
    assert!(rows.iter().all(|r| r.pass), "{rows:?}");
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
