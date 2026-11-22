//! Phase 7, part A: the near-to-far-field transform on solver fields, in
//! TMz and TEz. A soft line source radiates the same in every direction
//! (the acceptance gate, `cases::scattering::isotropy`); a second contour
//! gives the same far field; moving the source changes only the phase;
//! outside the contour it radiates nothing; and the far-field power equals
//! the flux through the contour.
//!
//! What is left in each is the grid's O((k h)^2) error, mostly the node
//! average of H, which is low by cos(k_n / 2) (1.2 percent at 20 cells per
//! wavelength) and so tips the J and M balance. I set the bounds before the
//! runs from that size: 1e-2 (and 0.1 dB, 5e-3 for power) at 40 cells per
//! wavelength. At 20 cells per wavelength the same numbers are 3 to 10
//! times larger, as a second-order error should be, so the test asserts
//! that each falls by at least 2.5 times from 20 to 40 cells per
//! wavelength. Measured (40 / 20 cells per wavelength): contour size
//! 3.7e-3 / 1.1e-2, phase 3.2e-3 / 1.3e-2 rad, moved-source ripple 0.030 /
//! 0.118 dB, extinction 3.0e-3 / 3.0e-2, power 1.8e-3 / 4.3e-3.

use fdtd_core::cases::scattering::{
    LINE_FREQS, contour, far_fields, isotropy, line_source_scene, ripple_db, run_spectra,
};
use fdtd_core::kernel::Spectrum;
use fdtd_core::monitor::Normal;
use fdtd_core::scene::{DEFAULT_ANGLES, Mode, Monitor};
use std::f64::consts::PI;

const N: usize = 121;
const C: usize = 60;
const A: usize = DEFAULT_ANGLES;

/// Frequency index `f` of a far field.
fn row(far: &[(f64, f64)], f: usize) -> &[(f64, f64)] {
    &far[f * A..(f + 1) * A]
}

fn peak(v: &[(f64, f64)]) -> f64 {
    v.iter().fold(0.0, |m, v| m.max(v.0.hypot(v.1)))
}

/// Checks a residual measured at `LINE_FREQS` (40, then 20 cells per
/// wavelength): under `bound` at 40, and at least 2.5 times smaller there.
fn second_order(what: &str, mode: Mode, v: [f64; 2], bound: f64) {
    println!("{mode:?} {what}: {:.3e} / {:.3e}", v[0], v[1]);
    assert!(v[0] < bound, "{mode:?} {what}: {:?}", v);
    assert!(
        v[1] > 2.5 * v[0],
        "{mode:?} {what} does not fall with h: {:?}",
        v
    );
}

#[test]
fn line_source_is_isotropic() {
    let rows = isotropy().unwrap();
    assert!(rows.iter().all(|r| r.pass), "{rows:?}");
}

#[test]
fn contour_size_and_source_position() {
    for mode in [Mode::Tm, Mode::Te] {
        let ring = |l: usize| contour(C - l, C - l, C + l, C + l, &LINE_FREQS);
        let far = far_fields(&line_source_scene(
            N,
            (C, C),
            vec![ring(20), ring(30)],
            mode,
        ))
        .unwrap();
        let (dx, dy) = (7.0, -4.0);
        let moved =
            &far_fields(&line_source_scene(N, (C + 7, C - 4), vec![ring(20)], mode)).unwrap()[0];
        let outside = contour(C + 10, C - 15, C + 40, C + 15, &LINE_FREQS);
        let empty = &far_fields(&line_source_scene(N, (C, C), vec![outside], mode)).unwrap()[0];
        let (mut size, mut phase, mut flat, mut ext) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 2]);
        for (f, &freq) in LINE_FREQS.iter().enumerate() {
            let k = 2.0 * PI * freq;
            let (a, b, m) = (row(&far[0], f), row(&far[1], f), row(moved, f));
            for i in 0..A {
                let d = (b[i].0 - a[i].0).hypot(b[i].1 - a[i].1) / a[i].0.hypot(a[i].1);
                size[f] = f64::max(size[f], d);
                let phi = 2.0 * PI * i as f64 / A as f64;
                let want = k * (phi.cos() * dx + phi.sin() * dy);
                let got =
                    (m[i].1 * a[i].0 - m[i].0 * a[i].1).atan2(m[i].0 * a[i].0 + m[i].1 * a[i].1);
                let e = ((got - want + PI).rem_euclid(2.0 * PI) - PI).abs();
                phase[f] = f64::max(phase[f], e);
            }
            flat[f] = ripple_db(moved, A, f);
            ext[f] = peak(row(empty, f)) / peak(a);
        }
        second_order("contour size", mode, size, 1e-2);
        second_order("moved source phase", mode, phase, 1e-2);
        second_order("moved source ripple dB", mode, flat, 0.1);
        second_order("source outside", mode, ext, 1e-2);
    }
}

/// `sum |F|^2 / (2 eta) dphi` (eta = 1) against the outward flux through
/// four flux lines on the contour's own box. The x lines take the corner
/// nodes and the y lines stop one node short, so each node is counted once.
#[test]
fn far_field_power_equals_contour_flux() {
    let (x0, x1) = (C - 20, C + 20);
    let flux = |normal, at, from, to| Monitor::Flux {
        normal,
        at,
        from,
        to,
        freqs: LINE_FREQS.to_vec(),
    };
    for mode in [Mode::Tm, Mode::Te] {
        let monitors = vec![
            contour(x0, x0, x1, x1, &LINE_FREQS),
            flux(Normal::X, x0, x0, x1),
            flux(Normal::X, x1, x0, x1),
            flux(Normal::Y, x0, x0 + 1, x1 - 1),
            flux(Normal::Y, x1, x0 + 1, x1 - 1),
        ];
        let spectra = run_spectra(&line_source_scene(N, (C, C), monitors, mode)).unwrap();
        let Spectrum::FarField(far) = &spectra[0] else {
            panic!("{:?}", spectra[0]);
        };
        let lines: Vec<&Vec<f64>> = spectra[1..]
            .iter()
            .map(|s| match s {
                Spectrum::Flux(v) => v,
                other => panic!("{other:?}"),
            })
            .collect();
        for f in 0..LINE_FREQS.len() {
            let p: f64 = row(far, f)
                .iter()
                .map(|v| (v.0 * v.0 + v.1 * v.1) / 2.0)
                .sum::<f64>();
            let p = p * 2.0 * PI / A as f64;
            let out = -lines[0][f] + lines[1][f] - lines[2][f] + lines[3][f];
            let err = (p / out - 1.0).abs();
            println!("{mode:?} power f {}: {err:.3e}", LINE_FREQS[f]);
            assert!(err < 5e-3, "{mode:?} f {}: {err}", LINE_FREQS[f]);
        }
    }
}
