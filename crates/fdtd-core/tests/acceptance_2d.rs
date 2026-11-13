//! Phase 1 acceptance tests for the 2D TMz reference solver. Public API only.
//! All quantities are normalized (dx = dy = 1, c0 = 1, dt = S).

mod common;

use common::{max_abs, wavenumber_from_probes};
use fdtd_core::dispersion::numerical_wavenumber;
use fdtd_core::material::Material;
use fdtd_core::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
use fdtd_core::source::{SourceKind, Waveform};
use fdtd_core::tm::SimTm;
use std::f64::consts::{FRAC_PI_4, PI, SQRT_2};

fn vacuum(nx: usize, ny: usize, s: f64) -> Scene {
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: s,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![],
        monitors: vec![],
    }
}

/// Discrete kx of the lowest mode of a PEC parallel-plate guide (ky = pi /
/// (ny - 1)), from the 2D relation in closed form:
/// sin^2(kx/2) = (sin(omega S/2)/S)^2 - sin^2(ky/2).
fn guide_kx(omega: f64, ky: f64, s: f64) -> f64 {
    let r = (omega * s / 2.0).sin() / s;
    2.0 * (r * r - (ky / 2.0).sin().powi(2)).sqrt().asin()
}

/// Measures kx of the lowest TMz mode in a PEC parallel-plate guide of
/// `ny` nodes at frequency `omega`. Returns (kx measured, ky).
///
/// A PEC box cannot hold a ky = 0 wave, so A5 uses guided modes: each is a
/// pair of plane waves at angle atan(ky / kx), and the transverse profile
/// sin(ky j) is an exact eigenvector of the discrete y-difference operator. A
/// column of soft sources with that profile excites that mode alone, so the
/// probe spectra obey the exact F2 = F1 exp(-j kx L) of `wavenumber_from_probes`.
/// Waves reflected off the left wall are also right-going in the same mode and
/// do not disturb the ratio. The domain is long enough that the right-wall
/// echo never reaches probe 2 inside the record.
fn guide_measure(ny: usize, omega: f64, tau: f64) -> (f64, f64) {
    let s = 0.5;
    let ky = PI / (ny - 1) as f64;
    let t0 = 6.0 * tau;
    let (src, p1, len) = (10, 60, 200);
    let p2 = p1 + len;
    let record_t = t0 + 6.0 * tau + p2 as f64 / 0.6 + 50.0;
    let steps = (record_t / s).ceil() as u64;
    let nx = p2 + (record_t / 2.0) as usize + 50;
    let jp = (ny - 1) / 2;

    let mut scene = vacuum(nx, ny, s);
    for j in 1..ny - 1 {
        scene.sources.push(Source {
            i: src,
            j,
            waveform: Waveform::ModulatedGaussian {
                t0,
                tau,
                f0: omega / (2.0 * PI),
            },
            amplitude: (ky * j as f64).sin(),
            kind: SourceKind::Soft,
        });
    }
    scene.monitors = vec![
        Monitor::Point { i: p1, j: jp },
        Monitor::Point { i: p2, j: jp },
    ];
    let mut sim = SimTm::from_scene(&scene).unwrap();
    sim.step_n(steps);
    let (r1, r2) = (sim.probes()[0].history(), sim.probes()[1].history());
    let tail = &r2[r2.len() - 100..];
    assert!(
        max_abs(tail) < 1e-13 * max_abs(r2),
        "record cut off the pulse at probe 2"
    );
    let k_guess = (omega * omega - ky * ky).sqrt();
    let k_meas = wavenumber_from_probes(r1, r2, omega, s, len as f64, k_guess);
    (k_meas, ky)
}

/// A5 along the diagonal: the guide height sets ky, and omega is chosen so
/// the discrete kx equals ky, a wave at exactly 45 degrees. Checked against
/// the closed form and against the library's Newton solver.
fn a5_diagonal(ny: usize, tau: f64) {
    let s = 0.5;
    let ky = PI / (ny - 1) as f64;
    let omega = 2.0 / s * (s * SQRT_2 * (ky / 2.0).sin()).asin();
    let k_newton = numerical_wavenumber(omega, FRAC_PI_4, s).unwrap() / SQRT_2;
    assert!(
        (k_newton / ky - 1.0).abs() < 1e-13,
        "setup: {k_newton} vs {ky}"
    );
    let (k_meas, _) = guide_measure(ny, omega, tau);
    let err = (k_meas / ky - 1.0).abs();
    let cpw = 2.0 * PI / (SQRT_2 * ky);
    let continuum = (SQRT_2 * ky / omega - 1.0).abs();
    println!(
        "A5 diagonal, {cpw:.2} cpw: kx_meas {k_meas:.12}, kx_d {ky:.12}, rel err {err:.3e}, discrete vs continuum {continuum:.3e}"
    );
    assert!(err < 1e-4, "relative error {err:e}");
}

/// A5 near the axis: a wide guide makes the mode a plane wave a few degrees
/// off the x axis.
fn a5_near_axis(cpw: f64, ny: usize) {
    let s = 0.5;
    let omega = 2.0 * PI / cpw;
    let (k_meas, ky) = guide_measure(ny, omega, 3.0 * cpw);
    let k_d = guide_kx(omega, ky, s);
    let k_c = (omega * omega - ky * ky).sqrt();
    let err = (k_meas / k_d - 1.0).abs();
    let gap = (k_d / k_c - 1.0).abs();
    let theta = ky.atan2(k_d).to_degrees();
    println!(
        "A5 near axis ({theta:.2} deg), {cpw} cpw: kx_meas {k_meas:.12}, kx_d {k_d:.12}, rel err {err:.3e}, discrete vs continuum {gap:.3e}"
    );
    assert!(err < 1e-4, "relative error {err:e}");
    // The gate must be able to tell grid physics from continuum physics.
    assert!(gap > 1e-3, "discrete and continuum too close: {gap:e}");
}

#[test]
fn a5_dispersion_diagonal_10_cpw() {
    a5_diagonal(8, 80.0);
}

#[test]
fn a5_dispersion_diagonal_20_cpw() {
    a5_diagonal(15, 130.0);
}

#[test]
fn a5_dispersion_near_axis_10_cpw() {
    a5_near_axis(10.0, 65);
}

#[test]
fn a5_dispersion_near_axis_20_cpw() {
    a5_near_axis(20.0, 65);
}
