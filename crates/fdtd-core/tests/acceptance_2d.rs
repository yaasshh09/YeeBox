//! Phase 1 acceptance tests for the 2D TMz reference solver. Public API only.
//! All quantities are normalized (dx = dy = 1, c0 = 1, dt = S).

mod common;

use common::{blackman_harris, golden_max, magnitude, max_abs, wavenumber_from_probes};
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

/// Discrete resonance of PEC cavity mode (m, n) for a cavity of `a` by `b`
/// cells: the 2D relation with kx = m pi / a, ky = n pi / b, solved for omega.
fn cavity_discrete(m: usize, n: usize, a: usize, b: usize, s: f64) -> f64 {
    let kx = m as f64 * PI / a as f64;
    let ky = n as f64 * PI / b as f64;
    let r = ((kx / 2.0).sin().powi(2) + (ky / 2.0).sin().powi(2)).sqrt();
    2.0 / s * (s * r).asin()
}

/// Continuum resonance `pi sqrt((m/a)^2 + (n/b)^2)` (c0 = 1, lengths in cells).
fn cavity_continuum(m: usize, n: usize, a: usize, b: usize) -> f64 {
    PI * ((m as f64 / a as f64).powi(2) + (n as f64 / b as f64).powi(2)).sqrt()
}

/// Runs a closed PEC cavity of `a` by `b` cells driven by a Ricker pulse at
/// `src` and returns `steps` samples of Ez at `probe`, starting once the pulse
/// is over (t > 2 t0), so the record is a pure sum of undamped cavity modes.
fn cavity_record(
    a: usize,
    b: usize,
    s: f64,
    f_peak: f64,
    src: (usize, usize),
    probe: (usize, usize),
    steps: u64,
) -> Vec<f64> {
    let t0 = 6.0 / (PI * f_peak);
    let skip = (2.0 * t0 / s).ceil() as usize;
    let mut scene = vacuum(a + 1, b + 1, s);
    scene.sources.push(Source {
        i: src.0,
        j: src.1,
        waveform: Waveform::Ricker { t0, f_peak },
        amplitude: 1.0,
        kind: SourceKind::Soft,
    });
    scene.monitors = vec![Monitor::Point {
        i: probe.0,
        j: probe.1,
    }];
    let mut sim = SimTm::from_scene(&scene).unwrap();
    sim.step_n(skip as u64 + steps);
    sim.probes()[0].history()[skip..].to_vec()
}

/// Every local maximum of the Blackman-Harris windowed spectrum in
/// (0, w_max] that is within 60 dB of the largest, refined by golden-section
/// search. The scan step is a quarter of a DFT bin; the window's main lobe is
/// 8 bins wide and its sidelobes are below -92 dB, so sidelobes never count.
fn spectral_peaks(rec: &[f64], s: f64, w_max: f64) -> Vec<f64> {
    let win = blackman_harris(rec.len());
    let bin = 2.0 * PI / (rec.len() as f64 * s);
    let step = bin / 4.0;
    let grid: Vec<f64> = (1..)
        .map(|k| k as f64 * step)
        .take_while(|&w| w <= w_max)
        .collect();
    let mag: Vec<f64> = grid.iter().map(|&w| magnitude(rec, &win, w, s)).collect();
    let top = mag.iter().cloned().fold(0.0, f64::max);
    let mut peaks = Vec::new();
    for k in 1..grid.len() - 1 {
        if mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] && mag[k] > 1e-3 * top {
            let (lo, hi) = (grid[k - 1], grid[k + 1]);
            let w = golden_max(|w| magnitude(rec, &win, w, s), lo, hi, 1e-13);
            assert!(
                w > lo + 1e-12 && w < hi - 1e-12,
                "peak stuck on its bracket"
            );
            peaks.push(w);
        }
    }
    peaks
}

/// A6. A 40 x 23 cell PEC cavity (lowest seven modes at least 4.3 percent
/// apart) rings after a Ricker pulse. Source and probe sit on nodes where
/// every one of the lowest seven modes is nonzero. A blind peak search must
/// find exactly six resonances below the midpoint of modes 6 and 7, and each
/// must match the discrete prediction to 0.01 percent.
#[test]
fn a6_cavity_first_six_resonances() {
    let (a, b, s) = (40, 23, 0.5);
    let mut predicted: Vec<(f64, usize, usize)> = (1..10)
        .flat_map(|m| (1..10).map(move |n| (cavity_discrete(m, n, a, b, s), m, n)))
        .collect();
    predicted.sort_by(|x, y| x.0.total_cmp(&y.0));
    let w_max = 0.5 * (predicted[5].0 + predicted[6].0);

    let rec = cavity_record(a, b, s, 0.04, (7, 5), (29, 16), 20_000);
    let peaks = spectral_peaks(&rec, s, w_max);
    assert_eq!(peaks.len(), 6, "peaks found: {peaks:?}");
    for (w, &(w_d, m, n)) in peaks.iter().zip(&predicted) {
        let err = (w / w_d - 1.0).abs();
        let gap = (w_d / cavity_continuum(m, n, a, b) - 1.0).abs();
        println!(
            "A6 mode ({m}, {n}): measured {w:.10}, discrete {w_d:.10}, rel err {err:.2e}, discrete vs continuum {gap:.2e}"
        );
        assert!(err < 1e-4, "mode ({m}, {n}): relative error {err:e}");
    }
}

/// Measured omega of mode (1, 1) of a 4L x 3L cell cavity, source and probe
/// at the center node so only odd modes ring and (1, 1) stays isolated.
fn cavity_11(l: usize, s: f64) -> f64 {
    let (a, b) = (4 * l, 3 * l);
    let w_c = cavity_continuum(1, 1, a, b);
    let steps = (60.0 * a as f64 / s) as u64;
    let rec = cavity_record(
        a,
        b,
        s,
        w_c / (2.0 * PI),
        (a / 2, b / 2),
        (a / 2, b / 2),
        steps,
    );
    let peaks = spectral_peaks(&rec, s, 1.5 * w_c);
    assert_eq!(peaks.len(), 1, "L = {l}: peaks {peaks:?}");
    peaks[0]
}

/// A7. The measured (1, 1) resonance converges to the continuum value
/// pi sqrt(1/a^2 + 1/b^2) at second order as the same physical cavity is
/// resolved with L = 4, 8, 16, 32 cells per unit (16 x 12 up to 128 x 96).
#[test]
fn a7_cavity_convergence_order() {
    let s = 0.5;
    let pts: Vec<(f64, f64)> = [4, 8, 16, 32]
        .iter()
        .map(|&l| {
            let w = cavity_11(l, s);
            let w_c = cavity_continuum(1, 1, 4 * l, 3 * l);
            let w_d = cavity_discrete(1, 1, 4 * l, 3 * l, s);
            let err = (w / w_c - 1.0).abs();
            println!(
                "A7 L = {l}: measured {w:.12}, continuum {w_c:.12}, rel err {err:.4e}, measured vs discrete {:.1e}",
                (w / w_d - 1.0).abs()
            );
            ((1.0 / l as f64).ln(), err.ln())
        })
        .collect();
    let m = pts.len() as f64;
    let (sx, sy) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
    let (mx, my) = (sx / m, sy / m);
    let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let order = num / den;
    println!("A7 fitted order {order:.4}");
    assert!((1.8..=2.2).contains(&order), "order {order}");
}
