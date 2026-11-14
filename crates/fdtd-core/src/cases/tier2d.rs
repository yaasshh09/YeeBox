//! A4 to A7: the 2D TMz solver in closed PEC boxes.

use super::signal::{
    blackman_harris, golden_max, magnitude, max_abs, random_fields, slope, wavenumber_from_probes,
};
use super::{CaseResult, Gate, ensure};
use crate::dispersion::numerical_wavenumber;
use crate::error::Result;
use crate::material::Material;
use crate::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
use crate::source::{SourceKind, Waveform};
use crate::tm::SimTm;
use std::f64::consts::{FRAC_PI_4, PI, SQRT_2};

/// An empty vacuum box with PEC walls.
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
        pml: None,
        periodic_y: false,
        plane_wave: None,
    }
}

/// A4: a random field in a 32 x 32 PEC box stays bounded for 20000 steps at
/// S = 0.70, and at S = 0.72, above 1/sqrt(2), the highest box mode grows
/// by about 1.44 per step. Measured: the largest |Ez| at S = 0.70 over the
/// starting peak. Guard: S = 0.72 must pass a million times its start
/// within 2000 steps.
pub fn a4() -> Result<Vec<CaseResult>> {
    let n = 32;
    let start = random_fields(7, n * n);
    let peak0 = max_abs(&start);
    let mut sim = SimTm::from_scene(&vacuum(n, n, 0.70))?;
    sim.set_ez(&start)?;
    let mut worst = 0.0_f64;
    for _ in 0..20_000 {
        sim.step();
        worst = worst.max(max_abs(sim.ez()));
    }
    let mut sim = SimTm::from_scene_unchecked(&vacuum(n, n, 0.72))?;
    sim.set_ez(&start)?;
    let mut steps = 0;
    while max_abs(sim.ez()) <= 1e6 * peak0 {
        ensure(steps < 2000, || {
            "A4: S = 0.72 did not blow up in 2000 steps".into()
        })?;
        sim.step();
        steps += 1;
    }
    Ok(vec![CaseResult::new(
        "A4",
        "2D stability, bounded at S = 0.70",
        worst / peak0,
        Gate::AtMost(10.0),
        "x start",
        format!("32 x 32 box, 20000 steps; S = 0.72 passed 1e6x at step {steps}"),
    )])
}

/// Discrete kx of the lowest mode of a PEC parallel-plate guide (ky = pi /
/// (ny - 1)): sin^2(kx/2) = (sin(omega S/2)/S)^2 - sin^2(ky/2).
fn guide_kx(omega: f64, ky: f64, s: f64) -> f64 {
    let r = (omega * s / 2.0).sin() / s;
    2.0 * (r * r - (ky / 2.0).sin().powi(2)).sqrt().asin()
}

/// kx of the lowest TMz mode in a PEC parallel-plate guide of `ny` nodes
/// at `omega`, measured between two probes 200 cells apart. Returns (kx, ky).
///
/// A PEC box cannot hold a ky = 0 wave, so A5 uses guided modes: each is a
/// pair of plane waves at angle atan(ky / kx), and the profile sin(ky j) is
/// an exact eigenvector of the discrete y-difference. A column of soft
/// sources with that profile excites that mode alone, so the probe spectra
/// obey F2 = F1 exp(-j kx L) exactly. Waves reflected off the left wall are
/// right-going in the same mode too. The right-wall echo never reaches
/// probe 2 inside the record.
fn guide_measure(ny: usize, omega: f64, tau: f64) -> Result<(f64, f64)> {
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
    let mut sim = SimTm::from_scene(&scene)?;
    sim.step_n(steps);
    let (r1, r2) = (sim.probes()[0].history(), sim.probes()[1].history());
    ensure(max_abs(&r2[r2.len() - 100..]) < 1e-13 * max_abs(r2), || {
        "A5: record cut off the pulse at probe 2".into()
    })?;
    let k_guess = (omega * omega - ky * ky).sqrt();
    Ok((
        wavenumber_from_probes(r1, r2, omega, s, len as f64, k_guess),
        ky,
    ))
}

/// A5: wavelength on the diagonal and near the axis against the discrete
/// relation, at 10 and 20 cells per wavelength, S = 0.5. On the diagonal
/// the guide height sets ky and omega is chosen so the discrete kx equals
/// ky (45 degrees, checked against the Newton solver); near the axis a wide
/// guide gives a plane wave a few degrees off x. Measured: the worst
/// relative error. Guard: near the axis, discrete and continuum differ by
/// more than 1e-3, so the gate tells them apart.
pub fn a5() -> Result<Vec<CaseResult>> {
    let s = 0.5;
    let mut worst = 0.0_f64;
    let mut notes = Vec::new();
    for (ny, tau) in [(8, 80.0), (15, 130.0)] {
        let ky = PI / (ny - 1) as f64;
        let omega = 2.0 / s * (s * SQRT_2 * (ky / 2.0).sin()).asin();
        let k_newton = numerical_wavenumber(omega, FRAC_PI_4, s)? / SQRT_2;
        ensure((k_newton / ky - 1.0).abs() < 1e-13, || {
            format!("A5 setup: {k_newton} vs {ky}")
        })?;
        let (k_meas, _) = guide_measure(ny, omega, tau)?;
        worst = worst.max((k_meas / ky - 1.0).abs());
        notes.push(format!("diagonal {:.1} cpw", 2.0 * PI / (SQRT_2 * ky)));
    }
    for cpw in [10.0, 20.0] {
        let omega = 2.0 * PI / cpw;
        let (k_meas, ky) = guide_measure(65, omega, 3.0 * cpw)?;
        let k_d = guide_kx(omega, ky, s);
        let k_c = (omega * omega - ky * ky).sqrt();
        worst = worst.max((k_meas / k_d - 1.0).abs());
        let gap = (k_d / k_c - 1.0).abs();
        ensure(gap > 1e-3, || {
            format!("A5: discrete vs continuum only {gap:e}")
        })?;
        notes.push(format!("{:.1} deg {cpw} cpw", ky.atan2(k_d).to_degrees()));
    }
    Ok(vec![CaseResult::new(
        "A5",
        "2D dispersion, axis and diagonal vs discrete",
        worst,
        Gate::AtMost(1e-4),
        "rel",
        format!("S = 0.5, guided modes: {}", notes.join(", ")),
    )])
}

/// Discrete resonance of PEC cavity mode (m, n), `a` by `b` cells.
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

/// A closed PEC cavity of `a` by `b` cells rung by a Ricker pulse at `src`:
/// `steps` samples of Ez at `probe` from t = 2 t0 on, so the record is a
/// pure sum of undamped modes.
fn cavity_record(
    a: usize,
    b: usize,
    s: f64,
    f_peak: f64,
    src: (usize, usize),
    probe: (usize, usize),
    steps: u64,
) -> Result<Vec<f64>> {
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
    let mut sim = SimTm::from_scene(&scene)?;
    sim.step_n(skip as u64 + steps);
    Ok(sim.probes()[0].history()[skip..].to_vec())
}

/// Every local maximum of the Blackman-Harris windowed spectrum in
/// (0, w_max] within 60 dB of the largest, refined by golden-section
/// search. The scan step is a quarter bin; the window's main lobe is 8 bins
/// wide and its sidelobes are below -92 dB, so sidelobes never count.
fn spectral_peaks(rec: &[f64], s: f64, w_max: f64) -> Result<Vec<f64>> {
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
            ensure(w > lo + 1e-12 && w < hi - 1e-12, || {
                "peak stuck on its bracket".into()
            })?;
            peaks.push(w);
        }
    }
    Ok(peaks)
}

/// A6: a 40 x 23 cell PEC cavity (lowest seven modes at least 4.3 percent
/// apart) rings after a Ricker pulse, source and probe on nodes where each
/// of those modes is nonzero. A blind peak search must find exactly six
/// resonances below the midpoint of modes 6 and 7. Measured: the worst
/// relative error against the discrete prediction.
pub fn a6() -> Result<Vec<CaseResult>> {
    let (a, b, s) = (40, 23, 0.5);
    let mut predicted: Vec<(f64, usize, usize)> = (1..10)
        .flat_map(|m| (1..10).map(move |n| (cavity_discrete(m, n, a, b, s), m, n)))
        .collect();
    predicted.sort_by(|x, y| x.0.total_cmp(&y.0));
    let w_max = 0.5 * (predicted[5].0 + predicted[6].0);
    let rec = cavity_record(a, b, s, 0.04, (7, 5), (29, 16), 20_000)?;
    let peaks = spectral_peaks(&rec, s, w_max)?;
    ensure(peaks.len() == 6, || {
        format!("A6: found {} peaks", peaks.len())
    })?;
    let worst = peaks
        .iter()
        .zip(&predicted)
        .map(|(w, &(w_d, _, _))| (w / w_d - 1.0).abs())
        .fold(0.0_f64, f64::max);
    let gap = (predicted[0].0 / cavity_continuum(1, 1, a, b) - 1.0).abs();
    Ok(vec![CaseResult::new(
        "A6",
        "PEC cavity, first 6 resonances vs discrete",
        worst,
        Gate::AtMost(1e-4),
        "rel",
        format!("40 x 23 cells, S = 0.5; mode (1, 1) discrete vs continuum {gap:.1e}"),
    )])
}

/// Measured omega of mode (1, 1) of a 4L x 3L cavity, source and probe at
/// the centre node so only odd modes ring and (1, 1) stays isolated.
fn cavity_11(l: usize, s: f64) -> Result<f64> {
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
    )?;
    let peaks = spectral_peaks(&rec, s, 1.5 * w_c)?;
    ensure(peaks.len() == 1, || {
        format!("A7 L = {l}: {} peaks", peaks.len())
    })?;
    Ok(peaks[0])
}

/// A7: mode (1, 1) of the same physical cavity resolved with L = 4, 8, 16,
/// 32 cells per unit (16 x 12 up to 128 x 96) converges to the continuum
/// pi sqrt(1/a^2 + 1/b^2). Measured: the fitted order.
pub fn a7() -> Result<Vec<CaseResult>> {
    let s = 0.5;
    let mut pts = Vec::new();
    let mut errs = Vec::new();
    for l in [4, 8, 16, 32] {
        let w = cavity_11(l, s)?;
        let err = (w / cavity_continuum(1, 1, 4 * l, 3 * l) - 1.0).abs();
        errs.push(err);
        pts.push(((1.0 / l as f64).ln(), err.ln()));
    }
    Ok(vec![CaseResult::new(
        "A7",
        "PEC cavity convergence to continuum",
        slope(&pts),
        Gate::Range(1.8, 2.2),
        "order",
        format!(
            "4L x 3L, L = 4..32; error {:.2e} at L = 4, {:.2e} at L = 32",
            errs[0], errs[3]
        ),
    )])
}
