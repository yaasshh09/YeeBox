//! Phase 0 acceptance tests for the 1D reference solver. Public API only.
//! All quantities are normalized (dx = 1, c0 = 1, dt = S).

use fdtd_core::dispersion::numerical_wavenumber_1d;
use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::source::{SourceKind, Waveform};
use std::f64::consts::PI;

/// Runs `steps` steps and records ez at each node in `probes` after every step.
fn record(sim: &mut Sim1d, steps: usize, probes: &[usize]) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(steps); probes.len()];
    for _ in 0..steps {
        sim.step();
        for (trace, &p) in out.iter_mut().zip(probes) {
            trace.push(sim.ez()[p]);
        }
    }
    out
}

fn gaussian_source(node: usize) -> Source1d {
    Source1d {
        node,
        waveform: Waveform::Gaussian {
            t0: 60.0,
            tau: 10.0,
        },
        amplitude: 1.0,
        kind: SourceKind::Soft,
    }
}

/// A1: at S = 1 the 1D update is the exact d'Alembert recurrence
/// E_i^{n+1} + E_i^{n-1} = E_{i+1}^n + E_{i-1}^n, so a pulse translates one cell
/// per step with no dispersion. Probe 2 must replay probe 1 exactly, 1000 steps late.
#[test]
fn a1_magic_time_step_1000_cells() {
    let (src, x0, dist) = (50, 100, 1000);
    let mut sim = Sim1d::new(1200, 1.0, Boundary1d::Mur1, Boundary1d::Mur1).unwrap();
    sim.add_source(gaussian_source(src)).unwrap();

    let traces = record(&mut sim, 1300, &[x0, x0 + dist]);
    let (p1, p2) = (&traces[0], &traces[1]);

    let peak = p1.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.1, "pulse never reached probe 1, peak {peak}");
    let err = (0..p1.len() - dist)
        .map(|n| (p2[n + dist] - p1[n]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "A1 shape error after {dist} cells: {:.3e} of peak",
        err / peak
    );
    assert!(err < 1e-12 * peak, "shape error {err:e}, peak {peak}");
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// DTFT `sum_n p[n] exp(-j omega n S)` of a probe record, as (re, im).
fn dtft(p: &[f64], omega: f64, s: f64) -> (f64, f64) {
    p.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = omega * n as f64 * s;
        (re + v * ph.cos(), im - v * ph.sin())
    })
}

/// A2 at `cpw` cells per wavelength. Returns (measured k, discrete k, continuum k).
///
/// In a source-free stretch of the discrete line, a right-going field has
/// probe spectra related exactly by F2(omega) = F1(omega) exp(-j k_d L), where
/// k_d is the discrete wavenumber. So -arg(F2 / F1) / L measures k_d directly.
/// The domain is long enough that nothing reflected off the right end reaches
/// probe 2 inside the record, so only right-going waves are ever recorded.
fn a2_measure(cpw: f64) -> (f64, f64, f64) {
    let s = 0.5;
    let omega = 2.0 * PI / cpw;
    let tau = 3.0 * cpw;
    let t0 = 6.0 * tau;
    let (src, p1, len) = (20, 100, 200);
    let p2 = p1 + len;
    // Pulse fully past probe 2 (group velocity above 0.9 here), plus margin.
    let record_t = t0 + 6.0 * tau + p2 as f64 / 0.9 + 50.0;
    let steps = (record_t / s).ceil() as usize;
    // Right-end echo needs at least 2 * (nx - p2) time units to come back to p2.
    let nx = p2 + (record_t / 2.0) as usize + 100;

    let mut sim = Sim1d::new(nx, s, Boundary1d::Mur1, Boundary1d::Mur1).unwrap();
    sim.add_source(Source1d {
        node: src,
        waveform: Waveform::ModulatedGaussian {
            t0,
            tau,
            f0: 1.0 / cpw,
        },
        amplitude: 1.0,
        kind: SourceKind::Soft,
    })
    .unwrap();
    let tr = record(&mut sim, steps, &[p1, p2]);
    assert!(
        max_abs(&tr[1][steps - 50..]) < 1e-13,
        "record cut off the pulse"
    );

    let (a, b) = dtft(&tr[0], omega, s);
    let (c, d) = dtft(&tr[1], omega, s);
    // arg(F2 / F1) = arg(F2 * conj(F1))
    let phase = (d * a - c * b).atan2(c * a + d * b);
    // Choose the 2 pi branch nearest the continuum k; valid while |k_d - k_c| L < pi.
    let m = ((omega * len as f64 + phase) / (2.0 * PI)).round();
    let k_meas = (2.0 * PI * m - phase) / len as f64;
    (k_meas, numerical_wavenumber_1d(omega, s).unwrap(), omega)
}

fn a2_check(cpw: f64) {
    let (k_meas, k_d, k_c) = a2_measure(cpw);
    let err = (k_meas / k_d - 1.0).abs();
    println!(
        "A2 {cpw} cpw: k_meas {k_meas:.12}, k_d {k_d:.12}, rel err {err:.3e}, k_d vs continuum {:.3e}",
        (k_d / k_c - 1.0).abs()
    );
    assert!(err < 1e-4, "{cpw} cpw: relative error {err:e}");
}

#[test]
fn a2_phase_velocity_10_cpw() {
    a2_check(10.0);
    // The test must be able to tell discrete physics from continuum physics:
    // at 10 cpw they differ by far more than the tolerance.
    let (_, k_d, k_c) = a2_measure(10.0);
    assert!((k_d / k_c - 1.0).abs() > 1e-3);
}

#[test]
fn a2_phase_velocity_20_cpw() {
    a2_check(20.0);
}

#[test]
fn a2_phase_velocity_40_cpw() {
    a2_check(40.0);
}

/// A3 at S = 1: off a PEC wall the pulse comes back inverted with equal
/// magnitude, delayed by exactly the round trip 2 (wall - probe) steps.
#[test]
fn a3_pec_reflection_inverted_equal() {
    let (src, probe, wall) = (50, 60, 400);
    let round_trip = 2 * (wall - probe);
    let mut sim = Sim1d::new(wall + 1, 1.0, Boundary1d::Mur1, Boundary1d::Pec).unwrap();
    sim.add_source(gaussian_source(src)).unwrap();
    let p = &record(&mut sim, 2 * round_trip, &[probe])[0];

    let peak = max_abs(&p[..round_trip]);
    assert!(peak > 0.1);
    // For n < round_trip the probe holds only the incident pulse, and p[n + R]
    // holds only the reflection of it.
    let err = (0..round_trip)
        .map(|n| (p[n + round_trip] + p[n]).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "A3 S=1 inverted reflection error: {:.3e} of peak",
        err / peak
    );
    assert!(err < 1e-10 * peak, "error {err:e}");
}

/// A3 at S = 0.5, by the image method: a PEC wall at node W is equivalent to
/// an open, mirror-symmetric line with a negated image source at 2W - a. The
/// odd symmetry forces ez[W] = 0 at every step, for any S.
#[test]
fn a3_pec_image_equivalence_s05() {
    let (s, a, w, steps) = (0.5, 60, 150, 1200);
    let source = |node, amplitude| Source1d {
        node,
        waveform: Waveform::Gaussian {
            t0: 60.0,
            tau: 10.0,
        },
        amplitude,
        kind: SourceKind::Soft,
    };

    let mut wall = Sim1d::new(w + 1, s, Boundary1d::Mur1, Boundary1d::Pec).unwrap();
    wall.add_source(source(a, 1.0)).unwrap();
    let mut image = Sim1d::new(2 * w + 1, s, Boundary1d::Mur1, Boundary1d::Mur1).unwrap();
    image.add_source(source(a, 1.0)).unwrap();
    image.add_source(source(2 * w - a, -1.0)).unwrap();

    let (mut err, mut peak) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        wall.step();
        image.step();
        for i in 0..=w {
            err = err.max((wall.ez()[i] - image.ez()[i]).abs());
            peak = peak.max(wall.ez()[i].abs());
        }
    }
    println!(
        "A3 S=0.5 image equivalence error: {:.3e} of peak",
        err / peak
    );
    assert!(peak > 0.1);
    assert!(err < 1e-12 * peak, "error {err:e}");
}
