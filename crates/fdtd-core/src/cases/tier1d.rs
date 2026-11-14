//! A1 to A3: the 1D reference solver.

use super::signal::{max_abs, wavenumber_from_probes};
use super::{CaseResult, Gate, ensure};
use crate::dispersion::numerical_wavenumber_1d;
use crate::error::Result;
use crate::fdtd1d::{Boundary1d, Sim1d, Source1d};
use crate::source::{SourceKind, Waveform};
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

fn gaussian_source(node: usize, amplitude: f64) -> Source1d {
    Source1d {
        node,
        waveform: Waveform::Gaussian {
            t0: 60.0,
            tau: 10.0,
        },
        amplitude,
        kind: SourceKind::Soft,
    }
}

/// A1: at S = 1 the 1D update is the exact d'Alembert recurrence
/// E_i^{n+1} + E_i^{n-1} = E_{i+1}^n + E_{i-1}^n, so a pulse translates one
/// cell per step with no dispersion. Probe 2 must replay probe 1 exactly,
/// 1000 steps late. Measured: the largest difference relative to the peak.
pub fn a1() -> Result<Vec<CaseResult>> {
    let (src, x0, dist) = (50, 100, 1000);
    let mut sim = Sim1d::new(1200, 1.0, Boundary1d::Mur1, Boundary1d::Mur1)?;
    sim.add_source(gaussian_source(src, 1.0))?;
    let traces = record(&mut sim, 1300, &[x0, x0 + dist]);
    let (p1, p2) = (&traces[0], &traces[1]);
    let peak = max_abs(p1);
    ensure(peak > 0.1, || {
        format!("A1: pulse never reached probe 1 ({peak})")
    })?;
    let err = (0..p1.len() - dist)
        .map(|n| (p2[n + dist] - p1[n]).abs())
        .fold(0.0_f64, f64::max);
    Ok(vec![CaseResult::new(
        "A1",
        "1D magic time step, shape after 1000 cells",
        err / peak,
        Gate::AtMost(1e-12),
        "rel",
        "S = 1, Gaussian tau 10".into(),
    )])
}

/// A2 at `cpw` cells per wavelength: (measured k, discrete k, continuum k).
///
/// In a source-free stretch of the discrete line, a right-going field has
/// probe spectra related exactly by F2(omega) = F1(omega) exp(-j k_d L), so
/// -arg(F2 / F1) / L measures k_d. The domain is long enough that nothing
/// reflected off the right end reaches probe 2 inside the record.
pub fn a2_measure(cpw: f64) -> Result<(f64, f64, f64)> {
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
    let mut sim = Sim1d::new(nx, s, Boundary1d::Mur1, Boundary1d::Mur1)?;
    sim.add_source(Source1d {
        node: src,
        waveform: Waveform::ModulatedGaussian {
            t0,
            tau,
            f0: 1.0 / cpw,
        },
        amplitude: 1.0,
        kind: SourceKind::Soft,
    })?;
    let tr = record(&mut sim, steps, &[p1, p2]);
    ensure(max_abs(&tr[1][steps - 50..]) < 1e-13, || {
        format!("A2 {cpw} cpw: record cut off the pulse")
    })?;
    let k_meas = wavenumber_from_probes(&tr[0], &tr[1], omega, s, len as f64, omega);
    Ok((k_meas, numerical_wavenumber_1d(omega, s)?, omega))
}

/// A2: measured phase velocity (as k) against the discrete relation at 10,
/// 20 and 40 cells per wavelength, S = 0.5. Measured: the worst relative
/// error. Guard: at 10 cpw the discrete and continuum k differ by more than
/// 1e-3, so the gate really separates grid physics from continuum physics.
pub fn a2() -> Result<Vec<CaseResult>> {
    let mut worst = 0.0_f64;
    let mut gap10 = 0.0;
    for cpw in [10.0, 20.0, 40.0] {
        let (k_meas, k_d, k_c) = a2_measure(cpw)?;
        worst = worst.max((k_meas / k_d - 1.0).abs());
        if cpw == 10.0 {
            gap10 = (k_d / k_c - 1.0).abs();
        }
    }
    ensure(gap10 > 1e-3, || {
        format!("A2: discrete vs continuum only {gap10:e} at 10 cpw")
    })?;
    Ok(vec![CaseResult::new(
        "A2",
        "1D phase velocity vs discrete relation",
        worst,
        Gate::AtMost(1e-4),
        "rel",
        format!("S = 0.5, 10, 20, 40 cpw; discrete vs continuum {gap10:.1e} at 10 cpw"),
    )])
}

/// A3: off a PEC wall the pulse comes back inverted with equal magnitude.
/// At S = 1 directly (delayed by the exact round trip); at S = 0.5 by the
/// image method (a wall at W equals an open mirror-symmetric line with a
/// negated image source at 2 W - a). Measured: the worse of the two errors
/// relative to the peak.
pub fn a3() -> Result<Vec<CaseResult>> {
    let (src, probe, wall) = (50, 60, 400);
    let round_trip = 2 * (wall - probe);
    let mut sim = Sim1d::new(wall + 1, 1.0, Boundary1d::Mur1, Boundary1d::Pec)?;
    sim.add_source(gaussian_source(src, 1.0))?;
    let p = &record(&mut sim, 2 * round_trip, &[probe])[0];
    let peak = max_abs(&p[..round_trip]);
    ensure(peak > 0.1, || "A3: pulse never reached the probe".into())?;
    let direct = (0..round_trip)
        .map(|n| (p[n + round_trip] + p[n]).abs())
        .fold(0.0_f64, f64::max)
        / peak;

    let (s, a, w, steps) = (0.5, 60, 150, 1200);
    let mut wall_sim = Sim1d::new(w + 1, s, Boundary1d::Mur1, Boundary1d::Pec)?;
    wall_sim.add_source(gaussian_source(a, 1.0))?;
    let mut image = Sim1d::new(2 * w + 1, s, Boundary1d::Mur1, Boundary1d::Mur1)?;
    image.add_source(gaussian_source(a, 1.0))?;
    image.add_source(gaussian_source(2 * w - a, -1.0))?;
    let (mut err, mut peak) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        wall_sim.step();
        image.step();
        for i in 0..=w {
            err = err.max((wall_sim.ez()[i] - image.ez()[i]).abs());
            peak = peak.max(wall_sim.ez()[i].abs());
        }
    }
    ensure(peak > 0.1, || "A3: image run has no pulse".into())?;
    let imaged = err / peak;
    Ok(vec![CaseResult::new(
        "A3",
        "1D PEC reflection inverted, equal magnitude",
        direct.max(imaged),
        Gate::AtMost(1e-10),
        "rel",
        format!("S = 1 direct {direct:.1e}, S = 0.5 image method {imaged:.1e}"),
    )])
}
