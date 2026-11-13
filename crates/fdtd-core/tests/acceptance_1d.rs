//! Phase 0 acceptance tests for the 1D reference solver. Public API only.
//! All quantities are normalized (dx = 1, c0 = 1, dt = S).

use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::source::{SourceKind, Waveform};

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
