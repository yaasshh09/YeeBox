//! Phase 0 acceptance tests for the 1D reference solver. A1 to A3 are
//! measured in `fdtd_core::cases::tier1d`; the invariants are here. Public
//! API only, normalized units (dx = 1, c0 = 1, dt = S).

mod common;

use common::{assert_cases, max_abs};
use fdtd_core::cases::tier1d;
use fdtd_core::fdtd1d::{Boundary1d, Sim1d, Source1d};
use fdtd_core::source::{SourceKind, Waveform};

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

#[test]
fn a1_magic_time_step_1000_cells() {
    assert_cases(tier1d::a1());
}

#[test]
fn a2_phase_velocity_10_20_40_cpw() {
    assert_cases(tier1d::a2());
}

#[test]
fn a3_pec_reflection_inverted_equal() {
    assert_cases(tier1d::a3());
}

/// With no source the fields stay exactly zero: nothing in the update can
/// create energy from nothing, for any boundary combination.
#[test]
fn quiescence_no_source_exact_zero() {
    let ends = [Boundary1d::Pec, Boundary1d::Mur1];
    for left in ends {
        for right in ends {
            let mut sim = Sim1d::new(300, 0.5, left, right).unwrap();
            sim.step_n(1000);
            assert!(sim.ez().iter().chain(sim.hy()).all(|&v| v == 0.0));
        }
    }
}

/// Doubling every source amplitude doubles every field value bit for bit,
/// since scaling by 2 is exact in binary floating point and the scheme is linear.
#[test]
fn linearity_double_amplitude_bit_exact() {
    let run = |scale: f64| {
        let mut sim = Sim1d::new(400, 0.5, Boundary1d::Mur1, Boundary1d::Pec).unwrap();
        let mut soft = gaussian_source(100);
        soft.amplitude = 0.7 * scale;
        let hard = Source1d {
            node: 250,
            waveform: Waveform::ModulatedGaussian {
                t0: 90.0,
                tau: 15.0,
                f0: 0.05,
            },
            amplitude: -1.3 * scale,
            kind: SourceKind::Hard,
        };
        sim.add_source(soft).unwrap();
        sim.add_source(hard).unwrap();
        sim.step_n(900);
        sim
    };
    let (one, two) = (run(1.0), run(2.0));
    assert!(max_abs(one.ez()) > 1e-3);
    for (a, b) in one
        .ez()
        .iter()
        .zip(two.ez())
        .chain(one.hy().iter().zip(two.hy()))
    {
        assert_eq!(2.0 * a, *b);
    }
}
