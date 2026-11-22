//! A18: TEz against TMz under duality.
//!
//! In vacuum the TEz update on the swapped layout (`te.rs`) is the TMz update
//! with Ez -> Hz, Hx -> -Ex, Hy -> -Ey, half a step later. After n steps a
//! TMz kernel holds Ez^n and a TEz kernel Hz^(n - 1/2), and they are the
//! same numbers if the TEz source injects the same samples: its waveform is
//! read at (n + 1/2) S, so its t0 is moved S/2 earlier. The CPML profiles
//! sit at the same positions in both modes, and the TEz ring (Hz = 0) is
//! the dual of the TMz PEC ring, so the identity holds in a closed box with
//! its reflections and in an open domain behind a CPML.
//!
//! Measured: the largest |Hz - Ez| over every node, every 50 steps, over
//! the running peak of |Ez|, worst of the two runs. Gate 1e-10. Guard: the
//! same measure with the shift left out sees the half-step slip.

use super::signal::max_abs;
use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::error::Result;
use crate::material::Material;
use crate::scene::{GridSpec, Mode, Monitor, SCENE_VERSION, Scene, Source};
use crate::source::{SourceKind, Waveform};
use crate::te::SimTe;
use crate::tm::SimTm;

const S: f64 = 0.5;

fn line_source(pml: Option<PmlSpec>, t0: f64) -> Scene {
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: 121,
            ny: 101,
            dx_phys_m: 1e-3,
            courant: S,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![Source {
            i: 45,
            j: 52,
            waveform: Waveform::Ricker { t0, f_peak: 0.05 },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        }],
        monitors: vec![Monitor::Point { i: 90, j: 30 }],
        pml,
        periodic_y: false,
        plane_wave: None,
        paint: None,
        view: None,
        mode: Mode::Tm,
    }
}

/// Worst relative node difference over the run, and the probe difference,
/// with the TEz waveform moved `shift` earlier.
fn compare(pml: Option<PmlSpec>, steps: u64, shift: f64) -> Result<(f64, f64)> {
    let t0 = 40.0;
    let mut tm = SimTm::from_scene(&line_source(pml, t0))?;
    let mut te_scene = line_source(pml, t0 - shift);
    te_scene.mode = Mode::Te;
    let mut te = SimTe::from_scene(&te_scene)?;
    let (mut peak, mut worst) = (0.0_f64, 0.0_f64);
    for _ in 0..steps / 50 {
        tm.step_n(50);
        te.step_n(50);
        peak = peak.max(max_abs(tm.ez()));
        let d = tm
            .ez()
            .iter()
            .zip(te.hz())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        worst = worst.max(d / peak);
    }
    let (a, b) = (tm.probes()[0].history(), te.probes()[0].history());
    ensure(a.len() == b.len() && max_abs(a) > 0.0, || {
        "A18: the probe saw no wave".into()
    })?;
    let probe = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / max_abs(a);
    Ok((worst, probe))
}

pub fn a18() -> Result<Vec<CaseResult>> {
    // 1500 steps is 750 time units: several round trips of the closed box.
    let (closed, closed_probe) = compare(None, 1500, 0.5 * S)?;
    let (open, open_probe) = compare(Some(PmlSpec::recommended(16)), 1500, 0.5 * S)?;
    // Guard: without the half-step shift the same measure must see the slip.
    let (slipped, _) = compare(None, 300, 0.0)?;
    ensure(slipped > 1e-3, || {
        format!("A18: an unshifted source differs by only {slipped:e}")
    })?;
    Ok(vec![CaseResult::new(
        "A18",
        "TEz vs TMz under duality, vacuum line source",
        closed.max(open),
        Gate::AtMost(1e-10),
        "rel",
        format!(
            "121 x 101, 1500 steps; closed box {closed:.1e}, CPML 16 {open:.1e}; \
             probe {:.1e}; unshifted source {slipped:.1e}",
            closed_probe.max(open_probe)
        ),
    )])
}
