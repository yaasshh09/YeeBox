//! A8 to A10: the open domain. A line source against the Hankel solution,
//! and the CPML reflection sweep.

use super::signal::{cdiv, dtft};
use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::dispersion::numerical_wavenumber;
use crate::error::Result;
use crate::material::Material;
use crate::reflection::{ReflectionSetup, reflection_db};
use crate::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
use crate::source::{SourceKind, Waveform};
use crate::tm::SimTm;
use std::f64::consts::PI;

const HANKEL: &str = include_str!("../../../../validation/golden/line_source_hankel.csv");

struct HankelRow {
    diagonal: bool,
    di: usize,
    dj: usize,
    k: f64,
    h: (f64, f64),
    h_c: (f64, f64),
}

fn hankel_rows() -> Result<Vec<HankelRow>> {
    ensure(HANKEL.contains("# S = 0.5, f0 = 0.05,"), || {
        "A8: golden data is for another setup".into()
    })?;
    let rows: Vec<HankelRow> = HANKEL
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("direction"))
        .filter_map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            let f = |k: usize| c.get(k)?.parse::<f64>().ok();
            Some(HankelRow {
                diagonal: c.first()? == &"diagonal",
                di: c.get(1)?.parse().ok()?,
                dj: c.get(2)?.parse().ok()?,
                k: f(4)?,
                h: (f(5)?, f(6)?),
                h_c: (f(7)?, f(8)?),
            })
        })
        .collect();
    ensure(rows.len() == 27, || {
        format!("A8: {} golden rows", rows.len())
    })?;
    Ok(rows)
}

/// A8: a soft CW line source in the open domain against the Hankel
/// solution, 2 to 5 wavelengths out, along the x axis and the diagonal.
/// 253 x 253 nodes, the 16-cell layer, f0 = 0.05 (20 cells per wavelength,
/// 40 steps per period at S = 0.5), a 3-period ramp and 3000 steps; the
/// phasor at each probe is the exact DFT over the last 10 whole periods.
/// Golden: `H0^(2)(k r) / H0^(2)(k r0)` with the numerical k of each
/// direction (SciPy). Measured: the worst `|F/F0 / (H/H0) - 1|`, which
/// bounds the amplitude to the gate and the phase to the gate in radians.
pub fn a8() -> Result<Vec<CaseResult>> {
    let (s, f0) = (0.5, 0.05);
    let omega = 2.0 * PI * f0;
    let rows = hankel_rows()?;
    for row in &rows {
        let theta = if row.diagonal { PI / 4.0 } else { 0.0 };
        let k = numerical_wavenumber(omega, theta, s)?;
        ensure((k - row.k).abs() < 1e-12 * k, || {
            format!("A8: k {k} vs golden {}", row.k)
        })?;
    }
    let (n, c) = (253, 126);
    let scene = Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: n,
            ny: n,
            dx_phys_m: 1e-3,
            courant: s,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![Source {
            i: c,
            j: c,
            waveform: Waveform::RampedCw {
                f0,
                ramp_periods: 3.0,
                phase: 0.0,
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        }],
        monitors: rows
            .iter()
            .map(|r| Monitor::Point {
                i: c + r.di,
                j: c + r.dj,
            })
            .collect(),
        pml: Some(PmlSpec::recommended(16)),
        periodic_y: false,
        plane_wave: None,
        paint: None,
        view: None,
    };
    let mut sim = SimTm::from_scene(&scene)?;
    let (steps, period) = (3000, 40);
    sim.step_n(steps as u64);
    let phasors: Vec<(f64, f64)> = sim
        .probes()
        .iter()
        .map(|p| dtft(&p.history()[steps + 1 - 10 * period..], None, omega, s))
        .collect();
    let (mut worst, mut worst_c) = (0.0_f64, 0.0_f64);
    for diagonal in [false, true] {
        let idx: Vec<usize> = (0..rows.len())
            .filter(|&k| rows[k].diagonal == diagonal)
            .collect();
        let f_ref = phasors[idx[0]];
        for &k in &idx {
            let ratio = cdiv(cdiv(phasors[k], f_ref), rows[k].h);
            worst = worst.max((ratio.0 - 1.0).hypot(ratio.1));
            let ratio_c = cdiv(cdiv(phasors[k], f_ref), rows[k].h_c);
            worst_c = worst_c.max((ratio_c.0 - 1.0).hypot(ratio_c.1));
        }
    }
    Ok(vec![CaseResult::new(
        "A8",
        "line source vs Hankel, 2 to 5 wavelengths",
        worst,
        Gate::AtMost(0.02),
        "rel",
        format!("axis and diagonal, f0 = 0.05, 16-cell layer; with the continuum k {worst_c:.1e}"),
    )])
}

/// A9 and A10: CPML reflection by the reference-domain method
/// (`reflection.rs`), from normal incidence to 70 degrees, the interface
/// corner included. A9: the recommended 10-cell layer, gate -60 dB. A10:
/// the tuned 16-cell layer; the spec asks for the number to be recorded
/// against a -80 dB target, and it is gated at the target too. Both share
/// one reference run (a PEC box about 700 x 600).
pub fn a9_a10() -> Result<Vec<CaseResult>> {
    let setup = ReflectionSetup::standard();
    let reference = setup.reference()?;
    let row = |id, name, d: usize, gate| -> Result<CaseResult> {
        let db = reflection_db(
            &setup.test_domain(Some(PmlSpec::recommended(d)))?,
            &reference,
        );
        let worst = setup.worst_db(&db, 70.0);
        let grazing = setup.worst_db(&db, 80.0);
        Ok(CaseResult::new(
            id,
            name,
            worst,
            Gate::AtMost(gate),
            "dB",
            format!("{d} cells, 0 to 70 deg, worst with 80 deg {grazing:.1} dB"),
        ))
    };
    Ok(vec![
        row("A9", "CPML reflection, 10 cells", 10, -60.0)?,
        row("A10", "CPML reflection, tuned 16 cells", 16, -80.0)?,
    ])
}
