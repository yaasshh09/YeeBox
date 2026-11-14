//! Phase 2 acceptance for the CPML, through the public API only. All
//! quantities are normalized (dx = dy = 1, c0 = 1, dt = S). Reflection is
//! measured by the reference-domain method in `fdtd_core::reflection`.

mod common;

use fdtd_core::cpml::PmlSpec;
use fdtd_core::reflection::{ReflectionSetup, reflection_db};
use std::sync::OnceLock;

/// The standard sweep and its reference records, computed once per test
/// binary. The reference is the expensive run (a PEC box about 700 x 600).
fn reference() -> &'static (ReflectionSetup, Vec<Vec<f64>>) {
    static REF: OnceLock<(ReflectionSetup, Vec<Vec<f64>>)> = OnceLock::new();
    REF.get_or_init(|| {
        let setup = ReflectionSetup::standard();
        let records = setup.reference().unwrap();
        (setup, records)
    })
}

/// Reflection in dB per probe for one layer, printed with each angle.
fn measure(spec: PmlSpec) -> (f64, Vec<f64>) {
    let (setup, reference) = reference();
    let db = reflection_db(&setup.test_domain(Some(spec)).unwrap(), reference);
    for (k, v) in db.iter().enumerate() {
        eprintln!(
            "  {:>2} cells, {:5.1} deg: {v:8.2} dB",
            spec.thickness,
            setup.angle_deg(k)
        );
    }
    (setup.worst_db(&db, 70.0), db)
}

/// A9: a 10-cell layer reflects at most -60 dB from normal incidence to 70
/// degrees, the interface corner included.
#[test]
fn a9_ten_cell_layer() {
    let (worst, _) = measure(PmlSpec::recommended(10));
    eprintln!("A9: worst 0 to 70 deg, 10 cells: {worst:.2} dB (gate -60)");
    assert!(worst <= -60.0, "{worst} dB");
}

/// A10: the tuned 16-cell layer. The spec asks for the best value to be
/// recorded; it also clears the -80 dB target, so that is asserted too.
#[test]
fn a10_tuned_sixteen_cell_layer() {
    let (worst, _) = measure(PmlSpec::recommended(16));
    eprintln!("A10: worst 0 to 70 deg, 16 cells: {worst:.2} dB (target -80)");
    assert!(worst <= -80.0, "{worst} dB");
}

/// Late-time stability: a 50 x 50 vacuum box inside the recommended 10-cell
/// layer, a Ricker pulse that is over by step 200, then 100000 steps. The
/// field energy `sum(eps Ez^2) + sum(Hx^2 + Hy^2)` is checked by its maximum
/// over each 2000-step window: once the source is done those maxima never
/// rise, and the last one sits at the floor.
///
/// Windowed, because inside a CPML the field energy alone is not monotone:
/// psi holds part of the energy and hands it back. With alpha > 0 the
/// leftover low-frequency field (wavelengths near the box size) beats with
/// that memory on a period of more than 500 steps, so 500-step maxima rise
/// now and then by up to 1.5x at the 1e-14 level. With 2000-step windows the
/// maxima fall monotonically for every layer I tried (alpha 0, 0.15, 0.5,
/// kappa 1), which is what separates a beat from growth.
#[test]
fn late_time_decay_no_growth() {
    use fdtd_core::material::Material;
    use fdtd_core::scene::{GridSpec, SCENE_VERSION, Scene, Source};
    use fdtd_core::source::{SourceKind, Waveform};
    use fdtd_core::tm::SimTm;

    let scene = Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: 50,
            ny: 50,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![Source {
            i: 22,
            j: 27,
            waveform: Waveform::Ricker {
                t0: 40.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        }],
        monitors: vec![],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: false,
    };
    let mut sim = SimTm::from_scene(&scene).unwrap();
    let energy = |sim: &SimTm| -> f64 {
        let e: f64 = sim.ez().iter().map(|v| v * v).sum();
        let h: f64 = sim.hx().iter().chain(sim.hy()).map(|v| v * v).sum();
        e + h
    };
    let (window, windows) = (2000, 50);
    let mut maxima = Vec::with_capacity(windows);
    for _ in 0..windows {
        let mut m = 0.0_f64;
        for _ in 0..window {
            sim.step();
            m = m.max(energy(&sim));
        }
        assert!(
            m.is_finite(),
            "energy not finite at step {}",
            sim.time_step()
        );
        maxima.push(m);
    }
    let peak = maxima[0];
    let last = maxima[windows - 1];
    eprintln!(
        "late time: peak {peak:.3e}, after 4000 steps {:.3e}, after 20000 {:.3e}, final {last:.3e} ({:.1e} of peak)",
        maxima[1],
        maxima[9],
        last / peak
    );
    // The source is done by step 200, inside the first window.
    for w in 1..windows {
        assert!(
            maxima[w] <= maxima[w - 1],
            "window {w}: {:e} > {:e}",
            maxima[w],
            maxima[w - 1]
        );
    }
    assert!(last <= 1e-20 * peak, "final {:e} of peak", last / peak);
}

/// Complex quotient a / b of (re, im) pairs.
fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// A8: a soft CW line source in the open domain against the Hankel
/// solution, from 2 to 5 wavelengths, along the x axis and the diagonal.
///
/// 253 x 253 nodes with the recommended 16-cell layer, source on the centre
/// node, f0 = 0.05 (20 cells per wavelength, 40 steps per period at S = 0.5),
/// a 3-period ramp and 3000 steps. The phasor at each probe is the exact DFT
/// over the last 10 whole periods. Golden values come from SciPy
/// (`validation/gen/line_source_hankel.py`): `H0^(2)(k r) / H0^(2)(k r0)` with
/// the numerical k of each direction. The gate is `|F/F0 / (H/H0) - 1| <= 0.02`,
/// which bounds the amplitude to 2 percent and the phase to 0.02 rad.
#[test]
fn a8_line_source_matches_hankel() {
    use fdtd_core::dispersion::numerical_wavenumber;
    use fdtd_core::material::Material;
    use fdtd_core::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
    use fdtd_core::source::{SourceKind, Waveform};
    use fdtd_core::tm::SimTm;
    use std::f64::consts::PI;

    const GOLDEN: &str = include_str!("../../../validation/golden/line_source_hankel.csv");
    let (s, f0) = (0.5, 0.05);
    let omega = 2.0 * PI * f0;
    assert!(
        GOLDEN.contains("# S = 0.5, f0 = 0.05,"),
        "golden data is for another setup"
    );

    struct Row {
        diagonal: bool,
        di: usize,
        dj: usize,
        r: f64,
        k: f64,
        h: (f64, f64),
        h_c: (f64, f64),
    }
    let rows: Vec<Row> = GOLDEN
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("direction"))
        .map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            let f = |k: usize| c[k].parse::<f64>().unwrap();
            Row {
                diagonal: c[0] == "diagonal",
                di: c[1].parse().unwrap(),
                dj: c[2].parse().unwrap(),
                r: f(3),
                k: f(4),
                h: (f(5), f(6)),
                h_c: (f(7), f(8)),
            }
        })
        .collect();
    assert_eq!(rows.len(), 27);

    // The closed-form k in the golden file equals this crate's dispersion solver.
    for row in &rows {
        let theta = if row.diagonal { PI / 4.0 } else { 0.0 };
        let k = numerical_wavenumber(omega, theta, s).unwrap();
        assert!((k - row.k).abs() < 1e-12 * k, "k {k} vs golden {}", row.k);
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
    };
    let mut sim = SimTm::from_scene(&scene).unwrap();
    let (steps, period) = (3000, 40);
    assert_eq!((1.0 / (f0 * s)) as usize, period);
    sim.step_n(steps as u64);

    let phasors: Vec<(f64, f64)> = sim
        .probes()
        .iter()
        .map(|p| common::dtft(&p.history()[steps + 1 - 10 * period..], None, omega, s))
        .collect();
    let (mut worst, mut worst_amp, mut worst_phase, mut worst_c) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for diagonal in [false, true] {
        let idx: Vec<usize> = (0..rows.len())
            .filter(|&k| rows[k].diagonal == diagonal)
            .collect();
        let f_ref = phasors[idx[0]];
        for &k in &idx {
            let ratio = cdiv(cdiv(phasors[k], f_ref), rows[k].h);
            let err = (ratio.0 - 1.0).hypot(ratio.1);
            let ratio_c = cdiv(cdiv(phasors[k], f_ref), rows[k].h_c);
            let err_c = (ratio_c.0 - 1.0).hypot(ratio_c.1);
            eprintln!(
                "  {} r = {:6.2}: |ratio| - 1 = {:+.2e}, phase {:+.2e} rad, err {:.2e} (continuum k: {:.2e})",
                if diagonal { "diag" } else { "axis" },
                rows[k].r,
                ratio.0.hypot(ratio.1) - 1.0,
                ratio.1.atan2(ratio.0),
                err,
                err_c
            );
            worst = worst.max(err);
            worst_amp = worst_amp.max((ratio.0.hypot(ratio.1) - 1.0).abs());
            worst_phase = worst_phase.max(ratio.1.atan2(ratio.0).abs());
            worst_c = worst_c.max(err_c);
        }
    }
    eprintln!(
        "A8: worst {worst:.2e} (amplitude {worst_amp:.2e}, phase {worst_phase:.2e} rad), gate 0.02; with continuum k {worst_c:.2e}"
    );
    assert!(worst <= 0.02, "A8 error {worst}");
}

/// Periodic y with a layer: the CPML sits on the x faces only. A y-uniform
/// line source must stay exactly y-uniform (a y layer near the seam would
/// absorb the rows next to it), and the x layer's normal-incidence
/// reflection must meet the A9 gate. Reflection is the reference-domain
/// method: the same grid extended 400 cells to the right, so the right
/// face's echo cannot reach the probe within the run, and the left face
/// echoes the same way in both and cancels in the difference.
#[test]
fn periodic_y_layer_on_x_faces_only() {
    use fdtd_core::material::Material;
    use fdtd_core::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
    use fdtd_core::source::{SourceKind, Waveform};
    use fdtd_core::tm::SimTm;

    let ny = 6;
    let line = |nx: usize| Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: (0..ny)
            .map(|j| Source {
                i: 40,
                j,
                waveform: Waveform::Ricker {
                    t0: 40.0,
                    f_peak: 0.05,
                },
                amplitude: 1.0,
                kind: SourceKind::Soft,
            })
            .collect(),
        monitors: vec![Monitor::Point { i: 70, j: 0 }],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: true,
    };
    let steps = 700;
    let run = |nx: usize| {
        let mut sim = SimTm::from_scene(&line(nx)).unwrap();
        for _ in 0..steps {
            sim.step();
            for j in 1..ny {
                let (r0, rj) = (&sim.ez()[..nx], &sim.ez()[j * nx..(j + 1) * nx]);
                assert_eq!(
                    r0,
                    rj,
                    "nx {nx}: row {j} differs at step {}",
                    sim.time_step()
                );
            }
        }
        sim.probes()[0].history().to_vec()
    };
    let test = run(130);
    let reference = run(530);
    let peak = common::max_abs(&reference);
    let diff: Vec<f64> = test.iter().zip(&reference).map(|(a, b)| a - b).collect();
    let db = 20.0 * (common::max_abs(&diff) / peak).log10();
    eprintln!("periodic y, x layer, normal incidence, 10 cells: {db:.2} dB");
    assert!(db <= -60.0, "{db} dB");
}

/// Periodic y with an x layer is translation invariant along y: moving a
/// point source by m rows moves every field by m rows, bit for bit, because
/// with no y layer every row runs exactly the same arithmetic. A y-uniform
/// drive cannot catch a stray y layer (every d/dy is zero there); this can.
#[test]
fn periodic_y_with_layer_is_translation_invariant() {
    use fdtd_core::material::Material;
    use fdtd_core::scene::{GridSpec, SCENE_VERSION, Scene, Source};
    use fdtd_core::source::{SourceKind, Waveform};
    use fdtd_core::tm::SimTm;

    let (nx, ny) = (60, 24);
    let at = |j: usize| Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![Source {
            i: 30,
            j,
            waveform: Waveform::Ricker {
                t0: 20.0,
                f_peak: 0.08,
            },
            amplitude: 1.0,
            kind: SourceKind::Soft,
        }],
        monitors: vec![],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: true,
    };
    let run = |j: usize| {
        let mut sim = SimTm::from_scene(&at(j)).unwrap();
        sim.step_n(150);
        (sim.ez().to_vec(), sim.hx().to_vec(), sim.hy().to_vec())
    };
    let base = run(12);
    assert!(
        common::max_abs(&base.0) > 1e-3,
        "the pulse must be on the grid"
    );
    for m in [1, 5, 12, 23] {
        let shifted = run((12 + m) % ny);
        let roll = |v: &[f64]| -> Vec<f64> {
            (0..nx * ny)
                .map(|k| v[((k / nx + ny - m) % ny) * nx + k % nx])
                .collect()
        };
        assert_eq!(roll(&base.0), shifted.0, "ez, shift {m}");
        assert_eq!(roll(&base.1), shifted.1, "hx, shift {m}");
        assert_eq!(roll(&base.2), shifted.2, "hy, shift {m}");
    }
}
