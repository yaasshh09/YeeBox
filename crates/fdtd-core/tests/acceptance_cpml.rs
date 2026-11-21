//! Phase 2 acceptance for the CPML, through the public API only. All
//! quantities are normalized (dx = dy = 1, c0 = 1, dt = S). Reflection is
//! measured by the reference-domain method in `fdtd_core::reflection`;
//! A8 to A10 themselves live in `fdtd_core::cases::boundary`.

mod common;

use common::assert_cases;
use fdtd_core::cases::boundary;
use fdtd_core::cpml::PmlSpec;

/// A9 (10-cell layer, gate -60 dB) and A10 (tuned 16-cell layer, -80 dB
/// target), on one shared reference run. See `cases::boundary::a9_a10`.
#[test]
fn a9_a10_cpml_reflection() {
    assert_cases(boundary::a9_a10());
}

/// A8: the line source against the Hankel solution. See `cases::boundary::a8`.
#[test]
fn a8_line_source_matches_hankel() {
    assert_cases(boundary::a8());
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
        plane_wave: None,
        paint: None,
        view: None,
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
        plane_wave: None,
        paint: None,
        view: None,
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
        plane_wave: None,
        paint: None,
        view: None,
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
