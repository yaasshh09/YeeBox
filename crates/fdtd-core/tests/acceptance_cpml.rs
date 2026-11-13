//! Phase 2 acceptance for the CPML, through the public API only. All
//! quantities are normalized (dx = dy = 1, c0 = 1, dt = S). Reflection is
//! measured by the reference-domain method in `fdtd_core::reflection`.

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
