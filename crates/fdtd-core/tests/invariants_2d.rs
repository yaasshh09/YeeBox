//! Tier 3 invariants for the 2D TMz kernel, through the public API only.
//! Each one holds for any correct linear Yee solver, whatever the scene.

use fdtd_core::material::Material;
use fdtd_core::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Shape, Source};
use fdtd_core::source::{SourceKind, Waveform};
use fdtd_core::tm::SimTm;

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn ricker_at(i: usize, j: usize, amplitude: f64) -> Source {
    Source {
        i,
        j,
        waveform: Waveform::Ricker {
            t0: 25.0,
            f_peak: 0.06,
        },
        amplitude,
        kind: SourceKind::Soft,
    }
}

/// Vacuum background with a dielectric block, a lossy block, and a PEC disk.
fn cluttered(nx: usize, ny: usize) -> Scene {
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![
            Material::VACUUM,
            Material::Dielectric {
                eps_inf: 3.0,
                sigma: 0.0,
            },
            Material::Dielectric {
                eps_inf: 2.0,
                sigma: 0.05,
            },
            Material::Pec,
        ],
        geometry: vec![
            Shape::Rect {
                x0: 20.0,
                y0: 5.0,
                x1: 30.0,
                y1: 25.0,
                material: 1,
            },
            Shape::Rect {
                x0: 33.0,
                y0: 20.0,
                x1: 40.0,
                y1: 44.0,
                material: 2,
            },
            Shape::Circle {
                cx: 15.0,
                cy: 35.0,
                r: 4.0,
                material: 3,
            },
        ],
        sources: vec![],
        monitors: vec![],
    }
}

/// Swapping a soft source and a probe, both on vacuum nodes, gives the same
/// recorded signal through dielectric, lossy, and PEC clutter.
#[test]
fn reciprocity_swap_source_and_probe() {
    let (a, b) = ((12, 15), (47, 38));
    let run = |from: (usize, usize), to: (usize, usize)| {
        let mut scene = cluttered(60, 50);
        scene.sources = vec![ricker_at(from.0, from.1, 1.0)];
        scene.monitors = vec![Monitor::Point { i: to.0, j: to.1 }];
        let mut sim = SimTm::from_scene(&scene).unwrap();
        sim.step_n(1500);
        sim.probes()[0].history().to_vec()
    };
    let ab = run(a, b);
    let ba = run(b, a);
    let peak = max_abs(&ab);
    let diff = ab
        .iter()
        .zip(&ba)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    eprintln!("reciprocity: diff {:.2e} of peak {peak:.3e}", diff / peak);
    assert!(peak > 1e-3, "signal must reach the probe");
    assert!(diff < 1e-9 * peak, "relative diff {:e}", diff / peak);
}

/// A scene symmetric about x = 30 with a source on that line gives
/// Ez(i, j) = Ez(60 - i, j), Hx even and Hy odd, to round-off.
#[test]
fn mirror_symmetry() {
    let (nx, ny) = (61, 45);
    let mut scene = cluttered(nx, ny);
    scene.geometry = vec![
        Shape::Rect {
            x0: 22.0,
            y0: 28.0,
            x1: 38.0,
            y1: 33.0,
            material: 1,
        },
        Shape::Circle {
            cx: 30.0,
            cy: 12.0,
            r: 5.5,
            material: 3,
        },
        Shape::Polygon {
            points: vec![[10.5, 38.0], [49.5, 38.0], [30.0, 42.5]],
            material: 2,
        },
    ];
    scene.sources = vec![ricker_at(30, 22, 1.0)];
    let mut sim = SimTm::from_scene(&scene).unwrap();
    sim.step_n(700);

    let (ez, hx, hy) = (sim.ez(), sim.hx(), sim.hy());
    let peak = max_abs(ez);
    let mut worst = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let (k, m) = (j * nx + i, j * nx + (nx - 1 - i));
            worst = worst.max((ez[k] - ez[m]).abs());
            worst = worst.max((hx[k] - hx[m]).abs());
            if i < nx - 1 {
                // Hy at i + 1/2 mirrors to Hy at nx - 2 - i, with a sign flip.
                worst = worst.max((hy[k] + hy[j * nx + (nx - 2 - i)]).abs());
            }
        }
    }
    eprintln!("mirror symmetry: worst {:.2e} of peak", worst / peak);
    assert!(peak > 1e-3);
    assert!(
        worst <= 1e-14 * peak,
        "relative asymmetry {:e}",
        worst / peak
    );
}

/// No source: every field stays exactly zero.
#[test]
fn quiescence_exact_zero() {
    let mut sim = SimTm::from_scene(&cluttered(40, 50)).unwrap();
    sim.step_n(300);
    for f in [sim.ez(), sim.hx(), sim.hy()] {
        assert!(f.iter().all(|&v| v == 0.0));
    }
}

/// Doubling the source amplitude doubles every field value exactly (scaling
/// by 2 is exact in binary floating point).
#[test]
fn linearity_double_amplitude_bit_exact() {
    let run = |amp: f64| {
        let mut scene = cluttered(50, 50);
        scene.sources = vec![ricker_at(10, 10, amp)];
        let mut sim = SimTm::from_scene(&scene).unwrap();
        sim.step_n(400);
        sim
    };
    let (one, two) = (run(1.0), run(2.0));
    for (a, b) in [
        (one.ez(), two.ez()),
        (one.hx(), two.hx()),
        (one.hy(), two.hy()),
    ] {
        assert!(a.iter().zip(b).all(|(x, y)| 2.0 * x == *y));
    }
    assert!(max_abs(one.ez()) > 1e-3);
}
