//! Tier 3 invariants for the 2D TMz kernel, through the public API only.
//! Each one holds for any correct linear Yee solver, whatever the scene.

use fdtd_core::cpml::PmlSpec;
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
        pml: None,
        periodic_y: false,
        plane_wave: None,
        paint: None,
        view: None,
        mode: fdtd_core::scene::Mode::Tm,
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

fn layer(thickness: usize) -> PmlSpec {
    PmlSpec {
        thickness,
        m: 3.0,
        sigma_scale: 1.0,
        kappa_max: 5.0,
        alpha_max: 0.05,
    }
}

/// 61 x 51 with a 10-cell CPML, symmetric about x = 30 and y = 25: two
/// dielectric bars, two PEC disks, a lossy disk at the centre, and a source
/// one node off centre along x so both mirrors are needed.
fn open_symmetric() -> Scene {
    let mut scene = cluttered(61, 51);
    scene.pml = Some(layer(10));
    let bar = |y0, y1| Shape::Rect {
        x0: 22.0,
        y0,
        x1: 38.0,
        y1,
        material: 1,
    };
    let disk = |cx, cy, r, material| Shape::Circle {
        cx,
        cy,
        r,
        material,
    };
    scene.geometry = vec![
        bar(13.0, 15.0),
        bar(35.0, 37.0),
        disk(15.0, 25.0, 2.5, 3),
        disk(45.0, 25.0, 2.5, 3),
        disk(30.0, 25.0, 1.5, 2),
    ];
    scene.sources = vec![ricker_at(30, 25, 1.0)];
    scene
}

/// With the CPML on, the field stays mirror symmetric about both axes. A
/// half-cell slip in one profile, or a low side that does not match the high
/// side, breaks this long before it shows in a reflection number. Run long
/// enough for every face and corner to absorb a wave.
#[test]
fn mirror_symmetry_with_cpml() {
    let scene = open_symmetric();
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let mut sim = SimTm::from_scene(&scene).unwrap();
    let mut worst = 0.0_f64;
    let mut peak = 0.0_f64;
    for _ in 0..4 {
        sim.step_n(150);
        let (ez, hx, hy) = (sim.ez(), sim.hx(), sim.hy());
        peak = peak.max(max_abs(ez));
        let at = |i: usize, j: usize| j * nx + i;
        for j in 0..ny {
            for i in 0..nx {
                let (mx, my) = (nx - 1 - i, ny - 1 - j);
                worst = worst.max((ez[at(i, j)] - ez[at(mx, j)]).abs());
                worst = worst.max((ez[at(i, j)] - ez[at(i, my)]).abs());
                if j < ny - 1 {
                    // Hx at (i, j + 1/2): even in x, odd in y (mirror row ny - 2 - j).
                    worst = worst.max((hx[at(i, j)] - hx[at(mx, j)]).abs());
                    worst = worst.max((hx[at(i, j)] + hx[at(i, ny - 2 - j)]).abs());
                }
                if i < nx - 1 {
                    // Hy at (i + 1/2, j): odd in x (mirror column nx - 2 - i), even in y.
                    worst = worst.max((hy[at(i, j)] + hy[at(nx - 2 - i, j)]).abs());
                    worst = worst.max((hy[at(i, j)] - hy[at(i, my)]).abs());
                }
            }
        }
    }
    eprintln!(
        "mirror symmetry with cpml: worst {:.2e} of peak",
        worst / peak
    );
    assert!(peak > 1e-3);
    assert!(
        worst <= 1e-14 * peak,
        "relative asymmetry {:e}",
        worst / peak
    );
}

/// Walls and padding stay exactly zero with the layer on.
#[test]
fn padding_and_walls_zero_with_cpml() {
    let scene = open_symmetric();
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let mut sim = SimTm::from_scene(&scene).unwrap();
    for _ in 0..300 {
        sim.step();
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
                    assert_eq!(sim.ez()[k], 0.0, "wall ez at ({i}, {j})");
                }
                if j == ny - 1 {
                    assert_eq!(sim.hx()[k], 0.0, "hx padding at ({i}, {j})");
                }
                if i == nx - 1 {
                    assert_eq!(sim.hy()[k], 0.0, "hy padding at ({i}, {j})");
                }
            }
        }
    }
    assert!(max_abs(sim.ez()) > 0.0);
}

/// Quiescence and exact linearity hold with the layer on: the psi recursions
/// are linear and start at zero.
#[test]
fn quiescence_and_linearity_with_cpml() {
    let mut scene = open_symmetric();
    scene.sources.clear();
    let mut sim = SimTm::from_scene(&scene).unwrap();
    sim.step_n(300);
    for f in [sim.ez(), sim.hx(), sim.hy()] {
        assert!(f.iter().all(|&v| v == 0.0));
    }

    let run = |amp: f64| {
        let mut scene = open_symmetric();
        scene.sources = vec![ricker_at(18, 14, amp)];
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
    assert!(max_abs(one.ez()) > 1e-6);
}

/// Reciprocity with a TF/SF plane wave and spectral monitors present. The
/// plane wave lights the cluttered box from the left; by linearity, the run
/// with the plane wave and a source at A, minus the run with the plane wave
/// alone, is A's own response. That response, recorded at B, equals B's
/// response recorded at A. DFT and flux monitors sit in every run, and they
/// are passive: the plane-wave-only fields are bit for bit the same with
/// and without them.
#[test]
fn reciprocity_with_tfsf_and_monitors() {
    use fdtd_core::monitor::Normal;
    use fdtd_core::tfsf::{Direction, PlaneWave};
    let (a, b) = ((12, 15), (47, 38));
    let base = |with_monitors: bool| {
        let mut scene = cluttered(60, 50);
        scene.plane_wave = Some(PlaneWave {
            x0: 8,
            y0: 3,
            x1: 52,
            y1: 47,
            direction: Direction::PlusX,
            waveform: Waveform::Ricker {
                t0: 30.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
        });
        if with_monitors {
            scene.monitors = vec![
                Monitor::Dft {
                    i: 25,
                    j: 30,
                    freqs: vec![0.03, 0.05],
                },
                Monitor::Flux {
                    normal: Normal::X,
                    at: 45,
                    from: 4,
                    to: 46,
                    freqs: vec![0.05],
                },
                Monitor::Flux {
                    normal: Normal::Y,
                    at: 2,
                    from: 1,
                    to: 58,
                    freqs: vec![0.05],
                },
            ];
        }
        scene
    };
    let probe_at = |scene: &Scene, src: Option<(usize, usize)>, to: (usize, usize)| {
        let mut scene = scene.clone();
        scene.sources = src.map(|p| ricker_at(p.0, p.1, 1.0)).into_iter().collect();
        scene.monitors.push(Monitor::Point { i: to.0, j: to.1 });
        let mut sim = SimTm::from_scene(&scene).unwrap();
        sim.step_n(1500);
        (sim.probes()[0].history().to_vec(), sim.ez().to_vec())
    };
    let scene = base(true);
    let response = |from, to| {
        let (with, _) = probe_at(&scene, Some(from), to);
        let (alone, _) = probe_at(&scene, None, to);
        with.iter()
            .zip(&alone)
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>()
    };
    let (ab, ba) = (response(a, b), response(b, a));
    let peak = max_abs(&ab);
    let diff = ab
        .iter()
        .zip(&ba)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    eprintln!(
        "reciprocity with tf/sf: diff {:.2e} of peak {peak:.3e}",
        diff / peak
    );
    assert!(peak > 1e-3, "signal must reach the probe");
    assert!(diff < 1e-9 * peak, "relative diff {:e}", diff / peak);

    let (_, with) = probe_at(&base(true), None, a);
    let (_, without) = probe_at(&base(false), None, a);
    assert_eq!(with, without, "monitors must not touch the fields");
    assert!(max_abs(&with) > 1e-3, "the plane wave must light the box");
}
