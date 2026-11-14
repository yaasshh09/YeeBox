//! Phase 3 acceptance for the TF/SF plane wave source, through the public
//! API only. Normalized units (dx = dy = 1, c0 = 1, dt = S).

mod common;

use fdtd_core::cpml::PmlSpec;
use fdtd_core::material::Material;
use fdtd_core::scene::{GridSpec, SCENE_VERSION, Scene};
use fdtd_core::source::Waveform;
use fdtd_core::tfsf::{Direction, PlaneWave};
use fdtd_core::tm::SimTm;

const RICKER: Waveform = Waveform::Ricker {
    t0: 40.0,
    f_peak: 0.05,
};

/// An empty `n x n` vacuum domain with a 10-cell layer and a plane wave
/// over the box `[lo, hi]^2` (or every row, with periodic y).
fn empty_domain(n: usize, lo: usize, hi: usize, dir: Direction, periodic: bool) -> Scene {
    let (y0, y1) = if periodic { (0, n - 1) } else { (lo, hi) };
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: n,
            ny: n,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![],
        monitors: vec![],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: periodic,
        plane_wave: Some(PlaneWave {
            x0: lo,
            y0,
            x1: hi,
            y1,
            direction: dir,
            waveform: RICKER,
            amplitude: 1.0,
        }),
    }
}

/// Runs the scene and returns (max |Ez| outside the box over the run, max
/// |Ez| inside it). The pulse enters, crosses the box, and leaves it.
fn leakage(scene: &Scene, steps: usize) -> (f64, f64) {
    let pw = scene.plane_wave.unwrap();
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let inside = |i: usize, j: usize| (pw.x0..=pw.x1).contains(&i) && (pw.y0..=pw.y1).contains(&j);
    let mut sim = SimTm::from_scene(scene).unwrap();
    let (mut sf, mut tf) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        sim.step();
        for j in 0..ny {
            for i in 0..nx {
                let v = sim.ez()[j * nx + i].abs();
                if inside(i, j) {
                    tf = tf.max(v);
                } else {
                    sf = sf.max(v);
                }
            }
        }
    }
    (sf, tf)
}

/// A15: in an empty domain the scattered-field region stays at or below
/// -120 dB of the incident peak, for every direction, with a closed box and
/// with periodic y. 600 steps take the pulse in, across the 50-cell box, and
/// out of it again; the incident peak itself must be about the waveform's 1.
#[test]
fn a15_tfsf_leakage_empty_domain() {
    let mut worst = f64::NEG_INFINITY;
    for periodic in [false, true] {
        for dir in [
            Direction::PlusX,
            Direction::MinusX,
            Direction::PlusY,
            Direction::MinusY,
        ] {
            if periodic && matches!(dir, Direction::PlusY | Direction::MinusY) {
                continue;
            }
            let (sf, tf) = leakage(&empty_domain(100, 25, 75, dir, periodic), 600);
            let db = 20.0 * (sf / tf).log10();
            eprintln!(
                "A15 {dir:?} {}: SF max {sf:.2e}, TF max {tf:.3}, {db:.1} dB",
                if periodic { "periodic" } else { "box" }
            );
            assert!((0.9..1.1).contains(&tf), "{dir:?}: incident peak {tf}");
            assert!(db <= -120.0, "{dir:?}: {db} dB");
            worst = worst.max(db);
        }
    }
    eprintln!("A15: worst leakage {worst:.1} dB (gate -120)");
}

/// Inside the box the total field is the incident wave alone: along a row
/// it equals the 1D auxiliary solution, which is a y-uniform 2D wave. For
/// +x travel every row in the box must be the same row, and the wave must
/// actually be on the grid.
#[test]
fn total_field_inside_is_a_plane_wave() {
    let scene = empty_domain(100, 25, 75, Direction::PlusX, false);
    let nx = scene.grid.nx;
    let mut sim = SimTm::from_scene(&scene).unwrap();
    // The Ricker peaks at the launch node (x = 23) at t0 = 40, step 80, and
    // moves about 0.5 cells per step, so at step 150 it is near x = 58.
    sim.step_n(150);
    let row = |j: usize| sim.ez()[j * nx + 25..=j * nx + 75].to_vec();
    let base = row(50);
    assert!(common::max_abs(&base) > 0.5, "the pulse must be in the box");
    let peak = common::max_abs(&base);
    for j in 25..=75 {
        let d: f64 = row(j)
            .iter()
            .zip(&base)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(d <= 1e-13 * peak, "row {j} differs by {d:e}");
    }
}
