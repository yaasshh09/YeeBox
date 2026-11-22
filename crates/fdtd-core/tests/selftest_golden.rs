//! The browser self-test's golden fields are what the CPU kernel computes
//! today: regenerated in memory and compared with the files under
//! `validation/golden/selftest/`. If a kernel change moves a number, rerun
//! `fdtd selftest-golden` and commit the files with the change.
//!
//! The comparison is by value, not by byte. `sin` and `exp` come from the
//! platform's math library, which can differ by one unit in the last place
//! between Windows and Linux, so a file written on one is a few ulps off on
//! the other. A real kernel change moves the numbers by far more than the
//! tolerance, and a guard below checks that a 0.1 percent change is caught.

use fdtd_core::scene::Scene;
use fdtd_core::selftest::{CHECKPOINTS, PEAK_EVERY, SCENES, golden};
use std::path::Path;

/// Peaks are f64, then the fields are f32 (`fdtd_core::selftest::golden`).
const N_PEAKS: usize = 3 * (CHECKPOINTS[CHECKPOINTS.len() - 1] / PEAK_EVERY) as usize;

/// `None` if `a` and `b` agree: peaks within 1e-9 of the largest peak,
/// fields within 1e-6 of the largest field value (about 16 f32 ulps there).
/// Otherwise what differs.
fn differ(a: &[u8], b: &[u8]) -> Option<String> {
    if a.len() != b.len() || a.len() < 8 * N_PEAKS {
        return Some(format!("lengths {} and {}", a.len(), b.len()));
    }
    let (pa, fa) = a.split_at(8 * N_PEAKS);
    let (pb, fb) = b.split_at(8 * N_PEAKS);
    let f64s = |v: &[u8]| -> Vec<f64> {
        v.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    };
    let f32s = |v: &[u8]| -> Vec<f64> {
        v.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect()
    };
    let worst = |x: &[f64], y: &[f64]| {
        let scale = x.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let d = x
            .iter()
            .zip(y)
            .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
        d / scale
    };
    let peaks = worst(&f64s(pa), &f64s(pb));
    let fields = worst(&f32s(fa), &f32s(fb));
    (peaks > 1e-9 || fields > 1e-6).then(|| format!("peaks {peaks:.1e}, fields {fields:.1e}"))
}

fn scene(root: &Path, name: &str) -> Scene {
    let text = std::fs::read_to_string(root.join("scenes").join(name)).unwrap();
    Scene::from_json(&text).unwrap()
}

#[test]
fn selftest_golden_files_are_fresh() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation");
    for name in SCENES {
        let file = root
            .join("golden/selftest")
            .join(name.replace(".json", ".bin"));
        let committed = std::fs::read(&file)
            .unwrap_or_else(|e| panic!("{}: {e}; run fdtd selftest-golden", file.display()));
        if let Some(why) = differ(&golden(&scene(&root, name)).unwrap(), &committed) {
            panic!(
                "{} is stale ({why}); run fdtd selftest-golden",
                file.display()
            );
        }
    }
}

/// The tolerance still catches a real change: the rod scene with its plane
/// wave 0.1 percent stronger does not match its file.
#[test]
fn a_small_real_change_is_caught() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation");
    let committed = std::fs::read(root.join("golden/selftest/tfsf_rod_minus_x.bin")).unwrap();
    let mut s = scene(&root, "tfsf_rod_minus_x.json");
    s.plane_wave.as_mut().unwrap().amplitude *= 1.001;
    assert!(differ(&golden(&s).unwrap(), &committed).is_some());
}
