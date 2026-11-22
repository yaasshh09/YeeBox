//! The Tier 4 scene set: every scene under `validation/scenes/` parses and
//! validates, and together they exercise every Phase 1 to 3 feature, so the
//! GPU against CPU diff (which runs on every file there) covers each one.
//! A feature is not done until a scene here uses it.

use fdtd_core::material::Material;
use fdtd_core::monitor::Normal;
use fdtd_core::scene::{Monitor, Scene, Shape};
use fdtd_core::source::{SourceKind, Waveform};
use fdtd_core::tfsf::Direction;
use std::collections::BTreeSet;
use std::path::Path;

fn scenes() -> Vec<(String, Scene)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/scenes");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).unwrap();
            let scene = Scene::from_json(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            scene.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
            out.push((name, scene));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn waveform_name(w: &Waveform) -> &'static str {
    match w {
        Waveform::Gaussian { .. } => "gaussian",
        Waveform::ModulatedGaussian { .. } => "modulated gaussian",
        Waveform::Ricker { .. } => "ricker",
        Waveform::RampedCw { .. } => "ramped cw",
    }
}

/// The features one scene exercises, as short names.
fn features(scene: &Scene) -> BTreeSet<&'static str> {
    let mut f = BTreeSet::new();
    match (&scene.pml, scene.periodic_y) {
        (None, false) => f.insert("closed pec box"),
        (Some(_), _) => f.insert("cpml"),
        _ => false,
    };
    if scene.periodic_y {
        f.insert("periodic y");
    }
    if scene.mode == fdtd_core::scene::Mode::Te {
        f.insert("te mode");
    }
    for shape in &scene.geometry {
        let id = match shape {
            Shape::Rect { material, .. } => {
                f.insert("rect");
                *material
            }
            Shape::Circle { material, .. } => {
                f.insert("circle");
                *material
            }
            Shape::Polygon { material, .. } => {
                f.insert("polygon");
                *material
            }
        };
        match scene.materials[id] {
            Material::Pec => f.insert("pec shape"),
            Material::Dielectric { sigma, .. } if sigma > 0.0 => f.insert("lossy material"),
            Material::Dielectric { .. } => f.insert("dielectric shape"),
            Material::Drude { .. } => f.insert("drude shape"),
            Material::Lorentz { .. } => f.insert("lorentz shape"),
        };
    }
    if let Some(paint) = &scene.paint {
        let ids: BTreeSet<usize> = (0..paint.len()).filter_map(|k| paint.get(k)).collect();
        for id in ids {
            f.insert(match scene.materials[id] {
                Material::Pec => "pec paint",
                _ if id == 0 => "erase paint",
                _ => "dielectric paint",
            });
        }
    }
    for s in &scene.sources {
        f.insert(match s.kind {
            SourceKind::Soft => "soft source",
            SourceKind::Hard => "hard source",
        });
        f.insert(waveform_name(&s.waveform));
    }
    for (a, b) in scene.sources.iter().zip(scene.sources.iter().skip(1)) {
        if (a.i, a.j) == (b.i, b.j) {
            f.insert("two sources on one node");
        }
    }
    let phases: BTreeSet<u64> = scene
        .sources
        .iter()
        .filter_map(|s| match s.waveform {
            Waveform::RampedCw { phase, .. } => Some(phase.to_bits()),
            _ => None,
        })
        .collect();
    if phases.len() >= 4 {
        f.insert("phased array");
    }
    if let Some(pw) = &scene.plane_wave {
        f.insert(waveform_name(&pw.waveform));
        f.insert(match pw.direction {
            Direction::PlusX => "plane wave +x",
            Direction::MinusX => "plane wave -x",
            Direction::PlusY => "plane wave +y",
            Direction::MinusY => "plane wave -y",
        });
        f.insert(if scene.periodic_y {
            "periodic tfsf box"
        } else {
            "closed tfsf box"
        });
        let along_x = matches!(pw.direction, Direction::PlusX | Direction::MinusX);
        if (along_x && pw.x0 == pw.x1) || (!along_x && pw.y0 == pw.y1) {
            f.insert("one node thick tfsf box");
        }
    }
    for m in &scene.monitors {
        f.insert(match m {
            Monitor::Point { .. } => "point monitor",
            Monitor::Dft { .. } => "dft monitor",
            Monitor::Flux {
                normal: Normal::X, ..
            } => "flux x",
            Monitor::Flux {
                normal: Normal::Y, ..
            } => "flux y",
        });
    }
    f
}

#[test]
fn scene_set_covers_every_feature() {
    let all = scenes();
    assert!(all.len() >= 8, "only {} scenes", all.len());
    let mut covered = BTreeSet::new();
    for (name, scene) in &all {
        // Kept small so 2000 CPU steps per scene stay quick in the test profile.
        assert!(
            scene.grid.nx * scene.grid.ny <= 200 * 160,
            "{name} is {} x {}",
            scene.grid.nx,
            scene.grid.ny
        );
        covered.extend(features(scene));
    }
    let wanted = [
        "closed pec box",
        "cpml",
        "periodic y",
        "rect",
        "circle",
        "polygon",
        "pec shape",
        "lossy material",
        "dielectric shape",
        "pec paint",
        "dielectric paint",
        "erase paint",
        "soft source",
        "hard source",
        "two sources on one node",
        "gaussian",
        "modulated gaussian",
        "ricker",
        "ramped cw",
        "phased array",
        "plane wave +x",
        "plane wave -x",
        "plane wave +y",
        "plane wave -y",
        "periodic tfsf box",
        "closed tfsf box",
        "one node thick tfsf box",
        "point monitor",
        "dft monitor",
        "flux x",
        "flux y",
        "te mode",
        "drude shape",
        "lorentz shape",
    ];
    let missing: Vec<_> = wanted.iter().filter(|w| !covered.contains(*w)).collect();
    assert!(missing.is_empty(), "no scene exercises {missing:?}");
}
