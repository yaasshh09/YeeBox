//! The web presets (`web/src/presets`, full size for the GPU) are valid
//! scenes, each with a compact twin in `validation/scenes` for the CPU
//! fallback, which the scene set and Tier 4 run.

use fdtd_core::scene::Scene;
use std::path::Path;

#[test]
fn presets_validate_and_have_compact_twins() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut n = 0;
    for entry in std::fs::read_dir(root.join("web/src/presets")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        n += 1;
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let full = Scene::from_json(&std::fs::read_to_string(&path).unwrap())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let twin = root.join(format!("validation/scenes/preset_{name}_compact.json"));
        let compact = Scene::from_json(&std::fs::read_to_string(&twin).unwrap())
            .unwrap_or_else(|e| panic!("{name} compact: {e}"));
        // The twin shows the same physical wavelength in the status strip.
        let (a, b) = (full.status(0).unwrap(), compact.status(0).unwrap());
        let wl = |s: &fdtd_core::scene::Status| {
            s.sources
                .iter()
                .flatten()
                .chain(&s.plane_wave)
                .next()
                .unwrap()
                .vacuum_wavelength_m
        };
        assert!((wl(&a) - wl(&b)).abs() / wl(&a) < 1e-12, "{name}");
        assert!(full.grid.nx * full.grid.ny <= 1024 * 1024, "{name}");
        // A share link carries the Rust JSON of the edited scene; it must
        // come back exactly, paint included.
        let mut edited = full.clone();
        edited
            .paint_stroke((300.0, 100.0), (300.0, 200.0), 2.0, 2)
            .unwrap();
        let text = edited.to_json().unwrap();
        assert_eq!(Scene::from_json(&text).unwrap(), edited, "{name}");
        assert_eq!(
            Scene::from_json(&text).unwrap().to_json().unwrap(),
            text,
            "{name}"
        );
    }
    assert_eq!(n, 8);
}
