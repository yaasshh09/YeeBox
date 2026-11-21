//! The browser self-test's golden fields are what the CPU kernel computes
//! today: regenerated in memory and compared byte for byte with the files
//! under `validation/golden/selftest/`. If a kernel change moves a number,
//! rerun `fdtd selftest-golden` and commit the files with the change.

use fdtd_core::scene::Scene;
use fdtd_core::selftest::{SCENES, golden};
use std::path::Path;

#[test]
fn selftest_golden_files_are_fresh() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation");
    for name in SCENES {
        let text = std::fs::read_to_string(root.join("scenes").join(name)).unwrap();
        let scene = Scene::from_json(&text).unwrap();
        let file = root
            .join("golden/selftest")
            .join(name.replace(".json", ".bin"));
        let committed = std::fs::read(&file)
            .unwrap_or_else(|e| panic!("{}: {e}; run fdtd selftest-golden", file.display()));
        assert!(
            golden(&scene).unwrap() == committed,
            "{} is stale; run fdtd selftest-golden",
            file.display()
        );
    }
}
