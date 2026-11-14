//! CPML reflection by the reference-domain method (`docs/02-numerics.md`).
//!
//! The same source runs in (a) the test domain, an interior box wrapped in
//! the layer under test, and (b) a reference domain, the same interior box
//! with PEC walls so far away that no echo reaches a probe inside the window.
//! For each probe,
//!
//!   R_dB = 20 log10( max_t |Ez_test - Ez_ref| / max_t |Ez_ref| ).
//!
//! Everything is normalized and runs on the f64 reference kernel; f32 cannot
//! resolve -80 dB. Coordinates in a setup are relative to the interior box:
//! node (0, 0) is its bottom-left corner, which is the interface corner of
//! the layer under test.

use crate::cpml::PmlSpec;
use crate::error::Result;
use crate::material::Material;
use crate::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Source};
use crate::source::{SourceKind, Waveform};
use crate::tm::SimTm;

/// One reflection measurement: grid, pulse, interior box, source, probes.
#[derive(Debug, Clone, PartialEq)]
pub struct ReflectionSetup {
    /// Courant number.
    pub s: f64,
    /// Soft source waveform at `source`.
    pub waveform: Waveform,
    /// Interior box in nodes, interface to interface inclusive.
    pub width: usize,
    pub height: usize,
    /// Source node, relative to the interior box.
    pub source: (usize, usize),
    /// Probe nodes, relative to the interior box.
    pub probes: Vec<(usize, usize)>,
    /// Steps in the observation window.
    pub steps: usize,
}

impl ReflectionSetup {
    /// The angle sweep used for A9 and A10. A Ricker pulse peaking at 20
    /// cells per wavelength starts 20 cells above the bottom interface. The
    /// probes sit on the interface row where the direct wave meets the face
    /// at 0, 10, ..., 80 degrees (rounded to nodes), and the last probe sits
    /// on the interface corner. 1000 steps at S = 0.5.
    pub fn standard() -> Self {
        let (sx, h) = (25, 20);
        let mut probes: Vec<(usize, usize)> = (0..=8)
            .map(|k| {
                let t = (10.0 * k as f64).to_radians().tan();
                (sx + (h as f64 * t).round() as usize, 0)
            })
            .collect();
        probes.push((0, 0));
        ReflectionSetup {
            s: 0.5,
            waveform: Waveform::Ricker {
                t0: 40.0,
                f_peak: 0.05,
            },
            width: 165,
            height: 60,
            source: (sx, h),
            probes,
            steps: 1000,
        }
    }

    /// Angle in degrees between the bottom face normal and the line from the
    /// source to probe `k`. Exact for the rounded probe position.
    pub fn angle_deg(&self, k: usize) -> f64 {
        let (px, py) = self.probes[k];
        let (sx, sy) = self.source;
        let dx = (px as f64 - sx as f64).abs();
        let dy = sy as f64 - py as f64;
        dx.atan2(dy).to_degrees()
    }

    /// Worst (largest) `R_dB` over probes whose angle is at most `max_deg`
    /// (half a degree of slack covers node rounding).
    pub fn worst_db(&self, db: &[f64], max_deg: f64) -> f64 {
        db.iter()
            .enumerate()
            .filter(|&(k, _)| self.angle_deg(k) <= max_deg + 0.5)
            .fold(f64::NEG_INFINITY, |m, (_, &v)| m.max(v))
    }

    /// Margin of the reference domain: with PEC walls this far outside the
    /// interior box, any echo travels at least `2 * margin` cells, which takes
    /// longer than the window plus 40 cells of slack.
    pub fn default_margin(&self) -> usize {
        (self.steps as f64 * self.s / 2.0).ceil() as usize + 20
    }

    /// The interior box placed `offset` nodes inside the outer wall, with an
    /// optional layer. Vacuum everywhere.
    fn scene(&self, offset: usize, pml: Option<PmlSpec>) -> Scene {
        let at = |(i, j): (usize, usize)| (i + offset, j + offset);
        let (si, sj) = at(self.source);
        Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx: self.width + 2 * offset,
                ny: self.height + 2 * offset,
                dx_phys_m: 1e-3,
                courant: self.s,
            },
            materials: vec![Material::VACUUM],
            geometry: vec![],
            sources: vec![Source {
                i: si,
                j: sj,
                waveform: self.waveform,
                amplitude: 1.0,
                kind: SourceKind::Soft,
            }],
            monitors: self
                .probes
                .iter()
                .map(|&p| {
                    let (i, j) = at(p);
                    Monitor::Point { i, j }
                })
                .collect(),
            pml,
            periodic_y: false,
        }
    }

    fn run(&self, scene: &Scene) -> Result<Vec<Vec<f64>>> {
        let mut sim = SimTm::from_scene(scene)?;
        sim.step_n(self.steps as u64);
        Ok(sim.probes().iter().map(|p| p.history().to_vec()).collect())
    }

    /// Probe records from the reference domain at the default margin.
    pub fn reference(&self) -> Result<Vec<Vec<f64>>> {
        self.reference_with_margin(self.default_margin())
    }

    /// Probe records from a PEC box `margin` nodes outside the interior box.
    pub fn reference_with_margin(&self, margin: usize) -> Result<Vec<Vec<f64>>> {
        self.run(&self.scene(margin, None))
    }

    /// Probe records from the test domain: the interior box inside `pml`, or
    /// with no layer, inside a PEC wall one node away.
    pub fn test_domain(&self, pml: Option<PmlSpec>) -> Result<Vec<Vec<f64>>> {
        let d = pml.map_or(0, |p| p.thickness).max(1);
        self.run(&self.scene(d, pml.filter(|p| p.thickness > 0)))
    }
}

/// `R_dB` per probe, from test and reference records of equal shape. An
/// exact match gives negative infinity.
pub fn reflection_db(test: &[Vec<f64>], reference: &[Vec<f64>]) -> Vec<f64> {
    test.iter()
        .zip(reference)
        .map(|(t, r)| {
            let diff = t
                .iter()
                .zip(r)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            let peak = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            20.0 * (diff / peak).log10()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpml::tests::spec;

    /// A smaller sweep for the harness self-checks.
    fn small() -> ReflectionSetup {
        ReflectionSetup {
            width: 60,
            height: 30,
            source: (20, 10),
            probes: vec![(20, 0), (30, 0), (45, 0), (0, 0)],
            steps: 400,
            ..ReflectionSetup::standard()
        }
    }

    #[test]
    fn standard_angles() {
        let s = ReflectionSetup::standard();
        assert_eq!(s.probes.len(), 10);
        for k in 0..9 {
            let a = s.angle_deg(k);
            assert!((a - 10.0 * k as f64).abs() < 1.5, "probe {k}: {a}");
        }
        assert!((s.angle_deg(9) - 51.34).abs() < 0.01);
        // worst_db keeps probes up to 70 degrees (and the 51 degree corner).
        let mut db: Vec<f64> = (0..10).map(|k| -100.0 + k as f64).collect();
        db[9] = -150.0;
        assert_eq!(s.worst_db(&db, 70.0), -93.0);
        assert_eq!(s.worst_db(&db, 80.0), -92.0);
        db[9] = -50.0;
        assert_eq!(s.worst_db(&db, 70.0), -50.0);
        // The window holds the direct pulse at the farthest probe.
        let far = s.probes[8].0 - s.source.0;
        assert!((far as f64).hypot(20.0) + 80.0 < s.steps as f64 * s.s);
    }

    /// The reference at the default margin equals one with 1.5 times the
    /// margin to far below anything measured, so no wall echo and no grid
    /// precursor from the far walls reaches a probe inside the window.
    #[test]
    fn reference_is_clean() {
        let s = small();
        let m = s.default_margin();
        let a = s.reference_with_margin(m).unwrap();
        let b = s.reference_with_margin(m + m / 2).unwrap();
        let r = reflection_db(&a, &b);
        eprintln!("reference vs bigger reference: {r:?} dB");
        assert!(r.iter().all(|&db| db < -140.0), "{r:?}");
    }

    /// A bare PEC box reflects at the 0 dB level, so the harness sees
    /// reflections; a 10-cell layer cuts that by tens of dB.
    #[test]
    fn harness_sees_reflection_and_absorption() {
        let s = small();
        let reference = s.reference().unwrap();
        let bare = reflection_db(&s.test_domain(None).unwrap(), &reference);
        let layer = reflection_db(&s.test_domain(Some(spec(10))).unwrap(), &reference);
        eprintln!("bare pec {bare:?} dB, 10-cell layer {layer:?} dB");
        assert!(bare.iter().all(|&db| db > -10.0), "{bare:?}");
        assert!(layer.iter().all(|&db| db < -40.0), "{layer:?}");
    }
}
