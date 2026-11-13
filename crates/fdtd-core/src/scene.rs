//! The scene: one value that fully determines a simulation. Both kernels are
//! built from it, and it round trips through JSON.
//!
//! Units: grid indices and geometry coordinates are in cells, with Ez node
//! (i, j) at (i, j). Time and frequency inside waveforms are normalized
//! (one time unit is dx / c0). `dx_phys_m` only matters at the UI boundary.

use crate::dispersion::courant_limit;
use crate::error::{FdtdError, Result};
use crate::material::Material;
use crate::source::{SourceKind, Waveform};
use serde::{Deserialize, Serialize};

/// The format version this build writes and reads.
pub const SCENE_VERSION: u32 = 1;

/// Largest grid accepted, in Ez nodes (4096 x 4096).
pub const MAX_NODES: usize = 1 << 24;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub version: u32,
    pub grid: GridSpec,
    /// Material table. Every node starts as material 0, so the table must not be empty.
    pub materials: Vec<Material>,
    /// Painted in order onto the material grid; the last shape containing a node wins.
    #[serde(default)]
    pub geometry: Vec<Shape>,
    #[serde(default)]
    pub sources: Vec<Source>,
    #[serde(default)]
    pub monitors: Vec<Monitor>,
}

/// Grid size in Ez nodes, physical cell size, and Courant number.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    pub dx_phys_m: f64,
    pub courant: f64,
}

/// A filled region in cell coordinates. `material` indexes the scene table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum Shape {
    /// Axis-aligned, edges included: `x0 <= x <= x1`, `y0 <= y <= y1`.
    Rect {
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        material: usize,
    },
    /// Disk, edge included: `(x - cx)^2 + (y - cy)^2 <= r^2`.
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        material: usize,
    },
    /// Simple or self-intersecting polygon, even-odd rule, at least 3 points.
    Polygon {
        points: Vec<[f64; 2]>,
        material: usize,
    },
}

/// A point source on Ez node (i, j). Must be an interior, non-PEC node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub i: usize,
    pub j: usize,
    pub waveform: Waveform,
    pub amplitude: f64,
    #[serde(default)]
    pub kind: SourceKind,
}

/// A field monitor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Monitor {
    /// Records Ez at node (i, j) once at n = 0 and after every step.
    Point { i: usize, j: usize },
}

fn finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FdtdError::InvalidParameter { name, value })
    }
}

impl Shape {
    fn material(&self) -> usize {
        match self {
            Shape::Rect { material, .. }
            | Shape::Circle { material, .. }
            | Shape::Polygon { material, .. } => *material,
        }
    }

    fn validate(&self, n_materials: usize) -> Result<()> {
        if self.material() >= n_materials {
            return Err(FdtdError::UnknownMaterial {
                id: self.material(),
            });
        }
        match self {
            Shape::Rect { x0, y0, x1, y1, .. } => {
                for (name, v) in [("x0", *x0), ("y0", *y0), ("x1", *x1), ("y1", *y1)] {
                    finite(name, v)?;
                }
                if x1 < x0 {
                    return Err(FdtdError::InvalidParameter {
                        name: "x1",
                        value: *x1,
                    });
                }
                if y1 < y0 {
                    return Err(FdtdError::InvalidParameter {
                        name: "y1",
                        value: *y1,
                    });
                }
            }
            Shape::Circle { cx, cy, r, .. } => {
                finite("cx", *cx)?;
                finite("cy", *cy)?;
                if !(r.is_finite() && *r >= 0.0) {
                    return Err(FdtdError::InvalidParameter {
                        name: "r",
                        value: *r,
                    });
                }
            }
            Shape::Polygon { points, .. } => {
                if points.len() < 3 {
                    return Err(FdtdError::InvalidParameter {
                        name: "points",
                        value: points.len() as f64,
                    });
                }
                for p in points {
                    finite("point x", p[0])?;
                    finite("point y", p[1])?;
                }
            }
        }
        Ok(())
    }
}

impl Scene {
    /// Parses and validates a scene.
    pub fn from_json(text: &str) -> Result<Scene> {
        let scene: Scene =
            serde_json::from_str(text).map_err(|e| FdtdError::Json(e.to_string()))?;
        scene.validate()?;
        Ok(scene)
    }

    /// Pretty JSON, readable in presets and diffs.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| FdtdError::Json(e.to_string()))
    }

    /// Checks everything that would make a kernel built from this scene wrong
    /// or unstable: version, grid size, Courant number against the 2D limit,
    /// material values, material indices, shape parameters, and source and
    /// monitor positions.
    pub fn validate(&self) -> Result<()> {
        self.validate_inner(true)
    }

    /// `validate` with the Courant check optional, for the stability test only.
    pub(crate) fn validate_inner(&self, check_courant: bool) -> Result<()> {
        if self.version != SCENE_VERSION {
            return Err(FdtdError::UnsupportedVersion {
                version: self.version,
            });
        }
        let GridSpec {
            nx,
            ny,
            dx_phys_m,
            courant,
        } = self.grid;
        if nx < 3 || ny < 3 || nx.checked_mul(ny).is_none_or(|n| n > MAX_NODES) {
            return Err(FdtdError::GridSize { nx, ny });
        }
        if !(dx_phys_m.is_finite() && dx_phys_m > 0.0) {
            return Err(FdtdError::InvalidParameter {
                name: "dx_phys_m",
                value: dx_phys_m,
            });
        }
        if !(courant.is_finite() && courant > 0.0) {
            return Err(FdtdError::InvalidParameter {
                name: "courant",
                value: courant,
            });
        }
        let limit = courant_limit(2)?;
        // Same few-ulp slack as the dispersion solver, so 1/sqrt(2) itself passes.
        if check_courant && courant > limit * (1.0 + 4.0 * f64::EPSILON) {
            return Err(FdtdError::CourantAboveLimit { s: courant, limit });
        }
        if self.materials.is_empty() {
            return Err(FdtdError::UnknownMaterial { id: 0 });
        }
        for m in &self.materials {
            m.validate()?;
        }
        for shape in &self.geometry {
            shape.validate(self.materials.len())?;
        }
        for src in &self.sources {
            src.waveform.validate()?;
            finite("amplitude", src.amplitude)?;
            // Wall nodes are PEC and never updated, so a source there is wrong.
            if !(1..nx - 1).contains(&src.i) || !(1..ny - 1).contains(&src.j) {
                return Err(FdtdError::OutOfDomain {
                    what: "source",
                    i: src.i,
                    j: src.j,
                });
            }
        }
        for m in &self.monitors {
            let Monitor::Point { i, j } = *m;
            if i >= nx || j >= ny {
                return Err(FdtdError::OutOfDomain {
                    what: "monitor",
                    i,
                    j,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small valid scene that exercises every field.
    pub(crate) fn sample() -> Scene {
        Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx: 40,
                ny: 30,
                dx_phys_m: 1e-3,
                courant: 0.5,
            },
            materials: vec![
                Material::VACUUM,
                Material::Dielectric {
                    eps_inf: 4.0,
                    sigma: 0.0,
                },
                Material::Pec,
            ],
            geometry: vec![
                Shape::Rect {
                    x0: 5.0,
                    y0: 5.0,
                    x1: 10.0,
                    y1: 12.0,
                    material: 1,
                },
                Shape::Circle {
                    cx: 25.0,
                    cy: 15.0,
                    r: 4.5,
                    material: 2,
                },
                Shape::Polygon {
                    points: vec![[30.0, 2.0], [38.0, 2.0], [34.0, 8.0]],
                    material: 1,
                },
            ],
            sources: vec![Source {
                i: 12,
                j: 20,
                waveform: Waveform::Ricker {
                    t0: 30.0,
                    f_peak: 0.05,
                },
                amplitude: 1.0,
                kind: SourceKind::Soft,
            }],
            monitors: vec![Monitor::Point { i: 30, j: 20 }],
        }
    }

    #[test]
    fn json_round_trip() {
        let scene = sample();
        let text = scene.to_json().unwrap();
        assert_eq!(Scene::from_json(&text).unwrap(), scene);
    }

    #[test]
    fn json_minimal_and_strict() {
        let text = r#"{"version": 1, "grid": {"nx": 10, "ny": 8, "dx_phys_m": 0.001,
            "courant": 0.5}, "materials": [{"kind": "dielectric", "eps_inf": 1, "sigma": 0}]}"#;
        let scene = Scene::from_json(text).unwrap();
        assert!(scene.geometry.is_empty() && scene.sources.is_empty());
        let typo = text.replace("\"courant\"", "\"courrant\"");
        assert!(matches!(Scene::from_json(&typo), Err(FdtdError::Json(_))));
    }

    fn rejects(f: impl FnOnce(&mut Scene)) -> FdtdError {
        let mut s = sample();
        f(&mut s);
        s.validate().unwrap_err()
    }

    #[test]
    fn validate_rejects_bad_scenes() {
        assert!(sample().validate().is_ok());
        assert_eq!(
            rejects(|s| s.version = 2),
            FdtdError::UnsupportedVersion { version: 2 }
        );
        assert!(matches!(
            rejects(|s| s.grid.courant = 0.7072),
            FdtdError::CourantAboveLimit { .. }
        ));
        assert!(matches!(
            rejects(|s| s.grid.courant = 0.0),
            FdtdError::InvalidParameter { .. }
        ));
        assert!(matches!(
            rejects(|s| s.grid.nx = 2),
            FdtdError::GridSize { .. }
        ));
        assert!(matches!(
            rejects(|s| s.grid.ny = MAX_NODES),
            FdtdError::GridSize { .. }
        ));
        assert!(matches!(
            rejects(|s| s.grid.nx = usize::MAX),
            FdtdError::GridSize { .. }
        ));
        assert!(matches!(
            rejects(|s| s.materials.clear()),
            FdtdError::UnknownMaterial { id: 0 }
        ));
        assert!(matches!(
            rejects(|s| s.materials[1] = Material::Dielectric {
                eps_inf: 0.5,
                sigma: 0.0
            }),
            FdtdError::InvalidParameter {
                name: "eps_inf",
                ..
            }
        ));
        assert!(matches!(
            rejects(|s| s.materials.truncate(2)),
            FdtdError::UnknownMaterial { id: 2 }
        ));
        assert!(matches!(
            rejects(|s| s.geometry[1] = Shape::Circle {
                cx: 1.0,
                cy: 1.0,
                r: -1.0,
                material: 0
            }),
            FdtdError::InvalidParameter { name: "r", .. }
        ));
        assert!(matches!(
            rejects(|s| s.geometry[2] = Shape::Polygon {
                points: vec![[0.0, 0.0], [1.0, 1.0]],
                material: 0
            }),
            FdtdError::InvalidParameter { .. }
        ));
        assert!(matches!(
            rejects(|s| s.geometry[0] = Shape::Rect {
                x0: 3.0,
                y0: 0.0,
                x1: 2.0,
                y1: 1.0,
                material: 0
            }),
            FdtdError::InvalidParameter { name: "x1", .. }
        ));
        for (i, j) in [(0, 5), (39, 5), (5, 0), (5, 29), (100, 5)] {
            assert_eq!(
                rejects(|s| (s.sources[0].i, s.sources[0].j) = (i, j)),
                FdtdError::OutOfDomain {
                    what: "source",
                    i,
                    j
                }
            );
        }
        assert!(matches!(
            rejects(|s| s.sources[0].amplitude = f64::NAN),
            FdtdError::InvalidParameter { .. }
        ));
        assert!(matches!(
            rejects(|s| s.sources[0].waveform = Waveform::Gaussian { t0: 1.0, tau: 0.0 }),
            FdtdError::InvalidParameter { .. }
        ));
        assert!(matches!(
            rejects(|s| s.monitors[0] = Monitor::Point { i: 40, j: 0 }),
            FdtdError::OutOfDomain {
                what: "monitor",
                ..
            }
        ));
        // At the limit itself, and on a wall node, a monitor is fine.
        let mut s = sample();
        s.grid.courant = std::f64::consts::FRAC_1_SQRT_2;
        s.monitors[0] = Monitor::Point { i: 0, j: 29 };
        assert!(s.validate().is_ok());
    }
}
