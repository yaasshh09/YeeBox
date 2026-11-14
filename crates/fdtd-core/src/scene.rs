//! The scene: one value that fully determines a simulation. Both kernels are
//! built from it, and it round trips through JSON.
//!
//! Units: grid indices and geometry coordinates are in cells, with Ez node
//! (i, j) at (i, j). Time and frequency inside waveforms are normalized
//! (one time unit is dx / c0). `dx_phys_m` only matters at the UI boundary.

use crate::cpml::PmlSpec;
use crate::dispersion::courant_limit;
use crate::error::{FdtdError, Result};
use crate::material::Material;
use crate::source::{SourceKind, Waveform};
use crate::tfsf::{Direction, PlaneWave, PlaneWaveFault};
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
    /// CPML layer inside the PEC wall. Absent means a closed PEC box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pml: Option<PmlSpec>,
    /// Wrap y: Ez row 0 neighbors row ny - 1, there are no y walls, and the
    /// CPML (if any) sits on the x faces only. For y-uniform plane waves.
    #[serde(default, skip_serializing_if = "is_false")]
    pub periodic_y: bool,
    /// TF/SF plane wave, axis-aligned. Absent means none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plane_wave: Option<PlaneWave>,
}

fn is_false(b: &bool) -> bool {
    !*b
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
    /// Whether the point (x, y), in cell units, is inside. Rect and circle
    /// edges count as inside. Polygons use the even-odd rule, and a point
    /// exactly on a polygon edge may land either way.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Shape::Rect { x0, y0, x1, y1, .. } => *x0 <= x && x <= *x1 && *y0 <= y && y <= *y1,
            Shape::Circle { cx, cy, r, .. } => (x - cx).powi(2) + (y - cy).powi(2) <= r * r,
            Shape::Polygon { points, .. } => {
                let mut inside = false;
                let mut prev = points[points.len() - 1];
                for &p in points {
                    if (p[1] > y) != (prev[1] > y)
                        && x < (prev[0] - p[0]) * (y - p[1]) / (prev[1] - p[1]) + p[0]
                    {
                        inside = !inside;
                    }
                    prev = p;
                }
                inside
            }
        }
    }

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
    /// Material index of Ez node (i, j): the last shape containing the node's
    /// position (i, j), or 0 if none does. Assumes a validated scene.
    pub fn material_at(&self, i: usize, j: usize) -> usize {
        let (x, y) = (i as f64, j as f64);
        self.geometry
            .iter()
            .rev()
            .find(|s| s.contains(x, y))
            .map_or(0, Shape::material)
    }

    /// Material index of every Ez node, laid out `j * nx + i`. The one
    /// rasterization rule shared by every kernel. Assumes a validated scene.
    pub fn rasterize(&self) -> Vec<usize> {
        let GridSpec { nx, ny, .. } = self.grid;
        (0..ny)
            .flat_map(|j| (0..nx).map(move |i| (i, j)))
            .map(|(i, j)| self.material_at(i, j))
            .collect()
    }

    /// CPML thickness in cells, 0 without a layer.
    pub fn pml_thickness(&self) -> usize {
        self.pml.map_or(0, |p| p.thickness)
    }

    /// Whether Ez node (i, j) is strictly inside the CPML layer (the wall
    /// ring counts as inside). Interface nodes, at depth 0, are not. With
    /// `periodic_y` only the x faces have a layer.
    pub fn in_pml(&self, i: usize, j: usize) -> bool {
        let d = self.pml_thickness();
        let GridSpec { nx, ny, .. } = self.grid;
        let in_y = !self.periodic_y && (j < d || j > ny - 1 - d);
        d > 0 && (i < d || i > nx - 1 - d || in_y)
    }

    /// Rows of Ez nodes the solver updates: all of them when y is periodic,
    /// otherwise everything but the two wall rows.
    pub fn live_rows(&self) -> std::ops::Range<usize> {
        if self.periodic_y {
            0..self.grid.ny
        } else {
            1..self.grid.ny - 1
        }
    }

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
    /// material values, material indices, shape parameters, the CPML layer
    /// (parameters, fit, and materials uniform along its normal), and source
    /// and monitor positions.
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
        if let Some(p) = &self.pml {
            p.validate()?;
            let d = p.thickness;
            if nx < 2 * d + 3 || (!self.periodic_y && ny < 2 * d + 3) {
                return Err(FdtdError::PmlTooThick {
                    thickness: d,
                    nx,
                    ny,
                });
            }
            self.check_pml_uniform()?;
        }
        for src in &self.sources {
            src.waveform.validate()?;
            finite("amplitude", src.amplitude)?;
            // Wall nodes are PEC and never updated, so a source there is wrong.
            if !(1..nx - 1).contains(&src.i) || !self.live_rows().contains(&src.j) {
                return Err(FdtdError::OutOfDomain {
                    what: "source",
                    i: src.i,
                    j: src.j,
                });
            }
            if self.in_pml(src.i, src.j) {
                return Err(FdtdError::SourceInPml { i: src.i, j: src.j });
            }
            if self.materials[self.material_at(src.i, src.j)] == Material::Pec {
                return Err(FdtdError::SourceOnPec { i: src.i, j: src.j });
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
        if let Some(pw) = &self.plane_wave {
            self.check_plane_wave(pw)?;
        }
        Ok(())
    }

    /// The TF/SF placement rules: a non-empty box whose faces, and the
    /// scattered-field nodes just across them, are live nodes outside the
    /// walls and the layer, all in material 0, which must be a lossless
    /// dielectric. With periodic y the wave travels along x and the box spans
    /// every row, so there are no y faces to check.
    fn check_plane_wave(&self, pw: &PlaneWave) -> Result<()> {
        let fault = |f| Err(FdtdError::PlaneWave(f));
        pw.waveform.validate()?;
        finite("amplitude", pw.amplitude)?;
        let GridSpec { nx, ny, .. } = self.grid;
        if pw.x0 > pw.x1 || pw.y0 > pw.y1 {
            return fault(PlaneWaveFault::EmptyBox);
        }
        if !matches!(self.materials[0], Material::Dielectric { sigma, .. } if sigma == 0.0) {
            return fault(PlaneWaveFault::LossyBackground);
        }
        if self.periodic_y {
            if matches!(pw.direction, Direction::PlusY | Direction::MinusY) {
                return fault(PlaneWaveFault::WrapsAlongY);
            }
            if pw.y0 != 0 || pw.y1 != ny - 1 {
                return fault(PlaneWaveFault::NotFullHeight);
            }
        }
        // Every node on a face and the node across it, as signed indices so
        // a face on row or column 0 shows up as out of range, not as a wrap.
        let (x0, x1, y0, y1) = (pw.x0 as i64, pw.x1 as i64, pw.y0 as i64, pw.y1 as i64);
        let mut nodes = Vec::new();
        for j in y0..=y1 {
            for i in [x0 - 1, x0, x1, x1 + 1] {
                nodes.push((i, j));
            }
        }
        if !self.periodic_y {
            for i in x0..=x1 {
                for j in [y0 - 1, y0, y1, y1 + 1] {
                    nodes.push((i, j));
                }
            }
        }
        for (i, j) in nodes {
            let live_x = (1..nx as i64 - 1).contains(&i);
            let live_y = self.periodic_y || (1..ny as i64 - 1).contains(&j);
            if !(live_x && live_y) || self.in_pml(i as usize, j as usize) {
                return fault(PlaneWaveFault::FaceTooClose);
            }
            if self.material_at(i as usize, j as usize) != 0 {
                return fault(PlaneWaveFault::NotBackground);
            }
        }
        Ok(())
    }

    /// Every non-wall node in the layer must have the material of the
    /// interior node it is clamped to. That covers the edges (uniform along
    /// the normal) and the corners (uniform along both) with one rule.
    fn check_pml_uniform(&self) -> Result<()> {
        let d = self.pml_thickness();
        let GridSpec { nx, ny, .. } = self.grid;
        let ids = self.rasterize();
        for j in self.live_rows() {
            for i in 1..nx - 1 {
                if !self.in_pml(i, j) {
                    continue;
                }
                let cj = if self.periodic_y {
                    j
                } else {
                    j.clamp(d, ny - 1 - d)
                };
                let ci = i.clamp(d, nx - 1 - d);
                if ids[j * nx + i] != ids[cj * nx + ci] {
                    return Err(FdtdError::PmlNotUniform { i, j });
                }
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
            pml: None,
            periodic_y: false,
            plane_wave: None,
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

    fn empty(nx: usize, ny: usize, geometry: Vec<Shape>) -> Scene {
        let mut s = sample();
        s.grid.nx = nx;
        s.grid.ny = ny;
        s.geometry = geometry;
        s.sources.clear();
        s.monitors.clear();
        s
    }

    fn cells(scene: &Scene, id: usize) -> Vec<(usize, usize)> {
        let nx = scene.grid.nx;
        scene
            .rasterize()
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m == id)
            .map(|(k, _)| (k % nx, k / nx))
            .collect()
    }

    #[test]
    fn rect_edges_inclusive() {
        let rect = Shape::Rect {
            x0: 2.0,
            y0: 3.0,
            x1: 5.0,
            y1: 4.0,
            material: 1,
        };
        let s = empty(10, 8, vec![rect]);
        let expect: Vec<_> = (3..=4).flat_map(|j| (2..=5).map(move |i| (i, j))).collect();
        assert_eq!(cells(&s, 1), expect);
        assert_eq!(s.rasterize().len(), 80);
    }

    /// Lattice points in a disk of radius r differ from pi r^2 by far less
    /// than the perimeter; the disk is also symmetric under both mirrors.
    #[test]
    fn circle_area_and_symmetry() {
        let r = 20.0;
        let s = empty(
            61,
            61,
            vec![Shape::Circle {
                cx: 30.0,
                cy: 30.0,
                r,
                material: 2,
            }],
        );
        let got = cells(&s, 2);
        let area = std::f64::consts::PI * r * r;
        assert!((got.len() as f64 - area).abs() < 2.0 * std::f64::consts::PI * r);
        for &(i, j) in &got {
            assert!(got.contains(&(60 - i, j)) && got.contains(&(i, 60 - j)));
        }
        assert!(got.contains(&(10, 30)) && got.contains(&(30, 50)));
        assert!(!got.contains(&(9, 30)));
    }

    /// Right triangle with hypotenuse x + y = 10.5: nodes with i, j >= 1 and
    /// i + j <= 10, which is 45 nodes. No node lies on an edge.
    #[test]
    fn triangle_known_cells() {
        let tri = Shape::Polygon {
            points: vec![[0.5, 0.5], [10.0, 0.5], [0.5, 10.0]],
            material: 1,
        };
        let s = empty(14, 14, vec![tri]);
        let got = cells(&s, 1);
        let expect: Vec<_> = (1..14)
            .flat_map(|j| (1..14).map(move |i| (i, j)))
            .filter(|&(i, j)| i + j <= 10)
            .collect();
        assert_eq!(got.len(), 45);
        assert_eq!(got, expect);
    }

    #[test]
    fn last_shape_wins() {
        let a = Shape::Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 6.0,
            y1: 6.0,
            material: 1,
        };
        let b = Shape::Rect {
            x0: 4.0,
            y0: 4.0,
            x1: 9.0,
            y1: 9.0,
            material: 2,
        };
        let s = empty(10, 10, vec![a.clone(), b.clone()]);
        assert_eq!(s.material_at(5, 5), 2);
        assert_eq!(s.material_at(3, 3), 1);
        assert_eq!(s.material_at(9, 0), 0);
        let s = empty(10, 10, vec![b, a]);
        assert_eq!(s.material_at(5, 5), 1);
        let raster = s.rasterize();
        for j in 0..10 {
            for i in 0..10 {
                assert_eq!(raster[j * 10 + i], s.material_at(i, j));
            }
        }
    }

    /// Vacuum 40 x 30 with a 5-cell layer and a source well inside.
    fn open_sample() -> Scene {
        let mut s = empty(40, 30, vec![]);
        s.pml = Some(crate::cpml::tests::spec(5));
        s.sources = sample().sources;
        s
    }

    #[test]
    fn pml_json_round_trip_and_absent_by_default() {
        let s = open_sample();
        let text = s.to_json().unwrap();
        assert!(text.contains("\"kappa_max\""));
        assert_eq!(Scene::from_json(&text).unwrap(), s);
        // A scene without a layer writes no pml key, so Phase 1 files are unchanged.
        let text = sample().to_json().unwrap();
        assert!(!text.contains("pml"));
        assert_eq!(Scene::from_json(&text).unwrap().pml, None);
    }

    #[test]
    fn pml_validation() {
        assert!(open_sample().validate().is_ok());
        let bad = |f: fn(&mut Scene)| {
            let mut s = open_sample();
            f(&mut s);
            s.validate().unwrap_err()
        };
        assert!(matches!(
            bad(|s| s.pml.as_mut().unwrap().kappa_max = 0.5),
            FdtdError::InvalidParameter {
                name: "kappa_max",
                ..
            }
        ));
        // ny = 30 leaves 30 - 2 * 14 = 2 nodes, one short of the minimum 3.
        assert!(matches!(
            bad(|s| s.pml.as_mut().unwrap().thickness = 14),
            FdtdError::PmlTooThick { thickness: 14, .. }
        ));
        let mut s = open_sample();
        s.pml.as_mut().unwrap().thickness = 13;
        (s.sources[0].i, s.sources[0].j) = (20, 15);
        assert!(s.validate().is_ok());

        // Sources: inside the layer is rejected, on the interface node is fine.
        for (i, j) in [(4, 15), (35, 15), (20, 4), (20, 25), (2, 2)] {
            let mut s = open_sample();
            (s.sources[0].i, s.sources[0].j) = (i, j);
            assert_eq!(s.validate(), Err(FdtdError::SourceInPml { i, j }));
        }
        for (i, j) in [(5, 15), (34, 15), (20, 5), (20, 24)] {
            let mut s = open_sample();
            (s.sources[0].i, s.sources[0].j) = (i, j);
            assert!(s.validate().is_ok(), "({i}, {j})");
        }
    }

    #[test]
    fn pml_materials_uniform_along_normal() {
        let with = |shape: Shape| {
            let mut s = open_sample();
            s.geometry = vec![shape];
            s.validate()
        };
        let rect = |x0, y0, x1, y1| Shape::Rect {
            x0,
            y0,
            x1,
            y1,
            material: 1,
        };
        // A slab crossing the left layer along x is uniform along the normal.
        assert!(with(rect(0.0, 10.0, 20.0, 14.0)).is_ok());
        // A block ending inside the layer is not.
        assert_eq!(
            with(rect(3.0, 10.0, 20.0, 14.0)),
            Err(FdtdError::PmlNotUniform { i: 1, j: 10 })
        );
        // A block that only touches the interface column from inside is not either.
        assert!(with(rect(5.0, 10.0, 20.0, 14.0)).is_err());
        // A slab reaching the bottom-left corner region must fill the corner too.
        assert!(with(rect(0.0, 0.0, 20.0, 14.0)).is_ok());
        assert!(with(rect(0.0, 3.0, 20.0, 14.0)).is_err());
        // Wall nodes are PEC whatever the raster says, so they are not checked.
        assert!(with(rect(0.0, 0.0, 0.0, 29.0)).is_ok());
    }

    /// `periodic_y` round trips, is omitted from the JSON when false (so
    /// older files are unchanged), and changes the rules: rows 0 and ny - 1
    /// take sources, the layer exists on the x faces only, so it only has to
    /// fit along x, and only the x faces need uniform materials.
    #[test]
    fn periodic_y_rules() {
        assert!(!sample().to_json().unwrap().contains("periodic_y"));
        let mut s = open_sample();
        s.periodic_y = true;
        let text = s.to_json().unwrap();
        assert!(text.contains("\"periodic_y\": true"));
        assert_eq!(Scene::from_json(&text).unwrap(), s);

        for (i, j) in [(20, 0), (20, 29), (20, 2)] {
            let mut s = open_sample();
            s.periodic_y = true;
            (s.sources[0].i, s.sources[0].j) = (i, j);
            assert!(s.validate().is_ok(), "({i}, {j})");
            s.periodic_y = false;
            assert!(s.validate().is_err(), "({i}, {j}) without the wrap");
        }
        // Still rejected inside the x layer and on the x walls.
        let mut s = open_sample();
        s.periodic_y = true;
        (s.sources[0].i, s.sources[0].j) = (3, 0);
        assert_eq!(s.validate(), Err(FdtdError::SourceInPml { i: 3, j: 0 }));
        assert!(s.in_pml(3, 15) && !s.in_pml(20, 0) && !s.in_pml(20, 29));

        // A 14-cell layer does not fit 30 rows, but with the wrap it needs
        // only 2 * 14 + 3 = 31 <= 40 columns.
        s.sources[0].i = 20;
        s.pml.as_mut().unwrap().thickness = 14;
        assert!(s.validate().is_ok());
        s.pml.as_mut().unwrap().thickness = 19;
        assert!(matches!(
            s.validate(),
            Err(FdtdError::PmlTooThick { thickness: 19, .. })
        ));

        // A block touching the old y faces inside the would-be y layer is
        // fine; one ending inside an x layer is still rejected.
        let mut s = open_sample();
        s.periodic_y = true;
        s.geometry = vec![Shape::Rect {
            x0: 10.0,
            y0: 0.0,
            x1: 20.0,
            y1: 2.0,
            material: 1,
        }];
        assert!(s.validate().is_ok());
        s.geometry[0] = Shape::Rect {
            x0: 3.0,
            y0: 0.0,
            x1: 20.0,
            y1: 2.0,
            material: 1,
        };
        assert_eq!(s.validate(), Err(FdtdError::PmlNotUniform { i: 1, j: 0 }));
    }

    /// A 60 x 50 grid with a 5-cell layer and a +x plane wave box over
    /// [10, 50] x [10, 40], all in vacuum.
    fn tfsf_sample() -> Scene {
        let mut s = empty(60, 50, vec![]);
        s.pml = Some(crate::cpml::tests::spec(5));
        s.plane_wave = Some(PlaneWave {
            x0: 10,
            y0: 10,
            x1: 50,
            y1: 40,
            direction: Direction::PlusX,
            waveform: Waveform::Ricker {
                t0: 40.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
        });
        s
    }

    fn tfsf_fault(f: impl FnOnce(&mut Scene)) -> Result<()> {
        let mut s = tfsf_sample();
        f(&mut s);
        s.validate()
    }

    fn pw(s: &mut Scene) -> &mut PlaneWave {
        s.plane_wave.as_mut().unwrap()
    }

    #[test]
    fn plane_wave_json_round_trip_and_absent_by_default() {
        let s = tfsf_sample();
        let text = s.to_json().unwrap();
        assert!(text.contains("\"direction\": \"+x\""));
        assert_eq!(Scene::from_json(&text).unwrap(), s);
        assert!(!sample().to_json().unwrap().contains("plane_wave"));
        let typo = text.replace("\"amplitude\": 1.0", "\"amplitude\": 1.0, \"angle\": 0");
        assert!(
            Scene::from_json(&typo).is_err(),
            "unknown keys must be rejected"
        );
    }

    #[test]
    fn plane_wave_placement_rules() {
        use PlaneWaveFault::*;
        let err = |f| Err(FdtdError::PlaneWave(f));
        assert!(tfsf_sample().validate().is_ok());
        assert_eq!(tfsf_fault(|s| pw(s).x1 = 9), err(EmptyBox));
        assert_eq!(tfsf_fault(|s| pw(s).y0 = 41), err(EmptyBox));

        // Layer is 5 cells: node 5 is the interface, so a face at 6 leaves
        // one scattered-field node (5) outside the layer, and a face at 5 does not.
        assert!(tfsf_fault(|s| pw(s).x0 = 6).is_ok());
        assert_eq!(tfsf_fault(|s| pw(s).x0 = 5), err(FaceTooClose));
        assert!(tfsf_fault(|s| pw(s).y1 = 43).is_ok());
        assert_eq!(tfsf_fault(|s| pw(s).y1 = 44), err(FaceTooClose));
        assert_eq!(tfsf_fault(|s| pw(s).x1 = 59), err(FaceTooClose));
        // Without a layer the walls are the limit: column 0 is a wall.
        assert!(tfsf_fault(|s| (s.pml, pw(s).x0) = (None, 2)).is_ok());
        assert!(
            tfsf_fault(|s| {
                s.pml = None;
                pw(s).x0 = 1;
            })
            .is_err()
        );
        assert!(
            tfsf_fault(|s| {
                s.pml = None;
                pw(s).y0 = 0;
            })
            .is_err()
        );

        // Material 0 on both sides of every face; shapes strictly inside are fine.
        let block = |x0, y0, x1, y1| Shape::Rect {
            x0,
            y0,
            x1,
            y1,
            material: 1,
        };
        assert!(tfsf_fault(|s| s.geometry = vec![block(11.0, 11.0, 49.0, 39.0)]).is_ok());
        assert_eq!(
            tfsf_fault(|s| s.geometry = vec![block(9.0, 20.0, 9.0, 20.0)]),
            err(NotBackground)
        );
        assert_eq!(
            tfsf_fault(|s| s.geometry = vec![block(30.0, 40.0, 30.0, 40.0)]),
            err(NotBackground)
        );
        assert_eq!(
            tfsf_fault(|s| s.geometry = vec![block(30.0, 9.0, 30.0, 9.0)]),
            err(NotBackground)
        );

        // The auxiliary grid is lossless: material 0 must be too.
        let lossy = Material::Dielectric {
            eps_inf: 2.0,
            sigma: 0.01,
        };
        assert_eq!(tfsf_fault(|s| s.materials[0] = lossy), err(LossyBackground));
        assert_eq!(
            tfsf_fault(|s| s.materials[0] = Material::Pec),
            err(LossyBackground)
        );
        assert!(
            tfsf_fault(|s| s.materials[0] = Material::Dielectric {
                eps_inf: 2.25,
                sigma: 0.0
            })
            .is_ok()
        );

        // Waveform and amplitude are checked like a point source's.
        assert!(tfsf_fault(|s| pw(s).amplitude = f64::NAN).is_err());
        assert!(
            tfsf_fault(|s| pw(s).waveform = Waveform::Ricker {
                t0: 1.0,
                f_peak: -1.0
            })
            .is_err()
        );
    }

    #[test]
    fn plane_wave_with_periodic_y() {
        use PlaneWaveFault::*;
        let err = |f| Err(FdtdError::PlaneWave(f));
        let full = |s: &mut Scene| {
            s.periodic_y = true;
            (pw(s).y0, pw(s).y1) = (0, 49);
        };
        assert!(tfsf_fault(full).is_ok());
        assert_eq!(
            tfsf_fault(|s| {
                full(s);
                pw(s).y1 = 48;
            }),
            err(NotFullHeight)
        );
        for d in [Direction::PlusY, Direction::MinusY] {
            assert_eq!(
                tfsf_fault(|s| {
                    full(s);
                    pw(s).direction = d;
                }),
                err(WrapsAlongY)
            );
        }
        // The x faces still need material 0 on every row, the seam rows included.
        assert_eq!(
            tfsf_fault(|s| {
                full(s);
                s.geometry = vec![Shape::Rect {
                    x0: 9.0,
                    y0: 0.0,
                    x1: 9.0,
                    y1: 0.0,
                    material: 1,
                }];
            }),
            err(NotBackground)
        );
    }

    #[test]
    fn source_on_pec_rejected() {
        assert_eq!(
            rejects(|s| (s.sources[0].i, s.sources[0].j) = (25, 15)),
            FdtdError::SourceOnPec { i: 25, j: 15 }
        );
    }
}
