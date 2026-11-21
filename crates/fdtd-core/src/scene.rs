//! The scene: one value that fully determines a simulation. Both kernels are
//! built from it, and it round trips through JSON.
//!
//! Units: grid indices and geometry coordinates are in cells, with Ez node
//! (i, j) at (i, j). Time and frequency inside waveforms are normalized
//! (one time unit is dx / c0). `dx_phys_m` only matters at the UI boundary.

use crate::cpml::PmlSpec;
use crate::dispersion::courant_limit;
use crate::error::{FdtdError, Result};
use crate::material::{Material, PALETTE};
use crate::monitor::{Dft, FluxLine, Normal};
use crate::source::{SourceKind, Waveform};
use crate::tfsf::{Direction, PlaneWave, PlaneWaveFault};
use crate::units::{C0, Units};
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
    /// Freehand paint over the geometry. Absent means none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paint: Option<Paint>,
    /// How the field is drawn. Display only: no kernel reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<View>,
}

/// Display settings carried with a scene so a shared link looks the same.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct View {
    pub field: ViewField,
    /// Field value drawn at full color is `1 / gain`. Positive.
    pub gain: f64,
    /// Rescale the gain from the field's running peak.
    pub auto_gain: bool,
}

/// The quantity on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewField {
    /// Instantaneous Ez, diverging colormap.
    Ez,
    /// Running mean of Ez^2, sequential colormap.
    Intensity,
}

/// Freehand paint: an override material index per Ez node, laid out
/// `j * nx + i`, or `None` where the geometry shows through. Painted after
/// every shape. Stored dense, written to JSON as runs of equal entries,
/// `{"runs": [[null, 500], [2, 12], ...]}`, so a stroke costs a few runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PaintRuns", into = "PaintRuns")]
pub struct Paint {
    ids: Vec<Option<usize>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PaintRuns {
    runs: Vec<(Option<usize>, usize)>,
}

impl TryFrom<PaintRuns> for Paint {
    type Error = String;

    /// Refuses more than `MAX_NODES` entries before allocating them, since
    /// paint arrives in shared links.
    fn try_from(r: PaintRuns) -> std::result::Result<Self, String> {
        let mut total = 0usize;
        for &(_, n) in &r.runs {
            total = total.saturating_add(n);
        }
        if total > MAX_NODES {
            return Err(format!("paint has {total} entries, more than {MAX_NODES}"));
        }
        let mut ids = Vec::with_capacity(total);
        for (id, n) in r.runs {
            ids.extend(std::iter::repeat_n(id, n));
        }
        Ok(Paint { ids })
    }
}

impl From<Paint> for PaintRuns {
    fn from(p: Paint) -> Self {
        let mut runs: Vec<(Option<usize>, usize)> = Vec::new();
        for id in p.ids {
            match runs.last_mut() {
                Some((last, n)) if *last == id => *n += 1,
                _ => runs.push((id, 1)),
            }
        }
        PaintRuns { runs }
    }
}

/// What one `Scene::paint_stroke` changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrokeOutcome {
    /// Live interior nodes under the brush that took its material.
    pub painted: usize,
    /// Nodes under the brush left as they were: sources, TF/SF faces and
    /// the nodes across them.
    pub skipped: usize,
    /// `[i0, j0, i1, j1]`, inclusive, around every node whose paint was
    /// written; `None` when nothing was. Layer nodes that follow them are
    /// covered by `kernel::material_edit`.
    pub bounds: Option<[usize; 4]>,
}

/// Squared distance from point `p` to the segment `a` to `b`, cell units.
fn segment_dist2(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (a.0 + t * dx, a.1 + t * dy);
    (p.0 - qx).powi(2) + (p.1 - qy).powi(2)
}

impl Paint {
    /// `len` entries, none painted. A scene needs `nx * ny`.
    pub fn blank(len: usize) -> Paint {
        Paint {
            ids: vec![None; len],
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The override at node index `k` (`j * nx + i`), `None` if unpainted
    /// or out of range.
    pub fn get(&self, k: usize) -> Option<usize> {
        self.ids.get(k).copied().flatten()
    }

    /// Sets or clears the override at node index `k`; out of range does
    /// nothing.
    pub fn set(&mut self, k: usize, id: Option<usize>) {
        if let Some(slot) = self.ids.get_mut(k) {
            *slot = id;
        }
    }
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

impl Source {
    /// The value this source applies at normalized time `t`:
    /// `amplitude * waveform(t)`. Both kernels call this, so the GPU uploads
    /// exactly the CPU's samples.
    pub fn value_at(&self, t: f64) -> f64 {
        self.amplitude * self.waveform.value(t)
    }
}

/// A phased array of `amplitudes.len()` soft point sources. Element k sits
/// on Ez node `origin + k * step` (cell units, steps may be negative) and
/// radiates `amplitudes[k] * sin(2 pi f0 t + k * phase_step)`, ramped on over
/// `ramp_periods` periods (`Waveform::RampedCw`). So each element leads the
/// one before it by `phase_step` radians. Frequencies are normalized.
///
/// Returns ordinary sources; `Scene::validate` checks where they land.
pub fn phased_array(
    origin: (usize, usize),
    step: (i64, i64),
    phase_step: f64,
    amplitudes: &[f64],
    f0: f64,
    ramp_periods: f64,
) -> Result<Vec<Source>> {
    if amplitudes.is_empty() {
        return Err(FdtdError::InvalidParameter {
            name: "array elements",
            value: 0.0,
        });
    }
    finite("phase_step", phase_step)?;
    let mut out = Vec::with_capacity(amplitudes.len());
    for (k, &amplitude) in amplitudes.iter().enumerate() {
        finite("amplitude", amplitude)?;
        let (i, j) = (
            origin.0 as i64 + k as i64 * step.0,
            origin.1 as i64 + k as i64 * step.1,
        );
        if i < 0 || j < 0 {
            return Err(FdtdError::InvalidParameter {
                name: "array element index",
                value: i.min(j) as f64,
            });
        }
        let waveform = Waveform::RampedCw {
            f0,
            ramp_periods,
            phase: k as f64 * phase_step,
            start: 0.0,
        };
        waveform.validate()?;
        out.push(Source {
            i: i as usize,
            j: j as usize,
            waveform,
            amplitude,
            kind: SourceKind::Soft,
        });
    }
    Ok(out)
}

/// What the status strip shows, in physical units. See `Scene::status`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Status {
    pub nx: usize,
    pub ny: usize,
    pub dx_m: f64,
    pub steps: u64,
    /// Simulated time of the E field, `steps * dt`, seconds.
    pub time_s: f64,
    /// One entry per scene source, in order; `None` for a waveform with no
    /// center frequency (plain Gaussian).
    pub sources: Vec<Option<Drive>>,
    pub plane_wave: Option<Drive>,
}

/// The frequency of one source or plane wave and how well the grid
/// resolves it in the medium where it is launched.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Drive {
    pub f_hz: f64,
    pub vacuum_wavelength_m: f64,
    /// In the medium at the source node (material 0 for the plane wave).
    pub cells_per_wavelength: f64,
    /// Below 10 cells per wavelength, where dispersion error grows fast.
    pub under_resolved: bool,
}

/// Cells per wavelength below which the status strip warns.
pub const WARN_CELLS_PER_WAVELENGTH: f64 = 10.0;

/// Shortest vacuum wavelength a source edit accepts, in cells. On the axis
/// the grid stops propagating below about 2.8 to 3 cells for S up to
/// 1/sqrt(2); 4 keeps a margin. Under 10 cells is allowed with a warning.
pub const MIN_WAVELENGTH_CELLS: f64 = 4.0;

/// Longest vacuum wavelength a source edit accepts, in cells (f = 0.005).
/// The recommended CPML holds about -101 dB there and falls to -41 dB at
/// f = 0.0025 (`PROGRESS.md`, CPML reflection vs frequency).
pub const MAX_WAVELENGTH_CELLS: f64 = 200.0;

/// A live edit of one source; absent fields stay as they are. Wavelengths
/// are in vacuum, in cells, so `f = 1 / wavelength_cells` normalized.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePatch {
    pub i: Option<usize>,
    pub j: Option<usize>,
    /// For CW, Ricker and modulated Gaussian sources.
    pub wavelength_cells: Option<f64>,
    pub amplitude: Option<f64>,
    /// Radians, CW sources only.
    pub phase: Option<f64>,
}

impl SourcePatch {
    /// The patched source at normalized time `t_now`. A new CW frequency
    /// shifts the phase so `2 pi f0 t + phase` is unchanged at `t_now`: the
    /// drive keeps going without a step. Positions are not checked here.
    pub fn apply(&self, src: &Source, t_now: f64) -> Result<Source> {
        let mut out = *src;
        out.i = self.i.unwrap_or(src.i);
        out.j = self.j.unwrap_or(src.j);
        if let Some(a) = self.amplitude {
            finite("amplitude", a)?;
            out.amplitude = a;
        }
        if let Some(cells) = self.wavelength_cells {
            if !(MIN_WAVELENGTH_CELLS..=MAX_WAVELENGTH_CELLS).contains(&cells) {
                return Err(FdtdError::InvalidParameter {
                    name: "wavelength_cells",
                    value: cells,
                });
            }
            finite("t_now", t_now)?;
            let f = 1.0 / cells;
            out.waveform = match out.waveform {
                Waveform::RampedCw {
                    f0,
                    ramp_periods,
                    phase,
                    start,
                } => Waveform::RampedCw {
                    f0: f,
                    ramp_periods,
                    phase: phase + 2.0 * std::f64::consts::PI * (f0 - f) * t_now,
                    start,
                },
                Waveform::Ricker { t0, .. } => Waveform::Ricker { t0, f_peak: f },
                Waveform::ModulatedGaussian { t0, tau, .. } => {
                    Waveform::ModulatedGaussian { t0, tau, f0: f }
                }
                Waveform::Gaussian { .. } => {
                    return Err(FdtdError::InvalidParameter {
                        name: "wavelength_cells",
                        value: cells,
                    });
                }
            };
        }
        if let Some(p) = self.phase {
            match &mut out.waveform {
                Waveform::RampedCw { phase, .. } => {
                    finite("phase", p)?;
                    *phase = p;
                }
                _ => {
                    return Err(FdtdError::InvalidParameter {
                        name: "phase",
                        value: p,
                    });
                }
            }
        }
        out.waveform.validate()?;
        Ok(out)
    }
}

/// A wall with two slits, in cells: columns `x..x + thickness`, rows
/// `y0..=y1`, opened wherever `|j - (center -+ separation / 2)| < width / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlitPair {
    pub x: usize,
    pub thickness: usize,
    pub y0: usize,
    pub y1: usize,
    /// Row halfway between the slit centers; may be fractional.
    pub center: f64,
    /// Center to center distance of the slits, in cells.
    pub separation: f64,
    /// Width of each slit, in cells.
    pub width: f64,
    pub material: usize,
}

/// The wall of a slit pair as ordinary rects, one per run of closed rows,
/// with integer edges so the open rows are exact. Overlapping slits just
/// merge into one opening. `Scene::validate` checks where the rects land.
pub fn slit_pair(spec: &SlitPair) -> Result<Vec<Shape>> {
    let SlitPair {
        x,
        thickness,
        y0,
        y1,
        center,
        separation,
        width,
        material,
    } = *spec;
    finite("center", center)?;
    for (name, v) in [("separation", separation), ("width", width)] {
        if !(v.is_finite() && v >= 0.0) {
            return Err(FdtdError::InvalidParameter { name, value: v });
        }
    }
    if width == 0.0 {
        return Err(FdtdError::InvalidParameter {
            name: "width",
            value: width,
        });
    }
    if thickness == 0 {
        return Err(FdtdError::InvalidParameter {
            name: "thickness",
            value: 0.0,
        });
    }
    if y1 < y0 {
        return Err(FdtdError::InvalidParameter {
            name: "y1",
            value: y1 as f64,
        });
    }
    let open = |j: usize| {
        [center - separation / 2.0, center + separation / 2.0]
            .iter()
            .any(|c| (j as f64 - c).abs() < width / 2.0)
    };
    let mut shapes = Vec::new();
    let mut run: Option<usize> = None;
    for j in y0..=y1 + 1 {
        let closed = j <= y1 && !open(j);
        match (closed, run) {
            (true, None) => run = Some(j),
            (false, Some(start)) => {
                shapes.push(Shape::Rect {
                    x0: x as f64,
                    y0: start as f64,
                    x1: (x + thickness - 1) as f64,
                    y1: (j - 1) as f64,
                    material,
                });
                run = None;
            }
            _ => {}
        }
    }
    Ok(shapes)
}

/// A field monitor. Frequencies are normalized (cycles per unit time).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Monitor {
    /// Records Ez at node (i, j) once at n = 0 and after every step.
    Point { i: usize, j: usize },
    /// Running DFT of Ez at node (i, j), sampled at t = n S after every step.
    Dft { i: usize, j: usize, freqs: Vec<f64> },
    /// Poynting flux spectrum through the Ez nodes `from..=to` of column `at`
    /// (x normal) or row `at` (y normal). See `monitor::FluxLine`.
    Flux {
        normal: Normal,
        at: usize,
        from: usize,
        to: usize,
        freqs: Vec<f64>,
    },
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

    /// `[x0, y0, x1, y1]` around the shape, cells, edges included: every
    /// node the shape can contain lies inside. Assumes a validated shape.
    pub fn bounds(&self) -> [f64; 4] {
        match self {
            Shape::Rect { x0, y0, x1, y1, .. } => [*x0, *y0, *x1, *y1],
            Shape::Circle { cx, cy, r, .. } => [cx - r, cy - r, cx + r, cy + r],
            Shape::Polygon { points, .. } => points.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |b, p| {
                    [
                        b[0].min(p[0]),
                        b[1].min(p[1]),
                        b[2].max(p[0]),
                        b[3].max(p[1]),
                    ]
                },
            ),
        }
    }

    /// The material index the shape paints.
    pub fn material(&self) -> usize {
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
    /// The node whose material Ez node (i, j) takes: itself in the interior,
    /// and in the CPML layer (or on the wall ring) the interface node it is
    /// clamped to along each normal (along x only with periodic y). So the
    /// layer always continues the medium it terminates, uniform along its
    /// normal, whatever the shapes or paint say inside it.
    pub fn material_node(&self, i: usize, j: usize) -> (usize, usize) {
        let d = self.pml_thickness();
        let GridSpec { nx, ny, .. } = self.grid;
        if d == 0 || nx < 2 * d + 1 || (!self.periodic_y && ny < 2 * d + 1) {
            return (i, j);
        }
        let cj = if self.periodic_y {
            j
        } else {
            j.clamp(d, ny - 1 - d)
        };
        (i.clamp(d, nx - 1 - d), cj)
    }

    /// Material index of Ez node (i, j), read at `material_node(i, j)`: its
    /// paint if painted, else the last shape containing that position, else
    /// 0. Assumes a validated scene.
    pub fn material_at(&self, i: usize, j: usize) -> usize {
        let (i, j) = self.material_node(i, j);
        if let Some(id) = self
            .paint
            .as_ref()
            .and_then(|p| p.get(j * self.grid.nx + i))
        {
            return id;
        }
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

    /// Makes sure every `material::PALETTE` entry is in the table, appending
    /// the missing ones at the end, and returns the table index of each
    /// entry in palette order. Existing entries and indices never change, so
    /// the rasterized grid stays the same. Entries match by exact value.
    pub fn with_palette(&mut self) -> Vec<usize> {
        PALETTE
            .iter()
            .map(|(_, m)| match self.materials.iter().position(|t| t == m) {
                Some(k) => k,
                None => {
                    self.materials.push(*m);
                    self.materials.len() - 1
                }
            })
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
        // Before anything below reads materials through `material_at`.
        if let Some(paint) = &self.paint {
            if paint.len() != nx * ny {
                return Err(FdtdError::InvalidParameter {
                    name: "paint length",
                    value: paint.len() as f64,
                });
            }
            if let Some(id) = paint
                .ids
                .iter()
                .flatten()
                .find(|&&id| id >= self.materials.len())
            {
                return Err(FdtdError::UnknownMaterial { id: *id });
            }
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
        }
        for src in &self.sources {
            self.check_source(src)?;
        }
        for m in &self.monitors {
            self.check_monitor(m)?;
        }
        if let Some(pw) = &self.plane_wave {
            self.check_plane_wave(pw)?;
        }
        if let Some(v) = &self.view
            && !(v.gain.is_finite() && v.gain > 0.0)
        {
            return Err(FdtdError::InvalidParameter {
                name: "gain",
                value: v.gain,
            });
        }
        Ok(())
    }

    /// One source on its own: waveform, amplitude, and a live interior node
    /// outside the layer and off PEC.
    pub fn check_source(&self, src: &Source) -> Result<()> {
        src.waveform.validate()?;
        finite("amplitude", src.amplitude)?;
        // Wall nodes are PEC and never updated, so a source there is wrong.
        if !(1..self.grid.nx - 1).contains(&src.i) || !self.live_rows().contains(&src.j) {
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
        Ok(())
    }

    /// The status strip numbers after `steps` steps, all through `Units`.
    /// Assumes a validated scene.
    pub fn status(&self, steps: u64) -> Result<Status> {
        let units = Units::new(self.grid.dx_phys_m, self.grid.courant)?;
        let drive = |w: &Waveform, eps: f64| {
            w.frequency().map(|f| {
                let f_hz = units.hz_from_omega_n(2.0 * std::f64::consts::PI * f);
                let cells_per_wavelength = units.cells_per_wavelength(f_hz, eps);
                Drive {
                    f_hz,
                    vacuum_wavelength_m: C0 / f_hz,
                    cells_per_wavelength,
                    under_resolved: cells_per_wavelength < WARN_CELLS_PER_WAVELENGTH,
                }
            })
        };
        Ok(Status {
            nx: self.grid.nx,
            ny: self.grid.ny,
            dx_m: units.dx_m(),
            steps,
            time_s: units.s_from_steps(steps as f64),
            sources: self
                .sources
                .iter()
                .map(|s| {
                    drive(
                        &s.waveform,
                        self.materials[self.material_at(s.i, s.j)].eps(),
                    )
                })
                .collect(),
            plane_wave: self
                .plane_wave
                .as_ref()
                .and_then(|pw| drive(&pw.waveform, self.materials[0].eps())),
        })
    }

    /// Applies `patch` to source `index` of a validated scene at normalized
    /// time `t_now`, checking only that source, so it is cheap enough for a
    /// slider or a drag. On error the scene is unchanged.
    pub fn patch_source(&mut self, index: usize, patch: &SourcePatch, t_now: f64) -> Result<()> {
        let old = self.sources.get(index).ok_or(FdtdError::InvalidParameter {
            name: "source index",
            value: index as f64,
        })?;
        let new = patch.apply(old, t_now)?;
        self.check_source(&new)?;
        self.sources[index] = new;
        Ok(())
    }

    /// Placement of one monitor. Points and DFT nodes must be grid nodes. An
    /// x-normal flux line averages Hy from `at - 1/2` and `at + 1/2`, so `at`
    /// is in `1..=nx-2`; a y-normal line averages Hx from `at -+ 1/2`, so `at`
    /// is in `1..=ny-2`, or any row with periodic y, where the edges wrap. The
    /// segment runs over existing nodes, and frequency lists are non-empty,
    /// finite and positive.
    fn check_monitor(&self, m: &Monitor) -> Result<()> {
        let GridSpec { nx, ny, .. } = self.grid;
        let out = |i, j| {
            Err(FdtdError::OutOfDomain {
                what: "monitor",
                i,
                j,
            })
        };
        match m {
            &Monitor::Point { i, j } => {
                if i >= nx || j >= ny {
                    return out(i, j);
                }
            }
            Monitor::Dft { i, j, freqs } => {
                Dft::new(freqs, 1)?;
                if *i >= nx || *j >= ny {
                    return out(*i, *j);
                }
            }
            Monitor::Flux {
                normal,
                at,
                from,
                to,
                freqs,
            } => {
                FluxLine::new(*normal, *at, *from, *to, freqs)?;
                let (at, to) = (*at, *to);
                match normal {
                    Normal::X => {
                        if !(1..nx - 1).contains(&at) || to >= ny {
                            return out(at, to);
                        }
                    }
                    Normal::Y => {
                        let row_ok = self.periodic_y && at < ny || (1..ny - 1).contains(&at);
                        if !row_ok || to >= nx {
                            return out(to, at);
                        }
                    }
                }
            }
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
        for (i, j) in self.tfsf_guard_nodes(pw) {
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

    /// Every Ez node on a TF/SF face and the scattered-field node across it,
    /// as signed indices so a face on row or column 0 shows up as out of
    /// range, not as a wrap. These must stay in material 0.
    fn tfsf_guard_nodes(&self, pw: &PlaneWave) -> Vec<(i64, i64)> {
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
        nodes
    }

    /// Paints `material` on every live Ez node within `radius` cells of the
    /// segment `from` to `to` (cell units, node (i, j) at (i, j)): a capsule
    /// brush. `material` 0 erases to the background. Nodes the scene needs
    /// as they are, sources and the TF/SF faces with the nodes across them,
    /// are skipped and counted. The brush never paints inside the CPML: the
    /// layer takes its material from the interface (`material_node`), so a
    /// wall drawn up to the layer runs on through it. Off-grid parts of a
    /// stroke are ignored. A stroke on a valid scene leaves it valid.
    pub fn paint_stroke(
        &mut self,
        from: (f64, f64),
        to: (f64, f64),
        radius: f64,
        material: usize,
    ) -> Result<StrokeOutcome> {
        for (name, v) in [
            ("stroke x", from.0),
            ("stroke y", from.1),
            ("stroke x", to.0),
            ("stroke y", to.1),
        ] {
            finite(name, v)?;
        }
        if !(radius.is_finite() && radius >= 0.0) {
            return Err(FdtdError::InvalidParameter {
                name: "brush radius",
                value: radius,
            });
        }
        if material >= self.materials.len() {
            return Err(FdtdError::UnknownMaterial { id: material });
        }
        let GridSpec { nx, ny, .. } = self.grid;
        let rows = self.live_rows();
        let mut protected: std::collections::HashSet<(usize, usize)> =
            self.sources.iter().map(|s| (s.i, s.j)).collect();
        if let Some(pw) = &self.plane_wave {
            protected.extend(
                self.tfsf_guard_nodes(pw)
                    .into_iter()
                    .filter(|&(i, j)| i >= 0 && j >= 0)
                    .map(|(i, j)| (i as usize, j as usize)),
            );
        }
        let mut out = StrokeOutcome {
            painted: 0,
            skipped: 0,
            bounds: None,
        };
        // The brush's bounding box, clipped to the live nodes.
        let lo_x = (from.0.min(to.0) - radius).ceil().max(1.0);
        let hi_x = (from.0.max(to.0) + radius).floor().min((nx - 2) as f64);
        let lo_y = (from.1.min(to.1) - radius).ceil().max(rows.start as f64);
        let hi_y = (from.1.max(to.1) + radius)
            .floor()
            .min((rows.end - 1) as f64);
        if lo_x > hi_x || lo_y > hi_y {
            return Ok(out);
        }
        let mut hit = Vec::new();
        for j in lo_y as usize..=hi_y as usize {
            for i in lo_x as usize..=hi_x as usize {
                if self.in_pml(i, j)
                    || segment_dist2((i as f64, j as f64), from, to) > radius * radius
                {
                    continue;
                }
                if protected.contains(&(i, j)) {
                    out.skipped += 1;
                } else {
                    hit.push((i, j));
                }
            }
        }
        out.painted = hit.len();
        if hit.is_empty() {
            return Ok(out);
        }
        let paint = self.paint.get_or_insert_with(|| Paint::blank(nx * ny));
        let mut b = [usize::MAX, usize::MAX, 0, 0];
        for (i, j) in hit {
            paint.set(j * nx + i, Some(material));
            b = [b[0].min(i), b[1].min(j), b[2].max(i), b[3].max(j)];
        }
        out.bounds = Some(b);
        Ok(out)
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
            paint: None,
            view: None,
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
    fn pml_takes_the_interface_material() {
        // open_sample: 40 x 30 with a 5-cell layer, interior 5..=34 x 5..=24.
        let with = |shape: Shape| {
            let mut s = open_sample();
            s.geometry = vec![shape];
            s.validate().unwrap();
            s
        };
        let rect = |x0, y0, x1, y1| Shape::Rect {
            x0,
            y0,
            x1,
            y1,
            material: 1,
        };
        // A block ending inside the layer, or touching the interface from
        // inside, fills the layer out to the wall along its rows.
        for x0 in [3.0, 5.0] {
            let s = with(rect(x0, 10.0, 20.0, 14.0));
            for i in 0..=4 {
                assert_eq!(s.material_at(i, 12), 1, "x0 {x0}, column {i}");
            }
            assert_eq!(s.material_at(2, 9), 0);
        }
        // Inside the layer only, it reaches no interface node: nothing.
        let s = with(rect(1.0, 10.0, 3.0, 14.0));
        assert_eq!(s.material_at(2, 12), 0);
        // At a corner the whole corner block follows the corner node.
        let s = with(rect(5.0, 5.0, 20.0, 14.0));
        assert_eq!((s.material_at(1, 1), s.material_at(4, 3)), (1, 1));
        let s = with(rect(6.0, 5.0, 20.0, 14.0));
        assert_eq!(s.material_at(1, 1), 0);
        let base = open_sample();
        assert_eq!(base.material_node(20, 12), (20, 12));
        assert_eq!(base.material_node(0, 29), (5, 24));
    }

    /// `periodic_y` round trips, is omitted from the JSON when false (so
    /// older files are unchanged), and changes the rules: rows 0 and ny - 1
    /// take sources, the layer exists on the x faces only, so it only has to
    /// fit along x, and only x is clamped for layer materials.
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
        // fine, and one ending inside an x layer runs out to the x wall.
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
        // With the wrap only x is clamped: the block runs out along x.
        s.validate().unwrap();
        assert_eq!((s.material_at(1, 0), s.material_at(1, 29)), (1, 0));
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

    /// Element k sits at origin + k * step, leads by k * phase_step, and has
    /// amplitude k; each one is a soft ramped CW source at f0.
    #[test]
    fn phased_array_elements() {
        let srcs = phased_array((10, 20), (0, 3), 0.25, &[1.0, 0.5, 2.0, 0.0], 0.04, 4.0).unwrap();
        assert_eq!(srcs.len(), 4);
        for (k, s) in srcs.iter().enumerate() {
            assert_eq!((s.i, s.j), (10, 20 + 3 * k));
            assert_eq!(s.kind, SourceKind::Soft);
            assert_eq!(s.amplitude, [1.0, 0.5, 2.0, 0.0][k]);
            assert_eq!(
                s.waveform,
                Waveform::RampedCw {
                    f0: 0.04,
                    ramp_periods: 4.0,
                    phase: 0.25 * k as f64,
                    start: 0.0,
                }
            );
        }
        // Steps may run backwards, as long as no element leaves the grid.
        let back = phased_array((10, 20), (-2, -1), 0.0, &[1.0; 3], 0.04, 4.0).unwrap();
        assert_eq!((back[2].i, back[2].j), (6, 18));
        // An expanded array is just point sources, so the scene checks it.
        let mut s = sample();
        s.sources = phased_array((5, 20), (3, 0), 0.5, &[1.0; 4], 0.04, 4.0).unwrap();
        assert!(s.validate().is_ok());
        s.sources = phased_array((30, 20), (3, 0), 0.5, &[1.0; 4], 0.04, 4.0).unwrap();
        assert!(matches!(
            s.validate(),
            Err(FdtdError::OutOfDomain { what: "source", .. })
        ));
    }

    #[test]
    fn slit_pair_open_cells() {
        // Slits centered on rows 11 and 19, 3 rows wide: 10..=12 and 18..=20.
        let spec = SlitPair {
            x: 20,
            thickness: 2,
            y0: 1,
            y1: 28,
            center: 15.0,
            separation: 8.0,
            width: 3.0,
            material: 2,
        };
        let shapes = slit_pair(&spec).unwrap();
        assert_eq!(shapes.len(), 3);
        let s = empty(40, 30, shapes);
        let walls = cells(&s, 2);
        for i in [20, 21] {
            let rows: Vec<usize> = walls.iter().filter(|c| c.0 == i).map(|c| c.1).collect();
            let want: Vec<usize> = (1..=28)
                .filter(|j| !(10..=12).contains(j) && !(18..=20).contains(j))
                .collect();
            assert_eq!(rows, want, "column {i}");
        }
        assert!(walls.iter().all(|c| c.0 == 20 || c.0 == 21));
        // An even width opens the rows strictly inside, here row 11 only.
        let narrow = slit_pair(&SlitPair { width: 2.0, ..spec }).unwrap();
        let s = empty(40, 30, narrow);
        assert!(!cells(&s, 2).contains(&(20, 11)));
        assert!(cells(&s, 2).contains(&(20, 10)) && cells(&s, 2).contains(&(20, 12)));
    }

    #[test]
    fn palette_appends_only_what_is_missing() {
        // The sample has vacuum, eps 4 and PEC; vacuum and PEC are reused.
        let mut s = sample();
        let before = s.rasterize();
        let ids = s.with_palette();
        assert_eq!(ids.len(), PALETTE.len());
        for (k, (name, m)) in PALETTE.iter().enumerate() {
            assert_eq!(s.materials[ids[k]], *m, "{name}");
        }
        assert_eq!(ids[0], 0);
        assert_eq!(ids[3], 2);
        assert_eq!(s.materials.len(), 3 + 3);
        assert_eq!(s.materials[1], sample().materials[1]);
        assert_eq!(s.rasterize(), before);
        s.validate().unwrap();
        // Twice is the same as once.
        let again = s.clone().with_palette();
        assert_eq!(again, ids);
        let mut twice = s.clone();
        twice.with_palette();
        assert_eq!(twice, s);
    }

    #[test]
    fn view_round_trips_and_is_checked() {
        let mut s = sample();
        s.view = Some(View {
            field: ViewField::Intensity,
            gain: 2.5,
            auto_gain: false,
        });
        let text = s.to_json().unwrap();
        assert!(text.contains(r#""field": "intensity""#));
        assert_eq!(Scene::from_json(&text).unwrap(), s);
        assert!(!sample().to_json().unwrap().contains("view"));
        for gain in [0.0, -1.0, f64::NAN] {
            let mut bad = s.clone();
            bad.view.as_mut().unwrap().gain = gain;
            assert!(bad.validate().is_err(), "gain {gain}");
        }
    }

    #[test]
    fn status_in_physical_units() {
        // dx = 1 mm, S = 0.5. The Ricker at 0.05 is 14.99 GHz, 20 mm, 20
        // cells in vacuum. A CW at 0.08 on the eps 4 node (7, 8) is 12.5 cells
        // in vacuum, 6.25 in the medium: under-resolved. A Gaussian has none.
        let mut s = sample();
        let mut cw = s.sources[0];
        (cw.i, cw.j) = (7, 8);
        cw.waveform = Waveform::RampedCw {
            f0: 0.08,
            ramp_periods: 3.0,
            phase: 0.0,
            start: 0.0,
        };
        let mut g = s.sources[0];
        g.waveform = Waveform::Gaussian {
            t0: 60.0,
            tau: 10.0,
        };
        s.sources.extend([cw, g]);
        let st = s.status(1000).unwrap();
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        assert_eq!((st.nx, st.ny, st.steps), (40, 30, 1000));
        assert!(rel(st.time_s, 1000.0 * 0.5 * 1e-3 / 299_792_458.0) < 1e-15);
        let r = st.sources[0].unwrap();
        assert!(rel(r.f_hz, 0.05 * 299_792_458.0 / 1e-3) < 1e-15);
        assert!(rel(r.vacuum_wavelength_m, 0.02) < 1e-15);
        assert!(rel(r.cells_per_wavelength, 20.0) < 1e-15);
        assert!(!r.under_resolved);
        let c = st.sources[1].unwrap();
        assert!(rel(c.cells_per_wavelength, 6.25) < 1e-15);
        assert!(c.under_resolved);
        assert_eq!(st.sources[2], None);
        assert_eq!(st.plane_wave, None);
        let st = tfsf_sample().status(0).unwrap();
        assert!(rel(st.plane_wave.unwrap().cells_per_wavelength, 20.0) < 1e-15);
    }

    fn cw_sample() -> Scene {
        let mut s = sample();
        s.sources[0].waveform = Waveform::RampedCw {
            f0: 0.05,
            ramp_periods: 3.0,
            phase: 0.0,
            start: 0.0,
        };
        s
    }

    fn wavelength(cells: f64) -> SourcePatch {
        SourcePatch {
            wavelength_cells: Some(cells),
            ..Default::default()
        }
    }

    #[test]
    fn source_patch_sets_wavelength() {
        let mut s = cw_sample();
        s.patch_source(0, &wavelength(40.0), 0.0).unwrap();
        assert!(matches!(s.sources[0].waveform, Waveform::RampedCw { f0, .. } if f0 == 1.0 / 40.0));
        let mut s = sample();
        s.patch_source(0, &wavelength(25.0), 0.0).unwrap();
        assert!(matches!(s.sources[0].waveform, Waveform::Ricker { f_peak, .. } if f_peak == 0.04));
        // The limits: 4 cells (above the grid cutoff) to 200 (the CPML floor).
        for (cells, ok) in [(3.99, false), (4.0, true), (200.0, true), (200.01, false)] {
            assert_eq!(
                cw_sample().patch_source(0, &wavelength(cells), 0.0).is_ok(),
                ok,
                "{cells}"
            );
        }
        // A plain Gaussian has no frequency, and only CW has a phase.
        s.sources[0].waveform = Waveform::Gaussian {
            t0: 60.0,
            tau: 10.0,
        };
        assert!(s.patch_source(0, &wavelength(30.0), 0.0).is_err());
        let phase = SourcePatch {
            phase: Some(1.0),
            ..Default::default()
        };
        assert!(sample().patch_source(0, &phase, 0.0).is_err());
        assert!(sample().patch_source(1, &phase, 0.0).is_err());
    }

    #[test]
    fn source_patch_keeps_cw_continuous() {
        // Past the ramp, a new frequency picks up the old phase at t_now, so
        // the drive has no step that would ring across the band.
        for t in [100.0, 1234.5, 20000.0] {
            let mut s = cw_sample();
            let before = s.sources[0].value_at(t);
            s.patch_source(0, &wavelength(33.0), t).unwrap();
            let after = s.sources[0].value_at(t);
            assert!((after - before).abs() < 1e-9, "t {t}: {before} vs {after}");
        }
    }

    #[test]
    fn source_patch_moves_and_checks_position() {
        let mut s = sample();
        let to = |i, j| SourcePatch {
            i: Some(i),
            j: Some(j),
            ..Default::default()
        };
        s.patch_source(0, &to(3, 4), 0.0).unwrap();
        assert_eq!((s.sources[0].i, s.sources[0].j), (3, 4));
        // Onto the PEC circle: refused, the source stays where it was.
        assert_eq!(
            s.patch_source(0, &to(25, 15), 0.0),
            Err(FdtdError::SourceOnPec { i: 25, j: 15 })
        );
        assert_eq!((s.sources[0].i, s.sources[0].j), (3, 4));
        assert!(s.patch_source(0, &to(0, 4), 0.0).is_err());
        let amp = SourcePatch {
            amplitude: Some(2.5),
            ..Default::default()
        };
        s.patch_source(0, &amp, 0.0).unwrap();
        assert_eq!(s.sources[0].amplitude, 2.5);
        s.validate().unwrap();
    }

    #[test]
    fn shape_bounds_hold_every_node() {
        for shape in sample().geometry {
            let [x0, y0, x1, y1] = shape.bounds();
            for j in 0..30 {
                for i in 0..40 {
                    let (x, y) = (i as f64, j as f64);
                    if shape.contains(x, y) {
                        assert!(
                            x0 <= x && x <= x1 && y0 <= y && y <= y1,
                            "{shape:?} ({i}, {j})"
                        );
                    }
                }
            }
        }
        let tri = Shape::Polygon {
            points: vec![[30.0, 2.0], [38.0, 2.0], [34.0, 8.0]],
            material: 1,
        };
        assert_eq!(tri.bounds(), [30.0, 2.0, 38.0, 8.0]);
    }

    #[test]
    fn slit_pair_rejects_bad_input() {
        let spec = SlitPair {
            x: 20,
            thickness: 2,
            y0: 1,
            y1: 28,
            center: 15.0,
            separation: 8.0,
            width: 3.0,
            material: 2,
        };
        assert!(
            slit_pair(&SlitPair {
                thickness: 0,
                ..spec
            })
            .is_err()
        );
        assert!(slit_pair(&SlitPair { width: 0.0, ..spec }).is_err());
        assert!(slit_pair(&SlitPair { y1: 0, ..spec }).is_err());
        assert!(
            slit_pair(&SlitPair {
                center: f64::INFINITY,
                ..spec
            })
            .is_err()
        );
        assert!(
            slit_pair(&SlitPair {
                separation: -1.0,
                ..spec
            })
            .is_err()
        );
    }

    #[test]
    fn phased_array_rejects_bad_input() {
        let ok = |o, st, ph, a: &[f64], f, r| phased_array(o, st, ph, a, f, r);
        assert!(
            ok((5, 5), (1, 0), 0.0, &[], 0.04, 4.0).is_err(),
            "no elements"
        );
        assert!(
            ok((5, 5), (-3, 0), 0.0, &[1.0; 3], 0.04, 4.0).is_err(),
            "below 0"
        );
        assert!(ok((5, 5), (1, 0), f64::NAN, &[1.0], 0.04, 4.0).is_err());
        assert!(ok((5, 5), (1, 0), 0.0, &[1.0, f64::INFINITY], 0.04, 4.0).is_err());
        assert!(ok((5, 5), (1, 0), 0.0, &[1.0], -0.04, 4.0).is_err());
        assert!(
            ok((5, 5), (1, 0), 0.0, &[1.0], 0.04, 2.0).is_err(),
            "ramp under 3"
        );
    }

    /// DFT and flux monitors round trip through their tagged JSON form and
    /// are placed by the same rules the kernels need: a DFT node inside the
    /// grid; an x-normal line on a column with an Hy edge on both sides
    /// (1..=nx-2), a y-normal line on a live row with an Hx edge on both
    /// sides (1..=ny-2, any row with periodic y), segments inside the grid,
    /// and valid frequency lists.
    #[test]
    fn spectral_monitors() {
        use crate::monitor::Normal;
        let freqs = vec![0.02, 0.05];
        let mut s = sample();
        s.monitors.push(Monitor::Dft {
            i: 30,
            j: 20,
            freqs: freqs.clone(),
        });
        s.monitors.push(Monitor::Flux {
            normal: Normal::X,
            at: 20,
            from: 1,
            to: 28,
            freqs: freqs.clone(),
        });
        let text = s.to_json().unwrap();
        assert!(text.contains("\"type\": \"flux\"") && text.contains("\"normal\": \"x\""));
        assert_eq!(Scene::from_json(&text).unwrap(), s);

        let flux = |normal, at, from, to| Monitor::Flux {
            normal,
            at,
            from,
            to,
            freqs: vec![0.05],
        };
        let check = |m: Monitor, periodic: bool| {
            let mut s = sample();
            s.periodic_y = periodic;
            s.monitors = vec![m];
            s.validate()
        };
        // sample() is 40 x 30.
        for (m, ok) in [
            (flux(Normal::X, 1, 0, 29), true),
            (flux(Normal::X, 38, 0, 29), true),
            (flux(Normal::X, 0, 0, 29), false),
            (flux(Normal::X, 39, 0, 29), false),
            (flux(Normal::X, 20, 0, 30), false),
            (flux(Normal::X, 20, 5, 4), false),
            (flux(Normal::Y, 1, 0, 39), true),
            (flux(Normal::Y, 28, 0, 39), true),
            (flux(Normal::Y, 0, 0, 39), false),
            (flux(Normal::Y, 29, 0, 39), false),
            (flux(Normal::Y, 10, 0, 40), false),
        ] {
            assert_eq!(check(m.clone(), false).is_ok(), ok, "{m:?}");
        }
        assert!(check(flux(Normal::Y, 0, 0, 39), true).is_ok());
        assert!(check(flux(Normal::Y, 29, 0, 39), true).is_ok());
        let dft = |i, j, freqs: Vec<f64>| Monitor::Dft { i, j, freqs };
        assert!(check(dft(39, 29, vec![0.1]), false).is_ok());
        assert!(check(dft(40, 0, vec![0.1]), false).is_err());
        assert!(check(dft(5, 5, vec![]), false).is_err());
        assert!(check(dft(5, 5, vec![0.1, -0.1]), false).is_err());
        assert!(
            check(
                Monitor::Flux {
                    normal: Normal::X,
                    at: 5,
                    from: 0,
                    to: 3,
                    freqs: vec![f64::NAN]
                },
                false
            )
            .is_err()
        );
    }

    #[test]
    fn source_on_pec_rejected() {
        assert_eq!(
            rejects(|s| (s.sources[0].i, s.sources[0].j) = (25, 15)),
            FdtdError::SourceOnPec { i: 25, j: 15 }
        );
    }

    fn painted(cells: &[(usize, usize)]) -> Paint {
        let mut p = Paint::blank(40 * 30);
        for &(k, id) in cells {
            p.set(k, Some(id));
        }
        p
    }

    #[test]
    fn paint_json_is_runs() {
        let mut s = sample();
        s.paint = Some(painted(&[(5, 2), (6, 2), (1199, 0)]));
        let text = s.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v["paint"]["runs"],
            serde_json::json!([[null, 5], [2, 2], [null, 1192], [0, 1]])
        );
        assert_eq!(Scene::from_json(&text).unwrap(), s);
        // No paint writes no key, so v1 files stay byte for byte the same.
        assert!(!sample().to_json().unwrap().contains("paint"));
    }

    #[test]
    fn paint_overrides_geometry() {
        let mut s = sample();
        let plain = s.rasterize();
        s.paint = Some(Paint::blank(40 * 30));
        assert_eq!(s.rasterize(), plain);
        // (7, 8) sits in the eps 4 rect, (20, 25) in vacuum.
        assert_eq!(plain[8 * 40 + 7], 1);
        s.paint = Some(painted(&[(8 * 40 + 7, 0), (25 * 40 + 20, 2)]));
        assert_eq!(s.material_at(7, 8), 0);
        assert_eq!(s.material_at(20, 25), 2);
        let r = s.rasterize();
        let changed: Vec<usize> = (0..r.len()).filter(|&k| r[k] != plain[k]).collect();
        assert_eq!(changed, vec![8 * 40 + 7, 25 * 40 + 20]);
    }

    #[test]
    fn paint_validation() {
        assert_eq!(
            rejects(|s| s.paint = Some(Paint::blank(10))),
            FdtdError::InvalidParameter {
                name: "paint length",
                value: 10.0
            }
        );
        assert_eq!(
            rejects(|s| s.paint = Some(painted(&[(0, 3)]))),
            FdtdError::UnknownMaterial { id: 3 }
        );
        // Paint counts for every rule that reads materials: PEC under a source.
        assert_eq!(
            rejects(|s| s.paint = Some(painted(&[(20 * 40 + 12, 2)]))),
            FdtdError::SourceOnPec { i: 12, j: 20 }
        );
    }

    #[test]
    fn stroke_paints_exact_cells() {
        // Every node within 1 of the segment (10, 10) to (14, 10).
        let mut s = empty(40, 30, vec![]);
        let out = s.paint_stroke((10.0, 10.0), (14.0, 10.0), 1.0, 2).unwrap();
        let mut want: Vec<(usize, usize)> = (9..=15).map(|i| (i, 10)).collect();
        want.extend((10..=14).flat_map(|i| [(i, 9), (i, 11)]));
        want.sort_by_key(|&(i, j)| (j, i));
        assert_eq!(cells(&s, 2), want);
        assert_eq!(
            out,
            StrokeOutcome {
                painted: 17,
                skipped: 0,
                bounds: Some([9, 9, 15, 11])
            }
        );
        // A diagonal with radius 1/2 takes the diagonal nodes only: the
        // neighbors sit 1/sqrt(2) away.
        let mut s = empty(40, 30, vec![]);
        s.paint_stroke((5.0, 5.0), (8.0, 8.0), 0.5, 1).unwrap();
        assert_eq!(cells(&s, 1), vec![(5, 5), (6, 6), (7, 7), (8, 8)]);
        // Erasing writes material 0 over a shape.
        let mut s = sample();
        s.paint_stroke((7.0, 8.0), (7.0, 8.0), 0.0, 0).unwrap();
        assert_eq!(s.material_at(7, 8), 0);
        assert_eq!(s.material_at(8, 8), 1);
    }

    #[test]
    fn stroke_skips_protected_nodes() {
        // PEC over the source at (12, 20) would make the scene invalid.
        let mut s = sample();
        let out = s.paint_stroke((10.0, 20.0), (14.0, 20.0), 0.0, 2).unwrap();
        assert_eq!((out.painted, out.skipped), (4, 1));
        assert_eq!(s.material_at(12, 20), 0);
        s.validate().unwrap();
        // A wall dragged across the plane wave box keeps its faces and the
        // nodes across them in the background.
        let mut s = tfsf_sample();
        let out = s.paint_stroke((30.0, 2.0), (30.0, 47.0), 0.0, 2).unwrap();
        assert_eq!(out.skipped, 4);
        for j in [9, 10, 40, 41] {
            assert_eq!(s.material_at(30, j), 0, "row {j}");
        }
        assert_eq!(s.material_at(30, 20), 2);
        s.validate().unwrap();
    }

    #[test]
    fn stroke_extends_into_the_layer() {
        // 40 x 30 with a 5-cell layer: interior nodes are i in 5..=34 and
        // j in 5..=24. Strokes at every edge and corner keep the scene valid.
        let base = || {
            let mut s = empty(40, 30, vec![]);
            s.pml = Some(crate::cpml::tests::spec(5));
            s
        };
        for (from, to) in [
            ((0.0, 15.0), (6.0, 15.0)),
            ((39.0, 15.0), (33.0, 15.0)),
            ((20.0, 0.0), (20.0, 6.0)),
            ((20.0, 29.0), (20.0, 23.0)),
            ((0.0, 0.0), (6.0, 6.0)),
            ((39.0, 29.0), (33.0, 23.0)),
            ((0.0, 29.0), (6.0, 23.0)),
            ((39.0, 0.0), (33.0, 6.0)),
        ] {
            let mut s = base();
            s.paint_stroke(from, to, 1.5, 2).unwrap();
            s.validate()
                .unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e}"));
        }
        // A horizontal wall through the left face runs out to the wall ring
        // (whose raster entry follows too; the kernels hold it PEC anyway).
        let mut s = base();
        s.paint_stroke((0.0, 15.0), (8.0, 15.0), 0.0, 2).unwrap();
        assert_eq!(cells(&s, 2), (0..=8).map(|i| (i, 15)).collect::<Vec<_>>());
        // Paint only inside the layer has no interior node to follow.
        let mut s = base();
        let out = s.paint_stroke((2.0, 15.0), (3.0, 15.0), 0.0, 2).unwrap();
        assert_eq!(out.painted, 0);
        assert!(cells(&s, 2).is_empty());
    }

    #[test]
    fn stroke_rejects_bad_input() {
        let mut s = sample();
        assert_eq!(
            s.paint_stroke((1.0, 1.0), (2.0, 2.0), 1.0, 3),
            Err(FdtdError::UnknownMaterial { id: 3 })
        );
        assert!(s.paint_stroke((f64::NAN, 1.0), (2.0, 2.0), 1.0, 1).is_err());
        assert!(s.paint_stroke((1.0, 1.0), (2.0, 2.0), -1.0, 1).is_err());
        assert!(s.paint.is_none());
        // Far outside the grid paints nothing, without error.
        let out = s
            .paint_stroke((-50.0, -50.0), (-40.0, -40.0), 2.0, 1)
            .unwrap();
        assert_eq!(out.bounds, None);
    }

    #[test]
    fn paint_runs_are_bounded() {
        // Shared links are untrusted: a huge run count must fail, not allocate.
        let text = format!(r#"{{"runs": [[null, {}]]}}"#, MAX_NODES + 1);
        assert!(serde_json::from_str::<Paint>(&text).is_err());
        let p: Paint = serde_json::from_str(r#"{"runs": [[null, 3], [1, 2]]}"#).unwrap();
        assert_eq!(p.len(), 5);
        assert_eq!((p.get(2), p.get(3)), (None, Some(1)));
    }
}
