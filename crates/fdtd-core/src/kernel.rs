//! The interface every kernel implements, so one harness can drive the f64
//! CPU reference and the f32 GPU kernel from the same `Scene` and diff them.

use crate::cpml::PmlSpec;
use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::material::Material;
use crate::scene::{GridSpec, Scene};
use crate::tfsf::PlaneWave;

pub trait Kernel: Sized {
    /// Validates the scene and builds the kernel at time step 0, fields zero.
    fn from_scene(scene: &Scene) -> Result<Self>;

    /// Advances `n` full steps.
    fn step(&mut self, n: u64);

    /// Full steps taken so far. E is at time n S, H at (n - 1/2) S.
    fn time_step(&self) -> u64;

    /// A copy of one field, normalized, `nx * ny` padded, index `j * nx + i`.
    /// Ez on nodes, Hx on y-edges, Hy on x-edges (see `tm.rs`).
    fn read_field(&mut self, which: Component) -> Result<Vec<f64>>;

    /// Ez history of every point monitor, in scene order. Each has
    /// `time_step() + 1` entries.
    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>>;

    /// The accumulated spectrum of every DFT and flux monitor, in scene
    /// order (point monitors are skipped), one entry per scene frequency.
    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>>;

    /// Takes the materials of `scene` on the Ez nodes in `bounds`
    /// (`[i0, j0, i1, j1]`, inclusive, cells), keeping every field and the
    /// time step. `scene` is the kernel's scene after a paint or shape edit,
    /// validated; a different `BuildKey` is `FdtdError::Rebuild`. A node
    /// that turns PEC keeps its Ez until the next step zeroes it, on every
    /// kernel alike.
    fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()>;

    /// Takes the sources of `scene` (positions, waveforms, amplitudes),
    /// keeping every field and the time step. Same `BuildKey` rule.
    fn set_sources(&mut self, scene: &Scene) -> Result<()>;
}

/// What a kernel is built from that a live edit must not change: the grid,
/// the y wrap, the material table, the CPML and the plane wave. Paint,
/// shapes and sources may change; anything else needs a new kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildKey {
    grid: GridSpec,
    periodic_y: bool,
    materials: Vec<Material>,
    pml: Option<PmlSpec>,
    plane_wave: Option<PlaneWave>,
}

impl BuildKey {
    pub fn of(scene: &Scene) -> BuildKey {
        BuildKey {
            grid: scene.grid,
            periodic_y: scene.periodic_y,
            materials: scene.materials.clone(),
            pml: scene.pml,
            plane_wave: scene.plane_wave,
        }
    }

    /// `FdtdError::Rebuild` unless `scene` has this key.
    pub fn check(&self, scene: &Scene) -> Result<()> {
        if *self == BuildKey::of(scene) {
            Ok(())
        } else {
            Err(FdtdError::Rebuild)
        }
    }
}

/// The nodes a material edit rewrites, the same for every kernel: columns
/// `i0..=i1` of each row in `rows`, the requested bounds clipped to the
/// live nodes (the wall ring never changes), with the new material id of
/// each, row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialEdit {
    pub i0: usize,
    pub i1: usize,
    pub rows: std::ops::Range<usize>,
    pub ids: Vec<usize>,
}

/// The `MaterialEdit` for `bounds` on `scene`, `None` if the bounds miss
/// every live node. Ids outside the material table are refused, so an
/// unvalidated scene cannot index past it.
pub fn material_edit(scene: &Scene, bounds: [usize; 4]) -> Result<Option<MaterialEdit>> {
    let [i0, j0, i1, j1] = bounds;
    let nx = scene.grid.nx;
    let live = scene.live_rows();
    let (i0, i1) = (i0.max(1), i1.min(nx - 2));
    let rows = j0.max(live.start)..(j1.saturating_add(1)).min(live.end);
    if i0 > i1 || rows.is_empty() {
        return Ok(None);
    }
    let mut ids = Vec::with_capacity((i1 - i0 + 1) * rows.len());
    for j in rows.clone() {
        for i in i0..=i1 {
            let id = scene.material_at(i, j);
            if id >= scene.materials.len() {
                return Err(FdtdError::UnknownMaterial { id });
            }
            ids.push(id);
        }
    }
    Ok(Some(MaterialEdit { i0, i1, rows, ids }))
}

/// What a spectral monitor has accumulated so far, per frequency of its list.
#[derive(Debug, Clone, PartialEq)]
pub enum Spectrum {
    /// The Ez transform at a node, `(re, im)`, normalized.
    Field(Vec<(f64, f64)>),
    /// Poynting flux through a line, positive along its normal, normalized.
    Flux(Vec<f64>),
}

/// How far kernel `b` is from the reference kernel `a` after `step` steps.
/// Every entry is `max |b - a|` over the whole array divided by the peak
/// `max |a|` of that array over this and every earlier checkpoint of the
/// run (0 when both are all zero, infinite when only `a` is). So 1e-4 means
/// "within 1e-4 of the reference peak field". An f32 kernel carries
/// round-off relative to the largest field it has held, so once a pulse has
/// left through the CPML the residue is compared against that, not against
/// itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Diff {
    pub step: u64,
    /// Ez, Hx, Hy, padded arrays included.
    pub fields: [f64; 3],
    /// All point probe histories together.
    pub probes: f64,
    /// All spectral monitors together, real and imaginary parts.
    pub spectra: f64,
}

impl Diff {
    /// The largest entry.
    pub fn worst(&self) -> f64 {
        self.fields
            .iter()
            .chain([&self.probes, &self.spectra])
            .fold(0.0, |m, &d| m.max(d))
    }
}

/// `max |b - a| / max |a|` of one snapshot, with the conventions of
/// `Diff`. Mismatched lengths are an error.
pub fn relative_diff(a: &[f64], b: &[f64]) -> Result<f64> {
    diff_against(a, b, &mut 0.0)
}

/// `max |b - a| / peak`, after raising `peak` to `max |a|`.
fn diff_against(a: &[f64], b: &[f64], peak: &mut f64) -> Result<f64> {
    if a.len() != b.len() {
        return Err(FdtdError::Case(format!(
            "compared arrays of length {} and {}",
            a.len(),
            b.len()
        )));
    }
    *peak = a.iter().fold(*peak, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((y - x).abs()));
    Ok(if worst == 0.0 { 0.0 } else { worst / *peak })
}

fn flatten(spectra: &[Spectrum]) -> Vec<f64> {
    let mut out = Vec::new();
    for s in spectra {
        match s {
            Spectrum::Field(v) => out.extend(v.iter().flat_map(|&(re, im)| [re, im])),
            Spectrum::Flux(v) => out.extend(v),
        }
    }
    out
}

/// Steps both kernels, built from the same scene, to each step count in
/// `checkpoints` (non-decreasing, none below where they already are) and
/// diffs every field, the probe histories and the spectra there. `a` is
/// the reference.
pub fn compare<A: Kernel, B: Kernel>(
    a: &mut A,
    b: &mut B,
    checkpoints: &[u64],
) -> Result<Vec<Diff>> {
    let mut out = Vec::with_capacity(checkpoints.len());
    // Running peaks of the reference: Ez, Hx, Hy, probes, spectra.
    let mut peak = [0.0f64; 5];
    for &step in checkpoints {
        for k in [a.time_step(), b.time_step()] {
            if step < k {
                return Err(FdtdError::Case(format!(
                    "checkpoint {step} is behind step {k}"
                )));
            }
        }
        a.step(step - a.time_step());
        b.step(step - b.time_step());
        let mut fields = [0.0; 3];
        for (f, c) in [Component::Ez, Component::Hx, Component::Hy]
            .into_iter()
            .enumerate()
        {
            fields[f] = diff_against(&a.read_field(c)?, &b.read_field(c)?, &mut peak[f])?;
        }
        let pa: Vec<f64> = a.probe_history()?.concat();
        let pb: Vec<f64> = b.probe_history()?.concat();
        let sa = flatten(&a.monitor_spectra()?);
        let sb = flatten(&b.monitor_spectra()?);
        out.push(Diff {
            step,
            fields,
            probes: diff_against(&pa, &pb, &mut peak[3])?,
            spectra: diff_against(&sa, &sb, &mut peak[4])?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tm::SimTm;
    use std::path::Path;

    fn scene(name: &str) -> Scene {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../validation/scenes")
            .join(name);
        Scene::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// The reference against itself is exactly zero everywhere, on a scene
    /// with TF/SF, flux lines and probes.
    #[test]
    fn same_kernel_is_exactly_zero() {
        let s = scene("tfsf_rod_minus_x.json");
        let (mut a, mut b) = (
            SimTm::from_scene(&s).unwrap(),
            SimTm::from_scene(&s).unwrap(),
        );
        for d in compare(&mut a, &mut b, &[0, 50, 300]).unwrap() {
            assert_eq!(d.worst(), 0.0, "{d:?}");
        }
    }

    /// A kernel whose Ez reads 1e-3 high, and whose spectra are off by the
    /// same factor, is caught in exactly those entries.
    struct Nudged(SimTm);

    impl Kernel for Nudged {
        fn from_scene(scene: &Scene) -> Result<Self> {
            SimTm::from_scene(scene).map(Nudged)
        }
        fn step(&mut self, n: u64) {
            self.0.step_n(n)
        }
        fn time_step(&self) -> u64 {
            self.0.time_step()
        }
        fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
            let v = self.0.read_field(which)?;
            let g = if which == Component::Ez { 1.001 } else { 1.0 };
            Ok(v.into_iter().map(|x| x * g).collect())
        }
        fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
            self.0.probe_history()
        }
        fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
            Ok(self
                .0
                .monitor_spectra()?
                .into_iter()
                .map(|s| match s {
                    Spectrum::Flux(v) => Spectrum::Flux(v.into_iter().map(|x| x * 1.001).collect()),
                    other => other,
                })
                .collect())
        }
        fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
            self.0.set_materials(scene, bounds)
        }
        fn set_sources(&mut self, scene: &Scene) -> Result<()> {
            self.0.set_sources(scene)
        }
    }

    #[test]
    fn perturbed_kernel_is_caught() {
        let s = scene("tfsf_rod_minus_x.json");
        let mut a = SimTm::from_scene(&s).unwrap();
        let mut b = Nudged::from_scene(&s).unwrap();
        let d = compare(&mut a, &mut b, &[300]).unwrap().remove(0);
        assert!((d.fields[0] - 1e-3).abs() < 1e-12, "{d:?}");
        assert!((d.spectra - 1e-3).abs() < 1e-12, "{d:?}");
        assert_eq!((d.fields[1], d.fields[2], d.probes), (0.0, 0.0, 0.0));
    }

    #[test]
    fn relative_diff_conventions() {
        assert_eq!(relative_diff(&[0.0, 0.0], &[0.0, 0.0]).unwrap(), 0.0);
        assert_eq!(relative_diff(&[0.0], &[1.0]).unwrap(), f64::INFINITY);
        assert_eq!(relative_diff(&[2.0, -4.0], &[2.0, -3.0]).unwrap(), 0.25);
        assert!(relative_diff(&[1.0], &[1.0, 2.0]).is_err());
    }

    /// A kernel whose fields read high by a fixed absolute amount on the
    /// wall ring, where the reference is exactly zero.
    struct Offset(SimTm);

    impl Kernel for Offset {
        fn from_scene(scene: &Scene) -> Result<Self> {
            SimTm::from_scene(scene).map(Offset)
        }
        fn step(&mut self, n: u64) {
            self.0.step_n(n)
        }
        fn time_step(&self) -> u64 {
            self.0.time_step()
        }
        fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
            let mut v = self.0.read_field(which)?;
            v[0] += 1e-6;
            Ok(v)
        }
        fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
            self.0.probe_history()
        }
        fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
            self.0.monitor_spectra()
        }
        fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
            self.0.set_materials(scene, bounds)
        }
        fn set_sources(&mut self, scene: &Scene) -> Result<()> {
            self.0.set_sources(scene)
        }
    }

    /// The denominator is the peak so far, not the current peak: as the
    /// pulse of a CPML scene leaves, a fixed absolute error keeps the ratio
    /// it had at the peak instead of growing with the decay.
    #[test]
    fn normalized_by_the_running_peak() {
        let s = scene("line_source_cpml.json");
        let mut s = s;
        s.sources[0].waveform = crate::source::Waveform::Ricker {
            t0: 60.0,
            f_peak: 0.05,
        };
        let mut a = SimTm::from_scene(&s).unwrap();
        let mut b = Offset::from_scene(&s).unwrap();
        let d = compare(&mut a, &mut b, &[200, 3000]).unwrap();
        let ez_at = |n: u64| {
            let mut c = SimTm::from_scene(&s).unwrap();
            c.step_n(n);
            c.ez().iter().fold(0.0f64, |m, v| m.max(v.abs()))
        };
        let (p200, p3000) = (ez_at(200), ez_at(3000));
        assert!(p3000 < 1e-2 * p200, "pulse did not leave: {p200} {p3000}");
        let peak = p200.max(p3000);
        assert!((d[1].fields[0] - 1e-6 / peak).abs() < 1e-9 / peak, "{d:?}");
        assert!(d[1].fields[0] <= d[0].fields[0] * (1.0 + 1e-9), "{d:?}");
    }

    #[test]
    fn checkpoints_cannot_go_back() {
        let s = scene("hard_sources.json");
        let (mut a, mut b) = (
            SimTm::from_scene(&s).unwrap(),
            SimTm::from_scene(&s).unwrap(),
        );
        assert!(compare(&mut a, &mut b, &[20, 10]).is_err());
    }
}
