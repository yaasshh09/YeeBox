//! The interface every kernel implements, so one harness can drive the f64
//! CPU reference and the f32 GPU kernel from the same `Scene` and diff them.

use crate::cpml::PmlSpec;
use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::material::Material;
use crate::scene::{GridSpec, Mode, Scene};
use crate::te::SimTe;
use crate::tfsf::PlaneWave;
use crate::tm::SimTm;

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

    /// The polarization this kernel runs. TMz unless a kernel says otherwise.
    fn mode(&self) -> Mode {
        Mode::Tm
    }
}

/// The f64 reference kernel of a scene's mode: what the CLI, the CPU
/// fallback and the harnesses build when they take any scene.
#[derive(Debug, Clone)]
pub enum CpuKernel {
    Tm(SimTm),
    Te(SimTe),
}

impl CpuKernel {
    /// The out-of-plane field on the nodes: Ez in TMz, Hz in TEz.
    pub fn node_field(&self) -> &[f64] {
        match self {
            CpuKernel::Tm(k) => k.ez(),
            CpuKernel::Te(k) => k.hz(),
        }
    }

    pub fn nx(&self) -> usize {
        match self {
            CpuKernel::Tm(k) => k.nx(),
            CpuKernel::Te(k) => k.nx(),
        }
    }

    pub fn ny(&self) -> usize {
        match self {
            CpuKernel::Tm(k) => k.ny(),
            CpuKernel::Te(k) => k.ny(),
        }
    }

    /// Zeroes every spectral accumulator (`SimTm::restart_spectra`).
    pub fn restart_spectra(&mut self) {
        match self {
            CpuKernel::Tm(k) => k.restart_spectra(),
            CpuKernel::Te(k) => k.restart_spectra(),
        }
    }

    /// The Courant number S, which is dt in normalized units.
    pub fn courant(&self) -> f64 {
        match self {
            CpuKernel::Tm(k) => k.courant(),
            CpuKernel::Te(k) => k.courant(),
        }
    }
}

impl Kernel for CpuKernel {
    fn from_scene(scene: &Scene) -> Result<Self> {
        Ok(match scene.mode {
            Mode::Tm => CpuKernel::Tm(SimTm::from_scene(scene)?),
            Mode::Te => CpuKernel::Te(SimTe::from_scene(scene)?),
        })
    }

    fn step(&mut self, n: u64) {
        match self {
            CpuKernel::Tm(k) => k.step_n(n),
            CpuKernel::Te(k) => k.step_n(n),
        }
    }

    fn time_step(&self) -> u64 {
        match self {
            CpuKernel::Tm(k) => k.time_step(),
            CpuKernel::Te(k) => k.time_step(),
        }
    }

    fn read_field(&mut self, which: Component) -> Result<Vec<f64>> {
        match self {
            CpuKernel::Tm(k) => k.read_field(which),
            CpuKernel::Te(k) => k.read_field(which),
        }
    }

    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>> {
        match self {
            CpuKernel::Tm(k) => k.probe_history(),
            CpuKernel::Te(k) => k.probe_history(),
        }
    }

    fn monitor_spectra(&mut self) -> Result<Vec<Spectrum>> {
        match self {
            CpuKernel::Tm(k) => k.monitor_spectra(),
            CpuKernel::Te(k) => k.monitor_spectra(),
        }
    }

    fn set_materials(&mut self, scene: &Scene, bounds: [usize; 4]) -> Result<()> {
        match self {
            CpuKernel::Tm(k) => k.set_materials(scene, bounds),
            CpuKernel::Te(k) => k.set_materials(scene, bounds),
        }
    }

    fn set_sources(&mut self, scene: &Scene) -> Result<()> {
        match self {
            CpuKernel::Tm(k) => k.set_sources(scene),
            CpuKernel::Te(k) => k.set_sources(scene),
        }
    }

    fn mode(&self) -> Mode {
        match self {
            CpuKernel::Tm(_) => Mode::Tm,
            CpuKernel::Te(_) => Mode::Te,
        }
    }
}

/// The Tier 4 gate: an f32 kernel within this fraction of the reference
/// peak field (`Comparator`). Shared by the native tests and the browser
/// self-test so both hold the same line.
pub const GPU_GATE: f64 = 1e-4;

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
    mode: crate::scene::Mode,
}

impl BuildKey {
    pub fn of(scene: &Scene) -> BuildKey {
        BuildKey {
            grid: scene.grid,
            periodic_y: scene.periodic_y,
            materials: scene.materials.clone(),
            pml: scene.pml,
            plane_wave: scene.plane_wave,
            mode: scene.mode,
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
    let [mut i0, mut j0, mut i1, mut j1] = bounds;
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    // Layer nodes follow the interface (`Scene::material_node`), so bounds
    // that reach an interface line reach on to the wall behind it.
    let d = scene.pml_thickness();
    if d > 0 {
        if i0 <= d {
            i0 = 0;
        }
        if i1 >= nx - 1 - d {
            i1 = nx - 1;
        }
        if !scene.periodic_y {
            if j0 <= d {
                j0 = 0;
            }
            if j1 >= ny - 1 - d {
                j1 = ny - 1;
            }
        }
    }
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
    /// Far-field amplitude `F(phi_m, f)` of a contour, `(re, im)`, entry
    /// `f * angles + m` (`ntff::Ntff::far_field`), normalized.
    FarField(Vec<(f64, f64)>),
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
            Spectrum::Field(v) | Spectrum::FarField(v) => {
                out.extend(v.iter().flat_map(|&(re, im)| [re, im]))
            }
            Spectrum::Flux(v) => out.extend(v),
        }
    }
    out
}

/// Everything `compare` diffs at one checkpoint: Ez, Hx, Hy, all point
/// probe histories together, and all spectra flattened (real and imaginary
/// parts of DFTs, then flux values, in scene order).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Snapshot {
    pub fields: [Vec<f64>; 3],
    pub probes: Vec<f64>,
    pub spectra: Vec<f64>,
}

impl Snapshot {
    /// Reads a kernel where it stands.
    pub fn of<K: Kernel>(k: &mut K) -> Result<Snapshot> {
        Ok(Snapshot {
            fields: {
                let [a, b, c] = k.mode().fields();
                [k.read_field(a)?, k.read_field(b)?, k.read_field(c)?]
            },
            probes: k.probe_history()?.concat(),
            spectra: flatten(&k.monitor_spectra()?),
        })
    }
}

/// The measure `compare` reports, fed one checkpoint at a time, so a caller
/// that reads its kernels asynchronously (the browser self-test) gets the
/// same numbers. It keeps the reference's running peaks: Ez, Hx, Hy,
/// probes, spectra.
#[derive(Debug, Clone, Default)]
pub struct Comparator {
    peak: [f64; 5],
}

impl Comparator {
    pub fn new() -> Comparator {
        Comparator::default()
    }

    /// Raises the field peaks (Ez, Hx, Hy) to reference maxima seen at a
    /// step where no comparison is made, so a caller that compares only at
    /// a few checkpoints still divides by the peak over the whole run.
    pub fn raise_field_peaks(&mut self, peaks: [f64; 3]) {
        for (p, v) in self.peak.iter_mut().zip(peaks) {
            *p = p.max(v);
        }
    }

    /// The `Diff` of candidate `b` against reference `a` at `step`.
    pub fn checkpoint(&mut self, step: u64, a: &Snapshot, b: &Snapshot) -> Result<Diff> {
        let mut fields = [0.0; 3];
        for (f, out) in fields.iter_mut().enumerate() {
            *out = diff_against(&a.fields[f], &b.fields[f], &mut self.peak[f])?;
        }
        Ok(Diff {
            step,
            fields,
            probes: diff_against(&a.probes, &b.probes, &mut self.peak[3])?,
            spectra: diff_against(&a.spectra, &b.spectra, &mut self.peak[4])?,
        })
    }
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
    let mut cmp = Comparator::new();
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
        out.push(cmp.checkpoint(step, &Snapshot::of(a)?, &Snapshot::of(b)?)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn scene(name: &str) -> Scene {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../validation/scenes")
            .join(name);
        Scene::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    /// `CpuKernel` builds the kernel of the scene's mode, steps exactly like
    /// it, and `Snapshot` reads the TEz fields by their own names.
    #[test]
    fn cpu_kernel_follows_the_mode() {
        let tm_scene = scene("tfsf_rod_minus_x.json");
        let mut te_scene = tm_scene.clone();
        te_scene.mode = crate::scene::Mode::Te;
        // The rod is a dielectric, not vacuum, so TEz runs it, just differently.
        let mut a = CpuKernel::from_scene(&tm_scene).unwrap();
        let mut b = SimTm::from_scene(&tm_scene).unwrap();
        let mut c = CpuKernel::from_scene(&te_scene).unwrap();
        let mut d = crate::te::SimTe::from_scene(&te_scene).unwrap();
        assert_eq!(
            (a.mode(), c.mode()),
            (crate::scene::Mode::Tm, crate::scene::Mode::Te)
        );
        for k in [&mut a, &mut c] {
            Kernel::step(k, 120);
        }
        b.step_n(120);
        d.step_n(120);
        assert_eq!(Snapshot::of(&mut a).unwrap(), Snapshot::of(&mut b).unwrap());
        let te = Snapshot::of(&mut c).unwrap();
        assert_eq!(te, Snapshot::of(&mut d).unwrap());
        assert_eq!(te.fields[0], d.hz());
        assert_eq!(c.node_field(), d.hz());
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
    fn comparator_keeps_the_running_peak() {
        let snap = |v: f64| Snapshot {
            fields: [vec![v], vec![0.0], vec![0.0]],
            ..Default::default()
        };
        let mut c = Comparator::new();
        let d = c.checkpoint(10, &snap(2.0), &snap(2.5)).unwrap();
        assert_eq!(d.fields, [0.25, 0.0, 0.0]);
        // The field fell to 1; the same 0.5 error is still 0.25 of the peak.
        let d = c.checkpoint(20, &snap(1.0), &snap(1.5)).unwrap();
        assert_eq!((d.step, d.fields[0], d.worst()), (20, 0.25, 0.25));
        assert!(c.checkpoint(30, &snap(1.0), &Snapshot::default()).is_err());
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
