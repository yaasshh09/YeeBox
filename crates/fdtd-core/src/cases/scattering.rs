//! Far-field cases (`docs/04-validation.md`, Phase 7): the line source
//! isotropy check of the near-to-far-field transform, A19 and A20 (cylinder
//! scattering width against `validation/gen/cylinder_series.py`), and the
//! shared runner the double slit and array cases use.
//!
//! Every run here is broadband: a Ricker pulse, monitors read once the
//! fields have died down (`spectra::run_until_quiet`), so the DFTs are the
//! exact transforms of the whole response. Normalized units.

use super::plane_wave::num;
use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::error::{FdtdError, Result};
use crate::kernel::{CpuKernel, Kernel, Spectrum};
use crate::material::Material;
use crate::ntff::ALL_SIDES;
use crate::ntff::scattering_width;
use crate::scene::{DEFAULT_ANGLES, GridSpec, Mode, Monitor, SCENE_VERSION, Scene, Shape, Source};
use crate::source::{SourceKind, Waveform};
use crate::spectra::run_until_quiet;
use crate::tfsf::{Direction, PlaneWave};
use std::f64::consts::PI;

const SERIES: &str = include_str!("../../../../validation/golden/cylinder_series.csv");

/// The frequencies the line source checks read: 40 and 20 cells per
/// wavelength in vacuum.
pub const LINE_FREQS: [f64; 2] = [0.025, 0.05];

/// Ricker pulse with its peak at `f_peak`, centered at `t0 = 2 / f_peak`
/// (its leading edge is below 1e-15 of peak at t = 0).
pub fn ricker(f_peak: f64) -> Waveform {
    Waveform::Ricker {
        t0: 2.0 / f_peak,
        f_peak,
    }
}

/// A closed contour on the node box `[x0, x1] x [y0, y1]` at `freqs`, 360
/// angles.
pub fn contour(x0: usize, y0: usize, x1: usize, y1: usize, freqs: &[f64]) -> Monitor {
    Monitor::Ntff {
        x0,
        y0,
        x1,
        y1,
        sides: ALL_SIDES.to_vec(),
        freqs: freqs.to_vec(),
        angles: DEFAULT_ANGLES,
    }
}

/// An `n` x `n` vacuum grid with a 16-cell recommended layer, one soft
/// Ricker line source (peak 0.04) at node `src`, the given monitors, in
/// `mode`.
pub fn line_source_scene(
    n: usize,
    src: (usize, usize),
    monitors: Vec<Monitor>,
    mode: Mode,
) -> Scene {
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: n,
            ny: n,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::VACUUM],
        geometry: vec![],
        sources: vec![Source {
            i: src.0,
            j: src.1,
            waveform: ricker(0.04),
            amplitude: 1.0,
            kind: SourceKind::Soft,
        }],
        monitors,
        pml: Some(PmlSpec::recommended(16)),
        periodic_y: false,
        plane_wave: None,
        paint: None,
        view: None,
        mode,
    }
}

/// Runs `scene` until its fields have died down and returns every spectral
/// monitor's spectrum, in scene order.
pub fn run_spectra(scene: &Scene) -> Result<Vec<Spectrum>> {
    let mut sim = CpuKernel::from_scene(scene)?;
    run_until_quiet(&mut sim, 400, 40_000)?;
    sim.monitor_spectra()
}

/// The far fields of `scene`'s contours, in scene order, each indexed
/// `f * angles + m` like `Ntff::far_field`.
pub fn far_fields(scene: &Scene) -> Result<Vec<Vec<(f64, f64)>>> {
    Ok(run_spectra(scene)?
        .into_iter()
        .filter_map(|s| match s {
            Spectrum::FarField(v) => Some(v),
            _ => None,
        })
        .collect())
}

/// |F| in dB.
pub fn db(f: (f64, f64)) -> f64 {
    20.0 * f.0.hypot(f.1).log10()
}

/// Peak-to-peak |F| in dB over all angles of frequency index `f`.
pub fn ripple_db(far: &[(f64, f64)], angles: usize, f: usize) -> f64 {
    let (lo, hi) = far[f * angles..(f + 1) * angles]
        .iter()
        .map(|&v| db(v))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    hi - lo
}

/// The line source isotropy check: a soft line source at the center of a
/// 121 x 121 vacuum grid, a closed contour 20 cells out on every side; the
/// peak-to-peak |F| over 360 angles, at 40 and 20 cells per wavelength, in
/// TMz and TEz. Gate 0.1 dB (Phase 7 acceptance).
pub fn isotropy() -> Result<Vec<CaseResult>> {
    let (n, c, l) = (121, 60, 20);
    let mut worst = 0.0_f64;
    let mut parts = Vec::new();
    for mode in [Mode::Tm, Mode::Te] {
        let ring = contour(c - l, c - l, c + l, c + l, &LINE_FREQS);
        let far = &far_fields(&line_source_scene(n, (c, c), vec![ring], mode))?[0];
        ensure(far.iter().all(|v| v.0.hypot(v.1) > 0.0), || {
            "line source: empty far field".into()
        })?;
        for (f, freq) in LINE_FREQS.iter().enumerate() {
            let r = ripple_db(far, DEFAULT_ANGLES, f);
            worst = worst.max(r);
            parts.push(format!("{mode:?} f {freq} {r:.4} dB"));
        }
    }
    Ok(vec![CaseResult::new(
        "NTFF",
        "line source far field isotropy",
        worst,
        Gate::AtMost(0.1),
        "dB",
        format!("contour 41 x 41 around the source; {}", parts.join(", ")),
    )])
}

/// The k a values of the cylinder cases, as in the golden file.
pub const KA: [f64; 3] = [1.0, 2.0, 5.0];

/// What the cylinder is made of.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cylinder {
    Pec,
    /// eps 4, lossless.
    Dielectric,
}

impl Cylinder {
    fn material(self) -> Material {
        match self {
            Cylinder::Pec => Material::Pec,
            Cylinder::Dielectric => Material::Dielectric {
                eps_inf: 4.0,
                sigma: 0.0,
            },
        }
    }

    /// The golden file's case name in `mode`.
    pub fn case(self, mode: Mode) -> &'static str {
        match (self, mode) {
            (Cylinder::Pec, Mode::Tm) => "pec_tm",
            (Cylinder::Pec, Mode::Te) => "pec_te",
            (Cylinder::Dielectric, Mode::Tm) => "dielectric_tm",
            (Cylinder::Dielectric, Mode::Te) => "dielectric_te",
        }
    }
}

/// The series sigma / lambda in dB for `case` at `ka`, one value per degree
/// from 0 to 359.
pub fn series(case: &str, ka: f64) -> Result<Vec<f64>> {
    let mut rows = Vec::new();
    for line in SERIES.lines() {
        let c: Vec<&str> = line.split(',').collect();
        if c.len() == 4 && c[0] == case && c[1].parse::<f64>().ok() == Some(ka) {
            rows.push(num(c[3], "cylinder series")?);
        }
    }
    ensure(rows.len() == DEFAULT_ANGLES, || {
        format!(
            "cylinder series: {} rows for {case} at k a {ka}",
            rows.len()
        )
    })?;
    Ok(rows)
}

/// The frequency at which a cylinder of radius `a` cells has each k a of
/// `KA` in vacuum: f = k a / (2 pi a).
pub fn cylinder_freqs(a: usize) -> Vec<f64> {
    KA.iter().map(|ka| ka / (2.0 * PI * a as f64)).collect()
}

/// The cylinder scene: radius `a` cells at the center node c, a +x TF/SF
/// Ricker plane wave (peak at k a = 2.5) on the box c +- (a + 3), a closed
/// far-field contour on c +- (a + 6), and the 16-cell recommended layer two
/// nodes beyond the contour. `cylinder: None` leaves the box empty, with a
/// DFT monitor at the center instead, for the incident field.
pub fn cylinder_scene(a: usize, mode: Mode, cylinder: Option<Cylinder>) -> Scene {
    let d = 16;
    let (tf, ring) = (a + 3, a + 6);
    let c = d + 2 + ring;
    let n = 2 * c + 1;
    let freqs = cylinder_freqs(a);
    let f_peak = 2.5 / (2.0 * PI * a as f64);
    let (geometry, monitors) = match cylinder {
        Some(_) => (
            vec![Shape::Circle {
                cx: c as f64,
                cy: c as f64,
                r: a as f64,
                material: 1,
            }],
            vec![contour(c - ring, c - ring, c + ring, c + ring, &freqs)],
        ),
        None => (vec![], vec![Monitor::Dft { i: c, j: c, freqs }]),
    };
    Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx: n,
            ny: n,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![
            Material::VACUUM,
            cylinder.map_or(Material::VACUUM, Cylinder::material),
        ],
        geometry,
        sources: vec![],
        monitors,
        pml: Some(PmlSpec::recommended(d)),
        periodic_y: false,
        plane_wave: Some(PlaneWave {
            x0: c - tf,
            y0: c - tf,
            x1: c + tf,
            y1: c + tf,
            direction: Direction::PlusX,
            waveform: ricker(f_peak),
            amplitude: 1.0,
        }),
        paint: None,
        view: None,
        mode,
    }
}

/// The incident field phasor at the cylinder center, one per k a, from the
/// scene without the cylinder.
pub fn incident(a: usize, mode: Mode) -> Result<Vec<(f64, f64)>> {
    match run_spectra(&cylinder_scene(a, mode, None))?.pop() {
        Some(Spectrum::Field(v)) => Ok(v),
        other => Err(FdtdError::Case(format!("cylinder incident: {other:?}"))),
    }
}

/// The step budget of a cylinder run of radius `a`: 1500 a.
pub fn cylinder_budget(a: usize) -> u64 {
    1500 * a as u64
}

/// sigma / lambda in dB per k a, one value per degree, from a far field.
fn sigma_db(far: &[(f64, f64)], a: usize, inc: &[(f64, f64)]) -> Vec<Vec<f64>> {
    cylinder_freqs(a)
        .iter()
        .enumerate()
        .map(|(f, freq)| {
            (0..DEFAULT_ANGLES)
                .map(|m| {
                    let sigma = scattering_width(far[f * DEFAULT_ANGLES + m], inc[f]);
                    10.0 * (sigma * freq).log10()
                })
                .collect()
        })
        .collect()
}

/// A cylinder run: sigma / lambda in dB per k a, one value per degree, at
/// the end of the run and at 3/4 of the step budget.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub sigma: Vec<Vec<f64>>,
    pub early: Vec<Vec<f64>>,
    pub steps: u64,
}

/// Runs the cylinder against the incident phasors `inc` of
/// `incident(a, mode)`. The run stops once the fields are quiet
/// (`run_until_quiet`) or at `cylinder_budget(a)`. The dielectric cylinder
/// has high-Q whispering gallery modes just above the band (one at k a =
/// 5.9 rings at 3e-4 of peak after 20000 steps at a = 20), so it may not go
/// quiet in any sensible time. A mode ringing at f_r adds at most
/// amplitude / (2 pi |f - f_r|) to the DFT at f from then on, however
/// slowly it decays, so the truncation shows directly as the change between
/// the far field at 3/4 of the budget and at the end; `errors` reports it.
pub fn measured(a: usize, mode: Mode, cylinder: Cylinder, inc: &[(f64, f64)]) -> Result<Measured> {
    let budget = cylinder_budget(a);
    let mut sim = CpuKernel::from_scene(&cylinder_scene(a, mode, Some(cylinder)))?;
    let far = |sim: &mut CpuKernel| -> Result<Vec<(f64, f64)>> {
        match sim.monitor_spectra()?.pop() {
            Some(Spectrum::FarField(v)) => Ok(v),
            other => Err(FdtdError::Case(format!("cylinder far field: {other:?}"))),
        }
    };
    let quiet = match run_until_quiet(&mut sim, 400, budget * 3 / 4) {
        Ok(_) => true,
        Err(FdtdError::NotDecayed { .. }) => false,
        Err(e) => return Err(e),
    };
    let early = sigma_db(&far(&mut sim)?, a, inc);
    if !quiet {
        sim.step(budget - sim.time_step());
    }
    Ok(Measured {
        sigma: sigma_db(&far(&mut sim)?, a, inc),
        early,
        steps: sim.time_step(),
    })
}

/// One run against the series: the worst and the RMS dB error over the
/// angles where the series is within 20 dB of its peak (deep nulls are
/// staircasing, `docs/04-validation.md`), per k a of `KA`.
#[derive(Debug, Clone, PartialEq)]
pub struct Errors {
    pub worst: [f64; 3],
    pub rms: [f64; 3],
    /// The largest change over the compared angles between 3/4 of the run
    /// and its end, dB (0 when the run went quiet first).
    pub settle: f64,
}

impl Errors {
    pub fn worst(&self) -> f64 {
        self.worst.iter().fold(0.0_f64, |m, &v| m.max(v))
    }

    /// RMS over all three k a together.
    pub fn rms(&self) -> f64 {
        (self.rms.iter().map(|v| v * v).sum::<f64>() / 3.0).sqrt()
    }
}

/// A run against the series for `case`.
pub fn errors(run: &Measured, case: &str) -> Result<Errors> {
    let (mut worst, mut rms, mut settle) = ([0.0; 3], [0.0; 3], 0.0_f64);
    let sim = &run.sigma;
    for (k, &ka) in KA.iter().enumerate() {
        let gold = series(case, ka)?;
        let top = gold.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let diffs: Vec<f64> = (0..DEFAULT_ANGLES)
            .filter(|&m| gold[m] >= top - 20.0)
            .map(|m| {
                settle = settle.max((sim[k][m] - run.early[k][m]).abs());
                (sim[k][m] - gold[m]).abs()
            })
            .collect();
        worst[k] = diffs.iter().fold(0.0_f64, |m, &v| m.max(v));
        rms[k] = (diffs.iter().map(|v| v * v).sum::<f64>() / diffs.len() as f64).sqrt();
    }
    Ok(Errors { worst, rms, settle })
}

/// One cylinder at one radius against the series.
pub fn cylinder_errors(a: usize, mode: Mode, cylinder: Cylinder) -> Result<Errors> {
    let inc = incident(a, mode)?;
    errors(&measured(a, mode, cylinder, &inc)?, cylinder.case(mode))
}

/// The radii of the convergence study, cells.
pub const RADII: [usize; 3] = [10, 20, 40];

/// A19 or A20: one cylinder against the series in TMz at the gated radius
/// (worst dB error over k a = 1, 2, 5, gate 1 dB), and the convergence row:
/// the pooled RMS dB error at each radius of `RADII`, which must fall at
/// every doubling (measured: the largest ratio RMS(2a) / RMS(a), gate
/// below 1; the note gives the fitted order). TEz runs the same way and is
/// recorded in the note, not gated. The truncation guard: the far field
/// moves less than 0.01 dB over the last quarter of every run.
fn cylinder_case(
    id: &'static str,
    conv_id: &'static str,
    name: &'static str,
    cylinder: Cylinder,
    gated: usize,
) -> Result<Vec<CaseResult>> {
    let mut by_mode = Vec::new();
    for mode in [Mode::Tm, Mode::Te] {
        let mut runs = Vec::new();
        for a in RADII {
            let e = cylinder_errors(a, mode, cylinder)?;
            ensure(e.settle < 0.01, || {
                format!(
                    "{id} {mode:?} a {a}: far field still moving {:.3} dB",
                    e.settle
                )
            })?;
            runs.push((a, e));
        }
        by_mode.push(runs);
    }
    let pick =
        |runs: &[(usize, Errors)], a: usize| runs.iter().find(|r| r.0 == a).map(|r| r.1.clone());
    let tm = pick(&by_mode[0], gated).ok_or(FdtdError::Case(format!("{id}: no gated run")))?;
    let te = pick(&by_mode[1], gated).ok_or(FdtdError::Case(format!("{id}: no gated run")))?;
    let rms_list =
        |runs: &[(usize, Errors)]| -> Vec<f64> { runs.iter().map(|r| r.1.rms()).collect() };
    let order = |runs: &[(usize, Errors)]| {
        let pts: Vec<(f64, f64)> = runs
            .iter()
            .map(|r| ((r.0 as f64).ln(), r.1.rms().ln()))
            .collect();
        -super::signal::slope(&pts)
    };
    let (tm_rms, te_rms) = (rms_list(&by_mode[0]), rms_list(&by_mode[1]));
    let ratio = tm_rms
        .windows(2)
        .map(|w| w[1] / w[0])
        .fold(0.0_f64, f64::max);
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Ok(vec![
        CaseResult::new(
            id,
            name,
            tm.worst(),
            Gate::AtMost(1.0),
            "dB",
            format!(
                "TMz, radius {gated} cells, k a = 1, 2, 5: {} dB; TEz recorded: {} dB; angles within 20 dB of the series peak",
                fmt(&tm.worst),
                fmt(&te.worst)
            ),
        ),
        CaseResult::new(
            conv_id,
            "error falls as the radius doubles",
            ratio,
            Gate::AtMost(1.0),
            "ratio",
            format!(
                "pooled RMS dB at a = 10, 20, 40: TMz {} (order {:.2}), TEz {} (order {:.2})",
                fmt(&tm_rms),
                order(&by_mode[0]),
                fmt(&te_rms),
                order(&by_mode[1])
            ),
        ),
    ])
}

/// A19: PEC cylinder, gated at a radius of 20 cells (lambda0 = 25 cells at
/// k a = 5).
pub fn a19() -> Result<Vec<CaseResult>> {
    cylinder_case(
        "A19",
        "A19 conv",
        "PEC cylinder scattering width",
        Cylinder::Pec,
        20,
    )
}

/// A20: eps 4 cylinder, gated at a radius of 40 cells, where the wavelength
/// inside it is 25 cells at k a = 5 (the 20-cell rule of A13 picks it).
pub fn a20() -> Result<Vec<CaseResult>> {
    cylinder_case(
        "A20",
        "A20 conv",
        "dielectric cylinder scattering width",
        Cylinder::Dielectric,
        40,
    )
}
