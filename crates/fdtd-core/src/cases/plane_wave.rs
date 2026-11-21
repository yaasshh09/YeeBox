//! A11 to A15: spectra and the plane wave source. Each case is gated on the
//! continuum answer and guarded against the exact discrete one from the
//! golden files, so a measurement that drifts from what the grid itself
//! predicts fails loudly even when it would still clear the gate.

use super::signal::slope;
use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::error::{FdtdError, Result};
use crate::kernel::{Kernel, Spectrum};
use crate::material::Material;
use crate::monitor::Dft;
use crate::scene::{GridSpec, Monitor, SCENE_VERSION, Scene, Shape, Source};
use crate::source::{SourceKind, Waveform};
use crate::spectra::{RtSpectrum, rt_spectrum, run_until_quiet};
use crate::tfsf::{Direction, PlaneWave};
use crate::tm::SimTm;
use std::f64::consts::PI;

const SLAB: &str = include_str!("../../../../validation/golden/slab_airy.csv");
const LOSSY: &str = include_str!("../../../../validation/golden/lossy_medium.csv");
const GUIDE: &str = include_str!("../../../../validation/golden/waveguide.csv");

/// The numeric rows of a golden CSV (the three comment lines and the column
/// header dropped), after checking its second line (the parameter line).
fn golden(text: &str, params: &str, what: &str) -> Result<Vec<Vec<String>>> {
    ensure(text.lines().nth(1) == Some(params), || {
        format!("{what}: golden data is for another setup")
    })?;
    Ok(text
        .lines()
        .skip(4)
        .map(|l| l.split(',').map(String::from).collect())
        .collect())
}

fn num(cell: &str, what: &str) -> Result<f64> {
    cell.parse()
        .map_err(|_| FdtdError::Case(format!("{what}: bad number {cell}")))
}

/// The graded preset without the CFS alpha, for scenes where every wave
/// meets the layer at normal incidence: alpha has nothing to do there and
/// costs absorption at long wavelengths (-41 dB at f = 0.0025 for the
/// preset, `fdtd1d` layer_reflection_against_frequency).
fn normal_layer(d: usize) -> PmlSpec {
    PmlSpec {
        alpha_max: 0.0,
        ..PmlSpec::recommended(d)
    }
}

/// An eps = 4 slab of `l` cells in a periodic-y vacuum line (ny = 3), lit
/// by a +x TF/SF Ricker pulse peaking at f_max / 2. Along x: 16-cell layer,
/// 4 cells, R line, 4 cells, entry face, 20 cells, slab (nodes s0 .. s0 +
/// l - 1, the golden file's "nodes 1..L"), 20 cells, T line, 4 cells, exit
/// face, 4 cells, layer. Returns the scene and the two line columns.
pub fn slab_scene(l: usize, f_max: f64) -> (Scene, usize, usize) {
    let d = 16;
    let r_at = d + 4;
    let x0 = r_at + 4;
    let s0 = x0 + 20;
    let t_at = s0 + l + 20;
    let x1 = t_at + 4;
    let nx = x1 + 5 + d;
    let f_peak = f_max / 2.0;
    let scene = Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny: 3,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![
            Material::VACUUM,
            Material::Dielectric {
                eps_inf: 4.0,
                sigma: 0.0,
            },
        ],
        geometry: vec![Shape::Rect {
            x0: s0 as f64,
            y0: -1.0,
            x1: (s0 + l - 1) as f64,
            y1: 3.0,
            material: 1,
        }],
        sources: vec![],
        monitors: vec![],
        pml: Some(normal_layer(d)),
        periodic_y: true,
        plane_wave: Some(PlaneWave {
            x0,
            y0: 0,
            x1,
            y1: 2,
            direction: Direction::PlusX,
            // u = pi f_peak t0 = 2 pi at t = 0, so the turn-on is below 1e-16.
            waveform: Waveform::Ricker {
                t0: 2.0 / f_peak,
                f_peak,
            },
            amplitude: 1.0,
        }),
        paint: None,
        view: None,
    };
    (scene, r_at, t_at)
}

/// One golden slab row: thickness, frequency, continuum and discrete R, T.
pub struct SlabRow {
    pub l: usize,
    pub f: f64,
    pub r_c: f64,
    pub t_c: f64,
    pub r_d: f64,
    pub t_d: f64,
}

pub fn slab_golden() -> Result<Vec<SlabRow>> {
    let rows = golden(
        SLAB,
        "# S = 0.5, eps = 4.0, base L = 20, slab on nodes 1..L",
        "A13",
    )?;
    rows.iter()
        .map(|r| {
            Ok(SlabRow {
                l: num(&r[0], "A13")? as usize,
                f: num(&r[1], "A13")?,
                r_c: num(&r[2], "A13")?,
                t_c: num(&r[3], "A13")?,
                r_d: num(&r[4], "A13")?,
                t_d: num(&r[5], "A13")?,
            })
        })
        .collect()
}

/// R and T of the `l`-cell slab at the golden frequencies for that `l`.
pub fn slab_spectrum(l: usize) -> Result<(Vec<SlabRow>, RtSpectrum)> {
    let rows: Vec<SlabRow> = slab_golden()?.into_iter().filter(|r| r.l == l).collect();
    ensure(rows.len() == 19, || {
        format!("A13: {} rows for L = {l}", rows.len())
    })?;
    let freqs: Vec<f64> = rows.iter().map(|r| r.f).collect();
    let f_max = freqs.iter().cloned().fold(0.0, f64::max);
    let (scene, r_at, t_at) = slab_scene(l, f_max);
    // One pulse length of quiet: the Ricker lasts about 2 t0 = 4 / f_peak.
    let window = (8.0 / f_max / 0.5) as u64;
    let rt = rt_spectrum(&scene, r_at, t_at, &freqs, window, 400_000)?;
    Ok((rows, rt))
}

/// A13 and the R + T = 1 check: an eps = 4 slab, 20 cells thick, at normal
/// incidence, over the band where the wavelength inside the slab is at
/// least 20 cells (f = 0.0025 to 0.025). Measured: the worst |R - R_c| and
/// |T - T_c| against Airy, and |R + T - 1|. Guards: the measurement within
/// 1e-4 of the exact discrete slab at L = 20, 40, 80, so what is left is
/// the grid's own error (predicted 4.9e-3 at the band edge), and the same
/// physical slab at L = 40 and 80 converging at order 1.8 to 2.2.
pub fn a13() -> Result<Vec<CaseResult>> {
    let mut worst_airy = Vec::new();
    let (mut base_rt, mut identity) = (0.0_f64, 0.0_f64);
    for l in [20, 40, 80] {
        let (rows, rt) = slab_spectrum(l)?;
        let (mut ea, mut ed, mut erl) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (k, row) in rows.iter().enumerate() {
            let (r, t) = (rt.r[k], rt.t[k]);
            ea = ea.max((r - row.r_c).abs()).max((t - row.t_c).abs());
            ed = ed.max((r - row.r_d).abs()).max((t - row.t_d).abs());
            erl = erl.max((r + t - 1.0).abs());
        }
        ensure(ed <= 1e-4, || {
            format!("A13 L = {l}: {ed:e} off the exact discrete slab")
        })?;
        identity = identity.max(ed);
        if l == 20 {
            base_rt = erl;
        }
        worst_airy.push(ea);
    }
    let order = -(worst_airy[2].log2() - worst_airy[0].log2()) / 2.0;
    ensure((1.8..=2.2).contains(&order), || {
        format!("A13: order {order}")
    })?;
    Ok(vec![
        CaseResult::new(
            "A13",
            "dielectric slab R and T vs Airy",
            worst_airy[0],
            Gate::AtMost(0.01),
            "abs",
            format!(
                "eps 4, 20 cells, f 0.0025 to 0.025; vs exact discrete {identity:.1e}; order {order:.3} over L = 20, 40, 80"
            ),
        ),
        CaseResult::new(
            "R+T",
            "R + T = 1, lossless slab",
            base_rt,
            Gate::AtMost(0.005),
            "abs",
            "the A13 slab at L = 20".into(),
        ),
    ])
}

/// A14: attenuation in a lossy medium (eps 2.25, sigma 0.05). The medium is
/// the background and a soft source on every row of a periodic-y line
/// launches a y-uniform pulse; right of the source the grid carries one
/// outgoing discrete wave, so alpha is minus the least-squares slope of
/// ln |E| over x = xs + 10 .. xs + 60. Band f = 0.008 to 0.02 (at least 33
/// cells per wavelength), picked from the discrete model so the predicted
/// error stays under half the gate. Measured: the worst relative error
/// against the continuum. Guard: within 1e-4 of the exact discrete alpha.
pub fn a14() -> Result<Vec<CaseResult>> {
    let rows = golden(LOSSY, "# S = 0.5, eps = 2.25, sigma = 0.05", "A14")?;
    let freqs = rows
        .iter()
        .map(|r| num(&r[0], "A14"))
        .collect::<Result<Vec<f64>>>()?;
    let (nx, ny, xs) = (220, 3, 100);
    let probes: Vec<usize> = (xs + 10..=xs + 60).step_by(5).collect();
    let scene = Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: 0.5,
        },
        materials: vec![Material::Dielectric {
            eps_inf: 2.25,
            sigma: 0.05,
        }],
        geometry: vec![],
        sources: (0..ny)
            .map(|j| Source {
                i: xs,
                j,
                waveform: Waveform::Ricker {
                    t0: 2.0 / 0.014,
                    f_peak: 0.014,
                },
                amplitude: 1.0,
                kind: SourceKind::Soft,
            })
            .collect(),
        monitors: probes
            .iter()
            .map(|&i| Monitor::Dft {
                i,
                j: 1,
                freqs: freqs.clone(),
            })
            .collect(),
        pml: Some(normal_layer(16)),
        periodic_y: true,
        plane_wave: None,
        paint: None,
        view: None,
    };
    let mut sim = SimTm::from_scene(&scene)?;
    run_until_quiet(&mut sim, 600, 200_000)?;
    let spectra = Kernel::monitor_spectra(&mut sim)?;
    let (mut worst_c, mut worst_d) = (0.0_f64, 0.0_f64);
    for (f, row) in rows.iter().enumerate() {
        let pts: Vec<(f64, f64)> = probes
            .iter()
            .zip(&spectra)
            .filter_map(|(&x, s)| match s {
                Spectrum::Field(v) => Some((x as f64, v[f].0.hypot(v[f].1).ln())),
                Spectrum::Flux(_) => None,
            })
            .collect();
        let alpha = -slope(&pts);
        worst_c = worst_c.max((alpha / num(&row[1], "A14")? - 1.0).abs());
        worst_d = worst_d.max((alpha / num(&row[2], "A14")? - 1.0).abs());
    }
    ensure(worst_d <= 1e-4, || {
        format!("A14: {worst_d:e} off the discrete alpha")
    })?;
    Ok(vec![CaseResult::new(
        "A14",
        "lossy medium attenuation vs analytic",
        worst_c,
        Gate::AtMost(0.01),
        "rel",
        format!("eps 2.25, sigma 0.05, f 0.008 to 0.02; vs exact discrete {worst_d:.1e}"),
    )])
}

/// Steady-state Ez phasors along `row` at columns `xs` in a parallel-plate
/// guide of height 40 (one PEC row at j = 0, periodic y, so its image closes
/// the guide at j = 40), driven by a soft TM1 source column at x = 56 at
/// `steps_per_period`. A DFT over 4 whole periods; steady once two
/// consecutive windows agree to 1e-6.
fn guide_phasors(steps_per_period: usize, row: usize, xs: &[usize]) -> Result<Vec<(f64, f64)>> {
    let (nx, ny, s) = (162, 40, 0.5);
    let f = 1.0 / (steps_per_period as f64 * s);
    let scene = Scene {
        version: SCENE_VERSION,
        grid: GridSpec {
            nx,
            ny,
            dx_phys_m: 1e-3,
            courant: s,
        },
        materials: vec![Material::VACUUM, Material::Pec],
        geometry: vec![Shape::Rect {
            x0: -1.0,
            y0: 0.0,
            x1: nx as f64,
            y1: 0.0,
            material: 1,
        }],
        sources: (1..ny)
            .map(|j| Source {
                i: 56,
                j,
                waveform: Waveform::RampedCw {
                    f0: f,
                    ramp_periods: 10.0,
                    phase: 0.0,
                },
                amplitude: (PI * j as f64 / ny as f64).sin(),
                kind: SourceKind::Soft,
            })
            .collect(),
        monitors: vec![],
        pml: Some(PmlSpec::recommended(16)),
        periodic_y: true,
        plane_wave: None,
        paint: None,
        view: None,
    };
    let mut sim = SimTm::from_scene(&scene)?;
    let window = 4 * steps_per_period;
    let mut last: Option<Vec<(f64, f64)>> = None;
    for _ in 0..200 {
        let mut d = Dft::new(&[f], xs.len())?;
        for _ in 0..window {
            sim.step();
            let t = sim.time_step() as f64 * s;
            let v: Vec<f64> = xs.iter().map(|&i| sim.ez()[row * nx + i]).collect();
            d.add(&v, t, s);
        }
        let now: Vec<(f64, f64)> = (0..xs.len()).map(|p| d.get(p, 0)).collect();
        if let Some(prev) = &last {
            let change = now
                .iter()
                .zip(prev)
                .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1) / a.0.hypot(a.1))
                .fold(0.0, f64::max);
            if change < 1e-6 {
                return Ok(now);
            }
        }
        last = Some(now);
    }
    Err(FdtdError::Case(format!(
        "no steady state at {steps_per_period} steps per period"
    )))
}

/// (fc_c, rows) of the waveguide golden file.
fn guide_golden() -> Result<(f64, Vec<Vec<String>>)> {
    let rows = golden(GUIDE, "# S = 0.5, a = 40, m = 1", "A11")?;
    let header = GUIDE.lines().nth(2).unwrap_or("");
    let fc = header
        .split("fc_c = ")
        .nth(1)
        .and_then(|v| v.split(',').next())
        .unwrap_or("");
    Ok((num(fc, "A11")?, rows))
}

/// A11: the TM1 mode of a parallel-plate guide, a = 40 cells, at 1.30 and
/// 1.60 times cutoff. beta is minus the unwrapped phase slope along the
/// mode's peak row right of the source. Measured: the worst of the guided
/// wavelength error (2 pi / beta vs continuum) and the cutoff error (the
/// zero of a line through (w^2, beta^2), vs f_c = 1 / (2 a)). Guard: beta
/// within 1e-4 of the exact discrete relation.
pub fn a11() -> Result<Vec<CaseResult>> {
    let (fc_c, rows) = guide_golden()?;
    let xs: Vec<usize> = (66..=126).step_by(5).collect();
    let (mut fit, mut lg_worst) = (Vec::new(), 0.0_f64);
    for r in rows.iter().filter(|r| r[0] == "beta") {
        let n = num(&r[1], "A11")? as usize;
        let f = num(&r[2], "A11")?;
        let (vc, vd) = (num(&r[4], "A11")?, num(&r[5], "A11")?);
        let ph = guide_phasors(n, 20, &xs)?;
        let mut pts = Vec::new();
        let (mut prev, mut offset) = (0.0_f64, 0.0);
        for (k, (&x, p)) in xs.iter().zip(&ph).enumerate() {
            let a = p.1.atan2(p.0);
            if k > 0 {
                while a + offset - prev > PI {
                    offset -= 2.0 * PI;
                }
                while a + offset - prev < -PI {
                    offset += 2.0 * PI;
                }
            }
            prev = a + offset;
            pts.push((x as f64, prev));
        }
        let beta = -slope(&pts);
        ensure((beta / vd - 1.0).abs() <= 1e-4, || {
            format!("A11: beta {beta} off the discrete {vd}")
        })?;
        lg_worst = lg_worst.max((vc / beta - 1.0).abs());
        fit.push(((2.0 * PI * f).powi(2), beta * beta));
    }
    ensure(fit.len() == 2, || {
        "A11: need two propagating frequencies".into()
    })?;
    let ((x1, y1), (x2, y2)) = (fit[0], fit[1]);
    let fc = (x1 - y1 * (x2 - x1) / (y2 - y1)).sqrt() / (2.0 * PI);
    let fc_err = (fc / fc_c - 1.0).abs();
    Ok(vec![CaseResult::new(
        "A11",
        "waveguide cutoff and guided wavelength",
        lg_worst.max(fc_err),
        Gate::AtMost(0.005),
        "rel",
        format!(
            "TM1, a = 40, 1.30 and 1.60 f_c; guided wavelength {lg_worst:.1e}, cutoff {fc_err:.1e}"
        ),
    )])
}

/// A12: the same guide at 0.70 times cutoff. alpha is minus the slope of
/// ln |E| right of the source. Measured: the relative error against the
/// continuum. Guard: within 1e-4 of the exact discrete relation.
pub fn a12() -> Result<Vec<CaseResult>> {
    let (_, rows) = guide_golden()?;
    let r = rows
        .iter()
        .find(|r| r[0] == "alpha")
        .ok_or_else(|| FdtdError::Case("A12: no alpha row".into()))?;
    let n = num(&r[1], "A12")? as usize;
    let (vc, vd) = (num(&r[4], "A12")?, num(&r[5], "A12")?);
    let xs: Vec<usize> = (61..=101).step_by(5).collect();
    let ph = guide_phasors(n, 20, &xs)?;
    let pts: Vec<(f64, f64)> = xs
        .iter()
        .zip(&ph)
        .map(|(&x, p)| (x as f64, p.0.hypot(p.1).ln()))
        .collect();
    let alpha = -slope(&pts);
    let ed = (alpha / vd - 1.0).abs();
    ensure(ed <= 1e-4, || {
        format!("A12: alpha {alpha} off the discrete {vd}")
    })?;
    Ok(vec![CaseResult::new(
        "A12",
        "waveguide evanescent decay below cutoff",
        (alpha / vc - 1.0).abs(),
        Gate::AtMost(0.01),
        "rel",
        format!("TM1, a = 40, 0.70 f_c; vs exact discrete {ed:.1e}"),
    )])
}

/// An empty `n x n` vacuum domain, 10-cell layer, plane wave over `[lo, hi]^2`
/// (every row with periodic y), Ricker pulse peaking at 20 cells per wavelength.
pub fn empty_tfsf(n: usize, lo: usize, hi: usize, dir: Direction, periodic: bool) -> Scene {
    let (y0, y1) = if periodic { (0, n - 1) } else { (lo, hi) };
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
        sources: vec![],
        monitors: vec![],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: periodic,
        plane_wave: Some(PlaneWave {
            x0: lo,
            y0,
            x1: hi,
            y1,
            direction: dir,
            waveform: Waveform::Ricker {
                t0: 40.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
        }),
        paint: None,
        view: None,
    }
}

/// (max |Ez| outside the box over the run, max |Ez| inside it).
pub fn leakage(scene: &Scene, steps: usize) -> Result<(f64, f64)> {
    let pw = scene
        .plane_wave
        .ok_or_else(|| FdtdError::Case("A15: no plane wave".into()))?;
    let (nx, ny) = (scene.grid.nx, scene.grid.ny);
    let inside = |i: usize, j: usize| (pw.x0..=pw.x1).contains(&i) && (pw.y0..=pw.y1).contains(&j);
    let mut sim = SimTm::from_scene(scene)?;
    let (mut sf, mut tf) = (0.0_f64, 0.0_f64);
    for _ in 0..steps {
        sim.step();
        for j in 0..ny {
            for i in 0..nx {
                let v = sim.ez()[j * nx + i].abs();
                if inside(i, j) {
                    tf = tf.max(v);
                } else {
                    sf = sf.max(v);
                }
            }
        }
    }
    Ok((sf, tf))
}

/// A15: in an empty domain the scattered-field region stays at or below
/// -120 dB of the incident peak, for all four directions in a closed box
/// and for +x and -x with periodic y. 600 steps take the pulse in, across
/// the 50-cell box, and out again. Measured: the worst leakage in dB.
/// Guard: the incident peak is about the waveform's 1.
pub fn a15() -> Result<Vec<CaseResult>> {
    let mut worst = f64::NEG_INFINITY;
    for periodic in [false, true] {
        for dir in [
            Direction::PlusX,
            Direction::MinusX,
            Direction::PlusY,
            Direction::MinusY,
        ] {
            if periodic && matches!(dir, Direction::PlusY | Direction::MinusY) {
                continue;
            }
            let (sf, tf) = leakage(&empty_tfsf(100, 25, 75, dir, periodic), 600)?;
            ensure((0.9..1.1).contains(&tf), || {
                format!("A15 {dir:?}: incident peak {tf}")
            })?;
            worst = worst.max(20.0 * (sf / tf).log10());
        }
    }
    Ok(vec![CaseResult::new(
        "A15",
        "TF/SF leakage, empty domain",
        worst,
        Gate::AtMost(-120.0),
        "dB",
        "all four directions, closed box and periodic y".into(),
    )])
}
