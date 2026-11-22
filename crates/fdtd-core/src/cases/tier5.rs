//! Tier 5: YeeBox against Meep (`docs/04-validation.md`). The Meep runs are
//! offline (`validation/meep/*.py`, in WSL, versions pinned in
//! `validation/meep/environment.txt`); their results are committed CSV, and
//! these functions run the same scenes here and compare. Both sides use
//! Courant 0.5, one Meep pixel per YeeBox cell and no subpixel smoothing,
//! so what is left is two independent implementations of the same scheme.
//! Recorded, not gated: Meep is a cross-check, not the reference.

use super::plane_wave::{num, slab_scene_with};
use super::scattering::{Cylinder, KA, incident, measured, series};
use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::error::{FdtdError, Result};
use crate::kernel::{CpuKernel, Kernel, Spectrum};
use crate::material::Material;
use crate::monitor::Normal;
use crate::scene::{DEFAULT_ANGLES, GridSpec, Mode, Monitor, SCENE_VERSION, Scene, Shape, Source};
use crate::source::{SourceKind, Waveform};
use crate::spectra::{rt_spectrum, run_until_quiet};

const CYLINDER: &str = include_str!("../../../../validation/meep/cylinder.csv");
const DRUDE: &str = include_str!("../../../../validation/meep/drude_slab.csv");
const BEND: &str = include_str!("../../../../validation/meep/bend.csv");

/// The numeric rows of a Meep CSV (comment and header lines dropped).
fn rows(text: &str, columns: usize, what: &str) -> Result<Vec<Vec<f64>>> {
    let out: Vec<Vec<f64>> = text
        .lines()
        .skip(2)
        .map(|l| {
            l.split(',')
                .map(|c| num(c, what))
                .collect::<Result<Vec<f64>>>()
        })
        .collect::<Result<_>>()?;
    ensure(
        out.iter().all(|r| r.len() == columns) && !out.is_empty(),
        || format!("{what}: malformed Meep data"),
    )?;
    Ok(out)
}

/// The A20 cylinder (eps 4, radius 40 cells, TMz): the worst dB difference
/// between YeeBox and Meep without smoothing over the angles A20 compares
/// (series within 20 dB of its peak), k a = 1, 2, 5. Also returns Meep's
/// own worst error against the series, for the record.
pub fn cylinder_vs_meep() -> Result<(f64, f64)> {
    let meep = rows(CYLINDER, 4, "meep cylinder")?;
    let a = 40;
    let inc = incident(a, Mode::Tm)?;
    let run = measured(a, Mode::Tm, Cylinder::Dielectric, &inc)?;
    let (mut vs_meep, mut meep_vs_series) = (0.0_f64, 0.0_f64);
    for (k, &ka) in KA.iter().enumerate() {
        let gold = series("dielectric_tm", ka)?;
        let top = gold.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let theirs: Vec<f64> = meep.iter().filter(|r| r[0] == ka).map(|r| r[2]).collect();
        ensure(theirs.len() == DEFAULT_ANGLES, || {
            format!("meep cylinder: {} rows at k a {ka}", theirs.len())
        })?;
        for m in (0..DEFAULT_ANGLES).filter(|&m| gold[m] >= top - 20.0) {
            vs_meep = vs_meep.max((run.sigma[k][m] - theirs[m]).abs());
            meep_vs_series = meep_vs_series.max((theirs[m] - gold[m]).abs());
        }
    }
    Ok((vs_meep, meep_vs_series))
}

/// The A16 Drude slab in TMz: the worst |R_yeebox - R_meep| and |T_yeebox -
/// T_meep| over the golden band, and Meep's own worst error against the
/// golden R_c, T_c.
pub fn drude_vs_meep() -> Result<(f64, f64)> {
    let meep = rows(DRUDE, 5, "meep drude slab")?;
    let (material, golden) = super::dispersive::golden_slab("drude", "tier 5")?;
    let freqs: Vec<f64> = golden.iter().map(|r| r.0).collect();
    ensure(meep.len() == freqs.len(), || {
        "meep drude slab: other frequencies".into()
    })?;
    let f_max = freqs.iter().cloned().fold(0.0, f64::max);
    let (scene, r_at, t_at) = slab_scene_with(super::dispersive::L, f_max, material, Mode::Tm);
    let window = (8.0 / f_max / 0.5) as u64;
    let rt = rt_spectrum(&scene, r_at, t_at, &freqs, window, 400_000)?;
    let (mut vs_meep, mut meep_vs_golden) = (0.0_f64, 0.0_f64);
    for (k, m) in meep.iter().enumerate() {
        ensure((m[0] - freqs[k]).abs() < 1e-12, || {
            "meep drude slab: frequency mismatch".into()
        })?;
        vs_meep = vs_meep
            .max((rt.r[k] - m[1]).abs())
            .max((rt.t[k] - m[2]).abs());
        meep_vs_golden = meep_vs_golden
            .max((m[1] - m[3]).abs())
            .max((m[2] - m[4]).abs());
    }
    Ok((vs_meep, meep_vs_golden))
}

/// The Meep tutorial bend in YeeBox, one node per Meep pixel at resolution
/// 10: a 161 x 161 grid (16 units), the 10-cell recommended layer, an eps
/// 12 guide on rows 41..=50 (10 nodes; Meep's guide is shifted half a pixel
/// so it covers the same 10 points) that, when `bent`, turns up on columns
/// 111..=120 and runs out through +y. A line of soft modulated-Gaussian
/// sources across the guide at column 15 (Meep's line source, f 0.15 and
/// width 0.1 in its units), and a flux line across twice the guide width:
/// column 145 for the straight guide, row 145 for the bent arm.
pub fn bend_scene(bent: bool) -> Scene {
    let n = 161;
    let rect = |x0: usize, y0: usize, x1: usize, y1: usize| Shape::Rect {
        x0: x0 as f64,
        y0: y0 as f64,
        x1: x1 as f64,
        y1: y1 as f64,
        material: 1,
    };
    let geometry = if bent {
        vec![rect(0, 41, 120, 50), rect(111, 41, 120, n - 1)]
    } else {
        vec![rect(0, 41, n - 1, 50)]
    };
    // Meep's GaussianSource is exp(-(t - t0)^2 / (2 w^2)) with w = 1 /
    // fwidth Meep time units, 100 YeeBox units here, so tau = sqrt(2) 100.
    let tau = 2.0_f64.sqrt() * 100.0;
    let waveform = Waveform::ModulatedGaussian {
        t0: 6.0 * tau,
        tau,
        f0: 0.015,
    };
    let freqs = bend_freqs();
    let monitor = if bent {
        Monitor::Flux {
            normal: Normal::Y,
            at: 145,
            from: 106,
            to: 125,
            freqs,
        }
    } else {
        Monitor::Flux {
            normal: Normal::X,
            at: 145,
            from: 36,
            to: 55,
            freqs,
        }
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
            Material::Dielectric {
                eps_inf: 12.0,
                sigma: 0.0,
            },
        ],
        geometry,
        sources: (41..=50)
            .map(|j| Source {
                i: 15,
                j,
                waveform,
                amplitude: 1.0,
                kind: SourceKind::Soft,
            })
            .collect(),
        monitors: vec![monitor],
        pml: Some(PmlSpec::recommended(10)),
        periodic_y: false,
        plane_wave: None,
        paint: None,
        view: None,
        mode: Mode::Tm,
    }
}

/// The bend frequencies, YeeBox units: Meep's 0.1 to 0.2 over 10.
pub fn bend_freqs() -> Vec<f64> {
    (0..100).map(|k| 0.01 + 0.01 * k as f64 / 99.0).collect()
}

/// T(f) of the bend: the flux into the bent arm over the flux down the
/// straight guide.
pub fn bend_transmission() -> Result<Vec<f64>> {
    let flux = |bent| -> Result<Vec<f64>> {
        let mut sim = CpuKernel::from_scene(&bend_scene(bent))?;
        run_until_quiet(&mut sim, 400, 200_000)?;
        match sim.monitor_spectra()?.pop() {
            Some(Spectrum::Flux(v)) => Ok(v),
            other => Err(FdtdError::Case(format!("bend flux: {other:?}"))),
        }
    };
    let (straight, bend) = (flux(false)?, flux(true)?);
    Ok(bend.iter().zip(&straight).map(|(b, s)| b / s).collect())
}

/// The worst |T_yeebox - T_meep| over the bend band.
pub fn bend_vs_meep() -> Result<f64> {
    let meep = rows(BEND, 2, "meep bend")?;
    let freqs = bend_freqs();
    ensure(meep.len() == freqs.len(), || {
        "meep bend: other frequencies".into()
    })?;
    let ours = bend_transmission()?;
    let mut worst = 0.0_f64;
    for ((m, t), f) in meep.iter().zip(&ours).zip(&freqs) {
        ensure((m[0] / 10.0 - f).abs() < 1e-12, || {
            "meep bend: frequency mismatch".into()
        })?;
        worst = worst.max((t - m[1]).abs());
    }
    Ok(worst)
}

/// The three cross-checks as recorded rows.
pub fn tier5() -> Result<Vec<CaseResult>> {
    let (cyl, cyl_meep) = cylinder_vs_meep()?;
    let (drude, drude_meep) = drude_vs_meep()?;
    let bend = bend_vs_meep()?;
    Ok(vec![
        CaseResult::new(
            "T5 cylinder",
            "eps 4 cylinder, YeeBox minus Meep",
            cyl,
            Gate::Record,
            "dB",
            format!(
                "radius 40 cells, TMz, k a = 1, 2, 5, the A20 angles; \
                 Meep itself is {cyl_meep:.3} dB from the series"
            ),
        ),
        CaseResult::new(
            "T5 Drude",
            "Drude slab, YeeBox minus Meep",
            drude,
            Gate::Record,
            "abs",
            format!(
                "the A16 slab, TMz, worst of R and T; \
                 Meep itself is {drude_meep:.2e} from the analytic R, T"
            ),
        ),
        CaseResult::new(
            "T5 bend",
            "waveguide bend transmission, YeeBox minus Meep",
            bend,
            Gate::Record,
            "abs",
            "the Meep tutorial bend, eps 12 guide 10 cells wide, 100 frequencies".into(),
        ),
    ])
}
