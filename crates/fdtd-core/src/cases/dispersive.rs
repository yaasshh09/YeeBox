//! A16, A17: Drude and Lorentz slabs at normal incidence against the
//! continuum Airy R and T with the complex eps(w) (`validation/gen/
//! slab_dispersive.py`, exp(+jwt) like `dispersive.rs`).
//!
//! The slab is the A13 setup (`plane_wave::slab_scene_with`): 20 cells on
//! nodes 1..L in a periodic-y vacuum line, a +x TF/SF Ricker pulse, R and T
//! from two flux lines and a normalization run. Each slab runs in TMz and in
//! TEz; at normal incidence both see the same physical slab (in TEz the two
//! face edges take the half-weight mean, a second-order interface).
//!
//! The band is the generator's: every frequency of 0.004 to 0.045 where the
//! wavelength inside the material, lambda0 / sqrt(|eps(w)|), is at least 20
//! cells. Both models put their plasma or resonance frequency at f = 0.02,
//! so the band crosses it. Measured: the worst |R - R_c| and |T - T_c| over
//! the band and both modes. Gate 0.02 absolute.

use super::plane_wave::{golden, num, slab_scene_with};
use super::{CaseResult, Gate, ensure};
use crate::error::Result;
use crate::material::Material;
use crate::scene::Mode;
use crate::spectra::rt_spectrum;

const GOLDEN: &str = include_str!("../../../../validation/golden/slab_dispersive.csv");
const PARAMS: &str =
    "# S = 0.5, L = 20, slab on nodes 1..L, exp(+jwt), lambda in material >= 20.0 cells";
const L: usize = 20;

/// One golden row: frequency, continuum R and T.
type Row = (f64, f64, f64);

/// The golden material of `kind` and its rows.
fn golden_slab(kind: &str, what: &str) -> Result<(Material, Vec<Row>)> {
    let rows: Vec<Vec<String>> = golden(GOLDEN, PARAMS, what)?
        .into_iter()
        .filter(|r| r[0] == kind)
        .collect();
    ensure(rows.len() > 20, || {
        format!("{what}: {} golden rows", rows.len())
    })?;
    let n = |r: &[String], c: usize| num(&r[c], what);
    let (eps_inf, a, b, c) = (
        n(&rows[0], 1)?,
        n(&rows[0], 2)?,
        n(&rows[0], 3)?,
        n(&rows[0], 4)?,
    );
    let material = if kind == "drude" {
        Material::Drude {
            eps_inf,
            wp: a,
            gamma: c,
        }
    } else {
        Material::Lorentz {
            eps_inf,
            d_eps: a,
            w0: b,
            delta: c,
        }
    };
    let data = rows
        .iter()
        .map(|r| Ok((n(r, 5)?, n(r, 6)?, n(r, 7)?)))
        .collect::<Result<_>>()?;
    Ok((material, data))
}

/// Worst |R - R_c|, |T - T_c| of the `kind` slab in `mode`.
fn slab_error(kind: &str, mode: Mode, what: &str) -> Result<f64> {
    let (material, rows) = golden_slab(kind, what)?;
    let freqs: Vec<f64> = rows.iter().map(|r| r.0).collect();
    let f_max = freqs.iter().cloned().fold(0.0, f64::max);
    let (scene, r_at, t_at) = slab_scene_with(L, f_max, material, mode);
    let window = (8.0 / f_max / 0.5) as u64;
    let rt = rt_spectrum(&scene, r_at, t_at, &freqs, window, 400_000)?;
    Ok(rows
        .iter()
        .enumerate()
        .map(|(k, &(_, r, t))| (rt.r[k] - r).abs().max((rt.t[k] - t).abs()))
        .fold(0.0, f64::max))
}

fn slab_case(id: &'static str, name: &'static str, kind: &str) -> Result<Vec<CaseResult>> {
    let (tm, te) = (
        slab_error(kind, Mode::Tm, id)?,
        slab_error(kind, Mode::Te, id)?,
    );
    let (_, rows) = golden_slab(kind, id)?;
    let (lo, hi) = (rows[0].0, rows[rows.len() - 1].0);
    Ok(vec![CaseResult::new(
        id,
        name,
        tm.max(te),
        Gate::AtMost(0.02),
        "abs",
        format!(
            "20 cells, f {lo} to {hi:.3} ({} points), TMz {tm:.1e}, TEz {te:.1e}",
            rows.len()
        ),
    )])
}

/// The A21 cylinder at radius `a` cells: the measured resonance and the
/// quasi-static one, wp / sqrt(eps_inf + 1), both angular, normalized.
///
/// A Drude cylinder (eps_inf 1) in a TEz vacuum, with wp set so that k a =
/// 0.1 at the resonance (quasi-static) and gamma = 0.1 of it (Q about 10).
/// A +x TF/SF Ricker pulse peaking at the resonance lights it; a probe on
/// the backscatter axis in the scattered-field region sees only the
/// cylinder's field. Its spectrum over the Ricker's analytic spectrum
/// (|W| proportional to (f / f_p)^2 exp(-(f / f_p)^2), so no normalization
/// run) peaks at the resonance; the record runs until the ring has decayed
/// by e^-10, so no window is needed.
pub fn drude_cylinder(a: usize) -> Result<(f64, f64)> {
    let w_res = 0.1 / a as f64;
    let (wp, gamma) = (w_res * 2.0_f64.sqrt(), 0.1 * w_res);
    let (d, gap) = (16, 4);
    let half = a + 6;
    let c = d + gap + 4 + half;
    let n = 2 * c + 1;
    let f_p = w_res / (2.0 * std::f64::consts::PI);
    let scene = crate::scene::Scene {
        version: crate::scene::SCENE_VERSION,
        grid: crate::scene::GridSpec {
            nx: n,
            ny: n,
            dx_phys_m: 1e-9,
            courant: 0.5,
        },
        materials: vec![
            Material::VACUUM,
            Material::Drude {
                eps_inf: 1.0,
                wp,
                gamma,
            },
        ],
        geometry: vec![crate::scene::Shape::Circle {
            cx: c as f64,
            cy: c as f64,
            r: a as f64,
            material: 1,
        }],
        sources: vec![],
        monitors: vec![crate::scene::Monitor::Point {
            i: c - half - 3,
            j: c,
        }],
        pml: Some(crate::cpml::PmlSpec::recommended(d)),
        periodic_y: false,
        plane_wave: Some(crate::tfsf::PlaneWave {
            x0: c - half,
            y0: c - half,
            x1: c + half,
            y1: c + half,
            direction: crate::tfsf::Direction::PlusX,
            waveform: crate::source::Waveform::Ricker {
                t0: 2.0 / f_p,
                f_peak: f_p,
            },
            amplitude: 1.0,
        }),
        paint: None,
        view: None,
        mode: Mode::Te,
    };
    let s = scene.grid.courant;
    let steps = ((4.0 / f_p + 20.0 / gamma) / s) as u64;
    let mut sim = crate::te::SimTe::from_scene(&scene)?;
    sim.step_n(steps);
    let rec = sim.probes()[0].history().to_vec();
    ensure(super::signal::max_abs(&rec) > 0.0, || {
        "A21: the probe saw nothing".into()
    })?;
    let ricker = |w: f64| {
        let x = w / w_res;
        x * x * (-x * x).exp()
    };
    let response = |w: f64| {
        let (re, im) = super::signal::dtft(&rec, None, w, s);
        re.hypot(im) / ricker(w)
    };
    let grid: Vec<f64> = (0..=200)
        .map(|k| w_res * (0.5 + k as f64 / 200.0))
        .collect();
    let best = (1..grid.len() - 1)
        .max_by(|&i, &j| response(grid[i]).total_cmp(&response(grid[j])))
        .unwrap_or(1);
    ensure(best > 1 && best < grid.len() - 2, || {
        format!("A21 a = {a}: the peak is on the edge of the search")
    })?;
    let w = super::signal::golden_max(response, grid[best - 1], grid[best + 1], 1e-12);
    Ok((w, wp / 2.0_f64.sqrt()))
}

/// A21: the small Drude cylinder resonance in TEz within 5 percent of
/// wp / sqrt(eps_inf + 1), gated at a = 10 cells, with a = 5 beside it for
/// the staircasing trend (the a = 20 run is in `fdtd report`).
pub fn a21() -> Result<Vec<CaseResult>> {
    let err = |a| -> Result<f64> {
        let (w, theory) = drude_cylinder(a)?;
        Ok((w - theory) / theory)
    };
    let (e5, e10) = (err(5)?, err(10)?);
    Ok(vec![CaseResult::new(
        "A21",
        "Drude cylinder resonance vs wp / sqrt(eps_inf + 1), TEz",
        e10.abs(),
        Gate::AtMost(0.05),
        "rel",
        format!("k a = 0.1, Q 10; radius 10 cells {e10:+.2e}, 5 cells {e5:+.2e}"),
    )])
}

/// A16: Drude slab, eps_inf 1, wp = 2 pi 0.02, gamma 0.005.
pub fn a16() -> Result<Vec<CaseResult>> {
    slab_case("A16", "Drude slab R and T vs complex eps", "drude")
}

/// A17: Lorentz slab, eps_inf 1, d_eps 2, w0 = 2 pi 0.02, delta 0.02.
pub fn a17() -> Result<Vec<CaseResult>> {
    slab_case("A17", "Lorentz slab R and T vs complex eps", "lorentz")
}
