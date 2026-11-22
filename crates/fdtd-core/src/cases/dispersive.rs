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

/// A16: Drude slab, eps_inf 1, wp = 2 pi 0.02, gamma 0.005.
pub fn a16() -> Result<Vec<CaseResult>> {
    slab_case("A16", "Drude slab R and T vs complex eps", "drude")
}

/// A17: Lorentz slab, eps_inf 1, d_eps 2, w0 = 2 pi 0.02, delta 0.02.
pub fn a17() -> Result<Vec<CaseResult>> {
    slab_case("A17", "Lorentz slab R and T vs complex eps", "lorentz")
}
