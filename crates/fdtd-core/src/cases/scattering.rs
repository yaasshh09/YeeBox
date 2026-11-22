//! Far-field cases (`docs/04-validation.md`, Phase 7): the line source
//! isotropy check of the near-to-far-field transform, and the shared
//! runner the scattering, double slit and array cases use.
//!
//! Every run here is broadband: a Ricker pulse, monitors read once the
//! fields have died down (`spectra::run_until_quiet`), so the DFTs are the
//! exact transforms of the whole response. Normalized units.

use super::{CaseResult, Gate, ensure};
use crate::cpml::PmlSpec;
use crate::error::Result;
use crate::kernel::{CpuKernel, Kernel, Spectrum};
use crate::material::Material;
use crate::ntff::ALL_SIDES;
use crate::scene::{DEFAULT_ANGLES, GridSpec, Mode, Monitor, SCENE_VERSION, Scene, Source};
use crate::source::{SourceKind, Waveform};
use crate::spectra::run_until_quiet;

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
