//! Phase 3 acceptance for spectra: A13 (slab R and T), through the public
//! API only. Normalized units (dx = dy = 1, c0 = 1, dt = S); frequencies in
//! cycles per unit time.

use fdtd_core::cpml::PmlSpec;
use fdtd_core::material::Material;
use fdtd_core::scene::{GridSpec, SCENE_VERSION, Scene, Shape};
use fdtd_core::source::Waveform;
use fdtd_core::spectra::{RtSpectrum, rt_spectrum};
use fdtd_core::tfsf::{Direction, PlaneWave};

const SLAB_GOLDEN: &str = include_str!("../../../validation/golden/slab_airy.csv");

/// One golden row: slab thickness, frequency, continuum and discrete R, T.
struct SlabRow {
    l: usize,
    f: f64,
    r_c: f64,
    t_c: f64,
    r_d: f64,
    t_d: f64,
}

fn slab_golden() -> Vec<SlabRow> {
    let mut lines = SLAB_GOLDEN.lines();
    assert_eq!(
        lines.nth(1),
        Some("# S = 0.5, eps = 4.0, base L = 20, slab on nodes 1..L")
    );
    lines
        .filter(|l| !l.starts_with('#') && !l.starts_with('L'))
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            SlabRow {
                l: v[0] as usize,
                f: v[1],
                r_c: v[2],
                t_c: v[3],
                r_d: v[4],
                t_d: v[5],
            }
        })
        .collect()
}

/// An eps = 4 slab of `l` cells in a periodic-y vacuum line (ny = 3), lit
/// by a +x TF/SF plane wave. Layout along x: 16-cell layer, 4 cells, R line,
/// 4 cells, entry face, 20 cells, slab, 20 cells, T line, 4 cells, exit
/// face, 4 cells, layer. Returns the scene and the two line columns.
fn slab_scene(l: usize, f_max: f64) -> (Scene, usize, usize) {
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
        // Nodes s0 ..= s0 + l - 1 on every row: the golden file's "nodes 1..L".
        geometry: vec![Shape::Rect {
            x0: s0 as f64,
            y0: -1.0,
            x1: (s0 + l - 1) as f64,
            y1: 3.0,
            material: 1,
        }],
        sources: vec![],
        monitors: vec![],
        // Every wave in this scene meets the x layers at normal incidence,
        // so the CFS alpha has nothing to do and costs absorption at long
        // wavelengths (-41 dB at f = 0.0025 for the preset, `fdtd1d`
        // layer_reflection_against_frequency). Graded preset, alpha 0.
        pml: Some(PmlSpec {
            alpha_max: 0.0,
            ..PmlSpec::recommended(d)
        }),
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
    };
    (scene, r_at, t_at)
}

fn measure(l: usize, rows: &[&SlabRow]) -> RtSpectrum {
    let freqs: Vec<f64> = rows.iter().map(|r| r.f).collect();
    let f_max = freqs.iter().cloned().fold(0.0, f64::max);
    let (scene, r_at, t_at) = slab_scene(l, f_max);
    // One pulse length of quiet: the Ricker lasts about 2 t0 = 4 / f_peak.
    let window = (8.0 / f_max / 0.5) as u64;
    rt_spectrum(&scene, r_at, t_at, &freqs, window, 400_000).unwrap()
}

/// Worst |R - R_c|, |T - T_c|, |R + T - 1|, and worst distance from the
/// exact discrete slab, max(|R - R_d|, |T - T_d|).
fn errors(rt: &RtSpectrum, rows: &[&SlabRow], verbose: bool) -> (f64, f64, f64, f64) {
    let mut e = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (k, row) in rows.iter().enumerate() {
        let (r, t) = (rt.r[k], rt.t[k]);
        if verbose {
            eprintln!(
                "  L {} f {:.5}: R {r:.6} (Airy {:.6}, discrete {:.6}), T {t:.6} (Airy {:.6}), R+T-1 {:+.1e}",
                row.l,
                row.f,
                row.r_c,
                row.r_d,
                row.t_c,
                r + t - 1.0
            );
        }
        e.0 = e.0.max((r - row.r_c).abs());
        e.1 = e.1.max((t - row.t_c).abs());
        e.2 = e.2.max((r + t - 1.0).abs());
        e.3 = e.3.max((r - row.r_d).abs().max((t - row.t_d).abs()));
    }
    e
}

/// A13: an eps = 4 slab, 20 cells thick, at normal incidence. Over the band
/// where the wavelength inside the slab is at least 20 cells (f from 0.0025
/// to 0.025), R and T are within 0.01 of the Airy formula and R + T = 1 to
/// 0.005. The measurement chain itself (TF/SF, flux lines, normalization)
/// is checked against the exact discrete slab to 1e-4: what is left over is
/// the grid's own discretization error, which the golden file predicts at
/// 4.9e-3 at the band edge. Then the same physical slab at L = 40 and 80,
/// frequencies scaled by 20 / L: the error against Airy falls at order 2.
#[test]
fn a13_dielectric_slab() {
    let golden = slab_golden();
    let mut worst_airy = Vec::new();
    for l in [20, 40, 80] {
        let rows: Vec<&SlabRow> = golden.iter().filter(|r| r.l == l).collect();
        assert_eq!(rows.len(), 19);
        let rt = measure(l, &rows);
        let (er, et, erl, ed) = errors(&rt, &rows, l == 20);
        eprintln!(
            "A13 L {l}: |R - R_c| {er:.2e}, |T - T_c| {et:.2e}, |R + T - 1| {erl:.1e}, vs discrete {ed:.1e}, steps {:?}",
            rt.steps
        );
        assert!(
            ed <= 1e-4,
            "L {l}: measurement off the exact discrete slab by {ed:e}"
        );
        if l == 20 {
            assert!(er <= 0.01 && et <= 0.01, "A13: R {er}, T {et}");
            assert!(erl <= 0.005, "R + T - 1 = {erl}");
        }
        worst_airy.push(er.max(et));
    }
    let order = |a: f64, b: f64| (a / b).log2();
    let (o1, o2) = (
        order(worst_airy[0], worst_airy[1]),
        order(worst_airy[1], worst_airy[2]),
    );
    // Least squares slope of log2(error) against log2(L) over the three points.
    let fit = -(worst_airy[2].log2() - worst_airy[0].log2()) / 2.0;
    eprintln!("A13 convergence: orders {o1:.3}, {o2:.3}, fitted {fit:.3}");
    assert!((1.8..=2.2).contains(&fit), "order {fit}");
}

const LOSSY_GOLDEN: &str = include_str!("../../../validation/golden/lossy_medium.csv");

/// A14: attenuation in a lossy medium (eps 2.25, sigma 0.05, normalized).
/// The medium is the background, and a soft source on every row of a
/// periodic-y line launches a y-uniform pulse. To the right of the source
/// the grid then carries exactly one outgoing discrete wave, so
/// |E(x)| = |A| exp(-alpha (x - xs)), and alpha is minus the least-squares
/// slope of ln |E| over x = xs + 10 .. xs + 60. Gate: within 1 percent of
/// the continuum alpha = -Im(w sqrt(eps - j sigma / w)). The exact discrete
/// alpha is matched to 1e-4, which checks the measurement itself. The band
/// (f = 0.008 to 0.02, at least 33 cells per wavelength) was picked before
/// any run so the predicted discretization error stays under half the gate.
#[test]
fn a14_lossy_attenuation() {
    use fdtd_core::kernel::{Kernel, Spectrum};
    use fdtd_core::scene::{Monitor, Source};
    use fdtd_core::source::SourceKind;
    use fdtd_core::spectra::run_until_quiet;
    use fdtd_core::tm::SimTm;

    let mut lines = LOSSY_GOLDEN.lines();
    assert_eq!(lines.nth(1), Some("# S = 0.5, eps = 2.25, sigma = 0.05"));
    let rows: Vec<Vec<f64>> = lines
        .filter(|l| !l.starts_with('#') && !l.starts_with('f'))
        .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
        .collect();
    let freqs: Vec<f64> = rows.iter().map(|r| r[0]).collect();

    let (nx, ny, xs) = (220, 3, 100);
    let xs_probe: Vec<usize> = (xs + 10..=xs + 60).step_by(5).collect();
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
        monitors: xs_probe
            .iter()
            .map(|&i| Monitor::Dft {
                i,
                j: 1,
                freqs: freqs.clone(),
            })
            .collect(),
        // Normal incidence only, as in A13: graded preset without alpha.
        pml: Some(PmlSpec {
            alpha_max: 0.0,
            ..PmlSpec::recommended(16)
        }),
        periodic_y: true,
        plane_wave: None,
    };
    let mut sim = SimTm::from_scene(&scene).unwrap();
    let steps = run_until_quiet(&mut sim, 600, 200_000).unwrap();
    let spectra = Kernel::monitor_spectra(&mut sim).unwrap();

    let (mut worst_c, mut worst_d) = (0.0_f64, 0.0_f64);
    for (f, row) in rows.iter().enumerate() {
        // Least-squares slope of ln |E| against x.
        let pts: Vec<(f64, f64)> = xs_probe
            .iter()
            .zip(&spectra)
            .map(|(&x, s)| match s {
                Spectrum::Field(v) => (x as f64, v[f].0.hypot(v[f].1).ln()),
                Spectrum::Flux(_) => unreachable!(),
            })
            .collect();
        let n = pts.len() as f64;
        let (mx, my) = (
            pts.iter().map(|p| p.0).sum::<f64>() / n,
            pts.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let slope = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
            / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>();
        let alpha = -slope;
        let (ec, ed) = ((alpha / row[1] - 1.0).abs(), (alpha / row[2] - 1.0).abs());
        eprintln!(
            "A14 f {:.3}: alpha {alpha:.8} (continuum {:.8}, discrete {:.8}), {:.3}% off continuum, {ed:.1e} off discrete",
            row[0],
            row[1],
            row[2],
            100.0 * ec
        );
        worst_c = worst_c.max(ec);
        worst_d = worst_d.max(ed);
    }
    eprintln!(
        "A14: worst {:.3}% off continuum (gate 1%), {worst_d:.1e} off discrete, {steps} steps",
        100.0 * worst_c
    );
    assert!(
        worst_d <= 1e-4,
        "measurement off the discrete alpha by {worst_d:e}"
    );
    assert!(worst_c <= 0.01, "A14 {worst_c}");
}

const GUIDE_GOLDEN: &str = include_str!("../../../validation/golden/waveguide.csv");

/// Steady-state Ez phasors along row `row` at the columns `xs` for a
/// parallel-plate guide of height 40 (one PEC row at j = 0, periodic y, so
/// its image closes the guide at j = 40) driven at `steps_per_period` by a
/// soft TM1 source column at x = 56. The phasor is a DFT over 4 whole
/// periods; it is steady once two consecutive windows agree to 1e-6.
/// Returns the phasors and the step count.
fn guide_phasors(steps_per_period: usize, row: usize, xs: &[usize]) -> (Vec<(f64, f64)>, u64) {
    use fdtd_core::monitor::Dft;
    use fdtd_core::scene::Source;
    use fdtd_core::source::SourceKind;
    use fdtd_core::tm::SimTm;
    use std::f64::consts::PI;

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
    };
    let mut sim = SimTm::from_scene(&scene).unwrap();
    let window = 4 * steps_per_period;
    let mut last: Option<Vec<(f64, f64)>> = None;
    for _ in 0..200 {
        let mut d = Dft::new(&[f], xs.len()).unwrap();
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
                return (now, sim.time_step());
            }
        }
        last = Some(now);
    }
    panic!("no steady state at {steps_per_period} steps per period");
}

/// Least-squares slope of y against x.
fn slope(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>()
        / pts.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>()
}

/// A11 and A12: the TM1 mode of a parallel-plate guide, a = 40 cells.
/// Above cutoff (1.30 and 1.60 f_c) beta is minus the phase slope along the
/// guide right of the source; the guided wavelength 2 pi / beta is gated at
/// 0.5 percent against continuum, and the cutoff, from a straight line
/// through (w^2, beta^2), at 0.5 percent against f_c = 1 / (2 a). Below
/// cutoff (0.70 f_c) alpha is minus the slope of ln |E|, gated at 1
/// percent. Against the exact discrete relation beta and alpha must agree
/// to 1e-4, which checks the measurement itself.
#[test]
fn a11_a12_parallel_plate_guide() {
    use std::f64::consts::PI;
    let mut lines = GUIDE_GOLDEN.lines();
    assert_eq!(lines.nth(1), Some("# S = 0.5, a = 40, m = 1"));
    let header = lines.next().unwrap();
    let fc_c: f64 = header
        .split("fc_c = ")
        .nth(1)
        .and_then(|v| v.split(',').next())
        .unwrap()
        .parse()
        .unwrap();
    let rows: Vec<Vec<String>> = lines
        .filter(|l| !l.starts_with("kind"))
        .map(|l| l.split(',').map(String::from).collect())
        .collect();

    let row = 20;
    let mut fit = Vec::new();
    for r in &rows {
        let n: usize = r[1].parse().unwrap();
        let f: f64 = r[2].parse().unwrap();
        let (vc, vd): (f64, f64) = (r[4].parse().unwrap(), r[5].parse().unwrap());
        if r[0] == "beta" {
            let xs: Vec<usize> = (66..=126).step_by(5).collect();
            let (ph, steps) = guide_phasors(n, row, &xs);
            // Unwrap the phase along x, then fit.
            let mut pts = Vec::new();
            let mut prev = 0.0_f64;
            let mut offset = 0.0;
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
            let lg_err = (vc / beta - 1.0).abs();
            eprintln!(
                "A11 f/fc {:.4}: beta {beta:.8} (continuum {vc:.8}, discrete {vd:.8}), guided wavelength {:.3}% off continuum, {:.1e} off discrete, {steps} steps",
                f / fc_c,
                100.0 * lg_err,
                (beta / vd - 1.0).abs()
            );
            assert!(
                (beta / vd - 1.0).abs() <= 1e-4,
                "beta off the discrete relation"
            );
            assert!(lg_err <= 0.005, "A11 guided wavelength {lg_err}");
            fit.push(((2.0 * PI * f).powi(2), beta * beta));
        } else {
            let xs: Vec<usize> = (61..=101).step_by(5).collect();
            let (ph, steps) = guide_phasors(n, row, &xs);
            let pts: Vec<(f64, f64)> = xs
                .iter()
                .zip(&ph)
                .map(|(&x, p)| (x as f64, p.0.hypot(p.1).ln()))
                .collect();
            let alpha = -slope(&pts);
            let err = (alpha / vc - 1.0).abs();
            eprintln!(
                "A12 f/fc {:.4}: alpha {alpha:.8} (continuum {vc:.8}, discrete {vd:.8}), {:.3}% off continuum, {:.1e} off discrete, {steps} steps",
                f / fc_c,
                100.0 * err,
                (alpha / vd - 1.0).abs()
            );
            assert!(
                (alpha / vd - 1.0).abs() <= 1e-4,
                "alpha off the discrete relation"
            );
            assert!(err <= 0.01, "A12 {err}");
        }
    }
    let ((x1, y1), (x2, y2)) = (fit[0], fit[1]);
    let wc = (x1 - y1 * (x2 - x1) / (y2 - y1)).sqrt();
    let fc = wc / (2.0 * PI);
    let err = (fc / fc_c - 1.0).abs();
    eprintln!(
        "A11 cutoff: {fc:.8} from the fit, continuum {fc_c}, {:.3}% off",
        100.0 * err
    );
    assert!(err <= 0.005, "A11 cutoff {err}");
}
