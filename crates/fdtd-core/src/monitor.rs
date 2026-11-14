//! Field monitors: point probes and running DFTs.
//!
//! Time convention for every DFT here: a sample enters at its true time.
//! After step n the solver holds E at t = n S and H at t = (n - 1/2) S, so
//! H samples are added with `t = (n - 1/2) S`. That is the half-step
//! correction of `docs/02-numerics.md`; the doc's `exp(-j w dt / 2)` factor is
//! the same thing written for H^(n + 1/2) stored under index n.

use crate::error::{FdtdError, Result};
use crate::tm::SimTm;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Running discrete Fourier transform of `points` real signals at a list of
/// frequencies: `F(f) += v(t) * exp(-j 2 pi f t) * dt`, accumulated during
/// stepping. Frequencies are normalized (cycles per unit time), t and dt are
/// normalized time. Complex values are `(re, im)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dft {
    freqs: Vec<f64>,
    points: usize,
    /// `sums[p * freqs.len() + f]`.
    sums: Vec<(f64, f64)>,
}

impl Dft {
    /// A zeroed transform of `points` signals at `freqs`, each finite and > 0.
    pub fn new(freqs: &[f64], points: usize) -> Result<Self> {
        if freqs.is_empty() {
            return Err(FdtdError::InvalidParameter {
                name: "dft frequencies",
                value: 0.0,
            });
        }
        if let Some(&f) = freqs.iter().find(|f| !(f.is_finite() && **f > 0.0)) {
            return Err(FdtdError::InvalidParameter {
                name: "dft frequency",
                value: f,
            });
        }
        Ok(Self {
            freqs: freqs.to_vec(),
            points,
            sums: vec![(0.0, 0.0); points * freqs.len()],
        })
    }

    /// Adds one sample of every signal, all taken at time `t`, weight `dt`.
    /// `values.len()` must equal `points`.
    pub fn add(&mut self, values: &[f64], t: f64, dt: f64) {
        debug_assert_eq!(values.len(), self.points);
        let nf = self.freqs.len();
        for (f, &freq) in self.freqs.iter().enumerate() {
            let ph = 2.0 * PI * freq * t;
            let (c, s) = (ph.cos() * dt, -ph.sin() * dt);
            for (p, &v) in values.iter().enumerate() {
                let acc = &mut self.sums[p * nf + f];
                acc.0 += v * c;
                acc.1 += v * s;
            }
        }
    }

    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Transform of signal `point` at frequency index `f`.
    pub fn get(&self, point: usize, f: usize) -> (f64, f64) {
        self.sums[point * self.freqs.len() + f]
    }
}

/// Which way a flux line's normal points. JSON form: `"x"` or `"y"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normal {
    /// A column segment at x = `at`, nodes j in `from..=to`; positive flux is +x.
    X,
    /// A row segment at y = `at`, nodes i in `from..=to`; positive flux is +y.
    Y,
}

/// Poynting flux spectrum through a line segment of Ez nodes, TMz.
///
/// With E = Ez z and H = (Hx, Hy), S = E x H gives `Sx = -Ez Hy` and
/// `Sy = Ez Hx`. H is averaged onto each node from its two half-cell edges
/// along the normal (Hy from i +- 1/2, Hx from j +- 1/2), and every sample
/// enters the DFT at its true time. The flux per frequency is
/// `sum over the segment of -1/2 Re(Ez conj(Hy))` (or `+1/2 Re(Ez conj(Hx))`),
/// in normalized units, cell width 1.
#[derive(Debug, Clone, PartialEq)]
pub struct FluxLine {
    pub normal: Normal,
    pub at: usize,
    pub from: usize,
    pub to: usize,
    e: Dft,
    h: Dft,
    e_buf: Vec<f64>,
    h_buf: Vec<f64>,
}

impl FluxLine {
    /// A line at `at` across nodes `from..=to`. An x-normal line needs the
    /// edge at `at - 1/2`, so `at >= 1`; the scene checks the rest of the
    /// placement against the grid.
    pub fn new(normal: Normal, at: usize, from: usize, to: usize, freqs: &[f64]) -> Result<Self> {
        if from > to {
            return Err(FdtdError::InvalidParameter {
                name: "flux line length",
                value: from as f64 - to as f64,
            });
        }
        if normal == Normal::X && at == 0 {
            return Err(FdtdError::InvalidParameter {
                name: "flux line position",
                value: 0.0,
            });
        }
        let n = to - from + 1;
        Ok(Self {
            normal,
            at,
            from,
            to,
            e: Dft::new(freqs, n)?,
            h: Dft::new(freqs, n)?,
            e_buf: vec![0.0; n],
            h_buf: vec![0.0; n],
        })
    }

    /// Adds the current step of `sim`: Ez at t = n S, the averaged H at
    /// t = (n - 1/2) S. With periodic y, a y-normal line on row 0 averages
    /// Hx rows ny - 1 and 0.
    pub fn record(&mut self, sim: &SimTm) {
        let (nx, ny, s) = (sim.nx(), sim.ny(), sim.courant());
        let (ez, hx, hy) = (sim.ez(), sim.hx(), sim.hy());
        for (p, q) in (self.from..=self.to).enumerate() {
            let (k, prev) = match self.normal {
                Normal::X => (q * nx + self.at, q * nx + self.at - 1),
                Normal::Y => {
                    let below = if self.at == 0 { ny - 1 } else { self.at - 1 };
                    (self.at * nx + q, below * nx + q)
                }
            };
            self.e_buf[p] = ez[k];
            self.h_buf[p] = match self.normal {
                Normal::X => 0.5 * (hy[prev] + hy[k]),
                Normal::Y => 0.5 * (hx[prev] + hx[k]),
            };
        }
        let n = sim.time_step() as f64;
        self.e.add(&self.e_buf, n * s, s);
        self.h.add(&self.h_buf, (n - 0.5) * s, s);
    }

    pub fn freqs(&self) -> &[f64] {
        self.e.freqs()
    }

    /// Flux per frequency, positive along +x (or +y).
    pub fn flux(&self) -> Vec<f64> {
        let sign = match self.normal {
            Normal::X => -0.5,
            Normal::Y => 0.5,
        };
        (0..self.freqs().len())
            .map(|f| {
                (0..self.e_buf.len())
                    .map(|p| {
                        let (e, h) = (self.e.get(p, f), self.h.get(p, f));
                        sign * (e.0 * h.0 + e.1 * h.1)
                    })
                    .sum()
            })
            .collect()
    }
}

/// Ez time history at one node (i, j), normalized. Entry n is Ez at time
/// step n: the kernel records once at construction and once after every step.
#[derive(Debug, Clone, PartialEq)]
pub struct PointProbe {
    pub i: usize,
    pub j: usize,
    history: Vec<f64>,
}

impl PointProbe {
    pub fn new(i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            history: Vec::new(),
        }
    }

    /// Appends Ez at this node from a `j * nx + i` node array.
    pub fn record(&mut self, ez: &[f64], nx: usize) {
        self.history.push(ez[self.j * nx + self.i]);
    }

    pub fn history(&self) -> &[f64] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane wave domain for the flux tests: vacuum, 32-cell x layers,
    /// TF/SF box [25, 175] x [y0, y1], the given direction, Ricker pulse.
    fn plane_wave_scene(
        dir: crate::tfsf::Direction,
        periodic: bool,
        ny: usize,
        y_box: (usize, usize),
    ) -> crate::scene::Scene {
        use crate::cpml::PmlSpec;
        use crate::material::Material;
        use crate::scene::{GridSpec, SCENE_VERSION, Scene};
        use crate::source::Waveform;
        use crate::tfsf::PlaneWave;
        Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx: 200,
                ny,
                dx_phys_m: 1e-3,
                courant: 0.5,
            },
            materials: vec![Material::VACUUM],
            geometry: vec![],
            sources: vec![],
            monitors: vec![],
            pml: Some(PmlSpec::recommended(16)),
            periodic_y: periodic,
            plane_wave: Some(PlaneWave {
                x0: 25,
                y0: y_box.0,
                x1: 175,
                y1: y_box.1,
                direction: dir,
                waveform: Waveform::Ricker {
                    t0: 40.0,
                    f_peak: 0.05,
                },
                amplitude: 1.0,
            }),
        }
    }

    const FREQS: [f64; 4] = [0.025, 0.04, 0.055, 0.07];

    /// |W(w)|^2 of the sampled launch waveform, W = sum w(n S) exp(-j w n S) S.
    fn launch_power(scene: &crate::scene::Scene, f: f64) -> f64 {
        let pw = scene.plane_wave.unwrap();
        let s = scene.grid.courant;
        let mut d = Dft::new(&[f], 1).unwrap();
        for n in 1..4000 {
            let t = n as f64 * s;
            d.add(&[pw.amplitude * pw.waveform.value(t)], t, s);
        }
        let (re, im) = d.get(0, 0);
        re * re + im * im
    }

    fn run_flux(scene: &crate::scene::Scene, lines: &mut [FluxLine], steps: u64) {
        let mut sim = crate::tm::SimTm::from_scene(scene).unwrap();
        for _ in 0..steps {
            sim.step();
            for l in lines.iter_mut() {
                l.record(&sim);
            }
        }
    }

    /// The flux of the plane wave through a line across the whole periodic
    /// height equals `ny cos(k/2) |W|^2 / 2` per frequency, with W the
    /// transform of the launch waveform: the 1D launch makes
    /// `E_p = W exp(-j k p)` at every node, and `Hy_avg = -cos(k/2) Ez`.
    /// A -x wave gives the same flux with the opposite sign.
    #[test]
    fn flux_matches_launch_spectrum() {
        use crate::dispersion::numerical_wavenumber_1d;
        use crate::tfsf::Direction;
        let ny = 4;
        for (dir, sign) in [(Direction::PlusX, 1.0), (Direction::MinusX, -1.0)] {
            let scene = plane_wave_scene(dir, true, ny, (0, ny - 1));
            let mut lines = [FluxLine::new(Normal::X, 100, 0, ny - 1, &FREQS).unwrap()];
            run_flux(&scene, &mut lines, 1500);
            for (f, &freq) in FREQS.iter().enumerate() {
                let k = numerical_wavenumber_1d(2.0 * PI * freq, 0.5).unwrap();
                let expect = sign * ny as f64 * (k / 2.0).cos() * launch_power(&scene, freq) / 2.0;
                let got = lines[0].flux()[f];
                eprintln!("{dir:?} f {freq}: flux {got:.10e}, expect {expect:.10e}");
                assert!(
                    ((got - expect) / expect).abs() < 1e-6,
                    "{dir:?} f {freq}: {got} vs {expect}"
                );
            }
        }
    }

    /// Normal y: a +y wave in a closed box crosses a row segment over the
    /// box columns with flux `columns cos(k/2) |W|^2 / 2`; a -y wave with the
    /// opposite sign. The x-flux line in a lossless empty periodic domain
    /// reads the same at two positions.
    #[test]
    fn flux_normal_y_and_conservation() {
        use crate::dispersion::numerical_wavenumber_1d;
        use crate::tfsf::Direction;
        for (dir, sign) in [(Direction::PlusY, 1.0), (Direction::MinusY, -1.0)] {
            let mut scene = plane_wave_scene(dir, false, 200, (25, 175));
            scene.plane_wave.as_mut().unwrap().x0 = 60;
            scene.plane_wave.as_mut().unwrap().x1 = 140;
            let mut lines = [FluxLine::new(Normal::Y, 100, 60, 140, &FREQS).unwrap()];
            run_flux(&scene, &mut lines, 1000);
            for (f, &freq) in FREQS.iter().enumerate() {
                let k = numerical_wavenumber_1d(2.0 * PI * freq, 0.5).unwrap();
                let expect = sign * 81.0 * (k / 2.0).cos() * launch_power(&scene, freq) / 2.0;
                let got = lines[0].flux()[f];
                assert!(
                    ((got - expect) / expect).abs() < 1e-6,
                    "{dir:?} f {freq}: {got} vs {expect}"
                );
            }
        }
        let scene = plane_wave_scene(Direction::PlusX, true, 3, (0, 2));
        let mut lines = [
            FluxLine::new(Normal::X, 50, 0, 2, &FREQS).unwrap(),
            FluxLine::new(Normal::X, 150, 0, 2, &FREQS).unwrap(),
        ];
        run_flux(&scene, &mut lines, 1500);
        for f in 0..FREQS.len() {
            let (a, b) = (lines[0].flux()[f], lines[1].flux()[f]);
            assert!(((a - b) / a).abs() < 1e-6, "f {}: {a} vs {b}", FREQS[f]);
        }
    }

    #[test]
    fn flux_line_input_checked() {
        assert!(
            FluxLine::new(Normal::X, 10, 5, 4, &FREQS).is_err(),
            "empty segment"
        );
        assert!(
            FluxLine::new(Normal::X, 0, 0, 4, &FREQS).is_err(),
            "needs a neighbor edge"
        );
        assert!(FluxLine::new(Normal::X, 10, 0, 4, &[]).is_err());
        assert!(FluxLine::new(Normal::Y, 10, 0, 4, &FREQS).is_ok());
    }

    /// A constant signal sampled N times from t = 0 has the closed-form
    /// transform dt (1 - z^N) / (1 - z), z = exp(-j w dt). Two signals in the
    /// same accumulator stay separate.
    #[test]
    fn dft_matches_geometric_sum() {
        let (dt, n) = (0.5, 400);
        let freqs = [0.013, 0.05, 0.31];
        let mut dft = Dft::new(&freqs, 2).unwrap();
        for k in 0..n {
            dft.add(&[1.0, -3.0], k as f64 * dt, dt);
        }
        for (f, &freq) in freqs.iter().enumerate() {
            let w = 2.0 * PI * freq;
            let z = |m: f64| ((w * dt * m).cos(), -(w * dt * m).sin());
            let (zn, z1) = (z(n as f64), z(1.0));
            let (nr, ni) = (1.0 - zn.0, -zn.1);
            let (dr, di) = (1.0 - z1.0, -z1.1);
            let d2 = dr * dr + di * di;
            let exact = (dt * (nr * dr + ni * di) / d2, dt * (ni * dr - nr * di) / d2);
            let got = dft.get(0, f);
            assert!((got.0 - exact.0).abs() < 1e-10 && (got.1 - exact.1).abs() < 1e-10);
            let other = dft.get(1, f);
            assert!(
                (other.0 + 3.0 * exact.0).abs() < 1e-9 && (other.1 + 3.0 * exact.1).abs() < 1e-9
            );
        }
        assert!(Dft::new(&[], 1).is_err());
        assert!(Dft::new(&[0.1, 0.0], 1).is_err());
        assert!(Dft::new(&[f64::NAN], 1).is_err());
    }

    /// The half-step identity. A TF/SF plane wave in a periodic vacuum domain
    /// is, at every frequency, `Ez_i = A exp(-j k i)` and
    /// `Hy_(i+1/2) = B exp(-j k (i + 1/2))` with the discrete k. The H update
    /// gives `B = -S A sin(k/2) / sin(w S / 2) = -A` on the dispersion
    /// relation, and averaging Hy onto the node multiplies by cos(k/2), so
    /// with H taken at its true time (n - 1/2) S,
    ///
    ///   Hy_avg(w) / Ez(w) = -cos(k / 2)   exactly, and real.
    ///
    /// Taking H at t = n S instead leaves a phase of w S / 2 (about 0.08 rad
    /// at the top frequency here), which this test sees by a factor of 1e4.
    #[test]
    fn half_step_identity_plane_wave() {
        use crate::cpml::PmlSpec;
        use crate::dispersion::numerical_wavenumber_1d;
        use crate::material::Material;
        use crate::scene::{GridSpec, SCENE_VERSION, Scene};
        use crate::source::Waveform;
        use crate::tfsf::{Direction, PlaneWave};
        use crate::tm::SimTm;

        let (nx, ny, s) = (160, 3, 0.5);
        let scene = Scene {
            version: SCENE_VERSION,
            grid: GridSpec {
                nx,
                ny,
                dx_phys_m: 1e-3,
                courant: s,
            },
            materials: vec![Material::VACUUM],
            geometry: vec![],
            sources: vec![],
            monitors: vec![],
            pml: Some(PmlSpec::recommended(16)),
            periodic_y: true,
            plane_wave: Some(PlaneWave {
                x0: 25,
                y0: 0,
                x1: 135,
                y1: ny - 1,
                direction: Direction::PlusX,
                waveform: Waveform::Ricker {
                    t0: 40.0,
                    f_peak: 0.05,
                },
                amplitude: 1.0,
            }),
        };
        let freqs = [0.02, 0.035, 0.05, 0.065, 0.08];
        let node = 60;
        let mut sim = SimTm::from_scene(&scene).unwrap();
        let (mut e, mut h, mut h_wrong) = (
            Dft::new(&freqs, 1).unwrap(),
            Dft::new(&freqs, 1).unwrap(),
            Dft::new(&freqs, 1).unwrap(),
        );
        // The pulse is past the node and in the exit layer well before 1500 steps.
        for _ in 0..1500 {
            sim.step();
            let n = sim.time_step() as f64;
            let hy = 0.5 * (sim.hy()[node - 1] + sim.hy()[node]);
            e.add(&[sim.ez()[node]], n * s, s);
            h.add(&[hy], (n - 0.5) * s, s);
            h_wrong.add(&[hy], n * s, s);
        }
        let ratio = |a: (f64, f64), b: (f64, f64)| {
            let d = b.0 * b.0 + b.1 * b.1;
            ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
        };
        let mut worst = (0.0_f64, 0.0_f64);
        for (f, &freq) in freqs.iter().enumerate() {
            let k = numerical_wavenumber_1d(2.0 * PI * freq, s).unwrap();
            let r = ratio(h.get(0, f), e.get(0, f));
            let expect = -(k / 2.0).cos();
            let (dre, dim) = ((r.0 - expect).abs(), r.1.abs());
            let rw = ratio(h_wrong.get(0, f), e.get(0, f));
            eprintln!(
                "f {freq}: ratio {:.12} {:+.2e}j, expect {expect:.12}; at t = n S the phase is {:+.3e}",
                r.0,
                r.1,
                rw.1.atan2(rw.0) - std::f64::consts::PI
            );
            worst = (worst.0.max(dre), worst.1.max(dim));
            assert!(rw.1.abs() > 1e-2, "the wrong time must show a phase");
        }
        eprintln!(
            "half-step identity: worst re {:.1e}, im {:.1e}",
            worst.0, worst.1
        );
        assert!(worst.0 < 1e-6 && worst.1 < 1e-6, "{worst:?}");
    }
}
