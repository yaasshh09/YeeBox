//! Field monitors: point probes and running DFTs.
//!
//! Time convention for every DFT here: a sample enters at its true time.
//! After step n the solver holds E at t = n S and H at t = (n - 1/2) S, so
//! H samples are added with `t = (n - 1/2) S`. That is the half-step
//! correction of `docs/02-numerics.md`; the doc's `exp(-j w dt / 2)` factor is
//! the same thing written for H^(n + 1/2) stored under index n.

use crate::error::{FdtdError, Result};
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
