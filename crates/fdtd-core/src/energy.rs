//! Discrete electromagnetic energy of the TMz Yee scheme, normalized
//! (eps0 = mu0 = 1, dx = dy = 1).
//!
//! After n steps the solver holds E^n and H^(n - 1/2). The quantity that a
//! lossless closed domain conserves exactly (up to round-off) is
//!
//!   W^n = 1/2 sum(eps Ez^n Ez^n) + 1/2 sum(H^(n - 1/2) . H^(n + 1/2)).
//!
//! H^(n + 1/2) is computed here from the current state with the same
//! expression the solver uses, so no copy of the previous H is stored.
//! Squaring H at a single half step instead gives a quantity that oscillates.

use crate::tm::SimTm;

/// W^n as defined above, at the solver's current time step. Assumes no
/// source acts between now and the next step, which holds for any closed
/// domain once the sources have died out.
pub fn energy(sim: &SimTm) -> f64 {
    let (nx, ny, s) = (sim.nx(), sim.ny(), sim.courant());
    let (ez, hx, hy, eps) = (sim.ez(), sim.hx(), sim.hy(), sim.eps());

    let mut e = 0.0;
    for k in 0..nx * ny {
        e += eps[k] * ez[k] * ez[k];
    }
    let mut h = 0.0;
    for j in 0..ny - 1 {
        for i in 0..nx {
            let k = j * nx + i;
            h += hx[k] * (hx[k] - s * (ez[k + nx] - ez[k]));
        }
    }
    for j in 0..ny {
        for i in 0..nx - 1 {
            let k = j * nx + i;
            h += hy[k] * (hy[k] + s * (ez[k + 1] - ez[k]));
        }
    }
    0.5 * (e + h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::scene::{Scene, Shape};
    use crate::tm::tests::{random_fields, vacuum_box};

    fn cavity_with_block(block: Material) -> Scene {
        let mut scene = vacuum_box(40, 30, 0.5);
        scene.materials.push(block);
        scene.geometry.push(Shape::Rect {
            x0: 10.0,
            y0: 8.0,
            x1: 22.0,
            y1: 17.0,
            material: 1,
        });
        scene
    }

    fn started(scene: &Scene) -> SimTm {
        let mut sim = SimTm::from_scene(scene).unwrap();
        sim.set_ez(&random_fields(3, 40 * 30)).unwrap();
        sim
    }

    /// Lossless closed cavity with a dielectric block: W is constant to 1e-10
    /// relative over 1e4 steps. The single-half-step H^2 energy, measured on
    /// the same run, moves by far more, which shows the test can tell them apart.
    #[test]
    fn conserved_lossless() {
        let mut sim = started(&cavity_with_block(Material::Dielectric {
            eps_inf: 4.0,
            sigma: 0.0,
        }));
        let w0 = energy(&sim);
        let naive = |sim: &SimTm| {
            let e: f64 = sim.eps().iter().zip(sim.ez()).map(|(p, e)| p * e * e).sum();
            let h: f64 = sim.hx().iter().chain(sim.hy()).map(|h| h * h).sum();
            0.5 * (e + h)
        };
        let (mut drift, mut naive_lo, mut naive_hi) = (0.0_f64, f64::MAX, 0.0_f64);
        for _ in 0..10_000 {
            sim.step();
            drift = drift.max(((energy(&sim) - w0) / w0).abs());
            let wn = naive(&sim);
            (naive_lo, naive_hi) = (naive_lo.min(wn), naive_hi.max(wn));
        }
        eprintln!(
            "energy drift {drift:.2e}, naive H^2 swing {:.2e}",
            (naive_hi - naive_lo) / w0
        );
        assert!(drift < 1e-10, "relative drift {drift:e}");
        assert!((naive_hi - naive_lo) / w0 > 1e-3);
    }

    /// With a lossy block and no source, W never increases.
    #[test]
    fn non_increasing_with_loss() {
        let mut sim = started(&cavity_with_block(Material::Dielectric {
            eps_inf: 2.0,
            sigma: 0.05,
        }));
        let w0 = energy(&sim);
        let mut last = w0;
        for n in 0..5000 {
            sim.step();
            let w = energy(&sim);
            assert!(w <= last + 1e-14 * w0, "step {n}: {w} > {last}");
            last = w;
        }
        assert!(
            last < 0.5 * w0,
            "loss must actually remove energy: {last} of {w0}"
        );
    }
}
