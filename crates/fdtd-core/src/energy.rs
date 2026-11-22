//! Discrete electromagnetic energy of the TMz and TEz Yee schemes,
//! normalized (eps0 = mu0 = 1, dx = dy = 1).
//!
//! After n steps the solver holds E^n and H^(n - 1/2). The quantity that a
//! lossless closed domain conserves exactly (up to round-off) is
//!
//!   W^n = 1/2 sum(eps Ez^n Ez^n) + 1/2 sum(H^(n - 1/2) . H^(n + 1/2)).
//!
//! H^(n + 1/2) is computed here from the current state with the same
//! expression the solver uses, so no copy of the previous H is stored.
//! Squaring H at a single half step instead gives a quantity that oscillates.
//!
//! TEz (`energy_te`) is the same identity with the roles swapped: E^n on
//! the edges weighted by the edge permittivity, and Hz^(n - 1/2) . Hz^(n + 1/2)
//! on the nodes.
//!
//! Closed boxes only. Inside a CPML layer the H update carries 1/kappa and
//! psi, so the H^(n + 1/2) recomputed here is not the solver's, and W is not
//! the quantity the layer conserves or dissipates.

use crate::te::SimTe;
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
    // With periodic y, Hx row ny - 1 is live and its upper neighbor is row 0.
    let hx_rows = if sim.periodic_y() { ny } else { ny - 1 };
    for j in 0..hx_rows {
        let jn = if j + 1 == ny { 0 } else { j + 1 };
        for i in 0..nx {
            let (k, up) = (j * nx + i, jn * nx + i);
            h += hx[k] * (hx[k] - s * (ez[up] - ez[k]));
        }
    }
    for j in 0..ny {
        for i in 0..nx - 1 {
            let k = j * nx + i;
            h += hy[k] * (hy[k] + s * (ez[k + 1] - ez[k]));
        }
    }
    0.5 * (e + h) + sim.dispersive_energy()
}

/// W^n of a TEz kernel, with the same assumptions as `energy`.
pub fn energy_te(sim: &SimTe) -> f64 {
    let (nx, ny, s) = (sim.nx(), sim.ny(), sim.courant());
    let (hz, ex, ey) = (sim.hz(), sim.ex(), sim.ey());
    let (eps_x, eps_y) = sim.eps();

    let mut e = 0.0;
    for k in 0..nx * ny {
        e += eps_x[k] * ex[k] * ex[k] + eps_y[k] * ey[k] * ey[k];
    }
    // Hz^(n + 1/2) recomputed on the nodes the solver updates.
    let rows = if sim.periodic_y() { 0..ny } else { 1..ny - 1 };
    let mut h = 0.0;
    for j in rows {
        let jp = if j == 0 { ny - 1 } else { j - 1 };
        for i in 1..nx - 1 {
            let (k, down) = (j * nx + i, jp * nx + i);
            let next = hz[k] + s * (-ey[k] + ey[k - 1] + ex[k] - ex[down]);
            h += hz[k] * next;
        }
    }
    0.5 * (e + h) + sim.dispersive_energy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;
    use crate::scene::Mode;
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

    /// The same cavity with periodic y (x walls only, every row live): W is
    /// still conserved to 1e-10 over 1e4 steps. A wrong wrap in the solver or
    /// in this function, or a missing Hx row, breaks the discrete identity.
    /// The block sits on rows 0 and ny - 1, so energy really crosses the seam.
    #[test]
    fn conserved_lossless_periodic_y() {
        let mut scene = cavity_with_block(Material::Dielectric {
            eps_inf: 4.0,
            sigma: 0.0,
        });
        scene.periodic_y = true;
        scene.geometry.push(Shape::Rect {
            x0: 25.0,
            y0: -0.5,
            x1: 30.0,
            y1: 2.0,
            material: 1,
        });
        scene.geometry.push(Shape::Rect {
            x0: 25.0,
            y0: 27.0,
            x1: 30.0,
            y1: 29.5,
            material: 1,
        });
        let mut sim = started(&scene);
        assert!(sim.ez()[3].abs() > 0.0, "row 0 must start with a field");
        let w0 = energy(&sim);
        let mut drift = 0.0_f64;
        for _ in 0..10_000 {
            sim.step();
            drift = drift.max(((energy(&sim) - w0) / w0).abs());
        }
        eprintln!("periodic energy drift {drift:.2e}");
        assert!(drift < 1e-10, "relative drift {drift:e}");
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

    fn started_te(scene: &Scene) -> SimTe {
        let mut scene = scene.clone();
        scene.mode = Mode::Te;
        let mut sim = SimTe::from_scene(&scene).unwrap();
        sim.set_hz(&random_fields(3, 40 * 30)).unwrap();
        sim
    }

    fn te_drift(scene: &Scene, steps: usize) -> f64 {
        let mut sim = started_te(scene);
        let w0 = energy_te(&sim);
        let mut drift = 0.0_f64;
        for _ in 0..steps {
            sim.step();
            drift = drift.max(((energy_te(&sim) - w0) / w0).abs());
        }
        drift
    }

    /// TEz: a lossless cavity with an eps 4 block, a PEC block, and the same
    /// with periodic y conserve W to 1e-10 over 1e4 steps. Averaged edges and
    /// PEC edges are both inside the identity.
    #[test]
    fn conserved_lossless_te() {
        let dielectric = cavity_with_block(Material::Dielectric {
            eps_inf: 4.0,
            sigma: 0.0,
        });
        let pec = cavity_with_block(Material::Pec);
        let mut periodic = dielectric.clone();
        periodic.periodic_y = true;
        periodic.geometry.push(Shape::Rect {
            x0: 25.0,
            y0: -0.5,
            x1: 30.0,
            y1: 2.0,
            material: 1,
        });
        for (name, scene) in [("eps 4", dielectric), ("pec", pec), ("periodic", periodic)] {
            let drift = te_drift(&scene, 10_000);
            eprintln!("TEz {name} energy drift {drift:.2e}");
            assert!(drift < 1e-10, "{name}: relative drift {drift:e}");
        }
    }

    /// TEz with a lossy block and no source: W never increases.
    #[test]
    fn non_increasing_with_loss_te() {
        let mut sim = started_te(&cavity_with_block(Material::Dielectric {
            eps_inf: 2.0,
            sigma: 0.05,
        }));
        let w0 = energy_te(&sim);
        let mut last = w0;
        for n in 0..5000 {
            sim.step();
            let w = energy_te(&sim);
            assert!(w <= last + 1e-14 * w0, "step {n}: {w} > {last}");
            last = w;
        }
        assert!(last < 0.5 * w0, "loss must remove energy: {last} of {w0}");
    }

    /// The energy of `scene` in TMz and TEz over `steps`: worst relative
    /// drift from the start, largest single-step rise, and the final level.
    fn passivity(scene: &Scene, steps: usize) -> [(f64, f64, f64); 2] {
        let mut tm = started(scene);
        let mut te = started_te(scene);
        let mut out = [(0.0, 0.0, 0.0); 2];
        for (m, w) in out.iter_mut().enumerate() {
            let energy_of =
                |tm: &SimTm, te: &SimTe| if m == 0 { energy(tm) } else { energy_te(te) };
            let w0 = energy_of(&tm, &te);
            let (mut drift, mut rise, mut last) = (0.0_f64, 0.0_f64, w0);
            for _ in 0..steps {
                if m == 0 {
                    tm.step();
                } else {
                    te.step();
                }
                let now = energy_of(&tm, &te);
                drift = drift.max(((now - w0) / w0).abs());
                rise = rise.max((now - last) / w0);
                last = now;
            }
            *w = (drift, rise, last / w0);
        }
        out
    }

    /// Lossless Drude and Lorentz blocks: the total energy, J and P terms
    /// included, is conserved to 1e-10 in both modes over 1e4 steps, and
    /// the field energy alone is not (so the dispersive terms carry real
    /// energy and the test can tell).
    #[test]
    fn dispersive_lossless_conserves_energy() {
        for (name, m) in [
            (
                "drude",
                Material::Drude {
                    eps_inf: 1.5,
                    wp: 0.4,
                    gamma: 0.0,
                },
            ),
            (
                "lorentz",
                Material::Lorentz {
                    eps_inf: 1.5,
                    d_eps: 2.0,
                    w0: 0.3,
                    delta: 0.0,
                },
            ),
        ] {
            let scene = cavity_with_block(m);
            for (mode, (drift, _, _)) in ["tm", "te"].iter().zip(passivity(&scene, 10_000)) {
                eprintln!("{name} {mode}: drift {drift:.2e}");
                assert!(drift < 1e-10, "{name} {mode}: drift {drift:e}");
            }
            let mut sim = started(&scene);
            let w0 = energy(&sim);
            sim.step_n(500);
            assert!(
                (sim.dispersive_energy() / w0).abs() > 1e-4,
                "{name}: the poles hold no energy"
            );
        }
    }

    /// Lossy Drude never gains energy, step by step, in either mode (proven,
    /// `Aux::energy`). Lossy Lorentz is not proven monotone per step; its
    /// worst measured rise is 1.1e-15 of the start, round-off, so it is held
    /// to the same 1e-14 as a regression guard.
    #[test]
    fn dispersive_loss_never_gains_energy() {
        let drude = cavity_with_block(Material::Drude {
            eps_inf: 1.0,
            wp: 0.4,
            gamma: 0.02,
        });
        for (mode, (_, rise, end)) in ["tm", "te"].iter().zip(passivity(&drude, 6000)) {
            eprintln!("lossy drude {mode}: worst rise {rise:.2e}, end {end:.3}");
            assert!(rise <= 1e-14, "drude {mode}: rose by {rise:e}");
            assert!(end < 0.9, "drude {mode}: loss must remove energy, {end}");
        }
        let lorentz = cavity_with_block(Material::Lorentz {
            eps_inf: 1.0,
            d_eps: 2.0,
            w0: 0.3,
            delta: 0.02,
        });
        for (mode, (drift, rise, end)) in ["tm", "te"].iter().zip(passivity(&lorentz, 6000)) {
            eprintln!(
                "lossy lorentz {mode}: worst rise {rise:.2e}, end {end:.3}, drift {drift:.2e}"
            );
            assert!(end < 0.9, "lorentz {mode}: loss must remove energy, {end}");
            assert!(rise <= 1e-14, "lorentz {mode}: rose by {rise:e}");
            assert!(drift <= 1.0, "lorentz {mode}: above its start");
        }
    }
}
