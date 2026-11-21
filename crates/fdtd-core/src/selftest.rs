//! The browser self-test's reference data: the native f64 CPU kernel's
//! fields on three committed scenes, so the page can check the browser's
//! GPU kernel against the native reference with the Tier 4 measure
//! (`kernel::Comparator`). `fdtd selftest-golden` writes the files under
//! `validation/golden/selftest/`, and a test fails if they go stale.
//!
//! File layout: little-endian f32, checkpoint by checkpoint, each Ez then
//! Hx then Hy, `nx * ny` values each (padded like the kernels). No header;
//! the scene gives the size. Rounding f64 to f32 costs about 6e-8 of the
//! peak, far under the 1e-4 gate.

use crate::error::{FdtdError, Result};
use crate::grid::Component;
use crate::kernel::{Kernel, Snapshot};
use crate::scene::Scene;
use crate::tm::SimTm;

/// The self-test scenes, in `validation/scenes`: a closed box with
/// dielectric and PEC, painted walls in a CPML, and a TF/SF plane wave on a
/// rod in a CPML.
pub const SCENES: [&str; 3] = [
    "cavity_tm.json",
    "painted_walls.json",
    "tfsf_rod_minus_x.json",
];

/// Steps at which the fields are compared.
pub const CHECKPOINTS: [u64; 2] = [1000, 2000];

/// The golden bytes of `scene`: the CPU kernel's fields at each checkpoint.
pub fn golden(scene: &Scene) -> Result<Vec<u8>> {
    let mut sim = SimTm::from_scene(scene)?;
    let mut out = Vec::new();
    for &n in &CHECKPOINTS {
        sim.step_n(n - sim.time_step());
        for c in [Component::Ez, Component::Hx, Component::Hy] {
            for v in sim.read_field(c)? {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Golden bytes back into one fields-only `Snapshot` per checkpoint, for a
/// grid of `nodes` nodes. A wrong length is an error.
pub fn snapshots(bytes: &[u8], nodes: usize) -> Result<Vec<Snapshot>> {
    let per = nodes * 4;
    if bytes.len() != per * 3 * CHECKPOINTS.len() {
        return Err(FdtdError::InvalidParameter {
            name: "golden length",
            value: bytes.len() as f64,
        });
    }
    let field = |at: usize| -> Vec<f64> {
        bytes[at..at + per]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect()
    };
    Ok((0..CHECKPOINTS.len())
        .map(|c| {
            let base = c * 3 * per;
            Snapshot {
                fields: [field(base), field(base + per), field(base + 2 * per)],
                ..Default::default()
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_round_trips_to_snapshots() {
        let s = crate::tm::tests::vacuum_box(12, 9, 0.5);
        let mut s = s;
        s.sources.push(crate::scene::Source {
            i: 5,
            j: 4,
            waveform: crate::source::Waveform::Ricker {
                t0: 30.0,
                f_peak: 0.05,
            },
            amplitude: 1.0,
            kind: Default::default(),
        });
        let bytes = golden(&s).unwrap();
        let snaps = snapshots(&bytes, 12 * 9).unwrap();
        assert_eq!(snaps.len(), CHECKPOINTS.len());
        let mut sim = SimTm::from_scene(&s).unwrap();
        sim.step_n(CHECKPOINTS[0]);
        let ez: Vec<f64> = sim.ez().iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(snaps[0].fields[0], ez);
        assert!(snaps[0].probes.is_empty() && snaps[0].spectra.is_empty());
        assert!(snapshots(&bytes[4..], 12 * 9).is_err());
    }
}
