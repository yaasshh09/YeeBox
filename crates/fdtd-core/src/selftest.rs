//! The browser self-test's reference data: the native f64 CPU kernel's
//! fields on three committed scenes, so the page can check the browser's
//! GPU kernel against the native reference with the Tier 4 measure
//! (`kernel::Comparator`). `fdtd selftest-golden` writes the files under
//! `validation/golden/selftest/`, and a test fails if they go stale.
//!
//! File layout, little-endian: first, every `PEAK_EVERY` steps up to the
//! last checkpoint, the reference's max |Ez|, |Hx|, |Hy| as three f64; then
//! at each checkpoint Ez, Hx and Hy as f32, `nx * ny` values each (padded
//! like the kernels). No header; the scene gives the size. The maxima let
//! the browser hold the Tier 4 measure (peak so far, checked every 250
//! steps) while comparing full fields only at the checkpoints: once a CPML
//! has drained a pulse, the field at a checkpoint is a small residue, and
//! dividing round-off by it would misreport. Rounding f64 to f32 costs
//! about 6e-8 of the peak, far under the 1e-4 gate.

use crate::error::{FdtdError, Result};
use crate::kernel::{CpuKernel, Kernel, Snapshot};
use crate::scene::Scene;

/// The self-test scenes, in `validation/scenes`: a closed box with
/// dielectric and PEC, painted walls in a CPML, a TF/SF plane wave on a
/// rod in a CPML, and a TEz plane wave on a Drude cylinder (the plasmon
/// path: TEz, dispersion and TF/SF by duality).
pub const SCENES: [&str; 4] = [
    "cavity_tm.json",
    "painted_walls.json",
    "tfsf_rod_minus_x.json",
    "drude_te.json",
];

/// Steps at which the fields are compared.
pub const CHECKPOINTS: [u64; 2] = [1000, 2000];

/// Spacing of the recorded reference maxima, as in Tier 4.
pub const PEAK_EVERY: u64 = 250;

fn peak_steps() -> impl Iterator<Item = u64> {
    (1..=CHECKPOINTS[CHECKPOINTS.len() - 1] / PEAK_EVERY).map(|k| k * PEAK_EVERY)
}

/// The golden bytes of `scene`: the CPU kernel's maxima every `PEAK_EVERY`
/// steps and its fields at each checkpoint.
pub fn golden(scene: &Scene) -> Result<Vec<u8>> {
    let mut sim = CpuKernel::from_scene(scene)?;
    let mut peaks = Vec::new();
    let mut fields = Vec::new();
    for n in peak_steps() {
        let now = sim.time_step();
        Kernel::step(&mut sim, n - now);
        for c in scene.mode.fields() {
            let v = sim.read_field(c)?;
            let m = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            peaks.extend_from_slice(&m.to_le_bytes());
            if CHECKPOINTS.contains(&n) {
                for x in v {
                    fields.extend_from_slice(&(x as f32).to_le_bytes());
                }
            }
        }
    }
    peaks.extend(fields);
    Ok(peaks)
}

/// A golden file read back: the reference maxima by step, and one
/// fields-only `Snapshot` per checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Golden {
    pub peaks: Vec<(u64, [f64; 3])>,
    pub snapshots: Vec<Snapshot>,
}

/// Golden bytes back into maxima and snapshots, for a grid of `nodes`
/// nodes. A wrong length is an error.
pub fn read_golden(bytes: &[u8], nodes: usize) -> Result<Golden> {
    let steps: Vec<u64> = peak_steps().collect();
    let head = steps.len() * 3 * 8;
    let per = nodes * 4;
    if bytes.len() != head + per * 3 * CHECKPOINTS.len() {
        return Err(FdtdError::InvalidParameter {
            name: "golden length",
            value: bytes.len() as f64,
        });
    }
    let f64_at = |at: usize| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[at..at + 8]);
        f64::from_le_bytes(b)
    };
    let peaks = steps
        .iter()
        .enumerate()
        .map(|(k, &n)| (n, [f64_at(k * 24), f64_at(k * 24 + 8), f64_at(k * 24 + 16)]))
        .collect();
    let field = |at: usize| -> Vec<f64> {
        bytes[at..at + per]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect()
    };
    let snapshots = (0..CHECKPOINTS.len())
        .map(|c| {
            let base = head + c * 3 * per;
            Snapshot {
                fields: [field(base), field(base + per), field(base + 2 * per)],
                ..Default::default()
            }
        })
        .collect();
    Ok(Golden { peaks, snapshots })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_round_trips() {
        let mut s = crate::tm::tests::vacuum_box(12, 9, 0.5);
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
        let g = read_golden(&bytes, 12 * 9).unwrap();
        assert_eq!(g.snapshots.len(), CHECKPOINTS.len());
        assert_eq!(g.peaks.len(), 8);
        let mut sim = crate::tm::SimTm::from_scene(&s).unwrap();
        sim.step_n(250);
        let peak = sim.ez().iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert_eq!(g.peaks[0], (250, [peak, g.peaks[0].1[1], g.peaks[0].1[2]]));
        sim.step_n(CHECKPOINTS[0] - 250);
        let ez: Vec<f64> = sim.ez().iter().map(|&v| v as f32 as f64).collect();
        assert_eq!(g.snapshots[0].fields[0], ez);
        assert!(read_golden(&bytes[4..], 12 * 9).is_err());
    }
}
