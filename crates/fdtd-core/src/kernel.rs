//! The interface every kernel implements, so one harness can drive the f64
//! CPU reference and the f32 GPU kernel from the same `Scene` and diff them.

use crate::error::Result;
use crate::grid::Component;
use crate::scene::Scene;

pub trait Kernel: Sized {
    /// Validates the scene and builds the kernel at time step 0, fields zero.
    fn from_scene(scene: &Scene) -> Result<Self>;

    /// Advances `n` full steps.
    fn step(&mut self, n: u64);

    /// Full steps taken so far. E is at time n S, H at (n - 1/2) S.
    fn time_step(&self) -> u64;

    /// A copy of one field, normalized, `nx * ny` padded, index `j * nx + i`.
    /// Ez on nodes, Hx on y-edges, Hy on x-edges (see `tm.rs`).
    fn read_field(&mut self, which: Component) -> Result<Vec<f64>>;

    /// Ez history of every point monitor, in scene order. Each has
    /// `time_step() + 1` entries.
    fn probe_history(&mut self) -> Result<Vec<Vec<f64>>>;
}
