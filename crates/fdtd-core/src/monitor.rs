//! Field monitors. Phase 1 has point probes only.

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
