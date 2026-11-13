//! The one error type for the reference solver.

use thiserror::Error;

/// Every fallible public function in `fdtd-core` returns this.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum FdtdError {
    /// The Courant number is above the stability limit for this dimension.
    #[error("Courant number {s} is above the stability limit {limit}")]
    CourantAboveLimit { s: f64, limit: f64 },
    /// A parameter is out of range, zero where it must be positive, or not finite.
    #[error("invalid parameter {name} = {value}")]
    InvalidParameter { name: &'static str, value: f64 },
    /// No real numerical wavenumber exists: the frequency is above what the grid
    /// can carry at this Courant number (the wave is evanescent on the grid).
    #[error("omega {omega} is above the grid cutoff at Courant number {s}")]
    AboveGridCutoff { omega: f64, s: f64 },
    /// The scene file was written by a format version this build cannot read.
    #[error("scene format version {version} is not supported")]
    UnsupportedVersion { version: u32 },
    /// A shape refers to a material index past the end of the table, or the
    /// table is empty (index 0 is the background and must exist).
    #[error("material index {id} is not in the material table")]
    UnknownMaterial { id: usize },
    /// A source or monitor sits outside the nodes it is allowed on.
    #[error("{what} at node ({i}, {j}) is outside the allowed region")]
    OutOfDomain {
        what: &'static str,
        i: usize,
        j: usize,
    },
    /// A source sits on a PEC node, where it would break the PEC condition.
    #[error("source at node ({i}, {j}) is on a PEC node")]
    SourceOnPec { i: usize, j: usize },
    /// Grid smaller than 3 by 3 or above the cell budget.
    #[error("grid {nx} x {ny} is outside the supported size")]
    GridSize { nx: usize, ny: usize },
    /// The scene JSON could not be parsed or written.
    #[error("scene json: {0}")]
    Json(String),
}

/// Shorthand used across the crate.
pub type Result<T> = std::result::Result<T, FdtdError>;
