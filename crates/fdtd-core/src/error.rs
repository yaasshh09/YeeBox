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
    /// The CPML layer leaves fewer than one interior node along x or y.
    #[error("pml of {thickness} cells does not fit a {nx} x {ny} grid")]
    PmlTooThick {
        thickness: usize,
        nx: usize,
        ny: usize,
    },
    /// A source sits strictly inside the CPML layer.
    #[error("source at node ({i}, {j}) is inside the pml")]
    SourceInPml { i: usize, j: usize },
    /// A node in the CPML layer has a different material from the interior
    /// node it faces along the layer normal.
    #[error("material at pml node ({i}, {j}) is not uniform along the pml normal")]
    PmlNotUniform { i: usize, j: usize },
    /// The TF/SF plane wave box breaks a placement rule.
    #[error("plane wave: {0}")]
    PlaneWave(crate::tfsf::PlaneWaveFault),
    /// A spectrum run hit its step budget before the fields died down, so
    /// its spectrum would ring.
    #[error("fields still at {level:e} of peak after {steps} steps")]
    NotDecayed { steps: u64, level: f64 },
    /// A validation case could not produce a trustworthy measurement: its
    /// golden data is for another setup, or a sanity guard failed (a record
    /// cut off a pulse, a peak search came up empty).
    #[error("validation case: {0}")]
    Case(String),
    /// The scene JSON could not be parsed or written.
    #[error("scene json: {0}")]
    Json(String),
    /// The GPU kernel could not get a device, build a pipeline, or read a
    /// buffer back, or the device reported an error while running.
    #[error("gpu: {0}")]
    Gpu(String),

    /// A live edit changed something a kernel is built from (grid, y wrap,
    /// material table, CPML, plane wave); build a new kernel instead.
    #[error("this edit needs a new kernel")]
    Rebuild,
}

/// Shorthand used across the crate.
pub type Result<T> = std::result::Result<T, FdtdError>;
