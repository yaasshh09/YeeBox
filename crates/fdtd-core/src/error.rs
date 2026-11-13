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
}

/// Shorthand used across the crate.
pub type Result<T> = std::result::Result<T, FdtdError>;
