//! Reference FDTD solver. Plain f64 loops, no cleverness. Its only job is to be
//! obviously correct. Everything inside works in normalized units: dx = 1,
//! c0 = eps0 = mu0 = 1, and dt = S (the Courant number).

pub mod error;
pub mod fdtd1d;
pub mod source;
pub mod units;

pub use error::{FdtdError, Result};
