//! Reference FDTD solver. Plain f64 loops, no cleverness. Its only job is to be
//! obviously correct. Everything inside works in normalized units: dx = 1,
//! c0 = eps0 = mu0 = 1, and dt = S (the Courant number).

pub mod cases;
pub mod cpml;
pub mod dispersion;
pub mod energy;
pub mod error;
pub mod fdtd1d;
pub mod grid;
pub mod kernel;
pub mod material;
pub mod monitor;
pub mod reflection;
pub mod scene;
pub mod selftest;
pub mod source;
pub mod spectra;
pub mod te;
pub mod tfsf;
pub mod tm;
pub mod units;

pub use error::{FdtdError, Result};
