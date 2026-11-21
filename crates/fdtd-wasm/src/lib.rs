//! The browser API: a thin `wasm-bindgen` layer over `fdtd-gpu` (the GPU
//! kernel and its renderer) and `fdtd-core` (the scene, edits, units and
//! the CPU fallback kernel). Glue only: every number is computed in those
//! crates, and TypeScript receives results, never formulas.
//!
//! Errors cross the boundary as JS `Error` objects with a stable `code`
//! property (`error_code`), never as panics. Debug builds install a panic
//! hook that prints the panic to the console.

use fdtd_core::FdtdError;
use wasm_bindgen::prelude::*;

pub mod cpu;
#[cfg(target_arch = "wasm32")]
mod selftest;
pub mod session;
#[cfg(target_arch = "wasm32")]
mod sim;

/// The stable code of an error, for the UI to branch on. Exhaustive, so a
/// new error variant does not compile until it has a code here.
pub fn error_code(e: &FdtdError) -> &'static str {
    match e {
        FdtdError::CourantAboveLimit { .. } => "courant_above_limit",
        FdtdError::InvalidParameter { .. } => "invalid_parameter",
        FdtdError::AboveGridCutoff { .. } => "above_grid_cutoff",
        FdtdError::UnsupportedVersion { .. } => "unsupported_version",
        FdtdError::UnknownMaterial { .. } => "unknown_material",
        FdtdError::OutOfDomain { .. } => "out_of_domain",
        FdtdError::SourceOnPec { .. } => "source_on_pec",
        FdtdError::GridSize { .. } => "grid_size",
        FdtdError::PmlTooThick { .. } => "pml_too_thick",
        FdtdError::SourceInPml { .. } => "source_in_pml",
        FdtdError::PlaneWave(_) => "plane_wave",
        FdtdError::NotDecayed { .. } => "not_decayed",
        FdtdError::Case(_) => "case",
        FdtdError::Json(_) => "json",
        FdtdError::Gpu(_) => "gpu",
        FdtdError::Rebuild => "rebuild",
    }
}

/// What the UI receives for an error.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorInfo {
    pub code: &'static str,
    pub message: String,
}

impl From<&FdtdError> for ErrorInfo {
    fn from(e: &FdtdError) -> Self {
        ErrorInfo {
            code: error_code(e),
            message: e.to_string(),
        }
    }
}

impl ErrorInfo {
    /// No WebGPU adapter, or none that can draw to the canvas.
    pub fn no_webgpu(message: impl Into<String>) -> ErrorInfo {
        ErrorInfo {
            code: "no_webgpu",
            message: message.into(),
        }
    }
}

impl From<ErrorInfo> for JsValue {
    /// A JS `Error` with the message, and `code` set on it.
    fn from(info: ErrorInfo) -> JsValue {
        let err = js_sys::Error::new(&info.message);
        let _ = js_sys::Reflect::set(&err, &"code".into(), &info.code.into());
        err.into()
    }
}

/// `Result` for exported functions: errors arrive in JS as coded `Error`s.
pub type JsResult<T> = std::result::Result<T, JsValue>;

/// Maps an `FdtdError` to its coded JS error.
pub fn js_err(e: FdtdError) -> JsValue {
    ErrorInfo::from(&e).into()
}

/// Runs once when the module loads. Debug builds print panics to the
/// console; release builds keep the default (abort with a message).
#[wasm_bindgen(start)]
pub fn start() {
    #[cfg(debug_assertions)]
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&format!("fdtd panic: {info}").into());
    }));
}

/// The crate version, for the Verify panel.
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fdtd_core::tfsf::PlaneWaveFault;
    use std::collections::BTreeSet;

    #[test]
    fn every_error_has_its_own_code() {
        let all = [
            FdtdError::CourantAboveLimit { s: 1.0, limit: 0.7 },
            FdtdError::InvalidParameter {
                name: "x",
                value: 1.0,
            },
            FdtdError::AboveGridCutoff { omega: 1.0, s: 0.5 },
            FdtdError::UnsupportedVersion { version: 9 },
            FdtdError::UnknownMaterial { id: 3 },
            FdtdError::OutOfDomain {
                what: "source",
                i: 0,
                j: 0,
            },
            FdtdError::SourceOnPec { i: 1, j: 1 },
            FdtdError::GridSize { nx: 1, ny: 1 },
            FdtdError::PmlTooThick {
                thickness: 9,
                nx: 3,
                ny: 3,
            },
            FdtdError::SourceInPml { i: 1, j: 1 },
            FdtdError::PlaneWave(PlaneWaveFault::EmptyBox),
            FdtdError::NotDecayed {
                steps: 1,
                level: 1.0,
            },
            FdtdError::Case("c".into()),
            FdtdError::Json("j".into()),
            FdtdError::Gpu("g".into()),
            FdtdError::Rebuild,
        ];
        let codes: BTreeSet<&str> = all.iter().map(error_code).collect();
        assert_eq!(codes.len(), all.len());
        assert!(!codes.contains("no_webgpu"));
        let info = ErrorInfo::from(&FdtdError::SourceOnPec { i: 4, j: 5 });
        assert_eq!(info.code, "source_on_pec");
        assert!(info.message.contains("(4, 5)"));
    }
}
