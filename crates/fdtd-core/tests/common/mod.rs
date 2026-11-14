//! Shared by the acceptance tests: the signal helpers live in
//! `fdtd_core::cases::signal`, and `assert_cases` prints and checks rows.
#![allow(dead_code, unused_imports)]

pub use fdtd_core::cases::signal::*;
use fdtd_core::cases::{CaseResult, fmt_num};

/// Prints every row of a case and asserts each one passed.
pub fn assert_cases(rows: fdtd_core::Result<Vec<CaseResult>>) {
    let rows = rows.unwrap_or_else(|e| panic!("{e}"));
    assert!(!rows.is_empty());
    for r in &rows {
        eprintln!(
            "{} {}: {} {} (gate {}) {}; {}",
            r.id,
            r.name,
            fmt_num(r.measured),
            r.unit,
            r.gate,
            if r.pass { "pass" } else { "FAIL" },
            r.note
        );
        assert!(r.pass, "{} failed: {} vs {}", r.id, r.measured, r.gate);
    }
}
