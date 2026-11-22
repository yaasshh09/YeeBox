//! The Tier 2 analytic acceptance cases (`docs/04-validation.md`), A1 to
//! A15 and the Phase 6 cases, as library functions. The acceptance tests, `fdtd validate` and
//! `fdtd report` all run these, so the table they print cannot drift from
//! what the tests check.
//!
//! Each case measures one number against its gate. Where a case also has a
//! sanity guard (a record that must not cut off a pulse, a measurement that
//! must agree with the exact discrete answer), a failed guard is an error,
//! and `all` reports it as a failed row. Normalized units throughout.

pub mod boundary;
pub mod dispersive;
pub mod duality;
pub mod plane_wave;
pub mod scattering;
pub mod signal;
pub mod tier1d;
pub mod tier2d;

use crate::error::{FdtdError, Result};

/// What a measured value has to satisfy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    /// measured <= the bound.
    AtMost(f64),
    /// lo <= measured <= hi.
    Range(f64, f64),
    /// No gate yet: the number is recorded until its reference is confirmed.
    Record,
}

impl Gate {
    pub fn check(&self, v: f64) -> bool {
        match *self {
            Gate::AtMost(b) => v <= b,
            Gate::Range(lo, hi) => (lo..=hi).contains(&v),
            Gate::Record => true,
        }
    }
}

impl std::fmt::Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Gate::AtMost(b) => write!(f, "<= {}", fmt_num(b)),
            Gate::Range(lo, hi) => write!(f, "{} to {}", fmt_num(lo), fmt_num(hi)),
            Gate::Record => write!(f, "recorded"),
        }
    }
}

/// Compact number for tables: plain for ordinary magnitudes, else scientific.
pub fn fmt_num(v: f64) -> String {
    let a = v.abs();
    if v == 0.0 || (1e-2..1e4).contains(&a) {
        let s = format!("{v:.4}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    } else {
        format!("{v:.2e}")
    }
}

/// One row of the validation table.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseResult {
    pub id: &'static str,
    pub name: &'static str,
    pub measured: f64,
    pub gate: Gate,
    /// What `measured` is: "rel", "abs", "dB", "order", "x start".
    pub unit: &'static str,
    pub pass: bool,
    /// Conditions and secondary numbers, one line.
    pub note: String,
}

impl CaseResult {
    pub fn new(
        id: &'static str,
        name: &'static str,
        measured: f64,
        gate: Gate,
        unit: &'static str,
        note: String,
    ) -> Self {
        CaseResult {
            id,
            name,
            measured,
            pass: measured.is_finite() && gate.check(measured),
            gate,
            unit,
            note,
        }
    }
}

/// A guard: `Ok(())` if `cond`, else a `Case` error with `msg`.
pub(crate) fn ensure(cond: bool, msg: impl FnOnce() -> String) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(FdtdError::Case(msg()))
    }
}

type CaseFn = fn() -> Result<Vec<CaseResult>>;

/// Every case, in table order, with the ids it reports. A9 and A10 share one
/// expensive reference run, and A13 reports the R + T = 1 check beside it.
pub const CASES: &[(&str, CaseFn)] = &[
    ("A1", tier1d::a1),
    ("A2", tier1d::a2),
    ("A3", tier1d::a3),
    ("A4", tier2d::a4),
    ("A5", tier2d::a5),
    ("A6", tier2d::a6),
    ("A7", tier2d::a7),
    ("A8", boundary::a8),
    ("A9, A10", boundary::a9_a10),
    ("A11", plane_wave::a11),
    ("A12", plane_wave::a12),
    ("A13, R+T", plane_wave::a13),
    ("A14", plane_wave::a14),
    ("A15", plane_wave::a15),
    ("A16", dispersive::a16),
    ("A17", dispersive::a17),
    ("A18", duality::a18),
    ("A21", dispersive::a21),
    ("NTFF", scattering::isotropy),
    ("A19", scattering::a19),
    ("A20", scattering::a20),
    ("A22", scattering::a22),
    ("A23", scattering::a23),
    ("A24", scattering::a24),
];

/// Runs every case. A case that errors becomes a failed row carrying the error.
pub fn all() -> Vec<CaseResult> {
    run(CASES)
}

/// Runs the given cases, errors turned into failed rows.
pub fn run(cases: &[(&'static str, CaseFn)]) -> Vec<CaseResult> {
    cases
        .iter()
        .flat_map(|&(id, f)| match f() {
            Ok(rows) => rows,
            Err(e) => vec![CaseResult {
                id,
                name: "did not run",
                measured: f64::NAN,
                gate: Gate::AtMost(0.0),
                unit: "",
                pass: false,
                note: e.to_string(),
            }],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_and_rows() {
        assert!(Gate::AtMost(1e-4).check(1e-4) && !Gate::AtMost(1e-4).check(2e-4));
        assert!(Gate::Range(1.8, 2.2).check(2.0) && !Gate::Range(1.8, 2.2).check(1.7));
        let r = CaseResult::new("X", "x", f64::NAN, Gate::AtMost(1.0), "rel", String::new());
        assert!(!r.pass, "NaN never passes");
        assert_eq!(Gate::AtMost(-60.0).to_string(), "<= -60");
        assert_eq!(Gate::AtMost(1e-12).to_string(), "<= 1.00e-12");
        assert_eq!(Gate::Range(1.8, 2.2).to_string(), "1.8 to 2.2");
        assert!(Gate::Record.check(123.0) && Gate::Record.to_string() == "recorded");
        assert_eq!(fmt_num(0.0049), "4.90e-3");
        let failed = run(&[("Z", || Err(FdtdError::Case("guard".into())))]);
        assert_eq!(failed.len(), 1);
        assert!(!failed[0].pass && failed[0].note.contains("guard"));
    }
}
