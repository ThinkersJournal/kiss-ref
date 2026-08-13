//! # kiss-ref-conformance
//!
//! The **coverage ledger** and the **conformance corpus** for `kiss-ref`.
//!
//! - The ledger ([`ledger`]) enumerates every op in the vocabulary and reports
//!   the `Done` / `Pending` split of the reference implementation, so the "always
//!   works" total-cover target is machine-visible and regressions are caught (the
//!   coverage-gate test asserts the mandatory core stays covered).
//! - The corpus (`tests/ops_conformance.rs`) mirrors KISS-Conform's `test_ops_*`
//!   test names (KISS-Ops §9.1 clause→test traceability) and checks the pinned
//!   numeric semantics of §6.

use kiss_ops_vocab::Op;
use kiss_ref_core::{implemented, ProvisionalPin, PROVISIONAL_PINS};

/// The coverage split of the reference implementation over the KISS-Ops vocab.
pub struct Ledger {
    /// Ops a reference path evaluates (float scalar on `f32`/`f64`, integer
    /// scalar on the integer dtypes, or the tensor path).
    pub done: Vec<Op>,
    /// Ops enumerated by the vocab but not yet evaluable in this seed.
    pub pending: Vec<Op>,
    /// The reference's provisional local pins (values fixed while the spec leaves
    /// them open — [`kiss_ref_core::PROVISIONAL_PINS`]). Surfaced here so the
    /// count sits **beside** the Done/Pending counts, not in prose; zero is the
    /// resolved state.
    pub provisional: &'static [ProvisionalPin],
}

impl Ledger {
    /// A one-line human summary for the coverage-gate test output.
    pub fn summary(&self) -> String {
        let total = self.done.len() + self.pending.len();
        let pins: String = self
            .provisional
            .iter()
            .map(|p| format!(" [{}={} pending {}]", p.site, p.value, p.issue))
            .collect();
        format!(
            "kiss-ref coverage — {} of {} ops evaluable, {} PENDING, {} PROVISIONAL PIN(S){}. \
             Paths: f16/bf16/f32/f64 float scalar (nextafter on f32/f64 only, §6.9-0003) \
             + integer scalar (i8..u64, i4/u4/b1) \
             + the §6.11 structural atoms, §6.13 tensor non-primitives & window family on \
             the float lane, plus the integer tensor lane (reduce/scan/gather/scatter/sort/ \
             argmax/any/all). Dtype breadth: + FP8 (f8e4m3fn/f8e5m2, promote-to-f32) + the bool \
             truth-valued lane + the §6.18 complex family (c64/c128, Annex-G-governed). sk4 \
             recognizes the reserved FP8 variants (f8e4m3fnuz/f8e5m2fnuz) and the MX scales \
             (f8e8m0/f8e6m2) but declines them for compute (NotApplicable, not Pending). Cell \
             coverage is three-state (Done/Pending/NotApplicable); only spec-legal cells \
             form the denominator. Provisional local pins are a separate site-level \
             axis (not a cell state); zero is the resolved state.",
            self.done.len(),
            total,
            self.pending.len(),
            self.provisional.len(),
            pins
        )
    }

    /// The pending ops as their KISS tokens (for a readable PENDING list).
    pub fn pending_tokens(&self) -> Vec<&'static str> {
        self.pending.iter().map(|o| o.token()).collect()
    }
}

/// Build the coverage ledger from what the core reference actually evaluates.
pub fn ledger() -> Ledger {
    let mut done = Vec::new();
    let mut pending = Vec::new();
    for &op in Op::ALL {
        if implemented(op) {
            done.push(op);
        } else {
            pending.push(op);
        }
    }
    Ledger {
        done,
        pending,
        provisional: PROVISIONAL_PINS,
    }
}
