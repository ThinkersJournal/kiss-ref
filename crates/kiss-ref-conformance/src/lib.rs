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
use kiss_ref_core::float_supported;

/// The coverage split of the reference implementation over the KISS-Ops vocab.
pub struct Ledger {
    /// Ops the `f32`/`f64` scalar path evaluates (Done on both float dtypes).
    pub done: Vec<Op>,
    /// Ops enumerated by the vocab but not yet evaluable in this seed
    /// (integer-only bitwise atoms, structural atoms, and the reductions /
    /// scans / normalizations that build on them).
    pub pending: Vec<Op>,
}

impl Ledger {
    /// A one-line human summary for the coverage-gate test output.
    pub fn summary(&self) -> String {
        let total = self.done.len() + self.pending.len();
        format!(
            "kiss-ref coverage — f32/f64 scalar path: {} of {} ops DONE, {} PENDING. \
             Dtypes: 2 of 20 DONE (f32, f64); 18 PENDING (f16/bf16/fp8/sub-byte/complex).",
            self.done.len(),
            total,
            self.pending.len()
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
        if float_supported(op) {
            done.push(op);
        } else {
            pending.push(op);
        }
    }
    Ledger { done, pending }
}
