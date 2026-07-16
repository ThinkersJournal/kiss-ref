//! The coverage gate: enumerate every (op × dtype) cell and enforce the
//! mandatory core stays covered. A regression that drops a covered op to
//! `Pending` fails this test — turning "always works" from a claim into an
//! enforced invariant over the supported dtypes.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_conformance::ledger;
use kiss_ref_core::{float_supported, support, Support};

#[test]
fn coverage_ledger_reports_done_and_pending() {
    let l = ledger();
    // Visible with `cargo test -- --nocapture`; the machine-readable ledger the
    // evaluating teams read to see what remains.
    println!("{}", l.summary());
    println!("PENDING ops ({}): {:?}", l.pending.len(), l.pending_tokens());

    // A meaningful chunk of the vocabulary is covered — the float floor atoms
    // plus the elementwise non-primitives that resolve through them.
    assert!(
        l.done.len() >= 50,
        "expected >= 50 ops covered in the seed, got {}",
        l.done.len()
    );
    assert_eq!(l.done.len() + l.pending.len(), Op::ALL.len());
}

#[test]
fn coverage_mandatory_core_is_done_on_both_float_dtypes() {
    // Every op the reference evaluates MUST be Done on BOTH f32 and f64 — a
    // consumer picking the reference as its correctness floor gets total cover
    // over the supported dtypes.
    for &op in &ledger().done {
        assert_eq!(support(op, Dtype::F32), Support::Done, "{op:?} on f32");
        assert_eq!(support(op, Dtype::F64), Support::Done, "{op:?} on f64");
    }
}

#[test]
fn coverage_support_matches_float_supported_exactly() {
    // `support()` on a float dtype is Done iff the op is float-supported —
    // the ledger and the evaluator never disagree.
    for &op in Op::ALL {
        let expect = if float_supported(op) {
            Support::Done
        } else {
            Support::Pending
        };
        assert_eq!(support(op, Dtype::F32), expect, "{op:?}/f32");
        assert_eq!(support(op, Dtype::F64), expect, "{op:?}/f64");
    }
}

#[test]
fn coverage_non_float_dtypes_are_all_pending_in_seed() {
    for &op in Op::ALL {
        for &d in &Dtype::ALL {
            if !matches!(d, Dtype::F32 | Dtype::F64) {
                assert_eq!(support(op, d), Support::Pending, "{op:?}/{d:?}");
            }
        }
    }
}
