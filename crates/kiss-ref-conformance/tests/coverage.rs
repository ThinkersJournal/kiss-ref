//! The coverage gate: enumerate every (op × dtype) cell and enforce the
//! implemented paths stay covered. A regression that drops a covered op to
//! `Pending` fails this test.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_conformance::ledger;
use kiss_ref_core::scalar_int::int_spec;
use kiss_ref_core::{float_supported, int_supported, support, Support};

const INT_DTYPES: [Dtype; 11] = [
    Dtype::S8,
    Dtype::S16,
    Dtype::I32,
    Dtype::I64,
    Dtype::U8,
    Dtype::U16,
    Dtype::U32,
    Dtype::U64,
    Dtype::S4,
    Dtype::U4,
    Dtype::B1,
];

#[test]
fn coverage_ledger_reports_done_and_pending() {
    let l = ledger();
    // Visible with `cargo test -- --nocapture` — the machine-readable ledger the
    // evaluating teams read to see what remains.
    println!("{}", l.summary());
    println!("PENDING ops ({}): {:?}", l.pending.len(), l.pending_tokens());

    assert!(
        l.done.len() >= 78,
        "expected >= 78 ops evaluable (float scalar + integer scalar), got {}",
        l.done.len()
    );
    assert_eq!(l.done.len() + l.pending.len(), Op::ALL.len());
}

#[test]
fn coverage_float_ops_done_on_f32_f64() {
    for &op in Op::ALL {
        if float_supported(op) {
            assert_eq!(support(op, Dtype::F32), Support::Done, "{op:?}/f32");
            assert_eq!(support(op, Dtype::F64), Support::Done, "{op:?}/f64");
        }
    }
}

#[test]
fn coverage_int_ops_done_on_every_integer_dtype() {
    // The integer scalar path covers the bitwise/arithmetic atoms uniformly
    // across all integer dtypes — a consumer's correctness floor for integers.
    for &op in Op::ALL {
        if int_supported(op) {
            for &d in &INT_DTYPES {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
            }
        }
    }
}

#[test]
fn coverage_narrow_floats_match_wide_except_nextafter() {
    // f16/bf16 cover the same float ops as f32/f64, minus nextafter (§6.9-0003).
    for &op in Op::ALL {
        for &d in &[Dtype::F16, Dtype::Bf16] {
            if float_supported(op) && op != Op::Nextafter {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
            }
        }
        // nextafter is declined on the narrow floats.
        assert_eq!(support(Op::Nextafter, Dtype::F16), Support::Pending);
        assert_eq!(support(Op::Nextafter, Dtype::Bf16), Support::Pending);
        // ...but supported on the wide floats.
        assert_eq!(support(Op::Nextafter, Dtype::F32), Support::Done);
    }
}

#[test]
fn coverage_support_consistency() {
    // support() is Done on a float dtype iff float_supported (narrow floats minus
    // nextafter), on an integer dtype iff int_supported, else never.
    for &op in Op::ALL {
        for &d in Dtype::ALL.iter() {
            let expect = match d {
                Dtype::F32 | Dtype::F64 => float_supported(op),
                Dtype::F16 | Dtype::Bf16 => float_supported(op) && op != Op::Nextafter,
                _ if int_spec(d).is_some() => int_supported(op),
                _ => false,
            };
            assert_eq!(support(op, d) == Support::Done, expect, "{op:?}/{d:?}");
        }
    }
}

#[test]
fn coverage_dtype_breadth_still_pending() {
    // FP8 / bool / complex have no reference path yet in the seed.
    for &d in &[
        Dtype::E4m3,
        Dtype::E5m2,
        Dtype::Bool,
        Dtype::C32,
        Dtype::C64,
    ] {
        for &op in Op::ALL {
            assert_eq!(support(op, d), Support::Pending, "{op:?}/{d:?}");
        }
    }
}
