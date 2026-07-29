//! The coverage gate: enumerate every (op × dtype) cell and enforce the
//! implemented paths stay covered. A regression that drops a covered op to
//! `Pending` fails this test.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_conformance::ledger;
use kiss_ref_core::scalar_int::int_spec;
use kiss_ref_core::{
    bool_supported, float_supported, int_supported, int_tensor_supported, support,
    tensor_supported, Support,
};

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
    println!(
        "PENDING ops ({}): {:?}",
        l.pending.len(),
        l.pending_tokens()
    );

    assert_eq!(l.done.len() + l.pending.len(), Op::ALL.len());
    // Every op is now evaluable on at least the float (or integer scalar) lane —
    // the full 106, including the window family. Remaining gaps are per-(op×dtype)
    // cells (FP8/bool/complex, and the integer/FP8/complex tensor lanes), not whole
    // ops, so the per-op ledger is complete.
    assert_eq!(
        l.done.len(),
        Op::ALL.len(),
        "every op should be evaluable; still pending: {:?}",
        l.pending_tokens()
    );
    assert!(
        l.pending.is_empty(),
        "no op should remain pending: {:?}",
        l.pending_tokens()
    );
}

#[test]
fn coverage_int_tensor_lane_done_on_integers() {
    // The integer-capable tensor ops (atoms + argmax/any/all/cum*) are Done on
    // every integer dtype, incl. the packed s4/u4/b1.
    for &op in Op::ALL {
        if int_tensor_supported(op) {
            for &d in &INT_DTYPES {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
            }
        }
    }
}

#[test]
fn coverage_tensor_layer_done_on_floats() {
    // The 6 structural atoms + the tensor non-primitives are Done on every float
    // dtype (the float lane: f16/bf16/f32/f64), Pending elsewhere in this cut.
    for &op in Op::ALL {
        if tensor_supported(op) {
            for &d in &[
                Dtype::F16,
                Dtype::Bf16,
                Dtype::F32,
                Dtype::F64,
                Dtype::E4m3,
                Dtype::E5m2,
            ] {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
            }
        }
    }
}

#[test]
fn coverage_fp8_float_cells_done() {
    // FP8 (e4m3/e5m2) covers the same float op-set as f16/bf16 (compute via
    // promotion to f32): every non-nextafter float op is Done; nextafter and
    // bitwise are NotApplicable.
    for &op in Op::ALL {
        for &d in &[Dtype::E4m3, Dtype::E5m2] {
            if float_supported(op) && op != Op::Nextafter {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
            }
        }
    }
    assert_eq!(support(Op::Nextafter, Dtype::E4m3), Support::NotApplicable);
    assert_eq!(support(Op::BitAnd, Dtype::E5m2), Support::NotApplicable);
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
        // nextafter is NOT APPLICABLE on the narrow floats (§6.9-0003) — spec-illegal,
        // not merely unimplemented.
        assert_eq!(support(Op::Nextafter, Dtype::F16), Support::NotApplicable);
        assert_eq!(support(Op::Nextafter, Dtype::Bf16), Support::NotApplicable);
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
                Dtype::F32 | Dtype::F64 => float_supported(op) || tensor_supported(op),
                Dtype::F16 | Dtype::Bf16 | Dtype::E4m3 | Dtype::E5m2 => {
                    (float_supported(op) && op != Op::Nextafter) || tensor_supported(op)
                }
                Dtype::Bool => bool_supported(op),
                _ if int_spec(d).is_some() => int_supported(op) || int_tensor_supported(op),
                _ => false,
            };
            assert_eq!(support(op, d) == Support::Done, expect, "{op:?}/{d:?}");
        }
    }
}

#[test]
fn coverage_complex_all_not_applicable() {
    // §6.16-0007: complex arithmetic is the deferred §6.18 op family, so NONE of
    // the 106 vocab ops apply to a complex compute dtype — every cell is
    // NotApplicable, not a pending backlog item.
    for &d in &[Dtype::C32, Dtype::C64] {
        for &op in Op::ALL {
            assert_eq!(support(op, d), Support::NotApplicable, "{op:?}/{d:?}");
        }
    }
}

#[test]
fn coverage_illegal_cells_are_not_applicable() {
    // Permanently-illegal (op × dtype) cells report NotApplicable, not Pending —
    // they leave the coverage backlog entirely.
    assert_eq!(support(Op::BitAnd, Dtype::F32), Support::NotApplicable); // bitwise×float §6.10-0001
    assert_eq!(support(Op::BitOr, Dtype::Bool), Support::NotApplicable); // bitwise×bool
    assert_eq!(support(Op::Div, Dtype::I32), Support::NotApplicable); // div×int §6.4-0002
    assert_eq!(support(Op::Exp, Dtype::I32), Support::NotApplicable); // transcendental×int §6.8
    assert_eq!(support(Op::Softmax, Dtype::U8), Support::NotApplicable); // normalization×int
    assert_eq!(support(Op::Nextafter, Dtype::F16), Support::NotApplicable); // §6.9-0003
                                                                            // ...FP8 float ops and the bool truth-valued ops are now Done.
    assert_eq!(support(Op::Add, Dtype::E4m3), Support::Done);
    assert_eq!(support(Op::LogicalAnd, Dtype::Bool), Support::Done);
}

#[test]
fn coverage_bool_truth_cells_done() {
    // The truth-valued bool ops (logical / eq / select / min-max / {0,1}-preserving
    // structural + data-movement) are Done; arithmetic, bitwise, transcendental,
    // and the sum/prod-bearing reductions are NotApplicable on bool.
    for &op in Op::ALL {
        if bool_supported(op) {
            assert_eq!(support(op, Dtype::Bool), Support::Done, "{op:?}/bool");
        }
    }
    assert_eq!(support(Op::Add, Dtype::Bool), Support::NotApplicable);
    assert_eq!(support(Op::BitAnd, Dtype::Bool), Support::NotApplicable);
    assert_eq!(support(Op::Exp, Dtype::Bool), Support::NotApplicable);
    assert_eq!(support(Op::ReduceMean, Dtype::Bool), Support::NotApplicable);
    assert_eq!(support(Op::ScatterAdd, Dtype::Bool), Support::NotApplicable); // sum escapes {0,1}
    assert_eq!(support(Op::CmpLt, Dtype::Bool), Support::NotApplicable); // ordered cmp declined
                                                                         // im2col is bool-LEGAL (pure {0,1}-preserving data movement, OOB → 0) and
                                                                         // is now backed by the shared integer tensor_int::im2col kernel that the
                                                                         // bool lane reuses over {0,1}, so it genuinely evaluates → Done.
    assert_eq!(support(Op::Im2col, Dtype::Bool), Support::Done);
}
