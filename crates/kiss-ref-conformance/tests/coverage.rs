// SPDX-License-Identifier: MIT OR Apache-2.0
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
    Dtype::I8,
    Dtype::I16,
    Dtype::I32,
    Dtype::I64,
    Dtype::U8,
    Dtype::U16,
    Dtype::U32,
    Dtype::U64,
    Dtype::I4,
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
    // Every op is now evaluable on at least one lane — the full 121, including the
    // window family and the §6.18 complex family (on c64/c128). Remaining gaps are
    // per-(op×dtype) cells (bool edges, the integer/FP8 tensor lanes), not whole
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
    // every integer dtype, incl. the packed i4/u4/b1.
    let mut checked = 0usize;
    for &op in Op::ALL {
        if int_tensor_supported(op) {
            for &d in &INT_DTYPES {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "vacuous: int_tensor_supported matched no op");
}

#[test]
fn coverage_tensor_layer_done_on_floats() {
    // The 6 structural atoms + the tensor non-primitives are Done on every float
    // dtype (the float lane: f16/bf16/f32/f64), Pending elsewhere in this cut.
    let mut checked = 0usize;
    for &op in Op::ALL {
        if tensor_supported(op) {
            for &d in &[
                Dtype::F16,
                Dtype::Bf16,
                Dtype::F32,
                Dtype::F64,
                Dtype::F8e4m3fn,
                Dtype::F8e5m2,
            ] {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "vacuous: tensor_supported matched no op");
}

#[test]
fn coverage_fp8_float_cells_done() {
    // FP8 (f8e4m3fn/f8e5m2) covers the same float op-set as f16/bf16 (compute via
    // promotion to f32): every non-nextafter float op is Done; nextafter and
    // bitwise are NotApplicable.
    let mut checked = 0usize;
    for &op in Op::ALL {
        for &d in &[Dtype::F8e4m3fn, Dtype::F8e5m2] {
            if float_supported(op) && op != Op::Nextafter {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "vacuous: float_supported matched no fp8 cell");
    assert_eq!(
        support(Op::Nextafter, Dtype::F8e4m3fn),
        Support::NotApplicable
    );
    assert_eq!(support(Op::BitAnd, Dtype::F8e5m2), Support::NotApplicable);
}

#[test]
fn coverage_float_ops_done_on_f32_f64() {
    let mut checked = 0usize;
    for &op in Op::ALL {
        if float_supported(op) {
            assert_eq!(support(op, Dtype::F32), Support::Done, "{op:?}/f32");
            assert_eq!(support(op, Dtype::F64), Support::Done, "{op:?}/f64");
            checked += 1;
        }
    }
    assert!(checked > 0, "vacuous: float_supported matched no op");
}

#[test]
fn coverage_int_ops_done_on_every_integer_dtype() {
    // The integer scalar path covers the bitwise/arithmetic atoms uniformly
    // across all integer dtypes — a consumer's correctness floor for integers.
    let mut checked = 0usize;
    for &op in Op::ALL {
        if int_supported(op) {
            for &d in &INT_DTYPES {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "vacuous: int_supported matched no op");
}

#[test]
fn coverage_narrow_floats_match_wide_except_nextafter() {
    // f16/bf16 cover the same float ops as f32/f64, minus nextafter (KISS-OPS-6.9-0003).
    let mut checked = 0usize;
    for &op in Op::ALL {
        for &d in &[Dtype::F16, Dtype::Bf16] {
            if float_supported(op) && op != Op::Nextafter {
                assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "vacuous: float_supported matched no narrow-float cell"
    );
    // The `nextafter` facts below are loop-INVARIANT — they name one op, so their
    // population is one and they are asserted once, outside the loop. Inside it they
    // executed `Op::ALL` times for a claim that holds once: the mirror of the vacuity
    // this file now guards against (a loop that asserts nothing vs. a loop that asserts
    // the same thing N times). Neither shape reports its own arity, so keep them apart.
    //
    // nextafter is NOT APPLICABLE on the narrow floats (KISS-OPS-6.9-0003) — spec-illegal,
    // not merely unimplemented...
    assert_eq!(support(Op::Nextafter, Dtype::F16), Support::NotApplicable);
    assert_eq!(support(Op::Nextafter, Dtype::Bf16), Support::NotApplicable);
    // ...but supported on the wide floats.
    assert_eq!(support(Op::Nextafter, Dtype::F32), Support::Done);
}

#[test]
fn coverage_support_consistency() {
    // support() is Done on a float dtype iff float_supported (narrow floats minus
    // nextafter), on an integer dtype iff int_supported, else never.
    for &op in Op::ALL {
        for &d in Dtype::ALL.iter() {
            let expect = match d {
                Dtype::F32 | Dtype::F64 => float_supported(op) || tensor_supported(op),
                Dtype::F16 | Dtype::Bf16 | Dtype::F8e4m3fn | Dtype::F8e5m2 => {
                    (float_supported(op) && op != Op::Nextafter) || tensor_supported(op)
                }
                Dtype::Bool => bool_supported(op),
                Dtype::C64 | Dtype::C128 => op.is_complex(),
                // sk4: reserved FP8 variants + MX scales are recognized but decline
                // compute — never Done (§6.1-0001).
                Dtype::F8e4m3fnuz | Dtype::F8e5m2fnuz | Dtype::F8e8m0 | Dtype::F8e6m2 => false,
                _ if int_spec(d).is_some() => int_supported(op) || int_tensor_supported(op),
                _ => false,
            };
            assert_eq!(support(op, d) == Support::Done, expect, "{op:?}/{d:?}");
        }
    }
}

#[test]
fn coverage_complex_cells_done_real_ops_not_applicable() {
    // §6.18: the complex-arithmetic family is Done on the complex compute dtypes;
    // every REAL op is NotApplicable on a complex dtype (and vice versa). The
    // (op × c64/c128) matrix is exactly the 15 complex ops × 2 dtypes.
    for &d in &[Dtype::C64, Dtype::C128] {
        for &op in Op::ALL {
            let expect = if op.is_complex() {
                Support::Done
            } else {
                Support::NotApplicable
            };
            assert_eq!(support(op, d), expect, "{op:?}/{d:?}");
        }
    }
    // A complex op is NotApplicable on a real dtype (float and integer).
    for &op in &[Op::Cadd, Op::Cmul, Op::Cexp, Op::Cabs, Op::Cmake] {
        assert_eq!(
            support(op, Dtype::F32),
            Support::NotApplicable,
            "{op:?}/f32"
        );
        assert_eq!(
            support(op, Dtype::I32),
            Support::NotApplicable,
            "{op:?}/i32"
        );
    }
}

#[test]
fn coverage_illegal_cells_are_not_applicable() {
    // Permanently-illegal (op × dtype) cells report NotApplicable, not Pending —
    // they leave the coverage backlog entirely.
    assert_eq!(support(Op::BitAnd, Dtype::F32), Support::NotApplicable); // bitwise×float KISS-OPS-6.10-0001
    assert_eq!(support(Op::BitOr, Dtype::Bool), Support::NotApplicable); // bitwise×bool
    assert_eq!(support(Op::Div, Dtype::I32), Support::NotApplicable); // div×int KISS-OPS-6.4-0002
    assert_eq!(support(Op::Exp, Dtype::I32), Support::NotApplicable); // transcendental×int §6.8
    assert_eq!(support(Op::Softmax, Dtype::U8), Support::NotApplicable); // normalization×int
    assert_eq!(support(Op::Nextafter, Dtype::F16), Support::NotApplicable); // KISS-OPS-6.9-0003
                                                                            // ...FP8 float ops and the bool truth-valued ops are now Done.
    assert_eq!(support(Op::Add, Dtype::F8e4m3fn), Support::Done);
    assert_eq!(support(Op::LogicalAnd, Dtype::Bool), Support::Done);
}

/// ⚠️ `Error::UnsupportedDtype` must keep meaning exactly ONE thing.
///
/// Its doc once claimed two: "an integer op on a float dtype" (**spec-illegal**, permanent)
/// and "a dtype with no reference path yet" (**incompleteness**, transient). A caller must
/// respond to those oppositely — *this will never work* versus *this does not work yet* —
/// and one code cannot say which. Measured 2026-09-06: every emission site is the first
/// kind (`int_spec` refusing a non-integer dtype in `scalar_int.rs` / `tensor_int.rs`, and
/// the bool guard in `boolean.rs`), so the second meaning is emitted nowhere.
///
/// The coverage ledger keeps the distinction the error type erases: `NotApplicable` is the
/// permanent typed compute-decline, `Pending` is the backlog. **While `pending` is EMPTY the
/// ambiguity is latent rather than live** — there is no cell in the incompleteness state for
/// the code to have to describe.
///
/// ⚠️ This pins that, and it is deliberately **independent of KISS #420**. The KISS-side
/// decline-code VOCABULARY is deferred to that ruling (mirroring a ruled set beats inventing
/// a private one), but the detector is not: the day a `Pending` cell appears, one code starts
/// meaning two things, and this test goes RED under any #420 outcome. **The fix then is to
/// SPLIT the variant — the enum is `#[non_exhaustive]`, so that is additive — never to relax
/// this assertion.**
#[test]
fn unsupported_dtype_is_unambiguously_spec_illegal_while_pending_is_empty() {
    let l = ledger();
    assert!(
        l.pending.is_empty(),
        "a Pending cell now exists, so Error::UnsupportedDtype can mean BOTH spec-illegal \
         and not-yet-implemented. SPLIT the variant (see KISS #420) — do not relax this: {:?}",
        l.pending_tokens()
    );

    // Both surfaces on one cell, so the linkage is pinned rather than incidental: a bitwise
    // atom on a float is spec-illegal (KISS-OPS-6.10-0001), the ledger calls it
    // NotApplicable, and the integer path's own predicate refuses the dtype — which is
    // precisely what raises UnsupportedDtype there.
    assert_eq!(support(Op::BitAnd, Dtype::F32), Support::NotApplicable);
    assert!(
        int_spec(Dtype::F32).is_none(),
        "int_spec is the predicate behind the decline; if it accepted a float the error \
         would no longer mean what this test says it means"
    );
    // Control: the same predicate ACCEPTS a legal integer dtype, so the assertion above is
    // discriminating rather than true of everything.
    assert!(
        int_spec(Dtype::I32).is_some(),
        "control: i32 is an integer dtype"
    );
}

#[test]
fn coverage_bool_truth_cells_done() {
    // The truth-valued bool ops (logical / eq / select / min-max / {0,1}-preserving
    // structural + data-movement) are Done; arithmetic, bitwise, transcendental,
    // and the sum/prod-bearing reductions are NotApplicable on bool.
    let mut checked = 0usize;
    for &op in Op::ALL {
        if bool_supported(op) {
            assert_eq!(support(op, Dtype::Bool), Support::Done, "{op:?}/bool");
            checked += 1;
        }
    }
    assert!(checked > 0, "vacuous: bool_supported matched no op");
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

#[test]
fn coverage_reserved_and_mx_dtypes_not_applicable() {
    // sk4 KISS-CLASSIFY-6.1-0001: the reserved FP8 variants (f8e4m3fnuz/f8e5m2fnuz)
    // and the MX scale dtypes (f8e8m0/f8e6m2) are RECOGNIZED members of the closed
    // vocabulary but carry no element-value compute semantics at this schema
    // version. Every op is NotApplicable on them — a typed compute-decline, never
    // a Pending backlog item.
    for &op in Op::ALL {
        for &d in &[
            Dtype::F8e4m3fnuz,
            Dtype::F8e5m2fnuz,
            Dtype::F8e8m0,
            Dtype::F8e6m2,
        ] {
            assert_eq!(support(op, d), Support::NotApplicable, "{op:?}/{d:?}");
        }
    }
    // recognized-vs-unknown: they parse (Some), distinct from an unknown token (None).
    for tok in ["f8e4m3fnuz", "f8e5m2fnuz", "f8e8m0", "f8e6m2"] {
        assert!(Dtype::from_token(tok).is_some(), "{tok} recognized");
    }
    assert!(
        Dtype::from_token("f8e9m9").is_none(),
        "unknown token declines as None"
    );
}

#[test]
fn coverage_ledger_surfaces_provisional_pin_count() {
    let l = ledger();
    // The provisional-pin COUNT sits in the machine-readable summary, beside the
    // Done/Pending counts (not in prose) — robust to the count being zero (the
    // resolved state, currently) or nonzero (Architect ruling, 2026-08-12).
    let s = l.summary();
    assert!(
        s.contains(&format!("{} PROVISIONAL PIN(S)", l.provisional.len())),
        "summary must surface the provisional-pin count beside Done/Pending: {s}"
    );
    // Whenever a genuinely-open decision IS pinned, each entry must render its
    // site=value=issue detail in the summary so it can't hide — the forcing-function
    // contract for the next pin. The sort_network index-output dtype pin was resolved
    // by KISS-OPS-6.11-0019, so the registry is currently empty and this loop is a
    // no-op; it keeps teeth for the next pin without asserting a resolved one.
    for p in l.provisional {
        assert!(
            s.contains(&format!("[{}={} pending {}]", p.site, p.value, p.issue)),
            "each provisional pin must render its detail in the summary: {s}"
        );
    }
}
