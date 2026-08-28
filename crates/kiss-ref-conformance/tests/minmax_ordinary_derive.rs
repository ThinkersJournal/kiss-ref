// SPDX-License-Identifier: MIT OR Apache-2.0
//! Independent eval_op derivation + orientation guard for KISS #333's
//! `ops-minmax-ordinary.json` — the 24 strict-inequality vectors that separate
//! max_prop/fmax_ieee from min_prop/fmin_ieee, which neither the signed-zero tie
//! file nor the #329 NaN file can (a NaN row short-circuits before the
//! `cmp_ge`/`cmp_le` branch, and a tie makes both comparisons true).
//!
//! KISS 1 hand-derives the file from the §6.13 decompositions; this is the
//! SECOND, independent derivation (kiss-ref's eval_op over the same inputs),
//! cited in the file's two-party provenance. The guard is the ORIENTATION check
//! the born-red max<->min swap control structurally cannot give: the swap proves
//! the pair is DISCRIMINATED, but a consistently-inverted corpus (max returns the
//! smaller) passes it. Here `want` is computed by NATIVE value comparison —
//! independent of eval_op's decomposition — so it pins WHICH member is which:
//! max-family = the larger operand, min-family = the smaller.
//!
//! Run with `--nocapture` to emit the 24 (op, dtype, direction, bits) for a
//! diff-verify against KISS 1's hand-derivation.

use half::bf16;
use kiss_ops_vocab::Op;
use kiss_ref_core::eval_op;

// (json op name, Op, returns-the-larger?)
const OPS: [(&str, Op, bool); 4] = [
    ("max_prop", Op::MaxProp, true),
    ("fmax_ieee", Op::FmaxIeee, true),
    ("min_prop", Op::MinProp, false),
    ("fmin_ieee", Op::FminIeee, false),
];

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn minmax_ordinary_derive_and_orient() {
    // d1: a=2.0 b=1.0 ; d2: a=1.0 b=2.0 — strict inequality, non-NaN, non-tie.
    let dirs = [
        ("d1(a=2,b=1)", 2.0f64, 1.0f64),
        ("d2(a=1,b=2)", 1.0f64, 2.0f64),
    ];
    println!("KISS333_DERIVE_BEGIN");
    let mut n = 0u32;
    for (dir, a, b) in dirs {
        // NATIVE, eval_op-independent orientation oracle: larger/smaller by value.
        let (larger, smaller) = if a > b { (a, b) } else { (b, a) };
        for (name, op, larger_family) in OPS.iter() {
            let want = if *larger_family { larger } else { smaller };

            let got_f32 = eval_op::<f32>(*op, &[a as f32, b as f32])
                .unwrap_or_else(|e| panic!("{name} f32 {dir}: eval_op failed: {e:?}"))
                .to_bits();
            let got_f64 = eval_op::<f64>(*op, &[a, b])
                .unwrap_or_else(|e| panic!("{name} f64 {dir}: eval_op failed: {e:?}"))
                .to_bits();
            let got_bf16 =
                eval_op::<bf16>(*op, &[bf16::from_f32(a as f32), bf16::from_f32(b as f32)])
                    .unwrap_or_else(|e| panic!("{name} bf16 {dir}: eval_op failed: {e:?}"))
                    .to_bits();

            // eval_op (decomposition) must match the native-value orientation on
            // every dtype — this is the "which member is which" check.
            assert_eq!(
                got_f32,
                (want as f32).to_bits(),
                "{name} f32 {dir}: eval_op orientation != native"
            );
            assert_eq!(
                got_f64,
                want.to_bits(),
                "{name} f64 {dir}: eval_op orientation != native"
            );
            assert_eq!(
                got_bf16,
                bf16::from_f32(want as f32).to_bits(),
                "{name} bf16 {dir}: eval_op orientation != native"
            );

            println!(
                "  {name:>9} {dir:>12}  f32={}  f64={}  bf16={}",
                hex(&got_f32.to_be_bytes()),
                hex(&got_f64.to_be_bytes()),
                hex(&got_bf16.to_be_bytes())
            );
            n += 3;
        }
    }
    println!("KISS333_DERIVE_END");
    assert_eq!(n, 24, "expected 4 ops x 3 dtypes x 2 directions = 24");
}
