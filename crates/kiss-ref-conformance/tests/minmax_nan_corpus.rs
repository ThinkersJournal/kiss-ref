// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generator + permanent guard for KISS #329's `ops-minmax-nan.json` — the NaN
//! axis of `max_prop` / `min_prop` / `fmax_ieee` / `fmin_ieee`, which the
//! `ops-minmax-signed-zero.json` corpus (48 vectors, zero NaN) cannot
//! discriminate: the four ops differ ONLY on NaN.
//!
//! Every `expected` is DECOMPOSITION-TRACED — produced by running `eval_op`,
//! which evaluates the §6.13 reference decomposition (`select(cmp_ne(a,a), …)`),
//! not a native `max`. A minmax NaN output is a MOVED `select` value, so per
//! **KISS-CONFORM-6.8-0010(a)** it compares **exact-byte, payload included** —
//! this file is all `exact-byte`, and the both-NaN rows discriminate by carrying
//! two DISTINCT payloads (max_prop returns a's bits, fmax_ieee returns b's).
//!
//! This test IS the permanent guard: it asserts eval_op matches an INDEPENDENT
//! propagate/suppress rule on every row, and that max_prop and fmax_ieee differ
//! (the swap-test) on every case. Run with `--nocapture` to emit the JSON.

use half::bf16;
use kiss_ops_vocab::Op;
use kiss_ref_core::eval_op;

// (json op name, Op, propagates?) — propagate = max_prop/min_prop (return the NaN
// operand); suppress = fmax_ieee/fmin_ieee (return the OTHER operand).
const OPS: [(&str, Op, bool); 4] = [
    ("max_prop", Op::MaxProp, true),
    ("min_prop", Op::MinProp, true),
    ("fmax_ieee", Op::FmaxIeee, false),
    ("fmin_ieee", Op::FminIeee, false),
];

struct Case {
    tags: &'static str,
    a: u64,
    an: bool, // a is NaN
    b: u64,
    bn: bool, // b is NaN
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}

macro_rules! gen_dtype {
    ($name:expr, $T:ty, $uint:ty, $finite:expr, $inf:expr, $qa:expr, $qb:expr, $sx:expr, $tc:ident, $rows:ident) => {{
        // signed NaN in `a`, in `b`, both-distinct, sNaN both positions,
        // sNaN-vs-qNaN (catches a quieted moved sNaN), and NaN-vs-inf both
        // positions (inf is not NaN, so the second cmp_ne branch differs).
        let cases: [Case; 8] = [
            Case { tags: r#"["nan","qnan","pos-a"]"#,                              a: $qa,     an: true,  b: $finite, bn: false },
            Case { tags: r#"["nan","qnan","pos-b"]"#,                              a: $finite, an: false, b: $qb,     bn: true  },
            Case { tags: r#"["nan","qnan","both-nan","distinct-payload"]"#,        a: $qa,     an: true,  b: $qb,     bn: true  },
            Case { tags: r#"["nan","snan","pos-a"]"#,                              a: $sx,     an: true,  b: $finite, bn: false },
            Case { tags: r#"["nan","snan","pos-b"]"#,                              a: $finite, an: false, b: $sx,     bn: true  },
            Case { tags: r#"["nan","snan","qnan","both-nan","distinct-payload"]"# ,a: $sx,     an: true,  b: $qb,     bn: true  },
            Case { tags: r#"["nan","qnan","inf","pos-a"]"#,                        a: $qa,     an: true,  b: $inf,    bn: false },
            Case { tags: r#"["nan","qnan","inf","pos-b"]"#,                        a: $inf,    an: false, b: $qb,     bn: true  },
        ];
        for c in cases.iter() {
            // every case must carry at least one NaN — that is what makes the
            // simplified propagate=(an?a:b) / suppress=(an?b:a) rule below exact
            // (no finite-ordering branch is ever reached).
            assert!(c.an || c.bn, "{}: case a=0x{:X} b=0x{:X} has no NaN operand", $name, c.a, c.b);
            let mut max_out = 0u64;
            let mut min_out = 0u64;
            let mut fmax_out = 0u64;
            let mut fmin_out = 0u64;
            for (op_name, op, propagates) in OPS.iter() {
                let got: $uint = eval_op::<$T>(
                    *op,
                    &[<$T>::from_bits(c.a as $uint), <$T>::from_bits(c.b as $uint)],
                )
                .unwrap()
                .to_bits();
                // INDEPENDENT rule (>=1 operand is NaN in every case here):
                // propagate -> the NaN operand (a first); suppress -> the OTHER.
                let want: $uint = if *propagates {
                    if c.an { c.a as $uint } else { c.b as $uint }
                } else if c.an {
                    c.b as $uint
                } else {
                    c.a as $uint
                };
                assert_eq!(
                    got, want,
                    "{} {}: a=0x{:X} b=0x{:X} — eval_op diverged from the decomposition rule",
                    $name, op_name, c.a, c.b
                );
                match *op_name {
                    "max_prop" => max_out = got as u64,
                    "min_prop" => min_out = got as u64,
                    "fmax_ieee" => fmax_out = got as u64,
                    "fmin_ieee" => fmin_out = got as u64,
                    _ => {}
                }
                $tc += 1;
                let a_bytes = hex(&(c.a as $uint).to_be_bytes());
                let b_bytes = hex(&(c.b as $uint).to_be_bytes());
                let e_bytes = hex(&got.to_be_bytes());
                $rows.push(format!(
                    r#"    {{"tcId": {}, "op": "{}", "dtype": "{}", "rounding": "roundTiesToEven", "inputs": [{{"role":"a","dtype":"{}","bits":"{}"}}, {{"role":"b","dtype":"{}","bits":"{}"}}], "expected": {{"dtype":"{}","bits":"{}"}}, "class": "exact-byte", "ulp_bound": 0, "provenance": "decomposition-traced", "tags": {}}}"#,
                    $tc, op_name, $name, $name, a_bytes, $name, b_bytes, $name, e_bytes, c.tags
                ));
            }
            // the swap-test, structurally: every case discriminates propagate vs suppress.
            assert_ne!(
                max_out, fmax_out,
                "{}: max_prop and fmax_ieee agree on a=0x{:X} b=0x{:X} — case does not discriminate",
                $name, c.a, c.b
            );
            // SCOPE STATEMENT (KISS #333's lesson — say which PAIRS a corpus
            // separates, not how many ops it names). This file separates
            // propagate from suppress; it is BLIND to max-from-min BY
            // CONSTRUCTION: a NaN row short-circuits at the `cmp_ne` guards and
            // never reaches the `cmp_ge`/`cmp_le` branch that is the only
            // difference between max_prop and min_prop. Asserted so the blindness
            // stays deliberate — max-from-min needs a finite-ordering vector,
            // which does not belong in a NaN file (KISS #333, not this file).
            assert_eq!(
                max_out, min_out,
                "{}: max_prop != min_prop on a NaN row — this file cannot separate them; something changed",
                $name
            );
            assert_eq!(
                fmax_out, fmin_out,
                "{}: fmax_ieee != fmin_ieee on a NaN row — unexpected",
                $name
            );
        }
    }};
}

#[test]
fn minmax_nan_corpus_generate_and_guard() {
    let mut tc = 0u32;
    let mut rows: Vec<String> = Vec::new();

    // f32: 1.0=3F800000, +inf=7F800000, qNaN payloads A/B, sNaN (quiet bit clear).
    gen_dtype!(
        "f32",
        f32,
        u32,
        0x3F80_0000u64,
        0x7F80_0000u64,
        0x7FC0_1234u64,
        0x7FC0_5678u64,
        0x7F80_1234u64,
        tc,
        rows
    );
    // f64.
    gen_dtype!(
        "f64",
        f64,
        u64,
        0x3FF0_0000_0000_0000u64,
        0x7FF0_0000_0000_0000u64,
        0x7FF8_0000_0000_1234u64,
        0x7FF8_0000_0000_5678u64,
        0x7FF0_0000_0000_1234u64,
        tc,
        rows
    );
    // bf16 (top 16 bits of f32): 1.0=3F80, +inf=7F80, qNaN A/B, sNaN.
    gen_dtype!("bf16", bf16, u16, 0x3F80u64, 0x7F80u64, 0x7FC1u64, 0x7FD2u64, 0x7F81u64, tc, rows);

    let json = format!(
        "{{\n  \"schema\": \"kiss-oracle-vectors-v1.json\",\n  \"kiss_substandard\": \"OPS\",\n  \"schema_version\": 1,\n  \"spec_clause\": \"KISS-CONFORM-6.5-0008\",\n  \"generator\": \"hand-drafted by kiss-ref; cosigned kiss-ref + Baracuda (ThinkersJournal/KISS#329)\",\n  \"number_of_vectors\": {},\n  \"byte_order\": \"hex is the value's bytes most-significant first, left to right\",\n  \"provenance_note\": \"non-normative: every cell is decomposition-traced by kiss-ref's eval_op over the §6.13 select-decompositions. A minmax NaN is a MOVED select output, so KISS-CONFORM-6.8-0010(a) pins it exact-byte, payload included; propagate ops return the NaN operand's bytes, suppress ops the other operand's. Both-NaN rows carry two distinct payloads so max_prop (returns a) and fmax_ieee (returns b) differ by bits. sNaN rows verify a moved sNaN is not quieted (host: x86 SSE2 preserves it). Discrimination scope (KISS #333): these vectors separate propagate (max_prop/min_prop) from suppress (fmax_ieee/fmin_ieee) but do NOT separate max from min: every NaN row short-circuits before the cmp_ge/cmp_le branch, so max_prop==min_prop and fmax_ieee==fmin_ieee on all 96. Separating max from min needs a finite-ordering vector, which does not belong in a NaN file.\",\n  \"vectors\": [\n{}\n  ]\n}}",
        tc,
        rows.join(",\n")
    );
    println!("KISS329_JSON_BEGIN\n{json}\nKISS329_JSON_END");
    assert_eq!(tc, 96, "expected 8 cases x 4 ops x 3 dtypes = 96 vectors");
}
