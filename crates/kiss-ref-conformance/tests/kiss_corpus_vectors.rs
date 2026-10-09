// SPDX-License-Identifier: MIT OR Apache-2.0
//! kiss-ref against KISS's own oracle-vector corpus (`kiss-oracle-vectors-v1.json`).
//!
//! The reference evaluator is run on every vector of the six op-corpus files vendored under
//! `fixtures/kiss-corpus/` (KISS `904a4b4`, see `PROVENANCE.md`) and must reproduce `expected`:
//!
//! - `exact-byte`: the result's raw bits equal `expected` — payload and sign of a NaN included
//!   (KISS-CONFORM-6.8-0010(a): a MOVED value compares exact-byte, and a value compare of
//!   `0.0 == -0.0` would pass vacuously).
//! - `ULP`: a non-NaN `expected` is matched within `ulp_bound` on the integer totalOrder
//!   distance; a NaN `expected` is a COMPUTED NaN (KISS-OPS-6.16-0010), so the result must be a NaN
//!   whose QUIETNESS equals `expected`'s — the payload is not compared — for a dtype that admits a
//!   signaling NaN.
//!
//! Without this test kiss-ref is checked only against itself (the other tests in this crate are
//! hand-derived). The vendored files are guarded by `corpus_files_are_the_vendored_blobs`, which
//! pins the vector count of each so a partial re-vendor cannot pass quietly.

use half::{bf16, f16};
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_op, E4m3, E5m2};
use serde_json::Value;
use std::path::PathBuf;

const FILES: &[(&str, usize)] = &[
    ("ops-arith", 5),
    ("ops-minmax-ordinary", 24),
    ("ops-minmax-signed-zero", 48),
    ("ops-narrow-move-nan", 30),
    ("ops-narrow-select-nan", 40),
    ("ops-transcendental-nan", 12),
];

fn load(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/kiss-corpus")
        .join(format!("{name}.json"));
    let s =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// `"7F 81"` -> `0x7F81` (bytes most-significant first; ' ' and '·' are grouping marks).
fn parse_hex(s: &str) -> u64 {
    let mut v = 0u64;
    for c in s.chars().filter(|c| c.is_ascii_hexdigit()) {
        v = (v << 4) | c.to_digit(16).unwrap() as u64;
    }
    v
}

/// Integer totalOrder key of a float's bits (`width` bits): monotone in the value.
fn order_key(bits: u64, width: u32) -> i128 {
    let sign = 1u64 << (width - 1);
    if bits & sign != 0 {
        -((bits & !sign) as i128)
    } else {
        bits as i128
    }
}

/// Run one vector; returns `Err(reason)` on a mismatch.
fn run_one(v: &Value) -> Result<(), String> {
    let op_name = v["op"].as_str().unwrap();
    let dtype = v["dtype"].as_str().unwrap();
    let class = v["class"].as_str().unwrap();
    let ulp = v["ulp_bound"].as_u64().unwrap_or(0);
    let op =
        Op::from_token(op_name).ok_or_else(|| format!("op `{op_name}` not in kiss-ops-vocab"))?;
    let inputs: Vec<u64> = v["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| parse_hex(i["bits"].as_str().unwrap()))
        .collect();
    let want = parse_hex(v["expected"]["bits"].as_str().unwrap());

    macro_rules! lane {
        ($T:ty, $width:expr, $qbit:expr, $has_snan:expr) => {{
            let args: Vec<$T> = inputs.iter().map(|&b| <$T>::from_bits(b as _)).collect();
            let got = eval_op::<$T>(op, &args)
                .map_err(|e| format!("eval error: {e:?}"))?
                .to_bits() as u64;
            (got, $width, $qbit, $has_snan)
        }};
    }
    let (got, width, qbit, has_snan): (u64, u32, u64, bool) = match dtype {
        "f32" => lane!(f32, 32, 1u64 << 22, true),
        "f64" => lane!(f64, 64, 1u64 << 51, true),
        "bf16" => lane!(bf16, 16, 1u64 << 6, true),
        "f16" => lane!(f16, 16, 1u64 << 9, true),
        "f8e5m2" => lane!(E5m2, 8, 1u64 << 1, true),
        "f8e4m3fn" => lane!(E4m3, 8, 0, false),
        other => return Err(format!("dtype `{other}` has no lane in this runner")),
    };

    let is_nan = |bits: u64| -> bool {
        let (e_bits, m_bits) = match width {
            64 => (11, 52),
            32 => (8, 23),
            16 if dtype == "bf16" => (8, 7),
            16 => (5, 10),
            8 if dtype == "f8e5m2" => (5, 2),
            _ => (4, 3),
        };
        let exp = (bits >> m_bits) & ((1u64 << e_bits) - 1);
        let man = bits & ((1u64 << m_bits) - 1);
        if dtype == "f8e4m3fn" {
            exp == 0xF && man == 0x7
        } else {
            exp == (1u64 << e_bits) - 1 && man != 0
        }
    };

    match class {
        "exact-byte" => {
            if got == want {
                Ok(())
            } else {
                Err(format!("got {got:#x}, expected {want:#x}"))
            }
        }
        "ULP" => {
            if is_nan(want) {
                if !is_nan(got) {
                    return Err(format!("expected a NaN ({want:#x}), got {got:#x}"));
                }
                if has_snan && ((got & qbit) != 0) != ((want & qbit) != 0) {
                    return Err(format!(
                        "NaN quietness differs: got {got:#x}, expected {want:#x}"
                    ));
                }
                Ok(())
            } else {
                let d = (order_key(got, width) - order_key(want, width)).unsigned_abs();
                if d <= ulp as u128 {
                    Ok(())
                } else {
                    Err(format!(
                        "{d} ulp from expected (bound {ulp}): got {got:#x}, expected {want:#x}"
                    ))
                }
            }
        }
        other => Err(format!("class `{other}` has no comparator in this runner")),
    }
}

#[test]
fn corpus_files_are_the_vendored_blobs() {
    for (name, n) in FILES {
        let d = load(name);
        let vectors = d["vectors"].as_array().unwrap();
        assert_eq!(
            vectors.len(),
            *n,
            "{name}: vector count moved; re-vendor and update FILES + PROVENANCE.md together"
        );
        assert_eq!(
            d["number_of_vectors"].as_u64(),
            Some(*n as u64),
            "{name}: header disagrees with body"
        );
    }
}

#[test]
fn kiss_ref_reproduces_every_vector_of_the_kiss_corpus() {
    let mut failures = Vec::new();
    let mut total = 0usize;
    for (name, _) in FILES {
        let d = load(name);
        for v in d["vectors"].as_array().unwrap() {
            total += 1;
            if let Err(why) = run_one(v) {
                failures.push(format!(
                    "{name} tcId {} ({} {}): {why}",
                    v["tcId"],
                    v["op"].as_str().unwrap(),
                    v["dtype"].as_str().unwrap()
                ));
            }
        }
    }
    assert!(
        total == FILES.iter().map(|f| f.1).sum::<usize>(),
        "ran {total} vectors"
    );
    assert!(
        failures.is_empty(),
        "{} of {total} vectors failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The runner can fail: a corrupted `expected` must be reported, not skipped.
#[test]
fn the_runner_detects_a_wrong_expected_value() {
    let mut d = load("ops-arith");
    let v = &mut d["vectors"][0];
    let good = v["expected"]["bits"].as_str().unwrap().to_owned();
    assert!(run_one(v).is_ok(), "control: the unmodified vector passes");
    v["expected"]["bits"] = Value::String(if good == "00 00 00 01" {
        "00 00 00 02".into()
    } else {
        "00 00 00 01".into()
    });
    assert!(run_one(v).is_err(), "a wrong exact-byte expected must fail");
}
