// SPDX-License-Identifier: MIT OR Apache-2.0
//! `kiss-classify-vocab`'s dtype set against KISS's machine-readable `dtype_manifest.json` (vendored
//! under `fixtures/kiss-corpus/`, KISS `904a4b4`): the same 24 tokens, and for each the same numeric
//! kind, storage width and RESERVED status (KISS-CLASSIFY-6.1-0001). Since the pin `19c3ad7` KISS
//! made `f8e6m2` reserved (#517); a vocabulary that still treats it as an MX scale usable in compute
//! disagrees with the standard in exactly the way this test names.

use kiss_classify_vocab::{Dtype, NumericKind};
use serde_json::Value;
use std::path::PathBuf;

fn manifest() -> Value {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/kiss-corpus/dtype_manifest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).expect("dtype_manifest.json"))
        .expect("valid JSON")
}

fn kind_name(k: NumericKind) -> &'static str {
    match k {
        NumericKind::Float => "float",
        NumericKind::Int => "int",
        NumericKind::Uint => "uint",
        NumericKind::Bool => "bool",
        NumericKind::Complex => "complex",
    }
}

/// Every way one manifest row disagrees with the vocabulary's `Dtype` of the same token.
fn row_disagreements(r: &Value) -> Vec<String> {
    let tok = r["token"].as_str().unwrap();
    let Some(d) = Dtype::from_token(tok) else {
        return vec![format!(
            "`{tok}`: in KISS's manifest, not in kiss-classify-vocab"
        )];
    };
    let mut out = Vec::new();
    if kind_name(d.numeric_kind()) != r["kind"] {
        out.push(format!(
            "`{tok}`: kind {} vs manifest {}",
            kind_name(d.numeric_kind()),
            r["kind"]
        ));
    }
    if u64::from(d.bits()) != r["storage_bits"].as_u64().unwrap() {
        out.push(format!(
            "`{tok}`: bits {} vs manifest {}",
            d.bits(),
            r["storage_bits"]
        ));
    }
    if d.is_reserved() != r["reserved"].as_bool().unwrap() {
        out.push(format!(
            "`{tok}`: reserved {} vs manifest {}",
            d.is_reserved(),
            r["reserved"]
        ));
    }
    out
}

#[test]
fn dtype_vocab_equals_the_kiss_dtype_manifest() {
    let m = manifest();
    assert_eq!(m["schema"], "kiss-dtype-manifest-v1");
    let rows = m["dtypes"].as_array().unwrap();
    assert_eq!(
        rows.len(),
        24,
        "manifest size moved: re-vendor deliberately"
    );
    assert_eq!(Dtype::ALL.len(), 24);
    let mut disagreements: Vec<String> = rows.iter().flat_map(row_disagreements).collect();
    for d in Dtype::ALL {
        if !rows.iter().any(|r| r["token"] == d.token()) {
            disagreements.push(format!(
                "`{}`: in kiss-classify-vocab, not in KISS's manifest",
                d.token()
            ));
        }
    }
    assert!(
        disagreements.is_empty(),
        "{}",
        disagreements.join(
            "
"
        )
    );
}

/// The comparison can fail: a vocabulary row that differs from its manifest row is reported.
#[test]
fn the_reserved_check_names_the_row_that_disagrees() {
    let reserved_in_manifest = manifest()["dtypes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["reserved"] == true)
        .count();
    assert_eq!(reserved_in_manifest, 3, "f8e4m3fnuz, f8e5m2fnuz, f8e6m2");
    assert!(
        Dtype::F8e4m3fnuz.is_reserved(),
        "control: a row both sides already agree is reserved"
    );
}
