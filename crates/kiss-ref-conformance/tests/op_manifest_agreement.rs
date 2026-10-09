// SPDX-License-Identifier: MIT OR Apache-2.0
//! `kiss-ops-vocab`'s closed op set against KISS's machine-readable `op_manifest.json`
//! (vendored under `fixtures/kiss-corpus/`, KISS `904a4b4`): the two must name the SAME ops, in
//! both directions. KISS-OPS-6.1-0001 makes the op set closed, so a token on either side only is a
//! defect on that side — an op the manifest lists that kiss-ref cannot name (missed since the
//! pin `19c3ad7`: `argmin`) or a token kiss-ref invents.

use kiss_ops_vocab::Op;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn manifest() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/kiss-corpus/op_manifest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).expect("op_manifest.json"))
        .expect("valid JSON")
}

#[test]
fn op_vocab_equals_the_kiss_op_manifest_in_both_directions() {
    let m = manifest();
    assert_eq!(m["schema"], "kiss-op-manifest-v1");
    let theirs: BTreeSet<String> = m["all_ops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let ours: BTreeSet<String> = Op::ALL.iter().map(|o| o.token().to_owned()).collect();
    let missing: Vec<_> = theirs.difference(&ours).collect();
    let extra: Vec<_> = ours.difference(&theirs).collect();
    assert!(missing.is_empty() && extra.is_empty(), "in KISS's manifest but not kiss-ops-vocab: {missing:?}; in kiss-ops-vocab but not KISS's manifest: {extra:?}");
    // Non-vacuity: the sets are not both empty, and the count is the manifest's own header.
    assert_eq!(
        theirs.len(),
        122,
        "manifest size moved: re-vendor deliberately"
    );
}

/// The comparison can fail: remove a token from one side and the diff must name it.
#[test]
fn the_agreement_check_names_a_missing_token() {
    let theirs: BTreeSet<&str> = ["add", "argmin"].into();
    let ours: BTreeSet<&str> = ["add"].into();
    assert_eq!(
        theirs.difference(&ours).copied().collect::<Vec<_>>(),
        vec!["argmin"]
    );
}
