// SPDX-License-Identifier: MIT OR Apache-2.0
//! **An RFC in this repo must not call itself a draft once this repo says its clauses are
//! normative.**
//!
//! `docs/rfc-accumulator-tolerance-cells-2026-07-24.md` carried `**Status:** draft` for four
//! weeks after KISS #90 closed COMPLETED, its direction was ruled, it was filed as KISS PR
//! #92, realized by KISS PR #96, and `KISS-OPS-6.17-0008` landed on KISS `main`.
//!
//! ⚠ **The contradiction was already inside this repository, not across a repo boundary.**
//! `narrow_tensor_lane.rs` says *"NORMATIVE now … these cells are no longer Provisional —
//! they BACK 6.17-0008/-0009"*. Two files, one repo, opposite claims about the same clause,
//! and the stale one was the field a reader reaches for first. Nothing connected them.
//!
//! So the coupling is local and needs no network: **if the conformance suite claims to back
//! a clause, the RFC that proposed that clause is not a draft.** A test that had to fetch
//! KISS `main` would be a different and more fragile thing; this asserts only what this
//! repository already says about itself, which is the part that was self-contradictory.
//!
//! At the next such RFC: give it a status the moment its ruling lands, or add it here.

use std::path::{Path, PathBuf};

fn repo_file(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {} — {e}", p.display()))
}

/// The `**Status:**` line of a document, or `None` when there is none.
///
/// Returns `None` rather than an empty string so a caller fails loudly instead of
/// comparing against nothing when the field is renamed or removed.
fn status_line(doc: &str) -> Option<&str> {
    doc.lines()
        .map(str::trim)
        .find(|l| l.starts_with("**Status:**"))
}

/// Lines of `doc` that are outside a blockquote.
///
/// The DISCHARGED note quotes the wording it retires — a retraction that cannot name what
/// it retracts is a weak one — so a whole-document scan would fire on the retraction and
/// could only be satisfied by deleting the history.
fn live_body(doc: &str) -> String {
    doc.lines()
        .filter(|l| !l.trim_start().starts_with('>'))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Does `line` carry `draft` as a WORD?
///
/// ⚠ Not `contains("draft")`. The corrected status reads "drafted 2026-07-24 by kiss-ref",
/// which is a true statement about when the document was written and must not read as a
/// live draft status — a substring check fired on it, and this test caught its own false
/// positive on the first run. A status naming its drafting date is the normal way to write
/// this, so the loose form would have been wrong for every RFC fixed the same way.
fn says_draft(line: &str) -> bool {
    line.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|w| w == "draft")
}

const RFC: &str = "../../docs/rfc-accumulator-tolerance-cells-2026-07-24.md";
const BACKING_TEST: &str = "tests/narrow_tensor_lane.rs";
const CLAUSE: &str = "6.17-0008";

#[test]
fn the_accumulator_rfc_is_not_a_draft_while_the_suite_backs_its_clause() {
    let backing = repo_file(BACKING_TEST);
    let rfc = repo_file(RFC);

    // Positive control on the premise. If the suite ever stops claiming to back the
    // clause, this test's reasoning no longer applies and it must say so rather than
    // pass silently on a premise that has evaporated.
    assert!(
        backing.contains(CLAUSE),
        "{BACKING_TEST} no longer mentions KISS-OPS-{CLAUSE}. Either the backing moved — \
         in which case point this test at its new home — or it was withdrawn, in which \
         case the RFC's status needs revisiting rather than this test deleting."
    );
    assert!(
        backing.contains("BACK 6.17-0008"),
        "{BACKING_TEST} no longer states that these cells BACK the clause; the premise of \
         this check is gone and it would otherwise pass having verified nothing"
    );

    let status = status_line(&live_body(&rfc))
        .map(str::to_string)
        .expect("no `**Status:**` line in the accumulator RFC — this test checked nothing");

    assert!(
        !says_draft(&status),
        "the accumulator RFC calls itself a draft while {BACKING_TEST} states its clause is \
         NORMATIVE and backed.\n\n  status: {status}\n\n\
         KISS #90 closed COMPLETED 2026-08-08; the RFC was filed as KISS PR #92 and realized \
         by KISS PR #96. Quoting the old wording inside the DISCHARGED blockquote is intended; \
         asserting it as the document's status is not."
    );
}

/// The extractors must be able to fail, or the assertion above can be green while
/// comparing against nothing.
#[test]
fn the_extractors_return_none_rather_than_a_false_match() {
    assert_eq!(status_line("no status here"), None);
    assert_eq!(
        status_line("**Status:** draft · 2026-07-24"),
        Some("**Status:** draft · 2026-07-24")
    );

    // A status inside a blockquote is history, not the document's status.
    assert_eq!(
        status_line(&live_body("> **Status:** draft\n\nbody")),
        None,
        "a quoted status must not be read as the live one"
    );
    assert_eq!(live_body("> quoted\n> more").trim(), "");
    assert_eq!(live_body("> quoted\nlive").trim(), "live");

    // ⚠ The word-boundary discrimination, pinned. A `contains("draft")` check fired on the
    // CORRECTED status's own "drafted 2026-07-24" the first time this test ran — the loose
    // form would have been wrong for every RFC fixed the same way, since naming the
    // drafting date is the normal way to write a ratified status.
    assert!(says_draft("**Status:** draft - 2026-07-24"));
    assert!(says_draft("**Status:** DRAFT"));
    assert!(!says_draft(
        "**Status:** RATIFIED - drafted 2026-07-24 by kiss-ref"
    ));
    assert!(!says_draft(
        "**Status:** ratified; superseded the drafting round"
    ));
}

/// The RFC path this test depends on must exist, so a rename fails here rather than
/// quietly removing the check.
#[test]
fn the_documents_this_check_reads_are_where_it_expects() {
    for rel in [RFC, BACKING_TEST] {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        assert!(
            p.exists(),
            "{rel} is missing — this check has lost its subject"
        );
    }
}
