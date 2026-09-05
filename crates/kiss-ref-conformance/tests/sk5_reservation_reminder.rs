// SPDX-License-Identifier: MIT OR Apache-2.0
//! **sk5 `#[non_exhaustive]` reservation reminder** — a build-participating to-do,
//! deliberately NOT a memory note or a code comment.
//!
//! A schema revision is inherently a *wire* event; it does not have to also be a
//! *source break* for downstream. It becomes one only when a schema-growable public
//! type is a plain enum/struct: adding a variant/field then fails every exhaustive
//! `match` in a consumer. Marking such types `#[non_exhaustive]` decouples the two —
//! a later variant is a wire event the consumer's wildcard arm already tolerates.
//!
//! kiss-ref's three highest-value growth types are **already reserved**: `Dtype`
//! (dtype variants), `Op` (op variants), `Error` (error set). The enums below are the
//! **residual** — public and schema-growable, but not yet reserved. Adding
//! `#[non_exhaustive]` is *itself* a source break, so it must ride the next major
//! (the **sk5** break); a standalone `0.4.0` whose only content is future-proofing is
//! unjustifiable to a consumer reading the changelog.
//!
//! **Why this is a test and not a to-do list:** "recorded as an sk5 to-do" is a claim
//! nothing re-checks — the fossil-comment shape, silently missed when sk5 is designed
//! by someone reading the design doc rather than the list (KISS Architect, 2026-08-13;
//! the same "detectable beats remembered" principle behind the `PROVISIONAL_PINS`
//! value-match test). This test is `#[ignore]`d (the reservation is deliberately
//! deferred, not overdue), but it is **listed** by `cargo test` and its `type_name`
//! references are **compile-coupled**, so a rename or removal breaks *this file* loudly
//! instead of orphaning the reminder.
//!
//! **At sk5:** add `#[non_exhaustive]` to each enum below (in its defining crate), then
//! **delete this file** (`crates/kiss-ref-conformance/tests/sk5_reservation_reminder.rs`).
//!
//! Optionally replace it with a stronger successor — a *cross-crate exhaustiveness
//! canary*: a wildcard-free `match` on each reserved enum from this crate, which fails
//! to compile the moment the enum becomes `#[non_exhaustive]`. If you write it, give it
//! the treatment this reminder can't self-apply, because **its success condition is its
//! own compile failure** — a guard that, undocumented, is a trap:
//!
//! > A compile error at the canary means the reservation **landed** — it is the SUCCESS
//! > signal, not a regression. The fix is to **delete the canary** (its job is done),
//! > NOT to remove `#[non_exhaustive]` to make the build green — that would undo the very
//! > reservation the canary exists to confirm.
//!
//! Put that sentence in the canary's own doc. The sk5 implementor will hit its red build
//! at the exact moment they did the right thing; without the message, the cheapest way to
//! green is to reverse the reservation. That one comment is the difference between a guard
//! and a trap (KISS Architect, 2026-08-13).

use kiss_classify_vocab::NumericKind;
use kiss_ops_vocab::Family;
use kiss_ref_core::{Combine, Direction, IndexRef, Monoid, Node, OobPolicy};

/// The residual schema-growable public enums to reserve `#[non_exhaustive]` at sk5.
/// (`#[non_exhaustive]` is a cross-crate *compile* property, not runtime-observable in
/// the defining crate, so this is a build-listed + compile-coupled reminder rather than
/// a runtime assertion of the end state.)
fn residual() -> Vec<&'static str> {
    vec![
        core::any::type_name::<NumericKind>(), // kiss-classify-vocab
        core::any::type_name::<Family>(),      // kiss-ops-vocab
        core::any::type_name::<Monoid>(),      // kiss-ref-core (attrs / OpAttrs stand-in)
        core::any::type_name::<OobPolicy>(),
        core::any::type_name::<Combine>(),
        core::any::type_name::<Direction>(),
        core::any::type_name::<Node>(), // kiss-ref-core (recipe grammar)
        core::any::type_name::<IndexRef>(),
    ]
}

#[test]
#[ignore = "sk5: reserve the listed enums with #[non_exhaustive] in the sk5 break, then delete this reminder (see file)"]
fn sk5_reserve_schema_growable_enums_non_exhaustive() {
    // Carries the reminder in `cargo test`'s listing, where a reader sees it. The
    // reservation itself is deferred, so this stays ignored — but the two guards below
    // RUN, because an assertion inside an ignored test is not a guard.
    assert!(!residual().is_empty());
}

/// The scope pin, and the arrival check — neither ignored.
///
/// ⚠ The count assertion used to live inside the ignored test above, where it never
/// executed. Measured 2026-09-05: setting it to `99` against an 8-element list still
/// reported `test result: ok. 0 passed; 0 failed; 1 ignored`. It only fails under
/// `--ignored`, which no ordinary run passes. **`#[ignore]` disarms every assertion in
/// the body, so a tripwire placed there is decorative** — the file's own doc argues that
/// "recorded as an sk5 to-do is a claim nothing re-checks", and this was that shape one
/// level in.
#[test]
fn the_sk5_reservation_scope_is_pinned_and_we_are_still_pre_sk5() {
    let residual = residual();

    // The count is the pinned reservation scope: if a residual type is added, or one is
    // reserved/removed early, update this list AND the count together.
    assert_eq!(
        residual.len(),
        8,
        "sk5 reservation scope changed — reconcile the reminder: {residual:?}"
    );

    // ⚠ The arrival check. The reservation must ride the next breaking release, which
    // for a 0.x crate is a minor bump. Nothing watched for that event: sk5 could ship
    // with the reservation undone and this file would sit here, ignored, saying so to
    // nobody. Coupling the reminder to the version couples it to the release that
    // discharges it.
    //
    // `CARGO_PKG_VERSION`, not a hand-parse of `../../Cargo.toml`. This crate declares
    // `version.workspace = true`, so the two are the same string by construction — and
    // the macro is resolved by cargo at compile time, so it cannot be defeated by an
    // indented key, a reordered manifest, or a relative path that is wrong under some
    // other working directory. Cargo treats the manifest as a fingerprint input, so a
    // version bump rebuilds this test rather than leaving a stale constant baked in.
    let version = env!("CARGO_PKG_VERSION");

    assert!(
        version.starts_with("0.3."),
        "workspace version is {version}, no longer 0.3.x. If this is the sk5 break: add \
         #[non_exhaustive] to the {} enums listed in this file (in their defining \
         crates), then DELETE this file. If it is a breaking release that is NOT sk5, \
         widen this pin and say which release sk5 is.",
        residual.len()
    );
}
