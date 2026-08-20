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
#[test]
#[ignore = "sk5: reserve the listed enums with #[non_exhaustive] in the sk5 break, then delete this reminder (see file)"]
fn sk5_reserve_schema_growable_enums_non_exhaustive() {
    let residual = [
        core::any::type_name::<NumericKind>(), // kiss-classify-vocab
        core::any::type_name::<Family>(),      // kiss-ops-vocab
        core::any::type_name::<Monoid>(),      // kiss-ref-core (attrs / OpAttrs stand-in)
        core::any::type_name::<OobPolicy>(),
        core::any::type_name::<Combine>(),
        core::any::type_name::<Direction>(),
        core::any::type_name::<Node>(), // kiss-ref-core (recipe grammar)
        core::any::type_name::<IndexRef>(),
    ];
    // The count is the pinned reservation scope: if a residual type is added or one of
    // these is reserved/removed early, update this list AND the count together.
    assert_eq!(
        residual.len(),
        8,
        "sk5 reservation scope changed — reconcile the reminder: {residual:?}"
    );
}
