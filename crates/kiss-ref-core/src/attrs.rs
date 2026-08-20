// SPDX-License-Identifier: MIT OR Apache-2.0
//! Attribute enums for the tensor-evaluation layer.
//!
//! These stand in for the deferred §6.19 `OpAttrs` channel: a structural atom's
//! parameters (the reduction monoid, gather out-of-bounds policy, scatter
//! combine, sort direction) that in the wire form travel as an `OpAttrs` blob.
//! Until §6.19 is bound in the vocab, the tensor kernels take them as these typed
//! enums. Kept deliberately small and named exactly after the spec vocabulary.

/// The associative reduction/scan monoid (§6.11-0002). `max`/`min` are
/// NaN-propagating; `sum`/`prod` over floats are order-invariant/nondeterministic
/// (§6.0-0004).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Monoid {
    /// Additive fold; identity `+0.0` / integer `0`.
    Sum,
    /// Multiplicative fold; identity `1`.
    Prod,
    /// Maximum; identity `-inf` / dtype minimum; NaN-propagating.
    Max,
    /// Minimum; identity `+inf` / dtype maximum; NaN-propagating.
    Min,
}

/// Out-of-bounds policy for a `gather` READ (§6.11-0004). A negative signed index
/// is always out of bounds (no from-end wrap).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OobPolicy {
    /// Leave the caller-provided base value at that output position (§6.11
    /// gather-skip semantics, ruled option 1 on KISS PR #75, 2026-07-23; base
    /// requirement is dynamic — no base + an actually-OOB read is the typed
    /// [`crate::Error::GatherSkipNoBase`] decline).
    Skip,
    /// Read the index clamped into `[0, extent)`.
    Clamp,
    /// Output `0` at that position.
    ZeroFill,
}

/// Scatter WRITE combine (§6.11-0005 / §6.19-0016). Out-of-bounds writes are
/// always skipped (oob is fixed `skip` for scatter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    /// Overwrite; duplicate-index tie-break = highest row-major source wins
    /// (deterministic, §6.11-0006).
    Assign,
    /// Add into the destination; duplicates accumulate. Float `atomic_add` is
    /// order-invariant/nondeterministic (§6.0-0004); integer is exact.
    AtomicAdd,
    /// Maximum into the destination; NaN-propagating; deterministic.
    AtomicMax,
    /// Minimum into the destination; NaN-propagating; deterministic.
    AtomicMin,
}

/// Sort direction (§6.11-0007). NaN orders as the greatest value in either
/// direction (ascending → last, descending → first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Ascending.
    Asc,
    /// Descending.
    Desc,
}
