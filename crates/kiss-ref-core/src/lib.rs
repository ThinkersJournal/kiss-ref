// SPDX-License-Identifier: MIT OR Apache-2.0
//! # kiss-ref-core
//!
//! The KISS base-op **reference kernels**: the floating-point floor atoms
//! (KISS-Ops §6.4–§6.9) computed spec-exactly per compute dtype, plus the
//! §6.13/§6.14 **recursive-resolution engine** that evaluates any non-primitive
//! op by expanding its reference decomposition to the floor. Deliberately naive,
//! obvious, and slow — correctness is the only goal.
//!
//! Because a total cover of the base map always runs, it doubles as the "it
//! always works" correctness floor and a differential *target* consumers test
//! against. It is **not** the KISS-Conform oracle: that oracle is independent by
//! mandate (shares no lowering code with any reference implementation, Conform
//! §6.5-0002/0003) and is the artifact that mints the conformance corpus. It
//! **never panics** — every failure is an [`Error`] — so it is safe on a
//! consumer's production/execution path (e.g. Fuel's never-panic backend
//! contract).
//!
//! ## Scope
//! - **Scalar floor + non-primitives** over `f16`/`bf16`/`f32`/`f64` (float, via
//!   `libm`) and every integer dtype (incl. packed `i4`/`u4`/`b1`, computed in
//!   `i128` and wrapped).
//! - **Tensor layer** ([`tensor`]/[`kernels`]/[`tensor_ops`]/[`window`]): the six
//!   §6.11 structural atoms + all §6.13 tensor non-primitives on the float lane,
//!   plus an [`tensor_int`] integer tensor lane.
//! - **FP8** ([`fp8`]): `f8e4m3fn`/`f8e5m2` as `u8` newtypes with a hand-rolled f32
//!   codec (RNE + saturation), computed via the narrow-float promote-to-f32 lane.
//! - **bool** ([`boolean`]): the truth-valued lane (§6.2-0006) over the integer
//!   engine, `{0,1}`-normalized.
//! - **complex** ([`complex`]): the §6.18 complex-arithmetic family
//!   (`cmake`/`cre`/`cim` bridge + `cadd`…`cpow`) on `c64`/`c128`, evaluating the
//!   real-atom decompositions in the `f32`/`f64` component lane (§6.18-0015) with
//!   the Annex-G special-value/recovery rules governing the inf/NaN edges
//!   (§6.18-0013). Every `(complex op, c64/c128)` cell is [`Support::Done`]; a real
//!   op on a complex dtype (and any complex op on a real dtype) is
//!   [`Support::NotApplicable`].
//!
//! Coverage is a three-state model — [`Support::Done`] / [`Support::Pending`] /
//! [`Support::NotApplicable`] — driven by the spec-derived [`legality`] function;
//! only legal cells form the denominator. The `Pending` set is empty — every legal
//! cell is `Done`. The integer-tensor float-only ops (`reduce_mean`/norms/… need
//! `div`/`sqrt`) are `NotApplicable` on integers, not backlog: KISS-OPS-6.4-0002
//! excludes integer division and remainder from the op set.

#![cfg_attr(not(test), no_std)]

pub mod attrs;
pub mod boolean;
pub mod bridge;
pub mod complex;
pub mod diff;
pub mod fp8;
pub mod kernels;
pub mod recipe;
pub mod recipe_int;
pub mod resolve;
pub mod rng;
pub mod scalar;
pub mod scalar_int;
pub mod tensor;
pub mod tensor_int;
pub mod tensor_ops;
pub mod window;

pub use diff::{
    arg_conforms_f32, arg_conforms_f64, complex_conforms_c32, complex_conforms_c64, diff_bf16,
    diff_e4m3, diff_e5m2, diff_expr, diff_expr_bf16, diff_expr_f16, diff_expr_f32, diff_f16,
    diff_f32, diff_f64, reference_bf16, reference_c32, reference_c64, reference_e4m3,
    reference_e5m2, reference_expr, reference_expr_bf16, reference_expr_f16, reference_expr_f32,
    reference_f16, reference_f32, reference_f64, reference_matmul_acc, reference_prefix_scan_acc,
    reference_reduce_acc, ulp_distance_bf16, ulp_distance_e4m3, ulp_distance_e5m2,
    ulp_distance_f16, ulp_distance_f32, ulp_distance_f64, DiffReport, Tolerance,
};
pub use resolve::{
    eval_expr, eval_op, float_supported, implemented, legality, support, tensor_supported,
};
pub use scalar::ScalarFloat;
pub use scalar_int::{eval_int_expr, eval_int_op, int_supported};
pub use tensor_int::int_tensor_supported;

pub use attrs::{Combine, Direction, Monoid, OobPolicy};
pub use boolean::{bool_supported, eval_bool_op};
pub use bridge::{DetClass, Evaluated};
pub use complex::{eval_complex_op, Cplx, CplxOut};
pub use fp8::{E4m3, E5m2};
pub use recipe::{eval_recipe, selection_det, FlatDag, IndexRef, Node, RecipeEval};
pub use recipe_int::eval_recipe_int;
pub use tensor::{IndexTensor, Tensor, View, MAX_OPERANDS, MAX_RANK};

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

/// A reference-evaluation failure. The reference never panics; every problem is
/// one of these (honoring the never-panic discipline of a consumer's execution
/// path).
// `#[non_exhaustive]`: this crate is published, and the error set has grown
// (the accumulator work added variants) and will grow again. Marking it here
// makes future additions non-breaking — downstream `match`es must already carry
// a wildcard arm, so no consumer churns on a new variant.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The op has no reference kernel and no bound §6.13 decomposition in this
    /// seed (a PENDING region — e.g. a structural-atom-based reduction).
    Unsupported(Op),
    /// Wrong number of arguments for the op.
    Arity { op: Op, expected: usize, got: usize },
    /// A decomposition referenced `input(i)` for which no value was supplied.
    MissingInput(u8),
    /// A stored §6.13 decomposition string failed to parse (a vocab bug — the
    /// conformance suite guards against this).
    BadDecomposition { op: Op, pos: usize },
    /// The op is not defined on this dtype in this seed (e.g. an integer op on a
    /// float dtype, or a dtype with no reference path yet).
    UnsupportedDtype(Dtype),
    /// A differential candidate slice length did not match the reference length.
    LengthMismatch { expected: usize, got: usize },

    // ---- tensor-evaluation layer (§6.11 structural atoms) ------------------
    /// A tensor rank exceeds `MAX_RANK` (§6.19-0037).
    RankExceeded { rank: usize, max: usize },
    /// A tensor did not have the shape an operation required (element count or a
    /// coordinate of the wrong arity).
    ShapeMismatch { expected: usize, got: usize },
    /// Two operand shapes are not broadcast-compatible (§6.11-0001 / §6.20-0007).
    BroadcastIncompatible,
    /// An axis argument is out of range for the operand rank.
    AxisOutOfRange { axis: usize, rank: usize },
    /// An index-operand dtype is not one of `{u32, i32, i64}` (§6.11-0009).
    IndexDtypeIllegal(Dtype),
    /// A `reduce`/`prefix_scan` axis selection was empty (`0x0000` forbidden,
    /// §6.19-0038), or a reduction/scan was asked for no axes.
    EmptyAxesMask,
    /// Shape element-count arithmetic overflowed `usize`.
    ShapeOverflow,
    /// A recipe's [`recipe::IndexRef::Node`] or [`recipe::FlatDag::index_outputs`]
    /// entry referenced a node with no index-lane output, or an out-of-range node
    /// id. (`usize`, not `u8` — node ids exceed 255.)
    IndexSourceInvalid { node: usize },
    /// A `gather`/`scatter` [`recipe::IndexRef::Slot`] referenced an external index
    /// operand `indices[slot]` that was not supplied. Distinct from
    /// [`Error::MissingInput`] (a missing value-lane input/param): this is the
    /// index lane, and — like [`Error::IndexSourceInvalid`] — the slot is `usize`,
    /// not `u8`, so a large slot number is reported faithfully, never truncated.
    MissingIndexOperand { slot: usize },
    /// A `gather` under `oob = skip` read an **actually-OOB** index with no
    /// `base` operand supplied. Per the ruled §6.11 gather-skip semantics
    /// (KISS PR #75, Gap 1 option 1, ruled 2026-07-23): the base requirement is
    /// **dynamic** — `skip` + `base: None` is legal while every index is in
    /// range (§6.13 `index_select` relies on exactly that), and only an actual
    /// OOB read declines.
    GatherSkipNoBase,
    /// A requested reduction/scan/contraction accumulator dtype is narrower than
    /// (or incomparable to) the storage/compute dtype. RFC #92 (direction b) C1:
    /// a narrower accumulator is forbidden — the storage→accumulator promotion
    /// would be lossy, so it is a typed decline, not a silent rounding.
    AccumulatorTooNarrow { storage: Dtype, acc: Dtype },
    /// A requested accumulator dtype is not a float. A reduction/scan/contraction
    /// accumulator MUST be a float dtype (RFC #92 C1). This is the genuine
    /// wrong-kind decline (e.g. an integer accumulator); a `NumericKind::Float`
    /// dtype that merely has **no compute semantics** declines as
    /// [`Error::ReservedOrScaleDtype`] instead, so the two are not conflated.
    NonFloatAccumulator(Dtype),
    /// A dtype that is a **recognized** member of the sk4 vocabulary but has **no
    /// element-value compute semantics** at this schema version — a reserved FP8
    /// variant (`f8e4m3fnuz`/`f8e5m2fnuz`) or an MX scale (`f8e8m0`/`f8e6m2`,
    /// §6.1-0013) — was supplied in a compute position (e.g. an accumulator dtype).
    /// These are `NumericKind::Float`, so this is a **typed compute-decline**
    /// distinct from [`Error::NonFloatAccumulator`] (§6.1-0001;
    /// [`kiss_classify_vocab::Dtype::declines_compute`]).
    ReservedOrScaleDtype(Dtype),
}

/// Coverage of an `(op, dtype)` cell. Three states: a cell is either
/// spec-**legal** (and then `Done` or `Pending`) or **not applicable** — the op
/// has no meaning on that dtype and the spec mandates a typed decline. Only legal
/// cells (`Done` + `Pending`) form the coverage denominator; `NotApplicable` cells
/// are excluded, not counted as a backlog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// A reference kernel exists and is conformance-tested.
    Done,
    /// Spec-legal on this dtype, but no reference path in this seed yet.
    Pending,
    /// The op is not applicable to this dtype — permanently illegal per spec
    /// (e.g. a bitwise op on a float, `div` on an integer, any of the 106 ops on
    /// a complex dtype whose arithmetic is the deferred §6.18 family). Excluded
    /// from the coverage denominator.
    NotApplicable,
}

/// A **provisional local pin**: a value the reference fixes to keep evaluating while
/// the KISS spec leaves it open, recorded so the reference never presents an
/// unratified choice as settled conformance.
///
/// The pin is a fact about a **site**, not an `(op × dtype)` coverage cell — so it
/// does not muddy the `Support` verdict of any cell (a cell whose *value* semantics
/// are verified stays [`Support::Done`]; only the named sub-decision is provisional).
/// It carries its **value** (not merely its existence): a test asserts the code still
/// produces `value`, so when `issue` rules differently the assertion **fails** and
/// forces reconciliation — the pin's resolution is driven, not remembered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvisionalPin {
    /// The site whose value is fixed (where the pin lives).
    pub site: &'static str,
    /// The pinned value, as its normative token (asserted against the code by a test).
    pub value: &'static str,
    /// The KISS tracking issue whose ruling resolves the pin.
    pub issue: &'static str,
}

/// The reference's **provisional local pins** — choices kiss-ref fixes to keep
/// evaluating while the spec leaves them open. Each is countable and surfaced in the
/// coverage ledger **beside the Done/Pending counts** (not in prose); when its `issue`
/// rules and the pin is ratified or re-pinned, the entry is removed and the count
/// drops. **Zero is the resolved state — and is the current state.**
///
/// Currently **empty**. The one prior pin — `sort_network` **index-output dtype** =
/// `i64` — is resolved by **KISS-OPS-6.11-0019** (the sort index-lane output MUST be
/// `i64`, producer-side, with no wire field; realized in KISS `daa5f43` / PR #83, and
/// already present in kiss-ref's bound spec artifact `19c3ad7`). The ruling *agreed*
/// with the pinned value, so reconciliation was a re-cite, not a re-pin: the `i64`
/// output is now asserted directly against the clause (`test_ops_sort_index_output_i64`
/// in the conformance crate, plus a kernel-level guard in `kernels.rs`), not against
/// this registry. The pin had tracked KISS **#133** — an `rfc` issue filed 2026-08-08,
/// fifteen days *after* the ruling merged (a re-file of an already-answered question).
/// A guard that cites an open issue rather than the merged clause re-transmits a stale
/// "waiting on X" to every reader, so the fix is to anchor to the clause. The mechanism
/// stays for the next genuinely-open decision.
pub const PROVISIONAL_PINS: &[ProvisionalPin] = &[];

/// Where a reference kernel came from — the provenance rule of `DESIGN.md`
/// ("reuse Fuel iff spec-exact").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// Ported from a Fuel kernel verified spec-exact against its KISS §6 clause.
    PortedFromFuel,
    /// Written fresh from the spec (or `libm`-grade math).
    Fresh,
}

/// Provenance of the reference kernel for `op` in this seed.
///
/// **Every first-cut kernel is [`Provenance::Fresh`]** — written from the spec /
/// `libm`, not copied from Fuel. The Fuel-port audit (does Fuel's kernel meet the
/// KISS §6 clause under its determinism class? — if so, re-tag `PortedFromFuel`)
/// is exactly the reconciliation the Fuel/Baracuda evaluation phase performs, so
/// the tags are filled then, not asserted now.
pub fn provenance(_op: Op) -> Provenance {
    Provenance::Fresh
}

/// A short **source-lineage** tag for the reference kernel of `op` — where its
/// numeric behavior comes from — so a consumer's differential harness can
/// attribute a result to a source (Baracuda's provenance ask). First cut: every
/// kernel is spec-derived over `libm 0.2`; per-op lineage is refined as kernels
/// are ported or replaced.
pub fn source_lineage(_op: Op) -> &'static str {
    "kiss-ref-core: KISS-Ops §6 semantics, libm 0.2 transcendentals (Fresh)"
}
