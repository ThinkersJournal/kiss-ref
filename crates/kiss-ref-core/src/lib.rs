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
//!   `libm`) and every integer dtype (incl. packed `s4`/`u4`/`b1`, computed in
//!   `i128` and wrapped).
//! - **Tensor layer** ([`tensor`]/[`kernels`]/[`tensor_ops`]/[`window`]): the six
//!   §6.11 structural atoms + all §6.13 tensor non-primitives on the float lane,
//!   plus an [`tensor_int`] integer tensor lane.
//! - **FP8** ([`fp8`]): `e4m3`/`e5m2` as `u8` newtypes with a hand-rolled f32 codec
//!   (RNE + saturation), computed via the narrow-float promote-to-f32 lane.
//! - **bool** ([`boolean`]): the truth-valued lane (§6.2-0006) over the integer
//!   engine, `{0,1}`-normalized.
//! - **complex** (`c32`/`c64`): **not applicable** — complex arithmetic is the
//!   deferred §6.18 op family (absent from the vocab), so [`support`] reports every
//!   `(op, c32/c64)` cell [`Support::NotApplicable`] (§6.16-0007), not `Pending`.
//!
//! Coverage is a three-state model — [`Support::Done`] / [`Support::Pending`] /
//! [`Support::NotApplicable`] — driven by the spec-derived [`legality`] function;
//! only legal cells form the denominator. The remaining `Pending` cells are the
//! integer-tensor float-only ops (`reduce_mean`/norms/… need `div`/`sqrt`) and a
//! few genuinely-legal-but-unimplemented edges.

#![cfg_attr(not(test), no_std)]

pub mod attrs;
pub mod boolean;
pub mod bridge;
pub mod diff;
pub mod fp8;
pub mod kernels;
pub mod recipe;
pub mod resolve;
pub mod scalar;
pub mod scalar_int;
pub mod tensor;
pub mod tensor_int;
pub mod tensor_ops;
pub mod window;

pub use diff::{
    diff_bf16, diff_e4m3, diff_e5m2, diff_expr, diff_expr_bf16, diff_expr_f16, diff_expr_f32,
    diff_f16, diff_f32, diff_f64, reference_bf16, reference_e4m3, reference_e5m2, reference_expr,
    reference_expr_bf16, reference_expr_f16, reference_expr_f32, reference_f16, reference_f32,
    reference_f64, reference_matmul_acc, reference_prefix_scan_acc, reference_reduce_acc,
    ulp_distance_bf16, ulp_distance_e4m3, ulp_distance_e5m2, ulp_distance_f16, ulp_distance_f32,
    ulp_distance_f64, DiffReport, Tolerance,
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
pub use fp8::{E4m3, E5m2};
pub use recipe::{eval_recipe, FlatDag, IndexRef, Node, RecipeEval};
pub use tensor::{IndexTensor, Tensor, View, MAX_OPERANDS, MAX_RANK};

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

/// A reference-evaluation failure. The reference never panics; every problem is
/// one of these (honoring the never-panic discipline of a consumer's execution
/// path).
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// accumulator MUST be a float dtype (RFC #92 C1).
    NonFloatAccumulator(Dtype),
    /// A legal `(narrow storage, wider-than-f32 accumulator)` cell whose exact C3
    /// reference kiss-ref cannot yet produce: the narrow storage types round via
    /// `f32`, so narrowing an `f64` accumulator would double-round (`f64→f32→S`)
    /// and miss the C3 "rounded once A→S" value by up to 1 ULP. Declined rather
    /// than returned wrong (a **Pending** cell — the follow-up is a correct
    /// round-to-odd `f64→narrow` codec). The common `acc = f32` path is exact and
    /// unaffected. RFC #92 (direction b).
    AccumulatorNarrowingUnsupported { storage: Dtype, acc: Dtype },
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
