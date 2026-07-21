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
//! ## Seed scope
//! The scalar path implements the float floor atoms + elementwise non-primitives
//! over `f16`/`bf16`/`f32`/`f64`, and the integer floor atoms over all integer
//! dtypes (incl. the packed `s4`/`u4`/`b1`), computed in `i128` and wrapped. The
//! FP8 (`e4m3`/`e5m2`), `bool`, and complex (`c32`/`c64`) dtypes; the structural
//! atoms (`element_map` … `sort_network`) and the reduction / scan / normalization
//! / contraction non-primitives that build on them; and the tensor (`element_map`)
//! wrapper are enumerated in the vocab and reported `Pending` by [`support`] — the
//! next wave for the evaluating teams.

#![cfg_attr(not(test), no_std)]

pub mod attrs;
pub mod bridge;
pub mod diff;
pub mod kernels;
pub mod resolve;
pub mod scalar;
pub mod scalar_int;
pub mod tensor;
pub mod tensor_int;
pub mod tensor_ops;
pub mod window;

pub use diff::{
    diff_bf16, diff_f16, diff_f32, diff_f64, reference_bf16, reference_f16, reference_f32,
    reference_f64, ulp_distance_bf16, ulp_distance_f16, ulp_distance_f32, ulp_distance_f64,
    DiffReport, Tolerance,
};
pub use resolve::{eval_expr, eval_op, float_supported, implemented, support, tensor_supported};
pub use scalar::ScalarFloat;
pub use scalar_int::{eval_int_expr, eval_int_op, int_supported};
pub use tensor_int::int_tensor_supported;

pub use attrs::{Combine, Direction, Monoid, OobPolicy};
pub use bridge::{DetClass, Evaluated};
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
}

/// Coverage of an `(op, dtype)` cell in this seed. The conformance coverage gate
/// enumerates every legal cell and reports the `Done` / `Pending` split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    /// A reference kernel exists and is conformance-tested.
    Done,
    /// Enumerated by the vocab but not yet implemented in this seed.
    Pending,
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
