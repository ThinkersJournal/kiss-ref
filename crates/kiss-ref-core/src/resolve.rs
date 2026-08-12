//! The recursive-resolution engine (KISS-Ops §6.13 / §6.14).
//!
//! [`eval_op`] evaluates any KISS-Ops op on scalar float arguments: a floor atom
//! is computed directly; a non-primitive is resolved by parsing its §6.13
//! reference decomposition and evaluating it on the floor (with the §6.13-0003
//! overflow-safe refined forms substituted for the overflow-unsafe cases). This
//! is the total-cover property — every op with a bound decomposition is
//! evaluable — expressed as code.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::{Dtype, NumericKind};
use kiss_ops_vocab::decomp::{parse, Expr};
use kiss_ops_vocab::{Family, Op};

use crate::scalar::{mish_stable, silu_stable, softplus_stable, ScalarFloat};
use crate::{Error, Support};

fn arity<T>(op: Op, args: &[T], n: usize) -> Result<(), Error> {
    if args.len() == n {
        Ok(())
    } else {
        Err(Error::Arity {
            op,
            expected: n,
            got: args.len(),
        })
    }
}

fn un<T: ScalarFloat>(op: Op, args: &[T], f: impl Fn(T) -> T) -> Result<T, Error> {
    arity(op, args, 1)?;
    Ok(f(args[0]))
}

fn bin<T: ScalarFloat>(op: Op, args: &[T], f: impl Fn(T, T) -> T) -> Result<T, Error> {
    arity(op, args, 2)?;
    Ok(f(args[0], args[1]))
}

/// Evaluate a KISS-Ops op on scalar float arguments, spec-exactly for the
/// compute dtype `T`. Floor atoms compute directly; non-primitives resolve
/// through their §6.13 decomposition. Never panics — every failure is an
/// [`Error`].
pub fn eval_op<T: ScalarFloat>(op: Op, args: &[T]) -> Result<T, Error> {
    match op {
        // arithmetic atoms (§6.4)
        Op::Add => bin(op, args, |a, b| a.add(b)),
        Op::Sub => bin(op, args, |a, b| a.sub(b)),
        Op::Mul => bin(op, args, |a, b| a.mul(b)),
        Op::Div => bin(op, args, |a, b| a.div(b)),
        Op::Neg => un(op, args, |a| a.neg()),
        Op::Abs => un(op, args, |a| a.abs()),

        // raw-bit select (§6.5): move the chosen arm unchanged (no arithmetic),
        // so its signed zero / NaN payload is preserved bit-for-bit.
        Op::Select => {
            arity(op, args, 3)?;
            Ok(if args[0].is_truthy() {
                args[1]
            } else {
                args[2]
            })
        }

        // comparison atoms (§6.6): 1/0 in the compute dtype. Rust's PartialOrd is
        // IEEE-unordered for NaN, so cmp_lt/le/gt/ge yield 0 on NaN and cmp_ne
        // yields 1 (KISS-OPS-6.6-0002/0003), and signed-zero equality holds.
        Op::CmpEq => bin(op, args, |a, b| T::from_bool(a == b)),
        Op::CmpNe => bin(op, args, |a, b| T::from_bool(a != b)),
        Op::CmpLt => bin(op, args, |a, b| T::from_bool(a < b)),
        Op::CmpLe => bin(op, args, |a, b| T::from_bool(a <= b)),
        Op::CmpGt => bin(op, args, |a, b| T::from_bool(a > b)),
        Op::CmpGe => bin(op, args, |a, b| T::from_bool(a >= b)),

        // rounding atoms (§6.7)
        Op::Floor => un(op, args, |a| a.floor()),
        Op::Ceil => un(op, args, |a| a.ceil()),
        Op::Trunc => un(op, args, |a| a.trunc()),
        Op::RoundEven => un(op, args, |a| a.round_even()),

        // transcendental atoms (§6.8, declared-ULP)
        Op::Exp => un(op, args, |a| a.exp()),
        Op::Log => un(op, args, |a| a.log()),
        Op::Sin => un(op, args, |a| a.sin()),
        Op::Cos => un(op, args, |a| a.cos()),
        Op::Sqrt => un(op, args, |a| a.sqrt()),
        Op::Erf => un(op, args, |a| a.erf()),
        Op::Atan => un(op, args, |a| a.atan()),
        Op::Lgamma => un(op, args, |a| a.lgamma()),

        // binary-math atoms (§6.9)
        Op::Atan2 => bin(op, args, |a, b| a.atan2(b)),
        Op::Copysign => bin(op, args, |a, b| a.copysign(b)),
        Op::Nextafter => {
            // §6.9-0003: nextafter declines the narrow floats (f16/bf16/e4m3/
            // e5m2) — stepping in a promoted f32 yields the wrong neighbor.
            if T::NARROW_FLOAT {
                return Err(Error::Unsupported(op));
            }
            bin(op, args, |a, b| a.nextafter(b))
        }

        // refined non-primitives (§6.13-0003, all 11 refine-marked ops):
        // computed directly because the literal decomposition overflows,
        // catastrophically cancels, or gets a pinned domain edge wrong.
        Op::Tanh => un(op, args, |a| a.tanh()),
        Op::Sinh => un(op, args, |a| a.sinh()),
        Op::Cosh => un(op, args, |a| a.cosh()),
        Op::Softplus => un(op, args, |a| softplus_stable(a)),
        Op::Silu => un(op, args, |a| silu_stable(a)),
        Op::Mish => un(op, args, |a| mish_stable(a)),
        // near-zero-accurate (literal forms catastrophically cancel).
        Op::Expm1 => un(op, args, |a| a.expm1()),
        Op::Log1p => un(op, args, |a| a.log1p()),
        // domain-exact: IEEE pow/hypot pin §6.13-0005 (pow(-2,3)=-8, pow(0,0)=1)
        // and §6.13-0007 (hypot(inf,NaN)=+inf), which the reference forms miss.
        Op::Pow => bin(op, args, |a, b| a.pow(b)),
        Op::Hypot => bin(op, args, |a, b| a.hypot(b)),
        // exact 2^b scaling via IEEE pow (§6.13-0003 permits the exact form).
        // The direct `a * 2^b` is exact for integer `b`, but materializing `2^b`
        // spuriously overflows to +inf for `b >= 1024` (f64) even when the
        // product `a * 2^b` is finite (e.g. `ldexp(1e-300, 1024)` = ~1.8e8, not
        // inf — found by the §6.13 decomposition self-differential). When `2^b`
        // genuinely overflows to infinity, recover the possibly-finite product
        // by splitting the exponent into two parts (`b = lo + hi` with
        // `lo = floor(b/2)`); each `2^k` stays in range for |b| < 2048, and the
        // floor/ceil split keeps every integer `b` exact (each half is an exact
        // power of two). Narrow-lane saturation (`2^b` -> a *finite* max under
        // §6.16-0004/-0005) is deliberately left alone — the guard is on an
        // actual infinity, not on saturation.
        Op::Ldexp => bin(op, args, |a, b| {
            let p = T::from_f64(2.0).pow(b);
            if p.to_f64().is_infinite() {
                let bf = b.to_f64();
                let lo = (bf * 0.5).floor();
                let hi = bf - lo;
                let sl = T::from_f64(2.0).pow(T::from_f64(lo));
                let sh = T::from_f64(2.0).pow(T::from_f64(hi));
                a.mul(sl).mul(sh)
            } else {
                a.mul(p)
            }
        }),

        // everything else: resolve via the §6.13 decomposition (§6.14).
        other => resolve_nonprimitive(other, args),
    }
}

fn resolve_nonprimitive<T: ScalarFloat>(op: Op, args: &[T]) -> Result<T, Error> {
    match op.reference_decomposition_src() {
        Some(src) => {
            let e = parse(src).map_err(|pe| Error::BadDecomposition { op, pos: pe.pos })?;
            eval_expr(&e, args)
        }
        None => Err(Error::Unsupported(op)),
    }
}

/// Evaluate a parsed §6.13 decomposition tree with the given operand values.
pub fn eval_expr<T: ScalarFloat>(e: &Expr, inputs: &[T]) -> Result<T, Error> {
    match e {
        Expr::Input(i) => inputs
            .get(*i as usize)
            .copied()
            .ok_or(Error::MissingInput(*i)),
        Expr::Const(c) => Ok(T::from_f64(c.value())),
        Expr::Apply(op, args) => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_expr(a, inputs)?);
            }
            eval_op(*op, &vals)
        }
    }
}

// ---- coverage support -------------------------------------------------------

/// A floor atom the scalar-float path computes directly (arithmetic, select,
/// comparison, rounding, transcendental, binary-math). Excludes the bitwise
/// atoms (integer-only) and the structural atoms (`element_map` … `sort_network`)
/// — both PENDING in this seed.
fn is_float_floor_atom(op: Op) -> bool {
    op.is_primitive_floor()
        && matches!(
            op.family(),
            Family::Arithmetic
                | Family::Select
                | Family::Comparison
                | Family::Rounding
                | Family::Transcendental
                | Family::BinaryMath
        )
}

fn is_refined(op: Op) -> bool {
    matches!(
        op,
        Op::Tanh
            | Op::Sinh
            | Op::Cosh
            | Op::Softplus
            | Op::Silu
            | Op::Mish
            | Op::Expm1
            | Op::Log1p
            | Op::Pow
            | Op::Hypot
            | Op::Ldexp
    )
}

/// Whether the scalar-float path can evaluate `op` — a directly-handled atom, a
/// refined form, or a non-primitive whose whole §6.13 decomposition resolves
/// through float-supported ops. Mirrors [`eval_op`]'s dispatch, so it stays
/// honest about what actually evaluates.
pub fn float_supported(op: Op) -> bool {
    fn go(op: Op, budget: u32) -> bool {
        if budget == 0 {
            return false;
        }
        if is_float_floor_atom(op) || is_refined(op) {
            return true;
        }
        match op.reference_decomposition_src() {
            Some(src) => match parse(src) {
                Ok(e) => expr_ok(&e, budget),
                Err(_) => false,
            },
            None => false,
        }
    }
    fn expr_ok(e: &Expr, budget: u32) -> bool {
        match e {
            Expr::Input(_) | Expr::Const(_) => true,
            Expr::Apply(op, args) => go(*op, budget - 1) && args.iter().all(|a| expr_ok(a, budget)),
        }
    }
    go(op, 32)
}

/// Whether the **tensor-evaluation layer** (float lane) evaluates `op` — the six
/// §6.11 structural atoms, the §6.13 tensor non-primitives that decompose through
/// them, and the window family (`avg_pool`/`max_pool`/`im2col`).
///
/// Float lane (`f16`/`bf16`/`f32`/`f64`); the integer tensor lane covers a subset
/// separately via `tensor_int::int_tensor_supported`.
pub fn tensor_supported(op: Op) -> bool {
    matches!(
        op,
        // the 6 structural floor atoms (§6.11)
        Op::ElementMap
            | Op::Reduce
            | Op::PrefixScan
            | Op::Gather
            | Op::Scatter
            | Op::SortNetwork
        // reductions (§6.13)
            | Op::ReduceMean
            | Op::ReduceNorm2
            | Op::ReduceVar
            | Op::ReduceStd
            | Op::Logsumexp
            | Op::Argmax
            | Op::Any
            | Op::All
        // scans
            | Op::Cumsum
            | Op::Cumprod
            | Op::Cummax
        // normalizations
            | Op::Softmax
            | Op::LogSoftmax
            | Op::RmsNorm
            | Op::LayerNorm
        // contraction
            | Op::Matmul
        // gather/scatter family
            | Op::IndexSelect
            | Op::Embedding
            | Op::ScatterAdd
        // window family
            | Op::AvgPool
            | Op::MaxPool
            | Op::Im2col
    )
}

/// Whether any reference path evaluates `op` (float scalar, integer scalar, or the
/// tensor path). Used by the coverage ledger.
pub fn implemented(op: Op) -> bool {
    // Every §6.18 complex op has a reference kernel (`crate::complex`), evaluable on
    // `c32`/`c64` — so a complex op counts as implemented for the per-op ledger.
    float_supported(op) || crate::int_supported(op) || tensor_supported(op) || op.is_complex()
}

/// Coverage of `(op, dtype)` in this seed: `Done` iff a reference path evaluates
/// `op` on `dtype` — the float scalar/tensor path on the six float lanes
/// (`f16`/`bf16`/`f32`/`f64`/`e4m3`/`e5m2`, the narrow lanes via promote-to-f32),
/// the integer path on the integer dtypes, the `bool` truth-valued lane, or the
/// §6.18 complex path on `c32`/`c64`. `Pending` is the residue of genuinely
/// legal-but-unimplemented edges (e.g. a `bool`-legal op with no backing integer
/// kernel). Illegal cells are `NotApplicable`, never `Pending` — `nextafter` on
/// the narrow floats is a §6.9-0003 legality decline, not a backlog item. Drives
/// the conformance coverage ledger.
pub fn support(op: Op, dtype: Dtype) -> Support {
    if !legality(op, dtype) {
        Support::NotApplicable
    } else if implemented_on(op, dtype) {
        Support::Done
    } else {
        Support::Pending
    }
}

/// Whether `(op, dtype)` is **spec-legal** — i.e. the op has a defined meaning on
/// that dtype's numeric kind. Derived **only** from the spec (never from what is
/// implemented): the op family × numeric kind, with the load-bearing per-op edges.
/// The index-operand `{u32, i32, i64}` restriction (§6.11-0009) is a **separate**
/// operand-role axis and is NOT folded in here.
pub fn legality(op: Op, dtype: Dtype) -> bool {
    // sk4 §6.1-0001: the reserved FP8 variants (`f8e4m3fnuz`/`f8e5m2fnuz`) and the
    // MX scale dtypes (`f8e8m0`/`f8e6m2`) are recognized members of the closed
    // vocabulary but have no element-value compute semantics at this schema
    // version — every op declines on them. This is a typed compute-decline
    // (NotApplicable, never Pending backlog); the recognized-vs-unknown distinction
    // lives in `Dtype::from_token` (`Some` here, `None` for an unknown token).
    if dtype.declines_compute() {
        return false;
    }
    let kind = dtype.numeric_kind();
    // §6.18: the complex-arithmetic family is defined EXACTLY on the complex
    // compute dtypes (`c32`/`c64`), and no real op is defined on a complex dtype —
    // so a complex op and a complex dtype are legal together or not at all.
    if op.is_complex() || kind == NumericKind::Complex {
        return op.is_complex() && kind == NumericKind::Complex;
    }
    // §6.9-0003: nextafter is defined only on the wide floats (stepping in a
    // promoted f32 gives the wrong neighbor in the narrow/FP8 lattice).
    if op == Op::Nextafter {
        return matches!(dtype, Dtype::F32 | Dtype::F64);
    }
    match kind {
        // Every non-bitwise op is defined on a float dtype; bitwise is integer-only
        // (§6.10-0001).
        NumericKind::Float => op.family() != Family::Bitwise,
        NumericKind::Int | NumericKind::Uint => int_legal(op),
        NumericKind::Bool => bool_legal(op),
        NumericKind::Complex => false, // handled above
    }
}

/// Whether `op` is legal on an **integer** dtype: everything except the ops that
/// fundamentally need `div` / `sqrt` / a transcendental (float-only, §6.4-0002 /
/// §6.7 / §6.8). Bitwise, comparisons, select, min/max, logical, `sign`, `sqr`,
/// the structural atoms, `matmul`, `argmax`/`any`/`all`, the scans, and the
/// gather/scatter family are all integer-meaningful.
fn int_legal(op: Op) -> bool {
    let float_only = matches!(
        op.family(),
        Family::Rounding | Family::Transcendental | Family::BinaryMath | Family::Normalization
    ) || matches!(
        op,
        Op::Div
            | Op::Recip
            | Op::ReduceMean
            | Op::ReduceVar
            | Op::ReduceStd
            | Op::ReduceNorm2
            | Op::Logsumexp
            | Op::AvgPool
            | Op::Sigmoid
            | Op::Silu
            | Op::Softplus
            | Op::Mish
            | Op::Gelu
            | Op::GeluTanh
    );
    !float_only
}

/// Whether `op` is legal on the truth-valued **bool** dtype (§6.2-0006): the
/// logical family, equality, raw-bit `select`, order-preserving min/max, and the
/// data-movement / `{0,1}`-preserving structural ops. Arithmetic, bitwise
/// (§6.10-0001), rounding, transcendental, ordered comparisons, and the
/// sum/prod-bearing reductions/scans are NOT applicable.
pub(crate) fn bool_legal(op: Op) -> bool {
    matches!(
        op,
        // logical predicates + equality + raw-bit select
        Op::LogicalAnd | Op::LogicalOr | Op::LogicalNot
            | Op::CmpEq | Op::CmpNe | Op::Select
        // order-preserving on {0,1}
            | Op::MaxProp | Op::MinProp
        // structural atoms (data movement + {0,1}-preserving folds; the runtime
        // gate rejects sum/prod monoids and scatter-add)
            | Op::ElementMap | Op::Reduce | Op::PrefixScan | Op::Gather | Op::Scatter | Op::SortNetwork
        // reduction/scan non-primitives that stay in {0,1}
            | Op::Any | Op::All | Op::Cummax
        // gather/scatter + shape data movement
            | Op::IndexSelect | Op::Embedding | Op::Im2col
    )
}

/// Whether a reference path evaluates `(op, dtype)` in this seed — the concrete
/// dispatch behind `Done`. (Legality is orthogonal and comes first in [`support`].)
fn implemented_on(op: Op, dtype: Dtype) -> bool {
    let float_ok = match dtype {
        Dtype::F32 | Dtype::F64 => float_supported(op) || tensor_supported(op),
        // narrow floats + FP8: same coverage as the wide floats, minus nextafter
        // (all compute via promotion to f32).
        Dtype::F16 | Dtype::Bf16 | Dtype::F8e4m3fn | Dtype::F8e5m2 => {
            (float_supported(op) && op != Op::Nextafter) || tensor_supported(op)
        }
        _ => false,
    };
    let int_ok = crate::scalar_int::int_spec(dtype).is_some()
        && (crate::int_supported(op) || crate::tensor_int::int_tensor_supported(op));
    // the bool truth-valued lane (scalar via eval_bool_op; tensor via the integer
    // kernels over {0,1}).
    let bool_ok = dtype == Dtype::Bool && crate::boolean::bool_supported(op);
    // §6.18: every complex op has a reference kernel (crate::complex) on both
    // complex dtypes, computing in the f32/f64 component lane (§6.18-0015).
    let complex_ok = dtype.is_complex() && op.is_complex();
    float_ok || int_ok || bool_ok || complex_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32() {
        assert_eq!(eval_op(Op::Add, &[1.0f32, 2.0]).unwrap(), 3.0);
    }

    #[test]
    fn select_is_raw_bit_signed_zero() {
        // -0.0 condition is false → pick b; the chosen arm is moved unchanged.
        let r = eval_op(Op::Select, &[-0.0f64, 1.0, 2.0]).unwrap();
        assert_eq!(r, 2.0);
    }

    #[test]
    fn select_moves_arm_bit_for_bit() {
        // §6.5: raw-bit select performs NO arithmetic on the chosen arm — it moves
        // the operand bit-for-bit. A value-only check (`== 2.0`) cannot see this,
        // because `-0.0 == +0.0` and `NaN != NaN`. Prove bit-identity via
        // `.to_bits()` for the two payloads arithmetic would clobber: a signed zero
        // (arithmetic could canonicalize the sign) and a NaN with a specific,
        // non-canonical mantissa payload (arithmetic could quiet/canonicalize it).
        let neg_zero = -0.0f64;
        let nan_payload = f64::from_bits(0x7FF8_0000_ABCD_1234);
        assert_ne!(neg_zero.to_bits(), 0.0f64.to_bits()); // sanity: -0.0 ≠ +0.0 in bits

        // TRUE branch (cond = 1.0, truthy) → arm b selected, moved unchanged.
        let r = eval_op(Op::Select, &[1.0f64, neg_zero, 9.0]).unwrap();
        assert_eq!(r.to_bits(), neg_zero.to_bits());
        let r = eval_op(Op::Select, &[1.0f64, nan_payload, 9.0]).unwrap();
        assert_eq!(r.to_bits(), nan_payload.to_bits());

        // FALSE branch (cond = +0.0, not truthy) → arm c selected, moved unchanged.
        let r = eval_op(Op::Select, &[0.0f64, 9.0, neg_zero]).unwrap();
        assert_eq!(r.to_bits(), neg_zero.to_bits());
        let r = eval_op(Op::Select, &[0.0f64, 9.0, nan_payload]).unwrap();
        assert_eq!(r.to_bits(), nan_payload.to_bits());
    }

    #[test]
    fn cmp_ne_nan_is_true() {
        let nan = f64::NAN;
        assert_eq!(eval_op(Op::CmpNe, &[nan, nan]).unwrap(), 1.0);
        assert_eq!(eval_op(Op::CmpEq, &[nan, nan]).unwrap(), 0.0);
    }

    #[test]
    fn gelu_resolves_and_is_finite() {
        // Non-primitive resolved through erf + arithmetic floor atoms.
        let y = eval_op(Op::Gelu, &[1.0f64]).unwrap();
        assert!(
            y.is_finite() && y > 0.8 && y < 0.9,
            "gelu(1) ≈ 0.8413, got {y}"
        );
    }

    #[test]
    fn tanh_is_overflow_safe() {
        // Literal decomposition (e^x - e^-x)/(e^x + e^-x) would be inf/inf = NaN.
        assert_eq!(eval_op(Op::Tanh, &[100.0f64]).unwrap(), 1.0);
    }

    #[test]
    fn silu_resolves_through_sigmoid() {
        let y = eval_op(Op::Silu, &[1.0f64]).unwrap();
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((y - 0.7310585786).abs() < 1e-6, "got {y}");
    }
}
