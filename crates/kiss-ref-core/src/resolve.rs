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

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::{parse, Expr};
use kiss_ops_vocab::{Family, Op};

use crate::scalar::{mish_stable, silu_stable, softplus_stable, ScalarFloat};
use crate::{Error, Support};

fn arity<T>(op: Op, args: &[T], n: usize) -> Result<(), Error> {
    if args.len() == n {
        Ok(())
    } else {
        Err(Error::Arity { op, expected: n, got: args.len() })
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
            Ok(if args[0].is_truthy() { args[1] } else { args[2] })
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
        Op::Nextafter => bin(op, args, |a, b| a.nextafter(b)),

        // refined non-primitives (§6.13-0003): computed directly because the
        // literal decomposition overflows to NaN while the true value is finite.
        Op::Tanh => un(op, args, |a| a.tanh()),
        Op::Sinh => un(op, args, |a| a.sinh()),
        Op::Cosh => un(op, args, |a| a.cosh()),
        Op::Softplus => un(op, args, |a| softplus_stable(a)),
        Op::Silu => un(op, args, |a| silu_stable(a)),
        Op::Mish => un(op, args, |a| mish_stable(a)),

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
        Op::Tanh | Op::Sinh | Op::Cosh | Op::Softplus | Op::Silu | Op::Mish
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

/// Coverage of `(op, dtype)` in this seed: `Done` iff the dtype is a supported
/// float dtype (`f32`/`f64`) and the op resolves through the float path;
/// `Pending` otherwise. Drives the conformance coverage ledger.
pub fn support(op: Op, dtype: Dtype) -> Support {
    if matches!(dtype, Dtype::F32 | Dtype::F64) && float_supported(op) {
        Support::Done
    } else {
        Support::Pending
    }
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
    fn cmp_ne_nan_is_true() {
        let nan = f64::NAN;
        assert_eq!(eval_op(Op::CmpNe, &[nan, nan]).unwrap(), 1.0);
        assert_eq!(eval_op(Op::CmpEq, &[nan, nan]).unwrap(), 0.0);
    }

    #[test]
    fn gelu_resolves_and_is_finite() {
        // Non-primitive resolved through erf + arithmetic floor atoms.
        let y = eval_op(Op::Gelu, &[1.0f64]).unwrap();
        assert!(y.is_finite() && y > 0.8 && y < 0.9, "gelu(1) ≈ 0.8413, got {y}");
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
