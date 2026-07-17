//! Scalar reference implementations of the **integer** floor atoms (KISS-Ops
//! §6.2-0002 wrapping arithmetic, §6.6 comparisons, §6.10 bitwise) plus the few
//! integer-valued non-primitives (`sign`, `max_prop`/`min_prop`, the `logical_*`
//! family).
//!
//! Every integer dtype is computed uniformly in `i128` and then **wrapped** to
//! the dtype's bit width in two's-complement (KISS-OPS-6.2-0002 / §6.4-0005:
//! `neg(INT_MIN) = INT_MIN`, never UB, never saturating). Bitwise ops operate on
//! the width-masked bit pattern; signed `shr` is arithmetic, unsigned `shr` is
//! logical (§6.10).

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

use crate::Error;

/// `(bit width, signed)` for an integer-kind dtype (including the packed
/// sub-byte `s4`/`u4`/`b1`), or `None` for a non-integer dtype. `bool` is
/// excluded — it is its own kind, produced by predicates, not an arithmetic
/// integer dtype here.
pub fn int_spec(d: Dtype) -> Option<(u32, bool)> {
    Some(match d {
        Dtype::S8 => (8, true),
        Dtype::S16 => (16, true),
        Dtype::I32 => (32, true),
        Dtype::I64 => (64, true),
        Dtype::S4 => (4, true),
        Dtype::U8 => (8, false),
        Dtype::U16 => (16, false),
        Dtype::U32 => (32, false),
        Dtype::U64 => (64, false),
        Dtype::U4 => (4, false),
        Dtype::B1 => (1, false),
        _ => return None,
    })
}

fn mask(bits: u32) -> i128 {
    if bits >= 128 {
        -1
    } else {
        (1i128 << bits) - 1
    }
}

/// Reinterpret an `i128` computation result as a value of `(bits, signed)` via
/// two's-complement wrapping + sign extension.
fn wrap(v: i128, bits: u32, signed: bool) -> i128 {
    if bits >= 128 {
        return v;
    }
    let m = v & mask(bits);
    if signed && ((m >> (bits - 1)) & 1) == 1 {
        m - (1i128 << bits)
    } else {
        m
    }
}

/// The width-masked bit pattern (unsigned low `bits` bits).
fn pattern(v: i128, bits: u32) -> i128 {
    v & mask(bits)
}

/// Whether `op` is evaluated by the integer path.
pub fn int_supported(op: Op) -> bool {
    matches!(
        op,
        // arithmetic atoms (div excluded — float-only, §6.4-0002)
        Op::Add | Op::Sub | Op::Mul | Op::Neg | Op::Abs
        // raw-bit select
        | Op::Select
        // comparisons
        | Op::CmpEq | Op::CmpNe | Op::CmpLt | Op::CmpLe | Op::CmpGt | Op::CmpGe
        // bitwise atoms (§6.10)
        | Op::BitAnd | Op::BitOr | Op::BitXor | Op::BitNot
        | Op::Shl | Op::Shr | Op::Popcount | Op::Clz | Op::Ctz
        // integer-valued non-primitives
        | Op::Sign | Op::MaxProp | Op::MinProp
        | Op::LogicalAnd | Op::LogicalOr | Op::LogicalNot
    )
}

fn arity(op: Op, args: &[i128], n: usize) -> Result<(), Error> {
    if args.len() == n {
        Ok(())
    } else {
        Err(Error::Arity { op, expected: n, got: args.len() })
    }
}

/// Evaluate an integer floor atom (or integer-valued non-primitive) on `i128`
/// operands interpreted in `dtype`, returning the wrapped `i128` result. Never
/// panics — every failure is an [`Error`].
pub fn eval_int_op(op: Op, dtype: Dtype, args: &[i128]) -> Result<i128, Error> {
    let (bits, signed) = int_spec(dtype).ok_or(Error::UnsupportedDtype(dtype))?;
    let w = |v: i128| wrap(v, bits, signed);
    let b = |v: bool| if v { 1 } else { 0 };

    match op {
        Op::Add => {
            arity(op, args, 2)?;
            Ok(w(args[0].wrapping_add(args[1])))
        }
        Op::Sub => {
            arity(op, args, 2)?;
            Ok(w(args[0].wrapping_sub(args[1])))
        }
        Op::Mul => {
            arity(op, args, 2)?;
            Ok(w(args[0].wrapping_mul(args[1])))
        }
        Op::Neg => {
            arity(op, args, 1)?;
            Ok(w(args[0].wrapping_neg()))
        }
        Op::Abs => {
            arity(op, args, 1)?;
            // wrapping abs: abs(INT_MIN) = INT_MIN (§6.4-0005).
            Ok(w(args[0].wrapping_abs()))
        }
        Op::Select => {
            arity(op, args, 3)?;
            Ok(if args[0] != 0 { args[1] } else { args[2] })
        }

        Op::CmpEq => {
            arity(op, args, 2)?;
            Ok(b(args[0] == args[1]))
        }
        Op::CmpNe => {
            arity(op, args, 2)?;
            Ok(b(args[0] != args[1]))
        }
        Op::CmpLt => {
            arity(op, args, 2)?;
            Ok(b(args[0] < args[1]))
        }
        Op::CmpLe => {
            arity(op, args, 2)?;
            Ok(b(args[0] <= args[1]))
        }
        Op::CmpGt => {
            arity(op, args, 2)?;
            Ok(b(args[0] > args[1]))
        }
        Op::CmpGe => {
            arity(op, args, 2)?;
            Ok(b(args[0] >= args[1]))
        }

        Op::BitAnd => {
            arity(op, args, 2)?;
            Ok(w(pattern(args[0], bits) & pattern(args[1], bits)))
        }
        Op::BitOr => {
            arity(op, args, 2)?;
            Ok(w(pattern(args[0], bits) | pattern(args[1], bits)))
        }
        Op::BitXor => {
            arity(op, args, 2)?;
            Ok(w(pattern(args[0], bits) ^ pattern(args[1], bits)))
        }
        Op::BitNot => {
            arity(op, args, 1)?;
            Ok(w((!pattern(args[0], bits)) & mask(bits)))
        }
        Op::Shl => {
            arity(op, args, 2)?;
            let sh = args[1];
            if sh < 0 || sh >= bits as i128 {
                Ok(0)
            } else {
                Ok(w(pattern(args[0], bits) << sh))
            }
        }
        Op::Shr => {
            arity(op, args, 2)?;
            let sh = args[1];
            if sh < 0 || sh >= bits as i128 {
                Ok(0)
            } else if signed {
                // arithmetic shift on the sign-extended value
                Ok(w(args[0] >> sh))
            } else {
                // logical shift on the bit pattern
                Ok(w(pattern(args[0], bits) >> sh))
            }
        }
        Op::Popcount => {
            arity(op, args, 1)?;
            Ok((pattern(args[0], bits) as u128).count_ones() as i128)
        }
        Op::Clz => {
            arity(op, args, 1)?;
            let p = pattern(args[0], bits) as u128;
            Ok(if p == 0 {
                bits as i128
            } else {
                (bits - (128 - p.leading_zeros())) as i128
            })
        }
        Op::Ctz => {
            arity(op, args, 1)?;
            let p = pattern(args[0], bits) as u128;
            Ok(if p == 0 {
                bits as i128
            } else {
                (p.trailing_zeros() as i128).min(bits as i128)
            })
        }

        // integer-valued non-primitives (no NaN, so the NaN guards in the §6.13
        // decompositions are inert).
        Op::Sign => {
            arity(op, args, 1)?;
            Ok(w(b(args[0] > 0) - b(args[0] < 0)))
        }
        Op::MaxProp => {
            arity(op, args, 2)?;
            Ok(if args[0] >= args[1] { args[0] } else { args[1] })
        }
        Op::MinProp => {
            arity(op, args, 2)?;
            Ok(if args[0] <= args[1] { args[0] } else { args[1] })
        }
        Op::LogicalAnd => {
            arity(op, args, 2)?;
            Ok(b(args[0] != 0 && args[1] != 0))
        }
        Op::LogicalOr => {
            arity(op, args, 2)?;
            Ok(b(args[0] != 0 || args[1] != 0))
        }
        Op::LogicalNot => {
            arity(op, args, 1)?;
            Ok(b(args[0] == 0))
        }

        _ => Err(Error::Unsupported(op)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_add_wraps_two_complement() {
        // s8: 127 + 1 wraps to -128 (§6.2-0002).
        assert_eq!(eval_int_op(Op::Add, Dtype::S8, &[127, 1]).unwrap(), -128);
        // u8: 255 + 1 wraps to 0.
        assert_eq!(eval_int_op(Op::Add, Dtype::U8, &[255, 1]).unwrap(), 0);
    }

    #[test]
    fn int_neg_abs_wrap_at_min() {
        // §6.4-0005: neg(INT_MIN) = abs(INT_MIN) = INT_MIN.
        assert_eq!(eval_int_op(Op::Neg, Dtype::S8, &[-128]).unwrap(), -128);
        assert_eq!(eval_int_op(Op::Abs, Dtype::S8, &[-128]).unwrap(), -128);
        assert_eq!(eval_int_op(Op::Abs, Dtype::S8, &[-5]).unwrap(), 5);
    }

    #[test]
    fn int_bitwise_and_not() {
        assert_eq!(eval_int_op(Op::BitAnd, Dtype::U8, &[0b1100, 0b1010]).unwrap(), 0b1000);
        // u8 bit_not(0) = 255.
        assert_eq!(eval_int_op(Op::BitNot, Dtype::U8, &[0]).unwrap(), 255);
        // s8 bit_not(0) = -1 (all ones sign-extended).
        assert_eq!(eval_int_op(Op::BitNot, Dtype::S8, &[0]).unwrap(), -1);
    }

    #[test]
    fn int_shifts_signed_vs_unsigned() {
        // u8 logical shr.
        assert_eq!(eval_int_op(Op::Shr, Dtype::U8, &[0b1000_0000, 1]).unwrap(), 0b0100_0000);
        // s8 arithmetic shr keeps the sign.
        assert_eq!(eval_int_op(Op::Shr, Dtype::S8, &[-8, 1]).unwrap(), -4);
        // shl wraps within width.
        assert_eq!(eval_int_op(Op::Shl, Dtype::U8, &[1, 7]).unwrap(), 128);
        // out-of-range shift → 0 (deterministic reference choice).
        assert_eq!(eval_int_op(Op::Shl, Dtype::U8, &[1, 8]).unwrap(), 0);
    }

    #[test]
    fn int_bit_counts() {
        assert_eq!(eval_int_op(Op::Popcount, Dtype::U8, &[0b1011]).unwrap(), 3);
        // clz over the 8-bit width.
        assert_eq!(eval_int_op(Op::Clz, Dtype::U8, &[1]).unwrap(), 7);
        assert_eq!(eval_int_op(Op::Clz, Dtype::U8, &[0]).unwrap(), 8);
        assert_eq!(eval_int_op(Op::Ctz, Dtype::U8, &[0b1000]).unwrap(), 3);
        assert_eq!(eval_int_op(Op::Ctz, Dtype::U8, &[0]).unwrap(), 8);
    }

    #[test]
    fn int_sign_and_logical() {
        assert_eq!(eval_int_op(Op::Sign, Dtype::S8, &[-7]).unwrap(), -1);
        assert_eq!(eval_int_op(Op::Sign, Dtype::S8, &[0]).unwrap(), 0);
        assert_eq!(eval_int_op(Op::LogicalAnd, Dtype::U8, &[5, 0]).unwrap(), 0);
        assert_eq!(eval_int_op(Op::LogicalNot, Dtype::U8, &[0]).unwrap(), 1);
    }
}
