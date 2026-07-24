//! Integer-path conformance corpus — the pinned integer semantics of KISS-Ops
//! §6.2 / §6.4 / §6.10, with KISS-Conform `test_ops_*` names.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_int_op, Error};

fn ev(op: Op, d: Dtype, args: &[i128]) -> i128 {
    eval_int_op(op, d, args).unwrap_or_else(|e| panic!("{op:?}/{d:?} failed: {e:?}"))
}

#[test]
fn test_ops_int_wrapping() {
    // KISS-OPS-6.2-0002: add/sub/mul are wrapping two's-complement.
    assert_eq!(ev(Op::Add, Dtype::S8, &[127, 1]), -128);
    assert_eq!(ev(Op::Mul, Dtype::U8, &[16, 16]), 0); // 256 mod 256
    assert_eq!(ev(Op::Sub, Dtype::U8, &[0, 1]), 255);
}

#[test]
fn test_ops_int_neg_abs_wrap() {
    // KISS-OPS-6.4-0005: neg/abs of INT_MIN stay INT_MIN, no UB, no saturation.
    assert_eq!(
        ev(Op::Neg, Dtype::I32, &[i32::MIN as i128]),
        i32::MIN as i128
    );
    assert_eq!(
        ev(Op::Abs, Dtype::I32, &[i32::MIN as i128]),
        i32::MIN as i128
    );
    assert_eq!(ev(Op::Neg, Dtype::I32, &[5]), -5);
}

#[test]
fn test_ops_bitwise_integer_only() {
    // KISS-OPS-6.10-0001: bitwise atoms reject a float operand (typed decline).
    assert!(matches!(
        eval_int_op(Op::BitAnd, Dtype::F32, &[1, 1]),
        Err(Error::UnsupportedDtype(_))
    ));
    // and compute correctly on integers.
    assert_eq!(ev(Op::BitXor, Dtype::U8, &[0b1100, 0b1010]), 0b0110);
    assert_eq!(ev(Op::Popcount, Dtype::U16, &[0xFFFF]), 16);
}

#[test]
fn test_ops_u32_ordinary_dtype() {
    // KISS-OPS-6.2-0007: u32 is an ordinary unsigned integer dtype.
    assert_eq!(ev(Op::Add, Dtype::U32, &[u32::MAX as i128, 1]), 0);
    assert_eq!(ev(Op::CmpGt, Dtype::U32, &[5, 3]), 1);
}
