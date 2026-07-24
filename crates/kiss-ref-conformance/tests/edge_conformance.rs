//! Broadened edge corpus — inf/subnormal arithmetic, `nextafter`, `copysign`,
//! and narrow-float (f16/bf16) compute — the cases the first corpus skipped.

use half::{bf16, f16};
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_op, Error};

fn ev(op: Op, args: &[f64]) -> f64 {
    eval_op(op, args).unwrap_or_else(|e| panic!("{op:?} failed: {e:?}"))
}

// ---- infinities (IEEE, §6.2-0001) -------------------------------------------

#[test]
fn test_ops_inf_arithmetic() {
    let inf = f64::INFINITY;
    assert!(ev(Op::Add, &[inf, 1.0]).is_infinite());
    assert!(ev(Op::Sub, &[inf, inf]).is_nan()); // inf - inf = NaN
    assert!(ev(Op::Mul, &[inf, 0.0]).is_nan()); // inf * 0 = NaN
    assert_eq!(ev(Op::Div, &[1.0, inf]), 0.0); // 1 / inf = 0
    assert!(ev(Op::Div, &[inf, inf]).is_nan()); // inf / inf = NaN
    assert!(ev(Op::Div, &[1.0, 0.0]).is_infinite()); // 1 / 0 = inf
}

// ---- subnormals -------------------------------------------------------------

#[test]
fn test_ops_subnormal_arithmetic() {
    let tiny = f64::from_bits(1); // smallest positive subnormal
    assert!(tiny.is_sign_positive() && tiny != 0.0);
    // Two subnormals add exactly.
    assert_eq!(ev(Op::Add, &[tiny, tiny]), f64::from_bits(2));
    // abs / neg preserve subnormal magnitude and flip sign.
    assert_eq!(ev(Op::Abs, &[-tiny]), tiny);
    assert!(ev(Op::Neg, &[tiny]).is_sign_negative());
    // Underflow of a normal * small toward a subnormal stays finite (>= 0).
    assert!(ev(Op::Mul, &[f64::MIN_POSITIVE, 0.5]) >= 0.0);
}

// ---- §6.9 binary-math atoms -------------------------------------------------

#[test]
fn test_ops_nextafter_own_lattice() {
    // KISS-OPS-6.9-0003: steps in the dtype's own lattice.
    let up = ev(Op::Nextafter, &[1.0, 2.0]);
    assert_eq!(up, f64::from_bits(1.0f64.to_bits() + 1));
    let down = ev(Op::Nextafter, &[1.0, 0.0]);
    assert_eq!(down, f64::from_bits(1.0f64.to_bits() - 1));
    // From zero toward +1 → smallest positive subnormal.
    assert_eq!(ev(Op::Nextafter, &[0.0, 1.0]), f64::from_bits(1));
}

#[test]
fn test_ops_nextafter_declines_narrow_floats() {
    // KISS-OPS-6.9-0003: nextafter rejects f16/bf16 with a typed decline.
    assert!(matches!(
        eval_op(Op::Nextafter, &[f16::from_f32(1.0), f16::from_f32(2.0)]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        eval_op(Op::Nextafter, &[bf16::from_f32(1.0), bf16::from_f32(2.0)]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn test_ops_copysign_raw_bit() {
    // KISS-OPS-6.9-0002: magnitude of a, sign bit of b (raw-bit).
    assert_eq!(ev(Op::Copysign, &[3.0, -1.0]), -3.0);
    assert_eq!(ev(Op::Copysign, &[-5.0, 1.0]), 5.0);
    // sign of a negative zero is carried.
    assert!(ev(Op::Copysign, &[3.0, -0.0]).is_sign_negative());
}

// ---- narrow-float compute (f16 / bf16) --------------------------------------

#[test]
fn test_ops_f16_computes_in_dtype() {
    // Elementwise ops compute in f16 and round to f16.
    let a = f16::from_f32(1.0);
    let b = f16::from_f32(2.0);
    assert_eq!(eval_op(Op::Add, &[a, b]).unwrap(), f16::from_f32(3.0));
    assert_eq!(eval_op(Op::Mul, &[a, b]).unwrap(), f16::from_f32(2.0));
    // A non-primitive resolves through the floor in f16 and stays finite.
    let g = eval_op(Op::Gelu, &[f16::from_f32(1.0)]).unwrap().to_f32();
    assert!(
        g.is_finite() && (g - 0.8413).abs() < 0.05,
        "gelu(1) in f16 ≈ 0.84, got {g}"
    );
}

#[test]
fn test_ops_bf16_computes_in_dtype() {
    let a = bf16::from_f32(1.5);
    let b = bf16::from_f32(2.5);
    assert_eq!(eval_op(Op::Add, &[a, b]).unwrap(), bf16::from_f32(4.0));
    // sigmoid(0) = 0.5 exactly representable in bf16.
    assert_eq!(
        eval_op(Op::Sigmoid, &[bf16::from_f32(0.0)]).unwrap(),
        bf16::from_f32(0.5)
    );
}
