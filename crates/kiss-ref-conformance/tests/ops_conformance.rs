//! Conformance corpus — the pinned numeric semantics of KISS-Ops §6, with test
//! names mirroring KISS-Conform's `test_ops_*` (KISS-Ops §9.1 clause→test
//! traceability). Each test cites the clause it exercises.

use kiss_ops_vocab::Op;
use kiss_ref_core::eval_op;

// Evaluate a floor/non-primitive op on f64 args, unwrapping the reference result.
fn ev(op: Op, args: &[f64]) -> f64 {
    eval_op(op, args).unwrap_or_else(|e| panic!("{op:?} eval failed: {e:?}"))
}

/// 4-ULP-relative tolerance around `expected` (§6.8 declared-ULP band). One ULP
/// near a normalized `x` is ≈ `|x| * f64::EPSILON`.
fn within_ulp(got: f64, expected: f64, ulps: f64) -> bool {
    (got - expected).abs() <= ulps * expected.abs().max(f64::MIN_POSITIVE) * f64::EPSILON
}

// ---- §6.2 shared conventions ------------------------------------------------

#[test]
fn test_ops_default_nan_propagation() {
    // KISS-OPS-6.2-0003: a NaN in a contributing operand propagates.
    let nan = f64::NAN;
    assert!(ev(Op::Add, &[nan, 1.0]).is_nan());
    assert!(ev(Op::Sub, &[1.0, nan]).is_nan());
    assert!(ev(Op::Mul, &[nan, 0.0]).is_nan());
}

#[test]
fn test_ops_signed_zero_preserved() {
    // KISS-OPS-6.2-0004: neg(-0.0) and abs(-0.0) are the pinned +0.0 exceptions.
    assert!(ev(Op::Neg, &[-0.0]).is_sign_positive());
    assert!(ev(Op::Abs, &[-0.0]).is_sign_positive());
    // trunc preserves the sign of a zero operand (§6.7-0002).
    assert!(ev(Op::Trunc, &[-0.0]).is_sign_negative());
}

// ---- §6.4 arithmetic atoms --------------------------------------------------

#[test]
fn test_ops_add_sub_mul() {
    // KISS-OPS-6.4-0001.
    assert_eq!(ev(Op::Add, &[2.0, 3.0]), 5.0);
    assert_eq!(ev(Op::Sub, &[2.0, 3.0]), -1.0);
    assert_eq!(ev(Op::Mul, &[2.0, 3.0]), 6.0);
}

#[test]
fn test_ops_div_float() {
    // KISS-OPS-6.4-0002: div is IEEE on floats.
    assert_eq!(ev(Op::Div, &[1.0, 4.0]), 0.25);
    assert!(ev(Op::Div, &[1.0, 0.0]).is_infinite()); // IEEE, not an error
}

#[test]
fn test_ops_neg() {
    // KISS-OPS-6.4-0003.
    assert_eq!(ev(Op::Neg, &[3.0]), -3.0);
    assert!(ev(Op::Neg, &[f64::NAN]).is_nan());
}

#[test]
fn test_ops_abs_raw_bit() {
    // KISS-OPS-6.4-0004: clears the sign bit; NaN stays NaN.
    assert_eq!(ev(Op::Abs, &[-3.0]), 3.0);
    assert!(ev(Op::Abs, &[f64::NAN]).is_nan());
}

// ---- §6.5 raw-bit select ----------------------------------------------------

#[test]
fn test_ops_select_order() {
    // KISS-OPS-6.5-0001: (cond, a, b) → a when cond != 0, else b.
    assert_eq!(ev(Op::Select, &[1.0, 10.0, 20.0]), 10.0);
    assert_eq!(ev(Op::Select, &[0.0, 10.0, 20.0]), 20.0);
}

#[test]
fn test_ops_select_raw_bit() {
    // KISS-OPS-6.5-0002: the chosen arm is moved unchanged — its signed zero is
    // preserved (a mask-multiply would perturb it).
    assert!(ev(Op::Select, &[1.0, -0.0, 5.0]).is_sign_negative());
}

#[test]
fn test_ops_select_cond_zero_nan() {
    // KISS-OPS-6.5-0003: -0.0 cond is false, any NaN cond is true.
    assert_eq!(ev(Op::Select, &[-0.0, 10.0, 20.0]), 20.0);
    assert_eq!(ev(Op::Select, &[f64::NAN, 10.0, 20.0]), 10.0);
}

// ---- §6.6 comparison atoms --------------------------------------------------

#[test]
fn test_ops_compare_predicates() {
    // KISS-OPS-6.6-0001: 1/0 results.
    assert_eq!(ev(Op::CmpLt, &[1.0, 2.0]), 1.0);
    assert_eq!(ev(Op::CmpGe, &[1.0, 2.0]), 0.0);
}

#[test]
fn test_ops_compare_nan_false() {
    // KISS-OPS-6.6-0002: ordered comparisons are false on any NaN operand.
    let nan = f64::NAN;
    for op in [Op::CmpEq, Op::CmpLt, Op::CmpLe, Op::CmpGt, Op::CmpGe] {
        assert_eq!(ev(op, &[nan, 1.0]), 0.0, "{op:?}(NaN,1) must be 0");
    }
}

#[test]
fn test_ops_cmp_ne_nan_true() {
    // KISS-OPS-6.6-0003: cmp_ne is true on NaN — cmp_ne(x,x) is the isnan predicate.
    assert_eq!(ev(Op::CmpNe, &[f64::NAN, f64::NAN]), 1.0);
    assert_eq!(ev(Op::CmpNe, &[1.0, 1.0]), 0.0);
}

#[test]
fn test_ops_compare_signed_zero() {
    // KISS-OPS-6.6-0004: -0.0 compares equal to +0.0.
    assert_eq!(ev(Op::CmpEq, &[-0.0, 0.0]), 1.0);
    assert_eq!(ev(Op::CmpLe, &[-0.0, 0.0]), 1.0);
    assert_eq!(ev(Op::CmpGe, &[-0.0, 0.0]), 1.0);
}

// ---- §6.7 rounding atoms ----------------------------------------------------

#[test]
fn test_ops_rounding_directions() {
    // KISS-OPS-6.7-0001.
    assert_eq!(ev(Op::Floor, &[2.7]), 2.0);
    assert_eq!(ev(Op::Ceil, &[2.1]), 3.0);
    assert_eq!(ev(Op::Trunc, &[-2.7]), -2.0);
    // round_even: ties to even.
    assert_eq!(ev(Op::RoundEven, &[0.5]), 0.0);
    assert_eq!(ev(Op::RoundEven, &[1.5]), 2.0);
    assert_eq!(ev(Op::RoundEven, &[2.5]), 2.0);
    assert_eq!(ev(Op::RoundEven, &[3.5]), 4.0);
    assert_eq!(ev(Op::RoundEven, &[-2.5]), -2.0);
}

// ---- §6.8 transcendental atoms (declared-ULP) -------------------------------

#[test]
fn test_ops_transcendental_declared_ulp() {
    // KISS-OPS-6.8-0001: within the declared ULP ceiling (erf ≤ 4, exp/log ≤ 4,
    // sqrt correctly-rounded/≤2).
    assert!(within_ulp(ev(Op::Erf, &[1.0]), 0.8427007929497149, 4.0));
    assert!(within_ulp(ev(Op::Exp, &[1.0]), std::f64::consts::E, 4.0));
    assert!(within_ulp(ev(Op::Log, &[std::f64::consts::E]), 1.0, 4.0));
    assert!(within_ulp(ev(Op::Sqrt, &[2.0]), std::f64::consts::SQRT_2, 2.0));
    assert_eq!(ev(Op::Exp, &[0.0]), 1.0);
}

// ---- §6.13 reference decompositions (non-primitives) ------------------------

#[test]
fn test_ops_reference_decompositions() {
    // KISS-OPS-6.13-0001: a non-primitive equals its reference decomposition.
    assert_eq!(ev(Op::Sqr, &[3.0]), 9.0);
    assert_eq!(ev(Op::Recip, &[4.0]), 0.25);
    assert!(within_ulp(ev(Op::Sigmoid, &[0.0]), 0.5, 1.0));
    assert_eq!(ev(Op::Relu, &[-2.0]), 0.0);
    assert_eq!(ev(Op::Relu, &[3.0]), 3.0);
    assert!(within_ulp(ev(Op::Erfc, &[0.0]), 1.0, 1.0));
    // gelu(1) = 0.5*(1+erf(1/sqrt2)) ≈ 0.8413447460685429
    assert!(within_ulp(ev(Op::Gelu, &[1.0]), 0.8413447460685429, 8.0));
    // pow via exp(b*log(a)) for a>0.
    assert!(within_ulp(ev(Op::Pow, &[2.0, 3.0]), 8.0, 8.0));
}

#[test]
fn test_ops_decomposition_accuracy_refinement() {
    // KISS-OPS-6.13-0003: the MUST-refine ops are overflow-safe where the literal
    // decomposition would produce NaN/inf.
    assert_eq!(ev(Op::Tanh, &[1000.0]), 1.0); // not inf/inf = NaN
    assert!(within_ulp(ev(Op::Softplus, &[1000.0]), 1000.0, 8.0)); // not log(inf) = inf
    assert!(ev(Op::Softplus, &[-1000.0]).abs() < 1e-300); // ≈ 0, finite
    assert!(ev(Op::Mish, &[1000.0]).is_finite());
}

#[test]
fn test_ops_min_max_prop_nan() {
    // §6.13 max_prop / min_prop propagate NaN (vs the IEEE fmax/fmin forms).
    assert!(ev(Op::MaxProp, &[f64::NAN, 1.0]).is_nan());
    assert_eq!(ev(Op::MaxProp, &[2.0, 1.0]), 2.0);
    // fmax_ieee returns the non-NaN operand.
    assert_eq!(ev(Op::FmaxIeee, &[f64::NAN, 1.0]), 1.0);
}
