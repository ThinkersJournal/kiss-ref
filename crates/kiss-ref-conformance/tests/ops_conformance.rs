//! Conformance corpus — the pinned numeric semantics of KISS-Ops §6, with test
//! names mirroring KISS-Conform's `test_ops_*` (KISS-Ops §9.1 clause→test
//! traceability). Each test cites the clause it exercises.
//!
//! Comparisons use the reference's own sign-magnitude ULP metric
//! ([`kiss_ref_core::ulp_distance_f64`]) — faithful near zero and across the
//! signed-zero boundary — with the tolerance taken from each op's §6.8 ULP
//! ceiling where one applies.

use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_op, ulp_distance_f64};

fn ev(op: Op, args: &[f64]) -> f64 {
    eval_op(op, args).unwrap_or_else(|e| panic!("{op:?} eval failed: {e:?}"))
}

/// Within `ulps` ULP by the sign-magnitude total-order metric.
fn within_ulp(got: f64, expected: f64, ulps: u64) -> bool {
    ulp_distance_f64(got, expected) <= ulps
}

/// The declared §6.8 ULP ceiling for an op, as the differential tolerance.
fn ceiling(op: Op) -> u64 {
    op.ulp_ceiling().expect("op has a declared ULP ceiling") as u64
}

// ---- §6.2 shared conventions ------------------------------------------------

#[test]
fn test_ops_default_nan_propagation() {
    // KISS-OPS-6.2-0003.
    let nan = f64::NAN;
    assert!(ev(Op::Add, &[nan, 1.0]).is_nan());
    assert!(ev(Op::Sub, &[1.0, nan]).is_nan());
    assert!(ev(Op::Mul, &[nan, 0.0]).is_nan());
}

#[test]
fn test_ops_signed_zero_preserved() {
    // KISS-OPS-6.2-0004.
    assert!(ev(Op::Neg, &[-0.0]).is_sign_positive());
    assert!(ev(Op::Abs, &[-0.0]).is_sign_positive());
    assert!(ev(Op::Trunc, &[-0.0]).is_sign_negative()); // §6.7-0002
}

// ---- §6.4 arithmetic atoms --------------------------------------------------

#[test]
fn test_ops_add_sub_mul() {
    assert_eq!(ev(Op::Add, &[2.0, 3.0]), 5.0);
    assert_eq!(ev(Op::Sub, &[2.0, 3.0]), -1.0);
    assert_eq!(ev(Op::Mul, &[2.0, 3.0]), 6.0);
}

#[test]
fn test_ops_div_float() {
    assert_eq!(ev(Op::Div, &[1.0, 4.0]), 0.25);
    assert!(ev(Op::Div, &[1.0, 0.0]).is_infinite());
}

#[test]
fn test_ops_neg() {
    assert_eq!(ev(Op::Neg, &[3.0]), -3.0);
    assert!(ev(Op::Neg, &[f64::NAN]).is_nan());
}

#[test]
fn test_ops_abs_raw_bit() {
    assert_eq!(ev(Op::Abs, &[-3.0]), 3.0);
    assert!(ev(Op::Abs, &[f64::NAN]).is_nan());
}

// ---- §6.5 raw-bit select ----------------------------------------------------

#[test]
fn test_ops_select_order() {
    assert_eq!(ev(Op::Select, &[1.0, 10.0, 20.0]), 10.0);
    assert_eq!(ev(Op::Select, &[0.0, 10.0, 20.0]), 20.0);
}

#[test]
fn test_ops_select_raw_bit() {
    // KISS-OPS-6.5-0002: signed zero of the chosen arm preserved.
    assert!(ev(Op::Select, &[1.0, -0.0, 5.0]).is_sign_negative());
}

#[test]
fn test_ops_select_cond_zero_nan() {
    // KISS-OPS-6.5-0003.
    assert_eq!(ev(Op::Select, &[-0.0, 10.0, 20.0]), 20.0);
    assert_eq!(ev(Op::Select, &[f64::NAN, 10.0, 20.0]), 10.0);
}

// ---- §6.6 comparison atoms --------------------------------------------------

#[test]
fn test_ops_compare_predicates() {
    assert_eq!(ev(Op::CmpLt, &[1.0, 2.0]), 1.0);
    assert_eq!(ev(Op::CmpGe, &[1.0, 2.0]), 0.0);
}

#[test]
fn test_ops_compare_nan_false() {
    let nan = f64::NAN;
    for op in [Op::CmpEq, Op::CmpLt, Op::CmpLe, Op::CmpGt, Op::CmpGe] {
        assert_eq!(ev(op, &[nan, 1.0]), 0.0, "{op:?}(NaN,1) must be 0");
    }
}

#[test]
fn test_ops_cmp_ne_nan_true() {
    assert_eq!(ev(Op::CmpNe, &[f64::NAN, f64::NAN]), 1.0);
    assert_eq!(ev(Op::CmpNe, &[1.0, 1.0]), 0.0);
}

#[test]
fn test_ops_compare_signed_zero() {
    assert_eq!(ev(Op::CmpEq, &[-0.0, 0.0]), 1.0);
    assert_eq!(ev(Op::CmpLe, &[-0.0, 0.0]), 1.0);
    assert_eq!(ev(Op::CmpGe, &[-0.0, 0.0]), 1.0);
}

// ---- §6.7 rounding atoms ----------------------------------------------------

#[test]
fn test_ops_rounding_directions() {
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
    // KISS-OPS-6.8-0001: within the op's declared ULP ceiling (from the vocab).
    assert!(within_ulp(ev(Op::Erf, &[1.0]), 0.8427007929497149, ceiling(Op::Erf)));
    assert!(within_ulp(ev(Op::Exp, &[1.0]), std::f64::consts::E, ceiling(Op::Exp)));
    assert!(within_ulp(
        ev(Op::Log, &[std::f64::consts::E]),
        1.0,
        ceiling(Op::Log)
    ));
    assert!(within_ulp(
        ev(Op::Sqrt, &[2.0]),
        std::f64::consts::SQRT_2,
        ceiling(Op::Sqrt)
    ));
    assert_eq!(ev(Op::Exp, &[0.0]), 1.0);
}

#[test]
fn test_ops_atan2_declared_ulp() {
    // atan2 carries a §6.8 4-ULP ceiling (see the vocab RFC note on its §6.9
    // definition). Quadrant + magnitude check.
    assert!(within_ulp(
        ev(Op::Atan2, &[1.0, 1.0]),
        std::f64::consts::FRAC_PI_4,
        ceiling(Op::Atan2)
    ));
}

// ---- §6.13 reference decompositions (non-primitives) ------------------------

#[test]
fn test_ops_reference_decompositions() {
    // KISS-OPS-6.13-0001.
    assert_eq!(ev(Op::Sqr, &[3.0]), 9.0);
    assert_eq!(ev(Op::Recip, &[4.0]), 0.25);
    assert!(within_ulp(ev(Op::Sigmoid, &[0.0]), 0.5, 1));
    assert_eq!(ev(Op::Relu, &[-2.0]), 0.0);
    assert_eq!(ev(Op::Relu, &[3.0]), 3.0);
    assert!(within_ulp(ev(Op::Erfc, &[0.0]), 1.0, 1));
    // gelu(1) = 0.5*(1+erf(1/sqrt2)) ≈ 0.8413447460685429
    assert!(within_ulp(ev(Op::Gelu, &[1.0]), 0.8413447460685429, 8));
}

#[test]
fn test_ops_pow_full_domain() {
    // KISS-OPS-6.13-0005: pow pinned over its full domain — the exact edges the
    // naive exp(b·log(a)) reference gets wrong for a ≤ 0.
    assert_eq!(ev(Op::Pow, &[2.0, 3.0]), 8.0);
    assert_eq!(ev(Op::Pow, &[-2.0, 3.0]), -8.0); // odd integer exponent
    assert_eq!(ev(Op::Pow, &[-2.0, 2.0]), 4.0); // even integer exponent
    assert_eq!(ev(Op::Pow, &[0.0, 0.0]), 1.0); // pinned
    assert_eq!(ev(Op::Pow, &[0.0, 2.0]), 0.0);
    assert!(ev(Op::Pow, &[-2.0, 0.5]).is_nan()); // a<0, non-integer b → NaN
    assert!(ev(Op::Pow, &[0.0, -1.0]).is_infinite()); // +0^-1 → +inf
}

#[test]
fn test_ops_hypot_inf_nan() {
    // KISS-OPS-6.13-0007: an infinite operand yields +inf even if the other is
    // NaN — the naive sqrt(a²+b²) would yield NaN.
    assert!(ev(Op::Hypot, &[f64::INFINITY, f64::NAN]).is_infinite());
    assert!(ev(Op::Hypot, &[f64::NAN, f64::NEG_INFINITY]).is_infinite());
    assert_eq!(ev(Op::Hypot, &[3.0, 4.0]), 5.0);
    assert!(ev(Op::Hypot, &[f64::NAN, 2.0]).is_nan()); // finite + NaN → NaN
}

#[test]
fn test_ops_decomposition_accuracy_refinement() {
    // KISS-OPS-6.13-0003: the refine-marked ops are computed accurately where the
    // literal decomposition overflows or catastrophically cancels.
    assert_eq!(ev(Op::Tanh, &[1000.0]), 1.0); // not inf/inf = NaN
    assert!(within_ulp(ev(Op::Softplus, &[1000.0]), 1000.0, 8));
    assert!(ev(Op::Softplus, &[-1000.0]).abs() < 1e-300);
    assert!(ev(Op::Mish, &[1000.0]).is_finite());
    // expm1 / log1p near zero: the naive exp(x)-1 / log(1+x) forms cancel to 0
    // (exp(1e-20) rounds to 1.0), while the stable forms stay ≈ x and nonzero
    // (true expm1(x), log1p(x) ≈ x to far below 1 ULP at this magnitude).
    let x = 1e-20;
    assert!(ev(Op::Expm1, &[x]) != 0.0 && within_ulp(ev(Op::Expm1, &[x]), x, 4));
    assert!(ev(Op::Log1p, &[x]) != 0.0 && within_ulp(ev(Op::Log1p, &[x]), x, 4));
    // ldexp exact for integer exponent.
    assert_eq!(ev(Op::Ldexp, &[1.5, 3.0]), 12.0);
}

#[test]
fn test_ops_min_max_prop_nan() {
    // §6.13 max_prop / min_prop propagate NaN; the IEEE fmax/fmin forms don't.
    assert!(ev(Op::MaxProp, &[f64::NAN, 1.0]).is_nan());
    assert_eq!(ev(Op::MaxProp, &[2.0, 1.0]), 2.0);
    assert_eq!(ev(Op::FmaxIeee, &[f64::NAN, 1.0]), 1.0);
}
