// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conformance corpus — the pinned numeric semantics of KISS-Ops §6, with test
//! names mirroring KISS-Conform's `test_ops_*` (KISS-Ops §9.1 clause→test
//! traceability). Each test cites the clause it exercises.
//!
//! Comparisons use the reference's own sign-magnitude ULP metric
//! ([`kiss_ref_core::ulp_distance_f64`]) — faithful near zero and across the
//! signed-zero boundary — with the tolerance taken from each op's §6.8 ULP
//! ceiling where one applies.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::kernels::scatter;
use kiss_ref_core::tensor_ops::matmul;
use kiss_ref_core::{eval_op, ulp_distance_f64, Combine, IndexTensor, Tensor};

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
    assert!(ev(Op::Trunc, &[-0.0]).is_sign_negative()); // KISS-OPS-6.7-0002
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

// The narrow-float sign ops `neg`/`abs`/`copysign` are RAW-BIT sign transforms, so a NaN
// operand's payload is preserved (result is a NaN with the same payload, sign
// flipped/cleared/copied): KISS-OPS-6.4-0003 / -6.4-0004 / -6.9-0002. For bf16/f16/f8e5m2,
// KISS-OPS-6.2-0001 routes to §6.16, where KISS-OPS-6.16-0009 names a
// promote-to-`f32`-and-round-back implementation NON-CONFORMING because it quiets a moved
// signaling NaN. (KISS #399 leaves the §6.4-vs-§6.16 scope citation open; the obligation
// holds under either reading.) These drive `eval_op` at the narrow *storage* dtype so the
// sign op's own body runs, not the `f64` path the atom tests above use. Teeth: the previous
// `from_f32(-self.to_f32())` body ran bf16 through `half`'s quieting widening, so
// `neg(0x7F81)` returned `0xFFC1` (payload `0x01` -> `0x41`), not `0xFF81`.
//
// sNaN / -1.0 bit patterns: bf16 0x7F81 / 0xBF80, f16 0x7C01 / 0xBC00, f8e5m2 0x7D / 0xBC.

#[test]
fn test_ops_neg_raw_bit_narrow() {
    // KISS-OPS-6.4-0003 (+ -6.16-0009 for narrow): flip the sign bit, keep the sNaN payload.
    use half::{bf16, f16};
    use kiss_ref_core::E5m2;
    assert_eq!(
        eval_op(Op::Neg, &[bf16::from_bits(0x7F81)])
            .unwrap()
            .to_bits(),
        0xFF81
    );
    assert_eq!(
        eval_op(Op::Neg, &[f16::from_bits(0x7C01)])
            .unwrap()
            .to_bits(),
        0xFC01
    );
    assert_eq!(
        eval_op(Op::Neg, &[E5m2::from_bits(0x7D)])
            .unwrap()
            .to_bits(),
        0xFD
    );
}

#[test]
fn test_ops_abs_raw_bit_narrow() {
    // KISS-OPS-6.4-0004 (+ -6.16-0009 for narrow): clear the sign bit, keep the sNaN payload.
    use half::{bf16, f16};
    use kiss_ref_core::E5m2;
    assert_eq!(
        eval_op(Op::Abs, &[bf16::from_bits(0xFF81)])
            .unwrap()
            .to_bits(),
        0x7F81
    );
    assert_eq!(
        eval_op(Op::Abs, &[f16::from_bits(0xFC01)])
            .unwrap()
            .to_bits(),
        0x7C01
    );
    assert_eq!(
        eval_op(Op::Abs, &[E5m2::from_bits(0xFD)])
            .unwrap()
            .to_bits(),
        0x7D
    );
}

#[test]
fn test_ops_copysign_raw_bit_narrow() {
    // KISS-OPS-6.9-0002 (+ -6.16-0009 for narrow): magnitude of a, sign of b, keep sNaN payload.
    use half::{bf16, f16};
    use kiss_ref_core::E5m2;
    assert_eq!(
        eval_op(
            Op::Copysign,
            &[bf16::from_bits(0x7F81), bf16::from_bits(0xBF80)]
        )
        .unwrap()
        .to_bits(),
        0xFF81
    );
    assert_eq!(
        eval_op(
            Op::Copysign,
            &[f16::from_bits(0x7C01), f16::from_bits(0xBC00)]
        )
        .unwrap()
        .to_bits(),
        0xFC01
    );
    assert_eq!(
        eval_op(
            Op::Copysign,
            &[E5m2::from_bits(0x7D), E5m2::from_bits(0xBC)]
        )
        .unwrap()
        .to_bits(),
        0xFD
    );
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
    assert!(within_ulp(
        ev(Op::Erf, &[1.0]),
        0.8427007929497149,
        ceiling(Op::Erf)
    ));
    assert!(within_ulp(
        ev(Op::Exp, &[1.0]),
        std::f64::consts::E,
        ceiling(Op::Exp)
    ));
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

#[test]
fn test_ops_scatter_atomic_minmax_nan() {
    // KISS-OPS-6.11-0010: the float scatter atomic-max / atomic-min combines MUST be
    // NaN-propagating, consistent with the KISS-OPS-6.11-0002 max/min reduce monoids —
    // a NaN in EITHER the update stream or the destination makes the combined result
    // NaN. scatter's AtomicMax/AtomicMin route through Op::MaxProp/Op::MinProp, the
    // same monoid `test_ops_min_max_prop_nan` pins scalar-side (two-angle coverage).
    let mk_dest = || Tensor::from_vec(vec![1.0f64, 5.0, 2.0], &[3]).unwrap();
    let index = IndexTensor::new(vec![0i64, 1, 1, 2], &[4], Dtype::I64).unwrap();
    let updates = Tensor::from_vec(vec![f64::NAN, 3.0, 9.0, 7.0], &[4]).unwrap();

    // AtomicMax: idx0 <- max_prop(1, NaN)=NaN; idx1 <- max_prop(max_prop(5,3),9)=9;
    //            idx2 <- max_prop(2,7)=7.
    let out = scatter(mk_dest(), &index, &updates.view(), 0, Combine::AtomicMax).unwrap();
    assert!(out.as_slice()[0].is_nan());
    assert_eq!(out.as_slice()[1], 9.0);
    assert_eq!(out.as_slice()[2], 7.0);

    // AtomicMin: idx0 <- min_prop(1, NaN)=NaN; idx1 <- min_prop(min_prop(5,3),9)=3;
    //            idx2 <- min_prop(2,7)=2.
    let out = scatter(mk_dest(), &index, &updates.view(), 0, Combine::AtomicMin).unwrap();
    assert!(out.as_slice()[0].is_nan());
    assert_eq!(out.as_slice()[1], 3.0);
    assert_eq!(out.as_slice()[2], 2.0);

    // Destination already holds NaN, finite update -> destination NaN wins.
    let dest2 = Tensor::from_vec(vec![f64::NAN, 5.0], &[2]).unwrap();
    let index2 = IndexTensor::new(vec![0i64, 1], &[2], Dtype::I64).unwrap();
    let upd2 = Tensor::from_vec(vec![100.0f64, 3.0], &[2]).unwrap();
    let out = scatter(dest2, &index2, &upd2.view(), 0, Combine::AtomicMax).unwrap();
    assert!(out.as_slice()[0].is_nan());
    assert_eq!(out.as_slice()[1], 5.0);
}

#[test]
fn test_ops_no_fma_contraction_exact_byte() {
    // KISS-OPS-6.17-0002: no fused multiply-add — matmul MUST round after the
    // multiply AND after the add separately, never as one fused rounding of a*b+c.
    // K=2 dot product engineered so the two-separate-roundings result is exactly
    // +0.0, while a single fused rounding would yield 2^-24 (0x3380_0000):
    //   k0: prod = -(1+2^-11) * 1        = -(1+2^-11)           (exact)
    //   k1: prod = (1+2^-12)^2 = 1+2^-11+2^-24 -> round_f32 = 1+2^-11 (ties-to-even)
    //       acc  = round_f32( -(1+2^-11) + (1+2^-11) ) = +0.0
    // A fused fma(1+2^-12, 1+2^-12, -(1+2^-11)) = round_f32(2^-24) = 2^-24 != 0.
    let a = Tensor::from_vec(
        vec![-(1.0f32 + 2f32.powi(-11)), 1.0 + 2f32.powi(-12)],
        &[1, 2],
    )
    .unwrap();
    let b = Tensor::from_vec(vec![1.0f32, 1.0 + 2f32.powi(-12)], &[2, 1]).unwrap();
    let out = matmul(&a.view(), &b.view()).unwrap();
    assert_eq!(out.as_slice()[0].to_bits(), 0u32); // +0.0, NOT 2^-24
}

#[test]
fn test_ops_minmax_tie_quartet_signed_zero() {
    // KISS #74 vectors: on any ±0 tie, ALL FOUR minmax forms return operand A
    // **bit-for-bit** — the §6.13 decompositions share the identical innermost
    // `cmp_ge → a` / `cmp_le → a` select (the NaN arms are the family's only
    // difference), and cmp_ge/cmp_le are both true under signed-zero equality
    // (§6.6). A value-compare (0.0 == -0.0) would pass vacuously, so these
    // assert RAW BITS. The max_prop cell is the seam that caught Baracuda's
    // a>b tie in all three of its backends (fixed at their 7297f17d); the
    // narrow dtypes prove the sign bit survives promote-compute-round.
    use kiss_ref_core::E4m3;
    let ops = [Op::MaxProp, Op::MinProp, Op::FmaxIeee, Op::FminIeee];
    let pairs = [(0.0f32, -0.0f32), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0)];
    macro_rules! quartet {
        ($t:ty, $mk:expr, $bits:expr) => {
            for op in ops {
                for (a, b) in pairs {
                    let (ta, tb): ($t, $t) = ($mk(a), $mk(b));
                    let got: $t = eval_op(op, &[ta, tb])
                        .unwrap_or_else(|e| panic!("{op:?} on {} failed: {e:?}", stringify!($t)));
                    assert_eq!(
                        $bits(got),
                        $bits(ta),
                        "{op:?}({a:?}, {b:?}) as {} must return A bit-for-bit",
                        stringify!($t)
                    );
                }
            }
        };
    }
    quartet!(f64, |v: f32| v as f64, |x: f64| x.to_bits());
    quartet!(f32, |v: f32| v, |x: f32| x.to_bits());
    quartet!(half::f16, half::f16::from_f32, |x: half::f16| x.to_bits());
    quartet!(E4m3, E4m3::from_f32, |x: E4m3| x.to_bits());
}
