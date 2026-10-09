// SPDX-License-Identifier: MIT OR Apache-2.0
//! `pow` over its FULL domain, rule by rule (KISS-OPS-6.13-0005 as rewritten by KISS #516).
//!
//! The clause pins `pow` by the IEEE 754-2019 clause 9.2.1 special-value table (the same table as
//! ISO C99/C11 Annex F.9.4.4), applied in order with first match winning, and says the
//! `exp(b·log a)` reference is wrong at exactly the values the rules override. kiss-ref evaluates
//! `pow` directly (not through the reference), so each rule is checked here on every lane that can
//! hold the operands. Expected values come from the clause text, not from running `libm`.

use half::{bf16, f16};
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_op, ScalarFloat};

const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

/// (rule, a, b, expected). `expected` NaN means "any NaN".
const CASES: &[(&str, f64, f64, f64)] = &[
    // (1) pow(a, ±0) = +1 for every a, including NaN and ±inf.
    ("1", 0.0, 0.0, 1.0),
    ("1", -0.0, 0.0, 1.0),
    ("1", NAN, 0.0, 1.0),
    ("1", NAN, -0.0, 1.0),
    ("1", INF, 0.0, 1.0),
    ("1", -INF, -0.0, 1.0),
    ("1", -2.0, 0.0, 1.0),
    // (2) pow(+1, b) = +1 for every b, including NaN and ±inf.
    ("2", 1.0, NAN, 1.0),
    ("2", 1.0, INF, 1.0),
    ("2", 1.0, -INF, 1.0),
    ("2", 1.0, 3.0, 1.0),
    // (3) otherwise a NaN operand yields NaN.
    ("3", NAN, 2.0, NAN),
    ("3", 2.0, NAN, NAN),
    ("3", NAN, NAN, NAN),
    ("3", -1.0, NAN, NAN),
    // (4) pow(-1, ±inf) = +1.
    ("4", -1.0, INF, 1.0),
    ("4", -1.0, -INF, 1.0),
    // (5) b = ±inf, |a| < 1: +0 for +inf, +inf for -inf; |a| > 1 (incl a = ±inf): the reverse.
    ("5", 0.5, INF, 0.0),
    ("5", -0.5, INF, 0.0),
    ("5", 0.5, -INF, INF),
    ("5", 2.0, INF, INF),
    ("5", -2.0, INF, INF),
    ("5", 2.0, -INF, 0.0),
    ("5", INF, INF, INF),
    ("5", -INF, INF, INF),
    ("5", INF, -INF, 0.0),
    ("5", 0.0, INF, 0.0),
    ("5", 0.0, -INF, INF),
    // (6) zero base, finite b.
    ("6", 0.0, 2.0, 0.0),
    ("6", 0.0, 0.5, 0.0),
    ("6", 0.0, -2.0, INF),
    ("6", 0.0, -0.5, INF),
    ("6", -0.0, 3.0, -0.0),
    ("6", -0.0, 2.0, 0.0),
    ("6", -0.0, 0.5, 0.0),
    ("6", -0.0, -3.0, -INF),
    ("6", -0.0, -2.0, INF),
    ("6", -0.0, -0.5, INF),
    // (7) infinite base, finite b.
    ("7", INF, 2.0, INF),
    ("7", INF, -2.0, 0.0),
    ("7", -INF, 3.0, -INF),
    ("7", -INF, 2.0, INF),
    ("7", -INF, 0.5, INF),
    ("7", -INF, -3.0, -0.0),
    ("7", -INF, -2.0, 0.0),
    ("7", -INF, -0.5, 0.0),
    // (8) finite a < 0, finite b: NaN unless b is an exact integer; |a|^b even, -(|a|^b) odd.
    ("8", -2.0, 0.5, NAN),
    ("8", -2.0, -0.5, NAN),
    ("8", -2.0, 2.0, 4.0),
    ("8", -2.0, 3.0, -8.0),
    ("8", -2.0, -2.0, 0.25),
    ("8", -2.0, -3.0, -0.125),
];

fn check<T: ScalarFloat>(lane: &str) {
    for &(rule, a, b, want) in CASES {
        let got = eval_op::<T>(Op::Pow, &[T::from_f64(a), T::from_f64(b)])
            .unwrap()
            .to_f64();
        let ok = if want.is_nan() {
            got.is_nan()
        } else {
            got.to_bits() == want.to_bits()
        };
        assert!(
            ok,
            "{lane} rule ({rule}): pow({a}, {b}) = {got}, expected {want}"
        );
    }
}

#[test]
fn pow_follows_the_ieee_9_2_1_special_value_table_on_every_lane() {
    check::<f64>("f64");
    check::<f32>("f32");
    check::<f16>("f16");
    check::<bf16>("bf16");
}

/// Rule (9): only for finite a > 0, a != 1 and finite non-zero b is `pow` the `exp(b·log a)` value.
#[test]
fn pow_agrees_with_the_reference_where_only_rule_9_applies() {
    for (a, b) in [(2.0f64, 10.0), (3.0, 0.5), (0.25, 3.0), (10.0, -2.0)] {
        let got = eval_op::<f64>(Op::Pow, &[a, b]).unwrap();
        let reference = (b * a.ln()).exp();
        let rel = ((got - reference) / reference).abs();
        assert!(
            rel < 1e-12,
            "pow({a}, {b}) = {got} vs exp(b ln a) = {reference}"
        );
    }
}

/// The table can fail: the reference `exp(b·log a)` gets rule (2) and (1) wrong, which is the
/// clause's own informative remark, so this guards that the test would notice a regression to it.
#[test]
fn the_exp_log_reference_is_wrong_exactly_where_the_rules_override_it() {
    let reference = |a: f64, b: f64| (b * a.ln()).exp();
    assert!(
        reference(1.0, INF).is_nan(),
        "exp(inf * log 1) = exp(inf * 0) is NaN, rule (2) requires 1"
    );
    assert!(
        reference(INF, 0.0).is_nan(),
        "0 * log(inf) is NaN, rule (1) requires 1"
    );
    let pinned = eval_op::<f64>(Op::Pow, &[1.0, INF]).unwrap();
    assert_eq!(pinned, 1.0);
}
