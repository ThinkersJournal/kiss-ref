//! Conformance corpus for the **§6.18 complex-arithmetic op family**, with test
//! names mirroring KISS-Conform's `test_ops_*` (KISS-Ops §9.1 clause→test
//! traceability). Each test cites the clause it exercises.
//!
//! The ULP complex ops (`carg`/`clog`/`csqrt`/`cexp`) are judged under the
//! §6.18-0017 split comparator ([`kiss_ref_core::complex_conforms_c64`] /
//! [`kiss_ref_core::arg_conforms_f64`]): ULP on component magnitudes, exact sign
//! on zero components and the ±π branch endpoint.

use core::f64::consts::PI;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::complex::{
    cabs, cadd, carg, cconj, cdiv, cexp, cim, clog, cmake, cmul, cneg, cpow, cre, csqrt, csub, Cplx,
};
use kiss_ref_core::tensor_ops::op_det;
use kiss_ref_core::{arg_conforms_f64, complex_conforms_c64, DetClass};

const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

fn c(re: f64, im: f64) -> Cplx<f64> {
    Cplx { re, im }
}

/// Same ±0 sign, same value, exact bits.
fn bit_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

// ---- §6.18-0001 op set / §6.18-0002 no new primitive / §6.18-0016 advertised --

#[test]
fn test_ops_complex_op_set() {
    // KISS-OPS-6.18-0001: exactly the 15 complex ops.
    let set: Vec<&str> = Op::ALL
        .iter()
        .filter(|o| o.is_complex())
        .map(|o| o.token())
        .collect();
    assert_eq!(set.len(), 15);
    for t in [
        "cmake", "cre", "cim", "cadd", "csub", "cneg", "cconj", "cmul", "cdiv", "cabs", "carg",
        "cexp", "clog", "csqrt", "cpow",
    ] {
        assert!(set.contains(&t), "{t} must be in the complex family");
    }
}

#[test]
fn test_ops_complex_no_new_primitive() {
    // KISS-OPS-6.18-0002: every complex op is non-primitive; floor unchanged.
    assert!(Op::ALL
        .iter()
        .filter(|o| o.is_complex())
        .all(|o| !o.is_primitive_floor()));
}

#[test]
fn test_ops_complex_advertised_high_level() {
    // KISS-OPS-6.18-0016: cmake/cre/cim are plumbing (not advertised); the rest are.
    for o in Op::ALL.iter().filter(|o| o.is_complex()) {
        let advertised = !o.is_complex_component_bridge();
        let is_bridge = matches!(o, Op::Cmake | Op::Cre | Op::Cim);
        assert_eq!(advertised, !is_bridge, "{o:?}");
    }
}

// ---- §6.18-0003 component bridge --------------------------------------------

#[test]
fn test_ops_complex_component_bridge() {
    // KISS-OPS-6.18-0003: cmake places (re,im); cre/cim read them back; raw-lane
    // moves preserve signed zero and NaN payload.
    let z = cmake(1.0, -0.0);
    assert_eq!(cre(z), 1.0);
    assert!(bit_eq(cim(z), -0.0)); // −0.0 preserved
    let n = cmake(f64::from_bits(0x7ff8_0000_dead_beef), 2.0);
    assert!(bit_eq(cre(n), f64::from_bits(0x7ff8_0000_dead_beef))); // NaN payload intact
    assert_eq!(cim(n), 2.0);
}

// ---- §6.18-0004 add / sub / neg / conj --------------------------------------

#[test]
fn test_ops_complex_add_sub_neg_conj() {
    // KISS-OPS-6.18-0004.
    assert_eq!(cadd(c(1.0, 2.0), c(3.0, 4.0)), c(4.0, 6.0));
    assert_eq!(csub(c(1.0, 2.0), c(3.0, 4.0)), c(-2.0, -2.0));
    assert_eq!(cneg(c(1.0, -2.0)), c(-1.0, 2.0));
    // cconj flips the imaginary sign bit: +0.0 → −0.0.
    let r = cconj(c(5.0, 0.0));
    assert_eq!(r.re, 5.0);
    assert!(bit_eq(r.im, -0.0));
}

// ---- §6.18-0005 cmul + Annex G ----------------------------------------------

#[test]
fn test_ops_cmul_annexg() {
    // Finite: (1+2i)(3+4i) = -5 + 10i.
    assert_eq!(cmul(c(1.0, 2.0), c(3.0, 4.0)), c(-5.0, 10.0));
    // Annex G: an infinite operand × a nonzero finite operand is a complex
    // infinity, not NaN+iNaN (the naive form would give both-NaN).
    let r = cmul(c(INF, INF), c(1.0, 0.0));
    assert!(!(r.re.is_nan() && r.im.is_nan()));
    assert!(r.re.is_infinite() || r.im.is_infinite());
}

// ---- §6.18-0006 cdiv + Annex G ----------------------------------------------

#[test]
fn test_ops_cdiv_annexg() {
    // Finite: (1)/(i) = -i.
    let q = cdiv(c(1.0, 0.0), c(0.0, 1.0));
    assert!(arg_conforms_f64(q.re, 0.0, 4) || q.re.abs() < 1e-12);
    assert!((q.im + 1.0).abs() < 1e-12);
    // Nonzero finite numerator over a complex ZERO → complex infinity, never a
    // trap or NaN (§6.18-0006 / Annex G G.5.1), and NOT via the real div atom.
    let z = cdiv(c(2.0, 3.0), c(0.0, 0.0));
    assert_eq!(z.re, INF);
    assert_eq!(z.im, INF);
    // Complex-infinity numerator over finite denom → complex infinity.
    let big = cdiv(c(INF, INF), c(1.0, 0.0));
    assert!(big.re.is_infinite() || big.im.is_infinite());
    assert!(!(big.re.is_nan() && big.im.is_nan()));
}

// ---- §6.18-0007 cabs / §6.18-0008 carg --------------------------------------

#[test]
fn test_ops_cabs_annexg() {
    // KISS-OPS-6.18-0007: cabs = hypot; an infinite lane dominates a NaN lane.
    assert_eq!(cabs(c(3.0, 4.0)), 5.0);
    assert_eq!(cabs(c(INF, NAN)), INF);
    assert_eq!(cabs(c(NAN, INF)), INF);
}

#[test]
fn test_ops_carg_principal() {
    // KISS-OPS-6.18-0008: principal argument via atan2, signed-zero quadrant.
    assert!(arg_conforms_f64(carg(c(1.0, 1.0)), PI / 4.0, 4));
    // Signed zero of the imaginary lane selects the ±π endpoint on the cut.
    assert!(arg_conforms_f64(carg(c(-1.0, 0.0)), PI, 4));
    assert!(arg_conforms_f64(carg(c(-1.0, -0.0)), -PI, 4));
    // A wrong endpoint sign is rejected by the split comparator (magnitude alone
    // would pass).
    assert!(!arg_conforms_f64(-PI, PI, 4));
}

// ---- §6.18-0009 cexp + Annex G ----------------------------------------------

#[test]
fn test_ops_cexp_annexg() {
    // Finite: cexp(0) = 1 + 0i.
    assert!(complex_conforms_c64(cexp(c(0.0, 0.0)), c(1.0, 0.0), 4));
    // +∞ + i·0 = +∞ + i·0 (naive gives ∞·0 = NaN imaginary).
    let r = cexp(c(INF, 0.0));
    assert_eq!(r.re, INF);
    assert!(bit_eq(r.im, 0.0));
    // −∞ + i·y = +0·(cos y + i sin y) → (+0, +0) for y in (0, π/2).
    let l = cexp(c(-INF, 1.0));
    assert_eq!(l.re, 0.0);
    assert_eq!(l.im, 0.0);
    // x + i·∞ (finite x) is invalid → NaN + iNaN.
    let inv = cexp(c(1.0, INF));
    assert!(inv.re.is_nan() && inv.im.is_nan());
}

// ---- §6.18-0010 clog + §6.18-0011 csqrt principal branches ------------------

#[test]
fn test_ops_clog_principal_branch() {
    // KISS-OPS-6.18-0010: principal log; signed zero of the imaginary lane selects
    // the ±π endpoint: clog(−1 ± i·0) = 0 ± iπ.
    assert!(complex_conforms_c64(clog(c(-1.0, 0.0)), c(0.0, PI), 4));
    assert!(complex_conforms_c64(clog(c(-1.0, -0.0)), c(0.0, -PI), 4));
    // Annex-G rows reproduced by the pinned-atom decomposition.
    assert!(complex_conforms_c64(clog(c(1.0, INF)), c(INF, PI / 2.0), 4));
    let l = clog(c(-INF, INF));
    assert_eq!(l.re, INF);
    assert!(arg_conforms_f64(l.im, 3.0 * PI / 4.0, 4));
}

#[test]
fn test_ops_csqrt_principal_branch() {
    // KISS-OPS-6.18-0011: principal sqrt; signed zero of imaginary lane → ±2i.
    assert!(complex_conforms_c64(csqrt(c(-4.0, 0.0)), c(0.0, 2.0), 2));
    assert!(complex_conforms_c64(csqrt(c(-4.0, -0.0)), c(0.0, -2.0), 2));
    assert!(complex_conforms_c64(csqrt(c(4.0, 0.0)), c(2.0, 0.0), 2));
    // Annex-G rows: −∞ + i·1 = +0 + i·∞ ; +∞ + i·1 = +∞ + i·0.
    let l = csqrt(c(-INF, 1.0));
    assert_eq!(l.re, 0.0);
    assert_eq!(l.im, INF);
    let r = csqrt(c(INF, 1.0));
    assert_eq!(r.re, INF);
    assert!(bit_eq(r.im, 0.0));
}

// ---- §6.18-0012 cpow --------------------------------------------------------

#[test]
fn test_ops_cpow_principal() {
    // KISS-OPS-6.18-0012.
    assert_eq!(cpow(c(2.0, 3.0), c(0.0, 0.0)), c(1.0, 0.0)); // z^0 = 1
    assert_eq!(cpow(c(0.0, 0.0), c(0.0, 0.0)), c(1.0, 0.0)); // 0^0 = 1
    assert_eq!(cpow(c(0.0, 0.0), c(2.0, 0.0)), c(0.0, 0.0)); // 0^w, Re(w)>0 → 0
    assert_eq!(cpow(c(0.0, 0.0), c(-2.0, 0.0)).re, INF); //     Re(w)<0 → ∞
                                                         // i^2 = -1 (principal). The imaginary lane carries the `sin(π)` residual
                                                         // (~1e-16), so compare with a small absolute tolerance rather than the
                                                         // near-zero-strict split comparator against the exact-zero ideal.
    let i2 = cpow(c(0.0, 1.0), c(2.0, 0.0));
    assert!((i2.re + 1.0).abs() < 1e-9 && i2.im.abs() < 1e-9);
}

// ---- §6.18-0013 Annex G governs ---------------------------------------------

#[test]
fn test_ops_complex_nan_inf_annexg() {
    // KISS-OPS-6.18-0013: on a non-finite lane, Annex G governs over the naive
    // decomposition — no complex op returns the spurious naive value where a row
    // pins another.
    assert_eq!(cabs(c(INF, NAN)), INF); // not NaN
    assert!(bit_eq(cexp(c(INF, 0.0)).im, 0.0)); // not NaN (naive ∞·0)
    assert_eq!(csqrt(c(-INF, 1.0)).re, 0.0); // not NaN (naive ∞−∞)
    let d = cdiv(c(2.0, 3.0), c(0.0, 0.0)); // not a trap
    assert!(d.re.is_infinite());
}

// ---- §6.18-0014 determinism class -------------------------------------------

#[test]
fn test_ops_complex_determinism_class() {
    // KISS-OPS-6.18-0014: algebraic complex ops exact-byte; transcendental-bearing
    // ones ULP/tolerance.
    for op in [
        Op::Cmake,
        Op::Cre,
        Op::Cim,
        Op::Cadd,
        Op::Csub,
        Op::Cneg,
        Op::Cconj,
        Op::Cmul,
        Op::Cdiv,
    ] {
        assert_eq!(op_det(op), DetClass::ExactByte, "{op:?} must be exact-byte");
    }
    for op in [Op::Cabs, Op::Carg, Op::Cexp, Op::Clog, Op::Csqrt, Op::Cpow] {
        assert!(
            matches!(op_det(op), DetClass::Ulp(_)),
            "{op:?} must be ULP/tolerance"
        );
    }
}

// ---- §6.18-0015 component dtype ---------------------------------------------

#[test]
fn test_ops_complex_component_dtype() {
    // KISS-OPS-6.18-0015: c32 evaluates in f32 lanes, c64 in f64 lanes.
    assert_eq!(Dtype::C32.component_dtype(), Some(Dtype::F32));
    assert_eq!(Dtype::C64.component_dtype(), Some(Dtype::F64));
    // The f32 lane rounds to f32; the f64 lane keeps full precision — a value that
    // differs between the two component dtypes.
    let v = 0.1_f64;
    let in_f64 = cadd(Cplx { re: v, im: 0.0 }, Cplx { re: v, im: 0.0 }).re;
    let in_f32 = kiss_ref_core::complex::cadd(
        Cplx {
            re: v as f32,
            im: 0.0,
        },
        Cplx {
            re: v as f32,
            im: 0.0,
        },
    )
    .re;
    assert_eq!(in_f64, 0.2);
    assert_eq!(in_f32, 0.1_f32 + 0.1_f32);
}

// ---- §6.18-0017 split comparator (exact sign / branch endpoint) -------------

#[test]
fn test_ops_complex_branch_sign_exact() {
    // KISS-OPS-6.18-0017: a wrong sign of a zero component, or a wrong ±π endpoint,
    // is non-conforming even when the magnitude is within tolerance.
    // Wrong sign of a zero imaginary component: reference +0, candidate −0.
    assert!(complex_conforms_c64(c(2.0, 0.0), c(2.0, 0.0), 4));
    assert!(!complex_conforms_c64(c(2.0, 0.0), c(2.0, -0.0), 4));
    // Wrong ±π endpoint on carg (magnitude identical, sign flipped).
    assert!(arg_conforms_f64(PI, PI, 4));
    assert!(!arg_conforms_f64(PI, -PI, 4));
    // A same-sign, within-ULP magnitude still passes.
    let near_pi = f64::from_bits(PI.to_bits() + 1);
    assert!(arg_conforms_f64(PI, near_pi, 4));
}
