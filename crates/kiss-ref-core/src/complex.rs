//! Scalar reference kernels for the **§6.18 complex-arithmetic op family**
//! (`c64`/`c128`).
//!
//! Every complex op is non-primitive (§6.18-0002): it carries a reference
//! decomposition into the real primitive floor (`add … copysign`), the real
//! non-primitive `hypot`, and the `cmake`/`cre`/`cim` component bridge. A complex
//! op on `c64` evaluates that decomposition with `f32` component lanes, on `c128`
//! with `f64` (§6.18-0015) — realized here by the generic `T: ScalarFloat`
//! instantiated at `f32`/`f64` (the only two dtypes with a complex container).
//!
//! **Annex G governs the edges (§6.18-0013).** On any input with an infinite or
//! NaN lane, the ISO C99/C11 Annex G special-value / recovery rules are the pinned
//! semantics and override the naive decomposition. Two things make this tractable
//! without hand-transcribing every table:
//!
//! * `cabs`/`carg`/`clog` are computed straight from the decomposition, because
//!   kiss-ref's `hypot`/`atan2`/`log` atoms already obey the pinned §6.13-0007 /
//!   §6.9-0001 inf/NaN rules — so the decomposition already equals the Annex-G
//!   value on every enumerated row (§6.18-0013: "the decomposition and Annex G do
//!   not disagree"). Verified row-by-row in the conformance tests.
//! * `cmul`/`cdiv`/`cexp`/`csqrt` genuinely need Annex-G recovery: the naive form
//!   produces a spurious `∞·0 = NaN` (`cexp`) or `∞−∞ = NaN` (`cmul`/`cdiv`/
//!   `csqrt`) that Annex G replaces with a determinate complex ∞ / component value.
//!   Those are implemented explicitly per §6.18-0005 / -0006 / -0009 / -0011.

use kiss_ops_vocab::Op;

use crate::scalar::ScalarFloat;
use crate::Error;

/// A scalar complex value: an interleaved `(re, im)` pair whose component lanes
/// are `T` (§6.16-0007). `T = f32` realizes `c64`, `T = f64` realizes `c128`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cplx<T> {
    pub re: T,
    pub im: T,
}

impl<T: ScalarFloat> Cplx<T> {
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Cplx { re, im }
    }
}

/// The result of a complex op: a complex value, or a real component lane
/// (`cre`/`cim`/`cabs`/`carg` yield a real, §6.18-0001).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CplxOut<T> {
    Complex(Cplx<T>),
    Real(T),
}

// ---- lane predicates / constants --------------------------------------------
// T is always f32/f64 in complex context, so the f64 widening is lossless and
// preserves the sign of zeros and infinities.

#[inline]
fn is_inf<T: ScalarFloat>(x: T) -> bool {
    x.to_f64().is_infinite()
}
#[inline]
fn is_finite<T: ScalarFloat>(x: T) -> bool {
    x.to_f64().is_finite()
}
#[inline]
fn is_pos_inf<T: ScalarFloat>(x: T) -> bool {
    x.to_f64() == f64::INFINITY
}
#[inline]
fn is_zero<T: ScalarFloat>(x: T) -> bool {
    // Both +0.0 and -0.0 compare equal to ZERO (§6.5-0003 truthiness convention).
    x == T::ZERO
}
#[inline]
fn inf<T: ScalarFloat>() -> T {
    T::from_f64(f64::INFINITY)
}
#[inline]
fn nan<T: ScalarFloat>() -> T {
    T::from_f64(f64::NAN)
}
#[inline]
fn konst<T: ScalarFloat>(v: f64) -> T {
    // A §6.12-0003 named-constant leaf: the correctly-rounded RNE image of the
    // exact real `v` in the component dtype.
    T::from_f64(v)
}

// ---- the component bridge (§6.18-0003) --------------------------------------
// Raw-lane moves: no arithmetic, so signed zero and NaN payload are preserved.

#[inline]
pub fn cmake<T: ScalarFloat>(re: T, im: T) -> Cplx<T> {
    Cplx { re, im }
}
#[inline]
pub fn cre<T: ScalarFloat>(z: Cplx<T>) -> T {
    z.re
}
#[inline]
pub fn cim<T: ScalarFloat>(z: Cplx<T>) -> T {
    z.im
}

// ---- exact-byte algebraic ops (§6.18-0004) ----------------------------------

#[inline]
pub fn cadd<T: ScalarFloat>(z: Cplx<T>, w: Cplx<T>) -> Cplx<T> {
    Cplx {
        re: z.re.add(w.re),
        im: z.im.add(w.im),
    }
}
#[inline]
pub fn csub<T: ScalarFloat>(z: Cplx<T>, w: Cplx<T>) -> Cplx<T> {
    Cplx {
        re: z.re.sub(w.re),
        im: z.im.sub(w.im),
    }
}
#[inline]
pub fn cneg<T: ScalarFloat>(z: Cplx<T>) -> Cplx<T> {
    // Negate both lanes (flips the sign bit of each, incl. ±0).
    Cplx {
        re: z.re.neg(),
        im: z.im.neg(),
    }
}
#[inline]
pub fn cconj<T: ScalarFloat>(z: Cplx<T>) -> Cplx<T> {
    // Negate only the imaginary lane; `+0.0 → −0.0` (§6.18-0004).
    Cplx {
        re: z.re,
        im: z.im.neg(),
    }
}

/// Box an infinite lane to `±1`, a finite lane to signed `±0` — the Annex-G
/// "recover the sign, drop the magnitude" step for `cmul`/`cdiv` recovery.
#[inline]
fn box_inf<T: ScalarFloat>(x: T) -> T {
    let mag = if is_inf(x) { T::ONE } else { T::ZERO };
    mag.copysign(x)
}
/// Box a NaN lane to a signed `±0` (Annex-G recovery for the finite operand).
#[inline]
fn box_nan_to_zero<T: ScalarFloat>(x: T) -> T {
    if x.is_nan() {
        T::ZERO.copysign(x)
    } else {
        x
    }
}

/// `cmul` — `(ac − bd) + (ad + bc)i` with Annex-G infinity recovery (§6.18-0005).
/// Recovery fires iff **both** result components are NaN **and** at least one
/// operand lane is infinite; then the result is a complex infinity with the
/// Annex-G-determined component signs. (The C-reference intermediate-overflow
/// branch is intentionally omitted: §6.18-0005 pins the trigger to an infinite
/// *operand* lane.)
pub fn cmul<T: ScalarFloat>(z: Cplx<T>, w: Cplx<T>) -> Cplx<T> {
    let (mut a, mut b) = (z.re, z.im);
    let (mut c, mut d) = (w.re, w.im);
    let mut x = a.mul(c).sub(b.mul(d));
    let mut y = a.mul(d).add(b.mul(c));
    if x.is_nan() && y.is_nan() {
        let mut recalc = false;
        if is_inf(a) || is_inf(b) {
            a = box_inf(a);
            b = box_inf(b);
            c = box_nan_to_zero(c);
            d = box_nan_to_zero(d);
            recalc = true;
        }
        if is_inf(c) || is_inf(d) {
            c = box_inf(c);
            d = box_inf(d);
            a = box_nan_to_zero(a);
            b = box_nan_to_zero(b);
            recalc = true;
        }
        if recalc {
            let i = inf::<T>();
            x = i.mul(a.mul(c).sub(b.mul(d)));
            y = i.mul(a.mul(d).add(b.mul(c)));
        }
    }
    Cplx { re: x, im: y }
}

/// `cdiv` — `((ac + bd) + (bc − ad)i) / (c² + d²)` with Annex-G recovery
/// (§6.18-0006). The zero-denominator and infinite-operand cases are produced by
/// recovery with Annex-G-pinned component signs and are NEVER routed through the
/// real `div` atom (the normal path divides only when the denominator is nonzero),
/// so the exact-byte guarantee holds independently of any target's `div`/0
/// behavior (§6.4-0002 target-UB).
pub fn cdiv<T: ScalarFloat>(z: Cplx<T>, w: Cplx<T>) -> Cplx<T> {
    let (mut a, mut b) = (z.re, z.im);
    let (mut c, mut d) = (w.re, w.im);
    let t = c.mul(c).add(d.mul(d));
    // Divide only for a genuinely nonzero denominator; a zero denominator yields
    // the NaN placeholder that trips the recovery block below (no real `div`/0).
    let (mut x, mut y) = if is_zero(t) {
        (nan::<T>(), nan::<T>())
    } else {
        (a.mul(c).add(b.mul(d)).div(t), b.mul(c).sub(a.mul(d)).div(t))
    };
    if x.is_nan() && y.is_nan() {
        if is_zero(t) && (!a.is_nan() || !b.is_nan()) {
            // Nonzero finite numerator over a complex zero → complex infinity
            // (Annex G G.5.1): `copysign(∞, c)·a  +  i·copysign(∞, c)·b`.
            let inf_c = inf::<T>().copysign(c);
            x = inf_c.mul(a);
            y = inf_c.mul(b);
        } else if (is_inf(a) || is_inf(b)) && is_finite(c) && is_finite(d) {
            // Infinite numerator over finite denominator → complex infinity.
            a = box_inf(a);
            b = box_inf(b);
            let i = inf::<T>();
            x = i.mul(a.mul(c).add(b.mul(d)));
            y = i.mul(b.mul(c).sub(a.mul(d)));
        } else if (is_inf(c) || is_inf(d)) && is_finite(a) && is_finite(b) {
            // Finite numerator over infinite denominator → complex zero.
            c = box_inf(c);
            d = box_inf(d);
            let zero = T::ZERO;
            x = zero.mul(a.mul(c).add(b.mul(d)));
            y = zero.mul(b.mul(c).sub(a.mul(d)));
        }
    }
    Cplx { re: x, im: y }
}

// ---- ULP/tolerance ops (§6.18-0014) -----------------------------------------

/// `cabs(z) = hypot(re, im)` → real (§6.18-0007). The pinned `hypot` (§6.13-0007)
/// gives `+∞` whenever a lane is infinite even if the other is NaN.
#[inline]
pub fn cabs<T: ScalarFloat>(z: Cplx<T>) -> T {
    z.re.hypot(z.im)
}

/// `carg(z) = atan2(im, re)` → real, the principal argument in `[−π, +π]` under
/// the IEEE ±0 quadrant conventions of `atan2` (§6.18-0008 / §6.9-0001).
#[inline]
pub fn carg<T: ScalarFloat>(z: Cplx<T>) -> T {
    z.im.atan2(z.re)
}

/// `clog(z) = cmake(log(cabs(z)), carg(z))`, the principal complex log
/// (§6.18-0010). Computed straight from the decomposition: kiss-ref's pinned
/// `hypot`/`atan2`/`log` already reproduce every enumerated Annex-G row (verified
/// in the conformance tests), and the signed-zero of the imaginary lane selects
/// the ±π branch endpoint through `atan2`.
#[inline]
pub fn clog<T: ScalarFloat>(z: Cplx<T>) -> Cplx<T> {
    Cplx {
        re: cabs(z).log(),
        im: carg(z),
    }
}

/// `cexp(z) = exp(re)·(cos(im) + i·sin(im))` for finite arguments, with the
/// enumerated Annex-G special values governing any infinite/NaN lane (§6.18-0009 —
/// the naive form would otherwise give a spurious `∞·0 = NaN`).
pub fn cexp<T: ScalarFloat>(z: Cplx<T>) -> Cplx<T> {
    let (a, b) = (z.re, z.im);

    if a.is_nan() {
        // NaN + i·0 = NaN + i·0 ; NaN + i·(nonzero|NaN) = NaN + i·NaN. The zero
        // imaginary lane is returned VERBATIM (`b`), preserving its sign —
        // Annex-G conjugate symmetry `cexp(z̄) = conj(cexp(z))` and the §6.18-0017
        // exact-sign requirement on zero result components (not forced to `+0`).
        return if is_zero(b) {
            Cplx { re: nan(), im: b }
        } else {
            Cplx {
                re: nan(),
                im: nan(),
            }
        };
    }

    if is_finite(a) {
        // Finite real lane: an infinite/NaN imaginary lane is invalid → NaN+iNaN;
        // otherwise the finite decomposition (which already yields 1+i0 at ±0+i0).
        if !is_finite(b) {
            return Cplx {
                re: nan(),
                im: nan(),
            };
        }
        let ea = a.exp();
        return Cplx {
            re: ea.mul(b.cos()),
            im: ea.mul(b.sin()),
        };
    }

    // a is ±∞.
    let pos = is_pos_inf(a);
    if is_finite(b) {
        return if pos {
            if is_zero(b) {
                // +∞ + i·0 = +∞ + i·0 — the zero imaginary lane is returned
                // verbatim (`b`) to preserve its sign (Annex-G conjugate symmetry /
                // §6.18-0017), matching the finite path where `sin(±0) = ±0`.
                Cplx { re: inf(), im: b }
            } else {
                // +∞ + i·y (finite nonzero y) = +∞·(cos y + i·sin y).
                Cplx {
                    re: inf::<T>().mul(b.cos()),
                    im: inf::<T>().mul(b.sin()),
                }
            }
        } else {
            // −∞ + i·y (finite y, incl. 0) = +0·(cos y + i·sin y) — signed zeros
            // flow from cos y / sin y.
            Cplx {
                re: T::ZERO.mul(b.cos()),
                im: T::ZERO.mul(b.sin()),
            }
        };
    }
    // a is ±∞, b is ±∞ or NaN.
    if pos {
        // +∞ + i·∞ = ±∞ + i·NaN (invalid) ; +∞ + i·NaN = ±∞ + i·NaN.
        Cplx {
            re: inf(),
            im: nan(),
        }
    } else {
        // −∞ + i·∞ = ±0 ± i·0 ; −∞ + i·NaN = ±0 ± i·0 (signs unspecified).
        Cplx {
            re: T::ZERO,
            im: T::ZERO,
        }
    }
}

/// `csqrt(z)` — principal square root (§6.18-0011). Finite arguments use the
/// decomposition `m=cabs(z); (sqrt((m+re)/2), copysign(1,im)·sqrt((m−re)/2))`;
/// any infinite/NaN lane is governed by the enumerated Annex-G rows (the naive
/// form would give a spurious `∞−∞ = NaN`).
pub fn csqrt<T: ScalarFloat>(z: Cplx<T>) -> Cplx<T> {
    let (a, b) = (z.re, z.im);

    // Annex-G special values (any non-finite lane). `x + i·∞ = +∞ + i·∞` for ANY
    // x takes precedence over the NaN-real row.
    if is_inf(b) {
        // x + i·∞ = +∞ + i·∞ (b = +∞); the imaginary sign follows b for −∞.
        return Cplx {
            re: inf(),
            im: inf::<T>().copysign(b),
        };
    }
    if a.is_nan() || b.is_nan() {
        if a.is_nan() && !is_inf(b) {
            // NaN + i·y (finite y) = NaN + i·NaN ; NaN + i·NaN = NaN + i·NaN.
            return Cplx {
                re: nan(),
                im: nan(),
            };
        }
        // a is ±∞ with b NaN.
        return if is_pos_inf(a) {
            // +∞ + i·NaN = +∞ + i·NaN.
            Cplx {
                re: inf(),
                im: nan(),
            }
        } else {
            // −∞ + i·NaN = NaN ± i·∞.
            Cplx {
                re: nan(),
                im: inf::<T>().copysign(b),
            }
        };
    }
    if is_inf(a) {
        // a = ±∞, b finite (incl. 0).
        return if is_pos_inf(a) {
            // +∞ + i·y (finite y) = +∞ + i·0, sign of the zero from b.
            Cplx {
                re: inf(),
                im: T::ZERO.copysign(b),
            }
        } else {
            // −∞ + i·y (finite y) = +0 + i·∞, sign of the ∞ from b.
            Cplx {
                re: T::ZERO,
                im: inf::<T>().copysign(b),
            }
        };
    }

    // Finite decomposition.
    let m = cabs(z);
    let two = konst::<T>(2.0);
    let re = m.add(a).div(two).sqrt();
    let im = T::ONE.copysign(b).mul(m.sub(a).div(two).sqrt());
    Cplx { re, im }
}

/// `cpow(z, w) = cexp(cmul(w, clog(z)))`, the principal value (§6.18-0012),
/// evaluated through the already-Annex-G-governed `clog`/`cmul`/`cexp` kernels.
/// The pinned corner cases (`cpow(z,0) = 1+0i`; `cpow(0,w)` by `Re(w)`) are
/// special-cased so they do not depend on the `−∞` composition chain.
pub fn cpow<T: ScalarFloat>(z: Cplx<T>, w: Cplx<T>) -> Cplx<T> {
    // cpow(z, 0) = 1 + 0i for every z (§6.18-0012), including z = 0.
    if is_zero(w.re) && is_zero(w.im) {
        return Cplx {
            re: T::ONE,
            im: T::ZERO,
        };
    }
    // cpow(0 + i0, w): 0 for Re(w) > 0, complex ∞ for Re(w) < 0, NaN for Re(w) = 0
    // (w ≠ 0). (w = 0 handled above.)
    if is_zero(z.re) && is_zero(z.im) {
        let rw = w.re.to_f64();
        return if rw > 0.0 {
            Cplx {
                re: T::ZERO,
                im: T::ZERO,
            }
        } else if rw < 0.0 {
            Cplx {
                re: inf(),
                im: inf(),
            }
        } else {
            Cplx {
                re: nan(),
                im: nan(),
            }
        };
    }
    cexp(cmul(w, clog(z)))
}

/// Evaluate an advertised complex op or `cneg`/`cconj` on complex operands.
/// `cmake`/`cre`/`cim` are the component bridge (real↔complex) and are called
/// through their typed functions, not this complex-in dispatch. A non-complex op
/// is [`Error::Unsupported`].
pub fn eval_complex_op<T: ScalarFloat>(op: Op, args: &[Cplx<T>]) -> Result<CplxOut<T>, Error> {
    let unary = |n: usize| -> Result<Cplx<T>, Error> {
        if args.len() == 1 {
            Ok(args[0])
        } else {
            Err(Error::Arity {
                op,
                expected: n,
                got: args.len(),
            })
        }
    };
    let binary = || -> Result<(Cplx<T>, Cplx<T>), Error> {
        if args.len() == 2 {
            Ok((args[0], args[1]))
        } else {
            Err(Error::Arity {
                op,
                expected: 2,
                got: args.len(),
            })
        }
    };
    Ok(match op {
        Op::Cneg => CplxOut::Complex(cneg(unary(1)?)),
        Op::Cconj => CplxOut::Complex(cconj(unary(1)?)),
        Op::Cexp => CplxOut::Complex(cexp(unary(1)?)),
        Op::Clog => CplxOut::Complex(clog(unary(1)?)),
        Op::Csqrt => CplxOut::Complex(csqrt(unary(1)?)),
        Op::Cabs => CplxOut::Real(cabs(unary(1)?)),
        Op::Carg => CplxOut::Real(carg(unary(1)?)),
        Op::Cadd => {
            let (z, w) = binary()?;
            CplxOut::Complex(cadd(z, w))
        }
        Op::Csub => {
            let (z, w) = binary()?;
            CplxOut::Complex(csub(z, w))
        }
        Op::Cmul => {
            let (z, w) = binary()?;
            CplxOut::Complex(cmul(z, w))
        }
        Op::Cdiv => {
            let (z, w) = binary()?;
            CplxOut::Complex(cdiv(z, w))
        }
        Op::Cpow => {
            let (z, w) = binary()?;
            CplxOut::Complex(cpow(z, w))
        }
        _ => return Err(Error::Unsupported(op)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Cplx<f64> {
        Cplx { re, im }
    }

    #[test]
    fn cadd_csub_are_componentwise() {
        assert_eq!(cadd(c(1.0, 2.0), c(3.0, 4.0)), c(4.0, 6.0));
        assert_eq!(csub(c(1.0, 2.0), c(3.0, 4.0)), c(-2.0, -2.0));
    }

    #[test]
    fn cconj_flips_imaginary_signed_zero() {
        // +0.0 imaginary → −0.0 (§6.18-0004).
        let r = cconj(c(1.0, 0.0));
        assert_eq!(r.re, 1.0);
        assert!(r.im.to_bits() == (-0.0f64).to_bits());
    }

    #[test]
    fn cmul_finite() {
        // (1+2i)(3+4i) = -5 + 10i.
        assert_eq!(cmul(c(1.0, 2.0), c(3.0, 4.0)), c(-5.0, 10.0));
    }

    #[test]
    fn cmul_annexg_inf_times_finite_is_inf_not_nan() {
        // (∞ + i·∞)·(1 + i·0): naive gives NaN components; Annex G recovers ∞.
        let r = cmul(c(f64::INFINITY, f64::INFINITY), c(1.0, 0.0));
        assert!(r.re.is_infinite() || r.im.is_infinite());
        assert!(!(r.re.is_nan() && r.im.is_nan()));
    }

    #[test]
    fn cdiv_by_complex_zero_is_infinity_never_traps() {
        // (2 + 3i)/(0 + 0i) = copysign(∞,+0)·2 + i·copysign(∞,+0)·3 = +∞ + i·∞.
        let r = cdiv(c(2.0, 3.0), c(0.0, 0.0));
        assert_eq!(r.re, f64::INFINITY);
        assert_eq!(r.im, f64::INFINITY);
    }

    #[test]
    fn cdiv_finite() {
        // (1+0i)/(0+1i) = -i.
        let r = cdiv(c(1.0, 0.0), c(0.0, 1.0));
        assert!((r.re - 0.0).abs() < 1e-12 && (r.im + 1.0).abs() < 1e-12);
    }

    #[test]
    fn cexp_infinity_rows() {
        // +∞ + i·0 = +∞ + i·0 (naive would give ∞·0 = NaN imaginary).
        let r = cexp(c(f64::INFINITY, 0.0));
        assert_eq!(r.re, f64::INFINITY);
        assert_eq!(r.im, 0.0);
        // −∞ + i·y = +0·(cos y + i sin y).
        let r = cexp(c(f64::NEG_INFINITY, 1.0));
        assert_eq!(r.re, 0.0);
        assert_eq!(r.im, 0.0);
    }

    #[test]
    fn clog_principal_signed_zero_branch() {
        // clog(−1 + i·0) = 0 + iπ ; clog(−1 − i·0) = 0 − iπ (§6.18-0010).
        let up = clog(c(-1.0, 0.0));
        assert!((up.re).abs() < 1e-12 && (up.im - core::f64::consts::PI).abs() < 1e-12);
        let dn = clog(c(-1.0, -0.0));
        assert!((dn.re).abs() < 1e-12 && (dn.im + core::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn csqrt_principal_signed_zero_branch() {
        // csqrt(−4 + i·0) = 0 + 2i ; csqrt(−4 − i·0) = 0 − 2i (§6.18-0011).
        let up = csqrt(c(-4.0, 0.0));
        assert!(up.re.abs() < 1e-12 && (up.im - 2.0).abs() < 1e-12);
        let dn = csqrt(c(-4.0, -0.0));
        assert!(dn.re.abs() < 1e-12 && (dn.im + 2.0).abs() < 1e-12);
    }

    #[test]
    fn csqrt_annexg_inf_rows() {
        // −∞ + i·1 = +0 + i·∞ ; +∞ + i·1 = +∞ + i·0.
        let l = csqrt(c(f64::NEG_INFINITY, 1.0));
        assert_eq!(l.re, 0.0);
        assert_eq!(l.im, f64::INFINITY);
        let r = csqrt(c(f64::INFINITY, 1.0));
        assert_eq!(r.re, f64::INFINITY);
        assert_eq!(r.im, 0.0);
    }

    #[test]
    fn cpow_zero_exponent_is_one() {
        assert_eq!(cpow(c(2.0, 3.0), c(0.0, 0.0)), c(1.0, 0.0));
        assert_eq!(cpow(c(0.0, 0.0), c(0.0, 0.0)), c(1.0, 0.0));
    }

    #[test]
    fn cpow_zero_base() {
        // 0^w: Re(w)>0 → 0 ; Re(w)<0 → ∞.
        assert_eq!(cpow(c(0.0, 0.0), c(2.0, 0.0)), c(0.0, 0.0));
        let neg = cpow(c(0.0, 0.0), c(-2.0, 0.0));
        assert_eq!(neg.re, f64::INFINITY);
    }

    #[test]
    fn cpow_matches_known_value() {
        // i^2 = -1 (within tolerance; principal branch).
        let r = cpow(c(0.0, 1.0), c(2.0, 0.0));
        assert!((r.re + 1.0).abs() < 1e-9 && r.im.abs() < 1e-9);
    }
}
