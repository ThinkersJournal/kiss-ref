// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scalar reference implementations of the floating-point floor atoms
//! (KISS-Ops §6.4–§6.9), plus the overflow-safe refined forms (§6.13-0003) the
//! resolver substitutes for the overflow-unsafe non-primitive decompositions.
//!
//! Compute happens **in the compute dtype** (KISS-OPS-6.2-0001): `f32` atoms use
//! the `f32` `libm` variants, `f64` atoms the `f64` variants, so each is the
//! IEEE-754 reference for its own dtype. Exact atoms (`+ - * /`, comparisons,
//! `select`, rounding, `copysign`) use native operators / `libm` directed
//! rounding; the transcendentals go through `libm` under their §6.8 declared-ULP.

use kiss_classify_vocab::Dtype;

/// Private supertrait that SEALS [`ScalarFloat`]: only this crate can name and
/// implement `sealed::Sealed`, so `ScalarFloat` is a CLOSED set — exactly the six
/// built-in float lanes below. This is deliberate. A differential reference must
/// DEFINE the arithmetic it checks against; if a consumer could implement
/// `ScalarFloat`, the party being checked would supply the very semantics the
/// oracle uses to check it, and "kiss-ref says X" would become "whose kiss-ref."
/// Exotic floats (FP8 variants, MX formats, posits) are added HERE — upstreamed
/// and reviewed — not plugged in downstream.
mod sealed {
    pub trait Sealed {}
}
// The sealed set: exactly the six built-in float lanes. These are the ONLY impls
// of `Sealed`, so no downstream crate can satisfy `ScalarFloat`'s supertrait bound.
impl sealed::Sealed for f64 {}
impl sealed::Sealed for f32 {}
impl sealed::Sealed for half::f16 {}
impl sealed::Sealed for half::bf16 {}
impl sealed::Sealed for crate::fp8::E4m3 {}
impl sealed::Sealed for crate::fp8::E5m2 {}

/// A floating-point compute dtype the reference evaluates atoms in. A **sealed**
/// trait: implemented for exactly `f64`, `f32`, `half::f16`, `half::bf16`, `E4m3`,
/// and `E5m2` — the six KISS float lanes. Downstream crates cannot implement it
/// (see the private `sealed` module), which is what lets kiss-ref be an authoritative differential
/// reference rather than a mirror of a consumer's own arithmetic.
pub trait ScalarFloat: sealed::Sealed + Copy + PartialEq + PartialOrd {
    const ZERO: Self;
    const ONE: Self;
    /// True for the narrow floats (`f16`/`bf16`). Used by the resolver to
    /// **decline `nextafter`** on them per §6.9-0003 (stepping in a promoted
    /// `f32` yields the wrong neighbor in the narrow lattice).
    const NARROW_FLOAT: bool = false;
    /// The runtime [`Dtype`] tag of this compute/accumulator type — the bridge
    /// from a runtime accumulator `Dtype` to the monomorphized `*_acc` behavior,
    /// and the diagonal predicate (`acc == T::DTYPE`) that routes an
    /// accumulator-parameterized reference back to the verbatim kernel
    /// (RFC #92 direction b, accumulator-dtype tolerance cells).
    const DTYPE: Dtype;

    /// Round an `f64` reference constant into this dtype (§6.12 `const(bits)`).
    fn from_f64(v: f64) -> Self;

    /// Reinterpret the low `Self`-width bits of `bits` as a `Self`, verbatim — the
    /// bit-exact `const(bits)` leaf (KISS-OPS-6.12-0002). Any pattern round-trips
    /// bit-for-bit (signalling-NaN payload, ±0, subnormal), unlike `from_f64` which
    /// rounds an `f64` real value into the dtype.
    fn from_bits(bits: u64) -> Self;

    /// Round an `f64` ACCUMULATOR into this storage dtype with a SINGLE
    /// correctly-rounded RNE, even when the accumulator is wider than this type's
    /// internal pivot (RFC #92 C3 "result rounded once A→S"). Default is `from_f64`
    /// — already one round for `f32`/`f64`, where the accumulator never exceeds the
    /// pivot. The narrow via-`f32` types override it with round-to-odd `f64→f32`
    /// then RNE `f32→narrow` (double-rounding-is-odd), reusing their existing
    /// correctly-rounded `from_f32` codec (§6.16). Byte-identical to `from_f64`
    /// whenever `v` is exactly an `f32` value — which holds for every admitted
    /// accumulator `A ⊆ f32`.
    #[inline]
    fn from_f64_single_round(v: f64) -> Self {
        Self::from_f64(v)
    }

    /// Widen to `f64` (for truthiness / diagnostics, never for compute).
    fn to_f64(self) -> f64;
    /// IEEE `isnan` (`x != x`).
    fn is_nan(self) -> bool;

    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn div(self, b: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn trunc(self) -> Self;
    /// Round to nearest, **ties to even** (§6.7); preserves sign of zero.
    fn round_even(self) -> Self;

    fn exp(self) -> Self;
    fn log(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
    fn erf(self) -> Self;
    fn atan(self) -> Self;
    fn lgamma(self) -> Self;

    fn atan2(self, x: Self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    fn nextafter(self, to: Self) -> Self;

    // Refined non-primitive forms (§6.13-0003) — computed directly rather than
    // via the overflow-unsafe / edge-wrong reference decompositions.
    fn tanh(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    /// `log(1 + x)`, accurate near zero (vs the cancelling `log(add(1,x))`).
    fn log1p(self) -> Self;
    /// `exp(x) - 1`, accurate near zero (vs the cancelling `sub(exp(x),1)`).
    fn expm1(self) -> Self;
    /// IEEE-754 `pow` — pins the full domain of §6.13-0005 (`pow(-2,3)=-8`,
    /// `pow(0,0)=1`, signed-zero rules) that the `exp(b·log(a))` reference gets
    /// wrong for `a ≤ 0`.
    fn pow(self, b: Self) -> Self;
    /// IEEE-754 `hypot` — pins §6.13-0007 (`hypot(±inf, NaN)=+inf`) that the
    /// `sqrt(a²+b²)` reference gets wrong on an infinite operand.
    fn hypot(self, b: Self) -> Self;

    /// `1` if `v`, else `0`, in this dtype (§6.2-0005 comparison result).
    #[inline]
    fn from_bool(v: bool) -> Self {
        if v {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    /// Truthiness of a `select` condition (§6.5-0003): `-0.0` is false (equals
    /// zero), any NaN is true (non-zero). `self != ZERO` gives exactly this.
    #[inline]
    fn is_truthy(self) -> bool {
        self != Self::ZERO
    }
}

macro_rules! impl_scalar_float {
    ($t:ty, dtype=$dtype:expr, bits=$uint:ty,
     exp=$exp:path, log=$log:path, sin=$sin:path, cos=$cos:path, sqrt=$sqrt:path,
     erf=$erf:path, atan=$atan:path, lgamma=$lgamma:path, atan2=$atan2:path,
     copysign=$copysign:path, nextafter=$nextafter:path, floor=$floor:path,
     ceil=$ceil:path, trunc=$trunc:path, fabs=$fabs:path, tanh=$tanh:path,
     sinh=$sinh:path, cosh=$cosh:path, log1p=$log1p:path, expm1=$expm1:path,
     pow=$pow:path, hypot=$hypot:path) => {
        impl ScalarFloat for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;
            const DTYPE: Dtype = $dtype;

            #[inline]
            fn from_f64(v: f64) -> $t {
                v as $t
            }
            #[inline]
            fn from_bits(bits: u64) -> $t {
                <$t>::from_bits(bits as $uint)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline]
            fn is_nan(self) -> bool {
                self != self
            }

            #[inline]
            fn add(self, b: $t) -> $t {
                self + b
            }
            #[inline]
            fn sub(self, b: $t) -> $t {
                self - b
            }
            #[inline]
            fn mul(self, b: $t) -> $t {
                self * b
            }
            #[inline]
            fn div(self, b: $t) -> $t {
                self / b
            }
            #[inline]
            fn neg(self) -> $t {
                -self
            }
            #[inline]
            fn abs(self) -> $t {
                // Raw-bit sign clear: abs(-0.0)=+0.0, NaN payload preserved
                // (KISS-OPS-6.4-0004). libm fabs is a bit op, not arithmetic.
                $fabs(self)
            }

            #[inline]
            fn floor(self) -> $t {
                $floor(self)
            }
            #[inline]
            fn ceil(self) -> $t {
                $ceil(self)
            }
            #[inline]
            fn trunc(self) -> $t {
                $trunc(self)
            }
            #[inline]
            fn round_even(self) -> $t {
                // Ties to even (§6.7), NaN/inf pass through, sign of zero
                // preserved (§6.7-0002).
                if self.is_nan() || self == <$t>::INFINITY || self == <$t>::NEG_INFINITY {
                    return self;
                }
                let f = $floor(self);
                let d = self - f;
                let r = if d < 0.5 {
                    f
                } else if d > 0.5 {
                    f + 1.0
                } else {
                    // exact tie: pick the even neighbor
                    let half = f * 0.5;
                    if $floor(half) * 2.0 == f {
                        f
                    } else {
                        f + 1.0
                    }
                };
                // Restore sign of zero: round_even(-0.5) = -0.0, round_even(-0.0) = -0.0.
                if r == 0.0 && self.is_sign_negative() {
                    -0.0
                } else {
                    r
                }
            }

            #[inline]
            fn exp(self) -> $t {
                $exp(self)
            }
            #[inline]
            fn log(self) -> $t {
                $log(self)
            }
            #[inline]
            fn sin(self) -> $t {
                $sin(self)
            }
            #[inline]
            fn cos(self) -> $t {
                $cos(self)
            }
            #[inline]
            fn sqrt(self) -> $t {
                $sqrt(self)
            }
            #[inline]
            fn erf(self) -> $t {
                $erf(self)
            }
            #[inline]
            fn atan(self) -> $t {
                $atan(self)
            }
            #[inline]
            fn lgamma(self) -> $t {
                $lgamma(self)
            }

            #[inline]
            fn atan2(self, x: $t) -> $t {
                $atan2(self, x)
            }
            #[inline]
            fn copysign(self, sign: $t) -> $t {
                $copysign(self, sign)
            }
            #[inline]
            fn nextafter(self, to: $t) -> $t {
                $nextafter(self, to)
            }

            #[inline]
            fn tanh(self) -> $t {
                $tanh(self)
            }
            #[inline]
            fn sinh(self) -> $t {
                $sinh(self)
            }
            #[inline]
            fn cosh(self) -> $t {
                $cosh(self)
            }
            #[inline]
            fn log1p(self) -> $t {
                $log1p(self)
            }
            #[inline]
            fn expm1(self) -> $t {
                $expm1(self)
            }
            #[inline]
            fn pow(self, b: $t) -> $t {
                $pow(self, b)
            }
            #[inline]
            fn hypot(self, b: $t) -> $t {
                $hypot(self, b)
            }
        }
    };
}

impl_scalar_float!(
    f64,
    dtype = Dtype::F64,
    bits = u64,
    exp = libm::exp,
    log = libm::log,
    sin = libm::sin,
    cos = libm::cos,
    sqrt = libm::sqrt,
    erf = libm::erf,
    atan = libm::atan,
    lgamma = libm::lgamma,
    atan2 = libm::atan2,
    copysign = libm::copysign,
    nextafter = libm::nextafter,
    floor = libm::floor,
    ceil = libm::ceil,
    trunc = libm::trunc,
    fabs = libm::fabs,
    tanh = libm::tanh,
    sinh = libm::sinh,
    cosh = libm::cosh,
    log1p = libm::log1p,
    expm1 = libm::expm1,
    pow = libm::pow,
    hypot = libm::hypot
);

impl_scalar_float!(
    f32,
    dtype = Dtype::F32,
    bits = u32,
    exp = libm::expf,
    log = libm::logf,
    sin = libm::sinf,
    cos = libm::cosf,
    sqrt = libm::sqrtf,
    erf = libm::erff,
    atan = libm::atanf,
    lgamma = libm::lgammaf,
    atan2 = libm::atan2f,
    copysign = libm::copysignf,
    nextafter = libm::nextafterf,
    floor = libm::floorf,
    ceil = libm::ceilf,
    trunc = libm::truncf,
    fabs = libm::fabsf,
    tanh = libm::tanhf,
    sinh = libm::sinhf,
    cosh = libm::coshf,
    log1p = libm::log1pf,
    expm1 = libm::expm1f,
    pow = libm::powf,
    hypot = libm::hypotf
);

// ---- Narrow floats (f16 / bf16) ----------------------------------------------
//
// Computed by promoting to f32, evaluating with the f32 reference, then rounding
// back to the narrow type. For `+ - *` this is correctly-rounded to the narrow
// format (the exact result of two narrow operands fits in f32); for the
// transcendentals it lands well inside the coarse narrow-dtype ULP ceiling; `bf16`
// has no native arithmetic and is *defined* to compute in f32 (§6.16). Caveat:
// narrow `div` double-rounds (f32-correct then narrowed), so it can sit ≤1 ULP from a
// single-rounded (correctly-rounded-to-narrow) divide — and it is classed
// `ExactByte`, so a candidate that divides directly in narrow precision would deviate
// by that ≤1 ULP. This is the compute-in-f32 reference model applied uniformly.
// OPEN (routed to KISS — not a silent "revisited later"): does §6.16's compute-in-f32
// rule bind narrow `div` (⇒ the double-round is conformant) or must narrow `div` be
// correctly-rounded (⇒ fix via a round-to-odd f32 intermediate)?
// `nextafter` is declined for these types by the resolver (§6.9-0003), so its
// f32-promoted body here is never reached through `eval_op`.
macro_rules! impl_scalar_float_via_f32 {
    ($t:ty, $dtype:expr, $uint:ty) => {
        impl ScalarFloat for $t {
            const ZERO: $t = <$t>::ZERO;
            const ONE: $t = <$t>::ONE;
            const NARROW_FLOAT: bool = true;
            const DTYPE: Dtype = $dtype;

            #[inline]
            fn from_f64(v: f64) -> $t {
                <$t>::from_f32(v as f32)
            }
            #[inline]
            fn from_bits(bits: u64) -> $t {
                <$t>::from_bits(bits as $uint)
            }
            #[inline]
            fn from_f64_single_round(v: f64) -> $t {
                <$t>::from_f32(round_to_odd_f64_to_f32(v))
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self.to_f32() as f64
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }

            #[inline]
            fn add(self, b: $t) -> $t {
                <$t>::from_f32(self.to_f32() + b.to_f32())
            }
            #[inline]
            fn sub(self, b: $t) -> $t {
                <$t>::from_f32(self.to_f32() - b.to_f32())
            }
            #[inline]
            fn mul(self, b: $t) -> $t {
                <$t>::from_f32(self.to_f32() * b.to_f32())
            }
            #[inline]
            fn div(self, b: $t) -> $t {
                <$t>::from_f32(self.to_f32() / b.to_f32())
            }
            #[inline]
            fn neg(self) -> $t {
                <$t>::from_f32(-self.to_f32())
            }
            #[inline]
            fn abs(self) -> $t {
                <$t>::from_f32(ScalarFloat::abs(self.to_f32()))
            }

            #[inline]
            fn floor(self) -> $t {
                <$t>::from_f32(ScalarFloat::floor(self.to_f32()))
            }
            #[inline]
            fn ceil(self) -> $t {
                <$t>::from_f32(ScalarFloat::ceil(self.to_f32()))
            }
            #[inline]
            fn trunc(self) -> $t {
                <$t>::from_f32(ScalarFloat::trunc(self.to_f32()))
            }
            #[inline]
            fn round_even(self) -> $t {
                <$t>::from_f32(ScalarFloat::round_even(self.to_f32()))
            }

            #[inline]
            fn exp(self) -> $t {
                <$t>::from_f32(ScalarFloat::exp(self.to_f32()))
            }
            #[inline]
            fn log(self) -> $t {
                <$t>::from_f32(ScalarFloat::log(self.to_f32()))
            }
            #[inline]
            fn sin(self) -> $t {
                <$t>::from_f32(ScalarFloat::sin(self.to_f32()))
            }
            #[inline]
            fn cos(self) -> $t {
                <$t>::from_f32(ScalarFloat::cos(self.to_f32()))
            }
            #[inline]
            fn sqrt(self) -> $t {
                <$t>::from_f32(ScalarFloat::sqrt(self.to_f32()))
            }
            #[inline]
            fn erf(self) -> $t {
                <$t>::from_f32(ScalarFloat::erf(self.to_f32()))
            }
            #[inline]
            fn atan(self) -> $t {
                <$t>::from_f32(ScalarFloat::atan(self.to_f32()))
            }
            #[inline]
            fn lgamma(self) -> $t {
                <$t>::from_f32(ScalarFloat::lgamma(self.to_f32()))
            }

            #[inline]
            fn atan2(self, x: $t) -> $t {
                <$t>::from_f32(ScalarFloat::atan2(self.to_f32(), x.to_f32()))
            }
            #[inline]
            fn copysign(self, sign: $t) -> $t {
                <$t>::from_f32(ScalarFloat::copysign(self.to_f32(), sign.to_f32()))
            }
            #[inline]
            fn nextafter(self, to: $t) -> $t {
                // Declined by the resolver for narrow floats (§6.9-0003); this
                // f32-promoted body is never reached through eval_op.
                <$t>::from_f32(ScalarFloat::nextafter(self.to_f32(), to.to_f32()))
            }

            #[inline]
            fn tanh(self) -> $t {
                <$t>::from_f32(ScalarFloat::tanh(self.to_f32()))
            }
            #[inline]
            fn sinh(self) -> $t {
                <$t>::from_f32(ScalarFloat::sinh(self.to_f32()))
            }
            #[inline]
            fn cosh(self) -> $t {
                <$t>::from_f32(ScalarFloat::cosh(self.to_f32()))
            }
            #[inline]
            fn log1p(self) -> $t {
                <$t>::from_f32(ScalarFloat::log1p(self.to_f32()))
            }
            #[inline]
            fn expm1(self) -> $t {
                <$t>::from_f32(ScalarFloat::expm1(self.to_f32()))
            }
            #[inline]
            fn pow(self, b: $t) -> $t {
                <$t>::from_f32(ScalarFloat::pow(self.to_f32(), b.to_f32()))
            }
            #[inline]
            fn hypot(self, b: $t) -> $t {
                <$t>::from_f32(ScalarFloat::hypot(self.to_f32(), b.to_f32()))
            }
        }
    };
}

impl_scalar_float_via_f32!(half::f16, Dtype::F16, u16);
impl_scalar_float_via_f32!(half::bf16, Dtype::Bf16, u16);

// FP8 (e4m3 / e5m2) — same promote-to-f32 lane as the narrow floats (§6.16), with
// the hand-rolled u8 codec in `crate::fp8` (the `half` crate has no FP8).
impl_scalar_float_via_f32!(crate::fp8::E4m3, Dtype::F8e4m3fn, u8);
impl_scalar_float_via_f32!(crate::fp8::E5m2, Dtype::F8e5m2, u8);

/// **Exact** promotion of a storage value `S` into a wider accumulator dtype `A`
/// (RFC #92 direction b). `to_f64` is lossless for every float (all ⊆ `f64`), and
/// `A::from_f64` is lossless whenever `A` subsumes `S` — which the accumulator
/// width guard (`kernels::guard_accumulator`) enforces before this is ever
/// called, so the promotion never rounds. On the diagonal `A == S` this is the
/// round-trip identity `S::from_f64(x.to_f64()) == x` for every finite value.
#[inline]
pub(crate) fn widen<S: ScalarFloat, A: ScalarFloat>(x: S) -> A {
    A::from_f64(x.to_f64())
}

/// Smallest `f32` strictly greater than a FINITE `x`. At the top finite this
/// yields +inf, which `round_to_odd_f64_to_f32` never selects (that x is odd).
#[inline]
fn next_up_f32(x: f32) -> f32 {
    let b = x.to_bits();
    if x > 0.0 {
        f32::from_bits(b + 1) // magnitude up
    } else if x < 0.0 {
        f32::from_bits(b - 1) // toward zero
    } else {
        f32::from_bits(1) // ±0 → +2^-149
    }
}

/// Largest `f32` strictly less than a FINITE `x` (mirror of `next_up_f32`).
#[inline]
fn next_down_f32(x: f32) -> f32 {
    let b = x.to_bits();
    if x > 0.0 {
        f32::from_bits(b - 1) // toward zero
    } else if x < 0.0 {
        f32::from_bits(b + 1) // more negative
    } else {
        f32::from_bits(0x8000_0001) // ±0 → -2^-149
    }
}

/// Round `x` from `f64` to `f32` with ROUND-TO-ODD (von Neumann / sticky
/// rounding). Composed with a following RNE `f32→narrow`, this equals a single
/// correctly-rounded RNE `f64→narrow` (double-rounding-is-odd; f32's 24
/// significand bits ≥ 2m+2 for every narrow mantissa m ≤ 10). It is the EXACT
/// identity on every `f32`-representable value, so routing all narrowings through
/// it leaves every accumulator `A ⊆ f32` cell byte-unchanged.
#[inline]
fn round_to_odd_f64_to_f32(x: f64) -> f32 {
    if !x.is_finite() {
        return x as f32; // ±inf / NaN: propagate; narrow from_f32 applies §6.16.
    }
    let r = x as f32; // RNE, ties-to-even
    if (r as f64) == x {
        return r; // x is exactly an f32 (incl. ±0 / exact finite): identity.
    }
    if !r.is_finite() {
        // A finite x overflowed the f32 cast: round-to-odd never overflows a
        // finite operand — return the FINITE ±MAX (all-ones mantissa is odd). This
        // is what lets each narrow from_f32 apply the CORRECT §6.16 overflow rule:
        // e5m2 finite-overflow SATURATES to 57344 (a true f64 inf would instead
        // reach here via the !is_finite branch above and yield inf), e4m3 → 448,
        // f16/bf16 → their own inf.
        return if x < 0.0 { f32::MIN } else { f32::MAX };
    }
    // r is one of the two consecutive f32s bracketing x; the OTHER bracket is r's
    // neighbor toward x. Exactly one of two consecutive f32s is odd — return it.
    let other = if (r as f64) < x {
        next_up_f32(r)
    } else {
        next_down_f32(r)
    };
    if (r.to_bits() & 1) == 1 {
        r
    } else {
        other
    }
}

/// Narrow an accumulator value `A` back to the storage/result dtype `S` (RFC #92
/// C3 "…and the result rounded to S"). A **single** correctly-rounded RNE for
/// every `(A, S)` pair the accumulator width guard admits: when `A ⊆ f32` (which
/// the guard guarantees for a narrow `S`), `a.to_f64()` is exactly an `f32`, and
/// `from_f64_single_round` collapses to one `from_f32` round — byte-identical to
/// the former `S::from_f64` path. The only accumulator wider than the narrow
/// pivot, `A = f64` narrowed to a via-`f32` narrow `S`, would double-round
/// (`f64→f32→S`); `from_f64_single_round` avoids it with round-to-odd `f64→f32`
/// then the existing RNE `f32→S` codec (double-rounding-is-odd), so this is a
/// single correctly-rounded RNE there too.
#[inline]
pub(crate) fn narrow<A: ScalarFloat, S: ScalarFloat>(a: A) -> S {
    S::from_f64_single_round(a.to_f64())
}

// ---- Refined non-primitive scalar forms (§6.13-0003) --------------------------
// The literal reference decompositions of these ops overflow to NaN for large
// arguments while the true function is finite; the spec REQUIRES an overflow-safe
// form that agrees with the reference's mathematical meaning within the declared
// ULP. `tanh`/`sinh`/`cosh`/`log1p` are on the trait; `sigmoid`/`softplus` and
// their dependents are composed here.

/// Numerically stable logistic sigmoid `1/(1+e^-x)` (used by `silu`).
#[inline]
pub fn sigmoid_stable<T: ScalarFloat>(x: T) -> T {
    if x.to_f64() >= 0.0 {
        let e = x.neg().exp();
        T::ONE.div(T::ONE.add(e))
    } else {
        let e = x.exp();
        e.div(T::ONE.add(e))
    }
}

/// Overflow-safe `softplus`: `max(x,0) + log1p(exp(-|x|))` (§6.13-0003 — the
/// literal `log(1+exp(x))` overflows to `+inf` for large `x`).
#[inline]
pub fn softplus_stable<T: ScalarFloat>(x: T) -> T {
    let m = if x.to_f64() > 0.0 { x } else { T::ZERO };
    m.add(x.abs().neg().exp().log1p())
}

/// `silu(x) = x * sigmoid(x)`, with the stable sigmoid.
#[inline]
pub fn silu_stable<T: ScalarFloat>(x: T) -> T {
    x.mul(sigmoid_stable(x))
}

/// `mish(x) = x * tanh(softplus(x))`, inheriting the overflow-safe parts.
#[inline]
pub fn mish_stable<T: ScalarFloat>(x: T) -> T {
    x.mul(softplus_stable(x).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp8::{E4m3, E5m2};

    #[test]
    fn round_to_odd_is_identity_on_f32_values() {
        // Every exact f32 value comes back byte-unchanged (the acc ⊆ f32 property
        // that keeps every accumulator-≤-f32 narrowing cell byte-identical).
        for &v in &[
            0.0f32,
            -0.0,
            1.0,
            -3.5,
            57344.0,
            f32::MIN_POSITIVE,
            f32::MAX,
        ] {
            assert_eq!(
                round_to_odd_f64_to_f32(v as f64).to_bits(),
                v.to_bits(),
                "{v}"
            );
        }
        assert!(round_to_odd_f64_to_f32(f64::NAN).is_nan());
        assert_eq!(round_to_odd_f64_to_f32(f64::INFINITY), f32::INFINITY);
        assert_eq!(
            round_to_odd_f64_to_f32(f64::NEG_INFINITY),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn round_to_odd_f64_to_f32_overflow_saturates_correctly() {
        // THE DISCRIMINATOR (design open-point 2): a FINITE f64 above f32::MAX
        // narrows to e5m2 by SATURATION to ±57344 (§6.16-0005), NOT inf — because
        // round-to-odd returns the finite ±f32::MAX for a finite operand. Only a
        // true f64 inf yields e5m2 inf.
        assert_eq!(
            E5m2::from_f32(round_to_odd_f64_to_f32(1e300)).to_bits(),
            0x7B
        ); // 57344
        assert_eq!(
            E5m2::from_f32(round_to_odd_f64_to_f32(-1e300)).to_bits(),
            0xFB
        ); // -57344
        assert_eq!(
            E5m2::from_f32(round_to_odd_f64_to_f32(f64::INFINITY)).to_bits(),
            0x7C
        ); // inf
        assert_eq!(
            E5m2::from_f32(round_to_odd_f64_to_f32(f64::NEG_INFINITY)).to_bits(),
            0xFC
        ); // -inf
           // e4m3fn has no inf encoding — finite-overflow AND inf both saturate to ±448.
        assert_eq!(
            E4m3::from_f32(round_to_odd_f64_to_f32(1e300)).to_bits(),
            0x7E
        ); // 448
        assert_eq!(
            E4m3::from_f32(round_to_odd_f64_to_f32(f64::INFINITY)).to_bits(),
            0x7E
        );
    }

    #[test]
    fn round_to_odd_then_rne_is_a_single_f64_to_f16_round() {
        // R = 1 + 2^-11 + 2^-24 rounds UP to 1 + 2^-10 under a single RNE f64→f16,
        // where a naive f64→f32→f16 double round collapses to 1.0. Verified through
        // the trait entry point used by narrow().
        let r = 1.0f64 + 2f64.powi(-11) + 2f64.powi(-24);
        let single = half::f16::from_f64_single_round(r);
        assert_eq!(
            single.to_bits(),
            half::f16::from_f32(1.0 + 2f32.powi(-10)).to_bits()
        );
        assert_ne!(single.to_bits(), half::f16::from_f32(1.0).to_bits());
    }
}
