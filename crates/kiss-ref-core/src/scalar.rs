//! Scalar reference implementations of the floating-point floor atoms
//! (KISS-Ops §6.4–§6.9), plus the overflow-safe refined forms (§6.13-0003) the
//! resolver substitutes for the overflow-unsafe non-primitive decompositions.
//!
//! Compute happens **in the compute dtype** (KISS-OPS-6.2-0001): `f32` atoms use
//! the `f32` `libm` variants, `f64` atoms the `f64` variants, so each is the
//! IEEE-754 reference for its own dtype. Exact atoms (`+ - * /`, comparisons,
//! `select`, rounding, `copysign`) use native operators / `libm` directed
//! rounding; the transcendentals go through `libm` under their §6.8 declared-ULP.

/// A floating-point compute dtype the reference evaluates atoms in. Implemented
/// for `f32` and `f64` (the dtypes whose native Rust arithmetic *is* the
/// IEEE-754 reference); `f16`/`bf16`/FP8 are a documented follow-up.
pub trait ScalarFloat: Copy + PartialEq + PartialOrd {
    const ZERO: Self;
    const ONE: Self;
    /// True for the narrow floats (`f16`/`bf16`). Used by the resolver to
    /// **decline `nextafter`** on them per §6.9-0003 (stepping in a promoted
    /// `f32` yields the wrong neighbor in the narrow lattice).
    const NARROW_FLOAT: bool = false;

    /// Round an `f64` reference constant into this dtype (§6.12 `const(bits)`).
    fn from_f64(v: f64) -> Self;
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
    ($t:ty,
     exp=$exp:path, log=$log:path, sin=$sin:path, cos=$cos:path, sqrt=$sqrt:path,
     erf=$erf:path, atan=$atan:path, lgamma=$lgamma:path, atan2=$atan2:path,
     copysign=$copysign:path, nextafter=$nextafter:path, floor=$floor:path,
     ceil=$ceil:path, trunc=$trunc:path, fabs=$fabs:path, tanh=$tanh:path,
     sinh=$sinh:path, cosh=$cosh:path, log1p=$log1p:path, expm1=$expm1:path,
     pow=$pow:path, hypot=$hypot:path) => {
        impl ScalarFloat for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;

            #[inline]
            fn from_f64(v: f64) -> $t {
                v as $t
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
    exp = libm::exp, log = libm::log, sin = libm::sin, cos = libm::cos, sqrt = libm::sqrt,
    erf = libm::erf, atan = libm::atan, lgamma = libm::lgamma, atan2 = libm::atan2,
    copysign = libm::copysign, nextafter = libm::nextafter, floor = libm::floor,
    ceil = libm::ceil, trunc = libm::trunc, fabs = libm::fabs, tanh = libm::tanh,
    sinh = libm::sinh, cosh = libm::cosh, log1p = libm::log1p, expm1 = libm::expm1,
    pow = libm::pow, hypot = libm::hypot
);

impl_scalar_float!(
    f32,
    exp = libm::expf, log = libm::logf, sin = libm::sinf, cos = libm::cosf, sqrt = libm::sqrtf,
    erf = libm::erff, atan = libm::atanf, lgamma = libm::lgammaf, atan2 = libm::atan2f,
    copysign = libm::copysignf, nextafter = libm::nextafterf, floor = libm::floorf,
    ceil = libm::ceilf, trunc = libm::truncf, fabs = libm::fabsf, tanh = libm::tanhf,
    sinh = libm::sinhf, cosh = libm::coshf, log1p = libm::log1pf, expm1 = libm::expm1f,
    pow = libm::powf, hypot = libm::hypotf
);

// ---- Narrow floats (f16 / bf16) ----------------------------------------------
//
// Computed by promoting to f32, evaluating with the f32 reference, then rounding
// back to the narrow type. For `+ - *` this is correctly-rounded to the narrow
// format (the exact result of two narrow operands fits in f32); for the
// transcendentals it lands well inside the coarse narrow-dtype ULP ceiling; `bf16`
// has no native arithmetic and is *defined* to compute in f32 (§6.16). Caveat:
// narrow `div` is a double-rounding (f32-correct then narrowed) — occasionally 1
// ULP off a true correctly-rounded narrow divide; documented, revisited later.
// `nextafter` is declined for these types by the resolver (§6.9-0003), so its
// f32-promoted body here is never reached through `eval_op`.
macro_rules! impl_scalar_float_via_f32 {
    ($t:ty) => {
        impl ScalarFloat for $t {
            const ZERO: $t = <$t>::ZERO;
            const ONE: $t = <$t>::ONE;
            const NARROW_FLOAT: bool = true;

            #[inline]
            fn from_f64(v: f64) -> $t {
                <$t>::from_f32(v as f32)
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

impl_scalar_float_via_f32!(half::f16);
impl_scalar_float_via_f32!(half::bf16);

// FP8 (e4m3 / e5m2) — same promote-to-f32 lane as the narrow floats (§6.16), with
// the hand-rolled u8 codec in `crate::fp8` (the `half` crate has no FP8).
impl_scalar_float_via_f32!(crate::fp8::E4m3);
impl_scalar_float_via_f32!(crate::fp8::E5m2);

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
