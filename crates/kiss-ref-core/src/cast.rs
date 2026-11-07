// SPDX-License-Identifier: MIT OR Apache-2.0
//! The conversion atom `cast` (KISS-Ops §6.21) and the join `concat` (§6.11-0020).
//!
//! `cast` works on **bit patterns** (`u64`, low `width` bits significant; integers in two's
//! complement), so every element lane (the float tensors, the `i128` integer lane, the FP8 and
//! MX bytes) can feed it. Every source is decoded to an *exact* value `±mant * 2^exp` and rounded
//! **once** to the destination (§6.21-0002): no intermediate dtype is ever staged.
//!
//! Decisions for what the clauses leave open, each a typed decline rather than a guess:
//! * a NaN into a destination with no NaN (the MX element types), a NaN into an integer, and an
//!   out-of-range float (infinity included) into an integer under `saturation = none` are the
//!   Appendix D item 12 cases, [`Error::CastUnpinned`] (§6.21-0004);
//! * a complex dtype is [`Error::UnsupportedDtype`] (§6.21-0001); a reserved or scale dtype is
//!   [`Error::ReservedOrScaleDtype`];
//! * `bool` as a **destination** is `UnsupportedDtype`: neither §6.21-0002 nor -0003 states a
//!   float-to-truth-value rule. As a source it is the integer 0 or 1.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

use crate::scalar_int::int_spec;
use crate::tensor::{alloc_exact, numel, Odometer, Tensor, View, MAX_OPERANDS, MAX_RANK};
use crate::Error;

/// The `rounding` attribute (§6.21-0002; ordinals 1..=4 of `cast_rounding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastRounding {
    /// To nearest, ties to even (ordinal 1, the default).
    NearestEven,
    /// Toward zero (2).
    TowardZero,
    /// Toward negative infinity (3).
    TowardNegInf,
    /// Toward positive infinity (4).
    TowardPosInf,
}

impl CastRounding {
    /// The frozen `u8` ordinal, or `None` for 0 and the unassigned values.
    pub const fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            1 => Some(CastRounding::NearestEven),
            2 => Some(CastRounding::TowardZero),
            3 => Some(CastRounding::TowardNegInf),
            4 => Some(CastRounding::TowardPosInf),
            _ => None,
        }
    }
}

/// The `saturation` attribute (§6.21-0003; ordinals 1..=2 of `cast_saturation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastSaturation {
    /// No saturation: the rounding direction decides (1).
    None,
    /// Clamp to the largest finite magnitude or the nearest integer bound (2).
    Clamp,
}

impl CastSaturation {
    /// The frozen `u8` ordinal, or `None` for 0 and the unassigned values.
    pub const fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            1 => Some(CastSaturation::None),
            2 => Some(CastSaturation::Clamp),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NanKind {
    /// IEEE: all-ones exponent, non-zero mantissa.
    Ieee,
    /// `f8e4m3fn`: all-ones exponent and all-ones mantissa, no infinity.
    AllOnes,
    /// The MX element types: no NaN, no infinity.
    Absent,
}

#[derive(Clone, Copy)]
struct FloatFmt {
    exp_bits: u32,
    man_bits: u32,
    bias: i32,
    has_inf: bool,
    nan: NanKind,
}

impl FloatFmt {
    const fn width(self) -> u32 {
        1 + self.exp_bits + self.man_bits
    }
    fn max_finite_fields(self) -> (u64, u64) {
        let all_e = (1u64 << self.exp_bits) - 1;
        let all_m = (1u64 << self.man_bits) - 1;
        match self.nan {
            NanKind::Ieee => (all_e - 1, all_m),
            NanKind::AllOnes => (all_e, all_m - 1),
            NanKind::Absent => (all_e, all_m),
        }
    }
    fn pack(self, neg: bool, e: u64, m: u64) -> u64 {
        (u64::from(neg) << (self.width() - 1)) | (e << self.man_bits) | m
    }
    fn max_finite(self, neg: bool) -> u64 {
        let (e, m) = self.max_finite_fields();
        self.pack(neg, e, m)
    }
}

fn float_fmt(d: Dtype) -> Option<FloatFmt> {
    let f = |exp_bits, man_bits, bias, has_inf, nan| FloatFmt {
        exp_bits,
        man_bits,
        bias,
        has_inf,
        nan,
    };
    Some(match d {
        Dtype::F16 => f(5, 10, 15, true, NanKind::Ieee),
        Dtype::Bf16 => f(8, 7, 127, true, NanKind::Ieee),
        Dtype::F32 => f(8, 23, 127, true, NanKind::Ieee),
        Dtype::F64 => f(11, 52, 1023, true, NanKind::Ieee),
        Dtype::F8e5m2 => f(5, 2, 15, true, NanKind::Ieee),
        Dtype::F8e4m3fn => f(4, 3, 7, false, NanKind::AllOnes),
        Dtype::F4e2m1 => f(2, 1, 1, false, NanKind::Absent),
        Dtype::F6e2m3 => f(2, 3, 1, false, NanKind::Absent),
        Dtype::F6e3m2 => f(3, 2, 3, false, NanKind::Absent),
        _ => return None,
    })
}

/// A decoded source value. `Fin` with `mant == 0` is a (signed) zero.
enum Exact {
    /// A NaN: its sign and its mantissa left-aligned in 64 bits (payload carried best effort).
    Nan {
        neg: bool,
        man_top: u64,
    },
    Inf {
        neg: bool,
    },
    Fin {
        neg: bool,
        mant: u128,
        exp: i32,
    },
}

fn decode_float(bits: u64, f: FloatFmt) -> Exact {
    let neg = (bits >> (f.width() - 1)) & 1 == 1;
    let e = (bits >> f.man_bits) & ((1u64 << f.exp_bits) - 1);
    let m = bits & ((1u64 << f.man_bits) - 1);
    let all_e = (1u64 << f.exp_bits) - 1;
    let man_top = m << (64 - f.man_bits);
    if e == all_e {
        match f.nan {
            NanKind::Ieee if m == 0 => return Exact::Inf { neg },
            NanKind::Ieee => return Exact::Nan { neg, man_top },
            NanKind::AllOnes if m == (1u64 << f.man_bits) - 1 => {
                return Exact::Nan { neg, man_top }
            }
            _ => {}
        }
    }
    let mb = f.man_bits as i32;
    if e == 0 {
        Exact::Fin {
            neg,
            mant: u128::from(m),
            exp: 1 - f.bias - mb,
        }
    } else {
        Exact::Fin {
            neg,
            mant: u128::from(m | (1u64 << f.man_bits)),
            exp: e as i32 - f.bias - mb,
        }
    }
}

fn int_value(bits: u64, width: u32, signed: bool) -> i128 {
    let m = if width >= 64 {
        bits
    } else {
        bits & ((1u64 << width) - 1)
    };
    if signed && width < 128 && (m >> (width - 1)) & 1 == 1 {
        i128::from(m) - (1i128 << width)
    } else {
        i128::from(m)
    }
}

fn decode(bits: u64, d: Dtype) -> Result<(Exact, bool), Error> {
    if let Some(f) = float_fmt(d) {
        return Ok((decode_float(bits, f), false));
    }
    let v = if d == Dtype::Bool {
        i128::from(bits & 0xFF != 0)
    } else {
        let (w, s) = int_spec(d).ok_or(Error::UnsupportedDtype(d))?;
        int_value(bits, w, s)
    };
    Ok((
        Exact::Fin {
            neg: v < 0,
            mant: v.unsigned_abs(),
            exp: 0,
        },
        true,
    ))
}

/// What was shifted out, against one half unit in the last place.
#[derive(PartialEq, Eq)]
enum Rem {
    Zero,
    Below,
    Half,
    Above,
}

/// Divide `mant` by `2^shift`, rounding the **magnitude** per `r` for a value of sign `neg`.
fn round_shift(mant: u128, shift: u32, neg: bool, r: CastRounding) -> u128 {
    let (q, rem) = if shift >= 128 {
        (0, if mant == 0 { Rem::Zero } else { Rem::Below })
    } else {
        let low = mant & ((1u128 << shift) - 1);
        let rem = if low == 0 {
            Rem::Zero
        } else {
            match low.cmp(&(1u128 << (shift - 1))) {
                core::cmp::Ordering::Less => Rem::Below,
                core::cmp::Ordering::Equal => Rem::Half,
                core::cmp::Ordering::Greater => Rem::Above,
            }
        };
        (mant >> shift, rem)
    };
    let bump = rem != Rem::Zero
        && match r {
            CastRounding::NearestEven => rem == Rem::Above || (rem == Rem::Half && q & 1 == 1),
            CastRounding::TowardZero => false,
            CastRounding::TowardNegInf => neg,
            CastRounding::TowardPosInf => !neg,
        };
    q + u128::from(bump)
}

fn zero_bits(f: FloatFmt, neg: bool) -> u64 {
    f.pack(neg, 0, 0)
}

fn overflow_bits(f: FloatFmt, neg: bool, r: CastRounding, s: CastSaturation) -> Result<u64, Error> {
    if s == CastSaturation::Clamp {
        return Ok(f.max_finite(neg));
    }
    let toward_inf = match r {
        CastRounding::NearestEven => true,
        CastRounding::TowardZero => false,
        CastRounding::TowardNegInf => neg,
        CastRounding::TowardPosInf => !neg,
    };
    if !toward_inf {
        Ok(f.max_finite(neg))
    } else if f.has_inf {
        Ok(f.pack(neg, (1u64 << f.exp_bits) - 1, 0))
    } else {
        Err(Error::CastUnpinned)
    }
}

fn round_to_float(
    neg: bool,
    mant: u128,
    exp: i32,
    f: FloatFmt,
    r: CastRounding,
    s: CastSaturation,
) -> Result<u64, Error> {
    if mant == 0 {
        return Ok(zero_bits(f, neg));
    }
    let mb = f.man_bits as i32;
    let bl = 128 - mant.leading_zeros() as i32;
    let qe = (bl - 1 + exp).max(1 - f.bias) - mb;
    let shift = qe - exp;
    let mut m = if shift <= 0 {
        mant << (-shift) as u32
    } else {
        round_shift(mant, shift as u32, neg, r)
    };
    let mut qe = qe;
    if m >> (mb + 1) != 0 {
        m >>= 1;
        qe += 1;
    }
    if m == 0 {
        return Ok(zero_bits(f, neg));
    }
    let (biased, man) = if m < (1u128 << mb) {
        (0u64, m as u64)
    } else {
        ((qe + mb + f.bias) as u64, (m - (1u128 << mb)) as u64)
    };
    let (max_e, max_m) = f.max_finite_fields();
    if biased > max_e || (biased == max_e && man > max_m) {
        return overflow_bits(f, neg, r, s);
    }
    Ok(f.pack(neg, biased, man))
}

fn encode_float(ex: Exact, f: FloatFmt, r: CastRounding, s: CastSaturation) -> Result<u64, Error> {
    match ex {
        Exact::Nan { neg, man_top } => match f.nan {
            NanKind::Absent => Err(Error::CastUnpinned),
            NanKind::AllOnes => Ok(f.pack(neg, (1u64 << f.exp_bits) - 1, (1u64 << f.man_bits) - 1)),
            NanKind::Ieee => {
                let quiet = 1u64 << (f.man_bits - 1);
                let m = (man_top >> (64 - f.man_bits)) | quiet;
                Ok(f.pack(neg, (1u64 << f.exp_bits) - 1, m))
            }
        },
        Exact::Inf { neg } => {
            if s == CastSaturation::Clamp {
                Ok(f.max_finite(neg))
            } else if f.has_inf {
                Ok(f.pack(neg, (1u64 << f.exp_bits) - 1, 0))
            } else {
                Err(Error::CastUnpinned)
            }
        }
        Exact::Fin { neg, mant, exp } => round_to_float(neg, mant, exp, f, r, s),
    }
}

/// The integer a finite value rounds to (`None` when it is too large for `i128`).
fn rounded_integer(neg: bool, mant: u128, exp: i32, r: CastRounding) -> Option<i128> {
    let mag = if exp >= 0 {
        let bl = 128 - mant.leading_zeros() as i32;
        if mant != 0 && bl + exp > 100 {
            return None;
        }
        mant << exp as u32
    } else {
        round_shift(mant, exp.unsigned_abs(), neg, r)
    };
    let v = i128::try_from(mag).ok()?;
    Some(if neg { -v } else { v })
}

fn encode_int(
    ex: Exact,
    (width, signed): (u32, bool),
    from_int: bool,
    r: CastRounding,
    s: CastSaturation,
) -> Result<u64, Error> {
    let (lo, hi) = if signed {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    let (neg, v) = match ex {
        Exact::Nan { .. } => return Err(Error::CastUnpinned),
        Exact::Inf { neg } => (neg, None),
        Exact::Fin { neg, mant, exp } => (neg, rounded_integer(neg, mant, exp, r)),
    };
    let mask = (1u128 << width) - 1;
    match v {
        Some(x) if (lo..=hi).contains(&x) => Ok((x as u128 & mask) as u64),
        _ if s == CastSaturation::Clamp => {
            let b = match v {
                Some(x) => {
                    if x < lo {
                        lo
                    } else {
                        hi
                    }
                }
                None => {
                    if neg {
                        lo
                    } else {
                        hi
                    }
                }
            };
            Ok((b as u128 & mask) as u64)
        }
        // saturation = none: integer sources wrap modulo 2^width; a float is unpinned (§6.21-0004).
        Some(x) if from_int => Ok((x as u128 & mask) as u64),
        _ => Err(Error::CastUnpinned),
    }
}

fn check_dtype(d: Dtype) -> Result<(), Error> {
    if d.is_complex() {
        Err(Error::UnsupportedDtype(d))
    } else if d.declines_compute() {
        Err(Error::ReservedOrScaleDtype(d))
    } else {
        Ok(())
    }
}

/// Whether `dst` pins nearest-even and clamp itself (§6.21-0002, §6.21-0003).
fn pins_rounding_and_saturation(dst: Dtype) -> bool {
    matches!(
        dst,
        Dtype::F8e4m3fn | Dtype::F8e5m2 | Dtype::F4e2m1 | Dtype::F6e2m3 | Dtype::F6e3m2
    )
}

/// The typed declines of §6.21-0001..-0003 for a `(src, dst, rounding, saturation)` request,
/// decidable without any value.
pub fn check_cast(
    src: Dtype,
    dst: Dtype,
    rounding: CastRounding,
    saturation: CastSaturation,
) -> Result<(), Error> {
    check_dtype(src)?;
    check_dtype(dst)?;
    if dst == Dtype::Bool {
        return Err(Error::UnsupportedDtype(dst));
    }
    if pins_rounding_and_saturation(dst) {
        if rounding != CastRounding::NearestEven {
            return Err(Error::CastRoundingNotDefined(dst));
        }
        if saturation != CastSaturation::Clamp {
            return Err(Error::CastSaturationRequired(dst));
        }
    }
    Ok(())
}

/// Convert one element's bit pattern from `src` to `dst` (§6.21). Floats are IEEE-style bit
/// patterns of their width; integers are two's complement in the low `width` bits; `bool` is 0/1.
/// `src == dst` is the identity and returns `bits` unchanged (masked to the width).
pub fn cast_bits(
    bits: u64,
    src: Dtype,
    dst: Dtype,
    rounding: CastRounding,
    saturation: CastSaturation,
) -> Result<u64, Error> {
    check_cast(src, dst, rounding, saturation)?;
    let (ex, from_int) = decode(bits, src)?;
    if src == dst {
        let w = float_fmt(src).map_or_else(|| int_spec(src).map_or(8, |(w, _)| w), |f| f.width());
        return Ok(if w >= 64 {
            bits
        } else {
            bits & ((1u64 << w) - 1)
        });
    }
    if let Some(f) = float_fmt(dst) {
        return encode_float(ex, f, rounding, saturation);
    }
    let spec = int_spec(dst).ok_or(Error::UnsupportedDtype(dst))?;
    encode_int(ex, spec, from_int, rounding, saturation)
}

/// `cast` over a tensor of bit patterns: same shape, element for element (§6.21-0001).
pub fn cast_tensor(
    x: &View<u64>,
    src: Dtype,
    dst: Dtype,
    rounding: CastRounding,
    saturation: CastSaturation,
) -> Result<Tensor<u64>, Error> {
    check_cast(src, dst, rounding, saturation)?;
    let shape = x.shape();
    let mut buf: Vec<u64> = alloc_exact(numel(shape)?)?;
    let mut od = Odometer::new(shape)?;
    while let Some(c) = od.next_coord() {
        buf.push(cast_bits(x.read(c)?, src, dst, rounding, saturation)?);
    }
    Tensor::from_vec(buf, shape)
}

/// `concat` (§6.11-0020): join `inputs` along `axis`. Pure data movement: elements are copied,
/// never rounded or promoted, so every bit (a NaN payload, a signed zero) is preserved.
///
/// Declines: fewer than 2 inputs or `n + 1 > MAX_OPERANDS` ([`Error::Arity`]); a rank disagreement
/// or an extent disagreement off `axis` ([`Error::ShapeMismatch`]); `axis >= rank`.
pub fn concat<T: Copy>(inputs: &[View<'_, T>], axis: usize) -> Result<Tensor<T>, Error> {
    let n = inputs.len();
    if n < 2 || n + 1 > MAX_OPERANDS {
        return Err(Error::Arity {
            op: Op::Concat,
            expected: 2,
            got: n,
        });
    }
    let first = inputs[0].shape();
    let rank = first.len();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let mut out_shape = [0usize; MAX_RANK];
    out_shape[..rank].copy_from_slice(first);
    out_shape[axis] = 0;
    for v in inputs {
        let s = v.shape();
        if s.len() != rank {
            return Err(Error::ShapeMismatch {
                expected: rank,
                got: s.len(),
            });
        }
        for k in (0..rank).filter(|&k| k != axis) {
            if s[k] != first[k] {
                return Err(Error::ShapeMismatch {
                    expected: first[k],
                    got: s[k],
                });
            }
        }
        out_shape[axis] = out_shape[axis]
            .checked_add(s[axis])
            .ok_or(Error::ShapeOverflow)?;
    }
    concat_fill(inputs, axis, &out_shape[..rank])
}

fn concat_fill<T: Copy>(
    inputs: &[View<'_, T>],
    axis: usize,
    shape: &[usize],
) -> Result<Tensor<T>, Error> {
    let mut buf: Vec<T> = alloc_exact(numel(shape)?)?;
    let mut od = Odometer::new(shape)?;
    let mut src = [0usize; MAX_RANK];
    let rank = shape.len();
    while let Some(c) = od.next_coord() {
        src[..rank].copy_from_slice(c);
        let mut at = c[axis];
        let mut k = 0;
        // an input with a zero `axis` extent is skipped (`at < 0` is never true for it).
        while at >= inputs[k].shape()[axis] {
            at -= inputs[k].shape()[axis];
            k += 1;
        }
        src[axis] = at;
        buf.push(inputs[k].read(&src[..rank])?);
    }
    Tensor::from_vec(buf, shape)
}

/// The shared dtype of `concat` operands; a disagreement is a typed decline (§6.11-0020).
pub fn concat_dtype(dtypes: &[Dtype]) -> Result<Dtype, Error> {
    let first = *dtypes.first().ok_or(Error::Arity {
        op: Op::Concat,
        expected: 2,
        got: 0,
    })?;
    match dtypes.iter().find(|&&d| d != first) {
        Some(&got) => Err(Error::DtypeMismatch {
            expected: first,
            got,
        }),
        None => Ok(first),
    }
}
