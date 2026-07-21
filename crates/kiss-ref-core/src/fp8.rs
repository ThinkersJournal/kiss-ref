//! The FP8 dtypes **e4m3** (OCP `E4M3FN`) and **e5m2** (OCP `E5M2`).
//!
//! Both are 8-bit floats stored in a `u8` newtype. Like `f16`/`bf16` (§6.16), the
//! reference computes by **promoting to `f32`, evaluating with the `f32` reference,
//! and rounding back** (round-to-nearest-even, with saturation); the coarse FP8 ULP
//! ceiling easily absorbs the double-rounding. The `half` crate has no FP8, so the
//! `f32` ↔ FP8 conversions are hand-rolled here.
//!
//! - **e4m3** — `1s+4e+3m`, bias 7. OCP `FN`: **no infinities**, a single NaN
//!   (`S.1111.111`), max finite ±448, subnormals, ±0. An overflowing or infinite
//!   input **saturates** to ±448 (§6.16-0004).
//! - **e5m2** — `1s+5e+2m`, bias 15. IEEE-style: ±inf (`exp=31, m=0`), NaN
//!   (`exp=31, m≠0`), max finite ±57344, subnormals, ±0. An overflowing *finite*
//!   input saturates to ±57344 (§6.16-0005); an actual inf input yields inf.

/// Round a non-negative finite `a` to an FP8 magnitude (the low 7 bits), RNE.
/// `sat` is the magnitude returned on overflow (max-finite for FP8). `e4m3fn`
/// carves out the `exp=max, mant=all-ones` NaN slot as a saturation target.
fn round_mag(a: f32, m_bits: u32, bias: i32, max_fin_e: u32, sat: u8, e4m3fn: bool) -> u8 {
    if a == 0.0 {
        return 0;
    }
    let b = a.to_bits();
    let e32 = ((b >> 23) & 0xFF) as i32;
    let m32 = b & 0x007F_FFFF;
    // 24-bit significand (implicit 1 for a normal f32) + unbiased exponent.
    let (sig, ue) = if e32 == 0 { (m32, -126i32) } else { (m32 | 0x0080_0000, e32 - 127) };
    let drop = 23 - m_bits; // f32 keeps 23 fraction bits; the target keeps m_bits
    let m_mask = (1u32 << m_bits) - 1;
    let te = ue + bias; // target biased exponent

    if te <= 0 {
        // Subnormal (or underflow) in the target: shift the significand right and RNE.
        let shift = drop as i64 + (1 - te as i64);
        if shift > 31 {
            return 0; // underflow to +0
        }
        let shift = shift as u32;
        let round_bit = (sig >> (shift - 1)) & 1;
        let sticky = (sig & ((1u32 << (shift - 1)) - 1)) != 0;
        let mut kept = sig >> shift;
        if round_bit == 1 && (sticky || (kept & 1) == 1) {
            kept += 1;
        }
        if kept > m_mask {
            // rounded up into the smallest normal (biased exp 1, mant 0)
            return (1u32 << m_bits) as u8;
        }
        return kept as u8; // biased exp 0 (subnormal), mantissa `kept`
    }

    // Normal: round the 24-bit significand to (1 + m_bits) bits, RNE.
    let round_bit = (sig >> (drop - 1)) & 1;
    let sticky = (sig & ((1u32 << (drop - 1)) - 1)) != 0;
    let mut kept = sig >> drop;
    if round_bit == 1 && (sticky || (kept & 1) == 1) {
        kept += 1;
    }
    let mut te = te;
    if kept >= (1u32 << (m_bits + 1)) {
        kept >>= 1; // mantissa carried into the exponent
        te += 1;
    }
    let mant = kept & m_mask;
    if te as u32 > max_fin_e {
        return sat;
    }
    if e4m3fn && te as u32 == max_fin_e && mant == m_mask {
        return sat; // the exp=max, mant=all-ones slot is NaN in e4m3fn → saturate
    }
    (((te as u32) << m_bits) | mant) as u8
}

/// Decode an FP8 magnitude/sign byte to `f32`.
fn decode(bits: u8, m_bits: u32, bias: i32, e4m3fn: bool) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e_bits = 7 - m_bits;
    let max_e = (1u32 << e_bits) - 1;
    let exp = ((bits as u32 >> m_bits) & max_e) as i32;
    let mant = (bits as u32) & ((1u32 << m_bits) - 1);
    let scale = (1u32 << m_bits) as f32;
    if e4m3fn {
        if exp as u32 == max_e && mant == (1u32 << m_bits) - 1 {
            return f32::NAN;
        }
    } else if exp as u32 == max_e {
        return if mant == 0 { sign * f32::INFINITY } else { f32::NAN };
    }
    if exp == 0 {
        // subnormal: 2^(1-bias) · (mant / 2^m_bits)
        sign * libm::ldexpf(mant as f32 / scale, 1 - bias)
    } else {
        // normal: 2^(exp-bias) · (1 + mant / 2^m_bits)
        sign * libm::ldexpf(1.0 + mant as f32 / scale, exp - bias)
    }
}

/// OCP **e4m3** (`E4M3FN`): `1s+4e+3m`, bias 7, no inf, single NaN, max ±448.
///
/// `PartialEq`/`PartialOrd` are **by value** (via `to_f32`), so `-0.0 == +0.0` and
/// `NaN != NaN` follow IEEE semantics — bit-pattern equality would be wrong for a
/// float. Use [`E4m3::to_bits`] for raw-bit comparison.
#[derive(Clone, Copy, Debug)]
pub struct E4m3(pub u8);

/// OCP **e5m2** (`E5M2`): `1s+5e+2m`, bias 15, IEEE-style inf/NaN, max ±57344.
///
/// `PartialEq`/`PartialOrd` are by value (see [`E4m3`]).
#[derive(Clone, Copy, Debug)]
pub struct E5m2(pub u8);

impl PartialEq for E4m3 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}
impl PartialOrd for E4m3 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}
impl PartialEq for E5m2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}
impl PartialOrd for E5m2 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl E4m3 {
    /// `+0.0`.
    pub const ZERO: E4m3 = E4m3(0x00);
    /// `1.0` (`exp=7`, `mant=0`).
    pub const ONE: E4m3 = E4m3(0x38);

    /// The raw storage byte.
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }
    /// Wrap a raw storage byte.
    #[inline]
    pub const fn from_bits(b: u8) -> E4m3 {
        E4m3(b)
    }
    /// Widen to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, 3, 7, true)
    }
    /// Round an `f32` to `e4m3` (RNE; inf/overflow saturate to ±448, §6.16-0004).
    #[inline]
    pub fn from_f32(v: f32) -> E4m3 {
        let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
        if v.is_nan() {
            return E4m3(sign | 0x7F);
        }
        let a = libm::fabsf(v);
        if a.is_infinite() {
            return E4m3(sign | 0x7E); // no inf encoding → saturate to max finite
        }
        E4m3(sign | round_mag(a, 3, 7, 15, 0x7E, true))
    }
    /// The single NaN code (`S.1111.111`).
    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7F) == 0x7F
    }
}

impl E5m2 {
    /// `+0.0`.
    pub const ZERO: E5m2 = E5m2(0x00);
    /// `1.0` (`exp=15`, `mant=0`).
    pub const ONE: E5m2 = E5m2(0x3C);

    /// The raw storage byte.
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }
    /// Wrap a raw storage byte.
    #[inline]
    pub const fn from_bits(b: u8) -> E5m2 {
        E5m2(b)
    }
    /// Widen to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, 2, 15, false)
    }
    /// Round an `f32` to `e5m2` (RNE; finite overflow saturates to ±57344,
    /// §6.16-0005; an inf input yields inf).
    #[inline]
    pub fn from_f32(v: f32) -> E5m2 {
        let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0 };
        if v.is_nan() {
            return E5m2(sign | 0x7E); // quiet NaN (exp=31, mant=10)
        }
        let a = libm::fabsf(v);
        if a.is_infinite() {
            return E5m2(sign | 0x7C); // inf (exp=31, mant=0)
        }
        E5m2(sign | round_mag(a, 2, 15, 30, 0x7B, false))
    }
    /// `exp=31, mant≠0`.
    #[inline]
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7C) == 0x7C && (self.0 & 0x03) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_layout_roundtrip() {
        // 1.0, 0.0, ONE.
        assert_eq!(E4m3::ONE.to_f32(), 1.0);
        assert_eq!(E4m3::ZERO.to_f32(), 0.0);
        assert_eq!(E4m3::from_f32(1.0), E4m3::ONE);
        // max finite = 448 (0x7E); 0x7F is NaN.
        assert_eq!(E4m3(0x7E).to_f32(), 448.0);
        assert!(E4m3(0x7F).is_nan() && E4m3(0x7F).to_f32().is_nan());
        // exact small powers of two roundtrip.
        for &x in &[0.5f32, 2.0, 4.0, -3.0, 0.015625] {
            assert_eq!(E4m3::from_f32(x).to_f32(), x, "e4m3 {x}");
        }
    }

    #[test]
    fn e4m3_saturates_and_rounds() {
        // overflow / inf → ±448 (no inf in e4m3fn).
        assert_eq!(E4m3::from_f32(1000.0).to_f32(), 448.0);
        assert_eq!(E4m3::from_f32(f32::INFINITY).to_f32(), 448.0);
        assert_eq!(E4m3::from_f32(f32::NEG_INFINITY).to_f32(), -448.0);
        // RNE: 1.0625 = 1 + 1/16; e4m3 step at 1.0 is 1/8 = 0.125 → nearest even to 1.0.
        assert_eq!(E4m3::from_f32(1.0625).to_f32(), 1.0);
        // 1.1875 = 1 + 3/16 rounds to 1.25 (nearest; 1.125 vs 1.25, 3/16 is halfway → even 1.25? 1.125=1+1/8, 1.25=1+2/8; 3/16 is exactly between → ties to even = 1.25 (even mantissa 2)).
        assert_eq!(E4m3::from_f32(1.1875).to_f32(), 1.25);
        // NaN round-trips to NaN.
        assert!(E4m3::from_f32(f32::NAN).is_nan());
        // signed zero.
        assert_eq!(E4m3::from_f32(-0.0).to_bits(), 0x80);
    }

    #[test]
    fn e4m3_subnormals() {
        // smallest subnormal = 2^-9 (mant 1, exp 0) = 0.001953125.
        assert_eq!(E4m3(0x01).to_f32(), 2f32.powi(-9));
        assert_eq!(E4m3::from_f32(2f32.powi(-9)).to_bits(), 0x01);
        // smallest normal = 2^-6 (0x08).
        assert_eq!(E4m3(0x08).to_f32(), 2f32.powi(-6));
    }

    #[test]
    fn e5m2_layout_and_inf() {
        assert_eq!(E5m2::ONE.to_f32(), 1.0);
        assert_eq!(E5m2::from_f32(1.0), E5m2::ONE);
        // max finite 57344 (0x7B); 0x7C = inf; exp=31,mant!=0 = NaN.
        assert_eq!(E5m2(0x7B).to_f32(), 57344.0);
        assert_eq!(E5m2(0x7C).to_f32(), f32::INFINITY);
        assert!(E5m2(0x7E).is_nan());
        // inf input → inf; finite overflow → saturate to 57344.
        assert_eq!(E5m2::from_f32(f32::INFINITY).to_f32(), f32::INFINITY);
        assert_eq!(E5m2::from_f32(1.0e30).to_f32(), 57344.0);
        for &x in &[0.5f32, 2.0, 4.0, -3.0, 0.25] {
            assert_eq!(E5m2::from_f32(x).to_f32(), x, "e5m2 {x}");
        }
    }
}
