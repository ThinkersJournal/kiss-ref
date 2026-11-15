// SPDX-License-Identifier: MIT OR Apache-2.0
//! The zero-operand counting source `iota` (KISS-Ops §6.11-0025, OpAttrs §6.19-0047).
//!
//! `iota(target_dtype, extents, axis)` has no input: at the coordinate `(c0, .., c(r-1))` the value is
//! `c(axis)` converted to `target_dtype` **exactly**. It computes no floating-point arithmetic, so the
//! result is exact-byte (§6.0-0002). Like [`crate::cast`], the output is a tensor of **bit patterns**
//! (`u64`, low `width` bits significant), so every element lane can consume it.
//!
//! Typed declines, each a distinct [`Error`] and never a guess:
//! * `target_dtype` other than an integer dtype (`i8`..`i64`, `u8`..`u64`) or `f16`, `bf16`, `f32`,
//!   `f64` (`bool`, a sub-byte, MX or `f8*` dtype, a complex dtype, a reserved one): [`Error::UnsupportedDtype`]
//!   or [`Error::ReservedOrScaleDtype`];
//! * a rank of `0` or above `MAX_RANK`, an extent of `0`, an `axis` at or above the rank;
//! * a last count `extents[axis] - 1` the dtype cannot hold exactly ([`Error::IotaUnrepresentable`]): a
//!   float holds every integer up to `2^p` (`p` the significand width with the implicit bit), an integer
//!   dtype up to its maximum, because a rounded count is a wrong count.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;

use crate::cast::{cast_bits, CastRounding, CastSaturation};
use crate::tensor::{alloc_exact, numel, Odometer, Tensor, MAX_RANK};
use crate::Error;

/// The largest integer `dtype` holds exactly as a count, or `None` for a dtype `iota` refuses
/// (§6.11-0025: floats `2^p`, integers their maximum).
pub const fn iota_max_exact(dtype: Dtype) -> Option<u64> {
    Some(match dtype {
        Dtype::I8 => 127,
        Dtype::U8 => 255,
        Dtype::I16 => 32_767,
        Dtype::U16 => 65_535,
        Dtype::I32 => 2_147_483_647,
        Dtype::U32 => 4_294_967_295,
        Dtype::I64 => 9_223_372_036_854_775_807,
        Dtype::U64 => u64::MAX,
        Dtype::F16 => 1 << 11,
        Dtype::Bf16 => 1 << 8,
        Dtype::F32 => 1 << 24,
        Dtype::F64 => 1 << 53,
        _ => return None,
    })
}

/// `iota` (§6.11-0025): the count along `axis` of a tensor of shape `extents`, as `dtype` bit patterns,
/// row-major. A typed decline, never a panic; the `u64` source of every conversion is exact by the
/// representability check, so the shared `cast` path rounds nothing.
pub fn iota(dtype: Dtype, extents: &[usize], axis: usize) -> Result<Tensor<u64>, Error> {
    let max = match iota_max_exact(dtype) {
        Some(m) => m,
        None if dtype.is_complex() => return Err(Error::UnsupportedDtype(dtype)),
        None if dtype.declines_compute() => return Err(Error::ReservedOrScaleDtype(dtype)),
        None => return Err(Error::UnsupportedDtype(dtype)),
    };
    if extents.is_empty() || extents.len() > MAX_RANK {
        return Err(Error::RankExceeded {
            rank: extents.len(),
            max: MAX_RANK,
        });
    }
    if extents.contains(&0) {
        return Err(Error::IotaZeroExtent);
    }
    if axis >= extents.len() {
        return Err(Error::AxisOutOfRange {
            axis,
            rank: extents.len(),
        });
    }
    let last = extents[axis] as u64 - 1;
    if last > max {
        return Err(Error::IotaUnrepresentable { last });
    }
    let mut buf: Vec<u64> = alloc_exact(numel(extents)?)?;
    let mut od = Odometer::new(extents)?;
    while let Some(c) = od.next_coord() {
        buf.push(cast_bits(
            c[axis] as u64,
            Dtype::U64,
            dtype,
            CastRounding::NearestEven,
            CastSaturation::None,
        )?);
    }
    Tensor::from_vec(buf, extents)
}
