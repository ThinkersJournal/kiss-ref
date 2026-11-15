// SPDX-License-Identifier: MIT OR Apache-2.0
//! `bitcast` (KISS-Ops §6.21-0005, OpAttrs §6.19-0048): reinterpret each element's bits as a dtype of the
//! **same width**. A pure bit move: nothing is computed and nothing canonicalised, so a NaN keeps its
//! payload, quiet/signalling bit and sign, a signed zero keeps its sign and a subnormal is not flushed;
//! a round trip is the identity on every pattern. Elements are bit patterns (`u64`, low `width` bits
//! significant), the same representation [`crate::cast`] and [`crate::iota`] use.
//!
//! Source and destination MUST each be an integer dtype (`i8..i64`, `u8..u64`) or one of `f16`, `bf16`,
//! `f32`, `f64`, `f8e4m3fn`, `f8e5m2`. Typed declines: `bool`, a sub-byte dtype, an MX element type, a
//! complex dtype ([`Error::UnsupportedDtype`]); `f8e8m0` and the reserved floats
//! ([`Error::ReservedOrScaleDtype`]); a destination of a different width ([`Error::BitcastWidth`]).

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;

use crate::tensor::{alloc_exact, Tensor};
use crate::Error;

/// Whether `dtype` is one `bitcast` accepts as source or destination (§6.21-0005).
pub const fn bitcast_dtype_ok(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::I8
            | Dtype::I16
            | Dtype::I32
            | Dtype::I64
            | Dtype::U8
            | Dtype::U16
            | Dtype::U32
            | Dtype::U64
            | Dtype::F16
            | Dtype::Bf16
            | Dtype::F32
            | Dtype::F64
            | Dtype::F8e4m3fn
            | Dtype::F8e5m2
    )
}

fn check_dtype(dtype: Dtype) -> Result<(), Error> {
    if bitcast_dtype_ok(dtype) {
        Ok(())
    } else if dtype.declines_compute() {
        Err(Error::ReservedOrScaleDtype(dtype))
    } else {
        Err(Error::UnsupportedDtype(dtype))
    }
}

/// Validate a `src -> dst` pair: both dtypes legal and of equal width; `dst == src` is the identity.
pub fn check_bitcast(src: Dtype, dst: Dtype) -> Result<(), Error> {
    check_dtype(src)?;
    check_dtype(dst)?;
    if src.bits() != dst.bits() {
        return Err(Error::BitcastWidth {
            src: src.bits(),
            dst: dst.bits(),
        });
    }
    Ok(())
}

/// One element: the same bits, read as `dst` (masked to the width). A typed decline, never a panic.
pub fn bitcast_bits(src: Dtype, dst: Dtype, bits: u64) -> Result<u64, Error> {
    check_bitcast(src, dst)?;
    Ok(if src.bits() >= 64 {
        bits
    } else {
        bits & ((1u64 << src.bits()) - 1)
    })
}

/// `bitcast` over a tensor of `src` bit patterns: same shape, every element reinterpreted as `dst`.
pub fn bitcast(src: Dtype, dst: Dtype, input: &Tensor<u64>) -> Result<Tensor<u64>, Error> {
    check_bitcast(src, dst)?;
    let shape = input.shape();
    let mut out: Vec<u64> = alloc_exact(input.len())?;
    let view = input.view();
    let mut od = crate::tensor::Odometer::new(shape)?;
    while let Some(c) = od.next_coord() {
        out.push(bitcast_bits(src, dst, view.read(c)?)?);
    }
    Tensor::from_vec(out, shape)
}
