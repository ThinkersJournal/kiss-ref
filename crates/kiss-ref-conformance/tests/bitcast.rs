// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.21-0005 `bitcast`, checked against KISS's own vectors (`conformance/tests/bitcast_op.rs`,
//! `test_ops_bitcast_semantics`): a same-width reinterpretation that preserves every bit. Elements are bit
//! patterns.

use kiss_classify_vocab::Dtype as D;
use kiss_ref_core::bitcast::{bitcast, bitcast_bits, bitcast_dtype_ok};
use kiss_ref_core::tensor::Tensor;
use kiss_ref_core::Error;

#[test]
fn bitcast_preserves_every_bit_and_round_trips() {
    let snan = 0x7F80_0001u64; // signalling NaN, lowest payload bit
    let qnan = 0x7FC0_0001u64; // quiet NaN, payload 1
    for bits in [
        snan,
        qnan,
        0x8000_0000,
        1,
        0x3F80_0000,
        0xFF80_0000,
        0x7F7F_FFFF,
    ] {
        assert_eq!(bitcast_bits(D::F32, D::U32, bits), Ok(bits), "{bits:#x}");
        assert_eq!(bitcast_bits(D::U32, D::F32, bits), Ok(bits), "{bits:#x}");
        assert_eq!(bitcast_bits(D::F32, D::I32, bits), Ok(bits), "{bits:#x}");
    }
    // the signed reading is two's complement: -inf's bits are a negative i32.
    assert_eq!(0xFF80_0000u32 as i32, -8_388_608);
    assert_eq!(bitcast_bits(D::F16, D::I16, 0x7E01), Ok(0x7E01));
    assert_eq!(bitcast_bits(D::Bf16, D::U16, 0xFFC1), Ok(0xFFC1));
    assert_eq!(
        bitcast_bits(D::F64, D::U64, 0x7FF8_0000_0000_0001),
        Ok(0x7FF8_0000_0000_0001)
    );
    assert_eq!(bitcast_bits(D::F64, D::I64, u64::MAX), Ok(u64::MAX));
    assert_eq!(bitcast_bits(D::F8e4m3fn, D::U8, 0x7F), Ok(0x7F));
    assert_eq!(bitcast_bits(D::F8e5m2, D::I8, 0xFC), Ok(0xFC));
    // identity, and integers among themselves.
    assert_eq!(bitcast_bits(D::F32, D::F32, snan), Ok(snan));
    assert_eq!(bitcast_bits(D::I16, D::U16, 0xFFFF), Ok(0xFFFF));
}

#[test]
fn bitcast_declines_are_typed() {
    assert_eq!(
        bitcast_bits(D::F32, D::U16, 0),
        Err(Error::BitcastWidth { src: 32, dst: 16 })
    );
    assert_eq!(
        bitcast_bits(D::U64, D::F32, 0),
        Err(Error::BitcastWidth { src: 64, dst: 32 })
    );
    // sub-byte, MX element, bool, complex: unsupported; f8e8m0 and the reserved floats: reserved/scale.
    for d in [
        D::I4,
        D::U4,
        D::B1,
        D::F4e2m1,
        D::F6e2m3,
        D::F6e3m2,
        D::Bool,
        D::C64,
        D::C128,
    ] {
        assert_eq!(
            bitcast_bits(d, D::U8, 0),
            Err(Error::UnsupportedDtype(d)),
            "src {d:?}"
        );
        assert_eq!(
            bitcast_bits(D::U8, d, 0),
            Err(Error::UnsupportedDtype(d)),
            "dst {d:?}"
        );
    }
    for d in [D::F8e8m0, D::F8e4m3fnuz, D::F8e5m2fnuz, D::F8e6m2] {
        assert_eq!(
            bitcast_bits(d, D::U8, 0),
            Err(Error::ReservedOrScaleDtype(d)),
            "src {d:?}"
        );
        assert_eq!(
            bitcast_bits(D::U8, d, 0),
            Err(Error::ReservedOrScaleDtype(d)),
            "dst {d:?}"
        );
    }
    // the legal set is exactly 8 integers + f16, bf16, f32, f64, f8e4m3fn, f8e5m2.
    assert_eq!(D::ALL.iter().filter(|d| bitcast_dtype_ok(**d)).count(), 14);
}

#[test]
fn bitcast_tensor_keeps_shape_and_every_element() {
    let data = vec![0x7F80_0001u64, 0x8000_0000, 1, 0xFFFF_FFFF, 0x3F80_0000, 0];
    let t = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();
    let out = bitcast(D::F32, D::I32, &t).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    let back = bitcast(D::I32, D::F32, &out).unwrap();
    let v = back.view();
    let mut od = kiss_ref_core::tensor::Odometer::new(&[2, 3]).unwrap();
    let mut got = Vec::new();
    while let Some(c) = od.next_coord() {
        got.push(v.read(c).unwrap());
    }
    assert_eq!(got, data, "there and back is the identity on every pattern");
    assert_eq!(
        bitcast(D::F32, D::U16, &t).err(),
        Some(Error::BitcastWidth { src: 32, dst: 16 })
    );
}
