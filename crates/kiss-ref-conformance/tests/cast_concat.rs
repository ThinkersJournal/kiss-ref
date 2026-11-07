// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cast` (KISS-Ops §6.21) and `concat` (§6.11-0020) evaluators. Vectors are the ones KISS's own
//! `conformance/tests/cast_atom.rs` pins (bf16 ties, the double-rounding control, saturation,
//! integer clamp/wrap, the unpinned cases), plus a differential against the `half` crate (an
//! independent single-rounding f64 -> f16/bf16 implementation) and Rust's own `as` casts.

use half::{bf16, f16};
use kiss_classify_vocab::Dtype as D;
use kiss_ref_core::cast::{
    cast_bits, cast_tensor, check_cast, concat, concat_dtype, CastRounding as R,
    CastSaturation as S,
};
use kiss_ref_core::{Error, Tensor};

const ALL_R: [R; 4] = [
    R::NearestEven,
    R::TowardZero,
    R::TowardNegInf,
    R::TowardPosInf,
];

fn f32_to_bf(x: f32, r: R, s: S) -> u64 {
    cast_bits(u64::from(x.to_bits()), D::F32, D::Bf16, r, s).unwrap()
}

/// `1 + 2^-8`: exactly halfway between the bf16 neighbours 0x3F80 and 0x3F81.
fn tie_low() -> f32 {
    f32::from_bits(0x3F80_8000)
}

#[test]
fn cast_rounding_modes_bf16_ties() {
    let up = |x, r| f32_to_bf(x, r, S::None);
    let x = tie_low();
    assert_eq!(up(x, R::NearestEven), 0x3F80);
    assert_eq!(up(x, R::TowardZero), 0x3F80);
    assert_eq!(up(x, R::TowardNegInf), 0x3F80);
    assert_eq!(up(x, R::TowardPosInf), 0x3F81);
    assert_eq!(up(-x, R::NearestEven), 0xBF80);
    assert_eq!(up(-x, R::TowardZero), 0xBF80);
    assert_eq!(up(-x, R::TowardNegInf), 0xBF81);
    assert_eq!(up(-x, R::TowardPosInf), 0xBF80);
    let odd = f32::from_bits(0x3F81_8000);
    assert_eq!(up(odd, R::NearestEven), 0x3F82);
    assert_eq!(up(odd, R::TowardZero), 0x3F81);
    for r in ALL_R {
        assert_eq!(
            up(1.5, r),
            0x3FC0,
            "an exact conversion ignores the mode: {r:?}"
        );
    }
}

#[test]
fn cast_rounds_the_exact_source_once() {
    // 1 + 2^-8 + 2^-30 is above the tie in f64; staged through f32 the tail is lost and the
    // value lands on the tie, which nearest-even sends DOWN. One rounding goes UP.
    let x = 1.0f64 + 2f64.powi(-8) + 2f64.powi(-30);
    let direct = cast_bits(x.to_bits(), D::F64, D::Bf16, R::NearestEven, S::None).unwrap();
    let staged = f32_to_bf(x as f32, R::NearestEven, S::None);
    assert_eq!(direct, 0x3F81);
    assert_eq!(
        staged, 0x3F80,
        "control: staging through f32 changes the answer"
    );
}

#[test]
fn cast_saturation_float_destinations() {
    let c = |x: f32, r, s| f32_to_bf(x, r, s);
    assert_eq!(c(f32::MAX, R::NearestEven, S::None), 0x7F80);
    assert_eq!(c(f32::MAX, R::NearestEven, S::Clamp), 0x7F7F);
    assert_eq!(c(-f32::MAX, R::NearestEven, S::Clamp), 0xFF7F);
    assert_eq!(c(f32::MAX, R::TowardZero, S::None), 0x7F7F);
    assert_eq!(c(f32::MAX, R::TowardNegInf, S::None), 0x7F7F);
    assert_eq!(c(f32::MAX, R::TowardPosInf, S::None), 0x7F80);
    assert_eq!(c(f32::INFINITY, R::NearestEven, S::Clamp), 0x7F7F);
    assert_eq!(c(f32::NEG_INFINITY, R::NearestEven, S::None), 0xFF80);
    // a NaN stays a NaN with its sign, whatever the attributes
    let nan = c(f32::from_bits(0xFFC0_0001), R::NearestEven, S::Clamp);
    assert!(
        nan & 0x7F80 == 0x7F80 && nan & 0x007F != 0 && nan & 0x8000 != 0,
        "{nan:#06x}"
    );
}

#[test]
fn cast_fp8_and_mx_destinations() {
    let f =
        |x: f32, d| cast_bits(u64::from(x.to_bits()), D::F32, d, R::NearestEven, S::Clamp).unwrap();
    assert_eq!(
        f(1000.0, D::F8e4m3fn),
        0x7E,
        "448 is the largest finite e4m3fn"
    );
    assert_eq!(
        f(-1.0e6, D::F8e5m2),
        0xFB,
        "-57344 is the largest finite e5m2 magnitude"
    );
    assert_eq!(f(100.0, D::F4e2m1), 0x7, "6.0, the largest finite e2m1");
    assert_eq!(f(-100.0, D::F4e2m1), 0xF);
    assert_eq!(f(100.0, D::F6e2m3), 0x1F, "7.5");
    assert_eq!(f(100.0, D::F6e3m2), 0x1F, "28.0");
    assert_eq!(
        f(0.25, D::F4e2m1),
        0x0,
        "0.25 is a tie between 0 and 0.5: to even, zero"
    );
    assert_eq!(
        f(0.75, D::F4e2m1),
        0x2,
        "0.75 is a tie between 0.5 and 1.0: to even, 1.0 (0x2)"
    );
    assert_eq!(f(0.5, D::F4e2m1), 0x1, "the one subnormal, 0.5");
    // a NaN into e4m3fn is its single NaN; into e5m2 a quiet NaN
    assert_eq!(f(f32::NAN, D::F8e4m3fn) & 0x7F, 0x7F);
    assert_eq!((f(f32::NAN, D::F8e5m2) >> 2) & 0x1F, 0x1F);
    assert_ne!(
        f(f32::NAN, D::F8e5m2) & 0x3,
        0,
        "a NaN keeps a non-zero mantissa"
    );
    // an infinity clamps
    assert_eq!(f(f32::INFINITY, D::F8e4m3fn), 0x7E);
}

#[test]
fn cast_integer_clamp_and_wrap() {
    let i = |v: i64, src, dst, s| cast_bits(v as u64, src, dst, R::NearestEven, s).unwrap();
    assert_eq!(i(300, D::I64, D::I8, S::Clamp), 127);
    assert_eq!(i(300, D::I64, D::I8, S::None), 44);
    assert_eq!(i(-129, D::I64, D::I8, S::None), 127);
    assert_eq!(i(-1, D::I64, D::U8, S::None), 255);
    assert_eq!(i(-1, D::I64, D::U8, S::Clamp), 0);
    assert_eq!(
        i(-1, D::I8, D::I64, S::None) as i64,
        -1,
        "sign extension in the 64-bit pattern"
    );
    assert_eq!(
        i(u64::MAX as i64, D::U64, D::I64, S::Clamp),
        i64::MAX as u64
    );
    assert_eq!(i(9, D::I8, D::I4, S::Clamp), 7, "i4 clamps to 7");
    assert_eq!(
        i(9, D::I8, D::I4, S::None),
        0b1001,
        "i4 wraps: 9 mod 16 = -7 in two's complement"
    );
    assert_eq!(
        i(2, D::I8, D::B1, S::Clamp),
        1,
        "b1 is a 1-bit unsigned integer"
    );
    let f = |x: f32, r, s| cast_bits(u64::from(x.to_bits()), D::F32, D::I8, r, s);
    assert_eq!(f(300.0, R::NearestEven, S::Clamp), Ok(127));
    assert_eq!(f(-300.0, R::NearestEven, S::Clamp), Ok(0x80));
    assert_eq!(f(2.5, R::NearestEven, S::Clamp), Ok(2));
    assert_eq!(f(2.5, R::TowardPosInf, S::Clamp), Ok(3));
    assert_eq!(f(-2.5, R::TowardNegInf, S::Clamp), Ok(0xFD));
    assert_eq!(f(-2.5, R::TowardZero, S::Clamp), Ok(0xFE));
    assert_eq!(f(f32::INFINITY, R::NearestEven, S::Clamp), Ok(127));
    assert_eq!(
        f(100.0, R::NearestEven, S::None),
        Ok(100),
        "in range: pinned even under none"
    );
}

#[test]
fn cast_unpinned_cases_are_typed_declines() {
    let f32b = |x: f32| u64::from(x.to_bits());
    assert_eq!(
        cast_bits(f32b(f32::NAN), D::F32, D::I32, R::NearestEven, S::Clamp),
        Err(Error::CastUnpinned)
    );
    assert_eq!(
        cast_bits(f32b(1.0e10), D::F32, D::I16, R::NearestEven, S::None),
        Err(Error::CastUnpinned)
    );
    assert_eq!(
        cast_bits(f32b(f32::INFINITY), D::F32, D::I16, R::NearestEven, S::None),
        Err(Error::CastUnpinned)
    );
    assert_eq!(
        cast_bits(f32b(f32::NAN), D::F32, D::F4e2m1, R::NearestEven, S::Clamp),
        Err(Error::CastUnpinned)
    );
    assert_eq!(
        cast_bits(f32b(1.0e10), D::F32, D::I16, R::NearestEven, S::Clamp),
        Ok(32767),
        "control: clamp has a value"
    );
}

#[test]
fn cast_typed_declines_by_attribute_and_dtype() {
    for dst in [D::F8e4m3fn, D::F8e5m2, D::F4e2m1, D::F6e2m3, D::F6e3m2] {
        assert_eq!(
            check_cast(D::F32, dst, R::NearestEven, S::Clamp),
            Ok(()),
            "{dst:?}"
        );
        for r in [R::TowardZero, R::TowardNegInf, R::TowardPosInf] {
            assert_eq!(
                check_cast(D::F32, dst, r, S::Clamp),
                Err(Error::CastRoundingNotDefined(dst))
            );
        }
        assert_eq!(
            check_cast(D::F32, dst, R::NearestEven, S::None),
            Err(Error::CastSaturationRequired(dst))
        );
    }
    for r in ALL_R {
        assert_eq!(
            check_cast(D::F32, D::Bf16, r, S::None),
            Ok(()),
            "control: bf16 takes every mode"
        );
    }
    assert_eq!(
        check_cast(D::C64, D::F32, R::NearestEven, S::None),
        Err(Error::UnsupportedDtype(D::C64))
    );
    assert_eq!(
        check_cast(D::F32, D::C128, R::NearestEven, S::None),
        Err(Error::UnsupportedDtype(D::C128))
    );
    assert_eq!(
        check_cast(D::F32, D::F8e4m3fnuz, R::NearestEven, S::Clamp),
        Err(Error::ReservedOrScaleDtype(D::F8e4m3fnuz))
    );
    assert_eq!(
        check_cast(D::F8e8m0, D::F32, R::NearestEven, S::None),
        Err(Error::ReservedOrScaleDtype(D::F8e8m0))
    );
}

#[test]
fn cast_identity_preserves_every_bit() {
    for (d, bits) in [
        (D::F32, 0xFFC0_1234u64),
        (D::F64, 0x7FF0_0000_0000_0001),
        (D::F16, 0x7C01),
        (D::I8, 0x80),
    ] {
        for r in ALL_R {
            for s in [S::None, S::Clamp] {
                assert_eq!(cast_bits(bits, d, d, r, s), Ok(bits), "{d:?} {r:?} {s:?}");
            }
        }
    }
}

/// Deterministic xorshift for the differential cases.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn cast_matches_exact_oracles_on_random_inputs() {
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..200_000 {
        let bits = next(&mut st);
        let x = f64::from_bits(bits);
        if x.is_nan() {
            continue;
        }
        // f64 -> f32: `as` rounds the exact value once, to nearest even, overflow to infinity.
        let got = cast_bits(bits, D::F64, D::F32, R::NearestEven, S::None).unwrap();
        assert_eq!(got as u32, (x as f32).to_bits(), "f64->f32 {bits:#x}");
    }
    // moderate magnitudes, so f16/bf16 normals, subnormals and ties are all exercised. The oracle
    // scales by an exact power of two, rounds ties-to-even in f64 (exact there), and scales back.
    // (The `half` crate, built `default-features = false` as here, is NOT used as the oracle: its
    // software `f16::from_f64` rounds 0xbf235a0069b85960 down to 0x88d6; the exact value is
    // 1238.5004 quanta, so 0x88d7 is right, and CPython's struct 'e' agrees. half-rs#151.)
    for _ in 0..200_000 {
        let m = (next(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
        let e = (next(&mut st) % 44) as i32 - 30;
        let x = (m * 2f64.powi(e)) * if next(&mut st) & 1 == 1 { -1.0 } else { 1.0 };
        let ex = if x == 0.0 {
            -1022
        } else {
            ((x.to_bits() >> 52) & 0x7FF) as i32 - 1023
        };
        for (d, man, emin) in [(D::F16, 10, -14), (D::Bf16, 7, -126)] {
            let qe = ex.max(emin) - man;
            let want = (x / 2f64.powi(qe)).round_ties_even() * 2f64.powi(qe);
            let got = cast_bits(x.to_bits(), D::F64, d, R::NearestEven, S::None).unwrap() as u16;
            let val = if d == D::F16 {
                f16::from_bits(got).to_f64()
            } else {
                bf16::from_bits(got).to_f64()
            };
            assert_eq!(val.to_bits(), want.to_bits(), "{d:?} {x:e}");
        }
    }
    // integers -> f32 and f64 round once (u64 -> f32 is the double-rounding trap)
    for _ in 0..100_000 {
        let v = next(&mut st);
        assert_eq!(
            cast_bits(v, D::U64, D::F32, R::NearestEven, S::None).unwrap() as u32,
            (v as f32).to_bits(),
            "u64->f32 {v:#x}"
        );
        assert_eq!(
            cast_bits(v, D::U64, D::F64, R::NearestEven, S::None).unwrap(),
            (v as f64).to_bits()
        );
        let s = v as i64;
        assert_eq!(
            cast_bits(s as u64, D::I64, D::F32, R::NearestEven, S::None).unwrap() as u32,
            (s as f32).to_bits()
        );
    }
    // float -> integer: Rust's `as` truncates and saturates, matching toward-zero + clamp
    for _ in 0..100_000 {
        let x = f64::from_bits(next(&mut st));
        if x.is_nan() {
            continue;
        }
        let got = cast_bits(x.to_bits(), D::F64, D::I32, R::TowardZero, S::Clamp).unwrap();
        assert_eq!(got as u32 as i32, x as i32, "f64->i32 {x:e}");
        let got = cast_bits(x.to_bits(), D::F64, D::U8, R::TowardZero, S::Clamp).unwrap();
        assert_eq!(got as u8, x as u8, "f64->u8 {x:e}");
    }
}

#[test]
fn cast_directed_rounding_brackets_the_exact_value() {
    // For every mode the result is a representable neighbour on the right side: down <= x <= up.
    let mut st = 0x1234_5678_9ABC_DEF1u64;
    for _ in 0..50_000 {
        let m = (next(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
        let x = (m - 0.5) * 2f64.powi((next(&mut st) % 20) as i32 - 10);
        let to = |r| {
            let b = cast_bits(x.to_bits(), D::F64, D::F16, r, S::None).unwrap() as u16;
            f64::from(f16::from_bits(b).to_f32())
        };
        let (dn, up, tz) = (to(R::TowardNegInf), to(R::TowardPosInf), to(R::TowardZero));
        assert!(dn <= x && x <= up, "{x:e}: {dn:e} {up:e}");
        assert_eq!(
            tz,
            if x < 0.0 { up } else { dn },
            "toward zero picks the neighbour nearer zero"
        );
    }
}

#[test]
fn cast_tensor_is_element_for_element_and_keeps_signed_zero() {
    let src: [f32; 6] = [1.5, -2.25, 0.0, -0.0, 65504.0, 1.0e-3];
    let bits: Vec<u64> = src.iter().map(|x| u64::from(x.to_bits())).collect();
    let t = Tensor::from_vec(bits, &[2, 3]).unwrap();
    let out = cast_tensor(&t.view(), D::F32, D::Bf16, R::NearestEven, S::None).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.as_slice()[3], 0x8000, "a signed zero keeps its sign");
    assert_eq!(out.as_slice()[0], 0x3FC0);
}

// ---- concat ---------------------------------------------------------------------------------

#[test]
fn concat_joins_along_the_axis_and_preserves_bits() {
    let a = Tensor::from_vec(vec![1u32, 2, 3, 4], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5u32, 6], &[2, 1]).unwrap();
    let c = Tensor::from_vec(vec![7u32, 8, 9, 10, 11, 12], &[2, 3]).unwrap();
    let out = concat(&[a.view(), b.view(), c.view()], 1).unwrap();
    assert_eq!(out.shape(), &[2, 6]);
    assert_eq!(out.as_slice(), &[1, 2, 5, 7, 8, 9, 3, 4, 6, 10, 11, 12]);
    let out0 = concat(&[a.view(), a.view()], 0).unwrap();
    assert_eq!(out0.shape(), &[4, 2]);
    assert_eq!(out0.as_slice(), &[1, 2, 3, 4, 1, 2, 3, 4]);
    // bits: a NaN payload and a signed zero survive unchanged
    let weird = [0x7FC0_1234u32, 0xFFA0_0001, 0x8000_0000, 0x0000_0001];
    let w = Tensor::from_vec(weird.to_vec(), &[4]).unwrap();
    let out = concat(&[w.view(), w.view()], 0).unwrap();
    assert_eq!(&out.as_slice()[..4], &weird);
    assert_eq!(&out.as_slice()[4..], &weird);
}

#[test]
fn concat_zero_extent_input_contributes_nothing() {
    let a = Tensor::from_vec(vec![1u8, 2], &[1, 2]).unwrap();
    let z: Tensor<u8> = Tensor::from_vec(vec![], &[0, 2]).unwrap();
    let out = concat(&[z.view(), a.view(), z.view(), a.view()], 0).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[1, 2, 1, 2]);
}

#[test]
fn concat_declines_arity_rank_shape_axis_and_dtype() {
    let a = Tensor::from_vec(vec![1u8, 2, 3, 4], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![1u8, 2, 3], &[1, 3]).unwrap();
    let r1 = Tensor::from_vec(vec![1u8, 2], &[2]).unwrap();
    assert!(matches!(
        concat(&[a.view()], 0),
        Err(Error::Arity { got: 1, .. })
    ));
    assert!(matches!(
        concat(&[a.view(), b.view()], 0),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        concat(&[a.view(), r1.view()], 0),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        concat(&[a.view(), a.view()], 2),
        Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
    ));
    // n + 1 <= MAX_OPERANDS: MAX_OPERANDS inputs leave no slot for the output
    let many: Vec<_> = (0..kiss_ref_core::MAX_OPERANDS).map(|_| a.view()).collect();
    assert!(matches!(concat(&many, 0), Err(Error::Arity { .. })));
    let ok: Vec<_> = (0..kiss_ref_core::MAX_OPERANDS - 1)
        .map(|_| a.view())
        .collect();
    assert!(
        concat(&ok, 0).is_ok(),
        "control: MAX_OPERANDS - 1 inputs is legal"
    );
    assert_eq!(concat_dtype(&[D::F32, D::F32]), Ok(D::F32));
    assert_eq!(
        concat_dtype(&[D::F32, D::F64]),
        Err(Error::DtypeMismatch {
            expected: D::F32,
            got: D::F64
        })
    );
}
