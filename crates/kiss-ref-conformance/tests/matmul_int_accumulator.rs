// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.17-0010: the accumulator of an integer `matmul` is the compute dtype (the `0x00` default,
//! wrapping) or a signed 32/64-bit integer at least as wide, and the fold is exact (two's-complement wrap
//! in the accumulator dtype, the same bits for every reduction order). Replays KISS's
//! `test_ops_matmul_integer_accumulator` (`conformance/tests/matmul_attrs.rs`).

use kiss_classify_vocab::Dtype as D;
use kiss_ref_core::tensor::Tensor;
use kiss_ref_core::tensor_int::{int_matmul_accumulator_ok as ok, matmul, matmul_acc};
use kiss_ref_core::Error;

fn t(data: &[i128], shape: &[usize]) -> Tensor<i128> {
    Tensor::from_vec(data.to_vec(), shape).unwrap()
}

#[test]
fn integer_accumulator_legality_table() {
    for (c, a) in [
        (D::I8, D::I32),
        (D::U8, D::I32),
        (D::I8, D::I64),
        (D::I16, D::I32),
        (D::I4, D::I32),
        (D::U4, D::I32),
        (D::I32, D::I32),
        (D::I32, D::I64),
        (D::U16, D::I32),
        (D::U32, D::I64),
        (D::I8, D::I8), // the diagonal (0x00)
        (D::U8, D::U8),
    ] {
        assert!(ok(c, a), "{c:?} into {a:?}");
    }
    for (c, a) in [
        (D::I8, D::I16),  // narrower than 32 bits
        (D::I8, D::U32),  // not signed
        (D::I8, D::F32),  // a float accumulator for an integer compute dtype
        (D::I64, D::I32), // narrower than the compute dtype
        (D::U32, D::I32), // unsigned compute into a same-width signed accumulator
        (D::U64, D::I64), // likewise at 64 bits
        (D::I8, D::U8),   // unsigned narrow
        (D::F32, D::I32), // an integer accumulator for a float compute dtype
    ] {
        assert!(!ok(c, a), "{c:?} into {a:?}");
    }
}

#[test]
fn declines_are_typed() {
    let (a, b) = (t(&[1, 2], &[1, 2]), t(&[3, 4], &[2, 1]));
    for (c, acc) in [
        (D::I8, D::I16),
        (D::I8, D::F32),
        (D::I64, D::I32),
        (D::U32, D::I32),
    ] {
        assert_eq!(
            matmul_acc(&a.view(), &b.view(), c, Some(acc)).err(),
            Some(Error::MatmulAccumulator {
                compute: c,
                accumulator: acc
            })
        );
    }
}

#[test]
fn the_fold_is_exact_in_the_accumulator_dtype() {
    // KISS's vectors, against an independent i128 sum wrapped to the accumulator width.
    let av: Vec<i128> = vec![127, -128, 99, -7, 55, 127, -128, 3];
    let bv: Vec<i128> = vec![-128, -128, 127, 90, -1, 127, 127, -100];
    let exact: i128 = av.iter().zip(&bv).map(|(x, y)| x * y).sum();
    let (a, b) = (t(&av, &[1, 8]), t(&bv, &[8, 1]));
    let r32 = matmul_acc(&a.view(), &b.view(), D::I8, Some(D::I32)).unwrap();
    assert_eq!(r32.as_slice(), &[exact as i32 as i128]);
    let r64 = matmul_acc(&a.view(), &b.view(), D::I8, Some(D::I64)).unwrap();
    assert_eq!(r64.as_slice(), &[exact as i64 as i128]);
    // every reduction order gives the same bits.
    let (ar, br): (Vec<i128>, Vec<i128>) = (
        av.iter().rev().cloned().collect(),
        bv.iter().rev().cloned().collect(),
    );
    let rr = matmul_acc(
        &t(&ar, &[1, 8]).view(),
        &t(&br, &[8, 1]).view(),
        D::I8,
        Some(D::I32),
    )
    .unwrap();
    assert_eq!(rr.as_slice(), r32.as_slice());
    // the default accumulator (compute dtype) wraps per atom, where an i32 accumulator keeps 100*100.
    let (p, q) = (t(&[100], &[1, 1]), t(&[100], &[1, 1]));
    assert_eq!(
        matmul(&p.view(), &q.view(), D::I8).unwrap().as_slice(),
        &[16]
    );
    assert_eq!(
        matmul_acc(&p.view(), &q.view(), D::I8, None)
            .unwrap()
            .as_slice(),
        &[16]
    );
    assert_eq!(
        matmul_acc(&p.view(), &q.view(), D::I8, Some(D::I32))
            .unwrap()
            .as_slice(),
        &[10_000]
    );
}

#[test]
fn overflow_wraps_at_the_accumulator_width() {
    let big = i128::from(i32::MAX);
    let (a, b) = (t(&[big, big], &[1, 2]), t(&[1, 1], &[2, 1]));
    let r32 = matmul_acc(&a.view(), &b.view(), D::I32, Some(D::I32)).unwrap();
    assert_eq!(
        r32.as_slice(),
        &[(2 * i64::from(i32::MAX)) as i32 as i128],
        "wraps at 32 bits"
    );
    let r64 = matmul_acc(&a.view(), &b.view(), D::I32, Some(D::I64)).unwrap();
    assert_eq!(r64.as_slice(), &[2 * big], "kept at 64 bits");
}
