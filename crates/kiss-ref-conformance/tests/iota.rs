// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.11-0025 `iota`: the zero-operand counting source, checked against KISS's own vectors
//! (`conformance/tests/iota_op.rs`, `test_ops_iota_semantics`): the count along one axis of a static shape,
//! constant along the others, exactly, in the stated dtype. Elements are bit patterns.

use kiss_classify_vocab::Dtype as D;
use kiss_ref_core::iota::{iota, iota_max_exact};
use kiss_ref_core::Error;

fn counts(d: D, extents: &[usize], axis: usize) -> Vec<u64> {
    let t = iota(d, extents, axis).expect("legal iota");
    let n: usize = extents.iter().product();
    let v = t.view();
    let mut od = kiss_ref_core::tensor::Odometer::new(extents).unwrap();
    let mut out = Vec::with_capacity(n);
    while let Some(c) = od.next_coord() {
        out.push(v.read(c).unwrap());
    }
    out
}

#[test]
fn iota_counts_along_one_axis_and_is_constant_along_the_others() {
    assert_eq!(
        counts(D::I32, &[5], 0),
        vec![0, 1, 2, 3, 4],
        "rank 1: arange(0, 5)"
    );
    // [2,3]: axis 0 counts rows, axis 1 counts columns; floats are checked as f32 bit patterns.
    let f = |xs: &[f32]| {
        xs.iter()
            .map(|x| u64::from(x.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(counts(D::F32, &[2, 3], 0), f(&[0., 0., 0., 1., 1., 1.]));
    assert_eq!(counts(D::F32, &[2, 3], 1), f(&[0., 1., 2., 0., 1., 2.]));
    assert_eq!(
        counts(D::U8, &[2, 2, 2], 1),
        vec![0, 0, 1, 1, 0, 0, 1, 1],
        "rank 3, the middle axis"
    );
    assert_eq!(
        counts(D::I64, &[1, 4], 0),
        vec![0, 0, 0, 0],
        "an extent of 1 counts only 0"
    );
    // wide and narrow float lanes: the exact bit patterns of 0, 1, 2, 3.
    let d = [0.0f64, 1.0, 2.0, 3.0].map(f64::to_bits).to_vec();
    assert_eq!(counts(D::F64, &[4], 0), d);
    assert_eq!(
        counts(D::F16, &[3], 0),
        vec![0x0000, 0x3C00, 0x4000],
        "f16: 0, 1, 2"
    );
    assert_eq!(
        counts(D::Bf16, &[3], 0),
        vec![0x0000, 0x3F80, 0x4000],
        "bf16: 0, 1, 2"
    );
}

#[test]
fn iota_is_exact_or_declines() {
    assert_eq!(iota_max_exact(D::F32), Some(1 << 24));
    assert_eq!(iota_max_exact(D::F16), Some(2048));
    assert_eq!(iota_max_exact(D::Bf16), Some(256));
    assert_eq!(iota_max_exact(D::F64), Some(1 << 53));
    assert_eq!(iota_max_exact(D::U64), Some(u64::MAX));
    // the last count must be representable: 2^24 + 1 is not in f32 (the shape is never materialised).
    assert!(
        matches!(iota(D::F32, &[(1 << 24) + 2], 0), Err(Error::IotaUnrepresentable { last }) if last == (1 << 24) + 1)
    );
    assert!(matches!(
        iota(D::Bf16, &[258], 0),
        Err(Error::IotaUnrepresentable { last: 257 })
    ));
    assert!(
        iota(D::Bf16, &[257], 0).is_ok(),
        "last count 256 is exact in bf16"
    );
    assert!(matches!(
        iota(D::I8, &[129], 0),
        Err(Error::IotaUnrepresentable { last: 128 })
    ));
    assert!(iota(D::I8, &[128], 0).is_ok());
    // dtypes that are not a count: bool, sub-byte, MX, f8, complex, reserved.
    for d in [
        D::Bool,
        D::I4,
        D::U4,
        D::B1,
        D::F8e4m3fn,
        D::F8e5m2,
        D::F4e2m1,
        D::C64,
        D::C128,
    ] {
        assert!(
            matches!(iota(d, &[4], 0), Err(Error::UnsupportedDtype(x)) if x == d),
            "{d:?}"
        );
    }
    for d in [D::F8e4m3fnuz, D::F8e5m2fnuz] {
        assert!(
            matches!(iota(d, &[4], 0), Err(Error::ReservedOrScaleDtype(x)) if x == d),
            "{d:?}"
        );
    }
    // shape and axis declines.
    assert!(matches!(
        iota(D::I32, &[], 0),
        Err(Error::RankExceeded { rank: 0, .. })
    ));
    assert!(matches!(
        iota(D::I32, &[1; 9], 0),
        Err(Error::RankExceeded { rank: 9, .. })
    ));
    assert!(matches!(
        iota(D::I32, &[3, 0], 0),
        Err(Error::IotaZeroExtent)
    ));
    assert!(matches!(
        iota(D::I32, &[3, 4], 2),
        Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
    ));
}

#[test]
fn iota_compositions_named_by_the_clause() {
    // arange(start = 3, limit, delta = 2) = iota * delta + start over a rank-1 extent.
    let arange: Vec<u64> = counts(D::I32, &[4], 0).iter().map(|i| i * 2 + 3).collect();
    assert_eq!(arange, vec![3, 5, 7, 9]);
    // lower-triangular mask over 3x3: cmp_ge(iota axis 0, iota axis 1).
    let (r, c) = (counts(D::I32, &[3, 3], 0), counts(D::I32, &[3, 3], 1));
    let tri: Vec<bool> = r.iter().zip(&c).map(|(r, c)| r >= c).collect();
    assert_eq!(
        tri,
        vec![true, false, false, true, true, false, true, true, true]
    );
    // one-hot of index 2 over 4 classes: cmp_eq(iota, 2).
    let hot: Vec<bool> = counts(D::I32, &[4], 0).iter().map(|i| *i == 2).collect();
    assert_eq!(hot, vec![false, false, true, false]);
}
