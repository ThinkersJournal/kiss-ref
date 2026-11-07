// SPDX-License-Identifier: MIT OR Apache-2.0
//! `squeeze` (KISS-Ops §6.11-0022) and `unsqueeze` (§6.11-0023): pure shape, bits preserved,
//! the plain `u8` axes mask of §6.19-0046, typed declines for every unpinned case.

use kiss_ref_core::cast::{squeeze, squeeze_shape, unsqueeze, unsqueeze_shape};
use kiss_ref_core::tensor::MAX_RANK;
use kiss_ref_core::{Error, Tensor};

/// Bit patterns a value-level copy would alter: a signalling NaN with payload, a negative NaN,
/// -0.0, +0.0, a subnormal, 1.0.
const BITS: [u32; 6] = [0x7FA0_0001, 0xFFC0_1234, 0x8000_0000, 0, 1, 0x3F80_0000];

fn t(shape: &[usize]) -> Tensor<u32> {
    let n: usize = shape.iter().product();
    let data = (0..n).map(|i| BITS[i % BITS.len()]).collect();
    Tensor::from_vec(data, shape).unwrap()
}

#[test]
fn ops_squeeze_semantics_shapes() {
    assert_eq!(squeeze_shape(&[1, 3, 1, 4], 0b0001).unwrap(), vec![3, 1, 4]);
    assert_eq!(squeeze_shape(&[1, 3, 1, 4], 0b0101).unwrap(), vec![3, 4]);
    // removing every axis of an all-ones operand gives rank 0.
    assert!(squeeze_shape(&[1, 1], 0b11).unwrap().is_empty());
    // the rank is the operand rank minus the popcount, whatever the extents.
    assert_eq!(squeeze_shape(&[1, 7, 1], 0b100).unwrap().len(), 2);
}

#[test]
fn ops_squeeze_semantics_declines() {
    assert_eq!(squeeze_shape(&[1, 3], 0), Err(Error::AxesMaskEmpty));
    assert_eq!(
        squeeze_shape(&[1, 3], 0b100),
        Err(Error::AxesMaskBitOutOfRange { bit: 2, rank: 2 })
    );
    assert_eq!(
        squeeze_shape(&[1, 3], 0b10),
        Err(Error::SqueezeExtentNotOne { axis: 1, extent: 3 })
    );
    // rank 0: every non-empty mask is out of range.
    assert_eq!(
        squeeze_shape(&[], 1),
        Err(Error::AxesMaskBitOutOfRange { bit: 0, rank: 0 })
    );
}

#[test]
fn ops_unsqueeze_semantics_shapes() {
    assert_eq!(unsqueeze_shape(&[3, 4], 0b001).unwrap(), vec![1, 3, 4]);
    assert_eq!(unsqueeze_shape(&[3, 4], 0b101).unwrap(), vec![1, 3, 1, 4]);
    // an output axis may be the last one.
    assert_eq!(unsqueeze_shape(&[3, 4], 0b100).unwrap(), vec![3, 4, 1]);
    // rank 0 operand: the output is all ones, one per mask bit.
    assert_eq!(unsqueeze_shape(&[], 0b11).unwrap(), vec![1, 1]);
}

#[test]
fn ops_unsqueeze_semantics_declines() {
    assert_eq!(unsqueeze_shape(&[3, 4], 0), Err(Error::AxesMaskEmpty));
    // output rank 3: bit 3 is out of range.
    assert_eq!(
        unsqueeze_shape(&[3, 4], 0b1000),
        Err(Error::AxesMaskBitOutOfRange { bit: 3, rank: 3 })
    );
    // an output rank above MAX_RANK.
    let big = vec![1usize; MAX_RANK];
    assert_eq!(
        unsqueeze_shape(&big, 0b1),
        Err(Error::RankExceeded {
            rank: MAX_RANK + 1,
            max: MAX_RANK
        })
    );
}

#[test]
fn ops_squeeze_unsqueeze_preserve_every_bit() {
    let x = t(&[1, 3, 1, 2]);
    let y = squeeze(&x.view(), 0b0101).unwrap();
    assert_eq!(y.shape(), &[3, 2]);
    assert_eq!(y.as_slice(), x.as_slice());
    let z = unsqueeze(&y.view(), 0b0101).unwrap();
    assert_eq!(z.shape(), x.shape());
    assert_eq!(z.as_slice(), x.as_slice());
    // the NaN payload and the signed zero survived as bit patterns, not merely as values.
    assert_eq!(z.as_slice()[0], 0x7FA0_0001);
    assert_eq!(z.as_slice()[2], 0x8000_0000);
}

/// `unsqueeze(squeeze(x, S), S) = x` over every legal `S` of an all-ones-and-twos operand.
#[test]
fn ops_squeeze_unsqueeze_round_trip_all_masks() {
    let shape = [1usize, 2, 1, 1, 3];
    let x = t(&shape);
    let mut legal = 0;
    for mask in 1u8..=0x1F {
        let Ok(y) = squeeze(&x.view(), mask) else {
            continue;
        };
        legal += 1;
        let z = unsqueeze(&y.view(), mask).unwrap();
        assert_eq!(z.shape(), &shape);
        assert_eq!(z.as_slice(), x.as_slice(), "mask {mask:#x}");
    }
    // the extent-1 axes are 0, 2, 3: every non-empty subset of them is legal, nothing else.
    assert_eq!(legal, 7);
}
