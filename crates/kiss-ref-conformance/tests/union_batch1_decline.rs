// SPDX-License-Identifier: MIT OR Apache-2.0
//! The seven KISS union-batch-1 dtype tokens are vocabulary only: the bit-level data-movement path declines them
//! by name (never a silent wrong width), and the coverage ledger marks them Pending, not Done.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::cast::{cast_bits, CastRounding, CastSaturation};
use kiss_ref_core::{support, Error, Support};

#[test]
fn union_batch1_tokens_decline_cast_by_name() {
    for d in [Dtype::F128, Dtype::I128, Dtype::U128, Dtype::I256, Dtype::U256, Dtype::F8e3m4] {
        // identity cast used to mask to a wrong width; it must now decline
        let r = cast_bits(0xAB, d, d, CastRounding::NearestEven, CastSaturation::None);
        assert!(matches!(r, Err(Error::UnsupportedDtype(x)) if x == d), "{d:?}: {r:?}");
        assert_ne!(support(Op::Cast, d), Support::Done, "{d:?} must not be Done for cast");
        assert_ne!(support(Op::Concat, d), Support::Done, "{d:?} must not be Done for concat");
    }
    // the scale type keeps its own decline
    let r = cast_bits(1, Dtype::F8ue4m3, Dtype::F32, CastRounding::NearestEven, CastSaturation::Clamp);
    assert!(matches!(r, Err(Error::ReservedOrScaleDtype(Dtype::F8ue4m3))), "{r:?}");
    // control: a supported dtype still casts and is Done
    assert_eq!(cast_bits(0x41, Dtype::U8, Dtype::U8, CastRounding::NearestEven, CastSaturation::None), Ok(0x41));
    assert_eq!(support(Op::Cast, Dtype::I32), Support::Done);
}
