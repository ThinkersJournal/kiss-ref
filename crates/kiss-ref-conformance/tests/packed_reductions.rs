// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.11-0026 to -0029: the tuple-state `reduce` monoids `welford`, `online_softmax`, `argmax`,
//! replayed over KISS's own vectors (`fixtures/kiss-corpus/packed_reduction_vectors.json`, f32 state; lanes
//! are 8-hex-digit words in trailing-axis order). `exact-byte` cases are compared bit for bit, `tolerance`
//! cases within 2 ULP of the vector (the reduction is a tolerance-cell, §6.17-0008).

use kiss_classify_vocab::Dtype as D;
use kiss_ref_core::packed::{
    online_softmax_finalise, packed_shape, reduce_packed, state_dtype, welford_finalise,
    TupleMonoid,
};
use kiss_ref_core::tensor::Tensor;
use kiss_ref_core::Error;
use serde_json::Value;

fn corpus() -> Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/kiss-corpus/packed_reduction_vectors.json");
    serde_json::from_str(&std::fs::read_to_string(p).expect("read corpus")).expect("parse corpus")
}

fn kind(s: &str) -> TupleMonoid {
    match s {
        "welford" => TupleMonoid::Welford,
        "online_softmax" => TupleMonoid::OnlineSoftmax,
        "argmax" => TupleMonoid::Argmax,
        other => panic!("unknown kind {other}"),
    }
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("hex word")
}

/// ULP distance between two f32 bit patterns of the same sign class (NaNs equal only if both NaN).
fn within_ulps(a: u64, b: u64, ulps: i64) -> bool {
    let (fa, fb) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
    if fa.is_nan() || fb.is_nan() {
        return fa.is_nan() && fb.is_nan();
    }
    (a as i64 - b as i64).abs() <= ulps
}

#[test]
fn packed_reduction_value_vectors() {
    let v = corpus();
    let cases = v["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        19,
        "corpus population moved: re-vendor deliberately"
    );
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let input: Vec<u64> = c["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| hex(h.as_str().unwrap()))
            .collect();
        let want: Vec<u64> = c["lanes"].as_str().unwrap().split(' ').map(hex).collect();
        let k = kind(c["kind"].as_str().unwrap());
        let x = Tensor::from_vec(input.clone(), &[input.len()]).unwrap();
        let got = reduce_packed(k, D::F32, &x.view(), &[0], false).unwrap();
        assert_eq!(got.shape(), &[k.lanes()], "{name}");
        let tol = c["class"] == "tolerance";
        for (i, (g, w)) in got.as_slice().iter().zip(&want).enumerate() {
            // the count / index lane is never a tolerance lane.
            let exact = !tol || i == 0 && k == TupleMonoid::Welford || k == TupleMonoid::Argmax;
            let ok = if exact {
                g == w
            } else {
                within_ulps(*g, *w, 2)
            };
            assert!(ok, "{name} lane {i}: got {g:#x}, want {w:#x}");
        }
    }
}

#[test]
fn packed_reduction_shape_vectors() {
    let v = corpus();
    let cases = v["shape_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    let (mut ok, mut declined) = (0, 0);
    for c in cases {
        let name = c["name"].as_str().unwrap();
        let shape: Vec<usize> = c["in_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let axes: Vec<usize> = c["reduced_axes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let got = packed_shape(
            &shape,
            &axes,
            c["keepdim"].as_bool().unwrap(),
            kind(c["kind"].as_str().unwrap()),
        );
        let out = c["out"].as_str().unwrap();
        match out.strip_prefix("decline:") {
            Some(d) if d.starts_with("ArgmaxAxisCount") => {
                assert!(
                    matches!(got, Err(Error::ArgmaxAxisCount { got: 2 })),
                    "{name}"
                );
                declined += 1;
            }
            Some(d) => {
                assert!(d.starts_with("RankExceedsMax { rank: 9 }"), "{name}: {d}");
                assert!(
                    matches!(got, Err(Error::RankExceeded { rank: 9, max: 8 })),
                    "{name}: {got:?}"
                );
                declined += 1;
            }
            None => {
                assert_eq!(format!("{:?}", got.unwrap()), out, "{name}");
                ok += 1;
            }
        }
    }
    assert_eq!((ok, declined), (6, 2));
}

#[test]
fn state_dtype_and_finalise() {
    assert_eq!(state_dtype(D::F32), Ok(D::F32));
    assert_eq!(state_dtype(D::F64), Ok(D::F64));
    for d in [D::F16, D::Bf16, D::F8e4m3fn, D::F8e5m2] {
        assert_eq!(state_dtype(d), Ok(D::F32), "{d:?}");
    }
    assert_eq!(state_dtype(D::I32), Err(Error::UnsupportedDtype(D::I32)));
    assert_eq!(
        state_dtype(D::F8e4m3fnuz),
        Err(Error::ReservedOrScaleDtype(D::F8e4m3fnuz))
    );
    // welford finalise: mean 2.5, M2 5, n 4 -> population 1.25, Bessel 5/3.
    let st = [4u64, 0x4020_0000, 0x40A0_0000];
    assert_eq!(
        f32::from_bits(welford_finalise(D::F32, st, false).unwrap() as u32),
        1.25
    );
    assert_eq!(
        f32::from_bits(welford_finalise(D::F32, st, true).unwrap() as u32),
        5.0f32 / 3.0
    );
    // an empty state (and a corrected single element) is 0/0 = NaN.
    assert!(f32::from_bits(welford_finalise(D::F32, [0, 0, 0], false).unwrap() as u32).is_nan());
    assert!(
        f32::from_bits(welford_finalise(D::F32, [1, 0x4060_0000, 0], true).unwrap() as u32)
            .is_nan()
    );
    // online_softmax finalise: the empty state is -inf; (2, 1 + e^-1) is the logsumexp of {1, 2}.
    assert_eq!(
        f32::from_bits(online_softmax_finalise(D::F32, [0xFF80_0000, 0]).unwrap() as u32),
        f32::NEG_INFINITY
    );
    let l =
        f32::from_bits(online_softmax_finalise(D::F32, [0x4000_0000, 0x3FAF_16AC]).unwrap() as u32);
    assert!((l - (1.0f32.exp() + 2.0f32.exp()).ln()).abs() < 1e-6, "{l}");
}

#[test]
fn f64_state_packs_u64_lanes_and_narrow_compute_widens_to_f32() {
    // f64: the count lane is a u64.
    let x = Tensor::from_vec(vec![1.0f64.to_bits(), 3.0f64.to_bits()], &[2]).unwrap();
    let w = reduce_packed(TupleMonoid::Welford, D::F64, &x.view(), &[0], true).unwrap();
    assert_eq!(w.shape(), &[1, 3]);
    assert_eq!(
        w.as_slice(),
        &[2, 2.0f64.to_bits(), 2.0f64.to_bits()],
        "n=2 mean=2 M2=2"
    );
    // f16 compute -> f32 state: 0x3C00 is 1.0, 0x4200 is 3.0 in f16.
    let h = Tensor::from_vec(vec![0x3C00u64, 0x4200], &[2]).unwrap();
    let wh = reduce_packed(TupleMonoid::Welford, D::F16, &h.view(), &[0], false).unwrap();
    assert_eq!(
        wh.as_slice(),
        &[2, u64::from(2.0f32.to_bits()), u64::from(2.0f32.to_bits())]
    );
    // a non-float compute dtype declines.
    assert_eq!(
        reduce_packed(TupleMonoid::Argmax, D::I32, &x.view(), &[0], false).err(),
        Some(Error::UnsupportedDtype(D::I32))
    );
    // argmax reduces one axis and reports its zero-based position, in the u64 lane at f64.
    let m = Tensor::from_vec(
        [5.0f64, 9.0, 9.0, 1.0, 7.0, 2.0]
            .iter()
            .map(|v| v.to_bits())
            .collect(),
        &[2, 3],
    )
    .unwrap();
    let a = reduce_packed(TupleMonoid::Argmax, D::F64, &m.view(), &[1], false).unwrap();
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(a.as_slice(), &[9.0f64.to_bits(), 1, 7.0f64.to_bits(), 1]);
}
