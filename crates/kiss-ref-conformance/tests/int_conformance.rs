// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integer-path conformance corpus — the pinned integer semantics of KISS-Ops
//! §6.2 / §6.4 / §6.10, with KISS-Conform `test_ops_*` names.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_int_op, tensor_int, Error, Monoid, Tensor};

fn ev(op: Op, d: Dtype, args: &[i128]) -> i128 {
    eval_int_op(op, d, args).unwrap_or_else(|e| panic!("{op:?}/{d:?} failed: {e:?}"))
}

fn t(data: &[i128], shape: &[usize]) -> Tensor<i128> {
    Tensor::from_vec(data.to_vec(), shape).unwrap_or_else(|e| panic!("tensor {shape:?}: {e:?}"))
}

#[test]
fn test_ops_int_wrapping() {
    // KISS-OPS-6.2-0002: add/sub/mul are wrapping two's-complement.
    assert_eq!(ev(Op::Add, Dtype::I8, &[127, 1]), -128);
    assert_eq!(ev(Op::Mul, Dtype::U8, &[16, 16]), 0); // 256 mod 256
    assert_eq!(ev(Op::Sub, Dtype::U8, &[0, 1]), 255);
}

#[test]
fn test_ops_int_packed_subbyte_wrap() {
    // KISS-OPS-6.2-0002 wrapping two's-complement at the PACKED sub-byte widths, and
    // KISS-OPS-6.4-0005 (neg/abs of INT_MIN stay INT_MIN). i4/u4/b1 were Done by
    // genericity (eval_int_op wraps via int_spec width) but never value-tested — these
    // pin the 4-bit and 1-bit boundaries.

    // i4: signed 4-bit, range -8..=7
    assert_eq!(ev(Op::Add, Dtype::I4, &[7, 1]), -8); // 8 & 0xF = 8, bit3 set -> 8-16
    assert_eq!(ev(Op::Add, Dtype::I4, &[-8, -1]), 7); // -9 & 0xF = 7
    assert_eq!(ev(Op::Sub, Dtype::I4, &[-8, 1]), 7);
    assert_eq!(ev(Op::Mul, Dtype::I4, &[3, 3]), -7); // 9 -> 9-16
    assert_eq!(ev(Op::Mul, Dtype::I4, &[-3, 3]), 7); // -9 & 0xF = 7
    assert_eq!(ev(Op::Neg, Dtype::I4, &[-8]), -8); // KISS-OPS-6.4-0005
    assert_eq!(ev(Op::Abs, Dtype::I4, &[-8]), -8);

    // u4: unsigned 4-bit, range 0..=15
    assert_eq!(ev(Op::Add, Dtype::U4, &[15, 1]), 0);
    assert_eq!(ev(Op::Add, Dtype::U4, &[15, 15]), 14); // 30 & 0xF
    assert_eq!(ev(Op::Sub, Dtype::U4, &[0, 1]), 15);
    assert_eq!(ev(Op::Mul, Dtype::U4, &[4, 4]), 0); // 16 & 0xF
    assert_eq!(ev(Op::Mul, Dtype::U4, &[3, 6]), 2); // 18 & 0xF
    assert_eq!(ev(Op::BitNot, Dtype::U4, &[0]), 15);
    assert_eq!(ev(Op::BitNot, Dtype::U4, &[5]), 10);

    // b1: unsigned 1-bit, range 0..=1
    assert_eq!(ev(Op::Add, Dtype::B1, &[1, 1]), 0); // 2 & 1
    assert_eq!(ev(Op::Sub, Dtype::B1, &[0, 1]), 1); // -1 & 1
    assert_eq!(ev(Op::Mul, Dtype::B1, &[1, 1]), 1);
    assert_eq!(ev(Op::BitXor, Dtype::B1, &[1, 1]), 0);
    assert_eq!(ev(Op::BitOr, Dtype::B1, &[1, 0]), 1);
    assert_eq!(ev(Op::BitAnd, Dtype::B1, &[1, 0]), 0);
    assert_eq!(ev(Op::BitNot, Dtype::B1, &[0]), 1);
    assert_eq!(ev(Op::LogicalAnd, Dtype::B1, &[1, 1]), 1);
    assert_eq!(ev(Op::LogicalNot, Dtype::B1, &[0]), 1);
    assert_eq!(ev(Op::CmpLt, Dtype::B1, &[0, 1]), 1);
}

#[test]
fn test_ops_int_neg_abs_wrap() {
    // KISS-OPS-6.4-0005: neg/abs of INT_MIN stay INT_MIN, no UB, no saturation.
    assert_eq!(
        ev(Op::Neg, Dtype::I32, &[i32::MIN as i128]),
        i32::MIN as i128
    );
    assert_eq!(
        ev(Op::Abs, Dtype::I32, &[i32::MIN as i128]),
        i32::MIN as i128
    );
    assert_eq!(ev(Op::Neg, Dtype::I32, &[5]), -5);
}

#[test]
fn test_ops_bitwise_integer_only() {
    // KISS-OPS-6.10-0001: bitwise atoms reject a float operand (typed decline).
    assert!(matches!(
        eval_int_op(Op::BitAnd, Dtype::F32, &[1, 1]),
        Err(Error::UnsupportedDtype(_))
    ));
    // and compute correctly on integers.
    assert_eq!(ev(Op::BitXor, Dtype::U8, &[0b1100, 0b1010]), 0b0110);
    assert_eq!(ev(Op::Popcount, Dtype::U16, &[0xFFFF]), 16);
}

#[test]
fn test_ops_u32_ordinary_dtype() {
    // KISS-OPS-6.2-0007: u32 is an ordinary unsigned integer dtype.
    assert_eq!(ev(Op::Add, Dtype::U32, &[u32::MAX as i128, 1]), 0);
    assert_eq!(ev(Op::CmpGt, Dtype::U32, &[5, 3]), 1);
}

// ---- §6.13 non-primitive activations / minmax on the integer lane -----------
// The reference decompositions (`decomp.rs`) for these ops are float-shaped —
// they carry NaN guards (`cmp_ne(a,a)`) and select on a strict/loose threshold —
// but over integers there is NO NaN and NO ±0.0, so those guards are inert and
// the ops degenerate to their pure integer forms. Goldens are hand-computed from
// the decomposition string, wrapped two's-complement per KISS-OPS-6.2-0002.

#[test]
fn test_ops_int_sqr_wraps() {
    // KISS-OPS-6.13: sqr(x) = "mul(x, x)"; the product wraps to width exactly
    // like Op::Mul (KISS-OPS-6.2-0002) — there is NO wide accumulator.
    assert_eq!(ev(Op::Sqr, Dtype::I8, &[5]), 25);
    assert_eq!(ev(Op::Sqr, Dtype::I8, &[-5]), 25); // (-5)^2 = 25
    assert_eq!(ev(Op::Sqr, Dtype::I8, &[16]), 0); // 256 ≡ 0 (mod 256)
    assert_eq!(ev(Op::Sqr, Dtype::I8, &[12]), -112); // 144 = 0b1001_0000 → sign-ext
    assert_eq!(ev(Op::Sqr, Dtype::U8, &[12]), 144); // fits the unsigned pattern
    assert_eq!(ev(Op::Sqr, Dtype::U8, &[20]), 144); // 400 ≡ 144 (mod 256)
                                                    // (2^64-1)^2 exceeds i128 — must use wrapping_mul; low 64 bits are 1.
    assert_eq!(ev(Op::Sqr, Dtype::U64, &[u64::MAX as i128]), 1);
}

#[test]
fn test_ops_int_step_strict_threshold() {
    // KISS-OPS-6.13: step(x) = "select(cmp_gt(x, const(0)), const(1), const(0))"
    // — strict threshold at 0, so step(0) = 0. Output {0,1} fits every width.
    assert_eq!(ev(Op::Step, Dtype::I8, &[7]), 1);
    assert_eq!(ev(Op::Step, Dtype::I8, &[0]), 0); // strict: not > 0
    assert_eq!(ev(Op::Step, Dtype::I8, &[-1]), 0);
    assert_eq!(ev(Op::Step, Dtype::U8, &[200]), 1);
    assert_eq!(ev(Op::Step, Dtype::U8, &[0]), 0);
    assert_eq!(ev(Op::Step, Dtype::B1, &[1]), 1); // fits the 1-bit lane
}

#[test]
fn test_ops_int_relu_clamps_at_zero() {
    // KISS-OPS-6.13: relu(x) = "select(cmp_lt(x, const(0)), const(0), x)" = max(x,0).
    assert_eq!(ev(Op::Relu, Dtype::I8, &[-5]), 0);
    assert_eq!(ev(Op::Relu, Dtype::I8, &[5]), 5);
    assert_eq!(ev(Op::Relu, Dtype::I8, &[0]), 0);
    assert_eq!(ev(Op::Relu, Dtype::I8, &[-128]), 0); // INT_MIN clamps to 0
                                                     // On an unsigned dtype `x < 0` is impossible → relu is the identity.
    assert_eq!(ev(Op::Relu, Dtype::U8, &[200]), 200);
}

#[test]
fn test_ops_int_fmax_fmin_degenerate_to_minmax() {
    // KISS-OPS-6.13: fmax_ieee/fmin_ieee carry NaN-propagation guards
    // (`cmp_ne(a,a)`) in their decompositions. On integers there is no NaN and no
    // ±0.0 ordering, so those guards are inert and the ops are plain integer
    // max/min — byte-identical to max_prop/min_prop.
    assert_eq!(ev(Op::FmaxIeee, Dtype::I8, &[-3, 7]), 7);
    assert_eq!(ev(Op::FminIeee, Dtype::I8, &[-3, 7]), -3);
    assert_eq!(ev(Op::FmaxIeee, Dtype::U8, &[200, 100]), 200);
    assert_eq!(ev(Op::FminIeee, Dtype::U8, &[200, 100]), 100);
    // parity with the ordered min/max props on the same inputs.
    assert_eq!(
        ev(Op::FmaxIeee, Dtype::I8, &[-3, 7]),
        ev(Op::MaxProp, Dtype::I8, &[-3, 7])
    );
    assert_eq!(
        ev(Op::FminIeee, Dtype::I8, &[-3, 7]),
        ev(Op::MinProp, Dtype::I8, &[-3, 7])
    );
}

// ---- integer tensor lane: matmul / max_pool / im2col ------------------------

#[test]
fn test_ops_int_matmul_wraps_per_atom() {
    // KISS-OPS-6.13: matmul = reduce(sum, axis=K) of element_map(mul(a, b)); the
    // sum identity is 0 (KISS-OPS-6.11-0002). Every multiply and add wraps IN the
    // dtype (KISS-OPS-6.2-0002) — no wide accumulator escapes it.
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]] (no wrap in i32).
    let a = t(&[1, 2, 3, 4], &[2, 2]);
    let b = t(&[5, 6, 7, 8], &[2, 2]);
    let r = tensor_int::matmul(&a.view(), &b.view(), Dtype::I32).unwrap();
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(r.as_slice(), &[19, 22, 43, 50]);

    // i8 1×1 · 1×1: the single product 100*100 = 10000 wraps IN i8 to 16
    // (10000 mod 256 = 16) — proof the *multiply* itself is at dtype width, not a
    // wide i128 accumulator (which would keep 10000).
    let rp = tensor_int::matmul(
        &t(&[100], &[1, 1]).view(),
        &t(&[100], &[1, 1]).view(),
        Dtype::I8,
    )
    .unwrap();
    assert_eq!(rp.as_slice(), &[16]);

    // u8 accumulator wrap: [200,200] · [1;1] = 400 ≡ 144 (mod 256); the wrap is
    // in the running sum, not saturation.
    let ru = tensor_int::matmul(
        &t(&[200, 200], &[1, 2]).view(),
        &t(&[1, 1], &[2, 1]).view(),
        Dtype::U8,
    )
    .unwrap();
    assert_eq!(ru.as_slice(), &[144]);
}

#[test]
fn test_ops_int_max_pool_window_and_identity() {
    // KISS-OPS-6.13: max_pool = reduce(max) over the window; the max identity is
    // the dtype minimum (KISS-OPS-6.11-0002), NOT −inf (no float infinities).
    // [1,3,2,5], kernel 2 stride 1 → [max(1,3), max(3,2), max(2,5)] = [3,3,5].
    let x = t(&[1, 3, 2, 5], &[4]);
    let y = tensor_int::max_pool(&x.view(), &[0], &[2], &[1], &[0], &[1], Dtype::I8).unwrap();
    assert_eq!(y.as_slice(), &[3, 3, 5]);

    // A wholly-out-of-bounds (padded) window yields the S8 max identity = -128,
    // not −inf: extent 1, kernel 2 stride 1 pad 2 → first window taps land at
    // -2,-1 (both OOB) → identity.
    let z = t(&[7], &[1]);
    let yz = tensor_int::max_pool(&z.view(), &[0], &[2], &[1], &[2], &[1], Dtype::I8).unwrap();
    assert_eq!(yz.as_slice()[0], -128);
}

#[test]
fn test_ops_int_im2col_zero_fill_oob() {
    // KISS-OPS-6.13: im2col is pure zero-filled data movement — OOB taps read 0,
    // exact and integer-closed (no arithmetic, so no wrap can occur).
    // [10,20,30], kernel 2 stride 1 pad 1 → 4 patches × 2 taps:
    // patch0=[0,10], patch1=[10,20], patch2=[20,30], patch3=[30,0].
    let x = t(&[10, 20, 30], &[3]);
    let y = tensor_int::im2col(&x.view(), &[0], &[2], &[1], &[1], &[1]).unwrap();
    assert_eq!(y.shape(), &[4, 2]);
    assert_eq!(y.as_slice(), &[0, 10, 10, 20, 20, 30, 30, 0]);
}

#[test]
fn test_ops_int_packed_subbyte_tensor_lane() {
    // The PACKED sub-byte dtypes evaluated through the TENSOR fold (tensor_int::reduce),
    // not just scalar eval_int_op — confirms the width-wrap propagates through the
    // reduction. KISS-OPS-6.11-0002 Sum monoid + KISS-OPS-6.2-0002 two's-complement wrap.
    // i4: sum [7,1] = 8 -> wraps to -8 (bit3 set).
    let s = tensor_int::reduce(&t(&[7, 1], &[2]).view(), Dtype::I4, Monoid::Sum, &[0]).unwrap();
    assert_eq!(s.as_slice(), &[-8]);
    // u4: sum [15,1] = 16 -> 16 & 0xF = 0.
    let u = tensor_int::reduce(&t(&[15, 1], &[2]).view(), Dtype::U4, Monoid::Sum, &[0]).unwrap();
    assert_eq!(u.as_slice(), &[0]);
    // b1: sum [1,1] = 2 -> 2 & 1 = 0.
    let b = tensor_int::reduce(&t(&[1, 1], &[2]).view(), Dtype::B1, Monoid::Sum, &[0]).unwrap();
    assert_eq!(b.as_slice(), &[0]);
}
