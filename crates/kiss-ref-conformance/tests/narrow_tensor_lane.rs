//! **Narrow-lane EXECUTION coverage** — the tensor layer and the recipe
//! evaluator actually run on `f16`, `bf16`, `e4m3fn`, and `e5m2`.
//!
//! ## Why this file exists
//!
//! `resolve::implemented_on()` maps `F16`/`Bf16`/`E4m3`/`E5m2` to `Done` whenever
//! `tensor_supported(op)` holds, and `tensor_supported` covers all 28 tensor ops.
//! So the coverage ledger asserted `Done` for 28 ops × 4 narrow dtypes = **112
//! cells purely because the code is generic over `T: ScalarFloat`** — no test had
//! ever *executed* the tensor layer or the recipe evaluator on a narrow dtype
//! (`tensor_ops.rs` and `recipe.rs` contain zero `f16`/`bf16`/FP8 mentions). This
//! file is the execution evidence behind those cells.
//!
//! ## The accumulation semantics this file pins
//!
//! `kernels::reduce` / `kernels::prefix_scan` / `tensor_ops::matmul` fold through
//! `eval_op::<T>`, so on a narrow lane every fold step promotes to `f32`,
//! computes, and **rounds back to the narrow dtype**: the accumulator is
//! **narrow, with per-step rounding**. What KISS says about that:
//!
//! - **KISS-OPS-6.2-0001** — for `bf16`/`e4m3fn`/`e5m2` "the arithmetic MUST
//!   follow the encodings, rounding, saturation, and NaN/infinity conventions
//!   pinned in §6.16"; §6.16-0003/-0004/-0005 pin round-to-nearest-even results
//!   and, for FP8, **saturation to max-finite**. Per-atom rounding *into the
//!   narrow dtype* is therefore mandatory — the reference is right about the
//!   per-atom step.
//! - **KISS-OPS-6.17-0002** — under `bit-stable` MathPrecision "every
//!   floating-point arithmetic atom in an op's evaluation MUST round
//!   independently at the storage dtype's full … precision".
//! - **KISS-OPS-6.17-0005** — for an order-invariant/nondeterministic op
//!   (a float reduction, scan, or contraction) bit-stable is verified "under a
//!   **pinned bit-stable reference profile** (ascending-index reduction order
//!   with the accumulator at the storage dtype's precision)". kiss-ref's fold is
//!   exactly that profile: ascending index order, accumulator at storage
//!   precision. **The reference is conformant.**
//! - **KISS-OPS-6.0-0004** — but for a *candidate*, "neither a canonical
//!   reduction order nor a canonical **accumulator width** is pinned by
//!   KISS-Ops", and ops.md's own open-questions list (item 6, "Float reduction
//!   determinism") records the accumulator dtype as deliberately **unpinned**.
//!
//! So a candidate that accumulates an `e4m3` reduce in `f32` (what every real
//! FP8 tensor core does) is *equally* conformant and can differ from this
//! reference by **arbitrarily much** — see
//! [`narrow_reduce_sum_stagnates_in_a_narrow_accumulator`] (128 vs 192 on the
//! same inputs) and [`narrow_matmul_saturates_in_e4m3`] (448 vs 512). No KISS
//! clause declares a tolerance that bounds this, which §6.17-0007 requires of
//! every order-invariant/nondeterministic cell. That is a genuine spec gap and
//! the tests below pin the reference's actual behavior rather than dodge it.
//!
//! Every expectation here is hand-derived from the format definitions (§6.16
//! mantissa widths and max-finite magnitudes) FIRST; the evaluator is checked
//! against it, never the source of it.
//!
//! Comparison mode follows §6.0: exact-byte cells (`element_map`, `gather`,
//! `scatter`(assign), `sort_network`, max/min folds, `flip`, `iota`) are compared
//! **raw-bit**; order-invariant/nondeterministic cells (sum/prod folds, `matmul`,
//! `scatter_add`, `softmax`) are *also* compared raw-bit here because this file
//! pins the REFERENCE's own bits (the §6.17-0005 profile), not a candidate's —
//! the class governs how a candidate is judged, which is exactly the gap above.

use half::{bf16, f16};
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::{ConstSym, Expr};
use kiss_ops_vocab::Op;
use kiss_ref_core::kernels::{
    element_map, gather, prefix_scan, reduce, reduce_ref, scatter, sort_network,
};
use kiss_ref_core::tensor_ops::{index_select, matmul, matmul_ref, scatter_add, softmax};
use kiss_ref_core::Combine;
use kiss_ref_core::{
    eval_op, eval_recipe, support, DetClass, Direction, E4m3, E5m2, Error, FlatDag, IndexRef,
    IndexTensor, Monoid, Node, OobPolicy, ScalarFloat, Support, Tensor,
};

// ---- the narrow-lane harness -------------------------------------------------

/// A narrow float lane under test: `f16`, `bf16`, `e4m3fn`, `e5m2`. Adds the
/// three things `ScalarFloat` deliberately does not carry — a name for failure
/// messages, an `f32` round-trip, and raw storage bits for exact-byte compares.
trait Narrow: ScalarFloat + core::fmt::Debug {
    /// The §6.16 dtype token.
    const NAME: &'static str;
    /// Explicit mantissa bits (§6.16): f16 10, bf16 7, e4m3fn 3, e5m2 2.
    const MANTISSA_BITS: u32;
    /// Round an `f32` into this lane (RNE, FP8 saturating — §6.16-0003/-0004/-0005).
    fn of(v: f32) -> Self;
    /// Widen back to `f32` (lossless — every narrow lane is an f32 subset).
    fn f32(self) -> f32;
    /// The raw storage bits, for exact-byte comparison (§6.0-0002).
    fn bits(self) -> u64;
}

impl Narrow for f16 {
    const NAME: &'static str = "f16";
    const MANTISSA_BITS: u32 = 10;
    fn of(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl Narrow for bf16 {
    const NAME: &'static str = "bf16";
    const MANTISSA_BITS: u32 = 7;
    fn of(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl Narrow for E4m3 {
    const NAME: &'static str = "e4m3fn";
    const MANTISSA_BITS: u32 = 3;
    fn of(v: f32) -> Self {
        E4m3::from_f32(v)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl Narrow for E5m2 {
    const NAME: &'static str = "e5m2";
    const MANTISSA_BITS: u32 = 2;
    fn of(v: f32) -> Self {
        E5m2::from_f32(v)
    }
    fn f32(self) -> f32 {
        self.to_f32()
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

/// Build a narrow tensor from exactly-representable `f32` literals.
fn tn<T: Narrow>(v: &[f32], shape: &[usize]) -> Tensor<T> {
    Tensor::from_vec(v.iter().map(|&x| T::of(x)).collect(), shape)
        .unwrap_or_else(|e| panic!("[{}] tensor {shape:?}: {e:?}", T::NAME))
}

/// Widen a narrow tensor's payload for diagnostics.
fn wide<T: Narrow>(t: &Tensor<T>) -> Vec<f32> {
    t.as_slice().iter().map(|&x| x.f32()).collect()
}

/// An `i64` index tensor.
fn ix(data: &[i64], shape: &[usize]) -> IndexTensor {
    IndexTensor::new(data.to_vec(), shape, Dtype::I64)
        .unwrap_or_else(|e| panic!("index {shape:?}: {e:?}"))
}

/// Raw-bit equality against `want` re-rounded into the lane (§6.0-0002): keeps
/// ±0 distinct and matches a NaN only by its actual payload.
fn assert_bits<T: Narrow>(got: &Tensor<T>, want: &[f32], shape: &[usize]) {
    assert_eq!(got.shape(), shape, "[{}] output shape", T::NAME);
    let g: Vec<u64> = got.as_slice().iter().map(|&x| x.bits()).collect();
    let w: Vec<u64> = want.iter().map(|&x| T::of(x).bits()).collect();
    assert_eq!(
        g,
        w,
        "[{}] raw bits: got {:?} want {:?}",
        T::NAME,
        wide(got),
        want
    );
}

/// The same fold this reference performs, but with an `f32` accumulator — the
/// *other* legal reading of §6.0-0004 (accumulator width unpinned). Used to make
/// the divergence explicit in a failing message, never as an expectation.
fn wide_sum(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |a, &b| a + b)
}

/// Assert that a value **is** representable in the lane, so a test that claims
/// "exactly representable inputs" cannot silently be testing rounded ones.
fn assert_exact<T: Narrow>(v: f32) {
    assert_eq!(
        T::of(v).f32(),
        v,
        "[{}] {v} must be exactly representable ({} mantissa bits)",
        T::NAME,
        T::MANTISSA_BITS
    );
}

// ---- element_map (§6.11-0001) ------------------------------------------------

fn check_element_map_mul<T: Narrow>() {
    for &v in &[1.0f32, 2.0, 4.0, 8.0, 0.5] {
        assert_exact::<T>(v);
    }
    // body = mul(input(0), input(1)) — the §6.11-0001 per-element scalar body.
    let body = Expr::Apply(Op::Mul, vec![Expr::Input(0), Expr::Input(1)]);
    let a: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0], &[2, 2]);
    // rank-1 `b` broadcasts over the leading axis (stride-0, §6.11-0001).
    let b: Tensor<T> = tn(&[2.0, 0.5], &[2]);
    let out = element_map(&body, &[a.view(), b.view()], &[2, 2])
        .unwrap_or_else(|e| panic!("[{}] element_map: {e:?}", T::NAME));
    // [[1,2],[4,8]] * [2,0.5] broadcast = [[2,1],[8,4]] — every product is a
    // power of two, exact in all four lanes.
    assert_bits(&out, &[2.0, 1.0, 8.0, 4.0], &[2, 2]);
}

#[test]
fn narrow_element_map_executes_on_every_narrow_dtype() {
    check_element_map_mul::<f16>();
    check_element_map_mul::<bf16>();
    check_element_map_mul::<E4m3>();
    check_element_map_mul::<E5m2>();
}

fn check_element_map_const_leaf<T: Narrow>() {
    // body = add(input(0), const(1)) — exercises the §6.12 const(bits) leaf
    // rounded into the narrow lane by `T::from_f64`.
    let body = Expr::Apply(Op::Add, vec![Expr::Input(0), Expr::Const(ConstSym::One)]);
    let x: Tensor<T> = tn(&[0.5, 1.0, 2.0], &[3]);
    let out = element_map(&body, &[x.view()], &[3])
        .unwrap_or_else(|e| panic!("[{}] element_map const: {e:?}", T::NAME));
    // 1.5 = 1.1b (1 mantissa bit) and 3 = 1.1b·2 — exact even in e5m2 (2 bits).
    assert_bits(&out, &[1.5, 2.0, 3.0], &[3]);
}

#[test]
fn narrow_element_map_const_leaf_rounds_into_the_lane() {
    check_element_map_const_leaf::<f16>();
    check_element_map_const_leaf::<bf16>();
    check_element_map_const_leaf::<E4m3>();
    check_element_map_const_leaf::<E5m2>();
}

// ---- reduce (§6.11-0002 / -0008) ---------------------------------------------

fn check_reduce_sum_exact<T: Narrow>(want: f32) {
    for &v in &[1.0f32, 2.0, 4.0, 8.0] {
        assert_exact::<T>(v);
    }
    let x: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0], &[4]);
    let r = reduce(&x.view(), Monoid::Sum, &[0])
        .unwrap_or_else(|e| panic!("[{}] reduce sum: {e:?}", T::NAME));
    // Ascending-index fold from the +0 identity (§6.11-0002, §6.17-0005 profile):
    // 0 → 1 → 3 → 7 → 15. Partials 1, 3, 7 need ≤ 2 mantissa bits, so they are
    // exact in every lane; the FINAL 15 = 1.111b·2^3 needs 3.
    assert_bits(&r, &[want], &[1]);
}

#[test]
fn narrow_reduce_sum_folds_in_the_narrow_dtype() {
    // f16 (10 bits) / bf16 (7) / e4m3 (3) all hold 15 exactly.
    check_reduce_sum_exact::<f16>(15.0);
    check_reduce_sum_exact::<bf16>(15.0);
    check_reduce_sum_exact::<E4m3>(15.0);
    // e5m2 has only 2 mantissa bits: 15 sits EXACTLY midway between 14
    // (1.11b·2^3, mantissa 11 = odd) and 16 (1.00b·2^4, mantissa 00 = even), so
    // §6.16-0005 round-half-to-even lands on 16. The reduction of four exactly
    // representable inputs is 1 ULP ABOVE the exact answer. Pinned, not dodged.
    check_reduce_sum_exact::<E5m2>(16.0);
    assert_eq!(wide_sum(&[1.0, 2.0, 4.0, 8.0]), 15.0, "the f32-accumulator answer");
}

fn check_reduce_sum_stagnation<T: Narrow>(want: f32) {
    assert_exact::<T>(128.0);
    assert_exact::<T>(8.0);
    // 128 followed by eight 8s. Exact sum = 192, itself exactly representable in
    // EVERY lane (192 = 1.100b·2^7 — 2 mantissa bits).
    let mut v = vec![128.0f32];
    v.extend(core::iter::repeat(8.0f32).take(8));
    assert_exact::<T>(192.0);
    let x: Tensor<T> = tn(&v, &[9]);
    let r = reduce(&x.view(), Monoid::Sum, &[0])
        .unwrap_or_else(|e| panic!("[{}] reduce sum: {e:?}", T::NAME));
    assert_bits(&r, &[want], &[1]);
    assert_eq!(wide_sum(&v), 192.0, "the f32-accumulator answer is 192");
}

#[test]
fn narrow_reduce_sum_stagnates_in_a_narrow_accumulator() {
    // THE HEADLINE. Identical inputs, identical (ascending) fold order; the only
    // variable is the ACCUMULATOR WIDTH, which KISS-OPS-6.0-0004 leaves unpinned
    // ("neither a canonical reduction order nor a canonical accumulator width is
    // pinned by KISS-Ops"; ops.md open-question 6 keeps it open on purpose).
    //
    // f16 ULP at 128 is 0.125 and bf16's is 1.0, so 128+8 is exact and the fold
    // reaches the true 192.
    check_reduce_sum_stagnation::<f16>(192.0);
    check_reduce_sum_stagnation::<bf16>(192.0);
    // e4m3 ULP at 128 is 16: 128+8 = 136 lands EXACTLY midway between 128
    // (mantissa 000, even) and 144 (mantissa 001, odd) → RNE picks 128. Every
    // subsequent +8 stagnates the same way, so the fold NEVER LEAVES 128.
    check_reduce_sum_stagnation::<E4m3>(128.0);
    // e5m2 ULP at 128 is 32: 136 rounds down to 128 outright. Same stagnation.
    check_reduce_sum_stagnation::<E5m2>(128.0);
    // => on e4m3/e5m2 this reference returns 128 where an f32-accumulator
    // implementation (i.e. every FP8 tensor core) returns 192 — a 33% relative
    // divergence between two implementations that are BOTH conformant, with no
    // KISS-declared tolerance bounding it (§6.17-0007 requires one).
}

// ---- RFC #92 (direction b, RATIFIED): accumulator-dtype tolerance cells --------
//
// The OFF-DIAGONAL companion to the stagnation goldens above: the SAME inputs,
// SAME ascending fold order, but a WIDER accumulator dtype declared via <acc> —
// which is the (compute, acc) tolerance cell the ratified RFC keys on. kiss-ref
// holds these Provisional until KISS's clause-text realization PR lands; the
// numeric reference itself is stable (the C3 per-cell reference).

#[test]
fn narrow_reduce_f32_accumulator_recovers_the_true_sum() {
    // THE HEADLINE RECOVERY. reduce with an f32 accumulator over [128, eight 8s]
    // folds to the true 192 in EVERY narrow storage lane (192 = 1.100b·2^7,
    // representable everywhere) — exactly where the diagonal (acc==storage)
    // stagnated to 128 on e4m3/e5m2. This is the (S=fp8, A=f32) cell RFC #92 b
    // exists to make conformant-and-bounded.
    fn f32acc<T: Narrow>() {
        let mut v = vec![128.0f32];
        v.extend(core::iter::repeat(8.0f32).take(8));
        let x: Tensor<T> = tn(&v, &[9]);
        let r = reduce_ref::<T>(&x.view(), Monoid::Sum, &[0], Dtype::F32)
            .unwrap_or_else(|e| panic!("[{}] reduce_ref f32: {e:?}", T::NAME));
        assert_bits(&r, &[192.0], &[1]);
    }
    f32acc::<f16>();
    f32acc::<bf16>();
    f32acc::<E4m3>(); // diagonal gave 128; f32 accumulator gives 192
    f32acc::<E5m2>(); // diagonal gave 128; f32 accumulator gives 192
}

#[test]
fn narrow_reduce_f32_accumulator_result_narrow_is_a_single_rne() {
    // reduce_ref(Sum, f32) of [1,2,4,8]: the f32 fold is 15 EXACTLY; the ONLY
    // storage rounding is the single result-narrow A→S. So the outcome must equal
    // narrowing 15 once: f16/bf16/e4m3 → 15, e5m2 → 16 (15 is midway 14|16, RNE to
    // even 16). The e5m2 → 16 IS identical to the diagonal's stagnated 16, proving
    // the result-narrow is a single RNE and does not double-round.
    fn case<T: Narrow>(want: f32) {
        let x: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0], &[4]);
        let r = reduce_ref::<T>(&x.view(), Monoid::Sum, &[0], Dtype::F32)
            .unwrap_or_else(|e| panic!("[{}] reduce_ref f32: {e:?}", T::NAME));
        assert_bits(&r, &[want], &[1]);
    }
    case::<f16>(15.0);
    case::<bf16>(15.0);
    case::<E4m3>(15.0);
    case::<E5m2>(16.0);
}

#[test]
fn narrow_matmul_f32_accumulator() {
    // matmul with an f32 accumulator: products AND adds happen in f32, only the
    // final result narrows to S. K=8 of 64·1 → f32 512; narrow: f16/bf16/e5m2 →
    // 512, e4m3 → 448 (512 > e4m3 max finite 448 → saturate, §6.16-0004 — the
    // e4m3 448 equals its diagonal, pinning the sole S rounding is the final
    // narrow). Then the [128, eight 8s]·1 contraction → f32 192 everywhere, vs the
    // diagonal e4m3/e5m2 stagnation to 128 — the multiply-and-add-in-A recovery.
    fn k8<T: Narrow>(want: f32) {
        let a: Tensor<T> = tn(&[64.0; 8], &[1, 8]);
        let b: Tensor<T> = tn(&[1.0; 8], &[8, 1]);
        let r = matmul_ref::<T>(&a.view(), &b.view(), Dtype::F32)
            .unwrap_or_else(|e| panic!("[{}] matmul_ref f32: {e:?}", T::NAME));
        assert_bits(&r, &[want], &[1, 1]);
    }
    k8::<f16>(512.0);
    k8::<bf16>(512.0);
    k8::<E5m2>(512.0);
    k8::<E4m3>(448.0);

    fn contract<T: Narrow>() {
        let mut av = vec![128.0f32];
        av.extend(core::iter::repeat(8.0f32).take(8));
        let a: Tensor<T> = tn(&av, &[1, 9]);
        let b: Tensor<T> = tn(&[1.0; 9], &[9, 1]);
        let r = matmul_ref::<T>(&a.view(), &b.view(), Dtype::F32)
            .unwrap_or_else(|e| panic!("[{}] matmul_ref contract f32: {e:?}", T::NAME));
        assert_bits(&r, &[192.0], &[1, 1]);
    }
    contract::<f16>();
    contract::<bf16>();
    contract::<E4m3>();
    contract::<E5m2>();
}

#[test]
fn narrow_reduce_cross_storage_f16_accumulator() {
    // Not only f32 recovers: an f16 accumulator over e5m2 storage also reaches
    // 192 (f16 ULP at 128 = 0.125, so 128+8=136 and the fold are exact in f16;
    // narrow 192 → e5m2 = 192). Exercises the legality set (e5m2 → f16 admitted:
    // f16 subsumes e5m2, exp 5≥5, mant 10≥2).
    let mut v = vec![128.0f32];
    v.extend(core::iter::repeat(8.0f32).take(8));
    let x: Tensor<E5m2> = tn(&v, &[9]);
    let r = reduce_ref::<E5m2>(&x.view(), Monoid::Sum, &[0], Dtype::F16)
        .expect("e5m2 storage / f16 accumulator");
    assert_bits(&r, &[192.0], &[1]);
}

#[test]
fn narrow_accumulator_width_guard_and_maxmin_route_verbatim() {
    // C1 "accumulator at least as wide as storage" — a narrower/incomparable/
    // non-float accumulator is a TYPED decline, never a silent rounding.
    let x16: Tensor<f16> = tn(&[1.0, 2.0], &[2]);
    let xbf: Tensor<bf16> = tn(&[1.0, 2.0], &[2]);
    // f32 storage isn't a `Narrow` lane (the test infra is narrow-only), so build
    // it directly to exercise the wide-storage arm of the width guard.
    let x32: Tensor<f32> = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap();
    assert!(matches!(
        reduce_ref::<f16>(&x16.view(), Monoid::Sum, &[0], Dtype::Bf16),
        Err(Error::AccumulatorTooNarrow { storage: Dtype::F16, acc: Dtype::Bf16 })
    )); // bf16 mantissa 7 < f16 mantissa 10 → incomparable, declined
    assert!(matches!(
        reduce_ref::<bf16>(&xbf.view(), Monoid::Sum, &[0], Dtype::F16),
        Err(Error::AccumulatorTooNarrow { storage: Dtype::Bf16, acc: Dtype::F16 })
    )); // f16 exp 5 < bf16 exp 8 → incomparable, declined
    assert!(matches!(
        reduce_ref::<f32>(&x32.view(), Monoid::Sum, &[0], Dtype::F16),
        Err(Error::AccumulatorTooNarrow { storage: Dtype::F32, acc: Dtype::F16 })
    ));
    assert!(matches!(
        reduce_ref::<f16>(&x16.view(), Monoid::Sum, &[0], Dtype::I32),
        Err(Error::NonFloatAccumulator(Dtype::I32))
    ));
    // (narrow storage, f64 accumulator): now a SINGLE correctly-rounded RNE
    // f64→narrow (round-to-odd f64→f32, then the RNE from_f32 codec), no longer a
    // decline. reduce Sum[1.0, 2.0] folds to 3.0 exactly in the f64 accumulator;
    // 3.0 is f16-exact → f16 3.0.
    assert_bits(
        &reduce_ref::<f16>(&x16.view(), Monoid::Sum, &[0], Dtype::F64)
            .expect("f16 storage / f64 accumulator single-rounds"),
        &[3.0],
        &[1],
    );
    let xe4: Tensor<E4m3> = tn(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    // e4m3 identity·identity: every partial product/sum (1.0, 0.0) is e4m3-exact.
    assert_bits(
        &matmul_ref::<E4m3>(&xe4.view(), &xe4.view(), Dtype::F64)
            .expect("e4m3 storage / f64 accumulator single-rounds"),
        &[1.0, 0.0, 0.0, 1.0],
        &[2, 2],
    );
    // f32 STORAGE with an f64 accumulator is fine (f32 is not a via-f32 narrow):
    assert!(reduce_ref::<f32>(&x32.view(), Monoid::Sum, &[0], Dtype::F64).is_ok());
    // Max/Min are accumulator-invariant (raw-bit selects, no rounding atom): a
    // non-diagonal <acc> routes to the verbatim kernel for ANY acc — byte-identical.
    let xe: Tensor<E4m3> = tn(&[1.0, 3.0, 2.0], &[3]);
    let via_ref = reduce_ref::<E4m3>(&xe.view(), Monoid::Max, &[0], Dtype::F32).unwrap();
    let via_kernel = reduce(&xe.view(), Monoid::Max, &[0]).unwrap();
    assert_eq!(via_ref.as_slice(), via_kernel.as_slice());
}

#[test]
fn narrow_f64_accumulator_single_rounds_not_double() {
    // Fold R = 1 + 2^-11 + 2^-24 EXACTLY in the f64 accumulator (all three inputs
    // are f16-exact, so widening f16→f64 is lossless and the sum is exact). R sits
    // just ABOVE the f16 midpoint M = 1 + 2^-11 (between 1.0 [even f16 mantissa]
    // and 1 + 2^-10 [odd]):
    //   correct single RNE f64→f16: R > M ⇒ rounds UP to 1 + 2^-10.
    //   naive f64→f32→f16 double round: R = M + 2^-24 is the EXACT f32 midpoint
    //     between M and M + 2^-23; RNE ties-to-even → M (even f32 LSB); then M is
    //     the exact f16 midpoint → RNE ties-to-even → 1.0. WRONG.
    //   round-to-odd f64→f32: R inexact ⇒ returns the ODD bracket M + 2^-23, which
    //     RNEs UP to 1 + 2^-10 in f16. CORRECT.
    let two = |k: i32| 2f32.powi(k);
    assert_exact::<f16>(two(-11)); // f16 normal, mantissa 0
    assert_exact::<f16>(two(-24)); // f16 smallest subnormal
    let x: Tensor<f16> = tn(&[1.0, two(-11), two(-24)], &[3]);
    let r = reduce_ref::<f16>(&x.view(), Monoid::Sum, &[0], Dtype::F64)
        .expect("f16 storage / f64 accumulator single-rounds");
    assert_bits(&r, &[1.0 + two(-10)], &[1]);
    // Guard against a silent regression to the double-round answer.
    assert_ne!(r.as_slice()[0].to_bits(), f16::from_f32(1.0).to_bits());
}

#[test]
fn narrow_accumulator_diagonal_is_byte_identical() {
    // THE NON-NEGOTIABLE PROPERTY. On the diagonal acc == T::DTYPE, the ref MUST
    // return the verbatim kernel bit-for-bit — including NaN payloads and signed
    // zero, which the f64 pivot in widen/narrow could canonicalize if the diagonal
    // were (wrongly) routed through *_acc::<T,T>. Fails loudly if a future refactor
    // re-routes the diagonal.
    fn diag<T: Narrow>() {
        let v = &[1.0f32, -0.0, f32::NAN, 2.0, 8.0, 128.0];
        let x: Tensor<T> = tn(v, &[6]);
        let r_ref = reduce_ref::<T>(&x.view(), Monoid::Sum, &[0], T::DTYPE).unwrap();
        let r_ker = reduce(&x.view(), Monoid::Sum, &[0]).unwrap();
        for (a, b) in r_ref.as_slice().iter().zip(r_ker.as_slice()) {
            assert_eq!(a.bits(), b.bits(), "[{}] reduce diagonal not byte-identical", T::NAME);
        }
        let a2: Tensor<T> = tn(&[1.0, -0.0, 2.0, 8.0], &[2, 2]);
        let b2: Tensor<T> = tn(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let m_ref = matmul_ref::<T>(&a2.view(), &b2.view(), T::DTYPE).unwrap();
        let m_ker = matmul(&a2.view(), &b2.view()).unwrap();
        for (a, b) in m_ref.as_slice().iter().zip(m_ker.as_slice()) {
            assert_eq!(a.bits(), b.bits(), "[{}] matmul diagonal not byte-identical", T::NAME);
        }
    }
    diag::<f16>();
    diag::<bf16>();
    diag::<E4m3>();
    diag::<E5m2>();
    // (f32/f64 diagonals take the same `acc == T::DTYPE` short-circuit and are
    // covered verbatim by the 90 kiss-ref-core lib tests on reduce/matmul.)
}

fn check_reduce_max<T: Narrow>() {
    let x: Tensor<T> = tn(&[1.0, -2.0, 4.0, 0.5], &[4]);
    let r = reduce(&x.view(), Monoid::Max, &[0])
        .unwrap_or_else(|e| panic!("[{}] reduce max: {e:?}", T::NAME));
    // max is exact-byte (§6.0-0002): a raw-bit select, no arithmetic, so it is
    // accumulator-width-independent and identical on every lane.
    assert_bits(&r, &[4.0], &[1]);

    // §6.11-0002: the max/min monoids are NaN-PROPAGATING.
    let n: Tensor<T> = tn(&[1.0, f32::NAN, 3.0], &[3]);
    let rn = reduce(&n.view(), Monoid::Max, &[0])
        .unwrap_or_else(|e| panic!("[{}] reduce max nan: {e:?}", T::NAME));
    assert!(
        ScalarFloat::is_nan(rn.as_slice()[0]),
        "[{}] max must propagate NaN, got {:?}",
        T::NAME,
        wide(&rn)
    );

    // keepdim (§6.11-0008): reducing axis 1 of [2,3] leaves [2,1].
    let m: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0, 0.5, 0.25], &[2, 3]);
    let rk = reduce(&m.view(), Monoid::Max, &[1])
        .unwrap_or_else(|e| panic!("[{}] reduce keepdim: {e:?}", T::NAME));
    assert_bits(&rk, &[4.0, 8.0], &[2, 1]);
}

#[test]
fn narrow_reduce_max_is_exact_byte_on_every_narrow_dtype() {
    check_reduce_max::<f16>();
    check_reduce_max::<bf16>();
    check_reduce_max::<E4m3>();
    check_reduce_max::<E5m2>();
}

fn check_reduce_empty_axis<T: Narrow>(max_ident: f32, min_ident: f32) {
    // §6.11-0002: a reduction over an EMPTY axis yields the monoid identity.
    let x: Tensor<T> = tn(&[], &[2, 0]);
    let s = reduce(&x.view(), Monoid::Sum, &[1])
        .unwrap_or_else(|e| panic!("[{}] empty sum: {e:?}", T::NAME));
    assert_bits(&s, &[0.0, 0.0], &[2, 1]); // +0.0 identity, raw-bit positive zero
    let mx = reduce(&x.view(), Monoid::Max, &[1])
        .unwrap_or_else(|e| panic!("[{}] empty max: {e:?}", T::NAME));
    assert_bits(&mx, &[max_ident, max_ident], &[2, 1]);
    let mn = reduce(&x.view(), Monoid::Min, &[1])
        .unwrap_or_else(|e| panic!("[{}] empty min: {e:?}", T::NAME));
    assert_bits(&mn, &[min_ident, min_ident], &[2, 1]);
}

#[test]
fn narrow_reduce_empty_axis_identity_is_the_representable_one() {
    // §6.11-0002 spells the max/min identities as "−∞ / +∞ for float"; the
    // reference materializes them with `T::from_f64(±inf)`.
    check_reduce_empty_axis::<f16>(f32::NEG_INFINITY, f32::INFINITY);
    check_reduce_empty_axis::<bf16>(f32::NEG_INFINITY, f32::INFINITY);
    // e5m2 has IEEE-style infinities (§6.16-0005), so ±∞ survives verbatim.
    check_reduce_empty_axis::<E5m2>(f32::NEG_INFINITY, f32::INFINITY);
    // e4m3fn has NO INFINITY ENCODING (§6.16-0004), and conversion saturates to
    // max-finite — so the "−∞" identity materializes as −448 and "+∞" as +448.
    // That is still a correct monoid identity for the lane (±448 ARE the dtype
    // min/max), but it is NOT the literal value §6.11-0002 names, and an
    // empty-axis max on e4m3 therefore returns a FINITE −448.
    check_reduce_empty_axis::<E4m3>(-448.0, 448.0);
}

// ---- prefix_scan (§6.11-0003) ------------------------------------------------

fn check_prefix_scan<T: Narrow>(inclusive: &[f32], exclusive: &[f32]) {
    let x: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0], &[4]);
    let inc = prefix_scan(&x.view(), Monoid::Sum, 0, false)
        .unwrap_or_else(|e| panic!("[{}] scan inc: {e:?}", T::NAME));
    assert_bits(&inc, inclusive, &[4]);
    let exc = prefix_scan(&x.view(), Monoid::Sum, 0, true)
        .unwrap_or_else(|e| panic!("[{}] scan exc: {e:?}", T::NAME));
    assert_bits(&exc, exclusive, &[4]);
    // max scan is exact-byte and lane-independent.
    let cm: Tensor<T> = tn(&[1.0, 4.0, 2.0, 8.0], &[4]);
    let mx = prefix_scan(&cm.view(), Monoid::Max, 0, false)
        .unwrap_or_else(|e| panic!("[{}] scan max: {e:?}", T::NAME));
    assert_bits(&mx, &[1.0, 4.0, 4.0, 8.0], &[4]);
}

#[test]
fn narrow_prefix_scan_runs_on_every_narrow_dtype() {
    check_prefix_scan::<f16>(&[1.0, 3.0, 7.0, 15.0], &[0.0, 1.0, 3.0, 7.0]);
    check_prefix_scan::<bf16>(&[1.0, 3.0, 7.0, 15.0], &[0.0, 1.0, 3.0, 7.0]);
    check_prefix_scan::<E4m3>(&[1.0, 3.0, 7.0, 15.0], &[0.0, 1.0, 3.0, 7.0]);
    // Same 15 → 16 RNE tie as the reduce (2 mantissa bits); the EXCLUSIVE scan
    // stops at 7 and is therefore exact on e5m2 while the inclusive one is not.
    check_prefix_scan::<E5m2>(&[1.0, 3.0, 7.0, 16.0], &[0.0, 1.0, 3.0, 7.0]);
}

// ---- matmul (§6.13 contraction) ----------------------------------------------

fn check_matmul_exact<T: Narrow>() {
    for &v in &[1.0f32, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0] {
        assert_exact::<T>(v);
    }
    let a: Tensor<T> = tn(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b: Tensor<T> = tn(&[2.0, 0.0, 1.0, 2.0], &[2, 2]);
    let c = matmul(&a.view(), &b.view())
        .unwrap_or_else(|e| panic!("[{}] matmul: {e:?}", T::NAME));
    // [[1,2],[3,4]]·[[2,0],[1,2]]:
    //   c00 = 1·2 + 2·1 = 4    c01 = 1·0 + 2·2 = 4
    //   c10 = 3·2 + 4·1 = 10   c11 = 3·0 + 4·2 = 8
    // Partial sums (0,2,4 / 0,0,4 / 0,6,10 / 0,0,8) all need ≤ 2 mantissa bits.
    assert_bits(&c, &[4.0, 4.0, 10.0, 8.0], &[2, 2]);
}

#[test]
fn narrow_matmul_small_contraction_is_exact_on_every_narrow_dtype() {
    check_matmul_exact::<f16>();
    check_matmul_exact::<bf16>();
    check_matmul_exact::<E4m3>();
    check_matmul_exact::<E5m2>();
}

fn check_matmul_k8<T: Narrow>(want: f32) {
    assert_exact::<T>(64.0);
    let a: Tensor<T> = tn(&[64.0; 8], &[1, 8]);
    let b: Tensor<T> = tn(&[1.0; 8], &[8, 1]);
    let c = matmul(&a.view(), &b.view())
        .unwrap_or_else(|e| panic!("[{}] matmul k=8: {e:?}", T::NAME));
    assert_bits(&c, &[want], &[1, 1]);
}

#[test]
fn narrow_matmul_saturates_in_e4m3() {
    // A K=8 contraction of 64·1. Ascending-K partials: 64,128,192,256,320,384,
    // 448, 512 — every one representable in e4m3 EXCEPT the last (max finite
    // ±448, §6.16-0004).
    check_matmul_k8::<f16>(512.0);
    check_matmul_k8::<bf16>(512.0);
    check_matmul_k8::<E5m2>(512.0); // max finite 57344, plenty of room
    // e4m3fn has no infinity encoding and §6.16-0004 mandates SATURATION, so the
    // eighth accumulation step clamps: the reference returns 448, not 512 and not
    // inf. A candidate accumulating in f32 and rounding once at the end returns
    // 448 too (512 → saturate) — but a candidate accumulating in f16 and storing
    // to e4m3 also returns 448, while one accumulating in f32 and OUTPUTTING f32
    // (the usual FP8 GEMM shape, §6.13 does not pin the output dtype of a
    // narrow contraction either) returns 512. Pinned as the reference's truth.
    check_matmul_k8::<E4m3>(448.0);
}

fn check_matmul_overflow_head<T: Narrow>(input_as_stored: f32, want: f32) {
    // 32768 = 2^15. Representable in f16 (max 65504), bf16, and e5m2 (max 57344)
    // — but NOT in e4m3fn, where storing it already saturates to 448. The test
    // states the stored input explicitly so the input rounding is visible.
    assert_eq!(
        T::of(32768.0).f32(),
        input_as_stored,
        "[{}] 32768 stores as {input_as_stored}",
        T::NAME
    );
    let a: Tensor<T> = tn(&[32768.0, 32768.0], &[1, 2]);
    let b: Tensor<T> = tn(&[1.0, 1.0], &[2, 1]);
    let c = matmul(&a.view(), &b.view())
        .unwrap_or_else(|e| panic!("[{}] matmul overflow: {e:?}", T::NAME));
    assert_bits(&c, &[want], &[1, 1]);
}

#[test]
fn narrow_matmul_overflow_is_inf_on_f16_but_saturates_on_fp8() {
    // f16 is IEEE 754-2019 binary16 (§6.16-0002): 65536 > 65504 overflows to +inf.
    check_matmul_overflow_head::<f16>(32768.0, f32::INFINITY);
    // bf16 carries the binary32 exponent range (§6.16-0003): 65536 is exact.
    check_matmul_overflow_head::<bf16>(32768.0, 65536.0);
    // e5m2 has IEEE-style infinities, but §6.16-0005 pins SATURATION of an
    // overflowing finite value → 57344, NOT inf. Two IEEE-ish lanes, opposite
    // overflow behavior, both per spec.
    check_matmul_overflow_head::<E5m2>(32768.0, 57344.0);
    // e4m3 cannot even hold the operand: 32768 stores as 448 (§6.16-0004
    // saturating conversion), so the contraction is 448+448 = 896 → 448 again.
    check_matmul_overflow_head::<E4m3>(448.0, 448.0);
}

// ---- gather / scatter (§6.11-0004 / -0005 / -0006) ---------------------------

fn check_gather<T: Narrow>() {
    let d: Tensor<T> = tn(&[2.0, 4.0, 6.0], &[3]);
    let idx = ix(&[0, 2, 5, -1], &[4]);
    // §6.11-0004: negative is always OOB (no from-end wrap).
    let zf = gather(&d.view(), &idx, 0, OobPolicy::ZeroFill, None)
        .unwrap_or_else(|e| panic!("[{}] gather zero-fill: {e:?}", T::NAME));
    assert_bits(&zf, &[2.0, 6.0, 0.0, 0.0], &[4]);
    let cl = gather(&d.view(), &idx, 0, OobPolicy::Clamp, None)
        .unwrap_or_else(|e| panic!("[{}] gather clamp: {e:?}", T::NAME));
    assert_bits(&cl, &[2.0, 6.0, 6.0, 2.0], &[4]);

    // gather is a pure RAW-BIT MOVE: a NaN's payload and a −0 survive untouched,
    // with no promote-round trip (the narrow lane changes nothing here).
    let n: Tensor<T> = Tensor::from_vec(
        vec![T::of(f32::NAN), T::of(-0.0), T::of(1.0)],
        &[3],
    )
    .unwrap();
    let g = gather(&n.view(), &ix(&[1, 0], &[2]), 0, OobPolicy::Clamp, None)
        .unwrap_or_else(|e| panic!("[{}] gather raw-bit: {e:?}", T::NAME));
    assert_eq!(
        g.as_slice()[0].bits(),
        T::of(-0.0).bits(),
        "[{}] gather must preserve −0 raw bits",
        T::NAME
    );
    assert_eq!(
        g.as_slice()[1].bits(),
        T::of(f32::NAN).bits(),
        "[{}] gather must preserve the NaN payload",
        T::NAME
    );

    // §6.13 index_select = gather(oob=skip) with a 1-D index (in-range ⇒ no base).
    let table: Tensor<T> = tn(&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], &[3, 2]);
    let sel = index_select(&table.view(), &ix(&[2, 0], &[2]), 0)
        .unwrap_or_else(|e| panic!("[{}] index_select: {e:?}", T::NAME));
    assert_bits(&sel, &[16.0, 32.0, 1.0, 2.0], &[2, 2]);
}

#[test]
fn narrow_gather_is_a_raw_bit_move_on_every_narrow_dtype() {
    check_gather::<f16>();
    check_gather::<bf16>();
    check_gather::<E4m3>();
    check_gather::<E5m2>();
}

fn check_scatter_assign<T: Narrow>() {
    // §6.11-0005 raw-bit assign scatter on a narrow lane (the exact-byte move —
    // the `AtomicAdd` path is exercised separately; this pins the `Assign`
    // combine so `Op::Scatter` in EXECUTED_HERE is execution-backed, not just
    // reached transitively through scatter_add). Values chosen exactly
    // representable in every narrow lane; highest row-major source wins a tie.
    let dest: Tensor<T> = tn(&[7.0, 7.0, 7.0], &[3]);
    let upd: Tensor<T> = tn(&[1.0, 2.0, 4.0], &[3]);
    let out = scatter(dest, &ix(&[0, 2, 0], &[3]), &upd.view(), 0, Combine::Assign)
        .unwrap_or_else(|e| panic!("[{}] scatter(assign): {e:?}", T::NAME));
    // idx [0,2,0]: slot 0 written 1 then 4 (higher source wins) = 4; slot 1
    // untouched = 7; slot 2 = 2.
    assert_bits(&out, &[4.0, 7.0, 2.0], &[3]);
}

fn check_scatter_add<T: Narrow>() {
    // §6.11-0006 float atomic-add, in pinned row-major source order.
    let dest: Tensor<T> = tn(&[0.0, 0.0, 0.0], &[3]);
    let upd: Tensor<T> = tn(&[1.0, 2.0, 4.0], &[3]);
    let out = scatter_add(dest, &ix(&[0, 1, 0], &[3]), &upd.view(), 0)
        .unwrap_or_else(|e| panic!("[{}] scatter_add: {e:?}", T::NAME));
    // slot 0 collects 1+4 = 5 (exact everywhere: 1.01b·2^2), slot 1 gets 2.
    assert_bits(&out, &[5.0, 2.0, 0.0], &[3]);
}

fn check_scatter_add_stagnation<T: Narrow>(want: f32) {
    // The same accumulator-width lesson at the §6.11-0006 atomic-add combine:
    // eight 8s folded into a destination already holding 128.
    let dest: Tensor<T> = tn(&[128.0, 0.0], &[2]);
    let upd: Tensor<T> = tn(&[8.0; 8], &[8]);
    let out = scatter_add(dest, &ix(&[0; 8], &[8]), &upd.view(), 0)
        .unwrap_or_else(|e| panic!("[{}] scatter_add stagnation: {e:?}", T::NAME));
    assert_bits(&out, &[want, 0.0], &[2]);
}

#[test]
fn narrow_scatter_add_folds_in_the_narrow_dtype() {
    // the Assign combine (raw-bit move) — Op::Scatter, execution-backed.
    check_scatter_assign::<f16>();
    check_scatter_assign::<bf16>();
    check_scatter_assign::<E4m3>();
    check_scatter_assign::<E5m2>();
    check_scatter_add::<f16>();
    check_scatter_add::<bf16>();
    check_scatter_add::<E4m3>();
    check_scatter_add::<E5m2>();
    // …and stagnates for exactly the same reason a reduce does.
    check_scatter_add_stagnation::<f16>(192.0);
    check_scatter_add_stagnation::<bf16>(192.0);
    check_scatter_add_stagnation::<E4m3>(128.0);
    check_scatter_add_stagnation::<E5m2>(128.0);
}

// ---- sort_network (§6.11-0007) -----------------------------------------------

fn check_sort_network<T: Narrow>() {
    let x: Tensor<T> = tn(&[3.0, 1.0, f32::NAN, 2.0], &[4]);
    let (vals, idx) = sort_network(&x.view(), 0, Direction::Asc)
        .unwrap_or_else(|e| panic!("[{}] sort asc: {e:?}", T::NAME));
    // NaN orders GREATEST (ascending → last); ties break by lower original index.
    assert_eq!(
        vals.as_slice()[..3].iter().map(|&v| v.f32()).collect::<Vec<_>>(),
        vec![1.0, 2.0, 3.0],
        "[{}] ascending values",
        T::NAME
    );
    assert!(
        ScalarFloat::is_nan(vals.as_slice()[3]),
        "[{}] NaN must sort last ascending",
        T::NAME
    );
    assert_eq!(idx.as_slice(), &[1, 3, 0, 2], "[{}] original-index lane", T::NAME);

    let (dvals, didx) = sort_network(&x.view(), 0, Direction::Desc)
        .unwrap_or_else(|e| panic!("[{}] sort desc: {e:?}", T::NAME));
    assert!(
        ScalarFloat::is_nan(dvals.as_slice()[0]),
        "[{}] NaN must sort first descending",
        T::NAME
    );
    assert_eq!(didx.as_slice(), &[2, 0, 3, 1], "[{}] descending index lane", T::NAME);
}

#[test]
fn narrow_sort_network_total_order_holds_on_every_narrow_dtype() {
    check_sort_network::<f16>();
    check_sort_network::<bf16>();
    check_sort_network::<E4m3>();
    check_sort_network::<E5m2>();
}

// ---- softmax (§6.13 normalization) -------------------------------------------

fn check_softmax_uniform<T: Narrow>(each: f32, row_sum: f32) {
    // Uniform row: m = 0, shifted = 0, e = exp(0) = 1, s = 1+1+1 = 3 (exact in
    // every lane), out = div(1, 3) rounded ONCE into the lane.
    let x: Tensor<T> = tn(&[0.0, 0.0, 0.0], &[3]);
    let s = softmax(&x.view(), 0).unwrap_or_else(|e| panic!("[{}] softmax: {e:?}", T::NAME));
    assert_bits(&s, &[each, each, each], &[3]);
    let got: f32 = s.as_slice().iter().map(|&v| v.f32()).sum();
    assert_eq!(
        got,
        row_sum,
        "[{}] softmax row sums to {got}, not 1.0",
        T::NAME
    );
}

#[test]
fn narrow_softmax_rows_do_not_sum_to_one() {
    // 1/3 is not representable in ANY binary float, so a narrow softmax row
    // cannot sum to 1. The deviation grows as the mantissa shrinks — worth
    // stating loudly because a consumer's "softmax rows sum to 1" invariant is
    // FALSE on the narrow lanes and there is no §6.13 clause promising it.
    check_softmax_uniform::<f16>(0.333251953125, 0.999755859375); // −2.4e−4
    check_softmax_uniform::<bf16>(0.333984375, 1.001953125); // +2.0e−3
    check_softmax_uniform::<E4m3>(0.34375, 1.03125); // +3.1e−2 (!)
    check_softmax_uniform::<E5m2>(0.3125, 0.9375); // −6.3e−2 (!)
}

#[test]
fn narrow_softmax_collapses_to_one_hot_on_e4m3() {
    // A row spanning 16 in log space. After the §6.13 max-shift the exponent
    // arguments are −16, −8, 0; exp(−16) ≈ 1.1e−7 and exp(−8) ≈ 3.4e−4 both fall
    // below half of e4m3's smallest subnormal (2^−9 ≈ 1.95e−3), so they flush to
    // +0 and softmax degenerates to a ONE-HOT. Pinned as real behavior: an FP8
    // attention kernel evaluated this way loses every non-maximal logit.
    let x: Tensor<E4m3> = tn(&[0.0, 8.0, 16.0], &[3]);
    let s = softmax(&x.view(), 0).expect("softmax e4m3");
    assert_bits(&s, &[0.0, 0.0, 1.0], &[3]);
    // f16 keeps both tails alive (min subnormal 2^−24), so the collapse is a
    // property of the FP8 exponent range, not of the decomposition.
    let xf: Tensor<f16> = tn(&[0.0, 8.0, 16.0], &[3]);
    let sf = softmax(&xf.view(), 0).expect("softmax f16");
    assert!(
        sf.as_slice()[0].f32() > 0.0 && sf.as_slice()[1].f32() > 0.0,
        "f16 softmax tails must stay non-zero, got {:?}",
        wide(&sf)
    );
}

// ---- eval_recipe end-to-end --------------------------------------------------

fn check_recipe_dot<T: Narrow>(want: f32) {
    // node 0: Bind(0)              node 1: Bind(1)
    // node 2: Apply{mul, [0,1]}    node 3: Reduce{sum, axes=[0], nokd, child=2}
    let dag = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Bind(1),
            Node::Apply { op: Op::Mul, children: vec![0, 1] },
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![0],
                keepdim: false,
                child: 2,
            },
        ],
        vec![3],
    );
    let a: Tensor<T> = tn(&[1.0, 2.0, 4.0], &[3]);
    let b: Tensor<T> = tn(&[2.0, 2.0, 2.0], &[3]);
    let ev = eval_recipe(&dag, &[a, b], &[], &[])
        .unwrap_or_else(|e| panic!("[{}] eval_recipe dot: {e:?}", T::NAME));
    // products [2,4,8]; ascending fold 0→2→6→14. 6 = 1.10b·2^2 and 14 = 1.11b·2^3
    // both need only 2 mantissa bits, so the dot product is EXACT on all four.
    assert_bits(&ev.outputs[0], &[want], &[]);
    // §6.0-0004: the float sum node is order-invariant/nondeterministic; the
    // elementwise mul above it is exact-byte.
    assert_eq!(ev.dets[2], DetClass::ExactByte, "[{}] mul det", T::NAME);
    assert_eq!(
        ev.dets[3],
        DetClass::OrderInvariantNondeterministic,
        "[{}] sum-reduce det",
        T::NAME
    );
}

#[test]
fn narrow_eval_recipe_dot_product_runs_on_every_narrow_dtype() {
    check_recipe_dot::<f16>(14.0);
    check_recipe_dot::<bf16>(14.0);
    check_recipe_dot::<E4m3>(14.0);
    check_recipe_dot::<E5m2>(14.0);
}

fn check_recipe_graph<T: Narrow>(scan: &[f32]) {
    // A graph touching every non-arithmetic node kind on the narrow lane:
    //   0 Bind(0)                      x = [4,1,3,2]
    //   1 SortNetwork{keys:0, asc}      values [1,2,3,4]; index lane [1,3,2,0]
    //   2 Gather{data:0, index:Node(1)} x permuted by the sort's index lane
    //   3 Const(1)
    //   4 Apply{add, [2,3]}
    //   5 PrefixScan{sum, inclusive}
    //   6 Iota{like:0, axis:0}          [0,1,2,3]
    //   7 Flip{child:6, axis:0}         [3,2,1,0]
    let dag = FlatDag {
        nodes: vec![
            Node::Bind(0),
            Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc },
            Node::Gather {
                data: 0,
                index: IndexRef::Node(1),
                axis: 0,
                oob: OobPolicy::Clamp,
                base: None,
            },
            Node::Const(1.0),
            Node::Apply { op: Op::Add, children: vec![2, 3] },
            Node::PrefixScan { monoid: Monoid::Sum, axis: 0, exclusive: false, child: 4 },
            Node::Iota { like: 0, axis: 0 },
            Node::Flip { child: 6, axis: 0 },
        ],
        outputs: vec![1, 2, 5, 7],
        index_outputs: vec![1],
    };
    let x: Tensor<T> = tn(&[4.0, 1.0, 3.0, 2.0], &[4]);
    let ev = eval_recipe(&dag, &[x], &[], &[])
        .unwrap_or_else(|e| panic!("[{}] eval_recipe graph: {e:?}", T::NAME));
    assert_bits(&ev.outputs[0], &[1.0, 2.0, 3.0, 4.0], &[4]); // sorted values
    assert_bits(&ev.outputs[1], &[1.0, 2.0, 3.0, 4.0], &[4]); // gather by the index lane
    assert_bits(&ev.outputs[2], scan, &[4]); // running sum of [2,3,4,5]
    assert_bits(&ev.outputs[3], &[3.0, 2.0, 1.0, 0.0], &[4]); // flip(iota)
    assert_eq!(ev.index_outputs[0].as_slice(), &[1, 3, 2, 0], "[{}] index lane", T::NAME);
    // sort/gather/flip are raw-bit moves → exact-byte; the sum scan is not.
    assert_eq!(ev.dets[1], DetClass::ExactByte, "[{}] sort det", T::NAME);
    assert_eq!(ev.dets[2], DetClass::ExactByte, "[{}] gather det", T::NAME);
    assert_eq!(
        ev.dets[5],
        DetClass::OrderInvariantNondeterministic,
        "[{}] scan det",
        T::NAME
    );
    assert_eq!(ev.dets[7], DetClass::ExactByte, "[{}] flip det", T::NAME);
}

#[test]
fn narrow_eval_recipe_full_graph_runs_on_every_narrow_dtype() {
    // Inclusive scan of [2,3,4,5]: partials 2, 5, 9, 14.
    check_recipe_graph::<f16>(&[2.0, 5.0, 9.0, 14.0]);
    check_recipe_graph::<bf16>(&[2.0, 5.0, 9.0, 14.0]);
    check_recipe_graph::<E4m3>(&[2.0, 5.0, 9.0, 14.0]); // 9 = 1.001b·2^3, 3 bits
    // e5m2 (2 mantissa bits) cannot hold 9: it is EXACTLY midway between 8
    // (mantissa 00, even) and 10 (mantissa 01, odd) → RNE gives 8; the fold then
    // continues from 8, and 8+5 = 13 is midway between 12 (mantissa 10, even) and
    // 14 (mantissa 11, odd) → 12. The scan drifts to [2,5,8,12].
    check_recipe_graph::<E5m2>(&[2.0, 5.0, 8.0, 12.0]);
}

// ---- declines + ledger cross-check -------------------------------------------

#[test]
fn narrow_nextafter_declines_on_fp8_as_well_as_f16_bf16() {
    // §6.9-0003: nextafter is undefined on every narrow lane — stepping in a
    // promoted f32 yields the wrong neighbor in the narrow lattice. The FP8 lanes
    // inherit the decline through `ScalarFloat::NARROW_FLOAT`.
    assert_eq!(
        eval_op(Op::Nextafter, &[E4m3::of(1.0), E4m3::of(2.0)]),
        Err(Error::Unsupported(Op::Nextafter))
    );
    assert_eq!(
        eval_op(Op::Nextafter, &[E5m2::of(1.0), E5m2::of(2.0)]),
        Err(Error::Unsupported(Op::Nextafter))
    );
    assert_eq!(support(Op::Nextafter, Dtype::E4m3), Support::NotApplicable);
    assert_eq!(support(Op::Nextafter, Dtype::E5m2), Support::NotApplicable);
}

/// The ops this file EXECUTES on the narrow lanes — the honest backing for the
/// ledger cells `implemented_on()` grants them. Keep in sync with the tests above.
const EXECUTED_HERE: &[Op] = &[
    Op::ElementMap,
    Op::Reduce,
    Op::PrefixScan,
    Op::Gather,
    Op::Scatter,
    Op::SortNetwork,
    Op::Matmul,
    Op::Softmax,
    Op::IndexSelect,
    Op::ScatterAdd,
    Op::Cumsum,
    Op::Cummax,
];

#[test]
fn narrow_executed_cells_are_the_cells_the_ledger_calls_done() {
    // Each op above was actually run on each of the four narrow dtypes in this
    // file, so its `Done` in the coverage ledger is backed by execution rather
    // than by genericity alone.
    for &op in EXECUTED_HERE {
        for &d in &[Dtype::F16, Dtype::Bf16, Dtype::E4m3, Dtype::E5m2] {
            assert_eq!(support(op, d), Support::Done, "{op:?}/{d:?}");
        }
    }
}
