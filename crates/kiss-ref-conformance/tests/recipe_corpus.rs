// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recipe conformance corpus — golden flat-DAG recipes with hand-derived
//! outputs and per-node [`DetClass`] vectors.
//!
//! This corpus is the recipe-level differential conformance artifact — kiss-ref
//! is the differential target/proxy, NOT the oracle (conformance role, ratified
//! 2026-07-21). Golden values are hand-derived from KISS-Ops first principles
//! (KISS-OPS-6.11-0002..0008, KISS-OPS-6.12-0001, §6.13); the evaluator is CHECKED against
//! them, never the source of them. Comparison mode is chosen by the ROOT node's
//! [`DetClass`] per KISS-OPS-6.0-0001..0005: `ExactByte` → `to_bits` equality; `Ulp(k)`
//! → within `k` by the sign-magnitude metric ([`ulp_distance_f64`], the
//! ops_conformance idiom); `OrderInvariantNondeterministic` → tolerance, NEVER
//! byte-exact. Recipes are the KISS-CONTRACT-6.4-0009/§6.19 logical flat-DAG form; ULP
//! ceilings are §6.8 (via the vocab); shapes KISS-OPS-6.20-0007/0008; stable sort ties
//! KISS-OPS-6.11-0007. Test naming mirrors KISS-Conform (`test_recipe_*`).

use half::{bf16, f16};
use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    eval_recipe, ulp_distance_f64, Combine, DetClass, Direction, E4m3, Error, FlatDag, IndexRef,
    IndexTensor, Monoid, Node, OobPolicy, ScalarFloat, Tensor,
};

// ---- helpers ----------------------------------------------------------------

fn tf<T: ScalarFloat>(data: &[f64], shape: &[usize]) -> Tensor<T> {
    Tensor::from_vec(data.iter().map(|&v| T::from_f64(v)).collect(), shape)
        .unwrap_or_else(|e| panic!("tensor fixture {shape:?} failed: {e:?}"))
}

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    tf::<f64>(data, shape)
}

fn ix(data: &[i64], shape: &[usize]) -> IndexTensor {
    IndexTensor::new(data.to_vec(), shape, Dtype::I64)
        .unwrap_or_else(|e| panic!("index fixture {shape:?} failed: {e:?}"))
}

/// Bit-exact compare in `T`'s own width (every corpus literal is dyadic with a
/// ≤24-bit mantissa, so `T::from_f64` is exact at f32 too).
trait Bits: ScalarFloat {
    fn bits(self) -> u64;
}
impl Bits for f64 {
    fn bits(self) -> u64 {
        self.to_bits()
    }
}
impl Bits for f32 {
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

/// Root class `ExactByte` → per-element raw-bit equality (KISS-OPS-6.0-0002).
fn assert_bits<T: Bits>(got: &Tensor<T>, want: &[f64], shape: &[usize]) {
    assert_eq!(got.shape(), shape, "output shape");
    assert_eq!(got.as_slice().len(), want.len(), "element count");
    for (i, (&g, &w)) in got.as_slice().iter().zip(want).enumerate() {
        let wt = T::from_f64(w);
        assert_eq!(
            g.bits(),
            wt.bits(),
            "elem {i}: got {} want {w} (raw-bit compare)",
            g.to_f64()
        );
    }
}

/// Root class `OrderInvariantNondeterministic` → tolerance compare, never
/// byte-exact (KISS-OPS-6.0-0004). NaN is never an expected corpus value.
fn assert_close<T: ScalarFloat>(got: &Tensor<T>, want: &[f64], shape: &[usize], tol: f64) {
    assert_eq!(got.shape(), shape, "output shape");
    assert_eq!(got.as_slice().len(), want.len(), "element count");
    for (i, (&g, &w)) in got.as_slice().iter().zip(want).enumerate() {
        let gv = g.to_f64();
        assert!(
            (gv - w).abs() <= tol,
            "elem {i}: got {gv} want {w} (abs tol {tol})"
        );
    }
}

/// Root class `Ulp(k)` → within `k` ULP by the sign-magnitude metric (KISS-OPS-6.0-0003).
fn assert_ulp(got: &Tensor<f64>, want: &[f64], shape: &[usize], ulps: u64) {
    assert_eq!(got.shape(), shape, "output shape");
    assert_eq!(got.as_slice().len(), want.len(), "element count");
    for (i, (&g, &w)) in got.as_slice().iter().zip(want).enumerate() {
        let d = ulp_distance_f64(g, w);
        assert!(d <= ulps, "elem {i}: got {g} want {w} ({d} ULP > {ulps})");
    }
}

const EB: DetClass = DetClass::ExactByte;
const OIN: DetClass = DetClass::OrderInvariantNondeterministic;
const U4: DetClass = DetClass::Ulp(4.0);

// ---- builders (reused by the goldens AND the fuzz mutators) -----------------

/// R1: `relu(matmul(A, B) + bias)` — nondet root through an elementwise epilogue.
fn r1_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 A
            Node::Bind(1),                   // 1 B
            Node::Matmul { lhs: 0, rhs: 1 }, // 2
            Node::Bind(2),                   // 3 bias
            Node::Apply {
                op: Op::Add,
                children: vec![2, 3],
            }, // 4
            Node::Apply {
                op: Op::Relu,
                children: vec![4],
            }, // 5
        ],
        vec![5],
    )
}
fn r1_inputs<T: ScalarFloat>() -> Vec<Tensor<T>> {
    vec![
        tf::<T>(&[1.0, 2.0, 0.5, -1.0], &[2, 2]),
        tf::<T>(&[2.0, 0.5, 1.0, -0.5], &[2, 2]),
        tf::<T>(&[-2.0, 0.25], &[2]),
    ]
}

/// R2: row softmax over `[2,2]` — Max-reduce EB, exp Ulp(4), Sum-reduce OIN.
fn r2_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // 0
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![1],
                keepdim: true,
                child: 0,
            }, // 1
            Node::Apply {
                op: Op::Sub,
                children: vec![0, 1],
            }, // 2
            Node::Apply {
                op: Op::Exp,
                children: vec![2],
            }, // 3
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 3,
            }, // 4
            Node::Apply {
                op: Op::Div,
                children: vec![3, 4],
            }, // 5
        ],
        vec![5],
    )
}
fn r2_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[0.0, 1.0, 2.0, 2.0], &[2, 2])]
}

/// R3: layernorm-shaped row reduce, `y = (x - mean)/sqrt(var + eps)`.
/// NOTE: `ReducedCount` resolves against `inputs[0]` by definition, so `x`
/// MUST stay inputs[0].
fn r3_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // 0 x
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 0,
            }, // 1
            Node::ReducedCount(vec![1]), // 2 = 4.0
            Node::Apply {
                op: Op::Div,
                children: vec![1, 2],
            }, // 3 mean
            Node::Apply {
                op: Op::Sub,
                children: vec![0, 3],
            }, // 4 centered
            Node::Apply {
                op: Op::Mul,
                children: vec![4, 4],
            }, // 5
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 5,
            }, // 6
            Node::Apply {
                op: Op::Div,
                children: vec![6, 2],
            }, // 7 var
            Node::RuntimeScalar(0), // 8 eps
            Node::Apply {
                op: Op::Add,
                children: vec![7, 8],
            }, // 9
            Node::Apply {
                op: Op::Sqrt,
                children: vec![9],
            }, // 10
            Node::Apply {
                op: Op::Div,
                children: vec![4, 10],
            }, // 11
        ],
        vec![11],
    )
}
fn r3_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[2.0, 3.0, 5.0, 6.0, 1.0, 2.0, -2.0, -1.0], &[2, 4])]
}

/// R4: argsort → gather reorder via `IndexRef::Node` — ExactByte end-to-end.
fn r4_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // keys
            Node::Bind(1), // data
            Node::SortNetwork {
                keys: 0,
                axis: 0,
                dir: Direction::Asc,
            }, // 2
            Node::Gather {
                data: 1,
                index: IndexRef::Node(2),
                axis: 0,
                oob: OobPolicy::ZeroFill,
                base: None,
            }, // 3
        ],
        vec![3],
    )
}
fn r4_inputs<T: ScalarFloat>() -> Vec<Tensor<T>> {
    vec![
        tf::<T>(&[4.0, 1.0, 3.0, 2.0], &[4]),
        tf::<T>(&[40.5, 10.25, 30.75, 20.5], &[4]),
    ]
}

/// R5: sort with BOTH lanes exported — stable ties + raw-bit ±0 permutation.
fn r5_dag() -> FlatDag {
    FlatDag {
        nodes: vec![
            Node::Bind(0),
            Node::SortNetwork {
                keys: 0,
                axis: 0,
                dir: Direction::Asc,
            },
        ],
        outputs: vec![1],
        index_outputs: vec![1],
    }
}
fn r5_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[2.0, 0.0, 2.0, -0.0, 3.0], &[5])]
}

/// R6: scatter_add histogram with one OOB write (skipped, KISS-OPS-6.11-0005).
fn r6_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // dest
            Node::Bind(1), // updates
            Node::Scatter {
                dest: 0,
                index: IndexRef::Slot(0),
                updates: 1,
                axis: 0,
                combine: Combine::AtomicAdd,
            }, // 2
        ],
        vec![2],
    )
}
fn r6_inputs() -> Vec<Tensor<f64>> {
    vec![
        t64(&[0.0, 0.0, 0.0, 0.0], &[4]),
        t64(&[1.0, 2.0, 0.5, 0.25, 4.0, 1.5], &[6]),
    ]
}
fn r6_indices() -> Vec<IndexTensor> {
    vec![ix(&[0, 2, 0, 3, 2, 7], &[6])]
}

/// R7: gather-or-default — `Skip` + rank-0 `Const` base; high AND negative OOB
/// (KISS-OPS-6.11-0004: negative is OOB, no from-end wrap).
fn r7_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),     // 0 data
            Node::Const(-0.5), // 1 base
            Node::Gather {
                data: 0,
                index: IndexRef::Slot(0),
                axis: 0,
                oob: OobPolicy::Skip,
                base: Some(1),
            }, // 2
        ],
        vec![2],
    )
}
fn r7_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[8.0, 16.0, 32.0], &[3])]
}
fn r7_indices() -> Vec<IndexTensor> {
    vec![ix(&[1, 5, 0, -1], &[4])]
}

/// R8: iota (KISS-OPS-6.12-0001 coord leaf) × data — pins signed zero through the
/// recipe path.
fn r8_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 x
            Node::Iota { like: 0, axis: 1 }, // 1
            Node::Apply {
                op: Op::Mul,
                children: vec![1, 0],
            }, // 2
        ],
        vec![2],
    )
}
fn r8_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[0.5, 1.5, 2.5, -1.0, 2.0, -4.0], &[2, 3])]
}

/// R9: exp → sort → `IndexRef::Node` → gather — the escalation pin.
fn r9_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // x
            Node::Bind(1), // data
            Node::Apply {
                op: Op::Exp,
                children: vec![0],
            }, // 2
            Node::SortNetwork {
                keys: 2,
                axis: 0,
                dir: Direction::Asc,
            }, // 3
            Node::Gather {
                data: 1,
                index: IndexRef::Node(3),
                axis: 0,
                oob: OobPolicy::ZeroFill,
                base: None,
            }, // 4
        ],
        vec![4],
    )
}
fn r9_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[1.0, 0.0, 2.0], &[3]), t64(&[10.0, 20.0, 30.0], &[3])]
}

/// The f64 fuzz/self-determinism view of every golden builder:
/// `(dag, inputs, params, indices)`. Ids 0..=8 are R1..R9.
type Case = (FlatDag, Vec<Tensor<f64>>, Vec<f64>, Vec<IndexTensor>);
fn base_case(id: usize) -> Case {
    match id {
        0 => (r1_dag(), r1_inputs::<f64>(), vec![], vec![]),
        1 => (r2_dag(), r2_inputs(), vec![], vec![]),
        2 => (r3_dag(), r3_inputs(), vec![1.5], vec![]),
        3 => (r4_dag(), r4_inputs::<f64>(), vec![], vec![]),
        4 => (r5_dag(), r5_inputs(), vec![], vec![]),
        5 => (r6_dag(), r6_inputs(), vec![], r6_indices()),
        6 => (r7_dag(), r7_inputs(), vec![], r7_indices()),
        7 => (r8_dag(), r8_inputs(), vec![], vec![]),
        8 => (r9_dag(), r9_inputs(), vec![], vec![]),
        _ => panic!("unknown base recipe id {id}"),
    }
}

// ---- the golden tests -------------------------------------------------------

/// R1 body, dtype-parametric. Hand arithmetic (every product/sum dyadic, exact
/// in f32 and f64 under ANY contraction order):
///   C = A·B = [[1·2+2·1, 1·0.5+2·(−0.5)], [0.5·2+(−1)·1, 0.5·0.5+(−1)·(−0.5)]]
///     = [[4, −0.5], [0, 0.75]]
///   plus bias[−2, 0.25] (broadcast over columns) = [[2, −0.25], [−2, 1]]
///   relu → [[2, 0], [0, 1]].
/// Root is OrderInvariantNondeterministic (matmul) → tolerance compare.
fn r1_case<T: ScalarFloat>(tol: f64) {
    let r = eval_recipe(&r1_dag(), &r1_inputs::<T>(), &[], &[]).expect("R1 must evaluate");
    assert_close(&r.outputs[0], &[2.0, 0.0, 0.0, 1.0], &[2, 2], tol);
    assert!(r.index_outputs.is_empty());
    assert_eq!(r.dets, vec![EB, EB, OIN, EB, OIN, OIN]);
}

#[test]
fn test_recipe_matmul_bias_relu() {
    // KISS-CONTRACT-6.4-0009 flat-DAG matmul+epilogue; KISS-OPS-6.20-0007 bias broadcast; KISS-OPS-6.0-0004
    // nondet root propagation. Identical logical results + dets at both dtypes.
    r1_case::<f64>(1e-12);
    r1_case::<f32>(1e-6);
}

#[test]
fn test_recipe_softmax_rows() {
    // §6.13 softmax shape; §6.8 exp ceiling pinned exactly (Ulp(4.0)); Max-
    // reduce ExactByte vs Sum-reduce OIN (KISS-OPS-6.0-0004). Hand arithmetic:
    //   row0: max=1, shifted=[−1,0], exp=[e⁻¹,1]=[0.36787944117144233, 1],
    //         sum=1.3678794411714423, softmax=[0.2689414213699951,
    //         0.7310585786300049] (= sigmoid(∓1));
    //   row1: max=2, shifted=[0,0], exp=[1,1], sum=2, softmax=[0.5,0.5] exact.
    let r = eval_recipe(&r2_dag(), &r2_inputs(), &[], &[]).expect("R2 must evaluate");
    assert_close(
        &r.outputs[0],
        &[0.2689414213699951, 0.7310585786300049, 0.5, 0.5],
        &[2, 2],
        1e-12,
    );
    let s = r.outputs[0].as_slice();
    assert!((s[0] + s[1] - 1.0).abs() <= 1e-12, "row0 must sum to 1");
    assert!((s[2] + s[3] - 1.0).abs() <= 1e-12, "row1 must sum to 1");
    // Pin the §6.8 exp ceiling literal through the recipe path.
    assert_eq!(r.dets[3], DetClass::Ulp(4.0));
    assert_eq!(r.dets, vec![EB, EB, EB, U4, OIN, OIN]);
    assert!(r.index_outputs.is_empty());
}

#[test]
fn test_recipe_layernorm_rowreduce() {
    // §6.13 fold-epilogue shape: mean/var via reduce(sum)+reduced_count, eps via
    // runtime_scalar, sqrt epilogue. Hand arithmetic (every step dyadic-exact
    // for any fold order; sqrt(4.0)=2.0 exactly correctly-rounded):
    //   row0: sum=16, mean=4, centered=[−2,−1,1,2], sumsq=10, var=2.5,
    //         var+eps=4, sqrt=2 → y=[−1,−0.5,0.5,1];
    //   row1: sum=0, mean=0, centered=[1,2,−2,−1], sumsq=10, var=2.5,
    //         var+eps=4, sqrt=2 → y=[0.5,1,−1,−0.5].
    let r = eval_recipe(&r3_dag(), &r3_inputs(), &[1.5], &[]).expect("R3 must evaluate");
    assert_close(
        &r.outputs[0],
        &[-1.0, -0.5, 0.5, 1.0, 0.5, 1.0, -1.0, -0.5],
        &[2, 4],
        1e-12,
    );
    // Sqrt's own Ulp(2.0) is absorbed by the joined OIN.
    assert_eq!(
        r.dets,
        vec![EB, OIN, EB, OIN, OIN, OIN, OIN, OIN, EB, OIN, OIN, OIN]
    );
    assert!(r.index_outputs.is_empty());
}

/// R4 body, dtype-parametric, BIT-EXACT. keys=[4,1,3,2] sort asc → perm
/// [1,3,2,0]; gather data=[40.5,10.25,30.75,20.5] by the perm →
/// [10.25, 20.5, 30.75, 40.5]. Root ExactByte → raw-bit compare in T's width.
fn r4_case<T: Bits>() {
    let r = eval_recipe(&r4_dag(), &r4_inputs::<T>(), &[], &[]).expect("R4 must evaluate");
    assert_bits(&r.outputs[0], &[10.25, 20.5, 30.75, 40.5], &[4]);
    assert!(r.index_outputs.is_empty());
    assert_eq!(r.dets, vec![EB, EB, EB, EB]);
}

#[test]
fn test_recipe_argsort_gather_reorder() {
    // KISS-OPS-6.11-0007 sort + KISS-OPS-6.11-0004 gather via IndexRef::Node — a real
    // scheduling edge, ExactByte end-to-end, identical at both dtypes.
    r4_case::<f64>();
    r4_case::<f32>();
}

#[test]
fn test_recipe_sort_dual_output_stable_ties() {
    // KISS-OPS-6.11-0007: total order has −0.0 == +0.0 (tie) and duplicate keys tie;
    // ties → lower original index, values are a RAW-BIT permutation. keys
    // [2.0, 0.0, 2.0, −0.0, 3.0] asc → (orig1,+0.0),(orig3,−0.0),(orig0,2.0),
    // (orig2,2.0),(orig4,3.0).
    let r = eval_recipe(&r5_dag(), &r5_inputs(), &[], &[]).expect("R5 must evaluate");
    assert_bits(&r.outputs[0], &[0.0, -0.0, 2.0, 2.0, 3.0], &[5]);
    // Pin the raw-bit ±0 permutation explicitly.
    assert_eq!(r.outputs[0].as_slice()[0].to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(r.outputs[0].as_slice()[1].to_bits(), 0x8000_0000_0000_0000);
    // Index lane: the KISS-OPS-6.11-0007 lower-original-index tie order.
    assert_eq!(r.index_outputs.len(), 1);
    assert_eq!(r.index_outputs[0].as_slice(), &[1, 3, 0, 2, 4]);
    // ONE class covers both lanes.
    assert_eq!(r.dets, vec![EB, EB]);
}

#[test]
fn test_recipe_scatter_add_histogram() {
    // KISS-OPS-6.11-0005/-0006: OOB write skipped, atomic_add folds colliding sources
    // in row-major source order, untouched dest cells keep dest values.
    //   bin0 ← src0(1.0), src2(0.5) → 1.5; bin2 ← src1(2.0), src4(4.0) → 6;
    //   bin3 ← src3(0.25); src5 (idx 7 ≥ 4) skipped; bin1 keeps dest 0.0.
    // Root is OIN → tolerance compare BY CLASS DISCIPLINE: these dyadic values
    // are exact under ANY fold order, but the class describes the candidate's
    // reassociation freedom, not this input's luck — OIN must NEVER be
    // compared byte-exact. Any wrong element differs by ≥ 0.25, so tol 1e-12
    // loses no power.
    let r = eval_recipe(&r6_dag(), &r6_inputs(), &[], &r6_indices()).expect("R6 must evaluate");
    assert_close(&r.outputs[0], &[1.5, 0.0, 6.0, 0.25], &[4], 1e-12);
    assert_eq!(r.dets, vec![EB, EB, OIN]);
    assert!(r.index_outputs.is_empty());
}

#[test]
fn test_recipe_gather_or_default_skip_base() {
    // Main: KISS-OPS-6.11-0004 (negative index is OOB, NO from-end wrap) + the cosigned
    // §6.11 gather-skip RFC base operand. idx [1,5,0,−1] over data [8,16,32]:
    // 1 in → 16; 5 OOB → base −0.5; 0 in → 8; −1 OOB → base −0.5.
    let r = eval_recipe(&r7_dag(), &r7_inputs(), &[], &r7_indices()).expect("R7 must evaluate");
    assert_bits(&r.outputs[0], &[16.0, -0.5, 8.0, -0.5], &[4]);
    assert_eq!(r.dets, vec![EB, EB, EB]);
    assert!(r.index_outputs.is_empty());

    // Det-pair variant: base = exp(const 0.0) carries Ulp(4.0). The base's
    // class joins ONLY under Skip — a static, policy-conditioned join, never
    // index-conditioned (KISS-OPS-6.0-0005 precision the comparator relies on).
    let mk = |oob| {
        FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Const(0.0),
                Node::Apply {
                    op: Op::Exp,
                    children: vec![1],
                },
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(0),
                    axis: 0,
                    oob,
                    base: Some(2),
                },
            ],
            vec![3],
        )
    };
    let skip = eval_recipe(&mk(OobPolicy::Skip), &r7_inputs(), &[], &r7_indices())
        .expect("R7 skip arm must evaluate");
    assert_eq!(skip.dets, vec![EB, EB, U4, U4]);
    // Root class Ulp(4.0) → within 4 ULP: [16, exp(0), 8, exp(0)] = [16,1,8,1].
    assert_ulp(&skip.outputs[0], &[16.0, 1.0, 8.0, 1.0], &[4], 4);

    let zf = eval_recipe(&mk(OobPolicy::ZeroFill), &r7_inputs(), &[], &r7_indices())
        .expect("R7 zerofill arm must evaluate");
    // Under ZeroFill the base's bits can never reach the output: no join.
    assert_eq!(zf.dets, vec![EB, EB, U4, EB]);
    assert_bits(&zf.outputs[0], &[16.0, 0.0, 8.0, 0.0], &[4]);

    // Discriminating arm (adversarial-review find): with ALL-IN-RANGE indices
    // the base is never read at runtime, yet under Skip its class MUST STILL
    // join — the KISS-OPS-6.0-0005 join is policy-conditioned (static), never
    // index-conditioned. A lazy join-only-if-a-Skip-fired evaluator passes the
    // two arms above (r7_indices has live OOB cells) but fails here.
    let in_range = ix(&[0, 1, 2, 2], &[4]);
    let static_join = eval_recipe(&mk(OobPolicy::Skip), &r7_inputs(), &[], &[in_range])
        .expect("R7 in-range skip arm must evaluate");
    assert_eq!(static_join.dets, vec![EB, EB, U4, U4]);
    assert_ulp(&static_join.outputs[0], &[8.0, 16.0, 32.0, 32.0], &[4], 4);
}

#[test]
fn test_recipe_iota_positional() {
    // KISS-OPS-6.12-0001 coord leaf: iota axis 1 over [2,3] = [0,1,2,0,1,2]; product
    // with x — element [1,0] = (+0.0)·(−1.0) = −0.0 pins KISS-OPS-6.2-0004 signed zero
    // through the recipe path (bit compare).
    let r = eval_recipe(&r8_dag(), &r8_inputs(), &[], &[]).expect("R8 must evaluate");
    assert_bits(&r.outputs[0], &[0.0, 1.5, 5.0, -0.0, 2.0, -8.0], &[2, 3]);
    assert_eq!(r.outputs[0].as_slice()[3].to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(r.dets, vec![EB, EB, EB]);
    assert!(r.index_outputs.is_empty());

    // Shape-only-edge det pin: iota over the shape of a matmul result stays
    // ExactByte although `like`'s det is OIN — the like edge is shape-only
    // (its values are never read; recipe.rs pins the non-propagation).
    let vdag = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Bind(1),
            Node::Matmul { lhs: 0, rhs: 1 },
            Node::Iota { like: 2, axis: 0 },
        ],
        vec![3],
    );
    let a = t64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // I2
    let b = t64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let v = eval_recipe(&vdag, &[a, b], &[], &[]).expect("R8 variant must evaluate");
    // iota axis 0 over [2,2] = [0,0,1,1].
    assert_bits(&v.outputs[0], &[0.0, 0.0, 1.0, 1.0], &[2, 2]);
    assert_eq!(v.dets, vec![EB, EB, OIN, EB]);
}

#[test]
fn test_recipe_nonexact_index_escalation() {
    // THE PIN (adversarial-review rule): a non-ExactByte index producer
    // consumed via IndexRef::Node ESCALATES the consumer to
    // OrderInvariantNondeterministic, never forwards Ulp(k) — a ≤k-ULP key
    // perturbation can flip the permutation and relocate whole elements
    // (deviation unbounded in ULP). exp keys exp(1)=2.718…, exp(0)=1,
    // exp(2)=7.389… are separated by factor ≥ e, so any ≤4-ULP-conforming exp
    // preserves the order: sort asc → orig [1,0,2]; gather → [20,10,30].
    let r = eval_recipe(&r9_dag(), &r9_inputs(), &[], &[]).expect("R9 must evaluate");
    assert_close(&r.outputs[0], &[20.0, 10.0, 30.0], &[3], 1e-9);
    // Sort's own value lane stays ULP-sound…
    assert_eq!(r.dets[3], DetClass::Ulp(4.0));
    // …but the index-mediated gather escalates.
    assert_eq!(r.dets[4], DetClass::OrderInvariantNondeterministic);
    assert_eq!(r.dets, vec![EB, EB, U4, U4, OIN]);
    assert!(r.index_outputs.is_empty());
}

#[test]
fn test_recipe_cmp_mask_selection_escalation() {
    // THE PIN (KISS-OPS-6.0-0007, the cmp-MASK analogue of the index escalation
    // above; steward ruling 2026-08-05, KISS main @ 325800a). A COMPARISON output
    // is a SELECTION — it reports WHICH value won, not a value — so over a
    // non-exact producer it escalates to OrderInvariantNondeterministic and NEVER
    // carries the value-lane Ulp(k) join. A mask lives in {0,1}: a ≤k-ULP
    // perturbation of a near-tie operand flips the bit a FULL UNIT, unbounded in
    // ULP — Ulp(k) on a bit is a category error. The class is fixed by op
    // SEMANTICS (Family::Comparison), not by which lane the impl computes it in
    // (the representation-independence 6.0-0008 will anchor). Mirrors the
    // §6.8-0011 test_ops_per_output_determinism_class node-5 witness.

    // Arm 1 — the node-5 mirror AND the two-output crystallization in ONE graph:
    // exp keys carry Ulp(4.0); the SAME exp VALUE output stays Ulp(4.0) while the
    // cmp_lt MASK over it escalates to OIN. Threshold 2.0 sits ≥0.7 from every exp
    // value (exp(0)=1, exp(1)=e, exp(2)=e²) — far beyond 4 ULP — so the mask is
    // order-robust (as R9's keys were).
    let dag = FlatDag::new(
        vec![
            Node::Bind(0), // 0 x
            Node::Apply {
                op: Op::Exp,
                children: vec![0],
            }, // 1 exp — Ulp(4.0)
            Node::Const(2.0), // 2 threshold — EB
            Node::Apply {
                op: Op::CmpLt,
                children: vec![1, 2],
            }, // 3 mask — SELECTION
        ],
        vec![1, 3], // export BOTH the value (exp) and the mask
    );
    let x = t64(&[0.0, 1.0, 2.0], &[3]);
    let r = eval_recipe(&dag, std::slice::from_ref(&x), &[], &[]).expect("cmp-mask must evaluate");
    // The value output stays Ulp — the max-permissive value-lane join…
    assert_eq!(r.dets[1], DetClass::Ulp(4.0));
    // …but the mask output escalates: a SELECTION over a non-exact producer.
    assert_eq!(r.dets[3], DetClass::OrderInvariantNondeterministic);
    assert_eq!(r.dets, vec![EB, U4, EB, OIN]);
    // exp VALUE (loose abs tol: DetClass is the assertion, exp precision is
    // pinned by test_recipe_softmax_rows) and mask (OIN → tolerance, never bits).
    let exp = [(0.0f64).exp(), (1.0f64).exp(), (2.0f64).exp()];
    assert_close(&r.outputs[0], &exp, &[3], 1e-9);
    assert_close(&r.outputs[1], &[1.0, 0.0, 0.0], &[3], 1e-12);

    // Arm 2 — escalation is a NO-OP when both operands are exact: a cmp over two
    // ExactByte producers is ExactByte (a mask is only nondeterministic when the
    // values it compares are). Root ExactByte → raw-bit compare.
    let exact = FlatDag::new(
        vec![
            Node::Bind(0),    // 0 a — EB
            Node::Const(2.0), // 1 threshold — EB
            Node::Apply {
                op: Op::CmpLt,
                children: vec![0, 1],
            }, // 2 mask — EB
        ],
        vec![2],
    );
    let a = t64(&[1.0, 3.0], &[2]);
    let re =
        eval_recipe(&exact, std::slice::from_ref(&a), &[], &[]).expect("exact cmp must evaluate");
    assert_eq!(re.dets, vec![EB, EB, EB]);
    assert_bits(&re.outputs[0], &[1.0, 0.0], &[2]);

    // Arm 3 — the steward's relu crystallization, faithfully: y = relu(a) and
    // mask = (a>0) over the SAME Ulp-classed a = exp(x)−2. relu is an activation
    // (a VALUE) so y stays Ulp(4.0); cmp_gt is a Comparison (a SELECTION) so the
    // a>0 mask escalates to OIN — same op-graph, two outputs, two classes. a =
    // [−1.0, e−2≈0.718], both far from 0, so the mask is robust.
    let relu = FlatDag::new(
        vec![
            Node::Bind(0), // 0 x
            Node::Apply {
                op: Op::Exp,
                children: vec![0],
            }, // 1 exp — U4
            Node::Const(2.0), // 2 — EB
            Node::Apply {
                op: Op::Sub,
                children: vec![1, 2],
            }, // 3 a = exp(x)−2 — U4
            Node::Apply {
                op: Op::Relu,
                children: vec![3],
            }, // 4 max(a,0) — VALUE, U4
            Node::Const(0.0), // 5 — EB
            Node::Apply {
                op: Op::CmpGt,
                children: vec![3, 5],
            }, // 6 a>0 — MASK, OIN
        ],
        vec![4, 6],
    );
    let xr = t64(&[0.0, 1.0], &[2]);
    let rr = eval_recipe(&relu, std::slice::from_ref(&xr), &[], &[])
        .expect("relu crystallization must evaluate");
    assert_eq!(rr.dets, vec![EB, U4, EB, U4, U4, EB, OIN]);
    assert_eq!(rr.dets[4], DetClass::Ulp(4.0)); // value output stays Ulp
    assert_eq!(rr.dets[6], DetClass::OrderInvariantNondeterministic); // mask escalates
    let a_val = [(0.0f64).exp() - 2.0, (1.0f64).exp() - 2.0];
    assert_close(
        &rr.outputs[0],
        &[a_val[0].max(0.0), a_val[1].max(0.0)],
        &[2],
        1e-9,
    );
    assert_close(&rr.outputs[1], &[0.0, 1.0], &[2], 1e-12);

    // Arm 4 — the escalation is PER-NODE, so it propagates through an
    // INTERMEDIATE mask feeding a VALUE combinator: band = (exp>1.5)·(exp<5.0)
    // via Mul (an Arithmetic value op, NOT a comparison). Each cmp node's class
    // is escalated to OIN AT THE NODE, so Mul joins OIN⊔OIN = OIN — it does NOT
    // see the cmps' local exact-byte + exp's Ulp and under-report Ulp (which a
    // per-ROOT model that escalated only at the queried output would). This pins
    // kiss-ref as per-node (a real KISS↔kiss-ref distinction the steward
    // surfaced 2026-08-05; beyond 6.0-0008's comparison-op-OUTPUT scope).
    let band = FlatDag::new(
        vec![
            Node::Bind(0), // 0 x
            Node::Apply {
                op: Op::Exp,
                children: vec![0],
            }, // 1 exp — U4
            Node::Const(1.5), // 2 — EB
            Node::Const(5.0), // 3 — EB
            Node::Apply {
                op: Op::CmpGt,
                children: vec![1, 2],
            }, // 4 exp>1.5 — MASK, OIN
            Node::Apply {
                op: Op::CmpLt,
                children: vec![1, 3],
            }, // 5 exp<5.0 — MASK, OIN
            Node::Apply {
                op: Op::Mul,
                children: vec![4, 5],
            }, // 6 band — VALUE combinator over two OIN masks → OIN
        ],
        vec![6],
    );
    let rb = eval_recipe(&band, std::slice::from_ref(&x), &[], &[])
        .expect("intermediate-mask band must evaluate");
    // exp = [1, e, e²] → (>1.5)=[0,1,1], (<5)=[1,1,0], band=[0,1,0].
    assert_eq!(rb.dets, vec![EB, U4, EB, EB, OIN, OIN, OIN]);
    assert_eq!(rb.dets[6], DetClass::OrderInvariantNondeterministic);
    assert_close(&rb.outputs[0], &[0.0, 1.0, 0.0], &[3], 1e-12);
}

#[test]
fn test_recipe_bincount_scalar_updates() {
    // R10 — bincount (KISS PR #75 companion, RULED general-broadcast
    // 2026-07-23; the cross-implementation witness golden):
    // scatter[0, atomic-add, skip, i32](const(1), in0) — Baracuda's exact 2c
    // emit shape. Rank-0 updates broadcast over the index count (KISS-OPS-6.11-0001).
    // x = [0,2,0,3,2,2] over 4 zeroed bins → hand fold: bin0←{0,0}=2,
    // bin1←{}=0, bin2←{2,2,2}=3, bin3←{3}=1 → [2,0,3,1].
    let dag = FlatDag::new(
        vec![
            Node::Bind(0), // dest: zeros[4]
            Node::Const(1.0),
            Node::Scatter {
                dest: 0,
                index: IndexRef::Slot(0),
                updates: 1,
                axis: 0,
                combine: Combine::AtomicAdd,
            },
        ],
        vec![2],
    );
    let dest = t64(&[0.0, 0.0, 0.0, 0.0], &[4]);
    // i32 index dtype per the emit; values widened per KISS-OPS-6.11-0009.
    let idx = IndexTensor::new(vec![0, 2, 0, 3, 2, 2], &[6], Dtype::I32)
        .unwrap_or_else(|e| panic!("bincount index fixture failed: {e:?}"));
    let r = eval_recipe(&dag, &[dest], &[], &[idx]).expect("R10 must evaluate");
    // Root class OIN (float atomic_add) → tolerance compare per KISS-OPS-6.0-0004,
    // never byte-exact — even though small-int sums are exact under any order.
    assert_close(&r.outputs[0], &[2.0, 0.0, 3.0, 1.0], &[4], 1e-12);
    assert_eq!(r.dets, vec![EB, EB, OIN]);
    assert!(r.index_outputs.is_empty());
}

#[test]
fn test_recipe_flip_reverse() {
    // R11 — flip (reverse along axis; two-consumer add per the KISS #76 flip
    // routing): out[c] = in[c′], c′[axis] = extent−1−c[axis]. A raw-bit move:
    // −0.0 and the NaN payload must cross unchanged → root ExactByte,
    // bit-exact compare. x=[1,−0,NaN / 4,5,6] flip axis1 → [NaN,−0,1 / 6,5,4].
    let dag = FlatDag::new(
        vec![Node::Bind(0), Node::Flip { child: 0, axis: 1 }],
        vec![1],
    );
    let x = t64(&[1.0, -0.0, f64::NAN, 4.0, 5.0, 6.0], &[2, 3]);
    let r = eval_recipe(&dag, std::slice::from_ref(&x), &[], &[]).expect("R11 must evaluate");
    assert_bits(
        &r.outputs[0],
        &[f64::NAN, -0.0, 1.0, 6.0, 5.0, 4.0],
        &[2, 3],
    );
    assert_eq!(r.dets, vec![EB, EB]);
    // flip∘flip = identity, bit-for-bit (raw-bit move both ways).
    let dag2 = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Flip { child: 0, axis: 1 },
            Node::Flip { child: 1, axis: 1 },
        ],
        vec![2],
    );
    let r2 = eval_recipe(&dag2, std::slice::from_ref(&x), &[], &[])
        .expect("R11 involution must evaluate");
    for (a, b) in r2.outputs[0].as_slice().iter().zip(x.as_slice()) {
        assert_eq!(a.to_bits(), b.to_bits(), "flip∘flip must be bit-identity");
    }
    // axis OOR is the typed decline.
    let bad = FlatDag::new(
        vec![Node::Bind(0), Node::Flip { child: 0, axis: 2 }],
        vec![1],
    );
    assert!(matches!(
        eval_recipe(&bad, &[x], &[], &[]),
        Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
    ));
}

// ---- invariant D: self-determinism ------------------------------------------

#[test]
fn test_recipe_corpus_self_determinism() {
    // The reference must be bit-stable with itself — outputs bitwise-identical
    // and dets equal across two evaluations — for ALL recipes, including
    // OIN-rooted ones (the class permits CANDIDATES to differ, never the
    // reference from itself).
    for id in 0..9 {
        let (dag, inputs, params, indices) = base_case(id);
        let a = eval_recipe(&dag, &inputs, &params, &indices)
            .unwrap_or_else(|e| panic!("R{} first eval failed: {e:?}", id + 1));
        let b = eval_recipe(&dag, &inputs, &params, &indices)
            .unwrap_or_else(|e| panic!("R{} second eval failed: {e:?}", id + 1));
        assert_eq!(a.outputs.len(), b.outputs.len(), "R{} output arity", id + 1);
        for (o, (x, y)) in a.outputs.iter().zip(&b.outputs).enumerate() {
            assert_eq!(x.shape(), y.shape(), "R{} output {o} shape", id + 1);
            for (i, (u, v)) in x.as_slice().iter().zip(y.as_slice()).enumerate() {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "R{} output {o} elem {i} not bit-identical",
                    id + 1
                );
            }
        }
        assert_eq!(
            a.index_outputs.len(),
            b.index_outputs.len(),
            "R{} index-output arity",
            id + 1
        );
        for (o, (x, y)) in a.index_outputs.iter().zip(&b.index_outputs).enumerate() {
            assert_eq!(x.shape(), y.shape(), "R{} index output {o} shape", id + 1);
            assert_eq!(x.as_slice(), y.as_slice(), "R{} index output {o}", id + 1);
        }
        assert_eq!(a.dets, b.dets, "R{} dets", id + 1);
    }
}

// ---- fuzz: mutation taxonomy, typed declines --------------------------------

/// xorshift64 — deterministic, in-file, never wall-clock.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }
    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// "KISSREF0".
const MASTER_SEED: u64 = 0x4B49_5353_5245_4630;
const ITERATIONS: usize = 128;

/// The settable value-lane child-edge slots of `node`.
fn edge_count(node: &Node) -> usize {
    match node {
        Node::Bind(_)
        | Node::Const(_)
        | Node::ConstBits(_)
        | Node::RuntimeScalar(_)
        | Node::ReducedCount(_) => 0,
        Node::Apply { children, .. } => children.len(),
        Node::Reduce { .. }
        | Node::PrefixScan { .. }
        | Node::SortNetwork { .. }
        | Node::Iota { .. }
        | Node::Flip { .. } => 1,
        Node::Matmul { .. } | Node::Scatter { .. } => 2,
        Node::Gather { base, .. } => 1 + usize::from(base.is_some()),
    }
}

fn set_edge(node: &mut Node, slot: usize, target: usize) {
    match node {
        Node::Apply { children, .. } => children[slot] = target,
        Node::Reduce { child, .. } | Node::PrefixScan { child, .. } => *child = target,
        Node::Matmul { lhs, rhs } => {
            if slot == 0 {
                *lhs = target;
            } else {
                *rhs = target;
            }
        }
        Node::Gather { data, base, .. } => {
            if slot == 0 {
                *data = target;
            } else if let Some(b) = base {
                *b = target;
            }
        }
        Node::Scatter { dest, updates, .. } => {
            if slot == 0 {
                *dest = target;
            } else {
                *updates = target;
            }
        }
        Node::SortNetwork { keys, .. } => *keys = target,
        Node::Iota { like, .. } => *like = target,
        Node::Flip { child, .. } => *child = target,
        _ => {}
    }
}

/// The mutable axis sites of `node` (the M7 taxonomy: Reduce axes entries /
/// PrefixScan / Gather / Scatter / SortNetwork / Iota).
fn axis_site_count(node: &Node) -> usize {
    match node {
        Node::Reduce { axes, .. } => axes.len(),
        Node::PrefixScan { .. }
        | Node::Gather { .. }
        | Node::Scatter { .. }
        | Node::SortNetwork { .. }
        | Node::Iota { .. }
        | Node::Flip { .. } => 1,
        _ => 0,
    }
}

fn set_axis(node: &mut Node, slot: usize, value: usize) {
    match node {
        Node::Reduce { axes, .. } => axes[slot] = value,
        Node::PrefixScan { axis, .. }
        | Node::Gather { axis, .. }
        | Node::Scatter { axis, .. }
        | Node::SortNetwork { axis, .. }
        | Node::Iota { axis, .. } => *axis = value,
        _ => {}
    }
}

fn edge_sites(dag: &FlatDag) -> Vec<(usize, usize)> {
    dag.nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| (0..edge_count(n)).map(move |s| (i, s)))
        .collect()
}

fn bind_nodes(dag: &FlatDag) -> Vec<usize> {
    dag.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, Node::Bind(_)))
        .map(|(i, _)| i)
        .collect()
}

fn index_carriers(dag: &FlatDag) -> Vec<usize> {
    dag.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, Node::Gather { .. } | Node::Scatter { .. }))
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn test_recipe_fuzz_mutations_decline_typed() {
    // Every mutated recipe MUST return Err(_) — a typed decline, never a panic
    // (the test completing IS the no-panic evidence under the never-panic
    // contract). Fixed seeds; the assert message carries (class, iter, base)
    // for replay.
    //
    // Applicable bases per class (0-based ids into base_case; reported R1..R9):
    let classes: [&[usize]; 11] = [
        &[0, 1, 2, 3, 4, 5, 6, 7, 8], // M0 child index out of range
        &[0, 1, 2, 3, 4, 5, 6, 7, 8], // M1 self-cycle
        &[3, 5, 6, 8],                // M2 IndexRef::Node → non-index-lane node
        &[5, 6],                      // M3 IndexRef::Slot out of range
        &[4],                         // M4 index_outputs invalid
        &[0, 1, 2, 3, 4, 5, 6, 7, 8], // M5 Bind out of range
        &[2],                         // M6 RuntimeScalar slot out of range
        &[1, 2, 3, 4, 5, 7],          // M7 axis out of range
        &[0, 5],                      // M8 shape mismatch
        &[0, 1, 2, 3, 4, 5, 6, 7, 8], // M9 outputs out of range
        &[6],                         // M10 Skip + base:None with an OOB index
    ];
    for (class_id, bases) in classes.iter().enumerate() {
        let seed = MASTER_SEED ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(class_id as u64 + 1);
        let mut rng = Rng(seed);
        for iter in 0..ITERATIONS {
            let base_id = bases[rng.pick(bases.len())];
            let (mut dag, inputs, params, mut indices) = base_case(base_id);
            match class_id {
                // M0: one child edge of a random non-leaf node → nodes.len()+k.
                0 => {
                    let sites = edge_sites(&dag);
                    let (ni, slot) = sites[rng.pick(sites.len())];
                    let target = dag.nodes.len() + rng.pick(8);
                    set_edge(&mut dag.nodes[ni], slot, target);
                }
                // M1: self-cycle (revisit-while-active guard).
                1 => {
                    let sites = edge_sites(&dag);
                    let (ni, slot) = sites[rng.pick(sites.len())];
                    set_edge(&mut dag.nodes[ni], slot, ni);
                }
                // M2: Gather/Scatter index → IndexRef::Node(a Bind).
                2 => {
                    let binds = bind_nodes(&dag);
                    let b = binds[rng.pick(binds.len())];
                    let carriers = index_carriers(&dag);
                    let c = carriers[rng.pick(carriers.len())];
                    match &mut dag.nodes[c] {
                        Node::Gather { index, .. } | Node::Scatter { index, .. } => {
                            *index = IndexRef::Node(b);
                        }
                        _ => {}
                    }
                }
                // M3: IndexRef::Slot(indices.len()+k).
                3 => {
                    let carriers = index_carriers(&dag);
                    let c = carriers[rng.pick(carriers.len())];
                    let slot = indices.len() + rng.pick(4);
                    match &mut dag.nodes[c] {
                        Node::Gather { index, .. } | Node::Scatter { index, .. } => {
                            *index = IndexRef::Slot(slot);
                        }
                        _ => {}
                    }
                }
                // M4: index_outputs naming (a) a non-index node, (b) an
                // out-of-range node id.
                4 => {
                    if rng.next() % 2 == 0 {
                        dag.index_outputs.push(0); // Bind — no index lane
                    } else {
                        dag.index_outputs.push(dag.nodes.len() + rng.pick(4));
                    }
                }
                // M5: Bind(inputs.len()+k).
                5 => {
                    let binds = bind_nodes(&dag);
                    let b = binds[rng.pick(binds.len())];
                    dag.nodes[b] = Node::Bind(inputs.len() + rng.pick(4));
                }
                // M6: RuntimeScalar(params.len()+k).
                6 => {
                    let rs: Vec<usize> = dag
                        .nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| matches!(n, Node::RuntimeScalar(_)))
                        .map(|(i, _)| i)
                        .collect();
                    let s = rs[rng.pick(rs.len())];
                    dag.nodes[s] = Node::RuntimeScalar(params.len() + rng.pick(4));
                }
                // M7: an axis site bumped to rank+1+k.
                7 => {
                    let sites: Vec<(usize, usize)> = dag
                        .nodes
                        .iter()
                        .enumerate()
                        .flat_map(|(i, n)| (0..axis_site_count(n)).map(move |s| (i, s)))
                        .collect();
                    let (ni, slot) = sites[rng.pick(sites.len())];
                    // max operand rank per base: R2/R3/R8 are rank-2, the rest rank-1.
                    let rank = match base_id {
                        1 | 2 | 7 => 2,
                        _ => 1,
                    };
                    set_axis(&mut dag.nodes[ni], slot, rank + 1 + rng.pick(4));
                }
                // M8: curated shape mismatches.
                8 => {
                    if base_id == 0 {
                        // R1: rewire matmul rhs to the [2] bias input (K mismatch).
                        if let Node::Matmul { rhs, .. } = &mut dag.nodes[2] {
                            *rhs = 3;
                        }
                    } else {
                        // R6: scatter index length != updates extent.
                        let old: Vec<i64> = indices[0].as_slice().to_vec();
                        let keep = old.len() - (1 + rng.pick(3));
                        indices[0] = ix(&old[..keep], &[keep]);
                    }
                }
                // M9: outputs out of range.
                9 => {
                    dag.outputs = vec![dag.nodes.len() + rng.pick(4)];
                }
                // M10: R7 Skip + base:None with an actually-OOB index — the
                // kernel's typed Unsupported(Gather) decline (deterministic;
                // all iterations identical by construction, kept for
                // uniformity).
                _ => {
                    if let Node::Gather { base, .. } = &mut dag.nodes[2] {
                        *base = None;
                    }
                }
            }
            let err = eval_recipe(&dag, &inputs, &params, &indices).err();
            let tag = format!("M{class_id} iter {iter} base R{}", base_id + 1);
            match class_id {
                2 | 4 => assert!(
                    matches!(err, Some(Error::IndexSourceInvalid { .. })),
                    "{tag}: want IndexSourceInvalid, got {err:?}"
                ),
                // M3 mutates a gather/scatter IndexRef::Slot out of range — a
                // missing INDEX operand, distinct from a missing input/param.
                3 => assert!(
                    matches!(err, Some(Error::MissingIndexOperand { .. })),
                    "{tag}: want MissingIndexOperand, got {err:?}"
                ),
                5 | 6 => assert!(
                    matches!(err, Some(Error::MissingInput(_))),
                    "{tag}: want MissingInput, got {err:?}"
                ),
                7 => assert!(
                    matches!(err, Some(Error::AxisOutOfRange { .. })),
                    "{tag}: want AxisOutOfRange, got {err:?}"
                ),
                // M10 pins the SPECIFIC kernel decline its comment claims — the
                // Skip+no-base OOB read is the §6.11 gather-skip RFC surface, so
                // its variant must not silently drift (adversarial-review find:
                // this was a catch-all `is_some` that no test in the repo backed).
                10 => assert!(
                    matches!(err, Some(Error::GatherSkipNoBase)),
                    "{tag}: want GatherSkipNoBase, got {err:?}"
                ),
                _ => assert!(err.is_some(), "{tag}: mutation must decline, got Ok"),
            }
        }
    }
}

#[test]
fn test_recipe_constbits_exact_roundtrip() {
    // KISS-OPS-6.12-0002: a const(bits) leaf carries its value as the EXACT dtype bit
    // pattern, round-tripping ±0, ±inf, subnormals, and quiet/signalling NaN payloads.
    // kiss-ref has two const leaves:
    //   * Node::ConstBits(u64) — the BIT-EXACT leaf (T::from_bits): the low T-width bits
    //     are reinterpreted verbatim, so ANY pattern round-trips bit-for-bit on EVERY
    //     lane. This is the 6.12-0002-compliant const(bits).
    //   * Node::Const(f64) — the ergonomic real-valued leaf (T::from_f64), re-rounded per
    //     lane; a signalling-NaN payload survives only on the f64 lane.
    // The signalling-NaN payload is the discriminating case.
    let bits_f64 = |n: Node| -> u64 {
        let dag = FlatDag::new(vec![n], vec![0]);
        eval_recipe::<f64>(&dag, &[], &[], &[]).unwrap().outputs[0].as_slice()[0].to_bits()
    };
    let bits_f32 = |n: Node| -> u32 {
        let dag = FlatDag::new(vec![n], vec![0]);
        eval_recipe::<f32>(&dag, &[], &[], &[]).unwrap().outputs[0].as_slice()[0].to_bits()
    };
    let bits_e4m3 = |n: Node| -> u8 {
        let dag = FlatDag::new(vec![n], vec![0]);
        eval_recipe::<E4m3>(&dag, &[], &[], &[]).unwrap().outputs[0].as_slice()[0].to_bits()
    };

    // --- Node::ConstBits: bit-exact on EVERY lane (6.12-0002 satisfied) ---
    let f64_snan = 0x7FF0_0000_0000_0001u64; // exp all-ones, mantissa-MSB clear, payload 1
    assert_eq!(bits_f64(Node::ConstBits(f64_snan)), f64_snan); // signalling payload survives
    assert_eq!(
        bits_f64(Node::ConstBits(0x8000_0000_0000_0000)),
        0x8000_0000_0000_0000
    ); // -0.0
    assert_eq!(bits_f64(Node::ConstBits(1)), 1); // smallest subnormal

    let f32_snan = 0x7F80_0001u32; // f32 signalling NaN, payload 1
    assert_eq!(bits_f32(Node::ConstBits(f32_snan as u64)), f32_snan); // payload survives on f32!
    assert_eq!(bits_f32(Node::ConstBits(0x8000_0000)), 0x8000_0000); // -0.0

    // e4m3fn: the 8-bit pattern is taken verbatim — even the canonical-NaN code and -0.0
    // round-trip, patterns no real-value from_f64 path could target distinctly.
    assert_eq!(bits_e4m3(Node::ConstBits(0x7F)), 0x7F); // e4m3 NaN code
    assert_eq!(bits_e4m3(Node::ConstBits(0x80)), 0x80); // e4m3 -0.0
    assert_eq!(bits_e4m3(Node::ConstBits(0x01)), 0x01); // e4m3 smallest subnormal

    // --- Node::Const(f64): the ergonomic real-valued leaf, for contrast ---
    // f64 lane: from_f64 is identity, so the signalling payload survives here too...
    assert_eq!(bits_f64(Node::Const(f64::from_bits(f64_snan))), f64_snan);
    assert_eq!(bits_f64(Node::Const(-0.0)), 0x8000_0000_0000_0000);
    // ...but on a narrow lane the real value is RE-ROUNDED, so the signalling payload is
    // NOT preserved (only guaranteed to still be *a* NaN) — exactly why the bit-exact
    // ConstBits leaf exists. Finite / ±0 / ±inf DO round-trip through Const:
    assert_eq!(bits_f32(Node::Const(-0.0)), 0x8000_0000);
    assert_eq!(bits_f32(Node::Const(f64::INFINITY)), 0x7F80_0000);
    let via_const = bits_f32(Node::Const(f64::from_bits(f64_snan)));
    assert_eq!(via_const & 0x7F80_0000, 0x7F80_0000); // still exp all-ones...
    assert_ne!(via_const & 0x007F_FFFF, 0); // ...a NaN (nonzero mantissa); payload not pinned
}

// ---- canonical transformer fragments (worked recipe examples) --------------
// Named, hand-checked goldens for the fragments a decoder forward is built from.
// A consumer models its own recipe on these rather than inventing the shape — the
// keepdim-reduce-then-broadcast, the additive -inf mask, and softmax-as-a-sub-DAG
// are exactly where graph-construction bugs hide. Exact where the arithmetic is
// dyadic; tolerance (per the root DetClass) where a transcendental or a
// nondeterministic reduction/matmul is involved. Goldens for the transcendental
// fragments are DERIVED from the math (std `exp`), not hand-typed literals.

/// R10: RMSNorm `y = x / sqrt(mean(x^2) + eps) * w`. The 11-node shape a fused
/// RMSNorm kernel is diffed against — KISS-OPS-6.11-0002 keepdim Sum, the
/// ReducedCount mean idiom, a RuntimeScalar eps, two broadcasts.
fn r10_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // 0 x [2,4]
            Node::Apply {
                op: Op::Mul,
                children: vec![0, 0],
            }, // 1 x^2
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 1,
            }, // 2 sum x^2 [2,1]
            Node::ReducedCount(vec![1]), // 3 d = 4
            Node::Apply {
                op: Op::Div,
                children: vec![2, 3],
            }, // 4 mean
            Node::RuntimeScalar(0), // 5 eps
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            }, // 6 mean+eps
            Node::Apply {
                op: Op::Sqrt,
                children: vec![6],
            }, // 7 rms [2,1]
            Node::Apply {
                op: Op::Div,
                children: vec![0, 7],
            }, // 8 x/rms [2,4] (broadcast [2,1] over [2,4])
            Node::Bind(1), // 9 w [4]
            Node::Apply {
                op: Op::Mul,
                children: vec![8, 9],
            }, // 10 * w [2,4] (broadcast [4] over [2,4])
        ],
        vec![10],
    )
}
fn r10_inputs() -> Vec<Tensor<f64>> {
    // row0 all-2 (mean x^2 = 4), row1 = [4,0,0,0] (mean x^2 = 4); eps=12 -> rms=4.
    vec![
        t64(&[2.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0], &[2, 4]),
        t64(&[1.0, 2.0, 1.0, 2.0], &[4]),
    ]
}

#[test]
fn test_recipe_rmsnorm_rows() {
    // eps=12 keeps rms exact (sqrt(4+12)=4), so the whole fragment is dyadic-exact;
    // the root chains through a Sum reduce (OIN) so it is compared with tolerance.
    let r = eval_recipe(&r10_dag(), &r10_inputs(), &[12.0], &[]).expect("R10 must evaluate");
    // row0: [2,2,2,2]/4 * [1,2,1,2] = [.5,1,.5,1]; row1: [4,0,0,0]/4 * w = [1,0,0,0].
    assert_close(
        &r.outputs[0],
        &[0.5, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0],
        &[2, 4],
        1e-12,
    );
}

/// R11: SwiGLU MLP `(silu(x·Wg) ⊙ (x·Wu)) · Wd` — two parallel matmuls, a silu
/// gate, an elementwise product, an output matmul. Golden derived from silu(1).
fn r11_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 x [1,2]
            Node::Bind(1),                   // 1 Wg [2,2]
            Node::Matmul { lhs: 0, rhs: 1 }, // 2 x·Wg [1,2]
            Node::Apply {
                op: Op::Silu,
                children: vec![2],
            }, // 3 silu [1,2]
            Node::Bind(2),                   // 4 Wu [2,2]
            Node::Matmul { lhs: 0, rhs: 4 }, // 5 x·Wu [1,2]
            Node::Apply {
                op: Op::Mul,
                children: vec![3, 5],
            }, // 6 gate ⊙ up [1,2]
            Node::Bind(3),                   // 7 Wd [2,2]
            Node::Matmul { lhs: 6, rhs: 7 }, // 8 · Wd [1,2]
        ],
        vec![8],
    )
}
fn r11_inputs() -> Vec<Tensor<f64>> {
    vec![
        t64(&[1.0, 1.0], &[1, 2]),           // x
        t64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), // Wg = I  -> x·Wg = [1,1]
        t64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]), // Wu = I  -> x·Wu = [1,1]
        t64(&[1.0, 1.0, 1.0, 1.0], &[2, 2]), // Wd = ones -> sums both lanes
    ]
}

#[test]
fn test_recipe_swiglu_mlp() {
    let r = eval_recipe(&r11_dag(), &r11_inputs(), &[], &[]).expect("R11 must evaluate");
    // x·Wg = x·Wu = [1,1]; gate ⊙ up = [silu(1), silu(1)]; ·Wd(ones) sums lanes.
    let silu1 = 1.0 / (1.0 + (-1.0_f64).exp());
    assert_close(&r.outputs[0], &[2.0 * silu1, 2.0 * silu1], &[1, 2], 1e-9);
}

/// R12: single-head CAUSAL scaled-dot-product attention `softmax(Q·Kᵀ·scale +
/// mask)·V`. Softmax is the R2 sub-DAG (max-shift / exp / sum / div); the causal
/// mask is the ADDITIVE -inf form (what most kernels use). K is passed
/// pre-transposed — layout is the consumer's job, kiss-ref gets the logical Kᵀ.
/// The interesting fragment: mask broadcast, softmax axis, the two matmuls.
fn r12_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 Q [2,2]
            Node::Bind(1),                   // 1 Kᵀ [2,2]
            Node::Matmul { lhs: 0, rhs: 1 }, // 2 scores [2,2]
            Node::RuntimeScalar(0),          // 3 scale = 1/sqrt(d)
            Node::Apply {
                op: Op::Mul,
                children: vec![2, 3],
            }, // 4 scaled [2,2]
            Node::Bind(2),                   // 5 causal mask [2,2] (0 / -inf)
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            }, // 6 masked [2,2]
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![1],
                keepdim: true,
                child: 6,
            }, // 7 rowmax [2,1]
            Node::Apply {
                op: Op::Sub,
                children: vec![6, 7],
            }, // 8 shifted [2,2]
            Node::Apply {
                op: Op::Exp,
                children: vec![8],
            }, // 9 exp [2,2]
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 9,
            }, // 10 denom [2,1]
            Node::Apply {
                op: Op::Div,
                children: vec![9, 10],
            }, // 11 weights [2,2]
            Node::Bind(3),                   // 12 V [2,2]
            Node::Matmul { lhs: 11, rhs: 12 }, // 13 out [2,2]
        ],
        vec![13],
    )
}
fn r12_inputs() -> Vec<Tensor<f64>> {
    vec![
        t64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),               // Q = I
        t64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),               // Kᵀ = I -> scores = I
        t64(&[0.0, f64::NEG_INFINITY, 0.0, 0.0], &[2, 2]), // causal: (0,1) = -inf
        t64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),               // V
    ]
}

#[test]
fn test_recipe_attention_causal_single_head() {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt(); // scale = 1/sqrt(d), d=2
    let r = eval_recipe(&r12_dag(), &r12_inputs(), &[inv_sqrt2], &[]).expect("R12 must evaluate");
    // scaled scores = diag(s); mask sends (0,1) -> -inf.
    // row0: softmax([s, -inf]) = [1, 0] -> out0 = V[0] = [1, 2].
    // row1: softmax([0, s]) = [e^-s, 1]/(e^-s + 1) -> out1 = w0·V[0] + w1·V[1].
    let e = (-inv_sqrt2).exp();
    let w0 = e / (e + 1.0);
    let w1 = 1.0 / (e + 1.0);
    assert_close(
        &r.outputs[0],
        &[1.0, 2.0, w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0],
        &[2, 2],
        1e-9,
    );
}

/// R13: RoPE `out = x·cos + rotate_half(x)·sin`, the rotate_half convention (a
/// single d=2 rotation pair). `rotate_half([x0,x1]) = [-x1, x0]` is built as
/// `Flip` (reverse the pair) × a `[-1, 1]` sign — the only corpus recipe that
/// exercises `Node::Flip` in a real pattern. cos/sin tables are inputs (the
/// position·freq schedule is the consumer's; kiss-ref gets the logical tables).
/// The interesting bug site: the half-swap indexing and the sign placement.
fn r13_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                    // 0 x [1,2]
            Node::Bind(1),                    // 1 cos [2]
            Node::Bind(2),                    // 2 sin [2]
            Node::Flip { child: 0, axis: 1 }, // 3 [x1, x0]
            Node::Bind(3),                    // 4 sign = [-1, 1]
            Node::Apply {
                op: Op::Mul,
                children: vec![3, 4],
            }, // 5 rotate_half = [-x1, x0]
            Node::Apply {
                op: Op::Mul,
                children: vec![0, 1],
            }, // 6 x·cos
            Node::Apply {
                op: Op::Mul,
                children: vec![5, 2],
            }, // 7 rotate_half·sin
            Node::Apply {
                op: Op::Add,
                children: vec![6, 7],
            }, // 8 out
        ],
        vec![8],
    )
}
fn r13_inputs() -> Vec<Tensor<f64>> {
    let (c, s) = (1.0_f64.cos(), 1.0_f64.sin()); // angle = 1 rad
    vec![
        t64(&[1.0, 0.0], &[1, 2]), // x = [1, 0]
        t64(&[c, c], &[2]),        // cos table
        t64(&[s, s], &[2]),        // sin table
        t64(&[-1.0, 1.0], &[2]),   // sign for rotate_half
    ]
}

#[test]
fn test_recipe_rope_rotate_half() {
    let r = eval_recipe(&r13_dag(), &r13_inputs(), &[], &[]).expect("R13 must evaluate");
    // x=[1,0]: rotate_half=[-0,1]; out = [1·cos + (-0)·sin, 0·cos + 1·sin] = [cos1, sin1]
    // (the 2-D rotation of the unit x-vector by 1 rad). Exact given the cos/sin inputs.
    let (c, s) = (1.0_f64.cos(), 1.0_f64.sin());
    assert_close(&r.outputs[0], &[c, s], &[1, 2], 1e-15);
}

/// R14: PRE-NORM RESIDUAL BLOCK `y = x + (RMSNorm(x) · W)` — the structural spine
/// of a decoder layer (the two defining patterns: pre-normalization before a
/// sublayer, and the residual add around it). Composes R10's RMSNorm into a
/// projection + residual; with R11 (SwiGLU) and R12 (attention) as the sublayers,
/// this is how a consumer wires a full layer. The bug site: threading the SAME `x`
/// into both the norm AND the residual add (a mis-wire silently drops the skip).
fn r14_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0), // 0 x [2,4]
            Node::Apply {
                op: Op::Mul,
                children: vec![0, 0],
            }, // 1 x^2
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 1,
            }, // 2 sum x^2 [2,1]
            Node::ReducedCount(vec![1]), // 3 d=4
            Node::Apply {
                op: Op::Div,
                children: vec![2, 3],
            }, // 4 mean
            Node::RuntimeScalar(0), // 5 eps
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            }, // 6 mean+eps
            Node::Apply {
                op: Op::Sqrt,
                children: vec![6],
            }, // 7 rms [2,1]
            Node::Apply {
                op: Op::Div,
                children: vec![0, 7],
            }, // 8 x/rms [2,4]
            Node::Bind(1), // 9 w_rms [4]
            Node::Apply {
                op: Op::Mul,
                children: vec![8, 9],
            }, // 10 normed = RMSNorm(x) [2,4]
            Node::Bind(2), // 11 W [4,4]
            Node::Matmul { lhs: 10, rhs: 11 }, // 12 sublayer = normed·W [2,4]
            Node::Apply {
                op: Op::Add,
                children: vec![0, 12],
            }, // 13 y = x + sublayer (residual; x is node 0, reused)
        ],
        vec![13],
    )
}
fn r14_inputs() -> Vec<Tensor<f64>> {
    vec![
        t64(&[2.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0], &[2, 4]), // x (R10's inputs)
        t64(&[1.0, 2.0, 1.0, 2.0], &[4]),                        // w_rms
        // W = 2·I -> sublayer = 2·normed (exercises the matmul, stays dyadic).
        t64(
            &[
                2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
            ],
            &[4, 4],
        ),
    ]
}

#[test]
fn test_recipe_prenorm_residual_block() {
    // eps=12 -> RMSNorm(x) = R10's [0.5,1,0.5,1, 1,0,0,0]; ·2I -> [1,2,1,2, 2,0,0,0];
    // residual + x([2,2,2,2, 4,0,0,0]) -> [3,4,3,4, 6,0,0,0]. Exact dyadic; root is a
    // residual Add over a Sum-reduce + Matmul chain (OIN) so compared with tolerance.
    let r = eval_recipe(&r14_dag(), &r14_inputs(), &[12.0], &[]).expect("R14 must evaluate");
    assert_close(
        &r.outputs[0],
        &[3.0, 4.0, 3.0, 4.0, 6.0, 0.0, 0.0, 0.0],
        &[2, 4],
        1e-12,
    );
}

/// R15: MULTI-HEAD causal attention — R12 lifted to a batched head dimension. The
/// ONE new thing over R12: the head axis is a batched-matmul leading dim
/// (`[H,S,D]·[H,D,S]→[H,S,S]`, KISS-OPS-6.20-0007 batch broadcast), the softmax
/// runs over the LAST axis (2, the key dimension, NOT the head axis), and the
/// causal mask `[S,S]` broadcasts across ALL heads. There is no reshape/split atom
/// in the grammar, so heads arrive pre-shaped — layout is the consumer's job,
/// exactly as R12 gets a pre-transposed Kᵀ. Per-head independence is the bug site a
/// fused MHA kernel must preserve: head 1 must not leak into head 0.
fn r15_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 Q  [2,2,2] (H,S,D)
            Node::Bind(1),                   // 1 Kᵀ [2,2,2] (H,D,S)
            Node::Matmul { lhs: 0, rhs: 1 }, // 2 scores [2,2,2] batched over H
            Node::RuntimeScalar(0),          // 3 scale = 1/sqrt(d)
            Node::Apply {
                op: Op::Mul,
                children: vec![2, 3],
            }, // 4 scaled [2,2,2]
            Node::Bind(2),                   // 5 causal mask [2,2] (0 / -inf), broadcast over H
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            }, // 6 masked [2,2,2]
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![2],
                keepdim: true,
                child: 6,
            }, // 7 rowmax [2,2,1] (over keys)
            Node::Apply {
                op: Op::Sub,
                children: vec![6, 7],
            }, // 8 shifted [2,2,2]
            Node::Apply {
                op: Op::Exp,
                children: vec![8],
            }, // 9 exp [2,2,2]
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![2],
                keepdim: true,
                child: 9,
            }, // 10 denom [2,2,1]
            Node::Apply {
                op: Op::Div,
                children: vec![9, 10],
            }, // 11 weights [2,2,2]
            Node::Bind(3),                   // 12 V [2,2,2] (H,S,D)
            Node::Matmul { lhs: 11, rhs: 12 }, // 13 out [2,2,2]
        ],
        vec![13],
    )
}
fn r15_inputs() -> Vec<Tensor<f64>> {
    let ninf = f64::NEG_INFINITY;
    vec![
        // Q: both heads = I2, so scores[h] = Kᵀ[h] (isolates the head-batching).
        t64(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]),
        // Kᵀ: head0 = 2·I2 → scores0 = [[2,0],[0,2]]; head1 = 2·swap →
        // scores1 = [[0,2],[2,0]]. Distinct heads.
        t64(&[2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0], &[2, 2, 2]),
        // causal mask [S,S]: position 0 cannot see position 1.
        t64(&[0.0, ninf, 0.0, 0.0], &[2, 2]),
        // V: head0 rows [1,2],[3,4]; head1 rows [10,20],[30,40].
        t64(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 2, 2]),
    ]
}

#[test]
fn test_recipe_attention_multi_head() {
    // scale s = 1/sqrt(d), d=2. Let a = exp(-2s) = exp(-sqrt2).
    // Head 0 (scores [[2,0],[0,2]]·s, +mask):
    //   row0 = [2s, -inf] -> softmax [1,0] -> out = V0[0] = [1,2].
    //   row1 = [0, 2s]    -> softmax [a/(1+a), 1/(1+a)]
    //                        -> [(a+3)/(1+a), (2a+4)/(1+a)].
    // Head 1 (scores [[0,2],[2,0]]·s, +mask):
    //   row0 = [0, -inf]  -> softmax [1,0] -> out = V1[0] = [10,20].
    //   row1 = [2s, 0]    -> softmax [1/(1+a), a/(1+a)]
    //                        -> [(10+30a)/(1+a), (20+40a)/(1+a)].
    // Root chains through a Sum-reduce + Matmul (OIN) -> tolerance compare.
    let s = 1.0 / 2.0_f64.sqrt();
    let a = (-2.0 * s).exp();
    let d = 1.0 + a;
    let r = eval_recipe(&r15_dag(), &r15_inputs(), &[s], &[]).expect("R15 must evaluate");
    assert_close(
        &r.outputs[0],
        &[
            1.0,
            2.0,
            (a + 3.0) / d,
            (2.0 * a + 4.0) / d,
            10.0,
            20.0,
            (10.0 + 30.0 * a) / d,
            (20.0 + 40.0 * a) / d,
        ],
        &[2, 2, 2],
        1e-12,
    );
    // Head independence has teeth: the two heads' outputs must differ (a fused
    // kernel that shares state across heads would collapse them).
    let o = r.outputs[0].as_slice();
    assert!(o[0..4] != o[4..8], "heads must be independent");
}

/// R16: KV-CACHE DECODE STEP — the autoregressive one-token attention. A single
/// query row `[1,D]` attends over the FULL cached keys/values `[S_kv,D]`; there is
/// no mask (a decode token sees every cached position). The physical cache append
/// is the consumer's memory job — there is no concat atom in §6.11/§6.12, so
/// kiss-ref receives the logical post-append cache, exactly as R12 receives a
/// logical Kᵀ. The bug site is the SINGLE-ROW softmax: a `[1,S_kv]` rowmax/denom
/// that a multi-row path gets right can still be mis-broadcast at S_q=1.
fn r16_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 q  [1,2]
            Node::Bind(1),                   // 1 Kᵀ [2,3] (D, S_kv)
            Node::Matmul { lhs: 0, rhs: 1 }, // 2 scores [1,3]
            Node::RuntimeScalar(0),          // 3 scale
            Node::Apply {
                op: Op::Mul,
                children: vec![2, 3],
            }, // 4 scaled [1,3]
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![1],
                keepdim: true,
                child: 4,
            }, // 5 rowmax [1,1]
            Node::Apply {
                op: Op::Sub,
                children: vec![4, 5],
            }, // 6 shifted [1,3]
            Node::Apply {
                op: Op::Exp,
                children: vec![6],
            }, // 7 exp [1,3]
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 7,
            }, // 8 denom [1,1]
            Node::Apply {
                op: Op::Div,
                children: vec![7, 8],
            }, // 9 weights [1,3]
            Node::Bind(2),                   // 10 V_full [3,2]
            Node::Matmul { lhs: 9, rhs: 10 }, // 11 out [1,2]
        ],
        vec![11],
    )
}
fn r16_inputs() -> Vec<Tensor<f64>> {
    vec![
        t64(&[1.0, 0.0], &[1, 2]),                     // q -> scores = Kᵀ row 0
        t64(&[0.0, 2.0, 4.0, 9.0, 9.0, 9.0], &[2, 3]), // Kᵀ: scores = [0,2,4]
        t64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), // V_full rows [1,2],[3,4],[5,6]
    ]
}

#[test]
fn test_recipe_kv_cache_decode() {
    // scores = [0,2,4]; scaled by s=1/sqrt2. max=4s, shifted=[-4s,-2s,0].
    // a = exp(-2s); exp = [a², a, 1]; D = a²+a+1.
    //   out = ([a²,a,1]/D)·V_full
    //       = [(a²+3a+5)/D, (2a²+4a+6)/D].
    let s = 1.0 / 2.0_f64.sqrt();
    let a = (-2.0 * s).exp();
    let dsum = a * a + a + 1.0;
    let r = eval_recipe(&r16_dag(), &r16_inputs(), &[s], &[]).expect("R16 must evaluate");
    assert_close(
        &r.outputs[0],
        &[
            (a * a + 3.0 * a + 5.0) / dsum,
            (2.0 * a * a + 4.0 * a + 6.0) / dsum,
        ],
        &[1, 2],
        1e-12,
    );
}

/// The KV-cache correctness contract, as a metamorphic invariant: incremental
/// decode of the LAST position MUST equal the last row of a full-sequence causal
/// recompute over the same K/V. This is the property that makes caching sound — if
/// it fails, cached generation silently diverges from a full forward pass.
#[test]
fn test_recipe_kv_cache_equals_full_recompute_last_row() {
    let s = 1.0 / 2.0_f64.sqrt();
    let ninf = f64::NEG_INFINITY;

    // Decode: the R16 single-row step (q is the last query, no mask).
    let decode = eval_recipe(&r16_dag(), &r16_inputs(), &[s], &[]).expect("decode must evaluate");

    // Full: S=3 causal attention. Q_full's LAST row equals the decode query
    // [1,0]; the causal mask's last row is all-visible [0,0,0], so the last
    // position sees the whole cache — exactly the decode step's condition.
    let full_dag = FlatDag::new(
        vec![
            Node::Bind(0),                   // 0 Q_full [3,2]
            Node::Bind(1),                   // 1 Kᵀ [2,3]
            Node::Matmul { lhs: 0, rhs: 1 }, // 2 scores [3,3]
            Node::RuntimeScalar(0),          // 3 scale
            Node::Apply {
                op: Op::Mul,
                children: vec![2, 3],
            }, // 4
            Node::Bind(2),                   // 5 causal mask [3,3]
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            }, // 6
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![1],
                keepdim: true,
                child: 6,
            }, // 7
            Node::Apply {
                op: Op::Sub,
                children: vec![6, 7],
            }, // 8
            Node::Apply {
                op: Op::Exp,
                children: vec![8],
            }, // 9
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 9,
            }, // 10
            Node::Apply {
                op: Op::Div,
                children: vec![9, 10],
            }, // 11 weights [3,3]
            Node::Bind(3),                   // 12 V_full [3,2]
            Node::Matmul { lhs: 11, rhs: 12 }, // 13 out [3,2]
        ],
        vec![13],
    );
    let full_inputs = vec![
        t64(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[3, 2]), // Q_full: last row = [1,0] = decode q
        t64(&[0.0, 2.0, 4.0, 9.0, 9.0, 9.0], &[2, 3]), // same Kᵀ as R16
        t64(&[0.0, ninf, ninf, 0.0, 0.0, ninf, 0.0, 0.0, 0.0], &[3, 3]), // causal
        t64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]), // same V_full as R16
    ];
    let full = eval_recipe(&full_dag, &full_inputs, &[s], &[]).expect("full must evaluate");

    // Last row (position 2) of the [3,2] full output vs the [1,2] decode output.
    let dec = decode.outputs[0].as_slice();
    let last = &full.outputs[0].as_slice()[4..6];
    assert_eq!(dec.len(), 2);
    for (i, (&g, &w)) in dec.iter().zip(last).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12,
            "kv-cache decode elem {i}: decode {g} != full-recompute last row {w}"
        );
    }
}

// ---- narrow-lane execution of the transformer composites --------------------
// The corpus proves these fragments at f64; narrow_tensor_lane proves individual
// ops + softmax at f16/bf16/FP8. This closes the remaining gap: a MULTI-STEP
// composite forward (RMSNorm's fold chain, attention's matmul→softmax→matmul) run
// at f16 and bf16 — the realistic inference precision — had no coverage.

#[test]
fn test_recipe_rmsnorm_narrow_lanes_are_exact() {
    // With the dyadic corpus inputs (eps=12 → rms=4 exact), every RMSNorm
    // intermediate (x², sum, mean, +eps, sqrt, x/rms, ·w) is a small dyadic value
    // representable in both f16 (10 mantissa bits) and bf16 (7), so the whole fold
    // composite is EXACT — it lands on the same [0.5,1,0.5,1, 1,0,0,0] as f64.
    fn rmsnorm_narrow<T: ScalarFloat>() {
        let inputs = vec![
            tf::<T>(&[2.0, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0], &[2, 4]),
            tf::<T>(&[1.0, 2.0, 1.0, 2.0], &[4]),
        ];
        let r = eval_recipe(&r10_dag(), &inputs, &[T::from_f64(12.0)], &[])
            .expect("R10 narrow must evaluate");
        // tol = 0.0: the claim is EXACTNESS, so assert it — every intermediate is
        // a dyadic value representable in f16/bf16, so the narrow forward lands on
        // the f64 result bit-for-bit. A loose tol would let an unintended
        // intermediate widening/rounding regress silently.
        assert_close(
            &r.outputs[0],
            &[0.5, 1.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0],
            &[2, 4],
            0.0,
        );
    }
    rmsnorm_narrow::<f16>();
    rmsnorm_narrow::<bf16>();
}

#[test]
fn test_recipe_attention_narrow_lanes_land_in_band() {
    // Attention's softmax exp() forces PER-ATOM narrow rounding, so the narrow
    // output is a structured approximation of the f64 golden. Pin that it executes,
    // stays finite, and lands within a precision-appropriate band. Row 0 is exact
    // (the -inf causal mask → one-hot softmax → V[0]); only row 1 carries the
    // transcendental rounding. A broken attention (dropped mask, wrong softmax
    // axis, wrong matmul) deviates by ≥ 0.5, far outside either band.
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let e = (-inv_sqrt2).exp();
    let (w0, w1) = (e / (e + 1.0), 1.0 / (e + 1.0));
    let golden = [1.0, 2.0, w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0];

    fn attention_narrow<T: ScalarFloat>(inv_sqrt2: f64, golden: &[f64], band: f64) {
        let ninf = f64::NEG_INFINITY;
        let inputs = vec![
            tf::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),  // Q = I
            tf::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]),  // Kᵀ = I
            tf::<T>(&[0.0, ninf, 0.0, 0.0], &[2, 2]), // causal mask
            tf::<T>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),  // V
        ];
        let r = eval_recipe(&r12_dag(), &inputs, &[T::from_f64(inv_sqrt2)], &[])
            .expect("R12 narrow must evaluate");
        let got = r.outputs[0].as_slice();
        assert_eq!(got.len(), golden.len(), "shape");
        for (i, (&g, &w)) in got.iter().zip(golden).enumerate() {
            let gv = g.to_f64();
            assert!(gv.is_finite(), "R12 narrow elem {i} must be finite");
            assert!(
                (gv - w).abs() <= band,
                "R12 narrow elem {i}: got {gv} want ~{w} (band {band})"
            );
        }
    }
    // f16: 10 mantissa bits; bf16: 7 — a coarser band.
    attention_narrow::<f16>(inv_sqrt2, &golden, 2e-2);
    attention_narrow::<bf16>(inv_sqrt2, &golden, 6e-2);
}

#[test]
fn test_recipe_per_output_determinism_class() {
    // KISS-OPS-6.0-0007: per-output determinism class. A VALUE output's class = the
    // most-permissive join over its producing sub-DAG (`dets[node]`); a SELECTION output
    // (an exported index/permutation) escalates via `selection_det` — ExactByte iff the
    // producer is all-exact, ELSE OIN, NEVER Ulp (a permutation over non-exact keys is
    // not ULP-boundable). Two roots over a non-exact producer, checked both ways.

    // Case A — OIN keys: root A = float sum-reduction VALUE (OIN); root B = a sort
    // PERMUTATION over it, exported as an index output. B's sub-DAG contains the OIN
    // reduction, so B is OIN. The realization propagates via the node-det join.
    let dag_oin = FlatDag {
        nodes: vec![
            Node::Bind(0),
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: false,
                child: 0,
            },
            Node::SortNetwork {
                keys: 1,
                axis: 0,
                dir: Direction::Asc,
            },
        ],
        outputs: vec![1],
        index_outputs: vec![2],
    };
    let r = eval_recipe(
        &dag_oin,
        &[t64(&[3.0, 1.0, 2.0, 6.0, 5.0, 4.0], &[2, 3])],
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(
        r.dets,
        vec![EB, OIN, OIN],
        "value-lane classes: the reduction and the sort's sorted values are OIN"
    );
    assert_eq!(
        r.index_output_dets(&dag_oin),
        vec![OIN],
        "OIN keys: the exported permutation is OIN"
    );

    // Case B — Ulp keys: root A = exp() VALUE (Ulp(4)); root B = a sort PERMUTATION over
    // it. THE case the plain join gets wrong: the sort node's value-lane det is
    // EB ⊔ Ulp = Ulp, but a permutation over Ulp keys is not ULP-boundable, so the
    // SELECTION output must escalate to OIN — never Ulp.
    let dag_ulp = FlatDag {
        nodes: vec![
            Node::Bind(0),
            Node::Apply {
                op: Op::Exp,
                children: vec![0],
            },
            Node::SortNetwork {
                keys: 1,
                axis: 0,
                dir: Direction::Asc,
            },
        ],
        outputs: vec![1],
        index_outputs: vec![2],
    };
    let r2 = eval_recipe(&dag_ulp, &[t64(&[1.0, 0.0, 2.0], &[3])], &[], &[]).unwrap();
    assert_eq!(
        r2.dets,
        vec![EB, U4, U4],
        "value-lane classes: exp and the sort's sorted VALUES are Ulp(4)"
    );
    assert_eq!(
        r2.index_output_dets(&dag_ulp),
        vec![OIN],
        "Ulp keys: the exported permutation MUST escalate to OIN, not the join's Ulp"
    );
}
