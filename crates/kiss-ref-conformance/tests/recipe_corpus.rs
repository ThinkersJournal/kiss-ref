//! Recipe conformance corpus — golden flat-DAG recipes with hand-derived
//! outputs and per-node [`DetClass`] vectors.
//!
//! This corpus is the recipe-level differential conformance artifact — kiss-ref
//! is the differential target/proxy, NOT the oracle (conformance role, ratified
//! 2026-07-21). Golden values are hand-derived from KISS-Ops first principles
//! (§6.11-0002..0008, §6.12-0001, §6.13); the evaluator is CHECKED against
//! them, never the source of them. Comparison mode is chosen by the ROOT node's
//! [`DetClass`] per §6.0-0001..0005: `ExactByte` → `to_bits` equality; `Ulp(k)`
//! → within `k` by the sign-magnitude metric ([`ulp_distance_f64`], the
//! ops_conformance idiom); `OrderInvariantNondeterministic` → tolerance, NEVER
//! byte-exact. Recipes are the §6.4-0009/§6.19 logical flat-DAG form; ULP
//! ceilings are §6.8 (via the vocab); shapes §6.20-0007/0008; stable sort ties
//! §6.11-0007. Test naming mirrors KISS-Conform (`test_recipe_*`).

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    eval_recipe, ulp_distance_f64, Combine, DetClass, Direction, Error, FlatDag, IndexRef,
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

/// Root class `ExactByte` → per-element raw-bit equality (§6.0-0002).
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
/// byte-exact (§6.0-0004). NaN is never an expected corpus value.
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

/// Root class `Ulp(k)` → within `k` ULP by the sign-magnitude metric (§6.0-0003).
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
            Node::Bind(0),                                     // 0 A
            Node::Bind(1),                                     // 1 B
            Node::Matmul { lhs: 0, rhs: 1 },                   // 2
            Node::Bind(2),                                     // 3 bias
            Node::Apply { op: Op::Add, children: vec![2, 3] }, // 4
            Node::Apply { op: Op::Relu, children: vec![4] },   // 5
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
            Node::Bind(0),                                                                // 0
            Node::Reduce { monoid: Monoid::Max, axes: vec![1], keepdim: true, child: 0 }, // 1
            Node::Apply { op: Op::Sub, children: vec![0, 1] },                            // 2
            Node::Apply { op: Op::Exp, children: vec![2] },                               // 3
            Node::Reduce { monoid: Monoid::Sum, axes: vec![1], keepdim: true, child: 3 }, // 4
            Node::Apply { op: Op::Div, children: vec![3, 4] },                            // 5
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
            Node::Bind(0),                                                                // 0 x
            Node::Reduce { monoid: Monoid::Sum, axes: vec![1], keepdim: true, child: 0 }, // 1
            Node::ReducedCount(vec![1]),                                                  // 2 = 4.0
            Node::Apply { op: Op::Div, children: vec![1, 2] },                            // 3 mean
            Node::Apply { op: Op::Sub, children: vec![0, 3] },                            // 4 centered
            Node::Apply { op: Op::Mul, children: vec![4, 4] },                            // 5
            Node::Reduce { monoid: Monoid::Sum, axes: vec![1], keepdim: true, child: 5 }, // 6
            Node::Apply { op: Op::Div, children: vec![6, 2] },                            // 7 var
            Node::RuntimeScalar(0),                                                       // 8 eps
            Node::Apply { op: Op::Add, children: vec![7, 8] },                            // 9
            Node::Apply { op: Op::Sqrt, children: vec![9] },                              // 10
            Node::Apply { op: Op::Div, children: vec![4, 10] },                           // 11
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
            Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc }, // 2
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
            Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc },
        ],
        outputs: vec![1],
        index_outputs: vec![1],
    }
}
fn r5_inputs() -> Vec<Tensor<f64>> {
    vec![t64(&[2.0, 0.0, 2.0, -0.0, 3.0], &[5])]
}

/// R6: scatter_add histogram with one OOB write (skipped, §6.11-0005).
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
/// (§6.11-0004: negative is OOB, no from-end wrap).
fn r7_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),    // 0 data
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

/// R8: iota (§6.12-0001 coord leaf) × data — pins signed zero through the
/// recipe path.
fn r8_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),                                     // 0 x
            Node::Iota { like: 0, axis: 1 },                   // 1
            Node::Apply { op: Op::Mul, children: vec![1, 0] }, // 2
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
            Node::Apply { op: Op::Exp, children: vec![0] }, // 2
            Node::SortNetwork { keys: 2, axis: 0, dir: Direction::Asc }, // 3
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
    // §6.4-0009 flat-DAG matmul+epilogue; §6.20-0007 bias broadcast; §6.0-0004
    // nondet root propagation. Identical logical results + dets at both dtypes.
    r1_case::<f64>(1e-12);
    r1_case::<f32>(1e-6);
}

#[test]
fn test_recipe_softmax_rows() {
    // §6.13 softmax shape; §6.8 exp ceiling pinned exactly (Ulp(4.0)); Max-
    // reduce ExactByte vs Sum-reduce OIN (§6.0-0004). Hand arithmetic:
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
    // §6.11-0007 sort + §6.11-0004 gather via IndexRef::Node — a real
    // scheduling edge, ExactByte end-to-end, identical at both dtypes.
    r4_case::<f64>();
    r4_case::<f32>();
}

#[test]
fn test_recipe_sort_dual_output_stable_ties() {
    // §6.11-0007: total order has −0.0 == +0.0 (tie) and duplicate keys tie;
    // ties → lower original index, values are a RAW-BIT permutation. keys
    // [2.0, 0.0, 2.0, −0.0, 3.0] asc → (orig1,+0.0),(orig3,−0.0),(orig0,2.0),
    // (orig2,2.0),(orig4,3.0).
    let r = eval_recipe(&r5_dag(), &r5_inputs(), &[], &[]).expect("R5 must evaluate");
    assert_bits(&r.outputs[0], &[0.0, -0.0, 2.0, 2.0, 3.0], &[5]);
    // Pin the raw-bit ±0 permutation explicitly.
    assert_eq!(r.outputs[0].as_slice()[0].to_bits(), 0x0000_0000_0000_0000);
    assert_eq!(r.outputs[0].as_slice()[1].to_bits(), 0x8000_0000_0000_0000);
    // Index lane: the §6.11-0007 lower-original-index tie order.
    assert_eq!(r.index_outputs.len(), 1);
    assert_eq!(r.index_outputs[0].as_slice(), &[1, 3, 0, 2, 4]);
    // ONE class covers both lanes.
    assert_eq!(r.dets, vec![EB, EB]);
}

#[test]
fn test_recipe_scatter_add_histogram() {
    // §6.11-0005/-0006: OOB write skipped, atomic_add folds colliding sources
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
    // Main: §6.11-0004 (negative index is OOB, NO from-end wrap) + the cosigned
    // §6.11 gather-skip RFC base operand. idx [1,5,0,−1] over data [8,16,32]:
    // 1 in → 16; 5 OOB → base −0.5; 0 in → 8; −1 OOB → base −0.5.
    let r = eval_recipe(&r7_dag(), &r7_inputs(), &[], &r7_indices()).expect("R7 must evaluate");
    assert_bits(&r.outputs[0], &[16.0, -0.5, 8.0, -0.5], &[4]);
    assert_eq!(r.dets, vec![EB, EB, EB]);
    assert!(r.index_outputs.is_empty());

    // Det-pair variant: base = exp(const 0.0) carries Ulp(4.0). The base's
    // class joins ONLY under Skip — a static, policy-conditioned join, never
    // index-conditioned (§6.0-0005 precision the comparator relies on).
    let mk = |oob| {
        FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Const(0.0),
                Node::Apply { op: Op::Exp, children: vec![1] },
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
    // join — the §6.0-0005 join is policy-conditioned (static), never
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
    // §6.12-0001 coord leaf: iota axis 1 over [2,3] = [0,1,2,0,1,2]; product
    // with x — element [1,0] = (+0.0)·(−1.0) = −0.0 pins §6.2-0004 signed zero
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
        Node::Bind(_) | Node::Const(_) | Node::RuntimeScalar(_) | Node::ReducedCount(_) => 0,
        Node::Apply { children, .. } => children.len(),
        Node::Reduce { .. }
        | Node::PrefixScan { .. }
        | Node::SortNetwork { .. }
        | Node::Iota { .. } => 1,
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
        | Node::Iota { .. } => 1,
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
                3 | 5 | 6 => assert!(
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
                    matches!(err, Some(Error::Unsupported(Op::Gather))),
                    "{tag}: want Unsupported(Gather), got {err:?}"
                ),
                _ => assert!(err.is_some(), "{tag}: mutation must decline, got Ok"),
            }
        }
    }
}
