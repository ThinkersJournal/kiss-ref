//! The **integer recipe evaluator**: `eval_recipe_int(dag, node_dtypes, inputs,
//! params, indices) -> RecipeEval<i128>` — the `i128`-backed sibling of
//! [`crate::recipe::eval_recipe`].
//!
//! It walks the SAME logical flat-DAG ([`FlatDag`]/[`Node`]) as the float lane
//! with the SAME never-panic post-order worklist, but every node is evaluated in
//! the integer tensor lane ([`crate::tensor_int`]/[`crate::scalar_int`]) and
//! wrapped to a **per-node** output dtype (`node_dtypes[idx]`). The per-node dtype
//! is what lets a heterogeneous graph type-check: a comparator child produces a
//! `b1` mask, and a `reduce(sum)` over it counts into `i64` — two different dtypes
//! on one edge. (The float lane needs no such vector: it is homogeneously `T`.)
//!
//! **Determinism.** Every node is [`DetClass::ExactByte`]. Integer two's-complement
//! arithmetic is a ring homomorphism, so *accumulate-wide-then-truncate* equals
//! *accumulate-at-width* byte-for-byte: an integer `reduce`/`matmul` is both
//! order-independent AND exact, unlike its float counterpart (which is
//! [`DetClass::OrderInvariantNondeterministic`] because float sums do not
//! associate). There is no NaN and no signed zero, so no ULP class ever arises.
//!
//! **Scope (this cut).** The value-lane int-closed nodes: the leaves
//! (`Bind`/`const`/`const_bits`/`runtime_scalar`/`reduced_count`), elementwise
//! `Apply` (via [`crate::tensor_int::element_map`]), `Reduce`, `PrefixScan`,
//! `Matmul`, `Iota`, `Flip`. The **index-lane** nodes (`gather`/`scatter`/
//! `sort_network`) and any `index_outputs` request are a typed decline for now —
//! their index-resolution machinery is a focused additive follow-up — so a DAG
//! that uses them gets an [`Error`], never a panic.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::Expr;
use kiss_ops_vocab::Op;

use crate::bridge::DetClass;
use crate::recipe::{children_of, rank0, read_children, squeeze, FlatDag, Node, RecipeEval};
use crate::scalar_int::wrap_to_dtype;
use crate::tensor::{alloc_exact, broadcast_shapes, numel, IndexTensor, Tensor, View};
use crate::tensor_int;
use crate::tensor_ops::flip;
use crate::Error;

/// Evaluate a recipe DAG on the **integer lane**. `node_dtypes[i]` is the output
/// dtype of node `i` (parallel to `dag.nodes`); `inputs`/`params` are `i128`
/// (`Bind(i) → inputs[i]`, `runtime_scalar(s) → params[s]`), the true
/// sign-extended integer values, so a value computed at one dtype feeds a node at
/// a wider dtype losslessly. `indices` is reserved for the index-lane follow-up.
/// Returns a [`RecipeEval`] whose `dets` are all [`DetClass::ExactByte`] and whose
/// `index_outputs` is empty. **Never panics** — every failure is an [`Error`].
pub fn eval_recipe_int(
    dag: &FlatDag,
    node_dtypes: &[Dtype],
    inputs: &[Tensor<i128>],
    params: &[i128],
    indices: &[IndexTensor],
) -> Result<RecipeEval<i128>, Error> {
    let _ = indices; // reserved for the index-lane follow-up.
    let n = dag.nodes.len();
    // The per-node output dtype must be 1:1 with the nodes.
    if node_dtypes.len() != n {
        return Err(Error::LengthMismatch {
            expected: n,
            got: node_dtypes.len(),
        });
    }
    // No integer node produces an index lane in this cut, so any `index_outputs`
    // entry is unsatisfiable — a typed decline, not a silently-dropped request.
    if let Some(&r) = dag.index_outputs.first() {
        return Err(Error::IndexSourceInvalid { node: r });
    }
    let mut memo: Vec<Option<(Tensor<i128>, DetClass)>> = (0..n).map(|_| None).collect();
    // Iterative post-order over an explicit HEAP worklist (never the call stack) —
    // identical in shape to `eval_recipe`, so a deep dependency chain returns an
    // Error rather than overflowing the stack. `state`: 0 unvisited, 1 active (a
    // revisit on the current path is a cycle), 2 done.
    let mut state: Vec<u8> = (0..n).map(|_| 0u8).collect();
    let mut stack: Vec<(usize, bool)> = Vec::new();
    for start in 0..n {
        if state[start] == 2 {
            continue;
        }
        stack.push((start, false));
        while let Some((idx, expanded)) = stack.pop() {
            if idx >= n {
                return Err(Error::MissingInput(0));
            }
            if state[idx] == 2 {
                continue;
            }
            if expanded {
                let (t, d) = compute_node_int(dag, idx, node_dtypes, inputs, params, &memo)?;
                memo[idx] = Some((t, d));
                state[idx] = 2;
            } else {
                if state[idx] == 1 {
                    // a cycle in a supposedly-acyclic DAG — decline, don't loop.
                    return Err(Error::BadDecomposition {
                        op: Op::Add,
                        pos: idx,
                    });
                }
                state[idx] = 1;
                stack.push((idx, true)); // revisit after its children
                for c in children_of(&dag.nodes[idx]) {
                    if c >= n {
                        return Err(Error::MissingInput(0));
                    }
                    if state[c] != 2 {
                        stack.push((c, false));
                    }
                }
            }
        }
    }
    let mut outputs = Vec::with_capacity(dag.outputs.len());
    for &r in &dag.outputs {
        let (t, _) = memo
            .get(r)
            .and_then(|x| x.as_ref())
            .ok_or(Error::MissingInput(0))?;
        outputs.push(t.clone());
    }
    let dets = memo
        .iter()
        .map(|x| x.as_ref().map(|(_, d)| *d).unwrap_or(DetClass::ExactByte))
        .collect();
    Ok(RecipeEval {
        outputs,
        index_outputs: Vec::new(),
        dets,
    })
}

/// Compute node `idx` on the integer lane from its already-memoized children.
/// Returns the `i128` value-lane result and its [`DetClass`] (always
/// [`DetClass::ExactByte`] in this lane). The index-lane nodes decline.
fn compute_node_int(
    dag: &FlatDag,
    idx: usize,
    node_dtypes: &[Dtype],
    inputs: &[Tensor<i128>],
    params: &[i128],
    memo: &[Option<(Tensor<i128>, DetClass)>],
) -> Result<(Tensor<i128>, DetClass), Error> {
    let dtype = node_dtypes[idx];
    match &dag.nodes[idx] {
        Node::Bind(i) => {
            let t = inputs
                .get(*i)
                .cloned()
                .ok_or(Error::MissingInput(*i as u8))?;
            Ok((t, DetClass::ExactByte))
        }
        Node::Const(v) => {
            // Round-ties-even into the integer lane — the integer analog of the
            // float lane's `from_f64` re-round. A converter needing a bit-exact
            // integer const emits [`Node::ConstBits`] instead.
            let r = libm::rint(*v) as i128;
            Ok((rank0(wrap_to_dtype(r, dtype)?)?, DetClass::ExactByte))
        }
        Node::ConstBits(bits) => Ok((
            rank0(wrap_to_dtype(*bits as i128, dtype)?)?,
            DetClass::ExactByte,
        )),
        Node::RuntimeScalar(s) => {
            let v = params
                .get(*s)
                .copied()
                .ok_or(Error::MissingInput(*s as u8))?;
            Ok((rank0(v)?, DetClass::ExactByte))
        }
        Node::ReducedCount(axes) => {
            let x0 = inputs.first().ok_or(Error::MissingInput(0))?;
            let shape = x0.shape();
            let mut c: usize = 1;
            for &a in axes {
                c = c
                    .checked_mul(*shape.get(a).ok_or(Error::AxisOutOfRange {
                        axis: a,
                        rank: shape.len(),
                    })?)
                    .ok_or(Error::ShapeOverflow)?;
            }
            Ok((
                rank0(wrap_to_dtype(c as i128, dtype)?)?,
                DetClass::ExactByte,
            ))
        }
        Node::Apply { op, children } => {
            let (ts, _) = read_children(children, memo)?;
            let views: Vec<View<i128>> = ts.iter().map(|t| t.view()).collect();
            let shapes: Vec<&[usize]> = views.iter().map(|v| v.shape()).collect();
            let (out_shape, r) = broadcast_shapes(&shapes)?;
            // An elementwise `Apply` is the body `op(input0, input1, …)`; reuse the
            // tested integer `element_map` by expressing it as that decomposition.
            let mut args = Vec::with_capacity(children.len());
            for k in 0..children.len() {
                args.push(Expr::Input(k as u8));
            }
            let body = Expr::Apply(*op, args);
            let out = tensor_int::element_map(&body, &views, dtype, &out_shape[..r])?;
            Ok((out, DetClass::ExactByte))
        }
        Node::Reduce {
            monoid,
            axes,
            keepdim,
            child,
        } => {
            let (ts, _) = read_children(core::slice::from_ref(child), memo)?;
            let out = tensor_int::reduce(&ts[0].view(), dtype, *monoid, axes)?;
            let out = if *keepdim { out } else { squeeze(out, axes)? };
            Ok((out, DetClass::ExactByte))
        }
        Node::PrefixScan {
            monoid,
            axis,
            exclusive,
            child,
        } => {
            let (ts, _) = read_children(core::slice::from_ref(child), memo)?;
            let out = tensor_int::prefix_scan(&ts[0].view(), dtype, *monoid, *axis, *exclusive)?;
            Ok((out, DetClass::ExactByte))
        }
        Node::Matmul { lhs, rhs } => {
            let (ts, _) = read_children(&[*lhs, *rhs], memo)?;
            // The key contrast with the float lane: an integer contraction is
            // order-independent AND exact, so `ExactByte`, not the float lane's
            // `OrderInvariantNondeterministic`.
            let out = tensor_int::matmul(&ts[0].view(), &ts[1].view(), dtype)?;
            Ok((out, DetClass::ExactByte))
        }
        Node::Iota { like, axis } => {
            // shape-only edge: read the child's SHAPE, never its values.
            let (lt, _) = memo
                .get(*like)
                .and_then(|x| x.as_ref())
                .ok_or(Error::MissingInput(0))?;
            let shape = lt.shape();
            let rank = shape.len();
            if *axis >= rank {
                return Err(Error::AxisOutOfRange { axis: *axis, rank });
            }
            let count = numel(shape)?;
            let mut stride = 1usize;
            for &d in &shape[*axis + 1..] {
                stride = stride.checked_mul(d).ok_or(Error::ShapeOverflow)?;
            }
            let extent = shape[*axis];
            let mut buf: Vec<i128> = alloc_exact(count)?;
            for i in 0..count {
                buf.push(wrap_to_dtype(((i / stride) % extent) as i128, dtype)?);
            }
            Ok((Tensor::from_vec(buf, shape)?, DetClass::ExactByte))
        }
        Node::Flip { child, axis } => {
            let (ts, _) = read_children(core::slice::from_ref(child), memo)?;
            let out = flip(&ts[0].view(), *axis)?; // pure data movement (shared kernel)
            Ok((out, DetClass::ExactByte))
        }
        // ---- index-lane nodes: deferred additive follow-up (typed decline) -----
        Node::Gather { .. } => Err(Error::Unsupported(Op::Gather)),
        Node::Scatter { .. } => Err(Error::Unsupported(Op::Scatter)),
        Node::SortNetwork { .. } => Err(Error::Unsupported(Op::SortNetwork)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{FlatDag, IndexRef, Node};
    use alloc::vec;

    fn t(data: &[i128], shape: &[usize]) -> Tensor<i128> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    fn ev(
        nodes: Vec<Node>,
        outputs: Vec<usize>,
        dtypes: &[Dtype],
        inputs: &[Tensor<i128>],
    ) -> Result<RecipeEval<i128>, Error> {
        let dag = FlatDag::new(nodes, outputs);
        eval_recipe_int(&dag, dtypes, inputs, &[], &[])
    }

    #[test]
    fn bind_passthrough_is_exact() {
        let r = ev(
            vec![Node::Bind(0)],
            vec![0],
            &[Dtype::S8],
            &[t(&[1, 2, 3], &[3])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[1, 2, 3]);
        assert_eq!(r.dets[0], DetClass::ExactByte);
        assert!(r.index_outputs.is_empty());
    }

    #[test]
    fn apply_add_wraps_at_s8() {
        // 100 + 100 = 200, which wraps to -56 in s8 (two's complement).
        let r = ev(
            vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Apply {
                    op: Op::Add,
                    children: vec![0, 1],
                },
            ],
            vec![2],
            &[Dtype::S8, Dtype::S8, Dtype::S8],
            &[t(&[100], &[1]), t(&[100], &[1])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[-56]);
        assert_eq!(r.dets[2], DetClass::ExactByte);
    }

    #[test]
    fn reduce_sum_counts_predicate_into_i64() {
        // The heterogeneous-dtype case Baracuda flagged: a `b1` comparator mask
        // reduced by `sum` counts into `i64`. data > 3 → [0,1,0,1]; sum = 2.
        let r = ev(
            vec![
                Node::Bind(0), // data  (s8)
                Node::Bind(1), // thresh (s8), broadcast
                Node::Apply {
                    // mask (b1)
                    op: Op::CmpGt,
                    children: vec![0, 1],
                },
                Node::Reduce {
                    // count (i64)
                    monoid: crate::attrs::Monoid::Sum,
                    axes: vec![0],
                    keepdim: false,
                    child: 2,
                },
            ],
            vec![3],
            &[Dtype::S8, Dtype::S8, Dtype::B1, Dtype::I64],
            &[t(&[1, 5, 2, 8], &[4]), t(&[3], &[1])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[2]);
        assert_eq!(r.outputs[0].shape(), &[] as &[usize]);
        assert_eq!(r.dets[2], DetClass::ExactByte); // comparator
        assert_eq!(r.dets[3], DetClass::ExactByte); // reduce/count
    }

    #[test]
    fn reduce_max_is_exact() {
        let r = ev(
            vec![
                Node::Bind(0),
                Node::Reduce {
                    monoid: crate::attrs::Monoid::Max,
                    axes: vec![0],
                    keepdim: false,
                    child: 0,
                },
            ],
            vec![1],
            &[Dtype::S16, Dtype::S16],
            &[t(&[3, -7, 9, 2], &[4])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[9]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
    }

    #[test]
    fn matmul_is_exact_byte_not_nondeterministic() {
        // The key contrast with the float lane: an integer contraction is
        // order-independent AND exact, so ExactByte — NOT the float lane's
        // OrderInvariantNondeterministic.
        let r = ev(
            vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Matmul { lhs: 0, rhs: 1 },
            ],
            vec![2],
            &[Dtype::I32, Dtype::I32, Dtype::I32],
            // [1 2; 3 4] · [5 6; 7 8] = [19 22; 43 50]
            &[t(&[1, 2, 3, 4], &[2, 2]), t(&[5, 6, 7, 8], &[2, 2])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[19, 22, 43, 50]);
        assert_eq!(r.dets[2], DetClass::ExactByte);
    }

    #[test]
    fn prefix_scan_cumsum_wraps_at_u8() {
        // running sum [200,100,50] over u8: 200, (200+100)=300→44, (44+50)=94.
        let r = ev(
            vec![
                Node::Bind(0),
                Node::PrefixScan {
                    monoid: crate::attrs::Monoid::Sum,
                    axis: 0,
                    exclusive: false,
                    child: 0,
                },
            ],
            vec![1],
            &[Dtype::U8, Dtype::U8],
            &[t(&[200, 100, 50], &[3])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[200, 44, 94]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
    }

    #[test]
    fn const_bits_reinterprets_at_dtype() {
        // 0xFF as s8 = -1 (sign bit set); as u8 = 255.
        let rs = ev(vec![Node::ConstBits(0xFF)], vec![0], &[Dtype::S8], &[]).unwrap();
        assert_eq!(rs.outputs[0].as_slice(), &[-1]);
        let ru = ev(vec![Node::ConstBits(0xFF)], vec![0], &[Dtype::U8], &[]).unwrap();
        assert_eq!(ru.outputs[0].as_slice(), &[255]);
    }

    #[test]
    fn const_real_rounds_into_lane() {
        // Const(2.6) rounds (ties-to-even) into the integer lane → 3.
        let r = ev(vec![Node::Const(2.6)], vec![0], &[Dtype::I32], &[]).unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[3]);
    }

    #[test]
    fn iota_coordinates_along_axis() {
        // coord(axis=1) of a [2,3] shape → rows [0,1,2],[0,1,2].
        let r = ev(
            vec![Node::Bind(0), Node::Iota { like: 0, axis: 1 }],
            vec![1],
            &[Dtype::I32, Dtype::I32],
            &[t(&[0, 0, 0, 0, 0, 0], &[2, 3])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[0, 1, 2, 0, 1, 2]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
    }

    #[test]
    fn flip_reverses_along_axis() {
        let r = ev(
            vec![Node::Bind(0), Node::Flip { child: 0, axis: 0 }],
            vec![1],
            &[Dtype::S8, Dtype::S8],
            &[t(&[1, 2, 3, 4], &[4])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[4, 3, 2, 1]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
    }

    #[test]
    fn rejects_node_dtype_length_mismatch() {
        // node_dtypes shorter than nodes → typed decline, never a panic.
        let dag = FlatDag::new(vec![Node::Bind(0), Node::Bind(0)], vec![0]);
        let err = eval_recipe_int(&dag, &[Dtype::S8], &[t(&[1], &[1])], &[], &[]).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }));
    }

    #[test]
    fn declines_index_lane_node_without_panic() {
        // sort_network is an index-lane node — deferred, so a typed decline.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::SortNetwork {
                    keys: 0,
                    axis: 0,
                    dir: crate::attrs::Direction::Asc,
                },
            ],
            vec![1],
        );
        let err = eval_recipe_int(
            &dag,
            &[Dtype::S8, Dtype::S8],
            &[t(&[3, 1, 2], &[3])],
            &[],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(Op::SortNetwork)));
    }

    #[test]
    fn declines_index_outputs_request() {
        // No int node produces an index lane in this cut → any index_outputs
        // entry is a typed decline.
        let mut dag = FlatDag::new(vec![Node::Bind(0)], vec![0]);
        dag.index_outputs = vec![0];
        let err = eval_recipe_int(&dag, &[Dtype::S8], &[t(&[1], &[1])], &[], &[]).unwrap_err();
        assert!(matches!(err, Error::IndexSourceInvalid { .. }));
    }

    #[test]
    fn declines_gather_with_node_index_without_panic() {
        // A gather whose index is a Node ref (index lane) is deferred; the walk
        // must decline, never panic, even though the index producer is present.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::SortNetwork {
                    keys: 0,
                    axis: 0,
                    dir: crate::attrs::Direction::Asc,
                },
                Node::Gather {
                    data: 0,
                    index: IndexRef::Node(1),
                    axis: 0,
                    oob: crate::attrs::OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![2],
        );
        let err = eval_recipe_int(
            &dag,
            &[Dtype::S8, Dtype::S8, Dtype::S8],
            &[t(&[3, 1, 2], &[3])],
            &[],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }
}
