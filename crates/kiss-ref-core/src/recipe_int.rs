// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! **Scope.** Every recipe node: the leaves
//! (`Bind`/`const`/`const_bits`/`runtime_scalar`/`reduced_count`), elementwise
//! `Apply` (via [`crate::tensor_int::element_map`]), `Reduce`, `PrefixScan`,
//! `Matmul`, `Iota`, `Flip`, and the **index-lane** nodes `gather`/`scatter`/
//! `sort_network` — with `IndexRef::Slot`/`Node` resolution and `index_outputs`
//! export, sharing the float lane's `resolve_index_ref` / `has_index_lane`
//! machinery. One index-lane distinction is load-bearing: integer `scatter` with
//! `atomic_add` folds colliding sources over a ring (associative AND commutative),
//! so it stays [`DetClass::ExactByte`] where the float lane must escalate to
//! [`DetClass::OrderInvariantNondeterministic`].

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::Expr;
use kiss_ops_vocab::Op;

use crate::bridge::DetClass;
use crate::recipe::{
    children_of, has_index_lane, rank0, read_children, resolve_index_ref, squeeze, FlatDag,
    IndexRef, Node, RecipeEval,
};
use crate::scalar_int::wrap_to_dtype;
use crate::tensor::{alloc_exact, broadcast_shapes, numel, IndexTensor, Tensor, View};
use crate::tensor_int;
use crate::tensor_ops::flip;
use crate::Error;

/// Evaluate a recipe DAG on the **integer lane**. `node_dtypes[i]` is the output
/// dtype of node `i` (parallel to `dag.nodes`); `inputs`/`params` are `i128`
/// (`Bind(i) → inputs[i]`, `runtime_scalar(s) → params[s]`), the true
/// sign-extended integer values, so a value computed at one dtype feeds a node at
/// a wider dtype losslessly. `indices` are the external integer index operands of
/// `gather`/`scatter` (addressed by [`IndexRef::Slot`]); `sort_network` populates
/// the node-addressed index lane ([`IndexRef::Node`]) and `dag.index_outputs`.
/// Returns a [`RecipeEval`] whose `dets` are all [`DetClass::ExactByte`].
/// **Never panics** — every failure is an [`Error`].
pub fn eval_recipe_int(
    dag: &FlatDag,
    node_dtypes: &[Dtype],
    inputs: &[Tensor<i128>],
    params: &[i128],
    indices: &[IndexTensor],
) -> Result<RecipeEval<i128>, Error> {
    let n = dag.nodes.len();
    // The per-node output dtype must be 1:1 with the nodes.
    if node_dtypes.len() != n {
        return Err(Error::LengthMismatch {
            expected: n,
            got: node_dtypes.len(),
        });
    }
    // Pre-validate every index-lane reference against the DAG's static shape, so a
    // bad reference is the dedicated typed error, not a generic decline (mirrors
    // the float lane). `SortNetwork` is the only index producer.
    for node in &dag.nodes {
        let r = match node {
            Node::Gather { index, .. } | Node::Scatter { index, .. } => *index,
            _ => continue,
        };
        if let IndexRef::Node(m) = r {
            if m >= n || !has_index_lane(&dag.nodes[m]) {
                return Err(Error::IndexSourceInvalid { node: m });
            }
        }
    }
    for &r in &dag.index_outputs {
        if r >= n || !has_index_lane(&dag.nodes[r]) {
            return Err(Error::IndexSourceInvalid { node: r });
        }
    }
    let mut memo: Vec<Option<(Tensor<i128>, DetClass)>> = (0..n).map(|_| None).collect();
    let mut imemo: Vec<Option<IndexTensor>> = (0..n).map(|_| None).collect();
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
                let (t, d, ix) = compute_node_int(
                    dag,
                    idx,
                    node_dtypes,
                    inputs,
                    params,
                    indices,
                    &memo,
                    &imemo,
                )?;
                memo[idx] = Some((t, d));
                imemo[idx] = ix;
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
    let mut index_outputs = Vec::with_capacity(dag.index_outputs.len());
    for &r in &dag.index_outputs {
        let ix = imemo
            .get(r)
            .and_then(|x| x.as_ref())
            .ok_or(Error::IndexSourceInvalid { node: r })?;
        index_outputs.push(ix.clone());
    }
    let dets = memo
        .iter()
        .map(|x| x.as_ref().map(|(_, d)| *d).unwrap_or(DetClass::ExactByte))
        .collect();
    Ok(RecipeEval {
        outputs,
        index_outputs,
        dets,
    })
}

/// Compute node `idx` on the integer lane from its already-memoized children.
/// Returns the `i128` value-lane result, its [`DetClass`] (always
/// [`DetClass::ExactByte`] in this lane), and the optional index-lane result
/// (`Some` only for `sort_network`).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compute_node_int(
    dag: &FlatDag,
    idx: usize,
    node_dtypes: &[Dtype],
    inputs: &[Tensor<i128>],
    params: &[i128],
    indices: &[IndexTensor],
    memo: &[Option<(Tensor<i128>, DetClass)>],
    imemo: &[Option<IndexTensor>],
) -> Result<(Tensor<i128>, DetClass, Option<IndexTensor>), Error> {
    let dtype = node_dtypes[idx];
    match &dag.nodes[idx] {
        Node::Bind(i) => {
            let raw = inputs.get(*i).ok_or(Error::MissingInput(*i as u8))?;
            // A leaf still honors its declared output dtype: wrap each element to
            // node_dtypes[idx] so the "output lies in this node's dtype range"
            // invariant holds for EVERY node, not only computed ones. Idempotent
            // for a valid in-range input (see `bind_passthrough_is_exact`); it
            // sanitizes an input a caller failed to pre-normalize rather than
            // leaking an out-of-range value downstream.
            let mut data = alloc_exact(raw.as_slice().len())?;
            for &v in raw.as_slice() {
                data.push(wrap_to_dtype(v, dtype)?);
            }
            Ok((
                Tensor::from_vec(data, raw.shape())?,
                DetClass::ExactByte,
                None,
            ))
        }
        Node::Const(v) => {
            // Round-ties-even into the integer lane — the integer analog of the
            // float lane's `from_f64` re-round. A converter needing a bit-exact
            // integer const emits [`Node::ConstBits`] instead.
            let r = libm::rint(*v) as i128;
            Ok((rank0(wrap_to_dtype(r, dtype)?)?, DetClass::ExactByte, None))
        }
        Node::ConstBits(bits) => Ok((
            rank0(wrap_to_dtype(*bits as i128, dtype)?)?,
            DetClass::ExactByte,
            None,
        )),
        Node::RuntimeScalar(s) => {
            let v = params
                .get(*s)
                .copied()
                .ok_or(Error::MissingInput(*s as u8))?;
            // Interpret the scalar AS the node's declared dtype (same leaf
            // contract as Bind): wrap before lifting to a rank-0 tensor.
            Ok((rank0(wrap_to_dtype(v, dtype)?)?, DetClass::ExactByte, None))
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
                None,
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
            Ok((out, DetClass::ExactByte, None))
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
            Ok((out, DetClass::ExactByte, None))
        }
        Node::PrefixScan {
            monoid,
            axis,
            exclusive,
            child,
        } => {
            let (ts, _) = read_children(core::slice::from_ref(child), memo)?;
            let out = tensor_int::prefix_scan(&ts[0].view(), dtype, *monoid, *axis, *exclusive)?;
            Ok((out, DetClass::ExactByte, None))
        }
        Node::Matmul { lhs, rhs } => {
            let (ts, _) = read_children(&[*lhs, *rhs], memo)?;
            // The key contrast with the float lane: an integer contraction is
            // order-independent AND exact, so `ExactByte`, not the float lane's
            // `OrderInvariantNondeterministic`.
            let out = tensor_int::matmul(&ts[0].view(), &ts[1].view(), dtype)?;
            Ok((out, DetClass::ExactByte, None))
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
            Ok((Tensor::from_vec(buf, shape)?, DetClass::ExactByte, None))
        }
        Node::Flip { child, axis } => {
            let (ts, _) = read_children(core::slice::from_ref(child), memo)?;
            let out = flip(&ts[0].view(), *axis)?; // pure data movement (shared kernel)
            Ok((out, DetClass::ExactByte, None))
        }
        // ---- index lane: shares the float lane's IndexRef resolution -----------
        Node::Gather {
            data,
            index,
            axis,
            oob,
            base,
        } => {
            let (ts, _) = read_children(core::slice::from_ref(data), memo)?;
            let ixt = resolve_index_ref(*index, indices, imemo)?;
            let basem = match base {
                Some(b) => Some(
                    memo.get(*b)
                        .and_then(|x| x.as_ref())
                        .ok_or(Error::MissingInput(0))?,
                ),
                None => None,
            };
            let bview = basem.map(|(t, _)| t.view());
            let out = tensor_int::gather(&ts[0].view(), ixt, *axis, *oob, bview.as_ref())?;
            // A raw i128 move — no NaN, no signed zero, no reassociation — so
            // ExactByte on every oob arm (including the base-carrying `Skip`).
            Ok((out, DetClass::ExactByte, None))
        }
        Node::Scatter {
            dest,
            index,
            updates,
            axis,
            combine,
        } => {
            let (ts, _) = read_children(&[*dest, *updates], memo)?;
            let ixt = resolve_index_ref(*index, indices, imemo)?;
            let out =
                tensor_int::scatter(ts[0].clone(), dtype, ixt, &ts[1].view(), *axis, *combine)?;
            // Integer atomic_add folds colliding sources over a ring (associative
            // AND commutative) → order-independent AND exact, so ExactByte — NOT
            // the float lane's OrderInvariantNondeterministic.
            Ok((out, DetClass::ExactByte, None))
        }
        Node::SortNetwork { keys, axis, dir } => {
            let (ts, _) = read_children(core::slice::from_ref(keys), memo)?;
            let (vals, perm) = tensor_int::sort_network(&ts[0].view(), *axis, *dir)?;
            // Total order over integers is exact; the one class covers both the
            // value lane and the exported permutation.
            Ok((vals, DetClass::ExactByte, Some(perm)))
        }
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
            &[Dtype::I8],
            &[t(&[1, 2, 3], &[3])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[1, 2, 3]);
        assert_eq!(r.dets[0], DetClass::ExactByte);
        assert!(r.index_outputs.is_empty());
    }

    #[test]
    fn bind_wraps_out_of_range_input_to_dtype() {
        // Every node's output must lie in node_dtypes[idx]'s range — a leaf is no
        // exception. An input a caller failed to pre-normalize is wrapped to the
        // declared dtype (idempotent for an in-range value, as bind_passthrough
        // shows). At i8: 200 -> -56, -129 -> 127.
        let r = ev(
            vec![Node::Bind(0)],
            vec![0],
            &[Dtype::I8],
            &[t(&[200, -129], &[2])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[-56, 127]);
        assert_eq!(r.dets[0], DetClass::ExactByte);
    }

    #[test]
    fn runtime_scalar_wraps_to_dtype() {
        // Same contract for a runtime scalar: params[slot] is interpreted AS the
        // node's declared dtype, so 200 -> -56 at i8.
        let dag = FlatDag::new(vec![Node::RuntimeScalar(0)], vec![0]);
        let r = eval_recipe_int(&dag, &[Dtype::I8], &[], &[200], &[]).unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[-56]);
    }

    #[test]
    fn apply_add_wraps_at_s8() {
        // 100 + 100 = 200, which wraps to -56 in i8 (two's complement).
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
            &[Dtype::I8, Dtype::I8, Dtype::I8],
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
                Node::Bind(0), // data  (i8)
                Node::Bind(1), // thresh (i8), broadcast
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
            &[Dtype::I8, Dtype::I8, Dtype::B1, Dtype::I64],
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
            &[Dtype::I16, Dtype::I16],
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
        // 0xFF as i8 = -1 (sign bit set); as u8 = 255.
        let rs = ev(vec![Node::ConstBits(0xFF)], vec![0], &[Dtype::I8], &[]).unwrap();
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
            &[Dtype::I8, Dtype::I8],
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
        let err = eval_recipe_int(&dag, &[Dtype::I8], &[t(&[1], &[1])], &[], &[]).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }));
    }

    #[test]
    fn declines_index_outputs_request() {
        // index_outputs must name an index producer (SortNetwork); a Bind has no
        // index lane → IndexSourceInvalid, never a silently-dropped request.
        let mut dag = FlatDag::new(vec![Node::Bind(0)], vec![0]);
        dag.index_outputs = vec![0];
        let err = eval_recipe_int(&dag, &[Dtype::I8], &[t(&[1], &[1])], &[], &[]).unwrap_err();
        assert!(matches!(err, Error::IndexSourceInvalid { .. }));
    }

    #[test]
    fn gather_node_index_to_non_index_node_declines() {
        // A gather whose IndexRef::Node points at a NON-index node (a Bind, which
        // has no index lane) is caught by pre-validation with the dedicated
        // IndexSourceInvalid — never a panic, never a wrong gather.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0), // data — NOT an index producer
                Node::Gather {
                    data: 0,
                    index: IndexRef::Node(0),
                    axis: 0,
                    oob: crate::attrs::OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![1],
        );
        let err = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8],
            &[t(&[3, 1, 2], &[3])],
            &[],
            &[],
        )
        .unwrap_err();
        assert!(matches!(err, Error::IndexSourceInvalid { .. }));
    }

    // ---- index lane (gather / scatter / sort_network) -----------------------

    fn ix(data: &[i64], shape: &[usize]) -> IndexTensor {
        IndexTensor::new(data.to_vec(), shape, Dtype::I64).unwrap()
    }

    #[test]
    fn gather_slot_index_moves_values() {
        // gather via IndexRef::Slot is a raw i128 move → ExactByte. data
        // [10,20,30,40] gathered by [3,1,0] → [40,20,10].
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(0),
                    axis: 0,
                    oob: crate::attrs::OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![1],
        );
        let r = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8],
            &[t(&[10, 20, 30, 40], &[4])],
            &[],
            &[ix(&[3, 1, 0], &[3])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[40, 20, 10]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
        assert!(r.index_outputs.is_empty());
    }

    #[test]
    fn scatter_add_is_exact_byte_not_nondeterministic() {
        // THE contrast with the float lane: integer atomic_add folds colliding
        // sources associatively AND commutatively (a ring), so the scatter is
        // ExactByte — NOT the float lane's OrderInvariantNondeterministic.
        // dest zeros[4], updates [1,2,3,4,5] at idx [0,2,0,3,2]:
        // bin0=1+3=4, bin1=0, bin2=2+5=7, bin3=4 → [4,0,7,4].
        let dag = FlatDag::new(
            vec![
                Node::Bind(0), // dest
                Node::Bind(1), // updates
                Node::Scatter {
                    dest: 0,
                    index: IndexRef::Slot(0),
                    updates: 1,
                    axis: 0,
                    combine: crate::attrs::Combine::AtomicAdd,
                },
            ],
            vec![2],
        );
        let r = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8, Dtype::I8],
            &[t(&[0, 0, 0, 0], &[4]), t(&[1, 2, 3, 4, 5], &[5])],
            &[],
            &[ix(&[0, 2, 0, 3, 2], &[5])],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[4, 0, 7, 4]);
        assert_eq!(r.dets[2], DetClass::ExactByte);
    }

    #[test]
    fn sort_network_sorts_and_exports_permutation() {
        // sort_network exports BOTH lanes: sorted values (value lane) and the
        // original-index permutation (index lane via index_outputs). keys
        // [3,1,2] asc → values [1,2,3], perm [1,2,0].
        let dag = FlatDag {
            nodes: vec![
                Node::Bind(0),
                Node::SortNetwork {
                    keys: 0,
                    axis: 0,
                    dir: crate::attrs::Direction::Asc,
                },
            ],
            outputs: vec![1],
            index_outputs: vec![1],
        };
        let r = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8],
            &[t(&[3, 1, 2], &[3])],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[1, 2, 3]);
        assert_eq!(r.index_outputs.len(), 1);
        assert_eq!(r.index_outputs[0].as_slice(), &[1, 2, 0]);
        assert_eq!(r.dets[1], DetClass::ExactByte);
    }

    #[test]
    fn gather_via_node_index_end_to_end() {
        // The IndexRef::Node scheduling edge: sort keys → permutation → gather
        // data by that permutation (an argsort reorder), ExactByte end to end.
        // keys [4,1,3,2] asc → perm [1,3,2,0]; gather data [40,10,30,20] →
        // [10,20,30,40].
        let dag = FlatDag::new(
            vec![
                Node::Bind(0), // keys
                Node::Bind(1), // data
                Node::SortNetwork {
                    keys: 0,
                    axis: 0,
                    dir: crate::attrs::Direction::Asc,
                },
                Node::Gather {
                    data: 1,
                    index: IndexRef::Node(2),
                    axis: 0,
                    oob: crate::attrs::OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![3],
        );
        let r = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8, Dtype::I8, Dtype::I8],
            &[t(&[4, 1, 3, 2], &[4]), t(&[40, 10, 30, 20], &[4])],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(r.outputs[0].as_slice(), &[10, 20, 30, 40]);
        assert_eq!(r.dets[3], DetClass::ExactByte);
    }

    #[test]
    fn gather_missing_index_slot_reports_full_usize_slot() {
        // A missing external index operand reports the FULL slot as usize, not a
        // u8-truncated MissingInput — slot 300 must not read back as 44 (300-256).
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(300),
                    axis: 0,
                    oob: crate::attrs::OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![1],
        );
        let err = eval_recipe_int(
            &dag,
            &[Dtype::I8, Dtype::I8],
            &[t(&[1, 2, 3], &[3])],
            &[],
            &[],
        )
        .unwrap_err();
        assert_eq!(err, Error::MissingIndexOperand { slot: 300 });
    }
}
