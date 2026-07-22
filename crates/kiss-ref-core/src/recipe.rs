//! The **recipe evaluator**: `eval_recipe(dag, inputs, params, indices) ->
//! (outputs, per-node DetClass)`.
//!
//! kiss-ref evaluates the **logical flat-DAG** (a decoded Rust structure, [`FlatDag`]),
//! not wire bytes — a decoder produces the DAG the evaluator walks (the §6.4-0009/
//! §6.19 wire encoding is pinned separately by the KISS 3B/Appendix-F work). The
//! grammar is the **KISS-owned** recipe grammar (Contract §2.3 / Ops §6.13 / §6.19 /
//! §6.4-0009); this evaluator is scoped against the consolidated extract while that
//! grammar firms, and kiss-ref is a co-owner/validator of it.
//!
//! Each node is evaluated to a [`Tensor<T>`] by dispatching to the tensor kernels;
//! `Bind(i)` resolves to `inputs[i]`, a fold node's output is read through the
//! ordinary child edge (the `Reduced(i)` reference). Per-node [`DetClass`] is
//! computed from `(op, attrs)` joined most-permissively with the inputs' classes
//! (§6.0-0005) for the consumer's comparator.
//!
//! **Scope (float lane):** the scalar atoms, `reduce`/`prefix_scan`, **batched**
//! `matmul`, the value leaves (`Bind`/`const`/`runtime_scalar`/`reduced_count`), and
//! the index-bearing nodes `gather`/`scatter`/`sort_network` — whose integer index
//! operands ride in the separate `indices` input array (referenced by slot), keeping
//! the value-DAG's node results homogeneously `Tensor<T>`. **Deferred:** the
//! `gather`-`skip` base operand (§6.11 gather-skip RFC), `sort_network`'s
//! original-index output (needs an integer node output), and `iota` (needs the §6.20
//! shape oracle).

extern crate alloc;
use alloc::vec::Vec;

use kiss_ops_vocab::Op;

use crate::attrs::{Combine, Direction, Monoid, OobPolicy};
use crate::bridge::{monoid_det, DetClass};
use crate::kernels::{gather, map_views, prefix_scan, reduce, scatter, sort_network};
use crate::resolve::eval_op;
use crate::scalar::ScalarFloat;
use crate::tensor::{broadcast_shapes, IndexTensor, Tensor, View, MAX_RANK};
use crate::tensor_ops::matmul;
use crate::Error;

/// A node in the canonical flat-DAG (the logical, decoded form).
#[derive(Clone, Debug)]
pub enum Node {
    /// Bind the fused op's `inputs[i]` (§6.4-0009 `Bind`).
    Bind(usize),
    /// A `const(bits)` leaf (the reference holds the value as `f64`).
    Const(f64),
    /// A `runtime_scalar(slot)` leaf → `params[slot]`.
    RuntimeScalar(usize),
    /// A `reduced_count(axes)` value leaf — ∏ of the reduced extents (resolved
    /// against `inputs[0]`; the leaf-vs-shape modeling is a routed grammar question).
    ReducedCount(Vec<usize>),
    /// An elementwise scalar op over child nodes (§6.12 body atom): `add`/`sub`/
    /// `mul`/`div`, the unary/binary math atoms, `select`.
    Apply { op: Op, children: Vec<usize> },
    /// A `reduce` fold node.
    Reduce { monoid: Monoid, axes: Vec<usize>, keepdim: bool, child: usize },
    /// A `prefix_scan` fold node.
    PrefixScan { monoid: Monoid, axis: usize, exclusive: bool, child: usize },
    /// A `matmul` contraction node (batched `[..b,M,K]·[..b,K,N]`).
    Matmul { lhs: usize, rhs: usize },
    /// A `gather` node: read `data` (a node) at a runtime `index` along `axis`.
    /// `index` is a **slot into the external `indices` inputs** (integer data rides
    /// separately from the float value-DAG). v1 has no `base` operand — an OOB
    /// `skip` errors; the base child edge is the §6.11 gather-skip RFC extension.
    Gather { data: usize, index: usize, axis: usize, oob: OobPolicy },
    /// A `scatter` node: write `updates` (a node) into `dest` (a node) at a runtime
    /// `index` (an `indices` slot) along `axis`, combined per `combine`.
    Scatter { dest: usize, index: usize, updates: usize, axis: usize, combine: Combine },
    /// A `sort_network` node producing the sorted **values** along `axis`. (The
    /// original-index output is a follow-up — it needs an integer node output.)
    SortNetwork { keys: usize, axis: usize, dir: Direction },
}

/// A recipe: a DAG of [`Node`]s plus the output (root) node indices.
#[derive(Clone, Debug)]
pub struct FlatDag {
    /// The nodes (child edges reference these by index; may be in any order — the
    /// evaluator resolves dependencies with memoization + a cycle guard).
    pub nodes: Vec<Node>,
    /// The output (root) node indices, in output order.
    pub outputs: Vec<usize>,
}

/// A rank-0 (scalar) tensor holding `v` — broadcasts to any shape in an
/// elementwise op (stride-0 on every axis).
fn rank0<T: Copy>(v: T) -> Result<Tensor<T>, Error> {
    Tensor::from_vec({ let mut d = Vec::new(); d.push(v); d }, &[])
}

/// Apply an elementwise scalar `op` over the broadcast of `children` (via the
/// unchanged scalar `eval_op`).
fn apply_elementwise<T: ScalarFloat>(op: Op, children: &[Tensor<T>]) -> Result<Tensor<T>, Error> {
    if children.is_empty() {
        return Err(Error::Arity { op, expected: 1, got: 0 });
    }
    let views: Vec<View<T>> = children.iter().map(|t| t.view()).collect();
    let shapes: Vec<&[usize]> = views.iter().map(|v| v.shape()).collect();
    let (out_shape, r) = broadcast_shapes(&shapes)?;
    map_views(&views, &out_shape[..r], |buf| eval_op(op, buf))
}

/// Drop the (extent-1, keepdim) reduced axes from a reduce result (`nokd`).
fn squeeze<T: Copy>(t: Tensor<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let rank = t.rank();
    let mut reduced = [false; MAX_RANK];
    for &a in axes {
        if a < rank {
            reduced[a] = true;
        }
    }
    let mut new_shape: Vec<usize> = Vec::new();
    for (k, &d) in t.shape().iter().enumerate() {
        if !reduced[k] {
            new_shape.push(d);
        }
    }
    // reduced axes are extent-1, so numel is unchanged and the contiguous data is
    // valid for the squeezed shape.
    Tensor::from_vec(t.into_data(), &new_shape)
}

/// The determinism class of an elementwise scalar `op` joined with its inputs'
/// classes (§6.0-0005 most-permissive): transcendentals carry their §6.8 ULP.
fn scalar_det(op: Op, child_dets: &[DetClass]) -> DetClass {
    let own = match op.ulp_ceiling() {
        Some(u) => DetClass::Ulp(u),
        None => DetClass::ExactByte,
    };
    child_dets.iter().fold(own, |acc, &d| acc.join(d))
}

/// Evaluate a recipe DAG on `inputs` (`Bind(i) → inputs[i]`), `params`
/// (`runtime_scalar(slot) → params[slot]`), and `indices` (the integer index
/// operands of `gather`/`scatter` nodes, referenced by slot), returning the output
/// tensor(s) and the **per-node** [`DetClass`]. Float lane. Never panics — every
/// failure is an [`Error`].
pub fn eval_recipe<T: ScalarFloat>(
    dag: &FlatDag,
    inputs: &[Tensor<T>],
    params: &[T],
    indices: &[IndexTensor],
) -> Result<(Vec<Tensor<T>>, Vec<DetClass>), Error> {
    let n = dag.nodes.len();
    let mut memo: Vec<Option<(Tensor<T>, DetClass)>> = (0..n).map(|_| None).collect();
    // Iterative post-order over an explicit HEAP worklist (not the call stack), so a
    // deep dependency chain is bounded to O(n) heap and returns a result/Error rather
    // than overflowing the stack (never-panic). `state`: 0 unvisited, 1 active (on the
    // current path — a revisit is a cycle), 2 done.
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
                // children are all computed (post-order): compute this node.
                let result = compute_node(dag, idx, inputs, params, indices, &memo)?;
                memo[idx] = Some(result);
                state[idx] = 2;
            } else {
                if state[idx] == 1 {
                    // a cycle in a supposedly-acyclic DAG — decline, don't loop.
                    return Err(Error::BadDecomposition { op: Op::Add, pos: idx });
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
    Ok((outputs, dets))
}

/// Read the memoized results of `children` (each already evaluated), cloning the
/// tensors and collecting their determinism classes.
fn read_children<T: Clone>(
    children: &[usize],
    memo: &[Option<(Tensor<T>, DetClass)>],
) -> Result<(Vec<Tensor<T>>, Vec<DetClass>), Error> {
    let mut ts = Vec::with_capacity(children.len());
    let mut ds = Vec::with_capacity(children.len());
    for &c in children {
        let (t, d) = memo
            .get(c)
            .and_then(|x| x.as_ref())
            .ok_or(Error::MissingInput(0))?;
        ts.push(t.clone());
        ds.push(*d);
    }
    Ok((ts, ds))
}

/// The child node indices of `node` (the edges the worklist must evaluate first).
fn children_of(node: &Node) -> Vec<usize> {
    let mut v = Vec::new();
    match node {
        Node::Bind(_) | Node::Const(_) | Node::RuntimeScalar(_) | Node::ReducedCount(_) => {}
        Node::Apply { children, .. } => v.extend_from_slice(children),
        Node::Reduce { child, .. } | Node::PrefixScan { child, .. } => v.push(*child),
        Node::Matmul { lhs, rhs } => {
            v.push(*lhs);
            v.push(*rhs);
        }
        // index inputs are external (the `indices` array), not node children.
        Node::Gather { data, .. } => v.push(*data),
        Node::Scatter { dest, updates, .. } => {
            v.push(*dest);
            v.push(*updates);
        }
        Node::SortNetwork { keys, .. } => v.push(*keys),
    }
    v
}

/// Compute node `idx` from its already-memoized children (the worklist guarantees
/// they are `Some`). No recursion — the driver walks the DAG iteratively.
fn compute_node<T: ScalarFloat>(
    dag: &FlatDag,
    idx: usize,
    inputs: &[Tensor<T>],
    params: &[T],
    indices: &[IndexTensor],
    memo: &[Option<(Tensor<T>, DetClass)>],
) -> Result<(Tensor<T>, DetClass), Error> {
    match &dag.nodes[idx] {
        Node::Bind(i) => {
            let t = inputs.get(*i).cloned().ok_or(Error::MissingInput(*i as u8))?;
            Ok((t, DetClass::ExactByte))
        }
        Node::Const(v) => Ok((rank0(T::from_f64(*v))?, DetClass::ExactByte)),
        Node::RuntimeScalar(s) => {
            let v = params.get(*s).copied().ok_or(Error::MissingInput(*s as u8))?;
            Ok((rank0(v)?, DetClass::ExactByte))
        }
        Node::ReducedCount(axes) => {
            let x0 = inputs.first().ok_or(Error::MissingInput(0))?;
            let shape = x0.shape();
            let mut c: usize = 1;
            for &a in axes {
                c = c
                    .checked_mul(*shape.get(a).ok_or(Error::AxisOutOfRange { axis: a, rank: shape.len() })?)
                    .ok_or(Error::ShapeOverflow)?;
            }
            Ok((rank0(T::from_f64(c as f64))?, DetClass::ExactByte))
        }
        Node::Apply { op, children } => {
            let (ts, ds) = read_children(children, memo)?;
            Ok((apply_elementwise(*op, &ts)?, scalar_det(*op, &ds)))
        }
        Node::Reduce { monoid, axes, keepdim, child } => {
            let (ts, ds) = read_children(core::slice::from_ref(child), memo)?;
            let r = reduce(&ts[0].view(), *monoid, axes)?;
            let r = if *keepdim { r } else { squeeze(r, axes)? };
            Ok((r, monoid_det(*monoid).join(ds[0])))
        }
        Node::PrefixScan { monoid, axis, exclusive, child } => {
            let (ts, ds) = read_children(core::slice::from_ref(child), memo)?;
            let r = prefix_scan(&ts[0].view(), *monoid, *axis, *exclusive)?;
            Ok((r, monoid_det(*monoid).join(ds[0])))
        }
        Node::Matmul { lhs, rhs } => {
            let (ts, ds) = read_children(&[*lhs, *rhs], memo)?;
            let r = matmul(&ts[0].view(), &ts[1].view())?;
            // float sum contraction → order-invariant/nondeterministic (§6.0-0004).
            Ok((r, DetClass::OrderInvariantNondeterministic.join(ds[0]).join(ds[1])))
        }
        Node::Gather { data, index, axis, oob } => {
            let (ts, ds) = read_children(core::slice::from_ref(data), memo)?;
            let idx = indices.get(*index).ok_or(Error::MissingInput(*index as u8))?;
            // v1: no base operand (the §6.11 gather-skip RFC extension) — skip on OOB errors.
            let r = gather(&ts[0].view(), idx, *axis, *oob, None)?;
            Ok((r, DetClass::ExactByte.join(ds[0]))) // raw-bit move
        }
        Node::Scatter { dest, index, updates, axis, combine } => {
            let (ts, ds) = read_children(&[*dest, *updates], memo)?;
            let idx = indices.get(*index).ok_or(Error::MissingInput(*index as u8))?;
            let r = scatter(ts[0].clone(), idx, &ts[1].view(), *axis, *combine)?;
            // float atomic_add → order-invariant/nondeterministic; else exact-byte.
            let own = match combine {
                Combine::AtomicAdd => DetClass::OrderInvariantNondeterministic,
                _ => DetClass::ExactByte,
            };
            Ok((r, own.join(ds[0]).join(ds[1])))
        }
        Node::SortNetwork { keys, axis, dir } => {
            let (ts, ds) = read_children(core::slice::from_ref(keys), memo)?;
            let (vals, _idx) = sort_network(&ts[0].view(), *axis, *dir)?;
            Ok((vals, DetClass::ExactByte.join(ds[0]))) // total order + raw-bit move
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn matmul_bias_relu_recipe() {
        // out = relu(matmul(a, b) + bias), a=[2,3] b=[3,2] bias=[2] (broadcast).
        // nodes: 0=Bind(0)=a, 1=Bind(1)=b, 2=matmul(0,1), 3=Bind(2)=bias,
        //        4=add(2,3), 5=relu(4).
        let dag = FlatDag {
            nodes: vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Matmul { lhs: 0, rhs: 1 },
                Node::Bind(2),
                Node::Apply { op: Op::Add, children: vec![2, 3] },
                Node::Apply { op: Op::Relu, children: vec![4] },
            ],
            outputs: vec![5],
        };
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let bias = t(&[-10.0, 0.0], &[2]);
        let (outs, dets) = eval_recipe(&dag, &[a, b, bias], &[], &[]).unwrap();
        // matmul: row0=[1+0+3, 0+2+3]=[4,5]; row1=[4+0+6, 0+5+6]=[10,11].
        // +bias[-10,0]: [-6,5],[0,11]; relu: [0,5],[0,11].
        assert_eq!(outs[0].shape(), &[2, 2]);
        close(outs[0].as_slice(), &[0.0, 5.0, 0.0, 11.0]);
        // matmul node (index 2) is nondeterministic; the relu root inherits it.
        assert_eq!(dets[2], DetClass::OrderInvariantNondeterministic);
        assert_eq!(dets[5], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn softmax_recipe_rows_sum_to_one() {
        // softmax(x, axis=1) = e/sum(e), e=exp(x - max(x)); keepdim reduces broadcast.
        // 0=Bind(0)=x, 1=reduce(max,axis1,keepdim), 2=sub(0,1), 3=exp(2),
        // 4=reduce(sum,axis1,keepdim,3), 5=div(3,4).
        let dag = FlatDag {
            nodes: vec![
                Node::Bind(0),
                Node::Reduce { monoid: Monoid::Max, axes: vec![1], keepdim: true, child: 0 },
                Node::Apply { op: Op::Sub, children: vec![0, 1] },
                Node::Apply { op: Op::Exp, children: vec![2] },
                Node::Reduce { monoid: Monoid::Sum, axes: vec![1], keepdim: true, child: 3 },
                Node::Apply { op: Op::Div, children: vec![3, 4] },
            ],
            outputs: vec![5],
        };
        let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let (outs, dets) = eval_recipe(&dag, &[x], &[], &[]).unwrap();
        let s = outs[0].as_slice();
        close(&[s[0] + s[1] + s[2]], &[1.0]);
        close(&[s[3] + s[4] + s[5]], &[1.0]);
        close(&s[3..6], &[1.0 / 3.0; 3]);
        // exp node carries a ULP class; the div root is nondeterministic (sum inside).
        assert!(matches!(dets[3], DetClass::Ulp(_)));
        assert_eq!(dets[5], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn runtime_scalar_and_reduced_count() {
        // mean(x) via reduce(sum)/reduced_count, then + a runtime scalar param.
        // 0=Bind(0), 1=reduce(sum,axis0,nokd), 2=reduced_count([0]),
        // 3=div(1,2), 4=runtime_scalar(0), 5=add(3,4).
        let dag = FlatDag {
            nodes: vec![
                Node::Bind(0),
                Node::Reduce { monoid: Monoid::Sum, axes: vec![0], keepdim: false, child: 0 },
                Node::ReducedCount(vec![0]),
                Node::Apply { op: Op::Div, children: vec![1, 2] },
                Node::RuntimeScalar(0),
                Node::Apply { op: Op::Add, children: vec![3, 4] },
            ],
            outputs: vec![5],
        };
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let (outs, _) = eval_recipe(&dag, &[x], &[10.0], &[]).unwrap();
        // mean = 2.5, + 10 = 12.5; nokd → rank-0 scalar.
        assert_eq!(outs[0].shape(), &[] as &[usize]);
        close(outs[0].as_slice(), &[12.5]);
    }

    #[test]
    fn gather_scatter_recipe_nodes() {
        use kiss_classify_vocab::Dtype;
        // gather(data, index=indices[0], axis 0, zero_fill).
        let dag = FlatDag {
            nodes: vec![
                Node::Bind(0),
                Node::Gather { data: 0, index: 0, axis: 0, oob: OobPolicy::ZeroFill },
            ],
            outputs: vec![1],
        };
        let data = t(&[10.0, 20.0, 30.0], &[3]);
        let idx = IndexTensor::new(vec![2, 0, 9], &[3], Dtype::I64).unwrap();
        let (outs, dets) = eval_recipe(&dag, &[data], &[], &[idx]).unwrap();
        close(outs[0].as_slice(), &[30.0, 10.0, 0.0]); // idx 9 OOB → zero-fill
        assert_eq!(dets[1], DetClass::ExactByte); // raw-bit move

        // scatter_add(dest, updates, index=indices[0], atomic_add).
        let sdag = FlatDag {
            nodes: vec![
                Node::Bind(0), // dest
                Node::Bind(1), // updates
                Node::Scatter { dest: 0, index: 0, updates: 1, axis: 0, combine: Combine::AtomicAdd },
            ],
            outputs: vec![2],
        };
        let dest = t(&[0.0, 0.0, 0.0], &[3]);
        let upd = t(&[5.0, 7.0, 4.0], &[3]);
        let sidx = IndexTensor::new(vec![0, 0, 1], &[3], Dtype::I64).unwrap();
        let (souts, sdets) = eval_recipe(&sdag, &[dest, upd], &[], &[sidx]).unwrap();
        close(souts[0].as_slice(), &[12.0, 4.0, 0.0]); // idx0: 5+7=12, idx1: 4
        // float atomic_add → order-invariant/nondeterministic (§6.0-0004).
        assert_eq!(sdets[2], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn deep_chain_does_not_overflow() {
        // A long descending-index chain (child index > parent — permitted) must
        // evaluate iteratively without a stack overflow (regression, adversarial
        // review). node[i]=neg(node[i+1]); node[n-1]=Bind(0).
        let n = 20_000usize;
        let mut nodes = Vec::new();
        for i in 0..n {
            if i + 1 < n {
                nodes.push(Node::Apply { op: Op::Neg, children: vec![i + 1] });
            } else {
                nodes.push(Node::Bind(0));
            }
        }
        let dag = FlatDag { nodes, outputs: vec![0] };
        let x = t(&[5.0], &[]);
        let (outs, _) = eval_recipe(&dag, &[x], &[], &[]).unwrap();
        // 19_999 negations (odd) of 5.0 → -5.0. No overflow, returns a value.
        close(outs[0].as_slice(), &[-5.0]);
    }

    #[test]
    fn cycle_and_bad_index_decline_not_panic() {
        // a self-referential node must decline, not loop forever.
        let dag = FlatDag {
            nodes: vec![Node::Apply { op: Op::Add, children: vec![0, 0] }],
            outputs: vec![0],
        };
        assert!(eval_recipe::<f64>(&dag, &[], &[], &[]).is_err());
        // an out-of-range child index is an error.
        let dag2 = FlatDag {
            nodes: vec![Node::Apply { op: Op::Neg, children: vec![9] }],
            outputs: vec![0],
        };
        assert!(eval_recipe::<f64>(&dag2, &[], &[], &[]).is_err());
    }
}
