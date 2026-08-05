//! The **recipe evaluator**: `eval_recipe(dag, inputs, params, indices) ->
//! RecipeEval { outputs, index_outputs, dets }`.
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
//! `matmul`, the value leaves (`Bind`/`const`/`runtime_scalar`/`reduced_count`),
//! the index-bearing nodes `gather`/`scatter`/`sort_network` (incl. the RFC-pinned
//! gather `base` operand), and `iota` (§6.12 `coord` leaf; shape-of-child v1).
//!
//! **Two lattices.** The DAG carries two parallel result lattices: the **float
//! value lattice** (every node; homogeneously `Tensor<T>`, the memo) and a sparse
//! **integer index lattice** (`IndexTensor`; today populated only by
//! `sort_network`'s §6.11-0007 original-index output). Integer data never rides
//! the float lane. An index operand of `gather`/`scatter` is an [`IndexRef`]:
//! either an external `indices[slot]` input or the index-lane output of another
//! node (a real scheduling edge). Index-lane results leave the evaluator via
//! [`FlatDag::index_outputs`], a second root list symmetric with `outputs`
//! (§6.19-encodable as a second varint root list; `IndexRef` as tag + varint).
//! One [`DetClass`] per node covers **both** lane products of that node.

extern crate alloc;
use alloc::vec::Vec;

use kiss_ops_vocab::Op;

use crate::attrs::{Combine, Direction, Monoid, OobPolicy};
use crate::bridge::{monoid_det, DetClass};
use crate::kernels::{gather, map_views, prefix_scan, reduce, scatter, sort_network};
use crate::resolve::eval_op;
use crate::scalar::ScalarFloat;
use crate::tensor::{alloc_exact, broadcast_shapes, numel, IndexTensor, Tensor, View, MAX_RANK};
use crate::tensor_ops::{flip, matmul};
use crate::Error;

/// A reference to an integer index operand of `gather`/`scatter`: either an
/// external `indices[slot]` input or the **index-lane output** of another node.
/// §6.19-encodable as tag + varint. (The extended-flat-slot-space alternative —
/// node-produced indices numbered above `indices.len()` — was rejected: a slot's
/// meaning would depend on the runtime `indices.len()`, which the DAG cannot see.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexRef {
    /// The external `indices[slot]` input.
    Slot(usize),
    /// The index-lane output of node `.0` (today: `SortNetwork`'s original-index
    /// permutation). A real scheduling edge — the evaluator computes that node
    /// first, and the cycle guard covers it. Referencing a node with no
    /// index-lane output is [`Error::IndexSourceInvalid`], never a panic.
    Node(usize),
}

/// A node in the canonical flat-DAG (the logical, decoded form).
#[derive(Clone, Debug)]
pub enum Node {
    /// Bind the fused op's `inputs[i]` (§6.4-0009 `Bind`).
    Bind(usize),
    /// A real-valued `const` leaf: the reference holds an `f64` and RE-ROUNDS it into
    /// the lane via `T::from_f64` — ergonomic and dtype-generic, but a signalling-NaN
    /// payload survives only on the f64 lane. For a bit-exact leaf use [`Node::ConstBits`].
    Const(f64),
    /// A bit-exact `const(bits)` leaf (KISS-OPS-6.12-0002): the low `T`-width bits of the
    /// `u64` are reinterpreted as a `T` verbatim (`T::from_bits`), so ANY pattern —
    /// signalling-NaN payload, ±0, subnormal — round-trips bit-for-bit on every lane.
    /// The bits are dtype-specific (the same node evaluated at a wider `T` reads more
    /// bits). `ExactByte`.
    ConstBits(u64),
    /// A `runtime_scalar(slot)` leaf → `params[slot]`.
    RuntimeScalar(usize),
    /// A `reduced_count(axes)` value leaf — ∏ of the reduced extents (resolved
    /// against `inputs[0]`; the leaf-vs-shape modeling is a routed grammar question).
    ReducedCount(Vec<usize>),
    /// An elementwise scalar op over child nodes (§6.12 body atom): `add`/`sub`/
    /// `mul`/`div`, the unary/binary math atoms, `select`.
    Apply { op: Op, children: Vec<usize> },
    /// A `reduce` fold node.
    Reduce {
        monoid: Monoid,
        axes: Vec<usize>,
        keepdim: bool,
        child: usize,
    },
    /// A `prefix_scan` fold node.
    PrefixScan {
        monoid: Monoid,
        axis: usize,
        exclusive: bool,
        child: usize,
    },
    /// A `matmul` contraction node (batched `[..b,M,K]·[..b,K,N]`).
    Matmul { lhs: usize, rhs: usize },
    /// A `gather` node: read `data` (a node) at a runtime `index` along `axis`.
    /// `base` is the §6.11 gather-skip operand (KISS PR #75 Gap 1, **ruled**
    /// option 1, 2026-07-23): an optional value-lane child node, broadcast to
    /// the output shape, whose value is kept (raw-bit) at a `Skip`ped OOB
    /// position. The base requirement is **dynamic**: `Skip` + `base: None` is
    /// legal until an index is actually OOB, which is the typed
    /// [`Error::GatherSkipNoBase`] decline.
    Gather {
        data: usize,
        index: IndexRef,
        axis: usize,
        oob: OobPolicy,
        base: Option<usize>,
    },
    /// A `scatter` node: write `updates` (a node) into `dest` (a node) at a runtime
    /// `index` along `axis`, combined per `combine`.
    Scatter {
        dest: usize,
        index: IndexRef,
        updates: usize,
        axis: usize,
        combine: Combine,
    },
    /// A `sort_network` node. **Value lane:** the sorted values along `axis` (a
    /// raw-bit permutation). **Index lane** (same node id): the §6.11-0007
    /// original-index permutation (`i64`, stable total order), consumable
    /// downstream via [`IndexRef::Node`] and exportable via
    /// [`FlatDag::index_outputs`].
    SortNetwork {
        keys: usize,
        axis: usize,
        dir: Direction,
    },
    /// An `iota` node (the §6.12-0001 `coord(axis)` leaf): the row-major
    /// coordinate along `axis` of the **shape** of node `like`, as `T`. The
    /// `like` edge is SHAPE-ONLY — its values are never read, so its [`DetClass`]
    /// does not propagate. v1 shape source = shape-of-child (the §6.20 shape
    /// oracle's expected primitive, pending that grammar); `axis >= rank(like)`
    /// is [`Error::AxisOutOfRange`]. Value-lane only; index-lane publication is a
    /// reserved additive follow-up. Coordinates beyond `T`'s exact-integer range
    /// round deterministically (RNE) — §6.12 exactness is a routed validator
    /// question.
    Iota { like: usize, axis: usize },
    /// A `flip` node: reverse `child` along `axis` (`out[c] = in[c′]`,
    /// `c′[axis] = extent−1−c[axis]`). A pure raw-bit move — exact-byte, NaN
    /// payload / −0 preserved. Added for the reverse-scan emission need
    /// (KISS #76 flip routing; grammar row rides the #67 consolidation).
    Flip { child: usize, axis: usize },
}

/// A recipe: a DAG of [`Node`]s plus the output (root) node indices.
#[derive(Clone, Debug)]
pub struct FlatDag {
    /// The nodes (child edges reference these by index; may be in any order — the
    /// evaluator resolves dependencies with memoization + a cycle guard).
    pub nodes: Vec<Node>,
    /// The output (root) node indices, in output order (the value lane).
    pub outputs: Vec<usize>,
    /// Node ids whose **index-lane** result is exported, in output order — a
    /// second root list symmetric with `outputs` (§6.19: a second varint root
    /// list). Listing a node with no index-lane output is
    /// [`Error::IndexSourceInvalid`]. Empty = value-lane-only (the pre-index-lane
    /// behavior).
    pub index_outputs: Vec<usize>,
}

impl FlatDag {
    /// A value-lane-only DAG — `index_outputs` defaults to empty (the pre-
    /// index-lane constructor shape).
    pub fn new(nodes: Vec<Node>, outputs: Vec<usize>) -> Self {
        FlatDag {
            nodes,
            outputs,
            index_outputs: Vec::new(),
        }
    }
}

/// The result of evaluating a recipe.
#[derive(Clone, Debug)]
pub struct RecipeEval<T> {
    /// The value-lane outputs, in `dag.outputs` order.
    pub outputs: Vec<Tensor<T>>,
    /// The index-lane outputs, in `dag.index_outputs` order (`i64`-widened,
    /// dtype-tagged). Empty when the DAG declares none.
    pub index_outputs: Vec<IndexTensor>,
    /// Per-node [`DetClass`] — the **value-lane** class, in node order, most-permissive
    /// over each node's producing sub-DAG (§6.0-0005 join). For a **selection** output
    /// (an index/permutation) do NOT read this directly: `dets[sort]` is the sorted
    /// VALUES' class (e.g. `Ulp(k)` for `Ulp`-classed keys), but the exported
    /// permutation's class is the [`selection_det`] escalation of it — use
    /// [`RecipeEval::index_output_dets`] (KISS-OPS-6.0-0007), because a permutation over
    /// non-exact keys is not ULP-boundable.
    pub dets: Vec<DetClass>,
}

impl<T> RecipeEval<T> {
    /// The per-**index-output** determinism classes, in `dag.index_outputs` order: the
    /// [`selection_det`] escalation of each index output's producing node's class. An
    /// index/permutation is a SELECTION, not a value, so its class is NOT `self.dets[m]`
    /// (the value-lane join, e.g. `Ulp(k)`) but `ExactByte`-or-nondeterministic
    /// (KISS-OPS-6.0-0007). Use THIS — not `dets` — to classify an index output for a
    /// differential comparator.
    pub fn index_output_dets(&self, dag: &FlatDag) -> Vec<DetClass> {
        dag.index_outputs
            .iter()
            .map(|&m| selection_det(self.dets.get(m).copied().unwrap_or(DetClass::ExactByte)))
            .collect()
    }
}

/// A rank-0 (scalar) tensor holding `v` — broadcasts to any shape in an
/// elementwise op (stride-0 on every axis).
// The `vec!` macro is not imported on this no_std non-test path, so the
// one-element buffer is built by hand.
#[allow(clippy::vec_init_then_push)]
pub(crate) fn rank0<T: Copy>(v: T) -> Result<Tensor<T>, Error> {
    Tensor::from_vec(
        {
            let mut d = Vec::new();
            d.push(v);
            d
        },
        &[],
    )
}

/// Apply an elementwise scalar `op` over the broadcast of `children` (via the
/// unchanged scalar `eval_op`).
fn apply_elementwise<T: ScalarFloat>(op: Op, children: &[Tensor<T>]) -> Result<Tensor<T>, Error> {
    if children.is_empty() {
        return Err(Error::Arity {
            op,
            expected: 1,
            got: 0,
        });
    }
    let views: Vec<View<T>> = children.iter().map(|t| t.view()).collect();
    let shapes: Vec<&[usize]> = views.iter().map(|v| v.shape()).collect();
    let (out_shape, r) = broadcast_shapes(&shapes)?;
    map_views(&views, &out_shape[..r], |buf| eval_op(op, buf))
}

/// Drop the (extent-1, keepdim) reduced axes from a reduce result (`nokd`).
pub(crate) fn squeeze<T: Copy>(t: Tensor<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
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

/// True iff `node` produces an index-lane result (consumable via
/// [`IndexRef::Node`] / exportable via [`FlatDag::index_outputs`]).
pub(crate) fn has_index_lane(node: &Node) -> bool {
    matches!(node, Node::SortNetwork { .. })
}

/// Resolve an index operand: an external `indices[slot]` input or the index-lane
/// output of an already-evaluated node. Shared with the integer lane
/// ([`crate::recipe_int`]) — the resolution is value-type-agnostic.
pub(crate) fn resolve_index_ref<'a>(
    r: IndexRef,
    indices: &'a [IndexTensor],
    imemo: &'a [Option<IndexTensor>],
) -> Result<&'a IndexTensor, Error> {
    match r {
        IndexRef::Slot(s) => indices.get(s).ok_or(Error::MissingIndexOperand { slot: s }),
        IndexRef::Node(m) => imemo
            .get(m)
            .and_then(|x| x.as_ref())
            .ok_or(Error::IndexSourceInvalid { node: m }),
    }
}

/// The determinism class of a **selection** output — an index / permutation / mask /
/// one-hot / bucket that reports WHICH value won, not a value — given its producing
/// sub-DAG's class (KISS-OPS-6.0-0007). A selection is never more deterministic than the
/// values it selects among, and a selection over non-exact values is NOT ULP-boundable:
/// a ≤k-ULP perturbation of the compared values can flip the selection and relocate it
/// without bound. So it is `ExactByte` iff the producer is entirely exact-byte, else
/// `OrderInvariantNondeterministic` — **never `Ulp(k)`**. This is the escalation the
/// value-lane most-permissive join must NOT apply to a selection output.
pub fn selection_det(producer: DetClass) -> DetClass {
    match producer {
        DetClass::ExactByte => DetClass::ExactByte,
        _ => DetClass::OrderInvariantNondeterministic,
    }
}

/// The [`DetClass`] contribution of an index operand to its CONSUMER's value-lane
/// class. An external slot is a given (exact, shared byte-identically by reference and
/// candidate). A node-produced index escalates via [`selection_det`]: a non-exact
/// producer (ULP-classed sort keys) can permute **differently** within its own
/// tolerance, relocating whole data elements — a deviation NOT ULP-boundable — so
/// anything short of `ExactByte` becomes `OrderInvariantNondeterministic`, never a
/// forwarded `Ulp(k)` the consumer's output cannot honor.
fn index_ref_det<T: Clone>(r: IndexRef, memo: &[Option<(Tensor<T>, DetClass)>]) -> DetClass {
    match r {
        IndexRef::Slot(_) => DetClass::ExactByte,
        IndexRef::Node(m) => selection_det(
            memo.get(m)
                .and_then(|x| x.as_ref())
                .map(|(_, d)| *d)
                .unwrap_or(DetClass::ExactByte),
        ),
    }
}

/// Evaluate a recipe DAG on `inputs` (`Bind(i) → inputs[i]`), `params`
/// (`runtime_scalar(slot) → params[slot]`), and `indices` (the external integer
/// index operands of `gather`/`scatter` nodes, addressed by [`IndexRef::Slot`]),
/// returning a [`RecipeEval`]: the value-lane output tensor(s), the index-lane
/// outputs, and the **per-node** [`DetClass`]. Float lane. Never panics — every
/// failure is an [`Error`].
pub fn eval_recipe<T: ScalarFloat>(
    dag: &FlatDag,
    inputs: &[Tensor<T>],
    params: &[T],
    indices: &[IndexTensor],
) -> Result<RecipeEval<T>, Error> {
    let n = dag.nodes.len();
    // Pre-validate every index-lane reference against the DAG's static shape, so
    // a bad reference is the dedicated typed error rather than a generic decline.
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
    let mut memo: Vec<Option<(Tensor<T>, DetClass)>> = (0..n).map(|_| None).collect();
    // The sparse index-lane memo (today only `SortNetwork` populates it).
    let mut imemo: Vec<Option<IndexTensor>> = (0..n).map(|_| None).collect();
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
                let (t, d, ix) = compute_node(dag, idx, inputs, params, indices, &memo, &imemo)?;
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

/// Read the memoized results of `children` (each already evaluated), cloning the
/// tensors and collecting their determinism classes.
pub(crate) fn read_children<T: Clone>(
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
pub(crate) fn children_of(node: &Node) -> Vec<usize> {
    let mut v = Vec::new();
    match node {
        Node::Bind(_)
        | Node::Const(_)
        | Node::ConstBits(_)
        | Node::RuntimeScalar(_)
        | Node::ReducedCount(_) => {}
        Node::Apply { children, .. } => v.extend_from_slice(children),
        Node::Reduce { child, .. } | Node::PrefixScan { child, .. } => v.push(*child),
        Node::Matmul { lhs, rhs } => {
            v.push(*lhs);
            v.push(*rhs);
        }
        // a Slot index operand is external (the `indices` array), not a child;
        // a Node index operand IS a scheduling edge (the producer runs first).
        Node::Gather {
            data, index, base, ..
        } => {
            v.push(*data);
            if let Some(b) = base {
                v.push(*b);
            }
            if let IndexRef::Node(m) = index {
                v.push(*m);
            }
        }
        Node::Scatter {
            dest,
            updates,
            index,
            ..
        } => {
            v.push(*dest);
            v.push(*updates);
            if let IndexRef::Node(m) = index {
                v.push(*m);
            }
        }
        Node::SortNetwork { keys, .. } => v.push(*keys),
        Node::Iota { like, .. } => v.push(*like),
        Node::Flip { child, .. } => v.push(*child),
    }
    v
}

/// Compute node `idx` from its already-memoized children (the worklist guarantees
/// they are `Some`). No recursion — the driver walks the DAG iteratively. Returns
/// the value-lane result, the node's [`DetClass`] (covering both lanes), and the
/// index-lane result (`Some` only for index-producing nodes).
#[allow(clippy::type_complexity)]
fn compute_node<T: ScalarFloat>(
    dag: &FlatDag,
    idx: usize,
    inputs: &[Tensor<T>],
    params: &[T],
    indices: &[IndexTensor],
    memo: &[Option<(Tensor<T>, DetClass)>],
    imemo: &[Option<IndexTensor>],
) -> Result<(Tensor<T>, DetClass, Option<IndexTensor>), Error> {
    match &dag.nodes[idx] {
        Node::Bind(i) => {
            let t = inputs
                .get(*i)
                .cloned()
                .ok_or(Error::MissingInput(*i as u8))?;
            Ok((t, DetClass::ExactByte, None))
        }
        Node::Const(v) => Ok((rank0(T::from_f64(*v))?, DetClass::ExactByte, None)),
        Node::ConstBits(bits) => Ok((rank0(T::from_bits(*bits))?, DetClass::ExactByte, None)),
        Node::RuntimeScalar(s) => {
            let v = params
                .get(*s)
                .copied()
                .ok_or(Error::MissingInput(*s as u8))?;
            Ok((rank0(v)?, DetClass::ExactByte, None))
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
            Ok((rank0(T::from_f64(c as f64))?, DetClass::ExactByte, None))
        }
        Node::Apply { op, children } => {
            let (ts, ds) = read_children(children, memo)?;
            Ok((apply_elementwise(*op, &ts)?, scalar_det(*op, &ds), None))
        }
        Node::Reduce {
            monoid,
            axes,
            keepdim,
            child,
        } => {
            let (ts, ds) = read_children(core::slice::from_ref(child), memo)?;
            let r = reduce(&ts[0].view(), *monoid, axes)?;
            let r = if *keepdim { r } else { squeeze(r, axes)? };
            Ok((r, monoid_det(*monoid).join(ds[0]), None))
        }
        Node::PrefixScan {
            monoid,
            axis,
            exclusive,
            child,
        } => {
            let (ts, ds) = read_children(core::slice::from_ref(child), memo)?;
            let r = prefix_scan(&ts[0].view(), *monoid, *axis, *exclusive)?;
            Ok((r, monoid_det(*monoid).join(ds[0]), None))
        }
        Node::Matmul { lhs, rhs } => {
            let (ts, ds) = read_children(&[*lhs, *rhs], memo)?;
            let r = matmul(&ts[0].view(), &ts[1].view())?;
            // float sum contraction → order-invariant/nondeterministic (§6.0-0004).
            Ok((
                r,
                DetClass::OrderInvariantNondeterministic
                    .join(ds[0])
                    .join(ds[1]),
                None,
            ))
        }
        Node::Gather {
            data,
            index,
            axis,
            oob,
            base,
        } => {
            let (ts, ds) = read_children(core::slice::from_ref(data), memo)?;
            let idx = resolve_index_ref(*index, indices, imemo)?;
            let basem = match base {
                Some(b) => Some(
                    memo.get(*b)
                        .and_then(|x| x.as_ref())
                        .ok_or(Error::MissingInput(0))?,
                ),
                None => None,
            };
            let bview = basem.map(|(t, _)| t.view());
            let r = gather(&ts[0].view(), idx, *axis, *oob, bview.as_ref())?;
            // raw-bit move ⊔ data ⊔ index producer; the base's class joins ONLY
            // under `Skip` — that is the only arm whose output can carry base
            // bits (a static, policy-conditioned join, never index-conditioned).
            let mut det = DetClass::ExactByte
                .join(ds[0])
                .join(index_ref_det(*index, memo));
            if *oob == OobPolicy::Skip {
                if let Some((_, bd)) = basem {
                    det = det.join(*bd);
                }
            }
            Ok((r, det, None))
        }
        Node::Scatter {
            dest,
            index,
            updates,
            axis,
            combine,
        } => {
            let (ts, ds) = read_children(&[*dest, *updates], memo)?;
            let idx = resolve_index_ref(*index, indices, imemo)?;
            let r = scatter(ts[0].clone(), idx, &ts[1].view(), *axis, *combine)?;
            // float atomic_add → order-invariant/nondeterministic; else exact-byte.
            let own = match combine {
                Combine::AtomicAdd => DetClass::OrderInvariantNondeterministic,
                _ => DetClass::ExactByte,
            };
            Ok((
                r,
                own.join(ds[0])
                    .join(ds[1])
                    .join(index_ref_det(*index, memo)),
                None,
            ))
        }
        Node::SortNetwork { keys, axis, dir } => {
            let (ts, ds) = read_children(core::slice::from_ref(keys), memo)?;
            let (vals, perm) = sort_network(&ts[0].view(), *axis, *dir)?;
            // total order + raw-bit move; the one class covers BOTH lanes (ULP-
            // different keys may permute differently).
            Ok((vals, DetClass::ExactByte.join(ds[0]), Some(perm)))
        }
        Node::Iota { like, axis } => {
            // shape-only edge: read the child's SHAPE, never its values (so its
            // DetClass deliberately does not join — shapes are static per run).
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
            // row-major stride of `axis`; if a trailing extent is 0 then
            // count == 0 and the loop below never divides.
            let mut stride = 1usize;
            for &d in &shape[*axis + 1..] {
                stride = stride.checked_mul(d).ok_or(Error::ShapeOverflow)?;
            }
            let extent = shape[*axis];
            let mut buf: Vec<T> = alloc_exact(count)?;
            for i in 0..count {
                buf.push(T::from_f64(((i / stride) % extent) as f64));
            }
            Ok((Tensor::from_vec(buf, shape)?, DetClass::ExactByte, None))
        }
        Node::Flip { child, axis } => {
            let (ts, ds) = read_children(core::slice::from_ref(child), memo)?;
            let r = flip(&ts[0].view(), *axis)?;
            Ok((r, DetClass::ExactByte.join(ds[0]), None)) // raw-bit move
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
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Bind(1),
                Node::Matmul { lhs: 0, rhs: 1 },
                Node::Bind(2),
                Node::Apply {
                    op: Op::Add,
                    children: vec![2, 3],
                },
                Node::Apply {
                    op: Op::Relu,
                    children: vec![4],
                },
            ],
            vec![5],
        );
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let bias = t(&[-10.0, 0.0], &[2]);
        let r = eval_recipe(&dag, &[a, b, bias], &[], &[]).unwrap();
        // matmul: row0=[1+0+3, 0+2+3]=[4,5]; row1=[4+0+6, 0+5+6]=[10,11].
        // +bias[-10,0]: [-6,5],[0,11]; relu: [0,5],[0,11].
        assert_eq!(r.outputs[0].shape(), &[2, 2]);
        close(r.outputs[0].as_slice(), &[0.0, 5.0, 0.0, 11.0]);
        assert!(r.index_outputs.is_empty());
        // matmul node (index 2) is nondeterministic; the relu root inherits it.
        assert_eq!(r.dets[2], DetClass::OrderInvariantNondeterministic);
        assert_eq!(r.dets[5], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn softmax_recipe_rows_sum_to_one() {
        // softmax(x, axis=1) = e/sum(e), e=exp(x - max(x)); keepdim reduces broadcast.
        // 0=Bind(0)=x, 1=reduce(max,axis1,keepdim), 2=sub(0,1), 3=exp(2),
        // 4=reduce(sum,axis1,keepdim,3), 5=div(3,4).
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Reduce {
                    monoid: Monoid::Max,
                    axes: vec![1],
                    keepdim: true,
                    child: 0,
                },
                Node::Apply {
                    op: Op::Sub,
                    children: vec![0, 1],
                },
                Node::Apply {
                    op: Op::Exp,
                    children: vec![2],
                },
                Node::Reduce {
                    monoid: Monoid::Sum,
                    axes: vec![1],
                    keepdim: true,
                    child: 3,
                },
                Node::Apply {
                    op: Op::Div,
                    children: vec![3, 4],
                },
            ],
            vec![5],
        );
        let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let r = eval_recipe(&dag, &[x], &[], &[]).unwrap();
        let s = r.outputs[0].as_slice();
        close(&[s[0] + s[1] + s[2]], &[1.0]);
        close(&[s[3] + s[4] + s[5]], &[1.0]);
        close(&s[3..6], &[1.0 / 3.0; 3]);
        // exp node carries a ULP class; the div root is nondeterministic (sum inside).
        assert!(matches!(r.dets[3], DetClass::Ulp(_)));
        assert_eq!(r.dets[5], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn runtime_scalar_and_reduced_count() {
        // mean(x) via reduce(sum)/reduced_count, then + a runtime scalar param.
        // 0=Bind(0), 1=reduce(sum,axis0,nokd), 2=reduced_count([0]),
        // 3=div(1,2), 4=runtime_scalar(0), 5=add(3,4).
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Reduce {
                    monoid: Monoid::Sum,
                    axes: vec![0],
                    keepdim: false,
                    child: 0,
                },
                Node::ReducedCount(vec![0]),
                Node::Apply {
                    op: Op::Div,
                    children: vec![1, 2],
                },
                Node::RuntimeScalar(0),
                Node::Apply {
                    op: Op::Add,
                    children: vec![3, 4],
                },
            ],
            vec![5],
        );
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let r = eval_recipe(&dag, &[x], &[10.0], &[]).unwrap();
        // mean = 2.5, + 10 = 12.5; nokd → rank-0 scalar.
        assert_eq!(r.outputs[0].shape(), &[] as &[usize]);
        close(r.outputs[0].as_slice(), &[12.5]);
    }

    #[test]
    fn gather_scatter_recipe_nodes() {
        use kiss_classify_vocab::Dtype;
        // gather(data, index=indices[0], axis 0, zero_fill).
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(0),
                    axis: 0,
                    oob: OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![1],
        );
        let data = t(&[10.0, 20.0, 30.0], &[3]);
        let idx = IndexTensor::new(vec![2, 0, 9], &[3], Dtype::I64).unwrap();
        let r = eval_recipe(&dag, &[data], &[], &[idx]).unwrap();
        close(r.outputs[0].as_slice(), &[30.0, 10.0, 0.0]); // idx 9 OOB → zero-fill
        assert_eq!(r.dets[1], DetClass::ExactByte); // raw-bit move

        // scatter_add(dest, updates, index=indices[0], atomic_add).
        let sdag = FlatDag::new(
            vec![
                Node::Bind(0), // dest
                Node::Bind(1), // updates
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
        let dest = t(&[0.0, 0.0, 0.0], &[3]);
        let upd = t(&[5.0, 7.0, 4.0], &[3]);
        let sidx = IndexTensor::new(vec![0, 0, 1], &[3], Dtype::I64).unwrap();
        let sr = eval_recipe(&sdag, &[dest, upd], &[], &[sidx]).unwrap();
        close(sr.outputs[0].as_slice(), &[12.0, 4.0, 0.0]); // idx0: 5+7=12, idx1: 4
                                                            // float atomic_add → order-invariant/nondeterministic (§6.0-0004).
        assert_eq!(sr.dets[2], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn sort_index_lane_exports_permutation() {
        // sort asc: values via the value lane, the §6.11-0007 original-index
        // permutation via index_outputs (same node id).
        let dag = FlatDag {
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
        };
        let keys = t(&[3.0, 1.0, 2.0], &[3]);
        let r = eval_recipe(&dag, &[keys], &[], &[]).unwrap();
        close(r.outputs[0].as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(r.index_outputs.len(), 1);
        assert_eq!(r.index_outputs[0].as_slice(), &[1, 2, 0]);
        assert_eq!(r.dets[1], DetClass::ExactByte); // one class covers both lanes
    }

    #[test]
    fn sort_gather_chain_composes() {
        // argsort-then-gather: reorder `data` by the sort permutation of `keys`
        // via IndexRef::Node — a real scheduling edge, no external index input.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0), // keys
                Node::Bind(1), // data
                Node::SortNetwork {
                    keys: 0,
                    axis: 0,
                    dir: Direction::Asc,
                },
                Node::Gather {
                    data: 1,
                    index: IndexRef::Node(2),
                    axis: 0,
                    oob: OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![3],
        );
        let keys = t(&[3.0, 1.0, 2.0], &[3]);
        let data = t(&[30.0, 10.0, 20.0], &[3]);
        let r = eval_recipe(&dag, &[keys, data], &[], &[]).unwrap();
        // perm [1,2,0] applied to data → [10,20,30].
        close(r.outputs[0].as_slice(), &[10.0, 20.0, 30.0]);
        assert_eq!(r.dets[3], DetClass::ExactByte); // exact chain end-to-end
    }

    #[test]
    fn nonexact_index_producer_escalates_consumer_det() {
        // Keys through exp carry Ulp(k); a ≤k-ULP key change can FLIP the sort
        // permutation, relocating whole data values — a deviation unbounded in
        // ULP. The gather consuming that permutation must therefore escalate to
        // the most-permissive class, NOT forward Ulp(k) (which would falsely
        // fail conforming candidates — adversarial-review regression).
        let dag = FlatDag::new(
            vec![
                Node::Bind(0), // x
                Node::Bind(1), // data
                Node::Apply {
                    op: Op::Exp,
                    children: vec![0],
                },
                Node::SortNetwork {
                    keys: 2,
                    axis: 0,
                    dir: Direction::Asc,
                },
                Node::Gather {
                    data: 1,
                    index: IndexRef::Node(3),
                    axis: 0,
                    oob: OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![4],
        );
        let x = t(&[0.0, 1.0], &[2]);
        let data = t(&[5.0, 7.0], &[2]);
        let r = eval_recipe(&dag, &[x, data], &[], &[]).unwrap();
        // the sort node itself stays ULP-classed (its VALUE lane is ULP-sound);
        // the index-mediated gather is order-invariant/nondeterministic.
        assert!(matches!(r.dets[3], DetClass::Ulp(_)));
        assert_eq!(r.dets[4], DetClass::OrderInvariantNondeterministic);
    }

    #[test]
    fn gather_base_skip_keeps_base_value() {
        use kiss_classify_vocab::Dtype;
        // gather(skip) with a rank-0 Const base: the OOB position keeps the
        // base's (broadcast) value — the cosigned §6.11 gather-skip RFC surface.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Const(-1.0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(0),
                    axis: 0,
                    oob: OobPolicy::Skip,
                    base: Some(1),
                },
            ],
            vec![2],
        );
        let data = t(&[10.0, 20.0, 30.0], &[3]);
        let idx = IndexTensor::new(vec![2, 9, 0], &[3], Dtype::I64).unwrap();
        let r = eval_recipe(&dag, &[data], &[], &[idx]).unwrap();
        close(r.outputs[0].as_slice(), &[30.0, -1.0, 10.0]);
        assert_eq!(r.dets[2], DetClass::ExactByte);

        // Skip + no base on an actually-OOB index stays a typed decline.
        let bad = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Slot(0),
                    axis: 0,
                    oob: OobPolicy::Skip,
                    base: None,
                },
            ],
            vec![1],
        );
        let data2 = t(&[10.0, 20.0, 30.0], &[3]);
        let idx2 = IndexTensor::new(vec![2, 9, 0], &[3], Dtype::I64).unwrap();
        assert!(eval_recipe(&bad, &[data2], &[], &[idx2]).is_err());
    }

    #[test]
    fn gather_base_det_joins_only_under_skip() {
        use kiss_classify_vocab::Dtype;
        // base = exp(const) carries a ULP class. Under Skip the gather joins it;
        // under ZeroFill the base's bits can never reach the output, so the
        // class must NOT join (precision the differential comparator relies on).
        let mk = |oob| {
            FlatDag::new(
                vec![
                    Node::Bind(0),
                    Node::Const(1.0),
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
        let data = t(&[10.0, 20.0, 30.0], &[3]);
        let idx = IndexTensor::new(vec![0, 1, 2], &[3], Dtype::I64).unwrap();
        let skip = eval_recipe(
            &mk(OobPolicy::Skip),
            std::slice::from_ref(&data),
            &[],
            std::slice::from_ref(&idx),
        )
        .unwrap();
        assert!(matches!(skip.dets[3], DetClass::Ulp(_)));
        let zf = eval_recipe(&mk(OobPolicy::ZeroFill), &[data], &[], &[idx]).unwrap();
        assert_eq!(zf.dets[3], DetClass::ExactByte);
    }

    #[test]
    fn scatter_updates_broadcast() {
        use kiss_classify_vocab::Dtype;
        // updates broadcast to the write shape (§6.11-0001 general rules —
        // scatter broadcast-updates, ruled general-broadcast 2026-07-23): rank-0
        // writes one scalar per index element (bincount form); extent-1 broadcasts
        // too; an incompatible extent stays a typed decline.
        let scat = Node::Scatter {
            dest: 0,
            index: IndexRef::Slot(0),
            updates: 1,
            axis: 0,
            combine: Combine::AtomicAdd,
        };
        let dest = t(&[0.0, 0.0, 0.0], &[3]);
        let idx = IndexTensor::new(vec![0, 1, 1, 9], &[4], Dtype::I64).unwrap(); // 9 OOB → skipped
                                                                                 // rank-0 const updates.
        let dag = FlatDag::new(vec![Node::Bind(0), Node::Const(2.0), scat.clone()], vec![2]);
        let r = eval_recipe(
            &dag,
            std::slice::from_ref(&dest),
            &[],
            std::slice::from_ref(&idx),
        )
        .unwrap();
        close(r.outputs[0].as_slice(), &[2.0, 4.0, 0.0]);
        // extent-1 input updates (general broadcast, not a rank-0 carve-out).
        let dag1 = FlatDag::new(vec![Node::Bind(0), Node::Bind(1), scat.clone()], vec![2]);
        let one = t(&[3.0], &[1]);
        let r1 = eval_recipe(&dag1, &[dest.clone(), one], &[], std::slice::from_ref(&idx)).unwrap();
        close(r1.outputs[0].as_slice(), &[3.0, 6.0, 0.0]);
        // extent 2 vs write shape [4]: not broadcast-compatible → typed decline.
        let dag2 = FlatDag::new(vec![Node::Bind(0), Node::Bind(1), scat], vec![2]);
        let bad = t(&[1.0, 1.0], &[2]);
        assert!(matches!(
            eval_recipe(&dag2, &[dest, bad], &[], &[idx]),
            Err(Error::BroadcastIncompatible)
        ));
    }

    #[test]
    fn iota_materializes_coordinates() {
        // iota over the shape of Bind(0) ([2,3]): axis 1 → column ids, axis 0 →
        // row ids; an out-of-range axis declines.
        let mk = |axis| FlatDag::new(vec![Node::Bind(0), Node::Iota { like: 0, axis }], vec![1]);
        let x = t(&[0.0; 6], &[2, 3]);
        let r1 = eval_recipe(&mk(1), std::slice::from_ref(&x), &[], &[]).unwrap();
        close(r1.outputs[0].as_slice(), &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        assert_eq!(r1.dets[1], DetClass::ExactByte);
        let r0 = eval_recipe(&mk(0), std::slice::from_ref(&x), &[], &[]).unwrap();
        close(r0.outputs[0].as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert!(matches!(
            eval_recipe(&mk(2), &[x], &[], &[]),
            Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
        ));
    }

    #[test]
    fn index_source_invalid_is_typed() {
        // IndexRef::Node naming a node with no index lane → the dedicated error.
        let dag = FlatDag::new(
            vec![
                Node::Bind(0),
                Node::Gather {
                    data: 0,
                    index: IndexRef::Node(0), // Bind has no index-lane output
                    axis: 0,
                    oob: OobPolicy::ZeroFill,
                    base: None,
                },
            ],
            vec![1],
        );
        let data = t(&[1.0], &[1]);
        assert!(matches!(
            eval_recipe(&dag, std::slice::from_ref(&data), &[], &[]),
            Err(Error::IndexSourceInvalid { node: 0 })
        ));
        // index_outputs naming a non-index node (or out of range) → same error.
        let dag2 = FlatDag {
            nodes: vec![Node::Bind(0)],
            outputs: vec![0],
            index_outputs: vec![0],
        };
        assert!(matches!(
            eval_recipe(&dag2, std::slice::from_ref(&data), &[], &[]),
            Err(Error::IndexSourceInvalid { node: 0 })
        ));
        let dag3 = FlatDag {
            nodes: vec![Node::Bind(0)],
            outputs: vec![0],
            index_outputs: vec![9],
        };
        assert!(matches!(
            eval_recipe(&dag3, &[data], &[], &[]),
            Err(Error::IndexSourceInvalid { node: 9 })
        ));
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
                nodes.push(Node::Apply {
                    op: Op::Neg,
                    children: vec![i + 1],
                });
            } else {
                nodes.push(Node::Bind(0));
            }
        }
        let dag = FlatDag::new(nodes, vec![0]);
        let x = t(&[5.0], &[]);
        let r = eval_recipe(&dag, &[x], &[], &[]).unwrap();
        // 19_999 negations (odd) of 5.0 → -5.0. No overflow, returns a value.
        close(r.outputs[0].as_slice(), &[-5.0]);
    }

    #[test]
    fn cycle_and_bad_index_decline_not_panic() {
        // a self-referential node must decline, not loop forever.
        let dag = FlatDag::new(
            vec![Node::Apply {
                op: Op::Add,
                children: vec![0, 0],
            }],
            vec![0],
        );
        assert!(eval_recipe::<f64>(&dag, &[], &[], &[]).is_err());
        // an out-of-range child index is an error.
        let dag2 = FlatDag::new(
            vec![Node::Apply {
                op: Op::Neg,
                children: vec![9],
            }],
            vec![0],
        );
        assert!(eval_recipe::<f64>(&dag2, &[], &[], &[]).is_err());
    }
}
