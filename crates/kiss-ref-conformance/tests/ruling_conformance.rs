//! Ruling-conformance vectors — the seven KISS-Ops clauses ruled 2026-07-23
//! (KISS #75/#76), pinned as the kiss-ref reference mirror of KISS-Conform's
//! `test_ops_*` (KISS-Ops §9.1 clause→test traceability). Every value here is
//! hand-derived from the §6.11 / §6.19 semantics FIRST — the evaluator is
//! CHECKED against it, never the source of it (conformance role, ratified
//! 2026-07-21). Each test cites its clause id.
//!
//! Comparison mode follows the §6.0 determinism class of the pinned op:
//! `ExactByte` → per-element raw-bit (`to_bits`) equality (§6.0-0002); float
//! `atomic_add` / `sum` → `OrderInvariantNondeterministic`, compared under
//! tolerance, NEVER byte-exact (§6.0-0004). The §6.11 structural atoms are
//! pinned at the kernel (op) level — the reference mirror of KISS-Conform's
//! op tests — including the §6.13 non-primitives (`index_select`/`scatter_add`)
//! the rulings flow through. The §6.19 recipe-encoding clauses are pinned at the
//! DECODED flat-DAG level (`eval_recipe`): kiss-ref walks the logical DAG, so
//! "wire form" is pinned as the decoded `IndexRef` / `index_outputs` rules.

use kiss_classify_vocab::Dtype;
use kiss_ref_core::kernels::{gather, prefix_scan, scatter, sort_network};
use kiss_ref_core::tensor_ops::{index_select, scatter_add};
use kiss_ref_core::{
    eval_recipe, Combine, DetClass, Direction, Error, FlatDag, IndexRef, IndexTensor, Monoid,
    Node, OobPolicy, Tensor,
};

// ---- helpers (ops_conformance idiom: raw-bit for ExactByte, tol for OIN) -----

fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), shape).unwrap_or_else(|e| panic!("tensor {shape:?}: {e:?}"))
}

fn ix(data: &[i64], shape: &[usize]) -> IndexTensor {
    IndexTensor::new(data.to_vec(), shape, Dtype::I64)
        .unwrap_or_else(|e| panic!("index {shape:?}: {e:?}"))
}

/// ExactByte class → per-element raw-bit equality (§6.0-0002): distinguishes
/// −0.0 from +0.0 and any NaN payload, so a value-compare cannot pass vacuously.
fn assert_bits(got: &Tensor<f64>, want: &[f64], shape: &[usize]) {
    assert_eq!(got.shape(), shape, "output shape");
    assert_eq!(got.as_slice().len(), want.len(), "element count");
    for (i, (&g, &w)) in got.as_slice().iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "elem {i}: got {g} want {w} (raw-bit)");
    }
}

/// OrderInvariantNondeterministic class → tolerance compare, never byte-exact
/// (§6.0-0004). The values here are exact under any fold order, but the class
/// describes the candidate's reassociation freedom, so OIN is never bit-pinned.
fn assert_close(got: &Tensor<f64>, want: &[f64], shape: &[usize], tol: f64) {
    assert_eq!(got.shape(), shape, "output shape");
    assert_eq!(got.as_slice().len(), want.len(), "element count");
    for (i, (&g, &w)) in got.as_slice().iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "elem {i}: got {g} want {w} (abs tol {tol})");
    }
}

// ---- §6.11 structural atoms (kernel/op level) --------------------------------

#[test]
fn test_ops_gather_skip_base_dynamic() {
    // KISS-OPS-6.11-0015 (ruled 2026-07-23, KISS PR #75 Gap 1 option 1): the
    // gather-skip `base` requirement is DYNAMIC. `skip` + `base:None` is a legal
    // plain read while every index is in range — the §6.13 `index_select`
    // non-primitive IS `gather(oob=skip, base=None)` and relies on exactly that;
    // only an ACTUALLY-OOB read with no base declines (Error::GatherSkipNoBase);
    // with a base, the base's (broadcast) value is kept raw-bit at the skipped
    // position.
    let data = t(&[10.0, 20.0, 30.0], &[3]);

    // (a) skip + no base + all-in-range → Ok plain read (ExactByte raw-bit).
    // idx [2,0,1] over [10,20,30] → [30,10,20].
    let in_range = ix(&[2, 0, 1], &[3]);
    let g = gather(&data.view(), &in_range, 0, OobPolicy::Skip, None)
        .expect("skip + no base + in-range must be Ok");
    assert_bits(&g, &[30.0, 10.0, 20.0], &[3]);
    let sel = index_select(&data.view(), &in_range, 0).expect("index_select in-range must be Ok");
    assert_bits(&sel, &[30.0, 10.0, 20.0], &[3]);

    // (b) skip + no base + an actually-OOB index (9) → the typed dynamic decline,
    // both through the raw kernel and through the §6.13 index_select wrapper.
    let oob = ix(&[2, 9, 0], &[3]);
    assert_eq!(
        gather(&data.view(), &oob, 0, OobPolicy::Skip, None).err(),
        Some(Error::GatherSkipNoBase)
    );
    assert_eq!(index_select(&data.view(), &oob, 0).err(), Some(Error::GatherSkipNoBase));

    // (c) skip + base + OOB → the base value kept at the skipped position. base is
    // rank-0 (−0.0) → broadcast to the output shape; the −0.0 sign bit survives
    // (raw-bit move). idx [2,9,0] → [data[2], base, data[0]] = [30, −0.0, 10].
    let base = t(&[-0.0], &[]);
    let gb = gather(&data.view(), &oob, 0, OobPolicy::Skip, Some(&base.view()))
        .expect("skip + base must be Ok");
    assert_bits(&gb, &[30.0, -0.0, 10.0], &[3]);
    assert_eq!(gb.as_slice()[1].to_bits(), 0x8000_0000_0000_0000, "base −0.0 kept raw-bit");
}

#[test]
fn test_ops_scatter_dest_operand() {
    // KISS-OPS-6.11-0016 (ruled 2026-07-23): scatter has an EXPLICIT `dest`
    // operand; the output shape equals `dest.shape`; unwritten and OOB-skipped
    // positions RETAIN the dest value; `atomic_add` accumulates ONTO the dest
    // (not from zero).

    // (a) retention + output shape, via the ExactByte `Assign` combine (raw-bit).
    // dest [100,−0.0,300]; index [0,9] (idx 9 ≥ extent 3 → OOB, skipped);
    // updates [7,5]. Only bin0 is written (←7); the −0.0 at bin1 and 300 at bin2
    // are never touched → dest kept; output shape == dest.shape == [3].
    let dest = t(&[100.0, -0.0, 300.0], &[3]);
    let idx = ix(&[0, 9], &[2]);
    let upd = t(&[7.0, 5.0], &[2]);
    let out = scatter(dest, &idx, &upd.view(), 0, Combine::Assign).expect("assign must be Ok");
    assert_bits(&out, &[7.0, -0.0, 300.0], &[3]);
    assert_eq!(out.as_slice()[1].to_bits(), 0x8000_0000_0000_0000, "unwritten −0.0 dest kept");
    assert_eq!(out.shape(), &[3], "output shape == dest.shape");

    // (b) atomic_add accumulates ONTO dest (OIN root → tolerance). dest
    // [100,200,300]; index [0,0,9]; updates [1,2,5]: bin0 ← 100 + 1 + 2 = 103;
    // idx-9 update (9 ≥ 3) skipped; bins 1,2 untouched → 200, 300.
    let dest2 = t(&[100.0, 200.0, 300.0], &[3]);
    let idx2 = ix(&[0, 0, 9], &[3]);
    let upd2 = t(&[1.0, 2.0, 5.0], &[3]);
    let out2 = scatter_add(dest2, &idx2, &upd2.view(), 0).expect("scatter_add must be Ok");
    assert_close(&out2, &[103.0, 200.0, 300.0], &[3], 1e-12);
}

#[test]
fn test_ops_scatter_updates_broadcast() {
    // KISS-OPS-6.11-0017 (ruled 2026-07-23): scatter `updates` GENERAL-broadcast
    // to the write shape (dest.shape with [axis] = index.len()) under the ordinary
    // §6.11-0001 rules — rank-0 is the DEGENERATE case, NOT a rank-0-only carve
    // out; an incompatible extent is the typed Error::BroadcastIncompatible.
    // dest [0,0,0]; index [0,1,1,9] (idx 9 OOB → skipped); write shape = [4].
    let dest = t(&[0.0, 0.0, 0.0], &[3]);
    let idx = ix(&[0, 1, 1, 9], &[4]);

    // (a) rank-0 updates 2.0 → broadcast [2,2,2,2] (bincount form): bin0 ← 2;
    // bin1 ← 2 + 2 = 4; idx-9 skipped → bin2 = 0.
    let r0 = scatter_add(dest.clone(), &idx, &t(&[2.0], &[]).view(), 0)
        .expect("rank-0 updates must broadcast");
    assert_close(&r0, &[2.0, 4.0, 0.0], &[3], 1e-12);

    // (b) extent-1 updates [3.0] → general broadcast [3,3,3,3] (proves NOT
    // rank-0-only): bin0 ← 3; bin1 ← 3 + 3 = 6; idx-9 skipped → bin2 = 0.
    let r1 = scatter_add(dest.clone(), &idx, &t(&[3.0], &[1]).view(), 0)
        .expect("extent-1 updates must broadcast");
    assert_close(&r1, &[3.0, 6.0, 0.0], &[3], 1e-12);

    // (c) extent-2 updates vs write shape [4] → 2 ≠ 4 and 2 ≠ 1 → typed decline.
    let bad = scatter_add(dest, &idx, &t(&[1.0, 1.0], &[2]).view(), 0);
    assert_eq!(bad.err(), Some(Error::BroadcastIncompatible));
}

#[test]
fn test_ops_structural_empty_axis() {
    // KISS-OPS-6.11-0018 (ruled 2026-07-23 — THE previously-untested gap): the
    // structural atoms over an empty axis produce the shape-implied result.

    // (1) prefix_scan over an empty axis → zero positions. x shape [0], axis 0.
    let s = prefix_scan(&t(&[], &[0]).view(), Monoid::Sum, 0, false)
        .expect("empty-axis scan must be Ok");
    assert_eq!(s.shape(), &[0], "scan preserves the empty axis");
    assert!(s.as_slice().is_empty(), "zero scanned positions");

    // (2) gather with an empty index → empty gathered axis (the gathered axis
    // becomes 0; trailing data axes preserved). data [3,2], axis 0, index []
    // → out shape [0,2]. Skip + base:None is legal here — no OOB read occurs, so
    // the DYNAMIC base rule (6.11-0015) does not fire.
    let data = t(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2]);
    let g = gather(&data.view(), &ix(&[], &[0]), 0, OobPolicy::Skip, None)
        .expect("empty-index gather must be Ok");
    assert_eq!(g.shape(), &[0, 2], "gathered axis empty, trailing axis kept");
    assert!(g.as_slice().is_empty());

    // (3) scatter with an empty index → dest unchanged (raw-bit; −0.0 kept). No
    // write coordinate exists, so the dest buffer is returned verbatim.
    let dest = t(&[100.0, -0.0, 300.0], &[3]);
    let out = scatter_add(dest, &ix(&[], &[0]), &t(&[], &[0]).view(), 0)
        .expect("empty-index scatter must be Ok");
    assert_bits(&out, &[100.0, -0.0, 300.0], &[3]);

    // (4) sort_network over an empty row → empty perm + empty index vector (i64).
    // keys [2,0], axis 1: two empty rows → 0 total elements.
    let (vals, idxs) = sort_network(&t(&[], &[2, 0]).view(), 1, Direction::Asc)
        .expect("empty-row sort must be Ok");
    assert_eq!(vals.shape(), &[2, 0]);
    assert!(vals.as_slice().is_empty(), "empty value permutation");
    assert_eq!(idxs.shape(), &[2, 0]);
    assert!(idxs.as_slice().is_empty(), "empty index vector");
    assert_eq!(idxs.dtype(), Dtype::I64, "empty sort index vector is still i64");
}

#[test]
fn test_ops_sort_index_output_i64() {
    // KISS-OPS-6.11-0019 (ruled 2026-07-23): the sort_network original-index
    // output vector dtype is i64. keys [3,1,2] asc → values [1,2,3] (ExactByte
    // raw-bit permutation), original indices [1,2,0].
    let (vals, idxs) = sort_network(&t(&[3.0, 1.0, 2.0], &[3]).view(), 0, Direction::Asc)
        .expect("sort must be Ok");
    assert_bits(&vals, &[1.0, 2.0, 3.0], &[3]);
    assert_eq!(idxs.as_slice(), &[1, 2, 0], "original-index permutation");
    assert_eq!(idxs.dtype(), Dtype::I64, "sort index output dtype is i64");
}

// ---- §6.19 recipe-encoding (decoded flat-DAG level) --------------------------

#[test]
fn test_ops_index_ref_wire_form() {
    // KISS-OPS-6.19-0039 (decoded semantics — no wire; kiss-ref walks the logical
    // flat-DAG, so the "wire form" is pinned as the decoded IndexRef rules): an
    // EXTERNAL index operand is an IndexRef::Slot resolving to indices[slot]; an
    // INTERNAL index operand is an IndexRef::Node resolving to a node's index-lane
    // output; a Node ref at a non-index producer, or an out-of-range node id, is
    // the typed Error::IndexSourceInvalid.

    // (a) Slot → the external indices[0] operand. gather ZeroFill: idx [2,0,9]
    // over [10,20,30] → [30,10,+0.0] (idx 9 OOB → zero-fill).
    let dag_slot = FlatDag::new(
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
    let r = eval_recipe(
        &dag_slot,
        &[t(&[10.0, 20.0, 30.0], &[3])],
        &[],
        &[ix(&[2, 0, 9], &[3])],
    )
    .expect("slot form must evaluate");
    assert_bits(&r.outputs[0], &[30.0, 10.0, 0.0], &[3]);
    assert_eq!(r.dets[1], DetClass::ExactByte, "external Slot index → exact gather");

    // (b) Node → node 2's (sort_network) index lane, a real scheduling edge.
    // keys [3,1,2] → perm [1,2,0]; gather data [30,10,20] by that perm →
    // [data[1],data[2],data[0]] = [10,20,30].
    let dag_node = FlatDag::new(
        vec![
            Node::Bind(0), // keys
            Node::Bind(1), // data
            Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc },
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
    let r = eval_recipe(
        &dag_node,
        &[t(&[3.0, 1.0, 2.0], &[3]), t(&[30.0, 10.0, 20.0], &[3])],
        &[],
        &[],
    )
    .expect("node form must evaluate");
    assert_bits(&r.outputs[0], &[10.0, 20.0, 30.0], &[3]);
    assert_eq!(r.dets[3], DetClass::ExactByte, "exact chain via IndexRef::Node");

    // (c) Node at a non-index producer (Bind) → typed decline.
    let dag_bad = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Gather {
                data: 0,
                index: IndexRef::Node(0), // Bind has no index lane
                axis: 0,
                oob: OobPolicy::ZeroFill,
                base: None,
            },
        ],
        vec![1],
    );
    assert_eq!(
        eval_recipe(&dag_bad, &[t(&[1.0], &[1])], &[], &[]).err(),
        Some(Error::IndexSourceInvalid { node: 0 })
    );

    // (d) Node out of range (id 9 ≥ nodes.len() 2) → typed decline.
    let dag_oor = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Gather {
                data: 0,
                index: IndexRef::Node(9),
                axis: 0,
                oob: OobPolicy::ZeroFill,
                base: None,
            },
        ],
        vec![1],
    );
    assert_eq!(
        eval_recipe(&dag_oor, &[t(&[1.0], &[1])], &[], &[]).err(),
        Some(Error::IndexSourceInvalid { node: 9 })
    );
}

#[test]
fn test_ops_index_outputs_root_list() {
    // KISS-OPS-6.19-0040 (decoded): FlatDag::index_outputs is a second root list
    // exporting the named nodes' index-lane outputs IN LIST ORDER; a listed
    // non-index node (or out-of-range id) is Error::IndexSourceInvalid; an empty
    // list is value-lane-only.

    // (a) two independent sorts, exported in REVERSED order to prove the export
    // order follows the list, not node order.
    //   k1 [3,1,2] → node2 perm [1,2,0], vals [1,2,3];
    //   k2 [10,30,20] → node3 perm [0,2,1], vals [10,20,30].
    // outputs [2,3] → values [[1,2,3],[10,20,30]];
    // index_outputs [3,2] → [ node3 perm [0,2,1], node2 perm [1,2,0] ].
    let dag = FlatDag {
        nodes: vec![
            Node::Bind(0), // k1
            Node::Bind(1), // k2
            Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc }, // 2
            Node::SortNetwork { keys: 1, axis: 0, dir: Direction::Asc }, // 3
        ],
        outputs: vec![2, 3],
        index_outputs: vec![3, 2],
    };
    let r = eval_recipe(
        &dag,
        &[t(&[3.0, 1.0, 2.0], &[3]), t(&[10.0, 30.0, 20.0], &[3])],
        &[],
        &[],
    )
    .expect("dual-sort export must evaluate");
    assert_bits(&r.outputs[0], &[1.0, 2.0, 3.0], &[3]);
    assert_bits(&r.outputs[1], &[10.0, 20.0, 30.0], &[3]);
    assert_eq!(r.index_outputs.len(), 2);
    assert_eq!(r.index_outputs[0].as_slice(), &[0, 2, 1], "node3 perm exported first");
    assert_eq!(r.index_outputs[1].as_slice(), &[1, 2, 0], "node2 perm exported second");
    assert_eq!(r.index_outputs[0].dtype(), Dtype::I64);
    assert_eq!(r.index_outputs[1].dtype(), Dtype::I64);
    assert_eq!(r.dets[2], DetClass::ExactByte);
    assert_eq!(r.dets[3], DetClass::ExactByte);

    // (b) a listed non-index node → typed decline; and an out-of-range id.
    let bad_nonindex =
        FlatDag { nodes: vec![Node::Bind(0)], outputs: vec![0], index_outputs: vec![0] };
    assert_eq!(
        eval_recipe(&bad_nonindex, &[t(&[1.0], &[1])], &[], &[]).err(),
        Some(Error::IndexSourceInvalid { node: 0 })
    );
    let bad_oor =
        FlatDag { nodes: vec![Node::Bind(0)], outputs: vec![0], index_outputs: vec![9] };
    assert_eq!(
        eval_recipe(&bad_oor, &[t(&[1.0], &[1])], &[], &[]).err(),
        Some(Error::IndexSourceInvalid { node: 9 })
    );

    // (c) empty index_outputs (FlatDag::new default) → value-lane-only: a sort
    // node whose index lane is simply not exported.
    let value_only = FlatDag::new(
        vec![Node::Bind(0), Node::SortNetwork { keys: 0, axis: 0, dir: Direction::Asc }],
        vec![1],
    );
    let r = eval_recipe(&value_only, &[t(&[3.0, 1.0, 2.0], &[3])], &[], &[])
        .expect("value-only must evaluate");
    assert_bits(&r.outputs[0], &[1.0, 2.0, 3.0], &[3]);
    assert!(r.index_outputs.is_empty(), "no index lane exported");
}
