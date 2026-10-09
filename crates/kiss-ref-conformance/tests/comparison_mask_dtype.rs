// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.2-0005 as rewritten by KISS #516: a comparison op produces a **`bool` mask** — dtype
//! `bool`, one byte per element, byte `1` or `0` — whatever the operands' compute dtype. It is NOT
//! encoded in the compute dtype, so an `f32` comparison yields a one-byte mask, not a 4-byte
//! `1.0`/`0.0`. A mask that feeds an arithmetic or `reduce` atom is read as the unsigned byte `0`/`1`.
//!
//! kiss-ref evaluates every value in the compute lane `T` (so a decomposition can compose a mask into
//! `select`/`mul`), which is why `eval_op` still returns the mask AS a `T` holding exactly `1`/`0`. The
//! OBSERVABLE dtype of a comparison result is what these tests pin: `result_dtype`, the static output
//! dtypes of a recipe, and the byte form of a mask tensor.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_recipe, mask_bytes, result_dtype, FlatDag, Node, Tensor};

const CMPS: [Op; 6] = [
    Op::CmpEq,
    Op::CmpNe,
    Op::CmpLt,
    Op::CmpLe,
    Op::CmpGt,
    Op::CmpGe,
];

#[test]
fn only_the_six_comparison_ops_yield_a_bool_mask() {
    for op in Op::ALL {
        for compute in [Dtype::F32, Dtype::F64, Dtype::F16, Dtype::Bf16, Dtype::I32] {
            let want = if CMPS.contains(op) {
                Dtype::Bool
            } else {
                compute
            };
            assert_eq!(result_dtype(*op, compute), want, "{op:?} on {compute:?}");
        }
    }
    // the mask is one byte per element, whatever the operand width.
    assert_eq!(Dtype::Bool.bits(), 8);
    assert_eq!(
        result_dtype(Op::CmpLt, Dtype::F64).bits(),
        8,
        "an f64 comparison is NOT 64-bit"
    );
}

fn cmp_dag(root: Node) -> FlatDag {
    FlatDag::new(vec![Node::Bind(0), Node::Bind(1), root], vec![2])
}

#[test]
fn a_recipe_whose_root_is_a_comparison_reports_a_bool_output() {
    let dag = cmp_dag(Node::Apply {
        op: Op::CmpLt,
        children: vec![0, 1],
    });
    assert_eq!(dag.output_dtypes(Dtype::F32), vec![Dtype::Bool]);
    // control: an arithmetic root keeps the compute dtype.
    let add = cmp_dag(Node::Apply {
        op: Op::Add,
        children: vec![0, 1],
    });
    assert_eq!(add.output_dtypes(Dtype::F32), vec![Dtype::F32]);
}

/// A 4-node DAG `[Bind0, Bind1, cmp(0,1), root]` and the output dtype of `root` at f32.
fn dtype_with_a_cmp_feeding(root: Node) -> Vec<Dtype> {
    FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Bind(1),
            Node::Apply {
                op: Op::CmpGt,
                children: vec![0, 1],
            },
            root,
        ],
        vec![3],
    )
    .output_dtypes(Dtype::F32)
}

#[test]
fn a_mask_moved_unchanged_stays_a_mask() {
    // flip is a pure move: the mask stays a mask.
    assert_eq!(
        dtype_with_a_cmp_feeding(Node::Flip { child: 2, axis: 0 }),
        vec![Dtype::Bool]
    );
}

#[test]
fn a_computed_use_of_a_mask_is_a_compute_value() {
    // a mask feeding arithmetic is read as the unsigned byte 0/1 and the result is a compute value.
    let mul = Node::Apply {
        op: Op::Mul,
        children: vec![2, 0],
    };
    assert_eq!(dtype_with_a_cmp_feeding(mul), vec![Dtype::F32]);
    // select returns one of its value arms: the arm's dtype, not the condition's.
    let select = Node::Apply {
        op: Op::Select,
        children: vec![2, 0, 1],
    };
    assert_eq!(dtype_with_a_cmp_feeding(select), vec![Dtype::F32]);
}

#[test]
fn a_mask_tensor_is_one_byte_per_element() {
    let a = Tensor::from_vec(vec![1.0f32, 5.0, f32::NAN, 2.0], &[4]).unwrap();
    let b = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], &[4]).unwrap();
    let dag = cmp_dag(Node::Apply {
        op: Op::CmpLt,
        children: vec![0, 1],
    });
    let out = eval_recipe::<f32>(&dag, &[a, b], &[], &[]).unwrap();
    let t = &out.outputs[0];
    assert_eq!(
        t.as_slice(),
        &[1.0, 0.0, 0.0, 0.0],
        "values in the compute lane, NaN compares false"
    );
    let bytes = mask_bytes(t);
    assert_eq!(bytes, vec![1u8, 0, 0, 0]);
    assert_eq!(bytes.len(), 4, "4 elements -> 4 bytes, not 16");
}
