//! Metamorphic / invariant conformance — relations that MUST hold between the
//! outputs of related recipes, checked without a hand-typed golden for the
//! relation itself.
//!
//! A golden pins one input→output pair; a metamorphic relation pins a whole family
//! (softmax is shift-invariant for EVERY input; flip is its own inverse on EVERY
//! tensor). These catch the bugs a fixed golden can miss — a wrong reduction axis,
//! a mis-broadcast keepdim, a saturating integer path — because the relation is
//! violated across the family even where one chosen point happens to look right.
//! kiss-ref stays the differential target/proxy, never the oracle: the relations
//! are KISS-Ops first principles (softmax §6.13 shift-invariance, flip §6.11
//! involution, the §6.11-0007 direction duality, the §6.0-0004 integer-exactness
//! that makes int matmul a ring homomorphism), and the evaluator is CHECKED
//! against them.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_recipe, eval_recipe_int, Direction, FlatDag, Monoid, Node, Tensor};

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), shape).expect("f64 fixture")
}
fn ti(data: &[i128], shape: &[usize]) -> Tensor<i128> {
    Tensor::from_vec(data.to_vec(), shape).expect("i128 fixture")
}

/// The row-softmax sub-DAG (R2 shape): max-shift / exp / sum / div over the last
/// axis. Shared by the shift-invariance relation.
fn softmax_rows_dag() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Reduce {
                monoid: Monoid::Max,
                axes: vec![1],
                keepdim: true,
                child: 0,
            }, // rowmax
            Node::Apply {
                op: Op::Sub,
                children: vec![0, 1],
            }, // shifted
            Node::Apply {
                op: Op::Exp,
                children: vec![2],
            },
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 3,
            }, // denom
            Node::Apply {
                op: Op::Div,
                children: vec![3, 4],
            },
        ],
        vec![5],
    )
}

#[test]
fn metamorphic_softmax_is_shift_invariant() {
    // softmax(x + c·1) == softmax(x) for any per-row-constant c — the property the
    // max-shift EXISTS to guarantee (KISS §6.13). With integer-valued inputs the
    // shift cancels bit-for-bit: shifted = (xᵢ+c) − (max+c) = xᵢ − max exactly, so
    // BOTH runs feed the exp identical bytes and must return identical bytes.
    // Teeth: a per-column (wrong-axis) max, or a keepdim rowmax that fails to
    // broadcast back over the row, breaks the cancellation and the equality.
    let dag = softmax_rows_dag();
    let x = t64(&[1.0, 2.0, 3.0, 0.0, -1.0, -2.0], &[2, 3]);
    // c = 1e6, added to every element — huge enough that a NON-shifted softmax
    // would overflow exp to +inf, yet the shifted DAG is exactly invariant.
    let c = 1_000_000.0;
    let x_shift = t64(
        &[1.0 + c, 2.0 + c, 3.0 + c, 0.0 + c, -1.0 + c, -2.0 + c],
        &[2, 3],
    );

    let base = eval_recipe(&dag, std::slice::from_ref(&x), &[], &[]).expect("softmax(x)");
    let shifted =
        eval_recipe(&dag, std::slice::from_ref(&x_shift), &[], &[]).expect("softmax(x+c)");

    let b = base.outputs[0].as_slice();
    let s = shifted.outputs[0].as_slice();
    assert_eq!(b.len(), s.len());
    for (i, (&g, &h)) in b.iter().zip(s).enumerate() {
        assert_eq!(
            g.to_bits(),
            h.to_bits(),
            "elem {i}: softmax(x)={g} != softmax(x+c)={h} (shift not invariant)"
        );
    }
    // Sanity: it really is a softmax — each row sums to 1 (within fp tolerance).
    assert!((b[0] + b[1] + b[2] - 1.0).abs() <= 1e-12, "row0 sums to 1");
    assert!((b[3] + b[4] + b[5] - 1.0).abs() <= 1e-12, "row1 sums to 1");
}

#[test]
fn metamorphic_flip_is_its_own_inverse() {
    // flip∘flip = identity, bit-for-bit — flip is pure data movement (KISS §6.11),
    // so −0.0 and a NaN payload must survive both passes unchanged. Teeth: an
    // off-by-one in the reversed index, or a wrong axis, makes the double-flip
    // land elements elsewhere.
    // Output BOTH the single flip (node 1) and the double flip (node 2): the
    // double proves involution, the single proves flip is not a silent no-op
    // (without which `flip∘flip == x` would pass vacuously for an identity flip).
    let dag = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Flip { child: 0, axis: 1 },
            Node::Flip { child: 1, axis: 1 },
        ],
        vec![2, 1],
    );
    let x = t64(&[1.0, -0.0, f64::NAN, 4.0, 5.0, 6.0], &[2, 3]);
    let r = eval_recipe(&dag, std::slice::from_ref(&x), &[], &[]).expect("flip∘flip");
    // node 2: flip∘flip == x, bit-for-bit.
    for (i, (&g, &w)) in r.outputs[0].as_slice().iter().zip(x.as_slice()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "elem {i}: flip∘flip changed the raw bits"
        );
    }
    // node 1: a single flip really reverses each row (x is not a palindrome).
    let single = r.outputs[1].as_slice();
    let want = [f64::NAN, -0.0, 1.0, 6.0, 5.0, 4.0];
    for (i, (&g, &w)) in single.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "elem {i}: single flip must reverse"
        );
    }
}

#[test]
fn metamorphic_sort_desc_equals_reverse_of_sort_asc() {
    // For DISTINCT keys, descending sort == reverse(ascending sort)
    // (KISS-OPS-6.11-0007 direction duality). Distinctness matters: a STABLE sort
    // keeps ties in original order in BOTH directions, so reverse(asc) would flip
    // tied runs that desc leaves in place — the relation is exact only with no
    // ties. Teeth: a `Direction::Desc` that silently behaves like `Asc` makes the
    // two sides disagree on every element.
    let asc_then_flip = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::SortNetwork {
                keys: 0,
                axis: 0,
                dir: Direction::Asc,
            },
            Node::Flip { child: 1, axis: 0 },
        ],
        vec![2],
    );
    let desc = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::SortNetwork {
                keys: 0,
                axis: 0,
                dir: Direction::Desc,
            },
        ],
        vec![1],
    );
    let x = t64(&[3.0, 1.0, 4.0, 2.0, 5.0], &[5]); // strictly distinct

    let a = eval_recipe(&asc_then_flip, std::slice::from_ref(&x), &[], &[]).expect("reverse(asc)");
    let d = eval_recipe(&desc, std::slice::from_ref(&x), &[], &[]).expect("desc");
    let av = a.outputs[0].as_slice();
    let dv = d.outputs[0].as_slice();
    assert_eq!(av, &[5.0, 4.0, 3.0, 2.0, 1.0], "reverse(asc) shape/values");
    for (i, (&g, &w)) in dv.iter().zip(av).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "elem {i}: desc != reverse(asc)");
    }
}

#[test]
fn metamorphic_int_matmul_is_exactly_distributive() {
    // A·(B+C) == A·B + A·C, BYTE-EXACT at s8 — integer two's-complement arithmetic
    // is a ring, so matmul is EXACTLY linear (unlike the float lane, whose rounding
    // makes the two sides differ and forces the OrderInvariantNondeterministic
    // class). Inputs are chosen to overflow s8 with MIXED signs so the ring
    // identity holds under wrapping while a SATURATING implementation would
    // diverge: A·B = 200→(sat)127, A·C = −180→(sat)−128, sat-sum = −1, but
    // A·(B+C) = 20 — so saturation breaks the equality this test asserts.
    let s8 = Dtype::S8;
    let a = ti(&[1, 1], &[1, 2]);
    let b = ti(&[100, 100], &[2, 1]);
    let c = ti(&[-90, -90], &[2, 1]);

    // LHS: A·(B+C).
    let lhs_dag = FlatDag::new(
        vec![
            Node::Bind(0), // A
            Node::Bind(1), // B
            Node::Bind(2), // C
            Node::Apply {
                op: Op::Add,
                children: vec![1, 2],
            }, // B+C
            Node::Matmul { lhs: 0, rhs: 3 },
        ],
        vec![4],
    );
    // RHS: A·B + A·C.
    let rhs_dag = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Bind(1),
            Node::Bind(2),
            Node::Matmul { lhs: 0, rhs: 1 }, // A·B
            Node::Matmul { lhs: 0, rhs: 2 }, // A·C
            Node::Apply {
                op: Op::Add,
                children: vec![3, 4],
            },
        ],
        vec![5],
    );

    let inputs = [a, b, c];
    let lhs = eval_recipe_int(&lhs_dag, &[s8; 5], &inputs, &[], &[]).expect("A·(B+C)");
    let rhs = eval_recipe_int(&rhs_dag, &[s8; 6], &inputs, &[], &[]).expect("A·B + A·C");

    let l = lhs.outputs[0].as_slice();
    let r = rhs.outputs[0].as_slice();
    // Both wrap to the same s8 value; pin it (=20) so a both-wrong-but-equal bug
    // is caught too.
    assert_eq!(l, &[20], "A·(B+C) must wrap to 20 at s8");
    assert_eq!(r, &[20], "A·B + A·C must wrap to 20 at s8");
    assert_eq!(l, r, "integer matmul must be exactly distributive");
}
