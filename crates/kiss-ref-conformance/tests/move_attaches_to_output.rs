// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.16-0009 / -0010 / -0011 on the recipe path: where the **move obligation attaches**.
//!
//! §6.16-0011: the obligation attaches to the value that reaches the op's OBSERVABLE OUTPUT, never an
//! internal intermediate. Trace the whole path from the inputs to that output, the fold itself included:
//! if EVERY transformation on it is a move (bit-preserving, at most a sign-bit edit), the output's bits are
//! preserved exactly (§6.16-0009); if ANY is arithmetic, the output is a computed value and §6.16-0010
//! applies (a quiet NaN for a signaling operand). An implementation MUST NOT classify by access variant, by
//! the fold operator alone, or by the epilogue alone — both directions fail:
//!
//! - a **max**-reduction under an arithmetic epilogue is a COMPUTED op, though its fold is a move;
//! - a **sum**-reduction is a COMPUTED op even when its epilogue is a pure move, because the fold is itself
//!   one of the transformations traced.
//!
//! Each lane that admits a signaling NaN is checked on the cases that separate those readings.

use half::{bf16, f16};
use kiss_ops_vocab::Op;
use kiss_ref_core::{eval_recipe, E5m2, FlatDag, Monoid, Node, ScalarFloat, Tensor};

/// Lane under test: how to read a result's raw bits and its quiet bit.
trait Lane: ScalarFloat {
    const NAME: &'static str;
    /// A signaling NaN with a recognizable payload.
    const SNAN: u64;
    const QUIET_BIT: u64;
    const BITS: u32;
    fn raw(self) -> u64;
}

macro_rules! lane {
    ($t:ty, $name:expr, $snan:expr, $qbit:expr, $bits:expr, $raw:expr) => {
        impl Lane for $t {
            const NAME: &'static str = $name;
            const SNAN: u64 = $snan;
            const QUIET_BIT: u64 = $qbit;
            const BITS: u32 = $bits;
            fn raw(self) -> u64 {
                ($raw)(self)
            }
        }
    };
}
lane!(f32, "f32", 0x7F80_0001, 1 << 22, 32, |x: f32| u64::from(
    x.to_bits()
));
lane!(f16, "f16", 0x7C01, 1 << 9, 16, |x: f16| u64::from(
    x.to_bits()
));
lane!(bf16, "bf16", 0x7F81, 1 << 6, 16, |x: bf16| u64::from(
    x.to_bits()
));
lane!(E5m2, "e5m2", 0x7D, 1 << 1, 8, |x: E5m2| u64::from(
    x.to_bits()
));

fn input<T: Lane>() -> Tensor<T> {
    Tensor::from_vec(
        vec![T::from_f64(1.0), T::from_bits(T::SNAN), T::from_f64(2.0)],
        &[3],
    )
    .unwrap()
}

fn reduce(monoid: Monoid) -> Node {
    Node::Reduce {
        monoid,
        axes: vec![0],
        keepdim: false,
        child: 0,
    }
}

/// Evaluate a one-input recipe whose last node is the root; return the first output element's raw bits.
fn out_bits<T: Lane>(nodes: Vec<Node>) -> u64 {
    let root = nodes.len() - 1;
    let dag = FlatDag::new(nodes, vec![root]);
    let out = eval_recipe::<T>(&dag, &[input::<T>()], &[], &[]).unwrap();
    out.outputs[0].as_slice()[0].raw()
}

fn is_quiet_nan<T: Lane>(bits: u64) -> bool {
    T::from_bits(bits).is_nan() && bits & T::QUIET_BIT != 0
}

fn check<T: Lane>() {
    let name = T::NAME;
    let snan = T::SNAN;
    // A max-reduction is a pure move: the signaling NaN's bits reach the output unchanged.
    let max = out_bits::<T>(vec![Node::Bind(0), reduce(Monoid::Max)]);
    assert_eq!(
        max, snan,
        "{name}: reduce(max) is a move, the sNaN must arrive bit-for-bit"
    );
    // A sum-reduction computes: a quiet NaN.
    let sum = out_bits::<T>(vec![Node::Bind(0), reduce(Monoid::Sum)]);
    assert!(
        is_quiet_nan::<T>(sum),
        "{name}: reduce(sum) computes, got {sum:#x}"
    );
    // A max-reduction under an ARITHMETIC epilogue is computed (fold is a move, epilogue is not).
    let max_then_mul = out_bits::<T>(vec![
        Node::Bind(0),
        reduce(Monoid::Max),
        Node::Const(1.0),
        Node::Apply {
            op: Op::Mul,
            children: vec![1, 2],
        },
    ]);
    assert!(
        is_quiet_nan::<T>(max_then_mul),
        "{name}: max then mul is computed, got {max_then_mul:#x}"
    );
    // A sum-reduction under a PURE-MOVE epilogue is still computed (the fold itself is arithmetic).
    let sum_then_neg = out_bits::<T>(vec![
        Node::Bind(0),
        reduce(Monoid::Sum),
        Node::Apply {
            op: Op::Neg,
            children: vec![1],
        },
    ]);
    let magnitude = sum_then_neg & !(1u64 << (T::BITS - 1)); // `neg` edits only the sign bit
    assert!(
        is_quiet_nan::<T>(magnitude),
        "{name}: sum then neg stays computed, got {sum_then_neg:#x}"
    );
    // A pure move (flip) preserves the bits.
    let flipped = out_bits::<T>(vec![Node::Bind(0), Node::Flip { child: 0, axis: 0 }]);
    assert_eq!(
        flipped,
        T::from_f64(2.0).raw(),
        "{name}: flip moves element 2 first"
    );
}

#[test]
fn the_move_obligation_attaches_to_the_observable_output_on_every_lane() {
    check::<f32>();
    check::<f16>();
    check::<bf16>();
    check::<E5m2>();
}
