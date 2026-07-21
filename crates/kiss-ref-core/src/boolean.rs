//! The truth-valued **bool** lane (§6.2-0006, §6.1-0007).
//!
//! `bool` is a 1-byte truth value: `0` = false, any non-zero = true, canonical
//! true = `1`. It is **not** an arithmetic dtype — `scalar_int::int_spec` returns
//! `None` for it (so §6.2-0002 wrapping arithmetic and the §6.10-0001 bitwise atoms
//! correctly decline). The legal ops are the logical family, equality, raw-bit
//! `select`, the order-preserving `max_prop`/`min_prop`, and the `{0,1}`-preserving
//! structural / data-movement ops.
//!
//! Scalar truth-valued ops are computed by **normalizing operands on read**
//! (non-zero → 1), delegating to the existing integer engine in a wide `i64`
//! accumulator, and **normalizing the result** to a canonical `{0,1}`. The tensor
//! bool ops reuse the integer tensor kernels (`crate::tensor_int`) over `{0,1}`
//! values, restricted to the `max`/`min`/`assign` combines that preserve `{0,1}`
//! (a `sum`/`prod` reduce or `atomic_add` scatter would escape the set).

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

use crate::resolve::bool_legal;
use crate::scalar_int::eval_int_op;
use crate::Error;

/// Normalize an integer to canonical bool `{0,1}` (§6.2-0006: non-zero → true).
#[inline]
pub fn norm(v: i128) -> i128 {
    (v != 0) as i128
}

/// Whether the bool lane **evaluates** `op` — the ops with an actual reference
/// path: the scalar truth ops ([`bool_scalar_supported`]) plus the tensor ops that
/// are both bool-legal *and* backed by an integer tensor kernel (reused over
/// `{0,1}` values). This is a **subset** of [`crate::legality`] for `bool`:
/// `im2col` is bool-*legal* (pure data movement) but has no integer/bool kernel in
/// this seed (only the float `window::im2col`), so `(im2col, bool)` stays
/// `Pending`, not over-claimed as `Done`.
pub fn bool_supported(op: Op) -> bool {
    bool_scalar_supported(op)
        || (bool_legal(op) && crate::tensor_int::int_tensor_supported(op))
}

/// The **scalar** subset the bool lane evaluates directly via [`eval_bool_op`]:
/// the logical family, equality, raw-bit `select`, and `max_prop`/`min_prop`. The
/// structural/tensor bool ops go through `crate::tensor_int` (over `{0,1}` i128
/// values with `max`/`min`/`assign` combines), not this scalar path.
pub fn bool_scalar_supported(op: Op) -> bool {
    matches!(
        op,
        Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalNot
            | Op::CmpEq
            | Op::CmpNe
            | Op::Select
            | Op::MaxProp
            | Op::MinProp
    )
}

/// Evaluate a truth-valued scalar op on bool operands. Operands are normalized on
/// read (§6.2-0006), evaluated in a wide `i64` integer accumulator (so a logical
/// op's transiently-`>1` reference form is contained), and the result is
/// normalized to a canonical `{0,1}`. Never panics — every failure is an [`Error`].
pub fn eval_bool_op(op: Op, args: &[i128]) -> Result<i128, Error> {
    if !bool_scalar_supported(op) {
        return Err(Error::UnsupportedDtype(Dtype::Bool));
    }
    if args.len() > 3 {
        return Err(Error::Arity { op, expected: 3, got: args.len() });
    }
    let mut buf = [0i128; 3];
    for (i, &a) in args.iter().enumerate() {
        buf[i] = norm(a);
    }
    // A wide i64 accumulator: {0,1} logic never wraps, and it lets the reference
    // decompositions (e.g. logical_or's transient `1 + 1`) be clamped by norm().
    let r = eval_int_op(op, Dtype::I64, &buf[..args.len()])?;
    Ok(norm(r))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_family_normalizes() {
        // non-canonical truthy inputs (5, -3) normalize to 1.
        assert_eq!(eval_bool_op(Op::LogicalAnd, &[5, 0]).unwrap(), 0);
        assert_eq!(eval_bool_op(Op::LogicalAnd, &[5, -3]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::LogicalOr, &[0, 0]).unwrap(), 0);
        assert_eq!(eval_bool_op(Op::LogicalOr, &[0, 7]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::LogicalNot, &[0]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::LogicalNot, &[9]).unwrap(), 0);
    }

    #[test]
    fn eq_select_minmax() {
        assert_eq!(eval_bool_op(Op::CmpEq, &[1, 1]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::CmpNe, &[1, 0]).unwrap(), 1);
        // select: cond true → arm a (both normalized to {0,1}).
        assert_eq!(eval_bool_op(Op::Select, &[1, 1, 0]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::Select, &[0, 1, 0]).unwrap(), 0);
        // max_prop = or-like, min_prop = and-like on {0,1}.
        assert_eq!(eval_bool_op(Op::MaxProp, &[1, 0]).unwrap(), 1);
        assert_eq!(eval_bool_op(Op::MinProp, &[1, 0]).unwrap(), 0);
    }

    #[test]
    fn arithmetic_and_bitwise_decline() {
        // add/bitwise are not truth-valued ops — they decline (never wrap-compute).
        assert!(matches!(eval_bool_op(Op::Add, &[1, 1]), Err(Error::UnsupportedDtype(_))));
        assert!(matches!(eval_bool_op(Op::BitAnd, &[1, 1]), Err(Error::UnsupportedDtype(_))));
    }

    #[test]
    fn bool_tensor_via_int_kernels() {
        // The bool tensor lane reuses the integer kernels over {0,1}: any = reduce
        // (max, cmp_ne), all = reduce(min, cmp_ne) — computed in i128/I64.
        use crate::tensor::Tensor;
        use crate::tensor_int::{all, any};
        let x = Tensor::from_vec(alloc::vec![0i128, 1, 0], &[3]).unwrap();
        assert_eq!(any(&x.view(), Dtype::I64, &[0]).unwrap().as_slice(), &[1]);
        assert_eq!(all(&x.view(), Dtype::I64, &[0]).unwrap().as_slice(), &[0]);
    }

    extern crate alloc;
}
