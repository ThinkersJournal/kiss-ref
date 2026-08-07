//! The bridge from the scalar reference engine to the tensor layer.
//!
//! The whole trick of the tensor layer is that it reuses the **unchanged** scalar
//! engine: `element_map` evaluates a per-element body with [`crate::resolve::eval_expr`],
//! and a `reduce`/`prefix_scan` monoid combine is one [`crate::resolve::eval_op`]
//! call, so NaN-propagation, signed-zero, and the refined forms are inherited, not
//! re-implemented. This module carries the small pieces that ride alongside: the
//! determinism class ([`DetClass`]) a result declares (§6.0), and the monoid
//! identity / combine-op / determinism helpers.
//!
//! Two lanes exist, both built: the **float lane** (`T: ScalarFloat` —
//! `f16`/`bf16`/`f32`/`f64`) and the **integer lane** (`reduce`/`gather`/… over
//! integer dtypes via `eval_int_op` / `eval_recipe_int`). The `(op × int-dtype)`
//! cells are `Done`, not `Pending`. What remains `Pending` in the coverage ledger is
//! the per-`(op × dtype)` gaps the resolver declines — FP8/`bool` for some ops, and
//! `nextafter` on the narrow floats (§6.9-0003) — not a whole lane.

use kiss_ops_vocab::Op;

use crate::attrs::Monoid;
use crate::scalar::ScalarFloat;
use crate::tensor::Tensor;

/// Determinism class of a tensor result (§6.0-0001) — the vehicle the differential
/// comparator reads to choose byte-exact vs tolerance comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DetClass {
    /// Bit-reproducible: compare byte-exact (§6.0-0002).
    ExactByte,
    /// Within a declared ULP of the wide-precision truth (§6.0-0003).
    Ulp(f64),
    /// Order-invariant in value but **not** bit-reproducible — float sum/prod
    /// reductions/scans and float scatter `atomic_add` (§6.0-0004). Compare under
    /// tolerance, NEVER byte-exact.
    OrderInvariantNondeterministic,
}

impl DetClass {
    /// The most-permissive combination of two classes (§6.0-0005):
    /// nondeterministic dominates ULP dominates exact-byte.
    pub fn join(self, other: DetClass) -> DetClass {
        use DetClass::*;
        match (self, other) {
            (OrderInvariantNondeterministic, _) | (_, OrderInvariantNondeterministic) => {
                OrderInvariantNondeterministic
            }
            (Ulp(a), Ulp(b)) => Ulp(if a >= b { a } else { b }),
            (Ulp(u), ExactByte) | (ExactByte, Ulp(u)) => Ulp(u),
            (ExactByte, ExactByte) => ExactByte,
        }
    }
}

/// A tensor result paired with its declared determinism class.
#[derive(Clone, Debug)]
pub struct Evaluated<T> {
    /// The computed tensor.
    pub tensor: Tensor<T>,
    /// How a conformance comparator must compare it (§6.0).
    pub det: DetClass,
}

impl<T> Evaluated<T> {
    /// Pair a tensor with a determinism class.
    pub fn new(tensor: Tensor<T>, det: DetClass) -> Self {
        Evaluated { tensor, det }
    }
}

/// The identity element of `monoid` in dtype `T` (§6.11-0002): `sum=+0`, `prod=1`,
/// `max=−inf`, `min=+inf`. Folding from the identity is exact (`0+x=x`,
/// `1·x=x`, `max(−inf,x)=x`, `min(+inf,x)=x`) and yields the identity for an
/// empty axis.
pub fn monoid_identity<T: ScalarFloat>(m: Monoid) -> T {
    match m {
        Monoid::Sum => T::ZERO,
        Monoid::Prod => T::ONE,
        Monoid::Max => T::from_f64(f64::NEG_INFINITY),
        Monoid::Min => T::from_f64(f64::INFINITY),
    }
}

/// The floor op whose scalar evaluation combines two elements under `monoid`.
/// `max`/`min` map to the NaN-**propagating** `max_prop`/`min_prop` (§6.11-0002),
/// so the scalar engine supplies the NaN semantics.
pub fn monoid_op(m: Monoid) -> Op {
    match m {
        Monoid::Sum => Op::Add,
        Monoid::Prod => Op::Mul,
        Monoid::Max => Op::MaxProp,
        Monoid::Min => Op::MinProp,
    }
}

/// The determinism class a **float** reduction/scan under `monoid` declares
/// (§6.0-0004): float `sum`/`prod` are order-invariant/nondeterministic; `max`/
/// `min` are exact-byte.
pub fn monoid_det(m: Monoid) -> DetClass {
    match m {
        Monoid::Sum | Monoid::Prod => DetClass::OrderInvariantNondeterministic,
        Monoid::Max | Monoid::Min => DetClass::ExactByte,
    }
}
