//! **The self-differential: hand-written kernel vs the §6.13 reference
//! decomposition it claims to transcribe.**
//!
//! `kiss-ops-vocab` holds the §6.13 decomposition SOURCE STRINGS; `kiss-ref-core`
//! evaluates ops. This file is the differential between the two — the
//! spec-exactness property kiss-ref exists to guarantee, needing no other project.
//!
//! # What is (and is not) a real check here — read this before trusting a green run
//!
//! **The scalar non-primitive differential is, for 30 of the 41 bound
//! decompositions, a TAUTOLOGY, and this file says so rather than pretending
//! otherwise.** `resolve::eval_op` handles the floor atoms and the eleven
//! §6.13 *refine-marked* ops directly and routes **everything else** to
//! `resolve_nonprimitive` → `parse(op.reference_decomposition_src())` →
//! `eval_expr` (`resolve.rs:117`). So for an unmarked op, "evaluate the op" and
//! "evaluate its decomposition" are literally the same call sequence and agree
//! bit-for-bit by construction. Those tests are kept as **pins**, not discoveries:
//! §6.13-0003 ends with *"an op **not** marked MUST reproduce the reference under
//! its determinism class"*, so the day anyone adds a direct kernel for an unmarked
//! op — the exact way an implementation silently drifts from the spec — the pin
//! turns red. They also prove **total cover**: every bound decomposition evaluates
//! (never `Err`, never a panic) over an adversarial value set, in both lanes.
//!
//! **The genuinely non-tautological checks are the eleven refine-marked ops**
//! (§6.13 *Refine* column: `expm1`, `log1p`, `tanh`, `sinh`, `cosh`, `silu`,
//! `softplus`, `mish`, `pow`, `hypot`, `ldexp`). For those the code has BOTH
//! paths: a direct refined kernel *and* the literal decomposition string. The spec
//! requires them to agree "within the op's declared ULP" **and** to diverge exactly
//! where the literal form overflows / cancels / gets a pinned domain edge wrong.
//! Both directions are asserted here:
//!   * [`test_decomp_refined_agree_with_literal_in_the_well_conditioned_domain`]
//!     and the `pow`/`ldexp`/`hypot` band tests — agreement,
//!   * the `*_diverge_*` tests — the refinement is *observable* and lands on the
//!     right side (a refinement that quietly did nothing would pass an agreement
//!     test alone).
//!
//! The literal path is [`eval_literal`], which expands **every** op carrying a
//! decomposition, refine-marked ones included, down to the primitive floor. That
//! is deliberately *not* what `eval_expr` does: `eval_expr` evaluates a
//! decomposition over the *pinned* meaning of each child op (a child `tanh` is the
//! mandatory overflow-safe `tanh`, §6.13-0002 + §6.13-0003), which is the correct
//! reading of a composite reference — see
//! [`test_decomp_gelu_tanh_inherits_the_mandatory_tanh_refinement`].
//!
//! # Tensor non-primitives
//!
//! §6.13 tensor decompositions are §6.13-0009 *structured* bodies (a named §6.11
//! structural op under an attribute record), explicitly **not** §6.13-0006
//! expression trees, so there is nothing to `parse` and no mechanical expansion.
//! What is checkable is a **re-composition from the structural atoms** plus
//! **mathematical equivalences implied by the decomposition**, and that is what the
//! `tensor` group does. [`test_decomp_tensor_coverage_is_declared_not_assumed`]
//! machine-checks the covered / not-covered partition so this file cannot silently
//! overclaim.
//!
//! Deterministic: fixed value tables, no randomness, no wall clock.

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::{parse, Expr};
use kiss_ops_vocab::Op;
use kiss_ref_core::kernels::{element_map, reduce};
use kiss_ref_core::tensor_ops as tops;
use kiss_ref_core::{
    eval_expr, eval_op, ulp_distance_f32, ulp_distance_f64, Error, IndexTensor, Monoid,
    ScalarFloat, Tensor,
};

// =============================================================================
// The two evaluation paths
// =============================================================================

/// **The literal path.** Evaluate `e` with EVERY op that carries a §6.13
/// decomposition expanded through that decomposition — *including* the
/// refine-marked ops — so only primitive-floor atoms ever reach `eval_op`. This is
/// the "literal reference decomposition" of §6.13-0003, with no refinement
/// anywhere in the tree.
///
/// `budget` bounds the recursion (KISS-OPS-6.14 guarantees termination at the
/// floor; the budget makes a spec/vocab cycle an `Err`, never a stack overflow).
fn eval_literal<T: ScalarFloat>(e: &Expr, inputs: &[T], budget: u32) -> Result<T, Error> {
    match e {
        Expr::Input(i) => inputs
            .get(*i as usize)
            .copied()
            .ok_or(Error::MissingInput(*i)),
        Expr::Const(c) => Ok(T::from_f64(c.value())),
        Expr::Apply(op, args) => {
            let mut vals: Vec<T> = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_literal(a, inputs, budget)?);
            }
            match op.reference_decomposition_src() {
                // a non-primitive: expand it, never call the (possibly refined) kernel.
                Some(src) => {
                    if budget == 0 {
                        return Err(Error::Unsupported(*op));
                    }
                    let sub = parse(src).map_err(|pe| Error::BadDecomposition {
                        op: *op,
                        pos: pe.pos,
                    })?;
                    eval_literal(&sub, &vals, budget - 1)
                }
                // a floor atom: the reference kernel IS the pinned semantics.
                None => eval_op(*op, &vals),
            }
        }
    }
}

/// The §6.13 decomposition tree of `op` (panics only on a vocab bug — the vocab's
/// own `decomp_all_bound_sources_parse` test guards the strings).
fn tree(op: Op) -> Expr {
    let src = op
        .reference_decomposition_src()
        .unwrap_or_else(|| panic!("{op:?} carries no §6.13 decomposition"));
    parse(src).unwrap_or_else(|e| panic!("{op:?} src {src:?} failed to parse: {e:?}"))
}

/// `op` evaluated through the fully-expanded literal decomposition, `f64` lane.
fn literal_f64(op: Op, args: &[f64]) -> f64 {
    eval_literal(&tree(op), args, 32).unwrap_or_else(|e| panic!("{op:?} literal path: {e:?}"))
}

/// `op` evaluated through the fully-expanded literal decomposition, `f32` lane.
fn literal_f32(op: Op, args: &[f32]) -> f32 {
    eval_literal(&tree(op), args, 32).unwrap_or_else(|e| panic!("{op:?} literal path: {e:?}"))
}

/// `op` evaluated by the reference (refined where §6.13-0003 marks it), `f64`.
fn pinned_f64(op: Op, args: &[f64]) -> f64 {
    eval_op::<f64>(op, args).unwrap_or_else(|e| panic!("{op:?} eval_op: {e:?}"))
}

/// `op` evaluated by the reference, `f32`.
fn pinned_f32(op: Op, args: &[f32]) -> f32 {
    eval_op::<f32>(op, args).unwrap_or_else(|e| panic!("{op:?} eval_op: {e:?}"))
}

/// The operand count a decomposition reads (`1 + max input index`): the §6.13-0004
/// naming makes `x`/`a` = `input(0)` and `b` = `input(1)`.
fn arity_of(e: &Expr) -> usize {
    match e {
        Expr::Input(i) => *i as usize + 1,
        Expr::Const(_) => 0,
        Expr::Apply(_, args) => args.iter().map(arity_of).max().unwrap_or(0),
    }
}

// =============================================================================
// The §6.13 *Refine* column, mirrored from the spec
// =============================================================================

/// The eleven ops marked **✓** in the *Refine* column of the §6.13 table, quoted
/// from KISS-OPS-6.13-0003: "(`expm1`, `log1p`, `tanh`, `sinh`, `cosh`, `silu`,
/// `softplus`, `mish`, `pow`, `hypot`, `ldexp`)". These are the cells where the
/// reference has BOTH a direct kernel and a decomposition string, i.e. the only
/// cells where this differential can find anything.
const REFINED: &[Op] = &[
    Op::Expm1,
    Op::Log1p,
    Op::Tanh,
    Op::Sinh,
    Op::Cosh,
    Op::Silu,
    Op::Softplus,
    Op::Mish,
    Op::Pow,
    Op::Hypot,
    Op::Ldexp,
];

fn is_refined(op: Op) -> bool {
    REFINED.contains(&op)
}

/// Every op carrying a bound §6.13 scalar decomposition, in vocab order.
fn bound_scalar_nonprimitives() -> Vec<Op> {
    Op::ALL
        .iter()
        .copied()
        .filter(|o| o.reference_decomposition_src().is_some())
        .collect()
}

/// Whether `op`'s decomposition transitively mentions a refine-marked op — the
/// cells where the *literal* expansion is not a legitimate target, because the
/// child's refinement is itself a spec MUST.
fn has_refined_descendant(op: Op, budget: u32) -> bool {
    fn walk(e: &Expr, budget: u32) -> bool {
        match e {
            Expr::Input(_) | Expr::Const(_) => false,
            Expr::Apply(op, args) => {
                is_refined(*op)
                    || has_refined_descendant(*op, budget.saturating_sub(1))
                    || args.iter().any(|a| walk(a, budget))
            }
        }
    }
    if budget == 0 || op.reference_decomposition_src().is_none() {
        return false;
    }
    walk(&tree(op), budget)
}

// =============================================================================
// The shared adversarial value set
// =============================================================================

/// The adversarial scalar set the brief calls for: zero and signed zero, ±1,
/// small, large, subnormal-adjacent, NaN, ±inf, plus a spread of ordinary
/// magnitudes. Drives the total-cover / tautology pins (bit-equality, so no
/// tolerance is involved and every value is fair game).
const ADVERSARIAL_F64: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    3.0,
    -3.0,
    0.125,
    -0.125,
    1e-8,
    -1e-8,
    1e8,
    -1e8,
    1e300,
    -1e300,
    1e-300,
    f64::EPSILON,
    f64::MIN_POSITIVE,  // smallest normal
    -f64::MIN_POSITIVE, // subnormal-adjacent from below
    5e-324,             // smallest positive subnormal
    -5e-324,
    f64::MAX,
    f64::MIN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

/// The `f32` mirror — built natively (not cast from `f64`, which would flush the
/// subnormal-adjacent members to zero and lose the interesting lattice edges).
const ADVERSARIAL_F32: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    3.0,
    -3.0,
    0.125,
    -0.125,
    1e-8,
    -1e-8,
    1e8,
    -1e8,
    1e30,
    -1e30,
    1e-30,
    f32::EPSILON,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    1e-45, // smallest positive f32 subnormal
    -1e-45,
    f32::MAX,
    f32::MIN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// A smaller cross-product base for the binary ops (29² rows would be needless).
const BINARY_BASE_F64: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    3.0,
    1e-300,
    1e300,
    5e-324,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
];

const BINARY_BASE_F32: &[f32] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    3.0,
    1e-30,
    1e30,
    1e-45,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
];

/// Rows of the right arity for `op`, `f64` lane.
fn rows_f64(arity: usize) -> Vec<Vec<f64>> {
    match arity {
        1 => ADVERSARIAL_F64.iter().map(|&x| vec![x]).collect(),
        _ => {
            let mut v = Vec::new();
            for &a in BINARY_BASE_F64 {
                for &b in BINARY_BASE_F64 {
                    v.push(vec![a, b]);
                }
            }
            v
        }
    }
}

fn rows_f32(arity: usize) -> Vec<Vec<f32>> {
    match arity {
        1 => ADVERSARIAL_F32.iter().map(|&x| vec![x]).collect(),
        _ => {
            let mut v = Vec::new();
            for &a in BINARY_BASE_F32 {
                for &b in BINARY_BASE_F32 {
                    v.push(vec![a, b]);
                }
            }
            v
        }
    }
}

// =============================================================================
// Group 1 — total cover, and the tautology pins (documented as such)
// =============================================================================

#[test]
fn test_decomp_bound_set_matches_the_section_6_13_scalar_table() {
    // Born-red guard: the shape of everything below depends on this partition.
    let bound = bound_scalar_nonprimitives();
    assert_eq!(
        bound.len(),
        41,
        "the §6.13 scalar (expression-body) table binds 41 ops"
    );
    let refined: Vec<Op> = bound.iter().copied().filter(|o| is_refined(*o)).collect();
    assert_eq!(
        refined.len(),
        11,
        "§6.13-0003 marks exactly 11 ops ✓ in the Refine column"
    );
    assert_eq!(
        bound.len() - refined.len(),
        30,
        "30 unmarked ops MUST reproduce the reference (§6.13-0003 last sentence)"
    );
    for op in REFINED {
        assert!(
            op.reference_decomposition_src().is_some(),
            "{op:?} is refine-marked, so it must still carry the reference string it refines"
        );
        assert!(
            !op.is_primitive_floor(),
            "{op:?} is a non-primitive (§6.3-0003)"
        );
    }
    // Every bound decomposition is scalar-evaluable — the total-cover property.
    for op in bound {
        assert!(
            kiss_ref_core::float_supported(op),
            "{op:?} must evaluate on the scalar float path"
        );
    }
}

#[test]
fn test_decomp_scalar_total_cover_over_the_adversarial_set() {
    // Never `Err`, never a panic, on any adversarial row, in either lane
    // (the never-panic discipline of kiss-ref-core, exercised through every §6.13
    // decomposition at once).
    for op in bound_scalar_nonprimitives() {
        let arity = arity_of(&tree(op));
        assert!(
            arity == 1 || arity == 2,
            "{op:?} unexpected decomposition arity {arity}"
        );
        for row in rows_f64(arity) {
            assert!(
                eval_op::<f64>(op, &row).is_ok(),
                "{op:?}{row:?} f64 must evaluate"
            );
            assert!(
                eval_expr::<f64>(&tree(op), &row).is_ok(),
                "{op:?}{row:?} f64 decomposition"
            );
        }
        for row in rows_f32(arity) {
            assert!(
                eval_op::<f32>(op, &row).is_ok(),
                "{op:?}{row:?} f32 must evaluate"
            );
            assert!(
                eval_expr::<f32>(&tree(op), &row).is_ok(),
                "{op:?}{row:?} f32 decomposition"
            );
        }
    }
}

#[test]
fn test_decomp_unmarked_ops_reproduce_the_reference_bit_for_bit() {
    // KISS-OPS-6.13-0003 (last sentence): an op NOT marked in the Refine column
    // MUST reproduce the reference decomposition under its determinism class.
    //
    // HONESTY: today this holds BY CONSTRUCTION — `eval_op` routes every unmarked
    // non-primitive straight into `parse(src)` + `eval_expr` (resolve.rs:117), so
    // the two sides are the same call sequence. This is a PIN, not a discovery: it
    // fails the moment a direct kernel is introduced for an unmarked op without the
    // spec marking it refine-permitted. Bit-equality (not ULP) is asserted, so even
    // a NaN-payload or signed-zero difference trips it.
    for op in bound_scalar_nonprimitives()
        .into_iter()
        .filter(|o| !is_refined(*o))
    {
        let e = tree(op);
        let arity = arity_of(&e);
        for row in rows_f64(arity) {
            let via_op = pinned_f64(op, &row);
            let via_decomp = eval_expr::<f64>(&e, &row).unwrap();
            assert_eq!(
                via_op.to_bits(),
                via_decomp.to_bits(),
                "{op:?}{row:?} f64: kernel {via_op:e} vs decomposition {via_decomp:e}"
            );
        }
        for row in rows_f32(arity) {
            let via_op = pinned_f32(op, &row);
            let via_decomp = eval_expr::<f32>(&e, &row).unwrap();
            assert_eq!(
                via_op.to_bits(),
                via_decomp.to_bits(),
                "{op:?}{row:?} f32: kernel {via_op:e} vs decomposition {via_decomp:e}"
            );
        }
    }
}

#[test]
fn test_decomp_unmarked_ops_reproduce_the_full_floor_expansion() {
    // The same pin one level deeper: recursively expanded to the primitive floor
    // (KISS-OPS-6.14 resolution), an unmarked op whose decomposition contains no
    // refine-marked descendant must still be bit-identical. This exercises the
    // §6.14 closure — the whole tree resolves and nothing on the way down changes a
    // bit — and excludes exactly the ops for which the literal expansion is NOT a
    // legitimate target (see the gelu_tanh tests).
    for op in bound_scalar_nonprimitives()
        .into_iter()
        .filter(|o| !is_refined(*o) && !has_refined_descendant(*o, 32))
    {
        let arity = arity_of(&tree(op));
        for row in rows_f64(arity) {
            let a = pinned_f64(op, &row);
            let b = literal_f64(op, &row);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{op:?}{row:?} f64 floor expansion"
            );
        }
        for row in rows_f32(arity) {
            let a = pinned_f32(op, &row);
            let b = literal_f32(op, &row);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{op:?}{row:?} f32 floor expansion"
            );
        }
    }
}

#[test]
fn test_decomp_gelu_tanh_is_the_only_unmarked_op_over_a_refined_child() {
    // Which unmarked ops sit above a refine-marked op? Exactly one: `gelu_tanh`
    // (its §6.13 body contains `tanh`). Pinned so a spec/vocab edit that creates
    // another such composite is reported here rather than silently widening the
    // set of cells where the literal expansion is not a legitimate target.
    let composites: Vec<Op> = bound_scalar_nonprimitives()
        .into_iter()
        .filter(|o| !is_refined(*o) && has_refined_descendant(*o, 32))
        .collect();
    assert_eq!(
        composites,
        vec![Op::GeluTanh],
        "unmarked ops standing over a refined child"
    );
}

#[test]
fn test_decomp_gelu_tanh_inherits_the_mandatory_tanh_refinement() {
    // The load-bearing reading of §6.13-0002/-0003 for a composite reference: a
    // decomposition names OPS, and each named op means its own pinned semantics.
    // `gelu_tanh` is unmarked, but its body contains `tanh`, whose overflow-safe
    // form is a MUST — so the reference for `gelu_tanh` is its body evaluated over
    // the PINNED `tanh`, which is what `eval_expr` does, and NOT the fully-literal
    // `(e^t − e^−t)/(e^t + e^−t)` expansion.
    let e = tree(Op::GeluTanh);

    // CLAIM A (the actual inheritance, bit-exact, over the WHOLE grid incl. overflow):
    // eval_op(gelu_tanh) IS eval_expr(body) — gelu_tanh is unmarked, so eval_op
    // resolves it through exactly this tree over the PINNED children. So the pinned
    // `tanh` inside is inherited node-for-node, and the two are identical bits even
    // at x=±50 where a literal `tanh` would already be NaN. This is the concrete
    // meaning of "inherits the mandatory tanh refinement".
    for &x in &[
        0.0f64, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, -1.0, -5.0, -30.0, -50.0,
    ] {
        let k = pinned_f64(Op::GeluTanh, &[x]);
        assert_eq!(
            k.to_bits(),
            eval_expr::<f64>(&e, &[x]).unwrap().to_bits(),
            "gelu_tanh({x})"
        );
    }

    // CLAIM B (overflow): the fully-literal expansion — tanh expanded to
    // (e^t−e^−t)/(e^t+e^−t) too — is NaN once the cubic argument overflows `exp`,
    // while the pinned reference is finite. This is WHY the inheritance matters.
    for &x in &[30.0f64, 50.0, -30.0, -50.0] {
        let k = pinned_f64(Op::GeluTanh, &[x]);
        assert!(
            literal_f64(Op::GeluTanh, &[x]).is_nan(),
            "literal gelu_tanh({x}) should be NaN"
        );
        assert!(
            !k.is_nan(),
            "pinned gelu_tanh({x}) must not be NaN, got {k:e}"
        );
    }
    // gelu_tanh(x) → x for large positive x, → −0.0 for large negative x.
    assert_eq!(pinned_f64(Op::GeluTanh, &[30.0]), 30.0);
    assert!(pinned_f64(Op::GeluTanh, &[-30.0]).is_sign_negative());

    // CLAIM C (agreement where the literal form is well conditioned): for
    // NON-NEGATIVE x below the overflow edge, the literal expansion and the pinned
    // reference agree within a small band. (x ≥ 0 only: for x < 0 the outer
    // `1 + tanh(arg)` cancels — arg is large-negative, tanh ≈ −1 — so a 1-ULP
    // difference in the literal `tanh` blows up the relative error without bound;
    // that ill-conditioning is asserted as Claim D, not swept under a band.)
    for &x in &[0.0f64, 0.125, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
        let d = ulp_distance_f64(
            pinned_f64(Op::GeluTanh, &[x]),
            literal_f64(Op::GeluTanh, &[x]),
        );
        assert!(d <= 8, "gelu_tanh({x}) literal vs pinned = {d} ULP > 8");
    }

    // CLAIM D (the negative tail is ill-conditioned for the LITERAL form — pinned):
    // at x=−5 the fully-literal expansion is finite but many ULP from the pinned
    // reference (the outer cancellation), so the literal expansion is NOT a valid
    // differential target there — which is exactly the point of using `eval_expr`
    // (pinned children) as the reference for a composite, not `eval_literal`.
    let d_tail = ulp_distance_f64(
        pinned_f64(Op::GeluTanh, &[-5.0]),
        literal_f64(Op::GeluTanh, &[-5.0]),
    );
    assert!(
        literal_f64(Op::GeluTanh, &[-5.0]).is_finite(),
        "literal is finite at -5, just wrong"
    );
    assert!(
        d_tail > 1_000,
        "the negative-tail cancellation must be visible, got {d_tail} ULP"
    );

    // f32 lane: the literal form fails earlier (exp overflows at ~88), and the
    // inheritance still holds bit-for-bit.
    assert!(literal_f32(Op::GeluTanh, &[20.0]).is_nan());
    assert_eq!(pinned_f32(Op::GeluTanh, &[20.0]), 20.0);
    assert_eq!(
        pinned_f32(Op::GeluTanh, &[20.0]).to_bits(),
        eval_expr::<f32>(&e, &[20.0f32]).unwrap().to_bits(),
        "f32 gelu_tanh inherits pinned tanh"
    );
}

// =============================================================================
// Group 2 — the real differential: refined kernel vs literal decomposition
// =============================================================================

/// The ULP band for the unary refine-marked ops **inside their well-conditioned
/// domain**. Every measured cell over the domains declared in
/// [`agreement_domain`] lands at ≤ 4 ULP in both lanes (worst: `expm1` at
/// x = −0.125, `softplus`/`mish` at x = −2); 8 leaves 2× headroom for a `libm`
/// patch release without letting a real regression through.
const UNARY_BAND: u64 = 8;

/// The §6.8 declared ULP of `sqrt` (2) doubled for the add-then-root chain — the
/// literal `sqrt(a²+b²)` is exact-op arithmetic plus one `sqrt`, and every measured
/// in-range cell is ≤ 1 ULP.
const HYPOT_BAND: u64 = 4;

/// The band an `exp(k · log(base))`-shaped literal form earns. `log`'s §6.8 error
/// (4 ULP) is multiplied by the exponent and then re-exponentiated, so the gap
/// grows **linearly in `|k · ln(base)|`**: 2 ULP per unit over a 4-ULP floor. Worst
/// measured cell: `pow(1e5, 7)` at 36 ULP with `|b·ln a| = 80.6` → band 166.
fn chain_band(magnitude: f64) -> u64 {
    4 + 2 * (magnitude.abs().ceil() as u64)
}

/// The grid the unary agreement runs on.
const UNARY_GRID: &[f64] = &[
    -20.0, -10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.125, -1e-2, -1e-4, -1e-8,
    -1e-16, 0.0, 1e-16, 1e-8, 1e-4, 1e-2, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0,
    20.0, 30.0, 50.0,
];

/// Where the literal decomposition of a refine-marked unary op is **well
/// conditioned**, i.e. where §6.13-0003's "MUST agree within the declared ULP" is
/// a testable statement at all. Outside it the literal form is not merely a few
/// ULP off — it is *catastrophically* wrong (see the divergence tests), and no
/// fixed band would be honest:
///   * `expm1`/`log1p`/`tanh`/`sinh` cancel as x → 0 (`e^x − e^−x` and
///     `log(1+x)` lose every significant bit; measured 3.6e15 ULP at 1e−16),
///   * `softplus`/`mish` cancel as x → −∞ (`log(1 + e^x)` with `e^x` below the
///     rounding of 1; measured 1.7e8 ULP at x = −20),
///   * `cosh`/`silu` have NO ill-conditioned region on this grid — both operands of
///     their sums are same-signed — so they are compared over the whole grid.
fn agreement_domain(op: Op, x: f64) -> bool {
    match op {
        Op::Expm1 | Op::Log1p | Op::Tanh | Op::Sinh => x.abs() >= 0.125,
        Op::Softplus | Op::Mish => x >= -2.0,
        Op::Cosh | Op::Silu => true,
        _ => unreachable!("not a unary refine-marked op"),
    }
}

#[test]
fn test_decomp_refined_agree_with_literal_in_the_well_conditioned_domain() {
    // KISS-OPS-6.13-0003: a refined kernel "MUST agree with the reference
    // decomposition's pinned mathematical meaning (the function it denotes) within
    // the op's declared ULP". This is the non-tautological half of the file: the
    // direct kernel and the literal §6.13 string are genuinely different code.
    let unary = [
        Op::Expm1,
        Op::Log1p,
        Op::Tanh,
        Op::Sinh,
        Op::Cosh,
        Op::Silu,
        Op::Softplus,
        Op::Mish,
    ];
    for op in unary {
        for &x in UNARY_GRID {
            if !agreement_domain(op, x) {
                continue;
            }
            let d = ulp_distance_f64(pinned_f64(op, &[x]), literal_f64(op, &[x]));
            assert!(
                d <= UNARY_BAND,
                "{op:?}({x:e}) f64: refined {:e} vs literal {:e} = {d} ULP > {UNARY_BAND}",
                pinned_f64(op, &[x]),
                literal_f64(op, &[x])
            );
            let xf = x as f32;
            let d32 = ulp_distance_f32(pinned_f32(op, &[xf]), literal_f32(op, &[xf])) as u64;
            assert!(
                d32 <= UNARY_BAND,
                "{op:?}({x:e}) f32: refined {:e} vs literal {:e} = {d32} ULP > {UNARY_BAND}",
                pinned_f32(op, &[xf]),
                literal_f32(op, &[xf])
            );
        }
    }
}

#[test]
fn test_decomp_refined_pow_agrees_with_exp_b_log_a_on_the_positive_base_domain() {
    // §6.13-0005: "for a>0, pow(a,b) equals the reference exp(mul(b, log(a)))
    // (refinement permitted)". Outside a>0 the reference form is undefined (log of
    // a non-positive) — those cells are the divergence test below.
    for &a in &[0.5f64, 1.0, 1.5, 2.0, 3.0, 10.0, 1e5] {
        for &b in &[-3.0f64, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 7.0] {
            let band = chain_band(b * a.ln());
            let d = ulp_distance_f64(pinned_f64(Op::Pow, &[a, b]), literal_f64(Op::Pow, &[a, b]));
            assert!(d <= band, "pow({a:e},{b}) f64: {d} ULP > band {band}");
            let (af, bf) = (a as f32, b as f32);
            let d32 = ulp_distance_f32(
                pinned_f32(Op::Pow, &[af, bf]),
                literal_f32(Op::Pow, &[af, bf]),
            ) as u64;
            assert!(d32 <= band, "pow({a:e},{b}) f32: {d32} ULP > band {band}");
        }
    }
}

#[test]
fn test_decomp_refined_ldexp_agrees_with_mul_a_exp2_b_and_is_exact_for_integer_b() {
    // §6.13: ldexp's reference is `mul(a, exp2(b))`, refine-permitted to "exact
    // scaling a·2^b for integer b" (§6.13-0003). Two obligations, both checked:
    // agreement with the reference within the chain band, AND exactness — the
    // literal form is NOT exact (measured 23 ULP at b = 52) while the refined one
    // must be.
    const LN2: f64 = core::f64::consts::LN_2;
    for &a in &[1.0f64, 1.5, -3.25, 7.0] {
        for &b in &[-20.0f64, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 20.0, 52.0, 100.0] {
            let band = chain_band(b * LN2);
            let d = ulp_distance_f64(
                pinned_f64(Op::Ldexp, &[a, b]),
                literal_f64(Op::Ldexp, &[a, b]),
            );
            assert!(d <= band, "ldexp({a},{b}) f64: {d} ULP > band {band}");
            let (af, bf) = (a as f32, b as f32);
            let d32 = ulp_distance_f32(
                pinned_f32(Op::Ldexp, &[af, bf]),
                literal_f32(Op::Ldexp, &[af, bf]),
            ) as u64;
            assert!(d32 <= band, "ldexp({a},{b}) f32: {d32} ULP > band {band}");
            // Exactness of the refinement: a·2^b to the bit, for integer b in range.
            let exact = a * (2.0f64).powi(b as i32);
            assert_eq!(
                pinned_f64(Op::Ldexp, &[a, b]).to_bits(),
                exact.to_bits(),
                "ldexp({a},{b}) must be the exact scaling"
            );
        }
    }
    // ... and the literal form demonstrably is not (the refinement is observable).
    assert_ne!(literal_f64(Op::Ldexp, &[1.0, 52.0]), 4503599627370496.0);
    assert_eq!(pinned_f64(Op::Ldexp, &[1.0, 52.0]), 4503599627370496.0);
}

#[test]
fn test_decomp_refined_hypot_agrees_with_sqrt_of_sum_of_squares_in_range() {
    // §6.13: `sqrt(add(sqr(a), sqr(b)))` — faithful while a²+b² neither overflows
    // nor underflows; those two edges are the divergence test.
    for &(a, b) in &[
        (3.0f64, 4.0),
        (1.0, 1.0),
        (1e-5, 3.0),
        (1e8, 1.0),
        (-3.0, -4.0),
        (0.0, 0.0),
        (-0.0, 0.0),
        (5.0, 12.0),
        (1e150, 1e150),
        (7.25, -0.5),
    ] {
        let d = ulp_distance_f64(
            pinned_f64(Op::Hypot, &[a, b]),
            literal_f64(Op::Hypot, &[a, b]),
        );
        assert!(
            d <= HYPOT_BAND,
            "hypot({a:e},{b:e}) f64: {d} ULP > {HYPOT_BAND}"
        );
    }
    // The pinned exact triples survive both paths identically.
    assert_eq!(pinned_f64(Op::Hypot, &[3.0, 4.0]), 5.0);
    assert_eq!(literal_f64(Op::Hypot, &[3.0, 4.0]), 5.0);
}

#[test]
fn test_decomp_refined_diverge_where_the_literal_form_overflows() {
    // §6.13-0003 MUST cases: "where the literal reference decomposition would
    // overflow ... while the true function is finite (the exp-of-large-argument
    // forms tanh, sinh, cosh, silu, softplus, mish)". A refinement that quietly did
    // nothing would sail through the agreement test — these assert that the
    // refinement is OBSERVABLE and lands on the correct side.
    //
    // NOTE (honest): `sinh`/`cosh` have no such witness in either lane. sinh/cosh of
    // an argument big enough to overflow `exp` overflow *mathematically* too, so the
    // literal `inf` is the right answer; their refinement permission buys near-zero
    // accuracy and signed zero (asserted in the cancellation / signed-zero tests),
    // not overflow rescue.
    let inf = f64::INFINITY;

    // tanh: literal inf/inf = NaN; pinned ±1 (the clause's named example).
    for &(x, want) in &[
        (1000.0f64, 1.0f64),
        (-1000.0, -1.0),
        (inf, 1.0),
        (-inf, -1.0),
    ] {
        assert!(
            literal_f64(Op::Tanh, &[x]).is_nan(),
            "literal tanh({x:e}) should be NaN"
        );
        assert_eq!(pinned_f64(Op::Tanh, &[x]), want, "pinned tanh({x:e})");
    }
    // softplus: literal log(inf) = inf; pinned ≈ x (the clause's named example).
    assert!(literal_f64(Op::Softplus, &[1000.0]).is_infinite());
    assert_eq!(pinned_f64(Op::Softplus, &[1000.0]), 1000.0);
    // mish: inherits both; literal NaN, pinned ≈ x.
    assert!(literal_f64(Op::Mish, &[1000.0]).is_nan());
    assert_eq!(pinned_f64(Op::Mish, &[1000.0]), 1000.0);
    assert!(literal_f64(Op::Mish, &[inf]).is_nan());
    assert!(pinned_f64(Op::Mish, &[inf]).is_infinite());
    // silu: literal 1/inf = 0 flushes the whole product; pinned keeps the subnormal.
    assert_eq!(
        literal_f64(Op::Silu, &[-745.0]).to_bits(),
        (-0.0f64).to_bits()
    );
    let s = pinned_f64(Op::Silu, &[-745.0]);
    assert!(
        s < 0.0 && s > -1e-300,
        "pinned silu(-745) should be a tiny negative, got {s:e}"
    );

    // The f32 lane fails earlier (exp overflows at ~88) — same shape, different edge.
    assert!(literal_f32(Op::Tanh, &[100.0]).is_nan());
    assert_eq!(pinned_f32(Op::Tanh, &[100.0]), 1.0);
    assert!(literal_f32(Op::Softplus, &[100.0]).is_infinite());
    assert_eq!(pinned_f32(Op::Softplus, &[100.0]), 100.0);
    assert!(literal_f32(Op::Mish, &[100.0]).is_nan());
    assert_eq!(pinned_f32(Op::Mish, &[100.0]), 100.0);
    assert_eq!(
        literal_f32(Op::Silu, &[-100.0]).to_bits(),
        (-0.0f32).to_bits()
    );
    assert!(pinned_f32(Op::Silu, &[-100.0]) < 0.0);
}

#[test]
fn test_decomp_refined_diverge_where_the_literal_form_cancels() {
    // §6.13-0003 MUST cases: "or catastrophically cancel while the true function is
    // finite". Near zero and on subnormal-adjacent inputs the literal forms collapse
    // to exactly 0 (or to the wrong low bits) while the true function is ≈ x.
    for &op in &[Op::Expm1, Op::Log1p, Op::Tanh, Op::Sinh] {
        for &x in &[1e-20f64, -1e-20, 5e-324, f64::MIN_POSITIVE] {
            let lit = literal_f64(op, &[x]);
            let pin = pinned_f64(op, &[x]);
            // f(x) ≈ x to far below 1 ULP at these magnitudes.
            assert!(
                ulp_distance_f64(pin, x) <= 1,
                "{op:?}({x:e}) pinned {pin:e} should be ≈ x"
            );
            assert_ne!(
                lit.to_bits(),
                pin.to_bits(),
                "{op:?}({x:e}) literal must diverge"
            );
            assert!(
                lit == 0.0 || ulp_distance_f64(lit, x) > 1_000,
                "{op:?}({x:e}) literal {lit:e} should have collapsed"
            );
        }
    }
    // softplus/mish cancel toward −∞ instead: log(1 + e^x) with e^x under 1's ULP.
    assert_eq!(literal_f64(Op::Softplus, &[-50.0]), 0.0);
    let sp = pinned_f64(Op::Softplus, &[-50.0]);
    assert!(
        sp > 0.0 && sp < 1e-21,
        "pinned softplus(-50) ≈ e^-50 = 1.93e-22, got {sp:e}"
    );
    assert_eq!(
        literal_f64(Op::Mish, &[-50.0]).to_bits(),
        (-0.0f64).to_bits()
    );
    assert!(pinned_f64(Op::Mish, &[-50.0]) < 0.0);
    // f32 lane.
    assert_eq!(literal_f32(Op::Softplus, &[-20.0]), 0.0);
    assert!(pinned_f32(Op::Softplus, &[-20.0]) > 0.0);
}

#[test]
fn test_decomp_refined_preserve_signed_zero_where_the_literal_form_loses_it() {
    // KISS-OPS-6.2-0004 (signed zero preserved). The literal forms round-trip −0.0
    // through an arithmetic identity that mints +0.0: `sub(exp(-0), 1)` = 1−1 = +0,
    // `(e^-0 − e^+0)/2` = +0. The refined kernels keep the sign. 1 ULP under the
    // sign-magnitude metric — invisible to a value compare, which is why this
    // asserts RAW BITS.
    for &op in &[Op::Expm1, Op::Tanh, Op::Sinh] {
        assert_eq!(
            literal_f64(op, &[-0.0]).to_bits(),
            (0.0f64).to_bits(),
            "{op:?}(-0.0) literal mints +0.0"
        );
        assert_eq!(
            pinned_f64(op, &[-0.0]).to_bits(),
            (-0.0f64).to_bits(),
            "{op:?}(-0.0) pinned must stay -0.0"
        );
        // +0.0 agrees on both paths.
        assert_eq!(
            literal_f64(op, &[0.0]).to_bits(),
            pinned_f64(op, &[0.0]).to_bits()
        );
    }
    // §6.13-0005 pins the same for pow's negative-zero base: pow(-0.0, 3) = -0.0,
    // which `exp(3·log(-0.0))` cannot express (log of −0 is −inf → +0).
    assert_eq!(
        literal_f64(Op::Pow, &[-0.0, 3.0]).to_bits(),
        (0.0f64).to_bits()
    );
    assert_eq!(
        pinned_f64(Op::Pow, &[-0.0, 3.0]).to_bits(),
        (-0.0f64).to_bits()
    );
}

#[test]
fn test_decomp_refined_pow_hypot_pin_domain_edges_the_literal_form_cannot_reach() {
    // §6.13-0005 (pow full domain) and §6.13-0007 (hypot inf/NaN): the pinned edges
    // the `exp(b·log a)` / `sqrt(a²+b²)` references get wrong. Each row asserts the
    // literal form's failure AND the pinned value, so the divergence is justified,
    // not merely tolerated.
    let (inf, nan) = (f64::INFINITY, f64::NAN);
    // pow: a < 0 with an integer exponent (log(a) is NaN in the reference form).
    assert!(literal_f64(Op::Pow, &[-2.0, 3.0]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[-2.0, 3.0]), -8.0);
    assert!(literal_f64(Op::Pow, &[-2.0, 2.0]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[-2.0, 2.0]), 4.0);
    assert!(literal_f64(Op::Pow, &[-inf, 3.0]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[-inf, 3.0]), -inf);
    // pow(0,0) = 1 (pinned); the reference computes exp(0·-inf) = exp(NaN) = NaN.
    assert!(literal_f64(Op::Pow, &[0.0, 0.0]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[0.0, 0.0]), 1.0);
    // IEEE pow's NaN-absorbing identities, likewise unreachable through exp/log.
    assert!(literal_f64(Op::Pow, &[1.0, nan]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[1.0, nan]), 1.0);
    assert!(literal_f64(Op::Pow, &[nan, 0.0]).is_nan());
    assert_eq!(pinned_f64(Op::Pow, &[nan, 0.0]), 1.0);
    // §6.13-0005 rows the reference DOES get right (regression guard on the guard).
    assert_eq!(pinned_f64(Op::Pow, &[0.0, 2.0]), 0.0);
    assert_eq!(literal_f64(Op::Pow, &[0.0, 2.0]), 0.0);
    assert!(pinned_f64(Op::Pow, &[-2.0, 0.5]).is_nan()); // a<0, non-integer b

    // hypot: §6.13-0007 — an infinite operand wins even against NaN.
    assert!(literal_f64(Op::Hypot, &[inf, nan]).is_nan());
    assert!(pinned_f64(Op::Hypot, &[inf, nan]).is_infinite());
    assert!(literal_f64(Op::Hypot, &[nan, -inf]).is_nan());
    assert!(pinned_f64(Op::Hypot, &[nan, -inf]).is_infinite());
    // hypot: intermediate overflow / underflow of a², which the pinned form avoids.
    assert!(literal_f64(Op::Hypot, &[1e200, 1e200]).is_infinite());
    assert!(pinned_f64(Op::Hypot, &[1e200, 1e200]).is_finite());
    assert_eq!(literal_f64(Op::Hypot, &[1e-200, 1e-200]), 0.0);
    assert!(pinned_f64(Op::Hypot, &[1e-200, 1e-200]) > 0.0);
    // finite + NaN still propagates NaN on both paths (§6.13-0007 second half).
    assert!(pinned_f64(Op::Hypot, &[nan, 2.0]).is_nan());
    assert!(literal_f64(Op::Hypot, &[nan, 2.0]).is_nan());
    // f32 lane: the overflow edge arrives at ~1e19.
    assert!(literal_f32(Op::Hypot, &[1e30, 1e30]).is_infinite());
    assert!(pinned_f32(Op::Hypot, &[1e30, 1e30]).is_finite());
}

#[test]
fn test_decomp_ldexp_refined_must_not_overflow_where_the_reference_does_not() {
    // Regression: the refined `ldexp` used to materialize `2^b` and overflow to
    // +inf for b >= 1024 even when `a * 2^b` is finite (found by THIS
    // differential; fixed in resolve.rs by splitting the exponent when `2^b`
    // overflows). §6.13-0003 lets `ldexp` be refined, but a refinement MUST
    // "agree with the reference decomposition's pinned mathematical meaning
    // within the op's declared ULP" — and +inf vs 1.797e8 is not within any ULP.
    // Witness:
    //   pinned : 1e-300 * pow(2, 1024) = 1e-300 * inf = +inf
    //   literal: 1e-300 * exp(1024·ln2) = 1.7976931348622733e8   (finite, ~correct)
    //   exact  : 1e-300 * 2^1024        = 1.7976931348623157e8
    let lit = literal_f64(Op::Ldexp, &[1e-300, 1024.0]);
    assert!(
        lit.is_finite(),
        "the reference form reaches this cell: {lit:e}"
    );
    let pin = pinned_f64(Op::Ldexp, &[1e-300, 1024.0]);
    assert!(
        pin.is_finite(),
        "refined ldexp must not spuriously overflow, got {pin:e}"
    );
    assert!(
        ulp_distance_f64(pin, 1.7976931348623157e8) <= 4,
        "got {pin:e}"
    );
}

// =============================================================================
// Group 3 — tensor non-primitives (§6.13-0009 structured bodies)
// =============================================================================

fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), shape).unwrap_or_else(|e| panic!("tensor build: {e:?}"))
}

/// Max ULP distance between two equal-length payloads.
fn max_ulp(a: &[f64], b: &[f64]) -> u64 {
    assert_eq!(a.len(), b.len(), "payload lengths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| ulp_distance_f64(*x, *y))
        .max()
        .unwrap_or(0)
}

#[test]
fn test_decomp_tensor_coverage_is_declared_not_assumed() {
    // Anti-overclaim gate. The §6.13 tensor non-primitives are exactly the ops the
    // tensor lane evaluates minus the six §6.11 structural atoms. Every one of them
    // is listed either as covered by this file or as explicitly NOT covered, with a
    // reason; a new tensor op cannot be added without landing in one list.
    const COVERED: &[Op] = &[
        Op::ReduceMean,
        Op::ReduceNorm2,
        Op::ReduceVar,
        Op::ReduceStd,
        Op::Logsumexp,
        Op::Argmax,
        Op::Any,
        Op::All,
        Op::Cumsum,
        Op::Cumprod,
        Op::Cummax,
        Op::Softmax,
        Op::LogSoftmax,
        Op::RmsNorm,
        Op::LayerNorm,
        Op::Matmul,
        Op::IndexSelect,
        Op::Embedding,
    ];
    // NOT differentially checked here, and why:
    // * `scatter_add` — §6.13-0009 structured op over `scatter(combine=atomic-add)`;
    //   float atomic-add is order-invariant/**nondeterministic** (§6.0-0004), so no
    //   byte-exact recomposition is legitimate and a tolerance recomposition would
    //   just re-run the same atom. Belongs with a scatter-semantics track.
    // * `avg_pool` / `max_pool` / `im2col` — §6.13-0009 structured ops whose bodies
    //   are `reduce_mean` / `reduce(max)` / a structured `gather` **over the pooled
    //   window view**, parameterized by the §6.13-0004 window attribute record. The
    //   window view IS the thing under test (`window.rs`); rebuilding it here would
    //   re-transcribe the same index arithmetic instead of differentially checking
    //   it. Belongs with a window-family track.
    const NOT_COVERED: &[Op] = &[Op::ScatterAdd, Op::AvgPool, Op::MaxPool, Op::Im2col];
    const ATOMS: &[Op] = &[
        Op::ElementMap,
        Op::Reduce,
        Op::PrefixScan,
        Op::Gather,
        Op::Scatter,
        Op::SortNetwork,
    ];

    let mut nonprimitives: Vec<Op> = Op::ALL
        .iter()
        .copied()
        .filter(|o| kiss_ref_core::tensor_supported(*o) && !ATOMS.contains(o))
        .collect();
    let mut declared: Vec<Op> = COVERED.iter().chain(NOT_COVERED).copied().collect();
    nonprimitives.sort_by_key(|o| o.token());
    declared.sort_by_key(|o| o.token());
    assert_eq!(
        declared, nonprimitives,
        "every §6.13 tensor non-primitive must be declared covered or not-covered here"
    );
    assert_eq!(COVERED.len(), 18);
    assert_eq!(NOT_COVERED.len(), 4);
}

#[test]
fn test_decomp_tensor_matmul_equals_element_map_then_reduce() {
    // §6.13 `matmul`: "reduce(sum, axis=K) of element_map(mul(input(0), input(1)))",
    // input(0) read at [m,k] broadcast over N and input(1) at [k,n] broadcast over M
    // (§6.11-0001). This is a REAL differential: `tensor_ops::matmul` is a
    // hand-written accumulation loop, while the right-hand side is built here from
    // the two structural atoms over the explicit (m,n,k) iteration space. Both fold
    // K ascending from the sum identity, so the §6.13 form is reproduced BIT-EXACTLY
    // (not merely within the §6.0-0004 nondeterminism the op declares).
    let (m, k, n) = (3usize, 4usize, 2usize);
    let a: Vec<f64> = (1..=(m * k)).map(|i| i as f64 * 0.5 - 3.0).collect();
    let b: Vec<f64> = (1..=(k * n)).map(|i| i as f64 * 0.25 - 1.0).collect();
    let direct = tops::matmul(&t(&a, &[m, k]).view(), &t(&b, &[k, n]).view()).unwrap();

    // materialize the (m,n,k) iteration space with the §6.11-0001 broadcast reads
    let mut a3 = Vec::with_capacity(m * n * k);
    let mut b3 = Vec::with_capacity(m * n * k);
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                a3.push(a[i * k + p]); // input(0)[m,k], stride 0 on N
                b3.push(b[p * n + j]); // input(1)[k,n], stride 0 on M
            }
        }
    }
    let prod = element_map(
        &parse("mul(a, b)").unwrap(),
        &[t(&a3, &[m, n, k]).view(), t(&b3, &[m, n, k]).view()],
        &[m, n, k],
    )
    .unwrap();
    let recomposed = reduce(&prod.view(), Monoid::Sum, &[2]).unwrap();

    assert_eq!(direct.shape(), &[m, n]);
    assert_eq!(recomposed.shape(), &[m, n, 1]); // keepdim on the contracted axis
    for (x, y) in direct.as_slice().iter().zip(recomposed.as_slice()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "matmul {x:e} vs atom recomposition {y:e}"
        );
    }
    // A hand-computed row, so the pair cannot be jointly wrong:
    // a = [[-2.5,-2,-1.5,-1],[-0.5,0,0.5,1],[1.5,2,2.5,3]],
    // b = [[-0.75,-0.5],[-0.25,0],[0.25,0.5],[0.75,1]]
    // row0·col0 = 1.875 + 0.5 - 0.375 - 0.75 = 1.25
    assert_eq!(direct.as_slice()[0], 1.25);
}

#[test]
fn test_decomp_tensor_reduce_mean_divisor_is_the_product_of_reduced_extents() {
    // §6.13 `reduce_mean`: `div(reduce(sum, x), reduced_count)`, "divisor is the
    // product of extents over ALL reduced axes (§6.12-0001)". The multi-axis case is
    // where a transcription goes wrong (dividing by one extent, or by the output
    // element count), so the divisor is recomputed independently here.
    let y = Tensor::from_vec((1..=24).map(|i| i as f64).collect::<Vec<_>>(), &[2, 3, 4]).unwrap();
    for axes in [
        vec![0usize],
        vec![1],
        vec![2],
        vec![0, 2],
        vec![1, 2],
        vec![0, 1, 2],
    ] {
        let mean = tops::reduce_mean(&y.view(), &axes).unwrap();
        let sum = reduce(&y.view(), Monoid::Sum, &axes).unwrap();
        let count: f64 = axes.iter().map(|&a| y.shape()[a] as f64).product();
        let expect: Vec<f64> = sum.as_slice().iter().map(|s| s / count).collect();
        assert_eq!(
            mean.as_slice(),
            expect.as_slice(),
            "reduce_mean over {axes:?}"
        );
    }
    // hand-computed: axes [0,2] of 1..24 shaped [2,3,4] → divisor 8, rows
    // {1..4,13..16}=68/8, {5..8,17..20}=100/8, {9..12,21..24}=132/8.
    let m = tops::reduce_mean(&y.view(), &[0, 2]).unwrap();
    assert_eq!(m.shape(), &[1, 3, 1]);
    assert_eq!(m.as_slice(), &[8.5, 12.5, 16.5]);
}

#[test]
fn test_decomp_tensor_reduce_var_equals_the_centered_form_and_where_it_stops() {
    // §6.13 `reduce_var`: `sub(reduce_mean(sqr(x)), sqr(reduce_mean(x)))` — the
    // textbook E[x²]−E[x]² form, population (§6.13-0004). Differential target: the
    // mathematically equal but numerically different centered form E[(x−μ)²],
    // computed here independently.
    for row in [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-5.5, 0.25, 3.0, 11.75],
        vec![1e6, 1e6 + 1.0, 1e6 + 2.0, 1e6 + 3.0],
    ] {
        let x = t(&row, &[4]);
        let v = tops::reduce_var(&x.view(), &[0]).unwrap();
        let mu: f64 = row.iter().sum::<f64>() / 4.0;
        let centered: Vec<f64> = row.iter().map(|q| (q - mu) * (q - mu)).collect();
        let cv = tops::reduce_mean(&t(&centered, &[4]).view(), &[0]).unwrap();
        assert_eq!(
            max_ulp(v.as_slice(), cv.as_slice()),
            0,
            "reduce_var {row:?}"
        );
    }
    // hand-computed population variance of 1,2,3,4 = 1.25; std = sqrt(1.25).
    assert_eq!(
        tops::reduce_var(&t(&[1.0, 2.0, 3.0, 4.0], &[4]).view(), &[0])
            .unwrap()
            .as_slice(),
        &[1.25]
    );
    let sd = tops::reduce_std(&t(&[1.0, 2.0, 3.0, 4.0], &[4]).view(), &[0]).unwrap();
    assert!(
        ulp_distance_f64(sd.as_slice()[0], 1.118_033_988_749_895) <= 4,
        "std {:e}",
        sd.as_slice()[0]
    );

    // AND the honest limit of the §6.13 form: at a 1e8 offset the pinned textbook
    // decomposition cancels — 2.0 against a true 1.25. That is a property of the
    // SPEC's decomposition (which kiss-ref transcribes faithfully), not a kiss-ref
    // bug; pinned here so consumers see the cell rather than discovering it on
    // device. §6.13-0004 lets an implementation declare a Bessel correction as an
    // attribute, but NOT change the decomposition silently — so this stays.
    let big = vec![1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0];
    let v = tops::reduce_var(&t(&big, &[4]).view(), &[0]).unwrap();
    assert_eq!(
        v.as_slice(),
        &[2.0],
        "the textbook form's cancellation, pinned"
    );
    let mu: f64 = big.iter().sum::<f64>() / 4.0;
    let centered: Vec<f64> = big.iter().map(|q| (q - mu) * (q - mu)).collect();
    let cv = tops::reduce_mean(&t(&centered, &[4]).view(), &[0]).unwrap();
    assert_eq!(cv.as_slice(), &[1.25], "the centered form is exact here");
}

#[test]
fn test_decomp_tensor_reduce_norm2_equals_sqrt_of_the_sum_of_squares() {
    // §6.13 `reduce_norm2`: `sqrt(reduce(sum, sqr(x)))`. Checked against the inverse
    // relation (norm² == Σx², an independent statement) plus an exact triple.
    for row in [
        vec![3.0, 4.0],
        vec![1.0, 2.0, 2.0],
        vec![-0.5, 0.25, 1.75, -2.0],
    ] {
        let x = t(&row, &[row.len()]);
        let n2 = tops::reduce_norm2(&x.view(), &[0]).unwrap().as_slice()[0];
        let sq: Vec<f64> = row.iter().map(|v| v * v).collect();
        let s = reduce(&t(&sq, &[row.len()]).view(), Monoid::Sum, &[0])
            .unwrap()
            .as_slice()[0];
        assert!(
            ulp_distance_f64(n2 * n2, s) <= 4,
            "norm2({row:?})² = {:e} vs Σx² = {s:e}",
            n2 * n2
        );
    }
    assert_eq!(
        tops::reduce_norm2(&t(&[3.0, 4.0], &[2]).view(), &[0])
            .unwrap()
            .as_slice(),
        &[5.0]
    );
}

#[test]
fn test_decomp_tensor_logsumexp_equals_the_naive_form_and_survives_its_overflow() {
    // §6.13 `logsumexp`: `m=reduce(max,x); out=add(m, log(reduce(sum, exp(sub(x,m)))))`
    // — the max-shifted form. The naive `log(Σ e^x)` is the same function; the shift
    // is exactly the tensor-lane analogue of a §6.13-0003 refinement, so it gets the
    // same two-sided treatment: agree in range, diverge where the naive form dies.
    let naive = |row: &[f64]| -> f64 {
        let x = t(row, &[row.len()]);
        let e = element_map(&parse("exp(x)").unwrap(), &[x.view()], &[row.len()]).unwrap();
        let s = reduce(&e.view(), Monoid::Sum, &[0]).unwrap();
        element_map(&parse("log(x)").unwrap(), &[s.view()], &[1])
            .unwrap()
            .as_slice()[0]
    };
    for row in [
        vec![0.0, 1.0, 2.0, 3.0],
        vec![-2.5, 0.25, 7.5, 1.0],
        vec![700.0, 701.0, 702.0, 699.0],
        vec![-1.0, -1.0, -1.0, -1.0],
    ] {
        let lse = tops::logsumexp(&t(&row, &[4]).view(), &[0])
            .unwrap()
            .as_slice()[0];
        let d = ulp_distance_f64(lse, naive(&row));
        assert!(
            d <= 4,
            "logsumexp {row:?}: shifted {lse:e} vs naive {:e} = {d} ULP",
            naive(&row)
        );
    }
    // hand-computed: ln(1+e+e²+e³) = 3.4401896985611953.
    let base = tops::logsumexp(&t(&[0.0, 1.0, 2.0, 3.0], &[4]).view(), &[0])
        .unwrap()
        .as_slice()[0];
    assert!(
        ulp_distance_f64(base, 3.4401896985611953) <= 4,
        "got {base:e}"
    );

    // Divergence: every exp overflows (naive → +inf) / underflows (naive → −inf)
    // while the shifted form is finite and equals the shifted-by-c answer.
    let hi = vec![719.0, 720.0, 721.0, 722.0];
    assert!(
        naive(&hi).is_infinite() && naive(&hi) > 0.0,
        "naive must overflow here"
    );
    let lse_hi = tops::logsumexp(&t(&hi, &[4]).view(), &[0])
        .unwrap()
        .as_slice()[0];
    assert!(
        ulp_distance_f64(lse_hi, base + 719.0) <= 4,
        "logsumexp shift-equivariance: {lse_hi:e}"
    );
    let lo = vec![-800.0, -801.0, -802.0, -799.0];
    assert!(
        naive(&lo).is_infinite() && naive(&lo) < 0.0,
        "naive must underflow to -inf here"
    );
    let lse_lo = tops::logsumexp(&t(&lo, &[4]).view(), &[0])
        .unwrap()
        .as_slice()[0];
    assert!(
        ulp_distance_f64(lse_lo, base - 802.0) <= 4,
        "logsumexp low end: {lse_lo:e}"
    );
}

#[test]
fn test_decomp_tensor_softmax_and_log_softmax_are_mutually_consistent() {
    // Cross-op equivalences implied by the §6.13 bodies — NOT a re-transcription:
    // `softmax` is built on reduce(max)+exp+reduce(sum)+div, `log_softmax` on
    // `sub(x, logsumexp(x))`, i.e. two independently written functions that the spec
    // forces to satisfy softmax == exp(log_softmax) and Σ softmax == 1.
    for row in [
        vec![1.0, 2.0, 3.0, 4.0],
        vec![-3.0, 0.5, 0.5, 2.25],
        vec![10.0, 10.0, 10.0, 10.0],
        vec![0.0, -700.0, 700.0, 1.0],
    ] {
        let x = t(&row, &[4]);
        let sm = tops::softmax(&x.view(), 0).unwrap();
        let ls = tops::log_softmax(&x.view(), 0).unwrap();
        let round = element_map(&parse("exp(x)").unwrap(), &[ls.view()], &[4]).unwrap();
        assert!(
            max_ulp(sm.as_slice(), round.as_slice()) <= 8,
            "softmax {row:?} vs exp(log_softmax) {:?}",
            round.as_slice()
        );
        let total: f64 = sm.as_slice().iter().sum();
        assert!(
            ulp_distance_f64(total, 1.0) <= 4,
            "softmax {row:?} sums to {total:e}"
        );
    }
    // uniform row → exactly 1/n (the max shift makes every exponent 0).
    let u = tops::softmax(&t(&[10.0, 10.0, 10.0, 10.0], &[4]).view(), 0).unwrap();
    assert_eq!(u.as_slice(), &[0.25, 0.25, 0.25, 0.25]);
}

#[test]
fn test_decomp_tensor_softmax_is_shift_invariant_where_the_naive_form_would_overflow() {
    // The max-shift in the §6.13 `softmax` body is what makes it survive a +1000
    // offset (a naive Σe^x overflows at ~709). Shift invariance is a property of the
    // decomposition, not of the transcription — and here it holds BIT-EXACTLY.
    let row = vec![1.0, 2.0, 3.0, 4.0];
    let base = tops::softmax(&t(&row, &[4]).view(), 0).unwrap();
    for shift in [1000.0f64, -1000.0, 700.0] {
        let moved: Vec<f64> = row.iter().map(|v| v + shift).collect();
        let s = tops::softmax(&t(&moved, &[4]).view(), 0).unwrap();
        for (a, b) in base.as_slice().iter().zip(s.as_slice()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "softmax shift {shift}: {a:e} vs {b:e}"
            );
        }
    }
}

#[test]
fn test_decomp_tensor_scans_agree_with_their_reductions() {
    // §6.13 `cumsum`/`cumprod`/`cummax` = `prefix_scan(monoid, inclusive)`; §6.13
    // reductions fold the same monoid over the same axis. The last element of an
    // inclusive scan is therefore the reduction — a cross-check of §6.11-0003
    // against §6.11-0002 through the two §6.13 op families. Both fold ascending from
    // the monoid identity, so this is byte-exact.
    let row = [0.1, 0.2, 0.3, 0.4, 0.5];
    let x = t(&row, &[5]);
    for (scan, monoid) in [
        (tops::cumsum(&x.view(), 0).unwrap(), Monoid::Sum),
        (tops::cumprod(&x.view(), 0).unwrap(), Monoid::Prod),
        (tops::cummax(&x.view(), 0).unwrap(), Monoid::Max),
    ] {
        let red = reduce(&x.view(), monoid, &[0]).unwrap();
        assert_eq!(
            scan.as_slice()[4].to_bits(),
            red.as_slice()[0].to_bits(),
            "{monoid:?}: scan tail {:e} vs reduction {:e}",
            scan.as_slice()[4],
            red.as_slice()[0]
        );
    }
    // hand-computed prefixes.
    assert_eq!(
        tops::cummax(&t(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]).view(), 0)
            .unwrap()
            .as_slice(),
        &[3.0, 3.0, 4.0, 4.0, 5.0]
    );
}

#[test]
fn test_decomp_tensor_any_all_are_the_max_min_of_cmp_ne() {
    // §6.13 `any` = `reduce(max, cmp_ne(x, const(0)))`, `all` = `reduce(min, ...)`.
    // Recomposed here from the atoms; the content is the monoid choice and the
    // {0,1} mapping — including NaN, which is non-zero (§6.6-0003: cmp_ne(NaN,0)=1).
    for row in [
        vec![0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![2.0, -3.0, 0.5],
        vec![0.0, f64::NAN, 0.0],
        vec![-0.0, 0.0, -0.0],
    ] {
        let n = row.len();
        let x = t(&row, &[n]);
        let ne = element_map(&parse("cmp_ne(x, const(0))").unwrap(), &[x.view()], &[n]).unwrap();
        let want_any = reduce(&ne.view(), Monoid::Max, &[0]).unwrap();
        let want_all = reduce(&ne.view(), Monoid::Min, &[0]).unwrap();
        assert_eq!(
            tops::any(&x.view(), &[0]).unwrap().as_slice(),
            want_any.as_slice(),
            "any {row:?}"
        );
        assert_eq!(
            tops::all(&x.view(), &[0]).unwrap().as_slice(),
            want_all.as_slice(),
            "all {row:?}"
        );
    }
    // hand-computed: NaN is truthy, -0.0 is not.
    assert_eq!(
        tops::any(&t(&[0.0, f64::NAN, 0.0], &[3]).view(), &[0])
            .unwrap()
            .as_slice(),
        &[1.0]
    );
    assert_eq!(
        tops::all(&t(&[-0.0, 0.0, -0.0], &[3]).view(), &[0])
            .unwrap()
            .as_slice(),
        &[0.0]
    );
}

#[test]
fn test_decomp_tensor_argmax_matches_an_independent_scan() {
    // §6.13 `argmax`: "original-index at rank 0 of sort_network(desc, keys=x)". The
    // differential target is an ordinary linear scan keeping the FIRST maximum —
    // which is what the §6.11-0007 stable sort (ties → lower original index) must
    // produce. Ties are the interesting cell; NaN is excluded on purpose (the
    // sort's NaN-greatest total order versus a max-reduction's NaN propagation is a
    // §6.11-0007 question this file does not adjudicate).
    for row in [
        vec![1.0, 9.0, 3.0, 2.0],
        vec![1.0, 9.0, 3.0, 9.0], // tie → lowest index
        vec![-5.0, -5.0, -7.0],
        vec![0.0, -0.0, 0.0], // signed-zero tie
        vec![2.5],
    ] {
        let n = row.len();
        let got = tops::argmax(&t(&row, &[n]).view(), 0).unwrap();
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        assert_eq!(got.as_slice(), &[best as i64], "argmax {row:?}");
    }
}

#[test]
fn test_decomp_tensor_layer_norm_and_rms_norm_match_hand_computed_values() {
    // §6.13 `layer_norm`: mu=reduce_mean(x); v=reduce_var(x);
    //   out = add(mul(mul(sub(x,mu), rsqrt(add(v,eps))), gamma), beta)
    // §6.13 `rms_norm`: ms=reduce_mean(sqr(x)); out = mul(mul(x, rsqrt(add(ms,eps))), gamma)
    // Both with gamma=1, beta=0, eps=0 so the expected values are hand-computable:
    //   x = [1,2,3,4] → mu = 2.5, var = 1.25, 1/sqrt(1.25) = 0.8944271909999159
    //   layer_norm = [-1.5,-0.5,0.5,1.5] * 0.8944271909999159
    //   ms = 7.5, 1/sqrt(7.5) = 0.36514837167011072, rms_norm = [1,2,3,4] * that
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let ones = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let zeros = t(&[0.0, 0.0, 0.0, 0.0], &[4]);

    let ln = tops::layer_norm(&x.view(), &ones.view(), &zeros.view(), 0.0, 0).unwrap();
    let k = 0.8944271909999159f64;
    let want_ln = [-1.5 * k, -0.5 * k, 0.5 * k, 1.5 * k];
    assert!(
        max_ulp(ln.as_slice(), &want_ln) <= 4,
        "layer_norm {:?}",
        ln.as_slice()
    );
    // centered and unit-variance: mean 0, mean square 1.
    let mean = tops::reduce_mean(&ln.view(), &[0]).unwrap().as_slice()[0];
    assert!(mean.abs() < 1e-15, "layer_norm output mean {mean:e}");
    let ms = tops::reduce_mean(
        &element_map(&parse("mul(x, x)").unwrap(), &[ln.view()], &[4])
            .unwrap()
            .view(),
        &[0],
    )
    .unwrap()
    .as_slice()[0];
    assert!(
        ulp_distance_f64(ms, 1.0) <= 8,
        "layer_norm output mean-square {ms:e}"
    );

    let rn = tops::rms_norm(&x.view(), &ones.view(), 0.0, 0).unwrap();
    let r = 0.365_148_371_670_110_7_f64;
    let want_rn = [r, 2.0 * r, 3.0 * r, 4.0 * r];
    assert!(
        max_ulp(rn.as_slice(), &want_rn) <= 4,
        "rms_norm {:?}",
        rn.as_slice()
    );
    // gamma/beta are applied, not ignored: gamma=2 scales, beta=1 shifts.
    let twos = t(&[2.0, 2.0, 2.0, 2.0], &[4]);
    let scaled = tops::rms_norm(&x.view(), &twos.view(), 0.0, 0).unwrap();
    for (a, b) in scaled.as_slice().iter().zip(rn.as_slice()) {
        assert_eq!(a.to_bits(), (b * 2.0).to_bits(), "rms_norm gamma");
    }
    let shifted =
        tops::layer_norm(&x.view(), &ones.view(), &t(&[1.0; 4], &[4]).view(), 0.0, 0).unwrap();
    for (a, b) in shifted.as_slice().iter().zip(ln.as_slice()) {
        assert_eq!(a.to_bits(), (b + 1.0).to_bits(), "layer_norm beta");
    }
}

#[test]
fn test_decomp_tensor_gather_family_matches_its_declared_oob_policy() {
    // §6.13 `index_select` = `gather(oob=skip)`, `embedding` = `gather(oob=zero-fill)`,
    // both with a 1-D index. The differential content is the POLICY difference: the
    // same OOB index must zero-fill under `embedding` and decline (no base supplied)
    // under `index_select` — the ruled dynamic base requirement (§6.11 gather-skip).
    let table = t(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2]);
    let idx = IndexTensor::new(vec![2, 0], &[2], Dtype::I64).unwrap();
    let sel = tops::index_select(&table.view(), &idx, 0).unwrap();
    assert_eq!(sel.as_slice(), &[30.0, 31.0, 10.0, 11.0]);
    let emb = tops::embedding(&table.view(), &idx).unwrap();
    assert_eq!(
        emb.as_slice(),
        sel.as_slice(),
        "in-range: the two policies coincide"
    );

    let oob = IndexTensor::new(vec![1, 9], &[2], Dtype::I64).unwrap();
    assert_eq!(
        tops::embedding(&table.view(), &oob).unwrap().as_slice(),
        &[20.0, 21.0, 0.0, 0.0],
        "embedding zero-fills the OOB row"
    );
    assert_eq!(
        tops::index_select(&table.view(), &oob, 0),
        Err(Error::GatherSkipNoBase),
        "index_select declines an actually-OOB read with no base"
    );
    // A raw-bit move: −0.0 survives the gather unchanged.
    let signed = t(&[-0.0, 1.0], &[2]);
    let one = IndexTensor::new(vec![0], &[1], Dtype::I64).unwrap();
    let moved = tops::index_select(&signed.view(), &one, 0).unwrap();
    assert_eq!(moved.as_slice()[0].to_bits(), (-0.0f64).to_bits());
}
