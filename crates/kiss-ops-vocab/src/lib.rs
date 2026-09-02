// SPDX-License-Identifier: MIT OR Apache-2.0
//! # kiss-ops-vocab
//!
//! A Rust binding of the **KISS-Ops** op vocabulary
//! (`../../KISS/spec/ops.md`): the closed op set of §6.1-0001 — the mandatory
//! **primitive floor** (§6.3), the **non-primitive** ops with reference
//! decompositions (§6.13), and (deferred in this seed) the complex family
//! (§6.18).
//!
//! KISS-Ops — not this crate — *owns* the op set. This crate is one conformant
//! embodiment with **no privilege**; the source of truth is the spec, and when
//! the spec changes this binding follows.
//!
//! **Foundational root.** Depends on nothing, and in particular never imports
//! `kiss-classify-vocab`: KISS-Ops and KISS-Classify are independent sibling
//! roots (KISS-Ops front-matter edge note; KISS-Classify §6.9-0001).
//!
//! The §6.13 reference decompositions live in [`decomp`], stored as the spec's
//! own expression strings (§6.13-0006 grammar) so they are auditable against and
//! regenerable from the spec.

#![cfg_attr(not(test), no_std)]

pub mod decomp;

/// The op-family tag each op carries (KISS-Ops §2.7). An op MUST NOT be
/// re-classified into a different family (KISS-OPS-6.1-0003).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    Arithmetic,
    Select,
    Comparison,
    Rounding,
    Transcendental,
    BinaryMath,
    Bitwise,
    Structural,
    Minmax,
    Activation,
    Logical,
    Reduction,
    Contraction,
    Normalization,
    Scan,
    Window,
    GatherScatter,
    Shape,
    /// The §6.18 complex-arithmetic family (`c32`/`c64`). Every member is
    /// non-primitive over the real floor; the family introduces no new primitive
    /// (§6.18-0002).
    Complex,
}

/// Generate the `Op` enum + its metadata from one spec-mirroring table, so the
/// vocabulary is a single auditable list rather than several parallel matches.
macro_rules! kiss_ops {
    ($($variant:ident => $token:literal, $family:ident, $floor:literal;)*) => {
        /// A KISS-Ops op token. Carries no attributes — parameters travel via the
        /// OpAttrs channel (KISS-Ops §6.19), deferred in this seed. The set is
        /// closed (KISS-OPS-6.1-0001).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum Op {
            $($variant,)*
        }

        impl Op {
            /// Every op in the bound set, in spec (family-grouped) order.
            pub const ALL: &'static [Op] = &[$(Op::$variant,)*];

            /// The normative, case-sensitive KISS token (KISS-OPS-6.1-0002).
            pub const fn token(self) -> &'static str {
                match self { $(Op::$variant => $token,)* }
            }

            /// The op-family tag (KISS-OPS-6.1-0003).
            pub const fn family(self) -> Family {
                match self { $(Op::$variant => Family::$family,)* }
            }

            /// Primitive-floor membership (KISS-OPS-6.1-0004 / §6.3): `true` iff
            /// this op is in the mandatory primitive floor.
            pub const fn is_primitive_floor(self) -> bool {
                match self { $(Op::$variant => $floor,)* }
            }
        }
    };
}

// The KISS-Ops op set. Floor rows (`true`) reproduce §6.3-0001 exactly; the
// remaining rows are the §6.13 non-primitive table. Complex family §6.18 is a
// documented PENDING region in this seed.
kiss_ops! {
    // ---- §6.3 primitive floor: arithmetic atoms (§6.4) ----
    Add => "add", Arithmetic, true;
    Sub => "sub", Arithmetic, true;
    Mul => "mul", Arithmetic, true;
    Div => "div", Arithmetic, true;
    Neg => "neg", Arithmetic, true;
    Abs => "abs", Arithmetic, true;
    // ---- floor: raw-bit select (§6.5) ----
    Select => "select", Select, true;
    // ---- floor: comparison atoms (§6.6) ----
    CmpEq => "cmp_eq", Comparison, true;
    CmpNe => "cmp_ne", Comparison, true;
    CmpLt => "cmp_lt", Comparison, true;
    CmpLe => "cmp_le", Comparison, true;
    CmpGt => "cmp_gt", Comparison, true;
    CmpGe => "cmp_ge", Comparison, true;
    // ---- floor: rounding atoms (§6.7) ----
    Floor => "floor", Rounding, true;
    Ceil => "ceil", Rounding, true;
    Trunc => "trunc", Rounding, true;
    RoundEven => "round_even", Rounding, true;
    // ---- floor: transcendental atoms (§6.8, declared-ULP) ----
    Exp => "exp", Transcendental, true;
    Log => "log", Transcendental, true;
    Sin => "sin", Transcendental, true;
    Cos => "cos", Transcendental, true;
    Sqrt => "sqrt", Transcendental, true;
    Erf => "erf", Transcendental, true;
    Atan => "atan", Transcendental, true;
    Lgamma => "lgamma", Transcendental, true;
    // ---- floor: binary-math atoms (§6.9) ----
    Atan2 => "atan2", BinaryMath, true;
    Copysign => "copysign", BinaryMath, true;
    Nextafter => "nextafter", BinaryMath, true;
    // ---- floor: bitwise atoms (§6.10) ----
    BitAnd => "bit_and", Bitwise, true;
    BitOr => "bit_or", Bitwise, true;
    BitXor => "bit_xor", Bitwise, true;
    BitNot => "bit_not", Bitwise, true;
    Shl => "shl", Bitwise, true;
    Shr => "shr", Bitwise, true;
    Popcount => "popcount", Bitwise, true;
    Clz => "clz", Bitwise, true;
    Ctz => "ctz", Bitwise, true;
    // ---- floor: structural-access atoms (§6.11) ----
    ElementMap => "element_map", Structural, true;
    Reduce => "reduce", Structural, true;
    PrefixScan => "prefix_scan", Structural, true;
    Gather => "gather", Structural, true;
    Scatter => "scatter", Structural, true;
    SortNetwork => "sort_network", Structural, true;

    // ---- §6.13 non-primitive ops (reference decompositions in `decomp`) ----
    Sqr => "sqr", Arithmetic, false;
    Recip => "recip", Arithmetic, false;
    Sign => "sign", Arithmetic, false;
    Rsqrt => "rsqrt", Transcendental, false;
    Exp2 => "exp2", Transcendental, false;
    Expm1 => "expm1", Transcendental, false;
    Log2 => "log2", Transcendental, false;
    Log10 => "log10", Transcendental, false;
    Log1p => "log1p", Transcendental, false;
    Tan => "tan", Transcendental, false;
    Tanh => "tanh", Transcendental, false;
    Sinh => "sinh", Transcendental, false;
    Cosh => "cosh", Transcendental, false;
    Asinh => "asinh", Transcendental, false;
    Acosh => "acosh", Transcendental, false;
    Atanh => "atanh", Transcendental, false;
    Asin => "asin", Transcendental, false;
    Acos => "acos", Transcendental, false;
    Cbrt => "cbrt", Transcendental, false;
    Erfc => "erfc", Transcendental, false;
    Frac => "frac", Rounding, false;
    Step => "step", Activation, false;
    Sigmoid => "sigmoid", Activation, false;
    Relu => "relu", Activation, false;
    Silu => "silu", Activation, false;
    Softplus => "softplus", Activation, false;
    Mish => "mish", Activation, false;
    Gelu => "gelu", Activation, false;
    GeluTanh => "gelu_tanh", Activation, false;
    MaxProp => "max_prop", Minmax, false;
    MinProp => "min_prop", Minmax, false;
    FmaxIeee => "fmax_ieee", Minmax, false;
    FminIeee => "fmin_ieee", Minmax, false;
    Pow => "pow", BinaryMath, false;
    Hypot => "hypot", BinaryMath, false;
    RemFloor => "rem_floor", BinaryMath, false;
    RemTrunc => "rem_trunc", BinaryMath, false;
    Ldexp => "ldexp", BinaryMath, false;
    LogicalAnd => "logical_and", Logical, false;
    LogicalOr => "logical_or", Logical, false;
    LogicalNot => "logical_not", Logical, false;
    ReduceMean => "reduce_mean", Reduction, false;
    ReduceNorm2 => "reduce_norm2", Reduction, false;
    ReduceVar => "reduce_var", Reduction, false;
    ReduceStd => "reduce_std", Reduction, false;
    Logsumexp => "logsumexp", Reduction, false;
    Argmax => "argmax", Reduction, false;
    Any => "any", Reduction, false;
    All => "all", Reduction, false;
    Matmul => "matmul", Contraction, false;
    Softmax => "softmax", Normalization, false;
    LogSoftmax => "log_softmax", Normalization, false;
    RmsNorm => "rms_norm", Normalization, false;
    LayerNorm => "layer_norm", Normalization, false;
    Cumsum => "cumsum", Scan, false;
    Cumprod => "cumprod", Scan, false;
    Cummax => "cummax", Scan, false;
    AvgPool => "avg_pool", Window, false;
    MaxPool => "max_pool", Window, false;
    IndexSelect => "index_select", GatherScatter, false;
    Embedding => "embedding", GatherScatter, false;
    ScatterAdd => "scatter_add", GatherScatter, false;
    Im2col => "im2col", Shape, false;

    // ---- §6.18 complex-arithmetic op family (c32 / c64) ----
    // Every member is non-primitive over the real floor (§6.18-0002), Annex-G-
    // governed (§6.18-0013). Order matches the §6.18-0001 set: the cmake/cre/cim
    // component bridge (decomposition plumbing, NOT advertised high-level per
    // §6.18-0016), then the advertised complex ops.
    Cmake => "cmake", Complex, false;
    Cre => "cre", Complex, false;
    Cim => "cim", Complex, false;
    Cadd => "cadd", Complex, false;
    Csub => "csub", Complex, false;
    Cneg => "cneg", Complex, false;
    Cconj => "cconj", Complex, false;
    Cmul => "cmul", Complex, false;
    Cdiv => "cdiv", Complex, false;
    Cabs => "cabs", Complex, false;
    Carg => "carg", Complex, false;
    Cexp => "cexp", Complex, false;
    Clog => "clog", Complex, false;
    Csqrt => "csqrt", Complex, false;
    Cpow => "cpow", Complex, false;
}

impl Op {
    /// Parse an op from its normative token. `None` for any token outside the
    /// bound set (KISS-OPS-6.1-0001: outside the set is not a KISS-Ops op).
    pub fn from_token(tok: &str) -> Option<Op> {
        Op::ALL.iter().copied().find(|o| o.token() == tok)
    }

    /// The **maximum ULP ceiling** for an op carrying a declared-ULP tolerance
    /// (§6.8). A conforming kernel MAY declare a tighter per-target ULP but MUST
    /// NOT declare one looser than this. `None` for exact ops (bitwise-to-the-pin)
    /// and for non-primitives (which inherit their decomposition's tolerance).
    /// `sqrt` reports the 2 ULP fallback ceiling — a target that guarantees
    /// correctly-rounded `sqrt` MUST meet 0.5 ULP (KISS-OPS-6.8-0003).
    ///
    /// **Includes `atan2` at 4 ULP.** The §6.8 ceiling table lists `atan2`, but
    /// `atan2` is *defined* as a §6.9 binary-math atom, not a §6.8 transcendental
    /// atom. This binding follows the §6.8 ceiling (the stricter, testable
    /// reading). The §6.8-vs-§6.9 placement is a genuine **spec inconsistency**
    /// (an atom appearing in the tolerance table of a section that does not
    /// define it) and is filed as an RFC to KISS — resolve by either moving the
    /// `atan2` row to §6.9 or cross-referencing it explicitly.
    pub const fn ulp_ceiling(self) -> Option<f64> {
        match self {
            Op::Sqrt => Some(2.0),
            Op::Exp | Op::Log | Op::Sin | Op::Cos | Op::Atan | Op::Erf | Op::Atan2 => Some(4.0),
            Op::Lgamma => Some(8.0),
            _ => None,
        }
    }

    /// True for the §6.18 complex-arithmetic family (`Family::Complex`) — the ops
    /// defined on the complex compute dtypes `c32`/`c64`.
    pub const fn is_complex(self) -> bool {
        matches!(self.family(), Family::Complex)
    }

    /// True for the `cmake`/`cre`/`cim` component bridge — decomposition plumbing
    /// that MUST NOT be advertised high-level (§6.18-0016). The remaining complex
    /// ops are the advertised (native-matchable) family.
    pub const fn is_complex_component_bridge(self) -> bool {
        matches!(self, Op::Cmake | Op::Cre | Op::Cim)
    }

    /// Whether a NaN this op emits has a bit pattern **determined by the input
    /// bits** — the predicate that selects an exact-byte comparison over a
    /// payload-blind one for a `Tolerance::Exact` differential (KISS-CONFORM
    /// §6.8-0010(a), the moved/determined arm, vs the §6.8-0010 main rule for a
    /// computed NaN).
    ///
    /// Derived, per **KISS-OPS-6.16-0009** (an op whose §6.13 reference
    /// decomposition contains no arithmetic computes nothing, so it can only MOVE
    /// its operands): the payload is determined iff nothing in the op's evaluation
    /// *mints* a value. A minting atom is an arithmetic primitive
    /// (`add`/`sub`/`mul`/`div`) or a transcendental (a declared-ULP atom) — these
    /// re-mint a NaN's payload per device (the sm_89-vs-x86 `add` witness).
    /// Everything else — a raw-bit move/transform (`abs`/`neg`/`copysign`,
    /// `select`, `nextafter`), a comparison, or a rounding atom — carries the
    /// input bits through, so `expected` binds bit-exactly.
    ///
    /// **Determinacy, not "arithmetic-free family":** `abs`/`neg` are
    /// `family = arithmetic` yet payload-determining bit-transforms (they clear /
    /// flip the sign bit, payload untouched), and `copysign` is raw-bit by
    /// KISS-OPS-6.9-0002. The predicate keys on whether the value is minted, not
    /// on the family tag — a classifier keyed on "arithmetic-free" mis-files all
    /// three.
    ///
    /// **Enforcement is partial (Convention 16(d), stated not discovered):** the
    /// derivation classifies on whether the decomposition CONTAINS a minting atom,
    /// not on whether the NaN PATH reaches it. An op whose NaN branch moved while
    /// another branch computed would be conservatively classified payload-blind
    /// (under-enforcing). No such op exists at the current op set, so the guard
    /// test cannot pin that case — a future op is not in the pin.
    ///
    /// Meaningful for scalar-evaluable ops; the structural/tensor atoms never
    /// reach the scalar differential seam (it errors before any comparator) and
    /// are outside this predicate's domain.
    pub fn nan_payload_is_determined(self) -> bool {
        !self.nan_payload_mints()
    }

    /// Whether the op MINTS (computes) its output rather than moving it — the
    /// inverse of [`Op::nan_payload_is_determined`]. `pub(crate)` because the
    /// [`crate::decomp`] Expr walk recurses through it to expand nested
    /// non-primitives to their floor.
    pub(crate) fn nan_payload_mints(self) -> bool {
        match self.reference_decomposition_src() {
            // primitive floor atom: mints iff it is an arithmetic or transcendental
            // atom. A move / bit-transform / comparison / rounding / bitwise atom
            // does not. (The structural floor atoms are out of domain — see above.)
            None => {
                matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)
                    || self.ulp_ceiling().is_some()
            }
            // non-primitive: mints iff any node in its §6.13 decomposition mints,
            // recursing into nested non-primitives. An unparseable BOUND
            // decomposition is a build-time bug, not a runtime input; degrade to
            // `true` (payload-blind — today's behaviour) rather than panic, so the
            // never-panic differential seam holds and no conformant impl is newly
            // failed.
            Some(src) => crate::decomp::parse(src)
                .map(|e| crate::decomp::expr_mints(&e))
                .unwrap_or(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §6.8-0010(a) determinacy predicate (nan_payload_is_determined) --------
    //
    // Pins BOTH directions, per KISS-OPS-6.16-0009: a flip determined→computed
    // under-enforces (false-PASS a wrong moved payload); a flip computed→determined
    // false-FAILS a conformant impl (the diff.rs sm_89-vs-x86 `add` witness). A
    // one-directional pin catches only the first.
    //
    // The pin arrays are kept ONE OP PER LINE deliberately: this is an exhaustive
    // control, and one-per-line makes a dropped op a VISIBLE DIFF rather than a
    // formatting change nobody reviews. Codacy's "≤50 LOC per fn" rule flags this
    // test — DECLINED: that rule is not enforced on this repo (it analyses only a
    // PR's changed lines, so it never touched e.g. decomp.rs::reference_decomposition_src,
    // a 40+-arm match well over 50 LOC that rides main unflagged). It reports "new to
    // my scope," not "risky," and reshaping an exhaustive pin to satisfy an unenforced
    // rule trades the reviewability property for nothing. Do not compact these arrays.
    #[test]
    fn nan_payload_determinacy_pins_both_directions() {
        use crate::decomp::{expr_nan_payload_is_determined, parse};

        // DETERMINED — a NaN these emit is bit-fixed by the inputs (a move, or a
        // deterministic bit-transform), so a Tolerance::Exact diff must bit-compare.
        // abs/neg are family=arithmetic yet payload-determining; copysign is raw-bit
        // by KISS-OPS-6.9-0002 — the counterexamples that break an "arithmetic-free"
        // predicate and force a determinacy one.
        let determined = [
            Op::Select,
            Op::Copysign,
            Op::Abs,
            Op::Neg,
            Op::Nextafter,
            Op::Floor,
            Op::Ceil,
            Op::Trunc,
            Op::RoundEven,
            Op::MaxProp,
            Op::MinProp,
            Op::FmaxIeee,
            Op::FminIeee,
            Op::Relu,
        ];
        for op in determined {
            assert!(
                op.nan_payload_is_determined(),
                "{} must be payload-determining (moved / bit-transform)",
                op.token()
            );
        }

        // COMPUTED — arithmetic or a transcendental MINTS the NaN payload per device
        // (the sm_89 `add`→0x7fffffff vs x86 0x7fc00000 witness), so Exact must stay
        // payload-blind. A flip of any of these to determined false-FAILS conformance.
        let computed = [
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Sqr,
            Op::Recip,
            Op::Exp,
            Op::Log,
            Op::Sin,
            Op::Cos,
            Op::Sqrt,
            Op::Erf,
            Op::Atan,
            Op::Lgamma,
            Op::Atan2,
            Op::Tanh,
            Op::Sigmoid,
            Op::Gelu,
            Op::Pow,
            Op::Hypot,
        ];
        for op in computed {
            assert!(
                !op.nan_payload_is_determined(),
                "{} MINTS its NaN payload; must stay payload-blind",
                op.token()
            );
        }

        // sign/step never PRODUCE a NaN (every arm is a const; a NaN input drives all
        // comparisons false → const(0)); the predicate is vacuous for them, and it
        // classifies them determined, which is harmless.
        assert!(Op::Sign.nan_payload_is_determined());
        assert!(Op::Step.nan_payload_is_determined());

        // the composed-Expr analogue: an abs node does NOT poison a region, an add
        // node does (determined iff EVERY node is payload-determining).
        let det = |s| expr_nan_payload_is_determined(&parse(s).unwrap());
        assert!(det("abs(neg(a))"));
        assert!(det("select(cmp_ne(a, a), a, b)"));
        assert!(!det("add(a, b)"));
        assert!(!det("select(cmp_ge(a, b), add(a, b), a)"));
    }

    /// The exact §6.3-0001 primitive-floor token set.
    const FLOOR_SPEC: &[&str] = &[
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "select",
        "cmp_eq",
        "cmp_ne",
        "cmp_lt",
        "cmp_le",
        "cmp_gt",
        "cmp_ge",
        "floor",
        "ceil",
        "trunc",
        "round_even",
        "exp",
        "log",
        "sin",
        "cos",
        "sqrt",
        "erf",
        "atan",
        "lgamma",
        "atan2",
        "copysign",
        "nextafter",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_not",
        "shl",
        "shr",
        "popcount",
        "clz",
        "ctz",
        "element_map",
        "reduce",
        "prefix_scan",
        "gather",
        "scatter",
        "sort_network",
    ];

    fn floor_tokens() -> Vec<&'static str> {
        let mut v: Vec<&str> = Op::ALL
            .iter()
            .filter(|o| o.is_primitive_floor())
            .map(|o| o.token())
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn ops_primitive_floor_set_matches_spec_exactly() {
        // KISS-OPS-6.3-0001: the floor MUST be exactly this set.
        let mut spec: Vec<&str> = FLOOR_SPEC.to_vec();
        spec.sort_unstable();
        assert_eq!(floor_tokens(), spec);
        assert_eq!(FLOOR_SPEC.len(), 43, "the floor is 43 atoms");
    }

    #[test]
    fn ops_tokens_unique_and_round_trip() {
        let mut toks: Vec<&str> = Op::ALL.iter().map(|o| o.token()).collect();
        let n = toks.len();
        toks.sort_unstable();
        toks.dedup();
        assert_eq!(toks.len(), n, "every op token must be distinct");
        for o in Op::ALL {
            assert_eq!(Op::from_token(o.token()), Some(*o), "round-trip {o:?}");
        }
    }

    #[test]
    fn ops_full_set_size() {
        // 43 floor + 78 non-primitive = 121 (incl. the 15-op §6.18 complex family).
        assert_eq!(Op::ALL.len(), 121);
        assert_eq!(
            Op::ALL.iter().filter(|o| !o.is_primitive_floor()).count(),
            78
        );
    }

    /// The exact §6.18-0001 complex-arithmetic op set.
    const COMPLEX_SPEC: &[&str] = &[
        "cmake", "cre", "cim", "cadd", "csub", "cneg", "cconj", "cmul", "cdiv", "cabs", "carg",
        "cexp", "clog", "csqrt", "cpow",
    ];

    #[test]
    fn ops_complex_op_set_matches_spec_exactly() {
        // KISS-OPS-6.18-0001: the complex family MUST be exactly this set.
        let mut got: Vec<&str> = Op::ALL
            .iter()
            .filter(|o| o.is_complex())
            .map(|o| o.token())
            .collect();
        got.sort_unstable();
        let mut spec: Vec<&str> = COMPLEX_SPEC.to_vec();
        spec.sort_unstable();
        assert_eq!(got, spec);
        assert_eq!(COMPLEX_SPEC.len(), 15, "the complex family is 15 ops");
    }

    #[test]
    fn ops_complex_no_new_primitive() {
        // KISS-OPS-6.18-0002: every complex op is non-primitive; the §6.3 floor
        // (43 atoms) is unchanged by the family.
        assert!(Op::ALL
            .iter()
            .filter(|o| o.is_complex())
            .all(|o| !o.is_primitive_floor()));
        assert_eq!(
            Op::ALL.iter().filter(|o| o.is_primitive_floor()).count(),
            43,
            "the primitive floor stays at 43 atoms"
        );
    }

    #[test]
    fn ops_complex_advertised_high_level() {
        // KISS-OPS-6.18-0016: exactly these 12 are advertised high-level; the
        // cmake/cre/cim bridge is plumbing and MUST NOT be advertised.
        let advertised: Vec<&str> = Op::ALL
            .iter()
            .filter(|o| o.is_complex() && !o.is_complex_component_bridge())
            .map(|o| o.token())
            .collect();
        for t in ["cmake", "cre", "cim"] {
            assert!(!advertised.contains(&t), "{t} is plumbing, not advertised");
        }
        for t in [
            "cadd", "csub", "cneg", "cconj", "cmul", "cdiv", "cabs", "carg", "cexp", "clog",
            "csqrt", "cpow",
        ] {
            assert!(advertised.contains(&t), "{t} must be advertised high-level");
        }
        assert_eq!(advertised.len(), 12);
    }

    #[test]
    fn ops_declared_ulp_ceilings_match_section_6_8() {
        // The exact §6.8 declared-ULP set (including atan2, per the §6.8 table).
        let expected: &[(&str, f64)] = &[
            ("sqrt", 2.0),
            ("exp", 4.0),
            ("log", 4.0),
            ("sin", 4.0),
            ("cos", 4.0),
            ("atan", 4.0),
            ("erf", 4.0),
            ("atan2", 4.0),
            ("lgamma", 8.0),
        ];
        for o in Op::ALL {
            match o.ulp_ceiling() {
                Some(c) => {
                    let (_, want) = expected
                        .iter()
                        .find(|(t, _)| *t == o.token())
                        .unwrap_or_else(|| panic!("{o:?} carries a ceiling but is not in §6.8"));
                    assert_eq!(c, *want, "{o:?} ceiling");
                }
                None => assert!(
                    expected.iter().all(|(t, _)| *t != o.token()),
                    "{o:?} should carry a §6.8 ceiling"
                ),
            }
        }
        for (tok, _) in expected {
            assert!(
                Op::from_token(tok).and_then(|o| o.ulp_ceiling()).is_some(),
                "{tok} must carry a ceiling"
            );
        }
    }

    #[test]
    fn ops_unknown_token_is_not_an_op() {
        for bad in ["remainder", "gelu_erf", "maximum", "relu6", "ADD", ""] {
            assert_eq!(Op::from_token(bad), None, "{bad:?} must not parse");
        }
    }
}
