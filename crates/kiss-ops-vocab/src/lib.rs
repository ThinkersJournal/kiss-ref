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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact §6.3-0001 primitive-floor token set.
    const FLOOR_SPEC: &[&str] = &[
        "add", "sub", "mul", "div", "neg", "abs", "select", "cmp_eq", "cmp_ne", "cmp_lt", "cmp_le",
        "cmp_gt", "cmp_ge", "floor", "ceil", "trunc", "round_even", "exp", "log", "sin", "cos",
        "sqrt", "erf", "atan", "lgamma", "atan2", "copysign", "nextafter", "bit_and", "bit_or",
        "bit_xor", "bit_not", "shl", "shr", "popcount", "clz", "ctz", "element_map", "reduce",
        "prefix_scan", "gather", "scatter", "sort_network",
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
        // 43 floor + 63 non-primitive = 106 (complex §6.18 deferred).
        assert_eq!(Op::ALL.len(), 106);
        assert_eq!(Op::ALL.iter().filter(|o| !o.is_primitive_floor()).count(), 63);
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
