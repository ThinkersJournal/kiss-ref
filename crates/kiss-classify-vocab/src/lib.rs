// SPDX-License-Identifier: MIT OR Apache-2.0
//! # kiss-classify-vocab
//!
//! A Rust binding of the **KISS-Classify** pinned scalar dtype set
//! (`../../KISS/spec/classify.md` §6.1, "The pinned scalar dtype set"), bound to
//! the **sk4** schema vocabulary (KISS `@f4cc3ad`).
//!
//! KISS-Classify — not this crate — *owns* the dtype set. This crate is one
//! conformant embodiment of it with **no privilege** (KISS-Classify front-matter).
//! The source of truth is the spec; when the spec changes, this binding follows.
//!
//! **Foundational root.** This crate depends on nothing. In particular it never
//! imports `kiss-ops-vocab`: KISS-Classify §6.9-0001 pins that the data vocabulary
//! and the computation vocabulary are independent sibling roots, neither importing
//! the other. The Rust binding preserves that so a consumer needing only dtypes
//! does not pull in the op vocabulary.
//!
//! Dtypes are pinned by **bit layout, not by spelling** (KISS-Classify §2.2). The
//! `bits` here is the logical bit width from the §6.1 table; sub-byte and complex
//! packing details (`i4`/`u4` packed pairs, `b1` packed byte, `c64`/`c128`
//! interleaved re/im) are documented per-variant and belong to the storage layer.
//!
//! **sk4 vocabulary (§6.1, 24 tokens).** Integers carry a uniform `i`-prefix
//! (`i8`/`i16`/`i4`; `i32`/`i64` unchanged). FP8 is width-prefixed and
//! variant-explicit (`f8e4m3fn`, `f8e5m2`, with the byte-incompatible `f8e4m3fnuz`
//! / `f8e5m2fnuz` **reserved** — recognized on parse, no compute semantics at this
//! schema version). The two OCP-Microscaling **scale** dtypes `f8e8m0`/`f8e6m2` are
//! additive at sk4 (sibling-operand scales, never element-value dtypes). Complex is
//! named by **total** width: `c64` = pair-of-`f32` (the sk3 `c32`), `c128` =
//! pair-of-`f64` (the sk3 `c64`) — the version prefix (§3.4) makes the flip loud.

#![cfg_attr(not(test), no_std)]

/// The five numeric kinds KISS-Classify §6.1 groups dtypes under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericKind {
    /// IEEE-754 or non-IEEE floating-point (`f16 bf16 f32 f64`, the FP8 variants,
    /// and the MX scale floats `f8e8m0`/`f8e6m2`).
    Float,
    /// Signed two's-complement integer (`i8 i16 i32 i64 i4`).
    Int,
    /// Unsigned integer (`u8 u16 u32 u64 u4 b1`).
    Uint,
    /// 1-byte truth value (`bool`); `0` = false, any non-zero = true.
    Bool,
    /// Interleaved (re, im) floating-point pair (`c64 c128`).
    Complex,
}

/// The KISS-Classify §6.1 pinned scalar dtype set — exactly 24 dtypes (sk4).
///
/// Token spelling is normative and case-sensitive (KISS-CLASSIFY §6.1 / the same
/// exact-spelling discipline KISS-OPS-6.1-0002 applies to ops). At sk4 the integer
/// prefixes are uniform `i` (`i8`/`i16`/`i4`, `i32`/`i64`); the sk3 `s`-prefixed
/// spellings (`s8`/`s16`/`s4`), the bare FP8 spellings (`e4m3`/`e5m2`), and the sk3
/// complex spelling `c32` are no longer dtype tokens of this version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// IEEE-754 binary16 (1s+5e+10m), half-precision storage.
    F16,
    /// bfloat16 (1s+8e+7m); f32 exponent range, reduced mantissa. Not IEEE-754.
    Bf16,
    /// IEEE-754 binary32 storage. (Compute precision is a KISS-Ops fidelity
    /// attribute, not a dtype — KISS-Classify §6.1-0005.)
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Signed 8-bit two's-complement (sk3 `s8`).
    I8,
    /// Signed 16-bit two's-complement (sk3 `s16`).
    I16,
    /// Unsigned 8-bit; also the physical storage of `bool`.
    U8,
    /// Unsigned 16-bit.
    U16,
    /// Signed 32-bit two's-complement.
    I32,
    /// Signed 64-bit two's-complement.
    I64,
    /// Ordinary unsigned 32-bit storage (container width matches `i32`). The
    /// index/address *role* is an operand role in KISS-Ops, not a dtype class
    /// (KISS-CLASSIFY-6.1-0006).
    U32,
    /// Unsigned 64-bit.
    U64,
    /// 1-byte truth value; storage width equals `u8`; ops normalize to 0/1.
    Bool,
    /// FP8 E4M3 OCP *finite* (1s+4e+3m, bias 7); max finite ±448, no infinities,
    /// single NaN (OCP OFP8, §6.1-0010). sk3 `e4m3fn` + the `f8` width prefix.
    F8e4m3fn,
    /// FP8 E4M3 AMD `fnuz` variant (bias 8, no −0, no infinities); byte-incompatible
    /// with `f8e4m3fn`. **Reserved** at sk4: recognized on parse but carries no
    /// computation semantics — any use in a compute position is a typed decline
    /// distinct from an unknown token (§6.1-0001).
    F8e4m3fnuz,
    /// FP8 E5M2 IEEE-style (1s+5e+2m, bias 15); max finite ±57344, IEEE-style
    /// inf/NaN (OCP OFP8, §6.1-0011). sk3 `e5m2` + the `f8` prefix; no variant suffix.
    F8e5m2,
    /// FP8 E5M2 AMD `fnuz` variant (bias 16, no −0, no infinities); byte-incompatible
    /// with `f8e5m2`. **Reserved** at sk4 (recognized, no compute; see `F8e4m3fnuz`).
    F8e5m2fnuz,
    /// MX shared-exponent **scale** (unsigned: 0s+8e+0m); OCP Microscaling, §6.1-0013.
    /// A per-block scale carried as a *sibling operand*, **never** an element-value
    /// dtype — recognized on parse, declines compute in a value position. New at sk4.
    F8e8m0,
    /// MX **scale** (unsigned: 0s+6e+2m); finer-granularity sibling of `f8e8m0`
    /// (§6.1-0013). A scale type, never an element-value dtype. New at sk4.
    F8e6m2,
    /// Signed 4-bit `[-8,+7]`; packed-pair storage, sign-extended on read (sk3 `s4`).
    I4,
    /// Unsigned 4-bit `[0,15]`; packed-pair storage, zero-extended on read.
    U4,
    /// 1-bit binary-GEMM operand; packed-byte storage (LSB = lowest index);
    /// xor+popcount accumulation, raw `i32` output.
    B1,
    /// Complex named by **total** width: interleaved (re, im) pair of `f32`, 64 bits
    /// total (§6.1-0012). **sk4 meaning-flip:** this is the sk3 `c32`.
    C64,
    /// Complex: interleaved (re, im) pair of `f64`, 128 bits total (§6.1-0012). This
    /// is the sk3 `c64`.
    C128,
}

impl Dtype {
    /// Every dtype in the KISS-Classify §6.1 set, in table order. The coverage
    /// gate iterates this to enumerate the (op × dtype) matrix.
    pub const ALL: [Dtype; 24] = [
        Dtype::F16,
        Dtype::Bf16,
        Dtype::F32,
        Dtype::F64,
        Dtype::I8,
        Dtype::I16,
        Dtype::U8,
        Dtype::U16,
        Dtype::I32,
        Dtype::I64,
        Dtype::U32,
        Dtype::U64,
        Dtype::Bool,
        Dtype::F8e4m3fn,
        Dtype::F8e4m3fnuz,
        Dtype::F8e5m2,
        Dtype::F8e5m2fnuz,
        Dtype::F8e8m0,
        Dtype::F8e6m2,
        Dtype::I4,
        Dtype::U4,
        Dtype::B1,
        Dtype::C64,
        Dtype::C128,
    ];

    /// The normative, case-sensitive KISS token for this dtype.
    pub const fn token(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::I8 => "i8",
            Dtype::I16 => "i16",
            Dtype::U8 => "u8",
            Dtype::U16 => "u16",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::U32 => "u32",
            Dtype::U64 => "u64",
            Dtype::Bool => "bool",
            Dtype::F8e4m3fn => "f8e4m3fn",
            Dtype::F8e4m3fnuz => "f8e4m3fnuz",
            Dtype::F8e5m2 => "f8e5m2",
            Dtype::F8e5m2fnuz => "f8e5m2fnuz",
            Dtype::F8e8m0 => "f8e8m0",
            Dtype::F8e6m2 => "f8e6m2",
            Dtype::I4 => "i4",
            Dtype::U4 => "u4",
            Dtype::B1 => "b1",
            Dtype::C64 => "c64",
            Dtype::C128 => "c128",
        }
    }

    /// Parse a dtype from its normative token. Returns `None` for any token
    /// outside the §6.1 set (KISS-CLASSIFY: a token outside the set is not a
    /// dtype of this version — the *unknown-token* verdict). A **reserved** dtype
    /// (`f8e4m3fnuz`/`f8e5m2fnuz`) and an MX **scale** (`f8e8m0`/`f8e6m2`) DO parse
    /// here — they are recognized members of the closed vocabulary — but decline
    /// compute in a value position ([`Dtype::declines_compute`]).
    pub fn from_token(tok: &str) -> Option<Dtype> {
        Dtype::ALL.into_iter().find(|d| d.token() == tok)
    }

    /// The numeric kind this dtype belongs to (§6.1-0003 table). The MX scales
    /// `f8e8m0`/`f8e6m2` are kind `float` (unsigned exponent-scales; unsigned is a
    /// packing fact of §6.1-0013, not a distinct kind).
    pub const fn numeric_kind(self) -> NumericKind {
        match self {
            Dtype::F16
            | Dtype::Bf16
            | Dtype::F32
            | Dtype::F64
            | Dtype::F8e4m3fn
            | Dtype::F8e4m3fnuz
            | Dtype::F8e5m2
            | Dtype::F8e5m2fnuz
            | Dtype::F8e8m0
            | Dtype::F8e6m2 => NumericKind::Float,
            Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64 | Dtype::I4 => NumericKind::Int,
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::U4 | Dtype::B1 => {
                NumericKind::Uint
            }
            Dtype::Bool => NumericKind::Bool,
            Dtype::C64 | Dtype::C128 => NumericKind::Complex,
        }
    }

    /// The logical bit width from the §6.1-0002 table (4 for `i4`/`u4`, 1 for `b1`,
    /// 8 for every FP8/MX-scale byte, 64 for `c64`, 128 for `c128`). This is the
    /// logical width, not the container/storage width (which differs for the packed
    /// sub-byte types). A scale's width self-check is `exp + mantissa` (§6.1-0013).
    pub const fn bits(self) -> u16 {
        match self {
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 | Dtype::U16 => 16,
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 32,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 64,
            Dtype::I8
            | Dtype::U8
            | Dtype::Bool
            | Dtype::F8e4m3fn
            | Dtype::F8e4m3fnuz
            | Dtype::F8e5m2
            | Dtype::F8e5m2fnuz
            | Dtype::F8e8m0
            | Dtype::F8e6m2 => 8,
            Dtype::I4 | Dtype::U4 => 4,
            Dtype::B1 => 1,
            Dtype::C64 => 64,
            Dtype::C128 => 128,
        }
    }

    /// True for the floating-point kinds (`Float`) — the dtypes IEEE-754 (and
    /// the pinned non-IEEE `bf16`/FP8/MX-scale forms) arithmetic applies to. Note
    /// the reserved and MX-scale floats are `Float` by kind but decline compute in
    /// a value position ([`Dtype::declines_compute`]).
    pub const fn is_float(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Float)
    }

    /// True for the signed/unsigned integer kinds — the dtypes the bitwise
    /// atoms (KISS-OPS-6.10) and wrapping integer arithmetic accept.
    pub const fn is_integer(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Int | NumericKind::Uint)
    }

    /// True for the complex kinds (`c64`/`c128`) — the §6.18 complex family.
    pub const fn is_complex(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Complex)
    }

    /// True for the **reserved** FP8 variants (`f8e4m3fnuz`/`f8e5m2fnuz`): part of
    /// the closed sk4 vocabulary (recognized on parse) but with **no computation
    /// semantics at this schema version** (§6.1-0001). Activating a reserved
    /// spelling is a future additive schema event.
    pub const fn is_reserved(self) -> bool {
        matches!(self, Dtype::F8e4m3fnuz | Dtype::F8e5m2fnuz)
    }

    /// True for the OCP-Microscaling **scale** dtypes (`f8e8m0`/`f8e6m2`): a
    /// per-block shared scale carried as a *sibling operand* (§6.1-0013), **never**
    /// an element-value dtype — so it declines compute in a value position.
    pub const fn is_mx_scale(self) -> bool {
        matches!(self, Dtype::F8e8m0 | Dtype::F8e6m2)
    }

    /// True for a dtype that is a **recognized** token of the closed sk4 vocabulary
    /// but has **no element-value compute semantics** at this schema version: the
    /// reserved FP8 variants and the MX scales. A compute cell over such a dtype is
    /// a *typed decline* distinct from the unknown-token verdict — the distinction
    /// [`Dtype::from_token`] preserves (`Some` here, `None` for unknown).
    pub const fn declines_compute(self) -> bool {
        self.is_reserved() || self.is_mx_scale()
    }

    /// The real component-lane dtype of a complex dtype — `c64 → f32`, `c128 → f64`
    /// — i.e. the `f32`/`f64` element of the interleaved (re,im) storage that a
    /// §6.18 complex op evaluates its real-atom decomposition in (§6.18-0015, and
    /// MUST NOT be promoted/demoted across the family's atoms). `None` for a
    /// non-complex dtype.
    pub const fn component_dtype(self) -> Option<Dtype> {
        match self {
            Dtype::C64 => Some(Dtype::F32),
            Dtype::C128 => Some(Dtype::F64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_dtype_set_has_exactly_twenty_four() {
        // KISS-CLASSIFY-6.1-0001 (sk4): the scalar dtype set is exactly 24 tokens.
        assert_eq!(Dtype::ALL.len(), 24);
    }

    #[test]
    fn classify_five_numeric_kinds_present() {
        use NumericKind::*;
        for k in [Float, Int, Uint, Bool, Complex] {
            assert!(
                Dtype::ALL.iter().any(|d| d.numeric_kind() == k),
                "numeric kind {k:?} must be represented"
            );
        }
    }

    #[test]
    fn classify_tokens_round_trip_and_are_unique() {
        for d in Dtype::ALL {
            assert_eq!(Dtype::from_token(d.token()), Some(d), "round-trip {d:?}");
        }
        // Uniqueness: 24 distinct tokens.
        let mut toks: Vec<&str> = Dtype::ALL.iter().map(|d| d.token()).collect();
        toks.sort_unstable();
        toks.dedup();
        assert_eq!(toks.len(), 24, "all 24 tokens must be distinct");
    }

    #[test]
    fn classify_integer_prefixes_uniform_i_at_sk4() {
        // sk4 §3.1.1: the sk3 mixed s/i prefixes become a uniform i-prefix.
        assert_eq!(Dtype::I8.token(), "i8");
        assert_eq!(Dtype::I16.token(), "i16");
        assert_eq!(Dtype::I4.token(), "i4");
        assert_eq!(Dtype::I32.token(), "i32");
        assert_eq!(Dtype::I64.token(), "i64");
    }

    #[test]
    fn classify_fp8_and_complex_sk4_spellings() {
        // FP8 width-prefixed + variant-explicit (§3.1.2/§3.1.5); complex by total
        // width (§3.1.4).
        assert_eq!(Dtype::F8e4m3fn.token(), "f8e4m3fn");
        assert_eq!(Dtype::F8e4m3fnuz.token(), "f8e4m3fnuz");
        assert_eq!(Dtype::F8e5m2.token(), "f8e5m2");
        assert_eq!(Dtype::F8e5m2fnuz.token(), "f8e5m2fnuz");
        assert_eq!(Dtype::F8e8m0.token(), "f8e8m0");
        assert_eq!(Dtype::F8e6m2.token(), "f8e6m2");
        assert_eq!(Dtype::C64.token(), "c64");
        assert_eq!(Dtype::C128.token(), "c128");
    }

    #[test]
    fn classify_sk3_spellings_are_no_longer_dtypes() {
        // The sk3 spellings (and never-valid tokens) must not parse at sk4.
        for bad in [
            "s8", "s16", "s4", "e4m3", "e5m2", "c32", // sk3 spellings, retired
            "s32", "s64", "f8", "int32", "F32", "", "u128", // never valid
        ] {
            assert_eq!(
                Dtype::from_token(bad),
                None,
                "{bad:?} must not parse at sk4"
            );
        }
    }

    #[test]
    fn classify_reserved_and_mx_recognized_but_decline_compute() {
        // §6.1-0001: reserved FP8 variants and MX scales are RECOGNIZED members of
        // the closed vocabulary (from_token is Some — distinct from unknown-token)
        // but carry no element-value compute semantics (declines_compute).
        let declining = [
            Dtype::F8e4m3fnuz,
            Dtype::F8e5m2fnuz,
            Dtype::F8e8m0,
            Dtype::F8e6m2,
        ];
        for d in declining {
            assert_eq!(Dtype::from_token(d.token()), Some(d), "recognized {d:?}");
            assert!(d.declines_compute(), "{d:?} declines compute");
        }
        assert!(Dtype::F8e4m3fnuz.is_reserved() && Dtype::F8e5m2fnuz.is_reserved());
        assert!(Dtype::F8e8m0.is_mx_scale() && Dtype::F8e6m2.is_mx_scale());
        // reserved and MX-scale partition the declining set (no overlap).
        for d in declining {
            assert_ne!(d.is_reserved(), d.is_mx_scale(), "{d:?} in exactly one");
        }
        // Ordinary compute dtypes do NOT decline.
        for d in [
            Dtype::F32,
            Dtype::I8,
            Dtype::F8e4m3fn,
            Dtype::F8e5m2,
            Dtype::C64,
        ] {
            assert!(!d.declines_compute(), "{d:?} is an ordinary compute dtype");
        }
    }

    #[test]
    fn classify_mx_scales_are_float_kind_8_bit() {
        for d in [Dtype::F8e8m0, Dtype::F8e6m2] {
            assert_eq!(d.numeric_kind(), NumericKind::Float, "{d:?} kind");
            assert_eq!(d.bits(), 8, "{d:?} width (exp+mantissa self-check)");
        }
    }

    #[test]
    fn classify_complex_component_dtype() {
        // §6.18-0015 (sk4 names): c64's component lane is f32, c128's is f64;
        // every non-complex dtype has no component lane.
        assert_eq!(Dtype::C64.component_dtype(), Some(Dtype::F32));
        assert_eq!(Dtype::C128.component_dtype(), Some(Dtype::F64));
        for d in Dtype::ALL {
            assert_eq!(
                d.component_dtype().is_some(),
                d.is_complex(),
                "{d:?}: component_dtype is Some iff complex"
            );
        }
    }

    #[test]
    fn classify_bit_widths_match_table() {
        assert_eq!(Dtype::B1.bits(), 1);
        assert_eq!(Dtype::I4.bits(), 4);
        assert_eq!(Dtype::U4.bits(), 4);
        assert_eq!(Dtype::F8e4m3fn.bits(), 8);
        assert_eq!(Dtype::C64.bits(), 64);
        assert_eq!(Dtype::C128.bits(), 128);
    }
}
