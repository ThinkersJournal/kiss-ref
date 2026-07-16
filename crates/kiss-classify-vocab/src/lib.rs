//! # kiss-classify-vocab
//!
//! A Rust binding of the **KISS-Classify** pinned scalar dtype set
//! (`../../KISS/spec/classify.md` §6.1, "The pinned scalar dtype set").
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
//! packing details (`s4`/`u4` packed pairs, `b1` packed byte, `c32`/`c64`
//! interleaved re/im) are documented per-variant and belong to the storage layer.

#![cfg_attr(not(test), no_std)]

/// The five numeric kinds KISS-Classify §6.1 groups dtypes under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericKind {
    /// IEEE-754 or non-IEEE floating-point (`f16 bf16 f32 f64 e4m3 e5m2`).
    Float,
    /// Signed two's-complement integer (`s8 s16 i32 i64 s4`).
    Int,
    /// Unsigned integer (`u8 u16 u32 u64 u4 b1`).
    Uint,
    /// 1-byte truth value (`bool`); `0` = false, any non-zero = true.
    Bool,
    /// Interleaved (re, im) floating-point pair (`c32 c64`).
    Complex,
}

/// The KISS-Classify §6.1 pinned scalar dtype set — exactly 20 dtypes.
///
/// Token spelling is normative and case-sensitive (KISS-CLASSIFY §6.1 / the same
/// exact-spelling discipline KISS-OPS-6.1-0002 applies to ops). Note the spec's
/// mixed signed-integer prefixes are intentional and reproduced verbatim: `s8`,
/// `s16`, `s4` but `i32`, `i64`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Dtype {
    /// IEEE-754 binary16 (1s+5e+10m), half-precision storage.
    F16,
    /// bfloat16 (1s+8e+7m); f32 exponent range, reduced mantissa. Not IEEE-754.
    Bf16,
    /// IEEE-754 binary32 storage. (Compute precision is a KISS-Ops fidelity
    /// attribute, not a dtype — KISS-Classify §6.1.)
    F32,
    /// IEEE-754 binary64.
    F64,
    /// Signed 8-bit two's-complement.
    S8,
    /// Signed 16-bit two's-complement.
    S16,
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
    /// (KISS-CLASSIFY-6.2-0007 / KISS-OPS-6.2-0007).
    U32,
    /// Unsigned 64-bit.
    U64,
    /// 1-byte truth value; storage width equals `u8`; ops normalize to 0/1.
    Bool,
    /// FP8 E4M3 (1s+4e+3m, bias 7); max finite ±448, no infinities, single NaN.
    E4m3,
    /// FP8 E5M2 (1s+5e+2m, bias 15); max finite ±57344, IEEE-style inf/NaN.
    E5m2,
    /// Signed 4-bit `[-8,+7]`; packed-pair storage, sign-extended on read.
    S4,
    /// Unsigned 4-bit `[0,15]`; packed-pair storage, zero-extended on read.
    U4,
    /// 1-bit binary-GEMM operand; packed-byte storage (LSB = lowest index);
    /// xor+popcount accumulation, raw `i32` output.
    B1,
    /// Single-precision complex: interleaved (re, im) pair of `f32`, 64 bits.
    C32,
    /// Double-precision complex: interleaved (re, im) pair of `f64`, 128 bits.
    C64,
}

impl Dtype {
    /// Every dtype in the KISS-Classify §6.1 set, in table order. The coverage
    /// gate iterates this to enumerate the (atom × dtype) matrix.
    pub const ALL: [Dtype; 20] = [
        Dtype::F16,
        Dtype::Bf16,
        Dtype::F32,
        Dtype::F64,
        Dtype::S8,
        Dtype::S16,
        Dtype::U8,
        Dtype::U16,
        Dtype::I32,
        Dtype::I64,
        Dtype::U32,
        Dtype::U64,
        Dtype::Bool,
        Dtype::E4m3,
        Dtype::E5m2,
        Dtype::S4,
        Dtype::U4,
        Dtype::B1,
        Dtype::C32,
        Dtype::C64,
    ];

    /// The normative, case-sensitive KISS token for this dtype.
    pub const fn token(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::S8 => "s8",
            Dtype::S16 => "s16",
            Dtype::U8 => "u8",
            Dtype::U16 => "u16",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::U32 => "u32",
            Dtype::U64 => "u64",
            Dtype::Bool => "bool",
            Dtype::E4m3 => "e4m3",
            Dtype::E5m2 => "e5m2",
            Dtype::S4 => "s4",
            Dtype::U4 => "u4",
            Dtype::B1 => "b1",
            Dtype::C32 => "c32",
            Dtype::C64 => "c64",
        }
    }

    /// Parse a dtype from its normative token. Returns `None` for any token
    /// outside the §6.1 set (KISS-CLASSIFY: a token outside the set is not a
    /// dtype of this version).
    pub fn from_token(tok: &str) -> Option<Dtype> {
        Dtype::ALL.into_iter().find(|d| d.token() == tok)
    }

    /// The numeric kind this dtype belongs to (§6.1 table).
    pub const fn numeric_kind(self) -> NumericKind {
        match self {
            Dtype::F16 | Dtype::Bf16 | Dtype::F32 | Dtype::F64 | Dtype::E4m3 | Dtype::E5m2 => {
                NumericKind::Float
            }
            Dtype::S8 | Dtype::S16 | Dtype::I32 | Dtype::I64 | Dtype::S4 => NumericKind::Int,
            Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64 | Dtype::U4 | Dtype::B1 => {
                NumericKind::Uint
            }
            Dtype::Bool => NumericKind::Bool,
            Dtype::C32 | Dtype::C64 => NumericKind::Complex,
        }
    }

    /// The logical bit width from the §6.1 table (4 for `s4`/`u4`, 1 for `b1`,
    /// 64 for `c32`, 128 for `c64`). This is the logical width, not the
    /// container/storage width (which differs for the packed sub-byte types).
    pub const fn bits(self) -> u16 {
        match self {
            Dtype::F16 | Dtype::Bf16 => 16,
            Dtype::F32 => 32,
            Dtype::F64 => 64,
            Dtype::S8 | Dtype::U8 | Dtype::Bool | Dtype::E4m3 | Dtype::E5m2 => 8,
            Dtype::S16 | Dtype::U16 => 16,
            Dtype::I32 | Dtype::U32 => 32,
            Dtype::I64 | Dtype::U64 => 64,
            Dtype::S4 | Dtype::U4 => 4,
            Dtype::B1 => 1,
            Dtype::C32 => 64,
            Dtype::C64 => 128,
        }
    }

    /// True for the floating-point kinds (`Float`) — the dtypes IEEE-754 (and
    /// the pinned non-IEEE `bf16`/`e4m3`/`e5m2`) arithmetic applies to.
    pub const fn is_float(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Float)
    }

    /// True for the signed/unsigned integer kinds — the dtypes the bitwise
    /// atoms (KISS-OPS-6.10) and wrapping integer arithmetic accept.
    pub const fn is_integer(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Int | NumericKind::Uint)
    }

    /// True for the complex kinds (`c32`/`c64`) — the §6.18 complex family.
    pub const fn is_complex(self) -> bool {
        matches!(self.numeric_kind(), NumericKind::Complex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_dtype_set_has_exactly_twenty() {
        // KISS-Classify §6.1: "Twenty dtypes, five numeric kinds."
        assert_eq!(Dtype::ALL.len(), 20);
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
        // Uniqueness: 20 distinct tokens.
        let mut toks: Vec<&str> = Dtype::ALL.iter().map(|d| d.token()).collect();
        toks.sort_unstable();
        toks.dedup();
        assert_eq!(toks.len(), 20, "all 20 tokens must be distinct");
    }

    #[test]
    fn classify_signed_int_prefixes_match_spec_verbatim() {
        // The spec's mixed prefixes are intentional and must be reproduced:
        // s8/s16/s4 but i32/i64 (KISS-CLASSIFY §6.1 spelling).
        assert_eq!(Dtype::S8.token(), "s8");
        assert_eq!(Dtype::S16.token(), "s16");
        assert_eq!(Dtype::S4.token(), "s4");
        assert_eq!(Dtype::I32.token(), "i32");
        assert_eq!(Dtype::I64.token(), "i64");
    }

    #[test]
    fn classify_unknown_token_is_not_a_dtype() {
        for bad in ["s32", "s64", "f8", "int32", "F32", "", "u128"] {
            assert_eq!(Dtype::from_token(bad), None, "{bad:?} must not parse");
        }
    }

    #[test]
    fn classify_bit_widths_match_table() {
        assert_eq!(Dtype::B1.bits(), 1);
        assert_eq!(Dtype::S4.bits(), 4);
        assert_eq!(Dtype::U4.bits(), 4);
        assert_eq!(Dtype::E4m3.bits(), 8);
        assert_eq!(Dtype::C32.bits(), 64);
        assert_eq!(Dtype::C64.bits(), 128);
    }
}
