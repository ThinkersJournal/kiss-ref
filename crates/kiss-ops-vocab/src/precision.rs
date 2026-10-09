// SPDX-License-Identifier: MIT OR Apache-2.0
//! The KISS-Ops **precision-class** vocabulary (KISS-OPS-6.8-0007..0009, KISS #518).
//!
//! KISS-Ops *owns* the closed set of five `precision_class` tokens that the KISS-Contract
//! Capabilities section carries (KISS-CONTRACT-6.7-0005), and the rule that DERIVES a class from a
//! kernel's declared accuracy tier and its `bit_stability`. The class is a derived label, never an
//! authored fact: [`derive_precision_class`] is the whole rule.
//!
//! Scope: this module binds the vocabulary and the derivation only. It does not read or write a
//! KISS-Contract document (kiss-ref models none).

/// The five precision-class tokens, spelled as KISS-OPS-6.8-0007 spells them, in that clause's order:
/// **tightest first**. `bit-reproducible` is deliberately not a member (it is the `bit_stability` axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrecisionClass {
    /// `strict` — tier 0 on a `portable` kernel.
    Strict,
    /// `correctly-rounded` — tier 0, not portable (0 ULP versus the NAMED reference function).
    CorrectlyRounded,
    /// `bounded-ulp` — a tier carrying `max_ulp > 0`.
    BoundedUlp,
    /// `bounded-tolerance` — a tier carrying only `max_relative` and/or `max_absolute`.
    BoundedTolerance,
    /// `unbounded` — no declared tier.
    Unbounded,
}

impl PrecisionClass {
    /// All five members, tightest to loosest (KISS-OPS-6.8-0007).
    pub const ALL: [PrecisionClass; 5] = [
        PrecisionClass::Strict,
        PrecisionClass::CorrectlyRounded,
        PrecisionClass::BoundedUlp,
        PrecisionClass::BoundedTolerance,
        PrecisionClass::Unbounded,
    ];

    /// The normative token, verbatim.
    pub const fn token(self) -> &'static str {
        match self {
            PrecisionClass::Strict => "strict",
            PrecisionClass::CorrectlyRounded => "correctly-rounded",
            PrecisionClass::BoundedUlp => "bounded-ulp",
            PrecisionClass::BoundedTolerance => "bounded-tolerance",
            PrecisionClass::Unbounded => "unbounded",
        }
    }

    /// Parse a token; `None` for anything outside the closed set (including `bit-reproducible`).
    pub fn from_token(tok: &str) -> Option<PrecisionClass> {
        PrecisionClass::ALL
            .iter()
            .copied()
            .find(|c| c.token() == tok)
    }

    /// Tightness rank: `strict` is the tightest (4), `unbounded` the loosest (0).
    pub const fn tightness(self) -> u8 {
        match self {
            PrecisionClass::Strict => 4,
            PrecisionClass::CorrectlyRounded => 3,
            PrecisionClass::BoundedUlp => 2,
            PrecisionClass::BoundedTolerance => 1,
            PrecisionClass::Unbounded => 0,
        }
    }

    /// The tier kind this class corresponds to (KISS-OPS-6.8-0009). Both tier-0 classes map to `T0`.
    pub const fn tier_kind(self) -> TierKind {
        match self {
            PrecisionClass::Strict | PrecisionClass::CorrectlyRounded => TierKind::T0,
            PrecisionClass::BoundedUlp => TierKind::Tulp,
            PrecisionClass::BoundedTolerance => TierKind::Tother,
            PrecisionClass::Unbounded => TierKind::None,
        }
    }
}

/// The tier kind of a declared accuracy tier (KISS-OPS-6.8-0008). Declared tightest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TierKind {
    /// Carries `max_ulp = 0`.
    T0,
    /// Carries `max_ulp > 0`.
    Tulp,
    /// Carries only `max_relative` and/or `max_absolute`.
    Tother,
    /// No bound declared.
    None,
}

impl TierKind {
    /// Looseness rank used to pick the governing tier: `none` is loosest (3), `T0` tightest (0).
    const fn looseness(self) -> u8 {
        match self {
            TierKind::T0 => 0,
            TierKind::Tulp => 1,
            TierKind::Tother => 2,
            TierKind::None => 3,
        }
    }

    /// The kind of one tier from the bounds it carries (KISS-OPS-6.8-0008): `T0` if `max_ulp = 0`;
    /// else `Tulp` if `max_ulp > 0`; else `Tother` if it carries `max_relative` or `max_absolute`;
    /// else `None`.
    pub fn of(max_ulp: Option<f64>, has_max_relative_or_absolute: bool) -> TierKind {
        match max_ulp {
            Some(0.0) => TierKind::T0,
            Some(u) if u > 0.0 => TierKind::Tulp,
            _ if has_max_relative_or_absolute => TierKind::Tother,
            _ => TierKind::None,
        }
    }

    /// The governing kind of a kernel that declares tiers for several targets: the LOOSEST of them
    /// (`none` loosest, then `Tother`, `Tulp`, `T0`). A kernel declaring no tier is `None`.
    pub fn governing(kinds: &[TierKind]) -> TierKind {
        kinds
            .iter()
            .copied()
            .max_by_key(|k| k.looseness())
            .unwrap_or(TierKind::None)
    }
}

/// The closed reproducibility-scope set of KISS-CONTRACT-6.8-0013 (the only input of the derivation
/// besides the tier kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BitStability {
    /// Bit-identical on any compatible hardware.
    Portable,
    /// Bit-identical on the same hardware.
    SameHardware,
    /// Run-to-run variation is possible.
    None,
}

/// KISS-OPS-6.8-0008: the precision class is a function of the governing tier kind and `bit_stability`.
/// `bit_stability` distinguishes the classes only at tier 0.
pub const fn derive_precision_class(tier: TierKind, bit_stability: BitStability) -> PrecisionClass {
    match (tier, bit_stability) {
        (TierKind::T0, BitStability::Portable) => PrecisionClass::Strict,
        (TierKind::T0, _) => PrecisionClass::CorrectlyRounded,
        (TierKind::Tulp, _) => PrecisionClass::BoundedUlp,
        (TierKind::Tother, _) => PrecisionClass::BoundedTolerance,
        (TierKind::None, _) => PrecisionClass::Unbounded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCOPES: [BitStability; 3] = [
        BitStability::Portable,
        BitStability::SameHardware,
        BitStability::None,
    ];
    const KINDS: [TierKind; 4] = [
        TierKind::T0,
        TierKind::Tulp,
        TierKind::Tother,
        TierKind::None,
    ];

    /// KISS-OPS-6.8-0007 / `test_ops_precision_class_token_set`.
    #[test]
    fn ops_precision_class_token_set() {
        let toks: [&str; 5] = [
            "strict",
            "correctly-rounded",
            "bounded-ulp",
            "bounded-tolerance",
            "unbounded",
        ];
        assert_eq!(PrecisionClass::ALL.map(|c| c.token()), toks);
        for c in PrecisionClass::ALL {
            assert_eq!(PrecisionClass::from_token(c.token()), Some(c));
        }
        assert_eq!(
            PrecisionClass::from_token("bit-reproducible"),
            None,
            "not a member (the bit_stability axis)"
        );
        assert_eq!(
            PrecisionClass::from_token("Strict"),
            None,
            "case-sensitive, verbatim"
        );
        // ordered tightest -> loosest, strictly.
        for w in PrecisionClass::ALL.windows(2) {
            assert!(
                w[0].tightness() > w[1].tightness(),
                "{:?} must be tighter than {:?}",
                w[0],
                w[1]
            );
        }
    }

    /// KISS-OPS-6.8-0008 / `test_ops_precision_class_derivation`: every one of the 12 table cells.
    #[test]
    fn ops_precision_class_derivation() {
        use BitStability::{None as N, Portable as P, SameHardware as S};
        use PrecisionClass::*;
        let table: [(TierKind, BitStability, PrecisionClass); 12] = [
            (TierKind::T0, P, Strict),
            (TierKind::T0, S, CorrectlyRounded),
            (TierKind::T0, N, CorrectlyRounded),
            (TierKind::Tulp, P, BoundedUlp),
            (TierKind::Tulp, S, BoundedUlp),
            (TierKind::Tulp, N, BoundedUlp),
            (TierKind::Tother, P, BoundedTolerance),
            (TierKind::Tother, S, BoundedTolerance),
            (TierKind::Tother, N, BoundedTolerance),
            (TierKind::None, P, Unbounded),
            (TierKind::None, S, Unbounded),
            (TierKind::None, N, Unbounded),
        ];
        for (k, b, want) in table {
            assert_eq!(derive_precision_class(k, b), want, "{k:?} x {b:?}");
        }
        // the table is total: every (kind, scope) pair is one of its rows.
        assert_eq!(KINDS.len() * SCOPES.len(), table.len());
    }

    /// The tier kind of one tier, and the governing kind of several (loosest wins).
    #[test]
    fn ops_tier_kind_and_governing_tier() {
        assert_eq!(TierKind::of(Some(0.0), false), TierKind::T0);
        assert_eq!(
            TierKind::of(Some(0.0), true),
            TierKind::T0,
            "max_ulp = 0 wins over a tolerance"
        );
        assert_eq!(TierKind::of(Some(4.0), true), TierKind::Tulp);
        assert_eq!(TierKind::of(None, true), TierKind::Tother);
        assert_eq!(TierKind::of(None, false), TierKind::None);
        assert_eq!(TierKind::governing(&[]), TierKind::None, "no tier declared");
        assert_eq!(
            TierKind::governing(&[TierKind::T0, TierKind::Tulp]),
            TierKind::Tulp
        );
        assert_eq!(
            TierKind::governing(&[TierKind::Tother, TierKind::T0, TierKind::Tulp]),
            TierKind::Tother
        );
        assert_eq!(
            TierKind::governing(&[TierKind::T0, TierKind::None]),
            TierKind::None
        );
    }

    /// KISS-OPS-6.8-0009 / `test_ops_precision_class_tier_correspondence`: each class maps to its tier
    /// kind, the derivation inverts it, and a looser class never corresponds to a tighter tier.
    #[test]
    fn ops_precision_class_tier_correspondence() {
        assert_eq!(PrecisionClass::Strict.tier_kind(), TierKind::T0);
        assert_eq!(PrecisionClass::CorrectlyRounded.tier_kind(), TierKind::T0);
        assert_eq!(PrecisionClass::BoundedUlp.tier_kind(), TierKind::Tulp);
        assert_eq!(
            PrecisionClass::BoundedTolerance.tier_kind(),
            TierKind::Tother
        );
        assert_eq!(PrecisionClass::Unbounded.tier_kind(), TierKind::None);
        for k in KINDS {
            for b in SCOPES {
                assert_eq!(
                    derive_precision_class(k, b).tier_kind(),
                    k,
                    "the derived class corresponds to its own tier"
                );
            }
        }
        for a in PrecisionClass::ALL {
            for b in PrecisionClass::ALL {
                if a.tightness() > b.tightness() {
                    assert!(
                        a.tier_kind().looseness() <= b.tier_kind().looseness(),
                        "{b:?} (looser) must not sit on a tighter tier than {a:?}"
                    );
                }
            }
        }
    }
}
