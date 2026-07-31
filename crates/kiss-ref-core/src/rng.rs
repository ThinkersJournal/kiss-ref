//! Counter-based RNG — the Philox core of the `RandomBits` floor atom (KISS-Classify
//! §6.8 `rnd` family; the cross-backend bit-identical generator seam co-designed with
//! Fuel, 2026-07-31).
//!
//! A counter-based generator is a **pure deterministic function** of `(key, counter)`
//! — not "randomness" in the nondeterministic sense — so it is a bit-exact integer
//! kernel: [`crate::bridge::DetClass::ExactByte`], exact-equality oracle, no ULP
//! tolerance and no reduction-accumulator ambiguity. That is what lets a stochastic op
//! (dropout etc.) stay testable by the same differential oracle as every other atom:
//! `bernoulli_mask(p) = RandomBits < floor(p·2^32)` is an integer compare producing a
//! `b1` mask, so it never leaves the exact lane.
//!
//! **Provenance / discipline.** kiss-ref is the differential TARGET, never the oracle —
//! and for RNG that means it is not the oracle for the algorithm's numbers either. The
//! authority for Philox-4x32-10 is the upstream published known-answer test set
//! (`DEShawResearch/random123`, `tests/kat_vectors`, blob @ `d8a0c25e`, 2021-01-17);
//! this implementation is **checked against** it (see the module test), never the source
//! of it. The KAT vectors in the test were pulled from upstream, not recalled — a
//! recalled value for the all-`ffff` case was wrong (`6b562217` vs the real `6d5451fd`),
//! which is exactly the poisoned-anchor failure the pull-from-upstream rule prevents.
//!
//! **Scope (this cut):** the Philox-4x32-10 core `(counter, key) -> 4×u32`. The
//! `RandomBits` atom — the §8 counter-derivation mapping (logical row-major index →
//! block/base/stream split → per-element word) and the two-layer conformance corpus —
//! is the next increment on top of this verified core.

/// Philox-4x32 first multiplier (`M0`, Salmon et al. 2011 / Random123 `philox.h`).
const PHILOX_M4X32_0: u32 = 0xD251_1F53;
/// Philox-4x32 second multiplier (`M1`).
const PHILOX_M4X32_1: u32 = 0xCD9E_8D57;
/// Weyl-sequence key increment for word 0 (golden ratio, `W32_0`).
const PHILOX_W32_0: u32 = 0x9E37_79B9;
/// Weyl-sequence key increment for word 1 (`sqrt(3)-1`, `W32_1`).
const PHILOX_W32_1: u32 = 0xBB67_AE85;

/// The 64-bit product of two `u32`s, split into `(high 32, low 32)`. Exact — a
/// `u32·u32` product always fits in `u64`, so there is no wrapping.
#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let prod = (a as u64) * (b as u64);
    ((prod >> 32) as u32, prod as u32)
}

/// One Philox-4x32 round (Random123 `_philox4x32round`).
#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(PHILOX_M4X32_0, ctr[0]);
    let (hi1, lo1) = mulhilo(PHILOX_M4X32_1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

/// Advance the key by the Weyl increments between rounds (Random123
/// `_philox4x32bumpkey`).
#[inline]
fn bumpkey(key: [u32; 2]) -> [u32; 2] {
    [
        key[0].wrapping_add(PHILOX_W32_0),
        key[1].wrapping_add(PHILOX_W32_1),
    ]
}

/// Philox-4x32-10: 10 rounds of the Philox-4x32 bijection over the 128-bit `counter`
/// keyed by the 64-bit `key`, returning 4×`u32`. Argument order matches the spec's
/// `philox4x32_R(10, counter, key)`. Pure and total — `ExactByte`.
pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    // Random123 `philox4x32_R` schedule: round 1 uses the original key; each of
    // rounds 2..=10 bumps the key first (9 bumps for 10 rounds).
    for i in 0..10 {
        if i > 0 {
            k = bumpkey(k);
        }
        ctr = round(ctr, k);
    }
    ctr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox4x32_10_matches_random123_kat() {
        // Anchor: DEShawResearch/random123, tests/kat_vectors, blob @ d8a0c25e
        // (2021-01-17). Fetched from upstream, NOT recalled — the all-ffff case is
        // 6d5451fd upstream, where a recalled value had the wrong 6b562217.
        // Format: philox4x32 10 <counter[4]> <key[2]> -> <expected[4]>.
        let kat: &[([u32; 4], [u32; 2], [u32; 4])] = &[
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
            ),
            (
                [0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
                [0xffff_ffff, 0xffff_ffff],
                [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
            ),
            (
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0],
                [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
            ),
        ];
        for (ctr, key, want) in kat {
            assert_eq!(
                philox4x32_10(*ctr, *key),
                *want,
                "philox4x32-10(ctr={ctr:08x?}, key={key:08x?})"
            );
        }
    }
}
