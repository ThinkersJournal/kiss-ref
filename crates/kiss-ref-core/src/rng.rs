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

    #[test]
    fn philox4x32_10_matches_random123_extended_kat() {
        // Extended coverage from the SAME upstream repo's superseded systematic set
        // (DEShawResearch/random123, tests/old_kat_vectors) — 52 vectors walking
        // single-bit and structured (counter, key) inputs. Redundant for proving THIS
        // core (the canonical three already are conclusive), but a broad systematic
        // surface is where a shared-blind-spot bug would surface in the eventual
        // independent cross-implementation diff — far more places to disagree than
        // three points. Machine-transcribed from the upstream file (awk → literals),
        // never hand-typed: transcription is recall with extra steps.
        let kat: &[([u32; 4], [u32; 2], [u32; 4])] = &[
            (
                [0x00000001, 0x00000000, 0x00000000, 0x00000000],
                [0x00000001, 0x00000000],
                [0xac08141b, 0xdfc5ccbe, 0x79c07a47, 0xa7f66093],
            ),
            (
                [0x00000000, 0x00000001, 0x00000000, 0x00000000],
                [0x00000001, 0x00000000],
                [0x833ff8dc, 0x225c963c, 0x232b88b3, 0x0b334b07],
            ),
            (
                [0x00000000, 0x00000000, 0x00000001, 0x00000000],
                [0x00000001, 0x00000000],
                [0x07071c12, 0x428264b6, 0x3909104b, 0x6da2bda2],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x00000001],
                [0x00000001, 0x00000000],
                [0x127bee41, 0x1e047488, 0x48842b20, 0x0a393496],
            ),
            (
                [0x00000000, 0xffffffff, 0x00000000, 0x00000000],
                [0x00000001, 0x00000000],
                [0x04e2eae1, 0x463cc999, 0xe7786310, 0x65b3dc49],
            ),
            (
                [0x00000000, 0x80000000, 0x00000000, 0x00000000],
                [0x00000001, 0x00000000],
                [0x34a4d61d, 0x340e4017, 0xf6945830, 0xe9f52b96],
            ),
            (
                [0x00000000, 0x00000000, 0xffffffff, 0x00000000],
                [0x00000001, 0x00000000],
                [0xfb0905e2, 0x78378790, 0xca37926b, 0xcdf58cfa],
            ),
            (
                [0x00000000, 0x00000000, 0x80000000, 0x00000000],
                [0x00000001, 0x00000000],
                [0xf2e15c7d, 0x3fedcd99, 0x90046f6c, 0x6657f0ca],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0xffffffff],
                [0x00000001, 0x00000000],
                [0xf50c9398, 0xb36cdab2, 0x436bdc89, 0xcc49431b],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x80000000],
                [0x00000001, 0x00000000],
                [0xdde26e4e, 0x17666f3b, 0xfc1f5b3a, 0x83ac805d],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0x00000001, 0x00000000],
                [0x38527b8f, 0xfad9a48c, 0x20e675c0, 0xf2fea26a],
            ),
            (
                [0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89],
                [0x00000001, 0x00000000],
                [0xfb9877f8, 0x482895e2, 0x60b01453, 0x198ea67b],
            ),
            (
                [0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c],
                [0x00000001, 0x00000000],
                [0xf7b7fa0a, 0x8d970e2c, 0x205c9c76, 0x83b16ce5],
            ),
            (
                [0x00000001, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000001],
                [0xccf3076a, 0x6306f434, 0xbea71965, 0xd96fe48a],
            ),
            (
                [0x00000000, 0x00000001, 0x00000000, 0x00000000],
                [0x00000000, 0x00000001],
                [0x532f1322, 0x36c890fb, 0x6504b937, 0x07bbfef7],
            ),
            (
                [0x00000000, 0x00000000, 0x00000001, 0x00000000],
                [0x00000000, 0x00000001],
                [0x7f6c13ff, 0x114b2447, 0x0550eced, 0xea1fc805],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x00000001],
                [0x00000000, 0x00000001],
                [0x96b27e8c, 0x28826d0f, 0x66d958ef, 0x081509dc],
            ),
            (
                [0x00000000, 0xffffffff, 0x00000000, 0x00000000],
                [0x00000000, 0x00000001],
                [0x00e94a67, 0x3524f44e, 0x22995230, 0xbb54df1e],
            ),
            (
                [0x00000000, 0x80000000, 0x00000000, 0x00000000],
                [0x00000000, 0x00000001],
                [0xce1afc4e, 0x4fbf97a1, 0x6110bee9, 0xa559be2f],
            ),
            (
                [0x00000000, 0x00000000, 0xffffffff, 0x00000000],
                [0x00000000, 0x00000001],
                [0x576241af, 0xd3b45bee, 0x619bd94d, 0x15dd8793],
            ),
            (
                [0x00000000, 0x00000000, 0x80000000, 0x00000000],
                [0x00000000, 0x00000001],
                [0xd9da65af, 0xeced57ba, 0xc34d2661, 0x91fe931f],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0xffffffff],
                [0x00000000, 0x00000001],
                [0x36241728, 0x4d42c858, 0xfea737d6, 0x49f40347],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x80000000],
                [0x00000000, 0x00000001],
                [0x0436a39b, 0x31be9efa, 0x6abd0c76, 0xe7a68f83],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0x00000000, 0x00000001],
                [0x126514c3, 0xd1898b69, 0xda2707b3, 0x6e6fb436],
            ),
            (
                [0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89],
                [0x00000000, 0x00000001],
                [0x1c28fa3f, 0x8c5bf977, 0xbb989783, 0x0c1a3737],
            ),
            (
                [0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c],
                [0x00000000, 0x00000001],
                [0xac87a8c1, 0xf240abdb, 0xadd6c012, 0x68a32a78],
            ),
            (
                [0x00000001, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0xffffffff],
                [0x473ab778, 0x330bacc0, 0xacd5fc9a, 0x5ede76b3],
            ),
            (
                [0x00000000, 0x00000001, 0x00000000, 0x00000000],
                [0x00000000, 0xffffffff],
                [0xcdfbd51c, 0xcaae5f2e, 0x10739cfc, 0x79e95162],
            ),
            (
                [0x00000000, 0x00000000, 0x00000001, 0x00000000],
                [0x00000000, 0xffffffff],
                [0x6c4b0e80, 0x357e5a94, 0x3255445d, 0x6f433f0b],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x00000001],
                [0x00000000, 0xffffffff],
                [0x5c4ecd0c, 0x64c12cf9, 0x8f9fb4c8, 0x40c4a01f],
            ),
            (
                [0x00000000, 0xffffffff, 0x00000000, 0x00000000],
                [0x00000000, 0xffffffff],
                [0x5f9c47aa, 0x19861d86, 0xff2be81d, 0x4c7db17e],
            ),
            (
                [0x00000000, 0x80000000, 0x00000000, 0x00000000],
                [0x00000000, 0xffffffff],
                [0x715a87cf, 0xf966a7f0, 0x3e6dd95f, 0xe884e2fd],
            ),
            (
                [0x00000000, 0x00000000, 0xffffffff, 0x00000000],
                [0x00000000, 0xffffffff],
                [0x07678411, 0x9e524307, 0xba634e67, 0x94ee8cb7],
            ),
            (
                [0x00000000, 0x00000000, 0x80000000, 0x00000000],
                [0x00000000, 0xffffffff],
                [0xb6f48e28, 0xf2fa9414, 0x00b64f66, 0x4810cdb4],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0xffffffff],
                [0x00000000, 0xffffffff],
                [0x9f18bb20, 0x62161056, 0x7ea8c028, 0x4c01fc6e],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x80000000],
                [0x00000000, 0xffffffff],
                [0x2a460458, 0x0a2709e5, 0x47bc9115, 0x0f96d8af],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0x00000000, 0xffffffff],
                [0x08f7720c, 0x845ee9e0, 0x502ccd3d, 0xa9a00e27],
            ),
            (
                [0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89],
                [0x00000000, 0xffffffff],
                [0x9fcce05d, 0x837d3f3c, 0x5cdf3e61, 0xdb2544c0],
            ),
            (
                [0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c],
                [0x00000000, 0xffffffff],
                [0x72ce44b3, 0xb44f9b78, 0x49af0267, 0x726813ac],
            ),
            (
                [0x00000001, 0x00000000, 0x00000000, 0x00000000],
                [0x00000000, 0x80000000],
                [0xe29b9f84, 0x2903d316, 0x42a1f56f, 0x1ad02eab],
            ),
            (
                [0x00000000, 0x00000001, 0x00000000, 0x00000000],
                [0x00000000, 0x80000000],
                [0x94f419f8, 0x1116c316, 0x99f2fa30, 0xbc048ad6],
            ),
            (
                [0x00000000, 0x00000000, 0x00000001, 0x00000000],
                [0x00000000, 0x80000000],
                [0x488b0258, 0x9eea6420, 0xafe7062e, 0x768f5637],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x00000001],
                [0x00000000, 0x80000000],
                [0xdbec6d70, 0x69cea528, 0xf7e22262, 0x2e9f3e8f],
            ),
            (
                [0x00000000, 0xffffffff, 0x00000000, 0x00000000],
                [0x00000000, 0x80000000],
                [0x905ee550, 0xc7520e9a, 0x9af9d0ad, 0x69b97e98],
            ),
            (
                [0x00000000, 0x80000000, 0x00000000, 0x00000000],
                [0x00000000, 0x80000000],
                [0x1b89d963, 0x0defe9af, 0x60c9d3fb, 0x3cfae97d],
            ),
            (
                [0x00000000, 0x00000000, 0xffffffff, 0x00000000],
                [0x00000000, 0x80000000],
                [0x601b5460, 0xfe29678b, 0x54510a02, 0xc72ed094],
            ),
            (
                [0x00000000, 0x00000000, 0x80000000, 0x00000000],
                [0x00000000, 0x80000000],
                [0xae56ca02, 0x7609ba67, 0x25abd7df, 0x46ce39a3],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0xffffffff],
                [0x00000000, 0x80000000],
                [0x09812238, 0x47b8470d, 0x3e0d0a26, 0x92d96c9a],
            ),
            (
                [0x00000000, 0x00000000, 0x00000000, 0x80000000],
                [0x00000000, 0x80000000],
                [0xf7a0b371, 0xe51dc93d, 0x56ce135d, 0x1d4b5580],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0x00000000, 0x80000000],
                [0x63675d2c, 0x422d7c4d, 0x3145d30d, 0x3bce33a4],
            ),
            (
                [0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89],
                [0x00000000, 0x80000000],
                [0xaa74e81c, 0xc9ddb97b, 0x2c2cd881, 0x9390c78d],
            ),
            (
                [0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c],
                [0x00000000, 0x80000000],
                [0x94ca5566, 0x14924b2f, 0x28214292, 0x59faeaee],
            ),
        ];
        assert_eq!(kat.len(), 52, "extended KAT count");
        for (ctr, key, want) in kat {
            assert_eq!(
                philox4x32_10(*ctr, *key),
                *want,
                "philox4x32-10(ctr={ctr:08x?}, key={key:08x?})"
            );
        }
    }
}
