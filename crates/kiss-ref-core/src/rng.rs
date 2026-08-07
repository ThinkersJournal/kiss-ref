//! Counter-based RNG — the `RandomBits` floor atom: the Philox-4x32-10 core plus the
//! §8 counter-derivation mapping (KISS-Classify §6.8 `rnd` family; the cross-backend
//! bit-identical generator seam co-designed with Fuel, 2026-07-31).
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
//! **Scope (this cut):** the Philox-4x32-10 core `(counter, key) -> 4×u32` and the
//! [`random_bits`] atom — the §8 mapping (logical row-major index `i` → `block_index =
//! i/4` little-endian in `counter[0..1]`, `base`/`stream` in `counter[2..3]`, word =
//! `philox(...)[i % 4]`) filling a fresh row-major `u32` tensor (held as non-negative
//! `i128` for the integer lane). Both layers of the conformance corpus are present: the
//! algorithm anchor (55 upstream KAT vectors) and the mapping (the block-0-is-KAT-words
//! join to the anchor, the little-endian split, and the increment-coherence *structural*
//! check that advancing a block equals one Random123 `incr()`). The three §6.13
//! distribution decompositions are here too: `bernoulli_mask` (integer compare,
//! `ExactByte`), `uniform_f32` (the §8.1 mantissa splice, `ExactByte`), and
//! `normal_f32` (§8.2 basic Box-Muller — NOT `ExactByte`, its transcendentals carry a
//! tolerance; tested for structure — no NaN, the Pythagorean pairing identity — never a
//! guessed bound). Deferred: a recipe-grammar `RandomBits` node (a breaking 0.3.0 no
//! consumer needs — the diff surface is these value functions; the sk4 coordinated
//! 0.3.0 is the natural window to add it IF a consumer materializes, currently none)
//! and `normal_f32`'s numeric conformance tolerance (open, sabotage-calibrated pending
//! a second backend).

extern crate alloc;
use alloc::vec::Vec;

use crate::tensor::{alloc_exact, numel, Tensor};
use crate::Error;

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

/// The §8 per-element counter derivation: a logical row-major `linear_index` maps to
/// the Philox counter. `block_index = linear_index / 4` occupies `counter[0..1]`
/// **little-endian** (`block_lo` in `counter[0]`, the fastest-varying word — so
/// advancing the block by one is exactly one Random123 `incr()`); the runtime-bound
/// `base` is `counter[2]` and the build-time `stream` id is `counter[3]`.
fn derive_counter(linear_index: u64, base: u32, stream: u32) -> [u32; 4] {
    let block_index = linear_index / 4;
    [
        block_index as u32,         // block_lo — low 32 bits, the fastest-varying word
        (block_index >> 32) as u32, // block_hi
        base,
        stream,
    ]
}

/// `RandomBits`: fill a fresh contiguous row-major tensor of `shape` with per-element
/// Philox-4x32-10 draws (KISS §6.8 `rnd`). `seed` is the build-time key
/// (`key = [seed_lo, seed_hi]`), `base` the per-dispatch step scalar, `stream` the
/// build-time stream id. Logical element `i` (row-major) draws
/// `philox(derive_counter(i), key)[i % 4]`, so four consecutive elements share one
/// Philox evaluation. Values are `u32`, held as non-negative `i128` for the integer
/// lane. Pure and deterministic — `ExactByte`.
pub fn random_bits(
    shape: &[usize],
    seed: u64,
    base: u32,
    stream: u32,
) -> Result<Tensor<i128>, Error> {
    let key = [seed as u32, (seed >> 32) as u32];
    let count = numel(shape)?;
    let mut data: Vec<i128> = alloc_exact(count)?;
    // Reference form — one Philox evaluation per element. The crate is deliberately
    // "obvious, slow"; a backend evaluates once per 4-aligned block and keeps all four
    // words (§8 non-normative), but that optimization is not the reference's job.
    for i in 0..count as u64 {
        let out = philox4x32_10(derive_counter(i, base, stream), key);
        data.push(out[(i % 4) as usize] as i128);
    }
    Tensor::from_vec(data, shape)
}

/// `bernoulli_mask` — the §6.13 dropout-mask decomposition over [`random_bits`]: a
/// `b1` mask where element `i` is `1` iff `draw[i] < threshold`, else `0`. This is a
/// pure **integer compare** on the u32 draw (KISS §6.13: `bernoulli_mask(p) =
/// RandomBits < floor(p·2^32)`), so it stays entirely on the exact lane — no float
/// touches the highest-traffic stochastic op. `threshold` is the caller-supplied
/// `floor(p·2^32)` in `[0, 2^32]` (`u64` so `p = 1 → 2^32` is representable, giving an
/// all-ones mask); the reference owns only the exact-integer compare, not the
/// `p → threshold` conversion (whose float rounding is a separate spec point).
/// `ExactByte`.
pub fn bernoulli_mask(
    shape: &[usize],
    seed: u64,
    base: u32,
    stream: u32,
    threshold: u64,
) -> Result<Tensor<i128>, Error> {
    let draws = random_bits(shape, seed, base, stream)?;
    let mut data: Vec<i128> = alloc_exact(draws.as_slice().len())?;
    for &d in draws.as_slice() {
        // `d` is a u32 draw held in i128 (0..2^32); an exact-integer compare.
        data.push(i128::from((d as u64) < threshold));
    }
    Tensor::from_vec(data, shape)
}

/// The §8.1 `uniform_f32` splice: a u32 draw's top 23 bits (`w >> 9`) become the
/// mantissa of an f32 in `[1, 2)`, minus `1.0` → `[0, 1)` half-open (`0.0` attainable,
/// `1.0` not; spacing `2^-23`). `ExactByte` — integer ops plus a subtraction that is
/// exact by Sterbenz for operands in `[1, 2)`.
fn uniform_word(w: u32) -> f32 {
    f32::from_bits(0x3F80_0000 | (w >> 9)) - 1.0
}

/// `uniform_f32` — §8.1: `uniform_word` applied to each draw of [`random_bits`],
/// yielding a `[0, 1)` f32 tensor. `ExactByte`.
pub fn uniform_f32(
    shape: &[usize],
    seed: u64,
    base: u32,
    stream: u32,
) -> Result<Tensor<f32>, Error> {
    let draws = random_bits(shape, seed, base, stream)?;
    let mut data: Vec<f32> = alloc_exact(draws.as_slice().len())?;
    for &d in draws.as_slice() {
        data.push(uniform_word(d as u32));
    }
    Tensor::from_vec(data, shape)
}

/// Basic Box-Muller for one PAIR of outputs (§8.2): `r = sqrt(-2·ln(1 - u1))`,
/// `theta = 2·pi·u2`, returning `(r·cos(theta), r·sin(theta))` — the even and odd
/// elements of the pair, which share this single `(r, theta)` computation (one `ln` +
/// `sqrt` per pair, not per element). The `1 - u1` reflection maps `u1 ∈ [0,1)` to
/// `(0,1]`, removing the `ln(0)` singularity a naive `ln(u1)` hits when a draw yields
/// `u1 = 0` (a `2^-23` event, routine at tensor scale) — `u1 = 0` then gives `r = 0`, so
/// both outputs are `0`, finite. **NOT `ExactByte`**: `ln`/`sqrt`/`cos`/`sin` are not
/// bit-identical across backends.
fn box_muller_pair(u1: f32, u2: f32) -> (f32, f32) {
    let r = libm::sqrtf(-2.0 * libm::logf(1.0 - u1));
    let theta = 2.0 * core::f32::consts::PI * u2;
    (r * libm::cosf(theta), r * libm::sinf(theta))
}

/// `normal_f32` — §8.2: basic Box-Muller over pairs of [`uniform_f32`] draws. Element
/// `i` belongs to pair `p = i/2` and consumes draws at logical indices `2p` (`u1`) and
/// `2p+1` (`u2`); even elements take `r·cos(theta)`, odd `r·sin(theta)`, so consecutive
/// elements share one `(r, theta)`. Position-pure (element `i` depends only on `i`); an
/// odd element count simply leaves the final pair's `z1` uncomputed. **NOT `ExactByte`**
/// (see `box_muller_pair`); its numeric conformance tolerance is an open, to-be-
/// sabotage-calibrated item, so the reference pins structure — never a guessed bound.
pub fn normal_f32(
    shape: &[usize],
    seed: u64,
    base: u32,
    stream: u32,
) -> Result<Tensor<f32>, Error> {
    let key = [seed as u32, (seed >> 32) as u32];
    let count = numel(shape)?;
    // word(k) = the §8 RandomBits draw at logical index k, computed on demand so the
    // last even element of an odd count can reach its pair partner at index `count`.
    let word =
        |k: u64| -> u32 { philox4x32_10(derive_counter(k, base, stream), key)[(k % 4) as usize] };
    let mut data: Vec<f32> = alloc_exact(count)?;
    // Iterate by PAIR: one (r, theta) per pair, emitting z0 = r·cos and z1 = r·sin. A
    // trailing odd element takes only z0 (its pair's z1 is left uncomputed). Element i
    // still depends only on i, so the op stays position-pure.
    let mut i = 0u64;
    while (i as usize) < count {
        let (z0, z1) = box_muller_pair(uniform_word(word(i)), uniform_word(word(i + 1)));
        data.push(z0);
        if (i as usize) + 1 < count {
            data.push(z1);
        }
        i += 2;
    }
    Tensor::from_vec(data, shape)
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

    // ---- mapping layer (§8 counter derivation / RandomBits atom) -------------

    #[test]
    fn random_bits_block0_is_the_four_philox_words() {
        // The mapping's join to the algorithm anchor: shape [4], seed=base=stream=0 →
        // block 0's counter is [0,0,0,0] with key [0,0], so the four elements ARE the
        // canonical KAT block-0 words in Random123 lane order. Ties the atom directly
        // to the verified core, not to itself.
        let t = random_bits(&[4], 0, 0, 0).unwrap();
        let want: [i128; 4] = [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];
        assert_eq!(t.as_slice(), &want);
        assert_eq!(t.shape(), &[4]);
    }

    #[test]
    fn derive_counter_little_endian_split() {
        // block_index = linear_index / 4 splits little-endian: block_lo (low 32) in
        // counter[0], block_hi in counter[1]; base -> counter[2], stream -> counter[3].
        // block_index 1:
        assert_eq!(derive_counter(4, 7, 9), [1, 0, 7, 9]);
        // block_index 2^32 (block_lo wraps to 0, block_hi becomes 1):
        assert_eq!(derive_counter((1u64 << 32) * 4, 7, 9), [0, 1, 7, 9]);
        // all four elements of a block share one counter:
        for i in 0..4u64 {
            assert_eq!(derive_counter(i, 7, 9), [0, 0, 7, 9], "elem {i} in block 0");
        }
        // base and stream occupy their own slots:
        assert_eq!(
            derive_counter(0, 0xdead_beef, 0xfeed_face),
            [0, 0, 0xdead_beef, 0xfeed_face]
        );
    }

    #[test]
    fn derive_counter_increment_coherence() {
        // The invariant behind putting block_lo in counter[0]: advancing the block by
        // one is byte-identical to one Random123 counter incr() — bump counter[0],
        // carry upward, never reaching base/stream. A *structural* check: value tests
        // catch a wrong byte, this catches a wrong LAYOUT.
        fn incr(mut c: [u32; 4]) -> [u32; 4] {
            c[0] = c[0].wrapping_add(1);
            if c[0] == 0 {
                c[1] = c[1].wrapping_add(1);
                if c[1] == 0 {
                    c[2] = c[2].wrapping_add(1);
                    if c[2] == 0 {
                        c[3] = c[3].wrapping_add(1);
                    }
                }
            }
            c
        }
        let (base, stream) = (0x1111_2222u32, 0x3333_4444u32);
        // Non-carrying block step.
        let b = 5u64;
        assert_eq!(
            derive_counter((b + 1) * 4, base, stream),
            incr(derive_counter(b * 4, base, stream))
        );
        // Carrying block step: block_lo 0xffffffff -> 0, carry into block_hi; base and
        // stream MUST stay put (the carry never reaches counter[2..3]).
        let bc = 0xffff_ffffu64;
        assert_eq!(
            derive_counter(bc * 4, base, stream),
            [0xffff_ffff, 0, base, stream]
        );
        assert_eq!(
            derive_counter((bc + 1) * 4, base, stream),
            [0, 1, base, stream]
        );
        assert_eq!(
            derive_counter((bc + 1) * 4, base, stream),
            incr(derive_counter(bc * 4, base, stream))
        );
    }

    #[test]
    fn random_bits_maps_every_element_per_spec() {
        // Each logical element i draws philox(derive_counter(i), key)[i % 4]; a 3-block
        // [3,4] shape exercises three distinct blocks (element 4 is block 1, element 8
        // block 2). Checked against the verified core + derive_counter directly.
        let seed = 0x0123_4567_89ab_cdefu64;
        let (base, stream) = (0x2au32, 0x7u32);
        let key = [seed as u32, (seed >> 32) as u32];
        let t = random_bits(&[3, 4], seed, base, stream).unwrap();
        assert_eq!(t.shape(), &[3, 4]);
        for i in 0..12u64 {
            let want = philox4x32_10(derive_counter(i, base, stream), key)[(i % 4) as usize];
            assert_eq!(t.as_slice()[i as usize], want as i128, "element {i}");
        }
        // The draws are u32-valued, held non-negative in i128.
        assert!(t.as_slice().iter().all(|&v| (0..1i128 << 32).contains(&v)));
    }

    // ---- bernoulli decomposition (§6.13 dropout mask) -----------------------

    #[test]
    fn bernoulli_mask_is_kat_anchored_integer_compare() {
        // Block-0 draws (seed=base=stream=0) are the canonical KAT words
        // [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]. threshold 2^31 (p=0.5):
        // mask[i] = (draw[i] < 0x8000_0000) — only the first is below → [1,0,0,0].
        let m = bernoulli_mask(&[4], 0, 0, 0, 0x8000_0000).unwrap();
        assert_eq!(m.as_slice(), &[1, 0, 0, 0]);
        assert_eq!(m.shape(), &[4]);
    }

    #[test]
    fn bernoulli_mask_threshold_extremes() {
        // threshold 0 (p=0) → nothing is < 0 → all-zero mask.
        let z = bernoulli_mask(&[4], 0, 0, 0, 0).unwrap();
        assert_eq!(z.as_slice(), &[0, 0, 0, 0]);
        // threshold 2^32 (p=1) → every u32 draw is < 2^32 → all-ones mask.
        let o = bernoulli_mask(&[4], 0, 0, 0, 1 << 32).unwrap();
        assert_eq!(o.as_slice(), &[1, 1, 1, 1]);
    }

    #[test]
    fn bernoulli_mask_matches_draws_and_threshold() {
        // Consistency with the atom: mask[i] == (random_bits[i] < threshold) over a
        // multi-block shape with a non-trivial threshold.
        let (seed, base, stream, thr) = (0xdead_beef_0000_0001u64, 3u32, 5u32, 0x4000_0000u64);
        let draws = random_bits(&[10], seed, base, stream).unwrap();
        let mask = bernoulli_mask(&[10], seed, base, stream, thr).unwrap();
        for i in 0..10 {
            let want = i128::from((draws.as_slice()[i] as u64) < thr);
            assert_eq!(mask.as_slice()[i], want, "elem {i}");
        }
    }

    // ---- uniform_f32 decomposition (§8.1) -----------------------------------

    #[test]
    fn uniform_word_known_points() {
        // The §8.1 splice on pinned inputs (exact, hand-derivable):
        assert_eq!(uniform_word(0), 0.0); // from_bits(0x3F80_0000) - 1.0 = 1.0 - 1.0
        assert_eq!(uniform_word(1 << 9), 2f32.powi(-23)); // smallest positive draw
                                                          // largest value, just below 1.0 — 1.0 is NOT attainable:
        assert_eq!(uniform_word(u32::MAX), 1.0 - 2f32.powi(-23));
        assert!(uniform_word(u32::MAX) < 1.0);
    }

    #[test]
    fn uniform_f32_maps_draws_and_stays_half_open() {
        let (seed, base, stream) = (0x0123_4567_89ab_cdefu64, 3u32, 5u32);
        let draws = random_bits(&[64], seed, base, stream).unwrap();
        let u = uniform_f32(&[64], seed, base, stream).unwrap();
        for i in 0..64 {
            assert_eq!(
                u.as_slice()[i],
                uniform_word(draws.as_slice()[i] as u32),
                "elem {i}"
            );
            assert!((0.0..1.0).contains(&u.as_slice()[i]), "elem {i} in [0,1)");
        }
    }

    // ---- normal_f32 decomposition (§8.2, basic Box-Muller) ------------------
    // Structural assertions only — normal_f32 is deliberately tolerance-free until
    // sabotage-calibration with a second backend, so these test the properties the
    // design rests on (needing no oracle), never an absolute normal value.

    #[test]
    fn box_muller_reflection_removes_ln_zero_singularity() {
        // u1 = 0 is attainable (a draw < 512); the 1 - u1 reflection makes
        // r = sqrt(-2·ln 1) = 0 — finite — instead of ln(0) = -inf -> NaN. A missing
        // reflection turns this exact case into NaN.
        for &u2 in &[0.0f32, 0.25, 0.5, 0.999] {
            assert_eq!(box_muller_pair(0.0, u2), (0.0, 0.0), "u2={u2}");
        }
        // Every output over a sweep of (u1, u2) is finite.
        for iu in 0..64u32 {
            let u1 = uniform_word(iu << 9);
            for iu2 in 0..8u32 {
                let u2 = uniform_word(iu2 << 26);
                let (z0, z1) = box_muller_pair(u1, u2);
                assert!(z0.is_finite() && z1.is_finite());
            }
        }
    }

    #[test]
    fn normal_f32_pairing_is_structurally_correct() {
        // The Pythagorean structural identity: element 2p (even, r·cosθ) and 2p+1
        // (odd, r·sinθ) share one (r, θ), so z_even² + z_odd² == r² == -2·ln(1 - u1)
        // regardless of the angle. Catches a cos/sin swap, a wrong-pair mapping, and a
        // mis-derived radius while asserting NO absolute value — a relation among
        // outputs, so it needs no ULP budget.
        let (seed, base, stream) = (0xfeed_face_dead_beefu64, 11u32, 2u32);
        let n = normal_f32(&[16], seed, base, stream).unwrap();
        let key = [seed as u32, (seed >> 32) as u32];
        let word = |k: u64| philox4x32_10(derive_counter(k, base, stream), key)[(k % 4) as usize];
        for p in 0..8u64 {
            let u1 = uniform_word(word(2 * p));
            let r2 = -2.0 * libm::logf(1.0 - u1);
            let z0 = n.as_slice()[(2 * p) as usize];
            let z1 = n.as_slice()[(2 * p + 1) as usize];
            // f32 square-sum rounding only (cos²+sin² ≈ 1 within a few ULP) — a
            // reference self-consistency check, NOT a cross-backend tolerance.
            assert!(
                (z0 * z0 + z1 * z1 - r2).abs() <= 1e-4 * (1.0 + r2),
                "pair {p}: z0²+z1²={} vs r²={r2}",
                z0 * z0 + z1 * z1
            );
        }
        assert!(n.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn normal_f32_is_position_pure_and_odd_safe() {
        // Element i is a pure function of i: the same (seed, base, stream) over a
        // LARGER shape reproduces the smaller shape's elements exactly (a prefix). An
        // odd count stays finite — the last even element reaches its pair partner at
        // logical index `count`, computed on demand.
        let (seed, base, stream) = (0x0102_0304_0506_0708u64, 1u32, 9u32);
        let small = normal_f32(&[5], seed, base, stream).unwrap(); // odd count
        let big = normal_f32(&[9], seed, base, stream).unwrap();
        for i in 0..5 {
            assert_eq!(
                small.as_slice()[i],
                big.as_slice()[i],
                "elem {i} not position-pure"
            );
        }
        assert!(
            small.as_slice().iter().all(|v| v.is_finite()),
            "odd count all finite"
        );
    }
}
