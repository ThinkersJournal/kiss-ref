# Changelog

All notable changes to the published crates (`kiss-classify-vocab`, `kiss-ops-vocab`,
`kiss-ref-core`) are recorded here. The three versions move in lockstep; the
conformance crate is unpublished. Format follows [Keep a Changelog]; versions are
[SemVer].

## [Unreleased]

> The workspace version is already **0.4.0** (allocated by the portfolio PM; `cargo package --workspace` cannot
> verify a vocab change against the published 0.3.4) but nothing is published. The next release is a **breaking** one: it rebinds the
> crates from the KISS spec commit `19c3ad7` to current KISS `main` in slices. This section lists
> what is on `main` and unreleased; entries are added slice by slice, not at release time.

### Changed — behaviour that can turn a passing comparison into a failing one

- **`diff`: a computed NaN's QUIETNESS is now compared at `Tolerance::Exact`** (#39, KISS #352,
  KISS-CONFORM-6.8-0010). Before, the computed-NaN arm was `distance == 0`, and the ULP distance
  between ANY two NaNs is `0`, so a candidate that returned a *signaling* NaN where a quiet one
  is required was silently accepted. Now such a candidate is rejected. Payload and sign remain
  uncompared; `f8e4m3fn` (single NaN, no quiet bit) is vacuous. **A consumer whose differential
  passed on 0.3.x with a signaling NaN in a computed result will fail on this release.** This is
  the main reason the next release is not a patch.
- **`eval_op`: the computing atoms deliver a QUIET NaN for a signaling-NaN operand**
  (KISS-OPS-6.16-0010). `libm` returns a NaN argument unchanged, so `exp`/`log` of a signaling
  `f32`/`f64` NaN used to return that signaling NaN; they now return the quiet one (found by
  running the KISS `ops-transcendental-nan` vectors, 3 of 12 failed). The move ops (`neg`, `abs`,
  `copysign`, `select`, minmax) are unchanged, and so are the rounding atoms, whose NaN KISS has not
  pinned (KISS #396).

- **`f8e6m2` is now RESERVED** (KISS #517, KISS-CLASSIFY-6.1-0013): `Dtype::is_reserved()` is true for it
  and `Dtype::is_mx_scale()` is no longer. Until now kiss-ref described it as an MX scale; KISS made it
  a reserved spelling (recognized on parse, no pinned encoding, no compute). Behaviour for callers that
  only use `declines_compute()` is unchanged; callers that branch on `is_mx_scale()` for `f8e6m2` change.

- **`reduce_var` / `reduce_std` now use the deviations-about-the-mean form** (KISS #516, §6.13-0004).
  They used to transcribe `E[x²] − E[x]²`, which the standard now FORBIDS: it cancels catastrophically
  (2.0 against a true 1.25 at a 1e8 offset) and can go negative, making `reduce_std` NaN. The result is now
  `mean(sqr(x − mean(x)))`, exact at that offset and `>= +0`. **Numbers change for any consumer that
  compared against the old one-pass values**; `layer_norm` inherits the change. The previous behaviour was
  pinned by a test as "the honest limit of the §6.13 form"; that test is reversed.
- **A comparison's result dtype is `bool`** (KISS #516, KISS-OPS-6.2-0005): one byte per element, not the
  compute dtype. `eval_op` and the recipe evaluator still carry the mask as a compute-lane value holding exactly
  `1`/`0` (so it can compose into `select`/`mul`); what changed is the documented observable dtype, exposed by
  the new `result_dtype`, `FlatDag::output_dtypes` and `mask_bytes`. A consumer comparing a kernel's one-byte
  mask against kiss-ref's output should compare `mask_bytes`, not a `T`-width value.

### Added

- `tensor_ops::reduce_var_bessel`: `reduce_var` with the `bessel_correction` attribute (§6.19-0030); divisor
  `reduced_count − 1`, plain IEEE division (a count of 1 yields NaN).
- `result_dtype`, `FlatDag::output_dtypes`, `mask_bytes` (see above).
- `kiss-ref-conformance`: `pow` checked rule by rule against the IEEE 754 9.2.1 table of KISS-OPS-6.13-0005
  on four lanes (`tests/pow_full_domain.rs`). `libm::pow` already conformed; no code change was needed.
- `Op::Argmin` (KISS #516): `argmin` joins the closed op set (122 ops, was 121), with `tensor_ops::argmin`
  and the integer-lane `tensor_int::argmin`. NaN orders greatest, so `argmin` skips a NaN unless every
  element is NaN (then index 0), whereas `argmax` returns the first NaN; ties go to the lower original
  index. This differs from NumPy/PyTorch, which return the NaN index for both.
- `kiss-ref-conformance`: `kiss-classify-vocab`'s dtype set and `kiss-ops-vocab`'s op set are now checked
  against KISS's `dtype_manifest.json` / `op_manifest.json` in both directions
  (`tests/dtype_manifest_agreement.rs`, `tests/op_manifest_agreement.rs`).
- `kiss-ops-vocab::precision` (KISS #518, KISS-OPS-6.8-0007..0009): the closed five-token `PrecisionClass`
  set (`strict`, `correctly-rounded`, `bounded-ulp`, `bounded-tolerance`, `unbounded`), `TierKind`,
  `BitStability`, and `derive_precision_class` (all 12 cells of the derivation table). Vocabulary and rule
  only; kiss-ref reads and writes no KISS-Contract document.
- `ScalarFloat::quiet_nan`.
- `kiss-ref-conformance`: the KISS oracle-vector corpus (six files, 159 vectors, KISS `904a4b4`,
  vendored under `fixtures/kiss-corpus/`) is now run against kiss-ref (`tests/kiss_corpus_vectors.rs`).

### Changed

- **MIT copyright holder is now `Thinker's Journal`** (was `Eric Evans`), in the root
  `LICENSE-MIT` and the three published crate copies. Ruled by the copyright owner:
  Thinker's Journal is the non-profit that should hold the rights to code written for
  it, and it is expected to gain members beyond its current sole member.

  - **Not a licence change.** The licence remains `MIT OR Apache-2.0` — a dual licence
    under which a consumer may take *either*. No manifest, no SPDX header, and no
    `LICENSE-APACHE` file changes; only the MIT holder line moves.
  - **No consumer action.** Both attributions were authorized by the same owner, so
    the already-published 0.3.4 (which carries `Eric Evans`) is valid as shipped. This
    is a go-forward convention, not a defect fix, and rides the next release rather
    than triggering one.

## [0.3.4] — 2026-09-05

### Fixed

- **Narrow-float `neg` / `abs` / `copysign` now preserve a signaling NaN's payload
  bit-for-bit** (`kiss-ref-core`). These three raw-bit sign operations were computed
  by promoting to `f32` and back; for `bf16` / `f16` / `f8e5m2` that promotion runs
  through `half`'s widening conversion, which **quiets a signaling NaN** — so a
  signaling NaN's payload was silently altered.

  - **Affected (incorrect before 0.3.4):** `neg`, `abs`, `copysign` on `bf16`,
    `f16`, and `f8e5m2` when the operand is a **signaling NaN**. Example: `bf16`
    `neg(0x7F81)` returned `0xFFC1` (quieted; payload `0x01` → `0x41`) instead of
    `0xFF81` (sign bit flipped, payload preserved).
  - **Not affected:** the same ops on `f32` / `f64` (already raw-bit); the same ops
    on any non-NaN or quiet-NaN operand; `f8e4m3fn` (single NaN encoding — no
    signaling/quiet distinction to lose); every other operation.
  - **Spec basis:** KISS-OPS-6.4-0003 (`neg`), -6.4-0004 (`abs`), -6.9-0002
    (`copysign`) pin these as raw-bit sign transforms that preserve a NaN operand's
    payload; for the narrow dtypes KISS-OPS-6.2-0001 routes to §6.16, where
    KISS-OPS-6.16-0009 names a promote-and-round-back implementation
    non-conforming. (The §6.4-vs-§6.16 scope citation is tracked as KISS #399; the
    obligation holds under either reading.)

`kiss-classify-vocab` and `kiss-ops-vocab` are unchanged from 0.3.3 apart from the
lockstep version bump.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[SemVer]: https://semver.org/spec/v2.0.0.html
[0.3.4]: https://github.com/ThinkersJournal/kiss-ref/releases/tag/v0.3.4
