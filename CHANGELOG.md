# Changelog

All notable changes to the published crates (`kiss-classify-vocab`, `kiss-ops-vocab`,
`kiss-ref-core`) are recorded here. The three versions move in lockstep; the
conformance crate is unpublished. Format follows [Keep a Changelog]; versions are
[SemVer].

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
