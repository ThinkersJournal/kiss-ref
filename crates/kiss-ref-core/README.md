# kiss-ref-core

The [KISS](https://github.com/ThinkersJournal/KISS) base-op **reference kernels**: the
primitive floor atoms plus the §6.13/§6.14 recursive-resolution engine that decomposes every
non-primitive op down to that floor. Deliberately naive, obvious, and slow — spec-exactness is
the only goal.

This is the core of [`kiss-ref`](https://github.com/ThinkersJournal/kiss-ref), a
project-agnostic reference implementation that other implementations test **against** — a
differential *target*, **not** the KISS-Conform oracle (that oracle is independent by mandate).
It exposes a differential export seam (`ulp_distance`, `reference_*`, `diff_*`) so a conformance
harness or an on-device kernel generator can dev-depend on it and cross-run.

- **Coverage:** all 121 ops evaluable over `f16`/`bf16`/`f32`/`f64`, the integer dtypes (incl.
  packed `i4`/`u4`/`b1`), `f8e4m3fn`/`f8e5m2` FP8, the bool lane, and the complex lane
  (`c64`/`c128`, the §6.18 arithmetic family) — only real-only ops are `NotApplicable` on complex
  dtypes. The reserved `f8e4m3fnuz`/`f8e5m2fnuz` and MX-scale `f8e8m0`/`f8e6m2` dtypes parse but
  decline to compute (`NotApplicable` by design). Cell coverage is three-state
  (Done / Pending / NotApplicable), driven by a spec-derived `legality(op, dtype)`.
- **`no_std`** (default features off); depends only on `libm` (pure-Rust transcendentals) and
  `half` (f16/bf16 storage), plus the two KISS vocabulary crates.

`Error` is `#[non_exhaustive]`. Pre-1.0, tracking KISS — each release bound to a frozen spec commit,
not a live draft.

See the [repository](https://github.com/ThinkersJournal/kiss-ref) for `DESIGN.md` (architecture,
KISS binding, provenance rule, scope) and the `kiss-ref-conformance` corpus.

## License

Licensed under either of Apache License 2.0 or MIT license at your option.
