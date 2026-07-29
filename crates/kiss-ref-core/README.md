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

- **Coverage:** all 106 ops evaluable over `f16`/`bf16`/`f32`/`f64`, the integer dtypes (incl.
  packed `s4`/`u4`/`b1`), `e4m3`/`e5m2` FP8, and the bool lane; complex is `NotApplicable`
  (the deferred §6.18 family). Cell coverage is three-state (Done / Pending / NotApplicable),
  driven by a spec-derived `legality(op, dtype)`.
- **`no_std`** (default features off); depends only on `libm` (pure-Rust transcendentals) and
  `half` (f16/bf16 storage), plus the two KISS vocabulary crates.

`Error` is `#[non_exhaustive]`. Pre-1.0, tracking the unfrozen KISS drafts.

See the [repository](https://github.com/ThinkersJournal/kiss-ref) for `DESIGN.md` (architecture,
KISS binding, provenance rule, scope) and the `kiss-ref-conformance` corpus.

## License

Licensed under either of Apache License 2.0 or MIT license at your option.
