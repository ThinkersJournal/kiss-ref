# kiss-ref

A project-agnostic, spec-exact **reference implementation** of the
[KISS](https://github.com/ThinkersJournal/KISS) base-op vocabulary (KISS-Ops + KISS-Classify) — a
differential *target* other implementations test against, **not** the KISS-Conform oracle (that
oracle is independent by mandate and mints the corpus; kiss-ref is *tested against* it — Conform
§6.5-0002/0003).

Deliberately naive, obvious, and slow — correctness is the only goal. The target is a *total cover*
of the op basis (so it doubles as an "it always works" correctness floor), but read the coverage
ledger, not the tagline: today the cover holds over the **scalar-resolvable subset** (see Status).
It exposes a **differential export seam** (`ulp_distance`, `reference_*`, `diff_*`) so KISS-Conform
and Baracuda's on-device harness can dev-depend on it and cross-run.

See [DESIGN.md](DESIGN.md) for the architecture, the KISS binding, the provenance rule, and scope.

## Layout

| Crate | Binds / provides | Depends on |
|---|---|---|
| `kiss-classify-vocab` | KISS-Classify §6.1 — the 20 pinned dtypes | nothing |
| `kiss-ops-vocab` | KISS-Ops §6.1/6.3/6.13/6.18 — op set + reference decompositions | nothing |
| `kiss-ref-core` | the floor atoms + §6.13/6.14 recursive resolver | both vocab crates, `libm`, `half` |
| `kiss-ref-conformance` | coverage gate + `test_ops_*` conformance corpus | all of the above |

## Status

Vocab complete (106 ops, 20 dtypes). **All 106 ops evaluable.** The scalar floor + non-primitives over
`f16`/`bf16`/`f32`/`f64` and every integer dtype (incl. packed `s4`/`u4`/`b1`); a full **tensor layer**
(the 6 §6.11 structural atoms + all §6.13 tensor non-primitives incl. the window family, on the float
lane, plus an integer tensor lane); **FP8** (`e4m3`/`e5m2`, hand-rolled f32 codec); and the truth-valued
**bool** lane. Complex (`c32`/`c64`) is **NotApplicable** — its arithmetic is the deferred §6.18 op
family (§6.16-0007). Cell coverage is three-state (Done / Pending / NotApplicable) driven by a
spec-derived `legality(op, dtype)`; only legal cells count. `nextafter` is `NotApplicable` on the
narrow/FP8 floats (§6.9-0003). Adversarially reviewed via multi-agent workflows. Pre-1.0, following
unfrozen KISS drafts.

```
cargo test            # runs the conformance corpus + coverage-ledger report
```
