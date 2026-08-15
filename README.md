# kiss-ref

A project-agnostic, spec-exact **reference implementation** of the
[KISS](https://github.com/ThinkersJournal/KISS) base-op vocabulary (KISS-Ops + KISS-Classify) — a
differential *target* other implementations test against, **not** the KISS-Conform oracle (that
oracle is independent by mandate and mints the corpus; kiss-ref is *tested against* it — Conform
§6.5-0002/0003).

Deliberately naive, obvious, and slow — correctness is the only goal. The target is a *total cover*
of the op basis (so it doubles as an "it always works" correctness floor), but read the coverage
ledger, not the tagline: every op is evaluable on at least one lane, while some individual
`(op × dtype)` cells are still `Pending` (see Status).
It exposes a **differential export seam** (`ulp_distance`, `reference_*`, `diff_*`) so KISS-Conform
and Baracuda's on-device harness can dev-depend on it and cross-run.

**No wire codec.** kiss-ref implements the *vocabulary and semantics*, not the KISS wire format:
there is no `SCHEMA_VERSION` and no `structure_key` encoder here. Its agreement with KISS is checked
at the **vocabulary level** — the dtype and op **token spellings** — not the wire `structure_key`
(which lives in KISS). "Reference implementation" is not "wire-format implementation."

See [DESIGN.md](DESIGN.md) for the architecture, the KISS binding, the provenance rule, and scope.

## Layout

| Crate | Binds / provides | Depends on |
|---|---|---|
| `kiss-classify-vocab` | KISS-Classify §6.1 — the 24 pinned dtypes | nothing |
| `kiss-ops-vocab` | KISS-Ops §6.1/6.3/6.13/6.18 — op set + reference decompositions | nothing |
| `kiss-ref-core` | the floor atoms + §6.13/6.14 recursive resolver | both vocab crates, `libm`, `half` |
| `kiss-ref-conformance` | coverage gate + `test_ops_*` conformance corpus | all of the above |

## Status

Vocab complete (121 ops, 24 dtypes). **All 121 ops evaluable** on at least one lane. The scalar
floor + non-primitives over `f16`/`bf16`/`f32`/`f64` and every integer dtype (incl. packed
`i4`/`u4`/`b1`); a full **tensor layer** (the 6 §6.11 structural atoms + all §6.13 tensor
non-primitives incl. the window family, on the float lane, plus an integer tensor lane); **FP8**
(`f8e4m3fn`/`f8e5m2`, hand-rolled f32 codec); the truth-valued **bool** lane; and the **complex**
lane (`c64` = pair-of-`f32`, `c128` = pair-of-`f64`), whose §6.18 arithmetic family is **Done** —
only the real-only ops are `NotApplicable` on complex dtypes.

Two dtype groups are **recognized but decline to compute**, by design: they parse (distinct from an
unknown token) yet every op reports `NotApplicable` — the reserved `f8e4m3fnuz`/`f8e5m2fnuz` variants
and the MX scale types `f8e8m0`/`f8e6m2`. That is a typed compute-decline, not backlog.

Cell coverage is three-state (Done / Pending / NotApplicable) driven by a spec-derived
`legality(op, dtype)`; only legal cells count. `nextafter` is `NotApplicable` on the narrow/FP8
floats (§6.9-0003). Adversarially reviewed via multi-agent workflows. Pre-1.0: tracks KISS (itself
pre-freeze), but **binds each release to a specific frozen spec commit**, never a live draft.

```
cargo test            # runs the conformance corpus + coverage-ledger report
```
