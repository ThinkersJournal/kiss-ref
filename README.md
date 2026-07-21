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

Seed. Vocab complete (106 ops, 20 dtypes). Kernels: **78/106 ops evaluable** — the float floor +
elementwise non-primitives over `f16`/`bf16`/`f32`/`f64` (minus `nextafter` on the narrow floats,
§6.9-0003) and the integer floor atoms over all integer dtypes, incl. the packed `s4`/`u4`/`b1`. All
11 §6.13-0003 refine-marked ops (incl. `pow`/`hypot` full-domain edges) are direct refined kernels.
PENDING: the structural atoms + the reductions/scans/norms/matmul/pooling that decompose through them
(a tensor-evaluation layer — see DESIGN.md), and the FP8 (`e4m3`/`e5m2`) / `bool` / complex
(`c32`/`c64`) dtype breadth. Reviewed by the KISS and Baracuda agents (2026-07-17). Pre-1.0,
following unfrozen KISS drafts.

```
cargo test            # runs the conformance corpus + coverage-ledger report
```
