# kiss-ref

A project-agnostic, spec-exact **reference implementation** and **correctness oracle** for the
[KISS](https://github.com/ThinkersJournal/KISS) base-op vocabulary (KISS-Ops + KISS-Classify).

Deliberately naive, obvious, and slow — correctness is the only goal. It is the thing every KISS
consumer (Fuel, Baracuda, Unpopped, and KISS-Conform itself) tests against, and, because a total
reference cover always runs, it doubles as an "it always works" correctness floor.

See [DESIGN.md](DESIGN.md) for the architecture, the KISS binding, the provenance rule, and scope.

## Layout

| Crate | Binds / provides | Depends on |
|---|---|---|
| `kiss-classify-vocab` | KISS-Classify §6.1 — the 20 pinned dtypes | nothing |
| `kiss-ops-vocab` | KISS-Ops §6.1/6.3/6.13/6.18 — op set + reference decompositions | nothing |
| `kiss-ref-core` | the floor atoms + §6.13/6.14 recursive resolver | both vocab crates, `libm`, `half` |
| `kiss-ref-conformance` | coverage gate + `test_ops_*` conformance corpus | all of the above |

## Status

Seed (first cut). Vocab is complete; kernels cover the mandatory core over the common dtypes. The
coverage gate reports what is DONE vs PENDING. Pre-1.0, following unfrozen KISS drafts.

```
cargo test            # runs the conformance corpus + coverage-ledger report
```
