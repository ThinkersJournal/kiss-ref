# kiss-ops-vocab

A Rust binding of the [KISS](https://github.com/ThinkersJournal/KISS) **KISS-Ops** op
vocabulary — the primitive floor, the non-primitive ops, and their reference
decompositions (§6.1/6.3/6.13/6.18) — a *conformant* implementation with no special
privilege over the spec.

A foundational root of [`kiss-ref`](https://github.com/ThinkersJournal/kiss-ref): it binds
KISS-Ops only and imports **nothing** — in particular never `kiss-classify-vocab`, because
the computation and data vocabularies are independent sibling roots (KISS-Ops §6.9).

`no_std`, zero dependencies. Pre-1.0, tracking KISS — each release bound to a frozen spec commit,
not a live draft.

See the [kiss-ref repository](https://github.com/ThinkersJournal/kiss-ref) for the full
reference implementation, its architecture (`DESIGN.md`), and the conformance corpus.

## License

Licensed under either of Apache License 2.0 or MIT license at your option.
