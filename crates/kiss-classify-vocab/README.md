# kiss-classify-vocab

A Rust binding of the [KISS](https://github.com/ThinkersJournal/KISS) **KISS-Classify**
pinned scalar dtype set (the 20 dtypes of §6.1) — a *conformant* implementation with no
special privilege over the spec.

A foundational root of [`kiss-ref`](https://github.com/ThinkersJournal/kiss-ref): it binds
KISS-Classify only and imports **nothing** — in particular never `kiss-ops-vocab`, because
the computation and data vocabularies are independent sibling roots (KISS-Classify §6.9-0001).

`no_std`, zero dependencies. Pre-1.0, tracking the unfrozen KISS drafts.

See the [kiss-ref repository](https://github.com/ThinkersJournal/kiss-ref) for the full
reference implementation, its architecture (`DESIGN.md`), and the conformance corpus.

## License

Licensed under either of Apache License 2.0 or MIT license at your option.
