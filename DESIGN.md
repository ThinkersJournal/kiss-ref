# kiss-ref — a reference implementation of the KISS base ops

**Status:** Pre-1.0, published on crates.io (0.3.x). Tracks KISS (itself pre-freeze); each release binds a specific frozen spec commit, not a live draft (see [Spec pin](#spec-pin)). First cut 2026-07-16.
**License:** spec-conformance code MIT-OR-Apache-2.0; it implements the CC0 KISS standard.

`kiss-ref` is a **project-agnostic, spec-exact reference implementation** of the KISS base-op
vocabulary — a differential *target* every KISS consumer can test against. It is deliberately
naive, obvious, and slow: correctness is the only goal. It makes no strategic decisions, has no
optimizer, and knows nothing about any one consumer's IR.

It is **not** the KISS-Conform oracle. That oracle (`../KISS/conformance/src/semantics.rs`) is a
**wide-precision authoring instrument** — independent by mandate (it shares no lowering code with any
reference implementation and evaluates wider than the compute dtype, rounding once) and it *mints*
the conformance corpus (Conform §6.5-0002/0003/0007); a vector derived from kiss-ref's run would be
rejected as circular. kiss-ref is a **corpus-conformant proxy**: *tested against* that corpus, and
one dissimilar implementation at the umbrella §5.3 freeze gate (interop / wire / foreign-reader),
never a corpus minter. Note that kiss-ref and that oracle both resolve non-primitives from the
*same* §6.13 decomposition table, so they are code-disjoint but **comprehension-correlated** on the
non-primitives — kiss-ref is a strong independent check on the *floor*, a weak one on the
decompositions. (General rule: any differential check whose reference is a shared decomposition
table is not decomposition-independent.) The full authority model + independence reconciliation live
in [`../KISS/docs/conformance-architecture.md`](../KISS/docs/conformance-architecture.md); this
DESIGN owns the kiss-ref-specific slice.

It exists to serve four consumers at once, none privileged:

- **KISS** (github.com/ThinkersJournal/KISS) — a spec-exact executable reference for KISS-Ops +
  KISS-Classify that KISS-Conform can differential against (distinct from its independent §6.5
  oracle, which mints the corpus).
- **Fuel** — as a *verify* reference (contract verification) **and** as a total-cover *execution
  route*: an "it always works" correctness-floor backend (see "Execution route" below).
- **Baracuda** — as a reference to test its kernels against and to reconcile its existing
  `baracuda-kernel-vocab` / `baracuda-kernelgen` seeds with.
- **Unpopped** (future) — same reference role from day one.

## The core idea: the total cover of the base map

Two totality properties come from KISS-Ops itself:

- **A closed, enumerated op basis** — the mandatory **primitive floor** (KISS-Ops §6.3), plus the
  non-primitive ops each with a normative **reference decomposition** to the floor (§6.13), plus
  the complex family (§6.18). The op set is closed (§6.1-0001).
- **Guaranteed termination** — every op has an integer level, the decomposition graph is acyclic,
  and recursive resolution terminates at the floor in finitely many steps (§6.14). This is the
  standard's own formalization of "lower everything to a primitive base map."

A reference implementation that (a) implements the **floor spec-exactly** and (b) can **resolve any
non-primitive by expanding its §6.13 decomposition to the floor** is therefore a *total cover* of the
op basis: every op is evaluable, so the same artifact can be a differential reference (verify
against it) and a correctness floor (execute on it when nothing else will).

**Total cover now spans all 121 ops — the scalar floor + a full tensor-evaluation layer + the §6.18 complex family.** The scalar
path covers the elementwise float floor atoms + the non-primitives that decompose through them, and
the integer floor atoms. The six **structural atoms** (`element_map`/`reduce`/`prefix_scan`/`gather`/
`scatter`/`sort_network`, §6.11) are hand-written strided-tensor kernels, and `matmul`/`softmax`/the
reductions/scans/norms/pooling decompose *through* them via a **tensor-evaluation layer that nests the
scalar resolver inside a row-major odometer** — `element_map`'s per-element body *is* the unchanged
`eval_expr`, so pointwise numerics (NaN, signed-zero, refined forms) are inherited, not re-implemented,
and a `reduce`/scan monoid combine is one `eval_op` call. The one order-sensitive op, float
`scatter_add`, is pinned to row-major source order and tagged order-invariant/nondeterministic
(§6.0-0004) — compared under tolerance, never byte-exact. The **window family** (`avg_pool`/`max_pool`/
`im2col`) is included, so **every op is evaluable on at least the float lane (121/121)**; the
**integer tensor lane** additionally covers the atoms + `argmax`/`any`/`all`/`cum*` over the integer
dtypes (via `eval_int_op` wrapping). FP8 (`f8e4m3fn`/`f8e5m2`), the `bool` lane, and the §6.18 complex
family (`c64`/`c128`) are now all `Done`; the remaining `Pending` cells are per-(op × dtype): the
float-only tensor ops (the reductions/normalizations needing `div`/`sqrt`) on the integer lane. Three §6.11
under-specifications surfaced (gather skip-read value; scatter base-state / output-shape; empty-axis
for `prefix_scan`/`gather`/`scatter`/`sort_network`); kiss-ref pinned each by local convention, filed
them as a KISS RFC (PR #75), and **all three were ruled kiss-ref's way 2026-07-23** (gather-skip =
option-1 base operand with a dynamic requirement; scatter = explicit dest, output shape = dest.shape;
empty-axis = the shape-implied behaviors) — plus the companion scatter broadcast-updates (general
§6.11-0001 broadcast). The provisional flags are lifted; this was the intended §6.13-divergence
signal working end-to-end. Even so,
"total cover" and "1:1 mirror of the spec" describe the *target*; the coverage ledger
(`kiss-ref-conformance`) reports exactly which (op × dtype) cells are actually `Done`, and the prose
should always be read against it. Baracuda's `oracle.rs` covers this structural/tensor region
independently — but kiss-ref builds its **own** structural layer from the §6.13 table and does **not**
converge it against `oracle.rs`. Where the two readings *differ*, that is a spec-ambiguity KISS issue
to file (e.g. the u32-gather `same_as` bug, `f7578df1`), not an implementation to reconcile:
reconciling against `oracle.rs` would only deepen the shared-§6.13 comprehension-correlation noted
above. Two honest readings diverging is the signal.

## Architecture

```
kiss-classify-vocab   binds KISS-Classify §6.1 — the 24 pinned dtypes + bit layouts + numeric kinds.
                      Depends on NOTHING (foundational root; sibling of ops-vocab, never imports it).

kiss-ops-vocab        binds KISS-Ops §6.1/§6.3/§6.13/§6.18 — op tokens, family tags, primitive-floor
                      flags, integer levels, the §6.13 reference-decomposition table (as data), and
                      the §6.8 transcendental ULP ceilings. Depends on NOTHING (foundational root;
                      sibling of classify-vocab, never imports it — mirrors KISS-Ops §6.9 / DAG).

kiss-ref-core         the actual reference kernels. Implements the floor atoms spec-exactly + the
                      §6.13/§6.14 recursive-resolution engine (non-primitives evaluate for free) +
                      the tensor layer (`tensor`/`bridge`/`kernels`/`attrs`/`tensor_ops`): the 6 §6.11
                      structural atoms as strided-tensor kernels + the §6.13 tensor non-primitives.
                      Depends on BOTH vocab crates + libm + half. NO consumer (Fuel/Baracuda) deps.

kiss-ref-conformance  the build-time (atom × legal-dtype) coverage gate that makes "always works" an
                      enforced invariant, plus the conformance corpus mirroring KISS-Conform's
                      `test_ops_*` names (KISS-Ops §9.1 clause→test traceability).
```

The two-vocab split is not stylistic: KISS-Ops and KISS-Classify are **independent foundational
roots** in the KISS DAG, and each normatively states it does not import the other. A faithful Rust
binding preserves that — a consumer needing only dtypes never pulls in the op vocabulary.

### Conformance host and test topology

`kiss-ref-conformance` is the **deliberate conformance host** — not a crate whose tests
happen to live away from the code they exercise. Its public surface is the coverage
**`Ledger`** (`ledger()` → the machine-readable **per-op** `Done`/`Pending` split plus
provisional pins that evaluating teams read): that API *is* the crate's product. (The
per-`(op × dtype)` three-state — `Done`/`Pending`/`NotApplicable` — is the spec-derived
`support(op, dtype)` function in `kiss-ref-core`, which the coverage tests enforce cell by
cell; the `Ledger` reports the per-op roll-up.) It is therefore the correct home for the
cross-crate conformance layer — the corpus mirroring KISS-Conform's `test_ops_*` names, the
self-differential (hand-written kernel vs its §6.13 decomposition), the metamorphic relations,
the fuzz suites, and the coverage gate.

**The guarantee is the workspace run.** `cargo test` at the root (equivalently
`cargo test --workspace`) exercises every layer — the vocab crates, `kiss-ref-core`'s own
tests, and the full conformance suite. That is the command the README names and CI runs, and it
is what validates spec-exactness end to end.

**Residual, named rather than left implicit:** `cargo test -p kiss-ref-core` runs core's
**colocated `src/` unit tests** — a real local guard on the kernels (150 tests across 15 of the
crate's 18 `src/` files) — but **not** the conformance corpus or the decomposition-differential, which
also guard core's spec-exactness *from the conformance crate*. So a contributor iterating on
`kiss-ref-core` alone gets a genuine but partial signal; run the workspace suite for the full
guarantee. This is deliberate layering, **not** the "core properties guarded only elsewhere"
defect — core carries substantial local tests and the conformance corpus sits on top of them,
never as a substitute.

### Vocabulary is *bound*, not invented

`kiss-*-vocab` does **not** define the op basis or dtype set — the **KISS standards own them**
(KISS-Ops §6.1/§6.3/§6.13, KISS-Classify §6.1). These crates are the Rust *embodiment* of the spec,
with the source of truth at `../KISS/spec/{ops,classify}.md`. Where the §6.13 decompositions are used,
they follow the spec's pinned grammar (§6.13-0006), so they can be regenerated from the spec rather
than hand-maintained. When a base op or dtype is missing from the standard, that is an RFC back to
KISS — not something this project invents locally.

### Spec pin

kiss-ref binds a **specific frozen KISS spec commit** per release, never the moving `main`. **The
0.3.x line binds `19c3ad7`** — the sk4-frozen §6.1 dtype set (the closed 24-token vocabulary) and the
KISS-Ops op vocabulary at `../KISS/spec/{ops,classify}.md`. A spec change is adopted by **re-binding to
a new commit in a new release**, not by tracking drafts — a differential result is only meaningful
against the commit it was measured on, so each release cites its pin. (0.3.1/0.3.2 are docs/metadata
only and bind the same `19c3ad7` as 0.3.0.)

### Consumers adapt to the core (core/adapter split)

`kiss-ref-core` speaks only KISS vocabulary and has zero consumer dependencies. Each consumer wraps it
with a thin adapter *in its own repo* — the same pattern Fuel already uses to wrap `baracuda` (CUDA)
and `vulkane` (Vulkan) behind a backend. Fuel's `fuel-kiss-ref-backend` adapter is written at
integration time and is **not** part of this project; nothing here depends on Fuel.

## Provenance rule: reuse Fuel iff spec-exact

Every kernel carries a `Provenance` tag:

- **`PortedFromFuel`** — Fuel already had a kernel that meets the KISS §6 clause for this op under its
  determinism class (bitwise for exact ops per §6.2; within the declared-ULP ceiling for
  transcendentals per §6.8), so it was ported. The conformance test proves the tag.
- **`Fresh`** — Fuel had no spec-exact kernel (a coverage/dtype gap, or a flagged divergence), so a
  clean, obvious implementation was written from the spec.

In practice reuse concentrates on the **scalar** atoms (Fuel's already-Fuel-free `erf`, plus
`add`/`mul`/`sqrt`/…); the **structural** atoms (`element_map`, `reduce`, `prefix_scan`,
`sort_network`) have no direct Fuel equivalent and are `Fresh`; Fuel's *composite* kernels
(`gelu`, `softmax`) are not ported as reference — the resolver produces them from §6.13, and Fuel's
versions can later serve only as a refined fast lane.

## Determinism & "spec-exact"

- **Exact ops** (integer arithmetic, comparisons, `select`, rounding, shape/cast) — **bitwise** to the
  KISS §6 pin, including the load-bearing edges: default NaN propagation (§6.2-0003), signed-zero
  preservation (§6.2-0004), raw-bit `select` (§6.5), comparison-NaN → false / `cmp_ne` → true (§6.6).
- **Transcendental atoms** (§6.8) — **declared-ULP**, not bit-exact. The reference declares a per-atom
  ULP no looser than the §6.8 ceiling; conformance checks under that ULP. KISS-Ops forbids mandating a
  specific polynomial, so the reference is free to use `libm`.
- **Non-primitive ops** — their pinned semantics ARE their §6.13 decomposition evaluated on the floor
  (§6.2-0009); the reference evaluates exactly that (with the §6.13-0003 overflow-safe refinements
  where the spec marks them MUST, e.g. `tanh`/`softplus`).

**Compiler-independence is observed, not just constructed.** The exact ops are bit-determined by IEEE
754 and the transcendentals are pinned to the `libm` *crate* (a versioned dependency, not the
compiler), so the reference's numbers should not depend on the Rust compiler. This is *measured*, not
merely intended: the full deterministic differential suite is **bit-identical across rustc 1.97.1 and
1.98.0** (22 binaries, 371 tests, byte-identical per-binary results, 2026-08-21) — the exact-byte cells
match the same goldens under both compilers, the ULP cells stay within bound under both. The toolchain
is pinned (`rust-toolchain.toml`) so every run uses a named compiler rather than a moving `stable`
alias, and a version bump re-runs this check. (Residual the check does *not* cover: cross-**target**
excess precision, e.g. 32-bit x87, is a target property rather than a compiler-version one, and is out
of scope for a two-toolchain run on a single SSE2 host.)

## Execution route (Fuel's "it always works" floor)

Because the reference is a total cover, Fuel can adopt it not only at the verify seam but as a
**correctness-floor execution backend**: a backend that covers every (floor op × legal dtype),
advertises honest (high) cost, declares contiguous-only layout caps, and is therefore picked by the
optimizer only when no faster kernel covers an op/dtype. This turns "the base map is always
executable" from an aspiration into a theorem (total `decompose` ∘ total reference cover). It is
consumed through Fuel's normal backend-contract seam via the thin adapter; it never touches Fuel's
optimizer/executor/IR. As an execution route it must honor Fuel's never-panic/`Result` discipline —
so `kiss-ref-core` returns typed errors, never panics.

## Scope

- **Vocab: complete.** Both vocab crates enumerate the *full* KISS op set + 24 dtypes, so the coverage
  ledger is a complete list even where a kernel is still pending.
- **Scalar kernels: the mandatory core across the common dtypes.** The floor atoms + resolver over the
  float dtypes (`f32`, `f64`, `f16`, `bf16`) and every legal integer dtype (incl. packed `i4`/`u4`/`b1`),
  with the elementwise non-primitives resolved end-to-end.
- **Tensor layer: the 6 structural atoms + all 22 tensor non-primitives on the float lane** (§6.11/
  §6.13). `element_map`/`reduce`/`prefix_scan`/`gather`/`scatter`/`sort_network` as strided-tensor
  kernels; the reductions, scans, normalizations (`softmax`/`log_softmax`/`rms_norm`/`layer_norm`/
  `logsumexp`), `matmul`, `argmax`, `any`/`all`, the gather/scatter family (`index_select`/`embedding`/
  `scatter_add`), and the window family (`avg_pool`/`max_pool`/`im2col`) as spec-faithful transcriptions
  of their §6.13 decompositions. With the §6.18 complex family, this **lands the ledger at 121/121.**
- **Integer tensor lane:** the 6 atoms + `argmax`/`any`/`all`/`cum*` over every integer dtype (via
  `eval_int_op` two's-complement wrapping). The float-only tensor ops (`reduce_mean`/`var`/`std`/
  `norm2`, the normalizations, `matmul`, pooling — anything needing `div`/`sqrt`/`exp`) stay float.
- **Dtype breadth:** **FP8** (`f8e4m3fn`/`f8e5m2`) as `u8` newtypes with a hand-rolled f32 codec (RNE +
  saturation), computed via the narrow-float promote-to-f32 lane; the truth-valued **bool** lane
  (§6.2-0006) over the integer engine, `{0,1}`-normalized; and the **complex** lane (`c64` = pair-`f32`,
  `c128` = pair-`f64`), whose §6.18 arithmetic family is **`Done`** — only a real-only op on a complex
  dtype (and any complex op on a real dtype) is `NotApplicable`.
- **Recognize-but-decline-compute dtypes:** the reserved `f8e4m3fnuz`/`f8e5m2fnuz` and the MX scales
  `f8e8m0`/`f8e6m2` parse (distinct from an unknown token) but report `NotApplicable` for every op **by
  design** — a typed compute-decline, not backlog.
- **Coverage is three-state** — `Done` / `Pending` / `NotApplicable` — driven by a spec-derived
  `legality(op, dtype)` (op family × numeric kind); only legal cells form the denominator, so
  permanently-illegal cells (bitwise × float, `div` × int, `nextafter` × narrow, real-only-op × complex)
  leave the backlog entirely — as do the float-only tensor ops on the integer lane
  (`reduce_mean`/norms/`avg_pool`): they decompose through float `div`/`sqrt`, and KISS-OPS-6.4-0002
  excludes integer division and remainder from the op set, so they are `NotApplicable` on integers, not
  backlog. The `Pending` set is empty — every legal cell is `Done`. (The three §6.11 spec-gap cells were
  held provisional until the KISS PR #75 rulings landed — 2026-07-23, all three kiss-ref's way — and are
  no longer flagged.)

The coverage gate reports DONE vs PENDING for every (atom × legal-dtype) cell, so what remains is
machine-visible to the evaluating teams. They fill cells; kiss-ref dictates *how*.

## Non-goals

No optimizer, no fusion, no scheduling, no device management, no performance. Not a framework. Not a
consumer's IR. **No wire codec** — kiss-ref binds the *vocabulary and semantics*, not the KISS wire
format: there is no `SCHEMA_VERSION` and no `structure_key` encoder here. Its agreement with KISS is
checked at the **vocabulary level** (dtype/op token spellings), never the wire `structure_key` (which
lives in KISS). It is the slow, correct thing everything else is measured against.
