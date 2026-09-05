# kiss-ref sk4 Phase-1 Regen — Implementation Plan

> ## ✅ STATUS: COMPLETE AND SUPERSEDED — THIS IS AN ARCHIVED RECORD, NOT A WORK QUEUE. DO NOT EXECUTE.
>
> Every step below shipped in **0.3.0** (published to crates.io 2026-08-13), since superseded by
> 0.3.3 and **0.3.4**. All 18 boxes are ticked because the work is done, not as bookkeeping:
> verified 2026-09-05 against the tree, not from recall — the 10 named `classify_*` tests all
> exist and pass (`classify_dtype_set_has_exactly_twenty_four`,
> `classify_reserved_and_mx_recognized_but_decline_compute`, `classify_sk3_spellings_are_no_longer_dtypes`,
> …), `declines_compute`/`is_reserved`/`is_mx_scale` exist, `Error::ReservedOrScaleDtype` exists,
> `coverage_reserved_and_mx_dtypes_not_applicable` exists, and the workspace is at 0.3.4.
> The two "out of this plan's execution" items also happened: the Architect's spec-currency
> statement was received (anchor `19c3ad7`) and Eric authorised the 0.3.0 publish.
>
> ⚠️ The instruction line below is retained for provenance and is **inert**. A completed plan whose
> boxes stay unticked under an executable header reads as a full unstarted work queue to any agent
> or auditor that finds it — the defect this header now prevents.
>
> **[historical] For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebind kiss-ref's op+dtype vocabulary and semantic/coverage layer to the merged **sk4 Phase-0** KISS spec (`ThinkersJournal/KISS@f4cc3ad`, main tip `19c3ad7`), landing as the breaking **0.3.0**.

**Architecture:** kiss-ref is the *differential semantic* reference — it derives the op+dtype token vocabulary and evaluates op semantics; it has **no structure_key wire codec** (attrs.rs is a typed OpAttrs stand-in). Phase-1 respells the frozen §6.1 dtype set, flips complex naming to total-width, adds the 4 additive recognize-but-decline dtypes, remaps complex component lanes, and adds a visible Provisional cell state for the #133 local pin. The §6.7-0013 acc+mp / §6.19 OpAttr **wire** codecs and `SCHEMA_VERSION` live in KISS's own `structure_key.rs` reference, not here (pending Architect confirm).

**Tech Stack:** Rust workspace (4 crates: `kiss-classify-vocab`, `kiss-ops-vocab`, `kiss-ref-core`, `kiss-ref-conformance`); `half` for f16/bf16; `no_std` core (thumbv7em CI target).

## Global Constraints

- **Bind the artifact, not a summary.** Every spelling/rule is read from the merged spec snapshot in scratchpad (`kiss-sk4-classify.md` / `kiss-sk4-ops.md` at `19c3ad7`). Re-verify against `f4cc3ad` before declaring Phase-1 done.
- **Closed 24-token dtype set** (KISS-CLASSIFY-6.1-0001), exact spellings, no 25th, no omission.
- **Op tokens are UNCHANGED at sk4** — `kiss-ops-vocab` needs no token change (it does not reference dtypes; independent sibling root). Corpus stays 121 op tokens.
- **Breaking → workspace 0.2.4 → 0.3.0**, lockstep across all crates + inter-crate dep versions (or `cargo package --workspace` fails; see [[workspace-lockstep-version-bump]]).
- **Gates that must stay green:** `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --workspace`, `cargo build -p kiss-ref-core --target thumbv7em-none-eabihf` (no_std), `cargo package --workspace --allow-dirty`.
- **Publish is OUT of scope** — 0.3.0 crates.io publish is Eric's trigger, gated on (a) Phase-1 byte-match leg complete [verification precedes publication] and (b) an Architect spec-currency statement by commit.

---

## sk4 dtype delta (from merged §6.1, verified against `19c3ad7`)

Current kiss-classify-vocab = **20** tokens (sk3, and already missing the 2 sk3 fnuz variants). sk4 = closed **24**-set:

| sk4 token | kind | bits | change from current |
|---|---|---|---|
| `f16 bf16 f32 f64` | float | 16/16/32/64 | unchanged |
| `i8` | int | 8 | **rename** `S8`/`s8` |
| `i16` | int | 16 | **rename** `S16`/`s16` |
| `u8 u16 u32 u64` | uint | 8/16/32/64 | unchanged |
| `i32 i64` | int | 32/64 | unchanged |
| `bool` | bool | 8 | unchanged |
| `f8e4m3fn` | float | 8 | **rename** `E4m3`/`e4m3` (also fixes bare-`e4m3` latent non-conformance) |
| `f8e4m3fnuz` | float | 8 | **NEW** — reserved (recognize, typed-decline, no compute) |
| `f8e5m2` | float | 8 | **rename** `E5m2`/`e5m2` |
| `f8e5m2fnuz` | float | 8 | **NEW** — reserved |
| `f8e8m0` | float | 8 | **NEW** — MX shared-exponent scale (never element-value; decline compute) |
| `f8e6m2` | float | 8 | **NEW** — MX scale |
| `i4` | int | 4 | **rename** `S4`/`s4` |
| `u4 b1` | uint | 4/1 | unchanged |
| `c64` | complex | 64 | **flip** — was `C32`/`c32` (pair-f32); component → `f32` |
| `c128` | complex | 128 | **flip** — was `C64`/`c64` (pair-f64); component → `f64` |

**Complex flip ordering hazard:** old `C64`("c64", pair-f64) → new `C128`("c128"); old `C32`("c32", pair-f32) → new `C64`("c64"). Never rename `C32→C64` before `C64→C128` in a blind find/replace — do it with the tests updated first as the guardrail.

## File Structure

- `crates/kiss-classify-vocab/src/lib.rs` — the whole respell (enum, ALL, token, from_token, numeric_kind, bits, component_dtype, new `is_reserved`/`is_mx_scale`, docs, tests). **Largest change.**
- `crates/kiss-ops-vocab/**` — **no change** (op tokens stable).
- `crates/kiss-ref-core/src/*.rs` — Dtype-variant rename ripple (compiler-guided) across resolve, complex, fp8, bridge, scalar, tensor*, kernels, recipe*, boolean, diff, lib; plus reserved/MX compute-decline in resolve.rs + a typed `Error`; plus `Support::Provisional` in lib.rs.
- `crates/kiss-ref-conformance/src/lib.rs` + `tests/coverage.rs` — ledger provisional surface; coverage matrix grows to 24 dtypes (4 new = NotApplicable for compute); summary string respell.
- `Cargo.toml` (+ each crate manifest) — 0.2.4 → 0.3.0 lockstep.

---

### Task 1: kiss-classify-vocab dtype respell to the sk4 24-set

**Files:** Modify `crates/kiss-classify-vocab/src/lib.rs` (enum, impls, tests).

**Interfaces produced (later tasks + core rely on these exact names):**
- `Dtype::{I8,I16,I4}` (were S8/S16/S4), `Dtype::{F8e4m3fn,F8e4m3fnuz,F8e5m2,F8e5m2fnuz,F8e8m0,F8e6m2}` (E4m3/E5m2 renamed; fnuz+MX new), `Dtype::{C64,C128}` (were C32/C64).
- `Dtype::ALL: [Dtype; 24]`.
- `pub const fn is_reserved(self) -> bool` — true for `F8e4m3fnuz`/`F8e5m2fnuz`.
- `pub const fn is_mx_scale(self) -> bool` — true for `F8e8m0`/`F8e6m2`.
- `pub const fn declines_compute(self) -> bool` — `is_reserved() || is_mx_scale()` (recognized token with no element-value compute semantics at sk4).
- `component_dtype`: `C64→Some(F32)`, `C128→Some(F64)`.

- [x] **Step 1 — failing tests first.** Update/add in the `tests` module: `classify_dtype_set_has_exactly_twenty` → `...twenty_four` (assert `ALL.len()==24`); `classify_tokens_round_trip_and_are_unique` (24 distinct); rename spelling test to assert `I8.token()=="i8"`, `I16=="i16"`, `I4=="i4"`, `F8e4m3fn.token()=="f8e4m3fn"`, `F8e5m2.token()=="f8e5m2"`; `component_dtype`: `C64→F32`, `C128→F64`; bit widths: `C64==64`, `C128==128`, `F8e8m0==8`, `F8e6m2==8`; unknown-token set now excludes the sk4 tokens but still rejects `s8`,`e4m3`,`c32` (the *sk3* spellings are now unknown); NEW: `classify_reserved_and_mx_recognized_but_decline` (from_token Some for all 4, `declines_compute()` true, `is_reserved`/`is_mx_scale` partition them), `classify_fnuz_and_mx_tokens` (exact spellings). Run: `cargo test -p kiss-classify-vocab` → expect FAIL (variants/tokens don't exist yet).
- [x] **Step 2 — respell the enum + impls.** Rewrite `Dtype` (rename S8/S16/S4→I8/I16/I4, E4m3/E5m2→F8e4m3fn/F8e5m2, C32/C64→C64/C128; add F8e4m3fnuz, F8e5m2fnuz, F8e8m0, F8e6m2 in §6.1 table order). Update `ALL` (24, table order), `token()`, `numeric_kind()` (MX scales = Float; complex = C64/C128), `bits()`, `component_dtype()`, and add `is_reserved`/`is_mx_scale`/`declines_compute`. Update crate + variant docs (drop the "must ride sk4" caveat — it's now done; NumericKind doc examples to sk4 spellings; "exactly 20"→"exactly 24").
- [x] **Step 3 — green.** Run: `cargo test -p kiss-classify-vocab` → PASS. Then `cargo fmt` + `cargo clippy -p kiss-classify-vocab --all-targets -- -D warnings`.
- [x] **Step 4 — commit.** `git add -A && git commit` — `feat(sk4)!: respell kiss-classify-vocab to the sk4 24-dtype set (§6.1)`.

### Task 2: kiss-ref-core Dtype-variant rename ripple (compiler-guided)

**Files:** Modify every `crates/kiss-ref-core/src/*.rs` that matches on / constructs `Dtype::{S8,S16,S4,E4m3,E5m2,C32,C64}`, plus conformance tests.

**Note — Rust FP8 *type* vs Dtype variant:** the Rust structs `E4m3`/`E5m2` in `fp8.rs` (re-exported as `kiss_ref_core::E4m3`) are compute/storage types, **not** tokens — KEEP their Rust names (renaming is churn with no conformance value). Only the `Dtype` *variant* + token respell. Document the `Dtype::F8e4m3fn ↔ E4m3`(type) mapping where it's non-obvious.

- [x] **Step 1 — compile as the test.** `cargo build -p kiss-ref-core` → expect a bounded set of E0599 "no variant" errors listing every rename site.
- [x] **Step 2 — mechanical rename**, respecting the complex-flip ordering hazard (update `C64`→`C128` occurrences that mean pair-f64 first, then `C32`→`C64`; verify by the bits()/component_dtype() semantics, not blind text). Sites known: `component_dtype` consumers, `complex.rs` (c32/c64 component), `fp8.rs`, `resolve.rs` (legality/support/implemented_on match arms), coverage.
- [x] **Step 3 — green.** `cargo test -p kiss-ref-core` → PASS; then workspace `cargo test --workspace` (conformance tests reference `Dtype::` too). Fix ripple in `crates/kiss-ref-conformance/tests/*.rs` (e.g. `coverage.rs` INT_DTYPES `S8,S16,S4` → `I8,I16,I4`; the `[E4m3,E5m2]` lists → `[F8e4m3fn,F8e5m2]`).
- [x] **Step 4 — commit.** `refactor(sk4)!: ripple the §6.1 dtype-variant respell through kiss-ref-core + conformance`.

### Task 3: reserved + MX dtypes decline compute (typed, distinct from unknown)

**Files:** `crates/kiss-ref-core/src/resolve.rs` (legality/support), `.../src/lib.rs` (Error), `crates/kiss-ref-conformance/tests/coverage.rs`.

**Design:** recognized-vs-unknown already splits at `from_token` (Some vs None). For compute: a reserved/MX dtype is **compute-illegal** → `legality(op, d)==false` → `support==NotApplicable` for every op; and the eval path returns a typed `Error::ReservedOrScaleDtype(Dtype)` distinct from an unknown-dtype path. The op×dtype matrix grows to 24 dtypes; the 4 new columns are entirely NotApplicable.

- [x] **Step 1 — failing tests.** In `coverage.rs`: `for op in Op::ALL { for d in [F8e4m3fnuz,F8e5m2fnuz,F8e8m0,F8e6m2] { assert_eq!(support(op,d), NotApplicable) } }`; assert `Dtype::from_token("f8e4m3fnuz").is_some()` (recognized) while these decline. Add an eval-path test asserting a reserved dtype yields the typed decline. Run → FAIL.
- [x] **Step 2 — implement.** `legality`: early-return false for `d.declines_compute()`. Add `Error::ReservedOrScaleDtype(Dtype)` and return it on the eval entry when a value operand's dtype `declines_compute()`. Update the `coverage_support_consistency` `expect` match to map the 4 new dtypes → not-Done.
- [x] **Step 3 — green.** `cargo test --workspace` → PASS.
- [x] **Step 4 — commit.** `feat(sk4): reserved-fnuz + MX-scale dtypes are recognized but decline compute (typed)`.
- [x] **⚠ CONFIRM WITH ARCHITECT:** is `NotApplicable` the intended ledger verdict for reserved/MX (vs. a distinct "reserved-at-this-schema" state)? My read: NotApplicable + a typed decline error + recognized-`from_token` captures the spec's recognize-vs-unknown split honestly. Adjust if the Architect wants a 5th state.

### Task 4: visible Provisional cell state + SortNetwork #133 pin  ⚠ PENDING ARCHITECT MODELING CONFIRM

**Files:** `crates/kiss-ref-core/src/lib.rs` (Support), `.../src/kernels.rs:617,658` (cite #133), `crates/kiss-ref-conformance/src/lib.rs` (ledger provisional surface + summary), `tests/coverage.rs`.

**Design:** add `Support::Provisional { pin: &'static str, issue: &'static str }` (fields `&'static str` to keep `Support: Copy`). The SortNetwork index-output dtype is locally pinned to `Dtype::I64` (kernels.rs) with the wire dtype (i64 vs i32/u32) UNPINNED (KISS #133). Surface a provisional **count** in the Ledger + summary so "the number goes to zero" is a readable result; cite #133 at the pin site.

**Open modeling question (send to Architect — this is binding-awkwardness = spec evidence):** the pin is on SortNetwork's *index output* dtype, which is not an (op × *value*-dtype) coverage cell — the value cells are genuinely Done. Two honest options: (a) `support(SortNetwork, d)` returns `Provisional{...}` on its legal cells (marks the whole cell as pin-dependent — coarse, risks implying the *values* are unverified); (b) keep value cells `Done` and add a distinct typed `PROVISIONAL_PINS` registry `[{site, pin, issue}]` surfaced with a count (cleaner separation, but is it the "visible cell state" the ruling meant?). Default to (b) unless the Architect says (a). **Do not execute Task 4 until confirmed.**

### Task 5: structure_key wire codec (acc+mp §6.7-0013 + §6.19 OpAttr)  ⚠ CONDITIONAL — PENDING ARCHITECT SCOPE ANSWER

Only if the Architect says kiss-ref grows a structure_key emitter this phase. Current read: **out of scope** — kiss-ref has no codec surface; KISS's `structure_key.rs` reference owns the wire form; kiss-ref keeps attrs.rs typed enums. If confirmed out-of-scope, delete this task. If in-scope, plan a dedicated follow-up (byte-exact rules a–e, gem-symmetric `<acc>/<mp>`, omit-when-default, §6.19 OpAttr `u8` accumulator-ordinal + `u8` mp fields) — a separate plan, not folded here.

### Task 6: workspace 0.3.0 bump + full gates + byte-match leg

**Files:** `Cargo.toml` + each crate manifest (version + inter-crate dep versions).

- [x] **Step 1** — bump workspace + all 4 crates + inter-crate deps `0.2.4` → `0.3.0`.
- [x] **Step 2** — run all gates (fmt, clippy -D, `cargo test --workspace`, no_std thumbv7em, `cargo package --workspace --allow-dirty`). All green.
- [x] **Step 3 — byte-match leg.** Derive the 24 dtype tokens + 121 op tokens; state the EXACT command+flags used; report **3-of-4** (KISS reference, Fuel, kiss-ref) with unpopped-vocab@sk4 **unpublished** (sk4 respell `7d2c5d7`, unpushed, crate 0.1.0) explicitly excluded. Re-verify the vocab against `f4cc3ad`.
- [x] **Step 4 — commit + PR** — `chore(sk4)!: bump workspace to 0.3.0 (breaking sk4 regen)`; open PR; watch CI + reviewer comments.
- [x] **Step 5** — request Architect spec-currency statement (by commit) → then Eric authorizes the publish. (Both out of this plan's execution.)

---

## Self-Review

- **Spec coverage:** §6.1 renames (T1), FP8 f8-prefix + reserved fnuz (T1/T3), MX scales (T1/T3), complex total-width flip + component remap (T1/T2), closed-24 set (T1), #133 provisional pin (T4), version 0.3.0 (T6), byte-match (T6). SCHEMA_VERSION 3→4 + acc/mp + §6.19 OpAttr wire = KISS `structure_key.rs`, not kiss-ref (T5 conditional). ✔ covered or explicitly out-of-scope.
- **Placeholder scan:** none — the two `⚠` tasks carry full designs gated on a named confirmation, not "TBD".
- **Type consistency:** `Dtype::{I8,I16,I4,F8e4m3fn,F8e4m3fnuz,F8e5m2,F8e5m2fnuz,F8e8m0,F8e6m2,C64,C128}`, `is_reserved`/`is_mx_scale`/`declines_compute`, `Support::Provisional{pin,issue}`, `Error::ReservedOrScaleDtype(Dtype)` used consistently across tasks.

## Execution order under the current block
Architect (peer `92l1v72u`) is unreachable (peers MCP down) → **Tasks 1–3 execute now** (answer-independent). **Task 4** holds on the modeling confirm; **Task 5** holds on the scope answer; **Task 6** runs last. Send the two pending questions the moment the link recovers.

---

## OUTCOME (2026-08-12, both Architect answers received)

- **Q1 → CONFIRMED: Task 5 is OUT OF SCOPE / deleted.** kiss-ref has no wire codec and does not grow one; the §6.7-0013 acc+mp codec, §6.19 OpAttr wire blob, and `SCHEMA_VERSION` live in KISS's `structure_key.rs`. A 4th codec from a shared comprehension lineage adds ceremony, not freeze-gate evidence (evidence convention 6). **The byte-match is two tiers:** token-level (KISS reference / Fuel / Unpopped = the 3 separate codec derivers) and **vocabulary-level (kiss-ref)**. kiss-ref reports its leg as *vocabulary-level, no wire codec by design* — never "3-of-4" (which would imply 4 codec derivers).
- **Q2 → option (b), DONE (Task 4).** Typed `PROVISIONAL_PINS` registry `{site, value, issue}` in kiss-ref-core, count surfaced beside Done/Pending in the ledger summary; the SortNetwork #133 pin records value `i64` and a kernel test asserts the actual index dtype == the recorded value (forces reconciliation on a #133 ruling). Value cells stay `Done` — no false uncertainty. No `Support::Provisional` variant.
- **Tasks 1, 2, 3, 4, 6 COMPLETE**; 370 tests, fmt, clippy `-D`, `cargo package --workspace` (exit 0), no_std thumbv7em all green; workspace at breaking **0.3.0**.
- **Byte-match leg (vocabulary tier) VERIFIED:** kiss-ref's 24 dtype tokens `diff`-identical to the merged §6.1 table at `19c3ad7`; 121 op tokens unchanged. Command: `cargo test -p kiss-classify-vocab` (round-trips all 24) + `cargo test -p kiss-ops-vocab` (121 op tokens); dtype set diffed against `spec/classify.md@19c3ad7` §6.1 table.
- **Remaining (out of this plan's execution):** PR review/merge; then request the Architect spec-currency statement (by commit); then Eric authorizes the 0.3.0 publish. **Verification precedes publication.**
