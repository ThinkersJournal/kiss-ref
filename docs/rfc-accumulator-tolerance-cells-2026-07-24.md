# RFC: Float-reduction accumulator width as a tolerance-cell key (KISS #90, direction b)

**Status:** draft · 2026-07-24 · authored by kiss-ref (the reference implementation)
**Direction:** **(b)** — signaled by Eric 2026-07-24; supersedes the open (a)/(b)
choice on KISS #90.
**Affects:** KISS-Ops §6.0-0004, §6.17-0005/-0007, §6.11-0002/-0003; KISS-Classify
§6.7-0006 (`<acc>` key coordinate); the KISS-Conform tolerance-cell machinery.
**Author role:** reference-implementation feedback (C-4). kiss-ref surfaced the gap
by *executing* the tensor layer on the narrow lanes (commit `e8ae0b5`,
`narrow_tensor_lane.rs`) and pinned the divergence as a golden rather than hiding
it.

## Summary

Float `sum`/`prod` reductions, scans, and contractions have an **unpinned
accumulator width** (§6.0-0004; open-question 6). §6.17-0005 pins a *bit-stable
reference profile* — ascending index order, **accumulator at the storage dtype's
precision** — which kiss-ref implements, so **kiss-ref is conformant, not wrong**.
But §6.17-0007 requires a *declared tolerance* to classify a candidate pass-or-flag,
and **no clause supplies one for a narrow compute dtype**. The result: two
provably-conformant implementations diverge with nothing to bound the gap.

**Measured (pinned goldens, `narrow_tensor_lane.rs`):** reduce-sum of `[128, 8×8]`
(wide-truth 192, representable in every lane) is **128** under the storage-precision
profile (`e4m3`/`e5m2`: `128+8` rounds back to 128, RNE stagnation) and **192** under
an f32 accumulator (every real FP8 tensor core). **33% divergence, both conformant,
no tolerance.** (Also: softmax rows don't sum to 1 — `e4m3` +3.1%, `e5m2` −6.3%.)

**Direction (b), refined:** treat these ops as **tolerance-cells keyed on
`(compute-dtype, accumulator-dtype)`**, reusing the existing sk3/KISS-Classify
`<acc>` key coordinate — **no new key field**. The accumulator width becomes a
determinism-class axis *orthogonal to `<mp>`* (Baracuda's framing; the `<acc>`
coordinate exists in the key precisely for this). This neither reverses §6.17-0005
nor constrains silicon: any accumulator dtype stays conformant — it is simply
compared against *its own* reference cell.

---

## The core move: an accumulator-parameterized reference

The 128-vs-192 gap is only a "failure" if both are compared against **one**
reference. Under (b) they are **different cells** with **different references**:

- A candidate declaring `<acc> = e4m3` is compared against the reduction computed
  **with an e4m3 accumulator** (the §6.17-0005 storage-precision profile) → **128**.
- A candidate declaring `<acc> = f32` is compared against the reduction computed
  **with an f32 accumulator** → **192**.
- The cross-`<acc>` divergence (128 vs 192) is **expected and conformant** — the two
  results live in different `(compute, acc)` cells and are never compared to each
  other.

Within a single cell the only remaining nondeterminism is **reduction reassociation**
(§6.0-0004 leaves order unpinned; §6.17-0005 pins ascending *for the reference*), so
the **declared tolerance is tight** — a small ULP band at the accumulator's
precision, not the catastrophic cross-accumulator gap.

This requires one reference-implementation change (below): kiss-ref's reduction
reference must be **parameterized by the accumulator dtype**, so it can emit the
reference value for *any* declared `<acc>`, not only the storage-precision diagonal.

---

## Proposed normative clauses

### C1 — classification (amends §6.0-0004 / §6.17-0005)
A float `sum`/`prod` reduction, scan, or contraction is a **tolerance-cell** whose
determinism class and declared tolerance are a function of the pair
**`(compute-dtype, accumulator-dtype)`**. The accumulator dtype MUST be a float
dtype at least as wide as the compute dtype (the reference forbids a *narrower*
accumulator — it cannot improve on storage precision and has no use). The
storage-precision profile of §6.17-0005 is the diagonal cell
`accumulator-dtype == compute-dtype`.

### C2 — the key coordinate (binds KISS-Classify §6.7-0006)
The tolerance cell is keyed on the existing `<acc>` coordinate; **no new key field
is introduced.** A kernel that does not declare `<acc>` is read as
`accumulator-dtype == compute-dtype` (the §6.17-0005 default), preserving today's
behavior for every kernel that never opted in.

### C3 — the reference value (binds §6.17-0007)
For a cell `(S, A)`, the **reference value** is the reduction evaluated in the
§6.17-0005 pinned order (ascending index) with each input rounded to the compute
dtype `S`, each accumulate atom rounded to the **accumulator dtype `A`**, and the
final result rounded to `S`. This is a *fully specified, deterministic* value (order
and both roundings pinned). The declared tolerance bounds a candidate's
reassociation freedom against THIS reference — **not** against wide-precision truth
directly, which for a narrow accumulator is unboundedly far (the stagnation case).
§6.17-0007's "bound against wide-precision truth" is satisfied transitively: the
`(S, f64)` / widest-`A` cell IS wide-precision truth, and every narrower-`A` cell
declares how far its reference legitimately sits from it.

### C4 — the tolerance form (calibratable; proposed)
The declared tolerance for cell `(S, A)` is a **small multiple of the accumulator
ULP** — the reassociation error of an `N`-element reduction with an `A`-precision
accumulator, i.e. `O(N · eps_A · |max partial sum|)`, expressed as a ULP-of-`A` band
so the sign-magnitude metric applies. It is **NOT** a fixed constant and **NOT** a
bound against wide truth; it is tight *within the cell*. The exact multiple is the
one open calibration point (see below).

### C5 — no-inf monoid identity (independent; already routed)
Orthogonal to (b): when a monoid identity is `±∞` (`max` = −∞, `min` = +∞,
§6.11-0002) and the compute dtype has no infinity encoding (`e4m3fn`), the identity
MUST materialize as the dtype's finite extremal magnitude of the correct sign
(±448 for `e4m3fn`). Both implementations already do this; landing separately as a
small PR.

---

## Reference-implementation obligations (what kiss-ref changes)

1. **Parameterize the reduction reference by accumulator dtype.** Today
   `kernels::reduce` / `prefix_scan` / `tensor_ops::matmul` accumulate at storage
   precision (`acc == compute`, the C1 diagonal). Add an accumulator-dtype parameter
   so the reference can emit the value for any declared `<acc>` — e.g. accumulate in
   `A` (via the existing promote-compute-round machinery at `A`'s precision) while
   rounding inputs to `S`. The current storage-precision path becomes the
   `A == S` case, unchanged.
2. **Expose the per-cell reference on the differential seam** (`diff.rs`), so
   Baracuda's 3b on-device FP8 diff and Fuel's advisory can request the reference for
   *their* declared `<acc>` and compare within the C4 band.
3. **Keep the goldens.** `narrow_tensor_lane.rs` already pins the `A == S` diagonal
   (128/128, saturations, softmax drift); add the `A == f32` cells (192, …) as the
   companion so both sides of a representative `(S, A)` pair are conformance-pinned.
4. kiss-ref holds the affected cells **Provisional** until this RFC rules, then
   conforms and un-flags (same discipline as the §6.11 gaps).

## Open calibration point (the one genuinely-open number)

The C4 tolerance **multiple** (`k` in `k · N · eps_A`) is not derivable from first
principles alone — it depends on the permitted reassociation set. kiss-ref proposes
to calibrate it empirically: for each `(S, A)`, sweep the reassociation orders a
conformant kernel may use (pairwise tree, blocked, sequential) against the pinned
ascending reference over an adversarial value set, and set `k` to the observed worst
case plus a margin. kiss-ref can produce that calibration table as a follow-up once
the clause shape is ratified.

## Why (b) over (a)

(a) — pinning one canonical accumulator per dtype — would make narrow reductions
byte-comparable again, but reverses §6.17-0005 for narrow lanes and renders a
genuinely-narrow-accumulator device non-conformant (constraining silicon). (b) keeps
every accumulator width conformant, matches the hardware reality that FP8 tensor
cores accumulate in f32 while some accelerators accumulate narrow, reuses the
existing `<acc>` key at **zero new key cost**, and turns the divergence from an
unclassifiable failure into a keyed, bounded, executable tolerance cell.
