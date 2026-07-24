# RFC: Float-reduction accumulator width as a tolerance-cell key (KISS #90, direction b)

**Status:** draft · 2026-07-24 · authored by kiss-ref (the reference implementation)
**Direction:** **(b)** — signaled by Eric 2026-07-24; supersedes the open (a)/(b)
choice on KISS #90.
**Clauses AMENDED (existing text changes):** §6.17-0007 (the "bound against
wide-precision truth" sentence — C3 rewrites it), §6.0-0004 / §6.17-0005 (C1 adds
the `(compute, acc)` classification), **KISS-Classify §6.7-0001 / §6.7-0006** (C2 —
adds an accumulator coordinate to **non-contraction** reduction/scan cells, which
have none today), **KISS-Conform §6.5-0007** (C3 — scoping: as written it covers
only transcendental atoms, so this RFC adds the parallel reduction-cell oracle
obligation rather than reusing it).
**Clauses BOUND/REUSED (no text change):** §6.6-0003 (extent-free key, C4),
§6.8-0004 (tolerance lives in the Contract Guarantees, C4).
**Clause CROSS-REFERENCED (independent):** §6.11-0002 (no-inf monoid identity, C5).

**Review note (2026-07-24):** two findings from the automated review of PR #92 were
verified correct against the spec and are folded into this revision — (1) `<acc>`
is contraction-cell-only in the current key grammar (§6.7-0006), so non-contraction
reductions/scans need a key extension (C2 no longer claims "zero new key field" for
them); (2) §6.5-0007 is transcendental-atom-scoped, so C3 no longer cites it as the
reduction oracle. The first correction weakens the original "zero key cost"
argument for the reduction/scan case (the motivating one) — it is stated plainly
below; (b) still wins over (a) on *not constraining silicon*, not on key cost.
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
`(compute-dtype, accumulator-dtype)`**. The accumulator width becomes a
determinism-class axis *orthogonal to `<mp>`* (Baracuda's framing). For
**contraction** (`matmul`) cells this reuses the existing `<acc>` coordinate at zero
key cost (it already exists, §6.7-0006). For **non-contraction** reductions/scans —
which is the motivating case — the key currently has **no** accumulator coordinate
(§6.7-0001/-0006, C2), so keying them costs a bounded, schema-versioned key
extension. Either way this neither reverses §6.17-0005 nor constrains silicon: any
accumulator dtype stays conformant, compared against *its own* reference cell.

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

### C2 — the key coordinate (**AMENDS KISS-Classify §6.7-0001 / §6.7-0006**)
The tolerance cell is keyed on the accumulator dtype, but the current key grammar
carries `<acc>` **only for dense-contraction cells** (§6.7-0006: field 9, present
only for a `gem` cell; §6.7-0001 fixes non-contraction cells at exactly nine
fields). So:
- **Contraction (`matmul`) cells** — keyed at **zero new key cost**; the existing
  `<acc>` in field 9 already carries it.
- **Non-contraction reduction/scan cells (`reduce`, `prefix_scan`)** — have **no
  accumulator coordinate today**, and the 128-vs-192 motivating divergence is a
  `reduce`, not a contraction. Keying them therefore **requires adding an
  accumulator coordinate to the non-contraction key** — a schema-versioned change
  (a new field, or an `<acc>`-bearing extension of the field-8 `<reduce>` code).
  The DTYPE *spelling* is additive per §6.7-0007, but the field-count change is
  schema-affecting (§6.7-0001), so this is **not** zero-cost for reductions/scans.
  *(This corrects the initial draft's "no new key field" over-claim, per the PR #92
  review; the honest cost is stated so the (a)/(b) comparison stays fair.)*

Default when the coordinate is absent: `accumulator-dtype == compute-dtype` (the
§6.17-0005 diagonal), so every existing key is unchanged in meaning and every kernel
that never opts in behaves exactly as today.

### C3 — the reference value (**AMENDS §6.17-0007**)
§6.17-0007 currently ends: *"A tolerance cell's declared tolerance MUST bound the
combined input-rounding-plus-reduction-order error against the wide-precision truth
— the KISS-Conform §6.5-0007 oracle evaluation."* For an accumulator-keyed reduction
cell this sentence is unsatisfiable in any useful form: a narrow-accumulator
reference sits ~33% from wide truth, so "bound against wide truth" forces a 33%
tolerance that flags nothing. It also directly contradicts the per-cell reference
below. **Replace it**, for the `(compute, acc)` reduction/scan/contraction cells,
with:

> *The reference value of a `(compute-dtype S, accumulator-dtype A)`
> reduction/scan/contraction cell is the §6.17-0005-ordered (ascending-index)
> evaluation with each input rounded to `S`, each accumulate atom rounded to `A`,
> and the result rounded to `S`. The cell's declared tolerance MUST bound a
> candidate's reduction-order (reassociation) error against **that per-cell
> reference**. The cell whose `A` is the widest permitted accumulator dtype is the
> **wide-precision reduction reference** (wide-precision truth by construction —
> accumulating in the widest float is the tightest reduction); every narrower-`A`
> cell's reference is a defined, representable point whose distance from that widest
> reference is the cell's **characteristic** (the semantics of using that
> accumulator), not an error to be toleranced away.*

**Oracle-scope note (Copilot finding 2, verified).** §6.5-0007 as written is the
oracle-accuracy floor for **transcendental atoms** (conform.md), not reductions — so
neither §6.17-0007's current text nor this RFC may treat it as the reduction oracle.
This RFC therefore introduces the **reduction-cell oracle obligation in parallel**:
the widest-`A` reduction reference is defined intrinsically (above), the reduction
analogue of §6.5-0007's transcendental floor. A companion one-line amendment SHOULD
either extend §6.5-0007 to name reduction cells or add a sibling clause; the RFC
does not depend on which, since the widest-`A` reference is self-defining.

This keeps §6.17-0007's intent — bound the error a *candidate* may introduce — while
removing the self-contradiction: the widest-`A` cell is wide-precision truth, and
narrower cells bind against their own fully-specified deterministic reference (order
and both roundings pinned). Wide-truth conformance is thus preserved **transitively**,
not discarded.

### C4 — the tolerance form (calibratable; lives in the **Contract Guarantees**, not the key)
The declared tolerance for cell `(S, A)` scales with the reduction length:
`k(S,A) · N · eps_A · |max partial sum|`, expressed as a ULP-of-`A` band so the
sign-magnitude metric applies. It is **NOT** a fixed constant and **NOT** a bound
against wide truth; it is tight *within the cell*.

**Keying discipline (§6.6-0003 / §6.8-0004) — load-bearing.** `N` (the reduction
extent) is a **runtime** quantity, and the KISS-Classify key is **extent-free**
(§6.6-0003), so `N` MUST NOT enter the key. The key carries only the `(compute, acc)`
**class** via the `<acc>` coordinate (C2). The numeric tolerance **formula** lives in
the Contract **Guarantees** and is evaluated **per-invocation** with the real `N`
(§6.8-0004: "the tolerance used MUST be the one declared in the contract
Guarantees"). `k(S,A)` is the per-cell constant declared there — the one open
calibration number (see below). A reader must not conclude `N` is keyed: the class is
keyed, the magnitude is computed per call.

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

The C4 constant **`k(S, A)` is declared per `(compute, acc)` cell** (the
summation-bound constant genuinely differs by accumulator width) and is not derivable
from first principles alone — it depends on the permitted reassociation set. kiss-ref
proposes to calibrate it empirically and **adversarially**: for each `(S, A)`, sweep
the reassociation orders a conformant kernel may use (pairwise tree, blocked,
sequential, and worst-case adversarial reorderings) against the pinned ascending
reference over an adversarial value set, and set `k(S,A)` to the observed worst case
plus a margin. kiss-ref can produce that per-cell calibration table as a follow-up
once the clause shape is ratified.

## Why (b) over (a)

(a) — pinning one canonical accumulator per dtype — would make narrow reductions
byte-comparable again, but reverses §6.17-0005 for narrow lanes and renders a
genuinely-narrow-accumulator device non-conformant (constraining silicon). (b) keeps
every accumulator width conformant, matches the hardware reality that FP8 tensor
cores accumulate in f32 while some accelerators accumulate narrow, and turns the
divergence from an unclassifiable failure into a keyed, bounded, executable
tolerance cell. **(b)'s advantage over (a) is that it does not constrain silicon —
NOT that it is key-free:** contraction cells reuse the existing `<acc>` at zero cost,
but non-contraction reductions/scans (the motivating case) need a bounded
schema-versioned key extension (C2). That cost is real and is the honest price of
keeping every accumulator width conformant.
