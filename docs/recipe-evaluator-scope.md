# kiss-ref recipe-evaluation API — scoping note

**Status:** **v1 implemented** in `crates/kiss-ref-core/src/recipe.rs` · 2026-07-21 ·
scoped against Baracuda's grammar extract
(`baracuda/docs/kiss-ops-recipe-grammar-spec-2026-07-21.md`), which is an INPUT to
the KISS-owned recipe grammar (Contract §2.3 / Ops §6.13/§6.19 / §6.4-0009-0010 +
Fuel co-design + mlgheozs's 3B). Reconcile toward KISS on any divergence.

**v1 covers** (float lane): the scalar atoms (via `element_map`+`eval_op`),
`reduce`/`prefix_scan` (with `nokd` squeeze), `matmul`, and the value leaves
(`Bind`/`const`/`runtime_scalar`/`reduced_count`), returning per-node `DetClass`.
An iterative worklist walks the DAG (bounded to heap, never a stack overflow). End
-to-end recipes (matmul+bias+relu, softmax) evaluate correctly.
**v1 defers:** the index-bearing nodes (`gather`/`scatter`/`sort_network` — mixed
float/integer operands) and `iota` (needs the §6.20 shape oracle).

## Target API

```rust
pub fn eval_recipe<T: ScalarFloat>(
    dag: &FlatDag,
    inputs: &[Tensor<T>],
    params: &[T],          // runtime_scalar slots
) -> Result<(Vec<Tensor<T>>, Vec<DetClass>), Error>
```

Walk the canonical flat-DAG; dispatch each `Op{name, attrs}` to a kernel; resolve
`Bind(i) → inputs[i]` and `Reduced(i) →` the i-th fold node's output; return the
output tensor(s) + per-node `DetClass` (from `(op, attrs)`). Float lane first.

## Node → kernel mapping (what's ready vs. what's needed)

**Ready — dispatch to existing kernels/`eval_op`:**
- `add`/`sub`/`mul`/`div`, every `<unary>`/`<binary>` KISS-Ops atom, `select` →
  `kernels::element_map` with the op as the per-element body (the bridge already
  evaluates all 106 scalar ops via `eval_op`). DetClass from the atom (exact-byte,
  or ULP for the §6.8 transcendentals).
- `reduce{monoid,axes}` → `kernels::reduce` (keepdim). `prefix_scan{monoid,axis,
  exclusive}` → `kernels::prefix_scan`. `gather{axis,oob,index_dtype}` →
  `kernels::gather`. `scatter{axis,combine,oob,index_dtype}` → `kernels::scatter`
  (kiss-ref is a superset: it has assign/max/min beyond the v1 `atomic-add` floor).
- `reduced_count{axes}` → product of extents over `axes` (already computed inside
  `reduce_mean`; lift to a node).
- `matmul{roles}` → `tensor_ops::matmul` for rank-2 `mk.kn`.
- Per-node `DetClass` → `bridge::monoid_det` for `reduce`/`prefix_scan`,
  `tensor_ops::op_det` for the fixed-semantics ops (already instance-aware).

**Small additions needed (scoped, each bounded):**
1. **`const{bits}`** — a raw-bit constant (recipe uses a bit-pattern, kiss-ref has
   symbolic `ConstSym`). Add a raw-bits→`T` const materialization.
2. **`iota{axis}` / `coord(axis)` leaf** — kiss-ref's `Expr` has only
   `Input`/`Const`/`Apply`; the §6.12-0001 body leaf set also includes
   `coord(axis)`. The evaluator must resolve `coord` against the *output
   coordinate* (a tensor-context leaf, not a scalar `Expr` leaf).
3. **`runtime_scalar{slot}` / `param(i)` leaf** — same: resolve against the
   `params` array.
4. **`flip{axis}`** — new small kernel (reverse along an axis; a negative-stride
   view or a copy). Not in the current kiss-ref tensor layer.
5. **`reduce` `nokd` (collapse)** — kiss-ref's `reduce` is keepdim-fixed
   (extent-1). Add an axis-squeeze for `nokd`. (See reconciliation item below.)
6. **The DAG walker + `Reduced(i)` fold-staging + RowReduce epilogue** — the new
   evaluator machinery: topological walk, compute each fold node, stage
   `Reduced(i)` outputs, evaluate the elementwise epilogue over them + full-width
   `Bind`s (the RmsNorm/LayerNorm shape), collect per-node DetClass.

**Partial / roadmap:**
- `matmul` batched / role-general (`b`/multi-role) — kiss-ref is rank-2 `mk.kn`
  only today; batched matmul is on the coverage roadmap. Ties to the sk3 schedule/
  DetClass discussion (pinned-schedule → exact-byte vs default tolerance).

## kiss-ref is AHEAD of the v1 grammar (a co-owner finding)

The extract's §5 honest-misses **pooling/window, sort, im2col** — but **kiss-ref
already implements all of them** (`window.rs`: `avg_pool`/`max_pool`/`im2col`;
`kernels::sort_network`; `tensor_ops::argmax`). So when the recipe grammar grows to
cover these, kiss-ref is already the ready reference — and kiss-ref's implemented
semantics can *inform* the grammar's expansion (esp. the pooling divisor /
count_include_pad and the sort total-order/NaN-greatest rules). Worth surfacing to
the KISS consolidation.

## "Needs-tighter-pinning" — routed to the KISS consolidation (mlgheozs + Ops editor)

As co-owner/validator, kiss-ref's evaluator scoping surfaces these grammar
questions (value semantics; the shape-oracle §6.20 DimExpr is separate):

1. **Body-leaf set — value vs shape grammar separation** (refined with Baracuda).
   The **value-recipe body leaves** are `{input(Bind), const, iota(coord),
   runtime_scalar(param), reduced_count}` (each a 0-child node). `reduced(i)` is a
   **contextual child-edge** to the i-th fold node (not a free leaf) — resolvable
   only in a reduction/scan/RowReduce epilogue. **`extent(axis)` is NOT a
   value-recipe leaf** — it is a SHAPE quantity (§6.20 `DimExpr::Extent`), so
   §6.12-0001 listing it as a value body leaf conflates the value and shape
   grammars. Evaluator resolution: `coord(axis)` = output coordinate;
   `reduced_count` = ∏ reduced extents (a value); `extent` comes from the tensor
   shape (shape-side, never a value leaf); `reduced(i)` = the i-th fold output.
   kiss-ref already treats `reduced_count` as a value and `extent` as shape-side.
2. **`reduce` keepdim.** The recipe grammar has `keepdim ∈ {kd, nokd}`, but
   KISS-OPS-6.11-0008 pins `reduce` keepdim = **fixed 1** for the carrier. Which is
   normative for the DAG — does `reduce` carry `keepdim`, or is `nokd` a separate
   reshape node? (kiss-ref implements keepdim-fixed; needs the ruling.)
3. **`matmul` roles + schedule.** The batched/role-general evaluation order and its
   DetClass (fixed-schedule → exact-byte vs. nondeterministic) should reconcile
   with the sk3 `(acc, mp)` reproducibility classification kiss-ref already signed
   off on — same underlying "is the reduction schedule pinned" question.
4. **`const{bits}` interpretation** on the narrow floats (raw-bit → dtype value).

## Build sequence (per CireSnave — no rush)

Foundation (`FlatDag` type + walker + `Reduced` staging) → leaf resolution
(coord/param/const-bits) → node dispatch (reuse kernels) → `flip` + `nokd` →
per-node DetClass collection → conformance corpus of recipes. Gated by the coverage
ledger (`support(op,dtype)==Done`) and the KISS grammar landing.
