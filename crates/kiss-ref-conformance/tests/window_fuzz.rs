// SPDX-License-Identifier: MIT OR Apache-2.0
//! Never-panic fuzz suite for the **window family** — `avg_pool`, `max_pool`,
//! `im2col` (`kiss-ref-core/src/window.rs`).
//!
//! CLAUSES: KISS-OPS-6.13-0004 (per-axis window geometry — `window_size`/`stride`/
//! `dilation`/`padding` + the `count_include_pad` divisor) and KISS-OPS-6.13-0009
//! (each structured op's semantics + inf/NaN/OOB edges MUST be those of its §6.11
//! decomposition). The cross-op oracles below ground on that decomposition: the
//! max-pool empty-window identity = the `reduce(max)` identity (KISS-OPS-6.11-0002;
//! finite −448 on e4m3fn per -0002a), im2col OOB taps zero-filled = `gather`
//! zero-fill (KISS-OPS-6.11-0004), avg/max OOB taps skipped per KISS-OPS-6.13-0004.
//!
//! WHY THIS FILE EXISTS: the window family is the one op region the adversarial
//! DAG fuzzer (`recipe_fuzz.rs`) cannot reach — those ops are not recipe `Node`
//! variants, so no generator ever drives their `(kernel, stride, padding,
//! dilation, shape)` parameter arithmetic. That arithmetic already produced one
//! real unchecked-overflow bug (the raw `2 * p`, since fixed to `checked_mul` —
//! see the `bad_window_params_error_not_panic` regression in `window.rs`), so
//! the space has a bug history and, until this file, zero generator-enforced
//! coverage.
//!
//! CONTRACT: `kiss-ref-core` never panics — every failure is a typed [`Error`].
//! Every call here is wrapped in `std::panic::catch_unwind`: `Ok` and `Err` both
//! PASS, only a panic FAILS, and the failing seed is printed for replay.
//!
//! ## What is actually checked (falsifiable, not vacuous)
//! 1. **Never-panic** — the primary contract.
//! 2. **An independent `u128` model of the window shape formula.** `Stage1` +
//!    [`expect_avg`]/[`expect_max`]/[`expect_im2col`] recompute the KISS-OPS-6.13-0004
//!    `floor((in + 2·pad − dilation·(k−1) − 1) / stride) + 1` in `u128` — an
//!    arithmetic width in which none of the intermediates can overflow — and
//!    pin the EXACT `Ok` shape or the EXACT `Err` value. HONESTY NOTE: the model
//!    is a re-derivation of the same spec formula, not a second independent
//!    spec; what it buys is that a `usize`-overflow/wrap bug in the
//!    implementation (exactly the prior `2 * p` defect's signature) shows up as
//!    a model/impl divergence rather than as a silently wrong extent.
//! 3. **Cross-op oracles** (`cross_check_*`): with `count_include_pad`,
//!    `avg_pool` must equal the tap-sum of `im2col` divided by the window size,
//!    bit-for-bit; with all-zero padding, `max_pool` must equal the tap-max of
//!    `im2col`, and the two `count_include_pad` settings must agree. These use
//!    `eval_op` for the scalar step (the window layer's job is index arithmetic,
//!    not scalar semantics) and independent code for the indexing.
//! 4. **Hand-computed goldens** (`anchor_*`) pin correct behavior end to end.
//!
//! ## The allocation budget guard (read this before trusting the skip counter)
//! `avg_pool`/`max_pool`/`im2col` size their output buffer from the *parameters*,
//! so a pathological-but-non-overflowing padding asks for a buffer larger than
//! the machine. The generator therefore never emits the mid magnitude band for
//! `padding` (roughly 2^20 … 2^62): those values are neither small enough to run
//! nor large enough to force a typed `ShapeOverflow`, and the resulting
//! allocation is an *abort*, which `catch_unwind` cannot contain (it would kill
//! the test process rather than report a finding). That band is instead owned by
//! the two named regression tests at the bottom of this file. Everything else is
//! backstopped by [`affordable`], which runs the `u128` model first and skips any
//! tuple that would neither decline early nor fit the budget. Skips are counted
//! and their rate is asserted, so the guard cannot silently eat the corpus.
//!
//! ## Reproducing a failure
//! The failure message prints `seed=0x…` and the mode. Rerun exactly that case:
//!   KISS_WINDOW_FUZZ_SEED=0x<seed> KISS_WINDOW_FUZZ_MODE=<plausible|adversarial> \
//!     cargo test -p kiss-ref-conformance --test window_fuzz repro_from_env -- --ignored --nocapture
//! Seeds are stable for a given generator version; changing the generator
//! invalidates old seeds (standard fuzzer caveat).
//!
//! Budget: the default tests total ~5k cases with bounded sizes (well under 10s
//! wall). `fuzz_soak` is `#[ignore]`d for on-demand soaks.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use half::f16;
use kiss_ops_vocab::Op;
use kiss_ref_core::window::{avg_pool, im2col, max_pool};
use kiss_ref_core::{eval_op, E4m3, Error, ScalarFloat, Tensor, MAX_RANK};

// ---- iteration budgets (tune here on a slow machine) ------------------------

const PLAUSIBLE_ITERS: u64 = 900;
const ADVERSARIAL_ITERS: u64 = 1800;
const NARROW_ITERS: u64 = 250;
const CROSS_ITERS: u64 = 500;
const DET_ITERS: u64 = 150;
const SOAK_ITERS_PER_MODE: u64 = 40_000;

/// Output-element cap for a single generated case (see the module note on the
/// allocation budget guard). A tuple whose predicted output exceeds this and is
/// NOT guaranteed to decline early is skipped, not run.
const OUT_CAP: u128 = 2_000;
/// Cap on total tap work (`out_numel · kernel_numel`) — also the `im2col`
/// element cap, since `im2col` materializes exactly that many elements.
const WORK_CAP: u128 = 20_000;
/// Cap on the element count of a generated INPUT tensor.
const INPUT_CAP: usize = 4_096;

// ---- seed bases (fixed; per-case seed is derived and printed on failure) ----

const SEED_PLAUSIBLE: u64 = 0xB0FA_0B1A_5EED_0101;
const SEED_ADVERSARIAL: u64 = 0xB0FA_0B1A_5EED_0102;
const SEED_NARROW: u64 = 0xB0FA_0B1A_5EED_0103;
const SEED_CROSS: u64 = 0xB0FA_0B1A_5EED_0104;
const SEED_DET: u64 = 0xB0FA_0B1A_5EED_0105;
const SEED_SOAK: u64 = 0xB0FA_0B1A_5EED_0106;

// ---- deterministic PRNG -----------------------------------------------------

/// xorshift64* — deterministic, seedable, ~5 lines. NOT crypto; fuzz only.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n.max(1) as u64)) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
    fn pick(&mut self, pool: &[usize]) -> usize {
        pool[self.below(pool.len())]
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-case seed. The FAILURE MESSAGE prints this value (not base+i), so a
/// repro needs only (seed, mode).
fn case_seed(base: u64, i: u64) -> u64 {
    splitmix64(base ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

// ---- case representation ----------------------------------------------------

#[derive(Clone, Debug)]
struct Case {
    shape: Vec<usize>,
    data: Vec<f64>,
    axes: Vec<usize>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    count_include_pad: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Plausible,
    Adversarial,
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Plausible => "plausible",
        Mode::Adversarial => "adversarial",
    }
}

#[derive(Clone, Copy, Debug)]
enum Phase {
    Generate,
    AvgPool,
    MaxPool,
    Im2col,
}

// ---- value pools ------------------------------------------------------------

const SPECIALS: [f64; 8] = [
    -0.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1e300,
    5e-324,
    f64::MAX,
    -1e-300,
];

/// Kernel extents. Every huge value here is SAFE at any magnitude: a kernel
/// larger than the padded input drives the pooled extent to 0, so the output is
/// empty and nothing is allocated (`eff > padded ⇒ 0`, `window.rs:41`).
const WILD_KERNEL: [usize; 12] = [
    0,
    1,
    2,
    3,
    4,
    7,
    64,
    255,
    1 << 32,
    1 << 62,
    usize::MAX - 1,
    usize::MAX,
];
/// Strides. A huge stride only SHRINKS the output (`(padded−eff)/s + 1 → 1`).
const WILD_STRIDE: [usize; 9] = [0, 1, 2, 3, 5, 64, 1 << 40, 1 << 63, usize::MAX];
/// Paddings. DELIBERATELY BIMODAL — small values that keep the output bounded,
/// then only values `>= 2^63`, for which `p.checked_mul(2)` provably fails and
/// the call declines with `ShapeOverflow` before touching memory. The mid band
/// (2^20 … 2^62) is excluded: see the module-level note on the budget guard.
const WILD_PADDING: [usize; 10] = [
    0,
    1,
    2,
    3,
    8,
    64,
    1 << 63,
    (1 << 63) + 1,
    usize::MAX - 1,
    usize::MAX,
];
/// Dilations. Huge values either overflow `dil·(k−1)+1` (typed `ShapeOverflow`)
/// or blow the effective window past the padded input (empty output).
const WILD_DILATION: [usize; 9] = [0, 1, 2, 3, 16, 1 << 32, 1 << 62, 1 << 63, usize::MAX];

/// Steps of 1/8 in [-125, +125] — exact in f32/f64 and representable in f16.
fn plain_tame(rng: &mut Rng) -> f64 {
    ((rng.next() % 2001) as i64 - 1000) as f64 / 8.0
}

fn tame_value(rng: &mut Rng) -> f64 {
    if rng.chance(6) {
        SPECIALS[rng.below(SPECIALS.len())]
    } else {
        plain_tame(rng)
    }
}

// ---- the u128 model (independent width) -------------------------------------

const USIZE_MAX_U128: u128 = usize::MAX as u128;

/// `floor((in + 2·pad − dilation·(k−1) − 1) / stride) + 1` computed in `u128`,
/// declining exactly where the `usize` implementation's `checked_*` chain does.
/// Every operand is `< 2^64`, so no `u128` intermediate can overflow.
fn model_pool_out_dim(in_d: u128, k: u128, s: u128, p: u128, dil: u128) -> Result<u128, Error> {
    if s == 0 || k == 0 {
        return Err(Error::ShapeMismatch {
            expected: 1,
            got: 0,
        });
    }
    // `dil.checked_mul(k - 1).and_then(|x| x.checked_add(1))`
    let eff = dil * (k - 1) + 1;
    if eff > USIZE_MAX_U128 {
        return Err(Error::ShapeOverflow);
    }
    // `p.checked_mul(2).and_then(|pp| in_d.checked_add(pp))`
    let padded = in_d + 2 * p;
    if padded > USIZE_MAX_U128 {
        return Err(Error::ShapeOverflow);
    }
    if padded < eff {
        return Ok(0);
    }
    // `((padded - eff) / s).checked_add(1)`
    let out = (padded - eff) / s + 1;
    if out > USIZE_MAX_U128 {
        return Err(Error::ShapeOverflow);
    }
    Ok(out)
}

/// The outcome of `window_out_shape` — stage 1 of all three ops.
#[derive(Clone, Debug, PartialEq)]
enum Stage1 {
    /// The op declines with exactly this error.
    Decline(Error),
    /// The pooled shape: the input shape with each spatial axis replaced (a
    /// duplicated axis is written twice — last write wins, as in `window.rs`).
    Shape(Vec<u128>),
}

fn model_stage1(c: &Case) -> Stage1 {
    let rank = c.shape.len();
    let n = c.axes.len();
    if c.kernel.len() != n || c.stride.len() != n || c.padding.len() != n || c.dilation.len() != n {
        return Stage1::Decline(Error::ShapeMismatch {
            expected: n,
            got: c.kernel.len(),
        });
    }
    let mut out: Vec<u128> = c.shape.iter().map(|&d| d as u128).collect();
    for (i, &ax) in c.axes.iter().enumerate() {
        if ax >= rank {
            return Stage1::Decline(Error::AxisOutOfRange { axis: ax, rank });
        }
        match model_pool_out_dim(
            c.shape[ax] as u128,
            c.kernel[i] as u128,
            c.stride[i] as u128,
            c.padding[i] as u128,
            c.dilation[i] as u128,
        ) {
            Ok(d) => out[ax] = d,
            Err(e) => return Stage1::Decline(e),
        }
    }
    Stage1::Shape(out)
}

/// Saturating `u128` product (a product of eight `usize` extents can exceed
/// `u128`; saturating is sound here because every use compares against a cap).
fn sat_prod(v: impl IntoIterator<Item = u128>) -> u128 {
    let mut acc: u128 = 1;
    for d in v {
        acc = match acc.checked_mul(d) {
            Some(x) => x,
            None => return u128::MAX,
        };
    }
    acc
}

/// The pinned expectation for one op: either an exact typed decline or an exact
/// output shape.
#[derive(Clone, Debug, PartialEq)]
enum Expect {
    Decline(Error),
    Shape(Vec<usize>),
}

fn to_usize_shape(s: &[u128]) -> Vec<usize> {
    s.iter().map(|&d| d as usize).collect()
}

/// `avg_pool`'s stage order: `window_out_shape` → `numel(kernel)` →
/// `numel(out_shape)` → alloc → per-cell `Odometer::new(kernel)`.
fn expect_avg(c: &Case, s1: &Stage1) -> Expect {
    let shape = match s1 {
        Stage1::Decline(e) => return Expect::Decline(e.clone()),
        Stage1::Shape(s) => s,
    };
    if sat_prod(c.kernel.iter().map(|&d| d as u128)) > USIZE_MAX_U128 {
        return Expect::Decline(Error::ShapeOverflow);
    }
    let out_numel = sat_prod(shape.iter().copied());
    if out_numel > USIZE_MAX_U128 {
        return Expect::Decline(Error::ShapeOverflow);
    }
    // The kernel odometer is built per output cell — an over-rank kernel only
    // declines when at least one cell exists.
    if out_numel > 0 && c.axes.len() > MAX_RANK {
        return Expect::Decline(Error::RankExceeded {
            rank: c.axes.len(),
            max: MAX_RANK,
        });
    }
    Expect::Shape(to_usize_shape(shape))
}

/// `max_pool` never computes `numel(kernel)` up front.
fn expect_max(c: &Case, s1: &Stage1) -> Expect {
    let shape = match s1 {
        Stage1::Decline(e) => return Expect::Decline(e.clone()),
        Stage1::Shape(s) => s,
    };
    let out_numel = sat_prod(shape.iter().copied());
    if out_numel > USIZE_MAX_U128 {
        return Expect::Decline(Error::ShapeOverflow);
    }
    if out_numel > 0 && c.axes.len() > MAX_RANK {
        return Expect::Decline(Error::RankExceeded {
            rank: c.axes.len(),
            max: MAX_RANK,
        });
    }
    Expect::Shape(to_usize_shape(shape))
}

/// `im2col` appends the tap axes: `out_rank = rank + n`, checked BEFORE any
/// element-count arithmetic.
fn expect_im2col(c: &Case, s1: &Stage1) -> Expect {
    let shape = match s1 {
        Stage1::Decline(e) => return Expect::Decline(e.clone()),
        Stage1::Shape(s) => s,
    };
    let out_rank = c.shape.len() + c.axes.len();
    if out_rank > MAX_RANK {
        return Expect::Decline(Error::RankExceeded {
            rank: out_rank,
            max: MAX_RANK,
        });
    }
    let total = sat_prod(
        shape
            .iter()
            .copied()
            .chain(c.kernel.iter().map(|&d| d as u128)),
    );
    if total > USIZE_MAX_U128 {
        return Expect::Decline(Error::ShapeOverflow);
    }
    let mut out = to_usize_shape(shape);
    out.extend_from_slice(&c.kernel);
    Expect::Shape(out)
}

/// Whether this case is safe AND cheap enough to actually execute.
///
/// A tuple is affordable when, for every op, either an early typed decline is
/// GUARANTEED (nothing is allocated and no loop runs) or the predicted work fits
/// the budget. This is a harness resource bound, not a claim about the
/// implementation: see the module note.
fn affordable(c: &Case, s1: &Stage1) -> bool {
    let shape = match s1 {
        Stage1::Decline(_) => return true, // declines before any allocation
        Stage1::Shape(s) => s,
    };
    let rank = c.shape.len();
    let n = c.axes.len();
    let out_numel = sat_prod(shape.iter().copied());
    let kernel_numel = sat_prod(c.kernel.iter().map(|&d| d as u128));
    let work = out_numel.saturating_mul(kernel_numel);
    let im_total = out_numel.saturating_mul(kernel_numel);

    // avg_pool / max_pool allocate `out_numel` elements.
    let alloc_ok = out_numel > USIZE_MAX_U128 || out_numel <= OUT_CAP;
    // ... then walk `out_numel · kernel_numel` taps, unless an over-rank kernel
    // or an unrepresentable kernel element count cuts the walk short.
    let walk_ok = n > MAX_RANK || kernel_numel > USIZE_MAX_U128 || work <= WORK_CAP;
    // im2col materializes `out_numel · kernel_numel` elements.
    let im_ok = rank + n > MAX_RANK || im_total > USIZE_MAX_U128 || im_total <= WORK_CAP;

    alloc_ok && walk_ok && im_ok
}

// ---- generators -------------------------------------------------------------

/// A shape whose element count stays under [`INPUT_CAP`] (extents past the cap
/// are forced to 1).
fn gen_shape(rng: &mut Rng, max_rank: usize) -> Vec<usize> {
    let rank = rng.below(max_rank + 1);
    let mut shape = Vec::with_capacity(rank);
    let mut acc = 1usize;
    for _ in 0..rank {
        let d = if rng.chance(10) { 0 } else { 1 + rng.below(5) };
        let next = acc.saturating_mul(d.max(1));
        if next > INPUT_CAP {
            shape.push(1);
        } else {
            acc = next;
            shape.push(d);
        }
    }
    shape
}

fn gen_data(rng: &mut Rng, shape: &[usize]) -> Vec<f64> {
    let mut n = 1usize;
    for &d in shape {
        n = n.saturating_mul(d);
    }
    (0..n).map(|_| tame_value(rng)).collect()
}

/// `n` distinct axes of `0..rank`, in random order (a partial Fisher-Yates).
fn distinct_axes(rng: &mut Rng, rank: usize, n: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..rank).collect();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n.min(rank) {
        let k = rng.below(pool.len());
        out.push(pool.swap_remove(k));
    }
    out
}

/// Structurally plausible: a real pooling/convolution parameterization, with
/// mild in-band noise (5% zero kernel/stride) so the typed-decline paths are
/// still exercised from this mode.
fn gen_plausible(rng: &mut Rng) -> Case {
    let rank = 1 + rng.below(3); // 1..=3
    let mut shape = Vec::with_capacity(rank);
    let mut acc = 1usize;
    for _ in 0..rank {
        let d = if rng.chance(8) { 0 } else { 1 + rng.below(4) };
        acc = acc.saturating_mul(d.max(1));
        shape.push(if acc > INPUT_CAP { 1 } else { d });
    }
    let n = 1 + rng.below(rank); // 1..=rank spatial axes, distinct
    let axes = distinct_axes(rng, rank, n);
    let mut kernel = Vec::with_capacity(n);
    let mut stride = Vec::with_capacity(n);
    let mut padding = Vec::with_capacity(n);
    let mut dilation = Vec::with_capacity(n);
    for _ in 0..n {
        kernel.push(if rng.chance(5) { 0 } else { 1 + rng.below(3) });
        stride.push(if rng.chance(5) { 0 } else { 1 + rng.below(3) });
        padding.push(rng.below(3));
        dilation.push(1 + rng.below(2));
    }
    let data = gen_data(rng, &shape);
    Case {
        shape,
        data,
        axes,
        kernel,
        stride,
        padding,
        dilation,
        count_include_pad: rng.chance(50),
    }
}

/// Every parameter drawn from the wild pools; rank up to `MAX_RANK + 2` axes so
/// the rank guards are hit from both sides.
fn gen_chaos(rng: &mut Rng) -> Case {
    let shape = gen_shape(rng, MAX_RANK);
    let rank = shape.len();
    let n = rng.below(11); // 0..=10 — spans MAX_RANK
    let mut axes = Vec::with_capacity(n);
    for _ in 0..n {
        let r = rng.below(100);
        axes.push(if r < 70 {
            rng.below(rank.max(1))
        } else if r < 85 {
            rank + rng.below(3)
        } else if r < 95 {
            rng.below(16)
        } else {
            usize::MAX
        });
    }
    let mut kernel = Vec::with_capacity(n);
    let mut stride = Vec::with_capacity(n);
    let mut padding = Vec::with_capacity(n);
    let mut dilation = Vec::with_capacity(n);
    for _ in 0..n {
        kernel.push(rng.pick(&WILD_KERNEL));
        stride.push(rng.pick(&WILD_STRIDE));
        padding.push(rng.pick(&WILD_PADDING));
        dilation.push(rng.pick(&WILD_DILATION));
    }
    let data = gen_data(rng, &shape);
    Case {
        shape,
        data,
        axes,
        kernel,
        stride,
        padding,
        dilation,
        count_include_pad: rng.chance(50),
    }
}

/// Truncate or extend one parallel parameter array so its length no longer
/// matches `axes` — the `ShapeMismatch` family.
fn mutate_lengths(rng: &mut Rng, c: &mut Case) {
    let which = rng.below(4);
    let arr = match which {
        0 => &mut c.kernel,
        1 => &mut c.stride,
        2 => &mut c.padding,
        _ => &mut c.dilation,
    };
    if rng.chance(50) && !arr.is_empty() {
        let keep = rng.below(arr.len());
        arr.truncate(keep);
    } else {
        for _ in 0..(1 + rng.below(3)) {
            arr.push(1 + rng.below(3));
        }
    }
}

fn mutate(rng: &mut Rng, c: &mut Case) {
    match rng.below(9) {
        // M0: one parameter slot replaced by a wild value.
        0 | 1 => {
            if c.axes.is_empty() {
                c.axes.push(0);
                c.kernel.push(1);
                c.stride.push(1);
                c.padding.push(0);
                c.dilation.push(1);
            }
            let i = rng.below(c.axes.len());
            match rng.below(4) {
                0 => {
                    if i < c.kernel.len() {
                        c.kernel[i] = rng.pick(&WILD_KERNEL);
                    }
                }
                1 => {
                    if i < c.stride.len() {
                        c.stride[i] = rng.pick(&WILD_STRIDE);
                    }
                }
                2 => {
                    if i < c.padding.len() {
                        c.padding[i] = rng.pick(&WILD_PADDING);
                    }
                }
                _ => {
                    if i < c.dilation.len() {
                        c.dilation[i] = rng.pick(&WILD_DILATION);
                    }
                }
            }
        }
        // M1: parallel-array length mismatch.
        2 => mutate_lengths(rng, c),
        // M2: axis abuse — out of range, usize::MAX, or a duplicate.
        3 => {
            if c.axes.is_empty() {
                c.axes.push(usize::MAX);
                c.kernel.push(1);
                c.stride.push(1);
                c.padding.push(0);
                c.dilation.push(1);
            } else {
                let i = rng.below(c.axes.len());
                c.axes[i] = match rng.below(4) {
                    0 => c.shape.len() + rng.below(3),
                    1 => usize::MAX,
                    2 => rng.below(16),
                    _ => c.axes[rng.below(c.axes.len())], // duplicate
                };
            }
        }
        // M3: append spatial axes until the rank guards trip (n up to 11).
        4 => {
            let extra = 1 + rng.below(9);
            for _ in 0..extra {
                c.axes.push(rng.below(c.shape.len().max(1)));
                c.kernel.push(1 + rng.below(2));
                c.stride.push(1 + rng.below(2));
                c.padding.push(rng.below(2));
                c.dilation.push(1);
            }
        }
        // M4: empty the axis selection (a legal no-op window).
        5 => {
            c.axes.clear();
            c.kernel.clear();
            c.stride.clear();
            c.padding.clear();
            c.dilation.clear();
        }
        // M5: shape pathology — zero extents, rank 0, or rank MAX_RANK.
        6 => {
            c.shape = match rng.below(3) {
                0 => Vec::new(),
                1 => {
                    let mut s = gen_shape(rng, 3);
                    if s.is_empty() {
                        s.push(0);
                    } else {
                        let k = rng.below(s.len());
                        s[k] = 0;
                    }
                    s
                }
                _ => vec![1 + rng.below(2); MAX_RANK],
            };
            c.data = gen_data(rng, &c.shape);
        }
        // M6: data pathology — every element from the special pool.
        7 => {
            for v in c.data.iter_mut() {
                *v = SPECIALS[rng.below(SPECIALS.len())];
            }
        }
        // M7: all four parameter arrays go wild at once.
        _ => {
            for k in c.kernel.iter_mut() {
                *k = rng.pick(&WILD_KERNEL);
            }
            for s in c.stride.iter_mut() {
                *s = rng.pick(&WILD_STRIDE);
            }
            for p in c.padding.iter_mut() {
                *p = rng.pick(&WILD_PADDING);
            }
            for d in c.dilation.iter_mut() {
                *d = rng.pick(&WILD_DILATION);
            }
        }
    }
}

/// Case-kind roll is rng-driven (not loop-index-driven) so the printed seed
/// alone fully reproduces the case: chaos 30%, mutated-plausible 70%.
fn gen_adversarial(rng: &mut Rng) -> Case {
    if rng.chance(30) {
        return gen_chaos(rng);
    }
    let mut c = gen_plausible(rng);
    for _ in 0..(1 + rng.below(3)) {
        mutate(rng, &mut c);
    }
    c
}

fn build_case(seed: u64, mode: Mode) -> Case {
    let mut rng = Rng::new(seed);
    match mode {
        Mode::Plausible => gen_plausible(&mut rng),
        Mode::Adversarial => gen_adversarial(&mut rng),
    }
}

// ---- harness ----------------------------------------------------------------

fn payload_text(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string payload>".to_string()
    }
}

fn dump_case(c: &Case) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "shape        = {:?}", c.shape);
    let _ = writeln!(s, "axes         = {:?}", c.axes);
    let _ = writeln!(s, "kernel       = {:?}", c.kernel);
    let _ = writeln!(s, "stride       = {:?}", c.stride);
    let _ = writeln!(s, "padding      = {:?}", c.padding);
    let _ = writeln!(s, "dilation     = {:?}", c.dilation);
    let _ = writeln!(s, "cip          = {}", c.count_include_pad);
    let head: Vec<f64> = c.data.iter().take(16).copied().collect();
    let _ = writeln!(s, "data[{}] head = {:?}", c.data.len(), head);
    let _ = writeln!(s, "model stage1 = {:?}", model_stage1(c));
    s
}

fn fail(
    seed: u64,
    mode: Mode,
    phase: Phase,
    dtype: &str,
    case: Option<&Case>,
    payload: Box<dyn Any + Send>,
) -> ! {
    let dump = case
        .map(dump_case)
        .unwrap_or_else(|| "<no case: generation itself panicked>".to_string());
    panic!(
        "window_fuzz PANIC: seed={seed:#018x} mode={mode} phase={phase:?} dtype={dtype}\n\
         payload: {payload}\n\
         rerun: KISS_WINDOW_FUZZ_SEED={seed:#x} KISS_WINDOW_FUZZ_MODE={mode} \
         cargo test -p kiss-ref-conformance --test window_fuzz repro_from_env -- --ignored --nocapture\n\
         {dump}",
        mode = mode_name(mode),
        payload = payload_text(&*payload),
    );
}

/// What one executed case observed — the raw material of the vacuity guards.
#[derive(Clone, Copy, Default)]
struct Obs {
    skipped: bool,
    ok_avg: bool,
    ok_max: bool,
    ok_im: bool,
    ok_nonempty: bool,
    ok_padded: bool,
    ok_dilated: bool,
    ok_empty_out: bool,
    ok_multi_axis: bool,
    /// bit 0 ShapeMismatch, 1 AxisOutOfRange, 2 ShapeOverflow, 3 RankExceeded,
    /// 4 anything else.
    declines: u8,
}

fn decline_bit(e: &Error) -> u8 {
    match e {
        Error::ShapeMismatch { .. } => 1 << 0,
        Error::AxisOutOfRange { .. } => 1 << 1,
        Error::ShapeOverflow => 1 << 2,
        Error::RankExceeded { .. } => 1 << 3,
        _ => 1 << 4,
    }
}

fn tensor_of<T: ScalarFloat>(c: &Case) -> Result<Tensor<T>, Error> {
    let data: Vec<T> = c.data.iter().map(|&v| T::from_f64(v)).collect();
    Tensor::from_vec(data, &c.shape)
}

/// Check one op's result against its pinned expectation. `Ok`/`Err` both PASS
/// the never-panic contract; the model then pins WHICH of the two, and the exact
/// value. Returns whether it was `Ok`, plus the decline bit.
fn check<T: ScalarFloat>(
    seed: u64,
    mode: Mode,
    op: &str,
    dtype: &str,
    got: &Result<Tensor<T>, Error>,
    want: &Expect,
) -> (bool, u8) {
    match (got, want) {
        (Ok(t), Expect::Shape(s)) => {
            assert_eq!(
                t.shape(),
                s.as_slice(),
                "{op}/{dtype}: output shape != u128 model (seed={seed:#018x} mode={})",
                mode_name(mode)
            );
            (true, 0)
        }
        (Err(e), Expect::Decline(w)) => {
            assert_eq!(
                e,
                w,
                "{op}/{dtype}: declined with the wrong error (seed={seed:#018x} mode={})",
                mode_name(mode)
            );
            (false, decline_bit(e))
        }
        (Ok(t), Expect::Decline(w)) => panic!(
            "{op}/{dtype}: returned Ok(shape {:?}) but the u128 model requires Err({w:?}) \
             (seed={seed:#018x} mode={}) — a wrapped/overflowed extent",
            t.shape(),
            mode_name(mode)
        ),
        // A `ShapeOverflow` decline is a legitimate never-panic outcome the u128
        // model CANNOT predict: the model computes the mathematically-true shape
        // in `u128`, so it cannot see the kernel's `usize` intermediate-overflow
        // declines (e.g. `2*padding` or `dilation*(k-1)` overflowing `usize`
        // while the final output dim is small), nor the `try_reserve` allocation
        // limit on a valid-but-enormous shape. Both are typed declines, not wrong
        // answers — accept them. (The anchor goldens + cross-check oracle guard
        // against a *spurious* overflow decline on a small, genuinely-Ok case.)
        (Err(Error::ShapeOverflow), Expect::Shape(_)) => {
            (false, decline_bit(&Error::ShapeOverflow))
        }
        (Err(e), Expect::Shape(s)) => panic!(
            "{op}/{dtype}: declined Err({e:?}) but the u128 model requires Ok(shape {s:?}) \
             (seed={seed:#018x} mode={})",
            mode_name(mode)
        ),
    }
}

/// Build + evaluate one case on lane `T`. `Ok` and `Err` both PASS; only a panic
/// (or a divergence from the `u128` model) FAILS.
fn run_one<T: ScalarFloat>(seed: u64, mode: Mode, dtype: &str) -> Obs {
    let mut obs = Obs::default();
    // Phase 1: generation, wrapped so even a generator bug fails WITH the seed.
    let case = match catch_unwind(AssertUnwindSafe(|| build_case(seed, mode))) {
        Ok(c) => c,
        Err(p) => fail(seed, mode, Phase::Generate, dtype, None, p),
    };
    let s1 = model_stage1(&case);
    if !affordable(&case, &s1) {
        obs.skipped = true;
        return obs;
    }
    // A rank > MAX_RANK input cannot be built at all (Tensor::from_vec gates it),
    // so the window fns can never see one; nothing to fuzz there.
    let x = match tensor_of::<T>(&case) {
        Ok(t) => t,
        Err(_) => return obs,
    };
    let v = x.view();

    let want_avg = expect_avg(&case, &s1);
    let r = catch_unwind(AssertUnwindSafe(|| {
        avg_pool::<T>(
            &v,
            &case.axes,
            &case.kernel,
            &case.stride,
            &case.padding,
            &case.dilation,
            case.count_include_pad,
        )
    }));
    match r {
        Err(p) => fail(seed, mode, Phase::AvgPool, dtype, Some(&case), p),
        Ok(res) => {
            let (ok, bit) = check(seed, mode, "avg_pool", dtype, &res, &want_avg);
            obs.ok_avg = ok;
            obs.declines |= bit;
        }
    }

    let want_max = expect_max(&case, &s1);
    let r = catch_unwind(AssertUnwindSafe(|| {
        max_pool::<T>(
            &v,
            &case.axes,
            &case.kernel,
            &case.stride,
            &case.padding,
            &case.dilation,
        )
    }));
    match r {
        Err(p) => fail(seed, mode, Phase::MaxPool, dtype, Some(&case), p),
        Ok(res) => {
            let (ok, bit) = check(seed, mode, "max_pool", dtype, &res, &want_max);
            obs.ok_max = ok;
            obs.declines |= bit;
        }
    }

    let want_im = expect_im2col(&case, &s1);
    let r = catch_unwind(AssertUnwindSafe(|| {
        im2col::<T>(
            &v,
            &case.axes,
            &case.kernel,
            &case.stride,
            &case.padding,
            &case.dilation,
        )
    }));
    match r {
        Err(p) => fail(seed, mode, Phase::Im2col, dtype, Some(&case), p),
        Ok(res) => {
            let (ok, bit) = check(seed, mode, "im2col", dtype, &res, &want_im);
            obs.ok_im = ok;
            obs.declines |= bit;
        }
    }

    // Coverage bookkeeping, only from cases that actually produced a result.
    if let Expect::Shape(s) = &want_max {
        let numel: u128 = sat_prod(s.iter().map(|&d| d as u128));
        if obs.ok_max {
            if numel > 0 {
                obs.ok_nonempty = true;
                // padding > 0 with a non-empty output ⇒ the o=0,t=0 tap sits at
                // −pad, which is out of bounds for every extent ⇒ the OOB branch
                // provably executed.
                if case.padding.iter().any(|&p| p > 0) {
                    obs.ok_padded = true;
                }
                if case.dilation.iter().any(|&d| d > 1) {
                    obs.ok_dilated = true;
                }
                if case.axes.len() >= 2 {
                    obs.ok_multi_axis = true;
                }
            } else {
                obs.ok_empty_out = true;
            }
        }
    }
    obs
}

// ---- fuzz tests -------------------------------------------------------------

struct Tally {
    n: u64,
    skipped: u64,
    ok_avg: u64,
    ok_max: u64,
    ok_im: u64,
    ok_nonempty: u64,
    ok_padded: u64,
    ok_dilated: u64,
    ok_empty_out: u64,
    ok_multi_axis: u64,
    declines: u8,
}

impl Tally {
    fn new() -> Self {
        Tally {
            n: 0,
            skipped: 0,
            ok_avg: 0,
            ok_max: 0,
            ok_im: 0,
            ok_nonempty: 0,
            ok_padded: 0,
            ok_dilated: 0,
            ok_empty_out: 0,
            ok_multi_axis: 0,
            declines: 0,
        }
    }
    fn add(&mut self, o: Obs) {
        self.n += 1;
        self.skipped += u64::from(o.skipped);
        self.ok_avg += u64::from(o.ok_avg);
        self.ok_max += u64::from(o.ok_max);
        self.ok_im += u64::from(o.ok_im);
        self.ok_nonempty += u64::from(o.ok_nonempty);
        self.ok_padded += u64::from(o.ok_padded);
        self.ok_dilated += u64::from(o.ok_dilated);
        self.ok_empty_out += u64::from(o.ok_empty_out);
        self.ok_multi_axis += u64::from(o.ok_multi_axis);
        self.declines |= o.declines;
    }
}

#[test]
fn fuzz_plausible_f64() {
    let mut t = Tally::new();
    for i in 0..PLAUSIBLE_ITERS {
        t.add(run_one::<f64>(
            case_seed(SEED_PLAUSIBLE, i),
            Mode::Plausible,
            "f64",
        ));
    }
    // Generator-health (vacuity) guards, NOT never-panic violations: if these
    // trip, the plausible generator has rotted into all-decline — a fuzzer that
    // only ever produces typed declines proves nothing. Tune the generator (or
    // reconcile with a legitimately-stricter implementation); do not chase a
    // panic.
    let floor = PLAUSIBLE_ITERS / 3;
    assert!(
        t.ok_avg >= floor,
        "vacuity: only {}/{} avg_pool cases returned Ok (floor {floor})",
        t.ok_avg,
        t.n
    );
    assert!(
        t.ok_max >= floor,
        "vacuity: only {}/{} max_pool cases returned Ok (floor {floor})",
        t.ok_max,
        t.n
    );
    assert!(
        t.ok_im >= floor,
        "vacuity: only {}/{} im2col cases returned Ok (floor {floor})",
        t.ok_im,
        t.n
    );
    assert!(
        t.ok_nonempty >= floor,
        "vacuity: too few Ok cases with a NON-EMPTY output"
    );
    assert!(
        t.ok_padded > 0,
        "vacuity: no Ok case provably executed the out-of-bounds tap branch"
    );
    assert!(t.ok_dilated > 0, "vacuity: no Ok case used dilation > 1");
    assert!(
        t.ok_empty_out > 0,
        "vacuity: no Ok case produced an empty (zero-extent) output"
    );
    assert!(
        t.ok_multi_axis > 0,
        "vacuity: no Ok case pooled over 2+ spatial axes"
    );
    assert_eq!(
        t.skipped, 0,
        "the plausible generator must be affordable by construction"
    );
    // The 5% zero-kernel/zero-stride noise must actually reach the decline path.
    assert!(
        t.declines & 1 != 0,
        "vacuity: the plausible mode never produced a ShapeMismatch decline"
    );
}

#[test]
fn fuzz_adversarial_f64() {
    let mut t = Tally::new();
    for i in 0..ADVERSARIAL_ITERS {
        t.add(run_one::<f64>(
            case_seed(SEED_ADVERSARIAL, i),
            Mode::Adversarial,
            "f64",
        ));
    }
    // Every decline family must be reachable from the adversarial generator,
    // otherwise the wild pools are not doing what their comments claim.
    for (bit, name) in [
        (
            0u8,
            "ShapeMismatch (zero kernel/stride, mismatched parallel arrays)",
        ),
        (1, "AxisOutOfRange"),
        (2, "ShapeOverflow (the padding/dilation overflow guards)"),
        (
            3,
            "RankExceeded (over-rank axis lists / im2col tap-axis blowup)",
        ),
    ] {
        assert!(
            t.declines & (1 << bit) != 0,
            "vacuity: the adversarial generator never produced a {name} decline"
        );
    }
    assert!(
        t.ok_avg > 0 && t.ok_max > 0 && t.ok_im > 0,
        "vacuity: adversarial mode is all-decline"
    );
    // The budget guard must not be silently eating the corpus.
    assert!(
        t.skipped * 4 < t.n,
        "budget guard skipped {}/{} adversarial cases (>25%) — the generator is emitting \
         unaffordable tuples; retune the wild pools, not the cap",
        t.skipped,
        t.n
    );
}

#[test]
fn fuzz_plausible_f32() {
    for i in 0..PLAUSIBLE_ITERS {
        run_one::<f32>(case_seed(SEED_PLAUSIBLE, i), Mode::Plausible, "f32");
    }
}

#[test]
fn fuzz_adversarial_f32() {
    for i in 0..ADVERSARIAL_ITERS {
        run_one::<f32>(case_seed(SEED_ADVERSARIAL, i), Mode::Adversarial, "f32");
    }
}

/// The narrow lanes go through the promote-to-f32 macro path (`scalar.rs`
/// `impl_scalar_float_via_f32!`) — same index arithmetic, different rounding.
#[test]
fn fuzz_narrow_lanes() {
    for i in 0..NARROW_ITERS {
        let s = case_seed(SEED_NARROW, i);
        run_one::<f16>(s, Mode::Plausible, "f16");
        run_one::<E4m3>(s, Mode::Plausible, "e4m3");
        let a = case_seed(SEED_NARROW, i + NARROW_ITERS);
        run_one::<f16>(a, Mode::Adversarial, "f16");
        run_one::<E4m3>(a, Mode::Adversarial, "e4m3");
    }
}

#[test]
fn determinism_spot_check() {
    for i in 0..DET_ITERS {
        let seed = case_seed(SEED_DET, i);
        let case = build_case(seed, Mode::Adversarial);
        let s1 = model_stage1(&case);
        if !affordable(&case, &s1) {
            continue;
        }
        let x: Tensor<f64> = match tensor_of(&case) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let v = x.view();
        let go = || {
            (
                avg_pool::<f64>(
                    &v,
                    &case.axes,
                    &case.kernel,
                    &case.stride,
                    &case.padding,
                    &case.dilation,
                    case.count_include_pad,
                ),
                max_pool::<f64>(
                    &v,
                    &case.axes,
                    &case.kernel,
                    &case.stride,
                    &case.padding,
                    &case.dilation,
                ),
                im2col::<f64>(
                    &v,
                    &case.axes,
                    &case.kernel,
                    &case.stride,
                    &case.padding,
                    &case.dilation,
                ),
            )
        };
        let (a1, m1, i1) = go();
        let (a2, m2, i2) = go();
        for (r1, r2, name) in [
            (&a1, &a2, "avg_pool"),
            (&m1, &m2, "max_pool"),
            (&i1, &i2, "im2col"),
        ] {
            match (r1, r2) {
                (Err(e1), Err(e2)) => {
                    assert_eq!(e1, e2, "{name}: Err values differ (seed={seed:#018x})")
                }
                (Ok(t1), Ok(t2)) => {
                    assert_eq!(
                        t1.shape(),
                        t2.shape(),
                        "{name}: shapes differ (seed={seed:#018x})"
                    );
                    for (k, (p, q)) in t1.as_slice().iter().zip(t2.as_slice()).enumerate() {
                        assert_eq!(
                            p.to_bits(),
                            q.to_bits(),
                            "{name}: element {k} bits differ (seed={seed:#018x})"
                        );
                    }
                }
                _ => panic!("{name}: Ok/Err flip between identical calls (seed={seed:#018x})"),
            }
        }
    }
}

// ---- cross-op oracles -------------------------------------------------------

/// Bitwise equality with NaN treated as equal to NaN (payloads are not part of
/// any window-family guarantee).
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Row-major element count (test-local; small shapes only).
fn numel_of(shape: &[usize]) -> usize {
    shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
}

#[test]
fn cross_check_avg_and_max_against_im2col() {
    let mut checked_avg = 0usize;
    let mut checked_max = 0usize;
    let mut checked_cip = 0usize;
    for i in 0..CROSS_ITERS {
        let seed = case_seed(SEED_CROSS, i);
        let mut case = build_case(seed, Mode::Plausible);
        // im2col must be shape-legal for the oracle to exist.
        if case.shape.len() + case.axes.len() > MAX_RANK {
            continue;
        }
        case.count_include_pad = true;
        let s1 = model_stage1(&case);
        if !affordable(&case, &s1) {
            continue;
        }
        let x: Tensor<f64> = match tensor_of(&case) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let v = x.view();
        let cols = match im2col::<f64>(
            &v,
            &case.axes,
            &case.kernel,
            &case.stride,
            &case.padding,
            &case.dilation,
        ) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let taps = numel_of(&case.kernel);
        let patches = cols.as_slice().len().checked_div(taps).unwrap_or(0);

        // ORACLE 1 — count_include_pad: avg_pool == (Σ taps of im2col) / |kernel|.
        // im2col zero-fills an out-of-bounds tap and avg_pool skips it; adding
        // +0.0 in the same tap order is exact for every finite/inf/NaN operand,
        // so the two must agree BIT FOR BIT, not merely to a tolerance.
        if let Ok(avg) = avg_pool::<f64>(
            &v,
            &case.axes,
            &case.kernel,
            &case.stride,
            &case.padding,
            &case.dilation,
            true,
        ) {
            assert_eq!(
                avg.as_slice().len(),
                patches,
                "cross: patch count (seed={seed:#018x})"
            );
            let divisor = f64::from_f64(taps as f64);
            for p in 0..patches {
                let mut acc = 0.0f64;
                for t in 0..taps {
                    acc = eval_op(Op::Add, &[acc, cols.as_slice()[p * taps + t]]).expect("add");
                }
                let want = eval_op(Op::Div, &[acc, divisor]).expect("div");
                let got = avg.as_slice()[p];
                assert!(
                    same_bits(got, want),
                    "cross(avg==mean of im2col taps): patch {p} got {got:?} want {want:?} \
                     (seed={seed:#018x})\n{}",
                    dump_case(&case)
                );
            }
            if patches > 0 {
                checked_avg += 1;
            }
        }

        // ORACLE 2 — with NO padding every tap is in bounds, so max_pool must
        // equal the tap-max of im2col (with padding they legitimately differ:
        // im2col's zero fill is a real 0.0, max_pool skips the tap).
        if case.padding.iter().all(|&p| p == 0) {
            if let Ok(mx) = max_pool::<f64>(
                &v,
                &case.axes,
                &case.kernel,
                &case.stride,
                &case.padding,
                &case.dilation,
            ) {
                for p in 0..patches {
                    let mut acc = f64::NEG_INFINITY;
                    for t in 0..taps {
                        acc = eval_op(Op::MaxProp, &[acc, cols.as_slice()[p * taps + t]])
                            .expect("max");
                    }
                    let got = mx.as_slice()[p];
                    assert!(
                        same_bits(got, acc),
                        "cross(max==max of im2col taps): patch {p} got {got:?} want {acc:?} \
                         (seed={seed:#018x})\n{}",
                        dump_case(&case)
                    );
                }
                if patches > 0 {
                    checked_max += 1;
                }
            }

            // ORACLE 3 — with no padding, `count_include_pad` cannot matter:
            // every window's in-bounds tap count IS the full window size.
            let a = avg_pool::<f64>(
                &v,
                &case.axes,
                &case.kernel,
                &case.stride,
                &case.padding,
                &case.dilation,
                true,
            );
            let b = avg_pool::<f64>(
                &v,
                &case.axes,
                &case.kernel,
                &case.stride,
                &case.padding,
                &case.dilation,
                false,
            );
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(
                    a.shape(),
                    b.shape(),
                    "cross(cip): shapes (seed={seed:#018x})"
                );
                for (k, (p, q)) in a.as_slice().iter().zip(b.as_slice()).enumerate() {
                    assert!(
                        same_bits(*p, *q),
                        "cross(cip irrelevant without padding): element {k} {p:?} vs {q:?} \
                         (seed={seed:#018x})"
                    );
                }
                if !a.as_slice().is_empty() {
                    checked_cip += 1;
                }
            }
        }
    }
    // Vacuity: an oracle that never ran proves nothing.
    assert!(
        checked_avg >= 50,
        "vacuity: avg/im2col oracle ran only {checked_avg} times"
    );
    assert!(
        checked_max >= 20,
        "vacuity: max/im2col oracle ran only {checked_max} times"
    );
    assert!(
        checked_cip >= 20,
        "vacuity: count_include_pad oracle ran only {checked_cip} times"
    );
}

// ---- hand-computed goldens --------------------------------------------------

fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_vec(data.to_vec(), shape).expect("golden tensor")
}

#[test]
fn anchor_avg_pool_dilated_1d() {
    // x = [1,2,3,4,5], k=2 s=1 p=0 dil=2.
    // eff = 2·(2−1)+1 = 3; padded = 5; out = (5−3)/1 + 1 = 3.
    // o=0: taps at 0,2 → (1+3)/2 = 2 ; o=1: 1,3 → (2+4)/2 = 3 ;
    // o=2: taps at 2,4 → (3+5)/2 = 4. Exact in f32 and f64.
    let x = t64(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let y = avg_pool(&x.view(), &[0], &[2], &[1], &[0], &[2], false).expect("avg");
    assert_eq!(y.shape(), &[3]);
    assert_eq!(y.as_slice(), &[2.0, 3.0, 4.0]);

    let x32 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).expect("t32");
    let y32 = avg_pool(&x32.view(), &[0], &[2], &[1], &[0], &[2], true).expect("avg32");
    assert_eq!(y32.as_slice(), &[2.0f32, 3.0, 4.0]);
}

#[test]
fn anchor_max_pool_2d() {
    // 3x3 = [[1,9,2],[8,3,7],[4,6,5]], k=2x2 s=1 p=0 dil=1 → out 2x2:
    // (0,0) max{1,9,8,3}=9 ; (0,1) max{9,2,3,7}=9 ;
    // (1,0) max{8,3,4,6}=8 ; (1,1) max{3,7,6,5}=7.
    let x = t64(&[1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0], &[3, 3]);
    let y = max_pool(&x.view(), &[0, 1], &[2, 2], &[1, 1], &[0, 0], &[1, 1]).expect("max");
    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(y.as_slice(), &[9.0, 9.0, 8.0, 7.0]);
}

#[test]
fn anchor_im2col_2d_stride2_pad1() {
    // 2x2 = [[1,2],[3,4]], k=2x2 s=2 p=1 dil=1.
    // eff = 2, padded = 2+2 = 4, out = (4−2)/2 + 1 = 2 per axis → [2,2,2,2].
    // source pos = o·2 + t − 1 per axis; out of bounds → 0.
    //  patch(0,0): {(−1,−1),(−1,0),(0,−1),(0,0)} → 0,0,0,1
    //  patch(0,1): {(−1,1),(−1,2),(0,1),(0,2)}   → 0,0,2,0
    //  patch(1,0): {(1,−1),(1,0),(2,−1),(2,0)}   → 0,3,0,0
    //  patch(1,1): {(1,1),(1,2),(2,1),(2,2)}     → 4,0,0,0
    let x = t64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = im2col(&x.view(), &[0, 1], &[2, 2], &[2, 2], &[1, 1], &[1, 1]).expect("im2col");
    assert_eq!(y.shape(), &[2, 2, 2, 2]);
    assert_eq!(
        y.as_slice(),
        &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn anchor_avg_pool_wholly_padded_window() {
    // x = [5], k=2 s=1 p=2 dil=1 → eff = 2, padded = 1+4 = 5, out = 4.
    // window source positions o+t−2 for t∈{0,1}:
    //  o=0 → {−2,−1}: no valid tap ; o=1 → {−1,0}: one ;
    //  o=2 → { 0, 1}: one (1 is out of range) ; o=3 → {1,2}: none.
    // count_include_pad=true  → divisor 2 always → [0/2, 5/2, 5/2, 0/2].
    // count_include_pad=false → divisor = valid count → [0/0, 5/1, 5/1, 0/0],
    // and 0/0 is NaN: the empty-window divisor is NOT special-cased.
    let x = t64(&[5.0], &[1]);
    let inc = avg_pool(&x.view(), &[0], &[2], &[1], &[2], &[1], true).expect("cip");
    assert_eq!(inc.shape(), &[4]);
    assert_eq!(inc.as_slice(), &[0.0, 2.5, 2.5, 0.0]);

    let exc = avg_pool(&x.view(), &[0], &[2], &[1], &[2], &[1], false).expect("nocip");
    assert!(exc.as_slice()[0].is_nan());
    assert_eq!(exc.as_slice()[1], 5.0);
    assert_eq!(exc.as_slice()[2], 5.0);
    assert!(exc.as_slice()[3].is_nan());
}

#[test]
fn anchor_max_pool_wholly_padded_window_is_identity() {
    // Same geometry: a window with no in-bounds tap yields the max identity −inf
    // (documented in window.rs), NOT an error.
    let x = t64(&[5.0], &[1]);
    let y = max_pool(&x.view(), &[0], &[2], &[1], &[2], &[1]).expect("max");
    assert_eq!(y.shape(), &[4]);
    assert_eq!(
        y.as_slice(),
        &[f64::NEG_INFINITY, 5.0, 5.0, f64::NEG_INFINITY]
    );
}

#[test]
fn anchor_zero_extent_output_is_ok_not_error() {
    // eff = 5 > padded = 2 ⇒ pooled extent 0 ⇒ an EMPTY tensor, not a decline.
    let x = t64(&[1.0, 2.0], &[2]);
    let y = avg_pool(&x.view(), &[0], &[5], &[1], &[0], &[1], true).expect("avg");
    assert_eq!(y.shape(), &[0]);
    assert!(y.as_slice().is_empty());
    let c = im2col(&x.view(), &[0], &[5], &[1], &[0], &[1]).expect("im2col");
    assert_eq!(c.shape(), &[0, 5]);
    assert!(c.as_slice().is_empty());
    // A kernel of usize::MAX is the same story — an empty output, not overflow.
    let big = im2col(&x.view(), &[0], &[usize::MAX], &[1], &[0], &[1]).expect("im2col big k");
    assert_eq!(big.shape(), &[0, usize::MAX]);
    // A huge stride only shrinks the output.
    let s = im2col(&x.view(), &[0], &[1], &[usize::MAX], &[0], &[1]).expect("im2col big s");
    assert_eq!(s.shape(), &[1, 1]);
    assert_eq!(s.as_slice(), &[1.0]);
}

#[test]
fn anchor_duplicate_axes_last_write_wins() {
    // axes=[0,0] with kernel=[1,2]: both entries write the SAME output axis and
    // the SAME src coordinate, so entry 1 (k=2) fixes the extent — out = [1,2] —
    // and the tap position comes from the last entry too.
    // out(0,j) = mean(x[0,j], x[1,j]) → [(1+3)/2, (2+4)/2] = [2,3].
    let x = t64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = avg_pool(&x.view(), &[0, 0], &[1, 2], &[1, 1], &[0, 0], &[1, 1], true).expect("avg");
    assert_eq!(y.shape(), &[1, 2]);
    assert_eq!(y.as_slice(), &[2.0, 3.0]);
    let c = im2col(&x.view(), &[0, 0], &[1, 2], &[1, 1], &[0, 0], &[1, 1]).expect("im2col");
    assert_eq!(c.shape(), &[1, 2, 1, 2]);
    assert_eq!(c.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn anchor_empty_axis_selection_is_identity() {
    // No spatial axes ⇒ no windowing ⇒ the input passes through unchanged.
    let x = t64(&[7.0], &[]);
    let y = avg_pool(&x.view(), &[], &[], &[], &[], &[], false).expect("avg");
    assert_eq!(y.shape(), &[] as &[usize]);
    assert_eq!(y.as_slice(), &[7.0]);
    let c = im2col(&x.view(), &[], &[], &[], &[], &[]).expect("im2col");
    assert_eq!(c.shape(), &[] as &[usize]);
    assert_eq!(c.as_slice(), &[7.0]);
}

#[test]
fn anchor_typed_declines() {
    let x = t64(&[1.0, 2.0], &[2]);
    // kernel 0 / stride 0 — the shared "degenerate window" decline.
    assert_eq!(
        max_pool(&x.view(), &[0], &[0], &[1], &[0], &[1]).unwrap_err(),
        Error::ShapeMismatch {
            expected: 1,
            got: 0
        }
    );
    assert_eq!(
        max_pool(&x.view(), &[0], &[1], &[0], &[0], &[1]).unwrap_err(),
        Error::ShapeMismatch {
            expected: 1,
            got: 0
        }
    );
    // parallel-array length mismatch (reported against kernel.len()).
    assert_eq!(
        avg_pool(&x.view(), &[0], &[1, 1], &[1], &[0], &[1], false).unwrap_err(),
        Error::ShapeMismatch {
            expected: 1,
            got: 2
        }
    );
    // axis out of range.
    assert_eq!(
        im2col(&x.view(), &[3], &[1], &[1], &[0], &[1]).unwrap_err(),
        Error::AxisOutOfRange { axis: 3, rank: 1 }
    );
    // The `2·p` doubling and the `dil·(k−1)+1` effective window are both checked
    // (the `2*p` case is the standing regression from the prior adversarial
    // review — kept here so this suite owns it too).
    assert_eq!(
        max_pool(&x.view(), &[0], &[1], &[1], &[usize::MAX], &[1]).unwrap_err(),
        Error::ShapeOverflow
    );
    assert_eq!(
        max_pool(&x.view(), &[0], &[1], &[1], &[usize::MAX / 2 + 1], &[1]).unwrap_err(),
        Error::ShapeOverflow
    );
    assert_eq!(
        max_pool(&x.view(), &[0], &[2], &[1], &[0], &[usize::MAX]).unwrap_err(),
        Error::ShapeOverflow
    );
    // Over-rank tap odometer (9 spatial entries on a rank-1 input).
    let nine = vec![0usize; 9];
    let ones = vec![1usize; 9];
    let zeros = vec![0usize; 9];
    assert_eq!(
        avg_pool(&x.view(), &nine, &ones, &ones, &zeros, &ones, false).unwrap_err(),
        Error::RankExceeded {
            rank: 9,
            max: MAX_RANK
        }
    );
    assert_eq!(
        max_pool(&x.view(), &nine, &ones, &ones, &zeros, &ones).unwrap_err(),
        Error::RankExceeded {
            rank: 9,
            max: MAX_RANK
        }
    );
    // im2col appends the tap axes, so it trips one axis earlier: 1 + 9 = 10.
    assert_eq!(
        im2col(&x.view(), &nine, &ones, &ones, &zeros, &ones).unwrap_err(),
        Error::RankExceeded {
            rank: 10,
            max: MAX_RANK
        }
    );
    // rank 3 + 6 tap axes = 9 > MAX_RANK.
    let r3 = t64(&[1.0; 8], &[2, 2, 2]);
    assert_eq!(
        im2col(
            &r3.view(),
            &[0, 1, 2, 0, 1, 2],
            &[1; 6],
            &[1; 6],
            &[0; 6],
            &[1; 6]
        )
        .unwrap_err(),
        Error::RankExceeded {
            rank: 9,
            max: MAX_RANK
        }
    );
}

// ---- known-bug regression tests ---------------------------------------------
//
// FOUND BY THIS SUITE (2026-07-23). All three window ops size their output
// buffer with `Vec::with_capacity(count)` where `count` comes from the
// PARAMETERS, not from any materialized tensor:
//     window.rs:107 (avg_pool), :153 (max_pool), :209 (im2col)
// `numel` only guards `usize` multiplication overflow, so a padding that is
// pathological but does NOT overflow `usize` yields a `count` the allocator
// cannot serve, and `Vec::with_capacity` PANICS ("capacity overflow") or
// ABORTS (allocation failure) — both violate the never-panic contract.
//
// FIX (the idiom already used by `Tensor::full`, tensor.rs:105):
//     let mut data: Vec<T> = Vec::new();
//     data.try_reserve(count).map_err(|_| Error::ShapeOverflow)?;
// `try_reserve` returns `Err` for BOTH the capacity-overflow and the
// allocation-failure case, turning both into the typed decline the contract
// promises.

/// `count · size_of::<T>() > isize::MAX` ⇒ `Vec::with_capacity` panics with
/// "capacity overflow" before it even asks the allocator. This is a CATCHABLE
/// panic, so this test can REPORT it cleanly rather than killing the process.
///
/// Geometry: `in = 2`, `k = s = dil = 1`, `p = usize::MAX/4`.
/// `eff = 1`; `2p = 2^63 − 2` does not overflow; `padded = 2^63`; the pooled
/// extent is `2^63`, which is a perfectly good `usize` — and then
/// `Vec::<f64>::with_capacity(2^63)` needs `2^66` bytes.
///
/// Regression guard for the `try_reserve` fix (landed 2026-07-23, found by this
/// suite): the pooled extent 2^63 is a valid `usize`, so `numel` passes; the fix
/// makes the buffer allocation a typed [`Error::ShapeOverflow`] decline instead
/// of a `Vec::with_capacity` capacity-overflow panic.
#[test]
fn never_panic_huge_padding_capacity_overflow() {
    let x = t64(&[1.0, 2.0], &[2]);
    let p = usize::MAX / 4;
    let r = catch_unwind(AssertUnwindSafe(|| {
        max_pool(&x.view(), &[0], &[1], &[1], &[p], &[1]).map(|t| t.shape().to_vec())
    }));
    match r {
        Ok(Ok(shape)) => panic!("expected a typed decline, got Ok(shape {shape:?})"),
        Ok(Err(_)) => {} // typed decline: contract honored
        Err(payload) => panic!(
            "NEVER-PANIC VIOLATION: max_pool(p={p:#x}) panicked: {}\n\
             The pooled extent 2^63 is a valid usize, so `numel` passes and \
             `Vec::with_capacity` (window.rs:153) asks for 2^66 bytes.\n\
             FIX: replace `Vec::with_capacity(count)` in avg_pool/max_pool/im2col \
             with `Vec::new()` + `try_reserve(count).map_err(|_| Error::ShapeOverflow)?`, \
             as `Tensor::full` (tensor.rs:105) already does.",
            payload_text(&*payload)
        ),
    }
}

/// The same defect one magnitude band lower, where the allocator is actually
/// consulted: `p = 2^40` asks for ~17.6 TB, which fails and ABORTS the process
/// (`memory allocation of 17592186044432 bytes failed`). An abort is not an
/// unwind, so `catch_unwind` cannot contain it and this test would take the
/// whole test binary down with it.
///
/// Regression guard: with the `try_reserve` fix, `try_reserve(2^41)` fails
/// WITHOUT asking the allocator for 17.6 TB, so this returns a typed
/// [`Error::ShapeOverflow`] decline instead of aborting the process — an
/// ordinary fast test now (found by this suite, fixed 2026-07-23).
#[test]
fn never_panic_huge_padding_allocation_failure() {
    let x = t64(&[1.0, 2.0], &[2]);
    let p = 1usize << 40;
    let r = catch_unwind(AssertUnwindSafe(|| {
        max_pool(&x.view(), &[0], &[1], &[1], &[p], &[1]).map(|t| t.shape().to_vec())
    }));
    match r {
        Ok(Ok(shape)) => panic!("expected a typed decline, got Ok(shape {shape:?})"),
        Ok(Err(_)) => {}
        Err(_) => panic!("NEVER-PANIC VIOLATION: max_pool(p={p:#x}) panicked instead of declining"),
    }
}

// ---- soak + repro -----------------------------------------------------------

/// On-demand soak: 40k plausible + 40k adversarial f64 cases.
/// Run locally: cargo test -p kiss-ref-conformance --test window_fuzz fuzz_soak -- --ignored
#[test]
#[ignore]
fn fuzz_soak() {
    for i in 0..SOAK_ITERS_PER_MODE {
        run_one::<f64>(case_seed(SEED_SOAK, i), Mode::Plausible, "f64");
    }
    for i in SOAK_ITERS_PER_MODE..(2 * SOAK_ITERS_PER_MODE) {
        run_one::<f64>(case_seed(SEED_SOAK, i), Mode::Adversarial, "f64");
    }
}

fn parse_seed(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

/// Rerun exactly one case from `KISS_WINDOW_FUZZ_SEED` + `KISS_WINDOW_FUZZ_MODE`
/// (the values a failure message prints). Missing vars → usage text, no failure.
#[test]
#[ignore]
fn repro_from_env() {
    let (seed_s, mode_s) = match (
        std::env::var("KISS_WINDOW_FUZZ_SEED").ok(),
        std::env::var("KISS_WINDOW_FUZZ_MODE").ok(),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            println!(
                "usage: KISS_WINDOW_FUZZ_SEED=0x<hex|decimal> \
                 KISS_WINDOW_FUZZ_MODE=<plausible|adversarial> \
                 cargo test -p kiss-ref-conformance --test window_fuzz repro_from_env -- --ignored --nocapture"
            );
            return;
        }
    };
    let seed = parse_seed(&seed_s)
        .unwrap_or_else(|| panic!("bad KISS_WINDOW_FUZZ_SEED {seed_s:?} (decimal or 0x-hex)"));
    let mode = match mode_s.as_str() {
        "plausible" => Mode::Plausible,
        "adversarial" => Mode::Adversarial,
        _ => panic!("bad KISS_WINDOW_FUZZ_MODE {mode_s:?} (plausible|adversarial)"),
    };
    let case = build_case(seed, mode);
    let s1 = model_stage1(&case);
    println!(
        "repro seed={seed:#018x} mode={}\n{}affordable = {}",
        mode_name(mode),
        dump_case(&case),
        affordable(&case, &s1)
    );
    println!("expect avg    = {:?}", expect_avg(&case, &s1));
    println!("expect max    = {:?}", expect_max(&case, &s1));
    println!("expect im2col = {:?}", expect_im2col(&case, &s1));
    let obs = run_one::<f64>(seed, mode, "f64");
    println!(
        "result: skipped={} ok_avg={} ok_max={} ok_im={}",
        obs.skipped, obs.ok_avg, obs.ok_max, obs.ok_im
    );
}
