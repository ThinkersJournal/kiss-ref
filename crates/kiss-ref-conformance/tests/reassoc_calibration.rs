//! PRELIMINARY reassociation-tolerance calibration for RFC #92 direction b
//! (C4 / KISS-CONTRACT-6.8-0012).
//!
//! Band NOW NORMATIVE (KISS-CONTRACT-6.8-0012, merged in KISS PR #96 2026-07-24
//! @ 46e69a8 — incl. the absolute-band usage note this calibration surfaced):
//! for a `(compute S, accumulator A)` reduction / scan / contraction cell,
//!
//!     tol = k(S,A) * N * eps_A * |max partial sum|          (a ULP-of-A band)
//!
//! with eps_A = 2^-mant(A) (spacing at 1.0), N the runtime extent, and k(S,A) an
//! extent-free per-cell constant. §6.0-0004 leaves the reduction ORDER and the
//! accumulator WIDTH unpinned; the pinned reference is the ascending-INDEX fold
//! with the accumulator held wholly in A and a single terminal narrow A->S
//! (== `kernels::reduce_acc` / `diff::reference_reduce_acc`). k is NOT derivable
//! from first principles — it depends on the permitted reassociation set — so it
//! is calibrated EMPIRICALLY and ADVERSARIALLY here: sweep the fold schedules a
//! conformant kernel may emit (sequential / pairwise-tree / blocked / strided,
//! all computed in A) against the ascending-index reference over an adversarial
//! value set, and take the observed worst case plus a x2 margin.
//!
//! MECHANISM (Design 1 "public-api-permute", recommended variant b): the
//! (widen S->A, fold-entirely-in-A, single narrow A->S) triple is reconstructed
//! from the PUBLIC `ScalarFloat` API with ZERO new kiss-ref-core surface:
//!   widen  = |x:S| A::from_f64(x.to_f64())              // == scalar::widen  (pub(crate))
//!   fold   = A::add (== eval_op::<A>(Op::Add)), seeded at A::ZERO, NO intermediate narrow
//!   narrow = |a:A| S::from_f64_single_round(a.to_f64()) // == scalar::narrow (pub(crate))
//! Every value set is gated by [`assert_fidelity`]: the reconstruction's
//! ascending-index narrowed result MUST equal `reduce_ref::<S>(x, Sum, [0], A)`
//! bit-for-bit, certifying the reconstruction against the shipped kernel.
//!
//! STATUS: #96 is merged, so the clause SHAPE is final; the k(S,A) NUMBERS remain
//! **preliminary-pending-Baracuda-step-3b** real value sets / tensor shapes / true
//! K ceiling (those can only SHARPEN — raise — the worst case; the x2 margin
//! absorbs modest sharpening). Do NOT freeze interop on the numbers. See the
//! module `emit_preliminary_k_table` test output for the computed table.

#![allow(clippy::needless_range_loop)]

use half::{bf16, f16};
use kiss_classify_vocab::Dtype;
use kiss_ref_core::kernels::reduce_ref;
use kiss_ref_core::tensor_ops::matmul_ref;
use kiss_ref_core::{
    ulp_distance_bf16, ulp_distance_e4m3, ulp_distance_e5m2, ulp_distance_f16, ulp_distance_f32,
    ulp_distance_f64,
};
use kiss_ref_core::{E4m3, E5m2, Monoid, ScalarFloat, Tensor};

// ---- deterministic PRNG: SplitMix64 (fixed-seed, no wall-clock, no thread_rng) --

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `0..n` (n > 0).
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    #[inline]
    fn sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

// ---- the calibration float: public `ScalarFloat` + the 3 things it omits -------

/// A float the calibration runs over. Adds the three things `ScalarFloat`
/// deliberately does not carry — a name, the mantissa width (for eps), raw bits
/// (for the fidelity gate), and the crate's own `ulp_distance_*` metric (the SAME
/// metric Baracuda's device diff uses) — plus the two format extents the
/// adversarial value families need.
trait CalFloat: ScalarFloat + Copy + core::fmt::Debug {
    const NAME: &'static str;
    /// eps = 2^-MANT: f16 10, bf16 7, f32 23, f64 52, e4m3 3, e5m2 2.
    const MANT: i32;
    fn bits(self) -> u64;
    fn ulp_to(self, other: Self) -> u64;
    /// Largest finite magnitude of the format (the adversary's "big").
    fn max_finite() -> f64;
    /// Smallest positive subnormal of the format (the adversary's "small").
    fn min_subnormal() -> f64;
    #[inline]
    fn eps() -> f64 {
        2f64.powi(-Self::MANT)
    }
}

impl CalFloat for f16 {
    const NAME: &'static str = "f16";
    const MANT: i32 = 10;
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_f16(self, other) as u64
    }
    fn max_finite() -> f64 {
        65504.0
    }
    fn min_subnormal() -> f64 {
        2f64.powi(-24)
    }
}
impl CalFloat for bf16 {
    const NAME: &'static str = "bf16";
    const MANT: i32 = 7;
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_bf16(self, other) as u64
    }
    fn max_finite() -> f64 {
        // (2 - 2^-7) * 2^127
        (2.0 - 2f64.powi(-7)) * 2f64.powi(127)
    }
    fn min_subnormal() -> f64 {
        2f64.powi(-133)
    }
}
impl CalFloat for f32 {
    const NAME: &'static str = "f32";
    const MANT: i32 = 23;
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_f32(self, other) as u64
    }
    fn max_finite() -> f64 {
        f32::MAX as f64
    }
    fn min_subnormal() -> f64 {
        2f64.powi(-149)
    }
}
impl CalFloat for f64 {
    const NAME: &'static str = "f64";
    const MANT: i32 = 52;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_f64(self, other)
    }
    fn max_finite() -> f64 {
        f64::MAX
    }
    fn min_subnormal() -> f64 {
        f64::from_bits(1)
    }
}
impl CalFloat for E4m3 {
    const NAME: &'static str = "e4m3";
    const MANT: i32 = 3;
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_e4m3(self, other) as u64
    }
    fn max_finite() -> f64 {
        448.0
    }
    fn min_subnormal() -> f64 {
        2f64.powi(-9)
    }
}
impl CalFloat for E5m2 {
    const NAME: &'static str = "e5m2";
    const MANT: i32 = 2;
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
    fn ulp_to(self, other: Self) -> u64 {
        ulp_distance_e5m2(self, other) as u64
    }
    fn max_finite() -> f64 {
        57344.0
    }
    fn min_subnormal() -> f64 {
        2f64.powi(-16)
    }
}

// ---- the (widen, fold-in-A, narrow) triple reconstructed from the public API ---

#[inline]
fn widen<S: ScalarFloat, A: ScalarFloat>(x: S) -> A {
    A::from_f64(x.to_f64())
}
#[inline]
fn narrow<A: ScalarFloat, S: ScalarFloat>(a: A) -> S {
    S::from_f64_single_round(a.to_f64())
}

/// A fold schedule a conformant kernel may emit (§6.0-0004 leaves ORDER unpinned).
/// Design 2's crisp taxonomy: every intermediate is held in A, one terminal narrow.
#[derive(Clone)]
enum Schedule {
    /// Left-leaning sequential fold over the given leaf order.
    Sequential(Vec<usize>),
    /// Balanced pairwise (adjacent) tree over the given leaf order.
    PairwiseTree(Vec<usize>),
    /// Blocked: sequential per block of `block`, block partials combined ascending
    /// (block=32 is a GPU warp reduction). Partials kept in A.
    Blocked { order: Vec<usize>, block: usize },
    /// Strided: `lanes` interleaved lane-accumulators (element k -> lane k%lanes),
    /// lanes combined ascending in A (the SIMD / tensor-core shape).
    Strided { order: Vec<usize>, lanes: usize },
}

/// Fold pre-widened A `t` per `s`, seeded at the Sum identity `A::ZERO`, WHOLLY in
/// A (no intermediate narrow). Returns the PRE-NARROW A accumulator.
/// `A::add` == `eval_op::<A>(Op::Add)` (resolve.rs:45).
fn fold_a<A: ScalarFloat>(t: &[A], s: &Schedule) -> A {
    let add = |a: A, b: A| a.add(b);
    match s {
        Schedule::Sequential(p) => p.iter().fold(A::ZERO, |a, &i| add(a, t[i])),
        Schedule::PairwiseTree(p) => {
            let mut lv: Vec<A> = p.iter().map(|&i| t[i]).collect();
            while lv.len() > 1 {
                lv = lv
                    .chunks(2)
                    .map(|c| if c.len() == 2 { add(c[0], c[1]) } else { c[0] })
                    .collect();
            }
            lv.first().copied().unwrap_or(A::ZERO)
        }
        Schedule::Blocked { order, block } => order
            .chunks((*block).max(1))
            .map(|b| b.iter().fold(A::ZERO, |a, &i| add(a, t[i])))
            .fold(A::ZERO, &add),
        Schedule::Strided { order, lanes } => {
            let l = (*lanes).max(1);
            let mut ln = vec![A::ZERO; l];
            for (k, &i) in order.iter().enumerate() {
                ln[k % l] = add(ln[k % l], t[i]);
            }
            ln.into_iter().fold(A::ZERO, add)
        }
    }
}

/// Reference = ascending-INDEX A-fold + |max partial sum| (as f64), the pinned
/// §6.17-0005 profile (== `reduce_acc`'s accumulator, pre-narrow).
fn reference<A: ScalarFloat>(t: &[A]) -> (A, f64) {
    let mut acc = A::ZERO;
    let mut mp = 0.0f64;
    for &x in t {
        acc = acc.add(x);
        let m = acc.to_f64().abs();
        if m > mp {
            mp = m;
        }
    }
    (acc, mp)
}

// ---- adversarial value families (generated in S so widen S->A is exact) --------

/// A named value set (all entries exactly representable in S, so `widen::<S,A>` is
/// lossless under the subsumption guard).
struct ValueSet<S> {
    label: &'static str,
    v: Vec<S>,
}

/// Random S value with magnitude ~ 2^exp * [1,2), optional sign.
fn rand_s<S: CalFloat>(rng: &mut SplitMix64, min_exp: i32, max_exp: i32, signed: bool) -> S {
    let span = (max_exp - min_exp).max(0) as usize + 1;
    let e = min_exp + rng.below(span) as i32;
    let m = 1.0 + rng.unit(); // [1, 2)
    let mut val = m * 2f64.powi(e);
    if signed && rng.sign() < 0.0 {
        val = -val;
    }
    S::from_f64(val)
}

/// The N sweep for reductions / scans.
const NS: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256, 1024];

/// Build the value families for storage dtype S. `A_MANT` sizes the stagnation
/// adversary to the accumulator being calibrated.
fn value_sets<S: CalFloat>(rng: &mut SplitMix64, a_mant: i32) -> Vec<ValueSet<S>> {
    let mut out: Vec<ValueSet<S>> = Vec::new();
    let big = S::max_finite();
    let min_sub = S::min_subnormal();

    for &n in NS {
        // ---- family 1: the narrow_tensor_lane stagnation golden, generalized:
        // [128, 8, 8, ..., 8] (exact sum 192, exactly representable in every lane).
        {
            let mut v = Vec::with_capacity(n);
            v.push(S::from_f64(128.0));
            for _ in 1..n {
                v.push(S::from_f64(8.0));
            }
            out.push(ValueSet {
                label: "golden-stagnation",
                v,
            });
        }

        // ---- family 1b: accumulator-sized stagnation adversary. N/2 copies of
        // S-max-finite then N/2 copies of a small S value tuned so a BLOCK of them
        // escapes A's rounding of the big partial while each one stagnates:
        // small ~ ulp_A((N/2)*big)/8. (For (e4m3,f32),N=1024 this recovers the
        // worked golden: 512x448 then 512x2^-9.)
        if n >= 4 {
            let half = n / 2;
            let p = half as f64 * big; // big-block partial (in f64)
            if p.is_finite() && p > 0.0 {
                let ulp_a = 2f64.powi(p.abs().log2().floor() as i32 - a_mant);
                let small_target = (ulp_a / 8.0).max(min_sub);
                let small = S::from_f64(small_target);
                if small.to_f64() > 0.0 {
                    let mut v = Vec::with_capacity(n);
                    for _ in 0..half {
                        v.push(S::from_f64(big));
                    }
                    for _ in half..n {
                        v.push(small);
                    }
                    out.push(ValueSet {
                        label: "acc-stagnation",
                        v,
                    });
                }
            }
        }

        // ---- family 2: catastrophic cancellation. [+B, s, s, ..., s, -B]: the
        // mass of the small terms is absorbed into the big partial in ascending
        // order but retained if a schedule sums them first; |maxpartial| ~ B while
        // Sum|x| ~ 2B + (n-2)|s|.
        if n >= 3 {
            let b_exp = ((S::max_finite().log2().floor() as i32) - 4).max(1);
            let bval = 2f64.powi(b_exp);
            let s_small = 2f64.powi((b_exp - a_mant - 2).max(-140));
            let bb = S::from_f64(bval);
            let ss = S::from_f64(s_small);
            let mut v = Vec::with_capacity(n);
            v.push(bb);
            for _ in 1..(n - 1) {
                v.push(ss);
            }
            v.push(S::from_f64(-bval));
            out.push(ValueSet {
                label: "cancellation",
                v,
            });

            // 2b: alternating growing signed (near-cancellation, wide spread).
            let mut v2 = Vec::with_capacity(n);
            for i in 0..n {
                let e = (i as i32 % (a_mant + 6)) - 2;
                let mag = 2f64.powi(e);
                let sgn = if i % 2 == 0 { 1.0 } else { -1.0 };
                v2.push(S::from_f64(sgn * mag));
            }
            out.push(ValueSet {
                label: "alt-cancellation",
                v: v2,
            });
        }

        // ---- family 3: wide dynamic range, mixed sign (the real A=f32 adversary):
        // S subnormal-ish .. ~1/4 max-finite.
        {
            let hi = (S::max_finite().log2().floor() as i32) - 2;
            let lo = hi - (a_mant + 12);
            let v: Vec<S> = (0..n).map(|_| rand_s::<S>(rng, lo, hi, true)).collect();
            out.push(ValueSet {
                label: "wide-dynamic",
                v,
            });
        }

        // ---- family 4: benign random control, [-2, 2].
        {
            let v: Vec<S> = (0..n).map(|_| rand_s::<S>(rng, -2, 1, true)).collect();
            out.push(ValueSet {
                label: "random-control",
                v,
            });
        }

        // ---- family 5: boundary / subnormal / RNE-tie sites.
        {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let pick = i % 5;
                let val = match pick {
                    0 => 1.0,
                    1 => 1.0 + 2f64.powi(-a_mant), // one ULP-of-A above 1
                    2 => min_sub,                  // min subnormal
                    3 => 2f64.powi(-(a_mant + 1)), // half-ULP tie site vs 1.0
                    _ => 3.0 * 2f64.powi(-(a_mant + 1)),
                };
                v.push(S::from_f64(val));
            }
            out.push(ValueSet {
                label: "boundary-tie",
                v,
            });
        }
    }
    out
}

/// All schedules to sweep for one widened value vector `t`.
fn schedules<A: ScalarFloat + CalFloat>(t: &[A], rng: &mut SplitMix64) -> Vec<Schedule> {
    let n = t.len();
    let mut out: Vec<Schedule> = Vec::new();
    if n < 2 {
        return out;
    }
    let idx: Vec<usize> = (0..n).collect();

    // index-order pairwise tree (differs from the sequential reference).
    out.push(Schedule::PairwiseTree(idx.clone()));

    // canonical leaf orders: reverse, sorted asc/desc magnitude, sorted asc/desc value.
    let rev: Vec<usize> = (0..n).rev().collect();
    let mut asc_mag = idx.clone();
    asc_mag.sort_by(|&a, &b| t[a].to_f64().abs().total_cmp(&t[b].to_f64().abs()));
    let mut desc_mag = asc_mag.clone();
    desc_mag.reverse();
    let mut asc_val = idx.clone();
    asc_val.sort_by(|&a, &b| t[a].to_f64().total_cmp(&t[b].to_f64()));
    let mut desc_val = asc_val.clone();
    desc_val.reverse();

    for p in [&rev, &asc_mag, &desc_mag, &asc_val, &desc_val] {
        out.push(Schedule::Sequential(p.clone()));
        out.push(Schedule::PairwiseTree(p.clone()));
    }

    // blocked over index + asc-magnitude orders (b=32 is a warp reduction).
    for &b in &[2usize, 4, 8, 16, 32, 64, 128] {
        if b < n {
            out.push(Schedule::Blocked {
                order: idx.clone(),
                block: b,
            });
            out.push(Schedule::Blocked {
                order: asc_mag.clone(),
                block: b,
            });
        }
    }
    // strided lanes (SIMD / tensor-core shape).
    for &l in &[2usize, 4, 8, 16, 32] {
        if l < n {
            out.push(Schedule::Strided {
                order: idx.clone(),
                lanes: l,
            });
        }
    }

    // R fixed-seed random permutations, each as Sequential AND PairwiseTree.
    let r = if n <= 64 { 64 } else { 48 };
    for _ in 0..r {
        let mut p = idx.clone();
        // Fisher-Yates.
        for i in (1..n).rev() {
            let j = rng.below(i + 1);
            p.swap(i, j);
        }
        out.push(Schedule::Sequential(p.clone()));
        out.push(Schedule::PairwiseTree(p));
    }
    out
}

// ---- the fidelity gate: reconstruction == the shipped kernel, bit-for-bit ------

/// The reconstruction's ascending-index narrowed result MUST equal
/// `reduce_ref::<S>(x, Sum, [0], A)` bit-for-bit — certifying the (widen, fold, narrow)
/// triple against the shipped kernel on every value set. Fails loudly if core's
/// fold semantics ever drift.
fn assert_fidelity<S: CalFloat, A: CalFloat>(inp: &[S]) {
    let t: Vec<A> = inp.iter().map(|&s| widen::<S, A>(s)).collect();
    let mine: S = narrow::<A, S>(reference::<A>(&t).0);
    let x = Tensor::from_vec(inp.to_vec(), &[inp.len()])
        .unwrap_or_else(|e| panic!("[{}/{}] tensor build: {e:?}", S::NAME, A::NAME));
    let r = reduce_ref::<S>(&x.view(), Monoid::Sum, &[0], A::DTYPE)
        .unwrap_or_else(|e| panic!("[{}/{}] reduce_ref: {e:?}", S::NAME, A::NAME));
    assert_eq!(
        mine.bits(),
        r.as_slice()[0].bits(),
        "[{}/{}] reconstruction != reduce_ref on n={}",
        S::NAME,
        A::NAME,
        inp.len()
    );
}

// ---- the per-cell calibration ------------------------------------------------

#[derive(Clone)]
struct KRow {
    s: &'static str,
    a: &'static str,
    k: f64,
    decl_k: f64,
    ulp_s_max: u64,
    straddle: u64,
    zero_mp: u64,
    satur: u64,
    worst: String,
}

fn calibrate<S: CalFloat, A: CalFloat>(seed: u64) -> KRow {
    let mut rng = SplitMix64::new(seed);
    let mut k = 0.0f64;
    let mut us = 0u64;
    let mut straddle = 0u64;
    let mut zmp = 0u64;
    let mut sat = 0u64;
    let mut worst = String::from("(none: all schedules matched the reference)");

    for set in value_sets::<S>(&mut rng, A::MANT) {
        let inp = &set.v;
        // FIRST assertion in every cell: the reconstruction IS the shipped kernel.
        assert_fidelity::<S, A>(inp);

        let t: Vec<A> = inp.iter().map(|&s| widen::<S, A>(s)).collect();
        let n = t.len() as f64;
        let (ref_acc, mp) = reference::<A>(&t);
        if !ref_acc.to_f64().is_finite() {
            sat += 1;
            continue; // saturation / non-finite reference: exact-match regime, excluded from k.
        }
        let ref_s: S = narrow::<A, S>(ref_acc);

        for sched in schedules::<A>(&t, &mut rng) {
            let acc = fold_a::<A>(&t, &sched);
            if !acc.to_f64().is_finite() {
                sat += 1;
                continue;
            }
            let dev = (acc.to_f64() - ref_acc.to_f64()).abs();
            if mp == 0.0 {
                if dev != 0.0 {
                    zmp += 1; // formula-violation flag: band is 0 but deviation is not.
                }
                continue;
            }
            let ks = dev / (n * A::eps() * mp);
            if ks > k {
                k = ks;
                worst = format!(
                    "{} n={} dev_A={:.6e} |maxpartial|={:.6e} k={:.4}",
                    set.label,
                    inp.len(),
                    dev,
                    mp,
                    ks
                );
            }
            // S-observable straddle (what a device diff compares, in ULP-of-S).
            let d = ref_s.ulp_to(narrow::<A, S>(acc));
            if d > us {
                us = d;
            }
            if d > 0 {
                straddle += 1;
            }
        }
    }
    let decl_k = (2.0 * k).max(1.0);
    KRow {
        s: S::NAME,
        a: A::NAME,
        k,
        decl_k,
        ulp_s_max: us,
        straddle,
        zero_mp: zmp,
        satur: sat,
        worst,
    }
}

// ---- the emit test: computes + prints the full preliminary k(S,A) table --------

/// All 18 legal `(S, A)` reduce cells (A subsumes S on BOTH (exp, mant) per
/// `guard_accumulator`), INCLUDING the diagonal A==S (order still unpinned there).
/// * = LOAD-BEARING Baracuda cell.
fn all_rows() -> Vec<KRow> {
    vec![
        // f16 -> {f16, f32, f64}
        calibrate::<f16, f16>(0xC4_0100),
        calibrate::<f16, f32>(0xC4_0101),
        calibrate::<f16, f64>(0xC4_0102),
        // bf16 -> {bf16, f32, f64}
        calibrate::<bf16, bf16>(0xC4_0110),
        calibrate::<bf16, f32>(0xC4_0111),
        calibrate::<bf16, f64>(0xC4_0112),
        // e4m3 -> {e4m3, f16, bf16, f32*, f64}
        calibrate::<E4m3, E4m3>(0xC4_0120),
        calibrate::<E4m3, f16>(0xC4_0121),
        calibrate::<E4m3, bf16>(0xC4_0122),
        calibrate::<E4m3, f32>(0xC4_0123), // LOAD-BEARING
        calibrate::<E4m3, f64>(0xC4_0124),
        // e5m2 -> {e5m2, f16, bf16, f32*, f64}
        calibrate::<E5m2, E5m2>(0xC4_0130),
        calibrate::<E5m2, f16>(0xC4_0131),
        calibrate::<E5m2, bf16>(0xC4_0132),
        calibrate::<E5m2, f32>(0xC4_0133), // LOAD-BEARING
        calibrate::<E5m2, f64>(0xC4_0134),
        // f32 -> {f32, f64}
        calibrate::<f32, f32>(0xC4_0140),
        calibrate::<f32, f64>(0xC4_0141),
    ]
}

#[test]
fn emit_preliminary_k_table() {
    let rows = all_rows();

    println!("\n=== PRELIMINARY k(S,A) reassociation-tolerance table (RFC #92 b / C4) ===");
    println!("band: tol = k(S,A) * N * eps_A * |max partial sum|   (ULP-of-A; eps_A = 2^-mant)");
    println!("declared_k = max(1.0, 2 * k_measured).  STATUS: KISS-CONTRACT-6.8-0012 normative (#96 merged); numbers preliminary-pending-Baracuda-3b.\n");
    println!(
        "| {:<5} | {:<5} | {:>9} | {:>9} | {:>8} | {:>8} | {:>6} | {:>5} |",
        "S", "A", "k_meas", "decl_k", "ulpS_max", "straddle", "zeroMP", "sat"
    );
    println!(
        "|{:-<7}|{:-<7}|{:-<11}|{:-<11}|{:-<10}|{:-<10}|{:-<8}|{:-<7}|",
        "", "", "", "", "", "", "", ""
    );
    for r in &rows {
        println!(
            "| {:<5} | {:<5} | {:>9.4} | {:>9.4} | {:>8} | {:>8} | {:>6} | {:>5} |",
            r.s, r.a, r.k, r.decl_k, r.ulp_s_max, r.straddle, r.zero_mp, r.satur
        );
    }
    println!("\n--- worst-case witness per cell ---");
    for r in &rows {
        println!("  ({:>4}, {:>4})  {}", r.s, r.a, r.worst);
    }
    println!();

    // Structural invariants (do NOT weaken the value sets to satisfy these).
    for r in &rows {
        assert!(r.k.is_finite(), "[{}/{}] k not finite", r.s, r.a);
        assert!(r.decl_k >= 1.0, "[{}/{}] declared_k floor", r.s, r.a);
        // The band SHAPE is inadequate if any cell trips the zero-|maxpartial|
        // guard with a nonzero deviation: that is a finding to route to the clause
        // authors, not something to paper over.
        assert_eq!(
            r.zero_mp, 0,
            "[{}/{}] FORMULA-VIOLATION: dev>0 while |maxpartial|=0 ({} times) — band SHAPE needs an absolute-floor term",
            r.s, r.a, r.zero_mp
        );
    }
}

/// Pins representative k values with margin so a regression (or a real change in
/// the observed worst case) fails loudly. Numbers are the ACTUAL computed
/// worst-case k for each cell, asserted below a provisional ceiling that already
/// carries headroom; they are NOT hand-chosen to look small.
#[test]
fn pin_representative_k_values() {
    let rows = all_rows();
    let get = |s: &str, a: &str| -> KRow {
        rows.iter()
            .find(|r| r.s == s && r.a == a)
            .cloned()
            .unwrap_or_else(|| panic!("missing cell ({s},{a})"))
    };

    // LOAD-BEARING (fp8 -> f32): the whole point of RFC #92 b — a wide f32
    // accumulator keeps reassociation far below 1 term-ULP-of-A.
    assert!(
        get("e4m3", "f32").k < 0.5,
        "(e4m3,f32) k regressed: {}",
        get("e4m3", "f32").k
    );
    assert!(
        get("e5m2", "f32").k < 0.5,
        "(e5m2,f32) k regressed: {}",
        get("e5m2", "f32").k
    );

    // The coarse-eps diagonals are the largest-k home, still O(1) (< 1).
    assert!(
        get("e4m3", "e4m3").k < 1.0,
        "(e4m3,e4m3) k regressed: {}",
        get("e4m3", "e4m3").k
    );
    assert!(
        get("e5m2", "e5m2").k < 1.0,
        "(e5m2,e5m2) k regressed: {}",
        get("e5m2", "e5m2").k
    );

    // Provisional ceiling: EVERY declared_k stays below this. If a future value
    // set (e.g. Baracuda step-3b) pushes past it, that is a real finding.
    const PROVISIONAL_CEILING: f64 = 8.0;
    for r in &rows {
        assert!(
            r.decl_k <= PROVISIONAL_CEILING,
            "[{}/{}] declared_k {} exceeds provisional ceiling {} — REPORT, do not raise silently",
            r.s,
            r.a,
            r.decl_k,
            PROVISIONAL_CEILING
        );
    }
}

// ---- secondary cross-check: the design-angle's named "permute + reduce_ref" -----

/// The design-angle's own method ("permute the reduced axis + call reduce_ref")
/// builds ONLY sequential folds and narrows A->S at every call boundary, so it can
/// express the Sequential subspace but NOT tree/blocked/strided. This proves the
/// Sequential subspace of the reconstruction EQUALS the public kernel (off the
/// diagonal, where `reduce_acc` genuinely folds in A and narrows once), while the
/// tree rows in the table exhibit deviations permute+reduce_ref provably cannot
/// reach.
#[test]
fn sequential_subspace_equals_permute_plus_reduce_ref() {
    let mut rng = SplitMix64::new(0xC4_0200);
    // Off-diagonal cell (e4m3 -> f32): the real reduce_acc path.
    for set in value_sets::<E4m3>(&mut rng, f32::MANT) {
        let inp = &set.v;
        let n = inp.len();
        let t: Vec<f32> = inp.iter().map(|&s| widen::<E4m3, f32>(s)).collect();
        // a couple of permutations.
        let perms: [Vec<usize>; 2] = [(0..n).rev().collect(), {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.below(i + 1);
                p.swap(i, j);
            }
            p
        }];
        for p in &perms {
            // reconstruction's Sequential(perm), then the single narrow A->S.
            let mine: E4m3 =
                narrow::<f32, E4m3>(fold_a::<f32>(&t, &Schedule::Sequential(p.clone())));
            // the named method: physically permute the tensor, call reduce_ref(acc=f32).
            let permuted: Vec<E4m3> = p.iter().map(|&i| inp[i]).collect();
            let x = Tensor::from_vec(permuted, &[n]).unwrap();
            let via = reduce_ref::<E4m3>(&x.view(), Monoid::Sum, &[0], Dtype::F32).unwrap();
            assert_eq!(
                mine.bits(),
                via.as_slice()[0].bits(),
                "[{}] Sequential(perm) reconstruction != permute+reduce_ref",
                set.label
            );
        }
    }
}

// ---- matmul & scan: same Sum engine, different N accounting --------------------

/// matmul is the Sum engine over realized A-products (terms = round_A(widen(a_p) *
/// widen(b_p)), N = K); the K-sum has identical reassociation structure. This gates
/// that identity: the ascending fold of the realized products, narrowed once,
/// equals `matmul_ref` bit-for-bit — so the reduce k(S,A) transfers to matmul with
/// N=K. (Prod / matmul multiplicative error is NOT calibrated here; per the design
/// it would be reported separately and flagged, never folded onto Sum's k.)
#[test]
fn matmul_reuses_sum_engine_over_realized_products() {
    fn check<S: CalFloat, A: CalFloat>(a_vals: &[S], b_vals: &[S]) {
        let k = a_vals.len();
        assert_eq!(k, b_vals.len());
        // realized A-products (mul rounds to A — the tensor-core atom).
        let prods: Vec<A> = a_vals
            .iter()
            .zip(b_vals)
            .map(|(&a, &b)| widen::<S, A>(a).mul(widen::<S, A>(b)))
            .collect();
        let mine: S = narrow::<A, S>(reference::<A>(&prods).0);
        let am = Tensor::from_vec(a_vals.to_vec(), &[1, k]).unwrap();
        let bm = Tensor::from_vec(b_vals.to_vec(), &[k, 1]).unwrap();
        let r = matmul_ref::<S>(&am.view(), &bm.view(), A::DTYPE).unwrap();
        assert_eq!(
            mine.bits(),
            r.as_slice()[0].bits(),
            "[{}/{}] matmul_ref != ascending fold of realized products (K={})",
            S::NAME,
            A::NAME,
            k
        );
    }
    // the narrow_tensor_lane K=8 golden: 64*1 x8, and the [128, eight 8s]*1 contraction.
    let a8: Vec<E4m3> = core::iter::repeat(E4m3::from_f32(64.0)).take(8).collect();
    let b8: Vec<E4m3> = core::iter::repeat(E4m3::from_f32(1.0)).take(8).collect();
    check::<E4m3, f32>(&a8, &b8);
    check::<E5m2, f32>(
        &core::iter::repeat(E5m2::from_f32(64.0))
            .take(8)
            .collect::<Vec<_>>(),
        &core::iter::repeat(E5m2::from_f32(1.0))
            .take(8)
            .collect::<Vec<_>>(),
    );
    let mut av = vec![E4m3::from_f32(128.0)];
    av.extend(core::iter::repeat(E4m3::from_f32(8.0)).take(8));
    let bv: Vec<E4m3> = core::iter::repeat(E4m3::from_f32(1.0)).take(9).collect();
    check::<E4m3, f32>(&av, &bv);
}

/// A prefix scan position j is a prefix fold of length j+1, so k(S,A) reuses with
/// N = prefix length. This gates the identity at a representative scan.
#[test]
fn scan_prefix_is_a_length_j_fold() {
    use kiss_ref_core::kernels::prefix_scan_ref;
    let inp: Vec<E4m3> = {
        let mut v = vec![E4m3::from_f32(128.0)];
        v.extend(core::iter::repeat(E4m3::from_f32(8.0)).take(8));
        v
    };
    let x = Tensor::from_vec(inp.clone(), &[inp.len()]).unwrap();
    let scan = prefix_scan_ref::<E4m3>(&x.view(), Monoid::Sum, 0, false, Dtype::F32).unwrap();
    // each inclusive-scan slot j == ascending fold of the first j+1 widened terms, narrowed once.
    for j in 0..inp.len() {
        let t: Vec<f32> = inp[..=j].iter().map(|&s| widen::<E4m3, f32>(s)).collect();
        let mine: E4m3 = narrow::<f32, E4m3>(reference::<f32>(&t).0);
        assert_eq!(
            mine.bits(),
            scan.as_slice()[j].bits(),
            "scan slot {j} != length-{} prefix fold",
            j + 1
        );
    }
}
