// SPDX-License-Identifier: MIT OR Apache-2.0
//! The **differential export seam** — the API other implementations dev-depend
//! on to test against this reference.
//!
//! Both the KISS-Conform harness and Baracuda's on-device differential harness
//! need the same thing: feed a batch of input vectors, get the reference's
//! outputs, and compare a candidate's outputs under a ULP tolerance. This module
//! is that seam. Distances use a **sign-magnitude IEEE total-order** ULP metric
//! (faithful near zero and across the signed-zero boundary — unlike a relative
//! epsilon), so signed zero reads as 1 ULP and both-NaN reads as 0.
//!
//! **Why both-NaN → 0 (NaN payload is class, not bits):** raw-bit *moves*
//! (`select`, compare-driven picks, gather/sort) carry a NaN's payload intact,
//! but *arithmetic* ops may re-mint the payload per device — a live cross-
//! hardware witness: an sm_89 float `add` producing NaN canonicalizes it to
//! `0x7fffffff` while x86 propagates the input's `0x7fc00000` (Baracuda device
//! diff, 2026-07-23). So the comparator must treat any-NaN-vs-any-NaN as a
//! match (class equality), while ±0 stays 1 ULP apart — that asymmetry is what
//! keeps the metric NaN-portable without losing signed-zero sensitivity (the
//! `max_prop` tie-bug catcher).
//!
//! **`Tolerance::Exact` is op-aware on NaN (§6.8-0010).** The both-NaN=0 rule
//! above is the COMPUTED-NaN convention — correct for arithmetic, whose payload
//! is architectural. But a MOVED NaN — a `select` / minmax / `copysign` / `abs`
//! / `neg` output (KISS-OPS-6.16-0009: an arithmetic-free decomposition moves,
//! it does not compute) — has a payload DETERMINED by the inputs, so §6.8-0010(a)
//! binds it exact-byte. So `Exact` bit-compares the determined-NaN ops and keeps
//! the payload-blind metric for the computed ones, selecting on
//! [`kiss_ops_vocab::Op::nan_payload_is_determined`]. `ulp_distance_*` itself is
//! unchanged — it stays the computed-NaN / ULP metric; only the `Exact` decision
//! gained the op branch.
//!
//! **…and a computed NaN's QUIETNESS is compared, though its payload is not.** The quiet
//! bit is carved explicitly OUT of the payload exemption: KISS-OPS-6.16-0010 makes
//! delivering a *quiet* NaN a MUST for a decomposition containing arithmetic, so exempting
//! the whole payload would leave that clause undetectable by any conformant harness. A
//! computed NaN therefore matches iff the observed value is also NaN and — **where the
//! dtype's encoding admits a signaling NaN** — agrees in quietness with the expected value.
//! For `f8e4m3fn`, whose single NaN encoding has no quiet bit, the comparison is **vacuous**
//! and MUST NOT be synthesized (§6.8-0010). This MIRRORS KISS #352's `compare_nan_output`
//! rather than re-deriving it; the rule lives once, in `computed_nan_conforms`.

extern crate alloc;
use alloc::vec::Vec;

use kiss_ops_vocab::decomp::Expr;
use kiss_ops_vocab::Op;

use crate::{eval_expr, eval_op, Error};

/// Map an `f64` bit pattern to a monotone total order (IEEE 754 totalOrder):
/// negatives reverse, positives shift above zero, so integer subtraction on the
/// keys is the ULP distance.
#[inline]
fn key_f64(x: f64) -> u64 {
    let b = x.to_bits();
    if b & (1 << 63) != 0 {
        !b
    } else {
        b | (1 << 63)
    }
}

#[inline]
fn key_f32(x: f32) -> u32 {
    let b = x.to_bits();
    if b & (1 << 31) != 0 {
        !b
    } else {
        b | (1 << 31)
    }
}

/// Sign-magnitude ULP distance between two `f64`s. Both-NaN → 0 (a reference NaN
/// matched by a candidate NaN is conforming); exactly-one NaN → `u64::MAX`;
/// `+0.0` vs `-0.0` → 1 (they are genuinely distinct, per the signed-zero pins).
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0,
        (true, false) | (false, true) => u64::MAX,
        (false, false) => {
            let (x, y) = (key_f64(a), key_f64(b));
            x.abs_diff(y)
        }
    }
}

/// Sign-magnitude ULP distance between two `f32`s (same rules as the `f64` form).
pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0,
        (true, false) | (false, true) => u32::MAX,
        (false, false) => {
            let (x, y) = (key_f32(a), key_f32(b));
            x.abs_diff(y)
        }
    }
}

/// KISS-CONFORM-6.8-0010's **computed-NaN** rule, in ONE place.
///
/// It is applied at seven comparison sites, and a rule copied seven times is a rule with
/// seven chances to be copied wrong — so the sites delegate here rather than restating it.
///
/// This MIRRORS KISS #352's `compare_nan_output` COMPUTED arm rather than re-deriving it
/// (a second independent copy of a normative rule is the defect that mechanism exists to
/// remove): a computed NaN matches **iff** the observed value is also NaN and, **where the
/// dtype's encoding admits a signaling NaN, agrees in quietness with the expected value**.
///
/// Payload and sign are NOT compared — a computed NaN's payload is architectural (the
/// sm_89-vs-x86 witness in this module's docs) and bit-comparing it would fail conformant
/// hardware. The **quiet bit is carved explicitly out of that exemption**: KISS-OPS-6.16-0010
/// makes delivering a *quiet* NaN a MUST for a decomposition containing arithmetic, so
/// exempting the whole payload would leave that clause undetectable by any harness.
///
/// ⚠️ For a dtype whose encoding admits **no** signaling NaN (`f8e4m3fn`, a single NaN
/// encoding) the quietness comparison is **vacuous**, and a comparator MUST NOT synthesize a
/// distinction the format cannot represent (§6.8-0010). That is what `admits_snan` gates.
#[inline]
fn computed_nan_conforms(
    e_nan: bool,
    g_nan: bool,
    admits_snan: bool,
    e_quiet: bool,
    g_quiet: bool,
) -> bool {
    e_nan && g_nan && (!admits_snan || e_quiet == g_quiet)
}

/// Is this NaN **quiet**? (The quiet bit is the mantissa's most-significant bit.)
#[inline]
fn quiet_f64(x: f64) -> bool {
    x.to_bits() & 0x0008_0000_0000_0000 != 0
}

/// Is this NaN **quiet**? (The quiet bit is the mantissa's most-significant bit.)
#[inline]
fn quiet_f32(x: f32) -> bool {
    x.to_bits() & 0x0040_0000 != 0
}

/// The whole `Tolerance::Exact` decision for the `f64` lane: §6.8-0010(a) binds a
/// **determined** (moved) NaN exact-byte, payload and sign included; a **computed** NaN
/// goes to [`computed_nan_conforms`]. `f64` admits a signaling NaN, so quietness is
/// compared and never vacuous here.
#[inline]
fn exact_conforms_f64(determined: bool, e: f64, g: f64, d: u64) -> bool {
    if determined {
        e.to_bits() == g.to_bits()
    } else if e.is_nan() || g.is_nan() {
        computed_nan_conforms(e.is_nan(), g.is_nan(), true, quiet_f64(e), quiet_f64(g))
    } else {
        d == 0
    }
}

/// The `f32` lane's [`Tolerance::Exact`] decision. Same rule as [`exact_conforms_f64`];
/// `f32` also admits a signaling NaN.
#[inline]
fn exact_conforms_f32(determined: bool, e: f32, g: f32, d: u64) -> bool {
    if determined {
        e.to_bits() == g.to_bits()
    } else if e.is_nan() || g.is_nan() {
        computed_nan_conforms(e.is_nan(), g.is_nan(), true, quiet_f32(e), quiet_f32(g))
    } else {
        d == 0
    }
}

/// A comparison tolerance for a differential run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tolerance {
    /// The exact-comparison tolerance. A finite result compares bitwise-identical
    /// (0 ULP). A **NaN** result compares **op-aware, per §6.8-0010**: a
    /// determined-NaN op (a move / bit-transform — `select`, the minmax family,
    /// `abs`/`neg`/`copysign`) binds the payload AND sign exact-byte
    /// (§6.8-0010(a)), while a computed-NaN op (arithmetic / transcendental) keeps
    /// its payload and sign UNcompared, because arithmetic re-mints the payload per
    /// device — but its **quietness IS compared** wherever the dtype's encoding admits
    /// a signaling NaN (vacuous for `f8e4m3fn`), because KISS-OPS-6.16-0010 makes
    /// delivering a quiet NaN a MUST. The seam selects the arm from the op's (or region's)
    /// [`kiss_ops_vocab::Op::nan_payload_is_determined`], so `Exact` is the honest
    /// comparison for BOTH exact-op populations — the integer, comparison,
    /// `select`, rounding, and signed-zero pins included.
    Exact,
    /// Within `n` ULP — for the declared-ULP transcendentals (§6.8). Use the
    /// op's [`kiss_ops_vocab::Op::ulp_ceiling`] as the bound.
    Ulp(u64),
}

/// The outcome of diffing a candidate against the reference over a batch.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffReport {
    /// Number of rows compared.
    pub n: usize,
    /// Rows outside tolerance.
    pub mismatches: usize,
    /// Worst ULP distance seen (across all rows).
    pub max_ulp: u64,
    /// First out-of-tolerance row: `(index, reference, candidate)`.
    pub first_mismatch: Option<(usize, f64, f64)>,
}

impl DiffReport {
    /// True iff every row was within tolerance.
    pub fn conforms(&self) -> bool {
        self.mismatches == 0
    }
}

/// Reference `f64` outputs of `op` over a batch of input rows — each row is the
/// op's argument tuple (`&[a]` for a unary op, `&[a, b]` for a binary op). Errors
/// if `op` is not evaluable in the scalar path (e.g. a structural/tensor op).
pub fn reference_f64(op: Op, rows: &[&[f64]]) -> Result<Vec<f64>, Error> {
    rows.iter().map(|r| eval_op::<f64>(op, r)).collect()
}

/// Reference `f32` outputs of `op` over a batch of input rows.
pub fn reference_f32(op: Op, rows: &[&[f32]]) -> Result<Vec<f32>, Error> {
    rows.iter().map(|r| eval_op::<f32>(op, r)).collect()
}

/// Diff a candidate's `f64` outputs against this reference over `rows`, under
/// `tol`. The candidate slice must have one entry per row.
pub fn diff_f64(
    op: Op,
    rows: &[&[f64]],
    candidate: &[f64],
    tol: Tolerance,
) -> Result<DiffReport, Error> {
    let reference = reference_f64(op, rows)?;
    let determined = op.nan_payload_is_determined();
    if candidate.len() != reference.len() {
        return Err(Error::LengthMismatch {
            expected: reference.len(),
            got: candidate.len(),
        });
    }
    let mut report = DiffReport {
        n: reference.len(),
        mismatches: 0,
        max_ulp: 0,
        first_mismatch: None,
    };
    for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
        let d = ulp_distance_f64(e, g);
        if d > report.max_ulp {
            report.max_ulp = d;
        }
        let ok = match tol {
            // §6.8-0010(a): a determined-NaN op (a move / bit-transform) compares
            // exact-byte, payload+sign — so a wrong moved NaN payload fails even at
            // 0 ULP. A computed-NaN op's payload and sign stay uncompared (arithmetic
            // re-mints the payload per device — the module witness), but its QUIETNESS
            // is compared per §6.8-0010; see `exact_conforms_f64`.
            Tolerance::Exact => exact_conforms_f64(determined, e, g, d),
            Tolerance::Ulp(n) => d <= n,
        };
        if !ok {
            report.mismatches += 1;
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some((i, e, g));
            }
        }
    }
    Ok(report)
}

/// Diff a candidate's `f32` outputs against this reference over `rows`, under
/// `tol`. Same shape as [`diff_f64`]; ULP distances are the `f32` metric.
pub fn diff_f32(
    op: Op,
    rows: &[&[f32]],
    candidate: &[f32],
    tol: Tolerance,
) -> Result<DiffReport, Error> {
    let reference = reference_f32(op, rows)?;
    let determined = op.nan_payload_is_determined();
    if candidate.len() != reference.len() {
        return Err(Error::LengthMismatch {
            expected: reference.len(),
            got: candidate.len(),
        });
    }
    let mut report = DiffReport {
        n: reference.len(),
        mismatches: 0,
        max_ulp: 0,
        first_mismatch: None,
    };
    for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
        let d = ulp_distance_f32(e, g) as u64;
        if d > report.max_ulp {
            report.max_ulp = d;
        }
        let ok = match tol {
            // §6.8-0010(a): a determined-NaN op (a move / bit-transform) compares
            // exact-byte, payload+sign — so a wrong moved NaN payload fails even at
            // 0 ULP. A computed-NaN op's payload and sign stay uncompared (arithmetic
            // re-mints the payload per device — the module witness), but its QUIETNESS
            // is compared per §6.8-0010; see `exact_conforms_f32`.
            Tolerance::Exact => exact_conforms_f32(determined, e, g, d),
            Tolerance::Ulp(n) => d <= n,
        };
        if !ok {
            report.mismatches += 1;
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some((i, e as f64, g as f64));
            }
        }
    }
    Ok(report)
}

// ---- composed expressions (fused elementwise regions) -----------------------
//
// Fuel fuses adjacent elementwise ops into one region; to diff a region against
// this reference, express it as a §6.13 `Expr` AST and evaluate it row-wise with
// `eval_expr` — the SAME scalar engine the per-op seam uses, so NaN-propagation,
// signed zero, and the refined forms are inherited node-for-node.
// `reference_expr`/`diff_expr` are the multi-node analogue of
// `reference_f64`/`diff_f64`.

/// Reference `f64` outputs of a composed [`Expr`] over a batch of input rows —
/// each row is the operand tuple the expression's `input(i)` leaves read
/// (`input(0)` = `row[0]`, `input(1)` = `row[1]`, …), evaluated through
/// [`eval_expr`]. This is the multi-node analogue of [`reference_f64`]: where
/// that seam exports one op's reference outputs, this exports a whole fused
/// elementwise region's, row by row. Errors if the expression is not scalar-
/// evaluable on a row (e.g. an `input(i)` with no value supplied —
/// [`Error::MissingInput`] — or a structural/tensor op inside it).
pub fn reference_expr(expr: &Expr, rows: &[&[f64]]) -> Result<Vec<f64>, Error> {
    rows.iter().map(|r| eval_expr::<f64>(expr, r)).collect()
}

/// Diff a candidate's `f64` outputs against this reference for a composed
/// [`Expr`] over `rows`, under `tol`. Same shape as [`diff_f64`]; the candidate
/// slice must have one entry per row.
///
/// **The determinism class of a composed `Expr` is the CALLER's to own, so the
/// ULP tolerance here is advisory — this seam deliberately does not pick one:**
/// - A chain of only exact ops (`+ - * /`, comparison, `select`, rounding,
///   `copysign`) is `DetClass::ExactByte`: every node is bit-reproducible, so the
///   region is, and `Tolerance::Exact` is the honest comparison.
/// - A chain that passes through a §6.8 transcendental is `DetClass::Ulp` **only
///   insofar as per-node ULP error adds linearly** — a bound that holds while
///   each node's relative error stays small and the composition does not amplify
///   it. `diff_expr` faithfully applies whatever `Tolerance::Ulp(n)` the caller
///   supplies, but it cannot know that `n`.
/// - **CANCELLATION defeats any linear band:** subtracting two near-equal
///   intermediates (e.g. `sub(add(x, eps), x)`) blows up the *relative* error of
///   the result without bound, so a candidate that reassociates or reorders the
///   region can land arbitrarily many ULP from this reference while each
///   evaluation is legitimate. No fixed per-region ULP is correct for such a chain.
///
/// So this module supplies the **mechanism** (row-wise reference + the ULP
/// comparator) and refuses to invent a tolerance: `diff_expr` applies exactly the
/// `Tolerance` it is given, and the `DetClass` / advisory band lives with the
/// caller (Fuel), which alone has the region's provenance to compute it — the
/// analogue of the most-permissive `bridge::DetClass::join` the tensor layer
/// already performs.
pub fn diff_expr(
    expr: &Expr,
    rows: &[&[f64]],
    candidate: &[f64],
    tol: Tolerance,
) -> Result<DiffReport, Error> {
    let reference = reference_expr(expr, rows)?;
    let determined = kiss_ops_vocab::decomp::expr_nan_payload_is_determined(expr);
    if candidate.len() != reference.len() {
        return Err(Error::LengthMismatch {
            expected: reference.len(),
            got: candidate.len(),
        });
    }
    let mut report = DiffReport {
        n: reference.len(),
        mismatches: 0,
        max_ulp: 0,
        first_mismatch: None,
    };
    for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
        let d = ulp_distance_f64(e, g);
        if d > report.max_ulp {
            report.max_ulp = d;
        }
        let ok = match tol {
            // §6.8-0010(a): a determined-NaN op (a move / bit-transform) compares
            // exact-byte, payload+sign — so a wrong moved NaN payload fails even at
            // 0 ULP. A computed-NaN op's payload and sign stay uncompared (arithmetic
            // re-mints the payload per device — the module witness), but its QUIETNESS
            // is compared per §6.8-0010; see `exact_conforms_f64`.
            Tolerance::Exact => exact_conforms_f64(determined, e, g, d),
            Tolerance::Ulp(n) => d <= n,
        };
        if !ok {
            report.mismatches += 1;
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some((i, e, g));
            }
        }
    }
    Ok(report)
}

/// Reference `f32` outputs of a composed [`Expr`] over a batch of input rows —
/// the `f32` mirror of [`reference_expr`] (evaluates each row through
/// [`eval_expr`] in the `f32` compute lane).
pub fn reference_expr_f32(expr: &Expr, rows: &[&[f32]]) -> Result<Vec<f32>, Error> {
    rows.iter().map(|r| eval_expr::<f32>(expr, r)).collect()
}

/// Diff a candidate's `f32` outputs against this reference for a composed
/// [`Expr`] — the `f32` mirror of [`diff_expr`] (ULP distances use the `f32`
/// metric). The advisory-tolerance caveat on [`diff_expr`] applies unchanged.
pub fn diff_expr_f32(
    expr: &Expr,
    rows: &[&[f32]],
    candidate: &[f32],
    tol: Tolerance,
) -> Result<DiffReport, Error> {
    let reference = reference_expr_f32(expr, rows)?;
    let determined = kiss_ops_vocab::decomp::expr_nan_payload_is_determined(expr);
    if candidate.len() != reference.len() {
        return Err(Error::LengthMismatch {
            expected: reference.len(),
            got: candidate.len(),
        });
    }
    let mut report = DiffReport {
        n: reference.len(),
        mismatches: 0,
        max_ulp: 0,
        first_mismatch: None,
    };
    for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
        let d = ulp_distance_f32(e, g) as u64;
        if d > report.max_ulp {
            report.max_ulp = d;
        }
        let ok = match tol {
            // §6.8-0010(a): a determined-NaN op (a move / bit-transform) compares
            // exact-byte, payload+sign — so a wrong moved NaN payload fails even at
            // 0 ULP. A computed-NaN op's payload and sign stay uncompared (arithmetic
            // re-mints the payload per device — the module witness), but its QUIETNESS
            // is compared per §6.8-0010; see `exact_conforms_f32`.
            Tolerance::Exact => exact_conforms_f32(determined, e, g, d),
            Tolerance::Ulp(n) => d <= n,
        };
        if !ok {
            report.mismatches += 1;
            if report.first_mismatch.is_none() {
                report.first_mismatch = Some((i, e as f64, g as f64));
            }
        }
    }
    Ok(report)
}

// ---- narrow floats (f16 / bf16) ---------------------------------------------
//
// The seam covers every float dtype the reference computes in, so a consumer can
// diff f16/bf16 kernels too. Same sign-magnitude total-order metric, on the
// 16-bit pattern.

#[inline]
fn key_u16(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}

macro_rules! narrow_diff {
    ($t:ty, $ulp:ident, $refr:ident, $diff:ident, $quiet:expr, $admits_snan:expr) => {
        /// Sign-magnitude ULP distance between two narrow-float values (both-NaN
        /// → 0, one-NaN → `u16::MAX`).
        pub fn $ulp(a: $t, b: $t) -> u16 {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => 0,
                (true, false) | (false, true) => u16::MAX,
                (false, false) => key_u16(a.to_bits()).abs_diff(key_u16(b.to_bits())),
            }
        }

        /// Reference outputs of `op` over a batch of narrow-float input rows.
        pub fn $refr(op: Op, rows: &[&[$t]]) -> Result<Vec<$t>, Error> {
            rows.iter().map(|r| eval_op::<$t>(op, r)).collect()
        }

        /// Diff a candidate's narrow-float outputs against this reference.
        pub fn $diff(
            op: Op,
            rows: &[&[$t]],
            candidate: &[$t],
            tol: Tolerance,
        ) -> Result<DiffReport, Error> {
            let reference = $refr(op, rows)?;
            let determined = op.nan_payload_is_determined();
            if candidate.len() != reference.len() {
                return Err(Error::LengthMismatch {
                    expected: reference.len(),
                    got: candidate.len(),
                });
            }
            let mut report = DiffReport {
                n: reference.len(),
                mismatches: 0,
                max_ulp: 0,
                first_mismatch: None,
            };
            for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
                let d = $ulp(e, g) as u64;
                if d > report.max_ulp {
                    report.max_ulp = d;
                }
                let ok = match tol {
                    // §6.8-0010(a): determined-NaN ops compare exact-byte
                    // (payload+sign). A computed NaN's payload and sign stay
                    // uncompared, but its QUIETNESS is compared where THIS dtype's
                    // encoding admits a signaling NaN — vacuous for `f8e4m3fn`, whose
                    // single NaN encoding cannot represent the distinction. See the
                    // scalar seam above and `computed_nan_conforms`.
                    Tolerance::Exact => {
                        if determined {
                            e.to_bits() == g.to_bits()
                        } else if e.is_nan() || g.is_nan() {
                            computed_nan_conforms(
                                e.is_nan(),
                                g.is_nan(),
                                $admits_snan,
                                ($quiet)(e.to_bits()),
                                ($quiet)(g.to_bits()),
                            )
                        } else {
                            d == 0
                        }
                    }
                    Tolerance::Ulp(n) => d <= n,
                };
                if !ok {
                    report.mismatches += 1;
                    if report.first_mismatch.is_none() {
                        report.first_mismatch = Some((i, e.to_f32() as f64, g.to_f32() as f64));
                    }
                }
            }
            Ok(report)
        }
    };
}

// The quiet bit is each format's mantissa MSB: f16 has 10 mantissa bits (bit 9),
// bf16 has 7 (bit 6). Both encodings admit a signaling NaN, so §6.8-0010's quietness
// comparison is live — not vacuous — on these lanes.
narrow_diff!(
    half::f16,
    ulp_distance_f16,
    reference_f16,
    diff_f16,
    |b: u16| b & 0x0200 != 0,
    true
);
narrow_diff!(
    half::bf16,
    ulp_distance_bf16,
    reference_bf16,
    diff_bf16,
    |b: u16| b & 0x0040 != 0,
    true
);

// The composed-[`Expr`] seam over the narrow lanes — the multi-node analogue of
// `reference_f16`/`diff_f16`, reusing the same `ulp_distance_*` narrow metric.
// Fuel's advisory covers f16/bf16, so migrating those region lanes onto
// `reference_expr` needs these mirrors (the `Expr` evaluates through the same
// `eval_expr` engine in the narrow compute lane — numbers are identical to a
// hand-rolled per-node narrow reference).
macro_rules! narrow_expr {
    ($t:ty, $ulp:ident, $refr:ident, $diff:ident, $quiet:expr, $admits_snan:expr) => {
        /// Reference outputs of a composed [`Expr`] over a batch of narrow-float
        /// input rows — the narrow mirror of [`reference_expr`] (each row through
        /// [`eval_expr`] in the narrow compute lane).
        pub fn $refr(expr: &Expr, rows: &[&[$t]]) -> Result<Vec<$t>, Error> {
            rows.iter().map(|r| eval_expr::<$t>(expr, r)).collect()
        }

        /// Diff a candidate's narrow-float outputs against this reference for a
        /// composed [`Expr`]. The advisory-tolerance caveat on [`diff_expr`]
        /// applies unchanged — the caller owns the band.
        pub fn $diff(
            expr: &Expr,
            rows: &[&[$t]],
            candidate: &[$t],
            tol: Tolerance,
        ) -> Result<DiffReport, Error> {
            let reference = $refr(expr, rows)?;
            let determined = kiss_ops_vocab::decomp::expr_nan_payload_is_determined(expr);
            if candidate.len() != reference.len() {
                return Err(Error::LengthMismatch {
                    expected: reference.len(),
                    got: candidate.len(),
                });
            }
            let mut report = DiffReport {
                n: reference.len(),
                mismatches: 0,
                max_ulp: 0,
                first_mismatch: None,
            };
            for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
                let d = $ulp(e, g) as u64;
                if d > report.max_ulp {
                    report.max_ulp = d;
                }
                let ok = match tol {
                    // §6.8-0010(a): determined-NaN ops compare exact-byte
                    // (payload+sign). A computed NaN's payload and sign stay
                    // uncompared, but its QUIETNESS is compared where THIS dtype's
                    // encoding admits a signaling NaN — vacuous for `f8e4m3fn`, whose
                    // single NaN encoding cannot represent the distinction. See the
                    // scalar seam above and `computed_nan_conforms`.
                    Tolerance::Exact => {
                        if determined {
                            e.to_bits() == g.to_bits()
                        } else if e.is_nan() || g.is_nan() {
                            computed_nan_conforms(
                                e.is_nan(),
                                g.is_nan(),
                                $admits_snan,
                                ($quiet)(e.to_bits()),
                                ($quiet)(g.to_bits()),
                            )
                        } else {
                            d == 0
                        }
                    }
                    Tolerance::Ulp(n) => d <= n,
                };
                if !ok {
                    report.mismatches += 1;
                    if report.first_mismatch.is_none() {
                        report.first_mismatch = Some((i, e.to_f32() as f64, g.to_f32() as f64));
                    }
                }
            }
            Ok(report)
        }
    };
}

narrow_expr!(
    half::f16,
    ulp_distance_f16,
    reference_expr_f16,
    diff_expr_f16,
    |b: u16| b & 0x0200 != 0,
    true
);
narrow_expr!(
    half::bf16,
    ulp_distance_bf16,
    reference_expr_bf16,
    diff_expr_bf16,
    |b: u16| b & 0x0040 != 0,
    true
);

// ---- FP8 (e4m3 / e5m2) ------------------------------------------------------
//
// Same sign-magnitude total-order metric on the 8-bit pattern. After NaN
// exclusion the key is monotone for both formats: e4m3's non-NaN patterns are
// `0x00..=0x7E` / `0x80..=0xFE` (keys `0x01..=0xFE`, max distance 253), and
// e5m2's ±inf (`0x7C`/`0xFC`) sit just above max-finite in the key order — so
// the one-NaN sentinel `u8::MAX` never collides with a real distance.

#[inline]
fn key_u8(bits: u8) -> u8 {
    if bits & 0x80 != 0 {
        !bits
    } else {
        bits | 0x80
    }
}

macro_rules! fp8_diff {
    ($t:ty, $ulp:ident, $refr:ident, $diff:ident, $quiet:expr, $admits_snan:expr) => {
        /// Sign-magnitude ULP distance between two FP8 values (both-NaN → 0,
        /// one-NaN → `u8::MAX`, `+0` vs `-0` → 1).
        pub fn $ulp(a: $t, b: $t) -> u8 {
            match (a.is_nan(), b.is_nan()) {
                (true, true) => 0,
                (true, false) | (false, true) => u8::MAX,
                (false, false) => key_u8(a.to_bits()).abs_diff(key_u8(b.to_bits())),
            }
        }

        /// Reference outputs of `op` over a batch of FP8 input rows.
        pub fn $refr(op: Op, rows: &[&[$t]]) -> Result<Vec<$t>, Error> {
            rows.iter().map(|r| crate::eval_op::<$t>(op, r)).collect()
        }

        /// Diff a candidate's FP8 outputs against this reference.
        pub fn $diff(
            op: Op,
            rows: &[&[$t]],
            candidate: &[$t],
            tol: Tolerance,
        ) -> Result<DiffReport, Error> {
            let reference = $refr(op, rows)?;
            let determined = op.nan_payload_is_determined();
            if candidate.len() != reference.len() {
                return Err(Error::LengthMismatch {
                    expected: reference.len(),
                    got: candidate.len(),
                });
            }
            let mut report = DiffReport {
                n: reference.len(),
                mismatches: 0,
                max_ulp: 0,
                first_mismatch: None,
            };
            for (i, (&e, &g)) in reference.iter().zip(candidate).enumerate() {
                let d = $ulp(e, g) as u64;
                if d > report.max_ulp {
                    report.max_ulp = d;
                }
                let ok = match tol {
                    // §6.8-0010(a): determined-NaN ops compare exact-byte
                    // (payload+sign). A computed NaN's payload and sign stay
                    // uncompared, but its QUIETNESS is compared where THIS dtype's
                    // encoding admits a signaling NaN — vacuous for `f8e4m3fn`, whose
                    // single NaN encoding cannot represent the distinction. See the
                    // scalar seam above and `computed_nan_conforms`.
                    Tolerance::Exact => {
                        if determined {
                            e.to_bits() == g.to_bits()
                        } else if e.is_nan() || g.is_nan() {
                            computed_nan_conforms(
                                e.is_nan(),
                                g.is_nan(),
                                $admits_snan,
                                ($quiet)(e.to_bits()),
                                ($quiet)(g.to_bits()),
                            )
                        } else {
                            d == 0
                        }
                    }
                    Tolerance::Ulp(n) => d <= n,
                };
                if !ok {
                    report.mismatches += 1;
                    if report.first_mismatch.is_none() {
                        report.first_mismatch = Some((i, e.to_f32() as f64, g.to_f32() as f64));
                    }
                }
            }
            Ok(report)
        }
    };
}

// ⚠️ `f8e4m3fn` admits NO signaling NaN — its only NaN encoding is `S.1111.111`, so there
// is no quiet bit and no quiet/signaling distinction to compare. §6.8-0010 makes the
// quietness comparison VACUOUS there, and says a comparator MUST NOT synthesize a
// distinction the format cannot represent. The two arguments say that twice over, and
// deliberately: the predicate `|_| false` is "there is no quiet bit to read", and the
// trailing `false` is `admits_snan` — the flag that actually gates the comparison off.
// `f8e5m2` is IEEE-shaped with 2 mantissa bits, so its quiet bit is bit 1 and live.
fp8_diff!(
    crate::fp8::E4m3,
    ulp_distance_e4m3,
    reference_e4m3,
    diff_e4m3,
    |_: u8| false,
    false
);
fp8_diff!(
    crate::fp8::E5m2,
    ulp_distance_e5m2,
    reference_e5m2,
    diff_e5m2,
    |b: u8| b & 0x02 != 0,
    true
);

// ---- accumulator-parameterized reduction reference (RFC #92 direction b) ------
//
// The export seam Baracuda's step-3b on-device FP8 diff and Fuel's advisory call
// to get the per-`<acc>` reduction/scan/contraction REFERENCE for their declared
// accumulator dtype. Each pairs the reference tensor with its determinism class:
// §6.17-0007 keeps these OrderInvariantNondeterministic for ANY accumulator
// (fixing the accumulator does NOT pin the bits — float accumulation stays
// non-associative across contraction order), so a consumer MUST compare under
// tolerance, never `Tolerance::Exact`. The reference itself is the pinned profile
// of RFC #92 C3 (a defined per-cell value), not a bit golden.

use kiss_classify_vocab::Dtype;

use crate::attrs::Monoid;
use crate::bridge::{monoid_det, DetClass, Evaluated};
use crate::scalar::ScalarFloat;
use crate::tensor::View;

/// The `(compute S, accumulator A)` **reduce** reference for a runtime accumulator
/// `acc` (RFC #92 direction b), paired with its determinism class. `acc ==
/// T::DTYPE` (and `Max`/`Min` for any `acc`) return the verbatim kernel; a
/// too-narrow / non-float `acc` is a typed decline.
pub fn reference_reduce_acc<T: ScalarFloat>(
    x: &View<T>,
    monoid: Monoid,
    axes: &[usize],
    acc: Dtype,
) -> Result<Evaluated<T>, Error> {
    let t = crate::kernels::reduce_ref::<T>(x, monoid, axes, acc)?;
    Ok(Evaluated::new(t, monoid_det(monoid)))
}

/// The `(compute S, accumulator A)` **prefix_scan** reference for a runtime `acc`.
pub fn reference_prefix_scan_acc<T: ScalarFloat>(
    x: &View<T>,
    monoid: Monoid,
    axis: usize,
    exclusive: bool,
    acc: Dtype,
) -> Result<Evaluated<T>, Error> {
    let t = crate::kernels::prefix_scan_ref::<T>(x, monoid, axis, exclusive, acc)?;
    Ok(Evaluated::new(t, monoid_det(monoid)))
}

/// The `(compute S, accumulator A)` **matmul** reference for a runtime `acc`
/// (always `OrderInvariantNondeterministic` — a float contraction, §6.0-0004).
pub fn reference_matmul_acc<T: ScalarFloat>(
    a: &View<T>,
    b: &View<T>,
    acc: Dtype,
) -> Result<Evaluated<T>, Error> {
    let t = crate::tensor_ops::matmul_ref::<T>(a, b, acc)?;
    Ok(Evaluated::new(t, DetClass::OrderInvariantNondeterministic))
}

// ---- §6.18 complex differential seam + split comparator (§6.18-0017) ----------
//
// The reference for a complex op over a batch of complex-operand rows (a row is
// `&[z]` for a unary op, `&[z, w]` for a binary op), and the §6.18-0017 SPLIT
// comparator the ULP complex ops (`carg`/`clog`/`csqrt`/`cexp`) must be judged
// under: a ULP/tolerance comparison on component MAGNITUDES, combined with an
// EXACT sign-bit check on the two places a magnitude-tolerance comparator is blind
// — (a) every zero-valued component (`+0.0` vs `−0.0`) and (b) the ±π branch
// endpoint of an imaginary/angle component (the sign of a value equal to π in
// magnitude). Returning the wrong sign of a zero or the wrong ±π endpoint is
// non-conforming even when the magnitude is within tolerance.

use crate::complex::{eval_complex_op, Cplx, CplxOut};

// NOTE (sk4 naming): these public helpers keep their sk3-era `_c64`/`_c32` suffixes
// for API stability — they are historical names, NOT sk4 dtype tokens. Under the sk4
// total-width flip, pair-`f64` is now `c128` and pair-`f32` is now `c64`, so a
// suffix-accurate rename (`_c64`→`_c128`, `_c32`→`_c64`) both breaks the public API
// and collides on `c64`; it is deferred to the next breaking release. The functions
// are keyed on the `f64`/`f32` **component** type, which is unambiguous across versions.

/// Reference outputs of a complex `op` over a batch of pair-of-`f64` (sk4 `c128`,
/// sk3 `c64`) operand rows. Keyed on the `f64` component lane; `_c64` is a historical
/// suffix (see the note above), not the sk4 token.
pub fn reference_c64(op: Op, rows: &[&[Cplx<f64>]]) -> Result<Vec<CplxOut<f64>>, Error> {
    rows.iter().map(|r| eval_complex_op::<f64>(op, r)).collect()
}

/// Reference outputs of a complex `op` over a batch of pair-of-`f32` (sk4 `c64`,
/// sk3 `c32`) operand rows. Keyed on the `f32` component lane; `_c32` is a historical
/// suffix (see the note above), not the sk4 token.
pub fn reference_c32(op: Op, rows: &[&[Cplx<f32>]]) -> Result<Vec<CplxOut<f32>>, Error> {
    rows.iter().map(|r| eval_complex_op::<f32>(op, r)).collect()
}

/// One component of a §6.18-0017 split comparison: ULP tolerance on the magnitude
/// always; when `sign_exact` (a zero component, or a ±π-endpoint angle), the sign
/// bit must additionally match exactly.
fn split_component_f64(reference: f64, candidate: f64, ulp: u64, sign_exact: bool) -> bool {
    if ulp_distance_f64(reference.abs(), candidate.abs()) > ulp {
        return false;
    }
    !sign_exact || (reference.is_sign_negative() == candidate.is_sign_negative())
}

fn split_component_f32(reference: f32, candidate: f32, ulp: u64, sign_exact: bool) -> bool {
    if u64::from(ulp_distance_f32(reference.abs(), candidate.abs())) > ulp {
        return false;
    }
    !sign_exact || (reference.is_sign_negative() == candidate.is_sign_negative())
}

#[inline]
fn is_pi_endpoint_f64(x: f64) -> bool {
    x.abs() == core::f64::consts::PI
}
#[inline]
fn is_pi_endpoint_f32(x: f32) -> bool {
    x.abs() == core::f32::consts::PI
}

/// §6.18-0017 split comparator for a complex→complex ULP op (`cexp`/`clog`/
/// `csqrt`): ULP on both component magnitudes, exact sign on any zero component
/// and on an imaginary component at the ±π branch endpoint. Operates on `f64`
/// components (sk4 `c128`); `_c64` is a historical suffix, not the sk4 token.
pub fn complex_conforms_c64(reference: Cplx<f64>, candidate: Cplx<f64>, ulp: u64) -> bool {
    let re_ok = split_component_f64(reference.re, candidate.re, ulp, reference.re == 0.0);
    let im_sign_exact = reference.im == 0.0 || is_pi_endpoint_f64(reference.im);
    let im_ok = split_component_f64(reference.im, candidate.im, ulp, im_sign_exact);
    re_ok && im_ok
}

/// The `f32`-component form of [`complex_conforms_c64`] (sk4 `c64`); `_c32` is a
/// historical suffix, not the sk4 token.
pub fn complex_conforms_c32(reference: Cplx<f32>, candidate: Cplx<f32>, ulp: u64) -> bool {
    let re_ok = split_component_f32(reference.re, candidate.re, ulp, reference.re == 0.0);
    let im_sign_exact = reference.im == 0.0 || is_pi_endpoint_f32(reference.im);
    let im_ok = split_component_f32(reference.im, candidate.im, ulp, im_sign_exact);
    re_ok && im_ok
}

/// §6.18-0017 split comparator for `carg` (complex→real angle): ULP on magnitude,
/// exact sign on a zero angle and on the ±π endpoint.
pub fn arg_conforms_f64(reference: f64, candidate: f64, ulp: u64) -> bool {
    let sign_exact = reference == 0.0 || is_pi_endpoint_f64(reference);
    split_component_f64(reference, candidate, ulp, sign_exact)
}

/// `f32` form of [`arg_conforms_f64`].
pub fn arg_conforms_f32(reference: f32, candidate: f32, ulp: u64) -> bool {
    let sign_exact = reference == 0.0 || is_pi_endpoint_f32(reference);
    split_component_f32(reference, candidate, ulp, sign_exact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_signed_zero_is_one() {
        assert_eq!(ulp_distance_f64(0.0, -0.0), 1);
        assert_eq!(ulp_distance_f32(0.0, -0.0), 1);
    }

    #[test]
    fn ulp_distance_both_nan_is_zero_one_nan_is_max() {
        assert_eq!(ulp_distance_f64(f64::NAN, f64::NAN), 0);
        assert_eq!(ulp_distance_f64(f64::NAN, 1.0), u64::MAX);
    }

    #[test]
    fn ulp_distance_adjacent_is_one() {
        let a = 1.0f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance_f64(a, b), 1);
    }

    #[test]
    fn diff_detects_a_planted_error() {
        // A candidate that matches the reference except one row.
        let rows: [&[f64]; 3] = [&[1.0], &[2.0], &[3.0]];
        let mut cand = reference_f64(Op::Sqr, &rows).unwrap();
        cand[1] = 999.0;
        let r = diff_f64(Op::Sqr, &rows, &cand, Tolerance::Exact).unwrap();
        assert!(!r.conforms());
        assert_eq!(r.mismatches, 1);
        assert_eq!(r.first_mismatch.unwrap().0, 1);
    }

    #[test]
    fn exact_binds_a_determined_nan_payload_but_not_a_computed_one() {
        // §6.8-0010(a): a determined-NaN op's output is a MOVED value, so Tolerance::
        // Exact must bind payload+sign bit-for-bit — a wrong moved payload fails even
        // at 0 ULP (this is a false-PASS under the old payload-blind Exact; the born-
        // red for the fix). min_prop on two distinct-payload NaNs propagates operand a
        // via a select-move.
        let a = f32::from_bits(0x7FC0_1234); // qNaN, payload …1234
        let b = f32::from_bits(0x7FC0_5678); // qNaN, payload …5678
        let rows: [&[f32]; 1] = [&[a, b]];
        let reference = reference_f32(Op::MinProp, &rows).unwrap();
        assert_eq!(
            reference[0].to_bits(),
            0x7FC0_1234,
            "min_prop moves operand a"
        );

        let wrong = [f32::from_bits(0x7FC0_FFFF)]; // a NaN, but the wrong payload
        assert!(
            !diff_f32(Op::MinProp, &rows, &wrong, Tolerance::Exact)
                .unwrap()
                .conforms(),
            "a determined-NaN op must REJECT a wrong moved payload at Exact"
        );
        let right = [f32::from_bits(0x7FC0_1234)];
        assert!(
            diff_f32(Op::MinProp, &rows, &right, Tolerance::Exact)
                .unwrap()
                .conforms(),
            "the correct moved payload conforms"
        );

        // COMPUTED contrast — the module doc's sm_89-vs-x86 witness: add(inf, -inf) is
        // a COMPUTED NaN whose payload is architectural, so Exact stays payload-blind
        // and a different-payload candidate still conforms. Bit-comparing HERE would
        // false-FAIL a conformant implementation, which is why the fix is op-aware.
        let crows: [&[f32]; 1] = [&[f32::INFINITY, f32::NEG_INFINITY]];
        assert!(reference_f32(Op::Add, &crows).unwrap()[0].is_nan());
        let other = [f32::from_bits(0x7FC0_ABCD)];
        assert!(
            diff_f32(Op::Add, &crows, &other, Tolerance::Exact)
                .unwrap()
                .conforms(),
            "a computed-NaN op must stay payload-blind at Exact"
        );

        // the composed-Expr seam mirrors it: a select-region is determined, an
        // add-region is computed (determined iff EVERY node is payload-determining).
        use kiss_ops_vocab::decomp::parse;
        let sel = parse("select(cmp_ne(a, a), a, b)").unwrap();
        assert!(
            !diff_expr_f32(&sel, &rows, &wrong, Tolerance::Exact)
                .unwrap()
                .conforms(),
            "a determined select-region binds the moved payload"
        );
        let plus = parse("add(a, b)").unwrap();
        assert!(
            diff_expr_f32(&plus, &crows, &other, Tolerance::Exact)
                .unwrap()
                .conforms(),
            "a computed add-region stays payload-blind"
        );
    }

    /// KISS-CONFORM-6.8-0010's COMPUTED arm, mirroring KISS #352's `compare_nan_output`:
    /// match iff the observed result is also NaN and, **where the dtype's encoding admits
    /// a signaling NaN, agrees in QUIETNESS with the expected value**. Payload and sign
    /// stay uncompared — the quiet bit is carved explicitly OUT of that exemption, because
    /// KISS-OPS-6.16-0010 makes delivering a quiet NaN a MUST for a decomposition
    /// containing arithmetic, and exempting the whole payload would leave that clause
    /// undetectable by any conformant harness.
    #[test]
    fn exact_compares_quietness_on_a_computed_nan_but_never_payload_or_sign() {
        const QUIET: u32 = 0x0040_0000;
        let computed: [&[f32]; 1] = [&[f32::INFINITY, f32::NEG_INFINITY]];
        let e = reference_f32(Op::Add, &computed).unwrap()[0];
        assert!(
            e.is_nan() && e.to_bits() & QUIET != 0,
            "add(inf,-inf) mints a QUIET NaN, so quietness is the axis under test"
        );

        // The architectural-payload carve-out STANDS: a computed NaN's payload and sign
        // are still not compared (the sm_89-vs-x86 witness in this module's docs).
        for (bits, why) in [
            (0x7FC0_ABCDu32, "a different PAYLOAD still conforms"),
            (0xFFC0_0001, "a different SIGN still conforms"),
        ] {
            assert!(
                diff_f32(
                    Op::Add,
                    &computed,
                    &[f32::from_bits(bits)],
                    Tolerance::Exact
                )
                .unwrap()
                .conforms(),
                "{why}"
            );
        }

        // …but QUIETNESS is compared, and f32's encoding admits a signaling NaN, so the
        // comparison is NOT vacuous here. A signaling result where quiet is expected is a
        // detectable §6.16-0010 violation and MUST fail.
        let signaling = f32::from_bits(0x7F80_0001);
        assert!(
            signaling.is_nan() && signaling.to_bits() & QUIET == 0,
            "0x7F800001 is a SIGNALING NaN"
        );
        assert!(
            !diff_f32(Op::Add, &computed, &[signaling], Tolerance::Exact)
                .unwrap()
                .conforms(),
            "a signaling NaN where §6.16-0010 requires quiet MUST be a mismatch"
        );
    }

    #[test]
    fn diff_conforms_within_ulp_tolerance() {
        let rows: [&[f64]; 2] = [&[0.5], &[1.0]];
        let reference = reference_f64(Op::Erf, &rows).unwrap();
        // Perturb the candidate by 1 ULP each; a 4-ULP tolerance still conforms.
        let cand: Vec<f64> = reference
            .iter()
            .map(|&x| f64::from_bits(x.to_bits() + 1))
            .collect();
        let r = diff_f64(Op::Erf, &rows, &cand, Tolerance::Ulp(4)).unwrap();
        assert!(r.conforms());
        assert_eq!(r.max_ulp, 1);
    }

    #[test]
    fn ulp_distance_straddling_zero_is_small() {
        // Golden edge from Fuel's fkc/verify/ulp.rs (validates two independent
        // impls agree): smallest +subnormal → +0 → -0 → smallest -subnormal = 3.
        let pos_min = f64::from_bits(1);
        let neg_min = f64::from_bits(0x8000_0000_0000_0001);
        assert_eq!(ulp_distance_f64(pos_min, neg_min), 3);
    }

    #[test]
    fn fp8_seam_edges() {
        use crate::fp8::{E4m3, E5m2};
        // signed zero is 1 ULP on 8 bits too.
        assert_eq!(
            ulp_distance_e4m3(E4m3::from_bits(0x00), E4m3::from_bits(0x80)),
            1
        );
        assert_eq!(
            ulp_distance_e5m2(E5m2::from_bits(0x00), E5m2::from_bits(0x80)),
            1
        );
        // e4m3: both NaN codes (0x7F / 0xFF) match at 0; one-NaN is the sentinel.
        assert_eq!(
            ulp_distance_e4m3(E4m3::from_bits(0x7F), E4m3::from_bits(0xFF)),
            0
        );
        assert_eq!(ulp_distance_e4m3(E4m3::from_bits(0x7F), E4m3::ONE), u8::MAX);
        // e4m3 full-range: -448 (0xFE) to +448 (0x7E) spans 253 — below the sentinel.
        assert_eq!(
            ulp_distance_e4m3(E4m3::from_bits(0xFE), E4m3::from_bits(0x7E)),
            253
        );
        // adjacent codes are 1 ULP.
        assert_eq!(ulp_distance_e4m3(E4m3::ONE, E4m3::from_bits(0x39)), 1);
        // e5m2: +inf (0x7C) matches itself at 0, sits 1 ULP above max finite
        // (0x7B = 57344), and any-NaN (exp=31, m≠0) hits the sentinel.
        assert_eq!(
            ulp_distance_e5m2(E5m2::from_bits(0x7C), E5m2::from_bits(0x7C)),
            0
        );
        assert_eq!(
            ulp_distance_e5m2(E5m2::from_bits(0x7C), E5m2::from_bits(0x7B)),
            1
        );
        assert_eq!(
            ulp_distance_e5m2(E5m2::from_bits(0x7D), E5m2::from_bits(0xFE)),
            0
        );
        assert_eq!(ulp_distance_e5m2(E5m2::from_bits(0x7E), E5m2::ONE), u8::MAX);
    }

    #[test]
    fn fp8_diff_run_catches_planted_error() {
        use crate::fp8::E4m3;
        let one = E4m3::from_f32(1.0);
        let two = E4m3::from_f32(2.0);
        let rows: [&[E4m3]; 2] = [&[one], &[two]];
        let reference = reference_e4m3(Op::Sqr, &rows).unwrap();
        // a 1-code perturbation is caught at Exact, tolerated at Ulp(1).
        let cand: Vec<E4m3> = reference
            .iter()
            .map(|&x| E4m3::from_bits(x.to_bits() + 1))
            .collect();
        assert!(!diff_e4m3(Op::Sqr, &rows, &cand, Tolerance::Exact)
            .unwrap()
            .conforms());
        assert!(diff_e4m3(Op::Sqr, &rows, &cand, Tolerance::Ulp(1))
            .unwrap()
            .conforms());
    }

    #[test]
    fn narrow_float_seam_covers_f16_bf16() {
        use half::{bf16, f16};
        // signed zero is 1 ULP in the narrow lattice too.
        assert_eq!(ulp_distance_f16(f16::from_f32(0.0), f16::from_f32(-0.0)), 1);
        assert_eq!(
            ulp_distance_bf16(bf16::from_f32(0.0), bf16::from_f32(-0.0)),
            1
        );
        // an f16 differential run: a planted 1-ULP error is caught at Exact,
        // tolerated at Ulp(1).
        let rows: [&[f16]; 2] = [&[f16::from_f32(1.0)], &[f16::from_f32(2.0)]];
        let reference = reference_f16(Op::Sqr, &rows).unwrap();
        let cand: Vec<f16> = reference
            .iter()
            .map(|&x| f16::from_bits(x.to_bits() + 1))
            .collect();
        assert!(!diff_f16(Op::Sqr, &rows, &cand, Tolerance::Exact)
            .unwrap()
            .conforms());
        assert!(diff_f16(Op::Sqr, &rows, &cand, Tolerance::Ulp(1))
            .unwrap()
            .conforms());
    }

    // ---- composed-expression seam --------------------------------------------

    #[test]
    fn diff_expr_exact_chain_is_byte_exact() {
        use kiss_ops_vocab::decomp::parse;
        // (a*a) - (b*b): a pure exact-op chain => DetClass::ExactByte.
        let e = parse("sub(mul(a, a), mul(b, b))").unwrap();
        let rows: [&[f64]; 2] = [&[3.0, 2.0], &[5.0, 1.0]];
        // 3^2 - 2^2 = 5 ; 5^2 - 1^2 = 24 (both exact in f64).
        let reference = reference_expr(&e, &rows).unwrap();
        assert_eq!(reference, [5.0, 24.0]);
        // an exact candidate conforms at Exact with 0 ULP.
        let r = diff_expr(&e, &rows, &reference, Tolerance::Exact).unwrap();
        assert!(r.conforms() && r.max_ulp == 0);
        // a planted 1-ULP error on row 1 is caught at Exact, tolerated at Ulp(1).
        let mut cand = reference.clone();
        cand[1] = f64::from_bits(cand[1].to_bits() + 1);
        let bad = diff_expr(&e, &rows, &cand, Tolerance::Exact).unwrap();
        assert!(!bad.conforms());
        assert_eq!(bad.mismatches, 1);
        assert_eq!(bad.first_mismatch.unwrap().0, 1);
        assert!(diff_expr(&e, &rows, &cand, Tolerance::Ulp(1))
            .unwrap()
            .conforms());
    }

    #[test]
    fn diff_expr_agrees_with_single_op_seam() {
        use kiss_ops_vocab::decomp::parse;
        // mul(x,x) is exactly Sqr's §6.13 decomposition, so the composed seam and the
        // per-op seam MUST return the identical reference (cross-check of the two seams).
        let e = parse("mul(x, x)").unwrap();
        let rows: [&[f64]; 3] = [&[1.0], &[2.0], &[3.0]];
        let via_expr = reference_expr(&e, &rows).unwrap();
        let via_op = reference_f64(Op::Sqr, &rows).unwrap();
        assert_eq!(via_expr, via_op);
        assert_eq!(via_expr, [1.0, 4.0, 9.0]);
    }

    #[test]
    fn diff_expr_transcendental_chain_applies_caller_ulp_band() {
        use kiss_ops_vocab::decomp::parse;
        // exp(x) - exp(-x): a transcendental-bearing chain. We do NOT hard-code the
        // transcendental value (libm-version-sensitive); we perturb the reference by
        // 1 ULP and show the caller-supplied band governs (the advisory mechanism),
        // exactly like `diff_conforms_within_ulp_tolerance` does for the per-op seam.
        let e = parse("sub(exp(x), exp(neg(x)))").unwrap();
        let rows: [&[f64]; 2] = [&[0.5], &[1.0]];
        let reference = reference_expr(&e, &rows).unwrap();
        let cand: Vec<f64> = reference
            .iter()
            .map(|&x| f64::from_bits(x.to_bits() + 1))
            .collect();
        // every row is 1 ULP off: caught at Exact, tolerated at Ulp(4).
        let exact = diff_expr(&e, &rows, &cand, Tolerance::Exact).unwrap();
        assert!(!exact.conforms());
        assert_eq!(exact.mismatches, 2);
        assert_eq!(exact.first_mismatch.unwrap().0, 0);
        let toler = diff_expr(&e, &rows, &cand, Tolerance::Ulp(4)).unwrap();
        assert!(toler.conforms());
        assert_eq!(toler.max_ulp, 1);
    }

    #[test]
    fn diff_expr_length_mismatch_is_typed_error() {
        use kiss_ops_vocab::decomp::parse;
        let e = parse("mul(x, x)").unwrap();
        let rows: [&[f64]; 2] = [&[1.0], &[2.0]];
        let cand = [1.0]; // one short of the 2 reference rows
        assert_eq!(
            diff_expr(&e, &rows, &cand, Tolerance::Exact),
            Err(Error::LengthMismatch {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn diff_expr_propagates_eval_error() {
        use kiss_ops_vocab::decomp::parse;
        // add(a, b) reads input(1), but the rows supply only input(0) => MissingInput.
        let e = parse("add(a, b)").unwrap();
        let rows: [&[f64]; 1] = [&[1.0]];
        assert_eq!(reference_expr(&e, &rows), Err(Error::MissingInput(1)));
        assert_eq!(
            diff_expr(&e, &rows, &[0.0], Tolerance::Exact),
            Err(Error::MissingInput(1))
        );
    }

    #[test]
    fn diff_expr_f32_mirror_is_byte_exact() {
        use kiss_ops_vocab::decomp::parse;
        let e = parse("sub(mul(a, a), mul(b, b))").unwrap();
        let rows: [&[f32]; 1] = [&[3.0, 2.0]];
        let reference = reference_expr_f32(&e, &rows).unwrap();
        assert_eq!(reference, [5.0f32]); // 3^2 - 2^2 = 5
        let r = diff_expr_f32(&e, &rows, &reference, Tolerance::Exact).unwrap();
        assert!(r.conforms() && r.max_ulp == 0);
    }

    #[test]
    fn composed_expr_seam_narrow_f16_bf16() {
        use half::{bf16, f16};
        use kiss_ops_vocab::decomp::parse;
        // a*a - b*b over the narrow lanes: a=3,b=2 => 5, exactly representable in
        // both f16 and bf16, so the exact-op chain is byte-exact.
        let e = parse("sub(mul(a, a), mul(b, b))").unwrap();
        // f16 lane.
        let rf: [&[f16]; 1] = [&[f16::from_f32(3.0), f16::from_f32(2.0)]];
        let reff = reference_expr_f16(&e, &rf).unwrap();
        assert_eq!(reff, [f16::from_f32(5.0)]);
        // a planted 1-ULP error is caught at Exact, tolerated at Ulp(1).
        let cand: Vec<f16> = reff
            .iter()
            .map(|&x| f16::from_bits(x.to_bits() + 1))
            .collect();
        assert!(!diff_expr_f16(&e, &rf, &cand, Tolerance::Exact)
            .unwrap()
            .conforms());
        assert!(diff_expr_f16(&e, &rf, &cand, Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        // bf16 lane.
        let rb: [&[bf16]; 1] = [&[bf16::from_f32(3.0), bf16::from_f32(2.0)]];
        let refb = reference_expr_bf16(&e, &rb).unwrap();
        assert_eq!(refb, [bf16::from_f32(5.0)]);
        let cb: Vec<bf16> = refb
            .iter()
            .map(|&x| bf16::from_bits(x.to_bits() + 1))
            .collect();
        assert!(!diff_expr_bf16(&e, &rb, &cb, Tolerance::Exact)
            .unwrap()
            .conforms());
        assert!(diff_expr_bf16(&e, &rb, &cb, Tolerance::Ulp(1))
            .unwrap()
            .conforms());
        // a MissingInput propagates on the narrow lanes too.
        let e2 = parse("add(a, b)").unwrap();
        let short: [&[f16]; 1] = [&[f16::from_f32(1.0)]];
        assert_eq!(reference_expr_f16(&e2, &short), Err(Error::MissingInput(1)));
    }

    // ---- FP8 exhaustive key-monotonicity ---------------------------------------

    macro_rules! fp8_total_order_case {
        ($t:ty, $ulp:ident, $expected_nonnan:expr, $expected_max:expr) => {{
            // All non-NaN byte patterns of the format.
            let nonnan: Vec<u8> = (0u16..=255)
                .map(|b| b as u8)
                .filter(|&b| !<$t>::from_bits(b).is_nan())
                .collect();
            assert_eq!(nonnan.len(), $expected_nonnan);

            // key_u8 is a correct total order: sort the codes by key, then assert the
            // decoded value is monotone non-decreasing and the keys are strictly
            // increasing (i.e. distinct — a genuine total order, no collisions).
            let mut by_key = nonnan.clone();
            by_key.sort_by_key(|&b| key_u8(b));
            for w in by_key.windows(2) {
                assert!(
                    key_u8(w[0]) < key_u8(w[1]),
                    "keys must be a strict total order"
                );
                assert!(
                    <$t>::from_bits(w[0]).to_f32() <= <$t>::from_bits(w[1]).to_f32(),
                    "value must be monotone across the key order"
                );
            }

            // The one-NaN sentinel u8::MAX is unreachable by any finite/inf pair: the
            // widest real ULP distance over ALL non-NaN pairs stays strictly below it.
            let mut max_dist = 0u8;
            for &a in &nonnan {
                for &b in &nonnan {
                    let d = $ulp(<$t>::from_bits(a), <$t>::from_bits(b));
                    if d > max_dist {
                        max_dist = d;
                    }
                }
            }
            assert_eq!(max_dist, $expected_max);
            assert!(
                max_dist < u8::MAX,
                "a real distance must never reach the one-NaN sentinel"
            );
        }};
    }

    #[test]
    fn fp8_key_is_total_order_and_sentinel_unreachable() {
        use crate::fp8::{E4m3, E5m2};
        // e4m3: 2 NaN codes (0x7F/0xFF, mask (b & 0x7F) == 0x7F) => 254 non-NaN,
        // keys span [1, 254], so the widest real distance is 253 (-448 <-> +448).
        fp8_total_order_case!(E4m3, ulp_distance_e4m3, 254, 253);
        // e5m2: 6 NaN codes (exp=31 & mant!=0 => 0x7D/0x7E/0x7F/0xFD/0xFE/0xFF)
        // => 250 non-NaN, keys span [3, 252], widest real distance 249 (-inf <-> +inf).
        fp8_total_order_case!(E5m2, ulp_distance_e5m2, 250, 249);
    }
}
