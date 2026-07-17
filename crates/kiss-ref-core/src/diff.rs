//! The **differential export seam** — the API other implementations dev-depend
//! on to test against this reference.
//!
//! Both the KISS-Conform harness and Baracuda's on-device differential harness
//! need the same thing: feed a batch of input vectors, get the reference's
//! outputs, and compare a candidate's outputs under a ULP tolerance. This module
//! is that seam. Distances use a **sign-magnitude IEEE total-order** ULP metric
//! (faithful near zero and across the signed-zero boundary — unlike a relative
//! epsilon), so signed zero reads as 1 ULP and both-NaN reads as 0.

extern crate alloc;
use alloc::vec::Vec;

use kiss_ops_vocab::Op;

use crate::{eval_op, Error};

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
            if x >= y {
                x - y
            } else {
                y - x
            }
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
            if x >= y {
                x - y
            } else {
                y - x
            }
        }
    }
}

/// A comparison tolerance for a differential run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tolerance {
    /// Bitwise-identical (0 ULP) — for the exact ops (integer, comparison,
    /// `select`, rounding, signed-zero pins).
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
            Tolerance::Exact => d == 0,
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
            Tolerance::Exact => d == 0,
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
    ($t:ty, $ulp:ident, $refr:ident, $diff:ident) => {
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
                    Tolerance::Exact => d == 0,
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

narrow_diff!(half::f16, ulp_distance_f16, reference_f16, diff_f16);
narrow_diff!(half::bf16, ulp_distance_bf16, reference_bf16, diff_bf16);

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
    fn narrow_float_seam_covers_f16_bf16() {
        use half::{bf16, f16};
        // signed zero is 1 ULP in the narrow lattice too.
        assert_eq!(ulp_distance_f16(f16::from_f32(0.0), f16::from_f32(-0.0)), 1);
        assert_eq!(ulp_distance_bf16(bf16::from_f32(0.0), bf16::from_f32(-0.0)), 1);
        // an f16 differential run: a planted 1-ULP error is caught at Exact,
        // tolerated at Ulp(1).
        let rows: [&[f16]; 2] = [&[f16::from_f32(1.0)], &[f16::from_f32(2.0)]];
        let reference = reference_f16(Op::Sqr, &rows).unwrap();
        let cand: Vec<f16> = reference
            .iter()
            .map(|&x| f16::from_bits(x.to_bits() + 1))
            .collect();
        assert!(!diff_f16(Op::Sqr, &rows, &cand, Tolerance::Exact).unwrap().conforms());
        assert!(diff_f16(Op::Sqr, &rows, &cand, Tolerance::Ulp(1)).unwrap().conforms());
    }
}
