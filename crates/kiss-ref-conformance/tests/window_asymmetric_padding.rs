// SPDX-License-Identifier: MIT OR Apache-2.0
//! KISS-OPS-6.13-0010 with independent `padding_lo` / `padding_hi` (board 426 5 "A", Breaking): the
//! output extent is `floor((n + p_lo + p_hi - D) / s) + 1` (`0` if no window fits), and tap `j` of
//! window `o` reads operand position `o*s - p_lo + j*d`, out of bounds being skipped. Checked here against
//! a brute-force enumeration of windows over the padded range, not against the closed form.

use kiss_ref_core::tensor::Tensor;
use kiss_ref_core::window::{avg_pool, im2col, max_pool};

/// The windows by enumeration: window `o` is placed at padded offset `o*s` and fits iff its last tap lies
/// within `[-p_lo, n + p_hi)`; its taps are at `o*s - p_lo + j*d`, `None` when out of `[0, n)`.
fn brute(n: usize, k: usize, s: usize, d: usize, lo: usize, hi: usize) -> Vec<Vec<Option<usize>>> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o * s + d * (k - 1) < n + lo + hi {
        out.push(
            (0..k)
                .map(|j| {
                    let pos = (o * s + j * d) as i64 - lo as i64;
                    (0 <= pos && pos < n as i64).then_some(pos as usize)
                })
                .collect(),
        );
        o += 1;
    }
    out
}

/// One grid cell: `im2col`, `max_pool` and `avg_pool` (count_include_pad) against the enumerated windows.
fn check_cell(x: &Tensor<f64>, p: [usize; 6]) {
    let [n, k, s, d, lo, hi] = p;
    let want = brute(n, k, s, d, lo, hi);
    let got = im2col(&x.view(), &[0], &[k], &[s], &[lo], &[hi], &[d]).unwrap();
    assert_eq!(got.shape(), &[want.len(), k], "{p:?}");
    let val = |q: &Option<usize>| q.map_or(0.0, |q| (q + 1) as f64);
    let flat: Vec<f64> = want.iter().flatten().map(val).collect();
    assert_eq!(got.as_slice(), &flat[..], "{p:?}");
    fn sums(w: &[Option<usize>]) -> impl Iterator<Item = f64> + '_ {
        w.iter().flatten().map(|q| (*q + 1) as f64)
    }
    let mp = max_pool(&x.view(), &[0], &[k], &[s], &[lo], &[hi], &[d]).unwrap();
    let wmax: Vec<f64> = want
        .iter()
        .map(|w| sums(w).fold(f64::NEG_INFINITY, f64::max))
        .collect();
    assert_eq!(mp.as_slice(), &wmax[..], "max {p:?}");
    let ap = avg_pool(&x.view(), &[0], &[k], &[s], &[lo], &[hi], &[d], true).unwrap();
    let wavg: Vec<f64> = want
        .iter()
        .map(|w| sums(w).sum::<f64>() / k as f64)
        .collect();
    assert_eq!(ap.as_slice(), &wavg[..], "avg {p:?}");
}

#[test]
fn asymmetric_window_matches_brute_force_over_the_grid() {
    let (mut checked, mut asym) = (0u32, 0u32);
    for n in 1..=7usize {
        let x = Tensor::from_vec((1..=n).map(|v| v as f64).collect(), &[n]).unwrap();
        for (k, d, s) in itertools3() {
            for lo in 0..=3usize {
                for hi in 0..=3usize {
                    check_cell(&x, [n, k, s, d, lo, hi]);
                    checked += 1;
                    asym += (lo != hi) as u32;
                }
            }
        }
    }
    assert!(checked > 1000 && asym > 800, "{checked} {asym}");
}

fn itertools3() -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for k in 1..=4 {
        for d in 1..=3 {
            for s in 1..=3 {
                v.push((k, d, s));
            }
        }
    }
    v
}

#[test]
fn the_two_sides_are_not_interchangeable() {
    // n = 3, k = 2, s = 1: low padding 2 puts two zero taps before the operand, high padding 2 after it.
    let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], &[3]).unwrap();
    let lo = im2col(&x.view(), &[0], &[2], &[1], &[2], &[0], &[1]).unwrap();
    let hi = im2col(&x.view(), &[0], &[2], &[1], &[0], &[2], &[1]).unwrap();
    assert_eq!(lo.as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(hi.as_slice(), &[1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0, 0.0][..]);
    assert_ne!(lo.as_slice(), hi.as_slice());
    // "same" padding at an even window needs the asymmetric pair: k = 2, s = 1, lo 0, hi 1 keeps extent 3.
    let same = im2col(&x.view(), &[0], &[2], &[1], &[0], &[1], &[1]).unwrap();
    assert_eq!(same.shape(), &[3, 2]);
}
