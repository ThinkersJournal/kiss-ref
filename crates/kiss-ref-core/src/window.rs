//! The window family (§6.13): `avg_pool`, `max_pool`, `im2col`.
//!
//! Each pools/gathers over a sliding window across a set of spatial `axes` with
//! per-axis `(kernel, stride, padding, dilation)`. The window source position for
//! output spatial index `o` and window tap `t` is `o·stride + t·dilation − padding`
//! (§6.13 im2col mapping); positions outside the input are out of bounds.
//!
//! - `avg_pool` — §6.13 `reduce_mean` over the window; divisor is the full window
//!   size when `count_include_pad`, else the count of in-bounds taps. Carries a
//!   float sum → order-invariant/nondeterministic (§6.0-0004).
//! - `max_pool` — §6.13 `reduce(max)` over the window; OOB taps skipped, NaN
//!   propagates; exact-byte.
//! - `im2col` — §6.13 closed-form zero-filled gather: output appends the window
//!   (tap) axes after the output-spatial axes; OOB taps are zero-filled.
//!
//! Float lane (`T: ScalarFloat`); every access checked, typed [`Error`], never panics.

extern crate alloc;
use alloc::vec::Vec;

use kiss_ops_vocab::Op;

use crate::resolve::eval_op;
use crate::scalar::ScalarFloat;
use crate::tensor::{alloc_exact, numel, Odometer, Tensor, View, MAX_RANK};
use crate::Error;

/// The output extent of a pooled axis: `floor((in + 2·pad − dilation·(k−1) − 1) /
/// stride) + 1`, or `0` when the effective window is larger than the padded input.
fn pool_out_dim(in_d: usize, k: usize, s: usize, p: usize, dil: usize) -> Result<usize, Error> {
    if s == 0 || k == 0 {
        return Err(Error::ShapeMismatch { expected: 1, got: 0 });
    }
    let eff = dil.checked_mul(k - 1).and_then(|x| x.checked_add(1)).ok_or(Error::ShapeOverflow)?;
    // Checked doubling of the padding, like every other multiply in the reference
    // (never-panic contract) — a raw `2 * p` would overflow on a pathological pad.
    let padded = p
        .checked_mul(2)
        .and_then(|pp| in_d.checked_add(pp))
        .ok_or(Error::ShapeOverflow)?;
    if padded < eff {
        return Ok(0);
    }
    ((padded - eff) / s).checked_add(1).ok_or(Error::ShapeOverflow)
}

/// Validate the parallel window-parameter arrays and build the output shape (input
/// shape with each spatial axis replaced by its pooled extent).
fn window_out_shape(
    in_shape: &[usize],
    axes: &[usize],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Result<([usize; MAX_RANK], usize), Error> {
    let rank = in_shape.len();
    let n = axes.len();
    if kernel.len() != n || stride.len() != n || padding.len() != n || dilation.len() != n {
        return Err(Error::ShapeMismatch { expected: n, got: kernel.len() });
    }
    let mut out = [1usize; MAX_RANK];
    out[..rank].copy_from_slice(in_shape);
    for (i, &ax) in axes.iter().enumerate() {
        if ax >= rank {
            return Err(Error::AxisOutOfRange { axis: ax, rank });
        }
        out[ax] = pool_out_dim(in_shape[ax], kernel[i], stride[i], padding[i], dilation[i])?;
    }
    Ok((out, rank))
}

/// The source spatial position of window tap `t` at output position `o`:
/// `o·stride + t·dilation − padding`. Returns `None` (out of bounds) if negative
/// or `>= extent`.
fn tap_pos(o: usize, t: usize, stride: usize, dilation: usize, padding: usize, extent: usize) -> Option<usize> {
    let pos = (o as isize)
        .wrapping_mul(stride as isize)
        .wrapping_add((t as isize).wrapping_mul(dilation as isize))
        .wrapping_sub(padding as isize);
    if pos < 0 || pos >= extent as isize {
        None
    } else {
        Some(pos as usize)
    }
}

/// `avg_pool` — §6.13 `reduce_mean` over the window. `count_include_pad` selects
/// the divisor: the full window size vs. the count of in-bounds taps.
#[allow(clippy::too_many_arguments)]
pub fn avg_pool<T: ScalarFloat>(
    x: &View<T>,
    axes: &[usize],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    count_include_pad: bool,
) -> Result<Tensor<T>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    let (out_shape_buf, _) = window_out_shape(in_shape, axes, kernel, stride, padding, dilation)?;
    let out_shape = &out_shape_buf[..rank];
    let count_total = numel(kernel)?;

    let count = numel(out_shape)?;
    let mut data: Vec<T> = alloc_exact(count)?;
    let mut src = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        let mut acc = T::ZERO;
        let mut valid = 0usize;
        let mut tod = Odometer::new(kernel)?;
        while let Some(tap) = tod.next_coord() {
            src[..rank].copy_from_slice(oc);
            let mut in_bounds = true;
            for (i, &ax) in axes.iter().enumerate() {
                match tap_pos(oc[ax], tap[i], stride[i], dilation[i], padding[i], in_shape[ax]) {
                    Some(p) => src[ax] = p,
                    None => {
                        in_bounds = false;
                        break;
                    }
                }
            }
            if in_bounds {
                acc = eval_op(Op::Add, &[acc, x.read(&src[..rank])?])?;
                valid += 1;
            }
        }
        let divisor = if count_include_pad { count_total } else { valid };
        data.push(eval_op(Op::Div, &[acc, T::from_f64(divisor as f64)])?);
    }
    Tensor::from_vec(data, out_shape)
}

/// `max_pool` — §6.13 `reduce(max)` over the window; OOB taps skipped, NaN
/// propagates. A wholly-out-of-bounds window yields the max identity (`−inf`).
pub fn max_pool<T: ScalarFloat>(
    x: &View<T>,
    axes: &[usize],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Result<Tensor<T>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    let (out_shape_buf, _) = window_out_shape(in_shape, axes, kernel, stride, padding, dilation)?;
    let out_shape = &out_shape_buf[..rank];

    let count = numel(out_shape)?;
    let mut data: Vec<T> = alloc_exact(count)?;
    let ident = T::from_f64(f64::NEG_INFINITY);
    let mut src = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        let mut acc = ident;
        let mut tod = Odometer::new(kernel)?;
        while let Some(tap) = tod.next_coord() {
            src[..rank].copy_from_slice(oc);
            let mut in_bounds = true;
            for (i, &ax) in axes.iter().enumerate() {
                match tap_pos(oc[ax], tap[i], stride[i], dilation[i], padding[i], in_shape[ax]) {
                    Some(p) => src[ax] = p,
                    None => {
                        in_bounds = false;
                        break;
                    }
                }
            }
            if in_bounds {
                acc = eval_op(Op::MaxProp, &[acc, x.read(&src[..rank])?])?;
            }
        }
        data.push(acc);
    }
    Tensor::from_vec(data, out_shape)
}

/// `im2col` — §6.13 closed-form zero-filled gather. Output shape = the pooled
/// output spatial shape (the "patch" axes) with the window (tap) axes appended;
/// `out[patch.., tap..] = x[patch with spatial → o·stride + t·dilation − padding]`,
/// zero-filled where a tap is out of bounds. Exact-byte (pure data movement).
pub fn im2col<T: ScalarFloat>(
    x: &View<T>,
    axes: &[usize],
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
) -> Result<Tensor<T>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    let n = axes.len();
    let (patch_shape_buf, _) = window_out_shape(in_shape, axes, kernel, stride, padding, dilation)?;
    let patch_shape = &patch_shape_buf[..rank];

    let out_rank = rank + n;
    if out_rank > MAX_RANK {
        return Err(Error::RankExceeded { rank: out_rank, max: MAX_RANK });
    }
    let mut out_shape = [1usize; MAX_RANK];
    out_shape[..rank].copy_from_slice(patch_shape);
    out_shape[rank..out_rank].copy_from_slice(&kernel[..n]);
    let out_shape = &out_shape[..out_rank];

    let count = numel(out_shape)?;
    let mut data: Vec<T> = alloc_exact(count)?;
    let mut src = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        // oc = patch coord (first `rank`) ++ tap coord (last `n`)
        src[..rank].copy_from_slice(&oc[..rank]);
        let mut in_bounds = true;
        for (i, &ax) in axes.iter().enumerate() {
            match tap_pos(oc[ax], oc[rank + i], stride[i], dilation[i], padding[i], in_shape[ax]) {
                Some(p) => src[ax] = p,
                None => {
                    in_bounds = false;
                    break;
                }
            }
        }
        data.push(if in_bounds { x.read(&src[..rank])? } else { T::ZERO });
    }
    Tensor::from_vec(data, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn avg_pool_1d_no_pad() {
        // [1,2,3,4], kernel 2 stride 2 → [(1+2)/2, (3+4)/2] = [1.5, 3.5]
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let y = avg_pool(&x.view(), &[0], &[2], &[2], &[0], &[1], false).unwrap();
        assert_eq!(y.shape(), &[2]);
        assert_eq!(y.as_slice(), &[1.5, 3.5]);
    }

    #[test]
    fn max_pool_1d_stride1() {
        // [1,3,2,5], kernel 2 stride 1 → [max(1,3),max(3,2),max(2,5)] = [3,3,5]
        let x = t(&[1.0, 3.0, 2.0, 5.0], &[4]);
        let y = max_pool(&x.view(), &[0], &[2], &[1], &[0], &[1]).unwrap();
        assert_eq!(y.as_slice(), &[3.0, 3.0, 5.0]);
    }

    #[test]
    fn max_pool_nan_propagates() {
        let x = t(&[1.0, f64::NAN, 2.0, 3.0], &[4]);
        let y = max_pool(&x.view(), &[0], &[2], &[2], &[0], &[1]).unwrap();
        assert!(y.as_slice()[0].is_nan()); // window [1,NaN] → NaN
        assert_eq!(y.as_slice()[1], 3.0);
    }

    #[test]
    fn avg_pool_count_include_pad() {
        // [1,2,3], kernel 2 stride 1 pad 1 → windows: [pad,1],[1,2],[2,3],[3,pad]
        // count_include_pad=true → divisor 2 each: [0.5,1.5,2.5,1.5]
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        let y = avg_pool(&x.view(), &[0], &[2], &[1], &[1], &[1], true).unwrap();
        assert_eq!(y.as_slice(), &[0.5, 1.5, 2.5, 1.5]);
        // count_include_pad=false → edge windows divide by valid count 1: [1,1.5,2.5,3]
        let y2 = avg_pool(&x.view(), &[0], &[2], &[1], &[1], &[1], false).unwrap();
        assert_eq!(y2.as_slice(), &[1.0, 1.5, 2.5, 3.0]);
    }

    #[test]
    fn im2col_1d_zero_fill_oob() {
        // [10,20,30], kernel 2 stride 1 pad 1 → patches 4, taps 2.
        // patch0 = [pad=0, 10]; patch1=[10,20]; patch2=[20,30]; patch3=[30, pad=0]
        let x = t(&[10.0, 20.0, 30.0], &[3]);
        let y = im2col(&x.view(), &[0], &[2], &[1], &[1], &[1]).unwrap();
        assert_eq!(y.shape(), &[4, 2]);
        assert_eq!(y.as_slice(), &[0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 0.0]);
    }

    #[test]
    fn bad_window_params_error_not_panic() {
        let x = t(&[1.0, 2.0], &[2]);
        // mismatched param arrays
        assert!(avg_pool(&x.view(), &[0], &[2], &[], &[0], &[1], false).is_err());
        // stride 0
        assert!(max_pool(&x.view(), &[0], &[2], &[0], &[0], &[1]).is_err());
        // axis out of range
        assert!(max_pool(&x.view(), &[5], &[2], &[1], &[0], &[1]).is_err());
        // pathological padding must return ShapeOverflow, never panic/wrap
        // (regression, adversarial review — the `2*p` doubling is now checked).
        assert!(matches!(
            avg_pool(&x.view(), &[0], &[2], &[1], &[usize::MAX], &[1], false),
            Err(Error::ShapeOverflow)
        ));
    }
}
