//! Dense, row-major, strided tensors for the reference tensor-evaluation layer.
//!
//! A [`Tensor<T>`] owns a contiguous `Vec<T>` in row-major (C) order; a
//! [`View<'a, T>`] is a borrowed strided window over a slice, the read surface the
//! structural kernels iterate. Broadcasting is expressed as a **stride-0 axis**
//! (§6.11-0001); a reduce keepdim axis is an **extent-1/stride-0 axis**
//! (§6.11-0008) — one descriptor mechanism serves both.
//!
//! Every access is checked and returns [`Error`] — the tensor layer never panics
//! (Fuel's execution-route contract). Rank is bounded by [`MAX_RANK`]
//! (§6.19-0037); element-count arithmetic is `checked_mul` → [`Error::ShapeOverflow`].

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;

use crate::Error;

/// Maximum tensor rank (§6.19-0037).
pub const MAX_RANK: usize = 8;
/// Maximum operand count of a structural op (§6.19-0037).
pub const MAX_OPERANDS: usize = 8;

/// Row-major element count of `shape`, with overflow guarded (§ never panics).
pub fn numel(shape: &[usize]) -> Result<usize, Error> {
    let mut n: usize = 1;
    for &d in shape {
        n = n.checked_mul(d).ok_or(Error::ShapeOverflow)?;
    }
    Ok(n)
}

/// An empty `Vec` with capacity for `count` elements reserved **without
/// panicking or aborting**. `Vec::with_capacity(count)` panics when
/// `count * size_of::<T>()` exceeds `isize::MAX` and *aborts the process* when
/// the allocator refuses (e.g. a shape-derived `count` of `2^41` on a huge
/// padding). A shape-derived count is attacker-influenceable, so every tensor
/// buffer sized from `numel` / an extent must use this instead — the count is a
/// valid `usize` that only fails at allocation, which `Vec::try_reserve` reports
/// as a typed [`Error::ShapeOverflow`] rather than a never-panic violation.
/// (Found by the window-family allocation fuzzer, 2026-07-23.)
pub(crate) fn alloc_exact<T>(count: usize) -> Result<Vec<T>, Error> {
    let mut v: Vec<T> = Vec::new();
    v.try_reserve(count).map_err(|_| Error::ShapeOverflow)?;
    Ok(v)
}

/// The row-major linear index of `coord` within `shape` (kernels that write along
/// a non-row-major axis address the contiguous output buffer through this).
/// Caller guarantees `coord` is in range (only used with validated coordinates).
pub fn row_major_index(coord: &[usize], shape: &[usize]) -> usize {
    let mut idx: usize = 0;
    let mut acc: usize = 1;
    for k in (0..shape.len()).rev() {
        idx = idx.wrapping_add(coord[k].wrapping_mul(acc));
        acc = acc.wrapping_mul(shape[k]);
    }
    idx
}

/// Row-major contiguous strides for `shape` (into a fixed `[isize; MAX_RANK]`).
fn contiguous_strides(shape: &[usize]) -> [isize; MAX_RANK] {
    let mut s = [0isize; MAX_RANK];
    let mut acc: isize = 1;
    let r = shape.len();
    for k in (0..r).rev() {
        s[k] = acc;
        acc = acc.wrapping_mul(shape[k] as isize);
    }
    s
}

/// Decode a row-major linear index into `coord[..shape.len()]`.
///
/// Caller guarantees `lin < numel(shape)` and every `shape[k] > 0` (only reached
/// inside a `0..numel` loop, which does not execute when any extent is 0).
fn decode(lin: usize, shape: &[usize], coord: &mut [usize]) {
    let mut rem = lin;
    for k in (0..shape.len()).rev() {
        let d = shape[k];
        coord[k] = rem % d;
        rem /= d;
    }
}

/// A dense, row-major-materialized tensor owning its data.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: [usize; MAX_RANK],
    rank: usize,
}

impl<T: Copy> Tensor<T> {
    /// Wrap a contiguous row-major `data` buffer as a tensor of `shape`.
    ///
    /// Errors: [`Error::RankExceeded`] if `shape.len() > MAX_RANK`,
    /// [`Error::ShapeMismatch`] if `data.len()` ≠ `numel(shape)`.
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::RankExceeded {
                rank: shape.len(),
                max: MAX_RANK,
            });
        }
        let n = numel(shape)?;
        if n != data.len() {
            return Err(Error::ShapeMismatch {
                expected: n,
                got: data.len(),
            });
        }
        let mut s = [1usize; MAX_RANK];
        s[..shape.len()].copy_from_slice(shape);
        Ok(Tensor {
            data,
            shape: s,
            rank: shape.len(),
        })
    }

    /// A tensor of `shape` filled with `value`.
    pub fn full(shape: &[usize], value: T) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::RankExceeded {
                rank: shape.len(),
                max: MAX_RANK,
            });
        }
        let n = numel(shape)?;
        let mut data = Vec::new();
        data.try_reserve(n).map_err(|_| Error::ShapeOverflow)?;
        data.resize(n, value);
        Tensor::from_vec(data, shape)
    }

    /// The logical shape (`&self.shape[..rank]`).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// The tensor rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The row-major element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The contiguous row-major backing slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consume into the row-major backing buffer.
    #[inline]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// A borrowed contiguous view over the whole tensor.
    pub fn view(&self) -> View<'_, T> {
        View {
            data: &self.data,
            shape: self.shape,
            strides: contiguous_strides(&self.shape[..self.rank]),
            offset: 0,
            rank: self.rank,
        }
    }
}

/// A borrowed, strided read window over a contiguous slice.
#[derive(Clone, Copy)]
pub struct View<'a, T> {
    data: &'a [T],
    shape: [usize; MAX_RANK],
    strides: [isize; MAX_RANK],
    offset: usize,
    rank: usize,
}

impl<'a, T: Copy> View<'a, T> {
    /// The logical shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// The view rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Read the element at `coord` (length must equal the view rank). Checked at
    /// every step — a bad coordinate or a stride that lands outside the buffer is
    /// an [`Error`], never a panic.
    pub fn read(&self, coord: &[usize]) -> Result<T, Error> {
        if coord.len() != self.rank {
            return Err(Error::ShapeMismatch {
                expected: self.rank,
                got: coord.len(),
            });
        }
        let mut idx: isize = self.offset as isize;
        for (&c, &stride) in coord.iter().zip(&self.strides[..self.rank]) {
            idx = idx.wrapping_add((c as isize).wrapping_mul(stride));
        }
        let i = usize::try_from(idx).map_err(|_| Error::ShapeMismatch {
            expected: self.data.len(),
            got: 0,
        })?;
        self.data.get(i).copied().ok_or(Error::ShapeMismatch {
            expected: self.data.len(),
            got: i,
        })
    }

    /// Broadcast this view to `target` (numpy right-aligned rules): a leading pad
    /// axis or an input extent of `1` becomes a **stride-0** axis (§6.11-0001).
    /// Errors [`Error::BroadcastIncompatible`] if a non-1 extent mismatches.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<View<'a, T>, Error> {
        let r_out = target.len();
        if r_out > MAX_RANK || r_out < self.rank {
            return Err(Error::BroadcastIncompatible);
        }
        let pad = r_out - self.rank;
        let mut shape = [1usize; MAX_RANK];
        let mut strides = [0isize; MAX_RANK];
        for k in 0..r_out {
            shape[k] = target[k];
            if k < pad {
                strides[k] = 0;
            } else {
                let ax = k - pad;
                let in_dim = self.shape[ax];
                if in_dim == target[k] {
                    strides[k] = self.strides[ax];
                } else if in_dim == 1 {
                    strides[k] = 0;
                } else {
                    return Err(Error::BroadcastIncompatible);
                }
            }
        }
        Ok(View {
            data: self.data,
            shape,
            strides,
            offset: self.offset,
            rank: r_out,
        })
    }
}

/// The numpy-style broadcast of several shapes into `[out; MAX_RANK]` + rank.
pub fn broadcast_shapes(shapes: &[&[usize]]) -> Result<([usize; MAX_RANK], usize), Error> {
    let mut r_out = 0usize;
    for s in shapes {
        if s.len() > r_out {
            r_out = s.len();
        }
    }
    if r_out > MAX_RANK {
        return Err(Error::RankExceeded {
            rank: r_out,
            max: MAX_RANK,
        });
    }
    let mut out = [1usize; MAX_RANK];
    for s in shapes {
        let pad = r_out - s.len();
        for (j, &d) in s.iter().enumerate() {
            let k = pad + j;
            if out[k] == 1 {
                out[k] = d;
            } else if d != 1 && d != out[k] {
                return Err(Error::BroadcastIncompatible);
            }
        }
    }
    Ok((out, r_out))
}

/// A row-major coordinate iterator over an output shape (the odometer the kernels
/// walk). Yields `numel(shape)` coordinates in row-major order.
pub struct Odometer<'s> {
    shape: &'s [usize],
    numel: usize,
    lin: usize,
    coord: [usize; MAX_RANK],
}

impl<'s> Odometer<'s> {
    /// A row-major walk over `shape`. Returns [`Error::RankExceeded`] if the rank
    /// exceeds [`MAX_RANK`] (the fixed `coord` buffer would otherwise be indexed
    /// out of range) or [`Error::ShapeOverflow`] if the element count overflows.
    pub fn new(shape: &'s [usize]) -> Result<Self, Error> {
        if shape.len() > MAX_RANK {
            return Err(Error::RankExceeded {
                rank: shape.len(),
                max: MAX_RANK,
            });
        }
        Ok(Odometer {
            shape,
            numel: numel(shape)?,
            lin: 0,
            coord: [0usize; MAX_RANK],
        })
    }

    /// The number of coordinates this odometer yields.
    #[inline]
    pub fn len(&self) -> usize {
        self.numel
    }

    /// Whether the odometer yields no coordinates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.numel == 0
    }

    /// The next row-major coordinate, or `None` when exhausted.
    pub fn next_coord(&mut self) -> Option<&[usize]> {
        if self.lin >= self.numel {
            return None;
        }
        let r = self.shape.len();
        decode(self.lin, self.shape, &mut self.coord[..r]);
        self.lin += 1;
        Some(&self.coord[..r])
    }
}

/// A tensor of integer indices for `gather`/`scatter`/`index_select`/`embedding`/
/// `scatter_add`. Values are held widened in `i64`; `dtype` records the source
/// index dtype so the `{u32, i32, i64}` restriction (§6.11-0009) and the
/// negative=out-of-bounds rule (§6.11-0004) are enforced by tag, not inferred.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexTensor {
    data: Vec<i64>,
    shape: [usize; MAX_RANK],
    rank: usize,
    dtype: Dtype,
}

impl IndexTensor {
    /// Build an index tensor of `shape` from widened `i64` values, validating the
    /// dtype is an index-legal role dtype (§6.11-0009: `{u32, i32, i64}`).
    pub fn new(data: Vec<i64>, shape: &[usize], dtype: Dtype) -> Result<Self, Error> {
        if !matches!(dtype, Dtype::U32 | Dtype::I32 | Dtype::I64) {
            return Err(Error::IndexDtypeIllegal(dtype));
        }
        if shape.len() > MAX_RANK {
            return Err(Error::RankExceeded {
                rank: shape.len(),
                max: MAX_RANK,
            });
        }
        let n = numel(shape)?;
        if n != data.len() {
            return Err(Error::ShapeMismatch {
                expected: n,
                got: data.len(),
            });
        }
        let mut s = [1usize; MAX_RANK];
        s[..shape.len()].copy_from_slice(shape);
        Ok(IndexTensor {
            data,
            shape: s,
            rank: shape.len(),
            dtype,
        })
    }

    /// The index values in row-major order.
    #[inline]
    pub fn as_slice(&self) -> &[i64] {
        &self.data
    }

    /// The logical shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    /// The recorded source index dtype.
    #[inline]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The index rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn from_vec_shape_check() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.len(), 6);
        // wrong element count is an error, not a panic.
        assert!(matches!(
            Tensor::from_vec(vec![1.0f32, 2.0], &[2, 3]),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn view_read_row_major() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let v = t.view();
        assert_eq!(v.read(&[0, 0]).unwrap(), 1.0);
        assert_eq!(v.read(&[1, 2]).unwrap(), 6.0);
        assert_eq!(v.read(&[0, 2]).unwrap(), 3.0);
        // out-of-shape coord is an error, never a panic.
        assert!(v.read(&[2, 0]).is_err());
        assert!(v.read(&[0]).is_err());
    }

    #[test]
    fn broadcast_stride_zero() {
        // [3] broadcast to [2,3]: leading axis stride 0.
        let t = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], &[3]).unwrap();
        let b = t.view().broadcast_to(&[2, 3]).unwrap();
        assert_eq!(b.read(&[0, 1]).unwrap(), 20.0);
        assert_eq!(b.read(&[1, 1]).unwrap(), 20.0); // same row value (stride 0)
    }

    #[test]
    fn broadcast_incompatible_is_error() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(matches!(
            t.view().broadcast_to(&[2, 4]),
            Err(Error::BroadcastIncompatible)
        ));
    }

    #[test]
    fn broadcast_shapes_numpy() {
        let (s, r) = broadcast_shapes(&[&[2, 1], &[3]]).unwrap();
        assert_eq!(&s[..r], &[2, 3]);
    }

    #[test]
    fn odometer_row_major_order() {
        let mut od = Odometer::new(&[2, 2]).unwrap();
        let mut seen = alloc::vec::Vec::new();
        while let Some(c) = od.next_coord() {
            seen.push([c[0], c[1]]);
        }
        assert_eq!(seen, vec![[0, 0], [0, 1], [1, 0], [1, 1]]);
    }

    #[test]
    fn index_tensor_dtype_role() {
        assert!(IndexTensor::new(vec![0i64, 1], &[2], Dtype::I64).is_ok());
        assert!(matches!(
            IndexTensor::new(vec![0i64], &[1], Dtype::F32),
            Err(Error::IndexDtypeIllegal(_))
        ));
    }

    #[test]
    fn numel_overflow_guarded() {
        assert!(matches!(numel(&[usize::MAX, 2]), Err(Error::ShapeOverflow)));
    }

    #[test]
    fn odometer_rejects_over_rank() {
        // rank 9 > MAX_RANK: an Error, never an out-of-range panic on the fixed
        // coord buffer (regression, adversarial review).
        assert!(matches!(
            Odometer::new(&[1usize; 9]),
            Err(Error::RankExceeded { .. })
        ));
    }
}
