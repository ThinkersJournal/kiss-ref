//! The **integer tensor lane**: the §6.11 structural atoms + the integer-valued
//! §6.13 non-primitives (`argmax`/`any`/`all`/`cum{sum,prod,max}`) over the integer
//! dtypes, computed in `i128` and wrapped to the dtype (mirroring `scalar_int`).
//!
//! Reuses the dtype-agnostic tensor machinery (`Tensor<i128>`/`View<i128>`/
//! `Odometer`/`IndexTensor`); the difference from the float lane is that combines
//! go through [`eval_int_op`] with an explicit `dtype` (for two's-complement
//! wrapping) rather than `eval_op`, and there is no NaN. Ops whose decomposition
//! needs `div`/`sqrt`/`exp` (`reduce_mean`/`reduce_var`/`softmax`/norms/`matmul`)
//! are float-only and are **not** in this lane.
//!
//! Never panics; every access checked; a bad shape/index is a typed [`Error`].

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::Expr;
use kiss_ops_vocab::Op;

use crate::attrs::{Combine, Direction, Monoid, OobPolicy};
use crate::bridge::monoid_op;
use crate::scalar_int::{eval_int_expr, eval_int_op, int_spec};
use crate::tensor::{
    numel, row_major_index, IndexTensor, Odometer, Tensor, View, MAX_OPERANDS, MAX_RANK,
};
use crate::Error;

/// The identity element of `monoid` in integer `dtype` (§6.11-0002): sum=0,
/// prod=1, max=dtype minimum, min=dtype maximum.
fn int_identity(m: Monoid, dtype: Dtype) -> Result<i128, Error> {
    let (bits, signed) = int_spec(dtype).ok_or(Error::UnsupportedDtype(dtype))?;
    Ok(match m {
        Monoid::Sum => 0,
        Monoid::Prod => 1,
        Monoid::Max => {
            if signed {
                -(1i128 << (bits - 1))
            } else {
                0
            }
        }
        Monoid::Min => {
            if signed {
                (1i128 << (bits - 1)) - 1
            } else {
                (1i128 << bits) - 1
            }
        }
    })
}

/// Whether the integer tensor lane evaluates `op` (over integer dtypes). The
/// float-only ops (`reduce_mean`/`reduce_var`/`reduce_std`/`reduce_norm2`/
/// `softmax`/`log_softmax`/`rms_norm`/`layer_norm`/`matmul`/`logsumexp`/window)
/// are excluded — their decompositions need `div`/`sqrt`/`exp`.
pub fn int_tensor_supported(op: Op) -> bool {
    matches!(
        op,
        Op::ElementMap
            | Op::Reduce
            | Op::PrefixScan
            | Op::Gather
            | Op::Scatter
            | Op::SortNetwork
            | Op::Argmax
            | Op::Any
            | Op::All
            | Op::Cumsum
            | Op::Cumprod
            | Op::Cummax
            | Op::IndexSelect
            | Op::Embedding
            | Op::ScatterAdd
    )
}

// ---- structural atoms (integer lane) -----------------------------------------

/// **element_map** (integer lane): per-element `body` (via [`eval_int_expr`]) over
/// `N` broadcast integer views.
pub fn element_map(
    body: &Expr,
    inputs: &[View<i128>],
    dtype: Dtype,
    out_shape: &[usize],
) -> Result<Tensor<i128>, Error> {
    let n = inputs.len();
    if n > MAX_OPERANDS {
        return Err(Error::RankExceeded { rank: n, max: MAX_OPERANDS });
    }
    if out_shape.len() > MAX_RANK {
        return Err(Error::RankExceeded { rank: out_shape.len(), max: MAX_RANK });
    }
    let mut bviews: Vec<View<i128>> = Vec::with_capacity(n);
    for v in inputs {
        bviews.push(v.broadcast_to(out_shape)?);
    }
    let count = numel(out_shape)?;
    let mut data: Vec<i128> = Vec::with_capacity(count);
    let mut buf = [0i128; MAX_OPERANDS];
    let mut od = Odometer::new(out_shape)?;
    while let Some(coord) = od.next_coord() {
        for (k, v) in bviews.iter().enumerate() {
            buf[k] = v.read(coord)?;
        }
        data.push(eval_int_expr(body, dtype, &buf[..n])?);
    }
    Tensor::from_vec(data, out_shape)
}

fn decode_reduced(mut r: usize, reduced: &[usize], in_shape: &[usize], coord: &mut [usize]) {
    for &a in reduced.iter().rev() {
        let d = in_shape[a];
        coord[a] = r % d;
        r /= d;
    }
}

/// **reduce** (integer lane): fold over `axes` with `monoid`, keepdim. Combine via
/// [`eval_int_op`] so two's-complement wrapping is inherited.
pub fn reduce(
    x: &View<i128>,
    dtype: Dtype,
    monoid: Monoid,
    axes: &[usize],
) -> Result<Tensor<i128>, Error> {
    if axes.is_empty() {
        return Err(Error::EmptyAxesMask);
    }
    let in_shape = x.shape();
    let rank = in_shape.len();
    let mut is_reduced = [false; MAX_RANK];
    for &a in axes {
        if a >= rank {
            return Err(Error::AxisOutOfRange { axis: a, rank });
        }
        is_reduced[a] = true;
    }
    let mut reduced: Vec<usize> = Vec::new();
    let mut out_shape = [1usize; MAX_RANK];
    let mut rnum: usize = 1;
    for k in 0..rank {
        if is_reduced[k] {
            reduced.push(k);
            rnum = rnum.checked_mul(in_shape[k]).ok_or(Error::ShapeOverflow)?;
            out_shape[k] = 1;
        } else {
            out_shape[k] = in_shape[k];
        }
    }
    let out_shape = &out_shape[..rank];
    let op = monoid_op(monoid);
    let ident = int_identity(monoid, dtype)?;

    let count = numel(out_shape)?;
    let mut data: Vec<i128> = Vec::with_capacity(count);
    let mut in_coord = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        in_coord[..rank].copy_from_slice(oc);
        let mut acc = ident;
        for r in 0..rnum {
            decode_reduced(r, &reduced, in_shape, &mut in_coord[..rank]);
            let xv = x.read(&in_coord[..rank])?;
            acc = eval_int_op(op, dtype, &[acc, xv])?;
        }
        data.push(acc);
    }
    Tensor::from_vec(data, out_shape)
}

/// **prefix_scan** (integer lane): running `monoid` fold along one `axis`.
pub fn prefix_scan(
    x: &View<i128>,
    dtype: Dtype,
    monoid: Monoid,
    axis: usize,
    exclusive: bool,
) -> Result<Tensor<i128>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let op = monoid_op(monoid);
    let ident = int_identity(monoid, dtype)?;
    let count = numel(in_shape)?;
    let mut data: Vec<i128> = if count == 0 { Vec::new() } else { vec![ident; count] };

    let mut line_shape = [1usize; MAX_RANK];
    line_shape[..rank].copy_from_slice(in_shape);
    line_shape[axis] = 1;
    let extent = in_shape[axis];
    let mut coord = [0usize; MAX_RANK];
    let mut od = Odometer::new(&line_shape[..rank])?;
    while let Some(start) = od.next_coord() {
        coord[..rank].copy_from_slice(start);
        let mut acc = ident;
        for j in 0..extent {
            coord[axis] = j;
            let xv = x.read(&coord[..rank])?;
            let lin = row_major_index(&coord[..rank], in_shape);
            let slot = data.get_mut(lin).ok_or(Error::ShapeMismatch { expected: count, got: lin })?;
            if exclusive {
                *slot = acc;
                acc = eval_int_op(op, dtype, &[acc, xv])?;
            } else {
                acc = eval_int_op(op, dtype, &[acc, xv])?;
                *slot = acc;
            }
        }
    }
    Tensor::from_vec(data, in_shape)
}

fn resolve_index(ival: i64, extent: usize, oob: OobPolicy) -> Result<Option<usize>, Error> {
    if ival >= 0 && (ival as u128) < extent as u128 {
        return Ok(Some(ival as usize));
    }
    match oob {
        OobPolicy::Clamp => {
            if extent == 0 {
                return Err(Error::AxisOutOfRange { axis: 0, rank: 0 });
            }
            Ok(Some(if ival < 0 { 0 } else { extent - 1 }))
        }
        OobPolicy::ZeroFill => Ok(None),          // → zero
        OobPolicy::Skip => Ok(Some(usize::MAX)),  // sentinel handled by caller (base)
    }
}

/// **gather** (integer lane): raw integer move. Same shape rule as the float
/// gather; `zero_fill` yields integer `0`; `skip` keeps the `base` value.
pub fn gather(
    data: &View<i128>,
    index: &IndexTensor,
    axis: usize,
    oob: OobPolicy,
    base: Option<&View<i128>>,
) -> Result<Tensor<i128>, Error> {
    let dshape = data.shape();
    let drank = dshape.len();
    if axis >= drank {
        return Err(Error::AxisOutOfRange { axis, rank: drank });
    }
    let ishape = index.shape();
    let irank = ishape.len();
    let out_rank = drank - 1 + irank;
    if out_rank > MAX_RANK {
        return Err(Error::RankExceeded { rank: out_rank, max: MAX_RANK });
    }
    let extent = dshape[axis];
    let mut out_shape = [1usize; MAX_RANK];
    let mut w = 0;
    for &d in &dshape[..axis] {
        out_shape[w] = d;
        w += 1;
    }
    for &d in ishape {
        out_shape[w] = d;
        w += 1;
    }
    for &d in &dshape[axis + 1..] {
        out_shape[w] = d;
        w += 1;
    }
    let out_shape = &out_shape[..out_rank];
    let base_v = match base {
        Some(b) => Some(b.broadcast_to(out_shape)?),
        None => None,
    };

    let count = numel(out_shape)?;
    let mut buf: Vec<i128> = Vec::with_capacity(count);
    let mut src = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        let ilin = row_major_index(&oc[axis..axis + irank], ishape);
        let ival = *index.as_slice().get(ilin).ok_or(Error::ShapeMismatch {
            expected: index.as_slice().len(),
            got: ilin,
        })?;
        let val = match resolve_index(ival, extent, oob)? {
            None => 0, // zero_fill
            Some(s) if oob == OobPolicy::Skip && s == usize::MAX => match &base_v {
                Some(b) => b.read(oc)?,
                None => return Err(Error::Unsupported(Op::Gather)),
            },
            Some(s) => {
                for k in 0..axis {
                    src[k] = oc[k];
                }
                src[axis] = s;
                for k in 0..(drank - 1 - axis) {
                    src[axis + 1 + k] = oc[axis + irank + k];
                }
                data.read(&src[..drank])?
            }
        };
        buf.push(val);
    }
    Tensor::from_vec(buf, out_shape)
}

/// **scatter** (integer lane): write into an explicit `dest` along `axis` with a
/// 1-D index, combined per `combine` (via [`eval_int_op`]). Integer `atomic_add`
/// is deterministic (integer add is associative).
pub fn scatter(
    dest: Tensor<i128>,
    dtype: Dtype,
    index: &IndexTensor,
    updates: &View<i128>,
    axis: usize,
    combine: Combine,
) -> Result<Tensor<i128>, Error> {
    let mut dshape = [1usize; MAX_RANK];
    let drank = dest.rank();
    dshape[..drank].copy_from_slice(dest.shape());
    if axis >= drank {
        return Err(Error::AxisOutOfRange { axis, rank: drank });
    }
    if index.rank() != 1 {
        return Err(Error::ShapeMismatch { expected: 1, got: index.rank() });
    }
    let ushape = updates.shape();
    if ushape.len() != drank {
        return Err(Error::ShapeMismatch { expected: drank, got: ushape.len() });
    }
    for k in 0..drank {
        let want = if k == axis { index.as_slice().len() } else { dshape[k] };
        if ushape[k] != want {
            return Err(Error::ShapeMismatch { expected: want, got: ushape[k] });
        }
    }
    let dest_extent = dshape[axis];
    let mut data = dest.into_data();
    let dlen = data.len();
    let mut dcoord = [0usize; MAX_RANK];
    let mut od = Odometer::new(ushape)?;
    while let Some(uc) = od.next_coord() {
        let jpos = uc[axis];
        let ival = *index.as_slice().get(jpos).ok_or(Error::ShapeMismatch {
            expected: index.as_slice().len(),
            got: jpos,
        })?;
        if ival < 0 || (ival as u128) >= dest_extent as u128 {
            continue;
        }
        dcoord[..drank].copy_from_slice(uc);
        dcoord[axis] = ival as usize;
        let dlin = row_major_index(&dcoord[..drank], &dshape[..drank]);
        let u = updates.read(uc)?;
        let cur = *data.get(dlin).ok_or(Error::ShapeMismatch { expected: dlen, got: dlin })?;
        let next = match combine {
            Combine::Assign => u,
            Combine::AtomicAdd => eval_int_op(Op::Add, dtype, &[cur, u])?,
            Combine::AtomicMax => eval_int_op(Op::MaxProp, dtype, &[cur, u])?,
            Combine::AtomicMin => eval_int_op(Op::MinProp, dtype, &[cur, u])?,
        };
        *data.get_mut(dlin).ok_or(Error::ShapeMismatch { expected: dlen, got: dlin })? = next;
    }
    Tensor::from_vec(data, &dshape[..drank])
}

/// **sort_network** (integer lane): stable per-line permutation along `axis` by
/// integer value (no NaN); ties → lower original index.
pub fn sort_network(
    keys: &View<i128>,
    axis: usize,
    dir: Direction,
) -> Result<(Tensor<i128>, IndexTensor), Error> {
    let in_shape = keys.shape();
    let rank = in_shape.len();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let count = numel(in_shape)?;
    if count == 0 {
        return Ok((
            Tensor::from_vec(Vec::new(), in_shape)?,
            IndexTensor::new(Vec::new(), in_shape, Dtype::I64)?,
        ));
    }
    let extent = in_shape[axis];
    let fill = keys.read(&[0usize; MAX_RANK][..rank])?;
    let mut vals = vec![fill; count];
    let mut idxs = vec![0i64; count];

    let mut line_shape = [1usize; MAX_RANK];
    line_shape[..rank].copy_from_slice(in_shape);
    line_shape[axis] = 1;
    let mut coord = [0usize; MAX_RANK];
    let mut od = Odometer::new(&line_shape[..rank])?;
    while let Some(start) = od.next_coord() {
        coord[..rank].copy_from_slice(start);
        let mut pairs: Vec<(i64, i128)> = Vec::with_capacity(extent);
        for j in 0..extent {
            coord[axis] = j;
            pairs.push((j as i64, keys.read(&coord[..rank])?));
        }
        pairs.sort_by(|&(_, av), &(_, bv)| {
            let o = av.cmp(&bv);
            match dir {
                Direction::Asc => o,
                Direction::Desc => o.reverse(),
            }
        });
        for (r, &(orig, val)) in pairs.iter().enumerate() {
            coord[axis] = r;
            let lin = row_major_index(&coord[..rank], in_shape);
            *vals.get_mut(lin).ok_or(Error::ShapeMismatch { expected: count, got: lin })? = val;
            *idxs.get_mut(lin).ok_or(Error::ShapeMismatch { expected: count, got: lin })? = orig;
        }
    }
    Ok((
        Tensor::from_vec(vals, in_shape)?,
        IndexTensor::new(idxs, in_shape, Dtype::I64)?,
    ))
}

// ---- integer-valued non-primitives -------------------------------------------

/// `argmax` (integer lane) — rank-0 of `sort_network(desc)` along `axis`.
pub fn argmax(x: &View<i128>, axis: usize) -> Result<IndexTensor, Error> {
    let rank = x.rank();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let (_, idx) = sort_network(x, axis, Direction::Desc)?;
    let in_shape = x.shape();
    let mut out_shape = [1usize; MAX_RANK];
    out_shape[..rank].copy_from_slice(in_shape);
    out_shape[axis] = 1;
    let out_shape = &out_shape[..rank];
    let n = numel(out_shape)?;
    let isl = idx.as_slice();
    let mut data: Vec<i64> = Vec::with_capacity(n);
    let mut od = Odometer::new(out_shape)?;
    let mut coord = [0usize; MAX_RANK];
    while let Some(oc) = od.next_coord() {
        coord[..rank].copy_from_slice(oc);
        coord[axis] = 0;
        let lin = row_major_index(&coord[..rank], in_shape);
        data.push(*isl.get(lin).ok_or(Error::ShapeMismatch { expected: isl.len(), got: lin })?);
    }
    IndexTensor::new(data, out_shape, Dtype::I64)
}

/// `x != 0` elementwise (integer `cmp_ne`), the boolean seed for `any`/`all`.
fn ne_zero(x: &View<i128>, dtype: Dtype) -> Result<Tensor<i128>, Error> {
    let shape = x.shape();
    let count = numel(shape)?;
    let mut data: Vec<i128> = Vec::with_capacity(count);
    let mut od = Odometer::new(shape)?;
    while let Some(c) = od.next_coord() {
        data.push(eval_int_op(Op::CmpNe, dtype, &[x.read(c)?, 0])?);
    }
    Tensor::from_vec(data, shape)
}

/// `any` (integer lane) — `reduce(max, cmp_ne(x, 0))`.
pub fn any(x: &View<i128>, dtype: Dtype, axes: &[usize]) -> Result<Tensor<i128>, Error> {
    let ne = ne_zero(x, dtype)?;
    reduce(&ne.view(), dtype, Monoid::Max, axes)
}

/// `all` (integer lane) — `reduce(min, cmp_ne(x, 0))`.
pub fn all(x: &View<i128>, dtype: Dtype, axes: &[usize]) -> Result<Tensor<i128>, Error> {
    let ne = ne_zero(x, dtype)?;
    reduce(&ne.view(), dtype, Monoid::Min, axes)
}

/// `cumsum`/`cumprod`/`cummax` (integer lane) — inclusive `prefix_scan`.
pub fn cumsum(x: &View<i128>, dtype: Dtype, axis: usize) -> Result<Tensor<i128>, Error> {
    prefix_scan(x, dtype, Monoid::Sum, axis, false)
}
pub fn cumprod(x: &View<i128>, dtype: Dtype, axis: usize) -> Result<Tensor<i128>, Error> {
    prefix_scan(x, dtype, Monoid::Prod, axis, false)
}
pub fn cummax(x: &View<i128>, dtype: Dtype, axis: usize) -> Result<Tensor<i128>, Error> {
    prefix_scan(x, dtype, Monoid::Max, axis, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[i128], shape: &[usize]) -> Tensor<i128> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn int_reduce_sum_wraps() {
        // s8: 100 + 100 = 200 wraps to -56 (two's-complement, §6.2-0002).
        let x = t(&[100, 100], &[2]);
        let r = reduce(&x.view(), Dtype::S8, Monoid::Sum, &[0]).unwrap();
        assert_eq!(r.as_slice(), &[-56]);
        // u8 identity/empty axis → 0.
        let e = t(&[], &[0]);
        let re = reduce(&e.view(), Dtype::U8, Monoid::Sum, &[0]).unwrap();
        assert_eq!(re.as_slice(), &[0]);
    }

    #[test]
    fn int_reduce_max_min_identity() {
        let x = t(&[-5, 3, -1], &[3]);
        assert_eq!(reduce(&x.view(), Dtype::S8, Monoid::Max, &[0]).unwrap().as_slice(), &[3]);
        assert_eq!(reduce(&x.view(), Dtype::S8, Monoid::Min, &[0]).unwrap().as_slice(), &[-5]);
    }

    #[test]
    fn int_cumsum_and_argmax() {
        let x = t(&[1, 2, 3, 4], &[4]);
        assert_eq!(cumsum(&x.view(), Dtype::I32, 0).unwrap().as_slice(), &[1, 3, 6, 10]);
        let a = argmax(&t(&[3, 9, 1], &[3]).view(), 0).unwrap();
        assert_eq!(a.as_slice(), &[1]);
    }

    #[test]
    fn int_any_all() {
        let z = t(&[0, 5, 0], &[3]);
        assert_eq!(any(&z.view(), Dtype::I32, &[0]).unwrap().as_slice(), &[1]);
        assert_eq!(all(&z.view(), Dtype::I32, &[0]).unwrap().as_slice(), &[0]);
    }

    #[test]
    fn int_gather_and_scatter_add() {
        let d = t(&[10, 20, 30], &[3]);
        let idx = IndexTensor::new(vec![2, 0, 9], &[3], Dtype::I64).unwrap();
        let g = gather(&d.view(), &idx, 0, OobPolicy::ZeroFill, None).unwrap();
        assert_eq!(g.as_slice(), &[30, 10, 0]);

        let dest = t(&[0, 0, 0], &[3]);
        let sidx = IndexTensor::new(vec![0, 0, 1], &[3], Dtype::I64).unwrap();
        let upd = t(&[5, 7, 9], &[3]);
        let s = scatter(dest, Dtype::I32, &sidx, &upd.view(), 0, Combine::AtomicAdd).unwrap();
        assert_eq!(s.as_slice(), &[12, 9, 0]); // idx0: 5+7=12, idx1: 9
    }

    #[test]
    fn int_element_map_wraps() {
        // add two u8 tensors via element_map(add(input(0),input(1))): 200+100=300→44.
        let a = t(&[200], &[1]);
        let b = t(&[100], &[1]);
        let body = kiss_ops_vocab::decomp::parse("add(input(0), input(1))").unwrap();
        let r = element_map(&body, &[a.view(), b.view()], Dtype::U8, &[1]).unwrap();
        assert_eq!(r.as_slice(), &[44]);
    }
}
