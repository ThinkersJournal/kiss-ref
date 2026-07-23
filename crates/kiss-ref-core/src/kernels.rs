//! The six structural floor atoms (KISS-Ops §6.11) — the irreducible hand-written
//! tensor kernels. Everything numeric routes back through the scalar engine:
//! `element_map` bodies via [`eval_expr`], monoid combines and scatter combines
//! via [`eval_op`], so NaN-propagation / signed-zero / refined forms are inherited.
//!
//! Float lane only in this cut (`T: ScalarFloat`). Every access is checked; a bad
//! shape/coord/index is an [`Error`], never a panic.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::decomp::Expr;

use crate::attrs::{Combine, Direction, Monoid, OobPolicy};
use crate::bridge::{monoid_identity, monoid_op};
use crate::resolve::{eval_expr, eval_op};
use crate::scalar::ScalarFloat;
use crate::tensor::{
    numel, row_major_index, IndexTensor, Odometer, Tensor, View, MAX_OPERANDS, MAX_RANK,
};
use crate::Error;

/// Apply `f` per output coordinate over the broadcast of `inputs` (§6.11-0001).
/// The shared inner loop of `element_map` and the elementwise non-primitive
/// wrappers.
pub(crate) fn map_views<T: ScalarFloat>(
    inputs: &[View<T>],
    out_shape: &[usize],
    f: impl Fn(&[T]) -> Result<T, Error>,
) -> Result<Tensor<T>, Error> {
    let n = inputs.len();
    if n > MAX_OPERANDS {
        return Err(Error::RankExceeded { rank: n, max: MAX_OPERANDS });
    }
    // Guard the output rank directly: with zero inputs the broadcast loop below
    // never runs, so this is the only rank check on `out_shape` (never-panic).
    if out_shape.len() > MAX_RANK {
        return Err(Error::RankExceeded { rank: out_shape.len(), max: MAX_RANK });
    }
    let mut bviews: Vec<View<T>> = Vec::with_capacity(n);
    for v in inputs {
        bviews.push(v.broadcast_to(out_shape)?);
    }
    let count = numel(out_shape)?;
    let mut data: Vec<T> = Vec::with_capacity(count);
    let mut buf = [T::ZERO; MAX_OPERANDS];
    let mut od = Odometer::new(out_shape)?;
    while let Some(coord) = od.next_coord() {
        for (k, v) in bviews.iter().enumerate() {
            buf[k] = v.read(coord)?;
        }
        data.push(f(&buf[..n])?);
    }
    Tensor::from_vec(data, out_shape)
}

/// **element_map** (§6.11-0001): evaluate a per-element scalar `body` over `N`
/// broadcast input views. The output shape is the broadcast of the inputs
/// (§6.20-0007); the body is the UNCHANGED [`eval_expr`], so all pointwise
/// numerics are inherited verbatim.
pub fn element_map<T: ScalarFloat>(
    body: &Expr,
    inputs: &[View<T>],
    out_shape: &[usize],
) -> Result<Tensor<T>, Error> {
    map_views(inputs, out_shape, |buf| eval_expr(body, buf))
}

/// A row-major walk over the reduced axes of `in_shape`, writing each combination
/// into `coord`. Only reached inside a `0..rnum` loop, so every reduced extent is
/// non-zero there.
fn decode_reduced(mut r: usize, reduced: &[usize], in_shape: &[usize], coord: &mut [usize]) {
    for &a in reduced.iter().rev() {
        let d = in_shape[a];
        coord[a] = r % d;
        r /= d;
    }
}

/// **reduce** (§6.11-0002/-0008): fold `x` over the axis set `axes` with `monoid`,
/// keepdim fixed (each reduced axis becomes extent-1). Empty axis → monoid
/// identity; `max`/`min` NaN-propagate; the combine is one [`eval_op`] call.
pub fn reduce<T: ScalarFloat>(x: &View<T>, monoid: Monoid, axes: &[usize]) -> Result<Tensor<T>, Error> {
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
    let ident = monoid_identity::<T>(monoid);

    let count = numel(out_shape)?;
    let mut data: Vec<T> = Vec::with_capacity(count);
    let mut in_coord = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        in_coord[..rank].copy_from_slice(oc);
        let mut acc = ident;
        for r in 0..rnum {
            decode_reduced(r, &reduced, in_shape, &mut in_coord[..rank]);
            let xv = x.read(&in_coord[..rank])?;
            acc = eval_op(op, &[acc, xv])?;
        }
        data.push(acc);
    }
    Tensor::from_vec(data, out_shape)
}

/// **prefix_scan** (§6.11-0003): inclusive/exclusive running `monoid` fold along
/// exactly one `axis`; length-preserving (output shape = input shape).
pub fn prefix_scan<T: ScalarFloat>(
    x: &View<T>,
    monoid: Monoid,
    axis: usize,
    exclusive: bool,
) -> Result<Tensor<T>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let op = monoid_op(monoid);
    let ident = monoid_identity::<T>(monoid);
    let count = numel(in_shape)?;
    let mut data: Vec<T> = if count == 0 {
        Vec::new()
    } else {
        vec![ident; count]
    };

    // Walk each line (fix non-axis coords) and accumulate along `axis`.
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
                acc = eval_op(op, &[acc, xv])?;
            } else {
                acc = eval_op(op, &[acc, xv])?;
                *slot = acc;
            }
        }
    }
    Tensor::from_vec(data, in_shape)
}

/// Resolution of one index value under an out-of-bounds policy.
enum Resolved {
    /// A valid in-range source position.
    In(usize),
    /// Output a zero (`zero_fill`).
    Zero,
    /// Keep the caller-provided base value (`skip`).
    Skip,
}

fn resolve_index(ival: i64, extent: usize, oob: OobPolicy) -> Result<Resolved, Error> {
    if ival >= 0 && (ival as u128) < extent as u128 {
        return Ok(Resolved::In(ival as usize));
    }
    // Negative or >= extent: out of bounds (no from-end wrap, §6.11-0004).
    match oob {
        OobPolicy::Clamp => {
            if extent == 0 {
                return Err(Error::AxisOutOfRange { axis: 0, rank: 0 });
            }
            let c = if ival < 0 { 0 } else { extent - 1 };
            Ok(Resolved::In(c))
        }
        OobPolicy::ZeroFill => Ok(Resolved::Zero),
        OobPolicy::Skip => Ok(Resolved::Skip),
    }
}

/// **gather** (§6.11-0004): data-dependent read replacing coordinate `axis` with a
/// runtime `index`. Output shape = `data[..axis] ++ index.shape ++ data[axis+1..]`
/// (§6.20-0008). Pure raw-bit move (NaN payload / −0 preserved). `skip` keeps the
/// `base` value at that position (RFC-pinned); `base=None` under `skip` on an OOB
/// index is an error.
pub fn gather<T: ScalarFloat>(
    data: &View<T>,
    index: &IndexTensor,
    axis: usize,
    oob: OobPolicy,
    base: Option<&View<T>>,
) -> Result<Tensor<T>, Error> {
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
    let mut buf: Vec<T> = Vec::with_capacity(count);
    let mut src = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        // oc = pre(axis) ++ idx_part(irank) ++ post(drank-1-axis)
        let ilin = row_major_index(&oc[axis..axis + irank], ishape);
        let ival = *index.as_slice().get(ilin).ok_or(Error::ShapeMismatch {
            expected: index.as_slice().len(),
            got: ilin,
        })?;
        let val = match resolve_index(ival, extent, oob)? {
            Resolved::In(s) => {
                for k in 0..axis {
                    src[k] = oc[k];
                }
                src[axis] = s;
                for k in 0..(drank - 1 - axis) {
                    src[axis + 1 + k] = oc[axis + irank + k];
                }
                data.read(&src[..drank])?
            }
            Resolved::Zero => T::ZERO,
            Resolved::Skip => match &base_v {
                Some(b) => b.read(oc)?,
                None => return Err(Error::GatherSkipNoBase),
            },
        };
        buf.push(val);
    }
    Tensor::from_vec(buf, out_shape)
}

/// **scatter** (§6.11-0005/-0006): data-dependent write into an explicit `dest`
/// along `axis`, indexed by a 1-D `index`, combined per `combine`. The write
/// shape is `dest.shape` with `[axis] = index.len()`; `updates` is **broadcast**
/// to it under the ordinary §6.11-0001 rules — a rank-0 updates writes one
/// scalar per index element (the bincount/histogram form; KISS PR #75 companion,
/// **ruled** general-broadcast 2026-07-23). OOB writes are skipped; `dest`
/// positions never written keep
/// their value. Assign tie-break = highest row-major source wins; float
/// `atomic_add` folds colliding contributions in pinned row-major source order.
pub fn scatter<T: ScalarFloat>(
    dest: Tensor<T>,
    index: &IndexTensor,
    updates: &View<T>,
    axis: usize,
    combine: Combine,
) -> Result<Tensor<T>, Error> {
    let mut dshape = [1usize; MAX_RANK];
    let drank = dest.rank();
    dshape[..drank].copy_from_slice(dest.shape());
    if axis >= drank {
        return Err(Error::AxisOutOfRange { axis, rank: drank });
    }
    if index.rank() != 1 {
        // This cut supports a 1-D index aligned to `axis` (the scatter_add form).
        return Err(Error::ShapeMismatch { expected: 1, got: index.rank() });
    }
    // The write shape; updates broadcast to it (stride-0 axes read the same
    // element repeatedly — a raw-bit move either way).
    let mut wshape = [0usize; MAX_RANK];
    wshape[..drank].copy_from_slice(&dshape[..drank]);
    wshape[axis] = index.as_slice().len();
    let updates = updates.broadcast_to(&wshape[..drank])?;
    let ushape = updates.shape();
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
        // OOB destination index → skip (§6.11-0005).
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
            Combine::AtomicAdd => eval_op(kiss_ops_vocab::Op::Add, &[cur, u])?,
            Combine::AtomicMax => eval_op(kiss_ops_vocab::Op::MaxProp, &[cur, u])?,
            Combine::AtomicMin => eval_op(kiss_ops_vocab::Op::MinProp, &[cur, u])?,
        };
        *data.get_mut(dlin).ok_or(Error::ShapeMismatch { expected: dlen, got: dlin })? = next;
    }
    Tensor::from_vec(data, &dshape[..drank])
}

/// Total-order comparison for sort (ascending, NaN greatest, §6.11-0007). Ties are
/// resolved by the stable sort (lower original index first).
fn cmp_key<T: ScalarFloat>(a: T, b: T) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.to_f64().partial_cmp(&b.to_f64()).unwrap_or(Ordering::Equal),
    }
}

/// **sort_network** (§6.11-0007): stable per-line permutation along `axis` under
/// the total order (NaN greatest, ties → lower original index). Two outputs: the
/// values as a raw-bit permutation (NaN payload / −0 preserved) and the
/// original-index vector (`i64`).
pub fn sort_network<T: ScalarFloat>(
    keys: &View<T>,
    axis: usize,
    dir: Direction,
) -> Result<(Tensor<T>, IndexTensor), Error> {
    let in_shape = keys.shape();
    let rank = in_shape.len();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let count = numel(in_shape)?;
    if count == 0 {
        let vals = Tensor::from_vec(Vec::new(), in_shape)?;
        let idxs = IndexTensor::new(Vec::new(), in_shape, Dtype::I64)?;
        return Ok((vals, idxs));
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
        let mut pairs: Vec<(i64, T)> = Vec::with_capacity(extent);
        for j in 0..extent {
            coord[axis] = j;
            pairs.push((j as i64, keys.read(&coord[..rank])?));
        }
        pairs.sort_by(|&(_, av), &(_, bv)| {
            let o = cmp_key(av, bv);
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
    let vals = Tensor::from_vec(vals, in_shape)?;
    let idxs = IndexTensor::new(idxs, in_shape, Dtype::I64)?;
    Ok((vals, idxs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    #[test]
    fn reduce_sum_keepdim() {
        // [[1,2,3],[4,5,6]] sum axis 1 -> [[6],[15]] (keepdim).
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = reduce(&x.view(), Monoid::Sum, &[1]).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_empty_axis_is_identity() {
        let x = t(&[], &[2, 0]);
        let r = reduce(&x.view(), Monoid::Sum, &[1]).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn reduce_max_nan_propagates() {
        let x = t(&[1.0, f64::NAN, 3.0], &[3]);
        let r = reduce(&x.view(), Monoid::Max, &[0]).unwrap();
        assert!(r.as_slice()[0].is_nan());
    }

    #[test]
    fn prefix_scan_inclusive_exclusive() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let inc = prefix_scan(&x.view(), Monoid::Sum, 0, false).unwrap();
        assert_eq!(inc.as_slice(), &[1.0, 3.0, 6.0, 10.0]);
        let exc = prefix_scan(&x.view(), Monoid::Sum, 0, true).unwrap();
        assert_eq!(exc.as_slice(), &[0.0, 1.0, 3.0, 6.0]);
    }

    #[test]
    fn gather_zero_fill_and_clamp() {
        let d = t(&[10.0, 20.0, 30.0], &[3]);
        let idx = IndexTensor::new(vec![0, 2, 5, -1], &[4], Dtype::I64).unwrap();
        let zf = gather(&d.view(), &idx, 0, OobPolicy::ZeroFill, None).unwrap();
        assert_eq!(zf.as_slice(), &[10.0, 30.0, 0.0, 0.0]);
        let cl = gather(&d.view(), &idx, 0, OobPolicy::Clamp, None).unwrap();
        assert_eq!(cl.as_slice(), &[10.0, 30.0, 30.0, 10.0]);
    }

    #[test]
    fn scatter_add_accumulates_duplicates() {
        let dest = t(&[0.0, 0.0, 0.0], &[3]);
        let idx = IndexTensor::new(vec![0, 1, 0], &[3], Dtype::I64).unwrap();
        let upd = t(&[1.0, 2.0, 4.0], &[3]);
        let out = scatter(dest, &idx, &upd.view(), 0, Combine::AtomicAdd).unwrap();
        // index 0 gets 1+4=5, index 1 gets 2.
        assert_eq!(out.as_slice(), &[5.0, 2.0, 0.0]);
    }

    #[test]
    fn scatter_assign_highest_source_wins() {
        let dest = t(&[0.0, 0.0], &[2]);
        let idx = IndexTensor::new(vec![0, 0], &[2], Dtype::I64).unwrap();
        let upd = t(&[7.0, 9.0], &[2]);
        let out = scatter(dest, &idx, &upd.view(), 0, Combine::Assign).unwrap();
        assert_eq!(out.as_slice(), &[9.0, 0.0]); // highest source (9) wins
    }

    #[test]
    fn element_map_over_rank_errors_not_panics() {
        // Empty inputs + rank-9 out_shape must return RankExceeded, not panic
        // (regression, adversarial review — the broadcast loop is skipped when
        // inputs is empty, so map_views' own rank guard is the safety net).
        use kiss_ops_vocab::decomp::Expr;
        let r = element_map::<f64>(&Expr::Input(0), &[], &[1usize; 9]);
        assert!(matches!(r, Err(Error::RankExceeded { .. })));
    }

    #[test]
    fn sort_network_nan_greatest_and_index() {
        let x = t(&[3.0, 1.0, f64::NAN, 2.0], &[4]);
        let (vals, idx) = sort_network(&x.view(), 0, Direction::Asc).unwrap();
        assert_eq!(&vals.as_slice()[..3], &[1.0, 2.0, 3.0]);
        assert!(vals.as_slice()[3].is_nan()); // NaN last (ascending)
        assert_eq!(idx.as_slice(), &[1, 3, 0, 2]);
    }
}
