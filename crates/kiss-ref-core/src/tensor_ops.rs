//! The tensor non-primitives (§6.13), each a spec-faithful transcription of its
//! §6.13 reference decomposition over the six structural atoms + the scalar engine.
//!
//! Every function's doc comment quotes the verbatim §6.13 string it implements, so
//! it is auditable against the spec exactly like the elementwise decompositions in
//! `kiss-ops-vocab/src/decomp.rs`. This first cut is the **float lane**
//! (`T: ScalarFloat`); reductions/scans keep the reduced axis (keepdim, §6.11-0008)
//! so the result broadcasts back for the normalization composites.
//!
//! Float `sum`/`prod` reductions/scans and float `scatter_add` are
//! order-invariant/nondeterministic (§6.0-0004): compare their results under
//! tolerance, never byte-exact. Use [`crate::bridge::DetClass`] /
//! [`op_det`] to classify. `avg_pool`/`max_pool`/`im2col` (the window family) are a
//! documented follow-up — they need a windowed-view machinery not built here.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;

use crate::attrs::{Combine, Direction, Monoid, OobPolicy};
use crate::bridge::DetClass;
use crate::kernels::{gather, map_views, prefix_scan, reduce, scatter, sort_network};
use crate::resolve::eval_op;
use crate::scalar::ScalarFloat;
use crate::tensor::{
    broadcast_shapes, numel, row_major_index, IndexTensor, Odometer, Tensor, View, MAX_RANK,
};
use crate::Error;

// ---- small elementwise helpers over the scalar engine ------------------------

/// Elementwise unary `op` over `a` (same shape).
fn un<T: ScalarFloat>(op: Op, a: &View<T>) -> Result<Tensor<T>, Error> {
    map_views(&[*a], a.shape(), |b| eval_op(op, &[b[0]]))
}

/// Elementwise binary `op` over the broadcast of `a` and `b` (§6.20-0007).
fn bin<T: ScalarFloat>(op: Op, a: &View<T>, b: &View<T>) -> Result<Tensor<T>, Error> {
    let (s, r) = broadcast_shapes(&[a.shape(), b.shape()])?;
    map_views(&[*a, *b], &s[..r], |v| eval_op(op, &[v[0], v[1]]))
}

/// Elementwise `op(a, scalar)`.
fn scalar_rhs<T: ScalarFloat>(op: Op, a: &View<T>, c: T) -> Result<Tensor<T>, Error> {
    map_views(&[*a], a.shape(), |b| eval_op(op, &[b[0], c]))
}

/// The number of elements folded away by reducing `shape` over `axes` — the
/// `reduced_count` leaf (§6.12-0001).
fn reduced_count<T: ScalarFloat>(shape: &[usize], axes: &[usize]) -> Result<T, Error> {
    let rank = shape.len();
    let mut c: usize = 1;
    for &a in axes {
        if a >= rank {
            return Err(Error::AxisOutOfRange { axis: a, rank });
        }
        c = c.checked_mul(shape[a]).ok_or(Error::ShapeOverflow)?;
    }
    Ok(T::from_f64(c as f64))
}

// ---- reductions --------------------------------------------------------------

/// `reduce_mean` — §6.13: `div(reduce(sum, x), reduced_count)`.
pub fn reduce_mean<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let s = reduce(x, Monoid::Sum, axes)?;
    let c = reduced_count::<T>(x.shape(), axes)?;
    scalar_rhs(Op::Div, &s.view(), c)
}

/// `reduce_norm2` — §6.13: `sqrt(reduce(sum, sqr(x)))`.
pub fn reduce_norm2<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let sq = map_views(&[*x], x.shape(), |b| eval_op(Op::Mul, &[b[0], b[0]]))?;
    let s = reduce(&sq.view(), Monoid::Sum, axes)?;
    un(Op::Sqrt, &s.view())
}

/// `reduce_var` — §6.13: `sub(reduce_mean(sqr(x)), sqr(reduce_mean(x)))`.
pub fn reduce_var<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let sq = map_views(&[*x], x.shape(), |b| eval_op(Op::Mul, &[b[0], b[0]]))?;
    let mean_sq = reduce_mean(&sq.view(), axes)?;
    let mean = reduce_mean(x, axes)?;
    let mean2 = map_views(&[mean.view()], mean.shape(), |b| eval_op(Op::Mul, &[b[0], b[0]]))?;
    bin(Op::Sub, &mean_sq.view(), &mean2.view())
}

/// `reduce_std` — §6.13: `sqrt(reduce_var(x))`.
pub fn reduce_std<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let v = reduce_var(x, axes)?;
    un(Op::Sqrt, &v.view())
}

/// `logsumexp` — §6.13: `m=reduce(max,x); out=add(m, log(reduce(sum, exp(sub(x,m)))))`.
pub fn logsumexp<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let m = reduce(x, Monoid::Max, axes)?; // keepdim
    let shifted = bin(Op::Sub, x, &m.view())?;
    let e = un(Op::Exp, &shifted.view())?;
    let se = reduce(&e.view(), Monoid::Sum, axes)?;
    let log_se = un(Op::Log, &se.view())?;
    bin(Op::Add, &m.view(), &log_se.view())
}

/// `any` — §6.13: `reduce(max, cmp_ne(x, const(0)))`.
pub fn any<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let ne = scalar_rhs(Op::CmpNe, x, T::ZERO)?;
    reduce(&ne.view(), Monoid::Max, axes)
}

/// `all` — §6.13: `reduce(min, cmp_ne(x, const(0)))`.
pub fn all<T: ScalarFloat>(x: &View<T>, axes: &[usize]) -> Result<Tensor<T>, Error> {
    let ne = scalar_rhs(Op::CmpNe, x, T::ZERO)?;
    reduce(&ne.view(), Monoid::Min, axes)
}

/// `argmax` — §6.13: the original-index at rank 0 of `sort_network(desc, keys=x)`
/// along `axis`. Returns an index tensor with `axis` kept as extent 1.
pub fn argmax<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<IndexTensor, Error> {
    let rank = x.rank();
    if axis >= rank {
        return Err(Error::AxisOutOfRange { axis, rank });
    }
    let (_, idx) = sort_network(x, axis, Direction::Desc)?;
    // out shape = x.shape with axis set to 1; value = sorted-rank-0 index. `idx`
    // is contiguous row-major with `idx.shape() == x.shape()`, so read its payload
    // directly at the axis-0 coordinate.
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
        coord[axis] = 0; // rank-0 of the descending sort = the argmax
        let lin = row_major_index(&coord[..rank], in_shape);
        data.push(*isl.get(lin).ok_or(Error::ShapeMismatch { expected: isl.len(), got: lin })?);
    }
    IndexTensor::new(data, out_shape, Dtype::I64)
}

// ---- scans -------------------------------------------------------------------

/// `cumsum` — §6.13: `prefix_scan(monoid=sum, inclusive)` along `axis`.
pub fn cumsum<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<Tensor<T>, Error> {
    prefix_scan(x, Monoid::Sum, axis, false)
}

/// `cumprod` — §6.13: `prefix_scan(monoid=prod, inclusive)` along `axis`.
pub fn cumprod<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<Tensor<T>, Error> {
    prefix_scan(x, Monoid::Prod, axis, false)
}

/// `cummax` — §6.13: `prefix_scan(monoid=max, inclusive)` along `axis`.
pub fn cummax<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<Tensor<T>, Error> {
    prefix_scan(x, Monoid::Max, axis, false)
}

// ---- normalizations ----------------------------------------------------------

/// `softmax` — §6.13: `m=reduce(max,x); e=exp(sub(x,m)); s=reduce(sum,e); out=div(e,s)`.
pub fn softmax<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<Tensor<T>, Error> {
    let m = reduce(x, Monoid::Max, &[axis])?;
    let shifted = bin(Op::Sub, x, &m.view())?;
    let e = un(Op::Exp, &shifted.view())?;
    let s = reduce(&e.view(), Monoid::Sum, &[axis])?;
    bin(Op::Div, &e.view(), &s.view())
}

/// `log_softmax` — §6.13: `sub(x, logsumexp(x))` along `axis`.
pub fn log_softmax<T: ScalarFloat>(x: &View<T>, axis: usize) -> Result<Tensor<T>, Error> {
    let lse = logsumexp(x, &[axis])?;
    bin(Op::Sub, x, &lse.view())
}

/// `rms_norm` — §6.13: `ms=reduce_mean(sqr(x)); out=mul(mul(x, rsqrt(add(ms, eps))), gamma)`.
pub fn rms_norm<T: ScalarFloat>(
    x: &View<T>,
    gamma: &View<T>,
    eps: T,
    axis: usize,
) -> Result<Tensor<T>, Error> {
    let sq = map_views(&[*x], x.shape(), |b| eval_op(Op::Mul, &[b[0], b[0]]))?;
    let ms = reduce_mean(&sq.view(), &[axis])?;
    let denom = map_views(&[ms.view()], ms.shape(), |b| {
        eval_op(Op::Rsqrt, &[eval_op(Op::Add, &[b[0], eps])?])
    })?;
    let normed = bin(Op::Mul, x, &denom.view())?;
    bin(Op::Mul, &normed.view(), gamma)
}

/// `layer_norm` — §6.13:
/// `mu=reduce_mean(x); v=reduce_var(x); out=add(mul(mul(sub(x,mu), rsqrt(add(v,eps))), gamma), beta)`.
pub fn layer_norm<T: ScalarFloat>(
    x: &View<T>,
    gamma: &View<T>,
    beta: &View<T>,
    eps: T,
    axis: usize,
) -> Result<Tensor<T>, Error> {
    let mu = reduce_mean(x, &[axis])?;
    let v = reduce_var(x, &[axis])?;
    let denom = map_views(&[v.view()], v.shape(), |b| {
        eval_op(Op::Rsqrt, &[eval_op(Op::Add, &[b[0], eps])?])
    })?;
    let centered = bin(Op::Sub, x, &mu.view())?;
    let normed = bin(Op::Mul, &centered.view(), &denom.view())?;
    let scaled = bin(Op::Mul, &normed.view(), gamma)?;
    bin(Op::Add, &scaled.view(), beta)
}

// ---- contraction -------------------------------------------------------------

/// `matmul` — §6.13: `reduce(sum, axis=K) of element_map(mul(input(0), input(1)))`.
/// **Batched:** `[..batch, M, K] · [..batch, K, N] → [..batch, M, N]` — the leading
/// batch dims broadcast (numpy-style), the trailing two are the matrix dims; `K` is
/// contracted in pinned ascending order (float sum → order-invariant/
/// nondeterministic, §6.0-0004). The rank-2 case is `batch = []`.
pub fn matmul<T: ScalarFloat>(a: &View<T>, b: &View<T>) -> Result<Tensor<T>, Error> {
    let ash = a.shape();
    let bsh = b.shape();
    if ash.len() < 2 || bsh.len() < 2 {
        return Err(Error::ShapeMismatch { expected: 2, got: ash.len().min(bsh.len()) });
    }
    let (ar, br) = (ash.len(), bsh.len());
    let (m, k) = (ash[ar - 2], ash[ar - 1]);
    let (k2, n) = (bsh[br - 2], bsh[br - 1]);
    if k != k2 {
        return Err(Error::ShapeMismatch { expected: k, got: k2 });
    }
    // Broadcast the batch dims (everything but the trailing two).
    let (batch_buf, batch_rank) = broadcast_shapes(&[&ash[..ar - 2], &bsh[..br - 2]])?;
    let batch = &batch_buf[..batch_rank];
    let out_rank = batch_rank + 2;
    if out_rank > MAX_RANK {
        return Err(Error::RankExceeded { rank: out_rank, max: MAX_RANK });
    }
    // Full (broadcast) operand shapes: `batch ++ [m,k]` and `batch ++ [k,n]`.
    let mut a_full = [1usize; MAX_RANK];
    let mut b_full = [1usize; MAX_RANK];
    a_full[..batch_rank].copy_from_slice(batch);
    b_full[..batch_rank].copy_from_slice(batch);
    a_full[batch_rank] = m;
    a_full[batch_rank + 1] = k;
    b_full[batch_rank] = k;
    b_full[batch_rank + 1] = n;
    let av = a.broadcast_to(&a_full[..out_rank])?;
    let bv = b.broadcast_to(&b_full[..out_rank])?;

    let mut out_shape = [1usize; MAX_RANK];
    out_shape[..batch_rank].copy_from_slice(batch);
    out_shape[batch_rank] = m;
    out_shape[batch_rank + 1] = n;
    let out_shape = &out_shape[..out_rank];

    let count = numel(out_shape)?;
    let mut data: Vec<T> = Vec::with_capacity(count);
    let mut acoord = [0usize; MAX_RANK];
    let mut bcoord = [0usize; MAX_RANK];
    let mut od = Odometer::new(out_shape)?;
    while let Some(oc) = od.next_coord() {
        // oc = batch(batch_rank) ++ [i, j]
        acoord[..batch_rank].copy_from_slice(&oc[..batch_rank]);
        bcoord[..batch_rank].copy_from_slice(&oc[..batch_rank]);
        acoord[batch_rank] = oc[batch_rank]; // a's M index = i
        bcoord[batch_rank + 1] = oc[batch_rank + 1]; // b's N index = j
        let mut acc = T::ZERO;
        for p in 0..k {
            acoord[batch_rank + 1] = p; // a's K index
            bcoord[batch_rank] = p; // b's K index
            let prod =
                eval_op(Op::Mul, &[av.read(&acoord[..out_rank])?, bv.read(&bcoord[..out_rank])?])?;
            acc = eval_op(Op::Add, &[acc, prod])?;
        }
        data.push(acc);
    }
    Tensor::from_vec(data, out_shape)
}

// ---- gather / scatter family -------------------------------------------------

/// `index_select` — §6.13: `gather(oob=skip)` with a 1-D index along `axis`.
/// (Skip on an in-range index is a plain read; an OOB index under skip with no
/// base is an error — the gather-skip RFC gap.)
pub fn index_select<T: ScalarFloat>(
    data: &View<T>,
    index: &IndexTensor,
    axis: usize,
) -> Result<Tensor<T>, Error> {
    gather(data, index, axis, OobPolicy::Skip, None)
}

/// `embedding` — §6.13: `gather(oob=zero-fill)` with a 1-D index, `axis=0`.
pub fn embedding<T: ScalarFloat>(
    table: &View<T>,
    index: &IndexTensor,
) -> Result<Tensor<T>, Error> {
    gather(table, index, 0, OobPolicy::ZeroFill, None)
}

/// `scatter_add` — §6.13: `scatter(combine=atomic-add, oob=skip)` into `dest`.
/// Float `atomic_add` is order-invariant/nondeterministic (§6.0-0004).
pub fn scatter_add<T: ScalarFloat>(
    dest: Tensor<T>,
    index: &IndexTensor,
    updates: &View<T>,
    axis: usize,
) -> Result<Tensor<T>, Error> {
    scatter(dest, index, updates, axis, Combine::AtomicAdd)
}

// ---- determinism classification ----------------------------------------------

/// The determinism class of a tensor non-primitive on the **float** lane (§6.0):
/// the float-summation ops (reductions/means/norms/softmaxes/matmul/scatter_add)
/// are order-invariant/nondeterministic; the transcendental-bearing ones inherit
/// a ULP bound; pure max/min/data-movement are exact-byte.
pub fn op_det(op: Op) -> DetClass {
    match op {
        // float sum/prod contraction → nondeterministic. `avg_pool` (a deferred
        // window op) is `reduce_mean` over the window, so it carries a float sum
        // and is named in §6.0-0004 — classify it here, not via the exact default.
        Op::ReduceMean
        | Op::ReduceVar
        | Op::ReduceStd
        | Op::ReduceNorm2
        | Op::Matmul
        | Op::ScatterAdd
        | Op::Cumsum
        | Op::Cumprod
        | Op::AvgPool => DetClass::OrderInvariantNondeterministic,
        // transcendental-bearing → ULP (exp/log 4 ULP, sqrt 2 ULP; also carry the
        // nondeterministic sum inside, so most-permissive is nondeterministic)
        Op::Softmax | Op::LogSoftmax | Op::Logsumexp | Op::RmsNorm | Op::LayerNorm => {
            DetClass::OrderInvariantNondeterministic
        }
        // max/min reductions, scans, gathers, sorts, argmax → exact-byte; also the
        // deferred `max_pool` (max reduction) and `im2col` (pure data movement).
        _ => DetClass::ExactByte,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_vec(data.to_vec(), shape).unwrap()
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let s = softmax(&x.view(), 1).unwrap();
        // each row sums to 1.
        close(&[s.as_slice()[0] + s.as_slice()[1] + s.as_slice()[2]], &[1.0]);
        close(&[s.as_slice()[3] + s.as_slice()[4] + s.as_slice()[5]], &[1.0]);
        // uniform row → 1/3 each.
        close(&s.as_slice()[3..6], &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn reduce_mean_and_var() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let m = reduce_mean(&x.view(), &[0]).unwrap();
        close(m.as_slice(), &[2.5]);
        let v = reduce_var(&x.view(), &[0]).unwrap();
        close(v.as_slice(), &[1.25]); // population variance
    }

    #[test]
    fn logsumexp_matches_naive() {
        let x = t(&[0.0, 1.0, 2.0], &[3]);
        let lse = logsumexp(&x.view(), &[0]).unwrap();
        let naive = (0f64.exp() + 1f64.exp() + 2f64.exp()).ln();
        close(lse.as_slice(), &[naive]);
    }

    #[test]
    fn matmul_2x3_3x2() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [1,2,3]·[7,9,11]=58 ; [1,2,3]·[8,10,12]=64 ; [4,5,6]·[7,9,11]=139 ; ·[8,10,12]=154
        close(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_batched_and_broadcast() {
        // 2 batches of [2,3]·[3,2].
        let seq: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let a = Tensor::from_vec(seq.clone(), &[2, 2, 3]).unwrap();
        let b = Tensor::from_vec(seq, &[2, 3, 2]).unwrap();
        let c = matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        // batch0 [[1,2,3],[4,5,6]]·[[1,2],[3,4],[5,6]] = [[22,28],[49,64]];
        // batch1 [[7,8,9],[10,11,12]]·[[7,8],[9,10],[11,12]] = [[220,244],[301,334]].
        close(c.as_slice(), &[22., 28., 49., 64., 220., 244., 301., 334.]);
        // broadcast b's (absent) batch: [2,2,3]·[3,2] → [2,2,2].
        let b2 = t(&[1., 0., 0., 1., 1., 1.], &[3, 2]);
        let c2 = matmul(&a.view(), &b2.view()).unwrap();
        assert_eq!(c2.shape(), &[2, 2, 2]);
    }

    #[test]
    fn argmax_picks_max_index() {
        let x = t(&[1.0, 9.0, 3.0, 2.0], &[4]);
        let a = argmax(&x.view(), 0).unwrap();
        assert_eq!(a.as_slice(), &[1]);
    }

    #[test]
    fn cumsum_and_any_all() {
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        close(cumsum(&x.view(), 0).unwrap().as_slice(), &[1.0, 3.0, 6.0]);
        let z = t(&[0.0, 1.0, 0.0], &[3]);
        close(any(&z.view(), &[0]).unwrap().as_slice(), &[1.0]);
        close(all(&z.view(), &[0]).unwrap().as_slice(), &[0.0]);
    }

    #[test]
    fn determinism_tags() {
        // avg_pool carries a float sum → nondeterministic (§6.0-0004); max_pool is
        // a max reduction → exact-byte (regression, adversarial review).
        assert_eq!(op_det(Op::AvgPool), DetClass::OrderInvariantNondeterministic);
        assert_eq!(op_det(Op::MaxPool), DetClass::ExactByte);
        assert_eq!(op_det(Op::ScatterAdd), DetClass::OrderInvariantNondeterministic);
        assert_eq!(op_det(Op::Argmax), DetClass::ExactByte);
    }

    #[test]
    fn index_select_and_embedding() {
        let table = t(&[10.0, 11.0, 20.0, 21.0, 30.0, 31.0], &[3, 2]);
        let idx = IndexTensor::new(alloc::vec![2, 0], &[2], Dtype::I64).unwrap();
        let sel = index_select(&table.view(), &idx, 0).unwrap();
        close(sel.as_slice(), &[30.0, 31.0, 10.0, 11.0]);
        let idx2 = IndexTensor::new(alloc::vec![1, 9], &[2], Dtype::I64).unwrap();
        let emb = embedding(&table.view(), &idx2).unwrap();
        // row 1 = [20,21]; OOB row 9 → zero-fill.
        close(emb.as_slice(), &[20.0, 21.0, 0.0, 0.0]);
    }
}
