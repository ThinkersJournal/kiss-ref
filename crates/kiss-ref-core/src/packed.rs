// SPDX-License-Identifier: MIT OR Apache-2.0
//! The tuple-state `reduce` monoids `welford`, `online_softmax` and `argmax` (KISS-Ops §6.11-0026 to
//! §6.11-0029, board 427 "2) d)"). A reduction over one of them returns **one packed tensor**: the
//! `keepdim` result of §6.11-0008 with one trailing state axis (extent 3 for `welford`, 2 for
//! `online_softmax` and `argmax`), every lane of the one state dtype and carried as a bit pattern
//! (`u64`, low `width` bits significant). A count or index lane is an unsigned integer of the state
//! dtype's width (`u32` beside `f32`, `u64` beside `f64`), so it is never converted through a float.
//!
//! The **state dtype** is the compute dtype when it is `f32` or `f64` and `f32` for a narrower float; a
//! non-float compute dtype is a typed decline. Folds are ascending: `state = merge(state, element)`,
//! element order row-major over the reduced axes, every operation rounded to the state dtype.

extern crate alloc;
use alloc::vec::Vec;

use kiss_classify_vocab::Dtype;

use crate::cast::{cast_bits, CastRounding, CastSaturation};
use crate::scalar::ScalarFloat;
use crate::tensor::{alloc_exact, numel, Odometer, Tensor, View, MAX_RANK};
use crate::Error;

/// The three tuple-state monoids (ordinals 5, 6, 7 of §6.19-0014).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TupleMonoid {
    Welford,
    OnlineSoftmax,
    Argmax,
}

impl TupleMonoid {
    /// The extent of the trailing state axis: 3 for `welford`, 2 for the others.
    pub const fn lanes(self) -> usize {
        match self {
            TupleMonoid::Welford => 3,
            _ => 2,
        }
    }
}

/// The state dtype for a compute dtype (§6.11-0026): `f32`/`f64` as is, `f32` for a narrower float, a typed
/// decline for anything else.
pub fn state_dtype(compute: Dtype) -> Result<Dtype, Error> {
    match compute {
        Dtype::F32 | Dtype::F64 => Ok(compute),
        Dtype::F16 | Dtype::Bf16 | Dtype::F8e4m3fn | Dtype::F8e5m2 => Ok(Dtype::F32),
        d if d.declines_compute() => Err(Error::ReservedOrScaleDtype(d)),
        d => Err(Error::UnsupportedDtype(d)),
    }
}

/// The packed output shape: the `keepdim` result with the reduced axes kept at extent 1 (or removed) and one
/// trailing state axis. A rank above `MAX_RANK` is [`Error::RankExceeded`]; `argmax` over other than exactly one
/// axis is [`Error::ArgmaxAxisCount`].
pub fn packed_shape(
    in_shape: &[usize],
    axes: &[usize],
    keepdim: bool,
    kind: TupleMonoid,
) -> Result<Vec<usize>, Error> {
    if axes.is_empty() {
        return Err(Error::EmptyAxesMask);
    }
    let rank = in_shape.len();
    let mut reduced = [false; MAX_RANK];
    for &a in axes {
        if a >= rank || a >= MAX_RANK {
            return Err(Error::AxisOutOfRange { axis: a, rank });
        }
        reduced[a] = true;
    }
    if kind == TupleMonoid::Argmax && reduced.iter().filter(|r| **r).count() != 1 {
        return Err(Error::ArgmaxAxisCount {
            got: reduced.iter().filter(|r| **r).count(),
        });
    }
    let mut out = Vec::new();
    for (k, &d) in in_shape.iter().enumerate() {
        if !reduced[k] {
            out.push(d);
        } else if keepdim {
            out.push(1);
        }
    }
    out.push(kind.lanes());
    if out.len() > MAX_RANK {
        return Err(Error::RankExceeded {
            rank: out.len(),
            max: MAX_RANK,
        });
    }
    Ok(out)
}

/// A state-dtype float that can be read from and written to its lane bits.
trait Lane: ScalarFloat {
    /// The largest count/index the unsigned lane holds (`2^w - 1`).
    const MAX_LANE: u64;
    fn to_lane(self) -> u64;
}
impl Lane for f32 {
    const MAX_LANE: u64 = u32::MAX as u64;
    fn to_lane(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl Lane for f64 {
    const MAX_LANE: u64 = u64::MAX;
    fn to_lane(self) -> u64 {
        self.to_bits()
    }
}

fn cvt<T: Lane>(n: u64) -> T {
    T::from_f64(n as f64)
}

fn neg_inf<T: Lane>() -> T {
    T::from_f64(f64::NEG_INFINITY)
}

fn nan<T: Lane>() -> T {
    T::from_f64(f64::NAN)
}

/// `welford` state `(n, mean, M2)` (§6.11-0027).
#[derive(Clone, Copy)]
struct Wel<T> {
    n: u64,
    mean: T,
    m2: T,
}

fn welford_merge<T: Lane>(a: Wel<T>, b: Wel<T>) -> Result<Wel<T>, Error> {
    if a.n == 0 {
        return Ok(b);
    }
    if b.n == 0 {
        return Ok(a);
    }
    let n =
        a.n.checked_add(b.n)
            .filter(|n| *n <= T::MAX_LANE)
            .ok_or(Error::PackedLaneOverflow)?;
    let (na, nb, nt) = (cvt::<T>(a.n), cvt::<T>(b.n), cvt::<T>(n));
    let delta = b.mean.sub(a.mean);
    let mean = a.mean.add(delta.mul(nb.div(nt)));
    let m2 = a.m2.add(b.m2).add(delta.mul(delta).mul(na.mul(nb).div(nt)));
    Ok(Wel { n, mean, m2 })
}

/// `online_softmax` state `(m, l)` (§6.11-0028).
#[derive(Clone, Copy)]
struct Soft<T> {
    m: T,
    l: T,
}

fn soft_identity<T: Lane>() -> Soft<T> {
    Soft {
        m: neg_inf(),
        l: T::ZERO,
    }
}

fn soft_element<T: Lane>(x: T) -> Soft<T> {
    if x.is_nan() || x.to_f64() == f64::INFINITY {
        Soft { m: nan(), l: nan() }
    } else if x.to_f64() == f64::NEG_INFINITY {
        soft_identity()
    } else {
        Soft { m: x, l: T::ONE }
    }
}

fn soft_merge<T: Lane>(a: Soft<T>, b: Soft<T>) -> Soft<T> {
    if a.m.is_nan() || a.l.is_nan() || b.m.is_nan() || b.l.is_nan() {
        return Soft { m: nan(), l: nan() };
    }
    let m = if b.m > a.m { b.m } else { a.m };
    if m.to_f64() == f64::NEG_INFINITY {
        return soft_identity();
    }
    let l = a.l.mul(a.m.sub(m).exp()).add(b.l.mul(b.m.sub(m).exp()));
    Soft { m, l }
}

/// `argmax` state `(value, index)` (§6.11-0029).
#[derive(Clone, Copy)]
struct Arg<T> {
    v: T,
    i: u64,
}

/// `b`'s value is greater than `a`'s under the §6.11-0007 total order (NaN greatest, `+0 == -0`).
fn gt_total<T: Lane>(b: T, a: T) -> bool {
    if b.is_nan() {
        !a.is_nan()
    } else {
        !a.is_nan() && b > a
    }
}

fn eq_total<T: Lane>(a: T, b: T) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

fn arg_merge<T: Lane>(a: Arg<T>, b: Arg<T>) -> Arg<T> {
    if gt_total(b.v, a.v) || (eq_total(b.v, a.v) && b.i < a.i) {
        b
    } else {
        a
    }
}

/// The running state of one output coordinate's fold (only the field of the active monoid is used).
struct Acc<T> {
    wel: Wel<T>,
    soft: Soft<T>,
    arg: Arg<T>,
}

impl<T: Lane> Acc<T> {
    fn identity() -> Self {
        Acc {
            wel: Wel {
                n: 0,
                mean: T::ZERO,
                m2: T::ZERO,
            },
            soft: soft_identity(),
            arg: Arg {
                v: neg_inf(),
                i: T::MAX_LANE,
            },
        }
    }

    /// `state = merge(state, element)` for the element `v` at position `pos` along the reduced axis.
    fn push(&mut self, kind: TupleMonoid, v: T, pos: u64) -> Result<(), Error> {
        match kind {
            TupleMonoid::Welford => {
                let one = Wel {
                    n: 1,
                    mean: v,
                    m2: T::ZERO,
                };
                self.wel = welford_merge(self.wel, one)?;
            }
            TupleMonoid::OnlineSoftmax => self.soft = soft_merge(self.soft, soft_element(v)),
            TupleMonoid::Argmax => {
                if pos >= T::MAX_LANE {
                    return Err(Error::PackedLaneOverflow);
                }
                self.arg = arg_merge(self.arg, Arg { v, i: pos });
            }
        }
        Ok(())
    }

    fn emit(&self, kind: TupleMonoid, out: &mut Vec<u64>) {
        match kind {
            TupleMonoid::Welford => {
                out.extend([self.wel.n, self.wel.mean.to_lane(), self.wel.m2.to_lane()])
            }
            TupleMonoid::OnlineSoftmax => {
                out.extend([self.soft.m.to_lane(), self.soft.l.to_lane()])
            }
            TupleMonoid::Argmax => out.extend([self.arg.v.to_lane(), self.arg.i]),
        }
    }
}

/// One output lane group per kept coordinate: fold the reduced coordinates ascending.
fn fold<T: Lane>(
    kind: TupleMonoid,
    x: &View<u64>,
    compute: Dtype,
    axes: &[usize],
    out_kept: &[usize],
) -> Result<Vec<u64>, Error> {
    let in_shape = x.shape();
    let rank = in_shape.len();
    let mut is_red = [false; MAX_RANK];
    for &a in axes {
        is_red[a] = true;
    }
    let reduced: Vec<usize> = (0..rank).filter(|k| is_red[*k]).collect();
    let mut kept_shape = [1usize; MAX_RANK];
    for k in 0..rank {
        kept_shape[k] = if is_red[k] { 1 } else { in_shape[k] };
    }
    let rext: Vec<usize> = reduced.iter().map(|&a| in_shape[a]).collect();
    let mut lanes: Vec<u64> = alloc_exact(numel(out_kept)?.saturating_mul(kind.lanes()))?;
    let state = state_dtype(compute)?;
    let mut od = Odometer::new(&kept_shape[..rank])?;
    let mut coord = [0usize; MAX_RANK];
    while let Some(oc) = od.next_coord() {
        coord[..rank].copy_from_slice(oc);
        let mut acc = Acc::<T>::identity();
        let mut rod = Odometer::new(&rext)?;
        while let Some(rc) = rod.next_coord() {
            for (j, &a) in reduced.iter().enumerate() {
                coord[a] = rc[j];
            }
            let mut bits = x.read(&coord[..rank])?;
            if compute != state {
                bits = cast_bits(
                    bits,
                    compute,
                    state,
                    CastRounding::NearestEven,
                    CastSaturation::None,
                )?;
            }
            acc.push(kind, T::from_bits(bits), rc[0] as u64)?;
        }
        acc.emit(kind, &mut lanes);
    }
    Ok(lanes)
}

/// `reduce` over a tuple-state monoid: the packed tensor of state lanes (bit patterns in the state dtype).
/// `x` holds `compute`-dtype bit patterns.
pub fn reduce_packed(
    kind: TupleMonoid,
    compute: Dtype,
    x: &View<u64>,
    axes: &[usize],
    keepdim: bool,
) -> Result<Tensor<u64>, Error> {
    let state = state_dtype(compute)?;
    let shape = packed_shape(x.shape(), axes, keepdim, kind)?;
    let kept = &shape[..shape.len() - 1];
    let lanes = if state == Dtype::F64 {
        fold::<f64>(kind, x, compute, axes, kept)?
    } else {
        fold::<f32>(kind, x, compute, axes, kept)?
    };
    Tensor::from_vec(lanes, &shape)
}

/// The `welford` finalise (§6.11-0027): population variance `M2 / n`, or with Bessel's correction
/// `M2 / (n - 1)`; an empty or (corrected) single-element state is `0 / 0`, NaN. Returns the lane bits.
pub fn welford_finalise(state: Dtype, lanes: [u64; 3], bessel: bool) -> Result<u64, Error> {
    fn go<T: Lane>(lanes: [u64; 3], bessel: bool) -> u64 {
        let denom = if bessel {
            lanes[0].saturating_sub(1)
        } else {
            lanes[0]
        };
        T::from_bits(lanes[2]).div(cvt::<T>(denom)).to_lane()
    }
    match state {
        Dtype::F32 => Ok(go::<f32>(lanes, bessel)),
        Dtype::F64 => Ok(go::<f64>(lanes, bessel)),
        d => Err(Error::UnsupportedDtype(d)),
    }
}

/// The `online_softmax` finalise (§6.11-0028): the logsumexp `m + log(l)`; an empty state is `-inf`.
pub fn online_softmax_finalise(state: Dtype, lanes: [u64; 2]) -> Result<u64, Error> {
    fn go<T: Lane>(l: [u64; 2]) -> u64 {
        T::from_bits(l[0]).add(T::from_bits(l[1]).log()).to_lane()
    }
    match state {
        Dtype::F32 => Ok(go::<f32>(lanes)),
        Dtype::F64 => Ok(go::<f64>(lanes)),
        d => Err(Error::UnsupportedDtype(d)),
    }
}
