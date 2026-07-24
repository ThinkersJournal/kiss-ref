//! Never-panic adversarial-DAG fuzz suite for `eval_recipe` (the recipe evaluator).
//!
//! CONTRACT: `eval_recipe` never panics — every failure is a typed `Error`.
//! Each case builds a (dag, inputs, params, indices) tuple from a deterministic
//! PRNG (xorshift64*, fixed seed bases — no wall-clock, no OS randomness) and
//! evaluates it inside `std::panic::catch_unwind`. `Ok` and `Err` both PASS;
//! a panic FAILS the test and prints the case seed + a Debug dump of the DAG.
//! Debug-profile overflow checks are on, so arithmetic overflow panics are
//! caught as failures too — that is intended.
//!
//! Two generator modes:
//! - MOSTLY-VALID: structurally plausible DAGs (topological children, tracked
//!   shapes, sized inputs) that drive every Node variant deep into the compute
//!   kernels, with mild in-band noise (occasional OOB indices, odd arities).
//! - ADVERSARIAL: a valid DAG plus injected mutations (out-of-range ids/axes/
//!   slots incl. usize::MAX, cycles, base self-reference, IndexRef::Node abuse,
//!   garbage index_outputs, poison Consts, rank-8/zero-extent shapes, starved
//!   runtime arrays, deep ~10_000-node chains, 1000-wide Apply, dup children),
//!   plus pure-chaos DAGs whose every field is wild.
//!
//! REPRODUCING A FAILURE: the failure message prints `seed=0x…` and the mode.
//! Rerun exactly that case with:
//!   KISS_RECIPE_FUZZ_SEED=0x<seed> KISS_RECIPE_FUZZ_MODE=<valid|adversarial> \
//!     cargo test -p kiss-ref-conformance --test recipe_fuzz repro_from_env -- --ignored --nocapture
//! Seeds are stable for a given generator version; changing the generator
//! invalidates old seeds (standard fuzzer caveat).
//!
//! Budget: the default tests total ~10k cases with bounded sizes (<~10s wall,
//! tests run in parallel). `fuzz_soak_100k` is #[ignore]d for on-demand soaks.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use kiss_classify_vocab::Dtype;
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    eval_recipe, Combine, DetClass, Direction, Error, FlatDag, IndexRef, IndexTensor, Monoid, Node,
    OobPolicy, RecipeEval, ScalarFloat, Tensor, MAX_OPERANDS, MAX_RANK,
};

// ---- iteration budgets (tune here on a slow machine) ------------------------

const VALID_F64_ITERS: u64 = 4000;
const VALID_F32_ITERS: u64 = 1000;
const ADV_F64_ITERS: u64 = 4000;
const ADV_F32_ITERS: u64 = 1000;
const DETCHECK_ITERS: u64 = 100;
const SOAK_ITERS_PER_MODE: u64 = 50_000;

// ---- seed bases (fixed; per-case seed is derived and printed on failure) ----

const SEED_VALID_F64: u64 = 0xD15C_0B1A_5EED_0001;
const SEED_VALID_F32: u64 = 0xD15C_0B1A_5EED_0002;
const SEED_ADV_F64: u64 = 0xD15C_0B1A_5EED_0003;
const SEED_ADV_F32: u64 = 0xD15C_0B1A_5EED_0004;
const SEED_SOAK: u64 = 0xD15C_0B1A_5EED_0005;
const SEED_DETCHECK: u64 = 0xD15C_0B1A_5EED_0006;

// ---- deterministic PRNG -----------------------------------------------------

/// xorshift64* — deterministic, seedable, ~5 lines. NOT crypto; fuzz only.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n.max(1) as u64)) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next() % 100 < pct
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-case seed. The FAILURE MESSAGE prints this value (not base+i), so a
/// repro needs only (seed, mode).
fn case_seed(base: u64, i: u64) -> u64 {
    splitmix64(base ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

// ---- case representation ----------------------------------------------------

#[derive(Clone, Debug)]
struct Case {
    dag: FlatDag,
    inputs: Vec<Tensor<f64>>, // canonical f64; the f32 pass converts via from_f64
    params: Vec<f64>,
    indices: Vec<IndexTensor>,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    MostlyValid,
    Adversarial,
}

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::MostlyValid => "valid",
        Mode::Adversarial => "adversarial",
    }
}

#[derive(Clone, Copy, Debug)]
enum Phase {
    Generate,
    Eval,
}

// ---- value pools ------------------------------------------------------------

const SPECIALS: [f64; 7] = [
    -0.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1e300,
    5e-324,
    f64::MAX,
];

const POISON: [f64; 8] = [
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    -0.0,
    1e308,
    -1e308,
    5e-324,
    f64::MAX,
];

const INDEX_POISON: [i64; 4] = [i64::MIN, i64::MAX, -1, 1 << 40];

const MONOIDS: [Monoid; 4] = [Monoid::Sum, Monoid::Prod, Monoid::Max, Monoid::Min];
const COMBINES: [Combine; 4] = [
    Combine::Assign,
    Combine::AtomicAdd,
    Combine::AtomicMax,
    Combine::AtomicMin,
];
const OOBS: [OobPolicy; 3] = [OobPolicy::Skip, OobPolicy::Clamp, OobPolicy::ZeroFill];

const UNARY_OPS: [Op; 14] = [
    Op::Neg,
    Op::Abs,
    Op::Floor,
    Op::Ceil,
    Op::Sqrt,
    Op::Exp,
    Op::Log,
    Op::Sin,
    Op::Relu,
    Op::Sigmoid,
    Op::Sign,
    Op::Recip,
    Op::Tanh,
    Op::Erf,
];
const BINARY_OPS: [Op; 11] = [
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::CmpLt,
    Op::CmpEq,
    Op::Atan2,
    Op::Pow,
    Op::Copysign,
    Op::MaxProp,
    Op::Hypot,
];

/// Steps of 1/8 in [-125, +125] — exact in both f32 and f64.
fn plain_tame(rng: &mut Rng) -> f64 {
    ((rng.next() % 2001) as i64 - 1000) as f64 / 8.0
}

/// 95% tame, 5% special-pool.
fn tame_value(rng: &mut Rng) -> f64 {
    if rng.chance(5) {
        SPECIALS[rng.below(SPECIALS.len())]
    } else {
        plain_tame(rng)
    }
}

fn chaos_const(rng: &mut Rng) -> f64 {
    if rng.chance(40) {
        POISON[rng.below(POISON.len())]
    } else {
        plain_tame(rng)
    }
}

// ---- shape/tensor generators ------------------------------------------------

const MAX_NODES_HARD: usize = 40;
const MAX_INPUTS: usize = 12;

fn gen_shape(rng: &mut Rng) -> Vec<usize> {
    let rank = rng.below(5); // 0..=4
    let mut s = Vec::with_capacity(rank);
    for _ in 0..rank {
        let e = if rng.chance(8) {
            0
        } else if rank <= 3 {
            1 + rng.below(4)
        } else {
            1 + rng.below(3)
        };
        s.push(e);
    }
    s
}

fn gen_tensor(rng: &mut Rng, shape: &[usize]) -> Tensor<f64> {
    let mut n = 1usize;
    for &d in shape {
        n = n.saturating_mul(d);
    }
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(tame_value(rng));
    }
    Tensor::from_vec(data, shape)
        .unwrap_or_else(|e| panic!("generator bug: tensor of shape {shape:?}: {e:?}"))
}

// ---- MOSTLY-VALID generator -------------------------------------------------

struct Meta {
    shape: Vec<usize>,
    sortlike: bool, // has an index-lane output (SortNetwork)
}

fn pick_where(rng: &mut Rng, metas: &[Meta], f: impl Fn(&Meta) -> bool) -> Option<usize> {
    let cand: Vec<usize> = metas
        .iter()
        .enumerate()
        .filter(|(_, m)| f(m))
        .map(|(k, _)| k)
        .collect();
    if cand.is_empty() {
        None
    } else {
        Some(cand[rng.below(cand.len())])
    }
}

/// A prior node of rank >= 1; falls back to appending a helper input + Bind, or
/// (inputs full) to any prior node — a later consumer just declines.
fn rank1_child(
    rng: &mut Rng,
    nodes: &mut Vec<Node>,
    metas: &mut Vec<Meta>,
    inputs: &mut Vec<Tensor<f64>>,
) -> usize {
    if let Some(c) = pick_where(rng, metas, |m| !m.shape.is_empty()) {
        return c;
    }
    if inputs.len() < MAX_INPUTS {
        let mut sh = gen_shape(rng);
        if sh.is_empty() {
            sh.push(1 + rng.below(4));
        }
        inputs.push(gen_tensor(rng, &sh));
        let slot = inputs.len() - 1;
        metas.push(Meta {
            shape: sh,
            sortlike: false,
        });
        nodes.push(Node::Bind(slot));
        nodes.len() - 1
    } else {
        rng.below(nodes.len().max(1))
    }
}

fn push_const(rng: &mut Rng, nodes: &mut Vec<Node>, metas: &mut Vec<Meta>) {
    let v = if rng.chance(10) {
        SPECIALS[rng.below(SPECIALS.len())]
    } else {
        plain_tame(rng)
    };
    nodes.push(Node::Const(v));
    metas.push(Meta {
        shape: Vec::new(),
        sortlike: false,
    });
}

/// A fresh small index tensor for `gather`/`scatter` against `extent`, with 15%
/// deliberately-OOB values (negatives only under signed dtypes).
fn push_index_tensor(
    rng: &mut Rng,
    indices: &mut Vec<IndexTensor>,
    shape: &[usize],
    extent: usize,
) -> usize {
    let dtype = [Dtype::U32, Dtype::I32, Dtype::I64][indices.len() % 3];
    let mut n = 1usize;
    for &d in shape {
        n = n.saturating_mul(d);
    }
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        let v = if rng.chance(85) {
            rng.below(extent.max(1)) as i64
        } else {
            let pool = [-1i64, -5, extent as i64, extent as i64 + 3];
            let mut v = pool[rng.below(4)];
            if v < 0 && matches!(dtype, Dtype::U32) {
                v = extent as i64 + 3;
            }
            v
        };
        vals.push(v);
    }
    let it = IndexTensor::new(vals, shape, dtype)
        .unwrap_or_else(|e| panic!("generator bug: index tensor of shape {shape:?}: {e:?}"));
    indices.push(it);
    indices.len() - 1
}

#[allow(clippy::too_many_lines)]
fn gen_valid_node(
    rng: &mut Rng,
    nodes: &mut Vec<Node>,
    metas: &mut Vec<Meta>,
    inputs: &mut Vec<Tensor<f64>>,
    params: &[f64],
    indices: &mut Vec<IndexTensor>,
) {
    let i = nodes.len(); // >= 1 (a leaf is always seeded first)
    let mut w = rng.below(23);
    // Near the hard node cap, degrade helper-appending variants to Const/Apply.
    if nodes.len() >= MAX_NODES_HARD - 3 && (11..=20).contains(&w) {
        w = if rng.chance(50) { 2 } else { 6 };
    }
    match w {
        // Bind (weight 2)
        0 | 1 => {
            let slot = rng.below(inputs.len());
            metas.push(Meta {
                shape: inputs[slot].shape().to_vec(),
                sortlike: false,
            });
            nodes.push(Node::Bind(slot));
        }
        // Const (weight 2)
        2 | 3 => push_const(rng, nodes, metas),
        // RuntimeScalar (weight 1)
        4 => {
            if params.is_empty() {
                push_const(rng, nodes, metas);
            } else {
                nodes.push(Node::RuntimeScalar(rng.below(params.len())));
                metas.push(Meta {
                    shape: Vec::new(),
                    sortlike: false,
                });
            }
        }
        // ReducedCount (weight 1)
        5 => {
            let rank = inputs[0].rank();
            let axes: Vec<usize> = (0..rank).filter(|_| rng.chance(40)).collect();
            nodes.push(Node::ReducedCount(axes));
            metas.push(Meta {
                shape: Vec::new(),
                sortlike: false,
            });
        }
        // Apply (weight 5)
        6..=10 => {
            if rng.chance(5) {
                // deliberate arity/support noise → typed Err paths
                let op = Op::ALL[rng.below(Op::ALL.len())];
                let arity = rng.below(4);
                let mut children = Vec::with_capacity(arity);
                for _ in 0..arity {
                    children.push(rng.below(i));
                }
                let shape = children
                    .first()
                    .map(|&c| metas[c].shape.clone())
                    .unwrap_or_default();
                nodes.push(Node::Apply { op, children });
                metas.push(Meta {
                    shape,
                    sortlike: false,
                });
            } else {
                let r = rng.below(100);
                let (op, arity) = if r < 45 {
                    (UNARY_OPS[rng.below(UNARY_OPS.len())], 1)
                } else if r < 90 {
                    (BINARY_OPS[rng.below(BINARY_OPS.len())], 2)
                } else {
                    (Op::Select, 3)
                };
                let carrier = rng.below(i);
                let cshape = metas[carrier].shape.clone();
                let mut children = Vec::with_capacity(arity);
                for _ in 0..arity {
                    let roll = rng.below(100);
                    let c = if roll < 70 {
                        pick_where(rng, metas, |m| m.shape == cshape).unwrap_or(carrier)
                    } else if roll < 95 {
                        pick_where(rng, metas, |m| m.shape.is_empty()).unwrap_or(carrier)
                    } else {
                        rng.below(i)
                    };
                    children.push(c);
                }
                nodes.push(Node::Apply { op, children });
                metas.push(Meta {
                    shape: cshape,
                    sortlike: false,
                });
            }
        }
        // Reduce (weight 2)
        11 | 12 => {
            let child = rank1_child(rng, nodes, metas, inputs);
            let cshape = metas[child].shape.clone();
            let rank = cshape.len();
            let monoid = MONOIDS[rng.below(4)];
            let mut axes: Vec<usize> = (0..rank).filter(|_| rng.chance(40)).collect();
            if axes.is_empty() {
                axes.push(rng.below(rank.max(1)));
            }
            let keepdim = rng.chance(50);
            let mut shape = Vec::new();
            for (k, &d) in cshape.iter().enumerate() {
                if axes.contains(&k) {
                    if keepdim {
                        shape.push(1);
                    }
                } else {
                    shape.push(d);
                }
            }
            nodes.push(Node::Reduce {
                monoid,
                axes,
                keepdim,
                child,
            });
            metas.push(Meta {
                shape,
                sortlike: false,
            });
        }
        // PrefixScan (weight 1)
        13 => {
            let child = rank1_child(rng, nodes, metas, inputs);
            let cshape = metas[child].shape.clone();
            let axis = rng.below(cshape.len().max(1));
            let monoid = MONOIDS[rng.below(4)];
            let exclusive = rng.chance(50);
            nodes.push(Node::PrefixScan {
                monoid,
                axis,
                exclusive,
                child,
            });
            metas.push(Meta {
                shape: cshape,
                sortlike: false,
            });
        }
        // Matmul (weight 1)
        14 => {
            if rng.chance(80) && inputs.len() + 2 <= MAX_INPUTS {
                let brank = rng.below(3);
                let mut batch = Vec::with_capacity(brank);
                for _ in 0..brank {
                    batch.push(1 + rng.below(2));
                }
                fn dim(rng: &mut Rng) -> usize {
                    if rng.chance(8) {
                        0
                    } else {
                        1 + rng.below(3)
                    }
                }
                let (m, k, nn) = (dim(rng), dim(rng), dim(rng));
                let mut sa = batch.clone();
                sa.push(m);
                sa.push(k);
                let mut sb = batch.clone();
                sb.push(k);
                sb.push(nn);
                inputs.push(gen_tensor(rng, &sa));
                let slot_a = inputs.len() - 1;
                inputs.push(gen_tensor(rng, &sb));
                let slot_b = inputs.len() - 1;
                nodes.push(Node::Bind(slot_a));
                metas.push(Meta {
                    shape: sa,
                    sortlike: false,
                });
                let na = nodes.len() - 1;
                nodes.push(Node::Bind(slot_b));
                metas.push(Meta {
                    shape: sb,
                    sortlike: false,
                });
                let nb = nodes.len() - 1;
                let mut so = batch;
                so.push(m);
                so.push(nn);
                nodes.push(Node::Matmul { lhs: na, rhs: nb });
                metas.push(Meta {
                    shape: so,
                    sortlike: false,
                });
            } else {
                // random-pair arm — usually a typed decline, fine.
                let lhs =
                    pick_where(rng, metas, |m| m.shape.len() >= 2).unwrap_or_else(|| rng.below(i));
                let rhs =
                    pick_where(rng, metas, |m| m.shape.len() >= 2).unwrap_or_else(|| rng.below(i));
                let shape = metas[lhs].shape.clone();
                nodes.push(Node::Matmul { lhs, rhs });
                metas.push(Meta {
                    shape,
                    sortlike: false,
                });
            }
        }
        // Gather (weight 2)
        15 | 16 => {
            let data = if rng.chance(90) {
                pick_where(rng, metas, |m| !m.shape.is_empty()).unwrap_or_else(|| rng.below(i))
            } else {
                rng.below(i)
            };
            let ds = metas[data].shape.clone();
            let axis = rng.below(ds.len().max(1));
            let extent = ds.get(axis).copied().unwrap_or(0);
            let sortlike_ids: Vec<usize> = metas
                .iter()
                .enumerate()
                .filter(|(_, m)| m.sortlike)
                .map(|(k, _)| k)
                .collect();
            let (index, ishape) = if rng.chance(30) && !sortlike_ids.is_empty() {
                let s = sortlike_ids[rng.below(sortlike_ids.len())];
                (IndexRef::Node(s), metas[s].shape.clone())
            } else {
                let irank = rng.below(3);
                let mut ish = Vec::with_capacity(irank);
                for _ in 0..irank {
                    ish.push(rng.below(4));
                }
                let slot = push_index_tensor(rng, indices, &ish, extent);
                (IndexRef::Slot(slot), ish)
            };
            let oob = OOBS[rng.below(3)];
            let base = if oob == OobPolicy::Skip {
                if rng.chance(80) {
                    match pick_where(rng, metas, |m| m.shape.is_empty()) {
                        Some(b) => Some(b),
                        None => {
                            push_const(rng, nodes, metas);
                            Some(nodes.len() - 1)
                        }
                    }
                } else {
                    None
                }
            } else if rng.chance(20) {
                Some(rng.below(i))
            } else {
                None
            };
            let mut shape = Vec::new();
            if axis < ds.len() {
                shape.extend_from_slice(&ds[..axis]);
                shape.extend_from_slice(&ishape);
                shape.extend_from_slice(&ds[axis + 1..]);
            } else {
                shape = ds.clone();
            }
            nodes.push(Node::Gather {
                data,
                index,
                axis,
                oob,
                base,
            });
            metas.push(Meta {
                shape,
                sortlike: false,
            });
        }
        // Scatter (weight 2)
        17 | 18 => {
            let dest = rank1_child(rng, nodes, metas, inputs);
            let s = metas[dest].shape.clone();
            let axis = rng.below(s.len().max(1));
            // Index source FIRST — it fixes the updates extent `l`. Like the
            // Gather arm, ~30% consume a rank-1 sort node's index lane, so the
            // Scatter escalated-producer path is genuinely generated
            // (adversarial-review find: this arm previously hardcoded Slot,
            // leaving the coverage scan's Scatter+Node arm dead code).
            let sortlike_r1: Vec<usize> = metas
                .iter()
                .enumerate()
                .filter(|(_, m)| m.sortlike && m.shape.len() == 1)
                .map(|(k, _)| k)
                .collect();
            let node_index = if rng.chance(30) && !sortlike_r1.is_empty() {
                Some(sortlike_r1[rng.below(sortlike_r1.len())])
            } else {
                None
            };
            let l = match node_index {
                Some(sid) => metas[sid].shape[0],
                None => rng.below(5), // 0..=4
            };
            let mut us = s.clone();
            if axis < us.len() {
                us[axis] = l;
            }
            // Chained gathers can grow a TRACKED shape past MAX_RANK (each hop
            // adds index.rank - 1); the helper input must stay constructible,
            // so over-rank falls back to `dest` (the node just declines at eval).
            let updates = match pick_where(rng, metas, |m| m.shape == us) {
                Some(u) => u,
                None => {
                    if inputs.len() < MAX_INPUTS && us.len() <= MAX_RANK {
                        inputs.push(gen_tensor(rng, &us));
                        nodes.push(Node::Bind(inputs.len() - 1));
                        metas.push(Meta {
                            shape: us.clone(),
                            sortlike: false,
                        });
                        nodes.len() - 1
                    } else {
                        dest
                    }
                }
            };
            let extent = s.get(axis).copied().unwrap_or(0);
            let index = match node_index {
                // sort's index lane is a permutation of 0..l-1 — OOB writes
                // against dest's extent are skipped per spec, so still Ok-able.
                Some(sid) => IndexRef::Node(sid),
                None => IndexRef::Slot(push_index_tensor(rng, indices, &[l], extent)),
            };
            let combine = COMBINES[rng.below(4)];
            nodes.push(Node::Scatter {
                dest,
                index,
                updates,
                axis,
                combine,
            });
            metas.push(Meta {
                shape: s,
                sortlike: false,
            });
        }
        // SortNetwork (weight 2)
        19 | 20 => {
            let keys = rank1_child(rng, nodes, metas, inputs);
            let ks = metas[keys].shape.clone();
            let axis = rng.below(ks.len().max(1));
            let dir = if rng.chance(50) {
                Direction::Asc
            } else {
                Direction::Desc
            };
            nodes.push(Node::SortNetwork { keys, axis, dir });
            metas.push(Meta {
                shape: ks,
                sortlike: true,
            });
        }
        // Iota (weight 1)
        21 => {
            let like = rng.below(i);
            let rank = metas[like].shape.len();
            let axis = if rng.chance(90) {
                rng.below(rank.max(1))
            } else {
                rank + rng.below(2) // deliberate OOR; rank-0 like always declines
            };
            let shape = metas[like].shape.clone();
            nodes.push(Node::Iota { like, axis });
            metas.push(Meta {
                shape,
                sortlike: false,
            });
        }
        // Flip (weight 1)
        _ => {
            let child = rng.below(i);
            let rank = metas[child].shape.len();
            let axis = if rng.chance(90) {
                rng.below(rank.max(1))
            } else {
                rank + rng.below(2) // deliberate OOR; rank-0 child always declines
            };
            let shape = metas[child].shape.clone();
            nodes.push(Node::Flip { child, axis });
            metas.push(Meta {
                shape,
                sortlike: false,
            });
        }
    }
}

fn gen_valid(rng: &mut Rng) -> Case {
    let mut params: Vec<f64> = Vec::new();
    for _ in 0..rng.below(4) {
        params.push(plain_tame(rng));
    }
    let mut inputs: Vec<Tensor<f64>> = Vec::new();
    let s0 = gen_shape(rng);
    let n_in = 1 + rng.below(3);
    for k in 0..n_in {
        let sh = if k == 0 || rng.chance(60) {
            s0.clone()
        } else {
            gen_shape(rng)
        };
        inputs.push(gen_tensor(rng, &sh));
    }
    let mut indices: Vec<IndexTensor> = Vec::new();
    let mut nodes: Vec<Node> = Vec::new();
    let mut metas: Vec<Meta> = Vec::new();
    let target = 2 + rng.below(23);

    // Seed leaf so every later child reference has a prior target.
    let slot = rng.below(inputs.len());
    metas.push(Meta {
        shape: inputs[slot].shape().to_vec(),
        sortlike: false,
    });
    nodes.push(Node::Bind(slot));

    while nodes.len() < target && nodes.len() < MAX_NODES_HARD {
        gen_valid_node(
            rng,
            &mut nodes,
            &mut metas,
            &mut inputs,
            &params,
            &mut indices,
        );
    }

    let n = nodes.len();
    let mut outputs = Vec::new();
    for _ in 0..(1 + rng.below(3)) {
        outputs.push(rng.below(n));
    }
    let sortlike: Vec<usize> = metas
        .iter()
        .enumerate()
        .filter(|(_, m)| m.sortlike)
        .map(|(k, _)| k)
        .collect();
    let mut index_outputs = Vec::new();
    if !sortlike.is_empty() && rng.chance(35) {
        for _ in 0..(1 + rng.below(2)) {
            index_outputs.push(sortlike[rng.below(sortlike.len())]);
        }
    }
    Case {
        dag: FlatDag {
            nodes,
            outputs,
            index_outputs,
        },
        inputs,
        params,
        indices,
    }
}

// ---- ADVERSARIAL generator --------------------------------------------------

/// Mutable references to every node-id-bearing child field of `node`.
fn child_slots(node: &mut Node) -> Vec<&mut usize> {
    let mut v: Vec<&mut usize> = Vec::new();
    match node {
        Node::Bind(_) | Node::Const(_) | Node::RuntimeScalar(_) | Node::ReducedCount(_) => {}
        Node::Apply { children, .. } => v.extend(children.iter_mut()),
        Node::Reduce { child, .. } | Node::PrefixScan { child, .. } => v.push(child),
        Node::Matmul { lhs, rhs } => {
            v.push(lhs);
            v.push(rhs);
        }
        Node::Gather { data, base, .. } => {
            v.push(data);
            if let Some(b) = base {
                v.push(b);
            }
        }
        Node::Scatter { dest, updates, .. } => {
            v.push(dest);
            v.push(updates);
        }
        Node::SortNetwork { keys, .. } => v.push(keys),
        Node::Iota { like, .. } => v.push(like),
        Node::Flip { child, .. } => v.push(child),
    }
    v
}

fn ids_with_child_edges(nodes: &[Node]) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, nd)| {
            !matches!(
                nd,
                Node::Bind(_) | Node::Const(_) | Node::RuntimeScalar(_) | Node::ReducedCount(_)
            )
        })
        .map(|(k, _)| k)
        .collect()
}

fn set_random_child(rng: &mut Rng, node: &mut Node, value: usize) -> bool {
    let mut slots = child_slots(node);
    if slots.is_empty() {
        return false;
    }
    let k = rng.below(slots.len());
    *slots.swap_remove(k) = value;
    true
}

fn point_first_child_at(node: &mut Node, to: usize) -> bool {
    match child_slots(node).into_iter().next() {
        Some(s) => {
            *s = to;
            true
        }
        None => false,
    }
}

fn wild_usize(rng: &mut Rng, n: usize) -> usize {
    let r = rng.below(100);
    if r < 45 {
        rng.below(16)
    } else if r < 65 {
        if rng.chance(50) {
            n.saturating_add(rng.below(4))
        } else {
            n.saturating_sub(rng.below(4))
        }
    } else if r < 75 {
        usize::MAX
    } else {
        rng.below(1 << 20)
    }
}

/// A wild-but-always-OOR runtime slot (occasionally > 255 to exercise the
/// `as u8` truncation in `MissingInput`).
fn wild_slot(rng: &mut Rng, len: usize) -> usize {
    let r = rng.below(100);
    if r < 25 {
        usize::MAX
    } else if r < 50 {
        256 + rng.below(50)
    } else {
        len + rng.below(3)
    }
}

/// Apply mutation `kind` (0..17) to `c`; returns false when no eligible target
/// exists (the caller retargets to the next kind).
#[allow(clippy::too_many_lines)]
fn apply_mutation(rng: &mut Rng, c: &mut Case, kind: usize) -> bool {
    let n = c.dag.nodes.len();
    match kind {
        // M0: child-id OOR (n+below(4) / usize::MAX).
        0 => {
            let cand = ids_with_child_edges(&c.dag.nodes);
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let v = if rng.chance(25) {
                usize::MAX
            } else {
                n + rng.below(4)
            };
            set_random_child(rng, &mut c.dag.nodes[k], v)
        }
        // M1: self-cycle on a child edge.
        1 => {
            let cand = ids_with_child_edges(&c.dag.nodes);
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            set_random_child(rng, &mut c.dag.nodes[k], k)
        }
        // M2: two-node long cycle i0 <-> j.
        2 => {
            if n < 2 {
                return false;
            }
            let a = rng.below(n);
            let b = rng.below(n);
            if a == b {
                return false;
            }
            let (i0, j) = if a < b { (a, b) } else { (b, a) };
            if !point_first_child_at(&mut c.dag.nodes[i0], j) {
                c.dag.nodes[i0] = Node::Apply {
                    op: Op::Add,
                    children: vec![j],
                };
            }
            if !point_first_child_at(&mut c.dag.nodes[j], i0) {
                c.dag.nodes[j] = Node::Apply {
                    op: Op::Add,
                    children: vec![i0],
                };
            }
            true
        }
        // M3: axis abuse.
        3 => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| {
                    matches!(
                        nd,
                        Node::Reduce { .. }
                            | Node::PrefixScan { .. }
                            | Node::Gather { .. }
                            | Node::Scatter { .. }
                            | Node::SortNetwork { .. }
                            | Node::Iota { .. }
                    )
                })
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            match &mut c.dag.nodes[k] {
                Node::Reduce { axes, .. } => {
                    *axes = match rng.below(3) {
                        0 => Vec::new(),
                        1 => vec![0, 0, 0],
                        _ => (0..9).map(|_| rng.below(10)).collect(),
                    };
                }
                Node::PrefixScan { axis, .. }
                | Node::Gather { axis, .. }
                | Node::Scatter { axis, .. }
                | Node::SortNetwork { axis, .. }
                | Node::Iota { axis, .. } => {
                    *axis = if rng.chance(25) {
                        usize::MAX
                    } else {
                        MAX_RANK + rng.below(3)
                    };
                }
                _ => return false,
            }
            true
        }
        // M4: runtime slot OOR (Bind / RuntimeScalar / IndexRef::Slot).
        4 => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| {
                    matches!(
                        nd,
                        Node::Bind(_)
                            | Node::RuntimeScalar(_)
                            | Node::Gather {
                                index: IndexRef::Slot(_),
                                ..
                            }
                            | Node::Scatter {
                                index: IndexRef::Slot(_),
                                ..
                            }
                    )
                })
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let (ni, np, nx) = (c.inputs.len(), c.params.len(), c.indices.len());
            match &mut c.dag.nodes[k] {
                Node::Bind(s) => *s = wild_slot(rng, ni),
                Node::RuntimeScalar(s) => *s = wild_slot(rng, np),
                Node::Gather { index, .. } | Node::Scatter { index, .. } => {
                    *index = IndexRef::Slot(wild_slot(rng, nx));
                }
                _ => return false,
            }
            true
        }
        // M5: Gather.base abuse.
        5 => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd, Node::Gather { .. }))
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let roll = rng.below(5);
            let flip_to = rng.below(n.max(1));
            match &mut c.dag.nodes[k] {
                Node::Gather { oob, base, .. } => match roll {
                    0 => *base = Some(k), // base self-reference: a cycle through base
                    1 => *base = Some(n + rng.below(3)),
                    2 => *base = Some(usize::MAX),
                    3 => *base = if base.is_some() { None } else { Some(flip_to) },
                    _ => {
                        *oob = OobPolicy::Skip;
                        *base = None;
                    }
                },
                _ => return false,
            }
            true
        }
        // M6: IndexRef::Node abuse (incl. the LEGAL forward sort edge).
        6 => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd, Node::Gather { .. } | Node::Scatter { .. }))
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let sorts: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd, Node::SortNetwork { .. }))
                .map(|(m, _)| m)
                .collect();
            let non_sorts: Vec<usize> = (0..n).filter(|m| !sorts.contains(m)).collect();
            let forward_sorts: Vec<usize> = sorts.iter().copied().filter(|&m| m > k).collect();
            let old = match &c.dag.nodes[k] {
                Node::Gather { index, .. } | Node::Scatter { index, .. } => *index,
                _ => return false,
            };
            let new_ref = match rng.below(5) {
                0 => {
                    if non_sorts.is_empty() {
                        IndexRef::Node(rng.below(n.max(1)))
                    } else {
                        IndexRef::Node(non_sorts[rng.below(non_sorts.len())])
                    }
                }
                1 => IndexRef::Node(k),
                2 => IndexRef::Node(usize::MAX),
                3 => {
                    if !forward_sorts.is_empty() {
                        IndexRef::Node(forward_sorts[rng.below(forward_sorts.len())])
                    } else if !sorts.is_empty() {
                        IndexRef::Node(sorts[rng.below(sorts.len())])
                    } else {
                        IndexRef::Node(rng.below(n.max(1)))
                    }
                }
                _ => match old {
                    IndexRef::Slot(s) => IndexRef::Node(s),
                    IndexRef::Node(m) => IndexRef::Slot(m),
                },
            };
            match &mut c.dag.nodes[k] {
                Node::Gather { index, .. } | Node::Scatter { index, .. } => *index = new_ref,
                _ => return false,
            }
            true
        }
        // M7: index_outputs garbage.
        7 => {
            match rng.below(5) {
                0 => c.dag.index_outputs.push(usize::MAX),
                1 => c.dag.index_outputs.push(n + rng.below(3)),
                2 => c.dag.index_outputs.push(rng.below(n.max(1))),
                3 => match c.dag.index_outputs.first().copied() {
                    Some(e) => {
                        for _ in 0..8 {
                            c.dag.index_outputs.push(e);
                        }
                    }
                    None => c.dag.index_outputs.push(wild_usize(rng, n)),
                },
                _ => {
                    c.dag.index_outputs = (0..16).map(|_| wild_usize(rng, n)).collect();
                }
            }
            true
        }
        // M8: outputs garbage.
        8 => {
            match rng.below(3) {
                0 => c.dag.outputs.push(if rng.chance(50) {
                    usize::MAX
                } else {
                    n + rng.below(4)
                }),
                1 => c.dag.outputs.clear(),
                _ => match c.dag.outputs.first().copied() {
                    Some(e) => {
                        for _ in 0..8 {
                            c.dag.outputs.push(e);
                        }
                    }
                    None => c.dag.outputs.push(wild_usize(rng, n)),
                },
            }
            true
        }
        // M9: Const poison rewired into a consumer.
        9 => {
            let v = POISON[rng.below(POISON.len())];
            let consts: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd, Node::Const(_)))
                .map(|(k, _)| k)
                .collect();
            let cid = if !consts.is_empty() && rng.chance(50) {
                let k = consts[rng.below(consts.len())];
                c.dag.nodes[k] = Node::Const(v);
                k
            } else {
                c.dag.nodes.push(Node::Const(v));
                c.dag.nodes.len() - 1
            };
            let cand = ids_with_child_edges(&c.dag.nodes);
            if !cand.is_empty() {
                let k = cand[rng.below(cand.len())];
                set_random_child(rng, &mut c.dag.nodes[k], cid);
            }
            true
        }
        // M10: input shape pathology (rank-8, zero-extent, rank-0).
        10 => {
            let sh: Vec<usize> = match rng.below(3) {
                0 => (0..8).map(|_| 1 + rng.below(2)).collect(),
                1 => {
                    if rng.chance(50) {
                        vec![0]
                    } else {
                        vec![2, 0, 3]
                    }
                }
                _ => Vec::new(),
            };
            let t = gen_tensor(rng, &sh);
            if c.inputs.is_empty() {
                c.inputs.push(t);
            } else {
                let k = rng.below(c.inputs.len());
                c.inputs[k] = t;
            }
            true
        }
        // M11: runtime starvation.
        11 => {
            if rng.chance(50) {
                c.inputs.clear();
            }
            if rng.chance(50) {
                c.params.clear();
            }
            if rng.chance(50) {
                c.indices.clear();
            }
            true
        }
        // M12: input regenerated with a fresh incompatible shape.
        12 => {
            if c.inputs.is_empty() {
                return false;
            }
            let k = rng.below(c.inputs.len());
            let sh = gen_shape(rng);
            c.inputs[k] = gen_tensor(rng, &sh);
            true
        }
        // M13: index-value poison / shape sabotage.
        13 => {
            if c.indices.is_empty() {
                let vals: Vec<i64> = (0..3).map(|_| INDEX_POISON[rng.below(4)]).collect();
                c.indices.push(
                    IndexTensor::new(vals, &[3], Dtype::I64)
                        .unwrap_or_else(|e| panic!("generator bug: poison index: {e:?}")),
                );
                return true;
            }
            let k = rng.below(c.indices.len());
            let rebuilt = match rng.below(3) {
                0 => {
                    let sh = c.indices[k].shape().to_vec();
                    let nvals = c.indices[k].as_slice().len();
                    let vals: Vec<i64> = (0..nvals).map(|_| INDEX_POISON[rng.below(4)]).collect();
                    IndexTensor::new(vals, &sh, Dtype::I64)
                }
                1 => {
                    if rng.chance(50) {
                        IndexTensor::new(vec![INDEX_POISON[rng.below(4)]], &[], Dtype::I64)
                    } else {
                        let vals: Vec<i64> = (0..4).map(|_| rng.below(4) as i64).collect();
                        IndexTensor::new(vals, &[2, 2], Dtype::I64)
                    }
                }
                _ => {
                    let l = rng.below(6);
                    let vals: Vec<i64> = (0..l).map(|_| rng.below(4) as i64).collect();
                    IndexTensor::new(vals, &[l], Dtype::I64)
                }
            };
            c.indices[k] =
                rebuilt.unwrap_or_else(|e| panic!("generator bug: rebuilt index: {e:?}"));
            true
        }
        // M14: injected wide Apply (> MAX_OPERANDS children), routed to an output.
        14 => {
            let width = (MAX_OPERANDS + 1) + rng.below(992); // 9..=1000
            let op = Op::ALL[rng.below(Op::ALL.len())];
            let children: Vec<usize> = (0..width).map(|_| wild_usize(rng, n)).collect();
            c.dag.nodes.push(Node::Apply { op, children });
            let id = c.dag.nodes.len() - 1;
            if c.dag.outputs.is_empty() {
                c.dag.outputs.push(id);
            } else {
                let oi = rng.below(c.dag.outputs.len());
                c.dag.outputs[oi] = id;
            }
            true
        }
        // M15: duplicate children.
        15 => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| {
                    matches!(
                        nd,
                        Node::Apply { .. } | Node::Matmul { .. } | Node::Scatter { .. }
                    )
                })
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let fallback = rng.below(n.max(1));
            match &mut c.dag.nodes[k] {
                Node::Apply { children, .. } => {
                    let ar = children.len().max(1);
                    let cval = children.first().copied().unwrap_or(fallback);
                    *children = vec![cval; ar];
                }
                Node::Matmul { lhs, rhs } => *rhs = *lhs,
                Node::Scatter { dest, updates, .. } => *updates = *dest,
                _ => return false,
            }
            true
        }
        // M16: gather rank blowup — rank-8 index so combined rank > MAX_RANK.
        _ => {
            let cand: Vec<usize> = c
                .dag
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, nd)| matches!(nd, Node::Gather { .. }))
                .map(|(k, _)| k)
                .collect();
            if cand.is_empty() {
                return false;
            }
            let k = cand[rng.below(cand.len())];
            let it = IndexTensor::new(vec![0], &[1; MAX_RANK], Dtype::I64)
                .unwrap_or_else(|e| panic!("generator bug: rank-8 index: {e:?}"));
            c.indices.push(it);
            let slot = c.indices.len() - 1;
            match &mut c.dag.nodes[k] {
                Node::Gather { index, .. } => *index = IndexRef::Slot(slot),
                _ => return false,
            }
            true
        }
    }
}

fn gen_mutated(rng: &mut Rng) -> Case {
    let mut c = gen_valid(rng);
    let count = 1 + rng.below(4);
    for _ in 0..count {
        let mut kind = rng.below(17);
        for _ in 0..17 {
            if apply_mutation(rng, &mut c, kind) {
                break;
            }
            kind = (kind + 1) % 17;
        }
    }
    c
}

fn gen_deep_chain(rng: &mut Rng) -> Case {
    let n = 10_000usize;
    let descending = rng.chance(50);
    let mut nodes: Vec<Node> = Vec::with_capacity(n);
    for k in 0..n {
        let op = if rng.chance(50) { Op::Neg } else { Op::Abs };
        if descending {
            if k + 1 < n {
                nodes.push(Node::Apply {
                    op,
                    children: vec![k + 1],
                });
            } else {
                nodes.push(Node::Bind(0));
            }
        } else if k == 0 {
            nodes.push(Node::Bind(0));
        } else {
            nodes.push(Node::Apply {
                op,
                children: vec![k - 1],
            });
        }
    }
    let outputs = if descending { vec![0] } else { vec![n - 1] };
    if rng.chance(25) {
        // splice a cycle: one random node's child rewritten to a random id
        let k = rng.below(n);
        let target = rng.below(n);
        point_first_child_at(&mut nodes[k], target);
    }
    if rng.chance(10) {
        // OOR leaf Bind slot
        let leaf = if descending { n - 1 } else { 0 };
        nodes[leaf] = Node::Bind(1 + rng.below(4));
    }
    let inputs = vec![gen_tensor(rng, &[])]; // one rank-0 tensor
    Case {
        dag: FlatDag::new(nodes, outputs),
        inputs,
        params: Vec::new(),
        indices: Vec::new(),
    }
}

fn gen_wide_apply(rng: &mut Rng) -> Case {
    let inputs = vec![gen_tensor(rng, &[2]), gen_tensor(rng, &[])];
    let mut nodes: Vec<Node> = Vec::with_capacity(9);
    for k in 0..8usize {
        if rng.chance(50) {
            nodes.push(Node::Bind(k % 2));
        } else {
            nodes.push(Node::Const(plain_tame(rng)));
        }
    }
    let width = (MAX_OPERANDS + 1) + rng.below(992); // 9..=1000, always > MAX_OPERANDS
    let op = Op::ALL[rng.below(Op::ALL.len())];
    let children: Vec<usize> = (0..width).map(|_| rng.below(8)).collect();
    nodes.push(Node::Apply { op, children });
    Case {
        dag: FlatDag::new(nodes, vec![8]),
        inputs,
        params: Vec::new(),
        indices: Vec::new(),
    }
}

fn gen_chaos_tensor(rng: &mut Rng) -> Tensor<f64> {
    let rank = rng.below(6); // 0..=5
    let mut shape: Vec<usize> = (0..rank).map(|_| rng.below(9)).collect(); // extents <= 8
    let mut n: usize = 1;
    let mut ok = true;
    for &d in &shape {
        match n.checked_mul(d) {
            Some(m) if m <= 4096 => n = m,
            _ => {
                ok = false;
                break;
            }
        }
    }
    if !ok {
        for d in shape.iter_mut() {
            *d = 1;
        }
        n = 1;
    }
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(chaos_const(rng));
    }
    Tensor::from_vec(data, &shape)
        .unwrap_or_else(|e| panic!("generator bug: chaos tensor {shape:?}: {e:?}"))
}

fn gen_chaos_index(rng: &mut Rng) -> IndexTensor {
    let rank = rng.below(3);
    let shape: Vec<usize> = (0..rank).map(|_| rng.below(5)).collect(); // numel <= 16
    let mut n = 1usize;
    for &d in &shape {
        n = n.saturating_mul(d);
    }
    let poison = rng.chance(30);
    let dtype = if poison {
        Dtype::I64 // negatives stay legal-by-tag
    } else {
        [Dtype::U32, Dtype::I32, Dtype::I64][rng.below(3)]
    };
    let data: Vec<i64> = (0..n)
        .map(|_| {
            if poison {
                INDEX_POISON[rng.below(INDEX_POISON.len())]
            } else {
                rng.below(8) as i64
            }
        })
        .collect();
    IndexTensor::new(data, &shape, dtype)
        .unwrap_or_else(|e| panic!("generator bug: chaos index {shape:?}: {e:?}"))
}

fn chaos_node(rng: &mut Rng, n: usize) -> Node {
    match rng.below(12) {
        0 => Node::Bind(wild_usize(rng, n)),
        1 => Node::Const(chaos_const(rng)),
        2 => Node::RuntimeScalar(wild_usize(rng, n)),
        3 => Node::ReducedCount((0..rng.below(6)).map(|_| wild_usize(rng, n)).collect()),
        4 => {
            let len = if rng.chance(10) {
                rng.below(200)
            } else {
                rng.below(6)
            };
            Node::Apply {
                op: Op::ALL[rng.below(Op::ALL.len())],
                children: (0..len).map(|_| wild_usize(rng, n)).collect(),
            }
        }
        5 => Node::Reduce {
            monoid: MONOIDS[rng.below(4)],
            axes: (0..rng.below(6)).map(|_| wild_usize(rng, n)).collect(),
            keepdim: rng.chance(50),
            child: wild_usize(rng, n),
        },
        6 => Node::PrefixScan {
            monoid: MONOIDS[rng.below(4)],
            axis: wild_usize(rng, n),
            exclusive: rng.chance(50),
            child: wild_usize(rng, n),
        },
        7 => Node::Matmul {
            lhs: wild_usize(rng, n),
            rhs: wild_usize(rng, n),
        },
        8 => Node::Gather {
            data: wild_usize(rng, n),
            index: if rng.chance(50) {
                IndexRef::Slot(wild_usize(rng, n))
            } else {
                IndexRef::Node(wild_usize(rng, n))
            },
            axis: wild_usize(rng, n),
            oob: OOBS[rng.below(3)],
            base: if rng.chance(50) {
                Some(wild_usize(rng, n))
            } else {
                None
            },
        },
        9 => Node::Scatter {
            dest: wild_usize(rng, n),
            index: if rng.chance(50) {
                IndexRef::Slot(wild_usize(rng, n))
            } else {
                IndexRef::Node(wild_usize(rng, n))
            },
            updates: wild_usize(rng, n),
            axis: wild_usize(rng, n),
            combine: COMBINES[rng.below(4)],
        },
        10 => Node::SortNetwork {
            keys: wild_usize(rng, n),
            axis: wild_usize(rng, n),
            dir: if rng.chance(50) {
                Direction::Asc
            } else {
                Direction::Desc
            },
        },
        _ => Node::Iota {
            like: wild_usize(rng, n),
            axis: wild_usize(rng, n),
        },
    }
}

fn gen_chaos(rng: &mut Rng) -> Case {
    let n = 1 + rng.below(12);
    let nodes: Vec<Node> = (0..n).map(|_| chaos_node(rng, n)).collect();
    let outputs: Vec<usize> = (0..rng.below(4)).map(|_| wild_usize(rng, n)).collect();
    let index_outputs: Vec<usize> = (0..rng.below(3)).map(|_| wild_usize(rng, n)).collect();
    let inputs: Vec<Tensor<f64>> = (0..rng.below(3)).map(|_| gen_chaos_tensor(rng)).collect();
    let params: Vec<f64> = (0..rng.below(3)).map(|_| chaos_const(rng)).collect();
    let indices: Vec<IndexTensor> = (0..rng.below(3)).map(|_| gen_chaos_index(rng)).collect();
    Case {
        dag: FlatDag {
            nodes,
            outputs,
            index_outputs,
        },
        inputs,
        params,
        indices,
    }
}

fn starve_vec<T>(rng: &mut Rng, v: &mut Vec<T>) {
    if rng.chance(60) {
        v.clear();
    } else {
        let keep = rng.below(v.len() + 1);
        v.truncate(keep);
    }
}

fn gen_starved(rng: &mut Rng) -> Case {
    let mut c = gen_valid(rng);
    starve_vec(rng, &mut c.inputs);
    starve_vec(rng, &mut c.params);
    starve_vec(rng, &mut c.indices);
    c
}

/// Case-kind roll is rng-driven (not loop-index-driven) so the printed seed
/// alone fully reproduces the case: deep 0.5%, wide 0.5%, mutate 59%,
/// chaos 25%, starved 15%.
fn gen_adversarial(rng: &mut Rng) -> Case {
    let roll = rng.next() % 1000;
    if roll < 5 {
        gen_deep_chain(rng)
    } else if roll < 10 {
        gen_wide_apply(rng)
    } else if roll < 600 {
        gen_mutated(rng)
    } else if roll < 850 {
        gen_chaos(rng)
    } else {
        gen_starved(rng)
    }
}

// ---- harness ----------------------------------------------------------------

fn build_case(seed: u64, mode: Mode) -> Case {
    let mut rng = Rng::new(seed);
    match mode {
        Mode::MostlyValid => gen_valid(&mut rng),
        Mode::Adversarial => gen_adversarial(&mut rng),
    }
}

fn payload_text(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string payload>".to_string()
    }
}

fn dump_case(c: &Case) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let n = c.dag.nodes.len();
    if n <= 200 {
        let _ = writeln!(s, "dag = {:#?}", c.dag);
    } else {
        let _ = writeln!(s, "<{n} nodes, dump suppressed — regenerate from seed>");
    }
    let _ = writeln!(s, "outputs = {:?}", c.dag.outputs);
    let _ = writeln!(s, "index_outputs = {:?}", c.dag.index_outputs);
    for (k, t) in c.inputs.iter().enumerate() {
        let head: Vec<f64> = t.as_slice().iter().take(16).copied().collect();
        let _ = writeln!(s, "input[{k}]: shape {:?} head {:?}", t.shape(), head);
    }
    let _ = writeln!(s, "params = {:?}", c.params);
    for (k, ix) in c.indices.iter().enumerate() {
        let head: Vec<i64> = ix.as_slice().iter().take(16).copied().collect();
        let _ = writeln!(
            s,
            "index[{k}]: shape {:?} dtype {:?} head {:?}",
            ix.shape(),
            ix.dtype(),
            head
        );
    }
    s
}

fn fail(
    seed: u64,
    mode: Mode,
    phase: Phase,
    dtype: &str,
    case: Option<&Case>,
    payload: Box<dyn Any + Send>,
) -> ! {
    let dump = case
        .map(dump_case)
        .unwrap_or_else(|| "<no case: generation itself panicked>".to_string());
    panic!(
        "recipe_fuzz PANIC: seed={seed:#018x} mode={mode} phase={phase:?} dtype={dtype}\n\
         payload: {payload}\n\
         rerun: KISS_RECIPE_FUZZ_SEED={seed:#x} KISS_RECIPE_FUZZ_MODE={mode} \
         cargo test -p kiss-ref-conformance --test recipe_fuzz repro_from_env -- --ignored --nocapture\n\
         {dump}",
        mode = mode_name(mode),
        payload = payload_text(&*payload),
    );
}

/// Shape-coherence invariants on an `Ok` result. HONESTY NOTE (adversarial-
/// review adjudicated): these three length equalities are invariants of the
/// CURRENT `eval_recipe` by construction — they cannot fail today. They are
/// kept as regression tripwires for future refactors (partial dets, filtered
/// outputs, a lazily-built index lane), NOT as evidence that Ok results are
/// checked here. The falsifiable checking of Ok results lives in
/// `determinism_spot_check` (bitwise double-eval), the `anchor_*` goldens
/// below, and the recipe_corpus golden suite.
fn check_postconditions<T>(seed: u64, mode: Mode, case: &Case, ev: &RecipeEval<T>) {
    assert_eq!(
        ev.dets.len(),
        case.dag.nodes.len(),
        "postcondition dets.len (seed={seed:#018x} mode={})",
        mode_name(mode)
    );
    assert_eq!(
        ev.outputs.len(),
        case.dag.outputs.len(),
        "postcondition outputs.len (seed={seed:#018x} mode={})",
        mode_name(mode)
    );
    assert_eq!(
        ev.index_outputs.len(),
        case.dag.index_outputs.len(),
        "postcondition index_outputs.len (seed={seed:#018x} mode={})",
        mode_name(mode)
    );
}

/// Build + evaluate one case. `Ok`/`Err` both PASS; only a caught panic (or a
/// postcondition violation on `Ok`) FAILS. Returns the case and whether the
/// f64 eval returned `Ok` (for the vacuity guards).
fn run_one(seed: u64, mode: Mode, run_f64: bool, run_f32: bool) -> (Case, bool) {
    // Phase 1: generation, wrapped so even a generator bug fails WITH the seed.
    let case = match catch_unwind(AssertUnwindSafe(|| build_case(seed, mode))) {
        Ok(c) => c,
        Err(p) => fail(seed, mode, Phase::Generate, "-", None, p),
    };
    let mut ok64 = false;
    if run_f64 {
        // Phase 2: ONLY eval_recipe inside the closure.
        let r = catch_unwind(AssertUnwindSafe(|| {
            eval_recipe::<f64>(&case.dag, &case.inputs, &case.params, &case.indices)
        }));
        match r {
            Err(p) => fail(seed, mode, Phase::Eval, "f64", Some(&case), p),
            Ok(Ok(ev)) => {
                check_postconditions(seed, mode, &case, &ev);
                ok64 = true;
            }
            Ok(Err(_)) => {} // typed decline: PASS
        }
    }
    if run_f32 {
        // Conversion outside the eval catch_unwind (cannot fail: same shapes).
        let inputs32: Vec<Tensor<f32>> = case
            .inputs
            .iter()
            .map(|t| {
                let data: Vec<f32> =
                    t.as_slice().iter().map(|&v| <f32 as ScalarFloat>::from_f64(v)).collect();
                Tensor::from_vec(data, t.shape()).unwrap_or_else(|e| {
                    panic!("recipe_fuzz: f32 conversion failed (generator bug) seed={seed:#018x}: {e:?}")
                })
            })
            .collect();
        let params32: Vec<f32> = case
            .params
            .iter()
            .map(|&v| <f32 as ScalarFloat>::from_f64(v))
            .collect();
        let r = catch_unwind(AssertUnwindSafe(|| {
            eval_recipe::<f32>(&case.dag, &inputs32, &params32, &case.indices)
        }));
        match r {
            Err(p) => fail(seed, mode, Phase::Eval, "f32", Some(&case), p),
            Ok(Ok(ev)) => check_postconditions(seed, mode, &case, &ev),
            Ok(Err(_)) => {}
        }
    }
    (case, ok64)
}

fn variant_bit(node: &Node) -> u16 {
    match node {
        Node::Bind(_) => 0,
        Node::Const(_) => 1,
        Node::RuntimeScalar(_) => 2,
        Node::ReducedCount(_) => 3,
        Node::Apply { .. } => 4,
        Node::Reduce { .. } => 5,
        Node::PrefixScan { .. } => 6,
        Node::Matmul { .. } => 7,
        Node::Gather { .. } => 8,
        Node::Scatter { .. } => 9,
        Node::SortNetwork { .. } => 10,
        Node::Iota { .. } => 11,
        Node::Flip { .. } => 12,
    }
}

// ---- anchor cases (hand-computed goldens) -----------------------------------

#[test]
fn anchor_mean_param_recipe() {
    // mean(x) + param: 0=Bind(0), 1=reduce(sum,axis0,nokd), 2=reduced_count([0]),
    // 3=div(1,2), 4=runtime_scalar(0), 5=add(3,4). x=[1,2,3,4], param=10.
    let dag = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![0],
                keepdim: false,
                child: 0,
            },
            Node::ReducedCount(vec![0]),
            Node::Apply {
                op: Op::Div,
                children: vec![1, 2],
            },
            Node::RuntimeScalar(0),
            Node::Apply {
                op: Op::Add,
                children: vec![3, 4],
            },
        ],
        vec![5],
    );
    let want_dets = vec![
        DetClass::ExactByte,                      // n0 Bind
        DetClass::OrderInvariantNondeterministic, // n1 float Sum
        DetClass::ExactByte,                      // n2 ReducedCount
        DetClass::OrderInvariantNondeterministic, // n3 Div: EB ⊔ OIN ⊔ EB
        DetClass::ExactByte,                      // n4 RuntimeScalar
        DetClass::OrderInvariantNondeterministic, // n5 Add: EB ⊔ OIN ⊔ EB
    ];

    let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], &[4]).unwrap();
    let r = eval_recipe::<f64>(&dag, &[x], &[10.0], &[]).unwrap();
    assert_eq!(r.outputs.len(), 1);
    assert_eq!(r.outputs[0].shape(), &[] as &[usize]);
    assert_eq!(r.outputs[0].as_slice(), &[12.5]); // 10/4 + 10, exact
    assert!(r.index_outputs.is_empty());
    assert_eq!(r.dets, want_dets);

    let x32 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
    let r32 = eval_recipe::<f32>(&dag, &[x32], &[10.0f32], &[]).unwrap();
    assert_eq!(r32.outputs[0].shape(), &[] as &[usize]);
    assert_eq!(r32.outputs[0].as_slice(), &[12.5f32]); // exact in f32 too
    assert!(r32.index_outputs.is_empty());
    assert_eq!(r32.dets, want_dets);
}

#[test]
fn anchor_sort_gather_iota_both_lanes() {
    // Both lanes + IndexRef::Node + Gather.base + Iota, all ExactByte.
    // 0=Bind(keys), 1=sort(0), 2=Bind(data), 3=Const(-1), 4=gather(2 by
    // Node(1), skip, base 3), 5=iota(like 2, axis 0).
    let dag = FlatDag {
        nodes: vec![
            Node::Bind(0),
            Node::SortNetwork {
                keys: 0,
                axis: 0,
                dir: Direction::Asc,
            },
            Node::Bind(1),
            Node::Const(-1.0),
            Node::Gather {
                data: 2,
                index: IndexRef::Node(1),
                axis: 0,
                oob: OobPolicy::Skip,
                base: Some(3),
            },
            Node::Iota { like: 2, axis: 0 },
        ],
        outputs: vec![4, 5],
        index_outputs: vec![1],
    };
    let keys = Tensor::from_vec(vec![3.0f64, 1.0, 2.0], &[3]).unwrap();
    let data = Tensor::from_vec(vec![10.0f64, 20.0, 30.0], &[3]).unwrap();
    let r = eval_recipe::<f64>(&dag, &[keys, data], &[], &[]).unwrap();
    // sort asc [3,1,2] → values [1,2,3], original-index perm [1,2,0]
    // gather data by perm → [20,30,10]; iota axis0 over shape [3] → [0,1,2]
    assert_eq!(r.outputs[0].as_slice(), &[20.0, 30.0, 10.0]);
    assert_eq!(r.outputs[1].as_slice(), &[0.0, 1.0, 2.0]);
    assert_eq!(r.index_outputs.len(), 1);
    assert_eq!(r.index_outputs[0].as_slice(), &[1, 2, 0]);
    assert_eq!(r.dets, vec![DetClass::ExactByte; 6]);
}

#[test]
fn anchor_typed_declines() {
    // A3a: self-cycle → the evaluator's cycle sentinel.
    let a = FlatDag::new(
        vec![Node::Apply {
            op: Op::Add,
            children: vec![0, 0],
        }],
        vec![0],
    );
    assert!(matches!(
        eval_recipe::<f64>(&a, &[], &[], &[]),
        Err(Error::BadDecomposition { pos: 0, .. })
    ));

    // A3b: IndexRef::Node at a non-sort node → pre-validation typed error.
    let b = FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Gather {
                data: 0,
                index: IndexRef::Node(0),
                axis: 0,
                oob: OobPolicy::ZeroFill,
                base: None,
            },
        ],
        vec![1],
    );
    let one = Tensor::from_vec(vec![1.0f64], &[1]).unwrap();
    assert!(matches!(
        eval_recipe::<f64>(&b, &[one], &[], &[]),
        Err(Error::IndexSourceInvalid { node: 0 })
    ));

    // A3c: garbage index_outputs → same typed error, usize::MAX intact.
    let c = FlatDag {
        nodes: vec![Node::Bind(0)],
        outputs: vec![0],
        index_outputs: vec![usize::MAX],
    };
    let scalar = Tensor::from_vec(vec![1.0f64], &[]).unwrap();
    assert!(matches!(
        eval_recipe::<f64>(&c, &[scalar], &[], &[]),
        Err(Error::IndexSourceInvalid { node: usize::MAX })
    ));

    // A3d: Iota on a rank-0 like → AxisOutOfRange{0, 0}.
    let d = FlatDag::new(
        vec![Node::Const(0.0), Node::Iota { like: 0, axis: 0 }],
        vec![1],
    );
    assert!(matches!(
        eval_recipe::<f64>(&d, &[], &[], &[]),
        Err(Error::AxisOutOfRange { axis: 0, rank: 0 })
    ));
}

// ---- fuzz tests -------------------------------------------------------------

#[test]
fn fuzz_mostly_valid_f64() {
    let mut ok_count = 0usize;
    let mut variant_mask = 0u16;
    let mut ok_index_outputs = false;
    let mut ok_index_ref_node = false;
    let mut ok_gather_base = false;
    for i in 0..VALID_F64_ITERS {
        let seed = case_seed(SEED_VALID_F64, i);
        let (case, ok64) = run_one(seed, Mode::MostlyValid, true, false);
        if ok64 {
            ok_count += 1;
            for node in &case.dag.nodes {
                variant_mask |= 1 << variant_bit(node);
                match node {
                    Node::Gather {
                        index, base, oob, ..
                    } => {
                        if matches!(index, IndexRef::Node(_)) {
                            ok_index_ref_node = true;
                        }
                        // The base is only READ on a Skip-policy OOB cell
                        // (kernels.rs Resolved::Skip arm) — mere presence is
                        // satisfiable with the base dead on every cell
                        // (adversarial-review find). A NEGATIVE index entry is
                        // OOB for EVERY extent (§6.11-0004, no from-end wrap),
                        // so Skip + base:Some + a negative Slot entry + Ok eval
                        // ⇒ the base-read path executed.
                        if *oob == OobPolicy::Skip && base.is_some() {
                            if let IndexRef::Slot(s) = index {
                                if case
                                    .indices
                                    .get(*s)
                                    .is_some_and(|t| t.as_slice().iter().any(|&v| v < 0))
                                {
                                    ok_gather_base = true;
                                }
                            }
                        }
                    }
                    Node::Scatter { index, .. } => {
                        if matches!(index, IndexRef::Node(_)) {
                            ok_index_ref_node = true;
                        }
                    }
                    _ => {}
                }
            }
            if !case.dag.index_outputs.is_empty() {
                ok_index_outputs = true;
            }
        }
    }
    // Generator-health guards, NOT never-panic violations: if these trip, the
    // mostly-valid generator has rotted into all-Err — tune the generator (or
    // reconcile with a legitimately-stricter evaluator), don't chase a panic.
    assert!(
        ok_count >= 400,
        "generator-health guard: only {ok_count}/{VALID_F64_ITERS} cases returned Ok (floor 400)"
    );
    assert_eq!(
        variant_mask, 0x1FFF,
        "generator-health guard: Node-variant Ok-coverage mask {variant_mask:#015b} != 0x1FFF (all 13 variants)"
    );
    assert!(
        ok_index_outputs,
        "generator-health guard: no Ok case with non-empty index_outputs"
    );
    assert!(
        ok_index_ref_node,
        "generator-health guard: no Ok case containing an IndexRef::Node operand"
    );
    assert!(
        ok_gather_base,
        "generator-health guard: no Ok case where a Skip-gather base was provably READ \
         (Skip + base:Some + negative index entry)"
    );
}

#[test]
fn fuzz_mostly_valid_f32() {
    for i in 0..VALID_F32_ITERS {
        let seed = case_seed(SEED_VALID_F32, i);
        run_one(seed, Mode::MostlyValid, false, true);
    }
}

#[test]
fn fuzz_adversarial_f64() {
    for i in 0..ADV_F64_ITERS {
        let seed = case_seed(SEED_ADV_F64, i);
        run_one(seed, Mode::Adversarial, true, false);
    }
}

#[test]
fn fuzz_adversarial_f32() {
    for i in 0..ADV_F32_ITERS {
        let seed = case_seed(SEED_ADV_F32, i);
        run_one(seed, Mode::Adversarial, false, true);
    }
}

#[test]
fn determinism_spot_check() {
    // eval each case twice; identical Err values, output bit patterns, dets,
    // and index-lane slices — pure-function regression guard.
    for i in 0..DETCHECK_ITERS {
        let seed = case_seed(SEED_DETCHECK, i);
        let case = match catch_unwind(AssertUnwindSafe(|| build_case(seed, Mode::MostlyValid))) {
            Ok(c) => c,
            Err(p) => fail(seed, Mode::MostlyValid, Phase::Generate, "-", None, p),
        };
        let mut evals = Vec::with_capacity(2);
        for _ in 0..2 {
            match catch_unwind(AssertUnwindSafe(|| {
                eval_recipe::<f64>(&case.dag, &case.inputs, &case.params, &case.indices)
            })) {
                Err(p) => fail(seed, Mode::MostlyValid, Phase::Eval, "f64", Some(&case), p),
                Ok(r) => evals.push(r),
            }
        }
        let b = evals.pop().expect("two evals");
        let a = evals.pop().expect("two evals");
        match (a, b) {
            (Err(e1), Err(e2)) => {
                assert_eq!(e1, e2, "determinism: Err values differ (seed={seed:#018x})");
            }
            (Ok(r1), Ok(r2)) => {
                assert_eq!(
                    r1.dets, r2.dets,
                    "determinism: dets differ (seed={seed:#018x})"
                );
                assert_eq!(
                    r1.outputs.len(),
                    r2.outputs.len(),
                    "determinism: outputs.len differ (seed={seed:#018x})"
                );
                for (k, (t1, t2)) in r1.outputs.iter().zip(&r2.outputs).enumerate() {
                    assert_eq!(
                        t1.shape(),
                        t2.shape(),
                        "determinism: output {k} shape differs (seed={seed:#018x})"
                    );
                    for (j, (x, y)) in t1.as_slice().iter().zip(t2.as_slice()).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "determinism: output {k} elem {j} bits differ (seed={seed:#018x})"
                        );
                    }
                }
                assert_eq!(
                    r1.index_outputs, r2.index_outputs,
                    "determinism: index_outputs differ (seed={seed:#018x})"
                );
            }
            _ => panic!("determinism: Ok/Err flip between identical evals (seed={seed:#018x})"),
        }
    }
}

/// On-demand soak: 50k mostly-valid + 50k adversarial f64 cases.
/// Run locally: cargo test -p kiss-ref-conformance --test recipe_fuzz fuzz_soak_100k -- --ignored
#[test]
#[ignore]
fn fuzz_soak_100k() {
    for i in 0..SOAK_ITERS_PER_MODE {
        run_one(case_seed(SEED_SOAK, i), Mode::MostlyValid, true, false);
    }
    for i in SOAK_ITERS_PER_MODE..(2 * SOAK_ITERS_PER_MODE) {
        run_one(case_seed(SEED_SOAK, i), Mode::Adversarial, true, false);
    }
}

fn parse_seed(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

/// Rerun exactly one case from `KISS_RECIPE_FUZZ_SEED` + `KISS_RECIPE_FUZZ_MODE`
/// (the values a failure message prints). Missing vars → usage text, no failure.
#[test]
#[ignore]
fn repro_from_env() {
    let (seed_s, mode_s) = match (
        std::env::var("KISS_RECIPE_FUZZ_SEED").ok(),
        std::env::var("KISS_RECIPE_FUZZ_MODE").ok(),
    ) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            println!(
                "usage: KISS_RECIPE_FUZZ_SEED=0x<hex|decimal> KISS_RECIPE_FUZZ_MODE=<valid|adversarial> \
                 cargo test -p kiss-ref-conformance --test recipe_fuzz repro_from_env -- --ignored --nocapture"
            );
            return;
        }
    };
    let seed = parse_seed(&seed_s)
        .unwrap_or_else(|| panic!("bad KISS_RECIPE_FUZZ_SEED {seed_s:?} (decimal or 0x-hex)"));
    let mode = match mode_s.as_str() {
        "valid" => Mode::MostlyValid,
        "adversarial" => Mode::Adversarial,
        _ => panic!("bad KISS_RECIPE_FUZZ_MODE {mode_s:?} (valid|adversarial)"),
    };
    let case = match catch_unwind(AssertUnwindSafe(|| build_case(seed, mode))) {
        Ok(c) => c,
        Err(p) => fail(seed, mode, Phase::Generate, "-", None, p),
    };
    println!(
        "repro seed={seed:#018x} mode={}\n{}",
        mode_name(mode),
        dump_case(&case)
    );
    let (_, ok64) = run_one(seed, mode, true, true);
    println!(
        "repro result: f64 {}",
        if ok64 { "Ok" } else { "Err (typed decline)" }
    );
}
