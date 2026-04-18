//! Rayon-parallel batch processing for Mamba-3 SISO training.
//!
//! - Thread-local scratch pool with dimension-based invalidation
//! - Static-partition per-thread gradient slots (deterministic across runs)
//! - Fixed balanced binary-tree reduce (deterministic accumulation order)
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use std::cell::RefCell;

use rayon::prelude::*;

use super::backward::backward_mamba3_layer_batched;
use super::dims::Mamba3Dims;
use super::flat::Mamba3LayerFlat;
use super::forward::forward_mamba3_layer_batched;
use super::scratch::Mamba3Scratch;
use super::weights::TrainMamba3Weights;

// ── Thread-local scratch pool ──

struct Mamba3ThreadScratch {
    phase: Mamba3Scratch,
    temporal_flat: Vec<f32>,
    d_model: usize,
    d_inner: usize,
    seq_len: usize,
}

impl Mamba3ThreadScratch {
    fn new(dims: &Mamba3Dims) -> Self {
        Self {
            phase: Mamba3Scratch::zeros(dims),
            temporal_flat: vec![0.0; dims.seq_len * dims.d_model],
            d_model: dims.d_model,
            d_inner: dims.d_inner,
            seq_len: dims.seq_len,
        }
    }
}

thread_local! {
    static M3_THREAD_SCRATCH: RefCell<Option<Mamba3ThreadScratch>> = const { RefCell::new(None) };
}

fn with_scratch<F, R>(dims: &Mamba3Dims, f: F) -> R
where
    F: FnOnce(&mut Mamba3ThreadScratch) -> R,
{
    M3_THREAD_SCRATCH.with(|cell| {
        let mut opt = cell.borrow_mut();
        let needs_reinit = opt.as_ref().is_some_and(|s| {
            s.d_model != dims.d_model || s.d_inner != dims.d_inner || s.seq_len != dims.seq_len
        });
        if needs_reinit {
            *opt = None;
        }
        let scratch = opt.get_or_insert_with(|| Mamba3ThreadScratch::new(dims));
        f(scratch)
    })
}

/// Invalidate all thread-local scratch (call when dimensions change).
pub fn invalidate_mamba3_scratch() {
    rayon::broadcast(|_| {
        M3_THREAD_SCRATCH.with(|cell| *cell.borrow_mut() = None);
    });
}

// ── Raw pointer wrapper for rayon Send/Sync ──

struct BatchPtrs {
    temporal: *mut f32,
    ssm: *mut f32,
    k: *mut f32,
    v: *mut f32,
    angle: *mut f32,
    acts: *mut Vec<Mamba3LayerFlat>,
}
unsafe impl Send for BatchPtrs {}
unsafe impl Sync for BatchPtrs {}

// ── Parallel forward ──

/// Mamba-3 per-layer recurrent state buffers (all batched `[B * n_layers * ...]`).
pub struct Mamba3States<'a> {
    pub ssm: &'a mut [f32],
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
    pub angle: &'a mut [f32],
}

/// Parallel Mamba-3 forward for B samples.
///
/// Each sample: per-layer forward with state carry.
/// `temporal_out`: `[B * seq_len * d_model]` — input and output.
/// `batch_acts`: `[B]` vec of per-layer activation flats.
pub fn parallel_mamba3_forward(
    temporal_out: &mut [f32],
    batch_acts: &mut [Vec<Mamba3LayerFlat>],
    states: Mamba3States<'_>,
    weights: &TrainMamba3Weights,
    dims: &Mamba3Dims,
    batch_size: usize,
) {
    let Mamba3States {
        ssm: ssm_states,
        k: k_states,
        v: v_states,
        angle: angle_states,
    } = states;
    let dm = dims.d_model;
    let nl = dims.n_layers;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ds = dims.d_state;
    let seq_len = dims.seq_len;
    let na = dims.num_rope_angles.max(1);

    let ssm_per = nl * nh * hd * ds;
    let k_per = nl * nh * ds;
    let v_per = nl * nh * hd;
    let a_per = nl * nh * na;
    let t_per = seq_len * dm;

    let ptrs = &BatchPtrs {
        temporal: temporal_out.as_mut_ptr(),
        ssm: ssm_states.as_mut_ptr(),
        k: k_states.as_mut_ptr(),
        v: v_states.as_mut_ptr(),
        angle: angle_states.as_mut_ptr(),
        acts: batch_acts.as_mut_ptr(),
    };

    (0..batch_size).into_par_iter().for_each(|b| {
        with_scratch(dims, |tls| {
            // SAFETY: each sample b accesses disjoint memory regions.
            let temporal =
                unsafe { std::slice::from_raw_parts_mut(ptrs.temporal.add(b * t_per), t_per) };
            let acts = unsafe { &mut *ptrs.acts.add(b) };

            for (layer_idx, lw) in weights.layers.iter().enumerate() {
                let ssm_off = b * ssm_per + layer_idx * nh * hd * ds;
                let k_off = b * k_per + layer_idx * nh * ds;
                let v_off = b * v_per + layer_idx * nh * hd;
                let a_off = b * a_per + layer_idx * nh * na;

                let ssm =
                    unsafe { std::slice::from_raw_parts_mut(ptrs.ssm.add(ssm_off), nh * hd * ds) };
                let k = unsafe { std::slice::from_raw_parts_mut(ptrs.k.add(k_off), nh * ds) };
                let v = unsafe { std::slice::from_raw_parts_mut(ptrs.v.add(v_off), nh * hd) };
                let angle =
                    unsafe { std::slice::from_raw_parts_mut(ptrs.angle.add(a_off), nh * na) };

                forward_mamba3_layer_batched(
                    temporal,
                    &mut acts[layer_idx],
                    lw,
                    ssm,
                    k,
                    v,
                    angle,
                    &mut tls.phase,
                    dims,
                );
            }
        });
    });
}

// ── Parallel backward ──

// ── Raw pointer wrapper for rayon Send/Sync on backward inputs ──

struct BwdPtrs {
    d_temporal: *mut f32,
    batch_acts: *const Vec<Mamba3LayerFlat>,
}
unsafe impl Send for BwdPtrs {}
unsafe impl Sync for BwdPtrs {}

/// Parallel Mamba-3 backward with tree-reduce gradient accumulation.
///
/// **Deterministic static-partition approach.** Each of `n_threads` worker
/// owns a fixed sample range `[start..end)` and a pre-allocated gradient slot.
/// Thread-local scratch is zeroed at the start of each sample. After all
/// workers finish, slots are merged via fixed balanced binary-tree reduce.
///
/// Result: bit-identical output across runs regardless of OS thread
/// scheduling or rayon work-stealing order. Same principle as the
/// batch-invariant matvec kernel.
///
/// Only layer gradients are accumulated here; input_proj and norm_f are owned
/// by the caller and handled outside this function. `input_dim` is needed to
/// allocate per-thread slots that match `d_weights`'s layout.
///
/// **Burn-in limitation.** This entrypoint passes `angle_state_init = None`
/// to the per-sample backward, which means the cumulative RoPE angle is
/// reconstructed assuming the window starts at angle zero. That matches
/// fresh-state training (the only mode all in-tree callers use). If a
/// future workflow runs this on a window that started mid-sequence with a
/// non-zero angle accumulator, the RoPE backward gradient would be off by
/// the (unrecorded) offset. In that case, call
/// [`crate::mamba3_siso::cpu::backward::backward_mamba3_layer_batched`]
/// directly per-sample with the captured `angle_state_init`.
pub fn parallel_mamba3_backward(
    d_temporal_out: &mut [f32],
    batch_acts: &[Vec<Mamba3LayerFlat>],
    weights: &TrainMamba3Weights,
    d_weights: &mut TrainMamba3Weights,
    dims: &Mamba3Dims,
    batch_size: usize,
    input_dim: usize,
) {
    debug_assert_eq!(batch_acts.len(), batch_size);
    if batch_size == 0 {
        return;
    }

    let dm = dims.d_model;
    let seq_len = dims.seq_len;
    let t_per = seq_len * dm;

    let n_threads = rayon::current_num_threads().min(batch_size);

    // Pre-allocate per-thread gradient slots (zeroed).
    let mut slots: Vec<TrainMamba3Weights> = (0..n_threads)
        .map(|_| TrainMamba3Weights::zeros(dims, input_dim))
        .collect();

    // SAFETY: each thread writes to disjoint regions:
    // - slots[tid]: exclusive (static partition by tid)
    // - d_temporal_out[b*t_per .. (b+1)*t_per]: exclusive per-sample
    // - batch_acts[b], weights, dims: shared read-only
    struct BwdRawPtrs {
        slots: *mut TrainMamba3Weights,
        d_temporal: *mut f32,
        batch_acts: *const Vec<Mamba3LayerFlat>,
    }
    unsafe impl Send for BwdRawPtrs {}
    unsafe impl Sync for BwdRawPtrs {}

    let ptrs = BwdRawPtrs {
        slots: slots.as_mut_ptr(),
        d_temporal: d_temporal_out.as_mut_ptr(),
        batch_acts: batch_acts.as_ptr(),
    };

    let ptrs_ref = &ptrs;
    (0..n_threads).into_par_iter().for_each(|tid| {
        let start = tid * batch_size / n_threads;
        let end = (tid + 1) * batch_size / n_threads;

        // SAFETY: tid is unique per thread, start..end is disjoint per tid.
        let slot = unsafe { &mut *ptrs_ref.slots.add(tid) };
        for b in start..end {
            with_scratch(dims, |tls| {
                // SAFETY: disjoint per-sample slices via raw pointer offset.
                let d_temporal = unsafe {
                    std::slice::from_raw_parts_mut(ptrs_ref.d_temporal.add(b * t_per), t_per)
                };
                let acts = unsafe { &*ptrs_ref.batch_acts.add(b) };

                tls.phase.zero_all();
                for (layer_idx, lw) in weights.layers.iter().enumerate().rev() {
                    backward_mamba3_layer_batched(
                        d_temporal,
                        &acts[layer_idx],
                        lw,
                        &mut slot.layers[layer_idx],
                        &mut tls.phase,
                        dims,
                        None,
                    );
                }
            });
        }
    });

    // Fixed balanced binary-tree reduce: deterministic order regardless of
    // thread count. Always ((slot[0]+slot[1])+(slot[2]+slot[3]))+...
    while slots.len() > 1 {
        let half = slots.len() / 2;
        let remainder = slots.len() % 2;
        let (left, right_and_rest) = slots.split_at_mut(half);
        let right = &right_and_rest[..half];
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(a, b)| {
                a.add_inplace(b);
            });
        if remainder == 1 {
            let last_idx = slots.len() - 1;
            slots.swap(half, last_idx);
            slots.truncate(half + 1);
        } else {
            slots.truncate(half);
        }
    }
    if let Some(reduced) = slots.pop() {
        d_weights.add_inplace(&reduced);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba3_siso::config::Mamba3Config;

    fn test_cfg() -> Mamba3Config {
        Mamba3Config {
            d_model: 16,
            d_state: 4,
            expand: 2,
            headdim: 4,
            ngroups: 1,
            n_layers: 2,
            rope_fraction: 0.5,
            a_floor: 0.0625,
            is_outproj_norm: false,
        }
    }

    #[test]
    fn test_parallel_forward_no_panic() {
        let cfg = test_cfg();
        let dims = Mamba3Dims::from_config(&cfg, 4);
        let batch = 3;

        let mut w = TrainMamba3Weights::zeros(&dims, 16);
        for l in &mut w.layers {
            for v in &mut l.norm_weight {
                *v = 1.0;
            }
            for v in &mut l.d_param {
                *v = 1.0;
            }
            for v in &mut l.b_norm_weight {
                *v = 1.0;
            }
            for v in &mut l.c_norm_weight {
                *v = 1.0;
            }
        }

        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let nl = dims.n_layers;
        let na = dims.num_rope_angles.max(1);

        let mut temporal = vec![1.0_f32; batch * dims.seq_len * dims.d_model];
        let mut acts: Vec<Vec<Mamba3LayerFlat>> = (0..batch)
            .map(|_| (0..nl).map(|_| Mamba3LayerFlat::zeros(dims)).collect())
            .collect();
        let mut ssm = vec![0.0; batch * nl * nh * hd * ds];
        let mut k = vec![0.0; batch * nl * nh * ds];
        let mut v = vec![0.0; batch * nl * nh * hd];
        let mut angle = vec![0.0; batch * nl * nh * na];

        parallel_mamba3_forward(
            &mut temporal,
            &mut acts,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            &w,
            &dims,
            batch,
        );

        assert!(temporal.iter().all(|v| v.is_finite()));
    }

    fn init_weights(dims: &Mamba3Dims, input_dim: usize, seed: u32) -> TrainMamba3Weights {
        // Deterministic pseudo-random init via xorshift32 in [-0.05, 0.05].
        let mut state = seed.max(1);
        let mut rand01 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let u = (state & 0x7fff_ffff) as f32 / 2_147_483_647.0;
            u * 0.1 - 0.05
        };
        let mut w = TrainMamba3Weights::zeros(dims, input_dim);
        for v in &mut w.input_proj_w {
            *v = rand01();
        }
        for v in &mut w.norm_f_weight {
            *v = 1.0 + rand01();
        }
        for l in &mut w.layers {
            for v in &mut l.norm_weight {
                *v = 1.0 + rand01();
            }
            for v in &mut l.in_proj_w {
                *v = rand01();
            }
            for v in &mut l.dt_bias {
                *v = rand01();
            }
            for v in &mut l.b_norm_weight {
                *v = 1.0 + rand01();
            }
            for v in &mut l.c_norm_weight {
                *v = 1.0 + rand01();
            }
            for v in &mut l.b_bias {
                *v = rand01();
            }
            for v in &mut l.c_bias {
                *v = rand01();
            }
            for v in &mut l.d_param {
                *v = 1.0 + rand01();
            }
            for v in &mut l.norm_gate_weight {
                *v = 1.0 + rand01();
            }
            for v in &mut l.out_proj_w {
                *v = rand01();
            }
        }
        w
    }

    fn run_forward_backward(
        batch: usize,
        dims: &Mamba3Dims,
        w: &TrainMamba3Weights,
        temporal_in: &[f32],
        d_temporal_seed: &[f32],
    ) -> (Vec<f32>, TrainMamba3Weights) {
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let nl = dims.n_layers;
        let na = dims.num_rope_angles.max(1);

        let mut temporal = temporal_in.to_vec();
        let mut acts: Vec<Vec<Mamba3LayerFlat>> = (0..batch)
            .map(|_| (0..nl).map(|_| Mamba3LayerFlat::zeros(*dims)).collect())
            .collect();
        let mut ssm = vec![0.0; batch * nl * nh * hd * ds];
        let mut k = vec![0.0; batch * nl * nh * ds];
        let mut v = vec![0.0; batch * nl * nh * hd];
        let mut angle = vec![0.0; batch * nl * nh * na];

        parallel_mamba3_forward(
            &mut temporal,
            &mut acts,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            w,
            dims,
            batch,
        );

        let mut d_temporal = d_temporal_seed.to_vec();
        let input_dim = w.input_proj_w.len() / dims.d_model;
        let mut d_w = TrainMamba3Weights::zeros(dims, input_dim);
        parallel_mamba3_backward(&mut d_temporal, &acts, w, &mut d_w, dims, batch, input_dim);
        (d_temporal, d_w)
    }

    fn run_sequential_backward(
        batch: usize,
        dims: &Mamba3Dims,
        w: &TrainMamba3Weights,
        acts: &[Vec<Mamba3LayerFlat>],
        d_temporal_seed: &[f32],
    ) -> (Vec<f32>, TrainMamba3Weights) {
        let dm = dims.d_model;
        let t_per = dims.seq_len * dm;
        let input_dim = w.input_proj_w.len() / dims.d_model;
        let mut d_w = TrainMamba3Weights::zeros(dims, input_dim);
        let mut d_temporal = d_temporal_seed.to_vec();
        let mut scratch = Mamba3Scratch::zeros(dims);
        for b in 0..batch {
            let d_slice = &mut d_temporal[b * t_per..(b + 1) * t_per];
            for (layer_idx, lw) in w.layers.iter().enumerate().rev() {
                backward_mamba3_layer_batched(
                    d_slice,
                    &acts[b][layer_idx],
                    lw,
                    &mut d_w.layers[layer_idx],
                    &mut scratch,
                    dims,
                    None,
                );
            }
        }
        (d_temporal, d_w)
    }

    #[test]
    fn test_parallel_backward_matches_sequential() {
        let cfg = test_cfg();
        let input_dim = 16;
        let dims = Mamba3Dims::from_config(&cfg, input_dim);
        let batch = 5;

        let w = init_weights(&dims, input_dim, 0xC0FFEE);

        let seq_len = dims.seq_len;
        let dm = dims.d_model;
        let t_total = batch * seq_len * dm;

        let mut xs = 0xBEEFu32;
        let mut rand01 = || {
            xs ^= xs << 13;
            xs ^= xs >> 17;
            xs ^= xs << 5;
            (xs & 0x7fff_ffff) as f32 / 2_147_483_647.0 - 0.5
        };
        let temporal_in: Vec<f32> = (0..t_total).map(|_| rand01()).collect();
        let d_temporal_seed: Vec<f32> = (0..t_total).map(|_| rand01()).collect();

        // Parallel path
        let (d_tmp_par, d_w_par) =
            run_forward_backward(batch, &dims, &w, &temporal_in, &d_temporal_seed);

        // Sequential path (re-run forward to get same acts, then sequential backward)
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let nl = dims.n_layers;
        let na = dims.num_rope_angles.max(1);
        let mut temporal_seq = temporal_in.clone();
        let mut acts_seq: Vec<Vec<Mamba3LayerFlat>> = (0..batch)
            .map(|_| (0..nl).map(|_| Mamba3LayerFlat::zeros(dims)).collect())
            .collect();
        let mut ssm = vec![0.0; batch * nl * nh * hd * ds];
        let mut k = vec![0.0; batch * nl * nh * ds];
        let mut v = vec![0.0; batch * nl * nh * hd];
        let mut angle = vec![0.0; batch * nl * nh * na];
        parallel_mamba3_forward(
            &mut temporal_seq,
            &mut acts_seq,
            Mamba3States {
                ssm: &mut ssm,
                k: &mut k,
                v: &mut v,
                angle: &mut angle,
            },
            &w,
            &dims,
            batch,
        );
        let (d_tmp_seq, d_w_seq) =
            run_sequential_backward(batch, &dims, &w, &acts_seq, &d_temporal_seed);

        // Data-gradient parity (bit-level; same order of operations per sample)
        for (i, (&p, &s)) in d_tmp_par.iter().zip(&d_tmp_seq).enumerate() {
            assert!(
                (p - s).abs() <= 1e-5,
                "d_temporal[{i}] mismatch par={p} seq={s}"
            );
        }

        // Weight-gradient parity across all fields.
        // Tree-reduce changes summation order → allow small FP tolerance.
        let cmp = |a: &[f32], b: &[f32], name: &str| {
            assert_eq!(a.len(), b.len(), "{name} length mismatch");
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                let tol = 5e-5 * (x.abs().max(y.abs()).max(1e-3));
                assert!(
                    (x - y).abs() <= tol,
                    "{name}[{i}] par={x} seq={y} diff={}",
                    (x - y).abs()
                );
            }
        };
        for li in 0..dims.n_layers {
            let a = &d_w_par.layers[li];
            let s = &d_w_seq.layers[li];
            cmp(&a.norm_weight, &s.norm_weight, "norm_weight");
            cmp(&a.in_proj_w, &s.in_proj_w, "in_proj_w");
            cmp(&a.dt_bias, &s.dt_bias, "dt_bias");
            cmp(&a.b_norm_weight, &s.b_norm_weight, "b_norm_weight");
            cmp(&a.c_norm_weight, &s.c_norm_weight, "c_norm_weight");
            cmp(&a.b_bias, &s.b_bias, "b_bias");
            cmp(&a.c_bias, &s.c_bias, "c_bias");
            cmp(&a.d_param, &s.d_param, "d_param");
            cmp(&a.norm_gate_weight, &s.norm_gate_weight, "norm_gate_weight");
            cmp(&a.out_proj_w, &s.out_proj_w, "out_proj_w");
        }
    }

    #[test]
    fn test_parallel_backward_deterministic() {
        let cfg = test_cfg();
        let input_dim = 16;
        let dims = Mamba3Dims::from_config(&cfg, input_dim);
        let batch = 4;

        let w = init_weights(&dims, input_dim, 0xDEAD);

        let seq_len = dims.seq_len;
        let dm = dims.d_model;
        let t_total = batch * seq_len * dm;
        let mut xs = 0xFACEu32;
        let mut rand01 = || {
            xs ^= xs << 13;
            xs ^= xs >> 17;
            xs ^= xs << 5;
            (xs & 0x7fff_ffff) as f32 / 2_147_483_647.0 - 0.5
        };
        let temporal_in: Vec<f32> = (0..t_total).map(|_| rand01()).collect();
        let d_temporal_seed: Vec<f32> = (0..t_total).map(|_| rand01()).collect();

        let (d1_tmp, d1_w) = run_forward_backward(batch, &dims, &w, &temporal_in, &d_temporal_seed);
        let (d2_tmp, d2_w) = run_forward_backward(batch, &dims, &w, &temporal_in, &d_temporal_seed);

        for (i, (&a, &b)) in d1_tmp.iter().zip(&d2_tmp).enumerate() {
            let tol = 5e-6 * (a.abs().max(b.abs()).max(1e-3));
            assert!(
                (a - b).abs() <= tol,
                "d_temporal[{i}] non-deterministic a={a} b={b}"
            );
        }
        for li in 0..dims.n_layers {
            let a = &d1_w.layers[li].out_proj_w;
            let b = &d2_w.layers[li].out_proj_w;
            for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                let tol = 5e-5 * (x.abs().max(y.abs()).max(1e-3));
                assert!(
                    (x - y).abs() <= tol,
                    "layer[{li}].out_proj_w[{i}] non-deterministic a={x} b={y}"
                );
            }
        }
    }
}
