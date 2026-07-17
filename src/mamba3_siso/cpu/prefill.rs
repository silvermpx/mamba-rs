//! Full-sequence Mamba-3 CPU inference forward ("prefill") — the M3 twin
//! of [`crate::mamba_ssm::cpu::prefill`].
//!
//! Runs the SAME math as the training layer forward
//! (`forward_mamba3_layer_batched`) WITHOUT saving the activation tape, on
//! inference types (`Mamba3Weights` + `Mamba3LayerState`), with the
//! recurrent SSM/K/V/angle states carried in AND out so `mamba3_step`
//! continues seamlessly (prefill-then-decode).
//!
//! Layout doctrine (the measured M1 lesson): the head-sequential F5 phase
//! reads ONLY head-major/channel-major buffers — `x_cm [di * T]` plus the
//! head-scalar columns (`dd_a`/`dd_dt`/`trap` as `[nh * T]`) are pulled out
//! of the interleaved `[T, in_proj_dim]` projection with cache-blocked
//! transposes; the shared per-position rows (BCNormed B/C, tanh-pi angle
//! velocities) stream row-sequentially and stay L2-resident across heads.
//! In [`PrefillMode::Parallel`] every phase parallelizes (GEMMs, rows,
//! transpose tiles, and F5 over HEADS — each head owns disjoint state and
//! `y` column slices) with no cross-task reductions, so Parallel is
//! bit-equal to Single by construction; both are bit-equal to the training
//! layer forward (the anchor test pins it).

use rayon::prelude::*;

use crate::mamba_ssm::cpu::prefill::{
    PrefillMode, for_rows, transpose_cm_to_rm, transpose_rows_to_cm, zip_rows,
};
use crate::mamba3_siso::cpu::dims::Mamba3Dims;
use crate::mamba3_siso::cpu::forward::{
    heavy_tail, simd_rms_scale, simd_ssm_recurrence, simd_sum_sq, softplus,
};
use crate::mamba3_siso::state::Mamba3LayerState;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::ops::blas::{sgemm_forward, sgemm_forward_par};
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_scalar};

/// Stack-array limits (match `forward.rs` / config validation).
const MAX_DS: usize = 64;
const MAX_ANGLES: usize = MAX_DS / 2;

type GemmFn = fn(&mut [f32], &[f32], &[f32], Option<&[f32]>, usize, usize, usize);

/// Three-slice row zip (dst, a, b), serial or rayon-parallel per `mode` —
/// the gating phase needs the y row AND the proj row per output row.
fn gate_rows<F>(
    mode: PrefillMode,
    dst: (&mut [f32], usize),
    a: (&[f32], usize),
    b: (&[f32], usize),
    f: F,
) where
    F: Fn(&mut [f32], &[f32], &[f32]) + Sync + Send,
{
    let (dst, dw) = dst;
    let (a, aw) = a;
    let (b, bw) = b;
    match mode {
        PrefillMode::Single => dst
            .chunks_mut(dw)
            .zip(a.chunks(aw))
            .zip(b.chunks(bw))
            .for_each(|((d, x), y)| f(d, x, y)),
        PrefillMode::Parallel => dst
            .par_chunks_mut(dw)
            .zip(a.par_chunks(aw))
            .zip(b.par_chunks(bw))
            .for_each(|((d, x), y)| f(d, x, y)),
    }
}

/// Reusable M3 prefill scratch. `ensure` re-allocates only when the dims
/// fingerprint changes — zero allocation on the serving hot path once warmed.
pub struct Mamba3PrefillScratch {
    fingerprint: (Mamba3Dims, usize),
    post_norm: Vec<f32>,
    proj: Vec<f32>,
    /// BCNormed B/C rows `[T * ngroups * d_state]` (shared reads in F5).
    b_normed: Vec<f32>,
    c_normed: Vec<f32>,
    /// Per-position RoPE angle velocities `tanh(raw) * pi`, `[T * n_angles]`.
    tanh_pi: Vec<f32>,
    /// Channel-major x branch `[di * T]`.
    x_cm: Vec<f32>,
    /// Head-major scalar columns `[nh * T]`.
    dd_a_cm: Vec<f32>,
    dd_dt_cm: Vec<f32>,
    trap_cm: Vec<f32>,
    /// SSM output, channel-major `[di * T]` / row-major `[T * di]`.
    y_cm: Vec<f32>,
    y_rm: Vec<f32>,
    gated: Vec<f32>,
    out: Vec<f32>,
}

impl Mamba3PrefillScratch {
    /// Allocate scratch sized for `dims` and `input_dim`.
    pub fn new(dims: &Mamba3Dims, input_dim: usize) -> Self {
        let t = dims.seq_len;
        let di = dims.d_inner;
        let na = dims.num_rope_angles;
        Self {
            fingerprint: (*dims, input_dim),
            post_norm: vec![0.0; t * dims.d_model],
            proj: vec![0.0; t * dims.in_proj_dim],
            b_normed: vec![0.0; t * dims.ngroups * dims.d_state],
            c_normed: vec![0.0; t * dims.ngroups * dims.d_state],
            tanh_pi: vec![0.0; t * na.max(1)],
            x_cm: vec![0.0; di * t],
            dd_a_cm: vec![0.0; dims.nheads * t],
            dd_dt_cm: vec![0.0; dims.nheads * t],
            trap_cm: vec![0.0; dims.nheads * t],
            y_cm: vec![0.0; di * t],
            y_rm: vec![0.0; t * di],
            gated: vec![0.0; t * di],
            out: vec![0.0; t * dims.d_model],
        }
    }

    /// Re-size for a different fingerprint (no-op when unchanged).
    pub fn ensure(&mut self, dims: &Mamba3Dims, input_dim: usize) {
        if (*dims, input_dim) != self.fingerprint {
            *self = Self::new(dims, input_dim);
        }
    }
}

/// F5 for ONE head over the full sequence: input-dependent A/DT, per-head
/// B/C bias + RoPE, trapezoidal SSM recurrence, D skip. All hot accesses
/// are head-owned contiguous columns or L2-streamed shared rows. The
/// per-element expression chains are IDENTICAL to
/// `forward_mamba3_layer_batched` F5, so the tape-free form is bit-exact.
struct HeadIo<'a> {
    y_cols: &'a mut [f32],
    ssm: &'a mut [f32],
    k_state: &'a mut [f32],
    v_state: &'a mut [f32],
    angle_state: &'a mut [f32],
}

/// Shared read-only inputs of the F5 head phase (head-major scalar columns
/// are pre-sliced to THIS head's `[T]` column).
struct HeadShared<'a> {
    x_cm: &'a [f32],
    dd_a_col: &'a [f32],
    dd_dt_col: &'a [f32],
    trap_col: &'a [f32],
    b_normed: &'a [f32],
    c_normed: &'a [f32],
    tanh_pi: &'a [f32],
}

fn ssm_head(
    h: usize,
    io: HeadIo<'_>,
    shared: HeadShared<'_>,
    lw: &Mamba3LayerWeights,
    dims: &Mamba3Dims,
) {
    let HeadIo {
        y_cols,
        ssm,
        k_state,
        v_state,
        angle_state,
    } = io;
    let HeadShared {
        x_cm,
        dd_a_col,
        dd_dt_col,
        trap_col,
        b_normed,
        c_normed,
        tanh_pi,
    } = shared;
    let ds = dims.d_state;
    let hd = dims.headdim;
    let nh = dims.nheads;
    let ng = dims.ngroups;
    let na = dims.num_rope_angles;
    let t_len = dims.seq_len;
    let a_floor = dims.a_floor;
    let g = h / (nh / ng);
    let gs = g * ds;

    for t in 0..t_len {
        let a_val = (-heavy_tail(dd_a_col[t])).min(-a_floor);
        let dt_val = softplus(dd_dt_col[t] + lw.dt_bias[h]);

        let mut k_local = [0.0_f32; MAX_DS];
        let mut q_local = [0.0_f32; MAX_DS];
        let b_row = t * ng * ds;
        for n in 0..ds {
            k_local[n] = b_normed[b_row + gs + n] + lw.b_bias[h * ds + n];
            q_local[n] = c_normed[b_row + gs + n] + lw.c_bias[h * ds + n];
        }

        if na > 0 {
            let tp_row = t * na;
            for (an, &tp) in angle_state[..na]
                .iter_mut()
                .zip(&tanh_pi[tp_row..tp_row + na])
            {
                let delta = tp * dt_val;
                let mut acc = *an as f64 + delta as f64;
                let two_pi_64 = 2.0 * std::f64::consts::PI;
                acc -= two_pi_64 * (acc / two_pi_64).floor();
                *an = acc as f32;
            }
            for (a, an) in angle_state[..na].iter().enumerate() {
                let (sin_a, cos_a) = an.sin_cos();
                let i0 = 2 * a;
                let i1 = 2 * a + 1;
                let b0 = k_local[i0];
                let b1 = k_local[i1];
                k_local[i0] = cos_a * b0 - sin_a * b1;
                k_local[i1] = sin_a * b0 + cos_a * b1;
                let c0 = q_local[i0];
                let c1 = q_local[i1];
                q_local[i0] = cos_a * c0 - sin_a * c1;
                q_local[i1] = sin_a * c0 + cos_a * c1;
            }
        }

        let alpha = fast_exp_scalar(a_val * dt_val);
        let trap_sig = 1.0 / (1.0 + fast_exp_scalar(-trap_col[t]));
        let beta = alpha * dt_val * (1.0 - trap_sig);
        let gamma = trap_sig * dt_val;

        for p in 0..hd {
            let x_val = x_cm[(h * hd + p) * t_len + t];
            let v_prev = v_state[p];
            let s_off = p * ds;
            let y_val = simd_ssm_recurrence(
                &mut ssm[s_off..s_off + ds],
                &k_state[..ds],
                &k_local[..ds],
                &q_local[..ds],
                alpha,
                beta * v_prev,
                gamma * x_val,
            );
            y_cols[p * t_len + t] = lw.d_param[h].mul_add(x_val, y_val);
        }

        k_state[..ds].copy_from_slice(&k_local[..ds]);
        for p in 0..hd {
            v_state[p] = x_cm[(h * hd + p) * t_len + t];
        }
    }
}

/// Full-sequence Mamba-3 CPU inference forward (serial mode).
///
/// - `temporal_out`: `[T * d_model]` — POST-norm_f output at EVERY position.
/// - `input_flat`: `[T * input_dim]`.
/// - `states`: one [`Mamba3LayerState`] per layer, carried in AND out
///   (prefill-then-decode into [`super::inference::mamba3_step`]).
pub fn forward_mamba3_backbone_prefill(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba3Weights,
    states: &mut [Mamba3LayerState],
    scratch: &mut Mamba3PrefillScratch,
    dims: &Mamba3Dims,
) {
    forward_mamba3_backbone_prefill_mode(
        temporal_out,
        input_flat,
        w,
        states,
        scratch,
        dims,
        PrefillMode::Single,
    );
}

/// [`forward_mamba3_backbone_prefill`] with an explicit execution mode.
pub fn forward_mamba3_backbone_prefill_mode(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba3Weights,
    states: &mut [Mamba3LayerState],
    scratch: &mut Mamba3PrefillScratch,
    dims: &Mamba3Dims,
    mode: PrefillMode,
) {
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.num_rope_angles;
    let t_len = dims.seq_len;
    let is_outproj_norm = dims.is_outproj_norm;
    let t_dm = t_len * dm;

    assert!(ds <= MAX_DS, "d_state > {MAX_DS} unsupported");
    assert!(
        na <= MAX_ANGLES,
        "num_rope_angles > {MAX_ANGLES} unsupported"
    );
    assert_eq!(temporal_out.len(), t_dm, "temporal_out shape");
    assert_eq!(states.len(), dims.n_layers, "state layer count");
    let input_dim = if w.input_proj_w.is_empty() {
        dm
    } else {
        w.input_proj_w.len() / dm
    };
    assert_eq!(input_flat.len(), t_len * input_dim, "input shape");
    scratch.ensure(dims, input_dim);

    let gemm: GemmFn = match mode {
        PrefillMode::Single => sgemm_forward,
        PrefillMode::Parallel => sgemm_forward_par,
    };

    // Input projection (identity D2D when input_proj is absent — the
    // mixed-trained checkpoint convention).
    if w.input_proj_w.is_empty() {
        temporal_out[..t_dm].copy_from_slice(&input_flat[..t_dm]);
    } else {
        gemm(
            temporal_out,
            input_flat,
            &w.input_proj_w,
            Some(&w.input_proj_b),
            t_len,
            input_dim,
            dm,
        );
    }

    for (layer_idx, lw) in w.layers.iter().enumerate() {
        let lstate = &mut states[layer_idx];

        // F1: RMSNorm rows (temporal_out doubles as the residual — read
        // here, updated in place at F7).
        zip_rows(
            mode,
            (&mut scratch.post_norm[..t_dm], dm),
            (&temporal_out[..t_dm], dm),
            |dst, src| {
                let sum_sq = simd_sum_sq(src);
                let inv_rms = 1.0 / (sum_sq / dm as f32 + RMS_NORM_EPS).sqrt();
                simd_rms_scale(dst, src, &lw.norm_weight[..dm], inv_rms);
            },
        );

        // F2: in_proj GEMM.
        gemm(
            &mut scratch.proj,
            &scratch.post_norm,
            &lw.in_proj_w,
            None,
            t_len,
            dm,
            ip,
        );

        // F3/F4 prep rows: BCNorm B and C + tanh-pi angle velocities. The
        // 8-way split stays implicit — later phases read `proj` columns
        // directly (row-sequential) or via the blocked transposes below.
        let bsize = ng * ds;
        let (b_off, c_off) = (2 * di, 2 * di + bsize);
        let angles_off = 2 * di + 2 * bsize + 3 * nh;
        zip_rows(
            mode,
            (&mut scratch.b_normed[..t_len * bsize], bsize),
            (&scratch.proj[..t_len * ip], ip),
            |dst, proj_row| {
                for g in 0..ng {
                    let raw = &proj_row[b_off + g * ds..b_off + g * ds + ds];
                    let sum_sq = simd_sum_sq(raw);
                    let inv_rms = 1.0 / (sum_sq / ds as f32 + RMS_NORM_EPS).sqrt();
                    for (i, (d, &r)) in dst[g * ds..g * ds + ds].iter_mut().zip(raw).enumerate() {
                        *d = r * inv_rms * lw.b_norm_weight[i];
                    }
                }
            },
        );
        zip_rows(
            mode,
            (&mut scratch.c_normed[..t_len * bsize], bsize),
            (&scratch.proj[..t_len * ip], ip),
            |dst, proj_row| {
                for g in 0..ng {
                    let raw = &proj_row[c_off + g * ds..c_off + g * ds + ds];
                    let sum_sq = simd_sum_sq(raw);
                    let inv_rms = 1.0 / (sum_sq / ds as f32 + RMS_NORM_EPS).sqrt();
                    for (i, (d, &r)) in dst[g * ds..g * ds + ds].iter_mut().zip(raw).enumerate() {
                        *d = r * inv_rms * lw.c_norm_weight[i];
                    }
                }
            },
        );
        if na > 0 {
            let pi = std::f32::consts::PI;
            zip_rows(
                mode,
                (&mut scratch.tanh_pi[..t_len * na], na),
                (&scratch.proj[..t_len * ip], ip),
                |dst, proj_row| {
                    for (d, &raw) in dst.iter_mut().zip(&proj_row[angles_off..angles_off + na]) {
                        *d = raw.tanh() * pi;
                    }
                },
            );
        }

        // Blocked transposes: x branch to channel-major, head scalars to
        // head-major (the F5 head loops must never stride through `proj`).
        transpose_rows_to_cm(&mut scratch.x_cm, &scratch.proj, di, t_len, (ip, di), mode);
        let sc_off = 2 * di + 2 * bsize;
        transpose_rows_to_cm(
            &mut scratch.dd_dt_cm,
            &scratch.proj,
            nh,
            t_len,
            (ip, sc_off),
            mode,
        );
        transpose_rows_to_cm(
            &mut scratch.dd_a_cm,
            &scratch.proj,
            nh,
            t_len,
            (ip, sc_off + nh),
            mode,
        );
        transpose_rows_to_cm(
            &mut scratch.trap_cm,
            &scratch.proj,
            nh,
            t_len,
            (ip, sc_off + 2 * nh),
            mode,
        );

        // F5: trapezoidal SSM, one task per head (disjoint state + y cols).
        {
            let x_cm = &scratch.x_cm[..];
            let dd_a_cm = &scratch.dd_a_cm[..];
            let dd_dt_cm = &scratch.dd_dt_cm[..];
            let trap_cm = &scratch.trap_cm[..];
            let b_normed = &scratch.b_normed[..];
            let c_normed = &scratch.c_normed[..];
            let tanh_pi = &scratch.tanh_pi[..];
            let na_st = na.max(1);
            let run_head = |h: usize, io: HeadIo<'_>| {
                ssm_head(
                    h,
                    io,
                    HeadShared {
                        x_cm,
                        dd_a_col: &dd_a_cm[h * t_len..(h + 1) * t_len],
                        dd_dt_col: &dd_dt_cm[h * t_len..(h + 1) * t_len],
                        trap_col: &trap_cm[h * t_len..(h + 1) * t_len],
                        b_normed,
                        c_normed,
                        tanh_pi,
                    },
                    lw,
                    dims,
                );
            };
            match mode {
                PrefillMode::Single => {
                    for (h, ((((y_cols, ssm), k_st), v_st), an_st)) in scratch
                        .y_cm
                        .chunks_mut(hd * t_len)
                        .zip(lstate.ssm_state.chunks_mut(hd * ds))
                        .zip(lstate.k_state.chunks_mut(ds))
                        .zip(lstate.v_state.chunks_mut(hd))
                        .zip(lstate.angle_state.chunks_mut(na_st))
                        .enumerate()
                    {
                        run_head(
                            h,
                            HeadIo {
                                y_cols,
                                ssm,
                                k_state: k_st,
                                v_state: v_st,
                                angle_state: an_st,
                            },
                        );
                    }
                }
                PrefillMode::Parallel => {
                    scratch
                        .y_cm
                        .par_chunks_mut(hd * t_len)
                        .zip(lstate.ssm_state.par_chunks_mut(hd * ds))
                        .zip(lstate.k_state.par_chunks_mut(ds))
                        .zip(lstate.v_state.par_chunks_mut(hd))
                        .zip(lstate.angle_state.par_chunks_mut(na_st))
                        .enumerate()
                        .for_each(|(h, ((((y_cols, ssm), k_st), v_st), an_st))| {
                            run_head(
                                h,
                                HeadIo {
                                    y_cols,
                                    ssm,
                                    k_state: k_st,
                                    v_state: v_st,
                                    angle_state: an_st,
                                },
                            );
                        });
                }
            }
        }
        transpose_cm_to_rm(&mut scratch.y_rm, &scratch.y_cm, di, t_len, mode);

        // F6: output gating rows (z read row-sequentially from proj).
        gate_rows(
            mode,
            (&mut scratch.gated[..t_len * di], di),
            (&scratch.y_rm[..t_len * di], di),
            (&scratch.proj[..t_len * ip], ip),
            |gated_row, y_row, proj_row| {
                let z_row = &proj_row[..di];
                if is_outproj_norm {
                    for g_start in (0..di).step_by(hd) {
                        let g_end = (g_start + hd).min(di);
                        let g_len = g_end - g_start;
                        let sum_sq = simd_sum_sq(&y_row[g_start..g_end]);
                        let rstd = 1.0 / (sum_sq / g_len as f32 + RMS_NORM_EPS).sqrt();
                        for d in g_start..g_end {
                            let z = z_row[d];
                            let silu = z / (1.0 + fast_exp_scalar(-z));
                            gated_row[d] = y_row[d] * rstd * lw.norm_gate_weight[d] * silu;
                        }
                    }
                } else {
                    for ((gv, &y), &z) in gated_row.iter_mut().zip(y_row).zip(z_row) {
                        let silu = z / (1.0 + fast_exp_scalar(-z));
                        *gv = y * silu;
                    }
                }
            },
        );

        // F7: out_proj GEMM + in-place residual add.
        gemm(
            &mut scratch.out,
            &scratch.gated,
            &lw.out_proj_w,
            None,
            t_len,
            di,
            dm,
        );
        zip_rows(
            mode,
            (&mut temporal_out[..t_dm], dm),
            (&scratch.out[..t_dm], dm),
            |y_row, o_row| {
                for (y, &o) in y_row.iter_mut().zip(o_row) {
                    *y += o;
                }
            },
        );
    }

    // norm_f at EVERY position, via the public in-place RMSNorm (the anchor
    // test reproduces the exact same call per row).
    for_rows(mode, &mut temporal_out[..t_dm], dm, |row| {
        crate::ops::norms::rms_norm_inplace(row, &w.norm_f_weight[..dm], RMS_NORM_EPS);
    });
}

/// Prefill a batch of independent sequences in parallel (one rayon task per
/// sample, each running [`PrefillMode::Single`] internally — the M3 twin of
/// [`crate::mamba_ssm::cpu::prefill::prefill_batch`]).
pub fn prefill3_batch(
    outputs: &mut [f32],
    inputs: &[f32],
    w: &Mamba3Weights,
    states: &mut [crate::mamba3_siso::state::Mamba3State],
    scratches: &mut [Mamba3PrefillScratch],
    dims: &Mamba3Dims,
) {
    let t_len = dims.seq_len;
    let dm = dims.d_model;
    let b = states.len();
    let input_dim = if w.input_proj_w.is_empty() {
        dm
    } else {
        w.input_proj_w.len() / dm
    };
    assert_eq!(scratches.len(), b, "one scratch per sample");
    assert_eq!(outputs.len(), b * t_len * dm, "outputs shape");
    assert_eq!(inputs.len(), b * t_len * input_dim, "inputs shape");

    outputs
        .par_chunks_mut(t_len * dm)
        .zip(states.par_iter_mut())
        .zip(scratches.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((out, state), scratch))| {
            let inp = &inputs[i * t_len * input_dim..(i + 1) * t_len * input_dim];
            forward_mamba3_backbone_prefill(out, inp, w, &mut state.layers, scratch, dims);
        });
}
