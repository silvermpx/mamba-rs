//! Mamba-3 MIMO training on the CPU: a forward that records a tape and the
//! backward through it.
//!
//! The forward runs the prefill's per-head step token by token (tokens
//! outer, heads inner: each head's chain is the prefill's), so its outputs
//! and states are the prefill's bits; the tape keeps each layer's input
//! rows, in_proj rows and gated rows, and the whole layer state every
//! `stride` tokens. The backward replays each segment of each head from
//! its checkpoint through the same step functions, keeping that segment's
//! intermediates, and walks every stage back in a fixed order, the gated
//! output norm (`is_outproj_norm`) included.
//!
//! In a reduced precision the window rounds where the GPU trainer stores a
//! half value: the normalized rows, the in_proj and out_proj outputs and
//! the gated rows in the forward; the residual gradient's copy for the
//! GEMMs, the gated rows' and the normalized rows' input gradients and the
//! in_proj gradient (once, after every term is in) in the backward. TF32
//! rounds every GEMM operand instead. The heads' math, the states, the
//! norms' statistics and every weight gradient stay f32.

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use super::core::{
    HeadScratch, HeadState, MimoRowsIo, TokenIn, advance_angles, block_norm_row, discretize,
    mimo_head_step, mimo_rows, prepare_token, rope_half_split,
};
use crate::mamba3_siso::cpu::forward::{clamp_a, heavy_tail, heavy_tail_deriv, simd_sum_sq};
use crate::mlp::cpu::RowsGemm;
use crate::mlp::cpu::{
    GatedMlpBackwardScratch, GatedMlpRowsScratch, gated_mlp_rows, gated_mlp_rows_backward,
    mlp_norm_row,
};
use crate::ops::blas::sgemm_forward;
use crate::ops::fast_math::fast_exp_scalar;
use crate::ops::lane::LaneGemm;
use crate::ops::norms::{layer_norm_row_backward_in_place, layer_norm_stats};
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

/// One layer's record.
struct LayerTape {
    x_in: Vec<f32>,
    proj: Vec<f32>,
    /// The gated rows `[T, d_inner]`, the out_proj input.
    o: Vec<f32>,
    /// The layer state before tokens 0, stride, 2 * stride, ...
    ckpt: Vec<Mamba3MimoLayerState>,
    mlp_in: Vec<f32>,
}

/// The tape of one training window.
pub struct Mamba3MimoTape {
    seq_len: usize,
    stride: usize,
    input_dim: usize,
    /// The precision the window computes in.
    dtype: WeightDtype,
    /// Set by a forward that completed, cleared when one starts: the
    /// backward reads only a window the forward wrote whole.
    filled: bool,
    input: Vec<f32>,
    final_in: Vec<f32>,
    layers: Vec<LayerTape>,
}

impl Mamba3MimoTape {
    pub fn new(cfg: &Mamba3MimoConfig, seq_len: usize, input_dim: usize, stride: usize) -> Self {
        let dm = cfg.d_model;
        Self {
            seq_len,
            stride: stride.max(1),
            input_dim,
            dtype: WeightDtype::F32,
            filled: false,
            input: vec![0.0; seq_len * input_dim],
            final_in: vec![0.0; seq_len * dm],
            layers: (0..cfg.n_layers)
                .map(|_| LayerTape {
                    x_in: vec![0.0; seq_len * dm],
                    proj: vec![0.0; seq_len * cfg.in_proj_out_dim()],
                    o: vec![0.0; seq_len * cfg.d_inner()],
                    ckpt: Vec::new(),
                    mlp_in: if cfg.mlp_hidden() > 0 {
                        vec![0.0; seq_len * dm]
                    } else {
                        Vec::new()
                    },
                })
                .collect(),
        }
    }

    /// The tape of a window computed in `dtype`: the weights it is given
    /// must hold the bulk matrices rounded as that precision keeps them.
    pub(crate) fn with_dtype(mut self, dtype: WeightDtype) -> Self {
        self.dtype = dtype;
        self
    }
}

/// Reusable buffers for the training forward and backward.
#[derive(Default)]
pub struct Mamba3MimoTrainScratch {
    norm: Vec<f32>,
    bn: Vec<f32>,
    cn: Vec<f32>,
    tanh_pi: Vec<f32>,
    out: Vec<f32>,
    d_o: Vec<f32>,
    d_proj: Vec<f32>,
    d_bn: Vec<f32>,
    d_cn: Vec<f32>,
    d_tanh: Vec<f32>,
    d_post: Vec<f32>,
    heads: Vec<HeadScratch>,
    /// The (heads, rank, d_state, headdim, angles) `heads` was sized for.
    head_shape: (usize, usize, usize, usize, usize),
    mlp: GatedMlpRowsScratch,
    mlp_bwd: GatedMlpBackwardScratch,
    /// The buffers of a head's replay and walk back, reused across heads,
    /// segments and windows.
    head_bwd: HeadBackward,
    /// The precision of the window in flight, from its tape.
    dtype: WeightDtype,
    gemm: LaneGemm,
}

/// The input projection's precision: f32 in the half lanes, as the GPU
/// trainer keeps it, TF32 operands in the TF32 lane.
fn input_proj_dtype(dtype: WeightDtype) -> WeightDtype {
    if dtype == WeightDtype::Tf32 {
        dtype
    } else {
        WeightDtype::F32
    }
}

fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

fn bias(b: &[f32]) -> Option<&[f32]> {
    if b.is_empty() { None } else { Some(b) }
}

/// The token's prepared inputs: B and C normalised, the angle increments.
fn prepare_rows(
    cfg: &Mamba3MimoConfig,
    lw: &Mamba3MimoLayerWeights,
    proj: &[f32],
    s: &mut Mamba3MimoTrainScratch,
    t_len: usize,
) {
    let (ip, bc, na) = (cfg.in_proj_out_dim(), cfg.bc_width(), cfg.num_rope_angles());
    grow(&mut s.bn, t_len * bc);
    grow(&mut s.cn, t_len * bc);
    grow(&mut s.tanh_pi, t_len * na.max(1));
    for (((row, bn), cn), tp) in proj[..t_len * ip]
        .chunks_exact(ip)
        .zip(s.bn.chunks_exact_mut(bc))
        .zip(s.cn.chunks_exact_mut(bc))
        .zip(s.tanh_pi.chunks_exact_mut(na.max(1)))
    {
        prepare_token(cfg, lw, row, (bn, cn, &mut tp[..na]));
    }
}

/// Token `t`'s view of the window's prepared rows.
fn token_at<'a>(
    cfg: &Mamba3MimoConfig,
    proj: &'a [f32],
    (bn, cn, tanh_pi): (&'a [f32], &'a [f32], &'a [f32]),
    t: usize,
) -> TokenIn<'a> {
    let (di, nh, bc, na, ip) = (
        cfg.d_inner(),
        cfg.nheads(),
        cfg.bc_width(),
        cfg.num_rope_angles(),
        cfg.in_proj_out_dim(),
    );
    let row = &proj[t * ip..(t + 1) * ip];
    let heads = 2 * di + 2 * bc;
    TokenIn {
        z: &row[..di],
        x: &row[di..2 * di],
        bn: &bn[t * bc..(t + 1) * bc],
        cn: &cn[t * bc..(t + 1) * bc],
        dd_dt: &row[heads..heads + nh],
        dd_a: &row[heads + nh..heads + 2 * nh],
        trap: &row[heads + 2 * nh..heads + 3 * nh],
        tanh_pi: &tanh_pi[t * na.max(1)..t * na.max(1) + na],
    }
}

fn head_state<'a>(
    st: &'a mut Mamba3MimoLayerState,
    cfg: &Mamba3MimoConfig,
    h: usize,
) -> HeadState<'a> {
    let (n, p_len, r, na) = (
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
    );
    HeadState {
        ssm: &mut st.ssm_state[h * p_len * n..(h + 1) * p_len * n],
        k: &mut st.k_state[h * r * n..(h + 1) * r * n],
        v: &mut st.v_state[h * p_len..(h + 1) * p_len],
        angle: &mut st.angle_state[h * na..(h + 1) * na],
    }
}

/// The training forward over a window: the prefill's outputs and states,
/// and the tape the backward needs.
pub fn forward_mamba3_mimo_train(
    out: &mut [f32],
    input: &[f32],
    w: &Mamba3MimoWeights,
    states: &mut [Mamba3MimoLayerState],
    tape: &mut Mamba3MimoTape,
    (cfg, s): (&Mamba3MimoConfig, &mut Mamba3MimoTrainScratch),
) -> Result<(), String> {
    let n = cfg.n_layers;
    if w.layers.len() != n || states.len() != n || tape.layers.len() != n {
        return Err(format!(
            "a {n}-layer model trains with {n} layer weights, states and tape layers; got {}, \
             {} and {}",
            w.layers.len(),
            states.len(),
            tape.layers.len()
        ));
    }
    tape.filled = false;
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        tape.filled = true;
        return Ok(());
    }
    tape.input.copy_from_slice(&input[..t_len * tape.input_dim]);
    s.dtype = tape.dtype;
    s.mlp.dtype = tape.dtype;
    let rows = &mut out[..t_len * dm];
    if w.input_proj_w.is_empty() {
        rows.copy_from_slice(&input[..t_len * dm]);
    } else {
        s.gemm.forward(
            (sgemm_forward as RowsGemm, input_proj_dtype(tape.dtype)),
            (rows, &tape.input),
            (&w.input_proj_w, bias(&w.input_proj_b)),
            (t_len, tape.input_dim, dm),
        );
    }
    let head_shape = (
        cfg.nheads(),
        cfg.mimo_rank,
        cfg.d_state,
        cfg.headdim,
        cfg.num_rope_angles(),
    );
    if s.head_shape != head_shape || s.heads.len() != cfg.nheads() {
        s.heads = (0..cfg.nheads()).map(|_| HeadScratch::new(cfg)).collect();
        s.head_shape = head_shape;
    }
    let stride = tape.stride;
    for (l, ((lw, st), lt)) in w
        .layers
        .iter()
        .zip(states.iter_mut())
        .zip(tape.layers.iter_mut())
        .enumerate()
    {
        layer_forward(rows, (lw, st, lt), (cfg, t_len, stride), s);
        if let Some(mlp) = w.mlp.get(l) {
            lt.mlp_in.copy_from_slice(rows);
            let eps = cfg.rms_norm_eps;
            gated_mlp_rows(rows, t_len, mlp, &mut s.mlp, sgemm_forward, |dst, src| {
                for (d_row, s_row) in dst.chunks_exact_mut(dm).zip(src.chunks_exact(dm)) {
                    mlp_norm_row(d_row, s_row, mlp, eps);
                }
            });
        }
    }
    tape.final_in.copy_from_slice(rows);
    for (o, x) in rows
        .chunks_exact_mut(dm)
        .zip(tape.final_in.chunks_exact(dm))
    {
        block_norm_row(o, x, (&w.norm_f_weight, &w.norm_f_bias), cfg.rms_norm_eps);
    }
    tape.filled = true;
    Ok(())
}

fn layer_forward(
    rows: &mut [f32],
    (lw, state, lt): (
        &Mamba3MimoLayerWeights,
        &mut Mamba3MimoLayerState,
        &mut LayerTape,
    ),
    (cfg, t_len, stride): (&Mamba3MimoConfig, usize, usize),
    s: &mut Mamba3MimoTrainScratch,
) {
    let (dm, di, ip, p_len) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.headdim,
    );
    grow(&mut s.norm, t_len * dm);
    grow(&mut s.out, t_len * dm);
    let lane = (sgemm_forward as RowsGemm, s.dtype);
    lt.x_in.copy_from_slice(rows);
    for (d, x) in s.norm.chunks_exact_mut(dm).zip(rows.chunks_exact(dm)) {
        block_norm_row(d, x, (&lw.norm_weight, &lw.norm_bias), cfg.rms_norm_eps);
    }
    store(s.dtype, &mut s.norm[..t_len * dm]);
    s.gemm.forward(
        lane,
        (&mut lt.proj, &s.norm[..t_len * dm]),
        (&lw.in_proj_w, None),
        (t_len, dm, ip),
    );
    prepare_rows(cfg, lw, &lt.proj, s, t_len);
    lt.ckpt.clear();
    for t in 0..t_len {
        if t % stride == 0 {
            lt.ckpt.push(state.clone());
        }
        let tok = token_at(cfg, &lt.proj, (&s.bn, &s.cn, &s.tanh_pi), t);
        for (h, hs) in s.heads.iter_mut().enumerate() {
            mimo_head_step(
                (cfg, lw, h),
                &tok,
                head_state(state, cfg, h),
                hs,
                &mut lt.o[t * di + h * p_len..t * di + (h + 1) * p_len],
            );
        }
    }
    lt.ckpt.push(state.clone());
    store(s.dtype, &mut lt.o[..t_len * di]);
    s.gemm.forward(
        lane,
        (&mut s.out[..t_len * dm], &lt.o),
        (&lw.out_proj_w, None),
        (t_len, di, dm),
    );
    for (r, &o) in rows.iter_mut().zip(&s.out[..t_len * dm]) {
        *r += o;
    }
}

/// d loss / d x of rows through a block norm: [`row_norm_backward`], or
/// the LayerNorm backward for a model whose norms carry a bias, its
/// statistics recomputed from the rows as the forward computed them. The
/// parameters' gradients accumulate.
fn block_norm_backward(
    d_rows: &mut [f32],
    x_rows: &[f32],
    (w, b): (&[f32], &[f32]),
    (gw, gb): (&mut [f32], &mut [f32]),
    eps: f32,
) {
    if b.is_empty() {
        return row_norm_backward(d_rows, x_rows, (w, gw), eps);
    }
    let n = w.len();
    for (d, x) in d_rows.chunks_exact_mut(n).zip(x_rows.chunks_exact(n)) {
        let stats = layer_norm_stats(x, eps);
        layer_norm_row_backward_in_place(d, (&mut *gw, &mut *gb), x, w, stats);
    }
}

/// d loss / d x of rows normalised by [`row_norm`]; the scale's gradient
/// accumulates.
fn row_norm_backward(d_rows: &mut [f32], x_rows: &[f32], (w, gw): (&[f32], &mut [f32]), eps: f32) {
    let n = w.len();
    for (d, x) in d_rows.chunks_exact_mut(n).zip(x_rows.chunks_exact(n)) {
        let r = 1.0 / (simd_sum_sq(x) / n as f32 + eps).sqrt();
        let mut c = 0.0f32;
        for ((g, &xv), (&dv, &wv)) in gw.iter_mut().zip(x).zip(d.iter().zip(w)) {
            let xh = xv * r;
            *g += dv * xh;
            c += (dv * wv) * xh;
        }
        let c = c / n as f32;
        for ((dv, &xv), &wv) in d.iter_mut().zip(x).zip(w) {
            *dv = r * (*dv * wv - (xv * r) * c);
        }
    }
}

/// The backward of [`forward_mamba3_mimo_train`]: `d_out` holds d loss /
/// d (the outputs) and is used as scratch; `d_input` (empty to skip)
/// receives d loss / d (the inputs); the weight gradients accumulate into
/// `grads`, shaped like `w`. The gradients stop at the states the window
/// started from (they are inputs of the window, not parameters). A tape no
/// completed forward filled, or one of another model, is refused.
pub fn backward_mamba3_mimo_train(
    d_out: &mut [f32],
    d_input: &mut [f32],
    (w, grads): (&Mamba3MimoWeights, &mut Mamba3MimoWeights),
    tape: &Mamba3MimoTape,
    (cfg, s): (&Mamba3MimoConfig, &mut Mamba3MimoTrainScratch),
) -> Result<(), String> {
    if !tape.filled {
        return Err("the tape holds no completed forward window".into());
    }
    let n = cfg.n_layers;
    if w.layers.len() != n || grads.layers.len() != n || tape.layers.len() != n {
        return Err(format!(
            "a {n}-layer model's backward needs {n} layer weights, gradients and tape layers"
        ));
    }
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        return Ok(());
    }
    s.dtype = tape.dtype;
    s.mlp_bwd.dtype = tape.dtype;
    let d_rows = &mut d_out[..t_len * dm];
    block_norm_backward(
        d_rows,
        &tape.final_in,
        (&w.norm_f_weight, &w.norm_f_bias),
        (&mut grads.norm_f_weight, &mut grads.norm_f_bias),
        cfg.rms_norm_eps,
    );
    for l in (0..cfg.n_layers).rev() {
        let lt = &tape.layers[l];
        if let Some(mlp) = w.mlp.get(l) {
            gated_mlp_rows_backward(
                d_rows,
                &lt.mlp_in,
                t_len,
                (mlp, &mut grads.mlp[l]),
                &mut s.mlp_bwd,
                cfg.rms_norm_eps,
            );
        }
        layer_backward(
            d_rows,
            (&w.layers[l], &mut grads.layers[l], lt),
            (cfg, t_len, tape.stride),
            s,
        );
    }
    if !w.input_proj_w.is_empty() {
        let mut discard;
        let dx: &mut [f32] = if d_input.is_empty() {
            discard = vec![0.0; t_len * tape.input_dim];
            &mut discard
        } else {
            &mut d_input[..t_len * tape.input_dim]
        };
        let db = if grads.input_proj_b.is_empty() {
            None
        } else {
            Some(&mut grads.input_proj_b[..])
        };
        s.gemm.backward(
            input_proj_dtype(tape.dtype),
            (dx, &mut grads.input_proj_w, db),
            (d_rows, &tape.input, &w.input_proj_w),
            (t_len, tape.input_dim, dm),
        );
    } else if !d_input.is_empty() {
        d_input[..t_len * dm].copy_from_slice(d_rows);
    }
    Ok(())
}

/// One token of one head, as the forward ran it, with what the backward
/// needs: the step coefficients, the rotation, the rotated keys and
/// queries, the rank inputs and the readouts.
#[derive(Clone, Default)]
struct TokenRecord {
    dt: f32,
    a: f32,
    alpha: f32,
    lam: f32,
    lam_c: f32,
    beta: f32,
    gamma: f32,
    /// (sin, cos) of the angles after this token's advance.
    sc: Vec<(f32, f32)>,
    /// Pre-rotation keys and queries `[R, N]` each.
    k_pre: Vec<f32>,
    q_pre: Vec<f32>,
    /// Rotated keys and queries `[R, N]` each.
    kc: Vec<f32>,
    q: Vec<f32>,
    /// The state after the token `[P, N]`.
    s_cur: Vec<f32>,
    /// `u = mimo_x x`, `up = mimo_x x_prev`, the readouts with D: `[P, R]`.
    u: Vec<f32>,
    up: Vec<f32>,
    y: Vec<f32>,
    /// The raw previous input `[P]`.
    x_prev: Vec<f32>,
}

/// The buffers of a head's backward: its segment's token records (a pool,
/// the first `t1 - t0` of them in use), the replay's running state and
/// rank inputs, and the gradients the walk back carries between tokens.
#[derive(Default)]
struct HeadBackward {
    rec: Vec<TokenRecord>,
    s_start: Vec<f32>,
    ssm: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    angle: Vec<f32>,
    bv: Vec<f32>,
    gx: Vec<f32>,
    dy: Vec<f32>,
    du: Vec<f32>,
    ds: Vec<f32>,
    dq: Vec<f32>,
    d_up: Vec<f32>,
    dkc: Vec<f32>,
    dkp: Vec<f32>,
    dx_prev: Vec<f32>,
    dtheta: Vec<f32>,
    ds_carry: Vec<f32>,
    dk_carry: Vec<f32>,
    dx_carry: Vec<f32>,
    dtheta_carry: Vec<f32>,
}

/// `v` holding `n` copies of `x`, its allocation reused.
fn refill<T: Copy>(v: &mut Vec<T>, n: usize, x: T) {
    v.clear();
    v.resize(n, x);
}

/// `v` holding a copy of `src`, its allocation reused.
fn copy_into(v: &mut Vec<f32>, src: &[f32]) {
    v.clear();
    v.extend_from_slice(src);
}

/// Replay one head over tokens `t0..t1` from `start`, filling the first
/// `t1 - t0` records of `hb.rec`; the arithmetic is [`mimo_head_step`]'s.
fn replay_head(
    (cfg, lw, h): (&Mamba3MimoConfig, &Mamba3MimoLayerWeights, usize),
    (proj, bn, cn, tanh_pi): (&[f32], &[f32], &[f32], &[f32]),
    start: &Mamba3MimoLayerState,
    (t0, t1): (usize, usize),
    hb: &mut HeadBackward,
) {
    let (n, p_len, r, na, g_n) = (
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
        cfg.ngroups,
    );
    let g = h / (cfg.nheads() / g_n);
    copy_into(
        &mut hb.ssm,
        &start.ssm_state[h * p_len * n..(h + 1) * p_len * n],
    );
    copy_into(&mut hb.k, &start.k_state[h * r * n..(h + 1) * r * n]);
    copy_into(&mut hb.v, &start.v_state[h * p_len..(h + 1) * p_len]);
    copy_into(&mut hb.angle, &start.angle_state[h * na..(h + 1) * na]);
    copy_into(&mut hb.s_start, &hb.ssm);
    if hb.rec.len() < t1 - t0 {
        hb.rec.resize_with(t1 - t0, TokenRecord::default);
    }
    refill(&mut hb.bv, p_len * r, 0.0);
    refill(&mut hb.gx, p_len * r, 0.0);
    let proj_rows = h * r * p_len..(h + 1) * r * p_len;
    let mimo_x = &lw.mimo_x[proj_rows];
    for (slot, t) in (t0..t1).enumerate() {
        let tok = token_at(cfg, proj, (bn, cn, tanh_pi), t);
        let disc = discretize(
            tok.dd_a[h],
            tok.dd_dt[h],
            lw.dt_bias[h],
            tok.trap[h],
            cfg.a_floor,
        );
        let rr = &mut hb.rec[slot];
        rr.dt = disc.dt;
        rr.a = clamp_a(-heavy_tail(tok.dd_a[h]), cfg.a_floor);
        rr.alpha = disc.alpha;
        rr.lam = 1.0 / (1.0 + fast_exp_scalar(-tok.trap[h]));
        rr.lam_c = 1.0 / (1.0 + fast_exp_scalar(tok.trap[h]));
        rr.beta = disc.beta;
        rr.gamma = disc.gamma;
        refill(&mut rr.sc, na, (0.0, 1.0));
        advance_angles(&mut hb.angle, tok.tanh_pi, disc.dt, &mut rr.sc);
        refill(&mut rr.k_pre, r * n, 0.0);
        refill(&mut rr.q_pre, r * n, 0.0);
        for i in 0..r {
            let row = (i * g_n + g) * n..(i * g_n + g + 1) * n;
            let bias = (h * r + i) * n..(h * r + i + 1) * n;
            for (j, ((&b, &bb), (&c, &cb))) in tok.bn[row.clone()]
                .iter()
                .zip(&lw.b_bias[bias.clone()])
                .zip(tok.cn[row].iter().zip(&lw.c_bias[bias]))
                .enumerate()
            {
                rr.k_pre[i * n + j] = b + bb;
                rr.q_pre[i * n + j] = c + cb;
            }
        }
        let TokenRecord {
            k_pre,
            q_pre,
            kc,
            q,
            sc,
            ..
        } = rr;
        copy_into(kc, k_pre);
        copy_into(q, q_pre);
        for i in 0..r {
            rope_half_split(&mut kc[i * n..(i + 1) * n], sc);
            rope_half_split(&mut q[i * n..(i + 1) * n], sc);
        }
        let x_h = &tok.x[h * p_len..(h + 1) * p_len];
        refill(&mut rr.u, p_len * r, 0.0);
        refill(&mut rr.up, p_len * r, 0.0);
        for p in 0..p_len {
            for i in 0..r {
                let psi = mimo_x[i * p_len + p];
                rr.u[p * r + i] = psi * x_h[p];
                hb.gx[p * r + i] = disc.gamma * rr.u[p * r + i];
                rr.up[p * r + i] = psi * hb.v[p];
                hb.bv[p * r + i] = disc.beta * rr.up[p * r + i];
            }
        }
        refill(&mut rr.y, p_len * r, 0.0);
        mimo_rows(
            r,
            MimoRowsIo {
                state: &mut hb.ssm,
                k_prev: &hb.k,
                k_cur: &rr.kc,
                q: &rr.q,
                bv: &hb.bv,
                gx: &hb.gx,
                alpha: disc.alpha,
                y: &mut rr.y,
            },
        );
        let d = lw.d_param[h];
        for (y, &u) in rr.y.iter_mut().zip(&rr.u) {
            *y = d.mul_add(u, *y);
        }
        copy_into(&mut rr.s_cur, &hb.ssm);
        copy_into(&mut rr.x_prev, &hb.v);
        hb.k.copy_from_slice(&rr.kc);
        hb.v.copy_from_slice(x_h);
    }
}

/// The backward of the gate with the per-rank output norm (core's
/// `gate_normed`): with yn_r = y_r rstd_r over the head's channels and
/// a_r = yn_r w, o = sum_r (a_r s_r) Phi_r. Phi, Z, the norm weight and the
/// z column take their gradients, and `dy` the readouts', through the RMS
/// backward per rank: dy_r = rstd_r (dyn_r - yn_r sum_p (dyn_r yn_r) / P).
fn gate_normed_backward(
    dy: &mut [f32],
    (dp, gw, d_o): (&mut [f32], &mut Mamba3MimoLayerWeights, &[f32]),
    (lw, y, z_h): (&Mamba3MimoLayerWeights, &[f32], &[f32]),
    (h, r, p_len, eps): (usize, usize, usize, f32),
) {
    let proj = h * r * p_len;
    let w = &lw.norm_gate_weight[h * p_len..(h + 1) * p_len];
    let mut rstd = [0f32; crate::mamba3_mimo::config::MAX_MIMO_RANK];
    for (i, rs) in rstd[..r].iter_mut().enumerate() {
        let sum_sq = y
            .iter()
            .skip(i)
            .step_by(r)
            .fold(0.0f32, |acc, &v| v.mul_add(v, acc));
        *rs = 1.0 / (sum_sq / p_len as f32 + eps).sqrt();
    }
    for p in 0..p_len {
        for i in 0..r {
            let zr = lw.mimo_z[proj + i * p_len + p];
            let phi = lw.mimo_o[proj + i * p_len + p];
            let v = zr * z_h[p];
            let sg = 1.0 / (1.0 + fast_exp_scalar(-v));
            let silu = v * sg;
            let yn = y[p * r + i] * rstd[i];
            let a = yn * w[p];
            gw.mimo_o[proj + i * p_len + p] += d_o[p] * (a * silu);
            let d_a = d_o[p] * phi * silu;
            gw.norm_gate_weight[h * p_len + p] += d_a * yn;
            dy[p * r + i] = d_a * w[p];
            let dv = d_o[p] * phi * a * (sg * (1.0 + v * (1.0 - sg)));
            gw.mimo_z[proj + i * p_len + p] += dv * z_h[p];
            dp[h * p_len + p] += dv * zr;
        }
    }
    for i in 0..r {
        let mut dot = 0f32;
        for p in 0..p_len {
            dot += dy[p * r + i] * (y[p * r + i] * rstd[i]);
        }
        let c = dot / p_len as f32;
        for p in 0..p_len {
            let yn = y[p * r + i] * rstd[i];
            dy[p * r + i] = rstd[i] * (dy[p * r + i] - yn * c);
        }
    }
}

fn layer_backward(
    d_rows: &mut [f32],
    (lw, gw, lt): (
        &Mamba3MimoLayerWeights,
        &mut Mamba3MimoLayerWeights,
        &LayerTape,
    ),
    (cfg, t_len, stride): (&Mamba3MimoConfig, usize, usize),
    s: &mut Mamba3MimoTrainScratch,
) {
    let (dm, di, ip, bc) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.bc_width(),
    );
    let (n, p_len, r, na, nh, g_n) = (
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
        cfg.nheads(),
        cfg.ngroups,
    );
    grow(&mut s.d_o, t_len * di);
    grow(&mut s.d_proj, t_len * ip);
    grow(&mut s.d_bn, t_len * bc);
    grow(&mut s.d_cn, t_len * bc);
    grow(&mut s.d_tanh, t_len * na.max(1));
    grow(&mut s.d_post, t_len * dm);
    grow(&mut s.norm, t_len * dm);

    // The products take the lane's copy of the residual gradient.
    s.gemm.backward(
        s.dtype,
        (&mut s.d_o[..t_len * di], &mut gw.out_proj_w, None),
        (d_rows, &lt.o, &lw.out_proj_w),
        (t_len, di, dm),
    );
    prepare_rows(cfg, lw, &lt.proj, s, t_len);
    s.d_proj[..t_len * ip].fill(0.0);
    s.d_bn[..t_len * bc].fill(0.0);
    s.d_cn[..t_len * bc].fill(0.0);
    s.d_tanh[..t_len * na.max(1)].fill(0.0);
    let heads_col = 2 * di + 2 * bc;

    let hb = &mut s.head_bwd;
    for h in 0..nh {
        let g = h / (nh / g_n);
        let proj_rows = h * r * p_len..(h + 1) * r * p_len;
        refill(&mut hb.ds_carry, p_len * n, 0.0);
        refill(&mut hb.dk_carry, r * n, 0.0);
        refill(&mut hb.dx_carry, p_len, 0.0);
        refill(&mut hb.dtheta_carry, na, 0.0);
        let segments = t_len.div_ceil(stride);
        for seg in (0..segments).rev() {
            let (t0, t1) = (seg * stride, ((seg + 1) * stride).min(t_len));
            replay_head(
                (cfg, lw, h),
                (&lt.proj, &s.bn, &s.cn, &s.tanh_pi),
                &lt.ckpt[seg],
                (t0, t1),
                hb,
            );
            for t in (t0..t1).rev() {
                let i_t = t - t0;
                let rec = &hb.rec;
                let rr = &rec[i_t];
                let s_prev: &[f32] = if i_t == 0 {
                    &hb.s_start
                } else {
                    &rec[i_t - 1].s_cur
                };
                let k_prev: &[f32] = if i_t == 0 {
                    &lt.ckpt[seg].k_state[h * r * n..(h + 1) * r * n]
                } else {
                    &rec[i_t - 1].kc
                };
                let tok = token_at(cfg, &lt.proj, (&s.bn, &s.cn, &s.tanh_pi), t);
                let x_h = &tok.x[h * p_len..(h + 1) * p_len];
                let z_h = &tok.z[h * p_len..(h + 1) * p_len];
                let d_o = &s.d_o[t * di + h * p_len..t * di + (h + 1) * p_len];
                let dp = &mut s.d_proj[t * ip..(t + 1) * ip];

                // The gate: o = sum_r ((y_r s_r) Phi_r), s_r = silu(Z_r z),
                // with the output norm y_r taking (y_r rstd_r) w first.
                refill(&mut hb.dy, p_len * r, 0.0);
                let dy = &mut hb.dy[..];
                if cfg.is_outproj_norm {
                    gate_normed_backward(
                        dy,
                        (&mut *dp, &mut *gw, d_o),
                        (lw, &rr.y, z_h),
                        (h, r, p_len, cfg.ssm_norm_eps),
                    );
                } else {
                    for p in 0..p_len {
                        for i in 0..r {
                            let zr = lw.mimo_z[proj_rows.start + i * p_len + p];
                            let phi = lw.mimo_o[proj_rows.start + i * p_len + p];
                            let v = zr * z_h[p];
                            let sg = 1.0 / (1.0 + fast_exp_scalar(-v));
                            let silu = v * sg;
                            let y = rr.y[p * r + i];
                            gw.mimo_o[proj_rows.start + i * p_len + p] += d_o[p] * (y * silu);
                            dy[p * r + i] = d_o[p] * phi * silu;
                            let dv = d_o[p] * phi * y * (sg * (1.0 + v * (1.0 - sg)));
                            gw.mimo_z[proj_rows.start + i * p_len + p] += dv * z_h[p];
                            dp[h * p_len + p] += dv * zr;
                        }
                    }
                }
                // D and the readout.
                let d = lw.d_param[h];
                refill(&mut hb.du, p_len * r, 0.0);
                copy_into(&mut hb.ds, &hb.ds_carry);
                refill(&mut hb.dq, r * n, 0.0);
                let (du, ds, dq) = (&mut hb.du[..], &mut hb.ds[..], &mut hb.dq[..]);
                for p in 0..p_len {
                    for i in 0..r {
                        let dyv = dy[p * r + i];
                        gw.d_param[h] += dyv * rr.u[p * r + i];
                        du[p * r + i] += dyv * d;
                        for j in 0..n {
                            ds[p * n + j] += dyv * rr.q[i * n + j];
                            dq[i * n + j] += dyv * rr.s_cur[p * n + j];
                        }
                    }
                }
                // The recurrence.
                let mut d_alpha = 0f32;
                let (mut d_beta, mut d_gamma) = (0f32, 0f32);
                refill(&mut hb.d_up, p_len * r, 0.0);
                copy_into(&mut hb.dkc, &hb.dk_carry);
                refill(&mut hb.dkp, r * n, 0.0);
                let (d_up, dkc, dkp) = (&mut hb.d_up[..], &mut hb.dkc[..], &mut hb.dkp[..]);
                let ds_carry = &mut hb.ds_carry[..];
                for p in 0..p_len {
                    for j in 0..n {
                        let dsv = ds[p * n + j];
                        d_alpha += dsv * s_prev[p * n + j];
                        ds_carry[p * n + j] = rr.alpha * dsv;
                        for i in 0..r {
                            let bv = rr.beta * rr.up[p * r + i];
                            let gx = rr.gamma * rr.u[p * r + i];
                            let dbv = dsv * k_prev[i * n + j];
                            let dgx = dsv * rr.kc[i * n + j];
                            dkp[i * n + j] += dsv * bv;
                            dkc[i * n + j] += dsv * gx;
                            d_beta += dbv * rr.up[p * r + i];
                            d_up[p * r + i] += dbv * rr.beta;
                            d_gamma += dgx * rr.u[p * r + i];
                            du[p * r + i] += dgx * rr.gamma;
                        }
                    }
                }
                std::mem::swap(&mut hb.dk_carry, &mut hb.dkp);
                // The rank projections of x and of the previous x.
                refill(&mut hb.dx_prev, p_len, 0.0);
                let dx_prev = &mut hb.dx_prev[..];
                for p in 0..p_len {
                    let mut dx = hb.dx_carry[p];
                    for i in 0..r {
                        let psi = lw.mimo_x[proj_rows.start + i * p_len + p];
                        gw.mimo_x[proj_rows.start + i * p_len + p] +=
                            du[p * r + i] * x_h[p] + d_up[p * r + i] * rr.x_prev[p];
                        dx += du[p * r + i] * psi;
                        dx_prev[p] += d_up[p * r + i] * psi;
                    }
                    dp[di + h * p_len + p] += dx;
                }
                std::mem::swap(&mut hb.dx_carry, &mut hb.dx_prev);
                // The rotation, back to the biased keys and queries and to the
                // angles.
                let half = n / 2;
                copy_into(&mut hb.dtheta, &hb.dtheta_carry);
                let dtheta = &mut hb.dtheta[..];
                for (vec_grad, pre) in [(&mut *dkc, &rr.k_pre), (&mut *dq, &rr.q_pre)] {
                    for i in 0..r {
                        let gv = &mut vec_grad[i * n..(i + 1) * n];
                        let pv = &pre[i * n..(i + 1) * n];
                        for (a, &(sn, cs)) in rr.sc.iter().enumerate() {
                            let (g0, g1) = (gv[a], gv[a + half]);
                            let (v0, v1) = (pv[a], pv[a + half]);
                            dtheta[a] += g0 * (-sn * v0 - cs * v1) + g1 * (cs * v0 - sn * v1);
                            gv[a] = cs * g0 + sn * g1;
                            gv[a + half] = -sn * g0 + cs * g1;
                        }
                    }
                }
                for i in 0..r {
                    let row = t * bc + (i * g_n + g) * n;
                    let bias = (h * r + i) * n;
                    for j in 0..n {
                        gw.b_bias[bias + j] += dkc[i * n + j];
                        gw.c_bias[bias + j] += dq[i * n + j];
                        s.d_bn[row + j] += dkc[i * n + j];
                        s.d_cn[row + j] += dq[i * n + j];
                    }
                }
                // The angles: theta_t = theta_{t-1} + tanh_pi dt.
                let mut d_dt = 0f32;
                let d_tanh_row = &mut s.d_tanh[t * na.max(1)..t * na.max(1) + na];
                for ((&dth, &tp), dtan) in dtheta.iter().zip(tok.tanh_pi).zip(d_tanh_row) {
                    d_dt += dth * tp;
                    *dtan += dth * rr.dt;
                }
                std::mem::swap(&mut hb.dtheta_carry, &mut hb.dtheta);
                // The discretisation.
                let d_lam_c = d_beta * (rr.alpha * rr.dt);
                d_alpha += d_beta * (rr.dt * rr.lam_c);
                d_dt += d_beta * (rr.alpha * rr.lam_c);
                let d_lam = d_gamma * rr.dt;
                d_dt += d_gamma * rr.lam;
                let d_a = d_alpha * rr.alpha * rr.dt;
                d_dt += d_alpha * rr.alpha * rr.a;
                // lam (1 - lam) is lam * lam_c exactly, and lam_c is computed as
                // sigmoid(-trap) so it keeps its digits when trap is large; the
                // subtraction from one would round the tails to zero.
                let d_trap = (d_lam - d_lam_c) * (rr.lam * rr.lam_c);
                let dd_a = tok.dd_a[h];
                let unclamped = -heavy_tail(dd_a) <= -cfg.a_floor;
                let d_dd_a = if unclamped {
                    -d_a * heavy_tail_deriv(dd_a)
                } else {
                    0.0
                };
                let sp_in = tok.dd_dt[h] + lw.dt_bias[h];
                let d_sp = if sp_in > 20.0 {
                    d_dt
                } else {
                    d_dt / (1.0 + fast_exp_scalar(-sp_in))
                };
                gw.dt_bias[h] += d_sp;
                dp[heads_col + h] += d_sp;
                dp[heads_col + nh + h] += d_dd_a;
                dp[heads_col + 2 * nh + h] += d_trap;
            }
        }
    }

    // The B and C norms, and the angle projections, per token.
    let pi = std::f32::consts::PI;
    for t in 0..t_len {
        let row = &lt.proj[t * ip..(t + 1) * ip];
        let dp = &mut s.d_proj[t * ip..(t + 1) * ip];
        let (b_raw, c_raw) = (
            &row[2 * di..2 * di + bc],
            &row[2 * di + bc..2 * di + 2 * bc],
        );
        let mut d_b = s.d_bn[t * bc..(t + 1) * bc].to_vec();
        let mut d_c = s.d_cn[t * bc..(t + 1) * bc].to_vec();
        row_norm_backward(
            &mut d_b,
            b_raw,
            (&lw.b_norm_weight, &mut gw.b_norm_weight),
            cfg.ssm_norm_eps,
        );
        row_norm_backward(
            &mut d_c,
            c_raw,
            (&lw.c_norm_weight, &mut gw.c_norm_weight),
            cfg.ssm_norm_eps,
        );
        for (dv, &g) in dp[2 * di..2 * di + bc].iter_mut().zip(&d_b) {
            *dv += g;
        }
        for (dv, &g) in dp[2 * di + bc..2 * di + 2 * bc].iter_mut().zip(&d_c) {
            *dv += g;
        }
        let ang = heads_col + 3 * nh;
        for a in 0..na {
            let th = row[ang + a].tanh();
            dp[ang + a] += s.d_tanh[t * na.max(1) + a] * pi * (1.0 - th * th);
        }
    }

    // in_proj and the block norm; the residual already sits in d_rows.
    // Every term of the in_proj gradient is in: a half lane rounds it once,
    // as the GPU stores each of its columns.
    store(s.dtype, &mut s.d_proj[..t_len * ip]);
    for (d, x) in s.norm.chunks_exact_mut(dm).zip(lt.x_in.chunks_exact(dm)) {
        block_norm_row(d, x, (&lw.norm_weight, &lw.norm_bias), cfg.rms_norm_eps);
    }
    store(s.dtype, &mut s.norm[..t_len * dm]);
    s.gemm.backward(
        s.dtype,
        (&mut s.d_post[..t_len * dm], &mut gw.in_proj_w, None),
        (
            &s.d_proj[..t_len * ip],
            &s.norm[..t_len * dm],
            &lw.in_proj_w,
        ),
        (t_len, dm, ip),
    );
    block_norm_backward(
        &mut s.d_post[..t_len * dm],
        &lt.x_in,
        (&lw.norm_weight, &lw.norm_bias),
        (&mut gw.norm_weight, &mut gw.norm_bias),
        cfg.rms_norm_eps,
    );
    for (d, &dp) in d_rows.iter_mut().zip(&s.d_post[..t_len * dm]) {
        *d += dp;
    }
}

impl Mamba3MimoWeights {
    /// A gradient container: the same shapes as `self`, all zero.
    pub fn zeros_like(&self) -> Self {
        let z = |v: &Vec<f32>| vec![0.0; v.len()];
        Self {
            input_proj_w: z(&self.input_proj_w),
            input_proj_b: z(&self.input_proj_b),
            layers: self
                .layers
                .iter()
                .map(|l| Mamba3MimoLayerWeights {
                    norm_weight: z(&l.norm_weight),
                    norm_bias: z(&l.norm_bias),
                    in_proj_w: z(&l.in_proj_w),
                    dt_bias: z(&l.dt_bias),
                    b_norm_weight: z(&l.b_norm_weight),
                    c_norm_weight: z(&l.c_norm_weight),
                    b_bias: z(&l.b_bias),
                    c_bias: z(&l.c_bias),
                    d_param: z(&l.d_param),
                    mimo_x: z(&l.mimo_x),
                    mimo_z: z(&l.mimo_z),
                    mimo_o: z(&l.mimo_o),
                    norm_gate_weight: z(&l.norm_gate_weight),
                    out_proj_w: z(&l.out_proj_w),
                })
                .collect(),
            norm_f_weight: z(&self.norm_f_weight),
            norm_f_bias: z(&self.norm_f_bias),
            mlp: self
                .mlp
                .iter()
                .map(|m| crate::mlp::GatedMlpLayerWeights {
                    norm_weight: z(&m.norm_weight),
                    norm_bias: z(&m.norm_bias),
                    fc1_w: z(&m.fc1_w),
                    fc2_w: z(&m.fc2_w),
                })
                .collect(),
        }
    }
}
