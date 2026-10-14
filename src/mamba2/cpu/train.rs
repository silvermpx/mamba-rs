//! Mamba-2 training on the CPU: a forward that records a tape and the
//! backward through it.
//!
//! The forward runs the prefill's math in the prefill's order (the same
//! per-token conv, dt, scan and gate functions, the same GEMMs), so its
//! outputs are the prefill's bits. The tape keeps each layer's input rows,
//! in_proj rows and scan outputs, the conv history the window started from,
//! and the SSM state verbatim every `stride` tokens. The backward recomputes
//! the rest: the conv, the gate, and each segment's states, replayed from
//! its checkpoint. Everything runs in a fixed order on one thread per
//! sample, so the gradients do not depend on the thread count or on the
//! checkpoint stride.
//!
//! In a reduced precision the window rounds where the GPU trainer stores a
//! half value: the normalized rows, the in_proj and out_proj outputs, the
//! conv output and the gated rows in the forward; the residual gradient's
//! copy for the GEMMs, the gated rows' and the normalized rows' input
//! gradients and the in_proj gradient (once, after every term is in) in
//! the backward. TF32 rounds every GEMM operand instead. The scan, the
//! states, the norms' statistics and every weight gradient stay f32.

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::{Mamba2LayerWeights, Mamba2Weights};
use super::core::{
    DSkip, HeadScan, block_norm_row, conv_hist_push, conv_silu_token, dt_da, gate_token, head_scan,
};
use crate::mlp::cpu::RowsGemm;
use crate::mlp::cpu::{
    GatedMlpBackwardScratch, GatedMlpRowsScratch, gated_mlp_rows, gated_mlp_rows_backward,
    mlp_norm_row,
};
use crate::ops::blas::sgemm_forward;
use crate::ops::fold16::sum_sq_fold16;
use crate::ops::lane::LaneGemm;
use crate::ops::norms::{layer_norm_row_backward_in_place, layer_norm_stats};
use crate::ops::portable_math::{
    exp_portable, sigmoid_portable, silu_grad_portable, silu_portable, softplus_portable,
};
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

/// One layer's record.
struct LayerTape {
    /// The layer's input rows `[T, d_model]`.
    x_in: Vec<f32>,
    /// in_proj rows `[T, in_proj_dim]`.
    proj: Vec<f32>,
    /// Scan outputs `[T, d_ssm]`, before the gate.
    y: Vec<f32>,
    /// The conv history the window started from.
    conv_hist0: Vec<f32>,
    /// The SSM state before tokens 0, stride, 2 * stride, ...: `[S + 1,
    /// nheads, d_state, headdim]` with S = ceil(T / stride).
    ckpt: Vec<f32>,
    /// The rows entering the MLP branch, `[T, d_model]` (empty without one).
    mlp_in: Vec<f32>,
}

/// The tape of one training window.
pub struct Mamba2Tape {
    seq_len: usize,
    stride: usize,
    input_dim: usize,
    /// The precision the window computes in.
    dtype: WeightDtype,
    input: Vec<f32>,
    /// The rows entering the final norm.
    final_in: Vec<f32>,
    layers: Vec<LayerTape>,
}

impl Mamba2Tape {
    /// A tape for windows of `seq_len` tokens of `input_dim` features,
    /// checkpointing the state every `stride` tokens.
    pub fn new(cfg: &Mamba2Config, seq_len: usize, input_dim: usize, stride: usize) -> Self {
        let (dm, ip, d_ssm, cd) = (
            cfg.d_model,
            cfg.in_proj_out_dim(),
            cfg.ssm_dim(),
            cfg.conv_dim(),
        );
        let stride = stride.max(1);
        let ckpts = seq_len.div_ceil(stride) + 1;
        let state = cfg.nheads() * cfg.d_state * cfg.headdim;
        let mlp = if cfg.mlp_hidden() > 0 {
            seq_len * dm
        } else {
            0
        };
        Self {
            seq_len,
            stride,
            input_dim,
            dtype: WeightDtype::F32,
            input: vec![0.0; seq_len * input_dim],
            final_in: vec![0.0; seq_len * dm],
            layers: (0..cfg.n_layers)
                .map(|_| LayerTape {
                    x_in: vec![0.0; seq_len * dm],
                    proj: vec![0.0; seq_len * ip],
                    y: vec![0.0; seq_len * d_ssm],
                    conv_hist0: vec![0.0; (cfg.d_conv - 1) * cd],
                    ckpt: vec![0.0; ckpts * state],
                    mlp_in: vec![0.0; mlp],
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

    /// Bytes the tape holds.
    pub fn bytes(&self) -> usize {
        let layer: usize = self
            .layers
            .iter()
            .map(|l| {
                l.x_in.len()
                    + l.proj.len()
                    + l.y.len()
                    + l.conv_hist0.len()
                    + l.ckpt.len()
                    + l.mlp_in.len()
            })
            .sum();
        4 * (self.input.len() + self.final_in.len() + layer)
    }
}

/// Reusable buffers for the training forward and backward.
#[derive(Default)]
pub struct Mamba2TrainScratch {
    norm: Vec<f32>,
    xbc: Vec<f32>,
    pre: Vec<f32>,
    y_h: Vec<f32>,
    gated: Vec<f32>,
    out: Vec<f32>,
    d_gated: Vec<f32>,
    d_y: Vec<f32>,
    d_proj: Vec<f32>,
    d_xbc: Vec<f32>,
    d_post: Vec<f32>,
    seg_states: Vec<f32>,
    carry: Vec<f32>,
    d_xdt: Vec<f32>,
    u: Vec<f32>,
    mlp: GatedMlpRowsScratch,
    mlp_bwd: GatedMlpBackwardScratch,
    /// The precision of the window in flight, from its tape.
    dtype: WeightDtype,
    gemm: LaneGemm,
}

fn bias(b: &[f32]) -> Option<&[f32]> {
    if b.is_empty() { None } else { Some(b) }
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

/// The training forward over a window: `input` is `[seq_len, input_dim]`,
/// `out` (`[seq_len, d_model]`) receives every position's final-norm
/// output, `states` carry in and out, and `tape` records what the backward
/// needs. The outputs and states are the prefill's bits.
pub fn forward_mamba2_train(
    out: &mut [f32],
    input: &[f32],
    w: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    tape: &mut Mamba2Tape,
    (cfg, s): (&Mamba2Config, &mut Mamba2TrainScratch),
) {
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        return;
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
    // A segment without the final norm hands out the residual stream.
    if w.norm_f_weight.is_empty() {
        return;
    }
    tape.final_in.copy_from_slice(rows);
    for (o, x) in rows
        .chunks_exact_mut(dm)
        .zip(tape.final_in.chunks_exact(dm))
    {
        block_norm_row(o, x, (&w.norm_f_weight, &w.norm_f_bias), cfg.rms_norm_eps);
    }
}

/// One mixer over the window with its record; the prefill's operations in
/// the prefill's order.
fn layer_forward(
    rows: &mut [f32],
    (lw, state, lt): (&Mamba2LayerWeights, &mut Mamba2LayerState, &mut LayerTape),
    (cfg, t_len, stride): (&Mamba2Config, usize, usize),
    s: &mut Mamba2TrainScratch,
) {
    let (dm, di, ip, cd) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.conv_dim(),
    );
    let (d_mlp, d_ssm, n, p_len, g_n, nh) = (
        cfg.d_mlp(),
        cfg.ssm_dim(),
        cfg.d_state,
        cfg.headdim,
        cfg.ngroups,
        cfg.nheads(),
    );
    let state_len = nh * n * p_len;
    grow(&mut s.norm, t_len * dm);
    grow(&mut s.xbc, t_len * cd);
    grow(&mut s.gated, t_len * di);
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
        (&lw.in_proj_w, bias(&lw.in_proj_b)),
        (t_len, dm, ip),
    );
    lt.conv_hist0.copy_from_slice(&state.conv_hist);
    let xbc_col = 2 * d_mlp + d_ssm;
    for (proj, xbc) in lt.proj.chunks_exact(ip).zip(s.xbc.chunks_exact_mut(cd)) {
        let raw = &proj[xbc_col..xbc_col + cd];
        conv_silu_token(xbc, raw, &state.conv_hist, lw);
        conv_hist_push(&mut state.conv_hist, raw);
    }
    store(s.dtype, &mut s.xbc[..t_len * cd]);

    // The scan, head by head, with a checkpoint every `stride` tokens.
    for (h, st) in state.ssm.chunks_exact_mut(n * p_len).enumerate() {
        let g = h / (nh / g_n);
        for t in 0..t_len {
            if t % stride == 0 {
                let j = t / stride;
                lt.ckpt[j * state_len + h * n * p_len..j * state_len + (h + 1) * n * p_len]
                    .copy_from_slice(st);
            }
            let row = &s.xbc[t * cd..(t + 1) * cd];
            let (dt, da) = dt_da(
                lt.proj[t * ip + xbc_col + cd + h],
                lw.dt_bias[h],
                lw.a_log[h],
                cfg.dt_limit,
            );
            head_scan(HeadScan {
                state: st,
                x: &row[h * p_len..(h + 1) * p_len],
                b: &row[d_ssm + g * n..d_ssm + (g + 1) * n],
                c: &row[d_ssm + (g_n + g) * n..d_ssm + (g_n + g + 1) * n],
                dt,
                da,
                d: d_skip(lw, cfg, h),
                y: &mut lt.y[t * d_ssm + h * p_len..t * d_ssm + (h + 1) * p_len],
            });
        }
        let last = t_len.div_ceil(stride);
        lt.ckpt[last * state_len + h * n * p_len..last * state_len + (h + 1) * n * p_len]
            .copy_from_slice(st);
    }

    for ((g, y), proj) in s
        .gated
        .chunks_exact_mut(di)
        .zip(lt.y.chunks_exact(d_ssm))
        .zip(lt.proj.chunks_exact(ip))
    {
        gate_token(g, y, proj, lw, cfg);
    }
    store(s.dtype, &mut s.gated[..t_len * di]);
    s.gemm.forward(
        lane,
        (&mut s.out[..t_len * dm], &s.gated[..t_len * di]),
        (&lw.out_proj_w, bias(&lw.out_proj_b)),
        (t_len, di, dm),
    );
    for (t, &o) in rows.iter_mut().zip(&s.out[..t_len * dm]) {
        *t += o;
    }
}

fn d_skip<'a>(lw: &'a Mamba2LayerWeights, cfg: &Mamba2Config, h: usize) -> DSkip<'a> {
    let p_len = cfg.headdim;
    if cfg.d_has_hdim {
        DSkip::Channel(&lw.d_param[h * p_len..(h + 1) * p_len])
    } else {
        DSkip::Head(lw.d_param[h])
    }
}

/// The backward of [`forward_mamba2_train`]. `d_out` holds d loss / d (the
/// outputs) and is used as scratch; `d_input` (`[seq_len, input_dim]`, may
/// be empty to skip it) receives d loss / d (the inputs). The weight
/// gradients accumulate into `grads`, a container shaped like `w`. The
/// gradients stop at the conv history and SSM state the window started
/// from: they are inputs of the window, as on the GPU.
pub fn backward_mamba2_train(
    d_out: &mut [f32],
    d_input: &mut [f32],
    (w, grads): (&Mamba2Weights, &mut Mamba2Weights),
    tape: &Mamba2Tape,
    (cfg, s): (&Mamba2Config, &mut Mamba2TrainScratch),
) {
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        return;
    }
    s.dtype = tape.dtype;
    s.mlp_bwd.dtype = tape.dtype;
    let d_rows = &mut d_out[..t_len * dm];
    // Without the final norm the output's gradient is the residual
    // stream's.
    if !w.norm_f_weight.is_empty() {
        norm_backward(
            d_rows,
            &tape.final_in,
            (&w.norm_f_weight, &w.norm_f_bias),
            (&mut grads.norm_f_weight, &mut grads.norm_f_bias),
            cfg.rms_norm_eps,
        );
    }
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
}

/// d loss / d x of rows through a block norm: [`rms_backward`], or the
/// LayerNorm backward for a model whose norms carry a bias, its statistics
/// recomputed from the rows as the forward computed them. `d_rows` holds
/// d loss / d (the normalised rows) and leaves holding d loss / d x; the
/// parameters' gradients accumulate.
fn norm_backward(
    d_rows: &mut [f32],
    x_rows: &[f32],
    (w, b): (&[f32], &[f32]),
    (gw, gb): (&mut [f32], &mut [f32]),
    eps: f32,
) {
    if b.is_empty() {
        return rms_backward(d_rows, x_rows, (w, gw), eps);
    }
    let n = w.len();
    for (d, x) in d_rows.chunks_exact_mut(n).zip(x_rows.chunks_exact(n)) {
        let stats = layer_norm_stats(x, eps);
        layer_norm_row_backward_in_place(d, (&mut *gw, &mut *gb), x, w, stats);
    }
}

/// d loss / d x of rows normalised as `(x * r) * w` with
/// `r = 1 / sqrt(mean(x^2) + eps)`, the `rms_row` form. `d_rows` holds
/// d loss / d (the normalised rows) and leaves holding d loss / d x; the
/// scale's gradient accumulates.
fn rms_backward(d_rows: &mut [f32], x_rows: &[f32], (w, gw): (&[f32], &mut [f32]), eps: f32) {
    let n = w.len();
    for (d, x) in d_rows.chunks_exact_mut(n).zip(x_rows.chunks_exact(n)) {
        let r = 1.0 / (sum_sq_fold16(x) / n as f32 + eps).sqrt();
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

/// The backward of one mixer over the window. `d_rows` holds d loss / d
/// (the layer's output rows) and leaves holding d loss / d (its input
/// rows), the residual included.
fn layer_backward(
    d_rows: &mut [f32],
    (lw, gw, lt): (&Mamba2LayerWeights, &mut Mamba2LayerWeights, &LayerTape),
    (cfg, t_len, stride): (&Mamba2Config, usize, usize),
    s: &mut Mamba2TrainScratch,
) {
    let (dm, di, ip, cd, width) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.in_proj_out_dim(),
        cfg.conv_dim(),
        cfg.d_conv,
    );
    let (d_mlp, d_ssm, n, p_len, g_n, nh) = (
        cfg.d_mlp(),
        cfg.ssm_dim(),
        cfg.d_state,
        cfg.headdim,
        cfg.ngroups,
        cfg.nheads(),
    );
    let xbc_col = 2 * d_mlp + d_ssm;
    let dt_col = xbc_col + cd;
    for (v, len) in [
        (&mut s.gated, t_len * di),
        (&mut s.d_gated, t_len * di),
        (&mut s.d_y, t_len * d_ssm),
        (&mut s.d_proj, t_len * ip),
        (&mut s.xbc, t_len * cd),
        (&mut s.pre, t_len * cd),
        (&mut s.d_xbc, t_len * cd),
        (&mut s.d_post, t_len * dm),
        (&mut s.norm, t_len * dm),
    ] {
        grow(v, len);
    }

    // out_proj: the gated rows again, then dG and the weight gradients.
    for ((g, y), proj) in s
        .gated
        .chunks_exact_mut(di)
        .zip(lt.y.chunks_exact(d_ssm))
        .zip(lt.proj.chunks_exact(ip))
    {
        gate_token(g, y, proj, lw, cfg);
    }
    store(s.dtype, &mut s.gated[..t_len * di]);
    let db = if gw.out_proj_b.is_empty() {
        None
    } else {
        Some(&mut gw.out_proj_b[..])
    };
    // The products take the lane's copy of the residual gradient, the bias
    // its f32 sum.
    s.gemm.backward(
        s.dtype,
        (&mut s.d_gated[..t_len * di], &mut gw.out_proj_w, db),
        (d_rows, &s.gated[..t_len * di], &lw.out_proj_w),
        (t_len, di, dm),
    );

    // The gate and its norm, row by row, into d_proj (z0, x0, z) and dy.
    s.d_proj[..t_len * ip].fill(0.0);
    grow(&mut s.u, d_ssm / g_n);
    for t in 0..t_len {
        let proj = &lt.proj[t * ip..(t + 1) * ip];
        let y = &lt.y[t * d_ssm..(t + 1) * d_ssm];
        let dg = &s.d_gated[t * di..(t + 1) * di];
        let dp = &mut s.d_proj[t * ip..(t + 1) * ip];
        let dy = &mut s.d_y[t * d_ssm..(t + 1) * d_ssm];
        gate_backward(
            dg,
            (y, proj),
            (dp, dy),
            (lw, &mut gw.gate_norm_weight, &mut s.u),
            cfg,
        );
    }

    // The conv again, keeping its pre-activations.
    {
        let mut hist = lt.conv_hist0.clone();
        for ((proj, pre), xbc) in lt
            .proj
            .chunks_exact(ip)
            .zip(s.pre.chunks_exact_mut(cd))
            .zip(s.xbc.chunks_exact_mut(cd))
        {
            let raw = &proj[xbc_col..xbc_col + cd];
            conv_pre_token(pre, raw, &hist, lw);
            for (x, &v) in xbc.iter_mut().zip(pre.iter()) {
                *x = silu_portable(v);
            }
            conv_hist_push(&mut hist, raw);
        }
    }
    store(s.dtype, &mut s.xbc[..t_len * cd]);

    // The scan, head by head, segment by segment from the last.
    s.d_xbc[..t_len * cd].fill(0.0);
    let state_len = nh * n * p_len;
    grow(&mut s.seg_states, (stride + 1) * n * p_len);
    grow(&mut s.carry, n * p_len);
    grow(&mut s.d_xdt, p_len);
    for h in 0..nh {
        let g = h / (nh / g_n);
        s.carry[..n * p_len].fill(0.0);
        let segments = t_len.div_ceil(stride);
        for seg in (0..segments).rev() {
            let t0 = seg * stride;
            let t1 = (t0 + stride).min(t_len);
            // Replay the segment's states from its checkpoint.
            let ck =
                &lt.ckpt[seg * state_len + h * n * p_len..seg * state_len + (h + 1) * n * p_len];
            s.seg_states[..n * p_len].copy_from_slice(ck);
            grow(&mut s.y_h, p_len);
            for t in t0..t1 {
                let (prev, next) = s.seg_states.split_at_mut((t - t0 + 1) * n * p_len);
                let cur = &mut next[..n * p_len];
                cur.copy_from_slice(&prev[(t - t0) * n * p_len..]);
                let row = &s.xbc[t * cd..(t + 1) * cd];
                let (dt, da) = dt_da(
                    lt.proj[t * ip + dt_col + h],
                    lw.dt_bias[h],
                    lw.a_log[h],
                    cfg.dt_limit,
                );
                head_scan(HeadScan {
                    state: cur,
                    x: &row[h * p_len..(h + 1) * p_len],
                    b: &row[d_ssm + g * n..d_ssm + (g + 1) * n],
                    c: &row[d_ssm + (g_n + g) * n..d_ssm + (g_n + g + 1) * n],
                    dt,
                    da,
                    d: d_skip(lw, cfg, h),
                    y: &mut s.y_h[..p_len],
                });
            }
            for t in (t0..t1).rev() {
                let i = t - t0;
                let s_prev = &s.seg_states[i * n * p_len..(i + 1) * n * p_len];
                let s_cur = &s.seg_states[(i + 1) * n * p_len..(i + 2) * n * p_len];
                scan_step_backward(
                    (s_prev, s_cur, &mut s.carry[..n * p_len]),
                    (h, g, t),
                    (lt, lw, gw),
                    (
                        &s.xbc[t * cd..(t + 1) * cd],
                        &s.d_y,
                        &mut s.d_xbc,
                        &mut s.d_proj,
                    ),
                    (cfg, &mut s.d_xdt[..p_len]),
                );
            }
        }
    }

    // The conv: dv = dxbc silu'(v), into the raw inputs and the weights,
    // rows from the last.
    for t in (0..t_len).rev() {
        for c in 0..cd {
            let dv = s.d_xbc[t * cd + c] * silu_grad_portable(s.pre[t * cd + c]);
            if let Some(b) = gw.conv_b.get_mut(c) {
                *b += dv;
            }
            for k in 0..width {
                let src = t as isize - (width - 1) as isize + k as isize;
                let x = if src >= 0 {
                    lt.proj[src as usize * ip + xbc_col + c]
                } else {
                    lt.conv_hist0[(src + (width - 1) as isize) as usize * cd + c]
                };
                gw.conv_w[c * width + k] += dv * x;
                if src >= 0 {
                    s.d_proj[src as usize * ip + xbc_col + c] += lw.conv_w[c * width + k] * dv;
                }
            }
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
    let db = if gw.in_proj_b.is_empty() {
        None
    } else {
        Some(&mut gw.in_proj_b[..])
    };
    s.gemm.backward(
        s.dtype,
        (&mut s.d_post[..t_len * dm], &mut gw.in_proj_w, db),
        (
            &s.d_proj[..t_len * ip],
            &s.norm[..t_len * dm],
            &lw.in_proj_w,
        ),
        (t_len, dm, ip),
    );
    norm_backward(
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

/// The conv's pre-activation for one token, the chain of
/// [`conv_silu_token`] without the SiLU.
fn conv_pre_token(pre: &mut [f32], cur: &[f32], hist: &[f32], lw: &Mamba2LayerWeights) {
    let cd = cur.len();
    let width = lw.conv_w.len() / cd;
    for (c, (o, &xc)) in pre.iter_mut().zip(cur).enumerate() {
        let taps = &lw.conv_w[c * width..(c + 1) * width];
        let mut v = lw.conv_b.get(c).copied().unwrap_or(0.0);
        for (k, &wk) in taps[..width - 1].iter().enumerate() {
            v = wk.mul_add(hist[k * cd + c], v);
        }
        *o = taps[width - 1].mul_add(xc, v);
    }
}

/// The gate's backward for one row: `dg` is d loss / d (the gated row);
/// writes d z0, d x0 and d z into `dp` and d y into `dy`, and accumulates
/// the norm scale's gradient.
fn gate_backward(
    dg: &[f32],
    (y, proj): (&[f32], &[f32]),
    (dp, dy): (&mut [f32], &mut [f32]),
    (lw, g_norm, u): (&Mamba2LayerWeights, &mut [f32], &mut [f32]),
    cfg: &Mamba2Config,
) {
    let (d_mlp, d_ssm) = (cfg.d_mlp(), cfg.ssm_dim());
    let (z0, rest) = proj.split_at(d_mlp);
    let (x0, rest) = rest.split_at(d_mlp);
    let z = &rest[..d_ssm];
    let (dg_mlp, dg_ssm) = dg.split_at(d_mlp);
    for j in 0..d_mlp {
        dp[j] = dg_mlp[j] * x0[j] * silu_grad_portable(z0[j]);
        dp[d_mlp + j] = dg_mlp[j] * silu_portable(z0[j]);
    }
    let dz = &mut dp[2 * d_mlp..2 * d_mlp + d_ssm];
    if !cfg.rmsnorm {
        for i in 0..d_ssm {
            dy[i] = dg_ssm[i] * silu_portable(z[i]);
            dz[i] = dg_ssm[i] * y[i] * silu_grad_portable(z[i]);
        }
        return;
    }
    let gs = d_ssm / cfg.ngroups;
    let eps = cfg.gated_norm_eps;
    for grp in 0..cfg.ngroups {
        let sl = grp * gs..(grp + 1) * gs;
        let (y_g, z_g, w_g) = (
            &y[sl.clone()],
            &z[sl.clone()],
            &lw.gate_norm_weight[sl.clone()],
        );
        let dg_g = &dg_ssm[sl.clone()];
        if cfg.norm_before_gate {
            // o = ((y r) w) silu(z), r over y.
            let r = 1.0 / (sum_sq_fold16(y_g) / gs as f32 + eps).sqrt();
            let mut c = 0.0f32;
            for i in 0..gs {
                let s = silu_portable(z_g[i]);
                let normed = (y_g[i] * r) * w_g[i];
                dz[sl.start + i] = dg_g[i] * normed * silu_grad_portable(z_g[i]);
                let dn = dg_g[i] * s;
                g_norm[sl.start + i] += dn * (y_g[i] * r);
                c += (dn * w_g[i]) * (y_g[i] * r);
            }
            let c = c / gs as f32;
            for i in 0..gs {
                let dn = dg_g[i] * silu_portable(z_g[i]);
                dy[sl.start + i] = r * (dn * w_g[i] - (y_g[i] * r) * c);
            }
        } else {
            // o = (u r) w with u = y silu(z), r over u.
            let u = &mut u[..gs];
            for ((uv, &yv), &zv) in u.iter_mut().zip(y_g).zip(z_g) {
                *uv = yv * silu_portable(zv);
            }
            let r = 1.0 / (sum_sq_fold16(u) / gs as f32 + eps).sqrt();
            let mut c = 0.0f32;
            for i in 0..gs {
                g_norm[sl.start + i] += dg_g[i] * (u[i] * r);
                c += (dg_g[i] * w_g[i]) * (u[i] * r);
            }
            let c = c / gs as f32;
            for i in 0..gs {
                let du = r * (dg_g[i] * w_g[i] - (u[i] * r) * c);
                dy[sl.start + i] = du * silu_portable(z_g[i]);
                dz[sl.start + i] = du * y_g[i] * silu_grad_portable(z_g[i]);
            }
        }
    }
}

/// One token of one head's scan, backwards. The state went
/// `s_cur = s_prev dA + B (x dt)` and the readout `y = s_cur C + D x`;
/// `carry` holds d loss / d s_cur from the later tokens and leaves holding
/// d loss / d s_prev.
fn scan_step_backward(
    (s_prev, s_cur, carry): (&[f32], &[f32], &mut [f32]),
    (h, g, t): (usize, usize, usize),
    (lt, lw, gw): (&LayerTape, &Mamba2LayerWeights, &mut Mamba2LayerWeights),
    (xbc, d_y, d_xbc, d_proj): (&[f32], &[f32], &mut [f32], &mut [f32]),
    (cfg, d_xdt): (&Mamba2Config, &mut [f32]),
) {
    let (n, p_len, g_n) = (cfg.d_state, cfg.headdim, cfg.ngroups);
    let (d_ssm, cd, ip) = (cfg.ssm_dim(), cfg.conv_dim(), cfg.in_proj_out_dim());
    let dt_col = 2 * cfg.d_mlp() + d_ssm + cd;
    let raw = lt.proj[t * ip + dt_col + h];
    let a = raw + lw.dt_bias[h];
    let sp = softplus_portable(a);
    let (lo, hi) = cfg.dt_limit;
    let dt = sp.clamp(lo, hi);
    let a_neg = -exp_portable(lw.a_log[h]);
    let da = exp_portable(dt * a_neg);
    let x = &xbc[h * p_len..(h + 1) * p_len];
    let b = &xbc[d_ssm + g * n..d_ssm + (g + 1) * n];
    let c = &xbc[d_ssm + (g_n + g) * n..d_ssm + (g_n + g + 1) * n];
    let dy = &d_y[t * d_ssm + h * p_len..t * d_ssm + (h + 1) * p_len];

    d_xdt.fill(0.0);
    let mut d_da = 0.0f32;
    let dxbc = &mut d_xbc[t * cd..(t + 1) * cd];
    for k in 0..n {
        let (mut db, mut dc) = (0.0f32, 0.0f32);
        for p in 0..p_len {
            let idx = k * p_len + p;
            let ds = c[k].mul_add(dy[p], carry[idx]);
            dc = s_cur[idx].mul_add(dy[p], dc);
            db = ds.mul_add(x[p] * dt, db);
            d_xdt[p] = ds.mul_add(b[k], d_xdt[p]);
            d_da = ds.mul_add(s_prev[idx], d_da);
            carry[idx] = ds * da;
        }
        dxbc[d_ssm + g * n + k] += db;
        dxbc[d_ssm + (g_n + g) * n + k] += dc;
    }
    let mut d_dt = 0.0f32;
    for p in 0..p_len {
        let d = match d_skip(lw, cfg, h) {
            DSkip::Head(d) => d,
            DSkip::Channel(d) => d[p],
        };
        dxbc[h * p_len + p] += dy[p].mul_add(d, d_xdt[p] * dt);
        d_dt = d_xdt[p].mul_add(x[p], d_dt);
        if cfg.d_has_hdim {
            gw.d_param[h * p_len + p] += dy[p] * x[p];
        } else {
            gw.d_param[h] += dy[p] * x[p];
        }
    }
    // dA = exp(dt a_neg): through dt and through a_neg = -exp(a_log).
    d_dt = (d_da * da).mul_add(a_neg, d_dt);
    gw.a_log[h] += (d_da * da * dt) * a_neg;
    let clamped = sp < lo || sp > hi;
    let d_raw = if clamped {
        0.0
    } else if a > 20.0 {
        d_dt
    } else {
        d_dt * sigmoid_portable(a)
    };
    gw.dt_bias[h] += d_raw;
    d_proj[t * ip + dt_col + h] += d_raw;
}

impl Mamba2Weights {
    /// A gradient container: the same shapes as `self`, all zero.
    pub fn zeros_like(&self) -> Self {
        let z = |v: &Vec<f32>| vec![0.0; v.len()];
        Self {
            input_proj_w: z(&self.input_proj_w),
            input_proj_b: z(&self.input_proj_b),
            layers: self
                .layers
                .iter()
                .map(|l| Mamba2LayerWeights {
                    norm_weight: z(&l.norm_weight),
                    norm_bias: z(&l.norm_bias),
                    in_proj_w: z(&l.in_proj_w),
                    in_proj_b: z(&l.in_proj_b),
                    conv_w: z(&l.conv_w),
                    conv_b: z(&l.conv_b),
                    dt_bias: z(&l.dt_bias),
                    a_log: z(&l.a_log),
                    d_param: z(&l.d_param),
                    gate_norm_weight: z(&l.gate_norm_weight),
                    out_proj_w: z(&l.out_proj_w),
                    out_proj_b: z(&l.out_proj_b),
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
