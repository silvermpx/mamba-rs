//! Mamba-3 MIMO training on the CPU: a forward that records a tape and the
//! backward through it.
//!
//! The forward runs the prefill's per-head step token by token (tokens
//! outer, heads inner: each head's chain is the prefill's), so its outputs
//! and states are the prefill's bits; the tape keeps each layer's input
//! rows, in_proj rows and gated rows, and the whole layer state every
//! `stride` tokens. The backward replays each segment of each head from
//! its checkpoint through the same step functions, keeping that segment's
//! intermediates, and walks every stage back in a fixed order. The gated
//! output norm (`is_outproj_norm`) does not train yet.

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use super::core::{
    HeadScratch, HeadState, MimoRowsIo, TokenIn, advance_angles, discretize, mimo_head_step,
    mimo_rows, prepare_token, rope_half_split, row_norm,
};
use crate::mamba3_siso::cpu::forward::{clamp_a, heavy_tail, heavy_tail_deriv, simd_sum_sq};
use crate::mlp::cpu::{
    GatedMlpBackwardScratch, GatedMlpRowsScratch, gated_mlp_rows, gated_mlp_rows_backward,
};
use crate::ops::blas::{sgemm_backward, sgemm_forward};
use crate::ops::fast_math::fast_exp_scalar;
use crate::ops::norms::rms_norm_weighted;

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
    mlp: GatedMlpRowsScratch,
    mlp_bwd: GatedMlpBackwardScratch,
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
    if cfg.is_outproj_norm {
        return Err("the gated output norm does not train on the CPU yet".into());
    }
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        return Ok(());
    }
    tape.input.copy_from_slice(&input[..t_len * tape.input_dim]);
    let rows = &mut out[..t_len * dm];
    if w.input_proj_w.is_empty() {
        rows.copy_from_slice(&input[..t_len * dm]);
    } else {
        sgemm_forward(
            rows,
            &input[..t_len * tape.input_dim],
            &w.input_proj_w,
            bias(&w.input_proj_b),
            t_len,
            tape.input_dim,
            dm,
        );
    }
    if s.heads.len() != cfg.nheads() {
        s.heads = (0..cfg.nheads()).map(|_| HeadScratch::new(cfg)).collect();
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
                    rms_norm_weighted(d_row, s_row, &mlp.norm_weight, eps);
                }
            });
        }
    }
    tape.final_in.copy_from_slice(rows);
    for (o, x) in rows
        .chunks_exact_mut(dm)
        .zip(tape.final_in.chunks_exact(dm))
    {
        row_norm(o, x, &w.norm_f_weight, cfg.rms_norm_eps);
    }
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
    lt.x_in.copy_from_slice(rows);
    for (d, x) in s.norm.chunks_exact_mut(dm).zip(rows.chunks_exact(dm)) {
        row_norm(d, x, &lw.norm_weight, cfg.rms_norm_eps);
    }
    sgemm_forward(
        &mut lt.proj,
        &s.norm[..t_len * dm],
        &lw.in_proj_w,
        None,
        t_len,
        dm,
        ip,
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
    sgemm_forward(
        &mut s.out[..t_len * dm],
        &lt.o,
        &lw.out_proj_w,
        None,
        t_len,
        di,
        dm,
    );
    for (r, &o) in rows.iter_mut().zip(&s.out[..t_len * dm]) {
        *r += o;
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
/// `grads`, shaped like `w`.
pub fn backward_mamba3_mimo_train(
    d_out: &mut [f32],
    d_input: &mut [f32],
    (w, grads): (&Mamba3MimoWeights, &mut Mamba3MimoWeights),
    tape: &Mamba3MimoTape,
    (cfg, s): (&Mamba3MimoConfig, &mut Mamba3MimoTrainScratch),
) {
    let t_len = tape.seq_len;
    let dm = cfg.d_model;
    if t_len == 0 {
        return;
    }
    let d_rows = &mut d_out[..t_len * dm];
    row_norm_backward(
        d_rows,
        &tape.final_in,
        (&w.norm_f_weight, &mut grads.norm_f_weight),
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
        sgemm_backward(
            dx,
            &mut grads.input_proj_w,
            db,
            d_rows,
            &tape.input,
            &w.input_proj_w,
            (t_len, tape.input_dim, dm),
        );
    } else if !d_input.is_empty() {
        d_input[..t_len * dm].copy_from_slice(d_rows);
    }
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

/// Replay one head over tokens `t0..t1` from `start`, filling `rec`; the
/// arithmetic is [`mimo_head_step`]'s.
fn replay_head(
    (cfg, lw, h): (&Mamba3MimoConfig, &Mamba3MimoLayerWeights, usize),
    (proj, bn, cn, tanh_pi): (&[f32], &[f32], &[f32], &[f32]),
    start: &Mamba3MimoLayerState,
    (t0, t1): (usize, usize),
    rec: &mut Vec<TokenRecord>,
    s_start: &mut Vec<f32>,
) {
    let (n, p_len, r, na, g_n) = (
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
        cfg.ngroups,
    );
    let g = h / (cfg.nheads() / g_n);
    let mut ssm = start.ssm_state[h * p_len * n..(h + 1) * p_len * n].to_vec();
    let mut k = start.k_state[h * r * n..(h + 1) * r * n].to_vec();
    let mut v = start.v_state[h * p_len..(h + 1) * p_len].to_vec();
    let mut angle = start.angle_state[h * na..(h + 1) * na].to_vec();
    s_start.clear();
    s_start.extend_from_slice(&ssm);
    rec.clear();
    let proj_rows = h * r * p_len..(h + 1) * r * p_len;
    let mimo_x = &lw.mimo_x[proj_rows];
    for t in t0..t1 {
        let tok = token_at(cfg, proj, (bn, cn, tanh_pi), t);
        let disc = discretize(
            tok.dd_a[h],
            tok.dd_dt[h],
            lw.dt_bias[h],
            tok.trap[h],
            cfg.a_floor,
        );
        let mut rr = TokenRecord {
            dt: disc.dt,
            a: clamp_a(-heavy_tail(tok.dd_a[h]), cfg.a_floor),
            alpha: disc.alpha,
            lam: 1.0 / (1.0 + fast_exp_scalar(-tok.trap[h])),
            lam_c: 1.0 / (1.0 + fast_exp_scalar(tok.trap[h])),
            beta: disc.beta,
            gamma: disc.gamma,
            sc: vec![(0.0, 1.0); na],
            ..TokenRecord::default()
        };
        advance_angles(&mut angle, tok.tanh_pi, disc.dt, &mut rr.sc);
        rr.k_pre = vec![0.0; r * n];
        rr.q_pre = vec![0.0; r * n];
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
        rr.kc = rr.k_pre.clone();
        rr.q = rr.q_pre.clone();
        for i in 0..r {
            rope_half_split(&mut rr.kc[i * n..(i + 1) * n], &rr.sc);
            rope_half_split(&mut rr.q[i * n..(i + 1) * n], &rr.sc);
        }
        let x_h = &tok.x[h * p_len..(h + 1) * p_len];
        rr.u = vec![0.0; p_len * r];
        rr.up = vec![0.0; p_len * r];
        let mut bv = vec![0.0; p_len * r];
        let mut gx = vec![0.0; p_len * r];
        for p in 0..p_len {
            for i in 0..r {
                let psi = mimo_x[i * p_len + p];
                rr.u[p * r + i] = psi * x_h[p];
                gx[p * r + i] = disc.gamma * rr.u[p * r + i];
                rr.up[p * r + i] = psi * v[p];
                bv[p * r + i] = disc.beta * rr.up[p * r + i];
            }
        }
        rr.y = vec![0.0; p_len * r];
        mimo_rows(
            r,
            MimoRowsIo {
                state: &mut ssm,
                k_prev: &k,
                k_cur: &rr.kc,
                q: &rr.q,
                bv: &bv,
                gx: &gx,
                alpha: disc.alpha,
                y: &mut rr.y,
            },
        );
        let d = lw.d_param[h];
        for (y, &u) in rr.y.iter_mut().zip(&rr.u) {
            *y = d.mul_add(u, *y);
        }
        rr.s_cur = ssm.clone();
        rr.x_prev = v.clone();
        k.copy_from_slice(&rr.kc);
        v.copy_from_slice(x_h);
        rec.push(rr);
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

    sgemm_backward(
        &mut s.d_o[..t_len * di],
        &mut gw.out_proj_w,
        None,
        d_rows,
        &lt.o,
        &lw.out_proj_w,
        (t_len, di, dm),
    );
    prepare_rows(cfg, lw, &lt.proj, s, t_len);
    s.d_proj[..t_len * ip].fill(0.0);
    s.d_bn[..t_len * bc].fill(0.0);
    s.d_cn[..t_len * bc].fill(0.0);
    s.d_tanh[..t_len * na.max(1)].fill(0.0);
    let heads_col = 2 * di + 2 * bc;

    let mut rec = Vec::new();
    let mut s_start = Vec::new();
    for h in 0..nh {
        let g = h / (nh / g_n);
        let proj_rows = h * r * p_len..(h + 1) * r * p_len;
        let mut ds_carry = vec![0f32; p_len * n];
        let mut dk_carry = vec![0f32; r * n];
        let mut dx_carry = vec![0f32; p_len];
        let mut dtheta_carry = vec![0f32; na];
        let segments = t_len.div_ceil(stride);
        for seg in (0..segments).rev() {
            let (t0, t1) = (seg * stride, ((seg + 1) * stride).min(t_len));
            replay_head(
                (cfg, lw, h),
                (&lt.proj, &s.bn, &s.cn, &s.tanh_pi),
                &lt.ckpt[seg],
                (t0, t1),
                &mut rec,
                &mut s_start,
            );
            for t in (t0..t1).rev() {
                let i_t = t - t0;
                let rr = &rec[i_t];
                let s_prev: &[f32] = if i_t == 0 {
                    &s_start
                } else {
                    &rec[i_t - 1].s_cur
                };
                let k_prev: Vec<f32> = if i_t == 0 {
                    lt.ckpt[seg].k_state[h * r * n..(h + 1) * r * n].to_vec()
                } else {
                    rec[i_t - 1].kc.clone()
                };
                let tok = token_at(cfg, &lt.proj, (&s.bn, &s.cn, &s.tanh_pi), t);
                let x_h = &tok.x[h * p_len..(h + 1) * p_len];
                let z_h = &tok.z[h * p_len..(h + 1) * p_len];
                let d_o = &s.d_o[t * di + h * p_len..t * di + (h + 1) * p_len];
                let dp = &mut s.d_proj[t * ip..(t + 1) * ip];

                // The gate: o = sum_r ((y_r s_r) Phi_r), s_r = silu(Z_r z).
                let mut dy = vec![0f32; p_len * r];
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
                // D and the readout.
                let d = lw.d_param[h];
                let mut du = vec![0f32; p_len * r];
                let mut ds = ds_carry.clone();
                let mut dq = vec![0f32; r * n];
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
                let mut d_up = vec![0f32; p_len * r];
                let mut dkc = dk_carry.clone();
                let mut dkp = vec![0f32; r * n];
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
                dk_carry = dkp;
                // The rank projections of x and of the previous x.
                let mut dx_prev = vec![0f32; p_len];
                for p in 0..p_len {
                    let mut dx = dx_carry[p];
                    for i in 0..r {
                        let psi = lw.mimo_x[proj_rows.start + i * p_len + p];
                        gw.mimo_x[proj_rows.start + i * p_len + p] +=
                            du[p * r + i] * x_h[p] + d_up[p * r + i] * rr.x_prev[p];
                        dx += du[p * r + i] * psi;
                        dx_prev[p] += d_up[p * r + i] * psi;
                    }
                    dp[di + h * p_len + p] += dx;
                }
                dx_carry = dx_prev;
                // The rotation, back to the biased keys and queries and to the
                // angles.
                let half = n / 2;
                let mut dtheta = dtheta_carry.clone();
                for (vec_grad, pre) in [(&mut dkc, &rr.k_pre), (&mut dq, &rr.q_pre)] {
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
                dtheta_carry = dtheta;
                // The discretisation.
                let d_lam_c = d_beta * (rr.alpha * rr.dt);
                d_alpha += d_beta * (rr.dt * rr.lam_c);
                d_dt += d_beta * (rr.alpha * rr.lam_c);
                let d_lam = d_gamma * rr.dt;
                d_dt += d_gamma * rr.lam;
                let d_a = d_alpha * rr.alpha * rr.dt;
                d_dt += d_alpha * rr.alpha * rr.a;
                let d_trap =
                    d_lam * rr.lam * (1.0 - rr.lam) - d_lam_c * rr.lam_c * (1.0 - rr.lam_c);
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
    for (d, x) in s.norm.chunks_exact_mut(dm).zip(lt.x_in.chunks_exact(dm)) {
        row_norm(d, x, &lw.norm_weight, cfg.rms_norm_eps);
    }
    sgemm_backward(
        &mut s.d_post[..t_len * dm],
        &mut gw.in_proj_w,
        None,
        &s.d_proj[..t_len * ip],
        &s.norm[..t_len * dm],
        &lw.in_proj_w,
        (t_len, dm, ip),
    );
    row_norm_backward(
        &mut s.d_post[..t_len * dm],
        &lt.x_in,
        (&lw.norm_weight, &mut gw.norm_weight),
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
            mlp: self
                .mlp
                .iter()
                .map(|m| crate::mlp::GatedMlpLayerWeights {
                    norm_weight: z(&m.norm_weight),
                    fc1_w: z(&m.fc1_w),
                    fc2_w: z(&m.fc2_w),
                })
                .collect(),
        }
    }
}
