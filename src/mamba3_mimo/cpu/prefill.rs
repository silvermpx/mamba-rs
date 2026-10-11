//! Mamba-3 MIMO prefill: a prompt window through the backbone in one pass,
//! carrying the recurrent state in and out.
//!
//! The projections run as GEMMs over every row of the window and the heads
//! run in parallel, each walking the window token by token with the same
//! [`mimo_head_step`] the decode runs. The BLAS-free GEMM adds every output
//! in ascending k whatever the number of rows, so in the default build the
//! prefill gives the decode chain's bits, for either [`PrefillMode`].

use rayon::prelude::*;

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use super::core::{HeadScratch, HeadState, TokenIn, mimo_head_step, prepare_token, row_norm};
use crate::mamba1::cpu::prefill::{PrefillMode, zip_rows};
use crate::mlp::cpu::{GatedMlpRowsScratch, gated_mlp_rows};
use crate::ops::blas::{sgemm_forward, sgemm_forward_par};
use crate::ops::norms::rms_norm_weighted;

/// Reusable prefill buffers, grown to the longest window seen.
pub struct Mamba3MimoPrefillScratch {
    rows: usize,
    norm: Vec<f32>,
    proj: Vec<f32>,
    bn: Vec<f32>,
    cn: Vec<f32>,
    tanh_pi: Vec<f32>,
    /// Head-major gate outputs `[nheads, T, headdim]`.
    o_hm: Vec<f32>,
    /// Row-major gate outputs `[T, d_inner]`, the out_proj input.
    o_rm: Vec<f32>,
    out: Vec<f32>,
    heads: Vec<HeadScratch>,
    mlp: GatedMlpRowsScratch,
}

impl Mamba3MimoPrefillScratch {
    pub fn new(cfg: &Mamba3MimoConfig) -> Self {
        Self {
            rows: 0,
            norm: Vec::new(),
            proj: Vec::new(),
            bn: Vec::new(),
            cn: Vec::new(),
            tanh_pi: Vec::new(),
            o_hm: Vec::new(),
            o_rm: Vec::new(),
            out: Vec::new(),
            heads: (0..cfg.nheads()).map(|_| HeadScratch::new(cfg)).collect(),
            mlp: GatedMlpRowsScratch::default(),
        }
    }

    fn ensure(&mut self, cfg: &Mamba3MimoConfig, rows: usize) {
        if rows <= self.rows {
            return;
        }
        let (dm, di) = (cfg.d_model, cfg.d_inner());
        self.norm.resize(rows * dm, 0.0);
        self.proj.resize(rows * cfg.in_proj_out_dim(), 0.0);
        self.bn.resize(rows * cfg.bc_width(), 0.0);
        self.cn.resize(rows * cfg.bc_width(), 0.0);
        self.tanh_pi
            .resize(rows * cfg.num_rope_angles().max(1), 0.0);
        self.o_hm.resize(rows * di, 0.0);
        self.o_rm.resize(rows * di, 0.0);
        self.out.resize(rows * dm, 0.0);
        self.rows = rows;
    }
}

type Gemm = fn(&mut [f32], &[f32], &[f32], Option<&[f32]>, usize, usize, usize);

fn gemm_for(mode: PrefillMode) -> Gemm {
    match mode {
        PrefillMode::Single => sgemm_forward,
        PrefillMode::Parallel => sgemm_forward_par,
    }
}

/// One MIMO mixer over a window of `t_len` rows of `temporal`
/// (`[t_len, d_model]`), residual included, advancing `state`.
fn mimo_layer_prefill(
    temporal: &mut [f32],
    t_len: usize,
    (lw, state): (&Mamba3MimoLayerWeights, &mut Mamba3MimoLayerState),
    s: &mut Mamba3MimoPrefillScratch,
    (cfg, mode): (&Mamba3MimoConfig, PrefillMode),
) {
    let (dm, di, p_len) = (cfg.d_model, cfg.d_inner(), cfg.headdim);
    let (na, ip, bc) = (cfg.num_rope_angles(), cfg.in_proj_out_dim(), cfg.bc_width());
    let gemm = gemm_for(mode);

    zip_rows(
        mode,
        (&mut s.norm[..t_len * dm], dm),
        (&temporal[..t_len * dm], dm),
        |d, x| row_norm(d, x, &lw.norm_weight, cfg.rms_norm_eps),
    );
    gemm(
        &mut s.proj[..t_len * ip],
        &s.norm[..t_len * dm],
        &lw.in_proj_w,
        None,
        t_len,
        dm,
        ip,
    );
    // Each token's normalised B, C and angle increments (the increment
    // rows are one wide for a model without angles, and stay unused).
    for (((proj, bn), cn), tp) in s.proj[..t_len * ip]
        .chunks_exact(ip)
        .zip(s.bn.chunks_exact_mut(bc))
        .zip(s.cn.chunks_exact_mut(bc))
        .zip(s.tanh_pi.chunks_exact_mut(na.max(1)))
    {
        prepare_token(cfg, lw, proj, (bn, cn, &mut tp[..na]));
    }

    // Each head walks the window on its own state.
    let proj = &s.proj[..t_len * ip];
    let (bn, cn, tanh_pi) = (&s.bn[..], &s.cn[..], &s.tanh_pi[..]);
    let head_state = state_heads(state, cfg);
    let run_head =
        |(h, ((st, hs), o)): (usize, ((HeadState<'_>, &mut HeadScratch), &mut [f32]))| {
            let HeadState { ssm, k, v, angle } = st;
            for (t, o_t) in o.chunks_exact_mut(p_len).enumerate() {
                let tok = token_at(cfg, proj, (bn, cn, tanh_pi), t);
                mimo_head_step(
                    (cfg, lw, h),
                    &tok,
                    HeadState {
                        ssm: &mut *ssm,
                        k: &mut *k,
                        v: &mut *v,
                        angle: &mut *angle,
                    },
                    hs,
                    o_t,
                );
            }
        };
    let work = head_state
        .into_iter()
        .zip(s.heads.iter_mut())
        .zip(s.o_hm[..t_len * di].chunks_exact_mut(t_len * p_len))
        .enumerate();
    match mode {
        PrefillMode::Single => work.for_each(run_head),
        PrefillMode::Parallel => work.collect::<Vec<_>>().into_par_iter().for_each(run_head),
    }

    // Head-major outputs to rows, out_proj, residual.
    for (h, o_h) in s.o_hm[..t_len * di].chunks_exact(t_len * p_len).enumerate() {
        for (t, o_t) in o_h.chunks_exact(p_len).enumerate() {
            s.o_rm[t * di + h * p_len..t * di + (h + 1) * p_len].copy_from_slice(o_t);
        }
    }
    gemm(
        &mut s.out[..t_len * dm],
        &s.o_rm[..t_len * di],
        &lw.out_proj_w,
        None,
        t_len,
        di,
        dm,
    );
    for (t, &o) in temporal[..t_len * dm].iter_mut().zip(&s.out[..t_len * dm]) {
        *t += o;
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
        tanh_pi: &tanh_pi[t * na..(t + 1) * na],
    }
}

/// The layer state split into one disjoint [`HeadState`] per head.
fn state_heads<'a>(
    state: &'a mut Mamba3MimoLayerState,
    cfg: &Mamba3MimoConfig,
) -> Vec<HeadState<'a>> {
    let (nh, n, p_len, r, na) = (
        cfg.nheads(),
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
        cfg.num_rope_angles(),
    );
    let mut angles: Vec<&mut [f32]> = if na == 0 {
        (0..nh).map(|_| &mut [][..]).collect()
    } else {
        state.angle_state.chunks_exact_mut(na).collect()
    };
    state
        .ssm_state
        .chunks_exact_mut(p_len * n)
        .zip(state.k_state.chunks_exact_mut(r * n))
        .zip(state.v_state.chunks_exact_mut(p_len))
        .zip(angles.drain(..))
        .map(|(((ssm, k), v), angle)| HeadState { ssm, k, v, angle })
        .collect()
}

/// A prompt window through the whole MIMO backbone, the decode's math on
/// every row. `input_flat` is `[seq_len, input_dim]`; `temporal_out`
/// (`[seq_len, d_model]`) receives every position's final-norm output, and
/// `states` carry in and out.
pub fn forward_mamba3_mimo_backbone_prefill(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba3MimoWeights,
    states: &mut [Mamba3MimoLayerState],
    scratch: &mut Mamba3MimoPrefillScratch,
    (cfg, seq_len): (&Mamba3MimoConfig, usize),
) {
    forward_mamba3_mimo_backbone_prefill_mode(
        temporal_out,
        input_flat,
        w,
        states,
        scratch,
        (cfg, seq_len, PrefillMode::Parallel),
    );
}

/// [`forward_mamba3_mimo_backbone_prefill`] with an explicit execution
/// mode; both modes give the same bits.
pub fn forward_mamba3_mimo_backbone_prefill_mode(
    temporal_out: &mut [f32],
    input_flat: &[f32],
    w: &Mamba3MimoWeights,
    states: &mut [Mamba3MimoLayerState],
    scratch: &mut Mamba3MimoPrefillScratch,
    (cfg, seq_len, mode): (&Mamba3MimoConfig, usize, PrefillMode),
) {
    let dm = cfg.d_model;
    if seq_len == 0 {
        return;
    }
    scratch.ensure(cfg, seq_len);
    let out = &mut temporal_out[..seq_len * dm];
    if w.input_proj_w.is_empty() {
        out.copy_from_slice(&input_flat[..seq_len * dm]);
    } else {
        let input_dim = input_flat.len() / seq_len;
        gemm_for(mode)(
            out,
            &input_flat[..seq_len * input_dim],
            &w.input_proj_w,
            Some(&w.input_proj_b),
            seq_len,
            input_dim,
            dm,
        );
    }
    for (l, (lw, st)) in w.layers.iter().zip(states.iter_mut()).enumerate() {
        mimo_layer_prefill(out, seq_len, (lw, st), scratch, (cfg, mode));
        if let Some(mlp) = w.mlp.get(l) {
            let eps = cfg.rms_norm_eps;
            gated_mlp_rows(
                out,
                seq_len,
                mlp,
                &mut scratch.mlp,
                gemm_for(mode),
                |dst, src| {
                    zip_rows(mode, (dst, dm), (src, dm), |d, x| {
                        rms_norm_weighted(d, x, &mlp.norm_weight, eps);
                    });
                },
            );
        }
    }
    let norm = &mut scratch.norm[..seq_len * dm];
    norm.copy_from_slice(out);
    zip_rows(mode, (out, dm), (norm, dm), |d, x| {
        row_norm(d, x, &w.norm_f_weight, cfg.rms_norm_eps);
    });
}
