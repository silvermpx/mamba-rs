//! Mamba-3 MIMO decode: one token through the backbone, mutating the
//! recurrent state.
//!
//! ```text
//! temporal -> RMSNorm -> in_proj -> split [z | x | B | C | dd_dt | dd_A | trap | angles]
//!   -> B/C RMSNorm per (rank, group) row
//!   -> per head: discretise, advance angles, bias + half-split rotate R keys and queries,
//!      u_r = mimo_x_r * x, state rows, y_r += D u_r, gate and mix the ranks through mimo_o
//!   -> out_proj + residual [-> gated MLP] -> ... -> final RMSNorm
//! ```

use super::super::config::Mamba3MimoConfig;
use super::super::state::Mamba3MimoLayerState;
use super::super::weights::{Mamba3MimoLayerWeights, Mamba3MimoWeights};
use super::core::{HeadScratch, HeadState, mimo_head_step, prepare_token, row_norm};
use crate::mlp::cpu::{GatedMlpStepScratch, gated_mlp_step};
use crate::ops::blas::matvec_forward;

/// Reusable buffers for [`mamba3_mimo_step`].
pub struct Mamba3MimoStepScratch {
    residual: Vec<f32>,
    norm_buf: Vec<f32>,
    proj: Vec<f32>,
    /// Normalised B and C, `[R * G * N]` each.
    bn: Vec<f32>,
    cn: Vec<f32>,
    /// The gated output of every head, `[d_inner]`.
    o: Vec<f32>,
    tanh_pi: Vec<f32>,
    head: HeadScratch,
    mlp: GatedMlpStepScratch,
}

impl Mamba3MimoStepScratch {
    pub fn new(cfg: &Mamba3MimoConfig) -> Self {
        let dm = cfg.d_model;
        Self {
            residual: vec![0.0; dm],
            norm_buf: vec![0.0; dm],
            proj: vec![0.0; cfg.in_proj_out_dim()],
            bn: vec![0.0; cfg.bc_width()],
            cn: vec![0.0; cfg.bc_width()],
            o: vec![0.0; cfg.d_inner()],
            tanh_pi: vec![0.0; cfg.num_rope_angles()],
            head: HeadScratch::new(cfg),
            mlp: GatedMlpStepScratch::new(dm, cfg.mlp_hidden()),
        }
    }
}

/// One token through one MIMO mixer; `temporal` (`[d_model]`) is both the
/// input and the output, residual included.
pub fn mamba3_mimo_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba3MimoStepScratch,
    lw: &Mamba3MimoLayerWeights,
    state: &mut Mamba3MimoLayerState,
    cfg: &Mamba3MimoConfig,
) {
    let (dm, di, n, p_len, r) = (
        cfg.d_model,
        cfg.d_inner(),
        cfg.d_state,
        cfg.headdim,
        cfg.mimo_rank,
    );
    let na = cfg.num_rope_angles();

    scratch.residual.copy_from_slice(&temporal[..dm]);
    row_norm(
        &mut scratch.norm_buf,
        &temporal[..dm],
        &lw.norm_weight,
        cfg.rms_norm_eps,
    );
    matvec_forward(
        &mut scratch.proj,
        &scratch.norm_buf,
        &lw.in_proj_w,
        None,
        dm,
        cfg.in_proj_out_dim(),
    );
    let tok = prepare_token(
        cfg,
        lw,
        &scratch.proj,
        (&mut scratch.bn, &mut scratch.cn, &mut scratch.tanh_pi),
    );
    for h in 0..cfg.nheads() {
        mimo_head_step(
            (cfg, lw, h),
            &tok,
            HeadState {
                ssm: &mut state.ssm_state[h * p_len * n..(h + 1) * p_len * n],
                k: &mut state.k_state[h * r * n..(h + 1) * r * n],
                v: &mut state.v_state[h * p_len..(h + 1) * p_len],
                angle: &mut state.angle_state[h * na..(h + 1) * na],
            },
            &mut scratch.head,
            &mut scratch.o[h * p_len..(h + 1) * p_len],
        );
    }

    matvec_forward(
        &mut temporal[..dm],
        &scratch.o,
        &lw.out_proj_w,
        None,
        di,
        dm,
    );
    for (t, &res) in temporal[..dm].iter_mut().zip(&scratch.residual) {
        *t += res;
    }
}

/// One token through the whole MIMO backbone: input projection, every
/// layer (mixer, then the MLP when the model has one) and the final norm.
/// `input` is `[input_dim]`; `temporal` (`[d_model]`) receives the output.
pub fn mamba3_mimo_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba3MimoStepScratch,
    weights: &Mamba3MimoWeights,
    states: &mut [Mamba3MimoLayerState],
    cfg: &Mamba3MimoConfig,
) {
    let dm = cfg.d_model;
    if weights.input_proj_w.is_empty() {
        temporal[..dm].copy_from_slice(&input[..dm]);
    } else {
        matvec_forward(
            &mut temporal[..dm],
            input,
            &weights.input_proj_w,
            Some(&weights.input_proj_b),
            input.len(),
            dm,
        );
    }
    for (l, (lw, st)) in weights.layers.iter().zip(states.iter_mut()).enumerate() {
        mamba3_mimo_layer_step(temporal, scratch, lw, st, cfg);
        if let Some(mlp) = weights.mlp.get(l) {
            gated_mlp_step(&mut temporal[..dm], mlp, &mut scratch.mlp, cfg.rms_norm_eps);
        }
    }
    scratch.norm_buf.copy_from_slice(&temporal[..dm]);
    row_norm(
        &mut temporal[..dm],
        &scratch.norm_buf,
        &weights.norm_f_weight,
        cfg.rms_norm_eps,
    );
}
