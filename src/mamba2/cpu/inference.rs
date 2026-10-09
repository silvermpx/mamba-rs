//! Mamba-2 decode: one token through the backbone, mutating the state.
//!
//! ```text
//! temporal -> RMSNorm -> in_proj -> split [z0 | x0 | z | xBC | dt]
//!   -> conv over the last d_conv inputs + SiLU -> x | B | C
//!   -> per head: dt, dA, scan along the head's channels, + D x
//!   -> [silu(z0) x0 | gated RMSNorm(y, z)] -> out_proj + residual
//!   [-> gated MLP] -> ... -> final RMSNorm
//! ```

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::{Mamba2LayerWeights, Mamba2Weights};
use super::core::{conv_hist_push, conv_silu_token, gate_token, rms_row, ssm_token};
use crate::mlp::cpu::{GatedMlpStepScratch, gated_mlp_step};
use crate::ops::blas::matvec_forward;

/// Reusable buffers for [`mamba2_step`].
pub struct Mamba2StepScratch {
    residual: Vec<f32>,
    norm_buf: Vec<f32>,
    proj: Vec<f32>,
    xbc: Vec<f32>,
    y: Vec<f32>,
    gated: Vec<f32>,
    mlp: GatedMlpStepScratch,
}

impl Mamba2StepScratch {
    pub fn new(cfg: &Mamba2Config) -> Self {
        let dm = cfg.d_model;
        Self {
            residual: vec![0.0; dm],
            norm_buf: vec![0.0; dm],
            proj: vec![0.0; cfg.in_proj_out_dim()],
            xbc: vec![0.0; cfg.conv_dim()],
            y: vec![0.0; cfg.ssm_dim()],
            gated: vec![0.0; cfg.d_inner()],
            mlp: GatedMlpStepScratch::new(dm, cfg.mlp_hidden()),
        }
    }
}

/// `y = x W (+ b)` for one row, the bias added by the GEMM.
fn project(y: &mut [f32], x: &[f32], w: &[f32], b: &[f32]) {
    let bias = if b.is_empty() { None } else { Some(b) };
    matvec_forward(y, x, w, bias, x.len(), y.len());
}

/// One token through one Mamba-2 mixer; `temporal` (`[d_model]`) is both
/// the input and the output, residual included.
pub fn mamba2_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    lw: &Mamba2LayerWeights,
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    let dm = cfg.d_model;
    let (d_mlp, d_ssm, cd) = (cfg.d_mlp(), cfg.ssm_dim(), cfg.conv_dim());
    scratch.residual.copy_from_slice(&temporal[..dm]);
    rms_row(
        &mut scratch.norm_buf,
        &temporal[..dm],
        &lw.norm_weight,
        cfg.rms_norm_eps,
    );
    project(
        &mut scratch.proj,
        &scratch.norm_buf,
        &lw.in_proj_w,
        &lw.in_proj_b,
    );
    let xbc_raw = &scratch.proj[2 * d_mlp + d_ssm..2 * d_mlp + d_ssm + cd];
    let dt_raw = &scratch.proj[2 * d_mlp + d_ssm + cd..];
    conv_silu_token(&mut scratch.xbc, xbc_raw, &state.conv_hist, lw);
    conv_hist_push(&mut state.conv_hist, xbc_raw);
    ssm_token(
        (&mut state.ssm, &mut scratch.y),
        &scratch.xbc,
        dt_raw,
        lw,
        cfg,
    );
    gate_token(&mut scratch.gated, &scratch.y, &scratch.proj, lw, cfg);
    project(
        &mut temporal[..dm],
        &scratch.gated,
        &lw.out_proj_w,
        &lw.out_proj_b,
    );
    for (t, &r) in temporal[..dm].iter_mut().zip(&scratch.residual) {
        *t += r;
    }
}

/// One token through the whole Mamba-2 backbone: input projection, every
/// layer (mixer, then the MLP when the model has one), final norm.
/// `input` is `[input_dim]`; `temporal` (`[d_model]`) receives the output.
pub fn mamba2_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    weights: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    let dm = cfg.d_model;
    if weights.input_proj_w.is_empty() {
        temporal[..dm].copy_from_slice(&input[..dm]);
    } else {
        project(
            &mut temporal[..dm],
            input,
            &weights.input_proj_w,
            &weights.input_proj_b,
        );
    }
    for (l, (lw, st)) in weights.layers.iter().zip(states.iter_mut()).enumerate() {
        mamba2_layer_step(temporal, scratch, lw, st, cfg);
        if let Some(mlp) = weights.mlp.get(l) {
            gated_mlp_step(&mut temporal[..dm], mlp, &mut scratch.mlp, cfg.rms_norm_eps);
        }
    }
    scratch.norm_buf.copy_from_slice(&temporal[..dm]);
    rms_row(
        &mut temporal[..dm],
        &scratch.norm_buf,
        &weights.norm_f_weight,
        cfg.rms_norm_eps,
    );
}
