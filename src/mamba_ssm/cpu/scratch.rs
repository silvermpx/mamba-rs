//! Scratch buffers for O2 batched SGEMM Mamba forward/backward pipelines.
//!
//! Allocated once per training session, reused every gradient step.
//! The 7-phase forward and 8-phase backward restructure per-timestep loops
//! into batched SGEMM calls across all timesteps, requiring flat `[T * dim]`
//! intermediate buffers.

use crate::ops::dims::MambaDims;

/// Scratch buffers for 7-phase optimized forward (O2).
///
/// Each buffer is `[T * dim]` where T = `dims.seq_len`.
pub struct PhaseScratch {
    /// Post-RmsNorm outputs: `[T * d_model]`.
    pub post_norm_flat: Vec<f32>,
    /// In-projection outputs (x + gate concatenated): `[T * 2*d_inner]`.
    pub proj_flat: Vec<f32>,
    /// Gate branch after SiLU: `[T * d_inner]`.
    pub gate_silu_flat: Vec<f32>,
    /// Gated SSM output (y * gate_silu): `[T * d_inner]`.
    pub gated_flat: Vec<f32>,
    /// Out-projection outputs: `[T * d_model]`.
    pub out_flat: Vec<f32>,
}

impl PhaseScratch {
    /// Allocate zero-filled scratch buffers for the given dimensions.
    pub fn zeros(dims: &MambaDims) -> Self {
        let t = dims.seq_len;
        Self {
            post_norm_flat: vec![0.0; t * dims.d_model],
            proj_flat: vec![0.0; t * 2 * dims.d_inner],
            gate_silu_flat: vec![0.0; t * dims.d_inner],
            gated_flat: vec![0.0; t * dims.d_inner],
            out_flat: vec![0.0; t * dims.d_model],
        }
    }
}

/// Scratch buffers for 8-phase optimized backward (O2).
///
/// Gradient buffers for each intermediate value in the backward pipeline.
/// Sequential SSM backward (phase B4) still processes per-timestep,
/// but all SGEMM backward calls are batched across T.
pub struct BackwardPhaseScratch {
    /// d(gated output): `[T * d_inner]`.
    pub d_gated_flat: Vec<f32>,
    /// d(SSM output y): `[T * d_inner]`.
    pub d_y_flat: Vec<f32>,
    /// d(gate after SiLU): `[T * d_inner]`.
    pub d_gate_flat: Vec<f32>,
    /// d(delta after softplus): `[T * d_inner]`.
    pub d_delta_flat: Vec<f32>,
    /// d(delta raw before softplus): `[T * d_inner]`.
    pub d_delta_raw_flat: Vec<f32>,
    /// d(u / x_branch input to SSM): `[T * d_inner]`.
    pub d_u_flat: Vec<f32>,
    /// d(u) accumulated from x_proj backward: `[T * d_inner]`.
    pub d_u_xproj_flat: Vec<f32>,
    /// d(xdbl = [delta_raw, B, C] concatenated): `[T * xdbl_dim]`.
    pub d_xdbl_flat: Vec<f32>,
    /// d(conv1d output): `[T * d_inner]`.
    pub d_conv_out_flat: Vec<f32>,
    /// d(x_branch before conv): `[T * d_inner]`.
    pub d_x_branch_flat: Vec<f32>,
    /// d(in_proj output = [x, gate]): `[T * 2*d_inner]`.
    pub d_proj_flat: Vec<f32>,
    /// d(post-norm): `[T * d_model]`.
    pub d_norm_flat: Vec<f32>,
    /// d(pre-norm / residual path): `[T * d_model]`.
    pub d_pre_norm_flat: Vec<f32>,
    /// d(dt_proj input): `[T * dt_rank]`.
    pub d_dt_input_flat: Vec<f32>,
    /// SSM hidden state gradient: `[d_inner * d_state]`.
    pub d_h: Vec<f32>,
    /// Conv1d carry gradient for BPTT: `[d_inner * (d_conv - 1)]`.
    pub d_conv_carry: Vec<f32>,
    /// Saved dt_proj input for batched backward: `[T * dt_rank]`.
    pub xdbl_dt_buf: Vec<f32>,
    /// Saved u values for batched x_proj backward: `[T * d_inner]`.
    pub u_buf: Vec<f32>,
    /// Saved gated values for batched out_proj backward: `[T * d_inner]`.
    pub gated_buf: Vec<f32>,
    /// Saved post_norm values for batched in_proj backward: `[T * d_model]`.
    pub post_norm_buf: Vec<f32>,
    /// Discarded dx for batched input projection backward: `[T * mamba_input_dim]`.
    pub d_input_proj_scratch: Vec<f32>,
    /// Local accumulator for norm_f weight gradient: `[d_model]`.
    pub d_norm_f_weight_local: Vec<f32>,
}

impl BackwardPhaseScratch {
    /// Allocate zero-filled scratch buffers for the given dimensions.
    pub fn zeros(dims: &MambaDims) -> Self {
        let t = dims.seq_len;
        let di = dims.d_inner;
        let dm = dims.d_model;
        Self {
            d_gated_flat: vec![0.0; t * di],
            d_y_flat: vec![0.0; t * di],
            d_gate_flat: vec![0.0; t * di],
            d_delta_flat: vec![0.0; t * di],
            d_delta_raw_flat: vec![0.0; t * di],
            d_u_flat: vec![0.0; t * di],
            d_u_xproj_flat: vec![0.0; t * di],
            d_xdbl_flat: vec![0.0; t * dims.xdbl_dim],
            d_conv_out_flat: vec![0.0; t * di],
            d_x_branch_flat: vec![0.0; t * di],
            d_proj_flat: vec![0.0; t * 2 * di],
            d_norm_flat: vec![0.0; t * dm],
            d_pre_norm_flat: vec![0.0; t * dm],
            d_dt_input_flat: vec![0.0; t * dims.dt_rank],
            d_h: vec![0.0; di * dims.d_state],
            d_conv_carry: vec![0.0; di * (dims.d_conv - 1)],
            xdbl_dt_buf: vec![0.0; t * dims.dt_rank],
            u_buf: vec![0.0; t * di],
            gated_buf: vec![0.0; t * di],
            post_norm_buf: vec![0.0; t * dm],
            d_input_proj_scratch: vec![0.0; t * dims.mamba_input_dim.max(dm)],
            d_norm_f_weight_local: vec![0.0; dm],
        }
    }
}
