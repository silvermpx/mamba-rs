//! Mixed-precision (bf16/f16) training activations for Mamba-1.
//!
//! Mirrors [`super::forward::GpuMambaBackboneActs`] structure but stores
//! activations in `DtypedBuf` (bf16/f16/f32) where appropriate. Tensors that
//! must stay f32 for numerical stability (recurrent state, reduction stats,
//! residual stream) keep their `GpuBuffer` (f32) type — matching PyTorch AMP
//! convention and state-spaces/mamba's `scan_t = float2` invariant.
//!
//! Precision rules:
//! - **f32 always**: `residual` (residual stream), `rms_vals`/`norm_f_rms`
//!   (reduction stats), `conv_states` (recurrent state), `h_saved` (BPTT
//!   carry), `da_exp` (state-derived).
//! - **typed (bf16/f16/f32)**: `post_norm`, `gate_pre_silu`, `gate_post_silu`,
//!   `post_conv`, `u`, `xdbl`, `delta_raw`, `delta`, `y`, `gated`,
//!   `input_proj_inputs`, `input_proj_outputs`, `norm_f_input` — these are
//!   GEMM I/O or elementwise activations where bf16 storage is safe.
//!
//! Reference: `docs/training-mixed-precision.md` for the full design.

use std::sync::Arc;

use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::GpuMambaDims;

/// Mixed-precision saved activations for one Mamba layer.
pub struct GpuMambaLayerMixedActs {
    /// f32 — pre-norm input saved for residual add `[B*T*d_model]`.
    pub residual: GpuBuffer,
    /// f32 — RMS scalar per (b,t) `[B*T]`.
    pub rms_vals: GpuBuffer,
    /// typed — post-norm output `[B*T*d_model]`.
    pub post_norm: DtypedBuf,
    /// typed — gate branch before SiLU `[B*T*d_inner]`.
    pub gate_pre_silu: DtypedBuf,
    /// typed — gate branch after SiLU `[B*T*d_inner]`.
    pub gate_post_silu: DtypedBuf,
    /// f32 — conv1d state saved per step `[B*T*d_inner*d_conv]` (recurrent).
    pub conv_states: GpuBuffer,
    /// typed — pre-SiLU conv output `[B*T*d_inner]`.
    pub post_conv: DtypedBuf,
    /// typed — post-SiLU conv output, SSM input `[B*T*d_inner]`.
    pub u: DtypedBuf,
    /// typed — x_proj output (dt_raw, B, C concatenated) `[B*T*xdbl_dim]`.
    pub xdbl: DtypedBuf,
    /// typed — delta before softplus `[B*T*d_inner]`.
    pub delta_raw: DtypedBuf,
    /// typed — delta after softplus `[B*T*d_inner]`.
    pub delta: DtypedBuf,
    /// f32 — hidden state saved BEFORE each step (T+1 entries)
    /// `[B*(T+1)*d_inner*d_state]`. STAYS f32 (BPTT recurrence).
    pub h_saved: GpuBuffer,
    /// f32 — discretization `exp(delta * A)` `[B*T*d_inner*d_state]`.
    pub da_exp: GpuBuffer,
    /// typed — SSM output before gating `[B*T*d_inner]`.
    pub y: DtypedBuf,
    /// typed — gated output `y * gate_silu` `[B*T*d_inner]`.
    pub gated: DtypedBuf,
}

/// Full Mamba backbone mixed-precision activations.
pub struct GpuMambaBackboneMixedActs {
    /// typed — input to input_proj `[B*T*mamba_input_dim]`.
    pub input_proj_inputs: DtypedBuf,
    /// typed — output of input_proj `[B*T*d_model]`.
    pub input_proj_outputs: DtypedBuf,
    /// Per-layer activations.
    pub layers: Vec<GpuMambaLayerMixedActs>,
    /// f32 — pre-norm_f input for backward `[B*T*d_model]` (residual stream).
    pub norm_f_input: GpuBuffer,
    /// f32 — RMS values per sample for norm_f backward `[B*T]`.
    pub norm_f_rms: GpuBuffer,
    /// Element dtype of all `DtypedBuf` fields.
    pub dtype: WeightDtype,
}

impl GpuMambaBackboneMixedActs {
    /// Allocate all activation buffers, sized for `dims`. Fires async memsets
    /// on `stream`. Caller MUST `stream.synchronize()` before any default-stream
    /// HtoD upload into one of these buffers (see commit a950648 for the race
    /// class this prevents).
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let batch = dims.batch;
        let seq_len = dims.seq_len;
        let d_model = dims.d_model;
        let d_inner = dims.d_inner;
        let d_state = dims.d_state;
        let d_conv = dims.d_conv;
        let dt_rank = dims.dt_rank;
        let n_layers = dims.n_layers;
        let mamba_input_dim = dims.mamba_input_dim;
        let bt = batch * seq_len;
        let xdbl_dim = dt_rank + 2 * d_state;

        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaLayerMixedActs {
                    // f32 — recurrent / reduction
                    residual: GpuBuffer::zeros(stream, bt * d_model)?,
                    rms_vals: GpuBuffer::zeros(stream, bt)?,
                    conv_states: GpuBuffer::zeros(stream, bt * d_inner * d_conv)?,
                    h_saved: GpuBuffer::zeros(stream, batch * (seq_len + 1) * d_inner * d_state)?,
                    da_exp: GpuBuffer::zeros(stream, bt * d_inner * d_state)?,
                    // typed — GEMM I/O / elementwise
                    post_norm: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
                    gate_pre_silu: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    gate_post_silu: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    post_conv: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    u: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    xdbl: DtypedBuf::zeros(stream, bt * xdbl_dim, dtype)?,
                    delta_raw: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    delta: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    y: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    gated: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let s = Self {
            input_proj_inputs: DtypedBuf::zeros(stream, bt * mamba_input_dim, dtype)?,
            input_proj_outputs: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
            dtype,
        };

        // Enforce the race-fix invariant from commit a950648: every later
        // HtoD path that targets these buffers must observe the zeros first.
        stream
            .synchronize()
            .map_err(|e| format!("sync after mixed acts alloc: {e:?}"))?;

        Ok(s)
    }
}
