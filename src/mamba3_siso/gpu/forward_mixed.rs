//! Mixed-precision (bf16/f16) activation arena for Mamba-3 training.
//!
//! Mirrors the f32 [`super::mamba3_gpu::GpuMamba3LayerActs`] layout with
//! per-tensor dtype decisions matching PyTorch AMP / state-spaces/mamba
//! conventions:
//!
//! - **f32**: residual stream (residual), BPTT state (h_saved,
//!   k_prev_saved, v_prev_saved, chunk_states_saved), reduction saves
//!   (rms_vals, b_rms, c_rms, gated_rms_vals, angle_cumsum,
//!   da_cumsum_saved, scale_saved, gamma_saved, qk_dot_saved), small
//!   T-length coefficients (dt, a_val, trap, alpha, beta, gamma), plus
//!   `norm_f_input` + `norm_f_rms`.
//! - **typed**: activation I/O at kernel boundaries — post_norm, z, x,
//!   b_raw, c_raw, b_normed, c_normed, b_biased, c_biased, k, q, y,
//!   gated, dd_dt_raw, dd_a_raw, trap_raw, angles_raw, k_scaled_saved,
//!   plus `input_proj_inputs` and `input_proj_outputs`.
//!
//! This scaffolding carries only the storage + allocation. The actual
//! forward wiring (Step 8) reads the compute weights from
//! [`super::weights_mixed_train::GpuMamba3TrainMixedWeights`] and writes
//! into these acts.

use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba3_siso::config::Mamba3Config;

/// Per-layer saved activations for M3 mixed-precision backward.
///
/// Field dtype decisions are documented inline; they match the precision
/// invariants from Step 5 (residual f32, BPTT f32, reduction saves f32,
/// activation I/O typed).
pub struct GpuMamba3LayerMixedActs {
    // F1: RMSNorm
    /// f32 residual input (pre-norm) — feeds layer output residual add too.
    pub residual: GpuBuffer,
    /// f32 RMS scalar per (b,t) pair.
    pub rms_vals: GpuBuffer,
    /// typed post-norm output → in_proj input.
    pub post_norm: DtypedBuf,

    // F3: Split (after in_proj)
    pub z: DtypedBuf,
    pub x: DtypedBuf,
    pub b_raw: DtypedBuf,
    pub c_raw: DtypedBuf,
    /// typed — raw (pre-nonlinearity) dt / A / trap scores from in_proj
    /// out. `m3_split` kernel writes these; backward re-reads before
    /// activating them.
    pub dd_dt_raw: DtypedBuf,
    pub dd_a_raw: DtypedBuf,
    pub trap_raw: DtypedBuf,
    pub angles_raw: DtypedBuf,

    /// f32 — post-softplus dt (small T-length coefficient, precision-
    /// sensitive).
    pub dt: GpuBuffer,
    /// f32 — -softplus(dd_A), clamped.
    pub a_val: GpuBuffer,
    /// f32 — post-sigmoid trap.
    pub trap: GpuBuffer,

    // F4: BCNorm + bias + RoPE
    pub b_normed: DtypedBuf,
    pub c_normed: DtypedBuf,
    /// f32 — per-group RMS saves.
    pub b_rms: GpuBuffer,
    pub c_rms: GpuBuffer,
    pub b_biased: DtypedBuf,
    pub c_biased: DtypedBuf,
    /// typed — post-RoPE B (key).
    pub k: DtypedBuf,
    /// typed — post-RoPE C (query).
    pub q: DtypedBuf,

    // F5: Angle accumulation
    /// f32 — cumulative sum of angles (precision-sensitive).
    pub angle_cumsum: GpuBuffer,
    /// f32 — per-timestep coefficients (precision-sensitive, small).
    pub alpha: GpuBuffer,
    pub beta: GpuBuffer,
    pub gamma: GpuBuffer,

    // F6: SSM
    /// f32 — hidden state BPTT save [B*(T+1)*d_inner*ds].
    pub h_saved: GpuBuffer,
    /// f32 — k_state saved per t.
    pub k_prev_saved: GpuBuffer,
    /// f32 — v_state saved per t.
    pub v_prev_saved: GpuBuffer,
    /// typed — SSM output before gating.
    pub y: DtypedBuf,

    // F6 parallel: saved intermediates (skip recompute in backward)
    pub da_cumsum_saved: GpuBuffer,
    /// typed — prescaled K (typed because it feeds a typed kernel in bwd).
    pub k_scaled_saved: DtypedBuf,
    pub scale_saved: GpuBuffer,
    pub gamma_saved: GpuBuffer,
    pub qk_dot_saved: GpuBuffer,
    /// f32 — chunk states (BPTT-class recurrent state).
    pub chunk_states_saved: GpuBuffer,

    // F7: RMSNormGated
    /// f32 — one rstd per head.
    pub gated_rms_vals: GpuBuffer,
    /// typed — gated output → out_proj input.
    pub gated: DtypedBuf,
}

/// Full M3 backbone mixed-precision activations.
pub struct GpuMamba3BackboneMixedActs {
    /// typed — input to input_proj.
    pub input_proj_inputs: DtypedBuf,
    /// typed — output of input_proj.
    pub input_proj_outputs: DtypedBuf,
    pub layers: Vec<GpuMamba3LayerMixedActs>,
    /// f32 — pre-norm_f input (residual stream).
    pub norm_f_input: GpuBuffer,
    /// f32 — RMS values per sample for norm_f backward.
    pub norm_f_rms: GpuBuffer,
    pub dtype: WeightDtype,
}

impl GpuMamba3BackboneMixedActs {
    /// Allocate all save buffers. Must be called once per trainer instance;
    /// re-used forward-pass to forward-pass (zero alloc on hot path, CUDA
    /// Graph capture safe).
    pub fn new(
        stream: &Arc<CudaStream>,
        cfg: &Mamba3Config,
        batch: usize,
        seq_len: usize,
        input_dim: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ng = cfg.ngroups;
        let na = cfg.num_rope_angles();
        let hd = cfg.headdim;
        let bt = batch * seq_len;
        let nc_max = seq_len; // chunk count upper bound — mirror f32 scratch

        let layers = (0..cfg.n_layers)
            .map(|_| {
                Ok(GpuMamba3LayerMixedActs {
                    residual: GpuBuffer::zeros(stream, bt * dm)?,
                    rms_vals: GpuBuffer::zeros(stream, bt)?,
                    post_norm: DtypedBuf::zeros(stream, bt * dm, dtype)?,
                    z: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    x: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    b_raw: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    c_raw: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    dd_dt_raw: DtypedBuf::zeros(stream, bt * nh, dtype)?,
                    dd_a_raw: DtypedBuf::zeros(stream, bt * nh, dtype)?,
                    trap_raw: DtypedBuf::zeros(stream, bt * nh, dtype)?,
                    angles_raw: DtypedBuf::zeros(stream, bt * na, dtype)?,
                    dt: GpuBuffer::zeros(stream, bt * nh)?,
                    a_val: GpuBuffer::zeros(stream, bt * nh)?,
                    trap: GpuBuffer::zeros(stream, bt * nh)?,
                    b_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    c_normed: DtypedBuf::zeros(stream, bt * ng * ds, dtype)?,
                    b_rms: GpuBuffer::zeros(stream, bt * ng)?,
                    c_rms: GpuBuffer::zeros(stream, bt * ng)?,
                    b_biased: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    c_biased: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    k: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    q: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    angle_cumsum: GpuBuffer::zeros(stream, bt * nh * na)?,
                    alpha: GpuBuffer::zeros(stream, bt * nh)?,
                    beta: GpuBuffer::zeros(stream, bt * nh)?,
                    gamma: GpuBuffer::zeros(stream, bt * nh)?,
                    h_saved: GpuBuffer::zeros(stream, batch * (seq_len + 1) * di * ds)?,
                    k_prev_saved: GpuBuffer::zeros(stream, bt * nh * ds)?,
                    v_prev_saved: GpuBuffer::zeros(stream, bt * nh * hd)?,
                    y: DtypedBuf::zeros(stream, bt * di, dtype)?,
                    da_cumsum_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    k_scaled_saved: DtypedBuf::zeros(stream, bt * nh * ds, dtype)?,
                    scale_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    gamma_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    qk_dot_saved: GpuBuffer::zeros(stream, bt * nh)?,
                    chunk_states_saved: GpuBuffer::zeros(stream, batch * nc_max * nh * hd * ds)?,
                    gated_rms_vals: GpuBuffer::zeros(stream, bt * nh)?,
                    gated: DtypedBuf::zeros(stream, bt * di, dtype)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let s = Self {
            input_proj_inputs: DtypedBuf::zeros(stream, bt * input_dim, dtype)?,
            input_proj_outputs: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * dm)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
            dtype,
        };

        // Race-fix invariant (a950648): callers may immediately upload from
        // host via default-stream HtoD; sync first.
        stream
            .synchronize()
            .map_err(|e| format!("sync after m3 mixed acts alloc: {e:?}"))?;

        Ok(s)
    }
}
