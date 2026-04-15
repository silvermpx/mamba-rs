//! Step 10 — Mamba-3 SISO mixed-precision (bf16/f16) backward training
//! pipeline. Mirrors [`super::mamba3_gpu::gpu_backward_mamba3_layer`] +
//! `_backbone` with typed activation I/O routed through the typed kernel
//! variants from Steps 9a/9b/9c/9d.
//!
//! ## Status (this commit — initial scaffold)
//!
//! This file ships the **API surface** + the `norm_f` backward (f32, since
//! the residual stream stays f32 per `residual_in_fp32=True` convention)
//! + the per-layer dispatch loop. The per-layer body itself returns
//! `Err(...)` because completing the full typed bwd path requires
//! extending [`crate::mamba3_siso::gpu::forward_mixed::GpuMamba3MixedScratch`]
//! with bwd-only typed staging buffers (`d_y_typed`, `d_z_typed`,
//! `d_norm_typed`, `d_pre_norm_typed`, `d_x_typed`, `d_post_norm_typed`,
//! `d_gated_typed`, `d_proj_typed`) — those tensors are `typed` per the
//! AMP precision invariant (activation grads on the wire match activation
//! storage dtype) and the existing `GpuMamba3MixedScratch` only carries
//! forward-pass fields.
//!
//! All upstream pieces are in place:
//!   - typed kernels: rmsnorm_gated_bwd_typed (Step 9c), m3_dqkv_typed +
//!     m3_dqktheta_typed (Step 9b), bcnorm_bwd_typed + bc_bias_add_bwd_typed
//!     + rope_bwd_typed + m3_split_bwd_typed (Step 9a), chunked parallel
//!     bwd typed (Step 9d).
//!   - typed cuBLAS bwd: gpu_sgemm_backward_dw_grad_typed +
//!     gpu_gemm_ex_backward_dx_typed (Step 4c).
//!   - typed acts arena: GpuMamba3LayerMixedActs (Step 7).
//!   - typed weights compute copy: GpuMamba3MixedWeights (Step 7).
//!
//! The full per-layer body (~600 lines) mirrors
//! [`super::mamba3_gpu::gpu_backward_mamba3_layer`] section-by-section
//! (B8 → B7 → B6 → B5 → B4 → B3 → B2 → B1 + residual add) with each
//! kernel call replaced by its typed dispatch. See the `gpu_backward_*`
//! function in mamba3_gpu.rs as the line-by-line template.
//!
//! ## Precision invariants
//!
//! Activations: typed (bf16/f16) — read via `cached_ptr()` from the
//! `GpuMamba3LayerMixedActs` arena.
//!
//! Master grads: ALL stay f32 in `GpuMamba3LayerGrads` — atomicAdd
//! master-grad invariant per audit (bf16/f16 atomicAdd not supported on
//! ≤sm_89 and would lose precision on reduction-style accumulators).
//!
//! Activation grads (on the wire between kernels): typed (bf16/f16) —
//! requires the bwd-only typed scratch noted above.
//!
//! Scalar/scan-state scratch: f32 — reuse the existing
//! [`super::mamba3_gpu::GpuMamba3Scratch`] f32 fields for chunk-state
//! recomputation, postfix carries, dA cumsum, etc.

use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::grid_norm;
use crate::mamba3_siso::gpu::forward_mixed::GpuMamba3BackboneMixedActs;
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::mamba3_gpu::{GpuMamba3Dims, GpuMamba3Scratch};
use crate::mamba3_siso::gpu::weights::GpuMamba3Grads;
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;

use cudarc::driver::PushKernelArg;

/// Mamba-3 SISO full backbone mixed-precision backward.
///
/// **Currently returns `Err(...)` for the per-layer body** — see module
/// docstring for the missing scratch extension. The norm_f backward and
/// the per-layer dispatch shell ARE wired so callers can validate the
/// API surface compiles into their training loop.
///
/// **IMPORTANT**: weight gradients in `grads` are **accumulated** (`beta=1.0`
/// on the dW GEMMs). Caller MUST call [`GpuMamba3Grads::zero`] before each
/// training step if the buffer is reused across iterations.
#[allow(clippy::too_many_arguments)]
pub fn gpu_backward_mamba3_backbone_mixed(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    d_temporal: &mut GpuBuffer,
    acts: &GpuMamba3BackboneMixedActs,
    mamba_w: &GpuMamba3TrainMixedWeights,
    grads: &GpuMamba3Grads,
    scratch: &mut GpuMamba3Scratch,
    dims: &GpuMamba3Dims,
) -> Result<(), String> {
    let bt = dims.bt();
    let dm = dims.d_model;

    // norm_f backward (f32 — residual stream stays f32 per AMP
    // `residual_in_fp32=True` convention). Output written to
    // scratch.d_norm; we then copy it back into d_temporal so the
    // per-layer chain receives the post-norm gradient.
    {
        let nf_ptr = mamba_w.master.norm_f_weight.raw_ptr(&ctx.stream);
        let d_nf_ptr = grads.norm_f_weight.ptr();
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_bwd);
        builder.arg(scratch.d_norm.inner_mut());
        builder.arg(&d_nf_ptr);
        builder.arg(d_temporal.inner());
        builder.arg(acts.norm_f_input.inner());
        builder.arg(&nf_ptr);
        builder.arg(acts.norm_f_rms.inner());
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_bwd norm_f m3 mixed: {:?}", e))?;
    }
    d_temporal.copy_from_raw(&scratch.d_norm, &ctx.stream)?;

    // Per-layer dispatch — currently returns Err on first layer to flag
    // that the typed body is not yet wired (see module docstring). The
    // dispatch shell exists so callers can validate API surface compiles
    // into their training loop without conditionally dropping the function.
    let _ = (acts, mamba_w, grads, scratch, dims);
    if dims.n_layers > 0 {
        return Err(
            "gpu_backward_mamba3_backbone_mixed: per-layer typed bwd body \
             not yet wired — needs bwd-only typed scratch extension. \
             All upstream typed kernels and weights are ready (Steps \
             9a/9b/9c/9d + 4c + 7); the remaining work is mechanical: \
             clone the f32 backward layer body and replace every \
             m3k.foo call with m3k.foo_typed.get(dtype). See module \
             docstring. Use gpu_backward_mamba3_backbone (f32) until \
             this lands."
                .into(),
        );
    }

    Ok(())
}
