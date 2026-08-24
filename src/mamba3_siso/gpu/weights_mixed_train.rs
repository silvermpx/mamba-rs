//! Mixed-precision training weights for Mamba-3 (SISO).
//!
//! Architecture mirrors M1's [`crate::mamba_ssm::gpu::weights_mixed_train`]:
//! - **Master weights** = f32, owned by [`GpuMamba3Weights`] (per-tensor
//!   `GpuBuffer`). Optimizer writes here.
//! - **Compute copies** = bf16/f16 (or f32 passthrough), owned by
//!   [`GpuMamba3MixedWeights`] (existing bulk_arena + f32_arena layout).
//!   GEMMs + typed kernels read these.
//! - **Sync** after each optimizer step via `sync_master_to_compute`:
//!   cast f32 master → typed compute for `in_proj_w`, `out_proj_w`, and
//!   `input_proj_w` (bulk); D2D copy for everything else (norm weights,
//!   biases, `d_param`, `dt_bias`, `b_bias`, `c_bias`, `norm_gate_weight`).
//!
//! Uses M1's existing `cast_f32_to_bf16` / `cast_f32_to_f16` NVRTC kernels
//! (elementwise.cu is compiled into both `MambaKernels` and `Mamba3Kernels`
//! modules, so the cast launch against `ctx.kernels` is safe from either
//! callsite).

use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::weights::{GpuMamba3MixedWeights, GpuMamba3Weights};
use crate::mamba3_siso::weights::Mamba3Weights;

/// Mixed-precision training weights for M3.
///
/// `master` is the f32 source-of-truth (optimizer updates).
/// `compute` is the typed shadow that every forward/backward kernel reads.
/// Call [`Self::sync_master_to_compute`] after each optimizer step.
pub struct GpuMamba3TrainMixedWeights {
    pub master: GpuMamba3Weights,
    pub compute: GpuMamba3MixedWeights,
    pub dtype: WeightDtype,
}

impl GpuMamba3TrainMixedWeights {
    /// Allocate master (f32) + compute (typed) copies and upload from CPU.
    pub fn from_cpu(
        stream: &Arc<CudaStream>,
        cpu: &Mamba3Weights,
        cfg: &Mamba3Config,
        input_dim: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let master = GpuMamba3Weights::from_cpu(stream, cpu, cfg, input_dim)?;
        let compute = GpuMamba3MixedWeights::from_cpu(stream, cpu, dtype)?;
        Ok(Self {
            master,
            compute,
            dtype,
        })
    }

    /// Cast every f32 master tensor into its typed compute slot.
    /// f32 mode = D2D copy. bf16/f16 = elementwise cast kernel.
    /// Must be called after every optimizer step, before the next forward.
    pub fn sync_master_to_compute(&self, ctx: &GpuCtx) -> Result<(), String> {
        // Nothing to do: EVERY compute shadow - the typed bulk tensors and
        // the f32-stays-f32 ones alike - is written by the fused AdamW
        // kernel in the same launch that updates its master, so the
        // per-tensor copy walk that used to live here is gone. `a_log`
        // has no compute shadow at all (the forward and backward ride
        // a_neg_all, recomputed from the master every step). Kept as a
        // no-op seam: if a future weight sharing ever needs a shadow the
        // optimizer does not write, it belongs here.
        let _ = ctx;
        Ok(())
    }
}
