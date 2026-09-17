//! Mixed-precision training weights for Mamba SSM.
//!
//! Architecture (PyTorch-AMP convention):
//! - **Master weights** = f32, owned by [`GpuMambaTrainWeights`] (existing
//!   per-tensor allocation). All optimizer updates touch only the master.
//! - **Compute copies** = bf16/f16 (or f32 for `WeightDtype::F32` and `Tf32`), owned by
//!   [`GpuMambaMixedWeights`] (existing inference structure: bulk_arena +
//!   f32_arena). All forward/backward GEMMs read these.
//! - **Sync** after every optimizer step: cast f32 master → typed compute
//!   for the bulk weights; D2D copy for the f32-stays-f32 weights
//!   (norm/conv1d/dt_proj_b/a_log/D).
//!
//! No new primitive types — reuses `GpuBuffer`, `WeightSliceDyn`, and
//! the existing `cast_f32_to_bf16` / `cast_f32_to_f16` kernels.

use std::sync::Arc;

use crate::config::MambaConfig;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::weights::{GpuMambaMixedWeights, GpuMambaTrainWeights};
use crate::weights::MambaWeights;

/// Mixed-precision training weights.
///
/// `master` is the source of truth (f32, optimizer-updated).
/// `compute` is the typed shadow used by every forward/backward GEMM.
/// Call [`Self::sync_master_to_compute`] after each optimizer step.
pub struct GpuMambaTrainMixedWeights {
    /// f32 source-of-truth weights (per-tensor `GpuBuffer`s).
    pub master: GpuMambaTrainWeights,
    /// bf16/f16 cast copy used by GEMMs (or f32 view when `dtype == F32`).
    pub compute: GpuMambaMixedWeights,
    /// Element dtype of `compute.bulk_arena`.
    pub dtype: WeightDtype,
}

impl GpuMambaTrainMixedWeights {
    /// Allocate master + compute copies and upload from CPU weights.
    pub fn from_cpu(
        stream: &Arc<cudarc::driver::CudaStream>,
        cpu: &MambaWeights,
        cfg: &MambaConfig,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let master = GpuMambaTrainWeights::from_cpu(stream, cpu)?;
        let compute = GpuMambaMixedWeights::from_cpu(stream, cpu, cfg, dtype)?;
        Ok(Self {
            master,
            compute,
            dtype,
        })
    }

    /// Cast every f32 master tensor into its typed compute slot.
    /// For `WeightDtype::F32` this is a D2D copy. For bf16/f16 this fires
    /// the per-tensor `cast_f32_to_bf16` / `cast_f32_to_f16` kernel.
    ///
    /// Call once per optimizer step, after `optimizer.step()` writes to the
    /// master weights, before the next forward pass reads from `compute`.
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
