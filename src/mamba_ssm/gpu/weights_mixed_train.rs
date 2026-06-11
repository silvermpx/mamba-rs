//! Mixed-precision training weights for Mamba SSM.
//!
//! Architecture (PyTorch-AMP convention):
//! - **Master weights** = f32, owned by [`GpuMambaTrainWeights`] (existing
//!   per-tensor allocation). All optimizer updates touch only the master.
//! - **Compute copies** = bf16/f16 (or f32 for `WeightDtype::F32`), owned by
//!   [`GpuMambaMixedWeights`] (existing inference structure: bulk_arena +
//!   f32_arena). All forward/backward GEMMs read these.
//! - **Sync** after every optimizer step: cast f32 master → typed compute
//!   for the bulk weights; D2D copy for the f32-stays-f32 weights
//!   (norm/conv1d/dt_proj_b/a_log/D).
//!
//! No new primitive types — reuses `GpuBuffer`, `WeightSliceDyn`, and
//! the existing `cast_f32_to_bf16` / `cast_f32_to_f16` kernels.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use crate::config::MambaConfig;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::grid_1d;
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
        // input_proj — bulk
        sync_one(
            ctx,
            &self.master.input_proj_w,
            &self.compute.input_proj_w,
            self.dtype,
        )?;
        // input_proj_b — f32 stays f32
        sync_f32(ctx, &self.master.input_proj_b, &self.compute.input_proj_b)?;

        for (mw, cw) in self.master.layers.iter().zip(&self.compute.layers) {
            // f32-stays-f32
            sync_f32(ctx, &mw.norm_weight, &cw.norm_weight)?;
            sync_f32(ctx, &mw.conv1d_weight, &cw.conv1d_weight)?;
            sync_f32(ctx, &mw.conv1d_bias, &cw.conv1d_bias)?;
            sync_f32(ctx, &mw.dt_proj_b, &cw.dt_proj_b)?;
            sync_f32(ctx, &mw.a_log, &cw.a_log)?;
            sync_f32(ctx, &mw.d_param, &cw.d_param)?;
            // bulk (cast to dtype)
            sync_one(ctx, &mw.in_proj_w, &cw.in_proj_w, self.dtype)?;
            sync_one(ctx, &mw.x_proj_w, &cw.x_proj_w, self.dtype)?;
            sync_one(ctx, &mw.dt_proj_w, &cw.dt_proj_w, self.dtype)?;
            sync_one(ctx, &mw.out_proj_w, &cw.out_proj_w, self.dtype)?;
        }

        sync_f32(ctx, &self.master.norm_f_weight, &self.compute.norm_f_weight)?;
        Ok(())
    }
}

/// Cast a single f32 master `GpuBuffer` into the matching typed compute slice.
fn sync_one(
    ctx: &GpuCtx,
    master: &crate::mamba_ssm::gpu::buffers::GpuBuffer,
    compute: &crate::mamba_ssm::gpu::buffers::WeightSliceDyn,
    dtype: WeightDtype,
) -> Result<(), String> {
    let n_elems = master.len();
    debug_assert_eq!(n_elems, compute.len_elems());

    // Empty master tensor (HF Mamba identity input_proj). Compute slot is
    // also empty by construction; skipping avoids a degenerate 0-element
    // kernel launch (which fails with CUDA_ERROR_INVALID_VALUE for an
    // empty grid).
    if n_elems == 0 {
        return Ok(());
    }

    if matches!(dtype, WeightDtype::F32) {
        // f32 → f32: D2D async copy on stream (no cast needed).
        let bytes = n_elems * 4;
        let dst_ptr = compute.ptr();
        let src_ptr = master.cached_ptr();
        let res = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                dst_ptr,
                src_ptr,
                bytes,
                ctx.stream.cu_stream(),
            )
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("sync_one f32 D2D failed: {res:?}"));
        }
        return Ok(());
    }

    // bf16/f16 cast kernel: dst = cast(src) elementwise.
    let kernel = match dtype {
        WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
        WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
        WeightDtype::F32 => unreachable!(),
    };
    let n_i32 = n_elems as i32;
    let dst_ptr = compute.ptr();
    let src_ptr = master.cached_ptr();
    let mut builder = ctx.stream.launch_builder(kernel);
    builder.arg(&dst_ptr);
    builder.arg(&src_ptr);
    builder.arg(&n_i32);
    unsafe { builder.launch(grid_1d(n_elems)) }
        .map_err(|e| format!("sync_one cast_f32_to_{dtype:?}: {e:?}"))?;
    Ok(())
}

/// f32 → f32 stream-ordered D2D copy (no dtype cast).
fn sync_f32(
    ctx: &GpuCtx,
    master: &crate::mamba_ssm::gpu::buffers::GpuBuffer,
    compute: &crate::mamba_ssm::gpu::buffers::WeightSliceDyn,
) -> Result<(), String> {
    let n_elems = master.len();
    debug_assert_eq!(n_elems, compute.len_elems());
    if n_elems == 0 {
        return Ok(());
    }
    let bytes = n_elems * 4;
    let res = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            compute.ptr(),
            master.cached_ptr(),
            bytes,
            ctx.stream.cu_stream(),
        )
    };
    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("sync_f32 D2D failed: {res:?}"));
    }
    Ok(())
}
