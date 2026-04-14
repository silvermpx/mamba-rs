//! Mixed-precision training weights for Mamba-3 (SISO).
//!
//! Architecture mirrors M1's [`crate::mamba_ssm::gpu::weights_mixed_train`]:
//! - **Master weights** = f32, owned by [`GpuMamba3Weights`] (per-tensor
//!   `GpuBuffer`). Optimizer writes here.
//! - **Compute copies** = bf16/f16 (or f32 passthrough), owned by
//!   [`GpuMamba3MixedWeights`] (existing bulk_arena + f32_arena layout).
//!   GEMMs + typed kernels read these.
//! - **Sync** after each optimizer step via [`Self::sync_master_to_compute`]:
//!   cast f32 master → typed compute for `in_proj_w`, `out_proj_w`, and
//!   `input_proj_w` (bulk); D2D copy for everything else (norm weights,
//!   biases, `d_param`, `dt_bias`, `b_bias`, `c_bias`, `norm_gate_weight`).
//!
//! Uses M1's existing `cast_f32_to_bf16` / `cast_f32_to_f16` NVRTC kernels
//! (elementwise.cu is compiled into both `MambaKernels` and `Mamba3Kernels`
//! modules, so the cast launch against `ctx.kernels` is safe from either
//! callsite).

use std::sync::Arc;

use cudarc::driver::{CudaStream, PushKernelArg};

use crate::mamba_ssm::gpu::buffers::{GpuBuffer, WeightSliceDyn};
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::grid_1d;
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
            sync_f32(ctx, &mw.dt_bias, &cw.dt_bias)?;
            sync_f32(ctx, &mw.b_norm_weight, &cw.b_norm_weight)?;
            sync_f32(ctx, &mw.c_norm_weight, &cw.c_norm_weight)?;
            sync_f32(ctx, &mw.b_bias, &cw.b_bias)?;
            sync_f32(ctx, &mw.c_bias, &cw.c_bias)?;
            sync_f32(ctx, &mw.d_param, &cw.d_param)?;
            sync_f32(ctx, &mw.norm_gate_weight, &cw.norm_gate_weight)?;
            // bulk (cast to dtype)
            sync_one(ctx, &mw.in_proj_w, &cw.in_proj_w, self.dtype)?;
            sync_one(ctx, &mw.out_proj_w, &cw.out_proj_w, self.dtype)?;
        }

        sync_f32(ctx, &self.master.norm_f_weight, &self.compute.norm_f_weight)?;
        Ok(())
    }
}

/// Cast a single f32 master `GpuBuffer` into the matching typed compute slice.
fn sync_one(
    ctx: &GpuCtx,
    master: &GpuBuffer,
    compute: &WeightSliceDyn,
    dtype: WeightDtype,
) -> Result<(), String> {
    let n_elems = master.len();
    debug_assert_eq!(n_elems, compute.len_elems());

    if matches!(dtype, WeightDtype::F32) {
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
            return Err(format!("sync_one f32 D2D failed: {res:?}"));
        }
        return Ok(());
    }

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
fn sync_f32(ctx: &GpuCtx, master: &GpuBuffer, compute: &WeightSliceDyn) -> Result<(), String> {
    let n_elems = master.len();
    debug_assert_eq!(n_elems, compute.len_elems());
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
