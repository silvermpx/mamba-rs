//! The vendor tails of the parent module in a build without the `cublas`
//! feature. A context of such a build never enters a vendor GEMM mode, so
//! each tail answers as a deterministic context reaching a vendor boundary
//! does, and names the feature a vendor mode would need.

use super::super::buffers::{GpuBuffer, GradSlice};
use super::super::context::GpuCtx;
use super::super::dtype::WeightDtype;
use super::{TiedLmDims, TypedPtr};
use cudarc::driver::sys::CUdeviceptr;

fn absent(ctx: &GpuCtx, label: &str) -> Result<(), String> {
    let mode = ctx.ensure_vendor_gemm(label)?;
    Err(format!(
        "{label}: the {} GEMM mode needs mamba-rs built with the `cublas` feature",
        mode.as_str()
    ))
}

pub(super) fn forward_f32_ptrs(
    ctx: &GpuCtx,
    _pointers: (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    _bias: Option<CUdeviceptr>,
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_f32_forward_ptrs")
}

pub(super) fn backward_dx_buffers(
    ctx: &GpuCtx,
    _buffers: (&mut GpuBuffer, &GpuBuffer),
    _w_ptr: CUdeviceptr,
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_bi_backward_dx_raw")
}

pub(super) fn backward_dw_buffers(
    ctx: &GpuCtx,
    _dw: &GradSlice,
    _buffers: (&GpuBuffer, &GpuBuffer),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_bi_backward_dw_grad")
}

pub(super) fn backward_dx_ptrs(
    ctx: &GpuCtx,
    _pointers: (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_f32_backward_dx_ptrs")
}

pub(super) fn backward_dw_ptrs(
    ctx: &GpuCtx,
    _pointers: (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_f32_backward_dw_ptrs")
}

pub(super) fn backward_dw_typed(
    ctx: &GpuCtx,
    _dw: &GradSlice,
    _operands: (TypedPtr, TypedPtr),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_bi_backward_dw_grad_typed")
}

pub(super) fn backward_dx_typed(
    ctx: &GpuCtx,
    _operands: (TypedPtr, TypedPtr, TypedPtr),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_ex_backward_dx_typed")
}

pub(super) fn tied_lm_head_f32(
    ctx: &GpuCtx,
    _pointers: (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_bi_tied_lm_head_raw")
}

pub(super) fn tied_lm_head_typed(
    ctx: &GpuCtx,
    _pointers: (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    _dtype: WeightDtype,
    _dims: TiedLmDims,
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_ex_tied_lm_head_raw")
}

pub(super) fn typed_forward(
    ctx: &GpuCtx,
    _operands: (TypedPtr, TypedPtr, TypedPtr),
    _bias_ptr: Option<CUdeviceptr>,
    _dims: (usize, usize, usize),
) -> Result<(), String> {
    absent(ctx, "gpu_gemm_typed_forward_raw")
}
