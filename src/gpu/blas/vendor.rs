//! The cuBLAS calls of the vendor GEMM modes. Each tail here is what its
//! namesake in the parent module runs once a vendor mode is selected, and
//! the public `_blas` entry points are the context-free cuBLAS boundaries.
//! The module is built only with the `cublas` feature; without it a context
//! never enters a vendor mode and `vendor_absent` answers the tails.

use super::super::buffers::{GpuBuffer, GradSlice};
use super::super::context::GpuCtx;
use super::super::dtype::WeightDtype;
use super::super::launch::grid_1d_wide;
use super::{TiedLmDims, TypedPtr, kernel_ints};
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUdeviceptr;
use std::ffi::{c_int, c_void};

/// The vendor tail of [`super::gpu_gemm_f32_forward_ptrs`].
pub(super) fn forward_f32_ptrs(
    ctx: &GpuCtx,
    (y, x, w): (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    bias: Option<CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    ctx.ensure_vendor_gemm("gpu_gemm_f32_forward_ptrs")?;
    let beta = if let Some(b_ptr) = bias {
        let [b_i, n_i] = kernel_ints([batch, n_out], "bias_broadcast")?;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.bias_broadcast);
        builder.arg(&y);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d_wide(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_ptrs: {e:?}"))?;
        1.0f32
    } else {
        0.0f32
    };

    let alpha: f32 = 1.0;
    let w_raw = w as *const f32;
    let x_raw = x as *const f32;
    let y_raw = y as *mut f32;

    let [n_out_c, batch_c, n_in_c] =
        cublas_ints([n_out, batch, n_in], "gpu_gemm_f32_forward_ptrs")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out_c,
            batch_c,
            n_in_c,
            &alpha as *const f32,
            w_raw,
            n_out_c,
            x_raw,
            n_in_c,
            &beta as *const f32,
            y_raw,
            n_out_c,
        )
        .map_err(|e| format!("cuBLAS sgemm_forward_ptrs failed: {e:?}"))?;
    }

    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_bi_backward_dx_raw`].
pub(super) fn backward_dx_buffers(
    ctx: &GpuCtx,
    (dx, dy): (&mut GpuBuffer, &GpuBuffer),
    w_ptr: CUdeviceptr,
    (batch, n_in, n_out): (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_vendor_gemm("gpu_gemm_bi_backward_dx_raw")?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let w_raw = w_ptr as *const f32;
    let dy_raw = dy.raw_ptr(&ctx.stream) as *const f32;
    let dx_raw = dx.raw_ptr(&ctx.stream) as *mut f32;

    let [n_in_c, batch_c, n_out_c] =
        cublas_ints([n_in, batch, n_out], "gpu_gemm_bi_backward_dx_raw")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_in_c,
            batch_c,
            n_out_c,
            &alpha as *const f32,
            w_raw,
            n_out_c,
            dy_raw,
            n_out_c,
            &beta as *const f32,
            dx_raw,
            n_in_c,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dx_raw failed: {e:?}"))?;
    }

    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_bi_backward_dw_grad`].
pub(super) fn backward_dw_buffers(
    ctx: &GpuCtx,
    dw: &GradSlice,
    (dy, x_saved): (&GpuBuffer, &GpuBuffer),
    (batch, n_in, n_out): (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_vendor_gemm("gpu_gemm_bi_backward_dw_grad")?;
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    let dy_ptr = dy.raw_ptr(&ctx.stream) as *const f32;
    let x_ptr = x_saved.raw_ptr(&ctx.stream) as *const f32;
    let dw_ptr = dw.ptr() as *mut f32;

    let [n_out_c, n_in_c, batch_c] =
        cublas_ints([n_out, n_in, batch], "gpu_gemm_bi_backward_dw_grad")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n_out_c,
            n_in_c,
            batch_c,
            &alpha as *const f32,
            dy_ptr,
            n_out_c,
            x_ptr,
            n_in_c,
            &beta as *const f32,
            dw_ptr,
            n_out_c,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dw_grad failed: {e:?}"))?;
    }
    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_f32_backward_dx_ptrs`].
pub(super) fn backward_dx_ptrs(
    ctx: &GpuCtx,
    (dx, dy, w): (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    dims: (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_vendor_gemm("gpu_gemm_f32_backward_dx_ptrs")?;
    let (batch, n_in, n_out) = dims;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let [n_in_c, batch_c, n_out_c] =
        cublas_ints([n_in, batch, n_out], "gpu_gemm_f32_backward_dx_ptrs")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_in_c,
            batch_c,
            n_out_c,
            &alpha as *const f32,
            w as *const f32,
            n_out_c,
            dy as *const f32,
            n_out_c,
            &beta as *const f32,
            dx as *mut f32,
            n_in_c,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dx_ptrs failed: {e:?}"))?;
    }
    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_f32_backward_dw_ptrs`].
pub(super) fn backward_dw_ptrs(
    ctx: &GpuCtx,
    (dw, dy, x_saved): (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    dims: (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_vendor_gemm("gpu_gemm_f32_backward_dw_ptrs")?;
    let (batch, n_in, n_out) = dims;
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let [n_out_c, n_in_c, batch_c] =
        cublas_ints([n_out, n_in, batch], "gpu_gemm_f32_backward_dw_ptrs")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n_out_c,
            n_in_c,
            batch_c,
            &alpha as *const f32,
            dy as *const f32,
            n_out_c,
            x_saved as *const f32,
            n_in_c,
            &beta as *const f32,
            dw as *mut f32,
            n_out_c,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dw_ptrs failed: {e:?}"))?;
    }
    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_bi_backward_dw_grad_typed`].
pub(super) fn backward_dw_typed(
    ctx: &GpuCtx,
    dw: &GradSlice,
    (dy, x_saved): (TypedPtr, TypedPtr),
    (batch, n_in, n_out): (usize, usize, usize),
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    let [n_out_c, n_in_c, batch_c] =
        cublas_ints([n_out, n_in, batch], "gpu_gemm_bi_backward_dw_grad_typed")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::gemm_ex(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n_out_c,
            n_in_c,
            batch_c,
            &alpha as *const f32 as *const c_void,
            dy.ptr as *const c_void,
            dy.dtype.cuda_data_type(),
            n_out_c,
            x_saved.ptr as *const c_void,
            x_saved.dtype.cuda_data_type(),
            n_in_c,
            &beta as *const f32 as *const c_void,
            dw.ptr() as *mut c_void,
            cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
            n_out_c,
            effective_compute(ctx, dy.dtype)?,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex backward dW typed failed: {e:?}"))?;
    }
    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_ex_backward_dx_typed`].
pub(super) fn backward_dx_typed(
    ctx: &GpuCtx,
    (dx, dy, w): (TypedPtr, TypedPtr, TypedPtr),
    (batch, n_in, n_out): (usize, usize, usize),
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let [n_in_c, batch_c, n_out_c] =
        cublas_ints([n_in, batch, n_out], "gpu_gemm_ex_backward_dx_typed")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::gemm_ex(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_in_c,
            batch_c,
            n_out_c,
            &alpha as *const f32 as *const c_void,
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out_c,
            dy.ptr as *const c_void,
            dy.dtype.cuda_data_type(),
            n_out_c,
            &beta as *const f32 as *const c_void,
            dx.ptr as *mut c_void,
            dx.dtype.cuda_data_type(),
            n_in_c,
            effective_compute(ctx, dy.dtype)?,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex backward dX typed failed: {e:?}"))?;
    }
    Ok(())
}

/// The vendor tail of [`super::gpu_gemm_bi_tied_lm_head_raw`].
pub(super) fn tied_lm_head_f32(
    ctx: &GpuCtx,
    (logits_ptr, temporal_ptr, embed_ptr): (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    (batch, d_model, vocab_padded): (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_vendor_gemm("gpu_gemm_bi_tied_lm_head_raw")?;
    gpu_gemm_bi_tied_lm_head_blas(
        &*ctx.cublas()?,
        logits_ptr,
        temporal_ptr,
        embed_ptr,
        batch,
        d_model,
        vocab_padded,
    )
}

/// The vendor tail of [`super::gpu_gemm_ex_tied_lm_head_raw`].
pub(super) fn tied_lm_head_typed(
    ctx: &GpuCtx,
    (logits_ptr, temporal_ptr, embed_ptr): (CUdeviceptr, CUdeviceptr, CUdeviceptr),
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    let compute = effective_compute(ctx, dtype)?;
    gpu_gemm_ex_tied_lm_head_with_compute(
        &*ctx.cublas()?,
        logits_ptr,
        temporal_ptr,
        embed_ptr,
        dtype,
        dims,
        compute,
    )
}

/// The vendor tail of [`super::gpu_gemm_typed_forward_raw`].
pub(super) fn typed_forward(
    ctx: &GpuCtx,
    (c, x, w): (TypedPtr, TypedPtr, TypedPtr),
    bias_ptr: Option<CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let beta = if let Some(b_ptr) = bias_ptr {
        let [b_i, n_i] = kernel_ints([batch, n_out], "bias_broadcast typed")?;
        let c_ptr = c.ptr;
        let bias_kernel = match c.dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => &ctx.kernels.bias_broadcast,
            d => ctx.kernels.bias_broadcast_typed.get(d),
        };
        let mut builder = ctx.stream.launch_builder(bias_kernel);
        builder.arg(&c_ptr);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d_wide(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_typed: {:?}", e))?;
        1.0f32
    } else {
        0.0f32
    };
    let alpha: f32 = 1.0;

    let [n_out_c, batch_c, n_in_c] =
        cublas_ints([n_out, batch, n_in], "gpu_gemm_typed_forward_raw")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::gemm_ex(
            ctx.cublas_handle()?,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out_c,
            batch_c,
            n_in_c,
            &alpha as *const f32 as *const c_void,
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out_c,
            x.ptr as *const c_void,
            x.dtype.cuda_data_type(),
            n_in_c,
            &beta as *const f32 as *const c_void,
            c.ptr as *mut c_void,
            c.dtype.cuda_data_type(),
            n_out_c,
            // Compute type follows the context's vendor mode for every dtype.
            effective_compute(ctx, w.dtype)?,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed failed: {e:?}"))?;
    }

    Ok(())
}

/// Effective cuBLAS compute type for a context-aware typed GEMM.
///
/// Fast uses ordinary f32 compute and Pedantic uses pedantic f32 compute for
/// every input dtype. Deterministic is rejected at this vendor boundary.
fn effective_compute(
    ctx: &GpuCtx,
    _dtype: WeightDtype,
) -> Result<cudarc::cublas::sys::cublasComputeType_t, String> {
    ctx.ensure_vendor_gemm("context-aware GemmEx")?
        .vendor_compute()
}

/// The sizes of a cuBLAS call as the 32-bit ints its API takes, or an error
/// naming the size that does not fit, where a plain cast would wrap it
/// silently.
fn cublas_ints<const N: usize>(sizes: [usize; N], call: &str) -> Result<[c_int; N], String> {
    let mut out = [0 as c_int; N];
    for (o, size) in out.iter_mut().zip(sizes) {
        *o = c_int::try_from(size)
            .map_err(|_| format!("{call}: size {size} is past the 32-bit sizes cuBLAS takes"))?;
    }
    Ok(out)
}

/// Vendor-only no-context twin of [`super::gpu_gemm_bi_tied_lm_head_raw`].
///
/// This compatibility boundary always uses cuBLAS and therefore must not be
/// called by high-level model paths that own a [`GpuCtx`].
pub fn gpu_gemm_bi_tied_lm_head_blas(
    blas: &cudarc::cublas::CudaBlas,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    d_model: usize,
    vocab_padded: usize,
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let [vocab_padded_c, batch_c, d_model_c] = cublas_ints(
        [vocab_padded, batch, d_model],
        "gpu_gemm_bi_tied_lm_head_blas",
    )?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::sgemm(
            *blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            vocab_padded_c,
            batch_c,
            d_model_c,
            &alpha as *const f32,
            embed_ptr as *const f32,
            d_model_c,
            temporal_ptr as *const f32,
            d_model_c,
            &beta as *const f32,
            logits_ptr as *mut f32,
            vocab_padded_c,
        )
        .map_err(|e| format!("cuBLAS tied sgemm failed: {e:?}"))?;
    }
    Ok(())
}

/// Vendor-only no-context twin of [`super::gpu_gemm_ex_tied_lm_head_raw`].
///
/// This compatibility boundary always uses cuBLAS and therefore must not be
/// called by high-level model paths that own a [`GpuCtx`].
pub fn gpu_gemm_ex_tied_lm_head_blas(
    blas: &cudarc::cublas::CudaBlas,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    gpu_gemm_ex_tied_lm_head_with_compute(
        blas,
        logits_ptr,
        temporal_ptr,
        embed_ptr,
        dtype,
        dims,
        dtype.compute_type(),
    )
}

fn gpu_gemm_ex_tied_lm_head_with_compute(
    blas: &cudarc::cublas::CudaBlas,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    dtype: WeightDtype,
    dims: TiedLmDims,
    compute: cudarc::cublas::sys::cublasComputeType_t,
) -> Result<(), String> {
    let TiedLmDims {
        batch,
        d_model,
        vocab_padded,
    } = dims;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let [vocab_padded_c, batch_c, d_model_c] = cublas_ints(
        [vocab_padded, batch, d_model],
        "gpu_gemm_ex_tied_lm_head_with_compute",
    )?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            vocab_padded_c,
            batch_c,
            d_model_c,
            &alpha as *const f32 as *const c_void,
            embed_ptr as *const c_void,
            dtype.cuda_data_type(),
            d_model_c,
            temporal_ptr as *const c_void,
            dtype.cuda_data_type(),
            d_model_c,
            &beta as *const f32 as *const c_void,
            logits_ptr as *mut c_void,
            cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
            vocab_padded_c,
            compute,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS tied gemm_ex failed: {e:?}"))?;
    }
    Ok(())
}

/// Vendor-only fully typed GEMM forward: `C[B,N] = A[B,K] @ W[K,N]`.
///
/// All three operand dtypes are independent (`a.dtype`, `w.dtype`, `c.dtype`).
/// This helper has no [`GpuCtx`], so it uses the dtype's fixed cuBLAS compute
/// type rather than a context-selected [`crate::gpu::GemmMode`].
///
/// Bias (if provided) is always stored f32 (Mamba convention) and is
/// broadcast into C via the typed `bias_broadcast_<c.dtype>` kernel,
/// which upcasts bias to f32, adds f32, and downcasts to `c.dtype`.
///
/// This no-context compatibility boundary always uses cuBLAS. High-level
/// model paths that own a [`GpuCtx`] must call [`super::gpu_gemm_typed_forward_raw`]
/// so the selected deterministic or vendor mode is honored.
pub fn gpu_gemm_typed_raw_no_bias(
    blas: &cudarc::cublas::CudaBlas,
    c: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let [n_out_c, batch_c, n_in_c] =
        cublas_ints([n_out, batch, n_in], "gpu_gemm_typed_raw_no_bias")?;
    unsafe {
        #[cfg(test)]
        super::vendor_gemm_test::boundary()?;
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out_c,
            batch_c,
            n_in_c,
            &alpha as *const f32 as *const c_void,
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out_c,
            x.ptr as *const c_void,
            x.dtype.cuda_data_type(),
            n_in_c,
            &beta as *const f32 as *const c_void,
            c.ptr as *mut c_void,
            c.dtype.cuda_data_type(),
            n_out_c,
            w.dtype.compute_type(), // No context is available to select a GemmMode.
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed (no-bias) failed: {e:?}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn context_aware_vendor_compute_maps_all_dtypes() {
        use crate::gpu::context::GemmMode;
        use crate::gpu::device::GpuDevice;
        use cudarc::cublas::sys::cublasComputeType_t;

        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::Bf16] {
            assert!(effective_compute(&ctx, dtype).is_err(), "{dtype:?}");
        }
        ctx.set_gemm_mode(GemmMode::CublasFast).unwrap();
        for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::Bf16] {
            assert_eq!(
                effective_compute(&ctx, dtype).unwrap(),
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                "{dtype:?}"
            );
        }
        ctx.set_gemm_mode(GemmMode::CublasPedantic).unwrap();
        for dtype in [WeightDtype::F32, WeightDtype::F16, WeightDtype::Bf16] {
            assert_eq!(
                effective_compute(&ctx, dtype).unwrap(),
                cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
                "{dtype:?}"
            );
        }
    }
}
