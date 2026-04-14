//! cuBLAS SGEMM wrappers for GPU training.
//!
//! All matrices are row-major in our code. cuBLAS is column-major.
//!
//! The standard trick: for row-major C = A @ B, call cuBLAS with:
//!   C^T = B^T @ A^T  (in cuBLAS column-major convention)
//!   gemm(N, N, n_out, batch, n_in, 1.0, W, n_out, X, n_in, beta, Y, n_out)

use super::buffers::{GpuBuffer, GradSlice};
use super::context::GpuCtx;
use super::dtype::WeightDtype;
use super::launch::grid_1d;
use cudarc::driver::PushKernelArg;
use std::ffi::{c_int, c_void};

/// Batched linear forward: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
/// `dims` = `(batch, n_in, n_out)`.
pub fn gpu_sgemm_forward_raw(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let beta = if let Some(b_ptr) = bias_ptr {
        let b_i = batch as i32;
        let n_i = n_out as i32;
        let y_ptr = y.cached_ptr();
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.bias_broadcast);
        builder.arg(&y_ptr); // raw ptr — no SyncOnDrop, CUDA Graph safe
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_raw: {:?}", e))?;
        1.0f32
    } else {
        0.0f32
    };

    let alpha: f32 = 1.0;
    let w_raw = w_ptr as *const f32;
    let x_raw = x.raw_ptr(&ctx.stream) as *const f32;
    let y_raw = y.raw_ptr(&ctx.stream) as *mut f32;

    unsafe {
        cudarc::cublas::result::sgemm(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out as c_int,
            batch as c_int,
            n_in as c_int,
            &alpha as *const f32,
            w_raw,
            n_out as c_int,
            x_raw,
            n_in as c_int,
            &beta as *const f32,
            y_raw,
            n_out as c_int,
        )
        .map_err(|e| format!("cuBLAS sgemm_forward_raw failed: {e:?}"))?;
    }

    Ok(())
}

/// Input gradient: `dX[B,K] = dY[B,N] @ W^T[N,K]`.
pub fn gpu_sgemm_backward_dx_raw(
    ctx: &GpuCtx,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let w_raw = w_ptr as *const f32;
    let dy_raw = dy.raw_ptr(&ctx.stream) as *const f32;
    let dx_raw = dx.raw_ptr(&ctx.stream) as *mut f32;

    unsafe {
        cudarc::cublas::result::sgemm(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_in as c_int,
            batch as c_int,
            n_out as c_int,
            &alpha as *const f32,
            w_raw,
            n_out as c_int,
            dy_raw,
            n_out as c_int,
            &beta as *const f32,
            dx_raw,
            n_in as c_int,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dx_raw failed: {e:?}"))?;
    }

    Ok(())
}

/// Weight gradient: `dW[K,N] += X^T[K,B] @ dY[B,N]`.
pub fn gpu_sgemm_backward_dw_grad(
    ctx: &GpuCtx,
    dw: &GradSlice,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    let dy_ptr = dy.raw_ptr(&ctx.stream) as *const f32;
    let x_ptr = x_saved.raw_ptr(&ctx.stream) as *const f32;
    let dw_ptr = dw.ptr() as *mut f32;

    unsafe {
        cudarc::cublas::result::sgemm(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n_out as c_int,
            n_in as c_int,
            batch as c_int,
            &alpha as *const f32,
            dy_ptr,
            n_out as c_int,
            x_ptr,
            n_in as c_int,
            &beta as *const f32,
            dw_ptr,
            n_out as c_int,
        )
        .map_err(|e| format!("cuBLAS sgemm_backward_dw_grad failed: {e:?}"))?;
    }
    Ok(())
}

/// Full backward: dW (accumulated), dX (overwritten), db (accumulated).
///
/// `grads` = `(dw, db)`. `dims` = `(batch, n_in, n_out)`.
pub fn gpu_sgemm_backward_grad_raw(
    ctx: &GpuCtx,
    dx: &mut GpuBuffer,
    grads: (&GradSlice, Option<&GradSlice>),
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (dw, db) = grads;
    let (batch, n_in, n_out) = dims;
    gpu_sgemm_backward_dw_grad(ctx, dw, dy, x_saved, batch, n_in, n_out)?;
    gpu_sgemm_backward_dx_raw(ctx, dx, dy, w_ptr, batch, n_in, n_out)?;

    if let Some(db) = db {
        let b_i = batch as i32;
        let n_i = n_out as i32;
        let db_ptr = db.ptr();
        let dy_ptr = dy.cached_ptr();
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.colsum_accumulate);
        builder.arg(&db_ptr);
        builder.arg(&dy_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(n_out)) }
            .map_err(|e| format!("colsum_accumulate_grad_raw: {:?}", e))?;
    }

    Ok(())
}

/// Dispatch SGEMM or GEMMex based on weight dtype.
///
/// Activations are always f32; this is the single entry point used by the
/// unified inference pipeline to support f32 / bf16 / fp16 weight storage
/// with identical activation dataflow.
pub fn gpu_gemm_forward_dispatch(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    w_dtype: WeightDtype,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    match w_dtype {
        WeightDtype::F32 => gpu_sgemm_forward_raw(ctx, y, x, w_ptr, bias_ptr, dims),
        WeightDtype::F16 | WeightDtype::Bf16 => gpu_gemm_ex_forward_raw(
            ctx,
            y,
            x.cached_ptr(),
            WeightDtype::F32,
            w_ptr,
            w_dtype,
            bias_ptr,
            dims,
        ),
    }
}

/// Mixed-precision GEMM forward: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
///
/// Inputs X and W are in `w_dtype` (f32/f16/bf16). Output Y is always f32.
/// Compute type is f32 (CUBLAS_COMPUTE_32F) — f32 accumulation regardless of input dtype.
///
/// For `WeightDtype::F32`, this is mathematically identical to `gpu_sgemm_forward_raw`
/// (callers should prefer sgemm path for f32 to avoid gemmEx overhead).
///
/// `dims` = `(batch, n_in, n_out)`. `x_ptr` and `w_ptr` are raw device pointers (CUDA
/// Graph safe). `x_dtype` typically matches `w_dtype` for Mamba inference.
///
/// Bias (if provided) is always f32 (Mamba convention: biases stay f32 regardless of
/// weight dtype). It is added via a separate broadcast kernel on the f32 output.
pub fn gpu_gemm_ex_forward_raw(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x_ptr: cudarc::driver::sys::CUdeviceptr,
    x_dtype: WeightDtype,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    w_dtype: WeightDtype,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let beta = if let Some(b_ptr) = bias_ptr {
        let b_i = batch as i32;
        let n_i = n_out as i32;
        let y_ptr = y.cached_ptr();
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.bias_broadcast);
        builder.arg(&y_ptr);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_ex: {:?}", e))?;
        1.0f32
    } else {
        0.0f32
    };
    let alpha: f32 = 1.0;

    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out as c_int,
            batch as c_int,
            n_in as c_int,
            &alpha as *const f32 as *const c_void,
            w_ptr as *const c_void,
            w_dtype.cuda_data_type(),
            n_out as c_int,
            x_ptr as *const c_void,
            x_dtype.cuda_data_type(),
            n_in as c_int,
            &beta as *const f32 as *const c_void,
            y.cached_ptr() as *mut c_void,
            cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
            n_out as c_int,
            w_dtype.compute_type(),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex failed: {e:?}"))?;
    }

    Ok(())
}
