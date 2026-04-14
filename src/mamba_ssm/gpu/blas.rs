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
/// Activations are always f32; when weights are bf16/f16, activations are
/// downcast to the weight dtype on-the-fly (via cast kernel) into a scratch
/// buffer, then GemmEx runs with matching input dtypes. Output Y stays f32.
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
        WeightDtype::F16 | WeightDtype::Bf16 => {
            // cuBLAS requires A and B to have matching dtype. Downcast x f32 -> w_dtype
            // into the ctx's reusable half-staging buffer.
            let (batch, n_in, _) = dims;
            let half_bytes = batch * n_in * w_dtype.size_bytes();
            ctx.ensure_half_staging(half_bytes)?;
            let half_ptr = ctx.half_staging_ptr();
            let n = (batch * n_in) as i32;
            let src_ptr = x.cached_ptr();
            let kernel = match w_dtype {
                WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
                WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
                _ => unreachable!(),
            };
            let mut builder = ctx.stream.launch_builder(kernel);
            builder.arg(&half_ptr);
            builder.arg(&src_ptr);
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(batch * n_in)) }
                .map_err(|e| format!("cast_f32_to_half: {e:?}"))?;
            gpu_gemm_ex_forward_raw(
                ctx,
                y,
                TypedPtr {
                    ptr: half_ptr,
                    dtype: w_dtype,
                },
                TypedPtr {
                    ptr: w_ptr,
                    dtype: w_dtype,
                },
                bias_ptr,
                dims,
            )
        }
    }
}

/// Tied lm_head: logits[B, V] = temporal[B, D] @ embed^T[D, V].
/// All three buffers row-major. `embed[V, D]` reused from input embedding (no copy).
///
/// Single GEMM via OP_T on embed + OP_N on temporal.
/// Derivation: row-major Y[B,V] = X[B,D]·E^T[D,V] ⇔
///             col-major Y^T[V,B] = E[V,D] · X^T[D,B]
///   `embed` row-major [V,D] = col-major [D,V], OP_T → logical [V,D].
///   `temporal` row-major [B,D] = col-major [D,B], OP_N → logical [D,B].
///   Output col-major [V,B] = row-major [B,V].
pub fn gpu_sgemm_tied_lm_head_raw(
    ctx: &GpuCtx,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    d_model: usize,
    vocab_padded: usize,
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        cudarc::cublas::result::sgemm(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            vocab_padded as c_int,
            batch as c_int,
            d_model as c_int,
            &alpha as *const f32,
            embed_ptr as *const f32,
            d_model as c_int,
            temporal_ptr as *const f32,
            d_model as c_int,
            &beta as *const f32,
            logits_ptr as *mut f32,
            vocab_padded as c_int,
        )
        .map_err(|e| format!("cuBLAS tied sgemm failed: {e:?}"))?;
    }
    Ok(())
}

/// Typed device pointer: raw ptr + element dtype.
#[derive(Copy, Clone)]
pub struct TypedPtr {
    pub ptr: cudarc::driver::sys::CUdeviceptr,
    pub dtype: WeightDtype,
}

/// Tied LM head dims: `(batch, d_model, vocab_padded)`.
#[derive(Copy, Clone)]
pub struct TiedLmDims {
    pub batch: usize,
    pub d_model: usize,
    pub vocab_padded: usize,
}

/// Half-precision twin of `gpu_sgemm_tied_lm_head_raw` for bf16/f16 embed.
/// `temporal_ptr` input activations must already be in `dtype` (not f32).
pub fn gpu_gemm_ex_tied_lm_head_raw(
    ctx: &GpuCtx,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    let TiedLmDims {
        batch,
        d_model,
        vocab_padded,
    } = dims;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            vocab_padded as c_int,
            batch as c_int,
            d_model as c_int,
            &alpha as *const f32 as *const c_void,
            embed_ptr as *const c_void,
            dtype.cuda_data_type(),
            d_model as c_int,
            temporal_ptr as *const c_void,
            dtype.cuda_data_type(),
            d_model as c_int,
            &beta as *const f32 as *const c_void,
            logits_ptr as *mut c_void,
            cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
            vocab_padded as c_int,
            dtype.compute_type(),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS tied gemm_ex failed: {e:?}"))?;
    }
    Ok(())
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
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    gpu_gemm_typed_forward_raw(
        ctx,
        TypedPtr {
            ptr: y.cached_ptr(),
            dtype: WeightDtype::F32,
        },
        x,
        w,
        bias_ptr,
        dims,
    )
}

/// Fully typed GEMM forward: `C[B,N] = A[B,K] @ W[K,N] + bias[N]`.
///
/// All three operand dtypes are independent (`a.dtype`, `w.dtype`, `c.dtype`).
/// Compute type is f32 (CUBLAS_COMPUTE_32F) regardless of I/O dtypes —
/// tensor-core accumulation stays f32 for numerical stability.
///
/// Bias (if provided) is always stored f32 (Mamba convention) and is
/// broadcast into C via the typed `bias_broadcast_<c.dtype>` kernel,
/// which upcasts bias to f32, adds f32, and downcasts to `c.dtype`.
///
/// Used for end-to-end bf16/f16 activation paths where GEMM writes
/// directly to half-precision output without a staging f32 copy.
/// No-context twin of `gpu_gemm_typed_forward_raw` for callers that don't
/// hold a `GpuCtx` (e.g., the Mamba-3 engine has its own blas/kernels and
/// never passes a bias through this helper). Takes only the cuBLAS handle.
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
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out as c_int,
            batch as c_int,
            n_in as c_int,
            &alpha as *const f32 as *const c_void,
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out as c_int,
            x.ptr as *const c_void,
            x.dtype.cuda_data_type(),
            n_in as c_int,
            &beta as *const f32 as *const c_void,
            c.ptr as *mut c_void,
            c.dtype.cuda_data_type(),
            n_out as c_int,
            w.dtype.compute_type(),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed (no-bias) failed: {e:?}"))?;
    }
    Ok(())
}

pub fn gpu_gemm_typed_forward_raw(
    ctx: &GpuCtx,
    c: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let beta = if let Some(b_ptr) = bias_ptr {
        let b_i = batch as i32;
        let n_i = n_out as i32;
        let c_ptr = c.ptr;
        let bias_kernel = match c.dtype {
            WeightDtype::F32 => &ctx.kernels.bias_broadcast,
            d => ctx.kernels.bias_broadcast_typed.get(d),
        };
        let mut builder = ctx.stream.launch_builder(bias_kernel);
        builder.arg(&c_ptr);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_typed: {:?}", e))?;
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
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out as c_int,
            x.ptr as *const c_void,
            x.dtype.cuda_data_type(),
            n_in as c_int,
            &beta as *const f32 as *const c_void,
            c.ptr as *mut c_void,
            c.dtype.cuda_data_type(),
            n_out as c_int,
            // Compute type derives from W dtype (f32 for F32 weights, f32 for
            // bf16/f16 — all our paths use CUBLAS_COMPUTE_32F accumulate).
            w.dtype.compute_type(),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed failed: {e:?}"))?;
    }

    Ok(())
}
