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
use super::gemm_bi_triad::{PhysicalArgumentRange, prepare_physical_observer};
use super::kernel_identity::{
    ModuleKind, NoPhysicalObserver, PhysicalConversionArguments, PhysicalCudaLaunchError,
    PhysicalLaunchKind, PhysicalLaunchObservation, PhysicalLaunchObserver, PolicyDtype,
    PreparedPhysicalCaptureManifest, RecordedPhysicalTrace, RecordingPhysicalObserver,
    ResolvedGemmOp, ResolvedPhysicalKernelLaunch, enqueue_prepared_physical_launch,
    enqueue_with_physical_observation, finish_recording_physical_observer,
    resolve_physical_launch_observation,
};
use super::launch::grid_1d;
use cudarc::driver::{CudaFunction, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg};
use std::ffi::{c_int, c_void};

/// Effective cuBLAS compute type for a typed GEMM: the PEDANTIC default,
/// or the opt-in non-PEDANTIC tensor-core mode (`ctx.set_fast_gemm` /
/// MAMBA_RS_FAST_GEMM). Separate numeric contract - see GpuCtx::fast_gemm.
fn effective_compute(
    ctx: &GpuCtx,
    dtype: super::dtype::WeightDtype,
) -> cudarc::cublas::sys::cublasComputeType_t {
    if ctx.fast_gemm() {
        dtype.compute_type_fast()
    } else {
        dtype.compute_type()
    }
}

pub fn gpu_gemm_bi_forward_raw(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;

    // Opt-in deterministic path, family-selected (`set_bi_gemm_family`).
    if ctx.batch_invariant() {
        return match ctx.bi_gemm_family() {
            // The triad dispatcher; bias is fused into its kernels (no
            // separate broadcast launch).
            super::context::BiGemmFamily::Triad => super::gemm_bi_triad::launch_cached_f32_forward(
                ctx,
                y,
                x,
                w_ptr,
                bias_ptr.unwrap_or(0),
                (batch, n_in, n_out),
            ),
            // Fixed-tile, invariant by construction. f32 operands take the
            // CUDA-core FMA instantiation.
            super::context::BiGemmFamily::Fixed => {
                let y_ptr = {
                    use cudarc::driver::DevicePtr;
                    let (p, _r) = y.inner().device_ptr(&ctx.stream);
                    p
                };
                let x_ptr = {
                    use cudarc::driver::DevicePtr;
                    let (p, _r) = x.inner().device_ptr(&ctx.stream);
                    p
                };
                gemm_bi_forward_raw(
                    ctx,
                    TypedPtr {
                        ptr: y_ptr,
                        dtype: WeightDtype::F32,
                    },
                    TypedPtr {
                        ptr: x_ptr,
                        dtype: WeightDtype::F32,
                    },
                    TypedPtr {
                        ptr: w_ptr,
                        dtype: WeightDtype::F32,
                    },
                    bias_ptr,
                    (batch, n_in, n_out),
                )
            }
        };
    }

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

/// Same as [`gpu_gemm_bi_forward_raw`] but the input is a raw device pointer
/// (e.g. the backbone's temporal buffer during decode — avoids a per-token
/// D2H + H2D round trip just to re-wrap an on-device tensor).
pub fn gpu_gemm_bi_forward_ptr(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x_ptr: cudarc::driver::sys::CUdeviceptr,
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
        builder.arg(&y_ptr);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_ptr: {:?}", e))?;
        1.0f32
    } else {
        0.0f32
    };

    let alpha: f32 = 1.0;
    let w_raw = w_ptr as *const f32;
    let x_raw = x_ptr as *const f32;
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
        .map_err(|e| format!("cuBLAS sgemm_forward_ptr failed: {e:?}"))?;
    }

    Ok(())
}

/// Input gradient: `dX[B,K] = dY[B,N] @ W^T[N,K]`.
pub fn gpu_gemm_bi_backward_dx_raw(
    ctx: &GpuCtx,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    if ctx.batch_invariant() {
        return match ctx.bi_gemm_family() {
            super::context::BiGemmFamily::Triad => {
                super::gemm_bi_triad::launch_cached_f32_backward_dx(
                    ctx,
                    dx,
                    dy,
                    w_ptr,
                    (batch, n_in, n_out),
                )
            }
            super::context::BiGemmFamily::Fixed => super::gemm_bi_triad::gemm_bi_backward_dx(
                &ctx.stream,
                &ctx.kernels,
                dx,
                dy,
                w_ptr,
                (batch, n_in, n_out),
            ),
        };
    }
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
pub fn gpu_gemm_bi_backward_dw_grad(
    ctx: &GpuCtx,
    dw: &GradSlice,
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    if ctx.batch_invariant() {
        return match ctx.bi_gemm_family() {
            super::context::BiGemmFamily::Triad => {
                super::gemm_bi_triad::launch_cached_f32_backward_dw(
                    ctx,
                    dw.ptr(),
                    dy,
                    x_saved,
                    (batch, n_in, n_out),
                )
            }
            super::context::BiGemmFamily::Fixed => super::gemm_bi_triad::gemm_bi_backward_dw(
                &ctx.stream,
                &ctx.kernels,
                dw.ptr(),
                dy,
                x_saved,
                (batch, n_in, n_out),
            ),
        };
    }
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

/// Typed dW backward GEMM. Matches the f32
/// [`gpu_gemm_bi_backward_dw_grad`] math with bf16/f16 inputs and f32 master
/// gradient accumulator.
///
/// Math: `dW[K=n_in, N=n_out] += X^T @ dY` where X is `[batch, n_in]` and
/// dY is `[batch, n_out]`. Mirrors NVIDIA Apex `mlp_bp` weight grad pattern:
/// - A = dY (typed, OP_N), `lda=n_out`
/// - B = X  (typed, OP_T), `ldb=n_in`
/// - C = dW (f32 master, accumulator), `ldc=n_out`
/// - alpha=1.0, beta=1.0 (f32 scalars; PEDANTIC requires host f32, not f64)
/// - compute = `CUBLAS_COMPUTE_32F_PEDANTIC` (true f32 accumulate; we
///   intentionally diverge from PyTorch's TF32 default — see commit 61325b3
///   for the 1.4b regression that motivated PEDANTIC).
///
/// `dy.dtype` and `x.dtype` MUST match (cuBLAS GemmEx requires same A/B
/// element type). Output buffer `dw` is always f32 (master grad).
pub fn gpu_gemm_bi_backward_dw_grad_typed(
    ctx: &GpuCtx,
    dw: &GradSlice,
    dy: TypedPtr,
    x_saved: TypedPtr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    debug_assert_eq!(
        dy.dtype, x_saved.dtype,
        "cuBLAS GemmEx requires A.dtype == B.dtype"
    );
    // Hard assert (not debug_assert): every bench and serving build is
    // --release with no [profile] override, so a debug_assert here would
    // let the exact condition it names happen silently in the build that
    // claims determinism.
    assert!(
        dy.dtype != WeightDtype::F32 || !ctx.batch_invariant(),
        "f32 TypedPtr under the batch-invariant flag would silently take \
         non-deterministic cuBLAS — use gpu_gemm_bi_backward_dw_grad instead"
    );
    if ctx.batch_invariant() && dy.dtype != WeightDtype::F32 {
        return gemm_bi_backward_dw_typed(ctx, dw.ptr(), dy, x_saved, (batch, n_in, n_out));
    }
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            n_out as c_int,
            n_in as c_int,
            batch as c_int,
            &alpha as *const f32 as *const c_void,
            dy.ptr as *const c_void,
            dy.dtype.cuda_data_type(),
            n_out as c_int,
            x_saved.ptr as *const c_void,
            x_saved.dtype.cuda_data_type(),
            n_in as c_int,
            &beta as *const f32 as *const c_void,
            dw.ptr() as *mut c_void,
            cudarc::cublas::sys::cudaDataType::CUDA_R_32F,
            n_out as c_int,
            effective_compute(ctx, dy.dtype),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex backward dW typed failed: {e:?}"))?;
    }
    Ok(())
}

/// Typed dX backward GEMM. Typed twin of
/// [`gpu_gemm_bi_backward_dx_raw`]: `dX[B,K] = dY[B,N] @ W^T[N,K]` with
/// bf16/f16 A,B,C and f32 master accumulate (no TC, PEDANTIC).
///
/// Layout mirrors the f32 twin exactly (OP_T on W, OP_N on dY,
/// m=n_in, n=batch, k=n_out, lda=n_out, ldb=n_out, ldc=n_in,
/// alpha=1.0, beta=0.0 — dX is overwritten, not accumulated).
///
/// `dy.dtype`, `w.dtype`, and `dx.dtype` MUST match (cuBLAS GemmEx
/// requires homogeneous A/B/C dtype for this compute mode). Pass all
/// three via `TypedPtr`. Compute type: `CUBLAS_COMPUTE_32F_PEDANTIC`
/// (true f32 accumulate, same reasoning as the dW twin — see commit
/// 61325b3 for the 1.4b regression that motivates disabling TF32).
pub fn gpu_gemm_ex_backward_dx_typed(
    ctx: &GpuCtx,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    debug_assert_eq!(
        dy.dtype, w.dtype,
        "cuBLAS GemmEx requires A.dtype == B.dtype"
    );
    debug_assert_eq!(
        dx.dtype, dy.dtype,
        "typed dX GEMM: dx.dtype must match dy/w for PEDANTIC path"
    );
    // Hard assert - same determinism rationale as the dW twin above.
    assert!(
        dx.dtype != WeightDtype::F32 || !ctx.batch_invariant(),
        "f32 TypedPtr under the batch-invariant flag would silently take \
         non-deterministic cuBLAS — use gpu_gemm_bi_backward_dx_raw instead"
    );
    if ctx.batch_invariant() && dx.dtype != WeightDtype::F32 {
        return gemm_bi_backward_dx_typed(ctx, dx, dy, w, (batch, n_in, n_out));
    }
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *ctx.blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_in as c_int,
            batch as c_int,
            n_out as c_int,
            &alpha as *const f32 as *const c_void,
            w.ptr as *const c_void,
            w.dtype.cuda_data_type(),
            n_out as c_int,
            dy.ptr as *const c_void,
            dy.dtype.cuda_data_type(),
            n_out as c_int,
            &beta as *const f32 as *const c_void,
            dx.ptr as *mut c_void,
            dx.dtype.cuda_data_type(),
            n_in as c_int,
            effective_compute(ctx, dy.dtype),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex backward dX typed failed: {e:?}"))?;
    }
    Ok(())
}

/// Elementwise upcast of a typed (bf16/f16) device buffer into f32 (exact —
/// 16-bit grids embed in f32 without rounding).
#[derive(Clone, Copy)]
struct HalfPhysicalContext {
    op: ResolvedGemmOp,
    dtype: WeightDtype,
    dims: (usize, usize, usize),
}

impl HalfPhysicalContext {
    fn policy_dtype(self) -> Result<PolicyDtype, String> {
        match self.dtype {
            WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
            WeightDtype::F16 => Ok(PolicyDtype::F16),
            WeightDtype::F32 => Err("half physical context does not accept f32".into()),
        }
    }

    fn strides(self) -> (usize, usize, usize) {
        let (_, k, n) = self.dims;
        match self.op {
            ResolvedGemmOp::Nn => (k, n, n),
            ResolvedGemmOp::Tn => (k, n, n),
            ResolvedGemmOp::Nt => (n, n, k),
        }
    }
}

#[inline(always)]
fn validate_half_physical_policy<O: PhysicalLaunchObserver>(ctx: &GpuCtx) -> Result<(), String> {
    if O::ENABLED
        && (!ctx.batch_invariant() || ctx.bi_gemm_family() != super::context::BiGemmFamily::Triad)
    {
        return Err(
            "recording a half launch requires the live batch-invariant Triad policy".into(),
        );
    }
    Ok(())
}

fn conversion_observation(
    physical: HalfPhysicalContext,
    kind: PhysicalLaunchKind,
    element_count: usize,
    source: cudarc::driver::sys::CUdeviceptr,
    destination: cudarc::driver::sys::CUdeviceptr,
) -> Result<PhysicalLaunchObservation, String> {
    let logical_dtype = physical.policy_dtype()?;
    let element_count_u64 = u64::try_from(element_count)
        .map_err(|_| "half conversion element count exceeds u64::MAX".to_string())?;
    let half_bytes = u64::try_from(physical.dtype.size_bytes())
        .ok()
        .and_then(|width| element_count_u64.checked_mul(width))
        .ok_or_else(|| "half conversion span overflows u64".to_string())?;
    let f32_bytes = element_count_u64
        .checked_mul(4)
        .ok_or_else(|| "f32 conversion span overflows u64".to_string())?;
    let (source_bytes, destination_bytes) = match kind {
        PhysicalLaunchKind::InputUpcast => (half_bytes, f32_bytes),
        PhysicalLaunchKind::OutputDowncast => (f32_bytes, half_bytes),
        PhysicalLaunchKind::Gemm => {
            return Err("conversion launch cannot use GEMM kind".into());
        }
    };
    Ok(PhysicalLaunchObservation::conversion(
        kind,
        physical.op,
        logical_dtype,
        physical.dims,
        physical.strides(),
        element_count_u64,
        PhysicalConversionArguments::new(source, source_bytes, destination, destination_bytes),
    ))
}

fn bi_upcast_to_f32<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    src: TypedPtr,
    dst_ptr: cudarc::driver::sys::CUdeviceptr,
    n: usize,
    physical: HalfPhysicalContext,
    observer: &mut O,
) -> Result<(), String> {
    let kernel = match src.dtype {
        WeightDtype::Bf16 => &ctx.kernels.cast_bf16_to_f32,
        WeightDtype::F16 => &ctx.kernels.cast_f16_to_f32,
        WeightDtype::F32 => return Err("bi_upcast_to_f32: src is already f32".into()),
    };
    let n_i = i32::try_from(n)
        .map_err(|_| format!("invalid GEMM dimensions: element count {n} exceeds i32::MAX"))?;
    let src_ptr = src.ptr;
    let mut b = ctx.stream.launch_builder(kernel);
    b.arg(&dst_ptr);
    b.arg(&src_ptr);
    b.arg(&n_i);
    let config = grid_1d(n);
    let observation = if O::ENABLED {
        Some(conversion_observation(
            physical,
            PhysicalLaunchKind::InputUpcast,
            n,
            src_ptr,
            dst_ptr,
        )?)
    } else {
        None
    };
    unsafe { enqueue_with_physical_observation(observer, &mut b, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("bi_upcast_to_f32")))
}

/// Elementwise RNE downcast of an f32 device buffer into a typed (bf16/f16)
/// buffer — the single rounding the typed-GEMM contract allows.
fn bi_downcast_from_f32<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    dst: TypedPtr,
    src_ptr: cudarc::driver::sys::CUdeviceptr,
    n: usize,
    physical: HalfPhysicalContext,
    observer: &mut O,
) -> Result<(), String> {
    let kernel = match dst.dtype {
        WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
        WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
        WeightDtype::F32 => return Err("bi_downcast_from_f32: dst is already f32".into()),
    };
    let n_i = i32::try_from(n)
        .map_err(|_| format!("invalid GEMM dimensions: element count {n} exceeds i32::MAX"))?;
    let dst_ptr = dst.ptr;
    let mut b = ctx.stream.launch_builder(kernel);
    b.arg(&dst_ptr);
    b.arg(&src_ptr);
    b.arg(&n_i);
    let config = grid_1d(n);
    let observation = if O::ENABLED {
        Some(conversion_observation(
            physical,
            PhysicalLaunchKind::OutputDowncast,
            n,
            src_ptr,
            dst_ptr,
        )?)
    } else {
        None
    };
    unsafe { enqueue_with_physical_observation(observer, &mut b, config, observation) }
        .map_err(|error| error.with_driver_context(format_args!("bi_downcast_from_f32")))
}

/// Batch-invariant typed NN forward with FULL shape coverage:
/// `Y[B,N] = X[B,K] @ W[K,N] (+ bias)` for homogeneous bf16/f16 operands.
/// Covered typed buckets run natively; every other shape routes through
/// "upcast inputs → f32 gemm_bi → RNE downcast Y", which produces the
/// SAME bits as a native typed kernel (typed kernels
/// keep f32 accumulation and the f32 twin's FMA chain, with exactly one
/// RNE downcast at the store). `dims` = `(batch, n_in, n_out)`.
/// Qualified CC12.0 BF16/F16 cells may use cached SM120 TMA/MMA16; all other
/// cells retain the existing tensor-core and exact fallback policy.
pub fn gemm_bi_forward_typed(
    ctx: &GpuCtx,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: cudarc::driver::sys::CUdeviceptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut observer = NoPhysicalObserver;
    gemm_bi_forward_typed_in(ctx, y, x, w, bias_ptr, dims, &mut observer).map(drop)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::mamba_ssm::gpu) enum HalfPolicyBranchSeal {
    Native(super::gemm_bi_triad::HalfNativeBranchSeal),
    Sm120(super::gemm_bi_triad::Sm120AutoBranchSeal),
    Sm100(super::gemm_bi_triad::Sm100AutoBranchSeal),
    Sm90a(super::gemm_bi_triad::Sm90aAutoBranchSeal),
    ExactF32Fallback,
}

fn gemm_bi_forward_typed_in<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: cudarc::driver::sys::CUdeviceptr,
    dims: (usize, usize, usize),
    observer: &mut O,
) -> Result<HalfPolicyBranchSeal, String> {
    let checked_dims = super::gemm_bi_triad::GemmDims::nn(dims, dims.1)?;
    validate_half_physical_policy::<O>(ctx)?;
    let physical = HalfPhysicalContext {
        op: ResolvedGemmOp::Nn,
        dtype: y.dtype,
        dims,
    };
    if ctx.bi_tensor_cores() {
        let request = super::gemm_bi_triad::Sm120AutoRequest {
            op: super::gemm_bi_triad::Sm120Op::Nn,
            dtype: y.dtype,
            shape: super::gemm_bi_triad::Sm120Shape::contiguous(
                super::gemm_bi_triad::Sm120Op::Nn,
                dims,
            ),
            a_ptr: x.ptr,
            b_ptr: w.ptr,
            multiprocessors: ctx.kernels.multiprocessor_count(),
            half_policy: ctx.half_triad_policy(),
            operands: super::gemm_bi_triad::Sm120LaunchOperands {
                output_ptr: y.ptr,
                bias_ptr,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm120_auto_observed(ctx, observer, request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm120(seal));
        }
        // The SM100 family reads its own measured table; it is empty until a
        // board of that capability qualifies its cells.
        let sm100_request = super::gemm_bi_triad::Sm100AutoRequest {
            op: super::gemm_bi_triad::Sm100Op::Nn,
            dtype: y.dtype,
            shape: super::gemm_bi_triad::Sm100Shape::contiguous(
                super::gemm_bi_triad::Sm100Op::Nn,
                dims,
            ),
            a_ptr: x.ptr,
            b_ptr: w.ptr,
            operands: super::gemm_bi_triad::Sm100LaunchOperands {
                output_ptr: y.ptr,
                bias_ptr,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm100_auto_observed(ctx, observer, sm100_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm100(seal));
        }
        // A Hopper board runs its own wgmma kernels by the same seal.
        let sm90a_request = super::gemm_bi_triad::Sm90aAutoRequest {
            op: super::gemm_bi_triad::Sm90aOp::Nn,
            dtype: y.dtype,
            shape: super::gemm_bi_triad::Sm90aShape::contiguous(
                super::gemm_bi_triad::Sm90aOp::Nn,
                dims,
            ),
            a_ptr: x.ptr,
            b_ptr: w.ptr,
            operands: super::gemm_bi_triad::Sm90aLaunchOperands {
                output_ptr: y.ptr,
                bias_ptr,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm90a_auto_observed(ctx, observer, sm90a_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm90a(seal));
        }
    }
    // The SM89 deep-K N=128 bucket keeps the exact
    // scalar contract when its measured Split-K plan wins. Other admitted
    // shapes use the separate tensor-core contract in `gemm_bi_forward_tc`.
    if ctx.bi_tensor_cores()
        && !super::gemm_bi_triad::tc_half_policy_prefers_scalar_forward(
            ctx.compute_capability(),
            dims,
            ctx.kernels.multiprocessor_count(),
        )?
    {
        let ops = super::gemm_bi_triad::TcFwdOperands { y, x, w, bias_ptr };
        match super::gemm_bi_triad::gemm_bi_forward_tc_observed(ctx, observer, &ops, dims) {
            Ok((_tile, seal)) => return Ok(HalfPolicyBranchSeal::Native(seal)),
            // Below-tile-gate shapes drop to the scalar tier; real launch
            // failures must surface, not be recomputed around.
            Err(e) if e.starts_with("UNCOVERED") => {}
            Err(e) => return Err(e),
        }
    }
    let ops = super::gemm_bi_triad::TcFwdOperands { y, x, w, bias_ptr };
    match super::gemm_bi_triad::gemm_bi_forward_typed_observed(ctx, observer, &ops, dims) {
        Ok(seal) => return Ok(HalfPolicyBranchSeal::Native(seal)),
        // Only a bucket miss may fall through to the upcast path; a real
        // launch failure must surface, not be recomputed around.
        Err(e) if e.starts_with("UNCOVERED") => {}
        Err(e) => return Err(e),
    }
    ctx.with_bi_upcast_scratch(
        (checked_dims.mk, checked_dims.kn, checked_dims.mn),
        |xs, ws, ys| {
            bi_upcast_to_f32(ctx, x, xs.cached_ptr(), checked_dims.mk, physical, observer)?;
            bi_upcast_to_f32(ctx, w, ws.cached_ptr(), checked_dims.kn, physical, observer)?;
            super::gemm_bi_triad::record_physical_exact_scalar_f32_forward(
                ctx,
                observer,
                ys,
                xs,
                ws.cached_ptr(),
                bias_ptr,
                super::gemm_bi_triad::ScalarFallbackPhysicalContext {
                    dims,
                    dtype: physical.dtype,
                },
            )?;
            bi_downcast_from_f32(ctx, y, ys.cached_ptr(), checked_dims.mn, physical, observer)
        },
    )
    .map(|()| HalfPolicyBranchSeal::ExactF32Fallback)
}

/// Batch-invariant typed dW backward with FULL shape coverage:
/// `dW[K,N] += X^T[K,B] @ dY[B,N]` — typed dY/X, f32 master dW (no
/// downcast; gradients accumulate in f32 by design). Uncovered typed
/// buckets upcast dY/X and run the f32 TN dispatcher — bit-identical to a
/// native typed kernel. `dims` = `(batch, n_in, n_out)`.
/// Qualified CC12.0 BF16/F16 cells may use cached SM120 TMA/MMA16; all other
/// cells retain the existing tensor-core and exact fallback policy.
pub fn gemm_bi_backward_dw_typed(
    ctx: &GpuCtx,
    dw_ptr: cudarc::driver::sys::CUdeviceptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut observer = NoPhysicalObserver;
    gemm_bi_backward_dw_typed_in(ctx, dw_ptr, dy, x_saved, dims, &mut observer).map(drop)
}

fn gemm_bi_backward_dw_typed_in<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    dw_ptr: cudarc::driver::sys::CUdeviceptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
    observer: &mut O,
) -> Result<HalfPolicyBranchSeal, String> {
    let checked_dims = super::gemm_bi_triad::GemmDims::tn(dims)?;
    validate_half_physical_policy::<O>(ctx)?;
    let physical = HalfPhysicalContext {
        op: ResolvedGemmOp::Tn,
        dtype: dy.dtype,
        dims,
    };
    if ctx.bi_tensor_cores() {
        let request = super::gemm_bi_triad::Sm120AutoRequest {
            op: super::gemm_bi_triad::Sm120Op::Tn,
            dtype: dy.dtype,
            shape: super::gemm_bi_triad::Sm120Shape::contiguous(
                super::gemm_bi_triad::Sm120Op::Tn,
                dims,
            ),
            a_ptr: x_saved.ptr,
            b_ptr: dy.ptr,
            multiprocessors: ctx.kernels.multiprocessor_count(),
            half_policy: ctx.half_triad_policy(),
            operands: super::gemm_bi_triad::Sm120LaunchOperands {
                output_ptr: dw_ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 1.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm120_auto_observed(ctx, observer, request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm120(seal));
        }
        // The SM100 family reads its own measured table; it is empty until a
        // board of that capability qualifies its cells.
        let sm100_request = super::gemm_bi_triad::Sm100AutoRequest {
            op: super::gemm_bi_triad::Sm100Op::Tn,
            dtype: dy.dtype,
            shape: super::gemm_bi_triad::Sm100Shape::contiguous(
                super::gemm_bi_triad::Sm100Op::Tn,
                dims,
            ),
            a_ptr: x_saved.ptr,
            b_ptr: dy.ptr,
            operands: super::gemm_bi_triad::Sm100LaunchOperands {
                output_ptr: dw_ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 1.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm100_auto_observed(ctx, observer, sm100_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm100(seal));
        }
        // A Hopper board runs its own wgmma kernels by the same seal.
        let sm90a_request = super::gemm_bi_triad::Sm90aAutoRequest {
            op: super::gemm_bi_triad::Sm90aOp::Tn,
            dtype: dy.dtype,
            shape: super::gemm_bi_triad::Sm90aShape::contiguous(
                super::gemm_bi_triad::Sm90aOp::Tn,
                dims,
            ),
            a_ptr: x_saved.ptr,
            b_ptr: dy.ptr,
            operands: super::gemm_bi_triad::Sm90aLaunchOperands {
                output_ptr: dw_ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 1.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm90a_auto_observed(ctx, observer, sm90a_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm90a(seal));
        }
        match super::gemm_bi_triad::gemm_bi_backward_dw_tc_observed(
            ctx, observer, dw_ptr, dy, x_saved, dims,
        ) {
            Ok((_tile, seal)) => return Ok(HalfPolicyBranchSeal::Native(seal)),
            Err(e) if e.starts_with("UNCOVERED") => {}
            Err(e) => return Err(e),
        }
    }
    match super::gemm_bi_triad::gemm_bi_backward_dw_typed_observed(
        ctx, observer, dw_ptr, dy, x_saved, dims,
    ) {
        Ok(seal) => return Ok(HalfPolicyBranchSeal::Native(seal)),
        Err(e) if e.starts_with("UNCOVERED") => {}
        Err(e) => return Err(e),
    }
    ctx.with_bi_upcast_scratch((checked_dims.mn, checked_dims.mk, 0), |dys, xs, _| {
        bi_upcast_to_f32(
            ctx,
            dy,
            dys.cached_ptr(),
            checked_dims.mn,
            physical,
            observer,
        )?;
        bi_upcast_to_f32(
            ctx,
            x_saved,
            xs.cached_ptr(),
            checked_dims.mk,
            physical,
            observer,
        )?;
        super::gemm_bi_triad::record_physical_exact_scalar_f32_backward_dw(
            ctx,
            observer,
            dw_ptr,
            dys,
            xs,
            super::gemm_bi_triad::ScalarFallbackPhysicalContext {
                dims,
                dtype: physical.dtype,
            },
        )
    })
    .map(|()| HalfPolicyBranchSeal::ExactF32Fallback)
}

/// Batch-invariant typed dX backward with FULL shape coverage:
/// `dX[B,K] = dY[B,N] @ W^T[N,K]` — typed dY/W/dX. Uncovered typed buckets
/// upcast dY/W, run the f32 NT dispatcher, and RNE-downcast dX —
/// bit-identical to a native typed kernel. `dims` = `(batch, n_in, n_out)`.
/// Qualified CC12.0 BF16/F16 cells may use cached SM120 TMA/MMA16; all other
/// cells retain the existing tensor-core and exact fallback policy.
pub fn gemm_bi_backward_dx_typed(
    ctx: &GpuCtx,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let mut observer = NoPhysicalObserver;
    gemm_bi_backward_dx_typed_in(ctx, dx, dy, w, dims, &mut observer).map(drop)
}

fn gemm_bi_backward_dx_typed_in<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
    observer: &mut O,
) -> Result<HalfPolicyBranchSeal, String> {
    let checked_dims = super::gemm_bi_triad::GemmDims::nt(dims)?;
    validate_half_physical_policy::<O>(ctx)?;
    let physical = HalfPhysicalContext {
        op: ResolvedGemmOp::Nt,
        dtype: dx.dtype,
        dims,
    };
    if ctx.bi_tensor_cores() {
        let request = super::gemm_bi_triad::Sm120AutoRequest {
            op: super::gemm_bi_triad::Sm120Op::Nt,
            dtype: dx.dtype,
            shape: super::gemm_bi_triad::Sm120Shape::contiguous(
                super::gemm_bi_triad::Sm120Op::Nt,
                dims,
            ),
            a_ptr: dy.ptr,
            b_ptr: w.ptr,
            multiprocessors: ctx.kernels.multiprocessor_count(),
            half_policy: ctx.half_triad_policy(),
            operands: super::gemm_bi_triad::Sm120LaunchOperands {
                output_ptr: dx.ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm120_auto_observed(ctx, observer, request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm120(seal));
        }
        // The SM100 family reads its own measured table; it is empty until a
        // board of that capability qualifies its cells.
        let sm100_request = super::gemm_bi_triad::Sm100AutoRequest {
            op: super::gemm_bi_triad::Sm100Op::Nt,
            dtype: dx.dtype,
            shape: super::gemm_bi_triad::Sm100Shape::contiguous(
                super::gemm_bi_triad::Sm100Op::Nt,
                dims,
            ),
            a_ptr: dy.ptr,
            b_ptr: w.ptr,
            operands: super::gemm_bi_triad::Sm100LaunchOperands {
                output_ptr: dx.ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm100_auto_observed(ctx, observer, sm100_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm100(seal));
        }
        // A Hopper board runs its own wgmma kernels by the same seal.
        let sm90a_request = super::gemm_bi_triad::Sm90aAutoRequest {
            op: super::gemm_bi_triad::Sm90aOp::Nt,
            dtype: dx.dtype,
            shape: super::gemm_bi_triad::Sm90aShape::contiguous(
                super::gemm_bi_triad::Sm90aOp::Nt,
                dims,
            ),
            a_ptr: dy.ptr,
            b_ptr: w.ptr,
            operands: super::gemm_bi_triad::Sm90aLaunchOperands {
                output_ptr: dx.ptr,
                bias_ptr: 0,
                alpha: 1.0,
                beta: 0.0,
            },
        };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm90a_auto_observed(ctx, observer, sm90a_request)?
        {
            return Ok(HalfPolicyBranchSeal::Sm90a(seal));
        }
        match super::gemm_bi_triad::gemm_bi_backward_dx_tc_observed(ctx, observer, dx, dy, w, dims)
        {
            Ok((_tile, seal)) => return Ok(HalfPolicyBranchSeal::Native(seal)),
            Err(e) if e.starts_with("UNCOVERED") => {}
            Err(e) => return Err(e),
        }
    }
    match super::gemm_bi_triad::gemm_bi_backward_dx_typed_observed(ctx, observer, dx, dy, w, dims) {
        Ok(seal) => return Ok(HalfPolicyBranchSeal::Native(seal)),
        Err(e) if e.starts_with("UNCOVERED") => {}
        Err(e) => return Err(e),
    }
    ctx.with_bi_upcast_scratch(
        (checked_dims.mn, checked_dims.kn, checked_dims.mk),
        |dys, ws, dxs| {
            bi_upcast_to_f32(
                ctx,
                dy,
                dys.cached_ptr(),
                checked_dims.mn,
                physical,
                observer,
            )?;
            bi_upcast_to_f32(ctx, w, ws.cached_ptr(), checked_dims.kn, physical, observer)?;
            super::gemm_bi_triad::record_physical_exact_scalar_f32_backward_dx(
                ctx,
                observer,
                dxs,
                dys,
                ws.cached_ptr(),
                super::gemm_bi_triad::ScalarFallbackPhysicalContext {
                    dims,
                    dtype: physical.dtype,
                },
            )?;
            bi_downcast_from_f32(
                ctx,
                dx,
                dxs.cached_ptr(),
                checked_dims.mk,
                physical,
                observer,
            )
        },
    )
    .map(|()| HalfPolicyBranchSeal::ExactF32Fallback)
}

#[derive(Clone, Copy, Debug)]
pub(in crate::mamba_ssm::gpu) struct HalfPhysicalTraceRequest {
    pub(in crate::mamba_ssm::gpu) op: ResolvedGemmOp,
    pub(in crate::mamba_ssm::gpu) output: cudarc::driver::sys::CUdeviceptr,
    pub(in crate::mamba_ssm::gpu) a: cudarc::driver::sys::CUdeviceptr,
    pub(in crate::mamba_ssm::gpu) b: cudarc::driver::sys::CUdeviceptr,
    pub(in crate::mamba_ssm::gpu) bias: cudarc::driver::sys::CUdeviceptr,
    pub(in crate::mamba_ssm::gpu) dtype: WeightDtype,
    pub(in crate::mamba_ssm::gpu) dims: (usize, usize, usize),
    pub(in crate::mamba_ssm::gpu) nn_strides: Option<(usize, usize, usize)>,
    pub(in crate::mamba_ssm::gpu) forced_tile: Option<super::gemm_bi_triad::TcTile>,
    pub(in crate::mamba_ssm::gpu) capacity: usize,
}

pub(in crate::mamba_ssm::gpu) struct F32PhysicalGraphPackageRequest<'a> {
    pub(in crate::mamba_ssm::gpu) prepared: &'a super::gemm_bi_triad::PreparedF32TriadLaunch,
    pub(in crate::mamba_ssm::gpu) output: &'a mut GpuBuffer,
    pub(in crate::mamba_ssm::gpu) a: &'a GpuBuffer,
    pub(in crate::mamba_ssm::gpu) b: &'a GpuBuffer,
    pub(in crate::mamba_ssm::gpu) capacity: usize,
}

const PHYSICAL_GRAPH_MAX_KERNEL_ARGUMENTS: usize = 16;
const PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES: usize = 64;

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct PhysicalGraphKernelArgument {
    bytes: [u8; PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES],
}

unsafe impl DeviceRepr for PhysicalGraphKernelArgument {}

impl PhysicalGraphKernelArgument {
    fn encode<T: Copy>(value: T) -> Result<Self, String> {
        let width = std::mem::size_of::<T>();
        if width > PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES {
            return Err(format!(
                "physical graph kernel argument uses {width} bytes; maximum is {PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES}"
            ));
        }
        let mut encoded = Self {
            bytes: [0; PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&value).cast::<u8>(),
                encoded.bytes.as_mut_ptr(),
                width,
            );
        }
        Ok(encoded)
    }
}

struct PhysicalGraphKernelArguments {
    values: [PhysicalGraphKernelArgument; PHYSICAL_GRAPH_MAX_KERNEL_ARGUMENTS],
    len: usize,
}

impl PhysicalGraphKernelArguments {
    fn new() -> Self {
        Self {
            values: [PhysicalGraphKernelArgument {
                bytes: [0; PHYSICAL_GRAPH_MAX_ARGUMENT_BYTES],
            }; PHYSICAL_GRAPH_MAX_KERNEL_ARGUMENTS],
            len: 0,
        }
    }

    fn push<T: Copy>(&mut self, value: T) -> Result<(), String> {
        let slot = self
            .values
            .get_mut(self.len)
            .ok_or_else(|| "physical graph kernel argument capacity exceeded".to_string())?;
        *slot = PhysicalGraphKernelArgument::encode(value)?;
        self.len += 1;
        Ok(())
    }

    fn values(&self) -> &[PhysicalGraphKernelArgument] {
        &self.values[..self.len]
    }
}

struct PreparedPhysicalGraphLaunch {
    function: CudaFunction,
    config: LaunchConfig,
    node: ResolvedPhysicalKernelLaunch,
    arguments: PhysicalGraphKernelArguments,
}

struct BoundPhysicalGraphLaunch<'a> {
    builder: LaunchArgs<'a>,
    config: LaunchConfig,
    node: ResolvedPhysicalKernelLaunch,
}

pub(super) struct BoundPhysicalGraphLaunches<'a> {
    prefix: Vec<BoundPhysicalGraphLaunch<'a>>,
    triad: Option<super::gemm_bi_triad::BoundTriadPhysicalGraphSequence<'a>>,
    suffix: Vec<BoundPhysicalGraphLaunch<'a>>,
    #[cfg(test)]
    fail_before_enqueue: bool,
}

impl BoundPhysicalGraphLaunches<'_> {
    #[inline(always)]
    pub(super) unsafe fn enqueue(
        &mut self,
        observer: &mut RecordingPhysicalObserver,
    ) -> Result<(), PhysicalCudaLaunchError> {
        #[cfg(test)]
        if self.fail_before_enqueue {
            return Err(PhysicalCudaLaunchError::Prepared(
                "expected prepared physical body error",
            ));
        }
        for launch in &mut self.prefix {
            unsafe {
                enqueue_prepared_physical_launch(
                    observer,
                    &mut launch.builder,
                    launch.config,
                    launch.node,
                )?;
            }
        }
        if let Some(triad) = &mut self.triad {
            unsafe { triad.enqueue(observer)? };
        }
        for launch in &mut self.suffix {
            unsafe {
                enqueue_prepared_physical_launch(
                    observer,
                    &mut launch.builder,
                    launch.config,
                    launch.node,
                )?;
            }
        }
        Ok(())
    }
}

pub(super) struct PreparedPhysicalGraphPackage<'a> {
    ctx: &'a GpuCtx,
    manifest: PreparedPhysicalCaptureManifest,
    observer: Option<RecordingPhysicalObserver>,
    prefix: Box<[PreparedPhysicalGraphLaunch]>,
    triad: Option<super::gemm_bi_triad::PreparedTriadPhysicalGraphSequence>,
    suffix: Box<[PreparedPhysicalGraphLaunch]>,
    launch_capacity: usize,
    context_token: u64,
    stream_token: usize,
    #[cfg(test)]
    fail_before_enqueue: bool,
    #[cfg(test)]
    drift_policy_after_capture: bool,
}

impl PreparedPhysicalGraphPackage<'_> {
    pub(super) fn context(&self) -> &GpuCtx {
        self.ctx
    }

    pub(super) fn manifest(&self) -> &PreparedPhysicalCaptureManifest {
        &self.manifest
    }

    pub(super) fn take_observer(&mut self) -> Result<RecordingPhysicalObserver, String> {
        self.observer
            .take()
            .ok_or_else(|| "physical graph package observer was already consumed".to_string())
    }

    pub(super) fn validate(&self) -> Result<(), String> {
        if self.context_token != self.ctx.instance_token() {
            return Err("physical graph package belongs to another GPU context".into());
        }
        if self.stream_token != self.ctx.stream_token() {
            return Err("physical graph package belongs to another CUDA stream".into());
        }
        let prepared_count = self.prefix.len()
            + self.triad.as_ref().map_or(0, |triad| triad.len())
            + self.suffix.len();
        if self.launch_capacity == 0
            || self.launch_capacity != prepared_count
            || self.launch_capacity != self.manifest.launch_capacity()
        {
            return Err("physical graph package has inconsistent exact launch capacity".into());
        }
        Ok(())
    }

    fn bind_direct_launches<'a>(
        &'a self,
        prepared: &'a [PreparedPhysicalGraphLaunch],
    ) -> Result<Vec<BoundPhysicalGraphLaunch<'a>>, String> {
        let mut launches = Vec::new();
        launches
            .try_reserve_exact(prepared.len())
            .map_err(|error| format!("reserve bound physical graph launches: {error}"))?;
        for launch in prepared {
            let mut builder = self.ctx.stream.launch_builder(&launch.function);
            for argument in launch.arguments.values() {
                builder.arg(argument);
            }
            launches.push(BoundPhysicalGraphLaunch {
                builder,
                config: launch.config,
                node: launch.node,
            });
        }
        if launches.len() != prepared.len() || launches.capacity() != prepared.len() {
            return Err("bound physical graph launch backing capacity is not exact".into());
        }
        Ok(launches)
    }

    pub(super) fn bind_launches(&self) -> Result<BoundPhysicalGraphLaunches<'_>, String> {
        let prefix = self.bind_direct_launches(&self.prefix)?;
        let triad = self
            .triad
            .as_ref()
            .map(|triad| triad.bind(&self.ctx.stream))
            .transpose()?;
        let suffix = self.bind_direct_launches(&self.suffix)?;
        Ok(BoundPhysicalGraphLaunches {
            prefix,
            triad,
            suffix,
            #[cfg(test)]
            fail_before_enqueue: self.fail_before_enqueue,
        })
    }

    #[cfg(test)]
    fn inject_body_failure(&mut self) {
        self.fail_before_enqueue = true;
    }

    #[cfg(test)]
    fn inject_driver_failure(&mut self) {
        self.prefix[0].config.grid_dim.0 = 0;
    }

    #[cfg(test)]
    fn inject_post_capture_policy_drift(&mut self) {
        self.drift_policy_after_capture = true;
    }

    #[cfg(test)]
    pub(super) fn apply_post_capture_test_mutation(&self) {
        if self.drift_policy_after_capture {
            self.ctx.set_bi_tensor_cores(false);
        }
    }
}

fn physical_argument_bytes(elements: usize, width: usize, name: &str) -> Result<u64, String> {
    elements
        .checked_mul(width)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| format!("physical graph {name} span overflows u64"))
}

fn physical_f32_storage_elements(
    rows: usize,
    width: usize,
    stride: usize,
    name: &str,
) -> Result<usize, String> {
    if rows == 0 || width == 0 || stride < width {
        return Err(format!("physical F32 {name} has invalid storage geometry"));
    }
    (rows - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(width))
        .ok_or_else(|| format!("physical F32 {name} storage span overflows usize"))
}

fn prepare_f32_physical_graph_observer(
    ctx: &GpuCtx,
    prepared: &super::gemm_bi_triad::PreparedF32TriadLaunch,
    capacity: usize,
) -> Result<RecordingPhysicalObserver, String> {
    let request = prepared.physical_graph_request();
    let operands = prepared.physical_graph_operands();
    let ranges = f32_physical_argument_ranges(request, operands)?;
    prepare_physical_observer(ctx, capacity, &ranges)
}

fn f32_physical_argument_ranges(
    request: super::gemm_bi_triad::F32TriadRequest,
    operands: super::gemm_bi_triad::F32TriadOperands,
) -> Result<Vec<PhysicalArgumentRange>, String> {
    request.shape.validate(request.op)?;
    let shape = request.shape;
    let (a_geometry, b_geometry, output_geometry) = match request.op {
        ResolvedGemmOp::Nn => (
            (shape.m, shape.k, shape.lda),
            (shape.k, shape.n, shape.ldb),
            (shape.m, shape.n, shape.ldc),
        ),
        ResolvedGemmOp::Tn => (
            (shape.m, shape.k, shape.lda),
            (shape.m, shape.n, shape.ldb),
            (shape.k, shape.n, shape.ldc),
        ),
        ResolvedGemmOp::Nt => (
            (shape.m, shape.n, shape.lda),
            (shape.k, shape.n, shape.ldb),
            (shape.m, shape.k, shape.ldc),
        ),
    };
    // Epilogue-only zero-reduction launches bind null A/B and never read
    // either allocation. Keep output/bias liveness and all nonzero operand
    // geometry checks; do not weaken the shared half-storage validator.
    let operand_ranges = if shape.reduction(request.op) == 0 {
        1
    } else {
        3
    };
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(operand_ranges + usize::from(operands.bias.is_some()))
        .map_err(|error| format!("reserve physical F32 argument ranges: {error}"))?;
    for (pointer, (rows, width, stride), name) in [
        (operands.output, output_geometry, "output"),
        (operands.a, a_geometry, "A"),
        (operands.b, b_geometry, "B"),
    ]
    .into_iter()
    .take(operand_ranges)
    {
        let elements = physical_f32_storage_elements(rows, width, stride, name)?;
        ranges.push(PhysicalArgumentRange {
            pointer,
            required_bytes: physical_argument_bytes(elements, std::mem::size_of::<f32>(), name)?,
        });
    }
    if let Some(bias) = operands.bias {
        ranges.push(PhysicalArgumentRange {
            pointer: bias,
            required_bytes: physical_argument_bytes(shape.n, std::mem::size_of::<f32>(), "bias")?,
        });
    }
    if ranges.capacity() != ranges.len() {
        return Err("physical F32 argument range capacity is not exact".into());
    }
    Ok(ranges)
}

fn prepare_half_physical_observer(
    ctx: &GpuCtx,
    request: HalfPhysicalTraceRequest,
) -> Result<RecordingPhysicalObserver, String> {
    let (_, _, n) = request.dims;
    let shape = half_physical_request_shape(request)?;
    let checked_dims = match request.op {
        ResolvedGemmOp::Nn => super::gemm_bi_triad::GemmDims::nn(request.dims, shape.lda)?,
        ResolvedGemmOp::Tn => super::gemm_bi_triad::GemmDims::tn(request.dims)?,
        ResolvedGemmOp::Nt => super::gemm_bi_triad::GemmDims::nt(request.dims)?,
    };
    let half_width = request.dtype.size_bytes();
    let (output_geometry, a_geometry, b_geometry) = match request.op {
        ResolvedGemmOp::Nn => (
            (shape.m, shape.n, shape.ldc),
            (shape.m, shape.k, shape.lda),
            (shape.k, shape.n, shape.ldb),
        ),
        ResolvedGemmOp::Tn => (
            (shape.k, shape.n, shape.ldc),
            (shape.m, shape.k, shape.lda),
            (shape.m, shape.n, shape.ldb),
        ),
        ResolvedGemmOp::Nt => (
            (shape.m, shape.k, shape.ldc),
            (shape.m, shape.n, shape.lda),
            (shape.k, shape.n, shape.ldb),
        ),
    };
    let output_width = if request.op == ResolvedGemmOp::Tn {
        std::mem::size_of::<f32>()
    } else {
        half_width
    };
    let output_elements = physical_f32_storage_elements(
        output_geometry.0,
        output_geometry.1,
        output_geometry.2,
        "output",
    )?;
    let a_elements = physical_f32_storage_elements(a_geometry.0, a_geometry.1, a_geometry.2, "A")?;
    let b_elements = physical_f32_storage_elements(b_geometry.0, b_geometry.1, b_geometry.2, "B")?;
    let mut ranges = Vec::with_capacity(7);
    ranges.push(PhysicalArgumentRange {
        pointer: request.output,
        required_bytes: physical_argument_bytes(output_elements, output_width, "output")?,
    });
    ranges.push(PhysicalArgumentRange {
        pointer: request.a,
        required_bytes: physical_argument_bytes(a_elements, half_width, "A")?,
    });
    ranges.push(PhysicalArgumentRange {
        pointer: request.b,
        required_bytes: physical_argument_bytes(b_elements, half_width, "B")?,
    });
    if request.bias != 0 {
        ranges.push(PhysicalArgumentRange {
            pointer: request.bias,
            required_bytes: physical_argument_bytes(n, std::mem::size_of::<f32>(), "bias")?,
        });
    }
    let scratch_sizes = match request.op {
        ResolvedGemmOp::Nn => (checked_dims.mk, checked_dims.kn, checked_dims.mn),
        ResolvedGemmOp::Tn => (checked_dims.mn, checked_dims.mk, 0),
        ResolvedGemmOp::Nt => (checked_dims.mn, checked_dims.kn, checked_dims.mk),
    };
    ctx.with_bi_upcast_scratch(scratch_sizes, |first, second, third| {
        for (buffer, elements) in [
            (first, scratch_sizes.0),
            (second, scratch_sizes.1),
            (third, scratch_sizes.2),
        ] {
            if elements != 0 {
                ranges.push(PhysicalArgumentRange {
                    pointer: buffer.cached_ptr(),
                    required_bytes: physical_argument_bytes(
                        elements,
                        std::mem::size_of::<f32>(),
                        "scratch",
                    )?,
                });
            }
        }
        Ok(())
    })?;
    prepare_physical_observer(ctx, request.capacity, &ranges)
}

fn half_physical_request_shape(
    request: HalfPhysicalTraceRequest,
) -> Result<super::gemm_bi_triad::F32TriadShape, String> {
    let mut shape = super::gemm_bi_triad::F32TriadShape::contiguous(request.op, request.dims);
    if let Some((lda, ldb, ldc)) = request.nn_strides {
        if request.op != ResolvedGemmOp::Nn || request.forced_tile.is_none() {
            return Err("padded half physical strides require a forced NN route".into());
        }
        (shape.lda, shape.ldb, shape.ldc) = (lda, ldb, ldc);
    }
    shape.validate(request.op)?;
    Ok(shape)
}

fn prepare_native_half_graph_launch(
    ctx: &GpuCtx,
    observer: &RecordingPhysicalObserver,
    request: HalfPhysicalTraceRequest,
    expected: ResolvedPhysicalKernelLaunch,
) -> Result<PreparedPhysicalGraphLaunch, String> {
    let identity =
        super::gemm_bi_triad::prepare_native_half_graph_identity(ctx, observer, expected, request)?;
    let (function, config, node, base) = identity.into_parts();
    let shape = half_physical_request_shape(request)?;
    let checked = match request.op {
        ResolvedGemmOp::Nn => super::gemm_bi_triad::GemmDims::nn(request.dims, shape.lda)?,
        ResolvedGemmOp::Tn => super::gemm_bi_triad::GemmDims::tn(request.dims)?,
        ResolvedGemmOp::Nt => super::gemm_bi_triad::GemmDims::nt(request.dims)?,
    };
    let alpha = 1.0_f32;
    let mut arguments = PhysicalGraphKernelArguments::new();
    match request.op {
        ResolvedGemmOp::Nn => {
            let beta = 0.0_f32;
            arguments.push(request.output)?;
            arguments.push(request.a)?;
            arguments.push(request.b)?;
            arguments.push(request.bias)?;
            arguments.push(alpha)?;
            arguments.push(beta)?;
            arguments.push(checked.m_i32)?;
            if base == "gemm_bi_nn_gemv" {
                arguments.push(checked.k_i32)?;
                arguments.push(checked.k_i32)?;
                arguments.push(1_i32)?;
            } else {
                arguments.push(checked.n_i32)?;
                arguments.push(checked.k_i32)?;
                arguments.push(i32::try_from(shape.lda).map_err(|_| "NN lda exceeds i32::MAX")?)?;
                arguments.push(i32::try_from(shape.ldb).map_err(|_| "NN ldb exceeds i32::MAX")?)?;
                arguments.push(i32::try_from(shape.ldc).map_err(|_| "NN ldc exceeds i32::MAX")?)?;
                if matches!(base, "gemm_bi_nn_narrow" | "gemm_bi_nn_narrow_small") {
                    arguments.push(0_i32)?;
                }
            }
        }
        ResolvedGemmOp::Tn => {
            arguments.push(request.output)?;
            arguments.push(request.a)?;
            arguments.push(request.b)?;
            arguments.push(alpha)?;
            arguments.push(checked.m_i32)?;
            arguments.push(checked.k_i32)?;
            if base == "gemm_bi_tn_gemv" {
                arguments.push(checked.k_i32)?;
                arguments.push(1_i32)?;
            } else {
                arguments.push(checked.n_i32)?;
            }
            if base == "gemm_bi_tn_tc64_streamk" {
                let (partial, flags) = super::gemm_bi_triad::sm80_streamk_workspace(
                    &ctx.stream,
                    &ctx.kernels,
                    config.grid_dim.0,
                )?;
                arguments.push(partial)?;
                arguments.push(flags)?;
            }
        }
        ResolvedGemmOp::Nt => {
            arguments.push(request.output)?;
            arguments.push(request.a)?;
            arguments.push(request.b)?;
            arguments.push(alpha)?;
            arguments.push(checked.m_i32)?;
            if base == "gemm_bi_nt_gemv" {
                arguments.push(checked.k_i32)?;
                arguments.push(checked.k_i32)?;
                arguments.push(1_i32)?;
            } else {
                arguments.push(checked.n_i32)?;
                arguments.push(checked.k_i32)?;
            }
        }
    }
    Ok(PreparedPhysicalGraphLaunch {
        function,
        config,
        node,
        arguments,
    })
}

fn prepare_conversion_graph_launch(
    ctx: &GpuCtx,
    observer: &RecordingPhysicalObserver,
    physical: HalfPhysicalContext,
    kind: PhysicalLaunchKind,
    conversion: (usize, u64, u64),
) -> Result<PreparedPhysicalGraphLaunch, String> {
    let (count, source, destination) = conversion;
    let config = grid_1d(count);
    let observation = conversion_observation(physical, kind, count, source, destination)?;
    let node = resolve_physical_launch_observation(observer, observation, config)?;
    let function = match (kind, request_half_dtype(physical)?) {
        (PhysicalLaunchKind::InputUpcast, WeightDtype::Bf16) => {
            ctx.kernels.cast_bf16_to_f32.clone()
        }
        (PhysicalLaunchKind::InputUpcast, WeightDtype::F16) => ctx.kernels.cast_f16_to_f32.clone(),
        (PhysicalLaunchKind::OutputDowncast, WeightDtype::Bf16) => {
            ctx.kernels.cast_f32_to_bf16.clone()
        }
        (PhysicalLaunchKind::OutputDowncast, WeightDtype::F16) => {
            ctx.kernels.cast_f32_to_f16.clone()
        }
        _ => {
            return Err(
                "physical graph conversion requires a half dtype and conversion kind".into(),
            );
        }
    };
    let count_i32 =
        i32::try_from(count).map_err(|_| "physical graph conversion count exceeds i32::MAX")?;
    let mut arguments = PhysicalGraphKernelArguments::new();
    match kind {
        PhysicalLaunchKind::InputUpcast => {
            arguments.push(destination)?;
            arguments.push(source)?;
        }
        PhysicalLaunchKind::OutputDowncast => {
            arguments.push(destination)?;
            arguments.push(source)?;
        }
        PhysicalLaunchKind::Gemm => {
            return Err("physical graph conversion cannot bind a GEMM node".into());
        }
    }
    arguments.push(count_i32)?;
    Ok(PreparedPhysicalGraphLaunch {
        function,
        config,
        node,
        arguments,
    })
}

fn request_half_dtype(physical: HalfPhysicalContext) -> Result<WeightDtype, String> {
    match physical.dtype {
        WeightDtype::Bf16 | WeightDtype::F16 => Ok(physical.dtype),
        WeightDtype::F32 => Err("physical half graph fallback does not accept f32".into()),
    }
}

type PreparedFallbackGraphSegments = (
    Vec<PreparedPhysicalGraphLaunch>,
    super::gemm_bi_triad::PreparedTriadPhysicalGraphSequence,
    Vec<PreparedPhysicalGraphLaunch>,
);

fn prepare_fallback_graph_segments(
    ctx: &GpuCtx,
    observer: &RecordingPhysicalObserver,
    request: HalfPhysicalTraceRequest,
) -> Result<PreparedFallbackGraphSegments, String> {
    let checked = match request.op {
        ResolvedGemmOp::Nn => super::gemm_bi_triad::GemmDims::nn(request.dims, request.dims.1)?,
        ResolvedGemmOp::Tn => super::gemm_bi_triad::GemmDims::tn(request.dims)?,
        ResolvedGemmOp::Nt => super::gemm_bi_triad::GemmDims::nt(request.dims)?,
    };
    let physical = HalfPhysicalContext {
        op: request.op,
        dtype: request.dtype,
        dims: request.dims,
    };
    request_half_dtype(physical)?;
    let scratch_sizes = match request.op {
        ResolvedGemmOp::Nn => (checked.mk, checked.kn, checked.mn),
        ResolvedGemmOp::Tn => (checked.mn, checked.mk, 0),
        ResolvedGemmOp::Nt => (checked.mn, checked.kn, checked.mk),
    };
    ctx.with_bi_upcast_scratch(scratch_sizes, |first, second, third| {
        let first_ptr = first.cached_ptr();
        let second_ptr = second.cached_ptr();
        let third_ptr = third.cached_ptr();
        let input_conversions = match request.op {
            ResolvedGemmOp::Nn => [
                (checked.mk, request.a, first_ptr),
                (checked.kn, request.b, second_ptr),
            ],
            ResolvedGemmOp::Tn => [
                (checked.mn, request.b, first_ptr),
                (checked.mk, request.a, second_ptr),
            ],
            ResolvedGemmOp::Nt => [
                (checked.mn, request.a, first_ptr),
                (checked.kn, request.b, second_ptr),
            ],
        };
        let mut prefix = Vec::new();
        prefix
            .try_reserve_exact(input_conversions.len())
            .map_err(|error| format!("reserve physical graph conversion prefix: {error}"))?;
        for conversion in input_conversions {
            prefix.push(prepare_conversion_graph_launch(
                ctx,
                observer,
                physical,
                PhysicalLaunchKind::InputUpcast,
                conversion,
            )?);
        }
        if prefix.capacity() != prefix.len() {
            return Err("physical graph conversion prefix capacity is not exact".into());
        }
        let scalar_physical = super::gemm_bi_triad::ScalarFallbackPhysicalContext {
            dims: request.dims,
            dtype: request.dtype,
        };
        let scalar = match request.op {
            ResolvedGemmOp::Nn => {
                super::gemm_bi_triad::prepare_exact_scalar_f32_forward_graph_sequence(
                    ctx,
                    observer,
                    third,
                    first,
                    second_ptr,
                    request.bias,
                    scalar_physical,
                )?
            }
            ResolvedGemmOp::Tn => {
                super::gemm_bi_triad::prepare_exact_scalar_f32_backward_dw_graph_sequence(
                    ctx,
                    observer,
                    request.output,
                    first,
                    second,
                    scalar_physical,
                )?
            }
            ResolvedGemmOp::Nt => {
                super::gemm_bi_triad::prepare_exact_scalar_f32_backward_dx_graph_sequence(
                    ctx,
                    observer,
                    third,
                    first,
                    second_ptr,
                    scalar_physical,
                )?
            }
        };
        let mut suffix = Vec::new();
        let downcast = match request.op {
            ResolvedGemmOp::Nn => Some((checked.mn, third_ptr, request.output)),
            ResolvedGemmOp::Tn => None,
            ResolvedGemmOp::Nt => Some((checked.mk, third_ptr, request.output)),
        };
        if let Some(conversion) = downcast {
            suffix
                .try_reserve_exact(1)
                .map_err(|error| format!("reserve physical graph conversion suffix: {error}"))?;
            suffix.push(prepare_conversion_graph_launch(
                ctx,
                observer,
                physical,
                PhysicalLaunchKind::OutputDowncast,
                conversion,
            )?);
            if suffix.capacity() != suffix.len() {
                return Err("physical graph conversion suffix capacity is not exact".into());
            }
        }
        Ok((prefix, scalar, suffix))
    })
}

pub(super) fn prepare_half_physical_graph_package<'a>(
    ctx: &'a GpuCtx,
    request: HalfPhysicalTraceRequest,
    manifest: &PreparedPhysicalCaptureManifest,
) -> Result<PreparedPhysicalGraphPackage<'a>, String> {
    manifest.validate_capture_request(ctx.gemm_route(), request.capacity)?;
    let observer = prepare_half_physical_observer(ctx, request)?;
    let first = manifest
        .nodes()
        .first()
        .ok_or_else(|| "prepared physical graph manifest is empty".to_string())?;
    let (prefix, triad, suffix) = match (first.kind(), first.module_kind()) {
        (PhysicalLaunchKind::Gemm, ModuleKind::TriadSm80 | ModuleKind::TriadScalar) => (
            vec![prepare_native_half_graph_launch(
                ctx, &observer, request, *first,
            )?],
            None,
            Vec::new(),
        ),
        (PhysicalLaunchKind::InputUpcast, ModuleKind::Fixed) => {
            let (prefix, triad, suffix) = prepare_fallback_graph_segments(ctx, &observer, request)?;
            (prefix, Some(triad), suffix)
        }
        (PhysicalLaunchKind::Gemm, ModuleKind::TriadSm120) => {
            let op = match request.op {
                ResolvedGemmOp::Nn => super::gemm_bi_triad::Sm120Op::Nn,
                ResolvedGemmOp::Tn => super::gemm_bi_triad::Sm120Op::Tn,
                ResolvedGemmOp::Nt => super::gemm_bi_triad::Sm120Op::Nt,
            };
            let auto = super::gemm_bi_triad::Sm120AutoRequest {
                op,
                dtype: request.dtype,
                shape: super::gemm_bi_triad::Sm120Shape::contiguous(op, request.dims),
                a_ptr: request.a,
                b_ptr: request.b,
                multiprocessors: ctx.kernels.multiprocessor_count(),
                half_policy: ctx.half_triad_policy(),
                operands: super::gemm_bi_triad::Sm120LaunchOperands {
                    output_ptr: request.output,
                    bias_ptr: request.bias,
                    alpha: 1.0,
                    beta: if op == super::gemm_bi_triad::Sm120Op::Tn {
                        1.0
                    } else {
                        0.0
                    },
                },
            };
            let triad =
                super::gemm_bi_triad::prepare_sm120_auto_graph_sequence(ctx, &observer, auto)?;
            if triad.len() != 1 || manifest.nodes().len() != 1 {
                return Err("prepared SM120 graph package must contain exactly one launch".into());
            }
            (Vec::new(), Some(triad), Vec::new())
        }
        (PhysicalLaunchKind::Gemm, ModuleKind::TriadSm100) => {
            let op = match request.op {
                ResolvedGemmOp::Nn => super::gemm_bi_triad::Sm100Op::Nn,
                ResolvedGemmOp::Tn => super::gemm_bi_triad::Sm100Op::Tn,
                ResolvedGemmOp::Nt => super::gemm_bi_triad::Sm100Op::Nt,
            };
            let auto = super::gemm_bi_triad::Sm100AutoRequest {
                op,
                dtype: request.dtype,
                shape: super::gemm_bi_triad::Sm100Shape::contiguous(op, request.dims),
                a_ptr: request.a,
                b_ptr: request.b,
                operands: super::gemm_bi_triad::Sm100LaunchOperands {
                    output_ptr: request.output,
                    bias_ptr: request.bias,
                    alpha: 1.0,
                    beta: if op == super::gemm_bi_triad::Sm100Op::Tn {
                        1.0
                    } else {
                        0.0
                    },
                },
            };
            let triad =
                super::gemm_bi_triad::prepare_sm100_auto_graph_sequence(ctx, &observer, auto)?;
            if triad.len() != 1 || manifest.nodes().len() != 1 {
                return Err("prepared SM100 graph package must contain exactly one launch".into());
            }
            (Vec::new(), Some(triad), Vec::new())
        }
        (PhysicalLaunchKind::Gemm, ModuleKind::TriadSm90a) => {
            let op = match request.op {
                ResolvedGemmOp::Nn => super::gemm_bi_triad::Sm90aOp::Nn,
                ResolvedGemmOp::Tn => super::gemm_bi_triad::Sm90aOp::Tn,
                ResolvedGemmOp::Nt => super::gemm_bi_triad::Sm90aOp::Nt,
            };
            let auto = super::gemm_bi_triad::Sm90aAutoRequest {
                op,
                dtype: request.dtype,
                shape: super::gemm_bi_triad::Sm90aShape::contiguous(op, request.dims),
                a_ptr: request.a,
                b_ptr: request.b,
                operands: super::gemm_bi_triad::Sm90aLaunchOperands {
                    output_ptr: request.output,
                    bias_ptr: request.bias,
                    alpha: 1.0,
                    beta: if op == super::gemm_bi_triad::Sm90aOp::Tn {
                        1.0
                    } else {
                        0.0
                    },
                },
            };
            let triad =
                super::gemm_bi_triad::prepare_sm90a_auto_graph_sequence(ctx, &observer, auto)?;
            if triad.len() != 1 || manifest.nodes().len() != 1 {
                return Err("prepared SM90a graph package must contain exactly one launch".into());
            }
            (Vec::new(), Some(triad), Vec::new())
        }
        _ => return Err("prepared physical graph manifest has an unsupported first node".into()),
    };
    let prepared_count =
        prefix.len() + triad.as_ref().map_or(0, |triad| triad.len()) + suffix.len();
    if prepared_count != request.capacity {
        return Err("prepared physical graph package backing capacity is not exact".into());
    }
    Ok(PreparedPhysicalGraphPackage {
        ctx,
        manifest: manifest.clone(),
        observer: Some(observer),
        prefix: prefix.into_boxed_slice(),
        triad,
        suffix: suffix.into_boxed_slice(),
        launch_capacity: request.capacity,
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        #[cfg(test)]
        fail_before_enqueue: false,
        #[cfg(test)]
        drift_policy_after_capture: false,
    })
}

struct PreparedF32PhysicalGraphParts {
    observer: RecordingPhysicalObserver,
    triad: super::gemm_bi_triad::PreparedTriadPhysicalGraphSequence,
}

fn prepare_f32_physical_graph_parts(
    ctx: &GpuCtx,
    request: F32PhysicalGraphPackageRequest<'_>,
) -> Result<PreparedF32PhysicalGraphParts, String> {
    let F32PhysicalGraphPackageRequest {
        prepared,
        output,
        a,
        b,
        capacity,
    } = request;
    let prepared_request = prepared.physical_graph_request();
    let operands = prepared.physical_graph_operands();
    if output.cached_ptr() != operands.output
        || a.cached_ptr() != operands.a
        || b.cached_ptr() != operands.b
    {
        return Err("prepared F32 graph package buffer binding changed".into());
    }
    let observer = prepare_f32_physical_graph_observer(ctx, prepared, capacity)?;
    let triad = if prepared.physical_graph_is_direct() {
        super::gemm_bi_triad::prepare_prepared_f32_direct_graph_sequence(ctx, &observer, prepared)?
    } else {
        match prepared_request.op {
            ResolvedGemmOp::Nn => {
                super::gemm_bi_triad::prepare_prepared_f32_forward_graph_sequence(
                    ctx, &observer, prepared, output, a,
                )?
            }
            ResolvedGemmOp::Tn => {
                super::gemm_bi_triad::prepare_prepared_f32_backward_dw_graph_sequence(
                    ctx, &observer, prepared, b, a,
                )?
            }
            ResolvedGemmOp::Nt => {
                super::gemm_bi_triad::prepare_prepared_f32_backward_dx_graph_sequence(
                    ctx, &observer, prepared, output, a,
                )?
            }
        }
    };
    if triad.len() != capacity {
        return Err("prepared F32 graph package launch capacity is not exact".into());
    }
    Ok(PreparedF32PhysicalGraphParts { observer, triad })
}

pub(in crate::mamba_ssm::gpu) fn prepare_f32_physical_graph_package<'a>(
    ctx: &'a GpuCtx,
    request: F32PhysicalGraphPackageRequest<'_>,
    manifest: &PreparedPhysicalCaptureManifest,
) -> Result<PreparedPhysicalGraphPackage<'a>, String> {
    manifest.validate_capture_request(ctx.gemm_route(), request.capacity)?;
    let capacity = request.capacity;
    let PreparedF32PhysicalGraphParts { observer, triad } =
        prepare_f32_physical_graph_parts(ctx, request)?;
    Ok(PreparedPhysicalGraphPackage {
        ctx,
        manifest: manifest.clone(),
        observer: Some(observer),
        prefix: Box::new([]),
        triad: Some(triad),
        suffix: Box::new([]),
        launch_capacity: capacity,
        context_token: ctx.instance_token(),
        stream_token: ctx.stream_token(),
        #[cfg(test)]
        fail_before_enqueue: false,
        #[cfg(test)]
        drift_policy_after_capture: false,
    })
}

pub(in crate::mamba_ssm::gpu) unsafe fn record_prepared_f32_physical_trace(
    ctx: &GpuCtx,
    request: F32PhysicalGraphPackageRequest<'_>,
) -> Result<RecordedPhysicalTrace, String> {
    let PreparedF32PhysicalGraphParts {
        mut observer,
        triad,
    } = prepare_f32_physical_graph_parts(ctx, request)?;
    let mut bound = triad.bind(&ctx.stream)?;
    unsafe { bound.enqueue(&mut observer) }
        .map_err(|error| error.with_driver_context(format_args!("prepared F32 eager enqueue")))?;
    finish_recording_physical_observer(observer, ctx.gemm_route())
}

/// Records one real eager half GEMM route through the production branch body.
///
/// # Safety
///
/// Every raw device pointer must belong to `ctx` and remain live until the
/// stream has completed the enqueued work. The pointer roles follow `op`:
/// NN is `(Y, X, W)`, TN is `(dW, X, dY)`, and NT is `(dX, dY, W)`.
pub(in crate::mamba_ssm::gpu) unsafe fn record_half_physical_trace(
    ctx: &GpuCtx,
    request: HalfPhysicalTraceRequest,
) -> Result<RecordedPhysicalTrace, String> {
    let mut observer = prepare_half_physical_observer(ctx, request)?;
    dispatch_half_physical_request(ctx, request, &mut observer)?;
    finish_recording_physical_observer(observer, ctx.gemm_route())
}

pub(in crate::mamba_ssm::gpu) fn launch_half_production_branch(
    ctx: &GpuCtx,
    request: HalfPhysicalTraceRequest,
) -> Result<HalfPolicyBranchSeal, String> {
    let mut observer = NoPhysicalObserver;
    dispatch_half_physical_request(ctx, request, &mut observer)
}

fn dispatch_half_physical_request<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    request: HalfPhysicalTraceRequest,
    observer: &mut O,
) -> Result<HalfPolicyBranchSeal, String> {
    if request.nn_strides.is_some() && request.forced_tile.is_none() {
        return Err("padded half physical strides require a forced NN route".into());
    }
    let a = TypedPtr {
        ptr: request.a,
        dtype: request.dtype,
    };
    let b = TypedPtr {
        ptr: request.b,
        dtype: request.dtype,
    };
    match (request.op, request.forced_tile) {
        (ResolvedGemmOp::Nn, None) => gemm_bi_forward_typed_in(
            ctx,
            TypedPtr {
                ptr: request.output,
                dtype: request.dtype,
            },
            a,
            b,
            request.bias,
            request.dims,
            observer,
        ),
        (ResolvedGemmOp::Tn, None) => {
            gemm_bi_backward_dw_typed_in(ctx, request.output, b, a, request.dims, observer)
        }
        (ResolvedGemmOp::Nt, None) => gemm_bi_backward_dx_typed_in(
            ctx,
            TypedPtr {
                ptr: request.output,
                dtype: request.dtype,
            },
            a,
            b,
            request.dims,
            observer,
        ),
        (ResolvedGemmOp::Nn, Some(tile)) => {
            let ops = super::gemm_bi_triad::TcFwdOperands {
                y: TypedPtr {
                    ptr: request.output,
                    dtype: request.dtype,
                },
                x: a,
                w: b,
                bias_ptr: request.bias,
            };
            super::gemm_bi_triad::gemm_bi_forward_tc_with_tile_observed(
                ctx,
                observer,
                &ops,
                half_physical_request_shape(request)?,
                tile,
            )
            .map(HalfPolicyBranchSeal::Native)
        }
        (ResolvedGemmOp::Tn, Some(tile)) => {
            super::gemm_bi_triad::gemm_bi_backward_dw_tc_with_tile_observed(
                ctx,
                observer,
                request.output,
                b,
                a,
                request.dims,
                tile,
            )
            .map(HalfPolicyBranchSeal::Native)
        }
        (ResolvedGemmOp::Nt, Some(tile)) => {
            super::gemm_bi_triad::gemm_bi_backward_dx_tc_with_tile_observed(
                ctx,
                observer,
                TypedPtr {
                    ptr: request.output,
                    dtype: request.dtype,
                },
                a,
                b,
                request.dims,
                tile,
            )
            .map(HalfPolicyBranchSeal::Native)
        }
    }
}

/// Full backward: dW (accumulated), dX (overwritten), db (accumulated).
///
/// `grads` = `(dw, db)`. `dims` = `(batch, n_in, n_out)`.
pub fn gpu_gemm_bi_backward_grad_raw(
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
    gpu_gemm_bi_backward_dw_grad(ctx, dw, dy, x_saved, batch, n_in, n_out)?;
    gpu_gemm_bi_backward_dx_raw(ctx, dx, dy, w_ptr, batch, n_in, n_out)?;

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
        WeightDtype::F32 => gpu_gemm_bi_forward_raw(ctx, y, x, w_ptr, bias_ptr, dims),
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
/// Derivation: row-major `Y[B,V] = X[B,D]·E^T[D,V]` ⇔
///             col-major `Y^T[V,B] = E[V,D] · X^T[D,B]`
///   `embed` row-major `[V,D]` = col-major `[D,V]`, OP_T → logical `[V,D]`.
///   `temporal` row-major `[B,D]` = col-major `[D,B]`, OP_N → logical `[D,B]`.
///   Output col-major `[V,B]` = row-major `[B,V]`.
pub fn gpu_gemm_bi_tied_lm_head_raw(
    ctx: &GpuCtx,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    d_model: usize,
    vocab_padded: usize,
) -> Result<(), String> {
    gpu_gemm_bi_tied_lm_head_blas(
        &ctx.blas,
        logits_ptr,
        temporal_ptr,
        embed_ptr,
        batch,
        d_model,
        vocab_padded,
    )
}

/// No-context twin of `gpu_gemm_bi_tied_lm_head_raw` — takes only the cuBLAS
/// handle so callers without a `GpuCtx` (e.g., Mamba-3 LLM wrapper) can use
/// the same OP_T row-major trick without synthesizing a context.
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
    unsafe {
        cudarc::cublas::result::sgemm(
            *blas.handle(),
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

#[cfg(test)]
mod physical_graph_tests {
    use super::*;
    #[test]
    fn f32_physical_ranges_omit_unread_zero_reduction_operands() {
        use super::super::gemm_bi_triad::{F32TriadOperands, F32TriadRequest, F32TriadShape};
        for (op, dims, expected_output_bytes) in [
            (ResolvedGemmOp::Nn, (3, 0, 5), 60),
            (ResolvedGemmOp::Tn, (0, 4, 5), 80),
            (ResolvedGemmOp::Nt, (3, 4, 0), 48),
        ] {
            for bias in [None, Some(0x4000)]
                .into_iter()
                .take(if op == ResolvedGemmOp::Nn { 2 } else { 1 })
            {
                let request = F32TriadRequest {
                    op,
                    shape: F32TriadShape::contiguous(op, dims),
                };
                let operands = F32TriadOperands {
                    output: 0x1000,
                    a: 0,
                    b: 0,
                    bias,
                    alpha: 1.0,
                    beta: 0.0,
                };
                // Raw A/B really are null: this epilogue-only kernel must not
                // require allocation identity for either unread operand.
                let ranges = f32_physical_argument_ranges(request, operands)
                    .unwrap_or_else(|error| panic!("{op:?} {dims:?} bias={bias:?}: {error}"));
                let observed: Vec<_> = ranges
                    .iter()
                    .map(|r| (r.pointer, r.required_bytes))
                    .collect();
                let mut expected = vec![(0x1000, expected_output_bytes)];
                if bias.is_some() {
                    expected.push((0x4000, (dims.2 * 4) as u64));
                }
                assert_eq!(observed, expected);
                assert_eq!(ranges.len(), ranges.capacity());
            }
        }
    }

    #[test]
    fn f32_physical_ranges_retain_nonempty_operand_bounds() {
        use super::super::gemm_bi_triad::{F32TriadOperands, F32TriadRequest, F32TriadShape};
        for (op, sizes) in [
            (ResolvedGemmOp::Nn, [60, 48, 80]),
            (ResolvedGemmOp::Tn, [80, 48, 60]),
            (ResolvedGemmOp::Nt, [48, 60, 80]),
        ] {
            let request = F32TriadRequest {
                op,
                shape: F32TriadShape::contiguous(op, (3, 4, 5)),
            };
            let operands = F32TriadOperands {
                output: 0x1000,
                a: 0x2000,
                b: 0x3000,
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            };
            let ranges = f32_physical_argument_ranges(request, operands).unwrap();
            assert_eq!(
                ranges
                    .iter()
                    .map(|r| (r.pointer, r.required_bytes))
                    .collect::<Vec<_>>(),
                vec![(0x1000, sizes[0]), (0x2000, sizes[1]), (0x3000, sizes[2])]
            );
            assert_eq!(ranges.len(), ranges.capacity());
            let mut invalid = request;
            invalid.shape.lda = 0;
            assert!(f32_physical_argument_ranges(invalid, operands).is_err());
        }
        assert!(physical_f32_storage_elements(3, 0, 0, "active").is_err());
        assert!(physical_f32_storage_elements(0, 4, 4, "active").is_err());
        assert!(physical_f32_storage_elements(3, 4, 3, "active").is_err());
        assert!(physical_f32_storage_elements(usize::MAX, 4, 4, "active").is_err());
    }

    use crate::mamba_ssm::gpu::buffers::DtypedBuf;
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy};
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::graph_capture::{
        CapturedPhysicalGraph, capture_into_graph, capture_into_graph_with_physical_plan,
    };
    use crate::mamba_ssm::gpu::kernel_identity::PreparedPhysicalCaptureManifest;

    fn half_branch_is_sm120(branch: HalfPolicyBranchSeal) -> bool {
        matches!(branch, HalfPolicyBranchSeal::Sm120(_))
    }

    #[test]
    fn half_policy_has_a_distinct_sm120_branch_seal() {
        let projection: fn(HalfPolicyBranchSeal) -> bool = half_branch_is_sm120;
        assert_eq!(
            std::mem::size_of_val(&projection),
            std::mem::size_of::<usize>()
        );
    }

    #[test]
    #[ignore = "requires a CC 12.0 CUDA device and NVRTC"]
    fn sm120_auto_cache_rejects_managed_epoch_aba_during_capture() {
        let ctx = physical_graph_context();
        if ctx.compute_capability() != (12, 0) {
            return;
        }
        ctx.set_bi_tensor_cores(true);
        let route = super::super::gemm_bi_triad::SM120_AUTO_CELLS_CC120
            .iter()
            .copied()
            .find(|route| {
                route.op == super::super::gemm_bi_triad::Sm120Op::Nn
                    && route.dtype == WeightDtype::Bf16
            })
            .expect("qualified CC12.0 BF16 NN route");
        let dims = (route.shape.m, route.shape.k, route.shape.n);
        let mut buffers =
            PhysicalGraphBuffers::new_for_op(&ctx, WeightDtype::Bf16, dims, ResolvedGemmOp::Nn)
                .unwrap();
        let output_ptr = buffers.output.cached_ptr();
        let a_ptr = buffers.a.cached_ptr();
        let b_ptr = buffers.b.cached_ptr();
        let launch = || {
            gemm_bi_forward_typed(
                &ctx,
                TypedPtr {
                    ptr: output_ptr,
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: a_ptr,
                    dtype: WeightDtype::Bf16,
                },
                TypedPtr {
                    ptr: b_ptr,
                    dtype: WeightDtype::Bf16,
                },
                0,
                dims,
            )
        };
        launch().expect("warm automatic SM120 cache");
        ctx.stream.synchronize().expect("finish SM120 warmup");
        buffers
            .output
            .replace_managed_allocation_generation_for_test()
            .expect("replace managed output generation at the same address");
        let error = match unsafe { capture_into_graph(&ctx.stream, launch) } {
            Ok(_) => panic!("managed-generation ABA unexpectedly captured"),
            Err(error) => error,
        };
        assert_eq!(
            error.strip_prefix("body: ").unwrap_or(&error),
            "prepared SM120 Triad allocation epoch changed during graph capture; run eager warmup again"
        );
    }

    struct PhysicalGraphBuffers {
        output: DtypedBuf,
        a: DtypedBuf,
        b: DtypedBuf,
    }

    impl PhysicalGraphBuffers {
        fn new(
            ctx: &GpuCtx,
            dtype: WeightDtype,
            dims: (usize, usize, usize),
        ) -> Result<Self, String> {
            Self::new_for_op(ctx, dtype, dims, ResolvedGemmOp::Nn)
        }

        fn new_for_op(
            ctx: &GpuCtx,
            dtype: WeightDtype,
            dims: (usize, usize, usize),
            op: ResolvedGemmOp,
        ) -> Result<Self, String> {
            let (m, k, n) = dims;
            let (output_len, output_dtype, a_len, b_len) = match op {
                ResolvedGemmOp::Nn => (m * n, dtype, m * k, k * n),
                ResolvedGemmOp::Tn => (k * n, WeightDtype::F32, m * k, m * n),
                ResolvedGemmOp::Nt => (m * k, dtype, m * n, k * n),
            };
            Ok(Self {
                output: DtypedBuf::zeros(&ctx.stream, output_len, output_dtype)?,
                a: DtypedBuf::zeros(&ctx.stream, a_len, dtype)?,
                b: DtypedBuf::zeros(&ctx.stream, b_len, dtype)?,
            })
        }

        fn request(
            &self,
            dtype: WeightDtype,
            dims: (usize, usize, usize),
            capacity: usize,
        ) -> HalfPhysicalTraceRequest {
            self.request_for_op(dtype, dims, ResolvedGemmOp::Nn, capacity)
        }

        fn request_for_op(
            &self,
            dtype: WeightDtype,
            dims: (usize, usize, usize),
            op: ResolvedGemmOp,
            capacity: usize,
        ) -> HalfPhysicalTraceRequest {
            HalfPhysicalTraceRequest {
                op,
                output: self.output.cached_ptr(),
                a: self.a.cached_ptr(),
                b: self.b.cached_ptr(),
                bias: 0,
                dtype,
                dims,
                nn_strides: None,
                forced_tile: None,
                capacity,
            }
        }
    }

    struct F32PhysicalGraphBuffers {
        output: GpuBuffer,
        a: GpuBuffer,
        b: GpuBuffer,
    }

    impl F32PhysicalGraphBuffers {
        fn new(ctx: &GpuCtx, dims: (usize, usize, usize)) -> Result<Self, String> {
            let (m, k, n) = dims;
            Ok(Self {
                output: GpuBuffer::zeros(&ctx.stream, m * n)?,
                a: GpuBuffer::zeros(&ctx.stream, m * k)?,
                b: GpuBuffer::zeros(&ctx.stream, k * n)?,
            })
        }

        fn operands(&self) -> super::super::gemm_bi_triad::F32TriadOperands {
            super::super::gemm_bi_triad::F32TriadOperands {
                output: self.output.cached_ptr(),
                a: self.a.cached_ptr(),
                b: self.b.cached_ptr(),
                bias: None,
                alpha: 1.0,
                beta: 0.0,
            }
        }

        fn package_request<'a>(
            &'a mut self,
            prepared: &'a super::super::gemm_bi_triad::PreparedF32TriadLaunch,
            capacity: usize,
        ) -> F32PhysicalGraphPackageRequest<'a> {
            F32PhysicalGraphPackageRequest {
                prepared,
                output: &mut self.output,
                a: &self.a,
                b: &self.b,
                capacity,
            }
        }
    }

    fn physical_graph_context() -> GpuCtx {
        let device = GpuDevice::new(0).expect("CUDA device for physical graph test");
        let ctx = GpuCtx::new(&device).expect("GPU context for physical graph test");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx
    }

    fn eager_manifest(
        ctx: &GpuCtx,
        request: HalfPhysicalTraceRequest,
    ) -> Result<PreparedPhysicalCaptureManifest, String> {
        let trace = unsafe { record_half_physical_trace(ctx, request) }?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize eager physical graph trace: {error:?}"))?;
        Ok(trace.manifest())
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn half_physical_trace_arguments_track_base_offsets_without_device_addresses() {
        let ctx = physical_graph_context();
        ctx.set_bi_tensor_cores(false);
        let dims = (64, 96, 80);
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let width = dtype.size_bytes() as u64;
            let output = DtypedBuf::zeros(&ctx.stream, dims.0 * dims.2 + 1, dtype).unwrap();
            let a = DtypedBuf::zeros(&ctx.stream, dims.0 * dims.1 + 1, dtype).unwrap();
            let b = DtypedBuf::zeros(&ctx.stream, dims.1 * dims.2 + 1, dtype).unwrap();
            let record = |offset: u64| unsafe {
                record_half_physical_trace(
                    &ctx,
                    HalfPhysicalTraceRequest {
                        op: ResolvedGemmOp::Nn,
                        output: output.cached_ptr() + offset,
                        a: a.cached_ptr() + offset,
                        b: b.cached_ptr() + offset,
                        bias: 0,
                        dtype,
                        dims,
                        nn_strides: None,
                        forced_tile: None,
                        capacity: 1,
                    },
                )
            };
            let base = record(0).unwrap();
            let offset = record(width).unwrap();
            ctx.stream.synchronize().unwrap();
            assert_ne!(
                base.nodes()[0].launch().arguments_digest,
                offset.nodes()[0].launch().arguments_digest,
                "{dtype:?}"
            );
        }
    }

    unsafe fn capture_case(
        ctx: &GpuCtx,
        request: HalfPhysicalTraceRequest,
        manifest: &PreparedPhysicalCaptureManifest,
    ) -> Result<CapturedPhysicalGraph, String> {
        let package = prepare_half_physical_graph_package(ctx, request, manifest)?;
        unsafe { capture_into_graph_with_physical_plan(package) }
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn physical_graph_captures_native_half_and_exact_typed_fallback() {
        let cases = [
            (ResolvedGemmOp::Nn, true, (128, 128, 128), 1),
            (ResolvedGemmOp::Tn, true, (128, 128, 128), 1),
            (ResolvedGemmOp::Nt, true, (128, 128, 128), 1),
            (ResolvedGemmOp::Nn, false, (64, 384, 512), 5),
            (ResolvedGemmOp::Tn, false, (64, 384, 512), 3),
            (ResolvedGemmOp::Nt, false, (64, 384, 512), 6),
        ];
        for (op, tensor_cores, dims, capacity) in cases {
            let ctx = physical_graph_context();
            ctx.set_bi_tensor_cores(tensor_cores);
            let buffers =
                PhysicalGraphBuffers::new_for_op(&ctx, WeightDtype::Bf16, dims, op).unwrap();
            let request = buffers.request_for_op(WeightDtype::Bf16, dims, op, capacity);
            let manifest = eager_manifest(&ctx, request).unwrap();
            assert_eq!(manifest.launch_capacity(), capacity, "{op:?} {dims:?}");
            let graph = unsafe { capture_case(&ctx, request, &manifest) }.unwrap();
            assert_eq!(graph.nodes(), manifest.nodes());
            assert_eq!(
                graph.launches(),
                super::super::kernel_identity::ResolvedPhysicalLaunchSet::from_nodes(
                    manifest.nodes()
                )
                .unwrap()
            );
            graph.launch(&ctx, "physical graph success").unwrap();
            ctx.stream.synchronize().unwrap();
        }
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn physical_graph_captures_prepared_f32_exact_and_tf32() {
        let dims = (128, 128, 128);
        for forced_tf32 in [false, true] {
            let ctx = physical_graph_context();
            ctx.set_f32_triad_policy(if forced_tf32 {
                F32TriadPolicy::AllowDeterministicTf32V1
            } else {
                F32TriadPolicy::ExactScalarFmaV1
            });
            let mut buffers = F32PhysicalGraphBuffers::new(&ctx, dims).unwrap();
            let request = super::super::gemm_bi_triad::F32TriadRequest {
                op: ResolvedGemmOp::Nn,
                shape: super::super::gemm_bi_triad::F32TriadShape::contiguous(
                    ResolvedGemmOp::Nn,
                    dims,
                ),
            };
            let operands = buffers.operands();
            let prepared = if forced_tf32 {
                let availability = ctx.kernels.f32_triad_availability();
                let module_kind = availability
                    .specialized
                    .or(availability.portable)
                    .expect("qualified TF32 module")
                    .module_kind;
                let specs: &[super::super::gemm_bi_triad::Tf32KernelSpec] = match module_kind {
                    ModuleKind::TriadSm80 => &super::super::gemm_bi_triad::SM80_TF32_ROUTE_SPECS,
                    ModuleKind::TriadSm90a => &super::super::gemm_bi_triad::SM90A_TF32_ROUTE_SPECS,
                    ModuleKind::TriadSm100 => &super::super::gemm_bi_triad::SM100_TF32_ROUTE_SPECS,
                    ModuleKind::TriadSm120 => &super::super::gemm_bi_triad::SM120_TF32_ROUTE_SPECS,
                    _ => panic!("non-Triad TF32 module {module_kind:?}"),
                };
                let route = specs
                    .iter()
                    .find(|spec| spec.op == ResolvedGemmOp::Nn)
                    .expect("NN TF32 route")
                    .route;
                super::super::gemm_bi_triad::prepare_f32_triad_forced(
                    &ctx, request, operands, route,
                )
                .unwrap()
            } else {
                super::super::gemm_bi_triad::prepare_f32_triad(&ctx, request, operands).unwrap()
            };
            let capacity = prepared.physical_graph_launch_count();
            let trace = unsafe {
                record_prepared_f32_physical_trace(
                    &ctx,
                    buffers.package_request(&prepared, capacity),
                )
            }
            .unwrap();
            ctx.stream.synchronize().unwrap();
            let manifest = trace.manifest();
            assert_eq!(manifest.launch_capacity(), capacity);
            let package = prepare_f32_physical_graph_package(
                &ctx,
                buffers.package_request(&prepared, capacity),
                &manifest,
            )
            .unwrap();
            let graph = unsafe { capture_into_graph_with_physical_plan(package) }.unwrap();
            assert_eq!(graph.nodes(), manifest.nodes());
            graph.launch(&ctx, "prepared F32 physical graph").unwrap();
            ctx.stream.synchronize().unwrap();
        }
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn physical_graph_rejects_a_valid_unobserved_raw_cuda_node() {
        let ctx = physical_graph_context();
        ctx.set_bi_tensor_cores(true);
        let dims = (128, 128, 128);
        let buffers = PhysicalGraphBuffers::new(&ctx, WeightDtype::Bf16, dims).unwrap();
        let request = buffers.request(WeightDtype::Bf16, dims, 1);
        let manifest = eager_manifest(&ctx, request).unwrap();
        let package = prepare_half_physical_graph_package(&ctx, request, &manifest).unwrap();
        assert_eq!(package.launch_capacity, manifest.launch_capacity());
        let graph = unsafe { capture_into_graph_with_physical_plan(package) }.unwrap();
        assert_eq!(graph.nodes(), manifest.nodes());
        graph
            .launch(&ctx, "physical graph excludes raw-node escape")
            .unwrap();
        ctx.stream.synchronize().unwrap();
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn physical_graph_rejects_failures_and_replay_drift_then_reuses_stream() {
        let ctx = physical_graph_context();
        ctx.set_bi_tensor_cores(true);
        let dims = (128, 128, 128);
        let buffers = PhysicalGraphBuffers::new(&ctx, WeightDtype::Bf16, dims).unwrap();
        let request = buffers.request(WeightDtype::Bf16, dims, 1);
        let manifest = eager_manifest(&ctx, request).unwrap();

        let mut body_failure =
            prepare_half_physical_graph_package(&ctx, request, &manifest).unwrap();
        body_failure.inject_body_failure();
        let error = match unsafe { capture_into_graph_with_physical_plan(body_failure) } {
            Ok(_) => panic!("body failure returned a physical graph"),
            Err(error) => error,
        };
        assert!(
            error.contains("expected prepared physical body error"),
            "{error}"
        );

        let mut driver_failure =
            prepare_half_physical_graph_package(&ctx, request, &manifest).unwrap();
        driver_failure.inject_driver_failure();
        let driver_error = match unsafe { capture_into_graph_with_physical_plan(driver_failure) } {
            Ok(_) => panic!("Driver failure returned a physical graph"),
            Err(error) => error,
        };
        assert!(
            driver_error.contains("CUDA") || driver_error.contains("Driver"),
            "{driver_error}"
        );

        let stale_package = prepare_half_physical_graph_package(&ctx, request, &manifest).unwrap();
        let graph = unsafe { capture_case(&ctx, request, &manifest) }.unwrap();
        ctx.set_bi_tensor_cores(false);
        assert!(graph.launch(&ctx, "changed physical policy").is_err());
        ctx.set_bi_tensor_cores(true);

        let other = physical_graph_context();
        other.set_bi_tensor_cores(true);
        let other_buffers = PhysicalGraphBuffers::new(&other, WeightDtype::Bf16, dims).unwrap();
        let other_request = other_buffers.request(WeightDtype::Bf16, dims, 1);
        let other_package =
            prepare_half_physical_graph_package(&other, other_request, &manifest).unwrap();
        let other_capture_error =
            match unsafe { capture_into_graph_with_physical_plan(other_package) } {
                Ok(_) => panic!("a different context returned a physical graph"),
                Err(error) => error,
            };
        assert!(
            other_capture_error.contains("context instance"),
            "{other_capture_error}"
        );
        assert!(graph.launch(&other, "changed physical context").is_err());

        drop(buffers.a);
        let stale_capture_error =
            match unsafe { capture_into_graph_with_physical_plan(stale_package) } {
                Ok(_) => panic!("stale allocation generation returned a physical graph"),
                Err(error) => error,
            };
        assert!(
            stale_capture_error.contains("allocation generation"),
            "{stale_capture_error}"
        );
        assert!(graph.launch(&ctx, "changed allocation generation").is_err());
        drop(graph);

        let recovered = PhysicalGraphBuffers::new(&ctx, WeightDtype::Bf16, dims).unwrap();
        let recovered_request = recovered.request(WeightDtype::Bf16, dims, 1);
        let recovered_manifest = eager_manifest(&ctx, recovered_request).unwrap();
        let recovered_graph =
            unsafe { capture_case(&ctx, recovered_request, &recovered_manifest) }.unwrap();
        recovered_graph
            .launch(&ctx, "recovered physical graph")
            .unwrap();
        ctx.stream.synchronize().unwrap();
    }

    #[test]
    #[ignore = "requires a CUDA device and NVRTC"]
    fn physical_graph_rejects_capture_and_post_capture_identity_drift() {
        let ctx = physical_graph_context();
        ctx.set_bi_tensor_cores(true);
        let dims = (128, 128, 128);
        let eager_buffers = PhysicalGraphBuffers::new(&ctx, WeightDtype::Bf16, dims).unwrap();
        let eager_request = eager_buffers.request(WeightDtype::Bf16, dims, 1);
        let manifest = eager_manifest(&ctx, eager_request).unwrap();

        let changed_buffers = PhysicalGraphBuffers::new(&ctx, WeightDtype::Bf16, dims).unwrap();
        let changed_request = changed_buffers.request(WeightDtype::Bf16, dims, 1);
        let changed_package =
            prepare_half_physical_graph_package(&ctx, changed_request, &manifest).unwrap();
        let changed_arguments =
            match unsafe { capture_into_graph_with_physical_plan(changed_package) } {
                Ok(_) => panic!("changed arguments returned a physical graph"),
                Err(error) => error,
            };
        assert!(changed_arguments.contains("exact"), "{changed_arguments}");

        let mut drift_package =
            prepare_half_physical_graph_package(&ctx, eager_request, &manifest).unwrap();
        drift_package.inject_post_capture_policy_drift();
        let post_capture_policy =
            match unsafe { capture_into_graph_with_physical_plan(drift_package) } {
                Ok(_) => panic!("post-capture policy drift returned a physical graph"),
                Err(error) => error,
            };
        assert!(
            post_capture_policy.contains("changed since capture"),
            "{post_capture_policy}"
        );
        ctx.set_bi_tensor_cores(true);

        let oversized_request = HalfPhysicalTraceRequest {
            capacity: 2,
            ..eager_request
        };
        let capacity_error =
            prepare_half_physical_graph_package(&ctx, oversized_request, &manifest)
                .err()
                .expect("wrong observer capacity must reject the package");
        assert!(capacity_error.contains("capacity"), "{capacity_error}");

        let graph = unsafe { capture_case(&ctx, eager_request, &manifest) }.unwrap();
        graph
            .launch(&ctx, "recovered exact physical graph")
            .unwrap();
        ctx.stream.synchronize().unwrap();
    }
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

/// Half-precision twin of `gpu_gemm_bi_tied_lm_head_raw` for bf16/f16 embed.
/// `temporal_ptr` input activations must already be in `dtype` (not f32).
pub fn gpu_gemm_ex_tied_lm_head_raw(
    ctx: &GpuCtx,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    gpu_gemm_ex_tied_lm_head_blas(&ctx.blas, logits_ptr, temporal_ptr, embed_ptr, dtype, dims)
}

/// No-context twin of `gpu_gemm_ex_tied_lm_head_raw` — blas-only variant.
pub fn gpu_gemm_ex_tied_lm_head_blas(
    blas: &cudarc::cublas::CudaBlas,
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
            *blas.handle(),
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
            dtype.compute_type(), // blas-only twin: no ctx, stays PEDANTIC (fast_gemm covers ctx paths)
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
/// For `WeightDtype::F32`, this is mathematically identical to `gpu_gemm_bi_forward_raw`
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
            w.dtype.compute_type(), // blas-only twin: no ctx, stays PEDANTIC (fast_gemm covers ctx paths)
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed (no-bias) failed: {e:?}"))?;
    }
    Ok(())
}

/// Pick the batch-invariant GEMM kernel for given I/O dtypes. Returns
/// `None` if we should fall back to cuBLAS (e.g. mixed bf16/f32 combos
/// we didn't compile — currently only homogeneous I/O paths have a
/// batch-invariant kernel).
fn pick_bi_gemm(
    ctx: &GpuCtx,
    a_dtype: WeightDtype,
    b_dtype: WeightDtype,
    c_dtype: WeightDtype,
) -> Option<(&cudarc::driver::CudaFunction, u32)> {
    if a_dtype != b_dtype {
        return None;
    }
    match (a_dtype, c_dtype) {
        (WeightDtype::Bf16, WeightDtype::Bf16) => Some((&ctx.kernels.gemm_bi_bf16_bf16, 256)),
        (WeightDtype::F16, WeightDtype::F16) => Some((&ctx.kernels.gemm_bi_f16_f16, 256)),
        (WeightDtype::Bf16, WeightDtype::F32) => Some((&ctx.kernels.gemm_bi_bf16_f32, 256)),
        (WeightDtype::F16, WeightDtype::F32) => Some((&ctx.kernels.gemm_bi_f16_f32, 256)),
        (WeightDtype::F32, WeightDtype::F32) => Some((&ctx.kernels.gemm_bi_f32_f32_s2, 128)),
        _ => None,
    }
}

/// Arguments for the batch-invariant GEMM kernel. All row-major:
///   A: `[m, k]` stride `k`
///   B: `[k, n]` stride `n`
///   C: `[m, n]` stride `n`
/// `bias`: nullable `[n]` f32. Pass `0` for "no bias".
struct BiGemmArgs {
    c: cudarc::driver::sys::CUdeviceptr,
    a: cudarc::driver::sys::CUdeviceptr,
    b: cudarc::driver::sys::CUdeviceptr,
    bias: cudarc::driver::sys::CUdeviceptr,
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
}

// The WMMA GEMM path stays registered in MambaKernels and is reachable
// through gemm_bi_forward_raw (the Fixed family's entry and the f32
// dispatch arm).
fn launch_bi_gemm(
    ctx: &GpuCtx,
    kernel: &cudarc::driver::CudaFunction,
    threads: u32,
    args: BiGemmArgs,
) -> Result<(), String> {
    // The selected kernel supplies its qualified thread count. Every
    // variant below still owns a 64x64 output tile; a mismatched block
    // size can return plausible garbage rather than a launch error.
    // Static smem only - shared_mem_bytes stays 0 here; the dynamic
    // K-buffer belongs to launch_bi_matvec alone.
    const BLOCK_M: i32 = 64;
    const BLOCK_N: i32 = 64;
    let num_pid_m = (args.m + BLOCK_M - 1) / BLOCK_M;
    let num_pid_n = (args.n + BLOCK_N - 1) / BLOCK_N;
    let grid = (num_pid_m as u32) * (num_pid_n as u32);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let lda = args.k;
    let ldb = args.n;
    let ldc = args.n;
    let mut builder = ctx.stream.launch_builder(kernel);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&args.alpha);
    builder.arg(&args.beta);
    builder.arg(&args.m);
    builder.arg(&args.n);
    builder.arg(&args.k);
    builder.arg(&lda);
    builder.arg(&ldb);
    builder.arg(&ldc);
    unsafe { builder.launch(cfg) }.map_err(|e| format!("gemm_bi launch failed: {e:?}"))?;
    Ok(())
}

/// Direct entry to the Fixed batch-invariant GEMM ladder
/// (`kernels/gemm_bi_fixed/`, `gemm_bi_*`). Every rung uses `SPLIT_K=1`
/// and preserves its architecture-specific bit family across scheduling
/// choices. Forward-only NN, f32/bf16/f16.
pub fn gemm_bi_forward_raw(
    ctx: &GpuCtx,
    c: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    super::gemm_bi_fixed::fixed_forward(ctx, c, x, w, bias_ptr, dims).map(|_| ())
}

/// The fixed family's LEGACY tile (64x64x32, strict, no buckets): the
/// narrow-N fallback of the inference ladder and the whole f32 arm (the
/// shipped serve route - its bits never move with ladder work).
pub(crate) fn fixed_legacy_forward(
    ctx: &GpuCtx,
    c: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    let Some((kernel, threads)) = pick_bi_gemm(ctx, x.dtype, w.dtype, c.dtype) else {
        return Err(format!(
            "gemm_bi: no kernel for operand dtypes a={:?} b={:?} c={:?}",
            x.dtype, w.dtype, c.dtype
        ));
    };
    launch_bi_gemm(
        ctx,
        kernel,
        threads,
        BiGemmArgs {
            c: c.ptr,
            a: x.ptr,
            b: w.ptr,
            bias: bias_ptr.unwrap_or(0),
            alpha: 1.0,
            beta: 0.0,
            m: batch as i32,
            n: n_out as i32,
            k: n_in as i32,
        },
    )
}

/// Pick the M=1 matvec kernel — much faster than gemm_bi at M=1 because
/// the GEMM tile wastes 98% of smem bandwidth on zero-padding at M=1.
fn pick_bi_matvec(
    ctx: &GpuCtx,
    a_dtype: WeightDtype,
    b_dtype: WeightDtype,
    c_dtype: WeightDtype,
) -> Option<&cudarc::driver::CudaFunction> {
    if a_dtype != b_dtype {
        return None;
    }
    match (a_dtype, c_dtype) {
        (WeightDtype::Bf16, WeightDtype::Bf16) => Some(&ctx.kernels.matvec_bi_bf16_bf16),
        (WeightDtype::F16, WeightDtype::F16) => Some(&ctx.kernels.matvec_bi_f16_f16),
        (WeightDtype::Bf16, WeightDtype::F32) => Some(&ctx.kernels.matvec_bi_bf16_f32),
        (WeightDtype::F16, WeightDtype::F32) => Some(&ctx.kernels.matvec_bi_f16_f32),
        (WeightDtype::F32, WeightDtype::F32) => Some(&ctx.kernels.matvec_bi_f32_f32),
        _ => None,
    }
}

fn launch_bi_matvec(
    ctx: &GpuCtx,
    kernel: &cudarc::driver::CudaFunction,
    args: BiGemmArgs,
    io_dtype: WeightDtype,
) -> Result<(), String> {
    // Must match kernel constants in kernels/gemm_bi_fixed/:
    //   BLOCK_N_MV = 32, WARPS_PER_BLOCK = 8, THREADS_PER_BLOCK = 256
    // Grid is 2D: (ceil(N / BLOCK_N_MV), M) — one CTA per (m_row, col_chunk).
    const BLOCK_N_MV: i32 = 32;
    const THREADS_PER_BLOCK: i32 = 256;
    let a_bytes = (args.k as u32) * (io_dtype.size_bytes() as u32);
    let smem_bytes = (a_bytes + 15) & !15;
    let num_pid_n = (args.n + BLOCK_N_MV - 1) / BLOCK_N_MV;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_pid_n as u32, args.m as u32, 1),
        block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let lda = args.k;
    let ldb = args.n;
    let ldc = args.n;
    let mut builder = ctx.stream.launch_builder(kernel);
    builder.arg(&args.c);
    builder.arg(&args.a);
    builder.arg(&args.b);
    builder.arg(&args.bias);
    builder.arg(&args.alpha);
    builder.arg(&args.beta);
    builder.arg(&args.m);
    builder.arg(&args.n);
    builder.arg(&args.k);
    builder.arg(&lda);
    builder.arg(&ldb);
    builder.arg(&ldc);
    unsafe { builder.launch(cfg) }.map_err(|e| format!("matvec_bi launch failed: {e:?}"))?;
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

    // Dispatch:
    //   M=1   → batch-invariant matvec (decode hot path; ~1000 tok/s
    //           target, same as cuBLAS gemv, PLUS trivially deterministic
    //           because M=1 has no batch dim).
    //   M≥2   → cuBLAS GemmEx (fast Tensor-Core path; deterministic for
    //           fixed M within a process — sufficient for fixed-batch RL
    //           and prefill workloads).
    //
    // The `gemm_bi_*` WMMA kernels are registered but not in the default
    // path — they hit ~30% of cuBLAS throughput in this form; the
    // fixed-tile family is the fast deterministic path. Keep the
    // reference alive for the compiler.
    let _ = pick_bi_gemm(ctx, x.dtype, w.dtype, c.dtype);

    // The matvec kernel handles any M ≥ 1 via a 2D grid (CTA per
    // (m_row, col_chunk)) and gives strict cross-batch bit-identity.
    // Opt-in only — default is cuBLAS gemv for maximum throughput.
    // Enable via `ctx.set_batch_invariant(true)` or the
    // `MAMBA_RS_BATCH_INVARIANT=1` environment variable.
    // Typed gemm_bi, homogeneous bf16/f16 operand triples only.
    // Routing:
    //   - TC tier ON, N >= 32: the forward tile ladder
    //     (Thin16/Tile64/Tile128 — bit-identical per output element)
    //     covers EVERY M, so one arithmetic family serves decode and
    //     prefill alike and a row's bits never depend on M. This removes
    //     the old matvec/TC family break at M=128 (the invariance-matrix
    //     bucket edge) at a measured M=1 cost of ~1.4-1.9x vs matvec
    //     (thin_rung_decode_bench); from M=4 the ladder is FASTER.
    //   - scalar tier (TC off), M >= 128: full-coverage typed entry —
    //     native typed buckets, else upcast → f32 gemm_bi → RNE
    //     downcast. Bit-identical by contract.
    //   - scalar tier M < 128, and N < 32 on either tier: matvec_bi
    //     below — one reduction order for every M within its band.
    // Inference batch-parity asserts STRICT all-M bit-invariance of
    // decode logits (KL ~1e-12 across batch sizes); both the ladder and
    // matvec hold it — each is one reduction order for every M it serves.
    // Mixed a/b dtype combos have NO matvec_bi kernel (the a==b guard in
    // pick_bi_matvec): under the batch-invariant contract they FAIL LOUD
    // below instead of silently taking non-deterministic cuBLAS.
    // Family selector: the fixed-tile family serves the typed forward
    // whole (its Tensor-Core instantiation covers bf16/f16), so it is
    // tried before the triad's buckets.
    if ctx.batch_invariant() && ctx.bi_gemm_family() == super::context::BiGemmFamily::Fixed {
        return gemm_bi_forward_raw(ctx, c, x, w, bias_ptr, dims);
    }

    let homogeneous_half = c.dtype != WeightDtype::F32 && c.dtype == x.dtype && x.dtype == w.dtype;
    if ctx.batch_invariant() && homogeneous_half && n_out >= 2 {
        let tc_ladder = ctx.bi_tensor_cores() && n_out >= 32;
        if tc_ladder || batch >= 128 {
            return gemm_bi_forward_typed(ctx, c, x, w, bias_ptr.unwrap_or(0), dims);
        }
    }

    if ctx.batch_invariant()
        && let Some(kernel) = pick_bi_matvec(ctx, x.dtype, w.dtype, c.dtype)
    {
        let bias_arg = bias_ptr.unwrap_or(0);
        return launch_bi_matvec(
            ctx,
            kernel,
            BiGemmArgs {
                c: c.ptr,
                a: x.ptr,
                b: w.ptr,
                bias: bias_arg,
                alpha: 1.0,
                beta: 0.0,
                m: batch as i32,
                n: n_out as i32,
                k: n_in as i32,
            },
            x.dtype,
        );
    }

    if ctx.batch_invariant() {
        // No deterministic kernel covers this operand triple; falling
        // through would silently run non-PEDANTIC cuBLAS in the build
        // that claims determinism.
        return Err(format!(
            "batch-invariant GEMM: no deterministic kernel for operand dtypes \
             a={:?} b={:?} c={:?} at m={batch} - mixed a/b dtypes have no \
             matvec_bi variant",
            x.dtype, w.dtype, c.dtype
        ));
    }

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
            effective_compute(ctx, w.dtype),
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed failed: {e:?}"))?;
    }

    Ok(())
}
