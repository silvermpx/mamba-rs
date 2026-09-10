//! cuBLAS SGEMM wrappers for GPU training.
//!
//! All matrices are row-major in our code. cuBLAS is column-major.
//!
//! The standard trick: for row-major C = A @ B, call cuBLAS with:
//!   C^T = B^T @ A^T  (in cuBLAS column-major convention)
//!   gemm(N, N, n_out, batch, n_in, 1.0, W, n_out, X, n_in, beta, Y, n_out)

use super::buffers::{
    GpuBuffer, GradSlice, ManagedAllocationEpochStamp, managed_allocation_epoch_for_ranges,
};
use super::context::{GemmMode, GpuCtx};
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

/// Test-only observation at the actual vendor GEMM FFI boundaries. A scoped
/// guard owns thread-local state; nested guards are rejected without changing
/// the enclosing guard, and Drop restores it on errors and unwinding.
#[cfg(test)]
pub(crate) mod vendor_gemm_test {
    use std::{cell::Cell, marker::PhantomData, rc::Rc};

    #[derive(Clone, Copy)]
    struct State {
        deny: bool,
        calls: usize,
    }

    thread_local! {
        static STATE: Cell<Option<State>> = const { Cell::new(None) };
    }

    pub(crate) struct Guard {
        _thread: PhantomData<Rc<()>>,
    }

    impl Guard {
        pub(crate) fn new(deny: bool) -> Result<Self, String> {
            STATE.with(|state| {
                if state.get().is_some() {
                    return Err("nested vendor GEMM test guard".into());
                }
                state.set(Some(State { deny, calls: 0 }));
                Ok(Self {
                    _thread: PhantomData,
                })
            })
        }

        pub(crate) fn calls(&self) -> usize {
            STATE.with(|state| state.get().expect("active vendor guard").calls)
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            STATE.with(|state| state.set(None));
        }
    }

    pub(super) fn boundary() -> Result<(), String> {
        STATE.with(|state| {
            if let Some(mut current) = state.get() {
                current.calls += 1;
                state.set(Some(current));
                if current.deny {
                    return Err("vendor GEMM denied before FFI".into());
                }
            }
            Ok(())
        })
    }

    #[test]
    fn vendor_gemm_guard_restores_on_error_unwind_and_rejects_nesting() {
        let guard = Guard::new(false).unwrap();
        boundary().unwrap();
        assert!(Guard::new(true).is_err());
        assert_eq!(guard.calls(), 1);
        boundary().unwrap();
        assert_eq!(guard.calls(), 2);
        drop(guard);
        let error = (|| -> Result<(), String> {
            let _guard = Guard::new(true)?;
            boundary()
        })();
        assert!(error.unwrap_err().contains("before FFI"));
        let unwind = std::panic::catch_unwind(|| {
            let _guard = Guard::new(false).unwrap();
            panic!("exercise vendor guard unwind");
        });
        assert!(unwind.is_err());
        let guard = Guard::new(false).unwrap();
        assert_eq!(guard.calls(), 0);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn vendor_gemm_counter_observes_both_modes_and_denies_before_ffi() {
        use super::*;
        use crate::mamba_ssm::gpu::{buffers::DtypedBuf, device::GpuDevice};
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();
        for mode in [GemmMode::CublasFast, GemmMode::CublasPedantic] {
            ctx.set_gemm_mode(mode).unwrap();
            for dtype in [WeightDtype::F32, WeightDtype::Bf16, WeightDtype::F16] {
                eprintln!("vendor control {mode:?} {dtype:?}");
                let x = DtypedBuf::zeros(&ctx.stream, 32, dtype).unwrap();
                let w = DtypedBuf::zeros(&ctx.stream, 32 * 16, dtype).unwrap();
                let y = DtypedBuf::zeros(&ctx.stream, 16, dtype).unwrap();
                x.upload_f32(&ctx.stream, &[1.0; 32]).unwrap();
                w.upload_f32(&ctx.stream, &[1.0; 32 * 16]).unwrap();
                let run = || {
                    gpu_gemm_typed_forward_raw(
                        &ctx,
                        TypedPtr {
                            ptr: y.cached_ptr(),
                            dtype,
                        },
                        TypedPtr {
                            ptr: x.cached_ptr(),
                            dtype,
                        },
                        TypedPtr {
                            ptr: w.cached_ptr(),
                            dtype,
                        },
                        None,
                        (1, 32, 16),
                    )
                };
                let counter = Guard::new(false).unwrap();
                run().unwrap();
                let mut output = [0.0; 16];
                y.download_f32(&ctx.stream, &mut output).unwrap();
                assert_eq!(output, [32.0; 16]);
                assert_eq!(counter.calls(), 1);
                drop(counter);
                y.upload_f32(&ctx.stream, &[7.0; 16]).unwrap();
                let deny = Guard::new(true).unwrap();
                assert!(run().unwrap_err().contains("before FFI"));
                assert_eq!(deny.calls(), 1);
                y.download_f32(&ctx.stream, &mut output).unwrap();
                assert_eq!(output, [7.0; 16], "denied FFI must not write output");
            }
        }
    }
}

/// Effective cuBLAS compute type for a context-aware typed GEMM.
///
/// Fast uses ordinary f32 compute and Pedantic uses pedantic f32 compute for
/// every input dtype. Deterministic is rejected at this vendor boundary.
fn effective_compute(
    ctx: &GpuCtx,
    _dtype: super::dtype::WeightDtype,
) -> Result<cudarc::cublas::sys::cublasComputeType_t, String> {
    ctx.ensure_vendor_gemm("context-aware GemmEx")?
        .vendor_compute()
}

/// Routes row-major F32 `Y[B,N] = X[B,K] @ W[K,N] + bias[N]` using `ctx`.
///
/// `dims` is `(B,K,N)`. All pointers represent naturally aligned F32 spans in
/// the context's managed allocation domain: `y[B*N]`, `x[B*K]`, `w[K*N]`, and
/// optional `bias[N]`. Inputs must not overlap `y`. Their owners must remain
/// alive on `ctx.stream` through stream completion and every captured replay.
/// Zero reduction permits null `x` and `w`. Invalid context state, dimensions,
/// spans, alignment, aliasing, or unsupported selected routes return an error.
///
/// # Safety
///
/// The caller must uphold the pointer, allocation-domain, aliasing, stream,
/// and captured-replay lifetime requirements above.
pub(crate) unsafe fn gpu_gemm_f32_forward_ptrs(
    ctx: &GpuCtx,
    y: cudarc::driver::sys::CUdeviceptr,
    x: cudarc::driver::sys::CUdeviceptr,
    w: cudarc::driver::sys::CUdeviceptr,
    bias: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    let (batch, n_in, n_out) = dims;
    let shape = super::gemm_bi_triad::F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims);
    let request = super::gemm_bi_triad::F32TriadRequest {
        op: ResolvedGemmOp::Nn,
        shape,
    };
    let reduction_is_zero = shape.reduction(request.op) == 0;
    let x = if reduction_is_zero { 0 } else { x };
    let w = if reduction_is_zero { 0 } else { w };
    let operands = super::gemm_bi_triad::F32TriadOperands {
        output: y,
        a: x,
        b: w,
        bias,
        alpha: 1.0,
        beta: 0.0,
    };
    if ctx.gemm_mode() == GemmMode::Deterministic {
        return match ctx.bi_gemm_family() {
            super::context::BiGemmFamily::Triad => unsafe {
                super::gemm_bi_triad::launch_cached_f32_forward_ptrs(
                    ctx,
                    y,
                    x,
                    w,
                    bias.unwrap_or(0),
                    dims,
                )
            },
            super::context::BiGemmFamily::Inference => {
                super::gemm_bi_triad::validate_f32_triad_pointer_request(ctx, request, operands)?;
                gemm_bi_forward_raw(
                    ctx,
                    TypedPtr {
                        ptr: y,
                        dtype: WeightDtype::F32,
                    },
                    TypedPtr {
                        ptr: x,
                        dtype: WeightDtype::F32,
                    },
                    TypedPtr {
                        ptr: w,
                        dtype: WeightDtype::F32,
                    },
                    bias,
                    dims,
                )
            }
        };
    }

    super::gemm_bi_triad::validate_f32_triad_pointer_request(ctx, request, operands)?;
    ctx.ensure_vendor_gemm("gpu_gemm_f32_forward_ptrs")?;
    let beta = if let Some(b_ptr) = bias {
        let b_i = batch as i32;
        let n_i = n_out as i32;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.bias_broadcast);
        builder.arg(&y);
        builder.arg(&b_ptr);
        builder.arg(&b_i);
        builder.arg(&n_i);
        unsafe { builder.launch(grid_1d(batch * n_out)) }
            .map_err(|e| format!("bias_broadcast_ptrs: {e:?}"))?;
        1.0f32
    } else {
        0.0f32
    };

    let alpha: f32 = 1.0;
    let w_raw = w as *const f32;
    let x_raw = x as *const f32;
    let y_raw = y as *mut f32;

    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
        .map_err(|e| format!("cuBLAS sgemm_forward_ptrs failed: {e:?}"))?;
    }

    Ok(())
}

pub fn gpu_gemm_bi_forward_raw(
    ctx: &GpuCtx,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    bias_ptr: Option<cudarc::driver::sys::CUdeviceptr>,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    unsafe { gpu_gemm_f32_forward_ptrs(ctx, y.cached_ptr(), x.cached_ptr(), w_ptr, bias_ptr, dims) }
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
    unsafe { gpu_gemm_f32_forward_ptrs(ctx, y.cached_ptr(), x_ptr, w_ptr, bias_ptr, dims) }
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
    ctx.ensure_gemm_usable()?;
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
            super::context::BiGemmFamily::Inference => {
                super::gemm_bi_triad::launch_cached_f32_backward_dx(
                    ctx,
                    dx,
                    dy,
                    w_ptr,
                    (batch, n_in, n_out),
                )
            }
        };
    }
    ctx.ensure_vendor_gemm("gpu_gemm_bi_backward_dx_raw")?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let w_raw = w_ptr as *const f32;
    let dy_raw = dy.raw_ptr(&ctx.stream) as *const f32;
    let dx_raw = dx.raw_ptr(&ctx.stream) as *mut f32;

    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
    ctx.ensure_gemm_usable()?;
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
            super::context::BiGemmFamily::Inference => super::gemm_bi_triad::gemm_bi_backward_dw(
                &ctx.stream,
                &ctx.kernels,
                dw.ptr(),
                dy,
                x_saved,
                (batch, n_in, n_out),
            ),
        };
    }
    ctx.ensure_vendor_gemm("gpu_gemm_bi_backward_dw_grad")?;
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    let dy_ptr = dy.raw_ptr(&ctx.stream) as *const f32;
    let x_ptr = x_saved.raw_ptr(&ctx.stream) as *const f32;
    let dw_ptr = dw.ptr() as *mut f32;

    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
/// - alpha=1.0, beta=1.0 (f32 host scalars)
/// - vendor compute follows [`super::GemmMode`]: ordinary f32 compute in
///   `CublasFast` and pedantic f32 compute in `CublasPedantic`.
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
    ctx.ensure_gemm_usable()?;
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
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
            effective_compute(ctx, dy.dtype)?,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex backward dW typed failed: {e:?}"))?;
    }
    Ok(())
}

/// Typed dX backward GEMM. Typed twin of
/// [`gpu_gemm_bi_backward_dx_raw`]: `dX[B,K] = dY[B,N] @ W^T[N,K]` with
/// bf16/f16 A,B,C and f32 accumulation.
///
/// Layout mirrors the f32 twin exactly (OP_T on W, OP_N on dY,
/// m=n_in, n=batch, k=n_out, lda=n_out, ldb=n_out, ldc=n_in,
/// alpha=1.0, beta=0.0 — dX is overwritten, not accumulated).
///
/// `dy.dtype`, `w.dtype`, and `dx.dtype` MUST match (cuBLAS GemmEx
/// requires homogeneous A/B/C dtype for this compute mode). Pass all
/// three via `TypedPtr`. Vendor compute follows [`super::GemmMode`]: ordinary
/// f32 compute in `CublasFast` and pedantic f32 compute in
/// `CublasPedantic`.
pub fn gpu_gemm_ex_backward_dx_typed(
    ctx: &GpuCtx,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    debug_assert_eq!(
        dy.dtype, w.dtype,
        "cuBLAS GemmEx requires A.dtype == B.dtype"
    );
    debug_assert_eq!(
        dx.dtype, dy.dtype,
        "typed dX GEMM: dx.dtype must match dy/w"
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
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
            effective_compute(ctx, dy.dtype)?,
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
        PhysicalLaunchKind::InputTransform => {
            return Err("conversion launch cannot use input-transform kind".into());
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
    ctx.ensure_gemm_usable()?;
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
        let ops = super::gemm_bi_triad::TcFwdOperands { y, x, w, bias_ptr };
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm89_half_nn_auto_observed(ctx, observer, &ops, dims)?
        {
            return Ok(HalfPolicyBranchSeal::Native(seal));
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
    ctx.ensure_gemm_usable()?;
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
        if let Some(seal) = super::gemm_bi_triad::launch_sm89_half_tn_auto_observed(
            ctx, observer, dw_ptr, dy, x_saved, dims,
        )? {
            return Ok(HalfPolicyBranchSeal::Native(seal));
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
    ctx.ensure_gemm_usable()?;
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
        if let Some(seal) =
            super::gemm_bi_triad::launch_sm89_half_nt_auto_observed(ctx, observer, dx, dy, w, dims)?
        {
            return Ok(HalfPolicyBranchSeal::Native(seal));
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

fn push_nn_half_graph_arguments(
    arguments: &mut PhysicalGraphKernelArguments,
    request: HalfPhysicalTraceRequest,
    shape: super::gemm_bi_triad::F32TriadShape,
    checked: super::gemm_bi_triad::GemmDims,
    base: &str,
) -> Result<(), String> {
    arguments.push(request.output)?;
    arguments.push(request.a)?;
    arguments.push(request.b)?;
    arguments.push(request.bias)?;
    if sm89_half_nn_graph_uses_parameter_bundle(base) {
        arguments.push(super::gemm_bi_triad::Sm89HalfNnParams {
            alpha: 1.0,
            beta: 0.0,
            m: checked.m_i32,
            n: checked.n_i32,
            k: checked.k_i32,
            lda: i32::try_from(shape.lda).map_err(|_| "NN lda exceeds i32::MAX")?,
            ldb: i32::try_from(shape.ldb).map_err(|_| "NN ldb exceeds i32::MAX")?,
            ldc: i32::try_from(shape.ldc).map_err(|_| "NN ldc exceeds i32::MAX")?,
        })?;
        return Ok(());
    }
    arguments.push(1.0_f32)?;
    arguments.push(0.0_f32)?;
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
    Ok(())
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
    let mut ranges = f32_physical_argument_ranges(request, operands)?;
    if let Some(scratch) = prepared.physical_graph_scratch_range() {
        ranges.push(scratch);
    }
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
            push_nn_half_graph_arguments(&mut arguments, request, shape, checked, base)?;
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

fn native_half_graph_module_supported(module: ModuleKind) -> bool {
    matches!(
        module,
        ModuleKind::TriadSm80 | ModuleKind::TriadScalar | ModuleKind::TriadSm89Half
    )
}

fn sm89_half_nn_graph_uses_parameter_bundle(base: &str) -> bool {
    base == "gemm_bi_nn_sm89_m128n128_bk64_s3_v1"
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
        PhysicalLaunchKind::InputTransform => {
            return Err("physical graph conversion cannot bind an input-transform node".into());
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
        (PhysicalLaunchKind::Gemm, module) if native_half_graph_module_supported(module) => (
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
    ctx.ensure_gemm_usable()?;
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
    ctx.ensure_gemm_usable()?;
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
    ctx.ensure_gemm_usable()?;
    let dims = (batch, vocab_padded, d_model);
    let request = super::gemm_bi_triad::F32TriadRequest {
        op: ResolvedGemmOp::Nt,
        shape: super::gemm_bi_triad::F32TriadShape::contiguous(ResolvedGemmOp::Nt, dims),
    };
    let operands = super::gemm_bi_triad::F32TriadOperands {
        output: logits_ptr,
        a: temporal_ptr,
        b: embed_ptr,
        bias: None,
        alpha: 1.0,
        beta: 0.0,
    };
    if ctx.gemm_mode() == GemmMode::Deterministic {
        return unsafe {
            super::gemm_bi_triad::launch_cached_f32_backward_dx_ptrs(
                ctx,
                logits_ptr,
                temporal_ptr,
                embed_ptr,
                dims,
            )
        };
    }
    super::gemm_bi_triad::validate_f32_triad_pointer_request(ctx, request, operands)?;
    ctx.ensure_vendor_gemm("gpu_gemm_bi_tied_lm_head_raw")?;
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

/// Vendor-only no-context twin of [`gpu_gemm_bi_tied_lm_head_raw`].
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
    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
    #[ignore = "needs a CUDA device"]
    fn context_aware_vendor_compute_maps_all_dtypes() {
        use super::super::context::GemmMode;
        use super::super::device::GpuDevice;
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

    #[test]
    fn sm89_half_is_a_native_prepared_graph_module() {
        assert!(native_half_graph_module_supported(
            ModuleKind::TriadSm89Half
        ));
    }

    #[test]
    fn sm89_half_nn_graph_uses_the_five_argument_bundle_abi() {
        let base = "gemm_bi_nn_sm89_m128n128_bk64_s3_v1";
        assert!(sm89_half_nn_graph_uses_parameter_bundle(base));
        assert!(!sm89_half_nn_graph_uses_parameter_bundle("gemm_bi_nn_tc"));
        let dims = (2048, 1536, 768);
        let request = HalfPhysicalTraceRequest {
            op: ResolvedGemmOp::Nn,
            output: 0x1000,
            a: 0x2000,
            b: 0x3000,
            bias: 0x4000,
            dtype: WeightDtype::F16,
            dims,
            nn_strides: None,
            forced_tile: None,
            capacity: 1,
        };
        let shape =
            super::super::gemm_bi_triad::F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims);
        let checked = super::super::gemm_bi_triad::GemmDims::nn(dims, shape.lda).unwrap();
        let mut arguments = PhysicalGraphKernelArguments::new();
        push_nn_half_graph_arguments(&mut arguments, request, shape, checked, base).unwrap();
        assert_eq!(arguments.values().len(), 5);
        for (argument, pointer) in
            arguments.values()[..4]
                .iter()
                .zip([request.output, request.a, request.b, request.bias])
        {
            assert_eq!(
                &argument.bytes[..std::mem::size_of::<usize>()],
                &pointer.to_ne_bytes()
            );
            assert!(
                argument.bytes[std::mem::size_of::<usize>()..]
                    .iter()
                    .all(|&byte| byte == 0)
            );
        }
        let expected =
            PhysicalGraphKernelArgument::encode(super::super::gemm_bi_triad::Sm89HalfNnParams {
                alpha: 1.0,
                beta: 0.0,
                m: 2048,
                n: 768,
                k: 1536,
                lda: 1536,
                ldb: 768,
                ldc: 768,
            })
            .unwrap();
        assert_eq!(arguments.values()[4].bytes, expected.bytes);
    }

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
        ctx.set_gemm_mode(crate::mamba_ssm::gpu::GemmMode::Deterministic)
            .unwrap();
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

    fn record_tied_half_f32_trace(dtype: WeightDtype, dims: TiedLmDims) -> RecordedPhysicalTrace {
        use super::super::buffers::DtypedBuf;
        use super::super::context::{BiGemmFamily, GemmMode};
        use super::super::device::GpuDevice;

        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_gemm_mode(crate::mamba_ssm::gpu::GemmMode::Deterministic)
            .unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Inference);
        let checked = checked_tied_lm_dims(dims).unwrap();
        let temporal = DtypedBuf::zeros(&ctx.stream, checked.temporal_elements, dtype).unwrap();
        let embed = DtypedBuf::zeros(&ctx.stream, checked.embed_elements, dtype).unwrap();
        let logits = GpuBuffer::zeros(&ctx.stream, checked.logits_elements).unwrap();
        let temporal = TypedPtr {
            ptr: temporal.cached_ptr(),
            dtype,
        };
        let embed = TypedPtr {
            ptr: embed.cached_ptr(),
            dtype,
        };

        let eager = ctx
            .record_eager_gemm_trace(|| {
                let mut observer = NoPhysicalObserver;
                gemm_bi_tied_half_f32_in(
                    &ctx,
                    logits.cached_ptr(),
                    temporal,
                    embed,
                    dims,
                    &mut observer,
                )
            })
            .unwrap();
        let scratch = ctx.bi_upcast_scratch_ptrs();
        let mut ranges = vec![
            PhysicalArgumentRange {
                pointer: logits.cached_ptr(),
                required_bytes: physical_argument_bytes(checked.logits_elements, 4, "tied logits")
                    .unwrap(),
            },
            PhysicalArgumentRange {
                pointer: temporal.ptr,
                required_bytes: physical_argument_bytes(
                    checked.temporal_elements,
                    dtype.size_bytes(),
                    "tied temporal",
                )
                .unwrap(),
            },
            PhysicalArgumentRange {
                pointer: embed.ptr,
                required_bytes: physical_argument_bytes(
                    checked.embed_elements,
                    dtype.size_bytes(),
                    "tied embed",
                )
                .unwrap(),
            },
            PhysicalArgumentRange {
                pointer: scratch[0],
                required_bytes: physical_argument_bytes(
                    checked.temporal_elements,
                    4,
                    "tied temporal scratch",
                )
                .unwrap(),
            },
            PhysicalArgumentRange {
                pointer: scratch[1],
                required_bytes: physical_argument_bytes(
                    checked.embed_elements,
                    4,
                    "tied embed scratch",
                )
                .unwrap(),
            },
        ];
        ranges.retain(|range| range.required_bytes != 0);
        let conversion_count =
            usize::from(checked.temporal_elements != 0) + usize::from(checked.embed_elements != 0);
        let mut observer =
            prepare_physical_observer(&ctx, conversion_count + eager.routes().len(), &ranges)
                .expect("prepare tied physical observer");
        gemm_bi_tied_half_f32_in(
            &ctx,
            logits.cached_ptr(),
            temporal,
            embed,
            dims,
            &mut observer,
        )
        .expect("record tied half-to-F32 composition");
        let trace = finish_recording_physical_observer(observer, ctx.gemm_route()).unwrap();
        ctx.stream.synchronize().unwrap();
        trace
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn tied_half_f32_observer_records_two_upcasts_all_nt_launches_and_no_downcast() {
        let dims = TiedLmDims {
            batch: 32,
            d_model: 128,
            vocab_padded: 96,
        };
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let trace = record_tied_half_f32_trace(dtype, dims);
            let nodes = trace.nodes();
            assert_eq!(nodes[0].kind(), PhysicalLaunchKind::InputUpcast);
            assert_eq!(nodes[1].kind(), PhysicalLaunchKind::InputUpcast);
            assert!(
                nodes.len() > 3,
                "fixture must select a multi-launch NT route"
            );
            assert!(nodes[2..].iter().all(|node| {
                node.kind() == PhysicalLaunchKind::Gemm
                    && node.logical_op() == ResolvedGemmOp::Nt
                    && node.execution_dtype() == PolicyDtype::F32
                    && node.shape() == (dims.batch, dims.vocab_padded, dims.d_model)
                    && node.strides() == (dims.d_model, dims.d_model, dims.vocab_padded)
            }));
            assert_eq!(
                nodes
                    .iter()
                    .filter(|node| node.kind() == PhysicalLaunchKind::OutputDowncast)
                    .count(),
                0
            );
            let expected_logical_dtype = match dtype {
                WeightDtype::Bf16 => PolicyDtype::Bf16,
                WeightDtype::F16 => PolicyDtype::F16,
                WeightDtype::F32 => unreachable!(),
            };
            assert!(
                nodes
                    .iter()
                    .all(|node| node.logical_dtype() == expected_logical_dtype)
            );
        }
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn tied_half_f32_zero_reduction_observer_records_only_f32_epilogue() {
        let dims = TiedLmDims {
            batch: 2,
            d_model: 0,
            vocab_padded: 96,
        };
        let trace = record_tied_half_f32_trace(WeightDtype::Bf16, dims);
        assert_eq!(trace.nodes().len(), 1);
        let epilogue = &trace.nodes()[0];
        assert_eq!(epilogue.kind(), PhysicalLaunchKind::Gemm);
        assert_eq!(epilogue.logical_op(), ResolvedGemmOp::Nt);
        assert_eq!(epilogue.execution_dtype(), PolicyDtype::F32);
        assert_eq!(epilogue.shape(), (2, 96, 0));
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn tied_half_f32_scratch_freeze_reuses_reserved_pointers_and_rejects_growth() {
        use super::super::buffers::DtypedBuf;
        use super::super::context::GemmMode;
        use super::super::device::GpuDevice;

        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        let dims = TiedLmDims {
            batch: 2,
            d_model: 37,
            vocab_padded: 96,
        };
        presize_tied_lm_head_scratch(&ctx, WeightDtype::Bf16, dims).unwrap();
        let temporal =
            DtypedBuf::zeros(&ctx.stream, dims.batch * dims.d_model, WeightDtype::Bf16).unwrap();
        let embed = DtypedBuf::zeros(
            &ctx.stream,
            dims.vocab_padded * dims.d_model,
            WeightDtype::Bf16,
        )
        .unwrap();
        let logits = GpuBuffer::zeros(&ctx.stream, dims.batch * dims.vocab_padded).unwrap();
        let before = ctx.bi_upcast_scratch_ptrs();
        ctx.freeze_graph_scratch();
        gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            logits.cached_ptr(),
            temporal.cached_ptr(),
            embed.cached_ptr(),
            WeightDtype::Bf16,
            dims,
        )
        .expect("same-size tied head after freeze");
        assert_eq!(ctx.bi_upcast_scratch_ptrs(), before);

        let sentinel = vec![321.5; dims.batch * dims.vocab_padded];
        let larger_logits = GpuBuffer::from_cpu(&ctx.stream, &sentinel).unwrap();
        let larger = TiedLmDims {
            d_model: dims.d_model + 1,
            ..dims
        };
        let larger_temporal = DtypedBuf::zeros(
            &ctx.stream,
            larger.batch * larger.d_model,
            WeightDtype::Bf16,
        )
        .unwrap();
        let larger_embed = DtypedBuf::zeros(
            &ctx.stream,
            larger.vocab_padded * larger.d_model,
            WeightDtype::Bf16,
        )
        .unwrap();
        let error = gpu_gemm_ex_tied_lm_head_raw(
            &ctx,
            larger_logits.cached_ptr(),
            larger_temporal.cached_ptr(),
            larger_embed.cached_ptr(),
            WeightDtype::Bf16,
            larger,
        )
        .unwrap_err();
        assert!(
            error.contains("cannot grow after CUDA graph capture"),
            "{error}"
        );
        assert_eq!(larger_logits.to_cpu(&ctx.stream).unwrap(), sentinel);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn tied_half_f32_rejects_mismatched_and_f32_private_inputs_before_execution() {
        use super::super::context::GemmMode;
        use super::super::device::GpuDevice;

        let device = GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("GPU context");
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        let dims = TiedLmDims {
            batch: 2,
            d_model: 1,
            vocab_padded: 96,
        };
        let mut observer = NoPhysicalObserver;
        let mismatch = gemm_bi_tied_half_f32_in(
            &ctx,
            0,
            TypedPtr {
                ptr: 0,
                dtype: WeightDtype::Bf16,
            },
            TypedPtr {
                ptr: 0,
                dtype: WeightDtype::F16,
            },
            dims,
            &mut observer,
        )
        .unwrap_err();
        assert!(mismatch.contains("dtypes must match"), "{mismatch}");
        let unsupported = gemm_bi_tied_half_f32_in(
            &ctx,
            0,
            TypedPtr {
                ptr: 0,
                dtype: WeightDtype::F32,
            },
            TypedPtr {
                ptr: 0,
                dtype: WeightDtype::F32,
            },
            dims,
            &mut observer,
        )
        .unwrap_err();
        assert!(
            unsupported.contains("requires bf16 or f16"),
            "{unsupported}"
        );
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

#[derive(Copy, Clone)]
struct CheckedTiedLmDims {
    temporal_elements: usize,
    embed_elements: usize,
    logits_elements: usize,
}

fn checked_tied_lm_dims(dims: TiedLmDims) -> Result<CheckedTiedLmDims, String> {
    let shape = super::gemm_bi_triad::F32TriadShape::contiguous(
        ResolvedGemmOp::Nt,
        (dims.batch, dims.vocab_padded, dims.d_model),
    );
    shape.validate(ResolvedGemmOp::Nt)?;
    Ok(CheckedTiedLmDims {
        temporal_elements: dims
            .batch
            .checked_mul(dims.d_model)
            .ok_or_else(|| "tied lm_head B*D element count overflows usize".to_string())?,
        embed_elements: dims
            .vocab_padded
            .checked_mul(dims.d_model)
            .ok_or_else(|| "tied lm_head Vpad*D element count overflows usize".to_string())?,
        logits_elements: dims
            .batch
            .checked_mul(dims.vocab_padded)
            .ok_or_else(|| "tied lm_head B*Vpad element count overflows usize".to_string())?,
    })
}

fn tied_lm_byte_span(elements: usize, width: usize, label: &str) -> Result<u64, String> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| {
            u64::try_from(width)
                .ok()
                .and_then(|width| elements.checked_mul(width))
        })
        .ok_or_else(|| format!("tied lm_head {label} byte span overflows u64"))
}

fn tied_lm_ranges_overlap(a: (u64, u64), b: (u64, u64)) -> Result<bool, String> {
    let a_end =
        a.0.checked_add(a.1)
            .ok_or_else(|| "tied lm_head pointer range overflows u64".to_string())?;
    let b_end =
        b.0.checked_add(b.1)
            .ok_or_else(|| "tied lm_head pointer range overflows u64".to_string())?;
    Ok(a.0 < b_end && b.0 < a_end)
}

fn validate_tied_half_f32_ranges(
    ctx: &GpuCtx,
    logits: cudarc::driver::sys::CUdeviceptr,
    temporal: TypedPtr,
    embed: TypedPtr,
    checked: CheckedTiedLmDims,
) -> Result<ManagedAllocationEpochStamp, String> {
    if logits == 0 || !logits.is_multiple_of(4) {
        return Err("tied lm_head F32 logits pointer must be non-null and 4-byte aligned".into());
    }
    for (name, input, elements) in [
        ("temporal", temporal, checked.temporal_elements),
        ("embed", embed, checked.embed_elements),
    ] {
        if elements != 0 && (input.ptr == 0 || !input.ptr.is_multiple_of(2)) {
            return Err(format!(
                "tied lm_head {name} pointer must be non-null and 2-byte aligned"
            ));
        }
    }

    let logits_range = (
        logits,
        tied_lm_byte_span(checked.logits_elements, 4, "logits")?,
    );
    let temporal_range = (
        temporal.ptr,
        tied_lm_byte_span(
            checked.temporal_elements,
            temporal.dtype.size_bytes(),
            "temporal",
        )?,
    );
    let embed_range = (
        embed.ptr,
        tied_lm_byte_span(checked.embed_elements, embed.dtype.size_bytes(), "embed")?,
    );
    for (name, input_range) in [("temporal", temporal_range), ("embed", embed_range)] {
        if input_range.1 != 0 && tied_lm_ranges_overlap(logits_range, input_range)? {
            return Err(format!(
                "tied lm_head {name} input overlaps F32 logits output"
            ));
        }
    }

    let mut ranges = vec![logits_range];
    if temporal_range.1 != 0 {
        ranges.push(temporal_range);
    }
    if embed_range.1 != 0 {
        ranges.push(embed_range);
    }
    managed_allocation_epoch_for_ranges(ctx.stream.context().cu_ctx() as usize, &ranges).ok_or_else(
        || {
            "tied lm_head pointers are not covered by live allocations in the CUDA context"
                .to_string()
        },
    )
}

fn gemm_bi_tied_half_f32_in<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    logits: cudarc::driver::sys::CUdeviceptr,
    temporal: TypedPtr,
    embed: TypedPtr,
    dims: TiedLmDims,
    observer: &mut O,
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    if temporal.dtype != embed.dtype {
        return Err("tied lm_head temporal and embed dtypes must match".into());
    }
    if temporal.dtype == WeightDtype::F32 {
        return Err("deterministic tied half-to-F32 path requires bf16 or f16 inputs".into());
    }
    if ctx.gemm_mode() != GemmMode::Deterministic {
        return Err("tied half-to-F32 composition requires deterministic GEMM mode".into());
    }
    let checked = checked_tied_lm_dims(dims)?;
    let _allocation_epoch = validate_tied_half_f32_ranges(ctx, logits, temporal, embed, checked)?;
    let physical = HalfPhysicalContext {
        op: ResolvedGemmOp::Nt,
        dtype: temporal.dtype,
        dims: (dims.batch, dims.vocab_padded, dims.d_model),
    };

    ctx.with_bi_upcast_scratch(
        (checked.temporal_elements, checked.embed_elements, 0),
        |temporal_f32, embed_f32, _| {
            if checked.temporal_elements != 0 {
                bi_upcast_to_f32(
                    ctx,
                    temporal,
                    temporal_f32.cached_ptr(),
                    checked.temporal_elements,
                    physical,
                    observer,
                )?;
            }
            if checked.embed_elements != 0 {
                bi_upcast_to_f32(
                    ctx,
                    embed,
                    embed_f32.cached_ptr(),
                    checked.embed_elements,
                    physical,
                    observer,
                )?;
            }
            unsafe {
                super::gemm_bi_triad::record_physical_exact_scalar_f32_backward_dx_ptrs(
                    ctx,
                    observer,
                    logits,
                    temporal_f32.cached_ptr(),
                    embed_f32.cached_ptr(),
                    super::gemm_bi_triad::ScalarFallbackPhysicalContext {
                        dims: physical.dims,
                        dtype: physical.dtype,
                    },
                )
            }
        },
    )
}

/// Reserves deterministic tied-head conversion scratch without launching work.
///
/// BF16/F16 tied heads produce caller-owned F32 logits by casting the two
/// inputs once and reducing with the exact-scalar F32 NT route. The persistent
/// scratch footprint is `(B + Vpad) * D * 4` bytes. Call this before freezing
/// CUDA-graph-visible scratch; vendor modes and F32 heads need no reservation.
#[cfg(any(feature = "hf", test))]
pub(crate) fn presize_tied_lm_head_scratch(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    if dtype == WeightDtype::F32 || ctx.gemm_mode() != GemmMode::Deterministic {
        return Ok(());
    }
    let checked = checked_tied_lm_dims(dims)?;
    ctx.with_bi_upcast_scratch(
        (checked.temporal_elements, checked.embed_elements, 0),
        |_, _, _| Ok(()),
    )
}

/// Computes a tied LM head into caller-owned F32 logits.
///
/// The row-major operation is `logits[B,Vpad] = temporal[B,D] * embed[Vpad,D]^T`.
/// `temporal_ptr` and `embed_ptr` must be matching `dtype` spans with `B*D`
/// and `Vpad*D` elements; `logits_ptr` must be an F32 span with `B*Vpad`
/// elements. All allocations must belong to `ctx` and remain live until the
/// context stream completes (and through every replay that uses the pointers).
///
/// In deterministic mode BF16/F16 inputs are each cast once into persistent
/// F32 scratch, then reduced by the exact-scalar F32 NT route directly into
/// `logits_ptr`; there is no half output round trip. This requires
/// `(B + Vpad) * D * 4` bytes of shared conversion scratch. Call
/// `presize_tied_lm_head_scratch` before graph capture can freeze scratch
/// addresses. F32 delegates to [`gpu_gemm_bi_tied_lm_head_raw`], while vendor
/// modes retain GemmEx with the context's canonical compute type.
///
/// Returns an error before enqueue for unhealthy context state, invalid or
/// overflowing dimensions, null/misaligned/unmanaged spans, output overlap,
/// unsupported deterministic input types, or scratch growth after capture.
pub fn gpu_gemm_ex_tied_lm_head_raw(
    ctx: &GpuCtx,
    logits_ptr: cudarc::driver::sys::CUdeviceptr,
    temporal_ptr: cudarc::driver::sys::CUdeviceptr,
    embed_ptr: cudarc::driver::sys::CUdeviceptr,
    dtype: WeightDtype,
    dims: TiedLmDims,
) -> Result<(), String> {
    ctx.ensure_gemm_usable()?;
    if dtype == WeightDtype::F32 {
        return gpu_gemm_bi_tied_lm_head_raw(
            ctx,
            logits_ptr,
            temporal_ptr,
            embed_ptr,
            dims.batch,
            dims.d_model,
            dims.vocab_padded,
        );
    }
    if ctx.gemm_mode() == GemmMode::Deterministic {
        let mut observer = NoPhysicalObserver;
        return gemm_bi_tied_half_f32_in(
            ctx,
            logits_ptr,
            TypedPtr {
                ptr: temporal_ptr,
                dtype,
            },
            TypedPtr {
                ptr: embed_ptr,
                dtype,
            },
            dims,
            &mut observer,
        );
    }
    let compute = effective_compute(ctx, dtype)?;
    gpu_gemm_ex_tied_lm_head_with_compute(
        &ctx.blas,
        logits_ptr,
        temporal_ptr,
        embed_ptr,
        dtype,
        dims,
        compute,
    )
}

/// Vendor-only no-context twin of [`gpu_gemm_ex_tied_lm_head_raw`].
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
    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
            compute,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS tied gemm_ex failed: {e:?}"))?;
    }
    Ok(())
}

/// Mixed-precision GEMM forward: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
///
/// Inputs X and W may be f32, f16, or bf16. Output Y is always f32. Vendor
/// compute follows [`super::GemmMode`]: ordinary f32 compute in `CublasFast`
/// and pedantic f32 compute in `CublasPedantic`.
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
    ctx.ensure_gemm_usable()?;
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

/// Vendor-only fully typed GEMM forward: `C[B,N] = A[B,K] @ W[K,N]`.
///
/// All three operand dtypes are independent (`a.dtype`, `w.dtype`, `c.dtype`).
/// This helper has no [`GpuCtx`], so it uses the dtype's fixed cuBLAS compute
/// type rather than a context-selected [`super::GemmMode`].
///
/// Bias (if provided) is always stored f32 (Mamba convention) and is
/// broadcast into C via the typed `bias_broadcast_<c.dtype>` kernel,
/// which upcasts bias to f32, adds f32, and downcasts to `c.dtype`.
///
/// This no-context compatibility boundary always uses cuBLAS. High-level
/// model paths that own a [`GpuCtx`] must call [`gpu_gemm_typed_forward_raw`]
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
    unsafe {
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
            w.dtype.compute_type(), // No context is available to select a GemmMode.
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
// through gemm_bi_forward_raw (the Inference family's entry and the f32
// dispatch arm).
fn launch_bi_gemm<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    kernel: &cudarc::driver::CudaFunction,
    threads: u32,
    args: BiGemmArgs,
    storage: [WeightDtype; 3],
    observer: &mut O,
) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
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
    let observation =
        super::gemm_bi_inference::identity::observation(ctx, observer, kernel, cfg, || {
            super::gemm_bi_inference::identity::Arguments::legacy(
                [args.c, args.a, args.b, args.bias],
                storage.map(super::gemm_bi_inference::identity::policy_dtype),
                [args.alpha, args.beta],
                [args.m, args.n, args.k, lda, ldb, ldc],
            )
        })?;
    unsafe { enqueue_with_physical_observation(observer, &mut builder, cfg, observation) }
        .map_err(|error| error.with_driver_context(format_args!("Inference legacy launch")))?;
    Ok(())
}

/// Direct entry to the Inference batch-invariant GEMM ladder
/// (`kernels/gemm_bi_inference/`, `gemm_bi_*`). Every rung uses `SPLIT_K=1`
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
    ctx.ensure_gemm_usable()?;
    super::gemm_bi_inference::inference_forward(ctx, c, x, w, bias_ptr, dims).map(|_| ())
}

/// The Inference family's LEGACY tile (64x64x32, strict, no buckets): the
/// narrow-N fallback of the inference ladder and the whole f32 arm (the
/// shipped serve route - its bits never move with ladder work).
pub(in crate::mamba_ssm::gpu) fn fixed_legacy_forward<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    operands: super::gemm_bi_inference::InferenceFwdOperands,
    shape: super::gemm_bi_inference::InferenceShape,
    observer: &mut O,
) -> Result<(), String> {
    let super::gemm_bi_inference::InferenceFwdOperands { c, x, w, bias_ptr } = operands;
    let (batch, n_in, n_out) = (shape.m, shape.k, shape.n);
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
            m: i32::try_from(batch).map_err(|_| "Inference M exceeds i32")?,
            n: i32::try_from(n_out).map_err(|_| "Inference N exceeds i32")?,
            k: i32::try_from(n_in).map_err(|_| "Inference K exceeds i32")?,
        },
        [x.dtype, w.dtype, c.dtype],
        observer,
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

fn launch_bi_matvec<O: PhysicalLaunchObserver>(
    ctx: &GpuCtx,
    kernel: &cudarc::driver::CudaFunction,
    args: BiGemmArgs,
    storage: [WeightDtype; 3],
    observer: &mut O,
) -> Result<(), String> {
    if args.m == 0 || args.n == 0 {
        return Ok(());
    }
    // Must match kernel constants in kernels/gemm_bi_inference/:
    //   BLOCK_N_MV = 32, WARPS_PER_BLOCK = 8, THREADS_PER_BLOCK = 256
    // Grid is 2D: (ceil(N / BLOCK_N_MV), M) — one CTA per (m_row, col_chunk).
    const BLOCK_N_MV: i32 = 32;
    const THREADS_PER_BLOCK: i32 = 256;
    let a_bytes = u32::try_from(args.k)
        .ok()
        .and_then(|k| k.checked_mul(storage[0].size_bytes() as u32))
        .ok_or("matvec input byte span exceeds u32")?;
    let smem_bytes = a_bytes
        .checked_add(15)
        .ok_or("matvec aligned byte span exceeds u32")?
        & !15;
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
    let observation =
        super::gemm_bi_inference::identity::observation(ctx, observer, kernel, cfg, || {
            super::gemm_bi_inference::identity::Arguments::legacy(
                [args.c, args.a, args.b, args.bias],
                storage.map(super::gemm_bi_inference::identity::policy_dtype),
                [args.alpha, args.beta],
                [args.m, args.n, args.k, lda, ldb, ldc],
            )
        })?;
    unsafe { enqueue_with_physical_observation(observer, &mut builder, cfg, observation) }
        .map_err(|error| error.with_driver_context(format_args!("Fixed matvec launch")))?;
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
    ctx.ensure_gemm_usable()?;
    let (batch, n_in, n_out) = dims;

    // Canonical routing is mode-first. All-F32 delegates to the shared NN
    // seam: deterministic mode follows the selected Inference/Triad family,
    // while cuBLAS Fast/Pedantic use the configured vendor handle. Remaining
    // deterministic homogeneous-half triples follow the selected fixed-tile,
    // Triad, or matvec coverage below; unsupported mixed triples fail closed.
    // Vendor modes reach GemmEx only after those deterministic branches are
    // dormant. Keep the registered legacy selector referenced.
    let _ = pick_bi_gemm(ctx, x.dtype, w.dtype, c.dtype);

    if c.dtype == WeightDtype::F32 && x.dtype == WeightDtype::F32 && w.dtype == WeightDtype::F32 {
        return unsafe { gpu_gemm_f32_forward_ptrs(ctx, c.ptr, x.ptr, w.ptr, bias_ptr, dims) };
    }

    // The matvec kernel handles any M ≥ 1 via a 2D grid (CTA per
    // (m_row, col_chunk)) and gives strict cross-batch bit-identity.
    // Selected by `GemmMode::Deterministic`; the environment default is also
    // deterministic. The deprecated batch-invariant environment variable is
    // accepted only as a legacy adapter.
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
    if ctx.batch_invariant() && ctx.bi_gemm_family() == super::context::BiGemmFamily::Inference {
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
            [x.dtype, w.dtype, c.dtype],
            &mut NoPhysicalObserver,
        );
    }

    if ctx.batch_invariant() {
        // No deterministic kernel covers this operand triple. Do not cross
        // the vendor boundary while deterministic mode is selected.
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
        #[cfg(test)]
        vendor_gemm_test::boundary()?;
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
            // Compute type follows the context's vendor mode for every dtype.
            effective_compute(ctx, w.dtype)?,
            cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .map_err(|e| format!("cuBLAS gemm_ex typed failed: {e:?}"))?;
    }

    Ok(())
}

#[cfg(test)]
mod matvec_inventory_cuda_tests {
    use super::*;
    use crate::mamba_ssm::gpu::buffers::DtypedBuf;
    use crate::mamba_ssm::gpu::context::BiGemmFamily;
    use crate::mamba_ssm::gpu::device::GpuDevice;
    use crate::mamba_ssm::gpu::kernel_identity::{PhysicalGemmBackend, ResolvedNumericContract};

    #[test]
    #[ignore = "needs a CUDA device"]
    fn triad_native_half_context_inventory_records_all_projection_terminals() {
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new_with_mode(&device, GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_bi_tensor_cores(true);
        let deny = vendor_gemm_test::Guard::new(true).unwrap();
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for batch in [1, 3] {
                eprintln!("native half context terminals {dtype:?} B{batch}");
                let shapes = [
                    (batch, 32, 128),
                    (batch, 64, 18),
                    (batch, 2, 64),
                    (batch, 64, 32),
                ];
                let buffers: Vec<_> = shapes
                    .iter()
                    .map(|&(m, k, n)| {
                        let x = DtypedBuf::zeros(&ctx.stream, m * k, dtype).unwrap();
                        let w = DtypedBuf::zeros(&ctx.stream, k * n, dtype).unwrap();
                        let y = DtypedBuf::zeros(&ctx.stream, m * n, dtype).unwrap();
                        x.upload_f32(&ctx.stream, &vec![1.0; m * k]).unwrap();
                        w.upload_f32(&ctx.stream, &vec![1.0; k * n]).unwrap();
                        (x, w, y)
                    })
                    .collect();
                let pointers = |i: usize| {
                    let (x, w, y) = &buffers[i];
                    (
                        TypedPtr {
                            ptr: y.cached_ptr(),
                            dtype,
                        },
                        TypedPtr {
                            ptr: x.cached_ptr(),
                            dtype,
                        },
                        TypedPtr {
                            ptr: w.cached_ptr(),
                            dtype,
                        },
                    )
                };
                let run = || -> Result<(), String> {
                    for (i, &shape) in shapes.iter().enumerate() {
                        let (y, x, w) = pointers(i);
                        gpu_gemm_typed_forward_raw(&ctx, y, x, w, None, shape)?;
                    }
                    Ok(())
                };
                run().unwrap();
                let trace = ctx.record_eager_gemm_trace(run).unwrap();
                assert_eq!(
                    trace
                        .routes()
                        .iter()
                        .map(|route| route.shape)
                        .collect::<Vec<_>>(),
                    shapes
                );
                for (i, &(m, k, n)) in shapes.iter().enumerate() {
                    let mut output = vec![0.0; m * n];
                    buffers[i].2.download_f32(&ctx.stream, &mut output).unwrap();
                    assert_eq!(output, vec![k as f32; m * n]);
                    ctx.validate_resolved_gemm_route(
                        &trace.routes()[i],
                        "native half direct projection",
                    )
                    .unwrap();
                }

                let (y, x, w) = pointers(0);
                let (m, k, n) = shapes[0];
                let ranges = [
                    PhysicalArgumentRange {
                        pointer: y.ptr,
                        required_bytes: (m * n * dtype.size_bytes()) as u64,
                    },
                    PhysicalArgumentRange {
                        pointer: x.ptr,
                        required_bytes: (m * k * dtype.size_bytes()) as u64,
                    },
                    PhysicalArgumentRange {
                        pointer: w.ptr,
                        required_bytes: (k * n * dtype.size_bytes()) as u64,
                    },
                ];
                let mut observer = prepare_physical_observer(&ctx, 1, &ranges).unwrap();
                gemm_bi_forward_typed_in(&ctx, y, x, w, 0, shapes[0], &mut observer).unwrap();
                let physical_only =
                    finish_recording_physical_observer(observer, ctx.gemm_route()).unwrap();
                let mut observer = prepare_physical_observer(&ctx, 1, &ranges).unwrap();
                let context_trace = ctx
                    .record_eager_gemm_trace(|| {
                        gemm_bi_forward_typed_in(&ctx, y, x, w, 0, shapes[0], &mut observer)
                            .map(drop)
                    })
                    .unwrap();
                let physical_with_context =
                    finish_recording_physical_observer(observer, ctx.gemm_route()).unwrap();
                assert_eq!(
                    physical_with_context.nodes(),
                    physical_only.nodes(),
                    "context recording must preserve physical allocation digests"
                );
                assert_eq!(physical_with_context.nodes().len(), 1);
                assert_eq!(
                    context_trace.routes(),
                    &trace.routes()[..1],
                    "one context record, no duplication"
                );
                assert_ne!(
                    physical_only.nodes()[0]
                        .gemm_route()
                        .unwrap()
                        .launch
                        .arguments_digest,
                    context_trace.routes()[0].launch.arguments_digest
                );

                // A full recorder must reject the real terminal before enqueue.
                buffers[0]
                    .2
                    .upload_f32(&ctx.stream, &vec![7.0; m * n])
                    .unwrap();
                let recording = ctx.begin_gemm_route_recording(0).unwrap();
                assert!(gpu_gemm_typed_forward_raw(&ctx, y, x, w, None, shapes[0]).is_err());
                drop(recording);
                let mut output = vec![0.0; m * n];
                buffers[0].2.download_f32(&ctx.stream, &mut output).unwrap();
                assert_eq!(
                    output,
                    vec![7.0; m * n],
                    "invalid context recording must not enqueue"
                );
            }
        }
        assert_eq!(deny.calls(), 0);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn typed_matvec_physical_observation_preserves_public_output_and_storage() {
        let device = GpuDevice::new(0).unwrap();
        let ctx = GpuCtx::new(&device).unwrap();
        ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        for tc in [false, true] {
            ctx.set_bi_tensor_cores(tc);
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for output in [dtype, WeightDtype::F32] {
                    for k in [0, 37] {
                        let x = DtypedBuf::zeros(&ctx.stream, (3 * k).max(1), dtype).unwrap();
                        let w = DtypedBuf::zeros(&ctx.stream, (k * 16).max(1), dtype).unwrap();
                        let y = DtypedBuf::zeros(&ctx.stream, 3 * 16, output).unwrap();
                        let bias = DtypedBuf::zeros(&ctx.stream, 16, WeightDtype::F32).unwrap();
                        x.upload_f32(&ctx.stream, &vec![1.0; (3 * k).max(1)])
                            .unwrap();
                        w.upload_f32(&ctx.stream, &vec![1.0; (k * 16).max(1)])
                            .unwrap();
                        bias.upload_f32(&ctx.stream, &[0.5; 16]).unwrap();
                        let input = TypedPtr {
                            ptr: if k == 0 { 0 } else { x.cached_ptr() },
                            dtype,
                        };
                        let weight = TypedPtr {
                            ptr: if k == 0 { 0 } else { w.cached_ptr() },
                            dtype,
                        };
                        let output_ptr = TypedPtr {
                            ptr: y.cached_ptr(),
                            dtype: output,
                        };
                        gpu_gemm_typed_forward_raw(
                            &ctx,
                            output_ptr,
                            input,
                            weight,
                            Some(bias.cached_ptr()),
                            (3, k, 16),
                        )
                        .unwrap();
                        let mut expected = vec![0.0; 48];
                        y.download_f32(&ctx.stream, &mut expected).unwrap();
                        let mut ranges = vec![
                            PhysicalArgumentRange {
                                pointer: y.cached_ptr(),
                                required_bytes: (48 * output.size_bytes()) as u64,
                            },
                            PhysicalArgumentRange {
                                pointer: bias.cached_ptr(),
                                required_bytes: 64,
                            },
                        ];
                        if k != 0 {
                            ranges.push(PhysicalArgumentRange {
                                pointer: x.cached_ptr(),
                                required_bytes: (3 * k * dtype.size_bytes()) as u64,
                            });
                            ranges.push(PhysicalArgumentRange {
                                pointer: w.cached_ptr(),
                                required_bytes: (k * 16 * dtype.size_bytes()) as u64,
                            });
                        }
                        let mut observer = prepare_physical_observer(&ctx, 1, &ranges).unwrap();
                        let kernel = pick_bi_matvec(&ctx, dtype, dtype, output).unwrap();
                        let eager = ctx
                            .record_eager_gemm_trace(|| {
                                launch_bi_matvec(
                                    &ctx,
                                    kernel,
                                    BiGemmArgs {
                                        c: y.cached_ptr(),
                                        a: input.ptr,
                                        b: weight.ptr,
                                        bias: bias.cached_ptr(),
                                        alpha: 1.0,
                                        beta: 0.0,
                                        m: 3,
                                        n: 16,
                                        k: k as i32,
                                    },
                                    [dtype, dtype, output],
                                    &mut observer,
                                )
                            })
                            .unwrap();
                        let trace =
                            finish_recording_physical_observer(observer, ctx.gemm_route()).unwrap();
                        assert_eq!(eager.routes().len(), 1);
                        assert_eq!(trace.nodes().len(), 1);
                        let node = trace.nodes()[0];
                        let route = node.gemm_route().unwrap();
                        assert_eq!(route.backend, PhysicalGemmBackend::FixedMatvecEightWarpV1);
                        assert_eq!(
                            route.numeric_contract,
                            ResolvedNumericContract::ScalarFmaEightWarpTreePostDotBiasV1
                        );
                        assert_eq!(route.tile, (1, 32));
                        assert_eq!(route.bk, 0);
                        assert_eq!(node.launch.arguments_digest, route.launch.arguments_digest);
                        assert_ne!(
                            route.launch.arguments_digest,
                            eager.routes()[0].launch.arguments_digest
                        );
                        let mut actual = vec![0.0; 48];
                        y.download_f32(&ctx.stream, &mut actual).unwrap();
                        assert_eq!(actual, expected);
                        assert!(actual.iter().all(|v| *v == k as f32 + 0.5));
                    }
                }
            }
        }
    }
}
