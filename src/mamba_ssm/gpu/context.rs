//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::dtype::WeightDtype;
use super::kernels::MambaKernels;
use crate::config::MambaConfig;
use std::cell::RefCell;
use std::sync::Arc;

/// GPU execution context — holds everything needed for kernel launches.
///
/// Created once at init, passed by reference to all GPU functions.
pub struct GpuCtx {
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub kernels: MambaKernels,
    pub blas: cudarc::cublas::CudaBlas,
    pub _blas_workspace: cudarc::driver::CudaSlice<u8>,
    /// Reusable GPU byte staging buffer for f32→bf16/f16 activation downcast
    /// before mixed-precision GEMM. Grown lazily on first use.
    half_staging: RefCell<Option<cudarc::driver::CudaSlice<u8>>>,
    half_staging_ptr: RefCell<cudarc::driver::sys::CUdeviceptr>,
    half_staging_bytes: RefCell<usize>,
}

impl GpuCtx {
    /// Create a GPU context: compile kernels, init cuBLAS with TF32.
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        let stream = device.fork_stream()?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = MambaKernels::compile(device.context(), arch)?;
        let (blas, ws) = device.create_cublas(&stream)?;
        Ok(Self {
            stream,
            kernels,
            blas,
            _blas_workspace: ws,
            half_staging: RefCell::new(None),
            half_staging_ptr: RefCell::new(0),
            half_staging_bytes: RefCell::new(0),
        })
    }

    /// Disable TF32 Tensor Cores — use full f32 SGEMM for parity tests.
    pub fn disable_tf32(&self) {
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *self.blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            );
        }
    }

    /// Pre-size the half-precision staging buffer for a known engine
    /// config, batch, and dtype. Eliminates lazy-grow during the hot path
    /// — critical for CUDA Graph capture safety: if a captured graph baked
    /// a staging pointer and a later call grew the buffer, the freed
    /// allocation would be dereferenced on replay (CUDA_ERROR_ILLEGAL_ADDRESS
    /// or silent corruption). Sizes for the worst-case step-time GEMM
    /// operand (in_proj input = batch × d_model, the largest staging consumer
    /// in step_kernels). Idempotent — safe to call multiple times.
    pub fn presize_half_staging_for_step(
        &self,
        cfg: &MambaConfig,
        batch: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let max_in_elems = batch * dm.max(di).max(dt_rank);
        let bytes = max_in_elems * dtype.size_bytes();
        self.ensure_half_staging(bytes)
    }

    /// Ensure the half-precision staging buffer is at least `bytes` in size.
    /// In the steady state this is a no-op when `presize_half_staging_for_step`
    /// was called at engine construction; the lazy grow path remains as a
    /// fallback for prefill (which runs outside any captured graph) or for
    /// callers that don't presize.
    pub fn ensure_half_staging(&self, bytes: usize) -> Result<(), String> {
        let mut cur = self.half_staging_bytes.borrow_mut();
        if *cur >= bytes {
            return Ok(());
        }
        // Grow by at least the requested size, rounded up to a 4 KiB page —
        // no speculative doubling that wastes memory at the plateau (the old
        // `bytes.max(*cur * 2)` rule could leave us at 4× the actual need
        // after a few growths).
        let page = 4096;
        let new_size = bytes.div_ceil(page) * page;
        let buf = self
            .stream
            .alloc_zeros::<u8>(new_size)
            .map_err(|e| format!("half_staging alloc {new_size}B failed: {e:?}"))?;
        let ptr = {
            use cudarc::driver::DevicePtr;
            let (p, _g) = buf.device_ptr(&self.stream);
            p
        };
        *self.half_staging.borrow_mut() = Some(buf);
        *self.half_staging_ptr.borrow_mut() = ptr;
        *cur = new_size;
        Ok(())
    }

    pub fn half_staging_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        *self.half_staging_ptr.borrow()
    }
}
