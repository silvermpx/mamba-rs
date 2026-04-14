//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::kernels::MambaKernels;
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

    /// Ensure the half-precision staging buffer is at least `bytes` in size.
    /// Lazy allocation; reused across GEMMs.
    pub fn ensure_half_staging(&self, bytes: usize) -> Result<(), String> {
        let mut cur = self.half_staging_bytes.borrow_mut();
        if *cur >= bytes {
            return Ok(());
        }
        let new_size = bytes.max(*cur * 2).max(4096);
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
