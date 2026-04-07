//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::kernels::MambaKernels;
use std::sync::Arc;

/// GPU execution context — holds everything needed for kernel launches.
///
/// Created once at init, passed by reference to all GPU functions.
pub struct GpuCtx {
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub kernels: MambaKernels,
    pub blas: cudarc::cublas::CudaBlas,
    pub _blas_workspace: cudarc::driver::CudaSlice<u8>,
}

impl GpuCtx {
    /// Create a GPU context: compile kernels, init cuBLAS with TF32.
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        // Use a forked stream (not default) — CUDA Graph capture requires non-default stream.
        let stream = device.fork_stream()?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = MambaKernels::compile(device.context(), arch)?;
        let (blas, ws) = device.create_cublas(&stream)?;
        Ok(Self {
            stream,
            kernels,
            blas,
            _blas_workspace: ws,
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
}
