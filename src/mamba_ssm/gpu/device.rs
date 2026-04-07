//! CudaContext wrapper with stream management.
//!
//! Thin wrapper around cudarc::driver::CudaContext that provides:
//! - Device initialization with compute capability detection
//! - Default stream creation
//! - Device info logging (name, VRAM, SM count)

use std::sync::Arc;

/// Wrapper around CudaContext with convenience methods.
pub struct GpuDevice {
    ctx: Arc<cudarc::driver::CudaContext>,
    /// Compute capability (major, minor). E.g., (9, 0) for Hopper/GH200.
    pub compute_capability: (u32, u32),
}

impl GpuDevice {
    /// Initialize CUDA device by ordinal (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = cudarc::driver::CudaContext::new(ordinal)
            .map_err(|e| format!("CUDA device {} init failed: {:?}", ordinal, e))?;

        let cc = Self::query_compute_capability(ordinal)?;

        Ok(Self {
            ctx,
            compute_capability: cc,
        })
    }

    /// Query GPU compute capability (major, minor).
    fn query_compute_capability(ordinal: usize) -> Result<(u32, u32), String> {
        use cudarc::driver::sys;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            let dev = ordinal as i32;
            let r1 = sys::cuDeviceGetAttribute(
                &mut major,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                dev,
            );
            if r1 != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!(
                    "cuDeviceGetAttribute(MAJOR) failed for device {ordinal}: {r1:?}"
                ));
            }
            let r2 = sys::cuDeviceGetAttribute(
                &mut minor,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            );
            if r2 != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!(
                    "cuDeviceGetAttribute(MINOR) failed for device {ordinal}: {r2:?}"
                ));
            }
        }
        Ok((major as u32, minor as u32))
    }

    /// Get the NVRTC real-architecture target string for this GPU.
    ///
    /// Returns `sm_XX` (real architecture) to compile directly to CUBIN (native SASS),
    /// bypassing PTX entirely. This avoids CUDA_ERROR_UNSUPPORTED_PTX_VERSION when
    /// the NVRTC toolkit version generates a PTX ISA version newer than what the
    /// installed driver's JIT compiler supports (e.g., CUDA 12.8 NVRTC + driver 590).
    ///
    /// Trade-off: CUBIN is GPU-specific and not forward-compatible with newer GPUs.
    /// This is fine — we detect the exact GPU at init and compile for it.
    pub fn nvrtc_arch(cc: (u32, u32)) -> &'static str {
        match cc {
            (12, _) => "sm_120", // Blackwell consumer (RTX 5090, RTX 5080, RTX 5070)
            (10, _) => "sm_100", // Blackwell datacenter (B100, B200, GB200)
            (9, _) => "sm_90",   // Hopper (H100, H200, GH200)
            (8, 9) => "sm_89",   // Ada Lovelace (RTX 4090, RTX 4080, RTX 6000 Ada)
            (8, 6) => "sm_86",   // Ampere consumer (RTX 3090, RTX 3080, RTX 3070)
            (8, 0) => "sm_80",   // Ampere datacenter (A100, A30)
            (7, 5) => "sm_75",   // Turing (RTX 2080, RTX 2070, T4)
            (7, 0) => "sm_70",   // Volta (V100, Titan V)
            (6, 1) => "sm_61",   // Pascal consumer (GTX 1080, GTX 1070)
            (6, 0) => "sm_60",   // Pascal datacenter (P100)
            _ => {
                if cc.0 > 12 {
                    "sm_120" // Future architectures — use latest known
                } else {
                    "sm_70" // Ancient GPUs — Volta fallback
                }
            }
        }
    }

    /// Get the default CUDA stream for this device.
    pub fn default_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.ctx.default_stream()
    }

    /// Create a new CUDA stream for async operations.
    pub fn fork_stream(&self) -> Result<Arc<cudarc::driver::CudaStream>, String> {
        self.ctx
            .default_stream()
            .fork()
            .map_err(|e| format!("stream fork failed: {:?}", e))
    }

    /// Get the underlying CudaContext for direct API access.
    pub fn context(&self) -> &Arc<cudarc::driver::CudaContext> {
        &self.ctx
    }

    /// Create a cuBLAS handle bound to the given compute stream.
    /// Enables TF32 Tensor Core math for ~8x SGEMM throughput on A100/GH200.
    /// Pre-allocates 32 MiB workspace for CUDA Graph compatibility.
    ///
    /// The handle MUST be bound to the same stream used for CUDA Graph capture,
    /// otherwise cuBLAS SGEMM operations will not be recorded into the graph.
    pub fn create_cublas(
        &self,
        compute_stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(cudarc::cublas::CudaBlas, cudarc::driver::CudaSlice<u8>), String> {
        let blas = cudarc::cublas::CudaBlas::new(compute_stream.clone())
            .map_err(|e| format!("cuBLAS init failed: {:?}", e))?;

        // Enable TF32 Tensor Cores for all SGEMM operations.
        unsafe {
            let status = cudarc::cublas::sys::cublasSetMathMode(
                *blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            );
            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err("cublasSetMathMode TF32 failed".into());
            }
        }

        // Pre-allocate cuBLAS workspace for CUDA Graph compatibility.
        // Without this, cuBLAS allocates workspace internally on each graph capture,
        // leaking memory on reinit(). 32 MiB recommended for Hopper kernels.
        let workspace_bytes: usize = 32 * 1024 * 1024;
        let workspace = compute_stream
            .alloc_zeros::<u8>(workspace_bytes)
            .map_err(|e| format!("cuBLAS workspace alloc failed: {:?}", e))?;
        unsafe {
            use cudarc::driver::DevicePtr;
            let (ws_ptr, _guard) = workspace.device_ptr(compute_stream);
            let status = cudarc::cublas::sys::cublasSetWorkspace_v2(
                *blas.handle(),
                ws_ptr as *mut std::ffi::c_void,
                workspace_bytes,
            );
            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err("cublasSetWorkspace_v2 failed".into());
            }
        }

        Ok((blas, workspace))
    }
}
