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
    nvrtc_target: &'static str,
}

impl GpuDevice {
    /// Initialize CUDA device by ordinal (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = cudarc::driver::CudaContext::new(ordinal)
            .map_err(|e| format!("CUDA device {} init failed: {:?}", ordinal, e))?;

        let cc = Self::query_compute_capability(ordinal)?;
        let nvrtc_target = Self::resolve_nvrtc_target(cc)?;

        Ok(Self {
            ctx,
            compute_capability: cc,
            nvrtc_target,
        })
    }

    /// Query GPU compute capability (major, minor).
    fn query_compute_capability(ordinal: usize) -> Result<(u32, u32), String> {
        use cudarc::driver::sys;
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            let dev = i32::try_from(ordinal)
                .map_err(|_| format!("CUDA device ordinal {ordinal} exceeds i32::MAX"))?;
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
        let major = u32::try_from(major)
            .map_err(|_| format!("CUDA device {ordinal} returned negative CC major {major}"))?;
        let minor = u32::try_from(minor)
            .map_err(|_| format!("CUDA device {ordinal} returned negative CC minor {minor}"))?;
        Ok((major, minor))
    }

    /// Resolve the NVRTC target without silently lowering the device family.
    pub fn resolve_nvrtc_target(cc: (u32, u32)) -> Result<&'static str, String> {
        let target = match cc {
            (8, 0) => "sm_80",
            (8, 6) => "sm_86",
            (8, 7) => "sm_87",
            (8, 9) => "sm_89",
            (9, 0) => "sm_90",
            (10, 0) => "sm_100",
            (10, 3) => "sm_103",
            (12, 0) => "sm_120",
            (12, 1) => "sm_121",
            (major, _) if major > 12 => "compute_120",
            _ => {
                return Err(format!(
                    "unsupported CUDA compute capability {}.{}; deterministic GPU kernels require SM80 or newer with a known target",
                    cc.0, cc.1
                ));
            }
        };
        Ok(target)
    }

    /// Compatibility wrapper for callers that already validated `cc`.
    /// Production compilation uses the target stored by [`Self::new`].
    pub fn nvrtc_arch(cc: (u32, u32)) -> &'static str {
        Self::resolve_nvrtc_target(cc)
            .expect("GpuDevice::nvrtc_arch requires a target validated by GpuDevice::new")
    }

    /// NVRTC target validated when the device was opened.
    pub fn nvrtc_target(&self) -> &'static str {
        self.nvrtc_target
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

        // Enable TF32 Tensor Cores for all SGEMM operations. Per-op
        // compute type for bf16/f16 paths is CUBLAS_COMPUTE_32F_PEDANTIC
        // so TF32 math mode doesn't actually apply to typed GEMMs —
        // this is purely an f32-SGEMM perf flag.
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

#[cfg(test)]
mod tests {
    use super::GpuDevice;

    #[test]
    fn nvrtc_target_accepts_supported_sm80_plus() {
        let cases = [
            ((8, 0), "sm_80"),
            ((8, 6), "sm_86"),
            ((8, 7), "sm_87"),
            ((8, 9), "sm_89"),
            ((9, 0), "sm_90"),
            ((10, 0), "sm_100"),
            ((10, 3), "sm_103"),
            ((12, 0), "sm_120"),
            ((12, 1), "sm_121"),
        ];

        for (cc, expected) in cases {
            assert_eq!(GpuDevice::resolve_nvrtc_target(cc), Ok(expected));
        }
    }

    #[test]
    fn nvrtc_target_uses_virtual_arch_for_future_major() {
        assert_eq!(GpuDevice::resolve_nvrtc_target((13, 0)), Ok("compute_120"));
    }

    #[test]
    fn nvrtc_target_rejects_cc_below_80() {
        for cc in [(6, 0), (7, 0), (7, 5)] {
            assert!(GpuDevice::resolve_nvrtc_target(cc).is_err());
        }
    }

    #[test]
    fn nvrtc_target_rejects_unknown_known_family_minor() {
        for cc in [(8, 1), (9, 1), (10, 1), (12, 2)] {
            assert!(GpuDevice::resolve_nvrtc_target(cc).is_err());
        }
    }
}
