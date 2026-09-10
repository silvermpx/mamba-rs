//! CudaContext wrapper with stream management.
//!
//! Thin wrapper around cudarc::driver::CudaContext that provides:
//! - Device initialization with compute capability detection
//! - Default stream creation
//! - Device info logging (name, VRAM, SM count)

use std::sync::Arc;

/// Wrapper around CudaContext with convenience methods.
///
/// Device topology is captured once and cannot drift from its identity.
///
/// ```compile_fail
/// use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
///
/// fn rewrite_topology(device: &mut GpuDevice) {
///     device.multiprocessor_count = 1;
/// }
/// ```
pub struct GpuDevice {
    ctx: Arc<cudarc::driver::CudaContext>,
    /// Compute capability (major, minor). E.g., (9, 0) for Hopper/GH200.
    pub compute_capability: (u32, u32),
    nvrtc_target: &'static str,
    identity: super::kernel_identity::DeviceIdentity,
}

impl GpuDevice {
    /// Initialize CUDA device by ordinal (0 = first GPU).
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let ctx = cudarc::driver::CudaContext::new(ordinal)
            .map_err(|e| format!("CUDA device {} init failed: {:?}", ordinal, e))?;

        let cc = Self::query_compute_capability(ordinal)?;
        let multiprocessor_count = Self::query_multiprocessor_count(ordinal)?;
        let nvrtc_target = Self::resolve_nvrtc_target_for_nvrtc(cc, Self::query_nvrtc_version()?)?;
        let target = super::kernel_identity::CudaTarget::new(Self::resolve_nvrtc_target(cc)?)?;
        let driver = super::kernel_identity::query_driver_identity()?;

        Ok(Self {
            ctx,
            compute_capability: cc,
            nvrtc_target,
            identity: super::kernel_identity::DeviceIdentity {
                compute_capability: cc,
                multiprocessor_count,
                target,
                driver,
            },
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

    fn query_multiprocessor_count(ordinal: usize) -> Result<u32, String> {
        use cudarc::driver::sys;
        let device = i32::try_from(ordinal)
            .map_err(|_| format!("CUDA device ordinal {ordinal} exceeds i32::MAX"))?;
        let mut count = 0;
        let result = unsafe {
            sys::cuDeviceGetAttribute(
                &mut count,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device,
            )
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!(
                "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed for device {ordinal}: {result:?}"
            ));
        }
        let count = u32::try_from(count).map_err(|_| {
            format!("CUDA device {ordinal} returned negative multiprocessor count {count}")
        })?;
        if count == 0 {
            return Err(format!(
                "CUDA device {ordinal} returned zero multiprocessors"
            ));
        }
        Ok(count)
    }

    /// Resolve the NVRTC target without silently lowering the device family.
    pub fn resolve_nvrtc_target(cc: (u32, u32)) -> Result<&'static str, String> {
        let target = match cc {
            (8, 0) => "sm_80",
            (8, 6) => "sm_86",
            (8, 7) => "sm_87",
            (8, 9) => "sm_89",
            (9, 0) => "sm_90a",
            (10, 0) => "sm_100a",
            (10, 1) => "sm_101a",
            (10, 3) => "sm_103a",
            // The arch-specific target, like the CC 10.x parts: the baseline
            // sm_110 compiles the portable ladder only and leaves the Fixed
            // tcgen05 rung out of the PTX.
            (11, 0) => "sm_110a",
            (12, 0) => "sm_120",
            (12, 1) => "sm_121",
            // A minor this table does not name still belongs to the family
            // and runs the family's virtual target.
            (12, _) => "compute_120",
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

    fn query_nvrtc_version() -> Result<(i32, i32), String> {
        let mut major = 0;
        let mut minor = 0;
        let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
        if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcVersion failed: {result:?}"));
        }
        Ok((major, minor))
    }

    fn resolve_nvrtc_target_for_nvrtc(
        cc: (u32, u32),
        nvrtc_version: (i32, i32),
    ) -> Result<&'static str, String> {
        match cc {
            (10, 1) if ((12, 8)..(13, 0)).contains(&nvrtc_version) => Ok("sm_101a"),
            (10, 1) => Err(format!(
                "CUDA {}.{} cannot compile compute capability 10.1; the SM101 target is available in CUDA 12.8 and 12.9",
                nvrtc_version.0, nvrtc_version.1
            )),
            (10, 3) if nvrtc_version >= (12, 9) => Ok("sm_103a"),
            (10, 3) => Err(format!(
                "CUDA {}.{} cannot compile compute capability 10.3; SM103 needs CUDA 12.9 or newer",
                nvrtc_version.0, nvrtc_version.1
            )),
            (11, 0) if nvrtc_version >= (13, 2) => Ok("sm_110a"),
            (11, 0) => Err(format!(
                "CUDA {}.{} cannot compile compute capability 11.0; SM110 needs CUDA 13.2 or newer",
                nvrtc_version.0, nvrtc_version.1
            )),
            (12, 0) if nvrtc_version >= (12, 8) => Ok("compute_120"),
            (12, 1) if nvrtc_version >= (12, 9) => Ok("compute_121"),
            (12, _) if nvrtc_version >= (12, 8) => Ok("compute_120"),
            (12, _) => Err(format!(
                "CUDA {}.{} cannot compile compute capability {}.{}; SM120 needs CUDA 12.8 and SM121 needs CUDA 12.9 for its native generic target",
                nvrtc_version.0, nvrtc_version.1, cc.0, cc.1
            )),
            _ => Self::resolve_nvrtc_target(cc),
        }
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

    /// Immutable device and driver domain used by graph route snapshots.
    pub fn identity(&self) -> super::kernel_identity::DeviceIdentity {
        self.identity
    }

    /// Physical SM count captured in the immutable device identity.
    pub fn multiprocessor_count(&self) -> u32 {
        self.identity.multiprocessor_count
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
    /// Configures handle math for the requested context GEMM mode.
    /// Pre-allocates 32 MiB workspace for CUDA Graph compatibility.
    ///
    /// The handle MUST be bound to the same stream used for CUDA Graph capture,
    /// otherwise cuBLAS SGEMM operations will not be recorded into the graph.
    pub fn create_cublas(
        &self,
        compute_stream: &Arc<cudarc::driver::CudaStream>,
        mode: super::GemmMode,
    ) -> Result<(cudarc::cublas::CudaBlas, cudarc::driver::CudaSlice<u8>), String> {
        let blas = cudarc::cublas::CudaBlas::new(compute_stream.clone())
            .map_err(|e| format!("cuBLAS init failed: {:?}", e))?;

        unsafe {
            let status = cudarc::cublas::sys::cublasSetMathMode(
                *blas.handle(),
                mode.cublas_math(),
            );
            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(format!(
                    "cublasSetMathMode for GEMM mode {} failed: {status:?}",
                    mode.as_str()
                ));
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

/// Whether `cc` belongs to the SM120 family (consumer Blackwell).
///
/// Every minor of major 12 runs the `compute_120` PTX the SM120 kernels are
/// built for, so the loader and every selector key on this one predicate:
/// a kernel is never compiled for a board that cannot select it, and never
/// selected on a board it was not compiled for. Measured tables stay scoped
/// to the exact board they were measured on; this only says which family the
/// kernels serve.
pub fn is_sm120_family(cc: (u32, u32)) -> bool {
    cc.0 == 12
}

#[cfg(test)]
mod tests {
    use super::{GpuDevice, is_sm120_family};

    #[test]
    fn nvrtc_target_accepts_supported_sm80_plus() {
        let cases = [
            ((8, 0), "sm_80"),
            ((8, 6), "sm_86"),
            ((8, 7), "sm_87"),
            ((8, 9), "sm_89"),
            ((9, 0), "sm_90a"),
            ((10, 0), "sm_100a"),
            ((10, 1), "sm_101a"),
            ((10, 3), "sm_103a"),
            ((11, 0), "sm_110a"),
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
    fn sm120_targets_follow_the_installed_nvrtc_floor() {
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 0), (12, 7)).is_err());
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 0), (12, 8)),
            Ok("compute_120")
        );
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 1), (12, 8)),
            Ok("compute_120")
        );
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 1), (12, 9)),
            Ok("compute_121")
        );
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 1), (13, 2)),
            Ok("compute_121")
        );
    }

    #[test]
    fn sm110_target_requires_cuda_13_2() {
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((11, 0), (13, 1)).is_err());
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((11, 0), (13, 2)),
            Ok("sm_110a")
        );
    }

    #[test]
    fn sm103_target_requires_cuda_12_9() {
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 3), (12, 8)).is_err());
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 3), (12, 9)),
            Ok("sm_103a")
        );
    }

    #[test]
    fn sm101_target_uses_its_cuda_12_native_name() {
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 1), (12, 7)).is_err());
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 1), (12, 8)),
            Ok("sm_101a")
        );
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 1), (12, 9)),
            Ok("sm_101a")
        );
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((10, 1), (13, 0)).is_err());
    }

    #[test]
    fn nvrtc_target_rejects_cc_below_80() {
        for cc in [(6, 0), (7, 0), (7, 5)] {
            assert!(GpuDevice::resolve_nvrtc_target(cc).is_err());
        }
    }

    #[test]
    fn nvrtc_target_rejects_unknown_known_family_minor() {
        for cc in [(8, 1), (9, 1), (10, 2), (11, 1)] {
            assert!(GpuDevice::resolve_nvrtc_target(cc).is_err());
        }
    }

    #[test]
    fn unmapped_sm120_minor_takes_the_family_virtual_target() {
        assert_eq!(
            GpuDevice::resolve_nvrtc_target((12, 2)).unwrap(),
            "compute_120"
        );
        assert_eq!(
            GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 2), (13, 2)).unwrap(),
            "compute_120"
        );
        assert!(GpuDevice::resolve_nvrtc_target_for_nvrtc((12, 2), (12, 7)).is_err());
        for cc in [(12, 0), (12, 1), (12, 2)] {
            assert!(is_sm120_family(cc));
        }
        for cc in [(8, 9), (9, 0), (10, 0), (13, 0)] {
            assert!(!is_sm120_family(cc));
        }
    }
}
