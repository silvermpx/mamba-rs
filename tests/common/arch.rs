//! NVRTC arch of CUDA device 0, resolved once per test binary.
//!
//! Every kernel compile in a test names the arch through this helper
//! instead of a literal, so a reading on a new board comes off that
//! board's SASS rather than PTX JIT-compiled for another one.

use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

/// NVRTC arch of device 0 — the cure for the hardcoded "sm_89" that had
/// every M3 isolated number on the 5090 read off JIT'd sm_89 PTX instead
/// of sm_120 SASS.
pub fn arch0() -> &'static str {
    static ARCH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ARCH.get_or_init(|| {
        let dev = GpuDevice::new(0).unwrap();
        GpuDevice::nvrtc_arch(dev.compute_capability).to_string()
    })
}
