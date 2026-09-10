//! The STAMP line: one printed record of the numeric route a bench arm
//! measured (GEMM tier, kernel route, reduce depth, state capacity, NVRTC
//! arch, shape, build). A number without its stamp is not a reading:
//! timings taken on kernels production never launches have cost whole
//! sessions before the stamp made the route part of the output.

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

/// One STAMP line carrying the whole measurement identity. `route` names
/// the kernel path the arm drives (e.g. "fold G=4" / "ungrouped" /
/// "pooled_graph"), `depth` the reducer depth in rows (0 where the arm
/// has no reduce stage).
pub fn bench_stamp(
    dev: &GpuDevice,
    ctx: &GpuCtx,
    shape: &str,
    route: &str,
    depth: usize,
) -> String {
    let (bi, tc, fast) = ctx.gemm_flags();
    let tier = match (bi, tc) {
        (true, true) => "bi+tc",
        (true, false) => "bi",
        (false, _) if ctx.tf32() => "cublas+tf32",
        (false, _) => "cublas",
    };
    let fast = if fast { "+fast" } else { "" };
    let build = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!(
        "STAMP tier={tier}{fast} route={route} depth={depth} state_cap={} arch={} shape={shape} build={build}-{}",
        ctx.state_cap(),
        GpuDevice::nvrtc_arch(dev.compute_capability),
        env!("CARGO_PKG_VERSION"),
    )
}
