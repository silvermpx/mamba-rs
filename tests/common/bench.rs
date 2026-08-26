//! Shared bench harness: the STAMP line, hardened timing, and FNV output
//! hashes.
//!
//! Every bench arm prints ONE stamp naming the full numeric route it
//! measured — tier (all four GEMM bits), kernel route, reduce depth,
//! state capacity, NVRTC arch, shape, build. The 0.6.3 wave lost half a
//! day to readings taken on the wrong tier and shipped two ledger rows
//! measured on kernels production never launches; the stamp is the
//! structural cure: a number without its stamp is not a reading.

use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
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

/// Hardened wall timing: ~1s warm spin (not a fixed iteration count — a
/// 0.1ms kernel and a 4ms window need different warm depths), sync, then
/// `iters` timed launches, sync, mean milliseconds per iteration.
pub fn timed(ctx: &GpuCtx, iters: usize, mut f: impl FnMut()) -> f64 {
    let warm_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while std::time::Instant::now() < warm_deadline {
        f();
    }
    ctx.stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    ctx.stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters.max(1) as f64
}

/// The digest law lives in `common::digest`; re-exported here because
/// every bench printer reaches for it alongside the timing helpers.
pub use super::digest::fnv1a_f32;

/// Print one `HASH <name> <hex>` line per buffer (f32 elements, `n`
/// leading elements each). The bit gate for kernels without digest
/// coverage: record once, compare forever.
pub fn hash_outputs(ctx: &GpuCtx, bufs: &[(&str, &GpuBuffer, usize)]) {
    for (name, buf, n) in bufs {
        let mut v = vec![0f32; *n];
        buf.download(&ctx.stream, &mut v).unwrap();
        ctx.stream.synchronize().unwrap();
        eprintln!("HASH {name} {:016x}", fnv1a_f32(&v));
    }
}
