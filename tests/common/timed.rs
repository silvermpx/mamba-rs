//! Hardened wall timing for GPU benches.

use mamba_rs::mamba_ssm::gpu::context::GpuCtx;

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
