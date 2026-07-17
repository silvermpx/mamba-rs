//! Deterministic global-norm gradient clipping — host side.
//!
//! The norm is computed by the fixed-grid `grad_sumsq_partial_f32` kernel
//! (kernels/grad_clip.cu): 512 blocks x 256 threads, fixed-stride element
//! assignment, per-thread f64 accumulation, fixed shared-memory tree reduce,
//! one f64 partial per block. The host then sums the 512 partials in order
//! and takes the square root — no atomics anywhere, so the norm is
//! bit-stable across runs. Scaling reuses the existing `scale_grads_f32`
//! elementwise kernel with the PyTorch `clip_grad_norm_` coefficient
//! `max_norm / (norm + 1e-6)`.
//!
//! This is an EAGER-path facility (the split `backward_step`): the norm is
//! an inherent host sync point. The fused captured step never computes it.

use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::buffers::{GpuBuffer, GpuByteBuffer};
use super::context::GpuCtx;
use super::launch::grid_1d;

/// Number of f64 partials the fixed-grid reduction produces. Must match
/// `GCLIP_BLOCKS` in kernels/grad_clip.cu.
pub const GRAD_CLIP_PARTIALS: usize = 512;

/// Threads per block of the reduction kernel. Must match `GCLIP_THREADS`
/// in kernels/grad_clip.cu.
const GRAD_CLIP_THREADS: u32 = 256;

/// Allocate the device-side partials buffer the norm kernel writes into.
pub fn alloc_partials(stream: &Arc<cudarc::driver::CudaStream>) -> Result<GpuByteBuffer, String> {
    GpuByteBuffer::zeros(stream, GRAD_CLIP_PARTIALS * std::mem::size_of::<f64>())
}

/// Compute the global L2 norm of the flat grad arena. Deterministic
/// (fixed-order reduction, f64 accumulation); synchronizes the stream
/// before the partials download.
pub fn global_grad_norm(
    ctx: &GpuCtx,
    grads_flat: &GpuBuffer,
    partials: &mut GpuByteBuffer,
    partials_host: &mut [f64],
) -> Result<f64, String> {
    assert_eq!(
        partials.len_bytes(),
        GRAD_CLIP_PARTIALS * std::mem::size_of::<f64>(),
        "grad-clip partials buffer has the wrong size"
    );
    assert_eq!(
        partials_host.len(),
        GRAD_CLIP_PARTIALS,
        "grad-clip host partials slice has the wrong length"
    );
    let n = grads_flat.len() as i32;
    let dst = partials.cached_ptr();
    let src = grads_flat.cached_ptr();
    let mut b = ctx
        .stream
        .launch_builder(&ctx.kernels.grad_sumsq_partial_f32);
    b.arg(&dst);
    b.arg(&src);
    b.arg(&n);
    // Geometry is part of the kernel's determinism contract — always the
    // full fixed grid, independent of n.
    let launch_cfg = LaunchConfig {
        grid_dim: (GRAD_CLIP_PARTIALS as u32, 1, 1),
        block_dim: (GRAD_CLIP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(launch_cfg) }.map_err(|e| format!("grad_sumsq_partial_f32: {e:?}"))?;
    ctx.stream
        .synchronize()
        .map_err(|e| format!("grad norm sync: {e:?}"))?;
    partials.download_f64(&ctx.stream, partials_host)?;
    // Ordered host sum — the final, fixed-order reduction stage.
    let mut sum = 0.0f64;
    for &p in partials_host.iter() {
        sum += p;
    }
    Ok(sum.sqrt())
}

/// In-place multiply of the flat grad arena by `factor` (the clip
/// coefficient). Reuses the AMP `scale_grads_f32` kernel.
pub fn scale_grads(ctx: &GpuCtx, grads_flat: &mut GpuBuffer, factor: f32) -> Result<(), String> {
    let n_elems = grads_flat.len();
    let n = n_elems as i32;
    let mut b = ctx.stream.launch_builder(&ctx.kernels.scale_grads_f32);
    b.arg(grads_flat.inner_mut());
    b.arg(&factor);
    b.arg(&n);
    unsafe { b.launch(grid_1d(n_elems)) }
        .map(|_| ())
        .map_err(|e| format!("scale_grads (clip): {e:?}"))
}
