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

/// Device-side clip: launches the sum-of-squares reduction, folds the
/// partials into `(norm, coef)` on device, and scales the arena by that
/// coefficient - three back-to-back launches with NO host round trip
/// between them. Returns the PRE-clip norm, downloaded once at the end
/// (the caller needs it for the metric and for the non-finite check);
/// the difference from the old path is that the scaling pass is already
/// enqueued when the host blocks, instead of the GPU idling through a
/// sync, a 4 KB download and a 512-iteration host fold.
///
/// Bit contract: identical ordered f64 fold, identical sqrt, identical
/// coefficient, identical element-to-thread mapping in the scaling pass;
/// a coefficient of exactly 1.0 makes the pass a bitwise no-op, matching
/// the old conditional skip.
pub fn clip_grads_device(
    ctx: &GpuCtx,
    grads_flat: &mut GpuBuffer,
    partials: &mut GpuByteBuffer,
    scratch: &mut GpuBuffer,
    max_norm: f32,
) -> Result<f32, String> {
    assert_eq!(
        partials.len_bytes(),
        GRAD_CLIP_PARTIALS * std::mem::size_of::<f64>(),
        "grad-clip partials buffer has the wrong size"
    );
    assert_eq!(
        scratch.len(),
        2,
        "grad-clip scratch must hold exactly [coef, norm]"
    );
    let n_elems = grads_flat.len();
    let n = n_elems as i32;
    // Stage 1: fixed-grid sum of squares (geometry is contractual).
    {
        let dst = partials.cached_ptr();
        let src = grads_flat.cached_ptr();
        let mut b = ctx
            .stream
            .launch_builder(&ctx.kernels.grad_sumsq_partial_f32);
        b.arg(&dst);
        b.arg(&src);
        b.arg(&n);
        let cfg = LaunchConfig {
            grid_dim: (GRAD_CLIP_PARTIALS as u32, 1, 1),
            block_dim: (GRAD_CLIP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(|e| format!("grad_sumsq_partial_f32: {e:?}"))?;
    }
    // Stage 2: ordered fold -> (coef, norm), single thread.
    {
        let coef_ptr = scratch.cached_ptr();
        let norm_ptr = coef_ptr + std::mem::size_of::<f32>() as u64;
        let part_ptr = partials.cached_ptr();
        let np = GRAD_CLIP_PARTIALS as i32;
        let mut b = ctx.stream.launch_builder(&ctx.kernels.grad_clip_coef_f32);
        b.arg(&coef_ptr);
        b.arg(&norm_ptr);
        b.arg(&part_ptr);
        b.arg(&np);
        b.arg(&max_norm);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(|e| format!("grad_clip_coef_f32: {e:?}"))?;
    }
    // Stage 3: scale by the device-resident coefficient.
    {
        let coef_ptr = scratch.cached_ptr();
        let mut b = ctx.stream.launch_builder(&ctx.kernels.scale_grads_dev_f32);
        b.arg(grads_flat.inner_mut());
        b.arg(&coef_ptr);
        b.arg(&n);
        unsafe { b.launch(grid_1d(n_elems)) }.map_err(|e| format!("scale_grads_dev_f32: {e:?}"))?;
    }
    let mut host = [0.0f32; 2];
    scratch.download(&ctx.stream, &mut host)?;
    Ok(host[1])
}

/// Geometry of a per-layer two-block region of a flat grad arena:
/// block A is an `a_rows x a_cols` stripe starting at `a_off` with row
/// stride `a_row_stride` (a column slice of a row-major matrix), block
/// B a contiguous `b_len` vector at `b_off`; both repeat every
/// `layer_stride` elements for `n_layers` layers. Offsets are absolute
/// arena element offsets for layer 0.
#[derive(Clone, Copy, Debug)]
pub struct GradRegionGeom {
    pub n_layers: i32,
    pub layer_stride: i32,
    pub a_off: i32,
    pub a_rows: i32,
    pub a_cols: i32,
    pub a_row_stride: i32,
    pub b_off: i32,
    pub b_len: i32,
}

/// Region twin of [`clip_grads_device`]: deterministic L2 norm over the
/// described region, PyTorch clip coefficient against `max_norm`, and
/// in-place scaling of ONLY that region — three back-to-back launches,
/// one host download of the pre-clip region norm at the end. Same bit
/// contract as the global clip (ordered f64 fold; coefficient exactly
/// 1.0 is a bitwise no-op).
pub fn clip_region_device(
    ctx: &GpuCtx,
    grads_flat: &mut GpuBuffer,
    partials: &mut GpuByteBuffer,
    scratch: &mut GpuBuffer,
    geom: GradRegionGeom,
    max_norm: f32,
) -> Result<f32, String> {
    assert_eq!(
        partials.len_bytes(),
        GRAD_CLIP_PARTIALS * std::mem::size_of::<f64>(),
        "grad-clip partials buffer has the wrong size"
    );
    assert_eq!(
        scratch.len(),
        2,
        "grad-clip scratch must hold exactly [coef, norm]"
    );
    let fixed = LaunchConfig {
        grid_dim: (GRAD_CLIP_PARTIALS as u32, 1, 1),
        block_dim: (GRAD_CLIP_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // Stage 1: fixed-grid region sum of squares.
    {
        let dst = partials.cached_ptr();
        let src = grads_flat.cached_ptr();
        let mut b = ctx
            .stream
            .launch_builder(&ctx.kernels.grad_region_sumsq_partial_f32);
        b.arg(&dst);
        b.arg(&src);
        b.arg(&geom.n_layers);
        b.arg(&geom.layer_stride);
        b.arg(&geom.a_off);
        b.arg(&geom.a_rows);
        b.arg(&geom.a_cols);
        b.arg(&geom.a_row_stride);
        b.arg(&geom.b_off);
        b.arg(&geom.b_len);
        unsafe { b.launch(fixed) }.map_err(|e| format!("grad_region_sumsq_partial_f32: {e:?}"))?;
    }
    // Stage 2: the SAME ordered fold as the global clip.
    {
        let coef_ptr = scratch.cached_ptr();
        let norm_ptr = coef_ptr + std::mem::size_of::<f32>() as u64;
        let part_ptr = partials.cached_ptr();
        let np = GRAD_CLIP_PARTIALS as i32;
        let mut b = ctx.stream.launch_builder(&ctx.kernels.grad_clip_coef_f32);
        b.arg(&coef_ptr);
        b.arg(&norm_ptr);
        b.arg(&part_ptr);
        b.arg(&np);
        b.arg(&max_norm);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(|e| format!("grad_clip_coef_f32 (region): {e:?}"))?;
    }
    // Stage 3: scale ONLY the region by the device-resident coefficient.
    {
        let coef_ptr = scratch.cached_ptr();
        let mut b = ctx
            .stream
            .launch_builder(&ctx.kernels.grad_region_scale_dev_f32);
        b.arg(grads_flat.inner_mut());
        b.arg(&coef_ptr);
        b.arg(&geom.n_layers);
        b.arg(&geom.layer_stride);
        b.arg(&geom.a_off);
        b.arg(&geom.a_rows);
        b.arg(&geom.a_cols);
        b.arg(&geom.a_row_stride);
        b.arg(&geom.b_off);
        b.arg(&geom.b_len);
        unsafe { b.launch(fixed) }.map_err(|e| format!("grad_region_scale_dev_f32: {e:?}"))?;
    }
    let mut host = [0.0f32; 2];
    scratch.download(&ctx.stream, &mut host)?;
    Ok(host[1])
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
