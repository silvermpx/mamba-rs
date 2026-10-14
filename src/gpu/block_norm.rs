//! The block norms on the device: the norm before each mixer, the one
//! before an MLP and the final norm. A model built with `rms_norm = true`
//! (every official checkpoint) runs them as RMSNorm, a scale and no bias;
//! `rms_norm = false` makes them LayerNorm with a bias, mamba_ssm's
//! `nn.LayerNorm` `norm_cls`. The norms inside the mixers are RMS either
//! way, as in the reference.
//!
//! A norm's kind follows its bias: a null bias pointer is RMSNorm. The RMS
//! launches here are the ones the call sites made by hand, the same kernels
//! with the same arguments and geometry, so a site routed through them
//! keeps its bits.

use cudarc::driver::PushKernelArg;

use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::grid_norm;

/// One block norm's f32 parameters on the device: the scale, and the bias
/// for LayerNorm (0 for RMSNorm).
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormParams {
    pub w: u64,
    pub b: u64,
}

impl NormParams {
    pub(crate) fn is_layer_norm(&self) -> bool {
        self.b != 0
    }
}

/// The norm of `rows` f32 rows of `x`, `dim` wide, into `y` stored as
/// `out` (f32 for F32 and TF32, else the half type); `stats` takes the
/// rows' statistics (`BlockNorm::stats_per_row` each, interleaved).
pub(crate) fn forward(
    ctx: &GpuCtx,
    out: WeightDtype,
    (y, stats, x): (u64, u64, u64),
    p: NormParams,
    (rows, dim, eps): (usize, usize, f32),
    what: &str,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let k = &ctx.kernels;
    let kernel = match (p.is_layer_norm(), out) {
        (true, dt) => k.layernorm_fwd_typed.get(dt),
        (false, WeightDtype::F32 | WeightDtype::Tf32) => &k.rmsnorm_fwd,
        (false, dt) => k.rmsnorm_fwd_f32in_typed.get(dt),
    };
    let (rows_i, dim_i) = (rows as i32, dim as i32);
    let mut bld = ctx.stream.launch_builder(kernel);
    bld.arg(&y);
    bld.arg(&stats);
    bld.arg(&x);
    bld.arg(&p.w);
    if p.is_layer_norm() {
        bld.arg(&p.b);
    }
    bld.arg(&rows_i);
    bld.arg(&dim_i);
    bld.arg(&eps);
    unsafe { bld.launch(grid_norm(rows, dim)) }
        .map(|_| ())
        .map_err(|e| format!("{what}: {e:?}"))
}

/// The previous branch's residual add fused into the norm: `resid +=
/// branch` in place (`branch` stored as `dtype`), then the norm of the sum
/// into `y`, stored as `dtype` too. The sum and the output carry the bits
/// of the residual add followed by [`forward`].
pub(crate) fn forward_resadd(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    (y, stats, resid, branch): (u64, u64, u64, u64),
    p: NormParams,
    (rows, dim, eps): (usize, usize, f32),
    what: &str,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let k = &ctx.kernels;
    let kernel = if p.is_layer_norm() {
        k.layernorm_fwd_resadd_typed.get(dtype)
    } else {
        k.rmsnorm_fwd_resadd_typed.get(dtype)
    };
    let (rows_i, dim_i) = (rows as i32, dim as i32);
    let mut bld = ctx.stream.launch_builder(kernel);
    bld.arg(&y);
    bld.arg(&stats);
    bld.arg(&resid);
    bld.arg(&branch);
    bld.arg(&p.w);
    if p.is_layer_norm() {
        bld.arg(&p.b);
    }
    bld.arg(&rows_i);
    bld.arg(&dim_i);
    bld.arg(&eps);
    unsafe { bld.launch(grid_norm(rows, dim)) }
        .map(|_| ())
        .map_err(|e| format!("{what}: {e:?}"))
}

/// The backward of [`forward`] over `rows` rows: `dx` (f32) takes the
/// input gradient, added onto it when `accumulate`, and its copy stored as
/// `dy_dtype` into `dx_typed` unless that is 0; `partials` takes the
/// scale's per-row gradients for the caller's reduction. For LayerNorm the
/// bias gradient, the column sum of `dy`, is added into `g_bias` first:
/// `dx` may alias `dy`.
pub(crate) fn backward(
    ctx: &GpuCtx,
    dy_dtype: WeightDtype,
    (dx, partials, dy, x, stats): (u64, u64, u64, u64, u64),
    (p, g_bias): (NormParams, u64),
    (rows, dim): (usize, usize),
    (accumulate, dx_typed): (bool, u64),
    what: &str,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    if p.is_layer_norm() {
        reduce_bias(ctx, dy_dtype, (g_bias, dy), (rows, dim), what)?;
    }
    let k = &ctx.kernels;
    let kernel = match (p.is_layer_norm(), dy_dtype) {
        (true, dt) => k.layernorm_bwd_typed.get(dt),
        (false, WeightDtype::F32 | WeightDtype::Tf32) => &k.rmsnorm_bwd,
        (false, dt) => k.rmsnorm_bwd_f32in_typed.get(dt),
    };
    let (rows_i, dim_i) = (rows as i32, dim as i32);
    let acc = i32::from(accumulate);
    let mut bld = ctx.stream.launch_builder(kernel);
    bld.arg(&dx);
    bld.arg(&partials);
    bld.arg(&dy);
    bld.arg(&x);
    bld.arg(&p.w);
    bld.arg(&stats);
    bld.arg(&rows_i);
    bld.arg(&dim_i);
    bld.arg(&acc);
    bld.arg(&dx_typed);
    unsafe { bld.launch(grid_norm(rows, dim)) }
        .map(|_| ())
        .map_err(|e| format!("{what}: {e:?}"))
}

/// `db[i] += sum over rows of dy[row, i]` for a `dy` stored as `dtype`,
/// into an f32 gradient: one block per column, a fixed tree, no atomics.
pub(crate) fn reduce_bias(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    (db, dy): (u64, u64),
    (rows, n): (usize, usize),
    what: &str,
) -> Result<(), String> {
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let (rows_i, n_i) = (rows as i32, n as i32);
    let mut bld = ctx
        .stream
        .launch_builder(ctx.kernels.reduce_bias_typed.get(dtype));
    bld.arg(&db);
    bld.arg(&dy);
    bld.arg(&rows_i);
    bld.arg(&n_i);
    // Threads sum over the rows; shared memory holds one partial per thread.
    let threads = 256u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: (threads as usize * std::mem::size_of::<f32>()) as u32,
    };
    unsafe { bld.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("{what} bias: {e:?}"))
}
