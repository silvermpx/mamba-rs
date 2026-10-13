//! FalconMamba's dt, B and C mixer norms on the device: the forward over
//! x_proj rows in place, and the backward folded into the d_xdbl assembly.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;

/// One block of three warps per row, the kernels' layout; rows past the
/// grid loop inside the kernel.
fn rows_cfg(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.min(1 << 20) as u32, 1, 1),
        block_dim: (96, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn int(v: usize, what: &str) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| format!("bcdt norm: {what} {v} past i32"))
}

/// FalconMamba's norms of `rows` x_proj rows at `xdbl`, `dt_rank + 2
/// d_state` values of `dtype` wide, in place. `rstd`, when non-zero, takes
/// each row's three scales.
pub(crate) fn launch_bcdt_norm(
    ctx: &GpuCtx,
    (xdbl, rstd): (u64, u64),
    dtype: WeightDtype,
    (rows, dt_rank, d_state): (usize, usize, usize),
    eps: f32,
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let (dr, ds) = (int(dt_rank, "dt_rank")?, int(d_state, "d_state")?);
    let stride = int(dt_rank + 2 * d_state, "row width")?;
    let rows_l = i64::try_from(rows).map_err(|_| "bcdt norm: rows past i64")?;
    let mut bld = ctx
        .stream
        .launch_builder(ctx.kernels.bcdt_norm_fwd_typed.get(dtype));
    bld.arg(&xdbl);
    bld.arg(&rstd);
    bld.arg(&rows_l);
    bld.arg(&stride);
    bld.arg(&dr);
    bld.arg(&ds);
    bld.arg(&eps);
    unsafe { bld.launch(rows_cfg(rows)) }
        .map(drop)
        .map_err(|e| format!("bcdt_norm_fwd: {e:?}"))
}

/// The pointers of the d_xdbl assembly through the norms' backward: the
/// output and the dt gradient in the compute dtype, the B and C gradients
/// in f32, the normalized rows the forward stored and their scales.
pub(crate) struct NormBackward {
    pub dst: u64,
    pub dt_src: u64,
    pub b_src: u64,
    pub c_src: u64,
    pub y: u64,
    pub rstd: u64,
}

/// `p.dst` takes the gradient of the raw x_proj rows: the gradient of the
/// normalized rows assembled from its three sources, taken back through
/// the norms, stored once in `dtype`.
pub(crate) fn launch_pack_xdbl_norm_bwd(
    ctx: &GpuCtx,
    p: &NormBackward,
    dtype: WeightDtype,
    (rows, dt_rank, d_state): (usize, usize, usize),
) -> Result<(), String> {
    if rows == 0 {
        return Ok(());
    }
    let (dr, ds) = (int(dt_rank, "dt_rank")?, int(d_state, "d_state")?);
    let rows_l = i64::try_from(rows).map_err(|_| "bcdt norm: rows past i64")?;
    let mut bld = ctx
        .stream
        .launch_builder(ctx.kernels.pack_xdbl_norm_bwd_typed.get(dtype));
    bld.arg(&p.dst);
    bld.arg(&p.dt_src);
    bld.arg(&p.b_src);
    bld.arg(&p.c_src);
    bld.arg(&p.y);
    bld.arg(&p.rstd);
    bld.arg(&rows_l);
    bld.arg(&dr);
    bld.arg(&ds);
    unsafe { bld.launch(rows_cfg(rows)) }
        .map(drop)
        .map_err(|e| format!("pack_xdbl_norm_bwd: {e:?}"))
}
