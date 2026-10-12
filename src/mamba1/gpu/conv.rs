//! The depthwise conv launches of the Mamba-1 forward, prefill and
//! backward. Up to [`CONV_WINDOW_TAPS`] taps the tiled kernels keep a
//! channel's window, its carries and its tap gradients in registers; a
//! wider conv takes the kernels that read the window from the rows
//! (kernels/conv1d.cu), which run the same chains in the same order.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{blocks_1d, grid_1d_wide, grid_conv_fwd_tiled, grid_conv_tiled, tiles_y};

/// The widest window the tiled conv kernels hold per thread.
pub const CONV_WINDOW_TAPS: usize = 8;

/// The tiled forward's tile depth (CONV1D_FWD_TILE_T). The kernel of any
/// width takes tiles at least as deep as its window, so no tile but the
/// first reads the carry-in it overwrites.
const FWD_TILE: usize = 32;

/// One conv forward with the fused SiLU, by device pointer.
pub(crate) struct ConvForward {
    /// The `[B * T, d_inner]` outputs.
    pub u: u64,
    /// The outputs again in t-major rows of the given stride, for the
    /// parallel scan; a null pointer writes none.
    pub u_t: (u64, usize),
    /// The layer's carry state `[B, d_inner, d_conv]`, read and written.
    pub state: u64,
    /// The carry-in window saved for the backward; null in a prefill.
    pub saved: u64,
    /// The x rows and their row stride.
    pub x: (u64, usize),
    pub weight: u64,
    pub bias: u64,
}

/// Launch the conv forward over `(batch, seq, d_inner, d_conv)`.
pub(crate) fn conv_forward(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    p: &ConvForward,
    (batch, seq, d_inner, d_conv): (usize, usize, usize, usize),
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    let k = &ctx.kernels;
    let (b_i, t_i, di_i, dc_i) = (batch as i32, seq as i32, d_inner as i32, d_conv as i32);
    let (xs_i, uts_i) = (p.x.1 as i32, p.u_t.1 as i32);
    if d_conv > CONV_WINDOW_TAPS {
        let tile = d_conv.max(FWD_TILE);
        let tile_i = tile as i32;
        let mut bld = ctx.stream.launch_builder(k.conv1d_fwd_any_typed.get(dtype));
        bld.arg(&p.u);
        bld.arg(&p.u_t.0);
        bld.arg(&uts_i);
        bld.arg(&p.state);
        bld.arg(&p.saved);
        bld.arg(&p.x.0);
        bld.arg(&p.weight);
        bld.arg(&p.bias);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&dc_i);
        bld.arg(&xs_i);
        bld.arg(&tile_i);
        let grid = blocks_1d(batch * d_inner)
            .and_then(|x| Ok((x, tiles_y(seq, tile)?)))
            .map_err(|e| format!("{what} conv forward: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (grid.0, grid.1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { bld.launch(cfg) }.map_err(|e| format!("{what} conv forward: {e:?}"))?;
        return Ok(());
    }
    let cfg = grid_conv_fwd_tiled(batch, d_inner, seq)
        .map_err(|e| format!("{what} conv forward: {e}"))?;
    if p.saved != 0 {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_burnin_fwd_tiled_typed.get(dtype));
        bld.arg(&p.u);
        bld.arg(&p.state);
        bld.arg(&p.saved);
        bld.arg(&p.x.0);
        bld.arg(&p.weight);
        bld.arg(&p.bias);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&dc_i);
        bld.arg(&xs_i);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("{what} conv forward: {e:?}"))?;
    } else {
        let mut bld = ctx
            .stream
            .launch_builder(k.conv1d_burnin_nosave_tiled_typed.get(dtype));
        bld.arg(&p.u);
        bld.arg(&p.u_t.0);
        bld.arg(&p.state);
        bld.arg(&p.x.0);
        bld.arg(&p.weight);
        bld.arg(&p.bias);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&dc_i);
        bld.arg(&xs_i);
        bld.arg(&uts_i);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("{what} conv forward: {e:?}"))?;
    }
    Ok(())
}

/// One conv backward, by device pointer: the x gradient into a strided
/// destination, and the tap and bias partials of every (sequence, tile)
/// row of CONV1D_TILE_T steps that the caller reduces.
pub(crate) struct ConvBackward {
    /// The x gradient's destination, its row stride and column offset.
    pub d_x: (u64, usize, usize),
    pub d_weight_partials: u64,
    pub d_bias_partials: u64,
    pub d_u: u64,
    /// The x rows and their row stride.
    pub x: (u64, usize),
    /// The carry-in window the forward saved.
    pub conv_init: u64,
    pub weight: u64,
    pub bias: u64,
    /// An f32 scratch of `[B * T, d_inner]` for a conv wider than
    /// [`CONV_WINDOW_TAPS`]; unused otherwise.
    pub dco: u64,
}

/// Launch the conv backward over `(batch, seq, d_inner, d_conv)`.
pub(crate) fn conv_backward(
    ctx: &GpuCtx,
    dtype: WeightDtype,
    p: &ConvBackward,
    (batch, seq, d_inner, d_conv): (usize, usize, usize, usize),
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    let k = &ctx.kernels;
    let (b_i, t_i, di_i, dc_i) = (batch as i32, seq as i32, d_inner as i32, d_conv as i32);
    let (os_i, oo_i, xs_i) = (p.d_x.1 as i32, p.d_x.2 as i32, p.x.1 as i32);
    let cfg =
        grid_conv_tiled(batch, d_inner, seq).map_err(|e| format!("{what} conv backward: {e}"))?;
    if d_conv > CONV_WINDOW_TAPS {
        let mut bld = ctx.stream.launch_builder(k.conv1d_bwd_dco_typed.get(dtype));
        bld.arg(&p.dco);
        bld.arg(&p.d_u);
        bld.arg(&p.x.0);
        bld.arg(&p.conv_init);
        bld.arg(&p.weight);
        bld.arg(&p.bias);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&dc_i);
        bld.arg(&xs_i);
        unsafe { bld.launch(grid_1d_wide(batch * seq * d_inner)) }
            .map_err(|e| format!("{what} conv pre-activation gradient: {e:?}"))?;
        let mut bld = ctx.stream.launch_builder(k.conv1d_bwd_any_typed.get(dtype));
        bld.arg(&p.d_x.0);
        bld.arg(&p.d_weight_partials);
        bld.arg(&p.d_bias_partials);
        bld.arg(&p.dco);
        bld.arg(&p.x.0);
        bld.arg(&p.conv_init);
        bld.arg(&p.weight);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&di_i);
        bld.arg(&dc_i);
        bld.arg(&os_i);
        bld.arg(&oo_i);
        bld.arg(&xs_i);
        unsafe { bld.launch(cfg) }.map_err(|e| format!("{what} conv backward: {e:?}"))?;
        return Ok(());
    }
    let mut bld = ctx
        .stream
        .launch_builder(k.conv1d_bwd_tiled_typed.get(dtype));
    bld.arg(&p.d_x.0);
    bld.arg(&p.d_weight_partials);
    bld.arg(&p.d_bias_partials);
    bld.arg(&p.d_u);
    bld.arg(&p.x.0);
    bld.arg(&p.conv_init);
    bld.arg(&p.weight);
    bld.arg(&p.bias);
    bld.arg(&b_i);
    bld.arg(&t_i);
    bld.arg(&di_i);
    bld.arg(&dc_i);
    bld.arg(&os_i);
    bld.arg(&oo_i);
    bld.arg(&xs_i);
    unsafe { bld.launch(cfg) }.map_err(|e| format!("{what} conv backward: {e:?}"))?;
    Ok(())
}

/// Elements of the f32 scratch the backward of a `d_conv`-tap conv needs
/// over `rows * d_inner` positions: all of them past the window the tiled
/// kernels hold, one otherwise.
pub(crate) fn conv_dco_len(rows: usize, d_inner: usize, d_conv: usize) -> usize {
    if d_conv > CONV_WINDOW_TAPS {
        rows * d_inner
    } else {
        1
    }
}
