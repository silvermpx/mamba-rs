//! The GPU side of the bidirectional mode's row reversal (see
//! [`crate::bidi`]).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::gpu::context::GpuCtx;

/// The rows of a window on the device: `[batch, seq_len, width]` elements
/// of `elem_bytes` (4, or 2 for the half types) at `ptr`.
#[derive(Clone, Copy)]
pub(crate) struct WindowRows {
    pub ptr: u64,
    pub elem_bytes: usize,
    pub batch: usize,
    pub seq_len: usize,
    pub width: usize,
}

/// Reverse every sequence's rows in place on the context's stream; a
/// packing (`seq_idx`, 0 for none) turns each sequence within its span.
pub(crate) fn reverse_rows(ctx: &GpuCtx, rows: WindowRows, seq_idx: u64) -> Result<(), String> {
    let WindowRows {
        ptr,
        elem_bytes,
        batch,
        seq_len,
        width,
    } = rows;
    let n_rows = batch * seq_len;
    if n_rows == 0 || width == 0 || seq_len < 2 {
        return Ok(());
    }
    let row_bytes = width * elem_bytes;
    let k = &ctx.kernels;
    // The widest unit that divides a row and keeps every row aligned: a
    // 16-byte move runs at more than twice the rate of a 4-byte one.
    let (kernel, unit) = if row_bytes % 16 == 0 && ptr % 16 == 0 {
        (&k.rows_reverse_u128, 16)
    } else if row_bytes % 4 == 0 && ptr % 4 == 0 {
        (&k.rows_reverse_u32, 4)
    } else {
        (&k.rows_reverse_u16, 2)
    };
    let units = row_bytes / unit;
    let block = units.next_multiple_of(32).min(256);
    let grid_rows = u32::try_from(n_rows).map_err(|_| "reverse rows: too many rows")?;
    let grid_units =
        u32::try_from(units.div_ceil(block)).map_err(|_| "reverse rows: rows too wide")?;
    let (t, units_i) = (
        i32::try_from(seq_len).map_err(|_| "reverse rows: seq_len past i32")?,
        i32::try_from(units).map_err(|_| "reverse rows: rows too wide")?,
    );
    let mut bld = ctx.stream.launch_builder(kernel);
    bld.arg(&ptr);
    bld.arg(&t);
    bld.arg(&units_i);
    bld.arg(&seq_idx);
    let cfg = LaunchConfig {
        grid_dim: (grid_rows, grid_units, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: `ptr` holds `n_rows` rows of `row_bytes` and `seq_idx`, when
    // set, `n_rows` ids; the kernel reads and writes within them.
    unsafe { bld.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("reverse rows: {e:?}"))
}

/// `bytes` zero bytes from `ptr` on, on the context's stream: a
/// bidirectional model's carried states at the start of every window. The
/// set is a memset node in a captured graph, so a replay empties them too.
pub(crate) fn zero_bytes(ctx: &GpuCtx, ptr: u64, bytes: usize) -> Result<(), String> {
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: the caller hands a device allocation of at least `bytes`
    // bytes from `ptr`, used on this stream only.
    let r = unsafe { cudarc::driver::sys::cuMemsetD8Async(ptr, 0, bytes, ctx.stream.cu_stream()) };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("zero the carried states: {r:?}"));
    }
    Ok(())
}
