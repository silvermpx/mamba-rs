//! The B and C columns of the x projection rows, gathered for the scan
//! each path runs: t-major `[b][n][t]` for the parallel scan, which reads a
//! (d, n) lane over consecutive tokens, and row-major `[b][t][n]` for the
//! sequential one.

use cudarc::driver::PushKernelArg;

use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{grid_1d_wide, grid_gather_bc_tiled, y_runs};

/// One gather: the rows `[batch * seq, src_stride]` at `src` hold B from
/// column `b_offset` and C right after it, `d_state` columns each.
pub(crate) struct BcGather {
    pub dst_b: u64,
    pub dst_c: u64,
    pub src: u64,
    pub batch: usize,
    pub seq: usize,
    pub src_stride: usize,
    pub d_state: usize,
    pub b_offset: usize,
}

fn int(v: usize, what: &str) -> Result<i32, String> {
    i32::try_from(v).map_err(|_| format!("B/C gather: {what} {v} past i32"))
}

impl BcGather {
    /// Launch the gather in `dt`: t-major through the staged tile, a slice
    /// of states a block, when `tmajor`, else row-major. A copy either way,
    /// so every value keeps its bits.
    pub(crate) fn launch(&self, ctx: &GpuCtx, dt: WeightDtype, tmajor: bool) -> Result<(), String> {
        let ld = int(self.src_stride, "row width")?;
        let ds = int(self.d_state, "d_state")?;
        let b_off = int(self.b_offset, "B column")?;
        let c_off = int(self.b_offset + self.d_state, "C column")?;
        if tmajor {
            let t = int(self.seq, "window")?;
            let cfg = grid_gather_bc_tiled(self.batch, self.seq, self.d_state, dt.size_bytes());
            for (y_first, cfg) in y_runs(cfg) {
                let mut bld = ctx
                    .stream
                    .launch_builder(ctx.kernels.gather_bc_cols_tmajor_tiled_typed.get(dt));
                bld.arg(&self.dst_b);
                bld.arg(&self.dst_c);
                bld.arg(&self.src);
                bld.arg(&t);
                bld.arg(&ld);
                bld.arg(&ds);
                bld.arg(&b_off);
                bld.arg(&c_off);
                bld.arg(&y_first);
                unsafe { bld.launch(cfg) }.map_err(|e| format!("t-major B/C gather: {e:?}"))?;
            }
        } else {
            let rows = int(self.batch * self.seq, "rows")?;
            let mut bld = ctx
                .stream
                .launch_builder(ctx.kernels.gather_bc_cols_typed.get(dt));
            bld.arg(&self.dst_b);
            bld.arg(&self.dst_c);
            bld.arg(&self.src);
            bld.arg(&rows);
            bld.arg(&ld);
            bld.arg(&ds);
            bld.arg(&b_off);
            bld.arg(&c_off);
            unsafe { bld.launch(grid_1d_wide(self.batch * self.seq * self.d_state)) }
                .map_err(|e| format!("B/C gather: {e:?}"))?;
        }
        Ok(())
    }
}
