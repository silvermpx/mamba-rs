//! The chunked SISO backward's dQKV stage for heads and states the warp
//! kernel cannot hold.
//!
//! Four launches: the per-chunk terms of the reverse state pass, the serial
//! fold of those terms into every chunk's entering state, the chain over the
//! saved chunk state that dADT needs, and the stage itself. Every operand
//! streams through fixed tiles, so one shared-memory budget serves any
//! headdim and any d_state, and every sum runs in the warp kernel's order:
//! where both kernels serve a shape their bits are the same.

use super::kernels::Mamba3Kernels;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{IndexWidth, grid_1d_cover, yz_runs};
use crate::mamba3_siso::route::chunked_backward_tiled_smem_bytes;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

/// Tile width of the stage. Any width up to 32 gives the same bits, and 16
/// keeps a block under the 48 KB default shared-memory window, so two
/// blocks of 256 threads share an SM on the consumer parts.
const TILE: usize = 16;

/// Threads per block of the tiled kernels.
const BLOCK: u32 = 256;

/// The shape of one layer's chunked backward.
#[derive(Clone, Copy)]
pub(super) struct DqkvShape {
    pub batch: usize,
    pub seq_len: usize,
    pub nheads: usize,
    pub headdim: usize,
    pub d_state: usize,
    pub chunk: usize,
}

/// The stage's device buffers. Q_rot, K_scaled, V and dO are in the
/// activation dtype; everything else is f32.
pub(super) struct DqkvTiledIo {
    pub d_q: CUdeviceptr,
    pub d_k: CUdeviceptr,
    pub d_v: CUdeviceptr,
    pub d_adt: CUdeviceptr,
    pub d_qk_dot: CUdeviceptr,
    pub d_partials: CUdeviceptr,
    pub q_rot: CUdeviceptr,
    pub k_scaled: CUdeviceptr,
    pub v: CUdeviceptr,
    pub da_cs: CUdeviceptr,
    pub da_cs_sum: CUdeviceptr,
    pub qk_dot: CUdeviceptr,
    pub chunk_states: CUdeviceptr,
    pub d_o: CUdeviceptr,
    pub d_param: CUdeviceptr,
    pub state_terms: CUdeviceptr,
    pub enter_state: CUdeviceptr,
    /// The gradient of the state the window entered with, `[B*nh*hd*ds]`.
    pub seed_grad: CUdeviceptr,
    pub dm_chain: CUdeviceptr,
    /// The sequences packed into the rows (see [`crate::varlen`]), i32
    /// `[B, T]`; 0 for one sequence a row.
    pub seq_idx: CUdeviceptr,
}

/// Runs the stage. The caller reduces `d_partials` into the D gradient,
/// as it does after the warp kernel.
pub(super) fn launch(
    stream: &Arc<CudaStream>,
    m3k: &Mamba3Kernels,
    dtype: WeightDtype,
    shape: DqkvShape,
    io: &DqkvTiledIo,
    width: IndexWidth,
) -> Result<(), String> {
    let DqkvShape {
        batch,
        seq_len,
        nheads,
        headdim,
        d_state,
        chunk,
    } = shape;
    let n_chunks = seq_len.div_ceil(chunk);
    let rows = batch * n_chunks * nheads;
    let dim = |v: usize| i32::try_from(v).map_err(|_| format!("dQKV: dimension {v} exceeds i32"));
    let (b, t, nh, hd, ds, cs) = (
        dim(batch)?,
        dim(seq_len)?,
        dim(nheads)?,
        dim(headdim)?,
        dim(d_state)?,
        dim(chunk)?,
    );
    let (nc, rows_i) = (dim(n_chunks)?, dim(rows)?);
    let grid = (nheads as u32, batch as u32, n_chunks as u32);

    let terms_cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    for (y_first, z_first, cfg) in yz_runs(terms_cfg) {
        let mut terms = stream.launch_builder(m3k.m3_dqkv_state_terms_tiled.get(dtype));
        for p in [&io.state_terms, &io.q_rot, &io.da_cs, &io.d_o] {
            terms.arg(p);
        }
        for d in [&b, &t, &nh, &hd, &ds, &cs] {
            terms.arg(d);
        }
        terms.arg(&y_first);
        terms.arg(&z_first);
        terms.arg(&io.seq_idx);
        unsafe { terms.launch(cfg) }.map_err(|e| format!("m3_dqkv_state_terms_tiled: {e:?}"))?;
    }

    let mut pass = stream.launch_builder(m3k.m3_dstate_passing_bwd.get(width));
    for p in [&io.enter_state, &io.state_terms, &io.da_cs_sum] {
        pass.arg(p);
    }
    for d in [&b, &nh, &hd, &ds, &nc] {
        pass.arg(d);
    }
    pass.arg(&io.seed_grad);
    pass.arg(&t);
    pass.arg(&cs);
    pass.arg(&io.seq_idx);
    unsafe { pass.launch(grid_1d_cover(batch * nheads * headdim * d_state)?) }
        .map_err(|e| format!("m3_dstate_passing_bwd tiled: {e:?}"))?;

    let mut chain = stream.launch_builder(&m3k.m3_dqkv_dm_chain);
    for p in [&io.dm_chain, &io.chunk_states, &io.enter_state] {
        chain.arg(p);
    }
    for d in [&rows_i, &hd, &ds] {
        chain.arg(d);
    }
    unsafe { chain.launch(grid_1d_cover(rows)?) }
        .map_err(|e| format!("m3_dqkv_dm_chain: {e:?}"))?;

    let tile = TILE as i32;
    let stage_cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: chunked_backward_tiled_smem_bytes(TILE) as u32,
    };
    for (y_first, z_first, cfg) in yz_runs(stage_cfg) {
        let mut stage = stream.launch_builder(m3k.m3_dqkv_tiled.get(dtype));
        for p in [
            &io.d_q,
            &io.d_k,
            &io.d_v,
            &io.d_adt,
            &io.d_qk_dot,
            &io.d_partials,
            &io.q_rot,
            &io.k_scaled,
            &io.v,
            &io.da_cs,
            &io.da_cs_sum,
            &io.qk_dot,
            &io.chunk_states,
            &io.d_o,
            &io.d_param,
            &io.enter_state,
            &io.dm_chain,
        ] {
            stage.arg(p);
        }
        for d in [&b, &t, &nh, &hd, &ds, &cs, &tile] {
            stage.arg(d);
        }
        stage.arg(&y_first);
        stage.arg(&z_first);
        stage.arg(&io.seq_idx);
        unsafe { stage.launch(cfg) }.map_err(|e| format!("m3_dqkv_tiled: {e:?}"))?;
    }
    Ok(())
}
