//! The Mamba-2 CUDA module: `kernels/gated_mlp.cu`, `mamba2_ops.cu`,
//! `mamba2_ssd.cu` and `mamba2_bwd.cu` after the typed prelude, compiled
//! with NVRTC and cached under its own module kind, so adding it moves no
//! identity of the Inference or Mamba-3 modules. The norms, the residual
//! adds and the optimizer come from the Common module through the context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::gpu::kernel_identity as ki;
use crate::gpu::kernels::{CudaModuleAnchors, HalfKernel, TypedKernel};
use crate::gpu::launch::opt_in_dynamic_smem;

/// The compiled Mamba-2 kernels.
pub struct Mamba2Kernels {
    /// The decode step of every head, by activation dtype.
    pub step_fwd: TypedKernel,
    /// The sequential prefill: the decode step's recurrence over a window.
    pub seq_fwd: TypedKernel,
    /// The training forward's sequential scan, writing replay checkpoints.
    pub seq_fwd_ckpt: TypedKernel,
    /// The grouped gated RMSNorm.
    pub gated_norm_fwd: TypedKernel,
    /// The gated norm of a group too wide to stage, with the staged
    /// kernel's bits.
    pub gated_norm_fwd_stream: TypedKernel,
    /// The gate without a norm.
    pub gate_only_fwd: TypedKernel,
    /// The mixer's inner MLP gate.
    pub mlp_gate_fwd: TypedKernel,
    /// The gated MLP after the mixer: value times silu(gate) of each row
    /// of its up projection.
    pub gated_mlp: TypedKernel,
    /// The prefill: conv over the window and its state, dt with the chunk
    /// cumsum, the CB products, the chunk states, the state passing and
    /// the chunk scan.
    pub conv_prefill: TypedKernel,
    /// The decode's conv, the prefill conv's chain at one token.
    pub conv_step: TypedKernel,
    pub conv_state: TypedKernel,
    pub dt_cumsum: TypedKernel,
    pub bmm_chunk: TypedKernel,
    pub chunk_state: TypedKernel,
    pub state_passing: CudaFunction,
    pub chunk_scan: TypedKernel,
    /// The chunk scan as register tiles, for shapes whose staging fits
    /// (see [`chunk_scan_tiles_smem_bytes`]).
    pub chunk_scan_tiles: TypedKernel,
    /// The training backward, f32.
    pub bwd: Mamba2BwdKernels,
    /// The dynamic shared memory each opted-in kernel (the scans, the gated
    /// norms, the sequential backward) may stage: the device's per-block
    /// opt-in less the largest static stage among them.
    pub smem_optin: usize,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
    /// The loaded module, held by every graph that launches these kernels
    /// so it outlives them.
    _modules: CudaModuleAnchors,
}

/// The backward of the window and the training helpers
/// (`kernels/mamba2_bwd.cu`); the kernels that read the stored projection
/// or conv rows come in every activation dtype.
pub struct Mamba2BwdKernels {
    pub a_neg: CudaFunction,
    pub dhloc: TypedKernel,
    pub state_passing: CudaFunction,
    pub chunk_dot: CudaFunction,
    pub dx: TypedKernel,
    pub rowsums: TypedKernel,
    /// `m2b_dx` and `m2b_rowsums` as one register-tile block per chunk and
    /// head, when [`dx_tiles_geom`] fits the head.
    pub dx_tiles: TypedKernel,
    /// The widest block `dx_tiles` launches, from its register use.
    pub dx_tiles_threads: usize,
    pub pairs: TypedKernel,
    pub db: TypedKernel,
    pub dc: TypedKernel,
    /// `m2b_db` and `m2b_dc` as register tiles, when
    /// [`bc_tiles_smem_bytes`] fits.
    pub db_tiles: TypedKernel,
    pub dc_tiles: TypedKernel,
    pub combine: TypedKernel,
    pub head_grads: CudaFunction,
    pub d_parts: TypedKernel,
    pub d_reduce: CudaFunction,
    pub gated_norm: TypedKernel,
    /// The backward of a group too wide to stage, with the staged
    /// kernel's bits.
    pub gated_norm_stream: TypedKernel,
    pub gate_only: TypedKernel,
    pub mlp_gate: TypedKernel,
    /// The backward of the gated MLP's gate, the shared kernel's bits.
    pub gated_mlp: TypedKernel,
    pub conv_dv: TypedKernel,
    pub conv_dx: TypedKernel,
    pub conv_dw: TypedKernel,
    /// The sequential path's backward of the recurrence (replay per
    /// segment) and the group sums of its dB and dC.
    pub seq_bwd: TypedKernel,
    /// `m2s_bwd` with the token's states in shared memory and its step
    /// scalars from `seq_scalars`, when [`seq_bwd_tiles_smem_bytes`] fits.
    pub seq_bwd_tiles: TypedKernel,
    /// `m2s_bwd` with the states read from the replay tape and dS in shared
    /// memory or a global scratch, for every head the other two cannot hold
    /// (see [`Mamba2Kernels::seq_bwd_route`]).
    pub seq_bwd_global: TypedKernel,
    /// The widest block each sequential backward kernel launches.
    pub seq_threads: SeqBwdThreads,
    pub seq_scalars: TypedKernel,
    pub seq_bc_groups: CudaFunction,
    /// The column sums of per-row weight-gradient partials, the shared
    /// reduction's bits read a row's neighbouring columns at a time.
    pub reduce_rows: CudaFunction,
    /// A bias gradient from bf16 or f16 rows.
    pub bias_grad: HalfKernel,
}

/// The scan's shared memory for a block of `rows` channels: B and C of two
/// tokens and the rows' state tile, each row padded by one.
pub fn step_smem_bytes(rows: usize, ds: usize) -> usize {
    (4 * ds + rows * (ds + 1)) * 4
}

/// The channels per block of the decode and sequential scans: a warp when
/// its state tile fits `cap` bytes, fewer otherwise. Each thread keeps one
/// channel's state and sums its readout alone, so the block size never
/// changes a value. `None` when even one channel does not fit.
pub fn step_rows(hd: usize, ds: usize, cap: usize) -> Option<usize> {
    [32usize, 16, 8, 4, 2, 1]
        .into_iter()
        .map(|r| r.min(hd.max(1)))
        .find(|&r| step_smem_bytes(r, ds) <= cap)
}

/// The channels per block of a decode step of `batch` sequences over `nh`
/// heads: the widest block (32, 16, 8, then 4 rows) that still leaves at
/// least `DECODE_BLOCKS` blocks, so a lone token spreads the heads' state
/// tiles over many blocks (a block moves its rows with 128 threads) while a
/// batch keeps the wide blocks that share each token's B and C; fewer rows
/// when the block does not fit `cap`. The rows never change a value.
pub fn decode_rows(
    (batch, nh): (usize, usize),
    (hd, ds): (usize, usize),
    cap: usize,
) -> Option<usize> {
    const DECODE_BLOCKS: usize = 384;
    let fits = |r: usize| step_smem_bytes(r, ds) <= cap;
    [32usize, 16, 8, 4]
        .into_iter()
        .map(|r| r.min(hd.max(1)))
        .find(|&r| fits(r) && batch * nh * hd.div_ceil(r) >= DECODE_BLOCKS)
        .or_else(|| {
            [4usize, 2, 1]
                .into_iter()
                .map(|r| r.min(hd.max(1)))
                .find(|&r| fits(r))
        })
}

/// Launch geometry of the decode and sequential scans with `rows` channels
/// per block (see [`step_rows`]), and helper thread rows up to 128 threads
/// that move the state tile, so a decode step is not bound by one warp's
/// loads.
pub fn step_cfg(batch: usize, nh: usize, (hd, ds): (usize, usize), rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, nh as u32, hd.div_ceil(rows) as u32),
        block_dim: (rows as u32, (128 / rows).max(1) as u32, 1),
        shared_mem_bytes: step_smem_bytes(rows, ds) as u32,
    }
}

/// Tokens per SSD chunk.
pub const CHUNK: usize = 64;

/// The gated norm's shared memory: the group, each head's row padded by
/// one, and its per-head sums.
pub fn gated_norm_smem_bytes(d_ssm: usize, ng: usize, hd: usize) -> usize {
    let gs = d_ssm / ng;
    (gs + 2 * (gs / hd)) * 4
}

/// The gated norm backward's shared memory: the group twice, each head's
/// row padded by one, and its per-head sums.
pub fn gated_norm_bwd_smem_bytes(d_ssm: usize, ng: usize, hd: usize) -> usize {
    let gs = d_ssm / ng;
    (2 * gs + 3 * (gs / hd)) * 4
}

/// Launch geometry of the gated norm: one block per (row, group), its
/// group and per-head sums staged in shared memory when they fit `cap`
/// bytes, else a pass of per-head sums of one float a thread. Returns
/// whether the staged kernel runs.
pub fn gated_norm_cfg(
    rows: usize,
    (d_ssm, ng, hd): (usize, usize, usize),
    cap: usize,
) -> (bool, LaunchConfig) {
    const THREADS: usize = 256;
    let staged = gated_norm_smem_bytes(d_ssm, ng, hd);
    let fits = staged <= cap;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, ng as u32, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: if fits { staged } else { THREADS * 4 } as u32,
    };
    (fits, cfg)
}

impl Mamba2Kernels {
    /// Compile (or load from the cache) the Mamba-2 module for `arch`.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let ComposedModule {
            module,
            compiler_identity,
            artifact_identity,
        } = compile_composed(
            ctx,
            arch,
            &ComposedSpec {
                sources: &[
                    include_str!("../../../kernels/_typed_prelude.cuh"),
                    include_str!("../../../kernels/gated_mlp.cu"),
                    include_str!("../../../kernels/mamba2_ops.cu"),
                    include_str!("../../../kernels/mamba2_ssd.cu"),
                    include_str!("../../../kernels/mamba2_bwd.cu"),
                ],
                kind: ki::ModuleKind::Mamba2Combined,
                cache_prefix: "mamba2-kernels-v1-",
                seed: "2405210601",
                options: &[],
                label: "Mamba-2",
            },
        )?;
        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("Mamba-2 kernel '{name}' not found: {e:?}"))
        };
        let typed = |base: &str| -> Result<TypedKernel, String> {
            Ok(TypedKernel {
                f32: get(base)?,
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        let smem_optin = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| format!("query opt-in shared memory: {e:?}"))?
            .max(0) as usize;
        let dx_tiles = typed("m2b_dx_tiles")?;
        let dx_tiles_threads = max_block(&dx_tiles, "dx tiles")?;
        let seq_bwd = typed("m2s_bwd")?;
        let seq_bwd_tiles = typed("m2s_bwd_tiles")?;
        let seq_bwd_global = typed("m2s_bwd_global")?;
        let seq_threads = SeqBwdThreads {
            tiles: max_block(&seq_bwd_tiles, "sequential backward tiles")?,
            shared: max_block(&seq_bwd, "sequential backward")?,
            global: max_block(&seq_bwd_global, "global sequential backward")?,
        };
        let mut kernels = Self {
            step_fwd: typed("m2_step_fwd")?,
            seq_fwd: typed("m2_seq_fwd")?,
            seq_fwd_ckpt: typed("m2_seq_fwd_ckpt")?,
            gated_norm_fwd: typed("m2_gated_norm_fwd")?,
            gated_norm_fwd_stream: typed("m2_gated_norm_fwd_stream")?,
            gate_only_fwd: typed("m2_gate_only_fwd")?,
            mlp_gate_fwd: typed("m2_mlp_gate_fwd")?,
            gated_mlp: typed("mlp_gate_fwd")?,
            conv_prefill: typed("m2_conv_prefill")?,
            conv_step: typed("m2_conv_step")?,
            conv_state: typed("m2_conv_state")?,
            dt_cumsum: typed("m2_dt_cumsum")?,
            bmm_chunk: typed("m2_bmm_chunk")?,
            chunk_state: typed("m2_chunk_state")?,
            state_passing: get("m2_state_passing")?,
            chunk_scan: typed("m2_chunk_scan")?,
            chunk_scan_tiles: typed("m2_chunk_scan_tiles")?,
            bwd: Mamba2BwdKernels {
                a_neg: get("m2_a_neg")?,
                dhloc: typed("m2b_dhloc")?,
                state_passing: get("m2b_state_passing")?,
                chunk_dot: get("m2b_chunk_dot")?,
                dx: typed("m2b_dx")?,
                rowsums: typed("m2b_rowsums")?,
                dx_tiles,
                dx_tiles_threads,
                pairs: typed("m2b_pairs")?,
                db: typed("m2b_db")?,
                dc: typed("m2b_dc")?,
                db_tiles: typed("m2b_db_tiles")?,
                dc_tiles: typed("m2b_dc_tiles")?,
                combine: typed("m2b_combine")?,
                head_grads: get("m2b_head_grads")?,
                d_parts: typed("m2b_d_parts")?,
                d_reduce: get("m2b_d_reduce")?,
                gated_norm: typed("m2b_gated_norm")?,
                gated_norm_stream: typed("m2b_gated_norm_stream")?,
                gate_only: typed("m2b_gate_only")?,
                mlp_gate: typed("m2b_mlp_gate")?,
                gated_mlp: typed("mlp_gate_bwd")?,
                conv_dv: typed("m2b_conv_dv")?,
                conv_dx: typed("m2b_conv_dx")?,
                conv_dw: typed("m2b_conv_dw")?,
                seq_bwd,
                seq_bwd_tiles,
                seq_bwd_global,
                seq_threads,
                seq_scalars: typed("m2s_scalars")?,
                seq_bc_groups: get("m2s_bc_groups")?,
                reduce_rows: get("m2b_reduce_rows")?,
                bias_grad: HalfKernel {
                    bf16: get("m2b_bias_grad_bf16")?,
                    f16: get("m2b_bias_grad_f16")?,
                },
            },
            smem_optin,
            compiler_identity,
            artifact_identity,
            _modules: CudaModuleAnchors::new(vec![module.clone()]),
        };
        kernels.smem_optin = {
            let optin = kernels.smem_optin;
            let mut budget = optin;
            for (k, name) in [
                (&kernels.step_fwd, "step"),
                (&kernels.seq_fwd, "sequential scan"),
                (&kernels.seq_fwd_ckpt, "checkpointing scan"),
                (&kernels.gated_norm_fwd, "gated norm"),
                (&kernels.bwd.gated_norm, "gated norm backward"),
                (&kernels.bwd.seq_bwd, "sequential backward"),
                (&kernels.bwd.seq_bwd_tiles, "sequential backward tiles"),
                (&kernels.bwd.seq_bwd_global, "global sequential backward"),
                (&kernels.chunk_scan_tiles, "chunk scan tiles"),
                (&kernels.bmm_chunk, "chunk bmm"),
                (&kernels.chunk_state, "chunk state"),
                (&kernels.bwd.dx_tiles, "dx tiles"),
                (&kernels.bwd.db_tiles, "dB tiles"),
                (&kernels.bwd.dc_tiles, "dC tiles"),
            ] {
                for f in [&k.f32, &k.bf16, &k.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            budget
        };
        Ok(kernels)
    }

    /// The loaded module, for a graph to hold.
    pub(crate) fn module_anchors(&self) -> CudaModuleAnchors {
        self._modules.clone()
    }

    /// The sequential backward's kernel for a head of `hd` channels and
    /// `ds` states: the tiled kernel when its three tiles and its block fit,
    /// else `m2s_bwd` when its two tiles and a thread per channel fit, else
    /// the global kernel, which serves every head.
    pub fn seq_bwd_route(&self, hd: usize, ds: usize) -> SeqBwdRoute {
        let (t, cap) = (self.bwd.seq_threads, self.smem_optin);
        let lanes = hd.next_multiple_of(32);
        if seq_bwd_tiles_smem_bytes(hd, ds) <= cap && seq_bwd_tiles_threads(hd) <= t.tiles {
            SeqBwdRoute::Tiles
        } else if lanes <= t.shared && seq_bwd_smem_bytes(hd, ds, lanes) <= cap {
            SeqBwdRoute::Shared
        } else {
            SeqBwdRoute::Global(seq_bwd_global_geom(hd, ds, t.global, cap))
        }
    }
}

/// The widest block every dtype of `k` launches, from its register use, a
/// whole number of warps.
fn max_block(k: &TypedKernel, what: &str) -> Result<usize, String> {
    use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
    let mut threads = 1024usize;
    for f in [&k.f32, &k.bf16, &k.f16] {
        let max = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|e| format!("threads per block of the {what}: {e:?}"))?;
        threads = threads.min(max.max(32) as usize);
    }
    Ok(threads / 32 * 32)
}

/// Tokens per block of the tiled chunk scan.
pub const CHUNK_SCAN_TT: usize = 32;

/// The tiled chunk scan's shared memory: the entering state rows padded by
/// one, the tile's C rows and weights, the x rows of the chunk.
pub fn chunk_scan_tiles_smem_bytes(hd: usize, ds: usize) -> usize {
    (hd * (ds + 1) + CHUNK_SCAN_TT * ds + CHUNK_SCAN_TT * CHUNK + CHUNK * hd) * 4
}

/// The chunk bmm's shared memory: the chunk's C and B rows transposed,
/// padded to 68, up to 128 state indices a pass.
pub fn bmm_smem_bytes(ds: usize) -> usize {
    2 * ds.min(128) * (CHUNK + 4) * 4
}

/// The chunk state kernel's state indices per block (32 per lane column,
/// one to four columns), the channels of its weighted-input tile and its
/// shared memory: the tile [64][channels], the B rows of the block's
/// indices and the per-token weights. The tile is the whole head rounded up
/// to 4 when that fits `cap` bytes, else the widest multiple of 4 that does.
pub fn chunk_state_geom(hd: usize, ds: usize, cap: usize) -> (usize, usize, usize) {
    let cols = 32 * ds.div_ceil(32).min(4).next_power_of_two().min(4);
    let bytes = |pt: usize| CHUNK * (pt + cols + 1) * 4;
    let fit = (cap / (CHUNK * 4)).saturating_sub(cols + 1) / 4 * 4;
    let pt = hd.next_multiple_of(4).min(fit.max(4));
    (cols, pt, bytes(pt))
}

/// The fused dx and token sums' state indices per staging pass and shared
/// memory: the staging (the B or C rows and the dS or H rows of `nk`
/// indices, later the three `[64][P + 1]` token rows), the weights
/// `[64][64]`, the dy rows `[64][P rounded up to 4]` and the acs and dt of
/// the chunk. The
/// largest pass that fits `cap` (all of N, then halves); `None` when the
/// block of 16 threads per 4 channels holds more than `threads`, the widest
/// block the kernel's registers allow, or when even one index does not fit.
pub fn dx_tiles_geom(hd: usize, ds: usize, cap: usize, threads: usize) -> Option<(usize, usize)> {
    if hd == 0 || 16 * hd.div_ceil(4) > threads.min(1024) || ds == 0 {
        return None;
    }
    let p4 = hd.next_multiple_of(4);
    let bytes = |nk: usize| {
        ((nk * (CHUNK + 4 + p4 + 4)).max(3 * CHUNK * (hd + 1))
            + CHUNK * CHUNK
            + CHUNK * p4
            + 2 * CHUNK)
            * 4
    };
    std::iter::successors(Some(ds), |&nk| (nk > 1).then(|| nk.div_ceil(2)))
        .find(|&nk| bytes(nk) <= cap)
        .map(|nk| (nk, bytes(nk)))
}

/// The tiled dB and dC's shared memory for a head of `hd` channels: the
/// head's x or dy rows transposed (padded to 68), its dS or H columns of a
/// 32-index tile, the group's dCB `[64][64]`, the C or B rows of the tile and
/// the rows' scales.
pub fn bc_tiles_smem_bytes(hd: usize) -> usize {
    ((CHUNK + 4) * hd + 32 * hd + CHUNK * CHUNK + CHUNK * 32 + CHUNK) * 4
}

/// The tiled sequential backward's shared memory for a head: the states
/// after and before the token and dS (each row padded by one), B and C, x
/// and dy of the token, two reduction slots per channel thread (the
/// channels rounded up to a warp).
pub fn seq_bwd_tiles_smem_bytes(hd: usize, ds: usize) -> usize {
    (3 * hd * (ds + 1) + 2 * ds + 2 * hd + 2 * hd.next_multiple_of(32)) * 4
}

/// The tiled sequential backward's block: a thread per channel and at
/// least 256 threads, since its dS update, dB and dC chains and tile moves
/// spread over the whole block while only the channel threads walk the
/// rows.
pub fn seq_bwd_tiles_threads(hd: usize) -> usize {
    hd.next_multiple_of(32).max(256)
}

/// The sequential backward's shared memory for a head: the replay state
/// rows and the state-gradient rows (each padded by one), B and C, the step
/// scalars, x and dy of the token, one reduction slot per thread.
pub fn seq_bwd_smem_bytes(hd: usize, ds: usize, threads: usize) -> usize {
    (2 * hd * (ds + 1) + 2 * ds + 4 + 2 * hd + threads) * 4
}

/// The widest block each sequential backward kernel launches, from the
/// compiled kernels' register use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqBwdThreads {
    pub tiles: usize,
    pub shared: usize,
    pub global: usize,
}

/// The sequential backward's kernel for a head
/// ([`Mamba2Kernels::seq_bwd_route`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqBwdRoute {
    /// `m2s_bwd_tiles`: the token's states and dS in shared memory.
    Tiles,
    /// `m2s_bwd`: the replay state and dS in shared memory, a thread per
    /// channel.
    Shared,
    /// `m2s_bwd_global` with its launch.
    Global(SeqGlobalGeom),
}

/// The launch of `m2s_bwd_global` for a head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqGlobalGeom {
    /// The block: a thread per channel when the head fits it, else a power
    /// of two whose threads take the channels in turns.
    pub threads: usize,
    /// dS in shared memory, else in a global scratch of P N floats per
    /// sequence and head.
    pub shared_ds: bool,
    /// The head is wider than the block: each block sum keeps a leaf per
    /// lane in a global scratch of 2 lanes floats per sequence and head.
    pub wide: bool,
    /// Dynamic shared memory in bytes.
    pub smem: usize,
}

/// The launch of `m2s_bwd_global` for a head of `hd` channels and `ds`
/// states, its blocks at most `max_threads` wide with `cap` bytes of shared
/// memory. A head that fits the block gets a thread per channel, at least
/// one per dC and dB chain and at least 256 for the passes over the state.
pub fn seq_bwd_global_geom(hd: usize, ds: usize, max_threads: usize, cap: usize) -> SeqGlobalGeom {
    let lanes = hd.next_multiple_of(32);
    let max_threads = (max_threads / 32 * 32).max(32);
    let wide = lanes > max_threads;
    let threads = if wide {
        1usize << max_threads.ilog2()
    } else {
        lanes
            .max((2 * ds).next_multiple_of(32))
            .max(256)
            .min(max_threads)
    };
    let leaves = if wide { threads } else { lanes };
    let rest = (2 * ds + 2 * leaves) * 4;
    let tile = hd * (ds + 1) * 4;
    let shared_ds = tile + rest <= cap;
    SeqGlobalGeom {
        threads,
        shared_ds,
        wide,
        smem: if shared_ds { tile + rest } else { rest },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No validated state is refused for shared memory on a supported part:
    /// one channel of the largest state fits the 99 KB every part since
    /// Ampere lets a block opt in to.
    #[test]
    fn one_channel_of_every_validated_state_fits_a_supported_part() {
        for ds in 1..=1024 {
            for hd in [1, 7, 32, 33, 4096] {
                assert!(step_rows(hd, ds, 99 * 1024).is_some(), "hd {hd} ds {ds}");
            }
        }
    }
}
