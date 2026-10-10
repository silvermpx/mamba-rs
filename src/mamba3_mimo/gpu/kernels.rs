//! The Mamba-3 MIMO CUDA module: `kernels/gated_mlp.cu`,
//! `kernels/mamba3_mimo.cu`, `kernels/mamba3_mimo_chunked.cu`,
//! `kernels/mamba3_mimo_bwd.cu` and `kernels/mamba3_mimo_chunked_bwd.cu`
//! after the typed prelude, under its own module kind. The norms, the
//! residual adds and the optimizer come from the Fixed module through the
//! context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::mamba_ssm::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::mamba_ssm::gpu::kernel_identity as ki;
use crate::mamba_ssm::gpu::kernels::TypedKernel;
use crate::mamba_ssm::gpu::launch::opt_in_dynamic_smem;

/// The compiled MIMO kernels.
pub struct Mamba3MimoKernels {
    /// B and C normalised per (rank, group) row.
    pub bc_norm: TypedKernel,
    /// The scan of every head over a window (decode is the window of one).
    pub scan: TypedKernel,
    /// The chunked path: step scalars and angles, each chunk's state, the
    /// state passing, the outputs, the carried keys and input.
    pub chunk_prep: TypedKernel,
    pub chunk_state: TypedKernel,
    pub state_pass: CudaFunction,
    pub chunk_scan: TypedKernel,
    pub chunk_writeback: TypedKernel,
    /// The MLP's gate, `value * silu(gate)`.
    pub mlp_gate: TypedKernel,
    /// The training forward's scan, writing the replay checkpoints.
    pub scan_ckpt: TypedKernel,
    /// The sequential backward of the scan (replay per segment).
    pub scan_bwd: TypedKernel,
    pub bc_norm_bwd: TypedKernel,
    pub angle_bwd: TypedKernel,
    pub mlp_gate_bwd: TypedKernel,
    /// The per-sequence weight partials summed over the sequences.
    pub sum_rows: CudaFunction,
    /// The chunked path's backward.
    pub cb: MimoChunkedBwd,
    /// The dynamic shared memory each kernel staging more than the default
    /// 48 KB may use: the device's per-block opt-in less the largest
    /// static stage among those kernels.
    pub smem_optin: usize,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
}

/// The chunked path's backward kernels (`kernels/mamba3_mimo_chunked_bwd.cu`).
pub struct MimoChunkedBwd {
    pub gate: TypedKernel,
    pub inter: CudaFunction,
    pub dsloc: CudaFunction,
    pub pass: CudaFunction,
    pub chunk_dot: CudaFunction,
    pub state_terms: TypedKernel,
    pub pairs: TypedKernel,
    pub rot_bwd: CudaFunction,
    pub angles: TypedKernel,
    pub combine: CudaFunction,
    pub lead: CudaFunction,
    pub scalars: TypedKernel,
    pub u_bwd: TypedKernel,
    pub sum_inner: CudaFunction,
    /// Per-row partials summed over the rows, 64-bit indexed.
    pub reduce_rows: CudaFunction,
}

/// The chunked path's launch geometry for one head shape, chosen to fit
/// the device's shared memory. A tile here sets how many tokens or state
/// indices a block stages at a time, never the order in which a thread
/// sums, so every choice gives the same bits; the largest tiles that fit are
/// taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkedGeom {
    /// `m3m_chunk_state`: state indices per block, tokens staged per pass,
    /// bytes.
    pub state: (usize, usize, usize),
    /// `m3m_chunk_scan`: query tokens per block, key tokens per pass, bytes.
    pub scan: (usize, usize, usize),
    /// `m3mb_dsloc`: state indices per block, bytes.
    pub dsloc: (usize, usize),
    /// `m3mb_pairs`: tokens per tile, bytes.
    pub pairs: (usize, usize),
}

/// The chunked geometry of a head of `hd` channels, `ds` state indices and
/// rank `rank` within `cap` bytes of shared memory, or `None` when a kernel
/// of the path cannot fit (or a block would pass 1024 threads); the
/// backward's two kernels count only when `training`.
pub fn chunked_geom(
    (hd, ds, rank): (usize, usize, usize),
    cap: usize,
    training: bool,
) -> Option<ChunkedGeom> {
    if hd == 0 || ds == 0 || rank == 0 || hd > 1024 {
        return None;
    }
    let wide = (256 / hd).max(1);
    // The state kernel: kt [sl][R][nt], xt [sl][P], wt [sl].
    let state = (1..=wide.min(ds)).rev().find_map(|nt| {
        [CHUNK, 32, 16, 8, 4, 2, 1].into_iter().find_map(|sl| {
            let bytes = (sl * rank * nt + sl * hd + sl) * 4;
            (bytes <= cap).then_some((nt, sl, bytes))
        })
    })?;
    // The output kernel: queries [tt][R][N], keys [st][R][N], their block
    // [tt R][st R], the keys' inputs [st][P].
    let scan = (1..=wide.min(CHUNK)).rev().find_map(|tt| {
        [8usize, 4, 2, 1].into_iter().find_map(|st| {
            let bytes = (tt * rank * ds + st * rank * ds + tt * rank * st * rank + st * hd) * 4;
            (bytes <= cap).then_some((tt, st, bytes))
        })
    })?;
    if !training {
        return Some(ChunkedGeom {
            state,
            scan,
            dsloc: (0, 0),
            pairs: (0, 0),
        });
    }
    // The local state gradient: queries [L][R][nt], decays [L].
    let dsloc = (1..=wide.min(ds)).rev().find_map(|nt| {
        let bytes = (CHUNK * rank * nt + CHUNK) * 4;
        (bytes <= cap).then_some((nt, bytes))
    })?;
    // The pair terms: queries, keys and the dq tile, dy and u, three
    // (tile R)^2 blocks.
    let pairs = [16usize, 8, 4, 2, 1].into_iter().find_map(|tl| {
        let tr = tl * rank;
        let bytes = (3 * tr * ds + 2 * tr * hd + 3 * tr * tr) * 4;
        (bytes <= cap).then_some((tl, bytes))
    })?;
    Some(ChunkedGeom {
        state,
        scan,
        dsloc,
        pairs,
    })
}

/// The scan's shared memory for a head: the state rows padded by one, the
/// previous keys, the keys and the queries, the angles' sines and cosines,
/// the step scalars.
pub fn scan_smem_bytes(hd: usize, ds: usize, rank: usize, na: usize) -> usize {
    (hd * (ds + 1) + 3 * rank * ds + 2 * na + 8) * 4
}

/// The training backward's shared memory for a head: the replay state rows
/// and the state-gradient rows (each padded by one), the replay keys, keys
/// and queries, the key and query gradients, the previous keys' carry and
/// the bias accumulators, three [P, R] tiles, the angle arrays, the step
/// scalars and one reduction slot per thread.
pub fn scan_bwd_smem_bytes(
    (hd, ds, rank, na): (usize, usize, usize, usize),
    threads: usize,
) -> usize {
    (2 * hd * (ds + 1) + 8 * rank * ds + 3 * hd * rank + 6 * na + 8 + threads) * 4
}

/// Tokens per chunk of the chunked path.
pub const CHUNK: usize = 64;

/// One token's record in the backward's replay buffer, in floats.
pub fn scan_bwd_record_floats(hd: usize, ds: usize, rank: usize, na: usize) -> usize {
    hd * ds + 4 * rank * ds + 3 * hd * rank + hd + 2 * na + 8
}

/// Launch geometry of the scan: one block per (batch, head), a whole
/// number of warps covering the head's channels.
pub fn scan_cfg(
    batch: usize,
    nh: usize,
    (hd, ds, rank, na): (usize, usize, usize, usize),
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, nh as u32, 1),
        block_dim: (hd.next_multiple_of(32) as u32, 1, 1),
        shared_mem_bytes: scan_smem_bytes(hd, ds, rank, na) as u32,
    }
}

impl Mamba3MimoKernels {
    /// Compile (or load from the cache) the MIMO module for `arch`.
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
                    include_str!("../../../kernels/mamba3_mimo.cu"),
                    include_str!("../../../kernels/mamba3_mimo_chunked.cu"),
                    include_str!("../../../kernels/mamba3_mimo_bwd.cu"),
                    include_str!("../../../kernels/mamba3_mimo_chunked_bwd.cu"),
                ],
                kind: ki::ModuleKind::Mamba3MimoCombined,
                cache_prefix: "mamba3-mimo-kernels-v1-",
                seed: "2603155690",
                label: "Mamba-3 MIMO",
            },
        )?;
        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("MIMO kernel '{name}' not found: {e:?}"))
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
        let mut kernels = Self {
            bc_norm: typed("mimo_bc_norm")?,
            scan: typed("mimo_scan")?,
            chunk_prep: typed("m3m_prep")?,
            chunk_state: typed("m3m_chunk_state")?,
            state_pass: get("m3m_state_pass")?,
            chunk_scan: typed("m3m_chunk_scan")?,
            chunk_writeback: typed("m3m_writeback")?,
            mlp_gate: typed("mlp_gate_fwd")?,
            scan_ckpt: typed("mimo_scan_ckpt")?,
            scan_bwd: typed("mimo_scan_bwd")?,
            bc_norm_bwd: typed("mimo_bc_norm_bwd")?,
            angle_bwd: typed("mimo_angle_bwd")?,
            mlp_gate_bwd: typed("mimo_mlp_gate_bwd")?,
            sum_rows: get("mimo_sum_rows")?,
            cb: MimoChunkedBwd {
                gate: typed("m3mb_gate")?,
                inter: get("m3mb_inter")?,
                dsloc: get("m3mb_dsloc")?,
                pass: get("m3mb_pass")?,
                chunk_dot: get("m3mb_chunk_dot")?,
                state_terms: typed("m3mb_state_terms")?,
                pairs: typed("m3mb_pairs")?,
                rot_bwd: get("m3mb_rot_bwd")?,
                angles: typed("m3mb_angles")?,
                combine: get("m3mb_combine")?,
                lead: get("m3mb_lead")?,
                scalars: typed("m3mb_scalars")?,
                u_bwd: typed("m3mb_u_bwd")?,
                sum_inner: get("m3mb_sum_inner")?,
                reduce_rows: get("m3mb_reduce_rows")?,
            },
            smem_optin,
            compiler_identity,
            artifact_identity,
        };
        kernels.smem_optin = {
            let optin = kernels.smem_optin;
            let mut budget = optin;
            for (k, name) in [
                (&kernels.scan, "scan"),
                (&kernels.scan_ckpt, "scan_ckpt"),
                (&kernels.scan_bwd, "scan_bwd"),
                (&kernels.chunk_state, "chunk_state"),
                (&kernels.chunk_scan, "chunk_scan"),
                (&kernels.cb.pairs, "pairs"),
            ] {
                for f in [&k.f32, &k.bf16, &k.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            budget.min(opt_in_dynamic_smem(&kernels.cb.dsloc, optin, "dsloc")?)
        };
        Ok(kernels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released head keeps the full tiles; a wide rank shrinks the
    /// scan's key tile; every chosen tile fits the limit; a head past 1024
    /// channels has no chunked geometry.
    #[test]
    fn chunked_geometry_fits_the_shared_memory() {
        let cap = 99 * 1024;
        let released = chunked_geom((64, 128, 4), cap, true).unwrap();
        assert_eq!((released.state.0, released.state.1), (4, CHUNK));
        assert_eq!((released.scan.0, released.scan.1), (4, 8));
        let wide = chunked_geom((16, 16, 16), cap, true).unwrap();
        assert_eq!((wide.scan.0, wide.scan.1), (16, 4));
        assert_eq!(wide.dsloc.0, 16);
        for shape in [
            (64, 128, 4),
            (16, 16, 16),
            (64, 256, 16),
            (40, 24, 3),
            (512, 16, 1),
            (1024, 256, 16),
            (8, 64, 6),
        ] {
            for cap in [48 * 1024, 64 * 1024, 99 * 1024, 227 * 1024] {
                let Some(g) = chunked_geom(shape, cap, true) else {
                    continue;
                };
                for bytes in [g.state.2, g.scan.2, g.dsloc.1, g.pairs.1] {
                    assert!(bytes <= cap, "{shape:?} at {cap}: {g:?}");
                }
                assert!(g.state.0 * shape.0 <= 1024 && g.scan.0 * shape.0 <= 1024);
            }
        }
        assert!(chunked_geom((2048, 16, 1), 227 * 1024, true).is_none());
    }
}
