//! The Mamba-3 MIMO CUDA module: `kernels/gated_mlp.cu`,
//! `kernels/mamba3_mimo.cu`, `kernels/mamba3_mimo_chunked.cu`,
//! `kernels/mamba3_mimo_bwd.cu` and `kernels/mamba3_mimo_chunked_bwd.cu`
//! after the typed prelude, under its own module kind. The norms, the
//! residual adds and the optimizer come from the Common module through the
//! context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::gpu::kernel_identity as ki;
use crate::gpu::kernels::TypedKernel;
use crate::gpu::launch::opt_in_dynamic_smem;

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
    pub chunk_state_tiles: TypedKernel,
    pub state_pass: CudaFunction,
    pub chunk_scan: TypedKernel,
    pub chunk_scan_tiles: TypedKernel,
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
    /// `m3mb_inter` and `m3mb_state_terms` for a tile of tokens at once,
    /// when [`ChunkedGeom::inter_tiles`] and [`ChunkedGeom::terms_tiles`]
    /// fit.
    pub inter_tiles: CudaFunction,
    pub state_terms_tiles: TypedKernel,
    pub dsloc: CudaFunction,
    /// `m3mb_dsloc` as register tiles, when [`ChunkedGeom::dsloc_tiles`]
    /// fits.
    pub dsloc_tiles: CudaFunction,
    pub pass: CudaFunction,
    pub chunk_dot: CudaFunction,
    pub state_terms: TypedKernel,
    pub pairs: TypedKernel,
    /// The pair terms split by owner: dq per t-tile, then dk, du and dw per
    /// s-tile, when [`ChunkedGeom::pairs_tiles`] fits.
    pub pairs_dq: TypedKernel,
    pub pairs_dk: TypedKernel,
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
    /// `m3m_chunk_state_tiles`: key columns per lane (1, 2 or 4; 0 when a
    /// head of more than 128 channels leaves it to `m3m_chunk_state`),
    /// tokens staged per pass, bytes.
    pub state_tiles: (usize, usize, usize),
    /// `m3m_chunk_scan`: query tokens per block, key tokens per pass, bytes.
    pub scan: (usize, usize, usize),
    /// `m3m_chunk_scan_tiles`: query tokens per block (0 when a head of
    /// more than 128 channels or a rank past 64 leaves it to
    /// `m3m_chunk_scan`), key tokens per tile, state indices per key pass,
    /// bytes.
    pub scan_tiles: (usize, usize, usize, usize),
    /// `m3mb_dsloc`: state indices per block, bytes.
    pub dsloc: (usize, usize),
    /// `m3mb_dsloc_tiles`: key columns per lane (0 when a head of more than
    /// 128 channels leaves it to `m3mb_dsloc`), tokens staged per pass,
    /// bytes.
    pub dsloc_tiles: (usize, usize, usize),
    /// `m3mb_inter_tiles` and `m3mb_state_terms_tiles`: tokens per block
    /// and bytes, (0, 0) when a head of more than 128 channels or state
    /// indices, or a rank past 64, leaves them to the per-token kernels.
    pub inter_tiles: (usize, usize),
    pub terms_tiles: (usize, usize),
    /// `m3mb_pairs`: tokens per tile, bytes.
    pub pairs: (usize, usize),
    /// `m3mb_pairs_dq` and `m3mb_pairs_dk` at the same tile: their bytes, or
    /// zeros when a tile of more than 64 rows, a head of more than 128
    /// channels or more than 128 state indices leaves it to `m3mb_pairs`.
    pub pairs_tiles: (usize, usize),
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
    // The register-tile state kernel: ut [sl][R][P rounded up to 4],
    // kt [sl][R][32 jc], wt [sl]. The largest pass that leaves room for a
    // second block on the multiprocessor, else the largest that fits. The
    // backward's local state gradient has the same shape.
    let state_tiles = if hd <= 128 {
        let jc = match ds.div_ceil(32) {
            1 => 1,
            2 => 2,
            _ => 4,
        };
        let bytes = |sl: usize| (sl * rank * (hd.next_multiple_of(4) + 32 * jc) + sl) * 4;
        let passes = [CHUNK, 32, 16, 8, 4, 2, 1];
        passes
            .into_iter()
            .find(|&sl| bytes(sl) <= cap / 2)
            .or_else(|| passes.into_iter().find(|&sl| bytes(sl) <= cap))
            .map_or((0, 0, 0), |sl| (jc, sl, bytes(sl)))
    } else {
        (0, 0, 0)
    };
    // The output kernel: queries [tt][R][N], keys [st][R][N + 1], their
    // block [tt R][st R], the keys' inputs [st][P].
    let scan = (1..=wide.min(CHUNK)).rev().find_map(|tt| {
        [8usize, 4, 2, 1].into_iter().find_map(|st| {
            let bytes =
                (tt * rank * ds + st * rank * (ds + 1) + tt * rank * st * rank + st * hd) * 4;
            (bytes <= cap).then_some((tt, st, bytes))
        })
    })?;
    // The register-tile output kernel: queries [N][QR + 4], keys [nk][68],
    // inputs [64][32 jp], weighted G [64][QR + 4], weights [tq][kt] and the
    // chunk's three scalars, QR = tq R rounded up to 8 (at most 64 rows, so
    // at most 512 threads); the keys take the most state indices per pass
    // that fit.
    let scan_tiles = if hd <= 128 && rank <= CHUNK {
        let tq = [16usize, 8, 4, 2, 1]
            .into_iter()
            .find(|&tq| tq * rank <= CHUNK)
            .unwrap_or(1);
        let kt = CHUNK / rank;
        let qrs = (tq * rank).next_multiple_of(8) + 4;
        let pp = 32 * hd.div_ceil(32).next_power_of_two();
        let bytes = |nk: usize| {
            (ds * qrs + nk * (CHUNK + 4) + CHUNK * pp + CHUNK * qrs + tq * kt + 3 * CHUNK) * 4
        };
        std::iter::successors(Some(ds), |&nk| (nk > 1).then(|| nk.div_ceil(2)))
            .find(|&nk| bytes(nk) <= cap)
            .map_or((0, 0, 0, 0), |nk| (tq, kt, nk, bytes(nk)))
    } else {
        (0, 0, 0, 0)
    };
    if !training {
        return Some(ChunkedGeom {
            state,
            state_tiles,
            scan,
            scan_tiles,
            dsloc: (0, 0),
            dsloc_tiles: (0, 0, 0),
            inter_tiles: (0, 0),
            terms_tiles: (0, 0),
            pairs: (0, 0),
            pairs_tiles: (0, 0),
        });
    }
    // The local state gradient: queries [L][R][nt], decays [L].
    let dsloc = (1..=wide.min(ds)).rev().find_map(|nt| {
        let bytes = (CHUNK * rank * nt + CHUNK) * 4;
        (bytes <= cap).then_some((nt, bytes))
    })?;
    // The pair terms: queries, keys and the dq tile, dy and u, three
    // (tile R)^2 blocks; the key and u rows padded by one.
    let pairs = [16usize, 8, 4, 2, 1].into_iter().find_map(|tl| {
        let tr = tl * rank;
        let bytes = (3 * tr * ds + 2 * tr * hd + 2 * tr + 3 * tr * tr) * 4;
        (bytes <= cap).then_some((tl, bytes))
    })?;
    // The token-tile inter and state-terms kernels: rows (t, r) of `tt`
    // tokens padded to a float4 multiple plus 4. inter: the entering state,
    // dy transposed, d, and the trees' partials past d when the first two do
    // not hold them; state terms: dS with rows padded by one, u and the keys
    // transposed, ste, the partials. The largest token tile that fits.
    let token_tiles = |bytes: &dyn Fn(usize, usize) -> usize| {
        if hd > 128 || ds > 128 || rank > CHUNK {
            return (0, 0);
        }
        [16usize, 8, 4, 2, 1]
            .into_iter()
            .filter(|&tt| tt * rank <= CHUNK)
            .map(|tt| (tt, bytes(tt, (tt * rank).next_multiple_of(4) + 4)))
            .find(|&(_, b)| b <= cap)
            .unwrap_or((0, 0))
    };
    let inter_tiles = token_tiles(&|tt, rs| {
        let front = hd * ds + hd * rs;
        let red = if tt * 256 > front { tt * 256 } else { 0 };
        (front + tt * rank * ds + red) * 4
    });
    let terms_tiles =
        token_tiles(&|tt, rs| (hd * (ds + 1) + hd * rs + ds * rs + tt * rank * hd + tt * 256) * 4);
    // The split pair kernels: dy (transposed), u, the keys, w M (transposed)
    // and w for dq; the keys and u (transposed), the queries and dy, G, w M
    // and w G for dk and du. Rows padded to a float4 multiple plus 4.
    let pairs_tiles = {
        let tl = pairs.0;
        let tr = tl * rank;
        let trs = tr.next_multiple_of(4) + 4;
        let dq = (hd * trs + tr * (hd + 1) + tr * ds + tr * trs + tl * tl) * 4;
        let dk = (ds * trs + hd * trs + tr * (ds + 1) + tr * (hd + 1) + 3 * tr * trs) * 4;
        if tr <= 64 && hd <= 128 && ds <= 128 && dq.max(dk) <= cap {
            (dq, dk)
        } else {
            (0, 0)
        }
    };
    Some(ChunkedGeom {
        state,
        state_tiles,
        scan,
        scan_tiles,
        dsloc,
        dsloc_tiles: state_tiles,
        inter_tiles,
        terms_tiles,
        pairs,
        pairs_tiles,
    })
}

/// The scan's shared memory for a head: the state rows padded by one, the
/// keys starting on 16 bytes in three buffers (the previous token's, the
/// current one's and the next one's while it is built), the queries and
/// the step scalars in two. The angles stay in their threads' registers.
pub fn scan_smem_bytes(hd: usize, ds: usize, rank: usize) -> usize {
    ((hd * (ds + 1)).next_multiple_of(4) + 5 * rank * ds + 8) * 4
}

/// The training backward's shared memory for a head: the replay state rows
/// and the state-gradient rows (each padded by one, each block of rows to a
/// multiple of four floats so the keys start on 16 bytes), the replay
/// keys, keys and queries, the key and query gradients, the previous keys'
/// carry and the bias accumulators, three [P, R] tiles, three angle arrays
/// (the token's angle gradient, its carry and tanh_pi) and three reduction
/// slots per channel thread (the channels rounded up to a warp).
pub fn scan_bwd_smem_bytes((hd, ds, rank, na): (usize, usize, usize, usize)) -> usize {
    (2 * (hd * (ds + 1)).next_multiple_of(4)
        + 8 * rank * ds
        + 3 * hd * rank
        + 3 * na
        + 3 * hd.next_multiple_of(32))
        * 4
}

/// The training backward's block: a thread per channel, and at least 256
/// threads, since its loops over state elements and (r, n) pairs spread
/// over the whole block while only the channel threads walk the rows.
pub fn scan_bwd_threads(hd: usize) -> usize {
    hd.next_multiple_of(32).max(256)
}

/// Tokens per chunk of the chunked path.
pub const CHUNK: usize = 64;

/// One token's record in the backward's replay buffer, in floats.
pub fn scan_bwd_record_floats(hd: usize, ds: usize, rank: usize, na: usize) -> usize {
    hd * ds + 4 * rank * ds + 3 * hd * rank + hd + 2 * na + 8
}

/// Launch geometry of the scan: one block per (batch, head), a whole
/// number of warps covering the head's channels and at least eight, so a
/// decode step moves the head's state through shared memory in a few
/// vector loads per thread (the scan has no block-wide reduction, so its
/// bits do not depend on the block size).
pub fn scan_cfg(
    batch: usize,
    nh: usize,
    (hd, ds, rank, na): (usize, usize, usize, usize),
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, nh as u32, 1),
        block_dim: (hd.max(na).next_multiple_of(32).max(256) as u32, 1, 1),
        shared_mem_bytes: scan_smem_bytes(hd, ds, rank) as u32,
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
            chunk_state_tiles: typed("m3m_chunk_state_tiles")?,
            state_pass: get("m3m_state_pass")?,
            chunk_scan: typed("m3m_chunk_scan")?,
            chunk_scan_tiles: typed("m3m_chunk_scan_tiles")?,
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
                inter_tiles: get("m3mb_inter_tiles")?,
                state_terms_tiles: typed("m3mb_state_terms_tiles")?,
                dsloc: get("m3mb_dsloc")?,
                dsloc_tiles: get("m3mb_dsloc_tiles")?,
                pass: get("m3mb_pass")?,
                chunk_dot: get("m3mb_chunk_dot")?,
                state_terms: typed("m3mb_state_terms")?,
                pairs: typed("m3mb_pairs")?,
                pairs_dq: typed("m3mb_pairs_dq")?,
                pairs_dk: typed("m3mb_pairs_dk")?,
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
                (&kernels.chunk_state_tiles, "chunk_state_tiles"),
                (&kernels.chunk_scan, "chunk_scan"),
                (&kernels.chunk_scan_tiles, "chunk_scan_tiles"),
                (&kernels.cb.pairs, "pairs"),
                (&kernels.cb.pairs_dq, "pairs dq"),
                (&kernels.cb.pairs_dk, "pairs dk"),
                (&kernels.cb.state_terms, "state_terms"),
                (&kernels.cb.state_terms_tiles, "state_terms tiles"),
            ] {
                for f in [&k.f32, &k.bf16, &k.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            budget = budget.min(opt_in_dynamic_smem(&kernels.cb.dsloc, optin, "dsloc")?);
            budget = budget.min(opt_in_dynamic_smem(
                &kernels.cb.inter_tiles,
                optin,
                "inter tiles",
            )?);
            budget.min(opt_in_dynamic_smem(
                &kernels.cb.dsloc_tiles,
                optin,
                "dsloc tiles",
            )?)
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
        assert_eq!((released.state_tiles.0, released.state_tiles.1), (4, 16));
        let tiles = released.scan_tiles;
        assert_eq!((tiles.0, tiles.1, tiles.2), (16, 16, 64));
        assert_eq!(released.pairs_tiles, (38784, 66304));
        assert_eq!(released.inter_tiles.0, 16);
        assert_eq!(released.terms_tiles.0, 8);
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
                let sizes = [
                    g.state.2,
                    g.state_tiles.2,
                    g.scan.2,
                    g.scan_tiles.3,
                    g.dsloc.1,
                    g.pairs.1,
                    g.pairs_tiles.0,
                    g.inter_tiles.1,
                    g.terms_tiles.1,
                    g.pairs_tiles.1,
                ];
                for bytes in sizes {
                    assert!(bytes <= cap, "{shape:?} at {cap}: {g:?}");
                }
                assert!(g.state.0 * shape.0 <= 1024 && g.scan.0 * shape.0 <= 1024);
                assert_eq!(g.state_tiles.0 == 0, shape.0 > 128, "{shape:?}: {g:?}");
            }
        }
        assert!(chunked_geom((2048, 16, 1), 227 * 1024, true).is_none());
    }
}
