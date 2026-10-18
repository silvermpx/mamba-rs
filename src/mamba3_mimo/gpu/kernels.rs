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
use crate::gpu::kernels::{CudaModuleAnchors, TypedKernel};
use crate::gpu::launch::{opt_in_dynamic_smem, smem_budget};
use crate::mamba3_mimo::config::MAX_MIMO_RANK;

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
    /// `m3m_chunk_scan_tiles`, one entry for each channel count a lane
    /// takes (see [`Self::chunk_scan_tiles_for`]).
    pub chunk_scan_tiles: [TypedKernel; 3],
    pub chunk_writeback: TypedKernel,
    /// The MLP's gate, `value * silu(gate)`.
    pub mlp_gate: TypedKernel,
    /// The gate with the per-rank output norm over the scans' readouts.
    pub gate_normed: TypedKernel,
    /// The training forward's scan, writing the replay checkpoints.
    pub scan_ckpt: TypedKernel,
    /// The widest block both scans launch, from their register use.
    pub scan_threads: usize,
    /// The sequential backward of the scan (replay per segment).
    pub scan_bwd: TypedKernel,
    /// `mimo_scan_bwd` for a head whose rows, keys or tiles do not fit
    /// shared memory (see [`scan_bwd_global_geom`]).
    pub scan_bwd_global: TypedKernel,
    /// The widest block each sequential backward launches, from its
    /// register use.
    pub scan_bwd_block: usize,
    pub scan_bwd_global_block: usize,
    pub bc_norm_bwd: TypedKernel,
    pub angle_bwd: TypedKernel,
    pub mlp_gate_bwd: TypedKernel,
    /// The per-sequence weight partials summed over the sequences.
    pub sum_rows: CudaFunction,
    /// The chunked path's backward.
    pub cb: MimoChunkedBwd,
    /// The dynamic shared memory each kernel staging more than the default
    /// 48 KB may use: the device's per-block opt-in less the largest
    /// static stage among those kernels, or less when
    /// `MAMBA_RS_SMEM_BUDGET` asks for it.
    pub smem_optin: usize,
    /// The widest blocks the chunked path's kernels launch, from their
    /// register use.
    pub chunk_threads: ChunkThreads,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
    /// The loaded module, held by every graph that launches these kernels
    /// so it outlives them.
    _modules: CudaModuleAnchors,
}

/// The chunked path's backward kernels (`kernels/mamba3_mimo_chunked_bwd.cu`).
pub struct MimoChunkedBwd {
    pub gate: TypedKernel,
    /// The gate's backward with the output norm, per (row, head).
    pub gate_normed: TypedKernel,
    /// The inter-chunk part of dq and its decay term, a tile of tokens at
    /// once, at every shape ([`ChunkedGeom::inter_tiles`]).
    pub inter_tiles: CudaFunction,
    /// The state terms for a tile of tokens at once, at every shape
    /// ([`ChunkedGeom::terms`]).
    pub state_terms_tiles: TypedKernel,
    pub dsloc: CudaFunction,
    /// `m3mb_dsloc` as register tiles, when [`ChunkedGeom::dsloc_tiles`]
    /// fits.
    pub dsloc_tiles: CudaFunction,
    pub pass: CudaFunction,
    pub chunk_dot: CudaFunction,
    /// The pair terms split by owner: dq per t-tile, then dk, du and dw per
    /// s-tile, at every shape ([`ChunkedGeom::pairs`]).
    pub pairs_dq: TypedKernel,
    pub pairs_dk: TypedKernel,
    /// The narrow builds of the pair kernels, at most 256 threads a block
    /// and one block a multiprocessor ([`PairsPlan::narrow`]).
    pub pairs_dq_narrow: TypedKernel,
    pub pairs_dk_narrow: TypedKernel,
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
    /// `m3m_chunk_state_tiles`: key columns per lane (1, 2 or 4; 0 when no
    /// pass fits the shared memory and `m3m_chunk_state` serves), tokens
    /// staged per pass, channels per block, bytes.
    pub state_tiles: (usize, usize, usize, usize),
    /// `m3m_chunk_scan`: query tokens per block, key tokens per pass, bytes.
    pub scan: (usize, usize, usize),
    /// `m3m_chunk_scan_tiles`: query tokens per block (0 when a rank past 64
    /// leaves it to `m3m_chunk_scan`), key tokens per tile, state indices per
    /// key pass, bytes. A head past [`SCAN_TILE_CHANNELS`] takes groups of
    /// that many channels (see [`chunk_scan_tiles_cfg`]).
    pub scan_tiles: (usize, usize, usize, usize),
    /// `m3mb_dsloc`: state indices per block, bytes.
    pub dsloc: (usize, usize),
    /// `m3mb_dsloc_tiles`: key columns per lane (0 when no pass fits the
    /// shared memory and `m3mb_dsloc` serves), tokens staged per pass,
    /// channels per block, bytes.
    pub dsloc_tiles: (usize, usize, usize, usize),
    /// `m3mb_inter_tiles`: tokens per block, channels per slice of the
    /// entering state, bytes. Every shape has one: the state is staged a
    /// slice of channels at a time, its columns 128 at a time.
    pub inter_tiles: (usize, usize, usize),
    /// `m3mb_state_terms_tiles`: its plan ([`terms_plan`]). Every shape has
    /// one.
    pub terms: TermsPlan,
    /// The pair terms' reference tile: the tokens a tile of the pair kernel
    /// as it was released (frozen in the tests), whose masked zeros the
    /// split kernels' chains keep.
    pub pairs_ref: usize,
    /// Channels per block of `m3m_chunk_state`, `m3m_chunk_scan` and
    /// `m3mb_dsloc`: the whole head when its channels fit a block, else a
    /// tile of them (each output keeps its one chain either way).
    pub pt: usize,
    /// `m3mb_pairs_dq` and `m3mb_pairs_dk`: their tile of tokens, slices and
    /// bytes ([`pairs_plan`]). Every shape has one.
    pub pairs: PairsPlan,
}

/// The split pair kernels' plan: their tile of tokens, the channels a slice
/// of dy and u in `m3mb_pairs_dq`, the state indices a slice of the keys and
/// the channels a slice of u in `m3mb_pairs_dk`, and each kernel's bytes. A
/// slice as wide as its operand stages it once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairsPlan {
    pub tq: usize,
    pub ps: usize,
    pub dq_bytes: usize,
    pub kns: usize,
    pub pss: usize,
    pub dk_bytes: usize,
    /// The narrow builds run the plan ([`pairs_narrow`]), whose tile keeps
    /// a block within [`PAIRS_THREADS_NARROW`] threads.
    pub narrow: bool,
}

/// The widest blocks the chunked path launches, a whole number of warps
/// each, from the registers every dtype of its kernels compiled to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkThreads {
    /// The channel-parallel kernels: the chunk state and scan, the local
    /// state gradient and the lead.
    pub generic: usize,
    /// The register-tile chunk state, `m3m_chunk_state_tiles`.
    pub state_tiles: usize,
    /// The register-tile local state gradient, `m3mb_dsloc_tiles`.
    pub dsloc_tiles: usize,
}

impl ChunkThreads {
    /// One limit for every kernel.
    pub fn uniform(threads: usize) -> Self {
        Self {
            generic: threads,
            state_tiles: threads,
            dsloc_tiles: threads,
        }
    }
}

/// The register-tile geometry of the chunk state and the local state
/// gradient for a head of `hd` channels, `ds` state indices and rank `rank`
/// under blocks of at most `threads`: key columns per lane, tokens staged
/// per pass, channels per block and bytes, or zeros when no pass fits. A
/// warp holds 4 channels, so a block of the whole head needs
/// 8 threads per channel; past `threads` the head goes in groups as even as
/// the limit allows. The staged rows are the group's channels and the keys:
/// ut [sl][R][pg], kt [sl][R][32 jc], wt [sl]. The largest pass that leaves
/// room for a second block on the multiprocessor is taken, else the largest
/// that fits.
fn tiles_geom(
    (hd, ds, rank): (usize, usize, usize),
    cap: usize,
    threads: usize,
) -> (usize, usize, usize, usize) {
    let quads = hd.div_ceil(4);
    let groups = quads.div_ceil((threads / 32).max(1));
    let pg = 4 * quads.div_ceil(groups);
    let jc = match ds.div_ceil(32) {
        1 => 1,
        2 => 2,
        _ => 4,
    };
    let bytes = |sl: usize| (sl * rank * (pg + 32 * jc) + sl) * 4;
    let passes = [CHUNK, 32, 16, 8, 4, 2, 1];
    passes
        .into_iter()
        .find(|&sl| bytes(sl) <= cap / 2)
        .or_else(|| passes.into_iter().find(|&sl| bytes(sl) <= cap))
        .map_or((0, 0, 0, 0), |sl| (jc, sl, pg, bytes(sl)))
}

/// The chunked geometry of a head of `hd` channels, `ds` state indices and
/// rank `rank` within `cap` bytes of shared memory and the blocks `threads`
/// allows, or `None` when a kernel of the path cannot fit. The backward's
/// kernels count only when `training`.
pub fn chunked_geom(
    (hd, ds, rank): (usize, usize, usize),
    cap: usize,
    training: bool,
    threads: ChunkThreads,
) -> Option<ChunkedGeom> {
    if hd == 0 || ds == 0 || rank == 0 {
        return None;
    }
    let pt = hd.min((threads.generic / 32 * 32).max(32));
    let wide = (256 / pt).max(1);
    // The state kernel: kt [sl][R][nt], xt [sl][P], wt [sl].
    let state = (1..=wide.min(ds)).rev().find_map(|nt| {
        [CHUNK, 32, 16, 8, 4, 2, 1].into_iter().find_map(|sl| {
            let bytes = (sl * rank * nt + sl * pt + sl) * 4;
            (bytes <= cap).then_some((nt, sl, bytes))
        })
    })?;
    let state_tiles = tiles_geom((hd, ds, rank), cap, threads.state_tiles);
    // The output kernel: queries [tt][R][N], keys [st][R][N + 1], their
    // block [tt R][st R], the keys' inputs [st][P].
    let scan = (1..=wide.min(CHUNK)).rev().find_map(|tt| {
        [8usize, 4, 2, 1].into_iter().find_map(|st| {
            let bytes =
                (tt * rank * ds + st * rank * (ds + 1) + tt * rank * st * rank + st * pt) * 4;
            (bytes <= cap).then_some((tt, st, bytes))
        })
    })?;
    // The register-tile output kernel: queries [N][QR + 4], keys [nk][68],
    // inputs [64][32 jp], weighted G [64][QR + 4], weights [tq][kt] and the
    // chunk's three scalars, QR = tq R rounded up to 8 (at most 64 rows, so
    // at most 512 threads); the keys take the most state indices per pass
    // that fit.
    let scan_tiles = if rank <= CHUNK {
        let tq = [16usize, 8, 4, 2, 1]
            .into_iter()
            .find(|&tq| tq * rank <= CHUNK)
            .unwrap_or(1);
        let kt = CHUNK / rank;
        let qrs = (tq * rank).next_multiple_of(8) + 4;
        let lanes = hd
            .div_ceil(32)
            .next_power_of_two()
            .min(SCAN_TILE_CHANNELS / 32);
        let pp = 32 * lanes;
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
            dsloc_tiles: (0, 0, 0, 0),
            inter_tiles: (0, 0, 0),
            terms: TermsPlan::default(),
            pairs_ref: 0,
            pt,
            pairs: PairsPlan::default(),
        });
    }
    // The local state gradient: queries [L][R][nt], decays [L].
    let dsloc = (1..=wide.min(ds)).rev().find_map(|nt| {
        let bytes = (CHUNK * rank * nt + CHUNK) * 4;
        (bytes <= cap).then_some((nt, bytes))
    })?;
    // The pair kernel as it was released took the largest token tile whose
    // queries, keys and dq tile, dy and u and three (tile R)^2 blocks (the
    // key and u rows padded by one) its shared memory held, else the
    // largest whose three blocks it held, its rows then in global memory.
    // The masked zeros of that tile are part of its chains, which the split
    // kernels keep, so the tile is still chosen as it chose it.
    let pairs_ref = [16usize, 8, 4, 2, 1]
        .into_iter()
        .find(|&tl| {
            let tr = tl * rank;
            (3 * tr * ds + 2 * tr * hd + 2 * tr + 3 * tr * tr) * 4 <= cap
        })
        .or_else(|| {
            [16usize, 8, 4, 2, 1]
                .into_iter()
                .find(|&tl| 3 * (tl * rank).pow(2) * 4 <= cap)
        })?;
    let inter_tiles = Some(inter_tiles_geom((hd, ds, rank), cap)).filter(|g| g.0 > 0)?;
    let terms = terms_plan((hd, ds, rank), cap)?;
    let pairs = pairs_plan((hd, ds, rank), cap, pairs_narrow(hd, ds, rank))?;
    Some(ChunkedGeom {
        state,
        state_tiles,
        scan,
        scan_tiles,
        dsloc,
        dsloc_tiles: tiles_geom((hd, ds, rank), cap, threads.dsloc_tiles),
        inter_tiles,
        terms,
        pairs_ref,
        pt,
        pairs,
    })
}

/// `m3mb_pairs_dq`'s shared memory at a tile of `tq` tokens (tr = tq R
/// rows, rows padded to a float4 multiple plus 4) and slices of `ps`
/// channels: dy transposed [ps][TR4 + 4], u [tr][ps + 1] and the keys' 128
/// state indices [tr][min(N, 128)], padded to a float4 multiple, then w M
/// transposed `[tr][TR4 + 4]` and w `[tq][tq]`.
pub fn pairs_dq_bytes((hd, ds, rank): (usize, usize, usize), tq: usize, ps: usize) -> usize {
    let tr = tq * rank;
    let trs = tr.next_multiple_of(4) + 4;
    let ps = ps.min(hd);
    (ps * trs + (tr * (ps + 1 + ds.min(128))).next_multiple_of(4) + tr * trs + tq * tq) * 4
}

/// `m3mb_pairs_dk`'s shared memory at a tile of `tq` tokens and slices of
/// `kns` state indices and `pss` channels: the keys and u transposed
/// ([kns][TR4 + 4], [pss][TR4 + 4]), the queries and dy a row each (their
/// whole width where a slice is whole, else the larger of the slice and the
/// 128 columns a block's role reads, plus one), padded to a float4
/// multiple, then G and M [tr][TR4 + 4] and the pairs' weights and live
/// flags `[tq][tq]`.
pub fn pairs_dk_bytes(
    (hd, ds, rank): (usize, usize, usize),
    tq: usize,
    (kns, pss): (usize, usize),
) -> usize {
    let tr = tq * rank;
    let trs = tr.next_multiple_of(4) + 4;
    let (kn, pn) = (kns.min(ds), pss.min(hd));
    let row = |slice: usize, width: usize| if slice == width { width } else { slice.max(128) } + 1;
    (kn * trs
        + pn * trs
        + (tr * (row(kn, ds) + row(pn, hd))).next_multiple_of(4)
        + 2 * tr * trs
        + 2 * tq * tq)
        * 4
}

/// The split pair kernels' plan for a head of `hd` channels, `ds` state
/// indices and rank `rank` under `cap` bytes: the largest tile of tokens (a
/// power of two up to 16, at most 64 rows) whose kernels fit with slices of
/// at least a warp's worth (32, or the whole width when it is narrower),
/// each kernel's slices the widest that fit (dk's keys and u halved in turn,
/// the wider first); else a tile of one token at the widest slices that fit
/// at all. A tile of one token in slices of one fits any card, so `None`
/// only under a cap no card has.
pub fn pairs_plan(
    (hd, ds, rank): (usize, usize, usize),
    cap: usize,
    narrow: bool,
) -> Option<PairsPlan> {
    let dims = (hd, ds, rank);
    let fit = |tq: usize, floor: usize| {
        let ps = std::iter::successors(Some(hd), |&ps| (ps > 1).then(|| ps.div_ceil(2)))
            .take_while(|&ps| ps >= hd.min(floor))
            .find(|&ps| pairs_dq_bytes(dims, tq, ps) <= cap)?;
        let (k_floor, p_floor) = (ds.min(floor), hd.min(floor));
        let (kns, pss) = std::iter::successors(Some((ds, hd)), |&(kn, pn)| {
            let (k2, p2) = (kn.div_ceil(2), pn.div_ceil(2));
            match (kn > 1 && k2 >= k_floor, pn > 1 && p2 >= p_floor) {
                (true, true) if kn >= pn => Some((k2, pn)),
                (_, true) => Some((kn, p2)),
                (true, false) => Some((k2, pn)),
                (false, false) => None,
            }
        })
        .find(|&slices| pairs_dk_bytes(dims, tq, slices) <= cap)?;
        Some(PairsPlan {
            tq,
            ps,
            dq_bytes: pairs_dq_bytes(dims, tq, ps),
            kns,
            pss,
            dk_bytes: pairs_dk_bytes(dims, tq, (kns, pss)),
            narrow,
        })
    };
    let fits_block = |tq: usize| match narrow {
        true => pairs_threads(tq, rank) <= PAIRS_THREADS_NARROW,
        false => tq * rank <= CHUNK,
    };
    [16usize, 8, 4, 2, 1]
        .into_iter()
        .filter(|&tq| fits_block(tq))
        .find_map(|tq| fit(tq, 32))
        .or_else(|| fit(1, 1))
}

/// The most threads a block of the narrow pair kernels takes. Mirrors
/// `M3MB_PAIRS_BOUNDS_NARROW` in mamba3_mimo_chunked_bwd.cu.
pub const PAIRS_THREADS_NARROW: usize = 256;

/// The threads of a pair kernel's block at a tile of `tq` tokens: eight a
/// row of the tile's `tq * rank` rows, rounded up to four rows.
pub fn pairs_threads(tq: usize, rank: usize) -> usize {
    8 * (tq * rank).next_multiple_of(4)
}

/// Whether the pair kernels take their narrow builds, which keep dk's
/// accumulators in registers at half the warps a multiprocessor where the
/// wide builds spill them. The spills cost more from a head of 128 channels,
/// and on a head of 64 or more at 256 states, at 128 states past rank 1 and,
/// from a head of 96, at rank 4 and up. Everywhere else the warps win. Both
/// give the same bits.
pub fn pairs_narrow(hd: usize, ds: usize, rank: usize) -> bool {
    if let Some(forced) = pairs_build_forced() {
        return forced;
    }
    hd >= 128 || (hd >= 64 && (ds >= 256 || (ds >= 128 && rank >= 2) || (hd >= 96 && rank >= 4)))
}

/// `MAMBA_RS_PAIRS_BUILD` picks the pair kernels' build for tests and
/// measurements: `narrow` or `wide` takes that build at every shape, and
/// unset or `auto` leaves the choice to [`pairs_narrow`]. Any other value is
/// refused loudly, so a typo cannot report a run that never left the rule.
fn pairs_build_forced() -> Option<bool> {
    static FORCED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| match std::env::var("MAMBA_RS_PAIRS_BUILD") {
        Err(_) => None,
        Ok(v) => match v.trim() {
            "auto" | "" => None,
            "narrow" => Some(true),
            "wide" => Some(false),
            other => panic!(
                "MAMBA_RS_PAIRS_BUILD={other:?} is not a recognized build (use auto, narrow or wide)"
            ),
        },
    })
}

/// The token-tile state terms' plan: tokens per block, whether dS, u and
/// the rotated keys stage whole, the channels a slice of u and dS's rows in
/// the dk pass and the state indices a slice of the keys and dS's columns
/// in the ste pass (the whole widths when they stage whole), whether the
/// ste rows go to a global scratch, and the bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermsPlan {
    pub tt: usize,
    pub whole: bool,
    pub ps: usize,
    pub ns: usize,
    pub steb_global: bool,
    pub bytes: usize,
}

/// The token-tile state terms' shared memory at `tt` tokens a block (rows
/// padded to a float4 multiple plus 4, rs) with dS, u and the keys whole:
/// dS `[P][N + 1]` padded to a float4 multiple, u `[P][rs]`, the keys `[N][rs]`,
/// ste `[tt R][P]` and the trees' `[tt][256]`.
pub fn terms_whole_bytes((hd, ds, rank): (usize, usize, usize), tt: usize) -> usize {
    let rs = (tt * rank).next_multiple_of(4) + 4;
    ((hd * (ds + 1)).next_multiple_of(4) + hd * rs + ds * rs + tt * rank * hd + tt * 256) * 4
}

/// The token-tile state terms' shared memory at `tt` tokens a block with
/// slices of `ps` channels and `ns` state indices: the larger of the dk
/// pass's u and dS slices (`[ps][rs]`, `[ps][min(N, 128)]`) and the ste pass's
/// key and dS slices (`[ns][rs]`, `[min(P, 128)][ns rounded up to odd]`),
/// padded to a float4 multiple, then ste `[tt R][P]` unless it goes to the
/// global scratch, then the trees' `[tt][256]` unless the slices' room holds
/// them.
pub fn terms_sliced_bytes(
    (hd, ds, rank): (usize, usize, usize),
    tt: usize,
    (ps, ns): (usize, usize),
    steb_global: bool,
) -> usize {
    let rs = (tt * rank).next_multiple_of(4) + 4;
    let (ps, ns) = (ps.min(hd), ns.min(ds));
    let room = (ps * rs + ps * ds.min(128))
        .max(ns * rs + hd.min(128) * ((ns & !1) + 1))
        .next_multiple_of(4);
    let steb = if steb_global { 0 } else { tt * rank * hd };
    let trees = if tt * 256 <= room { 0 } else { tt * 256 };
    (room + steb + trees) * 4
}

/// The token-tile state terms' plan for a head of `hd` channels, `ds`
/// state indices and rank `rank` under `cap` bytes: the largest tile of
/// tokens (a power of two up to 16, at most 64 rows) whose dS, u and keys
/// stage whole; else the largest whose slices of at least a warp's worth
/// (32, or the whole width when it is narrower) fit with ste in shared
/// memory, then with ste in the global scratch, each pass's slices the
/// widest that fit; else a tile of one token at the widest slices that fit
/// at all. One token in slices of one fits any card, so `None` only under a
/// cap no card has.
pub fn terms_plan((hd, ds, rank): (usize, usize, usize), cap: usize) -> Option<TermsPlan> {
    let dims = (hd, ds, rank);
    let tiles = || {
        [16usize, 8, 4, 2, 1]
            .into_iter()
            .filter(move |&tt| tt * rank <= CHUNK)
    };
    if let Some(tt) = tiles().find(|&tt| terms_whole_bytes(dims, tt) <= cap) {
        return Some(TermsPlan {
            tt,
            whole: true,
            ps: hd,
            ns: ds,
            steb_global: false,
            bytes: terms_whole_bytes(dims, tt),
        });
    }
    let halvings = |width: usize, floor: usize| {
        std::iter::successors(Some(width), |&w| (w > 1).then(|| w.div_ceil(2)))
            .take_while(move |&w| w >= width.min(floor))
    };
    let sliced = |tt: usize, floor: usize, steb_global: bool| {
        halvings(hd, floor).find_map(|ps| {
            halvings(ds, floor).find_map(|ns| {
                let bytes = terms_sliced_bytes(dims, tt, (ps, ns), steb_global);
                (bytes <= cap).then_some(TermsPlan {
                    tt,
                    whole: false,
                    ps,
                    ns,
                    steb_global,
                    bytes,
                })
            })
        })
    };
    tiles()
        .find_map(|tt| sliced(tt, 32, false))
        .or_else(|| tiles().find_map(|tt| sliced(tt, 32, true)))
        .or_else(|| sliced(1, 1, true))
}

/// The token-tile inter kernel's shared memory for `ds` state indices at
/// rank `rank`, `tt` tokens a block and slices of `ps` channels: per pass
/// of at most 128 state indices a slice of the entering state (`[ps][W]`),
/// padded to a float4 multiple, and dy's slice transposed after it
/// (`[ps][R4 + 4]`, R4 = tt R rounded up to 4), then d `[tt R][N]` for the
/// trees, whose `[tt][256]` partials take the slices' room when it holds
/// them and follow d otherwise.
pub fn inter_tiles_bytes((ds, rank): (usize, usize), tt: usize, ps: usize) -> usize {
    let slices =
        (ps * ds.min(128)).next_multiple_of(4) + ps * ((tt * rank).next_multiple_of(4) + 4);
    let red = if tt * 256 <= slices { 0 } else { tt * 256 };
    (slices + tt * rank * ds + red) * 4
}

/// The token-tile inter kernel's plan for a head of `hd` channels, `ds`
/// state indices and rank `rank` under `cap` bytes ([`inter_tiles_bytes`]):
/// tokens per block, channels per slice of the entering state, bytes. The
/// largest token tile whose slice holds the whole head serves, else the
/// largest whose slice takes at least a warp's worth of channels (the whole
/// head when it is narrower), else the smallest tile that fits at all.
/// (0, 0, 0) only under a cap no card has.
pub fn inter_tiles_geom(
    (hd, ds, rank): (usize, usize, usize),
    cap: usize,
) -> (usize, usize, usize) {
    let per = |tt: usize| ds.min(128) + (tt * rank).next_multiple_of(4) + 4;
    // The widest slice at a token tile: with the partials in the slices'
    // room when that holds them, else with them apart. Rounding the state
    // slice up costs at most 3 floats.
    let widest = |tt: usize| -> Option<usize> {
        let room = (cap / 4).checked_sub(tt * rank * ds)?;
        let fits = |ps: usize| ps > 0 && inter_tiles_bytes((ds, rank), tt, ps) <= cap;
        let shared = (room.saturating_sub(3) / per(tt)).min(hd);
        let apart = (room.saturating_sub(tt * 256 + 3) / per(tt)).min(hd);
        [shared, apart].into_iter().find(|&ps| fits(ps))
    };
    let plans = || {
        [16usize, 8, 4, 2, 1]
            .into_iter()
            .filter(|&tt| tt * rank <= CHUNK)
            .filter_map(|tt| widest(tt).map(|ps| (tt, ps)))
    };
    plans()
        .find(|&(_, ps)| ps == hd)
        .or_else(|| plans().find(|&(_, ps)| ps >= hd.min(32)))
        .or_else(|| plans().next_back())
        .map_or((0, 0, 0), |(tt, ps)| {
            (tt, ps, inter_tiles_bytes((ds, rank), tt, ps))
        })
}

/// The scan's shared memory for `rows` state rows: the rows padded by one,
/// the keys starting on 16 bytes, then with `pipe` the pipelined body's
/// three key buffers (the previous token's, the current one's and the next
/// one's while it is built) and two of queries and of step scalars, else
/// the plain body's two key buffers and one of queries. The angles stay in
/// their threads' registers.
pub fn scan_smem_bytes(rows: usize, ds: usize, rank: usize, pipe: bool) -> usize {
    let keys = if pipe {
        5 * rank * ds + 8
    } else {
        3 * rank * ds
    };
    ((rows * (ds + 1)).next_multiple_of(4) + keys) * 4
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

/// The launch of `mimo_scan_bwd_global` for a head (see
/// [`scan_bwd_global_geom`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanBwdGlobalGeom {
    pub threads: usize,
    /// dS rows, the key gradients (5 R N) and the [P, R] tiles each in
    /// shared memory, else in a global scratch.
    pub ds_shared: bool,
    pub keys_shared: bool,
    pub tiles_shared: bool,
    /// The head is wider than the block: the block sums keep their leaves
    /// past the block in a global scratch of 4 lanes floats per sequence
    /// and head.
    pub wide: bool,
    pub smem: usize,
}

/// The launch of `mimo_scan_bwd_global` for a head of `hd` channels, `ds`
/// states, rank `rank` and `na` angles, its blocks at most `max_threads`
/// wide with `cap` bytes of shared memory. A head that fits the block gets
/// a thread per channel, at least one per (rank, state) chain up to a
/// thousand and at least 256 for the passes over the state. The groups go
/// into shared memory while they fit, dS first (four passes a token read
/// it), then the key gradients, then the tiles.
pub fn scan_bwd_global_geom(
    (hd, ds, rank, na): (usize, usize, usize, usize),
    max_threads: usize,
    cap: usize,
) -> ScanBwdGlobalGeom {
    let lanes = hd.next_multiple_of(32);
    let max_threads = (max_threads / 32 * 32).max(32);
    let wide = lanes > max_threads;
    let threads = if wide {
        1usize << max_threads.ilog2()
    } else {
        lanes
            .max((rank * ds).min(1024).next_multiple_of(32))
            .max(256)
            .min(max_threads)
    };
    let leaves = if wide { threads } else { lanes };
    let mut floats = 3 * na + 3 * leaves;
    let mut take = |n: usize| {
        let fits = (floats + n) * 4 <= cap;
        if fits {
            floats += n;
        }
        fits
    };
    let ds_shared = take((hd * (ds + 1)).next_multiple_of(4));
    let keys_shared = take(5 * rank * ds);
    let tiles_shared = take(3 * hd * rank);
    ScanBwdGlobalGeom {
        threads,
        ds_shared,
        keys_shared,
        tiles_shared,
        wide,
        smem: floats * 4,
    }
}

/// Tokens per chunk of the chunked path.
pub const CHUNK: usize = 64;

/// The channels a block of `m3m_chunk_scan_tiles` takes at most: a head up
/// to this wide in one block, a wider one in groups of it.
pub const SCAN_TILE_CHANNELS: usize = 128;

/// The launch of `m3m_chunk_scan_tiles` for `b` rows of `nc` chunks, `nh`
/// heads of `hd` channels at rank `rank`, a query tile of `tq` tokens and
/// `smem` bytes: the query tiles of each channel group on the grid's z
/// axis, 8 threads a query row (`tq` rank rows rounded up to 8).
pub fn chunk_scan_tiles_cfg(
    (b, nc, nh): (usize, usize, usize),
    (hd, rank): (usize, usize),
    tq: usize,
    smem: usize,
) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (
            (b * nc) as u32,
            nh as u32,
            (CHUNK.div_ceil(tq) * hd.div_ceil(SCAN_TILE_CHANNELS)) as u32,
        ),
        block_dim: ((8 * (tq * rank).next_multiple_of(8)) as u32, 1, 1),
        shared_mem_bytes: smem as u32,
    }
}

/// The tokens the chunked backward's angle walk stages at a time
/// (`M3MB_ANG_STEPS` in `mamba3_mimo_chunked_bwd.cu`).
pub const ANGLE_WALK_STEPS: usize = 16;

/// The angle walk's shared memory for `na` angles: the staged angle
/// gradients and tanh_pi values, the products with rows padded by one
/// float, and the tokens' dt and sequence starts.
pub fn angle_walk_smem_bytes(na: usize) -> usize {
    let steps = ANGLE_WALK_STEPS;
    (2 * steps * na + steps * (na + 1) + 2 * steps) * 4
}

/// One token's record in the backward's replay buffer, in floats.
pub fn scan_bwd_record_floats(hd: usize, ds: usize, rank: usize, na: usize) -> usize {
    hd * ds + 4 * rank * ds + 3 * hd * rank + hd + 2 * na + 8
}

/// The sequential scan's launch for a head (see [`scan_geom`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanGeom {
    /// Channels per block: the whole head, or a tile of it.
    pub tile: usize,
    pub threads: usize,
    pub smem: usize,
    /// The shared memory holds the pipelined body's keys, so a block with
    /// builders to spare may run it.
    pub pipe: bool,
}

impl ScanGeom {
    /// This launch with the heads in more tiles of channels when `groups`
    /// (sequence, head) pairs in the current tiles would leave most of the
    /// device's `sms` multiprocessors idle, as a decode step of a few
    /// sequences does: enough tiles of at least a warp of channels that the
    /// grid covers every multiprocessor once, the pipelined body where it
    /// fits `cap` bytes and `max_threads`. The channels share nothing but
    /// the keys, which every block builds alike, so a tile never changes a
    /// value.
    pub fn spread(
        self,
        (hd, ds, rank, na): (usize, usize, usize, usize),
        (groups, sms): (usize, usize),
        (cap, max_threads): (usize, usize),
    ) -> ScanGeom {
        let tiles = hd.div_ceil(self.tile.max(1));
        let want = sms.div_ceil(groups.max(1)).min(hd.div_ceil(32));
        if want <= tiles {
            return self;
        }
        let tile = hd.div_ceil(want);
        [true, false]
            .into_iter()
            .map(|pipe| ScanGeom {
                tile,
                threads: scan_block_threads(tile, na, pipe),
                smem: scan_smem_bytes(tile, ds, rank, pipe),
                pipe,
            })
            .find(|g| g.threads <= max_threads / 32 * 32 && g.smem <= cap)
            .unwrap_or(self)
    }

    /// The launch for `batch` sequences of `nh` heads of `hd` channels: a
    /// block per (sequence, head, tile).
    pub fn cfg(&self, batch: usize, nh: usize, hd: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (batch as u32, nh as u32, hd.div_ceil(self.tile) as u32),
            block_dim: (self.threads as u32, 1, 1),
            shared_mem_bytes: self.smem as u32,
        }
    }
}

/// Threads of a sequential scan block over `tile` channels and `na` angles:
/// the pipelined body needs a builder per angle (and at least a warp) past
/// the tile's channels, the plain one a thread per angle.
fn scan_block_threads(tile: usize, na: usize, pipe: bool) -> usize {
    if pipe {
        (tile.next_multiple_of(32) + na.max(32).next_multiple_of(32)).max(256)
    } else {
        tile.max(na).next_multiple_of(32)
    }
}

/// The sequential scan's geometry for a head of `hd` channels, `ds` states,
/// rank `rank` and `na` angles, within `cap` bytes of shared memory and
/// blocks of at most `max_threads`. The whole head takes one block, a whole
/// number of warps covering its channels and the angles and at least
/// eight, when that fits: a decode step then moves the head's state
/// through shared memory in a few vector loads per thread. Otherwise the
/// head runs in the fewest tiles of channels that fit, the pipelined body
/// preferred, the tiles evened out. The scan has no block-wide reduction
/// and its channels share nothing but the keys, which every block builds
/// alike, so no choice here changes a value. None when not even one
/// channel fits.
pub fn scan_geom(
    (hd, ds, rank, na): (usize, usize, usize, usize),
    cap: usize,
    max_threads: usize,
) -> Option<ScanGeom> {
    let max_threads = max_threads / 32 * 32;
    let whole = ScanGeom {
        tile: hd,
        threads: hd.max(na).next_multiple_of(32).max(256),
        smem: scan_smem_bytes(hd, ds, rank, true),
        pipe: true,
    };
    if hd == 0 || (whole.threads <= max_threads && whole.smem <= cap) {
        return Some(whole);
    }
    let threads = |pt: usize, pipe: bool| scan_block_threads(pt, na, pipe);
    let fits = |pt: usize, pipe: bool| {
        threads(pt, pipe) <= max_threads && scan_smem_bytes(pt, ds, rank, pipe) <= cap
    };
    let (tiles, pipe) = [true, false]
        .into_iter()
        .filter_map(|pipe| {
            (1..=hd.min(max_threads))
                .rev()
                .find(|&pt| fits(pt, pipe))
                .map(|pt| (hd.div_ceil(pt), pipe))
        })
        .min_by_key(|&(tiles, pipe)| (tiles, !pipe))?;
    let tile = hd.div_ceil(tiles);
    Some(ScanGeom {
        tile,
        threads: threads(tile, pipe),
        smem: scan_smem_bytes(tile, ds, rank, pipe),
        pipe,
    })
}

impl Mamba3MimoKernels {
    /// Compile (or load from the cache) the MIMO module for `arch` at
    /// `rank`: the kernels whose arrays the rank sizes run at it as a
    /// constant, so the module serves models of that rank.
    pub fn compile(
        ctx: &Arc<CudaContext>,
        arch: &'static str,
        rank: usize,
    ) -> Result<Self, String> {
        if !(1..=MAX_MIMO_RANK).contains(&rank) {
            return Err(format!("MIMO rank {rank} lies outside 1..={MAX_MIMO_RANK}"));
        }
        let rank_define = [format!("-DMIMO_RANK={rank}")];
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
                    include_str!("../../../kernels/_varlen.cuh"),
                    include_str!("../../../kernels/gated_mlp.cu"),
                    include_str!("../../../kernels/mamba3_mimo.cu"),
                    include_str!("../../../kernels/mamba3_mimo_chunked.cu"),
                    include_str!("../../../kernels/mamba3_mimo_bwd.cu"),
                    include_str!("../../../kernels/mamba3_mimo_chunked_bwd.cu"),
                ],
                kind: ki::ModuleKind::Mamba3MimoCombined,
                cache_prefix: "mamba3-mimo-kernels-v1-",
                seed: "2603155690",
                options: &rank_define,
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
        let scan = typed("mimo_scan")?;
        let scan_ckpt = typed("mimo_scan_ckpt")?;
        let scan_threads =
            max_block(&scan, "scan")?.min(max_block(&scan_ckpt, "checkpointing scan")?);
        let scan_bwd = typed("mimo_scan_bwd")?;
        let scan_bwd_global = typed("mimo_scan_bwd_global")?;
        let scan_bwd_block = max_block(&scan_bwd, "scan backward")?;
        let scan_bwd_global_block = max_block(&scan_bwd_global, "global scan backward")?;
        let chunk_state = typed("m3m_chunk_state")?;
        let chunk_scan = typed("m3m_chunk_scan")?;
        let dsloc = get("m3mb_dsloc")?;
        let lead = get("m3mb_lead")?;
        let chunk_state_tiles = typed("m3m_chunk_state_tiles")?;
        let dsloc_tiles = get("m3mb_dsloc_tiles")?;
        let chunk_threads = ChunkThreads {
            generic: max_block(&chunk_state, "chunk state")?
                .min(max_block(&chunk_scan, "chunk scan")?)
                .min(max_block_of(&[&dsloc, &lead], "chunked backward")?),
            state_tiles: max_block(&chunk_state_tiles, "chunk state tiles")?,
            dsloc_tiles: max_block_of(&[&dsloc_tiles], "local state gradient tiles")?,
        };
        let mut kernels = Self {
            bc_norm: typed("mimo_bc_norm")?,
            scan,
            chunk_prep: typed("m3m_prep")?,
            chunk_state,
            chunk_state_tiles,
            state_pass: get("m3m_state_pass")?,
            chunk_scan,
            chunk_scan_tiles: [
                typed("m3m_chunk_scan_tiles_jp1")?,
                typed("m3m_chunk_scan_tiles_jp2")?,
                typed("m3m_chunk_scan_tiles_jp4")?,
            ],
            chunk_writeback: typed("m3m_writeback")?,
            mlp_gate: typed("mlp_gate_fwd")?,
            gate_normed: typed("mimo_gate_normed")?,
            scan_ckpt,
            scan_threads,
            scan_bwd,
            scan_bwd_global,
            scan_bwd_block,
            scan_bwd_global_block,
            bc_norm_bwd: typed("mimo_bc_norm_bwd")?,
            angle_bwd: typed("mimo_angle_bwd")?,
            mlp_gate_bwd: typed("mlp_gate_bwd")?,
            sum_rows: get("mimo_sum_rows")?,
            cb: MimoChunkedBwd {
                gate: typed("m3mb_gate")?,
                gate_normed: typed("m3mb_gate_normed")?,
                inter_tiles: get("m3mb_inter_tiles")?,
                state_terms_tiles: typed("m3mb_state_terms_tiles")?,
                dsloc,
                dsloc_tiles,
                pass: get("m3mb_pass")?,
                chunk_dot: get("m3mb_chunk_dot")?,
                pairs_dq: typed("m3mb_pairs_dq")?,
                pairs_dk: typed("m3mb_pairs_dk")?,
                pairs_dq_narrow: typed("m3mb_pairs_dq_narrow")?,
                pairs_dk_narrow: typed("m3mb_pairs_dk_narrow")?,
                rot_bwd: get("m3mb_rot_bwd")?,
                angles: typed("m3mb_angles")?,
                combine: get("m3mb_combine")?,
                lead,
                scalars: typed("m3mb_scalars")?,
                u_bwd: typed("m3mb_u_bwd")?,
                sum_inner: get("m3mb_sum_inner")?,
                reduce_rows: get("m3mb_reduce_rows")?,
            },
            smem_optin,
            chunk_threads,
            compiler_identity,
            artifact_identity,
            _modules: CudaModuleAnchors::new(vec![module.clone()]),
        };
        kernels.smem_optin = {
            let optin = kernels.smem_optin;
            let mut budget = optin;
            for (k, name) in [
                (&kernels.scan, "scan"),
                (&kernels.scan_ckpt, "scan_ckpt"),
                (&kernels.scan_bwd, "scan_bwd"),
                (&kernels.scan_bwd_global, "scan_bwd_global"),
                (&kernels.chunk_state, "chunk_state"),
                (&kernels.chunk_state_tiles, "chunk_state_tiles"),
                (&kernels.chunk_scan, "chunk_scan"),
                (&kernels.chunk_scan_tiles[0], "chunk_scan_tiles"),
                (&kernels.chunk_scan_tiles[1], "chunk_scan_tiles"),
                (&kernels.chunk_scan_tiles[2], "chunk_scan_tiles"),
                (&kernels.cb.pairs_dq, "pairs dq"),
                (&kernels.cb.pairs_dk, "pairs dk"),
                (&kernels.cb.pairs_dq_narrow, "pairs dq narrow"),
                (&kernels.cb.pairs_dk_narrow, "pairs dk narrow"),
                (&kernels.cb.state_terms_tiles, "state_terms tiles"),
            ] {
                for f in [&k.f32, &k.bf16, &k.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            budget = budget.min(opt_in_dynamic_smem(&kernels.cb.dsloc, optin, "dsloc")?);
            // The chunk dot's 256-float tree sits in static shared memory, so
            // it holds the budget 1 KB under the opt-in, as the per-token
            // state terms kernel with the same tree did. The budget plans the
            // pair terms' reference tile, whose chains the split kernels
            // keep, so it must not move when a kernel leaves the module.
            budget = budget.min(opt_in_dynamic_smem(
                &kernels.cb.chunk_dot,
                optin,
                "chunk dot",
            )?);
            budget = budget.min(opt_in_dynamic_smem(
                &kernels.cb.inter_tiles,
                optin,
                "inter tiles",
            )?);
            smem_budget(budget.min(opt_in_dynamic_smem(
                &kernels.cb.dsloc_tiles,
                optin,
                "dsloc tiles",
            )?))
        };
        Ok(kernels)
    }

    /// The tiled chunk scan for a head of `hd` channels: the entry whose
    /// lanes take 1 channel of a group of [`SCAN_TILE_CHANNELS`] up to 32
    /// channels, 2 up to 64 and 4 past that.
    pub fn chunk_scan_tiles_for(&self, hd: usize) -> &TypedKernel {
        &self.chunk_scan_tiles[match hd {
            0..=32 => 0,
            33..=64 => 1,
            _ => 2,
        }]
    }

    /// The loaded module, for a graph to hold.
    pub(crate) fn module_anchors(&self) -> CudaModuleAnchors {
        self._modules.clone()
    }
}

/// The widest block every dtype of `k` launches, from its register use, a
/// whole number of warps.
fn max_block(k: &TypedKernel, what: &str) -> Result<usize, String> {
    max_block_of(&[&k.f32, &k.bf16, &k.f16], what)
}

/// The widest block every one of `fs` launches, a whole number of warps.
fn max_block_of(fs: &[&CudaFunction], what: &str) -> Result<usize, String> {
    use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
    let mut threads = 1024usize;
    for f in fs {
        let max = f
            .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|e| format!("threads per block of the MIMO {what}: {e:?}"))?;
        threads = threads.min(max.max(32) as usize);
    }
    Ok(threads / 32 * 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The angle walk's step count is the kernel's, and its shared memory
    /// at the most angles a model has (d_state 256 at rope fraction 1)
    /// stays under the 48 KB default window.
    #[test]
    fn the_angle_walk_stages_the_kernel_s_steps() {
        let source = include_str!("../../../kernels/mamba3_mimo_chunked_bwd.cu");
        assert!(source.contains("#define M3MB_ANG_STEPS 16"));
        assert_eq!(ANGLE_WALK_STEPS, 16);
        assert_eq!(angle_walk_smem_bytes(128), 24_768);
        assert!(angle_walk_smem_bytes(128) <= 48 * 1024);
    }

    /// The released head keeps the full tiles, a wide rank shrinks the
    /// scan's key tile, and every chosen tile fits the limit. A head wider
    /// than a block runs in tiles of channels, and a head whose rows the
    /// released pair kernel kept in global memory takes the split kernels in
    /// slices.
    #[test]
    fn chunked_geometry_fits_the_shared_memory() {
        let all = ChunkThreads::uniform(1024);
        let cap = 99 * 1024;
        let released = chunked_geom((64, 128, 4), cap, true, all).unwrap();
        assert_eq!((released.state.0, released.state.1), (4, CHUNK));
        assert_eq!(
            (
                released.state_tiles.0,
                released.state_tiles.1,
                released.state_tiles.2
            ),
            (4, 16, 64)
        );
        let tiles = released.scan_tiles;
        assert_eq!((tiles.0, tiles.1, tiles.2), (16, 16, 64));
        assert_eq!(
            released.pairs,
            PairsPlan {
                tq: 8,
                ps: 64,
                dq_bytes: 38_784,
                kns: 128,
                pss: 64,
                dk_bytes: 62_208,
                narrow: pairs_narrow(64, 128, 4),
            }
        );
        assert_eq!(released.inter_tiles.0, 16);
        assert_eq!(
            (
                released.terms.tt,
                released.terms.whole,
                released.terms.bytes
            ),
            (8, true, terms_whole_bytes((64, 128, 4), 8))
        );
        assert_eq!((released.scan.0, released.scan.1), (4, 8));
        assert_eq!((released.pt, released.pairs_ref), (64, 8));
        let wide = chunked_geom((16, 16, 16), cap, true, all).unwrap();
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
            (2048, 16, 1),
            (1280, 256, 16),
        ] {
            for cap in [48 * 1024, 64 * 1024, 99 * 1024, 227 * 1024] {
                let Some(g) = chunked_geom(shape, cap, true, all) else {
                    continue;
                };
                let sizes = [
                    g.state.2,
                    g.state_tiles.3,
                    g.dsloc_tiles.3,
                    g.scan.2,
                    g.scan_tiles.3,
                    g.dsloc.1,
                    g.pairs.dq_bytes,
                    g.inter_tiles.2,
                    g.terms.bytes,
                    g.pairs.dk_bytes,
                ];
                for bytes in sizes {
                    assert!(bytes <= cap, "{shape:?} at {cap}: {g:?}");
                }
                assert!(g.state.0 * g.pt <= 1024 && g.scan.0 * g.pt <= 1024);
                assert_eq!(g.pt, shape.0.min(1024), "{shape:?}: {g:?}");
                // The register tiles take any head, in channel groups past
                // what one block holds.
                assert!(
                    g.state_tiles.0 > 0 && g.dsloc_tiles.0 > 0,
                    "{shape:?}: {g:?}"
                );
            }
        }
        let tiled = chunked_geom((2048, 16, 1), 227 * 1024, true, all).unwrap();
        assert_eq!((tiled.pt, tiled.state.0, tiled.scan.0), (1024, 1, 1));
        // A head whose rows the released pair kernel kept in global memory:
        // its tile is the one whose three blocks fit, and the split kernels
        // take the head and the state in slices.
        let sliced = chunked_geom((1024, 256, 16), 99 * 1024, true, all).unwrap();
        assert_eq!(sliced.pairs_ref, 4, "{sliced:?}");
        assert_eq!(
            sliced.pairs,
            PairsPlan {
                tq: 2,
                ps: 256,
                dq_bytes: 90_768,
                kns: 128,
                pss: 128,
                dk_bytes: 79_136,
                narrow: pairs_narrow(1024, 256, 16),
            }
        );
    }

    /// The pair kernels plan every shape the config admits, at every card's
    /// limit and in both builds: a power-of-two tile of at most 16 tokens
    /// whose block fits the build (64 rows wide, 256 threads narrow), slices
    /// within their widths, both kernels within the cap.
    #[test]
    fn the_pairs_plan_takes_any_head_and_state_in_slices() {
        for (hd, ds, rank) in [
            (64, 128, 4),
            (256, 128, 2),
            (2048, 256, 16),
            (4096, 256, 1),
            (136, 136, 9),
            (40, 16, 3),
            (1, 1, 1),
            (512, 256, 16),
        ] {
            for (cap, narrow) in [48 * 1024, 64 * 1024, 99 * 1024, 227 * 1024]
                .into_iter()
                .flat_map(|cap| [(cap, false), (cap, true)])
            {
                let p = pairs_plan((hd, ds, rank), cap, narrow)
                    .unwrap_or_else(|| panic!("head {hd} state {ds} rank {rank} at {cap}"));
                let fits = match narrow {
                    true => pairs_threads(p.tq, rank) <= PAIRS_THREADS_NARROW,
                    false => p.tq * rank <= CHUNK,
                };
                assert!(
                    p.tq.is_power_of_two()
                        && p.tq <= 16
                        && fits
                        && p.narrow == narrow
                        && (1..=hd).contains(&p.ps)
                        && (1..=ds).contains(&p.kns)
                        && (1..=hd).contains(&p.pss)
                        && p.dq_bytes <= cap
                        && p.dk_bytes <= cap,
                    "head {hd} state {ds} rank {rank} at {cap}: {p:?}"
                );
            }
        }
        assert_eq!(
            pairs_plan((1, 1, 1), 99 * 1024, false),
            Some(PairsPlan {
                tq: 16,
                ps: 1,
                dq_bytes: 2_576,
                kns: 1,
                pss: 1,
                dk_bytes: 5_024,
                narrow: false,
            })
        );
    }

    /// The pair kernels take the narrow builds where the wide builds' spills
    /// cost more than half the warps, and the wide builds everywhere else.
    #[test]
    fn the_pair_kernels_take_the_narrow_build_where_the_spills_cost_more() {
        if std::env::var("MAMBA_RS_PAIRS_BUILD").is_ok() {
            return;
        }
        for (hd, ds, rank) in [
            (128, 64, 1),
            (256, 16, 1),
            (64, 256, 1),
            (64, 128, 2),
            (96, 64, 4),
            (1024, 256, 16),
        ] {
            assert!(
                pairs_narrow(hd, ds, rank),
                "head {hd} state {ds} rank {rank}"
            );
        }
        for (hd, ds, rank) in [
            (32, 256, 4),
            (32, 64, 16),
            (64, 64, 16),
            (64, 128, 1),
            (96, 64, 2),
            (96, 128, 1),
            (16, 16, 16),
        ] {
            assert!(
                !pairs_narrow(hd, ds, rank),
                "head {hd} state {ds} rank {rank}"
            );
        }
    }

    /// The state terms plan every shape the config admits, at every card's
    /// limit: a power-of-two tile of at most 16 tokens and 64 rows within
    /// the cap; dS, u and the keys whole where they fit, which keeps the
    /// released shapes' tiles, else in slices within their widths, and ste
    /// in the global scratch only when a token's ste rows cannot sit beside
    /// the slices.
    #[test]
    fn the_terms_plan_takes_any_head_and_state_in_slices() {
        for (hd, ds, rank) in [
            (64, 128, 4),
            (256, 128, 2),
            (1024, 256, 16),
            (136, 256, 9),
            (4096, 256, 1),
            (2048, 256, 16),
            (512, 64, 1),
            (32, 64, 16),
            (40, 16, 3),
            (1, 1, 1),
        ] {
            for cap in [48 * 1024, 64 * 1024, 99 * 1024, 227 * 1024] {
                let p = terms_plan((hd, ds, rank), cap)
                    .unwrap_or_else(|| panic!("head {hd} state {ds} rank {rank} at {cap}"));
                let bytes = if p.whole {
                    terms_whole_bytes((hd, ds, rank), p.tt)
                } else {
                    terms_sliced_bytes((hd, ds, rank), p.tt, (p.ps, p.ns), p.steb_global)
                };
                assert!(
                    p.tt.is_power_of_two()
                        && p.tt <= 16
                        && p.tt * rank <= CHUNK
                        && (1..=hd).contains(&p.ps)
                        && (1..=ds).contains(&p.ns)
                        && p.bytes == bytes
                        && bytes <= cap,
                    "head {hd} state {ds} rank {rank} at {cap}: {p:?}"
                );
            }
        }
        let plan = |dims, cap| terms_plan(dims, cap).unwrap();
        let released = plan((64, 128, 4), 99 * 1024);
        assert_eq!((released.tt, released.whole), (8, true));
        assert_eq!(
            plan((1024, 256, 16), 99 * 1024),
            TermsPlan {
                tt: 1,
                whole: false,
                ps: 32,
                ns: 32,
                steb_global: false,
                bytes: 84_992,
            }
        );
        assert_eq!(
            plan((2048, 256, 16), 99 * 1024),
            TermsPlan {
                tt: 4,
                whole: false,
                ps: 128,
                ns: 128,
                steb_global: true,
                bytes: 100_864,
            }
        );
    }

    /// The inter kernel plans every shape: the entering state whole where it
    /// fits beside the largest token tile, else in slices of channels, within
    /// the cap at every card's limit.
    #[test]
    fn the_inter_plan_takes_any_state_in_slices_of_channels() {
        for (hd, ds, rank) in [
            (64, 128, 4),
            (256, 128, 2),
            (2048, 256, 16),
            (4096, 256, 1),
            (136, 136, 1),
            (40, 16, 3),
            (1, 1, 1),
        ] {
            for cap in [48 * 1024, 64 * 1024, 99 * 1024, 227 * 1024] {
                let (tt, ps, bytes) = inter_tiles_geom((hd, ds, rank), cap);
                assert!(
                    tt > 0 && tt * rank <= CHUNK && (1..=hd).contains(&ps) && bytes <= cap,
                    "head {hd} state {ds} rank {rank} at {cap}: {:?}",
                    (tt, ps, bytes)
                );
            }
        }
        assert_eq!(inter_tiles_geom((64, 128, 4), 99 * 1024), (16, 64, 82_944));
        assert_eq!(inter_tiles_geom((1, 1, 1), 99 * 1024), (16, 1, 16_544));
        assert_eq!(
            inter_tiles_geom((256, 128, 2), 99 * 1024),
            (16, 129, 101_008)
        );
        assert_eq!(inter_tiles_geom((2048, 256, 16), 48 * 1024).0, 1);
    }

    /// Register tiles under a kernel that holds fewer threads than the whole
    /// head needs take the head in channel groups as even as the limit
    /// allows, each within the limit and together covering the head.
    #[test]
    fn register_tiles_take_the_head_in_groups_under_their_register_limit() {
        let cap = 99 * 1024;
        for (hd, state, dsloc) in [(128, 416, 384), (64, 416, 384), (72, 256, 64), (40, 32, 32)] {
            let threads = ChunkThreads {
                generic: 1024,
                state_tiles: state,
                dsloc_tiles: dsloc,
            };
            let g = chunked_geom((hd, 64, 4), cap, true, threads).unwrap();
            for (tiles, limit) in [(g.state_tiles, state), (g.dsloc_tiles, dsloc)] {
                let pg = tiles.2;
                assert!(pg > 0 && pg % 4 == 0, "{hd}: {tiles:?}");
                assert!(8 * pg <= limit, "{hd} under {limit}: {tiles:?}");
                let groups = hd.div_ceil(pg);
                assert!(
                    groups * pg >= hd && (groups - 1) * pg < hd,
                    "{hd}: {tiles:?}"
                );
                assert!(tiles.3 <= cap);
            }
        }
        // The whole head when it fits: one group of every channel.
        let g = chunked_geom((72, 24, 4), cap, true, ChunkThreads::uniform(1024)).unwrap();
        assert_eq!((g.state_tiles.2, g.dsloc_tiles.2), (72, 72));
    }
    /// No validated head is refused for shared memory on a supported part:
    /// within the 99 KB every part since Ampere lets a block opt in to, or
    /// the least budget the forced-route knob takes, and fewer threads than
    /// any kernel leaves a block, the recurrence fits a tile of one channel
    /// and the chunked kernels a tile of every shape.
    #[test]
    fn every_validated_head_fits_a_supported_part() {
        for cap in [crate::gpu::launch::SMEM_BUDGET_FLOOR, 99 * 1024] {
            every_validated_head_fits(cap);
        }
    }

    /// A grid of few (sequence, head) pairs spreads its heads over tiles of
    /// at least a warp of channels until it covers the multiprocessors, and
    /// stays as it is when it already does or a head is one warp wide.
    #[test]
    fn a_small_scan_grid_spreads_its_heads() {
        let cap = 99 * 1024;
        let shape = (64usize, 64usize, 4usize, 16usize);
        let whole = scan_geom(shape, cap, 1024).unwrap();
        assert_eq!(whole.tile, 64);
        let spread = whole.spread(shape, (4, 142), (cap, 1024));
        assert_eq!(spread.tile, 32);
        assert!(spread.threads <= 1024 && spread.smem <= cap);
        assert_eq!(whole.spread(shape, (142, 142), (cap, 1024)), whole);
        let narrow = (32usize, 64usize, 4usize, 16usize);
        let w = scan_geom(narrow, cap, 1024).unwrap();
        assert_eq!(w.spread(narrow, (1, 142), (cap, 1024)), w);
        let wide = (512usize, 128usize, 2usize, 0usize);
        let g = scan_geom(wide, cap, 1024).unwrap();
        let s = g.spread(wide, (2, 170), (cap, 1024));
        assert!(wide.0.div_ceil(s.tile) * 2 >= 16, "tile {}", s.tile);
        assert!(s.tile >= 32 && s.smem <= cap && s.threads <= 1024);
    }

    fn every_validated_head_fits(cap: usize) {
        use crate::mamba3_mimo::config::{MAX_D_STATE, MAX_MIMO_RANK};
        for rank in 1..=MAX_MIMO_RANK {
            for ds in 1..=MAX_D_STATE {
                for hd in [1, 2, 31, 32, 33, 64, 128, 129, 1024, 4096] {
                    for na in [0, ds / 4, ds / 2] {
                        let scan = scan_geom((hd, ds, rank, na), cap, 128);
                        assert!(scan.is_some(), "scan hd {hd} ds {ds} rank {rank} na {na}");
                    }
                    for training in [false, true] {
                        let geom =
                            chunked_geom((hd, ds, rank), cap, training, ChunkThreads::uniform(32));
                        assert!(geom.is_some(), "chunked hd {hd} ds {ds} rank {rank}");
                    }
                }
            }
        }
    }
}
