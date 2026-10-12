//! Kernel launch helpers — grid/block calculation.
//!
//! Standard block size: 256 threads for 1D element-wise kernels.
//! SSM/conv1d kernels use batch*d_inner threads.
//! Norm kernels use batch blocks × dim threads (2D).

use cudarc::driver::LaunchConfig;

/// Standard block size for 1D element-wise kernels.
const BLOCK_1D: u32 = 256;

/// The largest value [`check_int_args`] lets a kernel take as an `int`:
/// `i32::MAX` less the headroom kernels step past their bounds with. A loop
/// counter strided by the block ends up to a block past its bound, and a
/// time row is padded by a few elements, so a value right under `i32::MAX`
/// would wrap them.
const INT_ARG_MAX: usize = i32::MAX as usize - (1 << 16);

/// Refuse a launch whose 32-bit kernel arguments would wrap. Each named value
/// is a dimension, a row count or a row width a kernel takes as an `int`;
/// element counts and offsets are 64-bit in the kernels that check with
/// this, so the size of a tensor is bounded by memory alone.
pub fn check_int_args(args: &[(&str, usize)]) -> Result<(), String> {
    match args.iter().find(|(_, v)| *v > INT_ARG_MAX) {
        Some((name, v)) => Err(format!(
            "{name} {v} is past the 32-bit range the kernels take it in"
        )),
        None => Ok(()),
    }
}

/// The integer type a kernel built twice forms its offsets in. The parallel
/// scan forwards exist as an `int` build and an `_i64` build of one body:
/// 64-bit offsets cost them enough registers on sm_89 to lose a resident
/// block per SM, so a launch whose offsets all fit takes the `int` build.
/// Both builds give the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexWidth {
    I32,
    I64,
}

impl IndexWidth {
    /// `I32` while no element count in `extents` passes the range
    /// [`check_int_args`] lets a kernel take as an `int`, `I64` past it.
    pub fn covering(extents: &[usize]) -> Self {
        if extents.iter().all(|&n| n <= INT_ARG_MAX) {
            Self::I32
        } else {
            Self::I64
        }
    }
}

/// Let `f` stage as much dynamic shared memory as the device's per-block
/// `optin` leaves beside its own static stage, and return that amount.
pub(crate) fn opt_in_dynamic_smem(
    f: &cudarc::driver::CudaFunction,
    optin: usize,
    name: &str,
) -> Result<usize, String> {
    use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
    let fixed = f
        .get_attribute(FnAttr::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
        .map_err(|e| format!("static shared memory of {name}: {e:?}"))?
        .max(0) as usize;
    let dynamic = optin.saturating_sub(fixed);
    f.set_attribute(
        FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        dynamic as i32,
    )
    .map_err(|e| format!("shared-memory opt-in for {name}: {e:?}"))?;
    Ok(dynamic)
}

/// Launch config for 1D element-wise kernel on `n` elements.
///
/// Grid: `ceil(n / 256)` blocks of 256 threads.
/// Suitable for: activations, vec_add, elementwise_mul, gating, etc.
pub fn grid_1d(n: usize) -> LaunchConfig {
    // `n as u32` would silently truncate for n >= 2^32 and launch a grid
    // covering a fraction of the elements, and the kernels this geometry
    // serves index with 32 bits; this is the cheap backstop.
    assert!(
        n <= i32::MAX as usize,
        "grid_1d: element count {n} exceeds i32::MAX"
    );
    let num_blocks = (n as u32).div_ceil(BLOCK_1D);
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The block count [`grid_1d`] reaches at `i32::MAX` elements: the largest
/// grid a grid-stride element kernel is launched with.
const GRID_1D_WIDE_MAX_BLOCKS: usize = (i32::MAX as usize).div_ceil(BLOCK_1D as usize);

/// Launch config for grid-stride element kernels that take a 64-bit count.
///
/// Up to `i32::MAX` elements this is [`grid_1d`]'s geometry, so a kernel
/// moved to a 64-bit count keeps its element-to-thread mapping and its bits.
/// Past that the grid stops growing and every thread strides over the rest.
pub fn grid_1d_wide(n: usize) -> LaunchConfig {
    let blocks = n
        .div_ceil(BLOCK_1D as usize)
        .clamp(1, GRID_1D_WIDE_MAX_BLOCKS);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch config of one thread per element for kernels that index with 64
/// bits and return past their count: [`grid_1d`]'s geometry, so every
/// element keeps its thread, without its 32-bit ceiling on the count.
pub fn grid_1d_cover(n: usize) -> Result<LaunchConfig, String> {
    Ok(LaunchConfig {
        grid_dim: (blocks_1d(n)?, 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Blocks of [`BLOCK_1D`] threads covering `n` threads, within the grid's
/// x extent of 2^31 - 1 blocks.
pub(crate) fn blocks_1d(n: usize) -> Result<u32, String> {
    let blocks = n.div_ceil(BLOCK_1D as usize).max(1);
    u32::try_from(blocks)
        .ok()
        .filter(|&b| b <= i32::MAX as u32)
        .ok_or_else(|| format!("{n} threads need {blocks} blocks, past the grid's x extent"))
}

/// Tiles of `tile` steps covering `t` steps on the grid's y axis, whose
/// extent is 65535.
pub(crate) fn tiles_y(t: usize, tile: usize) -> Result<u32, String> {
    let tiles = t.div_ceil(tile);
    u32::try_from(tiles)
        .ok()
        .filter(|&n| n <= 65535)
        .ok_or_else(|| format!("{t} steps need {tiles} tiles of {tile}, past the grid's y extent"))
}

/// Launch a one-dimensional element kernel whose last argument is its
/// element count as a 64-bit integer: the pointer arguments in order, then
/// the count, over [`grid_1d_wide`]. Those kernels stride over their count,
/// so a window past `i32::MAX` elements runs whole, and below it every
/// element keeps the thread [`grid_1d`] gives it.
pub(crate) fn launch_elems(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    ptrs: &[u64],
    n: usize,
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    use cudarc::driver::PushKernelArg;
    if n == 0 {
        return Ok(());
    }
    let count = i64::try_from(n).map_err(|_| format!("{what}: {n} elements exceed i64"))?;
    let mut bld = stream.launch_builder(f);
    for p in ptrs {
        bld.arg(p);
    }
    bld.arg(&count);
    unsafe { bld.launch(grid_1d_wide(n)) }.map_err(|e| format!("{what}: {e:?}"))?;
    Ok(())
}

/// Launch `fill_scalar`: `n` floats from `dst` on set to `val`.
pub(crate) fn launch_fill(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    (dst, val): (u64, f32),
    n: usize,
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    use cudarc::driver::PushKernelArg;
    if n == 0 {
        return Ok(());
    }
    let count = i64::try_from(n).map_err(|_| format!("{what}: {n} elements exceed i64"))?;
    let mut bld = stream.launch_builder(f);
    bld.arg(&dst);
    bld.arg(&val);
    bld.arg(&count);
    unsafe { bld.launch(grid_1d_wide(n)) }.map_err(|e| format!("{what}: {e:?}"))?;
    Ok(())
}

/// The most elements one launch of a 32-bit counted element kernel takes:
/// whole blocks within `i32::MAX`.
const I32_PIECE: usize = (i32::MAX as usize / BLOCK_1D as usize) * BLOCK_1D as usize;

/// Launch a conversion kernel `(dst, src, int n)` over `n` elements, each
/// pointer with the size of its element in bytes. These are the casts of the
/// Inference module, whose source is pinned with its 32-bit count, so past
/// `i32::MAX` elements they run in pieces the kernel can count, each piece
/// moving both pointers past the elements before it. Every element is
/// converted alone, so the pieces write what one launch would, and below
/// `i32::MAX` this is the one launch of [`grid_1d`].
pub fn launch_cast(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    dst: (u64, usize),
    src: (u64, usize),
    n: usize,
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    launch_cast_pieces(stream, f, (dst, src), (n, I32_PIECE), what)
}

/// [`launch_cast`] in pieces of at most `piece` elements.
fn launch_cast_pieces(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    f: &cudarc::driver::CudaFunction,
    ((dst, dst_bytes), (src, src_bytes)): ((u64, usize), (u64, usize)),
    (n, piece): (usize, usize),
    what: std::fmt::Arguments<'_>,
) -> Result<(), String> {
    use cudarc::driver::PushKernelArg;
    let mut done = 0;
    while done < n {
        let len = (n - done).min(piece);
        let (d, s) = (
            dst + (done * dst_bytes) as u64,
            src + (done * src_bytes) as u64,
        );
        let count = len as i32;
        let mut bld = stream.launch_builder(f);
        bld.arg(&d);
        bld.arg(&s);
        bld.arg(&count);
        unsafe { bld.launch(grid_1d(len)) }.map_err(|e| format!("{what}: {e:?}"))?;
        done += len;
    }
    Ok(())
}

/// Launch config for SSM/conv1d kernels: `batch * d_inner` threads.
///
/// Each thread handles one (b, d) pair, sequential across T/d_state.
pub fn grid_ssm(batch: usize, d_inner: usize) -> LaunchConfig {
    grid_1d(batch * d_inner)
}

/// Launch config for reduction kernels (d_B, d_C, d_D, d_a_log).
pub fn grid_reduce(total_elements: usize) -> LaunchConfig {
    grid_1d(total_elements)
}

/// Launch config for `colsum_accumulate`: one block of COLSUM_THREADS per
/// COLSUM_COLS columns (mirrors the defines in elementwise.cu).
pub fn grid_colsum(n_out: usize) -> LaunchConfig {
    const COLSUM_COLS: usize = 8;
    const COLSUM_THREADS: u32 = 256;
    LaunchConfig {
        grid_dim: (n_out.div_ceil(COLSUM_COLS) as u32, 1, 1),
        block_dim: (COLSUM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch config for the deterministic column tree-reduce used by the
/// batch-invariant SGEMM bias/colsum path.
///
/// Grid: `n_cols` blocks. Block: 256 threads. Shared memory: 256 * sizeof(f32).
/// Each block reduces one column via strided loop + smem tree + warp shuffle.
pub fn grid_col_tree_reduce(n_cols: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n_cols as u32, 1, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: (BLOCK_1D as usize * std::mem::size_of::<f32>()) as u32,
    }
}

/// Launch config for norm kernels (L2Norm, RMSNorm).
///
/// Grid: `batch` blocks. Block: the power of two at or above `dim`, at least
/// one warp and at most 1024 threads.
/// Shared memory: one f32 per thread for the reduction workspace.
pub fn grid_norm(batch: usize, dim: usize) -> LaunchConfig {
    // The reduction ends in a full-mask warp shuffle, so the block holds at
    // least one whole warp: below 32 the shuffle would read lanes that do
    // not exist. The extra lanes hold +0.0, the identity of the sum.
    let block = (dim as u32).clamp(32, 1024).next_power_of_two();
    LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (block as usize * std::mem::size_of::<f32>()) as u32,
    }
}

/// Launch config for the parallel prefix scan forward kernels (f32 and
/// typed alike: every shared-memory cell is f32).
///
/// Grid: `(batch, d_inner)` — one block per (b, d) pair.
/// Block: 128 threads (matches NTHREADS in mamba_ssm_parallel.cu).
/// Shared memory: the two block-scan rows, the two running-prefix rows at
/// the actual d_state, and one exchange slot per warp for the prefix
/// hand-off; the kernels load and store directly, so there is no staging
/// region. Layout (floats): 2*NWARPS + 2*d_state + 2*NWARPS.
/// Parallel-scan launch geometry — MUST mirror NTHREADS/NITEMS in
/// mamba_ssm_parallel.cu (block size, warp count, chunk length and the
/// slim-tape chunk count all derive from these two numbers).
pub const SCAN_NTHREADS: usize = 128;
pub const SCAN_NITEMS: usize = 8;
/// Timesteps per scan chunk.
pub const SCAN_CHUNK: usize = SCAN_NTHREADS * SCAN_NITEMS;

/// Slim h-tape switch: `MAMBA_RS_SCAN_TAPE=full` restores the
/// (T+1)-step `h_saved` tape on the parallel route (escape hatch for one
/// release); the default `slim` keeps only per-chunk
/// (run_a, run_b, h_entry) rows and the backward replays h bit-exactly
/// in-kernel (same thread-local scan, same block scan, same compose
/// chain on the same inputs).
/// NOTE: OnceLock-memoized - an in-process A/B measures one mode twice;
/// verifying both tape modes takes two separate process launches.
pub fn scan_tape_slim() -> bool {
    static SLIM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SLIM.get_or_init(|| match std::env::var("MAMBA_RS_SCAN_TAPE") {
        Err(_) => true,
        // Strict: a typo ("ful", "Full ") must fail loudly, not silently
        // select the slim default and report green.
        Ok(v) => match v.trim() {
            "full" => false,
            "slim" | "" => true,
            other => {
                panic!("MAMBA_RS_SCAN_TAPE={other:?} is not a recognized mode (use full or slim)")
            }
        },
    })
}

/// Slim-tape length in floats: 3 entries per (b, d, n, chunk). The chunk
/// count mirrors CHUNK_SIZE = NTHREADS * NITEMS = 1024 in
/// mamba_ssm_parallel.cu.
pub fn scan_tape_len(batch: usize, seq_len: usize, d_inner: usize, d_state: usize) -> usize {
    let n_chunks = seq_len.div_ceil(SCAN_CHUNK);
    batch * d_inner * d_state * 3 * n_chunks
}

pub fn grid_parallel_scan(batch: usize, d_inner: usize, d_state: usize) -> LaunchConfig {
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    assert!(
        d_state <= 256,
        "grid_parallel_scan: d_state {d_state} exceeds the kernel MAX_DSTATE guard - \
         the kernel would return without writing y"
    );
    const NWARPS: usize = SCAN_NTHREADS / 32;
    let smem_floats = 2 * NWARPS + 2 * d_state + 2 * NWARPS;
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: (smem_floats * std::mem::size_of::<f32>()) as u32,
    }
}

/// Offset width of a parallel scan forward that saves for the backward. Its
/// offsets reach the state and A rows `[batch, d_inner, d_state]`, the B and
/// C rows `[batch, d_state, T]`, the activations `[batch, T, d_inner]` and
/// the h tape: the chunk rows of the slim tape or every step of the full one.
pub fn scan_fwd_index_width(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
    slim_tape: bool,
) -> IndexWidth {
    let state = batch.saturating_mul(d_inner).saturating_mul(d_state);
    let tape = if slim_tape {
        state
            .saturating_mul(3)
            .saturating_mul(seq_len.div_ceil(SCAN_CHUNK))
    } else {
        state.saturating_mul(seq_len.saturating_add(1))
    };
    IndexWidth::covering(&[
        state,
        batch.saturating_mul(d_state).saturating_mul(seq_len),
        batch.saturating_mul(seq_len).saturating_mul(d_inner),
        tape,
    ])
}

/// Offset width of a parallel scan forward without saves. Its offsets reach
/// the state and A rows, the B and C rows, the activations while they are
/// row-major (t-major rows take a 64-bit base in both builds) and, when the
/// gate fuses into a row-major store, the in_proj rows of `gate_stride`
/// columns the gate is read from.
pub fn scan_fwd_nosave_index_width(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
    gate_stride: usize,
    tmajor: bool,
) -> IndexWidth {
    let rows = if tmajor {
        0
    } else {
        batch.saturating_mul(seq_len).saturating_mul(d_inner)
    };
    let gate = if tmajor || gate_stride == 0 {
        0
    } else {
        batch
            .saturating_mul(seq_len)
            .saturating_mul(gate_stride)
            .saturating_add(d_inner.saturating_mul(2))
    };
    IndexWidth::covering(&[
        batch.saturating_mul(d_inner).saturating_mul(d_state),
        batch.saturating_mul(d_state).saturating_mul(seq_len),
        rows,
        gate,
    ])
}

/// Launch config for the M1 parallel scan BACKWARD kernel.
///
/// Smem layout = forward layout + reverse-scan workspace + postfix carry +
/// next-thread δA exchange + d_a_log block-reduce + chunk-first-a boundary.
/// Total = SMEM_BWD_FLOATS = 2832 floats = 11 328 bytes (well under 48 KB).
///
/// Unlike the forward (`grid_parallel_scan_typed`), the allocation does NOT
/// shrink for bf16/f16: the bwd-extra regions live at fixed f32 offsets
/// AFTER the full `CHUNK_SIZE * sizeof(f32)` stage region (SMEM_REV_WA_OFF
/// = SMEM_TOTAL_FLOATS in mamba_ssm_parallel.cu). The typed kernels merely
/// reinterpret the stage slots as T_ACT in place, so the byte size must
/// always be the f32 layout size.
/// d-group size of the fold backward — MUST mirror SCAN_BWD_DGROUP in
/// mamba_ssm_parallel.cu.
pub const SCAN_BWD_DGROUP: usize = 4;

/// Launch config for the d-group fold backward: one block per
/// (batch, d-group of SCAN_BWD_DGROUP lanes). Smem = the f32 workspace
/// (warp scans, exchanges, per-group postfix/carry lanes, reduce and
/// boundary tiles) plus the typed delta/u/dy stage.
pub fn grid_parallel_scan_bwd_fold(
    batch: usize,
    d_inner: usize,
    d_state: usize,
    bytes_per_act: usize,
) -> LaunchConfig {
    const NWARPS: usize = SCAN_NTHREADS / 32;
    let g = SCAN_BWD_DGROUP;
    // Slot stride is the RUNTIME d_state (the kernel guard still bounds
    // it by its compile-time capacity). Two warp-scan workspaces, four
    // per-warp hand-off rows (reverse postfix, replay prefix), the
    // per-group postfix/carry/boundary lanes, the next-decay and
    // replayed-state hand-off rows, and one d_a reduce tile per group.
    let f32_floats =
        4 * NWARPS + 4 * NWARPS + 3 * g * d_state + NWARPS + g * SCAN_NTHREADS + NWARPS;
    // The f32 kernel stages three of its four lanes and reads the fourth
    // from global memory, so two blocks fit the SM's shared memory; the
    // half kernels stage all four. Mirrors the STAGED argument of
    // DEFINE_SSM_PARALLEL_SCAN_BWD_FOLD in mamba_ssm_parallel.cu.
    let staged = if bytes_per_act == 4 { g - 1 } else { g };
    let stage_bytes = (3 * staged + 1) * SCAN_CHUNK * bytes_per_act;
    LaunchConfig {
        grid_dim: (batch as u32, (d_inner / g) as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: (f32_floats * std::mem::size_of::<f32>() + stage_bytes) as u32,
    }
}

pub fn grid_parallel_scan_bwd(batch: usize, d_inner: usize) -> LaunchConfig {
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan_bwd: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    const NWARPS: usize = SCAN_NTHREADS / 32;
    const MAX_DSTATE: usize = 256;
    // SMEM_TOTAL_FLOATS (fwd layout incl. f32-sized stage) + bwd-extra
    // (reverse warp scan + postfix + next-A exchange + da-reduce +
    // chunk-first-A boundary). Must match SMEM_BWD_FLOATS in the kernel.
    let fwd_total_floats = 2 * NWARPS + 2 * MAX_DSTATE + 2 * SCAN_NTHREADS + SCAN_CHUNK;
    let bwd_extra_floats = 2 * NWARPS + 3 * MAX_DSTATE + 2 * SCAN_NTHREADS;
    let total_bytes = (fwd_total_floats + bwd_extra_floats) * std::mem::size_of::<f32>();
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: total_bytes as u32,
    }
}

/// Tile depth of `gather_bc_cols_tmajor_tiled` (must match GBC_TILE_T in
/// mamba_ssm_elementwise.cu). One block per (t-tile, b); dynamic smem
/// carries the B and C tiles at `[d_state][GBC_TILE_T + 1]` each (+1 pad
/// kills bank conflicts on the transposed read).
pub const GBC_TILE_T: usize = 32;

/// Grid for the staged t-major B/C gather.
pub fn grid_gather_bc_tiled(
    batch: usize,
    t: usize,
    d_state: usize,
    elem_bytes: usize,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((t as u32).div_ceil(GBC_TILE_T as u32), batch as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: (2 * d_state * (GBC_TILE_T + 1) * elem_bytes) as u32,
    }
}

/// Grid of the tiled conv forward kernels, the saving and the inference
/// one: x covers (b * d_inner) threads at 256/block, y covers T in tiles of
/// CONV1D_FWD_TILE_T = 32 (kernels/conv1d.cu). The serial per-(b,d) walk
/// left 146 SMs idle at the production shape (24 blocks); a tile walks from
/// a halo window exactly as the serial walk would, so the tile depth never
/// changes an output, and shallow tiles keep more blocks in flight.
pub fn grid_conv_fwd_tiled(batch: usize, d_inner: usize, t: usize) -> Result<LaunchConfig, String> {
    const TILE_T: usize = 32;
    Ok(LaunchConfig {
        grid_dim: (blocks_1d(batch * d_inner)?, tiles_y(t, TILE_T)?, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Grid of the t-major transposing kernels, the softplus into t-major rows
/// and the gate out of them (kernels/activations.cu): one 32 x 32 tile of
/// (d, t) per block and 32 x 8 threads, each thread four rows of the tile.
pub fn grid_tmajor_tiles(batch: usize, t: usize, d: usize) -> LaunchConfig {
    const TILE: u32 = 32;
    LaunchConfig {
        grid_dim: (
            (d as u32).div_ceil(TILE),
            (t as u32).div_ceil(TILE),
            batch as u32,
        ),
        block_dim: (TILE, 8, 1),
        shared_mem_bytes: 0,
    }
}

/// CONV1D_TILE_T (kernels/conv1d.cu): the tile depth of the conv backward,
/// which also fixes how many partial rows the tap gradients reduce.
const CONV_BWD_TILE_T: usize = 128;

/// Grid of the tiled conv backward: tiles of CONV1D_TILE_T = 128, the depth
/// that also fixes how many partial rows the tap gradients reduce.
pub fn grid_conv_tiled(batch: usize, d_inner: usize, t: usize) -> Result<LaunchConfig, String> {
    const TILE_T: usize = CONV_BWD_TILE_T;
    Ok(LaunchConfig {
        grid_dim: (blocks_1d(batch * d_inner)?, tiles_y(t, TILE_T)?, 1),
        block_dim: (BLOCK_1D, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Offset width of the Mamba-1 conv kernels over `(batch, seq, d_inner,
/// d_conv)`. Their offsets reach the rows they read and write, `[batch,
/// seq, widest_row]` with the widest row stride of the launch (the in_proj
/// rows the x half is read from, the d_proj rows the x gradient lands in),
/// the state and carry-in windows `[batch, d_inner, d_conv]` and the tap
/// partials of the backward, one window per tile. A decode step is `seq`
/// 1.
pub fn conv_index_width(
    batch: usize,
    seq: usize,
    d_inner: usize,
    d_conv: usize,
    widest_row: usize,
) -> IndexWidth {
    let tiles = seq.div_ceil(CONV_BWD_TILE_T).max(1);
    IndexWidth::covering(&[
        batch
            .saturating_mul(seq)
            .saturating_mul(widest_row.max(d_inner)),
        batch
            .saturating_mul(d_inner)
            .saturating_mul(d_conv)
            .saturating_mul(tiles),
    ])
}

/// Elements per 16-byte vector for an activation dtype.
pub fn vec8_width(elem_bytes: usize) -> usize {
    16 / elem_bytes
}

/// Can the 16-byte vectorized elementwise twins run for this shape and
/// these operands? Requires the element count to divide the vector width
/// and EVERY operand base pointer to be 16-byte aligned - a misaligned
/// `uint4` reinterpret faults. Buffer bases from the allocator are far
/// more aligned than this, but sliced or offset pointers are not, so the
/// check is per call site and the scalar kernel is always the fallback.
pub fn vec8_ok(
    n_elems: usize,
    elem_bytes: usize,
    ptrs: &[cudarc::driver::sys::CUdeviceptr],
) -> bool {
    let w = vec8_width(elem_bytes);
    n_elems.is_multiple_of(w) && ptrs.iter().all(|p| p.is_multiple_of(16))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAST_I32: usize = i32::MAX as usize + 4097;

    /// Below `i32::MAX` threads the 64-bit geometries are `grid_1d`'s, so a
    /// kernel moved to 64-bit indexing keeps every element on its thread.
    #[test]
    fn the_wide_geometries_are_grid_1d_below_i32_max() {
        for n in [1, 255, 256, 257, 1 << 20, 3_000_017, i32::MAX as usize] {
            let base = grid_1d(n);
            assert_eq!(grid_1d_wide(n).grid_dim, base.grid_dim, "wide, {n}");
            assert_eq!(
                grid_1d_cover(n).unwrap().grid_dim,
                base.grid_dim,
                "cover, {n}"
            );
        }
    }

    /// Past `i32::MAX` threads the strided grid stops growing and the
    /// covering grid keeps one thread per element.
    #[test]
    fn past_i32_max_the_wide_grid_strides_and_the_cover_grid_grows() {
        assert_eq!(
            grid_1d_wide(PAST_I32).grid_dim.0 as usize,
            GRID_1D_WIDE_MAX_BLOCKS
        );
        let cover = grid_1d_cover(PAST_I32).unwrap();
        assert_eq!(cover.grid_dim.0 as usize, PAST_I32.div_ceil(256));
        assert!(cover.grid_dim.0 as usize * 256 >= PAST_I32);
    }

    /// A grid past the x extent or a tile count past the y extent is an
    /// error, not a truncated launch.
    #[test]
    fn a_grid_past_its_extent_is_refused() {
        let most = i32::MAX as usize * 256;
        assert_eq!(blocks_1d(most).unwrap(), i32::MAX as u32);
        assert!(blocks_1d(most + 1).is_err());
        assert_eq!(tiles_y(65535 * 32, 32).unwrap(), 65535);
        assert!(tiles_y(65535 * 32 + 1, 32).is_err());
        assert!(grid_conv_fwd_tiled(1, 1, 65535 * 32 + 1).is_err());
        assert!(grid_conv_tiled(most, 1, 1).is_ok());
        assert!(grid_conv_tiled(most + 1, 1, 1).is_err());
    }

    /// The argument check names the first value a kernel would take
    /// wrapped.
    #[test]
    fn the_int_argument_check_names_what_would_wrap() {
        assert!(check_int_args(&[("a", 1), ("b", INT_ARG_MAX)]).is_ok());
        assert!(check_int_args(&[("b", INT_ARG_MAX + 1)]).is_err());
        let err = check_int_args(&[("a", 1), ("rows", PAST_I32), ("c", PAST_I32)]).unwrap_err();
        assert!(err.starts_with("rows "), "{err}");
    }

    /// A scan forward takes the int build up to the last shape whose offsets
    /// all fit and the `_i64` build one element past it, whichever tensor
    /// crosses: the state, the B and C rows, the row-major activations, the
    /// fused gate rows or the h tape. T-major rows and an ungated store do
    /// not count, the slim tape is counted by its chunk rows.
    #[test]
    fn a_scan_forward_takes_the_int_build_while_its_offsets_fit() {
        use IndexWidth::{I32, I64};
        let m = INT_ARG_MAX;
        assert_eq!(IndexWidth::covering(&[]), I32);
        assert_eq!(IndexWidth::covering(&[0, m]), I32);
        assert_eq!(IndexWidth::covering(&[1, m + 1]), I64);
        let nosave = scan_fwd_nosave_index_width;
        // The state [batch, d_inner, d_state].
        assert_eq!(nosave(m, 0, 1, 1, 0, false), I32);
        assert_eq!(nosave(m + 1, 0, 1, 1, 0, false), I64);
        // B and C rows [batch, d_state, T]: two states over half the range.
        let half = m / 2 + 1;
        assert_eq!(nosave(1, half - 1, 1, 2, 0, false), I32);
        assert_eq!(nosave(1, half, 1, 2, 0, false), I64);
        // Row-major activations [batch, T, d_inner], counted only row-major.
        assert_eq!(nosave(1, half - 1, 2, 1, 0, false), I32);
        assert_eq!(nosave(1, half, 2, 1, 0, false), I64);
        assert_eq!(nosave(1, half, 2, 1, 0, true), I32);
        // The fused gate's in_proj rows, counted only when the gate fuses.
        assert_eq!(nosave(1, 1, 1, 1, m - 2, false), I32);
        assert_eq!(nosave(1, 1, 1, 1, m - 1, false), I64);
        assert_eq!(nosave(1, 1, 1, 1, m - 1, true), I32);
        // The full tape holds T + 1 steps of every state, the slim one a
        // few floats per chunk.
        assert_eq!(scan_fwd_index_width(1, m - 1, 1, 1, false), I32);
        assert_eq!(scan_fwd_index_width(1, m, 1, 1, false), I64);
        assert_eq!(scan_fwd_index_width(1, m, 1, 1, true), I32);
        // The vision classifier's training shape fits; a batch whose B and
        // C rows end past 2^31 does not.
        assert_eq!(scan_fwd_index_width(2, 4621, 768, 16, true), I32);
        assert_eq!(scan_fwd_index_width(8193, 1024, 1, 256, true), I64);
    }

    /// A conv launch takes the int build while its rows, its windows and its
    /// tap partials fit, whichever crosses first.
    #[test]
    fn a_conv_launch_takes_the_int_build_while_its_offsets_fit() {
        use IndexWidth::{I32, I64};
        let m = INT_ARG_MAX;
        // Rows [batch, seq, widest row]: the in_proj stride counts.
        assert_eq!(conv_index_width(1, m / 2, 1, 1, 2), I32);
        assert_eq!(conv_index_width(1, m / 2 + 1, 1, 1, 2), I64);
        // Windows of one tile [batch, d_inner, d_conv].
        assert_eq!(conv_index_width(m / 4, 1, 1, 4, 1), I32);
        assert_eq!(conv_index_width(m / 4 + 1, 1, 1, 4, 1), I64);
        // The tap partials hold a window per tile of the backward: a window
        // wider than a tile is deep outgrows the rows.
        let (seq, dc) = (3 * CONV_BWD_TILE_T, 2 * CONV_BWD_TILE_T);
        assert_eq!(conv_index_width(m / (3 * dc), seq, 1, dc, 1), I32);
        assert_eq!(conv_index_width(m / (3 * dc) + 1, seq, 1, dc, 1), I64);
        // The vision classifier's shape and a decode step of 512 sequences
        // fit; 1025 sequences of 1024 steps read through 2048-column rows
        // do not.
        assert_eq!(conv_index_width(2, 4621, 768, 4, 1536), I32);
        assert_eq!(conv_index_width(512, 1, 2048, 4, 4096), I32);
        assert_eq!(conv_index_width(1025, 1024, 1024, 4, 2048), I64);
    }

    /// The pieces of a cast move each pointer by its own element size: a
    /// downcast and an upcast in pieces of three blocks and a tail write the
    /// bits of one launch over every element.
    #[test]
    fn a_cast_in_pieces_writes_what_one_launch_writes() {
        use crate::gpu::buffers::{DtypedBuf, GpuBuffer};
        use crate::gpu::context::GpuCtx;
        use crate::gpu::device::GpuDevice;
        use crate::gpu::dtype::WeightDtype;

        let device = GpuDevice::new(0).expect("open CUDA device");
        let ctx = GpuCtx::new(&device).expect("context");
        let s = &ctx.stream;
        let n = 2500;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 1250.0) * 0.3711).collect();
        let mut src = GpuBuffer::zeros(s, n).expect("src");
        src.upload(s, &values).expect("upload");
        let half = |piece: usize| -> DtypedBuf {
            let dst = DtypedBuf::zeros(s, n, WeightDtype::Bf16).expect("dst");
            launch_cast_pieces(
                s,
                &ctx.kernels.cast_f32_to_bf16,
                ((dst.cached_ptr(), 2), (src.cached_ptr(), 4)),
                (n, piece),
                format_args!("downcast"),
            )
            .expect("downcast");
            dst
        };
        let (one, pieces) = (half(n), half(3 * 256));
        let bits = |b: &DtypedBuf| -> Vec<u32> {
            let mut v = vec![0f32; n];
            b.download_f32(s, &mut v).expect("download");
            s.synchronize().expect("sync");
            v.iter().map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&pieces), bits(&one), "downcast");
        let full = |piece: usize| -> Vec<u32> {
            let dst = GpuBuffer::zeros(s, n).expect("dst");
            launch_cast_pieces(
                s,
                &ctx.kernels.cast_bf16_to_f32,
                ((dst.cached_ptr(), 4), (one.cached_ptr(), 2)),
                (n, piece),
                format_args!("upcast"),
            )
            .expect("upcast");
            s.synchronize().expect("sync");
            dst.to_cpu(s)
                .expect("download")
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        assert_eq!(full(3 * 256), full(n), "upcast");
    }
}
