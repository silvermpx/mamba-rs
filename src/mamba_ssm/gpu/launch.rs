//! Kernel launch helpers — grid/block calculation.
//!
//! Standard block size: 256 threads for 1D element-wise kernels.
//! SSM/conv1d kernels use batch*d_inner threads.
//! Norm kernels use batch blocks × dim threads (2D).

use cudarc::driver::LaunchConfig;

/// Standard block size for 1D element-wise kernels.
const BLOCK_1D: u32 = 256;

/// Validate that the largest per-tensor element count of a run fits in i32.
///
/// CUDA kernels take element counts and strides as 32-bit `int`s; a count
/// of 2^31 or more wraps to a negative argument and the kernel silently
/// processes nothing (or indexes out of bounds via overflowed products in
/// device code). The largest tensor in the training pipeline is `h_saved`
/// at `B * (T+1) * d_inner * d_state`. Call this once at trainer/state
/// construction — per-launch checks would be redundant after this.
pub fn validate_kernel_arg_capacity(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) -> Result<(), String> {
    // T=0 survives every arithmetic check but
    // produces grid_dim.x = 0 launches that surface as an opaque CUDA error
    // on a NEIGHBOURING kernel — reject it with a name at the front door.
    if seq_len == 0 {
        return Err("seq_len must be > 0".into());
    }
    if batch == 0 {
        return Err("batch must be > 0".into());
    }
    let elems = batch
        .checked_mul(seq_len + 1)
        .and_then(|v| v.checked_mul(d_inner))
        .and_then(|v| v.checked_mul(d_state))
        .ok_or("batch * (seq_len+1) * d_inner * d_state overflows usize")?;
    if elems > i32::MAX as usize {
        return Err(format!(
            "batch({batch}) * (seq_len({seq_len})+1) * d_inner({d_inner}) * d_state({d_state}) \
             = {elems} elements exceeds i32::MAX; CUDA kernels take element counts as 32-bit ints"
        ));
    }
    Ok(())
}

/// Launch config for 1D element-wise kernel on `n` elements.
///
/// Grid: `ceil(n / 256)` blocks of 256 threads.
/// Suitable for: activations, vec_add, elementwise_mul, gating, etc.
pub fn grid_1d(n: usize) -> LaunchConfig {
    // `n as u32` would silently truncate for n >= 2^32 and launch a grid
    // covering a fraction of the elements. Capacity is validated up front
    // by `validate_kernel_arg_capacity`; this is the cheap backstop.
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
/// Grid: `batch` blocks. Block: min(dim, 1024) threads.
/// Shared memory: `dim * sizeof(f32)` for reduction workspace.
pub fn grid_norm(batch: usize, dim: usize) -> LaunchConfig {
    let block = (dim as u32).min(1024);
    // Round up to next power of 2 for efficient warp reduction
    let block = block.next_power_of_two();
    LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: (block as usize * std::mem::size_of::<f32>()) as u32,
    }
}

/// Launch config for parallel prefix scan SSM kernel.
///
/// Grid: `(batch, d_inner)` — one block per (b, d) pair.
/// Block: 128 threads (matches NTHREADS in mamba_ssm_parallel.cu).
/// Shared memory: for block scan, running prefix, exchange, and coalesced staging.
///   Layout (floats): 2*NWARPS + 2*MAX_DSTATE + 2*NTHREADS + CHUNK_SIZE
///   = 2*4 + 2*256 + 2*128 + 1024 = 1800 floats = 7200 bytes.
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
    // Runtime d_state: the kernel packs its run/exchange/stage regions at
    // the actual d_state (address-only vs the padded MAX_DSTATE layout).
    let smem_floats = 2 * NWARPS + 2 * d_state + 2 * SCAN_NTHREADS + SCAN_CHUNK;
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: (smem_floats * std::mem::size_of::<f32>()) as u32,
    }
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
    // it by its compile-time capacity): at ds=16 this returns ~11.5 KB of
    // smem per block vs the old 256-slot stride and lifts residency.
    let f32_floats = 4 * NWARPS + 4 * SCAN_NTHREADS + 3 * g * d_state + 3 * SCAN_NTHREADS;
    let stage_bytes = (3 * g + 1) * SCAN_CHUNK * bytes_per_act;
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

/// Launch config for the typed (bf16/f16) M1 parallel scan forward kernel.
///
/// Differs from [`grid_parallel_scan`] by allocating only `CHUNK_SIZE *
/// sizeof(T_ACT)` bytes for the smem staging area (vs 4 bytes per slot for
/// f32). On bf16/f16 this saves 2 KB per block, taking total smem from
/// 7200 B → 5152 B and enabling the kernel's `__launch_bounds__(128, 4)`
/// to actually fit 4 resident blocks per SM on Ada (~10–15 % throughput
/// lift on memory-bound configs on memory-bound configs).
///
/// `bytes_per_act` must be `2` for bf16/f16 or `4` for f32 (in which case
/// this is identical to [`grid_parallel_scan`]).
pub fn grid_parallel_scan_typed(
    batch: usize,
    d_inner: usize,
    bytes_per_act: usize,
    d_state: usize,
) -> LaunchConfig {
    debug_assert!(bytes_per_act == 2 || bytes_per_act == 4);
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan_typed: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    assert!(
        d_state <= 256,
        "grid_parallel_scan_typed: d_state {d_state} exceeds the kernel MAX_DSTATE guard - \
         the kernel would return without writing y"
    );
    const NWARPS: usize = SCAN_NTHREADS / 32;
    // Fixed f32 region (block scan, running prefix, exchange) at the
    // actual d_state (address-only vs the padded MAX_DSTATE layout).
    let fixed_floats = 2 * NWARPS + 2 * d_state + 2 * SCAN_NTHREADS;
    let fixed_bytes = fixed_floats * std::mem::size_of::<f32>();
    let stage_bytes = SCAN_CHUNK * bytes_per_act;
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (SCAN_NTHREADS as u32, 1, 1),
        shared_mem_bytes: (fixed_bytes + stage_bytes) as u32,
    }
}

/// Tiled conv1d grid: x covers (b * d_inner) threads at 256/block, y covers
/// T in CONV1D_TILE_T=128 tiles. The serial per-(b,d) walk left 146 SMs
/// idle at the campaign shape (24 blocks); tiling fills the machine while
/// every output element keeps the identical 4-tap arithmetic.
/// Tile depth of `gather_bc_cols_tmajor_tiled` (must match GBC_TILE_T in
/// elementwise.cu). One block per (t-tile, b); dynamic smem carries the
/// B and C tiles at `[d_state][GBC_TILE_T + 1]` each (+1 pad kills bank
/// conflicts on the transposed read).
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

pub fn grid_conv_tiled(batch: usize, d_inner: usize, t: usize) -> LaunchConfig {
    const TILE_T: usize = 128;
    LaunchConfig {
        grid_dim: (
            ((batch * d_inner) as u32).div_ceil(256),
            (t as u32).div_ceil(TILE_T as u32),
            1,
        ),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
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
    n_elems.is_multiple_of(w) && ptrs.iter().all(|p| p % 16 == 0)
}
