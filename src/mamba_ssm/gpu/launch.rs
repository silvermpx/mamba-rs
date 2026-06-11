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
pub fn grid_parallel_scan(batch: usize, d_inner: usize) -> LaunchConfig {
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    const NTHREADS: u32 = 128;
    const NWARPS: usize = NTHREADS as usize / 32;
    const MAX_DSTATE: usize = 256;
    const CHUNK_SIZE: usize = NTHREADS as usize * 8; // NTHREADS * NITEMS
    let smem_floats = 2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS as usize + CHUNK_SIZE;
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (NTHREADS, 1, 1),
        shared_mem_bytes: (smem_floats * std::mem::size_of::<f32>()) as u32,
    }
}

/// Launch config for the M1 parallel scan BACKWARD kernel (Step 8e).
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
pub fn grid_parallel_scan_bwd(batch: usize, d_inner: usize) -> LaunchConfig {
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan_bwd: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    const NTHREADS: u32 = 128;
    const NWARPS: usize = NTHREADS as usize / 32;
    const MAX_DSTATE: usize = 256;
    const CHUNK_SIZE: usize = NTHREADS as usize * 8;
    // SMEM_TOTAL_FLOATS (fwd layout incl. f32-sized stage) + bwd-extra
    // (reverse warp scan + postfix + next-A exchange + da-reduce +
    // chunk-first-A boundary). Must match SMEM_BWD_FLOATS in the kernel.
    let fwd_total_floats = 2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS as usize + CHUNK_SIZE;
    let bwd_extra_floats = 2 * NWARPS + 3 * MAX_DSTATE + 2 * NTHREADS as usize;
    let total_bytes = (fwd_total_floats + bwd_extra_floats) * std::mem::size_of::<f32>();
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (NTHREADS, 1, 1),
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
/// lift on memory-bound configs per audit Agent 5 #1).
///
/// `bytes_per_act` must be `2` for bf16/f16 or `4` for f32 (in which case
/// this is identical to [`grid_parallel_scan`]).
pub fn grid_parallel_scan_typed(
    batch: usize,
    d_inner: usize,
    bytes_per_act: usize,
) -> LaunchConfig {
    debug_assert!(bytes_per_act == 2 || bytes_per_act == 4);
    assert!(
        d_inner <= 65535,
        "grid_parallel_scan_typed: d_inner {d_inner} exceeds CUDA grid.y limit 65535"
    );
    const NTHREADS: u32 = 128;
    const NWARPS: usize = NTHREADS as usize / 32;
    const MAX_DSTATE: usize = 256;
    const CHUNK_SIZE: usize = NTHREADS as usize * 8;
    // Fixed f32 region (block scan, running prefix, exchange).
    let fixed_floats = 2 * NWARPS + 2 * MAX_DSTATE + 2 * NTHREADS as usize;
    let fixed_bytes = fixed_floats * std::mem::size_of::<f32>();
    let stage_bytes = CHUNK_SIZE * bytes_per_act;
    LaunchConfig {
        grid_dim: (batch as u32, d_inner as u32, 1),
        block_dim: (NTHREADS, 1, 1),
        shared_mem_bytes: (fixed_bytes + stage_bytes) as u32,
    }
}
