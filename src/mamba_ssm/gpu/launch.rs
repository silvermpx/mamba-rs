//! Kernel launch helpers — grid/block calculation.
//!
//! Standard block size: 256 threads for 1D element-wise kernels.
//! SSM/conv1d kernels use batch*d_inner threads.
//! Norm kernels use batch blocks × dim threads (2D).

use cudarc::driver::LaunchConfig;

/// Standard block size for 1D element-wise kernels.
const BLOCK_1D: u32 = 256;

/// Launch config for 1D element-wise kernel on `n` elements.
///
/// Grid: `ceil(n / 256)` blocks of 256 threads.
/// Suitable for: activations, vec_add, elementwise_mul, gating, etc.
pub fn grid_1d(n: usize) -> LaunchConfig {
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
