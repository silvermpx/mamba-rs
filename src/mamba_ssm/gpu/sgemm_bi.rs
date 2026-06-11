//! Deterministic batch-invariant f32 SGEMM dispatcher (training triad).
//!
//! Ported from SQV-RS `sqv_uaac` (`blas_gpu.rs` + `kernels/sgemm_bi.cu`,
//! siboehm warptiling lineage). Three entry points used when
//! `ctx.batch_invariant()` is enabled:
//!
//!   - [`sgemm_bi_forward`]      NN: `Y = X @ W + bias`
//!   - [`sgemm_bi_backward_dw`]  TN: `dW += X^T @ dY` (accumulated)
//!   - [`sgemm_bi_backward_dx`]  NT: `dX = dY @ W^T`
//!
//! Every shape routes through a fixed-tile custom kernel (Big / Slim /
//! narrow / GEMV / split-K with deterministic tree reduce) — never cuBLAS.
//! Guarantees, in decreasing strength:
//!   - bit-identical across RUNS for a fixed shape (always);
//!   - bit-identical across BATCH SIZES that route to the same dispatch
//!     bucket (the per-cell K order is fixed within a bucket; crossing a
//!     bucket boundary — e.g. ultra-thin M<32 vs split-K M>=32 — changes
//!     the reduction association deterministically);
//!   - full f32 precision (no TF32 mantissa truncation).
//!
//! Unsupported shapes return `Err` (instead of SQV's panic): callers should
//! disable the batch-invariant flag for such configs rather than silently
//! falling back to non-deterministic cuBLAS.

use super::buffers::GpuBuffer;
use super::kernels::MambaKernels as GpuKernels;
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

type CUptr = cudarc::driver::sys::CUdeviceptr;

// ── Split-M TN partition heuristic (ported from SQV-RS blas_bi.rs) ──

/// Target CTA count factor for the split-M TN partition: aim to fill the
/// GPU with at least this many blocks when the base (K-tile × N-tile) grid
/// underfills it.
const SPLITM_TN_TARGET_GRID_FACTOR: u32 = 284;
/// Scratch cap for split-M partials, in f32 elements. Must not exceed the
/// `splitk_scratch` allocation in kernels.rs.
const SPLITM_TN_SCRATCH_CAP: usize = 1 << 23;
/// m_chunk alignment (BK of the TN tile).
const SPLITM_TN_BK_ALIGN: u32 = 16;

/// Decide the split-M factor for the TN (dW) kernel on underfilled grids.
/// Returns `(m_chunk, f_final)` or `None` when the plain kernel is fine.
#[inline]
fn splitm_tn_partition(batch: usize, n_in: usize, n_out: usize) -> Option<(usize, usize)> {
    if !(n_in >= 128 && n_out >= 128 && batch >= 256) {
        return None;
    }
    let k_tiles = (n_in as u32).div_ceil(128);
    let n_tiles = (n_out as u32).div_ceil(128);
    let base_blocks = k_tiles * n_tiles;
    if base_blocks == 0 || base_blocks >= SPLITM_TN_TARGET_GRID_FACTOR {
        return None;
    }
    let f_grid = SPLITM_TN_TARGET_GRID_FACTOR.div_ceil(base_blocks);
    let f_scratch_cap = (SPLITM_TN_SCRATCH_CAP / (n_in * n_out)) as u32;
    let f = f_grid.min(f_scratch_cap).max(1);
    let m_chunk_raw = (batch as u32).div_ceil(f);
    let m_chunk = (m_chunk_raw + SPLITM_TN_BK_ALIGN - 1) & !(SPLITM_TN_BK_ALIGN - 1);
    let f_final = (batch as u32).div_ceil(m_chunk);
    if f_final < 2 || (f_final as usize) * n_in * n_out > SPLITM_TN_SCRATCH_CAP {
        return None;
    }
    Some((m_chunk as usize, f_final as usize))
}

/// Minimum N (output cols) before the dispatcher switches from Slim-N tiles
/// to Big-N tiles. Below this, Slim-N (BN=64) packs better; above it Big-N
/// (BN=128) wins on wave occupancy. Historic name (`SGEMM_CUSTOM_MIN`) is a
/// leftover from when the threshold gated a cuBLAS fallback — the fallback
/// is gone (zero-cuBLAS contract), the constant remains as a tile-pick
/// boundary only.
const SGEMM_CUSTOM_MIN: usize = 128;

/// Boundary between Slim-N and Big tile variants (by output N dimension).
const SGEMM_SLIM_MAX: usize = 512;

/// v6.5 Phase C-1.5av: separate Slim Split-K NT-via-T n_in cap for backward dx.
/// The forward Slim NN path uses N as output dim → SGEMM_SLIM_MAX=512 bounds
/// wave-fill correctness there. But NT-via-T backward dx reads n_in (input dim
/// of original forward), and the kernel itself tiles arbitrary n_in via N-axis
/// tiling — the 512 cap is conservative, not load-bearing. v6.5 multi-step
/// critic_in_proj has n_in = critic_in = d_model + action_dim + 2*emb = 641
/// at default config. Bumping to 768 lets this shape hit Slim Split-K NT-via-T
/// with F=4 K-tile partials (576 blocks vs plain Big NT 144 blocks).
/// Determinism preserved: F is shape-keyed (function of n_out, not batch).
const SGEMM_SLIM_NT_NIN_MAX: usize = 768;

/// M threshold below which we force Slim-N even for N ≥ 129 (wave underfill protection).
/// At M < 512, Big tile BM=128 gives ≤4 M-blocks; adding N-blocks via Slim's BN=64 (vs Big's BN=128)
/// doubles grid to reduce wave underfill on Ada's 142 SMs. Only matters when N ≥ 129 (otherwise slim already chosen).
const SGEMM_M_SLIM_FORCE: usize = 512;

/// Single source of truth for Split-K/M scratch buffer cap, in f32 elements.
/// Must match `splitk_scratch` allocation in `kernels.rs` (1 << 23 = 8M f32 = 32 MB).
/// All Split-K dispatch gates (NN fwd, NT bwd_dx, Split-M TN bwd_dw) read this.
pub(super) const SPLITK_SCRATCH_CAP: usize = 1 << 23;

/// SM count for dispatch wave-fill heuristics. Calibrated for Ada RTX 6000 (142 SMs).
/// Over-shoot on smaller GPUs (A100=108) is correctness-safe — Split-K gates fire
/// slightly more aggressively. TODO: query `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`
/// at init for true per-GPU tuning; for now a single source-of-truth constant.
pub(super) const NUM_SMS: u32 = 142;

/// Pick (kernel function, BN tile size) with M-aware wave-quantization fix.
/// Slim-N for narrow output, or for small M with wide N.
/// Phase 2 will extend this dispatcher with narrow / GEMV / small-K buckets.
fn dispatch_slim_or_big<'k>(
    _kernels: &'k GpuKernels,
    m: usize,
    n_out: usize,
    func_slim: &'k cudarc::driver::CudaFunction,
    func_big: &'k cudarc::driver::CudaFunction,
) -> (&'k cudarc::driver::CudaFunction, u32) {
    let slim = n_out <= SGEMM_SLIM_MAX || (m < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    let func = if slim { func_slim } else { func_big };
    let bn: u32 = if slim { 64 } else { 128 };
    (func, bn)
}

/// Batched linear forward on GPU: `Y[B,N] = X[B,K] @ W[K,N] + bias[N]`.
///
/// cuBLAS computes: `Y^T[N,B] = W^T[N,K] @ X^T[K,B]` (column-major).
/// With row-major data, this is equivalent to: `Y[B,N] = X[B,K] @ W[K,N]`.
///
/// Bias is broadcast via pre-fill + beta=1.0 accumulate.
///
/// # Arguments
/// - `y`: output `[B * N]`, overwritten
/// - `x`: input `[B * K]`
/// - `w`: weights `[K * N]`
/// - `bias`: optional `[N]`, broadcast to each row
/// - `batch`: B (number of samples)
/// - `n_in`: K (input dimension)
/// - `n_out`: N (output dimension)
pub fn sgemm_bi_forward(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: &mut GpuBuffer,
    x: &GpuBuffer,
    w_ptr: CUptr,
    bias_ptr: CUptr, // 0 = no bias
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    // Shape-A Ultra-Thin-M NN dispatch: batch ∈ [1, 31] (actor inference rollout).
    // Covers shapes that fall through Split-K (min 32) and Big/Slim (min 128).
    // Grid: (ceil(N/32), M, 1). smem = K*4 bytes ≤ 8 KB (K ≤ 2048) — within the
    // 48 KB default dynamic-smem limit on sm_80+.
    // Non-mod-32 N handled by kernel's `col < N` predication (tail tile partial).
    // K up to 2048 covers SimbaV2 w2 forward (K=2048 N=512 when batch < 32).
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n_out as u32).div_ceil(32), batch as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n_in * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nn_ultra_thin);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&k_i); // lda
        builder.arg(&n_i); // ldb
        builder.arg(&n_i); // ldc
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nn_ultra_thin forward: {:?}", e))?;
        return Ok(());
    }

    // Narrow-N NN small-tile dispatch: N∈[2..127] AND batch ≤ 64.
    // Production target: TQC critic qhead w2 (M=64, K=512, N=25). Tile
    // NBM=16 NBN=16 NBK=16, 64 threads (2 warps). At M=64 N=25 grid is
    // ceil(64/16) × ceil(25/16) = 4 × 2 = 8 CTAs (vs 1 for the big-tile
    // narrow kernel). Per-output FMA chain is byte-identical to
    // sgemm_bi_nn_narrow regardless of tile — same ascending K __fmaf_rn,
    // same bias pre-seed at K=0, same scalar N-tail epilogue. ZERO ULP
    // downstream drift; CPU mirror (narrow_nn_sgemm_nn in blas_bi.rs) is
    // tile-agnostic and matches both GPU variants.
    if (2..=127).contains(&n_out) && (1..=64).contains(&batch) && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let post_op: i32 = 0;
        let num_pid_m = (batch as u32).div_ceil(16);
        let num_pid_n = (n_out as u32).div_ceil(16);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nn_narrow_small);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        builder.arg(&n_i);
        builder.arg(&post_op);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nn_narrow_small forward: {:?}", e))?;
        return Ok(());
    }

    // Phase 2.1 Narrow-N NN dispatch: N∈[2..127], batch > 64.
    // Tile BM=64 BN=32 BK=16, 128 threads, 2x2 warps. Scalar N-epilogue.
    // Kernel has M-predication (`if (g_row >= M) continue;`) and N-predication
    // (`if (g_col >= N) continue;`) → safe for any batch and any N via tile count.
    // Covers test-config shapes (M=32, K=32..64, N=32..64) that otherwise fall
    // to cuBLAS (non-deterministic, violates zero-cuBLAS contract).
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let post_op: i32 = 0;
        let num_pid_m = (batch as u32).div_ceil(64);
        let num_pid_n = (n_out as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nn_narrow);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        builder.arg(&n_i);
        builder.arg(&post_op);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nn_narrow forward: {:?}", e))?;
        return Ok(());
    }

    // Phase 2.2 GEMV-N1 dispatch: N=1 output (actor mean/log_std heads).
    // 4 rows/block, warp-shuffle K-reduction, deterministic batch-invariant.
    //
    // Phase 2.2.1 (2026-05-04): batch lower bound relaxed 4 → 1. Kernel
    // sgemm_bi_nn_gemv has `if (row >= M) return;` predication (kernels/sgemm_bi.cu:2201)
    // so M<4 is safe — partial last block. Closes single-env eval gap
    // (M=1 N=1 K=512 was hitting cuBLAS-fallback panic в gpu_eval_parity test).
    // Determinism preserved (kernel unchanged; same warp-shuffle butterfly).
    if n_out == 1 && batch >= 1 && n_in >= 32 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let lda_i = n_in as i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch as u32).div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nn_gemv);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&ldy_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_gemv forward: {:?}", e))?;
        return Ok(());
    }

    // Split-K Thin-M NN + K-tail dispatch for M<128 shapes with non-%32 K.
    // Decompose K = K_main + K_tail, where K_main = K - K%32 (multiple of 32),
    // K_tail = K%32 (1..31). Main is processed by the existing Split-K NN kernel
    // with lda = full K (so it reads only columns [0..K_main) of each row).
    // Tail is folded into the reducer as Σ_k X[m, K_main+k] · W[K_main+k, n].
    // Universal: works for K ∈ {33..65535} with any K%32 ≠ 0 — covers K=129,
    // K=257 (SALE action), K=385, K=513, K=642 (hypermlp forward), etc.
    // Use module-level SPLITK_SCRATCH_CAP (was per-function duplicated).
    //
    // Phase C-1.5bo: Slim NN underfill guard. Even with cap≤1024, wide-N shapes
    // (n_out=2048 at batch=1024) give Slim NN base_blocks=8*32=256 = 1.8 waves —
    // already saturated; Split-K's partial+reducer (DRAM scratch round-trip) is
    // strictly negative. Threshold 1*NUM_SMS=142 = true underfill only. Tighter
    // than the Slim Split-K's `3*NUM_SMS` because Thin-M's BM=32 produces 4× the
    // tile count of Slim NN's BM=128 — proportionally less SM headroom needed.
    // Replaces the implicit "batch≤1024" guard with an explicit M_tiles*N_tiles
    // check that doesn't rely on cap-relax envelope.
    let plain_slim_blocks_nn_ktail = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(64);
    let underfill_nn_ktail = plain_slim_blocks_nn_ktail < NUM_SMS;
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill_nn_ktail
    {
        let k_tail = n_in % 32;
        let k_main = n_in - k_tail;
        let partial_size = (k_main / 32) * batch * n_out;
        if k_main >= 32 && partial_size <= SPLITK_SCRATCH_CAP {
            let m_i = batch as i32;
            let n_i = n_out as i32;
            let k_chunks = (k_main / 32) as i32;
            let lda_i = n_in as i32; // actual stride (full K)
            let alpha: f32 = 1.0;
            let num_pid_m = (batch as u32).div_ceil(32);
            let num_pid_n = (n_out as u32).div_ceil(64);
            let partial_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_pid_m * num_pid_n * k_chunks as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let partial_ptr = {
                use cudarc::driver::DevicePtr;
                let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
                ptr
            };
            // Main Split-K partial on A columns [0..k_main), B rows [0..k_main).
            let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk32_partial);
            pb.arg(&partial_ptr);
            pb.arg(x.inner());
            pb.arg(&w_ptr);
            pb.arg(&m_i);
            pb.arg(&n_i);
            pb.arg(&k_chunks);
            pb.arg(&lda_i);
            unsafe { pb.launch(partial_cfg) }
                .map_err(|e| format!("sgemm_bi_nn_splitk32_partial (K-tail main): {:?}", e))?;

            // Tail fold via reducer: tail_cnt iterations of X[m, K_main+k] · W[K_main+k, n].
            let total = (batch * n_out) as u32;
            let reduce_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (total.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let zero_i32: i32 = 0;
            let tail_cnt_i = k_tail as i32;
            use cudarc::driver::DevicePtr;
            let (x_base_ptr, _r_x) = x.inner().device_ptr(stream);
            let x_tail_ptr: u64 = x_base_ptr + (k_main as u64) * 4; // X[:, k_main]
            let w_tail_ptr: u64 = w_ptr + ((k_main * n_out) as u64) * 4; // W[k_main, :]
            let x_tail_stride_i = n_in as i32; // stride between X[m, k_main] rows = K_full
            let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
            rb.arg(y.inner_mut());
            rb.arg(&partial_ptr);
            rb.arg(&bias_ptr);
            rb.arg(&x_tail_ptr);
            rb.arg(&w_tail_ptr);
            rb.arg(&alpha);
            rb.arg(&m_i);
            rb.arg(&n_i);
            rb.arg(&k_chunks);
            rb.arg(&x_tail_stride_i);
            rb.arg(&zero_i32); // out_col_stride default = N
            rb.arg(&tail_cnt_i);
            unsafe { rb.launch(reduce_cfg) }
                .map_err(|e| format!("sgemm_bi_splitk_reduce (K-tail): {:?}", e))?;
            return Ok(());
        }
    }

    // Split-K Thin-M NN dispatch for M<128 shapes (SALE L1/L2/L3 in gradient step,
    // M=batch=64). K split into 32-wide chunks, partial GEMMs run per (m,n,kc) block,
    // followed by deterministic tree-reduce. Grid fills 32+ blocks vs the 2-4 the
    // full-K Slim kernel would spawn → 4-8× better SM utilization.
    //
    // Envelope: M ∈ [32, 127], N ∈ [64, 512], K % 32 == 0, K ≥ 32,
    // and partial_size = K_CHUNKS*M*N ≤ 2M floats (scratch capacity).
    //
    // Phase C-1.5bo: same Slim NN underfill guard as K-tail variant above.
    let partial_size = (n_in / 32) * batch * n_out;
    let plain_slim_blocks_nn_main = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(64);
    let underfill_nn_main = plain_slim_blocks_nn_main < NUM_SMS;
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && partial_size <= SPLITK_SCRATCH_CAP
        && underfill_nn_main
    {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_chunks = (n_in / 32) as i32;
        let alpha: f32 = 1.0;

        // Partial kernel launch: grid = M_tiles × N_tiles × K_CHUNKS
        let num_pid_m = (batch as u32).div_ceil(32);
        let num_pid_n = (n_out as u32).div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n * k_chunks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            // Raw device pointer to pre-allocated scratch (1 MB, shared across calls).
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
            ptr
        };
        let lda_i = n_in as i32; // A row stride = full K (no tail in this branch)
        let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk32_partial);
        pb.arg(&partial_ptr);
        pb.arg(x.inner());
        pb.arg(&w_ptr);
        pb.arg(&m_i);
        pb.arg(&n_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_i);
        unsafe { pb.launch(partial_cfg) }
            .map_err(|e| format!("sgemm_bi_nn_splitk32_partial: {:?}", e))?;

        // Reduce kernel launch: grid covers M*N outputs, 256 threads/block.
        let total = (batch * n_out) as u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let null_tail: u64 = 0;
        let zero_i32: i32 = 0;
        let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
        rb.arg(y.inner_mut());
        rb.arg(&partial_ptr);
        rb.arg(&bias_ptr);
        rb.arg(&null_tail); // x_tail_ptr (none)
        rb.arg(&null_tail); // w_tail_ptr (none)
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&n_i);
        rb.arg(&k_chunks);
        rb.arg(&zero_i32); // x_tail_stride (unused)
        rb.arg(&zero_i32); // out_col_stride = N (default)
        rb.arg(&zero_i32); // tail_cnt = 0 (no tail)
        unsafe { rb.launch(reduce_cfg) }.map_err(|e| format!("sgemm_bi_splitk_reduce: {:?}", e))?;
        return Ok(());
    }

    // Phase 6 v2: Split-K Slim NN for fat-M shapes (M > 1024) that underfill
    // the Slim grid. Targets Mamba layer shapes at b=64 seq=33 → M=2112.
    // Tile BM=128 BN=64 BK=32 (same as sgemm_bi_nn_slim) — each fc's K-slice
    // has identical per-block FMA order to Slim NN on that K-range. Reducer
    // sgemm_bi_splitk_reduce applies alpha + bias, overwrites y.
    //
    // Determinism: K_CHUNK is a COMPILE-TIME CONSTANT → F = ceil(K / K_CHUNK)
    // depends ONLY on K. Same K always produces same F (and same per-fc
    // k-range) regardless of M, N, batch, stream, or SM scheduling.
    // Reducer f32 ascending-fc order (`sgemm_bi_splitk_reduce`; only the
    // split-M TN reducer is f64). Batch-invariant by construction.
    //
    // Gate ordering: fires AFTER Thin-M cap (batch > 1024) so it never steals
    // shapes Thin-M handles well (M ≤ 1024 has 4× BM=32 tiles vs 1× BM=128,
    // better fill under Thin-M). Only fat-M Mamba shapes land here.
    //
    // K_CHUNK choice: 64 (2× BK=32) — gives F=2 for K=128 (Mamba in_proj),
    // F=4 for K=256 (out_proj). Note post-Phase-1 the Mamba input_proj K is
    // `obs_dim` (was `obs_dim + emb = 384`), which falls below the F≥6 gate
    // below — Mamba input_proj routes via regular Slim NN dispatch now.
    // Sweet spot: small enough to split K=128, big enough that each fc does
    // 2+ BK iterations to amortize kernel launch overhead.
    //
    // Pitfall (learned from sgemm_batch_invariance_dispatch_matrix failure
    // commit cc9788b9 → fix commit):
    //   EARLIER: F derived from base_blocks (M_tiles*N_tiles/SMs). Failed
    //   because batch=4224 and batch=2048 produced different F → different
    //   reduction order → bit drift. DO NOT reintroduce batch-dependent F.
    const SPLITK_SLIM_K_CHUNK: u32 = 64; // PURE K-BASED, batch-invariant
    if batch > 1024
        && (128..=SGEMM_SLIM_MAX).contains(&n_out)
        && n_in >= SPLITK_SLIM_K_CHUNK as usize  // need ≥1 chunk (actually ≥4 below)
        && n_in.is_multiple_of(32)
    {
        // F is a pure function of K. Same K → same F, always.
        let f_final = (n_in as u32).div_ceil(SPLITK_SLIM_K_CHUNK);
        // F ≥ 6 (K ≥ 384). Ncu profiles:
        //   - SALE L1 (M=4224 K=128 N=128, F=2): reducer 35% of time → moved
        //     out of splitk_slim in a prior commit (F≥4 gate).
        //   - SALE L2/L3 (M=4224 K=256 N=256, F=4): splitk_reduce kernel is
        //     DRAM-bound at 83% throughput, partial kernel SM at 30%. The
        //     132 M-N output tiles (≥ 128 SMs) already saturate without
        //     K-split → splitk_slim only adds reducer overhead. Moved out
        //     of splitk_slim at the F≥6 raise.
        //   - (Historical) Mamba input_proj (M=3840 K=384 N=128, F=6): 60 M-N
        //     tiles < SM count, F=6 wave-fill was a win. Post-Phase-1 (z_s
        //     dropped from Mamba input) K=obs_dim falls below the F≥6 gate
        //     and routes via regular Slim NN. Gate retained for any future
        //     K∈[384,512] fat-M shape.
        //
        // After this gate raise, shapes with K < 384 fall to the regular
        // Slim NN dispatch below (single kernel, no reducer overhead).
        // CPU mirror gate at blas_bi.rs:136-139 mirrors this exact threshold.
        if f_final >= 6 && (f_final as usize) * batch * n_out <= SPLITK_SCRATCH_CAP {
            // Wave-fill heuristic: skip if Slim grid is already well-filled
            // (perf guard only, not correctness — F is batch-invariant above).
            let m_tiles = (batch as u32).div_ceil(128);
            let n_tiles = (n_out as u32).div_ceil(64);
            let base_blocks = m_tiles * n_tiles;
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                let k_chunk = SPLITK_SLIM_K_CHUNK;
                let m_i = batch as i32;
                let n_i = n_out as i32;
                let k_i = n_in as i32;
                let lda_i = n_in as i32; // A is [M, K], row-major
                let ldb_i = n_out as i32; // B is [K, N], row-major
                let k_chunk_i = k_chunk as i32;
                let alpha: f32 = 1.0;

                let partial_ptr = {
                    use cudarc::driver::DevicePtr;
                    let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
                    ptr
                };

                let partial_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (base_blocks, 1, f_final),
                    block_dim: (128, 1, 1), // Slim tile: 128 threads
                    shared_mem_bytes: 0,    // static smem
                };
                let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk_slim_partial);
                pb.arg(&partial_ptr);
                pb.arg(x.inner());
                pb.arg(&w_ptr);
                pb.arg(&m_i);
                pb.arg(&n_i);
                pb.arg(&k_i);
                pb.arg(&lda_i);
                pb.arg(&ldb_i);
                pb.arg(&k_chunk_i);
                unsafe { pb.launch(partial_cfg) }
                    .map_err(|e| format!("sgemm_bi_nn_splitk_slim_partial: {:?}", e))?;

                let total = (batch * n_out) as u32;
                let reduce_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let null_tail: u64 = 0;
                let zero_i32_local: i32 = 0;
                let f_i = f_final as i32;
                let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
                rb.arg(y.inner_mut());
                rb.arg(&partial_ptr);
                rb.arg(&bias_ptr);
                rb.arg(&null_tail); // x_tail_ptr (none)
                rb.arg(&null_tail); // w_tail_ptr (none)
                rb.arg(&alpha);
                rb.arg(&m_i);
                rb.arg(&n_i);
                rb.arg(&f_i); // K_chunks = F
                rb.arg(&zero_i32_local); // x_tail_stride (unused)
                rb.arg(&zero_i32_local); // out_col_stride default = N
                rb.arg(&zero_i32_local); // tail_cnt = 0 (K % 32 == 0 enforced)
                unsafe { rb.launch(reduce_cfg) }
                    .map_err(|e| format!("sgemm_bi_splitk_reduce (slim): {:?}", e))?;
                return Ok(());
            }
        }
    }

    // ===== Gap-fill: thin-M wide-N shapes not caught by specialized branches =====
    // Closes the dispatcher gap at (M < 128, N >= 128) that ultra-thin (M < 32,
    // K ≤ 2048), narrow tier 1/2 (N ≤ 127), splitk-thin (N ≤ 2048, requires
    // N%4==0 + K-tail or K%32==0), splitk-slim (M > 1024), and big-NN (M >= 128)
    // all miss. Example shapes: M=32 K=32 N=194 (N%4=2 fails splitk-thin, M<128
    // fails big-NN); Mamba-1 in_proj at micro-batch — M ∈ [32,128) with
    // N = 2·d_inner > 2048 for d_model ≥ 576, and M < 32 with K = d_model >
    // 2048 (d_model = 2560) where ultra-thin's smem K-cap excludes it.
    //
    // No upper N bound: `sgemm_nn_narrow` tiles N via ceil(N/32) CTAs with
    // M/N predication — unbounded by construction. K likewise unbounded
    // (strict ascending-K loop, no smem K staging).
    //
    // Re-uses `sgemm_nn_narrow` kernel (BM=64 BN=32, M/N predicated) — per-output
    // FMA chain is byte-identical to CPU mirror `narrow_nn_sgemm_nn` (strict
    // ascending K + `mul_add`) regardless of tile grid. Determinism preserved
    // by the per-output independence: tile boundary doesn't enter rounding.
    //
    // Perf: ~43% tile fill at boundary shapes (M=32 padded to BM=64) —
    // acceptable for shapes that no specialized branch handles. Specialized
    // branches above always take priority via gate ordering.
    if batch < 128 && n_out >= 128 && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let post_op: i32 = 0;
        let num_pid_m = (batch as u32).div_ceil(64);
        let num_pid_n = (n_out as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nn_narrow);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        builder.arg(&n_i);
        builder.arg(&post_op);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nn_narrow (gap-fill thin-M wide-N): {:?}", e))?;
        return Ok(());
    }

    // Custom deterministic SGEMM.
    // Envelope: M ≥ 128, N ≥ 128, K ≥ 1. Non-%4 N handled by kernel scalar N-epilogue.
    // Non-%4 K handled by kernel scalar K-fallback (runtime lda%4 check).
    // K<BK: kernel's scalar bounds check zero-fills smem for dotIdx≥K; wastes a few FMAs
    // but correct (handles Mamba-1 dt_proj K=4,8). Phase 2.4 — dropped `n_in >= 16` guard.
    if batch >= SGEMM_CUSTOM_MIN && n_out >= SGEMM_CUSTOM_MIN && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (func, bn) = dispatch_slim_or_big(
            kernels,
            batch,
            n_out,
            &kernels.sgemm_nn_slim,
            &kernels.sgemm_nn,
        );
        let slim = bn == 64;
        // Opt1: Big uses 256 threads/block for TLP; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // T1 v2: Big NN uses dynamic smem (2-stage cp.async). 33 KB needed.
        // Slim still uses static smem (single-stage). Set shared_mem_bytes only for Big.
        let smem_bytes: u32 = if slim { 0 } else { 34 * 1024 };
        // 2026-05-13 — Stage-4 persistent-CTA cap removed. Kernel body is now
        // data-parallel (one tile per CTA), so grid_dim == total_tiles. See
        // sgemm_bi.cu for the kernel-side unwrap rationale.
        let total_tiles = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(bn);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(y.inner_mut());
        builder.arg(x.inner());
        builder.arg(&w_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        builder.arg(&k_i); // lda = n_in (A is [M, K])
        builder.arg(&n_i); // ldb = n_out (B is [K, N])
        builder.arg(&n_i); // ldc = n_out (C is [M, N])
        unsafe { builder.launch(cfg) }.map_err(|e| {
            format!(
                "sgemm_bi_nn{} forward: {:?}",
                if slim { "_slim" } else { "" },
                e
            )
        })?;
        return Ok(());
    }

    // T2.11: zero-cuBLAS contract — all training paths must route through
    // custom deterministic kernels. A reachable cuBLAS fallback breaks
    // CPU↔GPU parity and is non-deterministic. Panic loudly so missing
    // dispatch coverage is caught at first hit, not as a silent training
    // regression months later.
    panic!(
        "gpu_sgemm_forward: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

/// Weight gradient: `dW[K,N] += X^T[K,B] @ dY[B,N]` (accumulated, beta=1.0).
///
/// cuBLAS: `dW^T[N,K] += dY^T[N,B] @ X[B,K]`
/// In col-major: A=dY (transa=N gives `dY^T[N,B]`), B=X_saved (transb=T gives `X[B,K]`)
/// gemm(N, T, N, K, B, 1.0, dY, N, X_saved, K, 1.0, dW, N)
///
/// Note: beta=1.0 for gradient accumulation.
pub fn sgemm_bi_backward_dw(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr, // accumulated in place (+=)
    dy: &GpuBuffer,
    x_saved: &GpuBuffer,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    // Phase 2.2b GEMV-N1 TN dispatch: dW[K,1] += X^T[K,M] @ dY[M,1]
    if n_out == 1 && n_in >= 4 && batch >= 32 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let lda_i = n_in as i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n_in as u32).div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_tn_gemv);
        builder.arg(&dw_ptr);
        builder.arg(x_saved.inner());
        builder.arg(dy.inner());
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&lda_i);
        builder.arg(&ldy_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_tn_gemv backward_dw: {:?}", e))?;
        return Ok(());
    }

    // Phase 2.1b Narrow-N TN dispatch: N∈[2..127] (critic qhead + gap-fill for
    // N∈[49..127] where slim/big kernels (N>=128) don't apply).
    // T3.3 (2026-05-01): comment fixed — gate was relaxed to N≥2 in Stage 4
    // shape coverage; the stale `9..127` text predated that change.
    // Kernel has `if (g_row >= K_out) continue;` and N-tile predication via
    // `div_ceil(N, 32)` blocks → safe for any n_in and any N.
    // Relaxed to n_in>=1, batch>=1 covers test shapes (M=32, K=32..64, N=32..64)
    // that otherwise fall to cuBLAS (zero-cuBLAS contract violation).
    if (2..=127).contains(&n_out) && n_in >= 1 && batch >= 1 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let n_i = n_out as i32;
        let alpha: f32 = 1.0;
        let num_pid_m = (n_in as u32).div_ceil(64);
        let num_pid_n = (n_out as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_tn_narrow);
        builder.arg(&dw_ptr);
        builder.arg(x_saved.inner());
        builder.arg(dy.inner());
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_tn_narrow backward_dw: {:?}", e))?;
        return Ok(());
    }

    // Split-M TN dispatch: M-axis split for underfilled Big TN grids.
    // CUTLASS parallel-split + deterministic ascending-fc reducer.
    //
    // F-SPLITM-TN-CONST (2026-05-17): partitioning math hoisted to
    // `blas_bi::splitm_tn_partition` so CPU mirror computes identical
    // (m_chunk, f_final). Replaces former `2*NUM_SMS`-dependent heuristic
    // (which made bit-exactness depend on GPU model) with a portable
    // `SPLITM_TN_TARGET_GRID_FACTOR = 284` (= historical Ada NUM_SMS=142×2).
    // Run-to-run bit-exact AND CPU↔GPU bit-exact at every batch ≥ 256.
    // Backward_dw is intentionally NOT batch-invariant (sums over M) but
    // for each fixed batch the (m_chunk, f_final) is deterministic.
    if let Some((m_chunk, f_final)) = splitm_tn_partition(batch, n_in, n_out) {
        let base_blocks = (n_in as u32).div_ceil(128) * (n_out as u32).div_ceil(128);
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let n_i = n_out as i32;
        let m_chunk_i = m_chunk as i32;
        let alpha: f32 = 1.0;
        let f_i = f_final as i32;
        let f_final_u32 = f_final as u32;

        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
            ptr
        };

        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (base_blocks, 1, f_final_u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut pb = stream.launch_builder(&kernels.sgemm_tn_splitm_partial);
        pb.arg(&partial_ptr);
        pb.arg(x_saved.inner());
        pb.arg(dy.inner());
        pb.arg(&m_i);
        pb.arg(&k_i);
        pb.arg(&n_i);
        pb.arg(&m_chunk_i);
        unsafe { pb.launch(partial_cfg) }
            .map_err(|e| format!("sgemm_bi_tn_splitm_partial: {:?}", e))?;

        let total = (n_in * n_out) as u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut rb = stream.launch_builder(&kernels.sgemm_splitm_reduce);
        rb.arg(&dw_ptr);
        rb.arg(&partial_ptr);
        rb.arg(&alpha);
        rb.arg(&k_i);
        rb.arg(&n_i);
        rb.arg(&f_i);
        unsafe { rb.launch(reduce_cfg) }.map_err(|e| format!("sgemm_bi_splitm_reduce: {:?}", e))?;
        return Ok(());
    }

    // Custom: dW[K,N] += X^T[K,M] @ dY[M,N]
    // Envelope: K_out ≥ 1, N ≥ 128. Kernel A-load is scalar per-row (handles non-%4 M),
    // B-load has runtime N%4 scalar fallback. K scalar fallback handles non-%4 K.
    // Phase 2.4: dropped `n_in >= 128` — kernel grid handles K_out<128 correctly;
    // covers Mamba-1 dt_proj backward (K_out=8).
    if n_in >= 1 && n_out >= SGEMM_CUSTOM_MIN {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let n_i = n_out as i32;
        let alpha: f32 = 1.0;
        let (func, bn) = dispatch_slim_or_big(
            kernels,
            n_in, // TN output rows = n_in (K_out); M-aware over output's leading dim
            n_out,
            &kernels.sgemm_tn_slim,
            &kernels.sgemm_tn,
        );
        let slim = bn == 64;
        // Opt1: Big uses 256 threads/block; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // T1 v2: Big TN uses dynamic smem for 2-stage cp.async (34 KB); Slim stays static.
        let smem_bytes: u32 = if slim { 0 } else { 34 * 1024 };
        // 2026-05-13 — data-parallel launch (no persistent-CTA cap). See
        // gpu_sgemm_forward note and sgemm_bi.cu for the kernel-side unwrap.
        let total_tiles = (n_in as u32).div_ceil(128) * (n_out as u32).div_ceil(bn);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(&dw_ptr);
        builder.arg(x_saved.inner());
        builder.arg(dy.inner());
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&n_i);
        unsafe { builder.launch(cfg) }.map_err(|e| {
            format!(
                "sgemm_bi_tn{} backward_dw: {:?}",
                if slim { "_slim" } else { "" },
                e
            )
        })?;
        return Ok(());
    }

    // T2.11: zero-cuBLAS contract — see gpu_sgemm_forward.
    panic!(
        "gpu_sgemm_backward_dw: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

/// Input gradient: `dX[B,K] = dY[B,N] @ W^T[N,K]` (overwritten, beta=0.0).
///
/// cuBLAS: `dX^T[K,B] = W[K,N] @ dY^T[N,B]`
/// But we want dX row-major, so:
/// `dX^T[K,B] = W[K,N](as col-major=W^T[N,K]) @ dY^T[N,B]`
///
/// Actually, row-major trick:
/// For C = A @ B^T in row-major:
/// C^T = B @ A^T in col-major
/// gemm(T, N, K, B, N, 1.0, W, N, dY, N, 0.0, dX, K)
pub fn sgemm_bi_backward_dx(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: &mut GpuBuffer,
    dy: &GpuBuffer,
    w_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    // Phase 2.1c Narrow-N NT dispatch: N∈[2..127] (critic qhead + gap-fill for
    // N∈[49..127] where slim/big kernels (N>=128) don't apply).
    // T3.3 (2026-05-01): comment fixed — gate was relaxed to N≥2 in Stage 4
    // shape coverage; the stale `9..127` text predated that change.
    // Kernel has `if (g_row >= M) continue;` M-predication → safe for any batch.
    // Relaxed to n_in>=1, batch>=1 covers test-config (M=32, K=32..64, N=32..64)
    // that otherwise falls to cuBLAS (zero-cuBLAS contract violation).
    if (2..=127).contains(&n_out) && n_in >= 1 && batch >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let num_pid_m = (batch as u32).div_ceil(64);
        let num_pid_n = (n_in as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nt_narrow);
        builder.arg(dx.inner_mut());
        builder.arg(dy.inner());
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nt_narrow backward_dx: {:?}", e))?;
        return Ok(());
    }

    // Phase 2.1c+ (2026-05-04) Small-batch wide-N NT dispatch.
    // Gap: batch ∈ [1, 31], N >= 128 — Narrow NT capped at N=127, Split-K
    // NT-via-T requires batch >= 32, Big/Slim NT requires batch >= 128.
    // Solution: reuse sgemm_nt_narrow kernel — N is reduction-axis, kernel
    // iterates `for nIdx in [0, N) by NBK=16` (sgemm_bi.cu:2635), no upper
    // bound on N. Tile dims (BM=64, BN=32) fit any small batch; M/K_out
    // predication inside kernel handles partial last block.
    // Determinism: kernel unchanged → bit-exact с N≤127 path.
    // Production unaffected: training uses batch=128 (Big/Slim path).
    // Closes test_gpu_correctness M=4 K=32 N=128 cuBLAS-fallback panic.
    if batch < 32 && n_in >= 1 && n_out >= 128 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let num_pid_m = (batch as u32).div_ceil(64);
        let num_pid_n = (n_in as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nt_narrow);
        builder.arg(dx.inner_mut());
        builder.arg(dy.inner());
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        unsafe { builder.launch(cfg) }.map_err(|e| {
            format!(
                "sgemm_bi_nt_narrow (small-batch wide-N) backward_dx: {:?}",
                e
            )
        })?;
        return Ok(());
    }

    // Phase 2.2c GEMV-N1 NT dispatch: dX[M,K] = dY[M,1] @ W^T[1,K] (outer product)
    // Phase 2.2.1c (2026-05-04): batch lower bound relaxed 4 → 1.
    // Kernel sgemm_bi_nt_gemv computes per-element dX[m,k] = alpha*dY[m]*W[k]
    // с total = M*K total threads и `if (tid >= total) return;` predication
    // (kernels/sgemm_bi.cu:2296) — safe для M<4. Closes single-env eval gap.
    if n_out == 1 && n_in >= 1 && batch >= 1 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let ldx_i = n_in as i32;
        let ldy_i: i32 = 1;
        let total = (batch * n_in) as u32;
        let block = 256u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nt_gemv);
        builder.arg(dx.inner_mut());
        builder.arg(dy.inner());
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&k_i);
        builder.arg(&ldx_i);
        builder.arg(&ldy_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nt_gemv backward_dx: {:?}", e))?;
        return Ok(());
    }

    // Split-K NT-via-transpose + K-tail for M<128 shapes with K_out%32 != 0.
    // Covers SALE action backward_dx (K_out=257, tail=1), SimbaV2 hyperplane
    // backward_dx (K_out=642, tail=2), and any K_out%32 ∈ {1..31}: main is the
    // first K_out - (K_out%32) rows (multiple of 32 → %4 safe for vectorized
    // stores), tail is K_out%32 columns filled via sequential dx_col_gemv calls.
    // Same transpose_scratch (4 M f32) + splitk_scratch (8 M f32) as the main
    // NT-via-T path below, so envelope caps match: K_out ≤ 4096, N ≤ 2048.
    //
    // Phase C-1.5bo: Slim NT-via-T underfill guard. Slim NT tile BM=128, BN=64
    // along K_out (= n_in here — backward dx output column axis). When grid ≥
    // NUM_SMS, Slim NT already saturates; Split-K transpose + partial + reducer
    // adds DRAM round-trips for no occupancy benefit. Threshold 1*NUM_SMS
    // matches forward NN guards (same Slim BM=128 vs Thin-M BM=32 geometry).
    let plain_slim_blocks_nt_ktail = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(64);
    let underfill_nt_ktail = plain_slim_blocks_nt_ktail < NUM_SMS;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill_nt_ktail
    {
        let k_tail_cnt = n_in % 32;
        let k_main = n_in - k_tail_cnt;
        let w_size_main = k_main * n_out;
        let partial_size_main = (n_out / 32) * batch * k_main;
        // F-KTAIL-CAP-PARITY (2026-05-17): w_size cap = SPLITK_NT_TRANSPOSE_CAP
        // (the GPU transpose_scratch capacity), partial cap = SPLITK_SCRATCH_CAP
        // (the GPU splitk_scratch capacity). Earlier hardcoded `1<<23` partial
        // cap was tighter than the underlying scratch (1<<23) and caused k_tail
        // to fall through at batch=1024 (partial=10.5M > 8M cap) while CPU has
        // no cap → catastrophic dispatch divergence on critic.head0.input_proj
        // dX at BATCH=1024 (max_ulp=2.1M on synthetic LCG). Lifting matches the
        // actual scratch sizes — bit-exact + no perf regression (k_tail is the
        // optimal path; the previous cap unnecessarily routed to slower default).
        if k_main >= 32
            && w_size_main <= SPLITK_NT_TRANSPOSE_CAP
            && partial_size_main <= SPLITK_SCRATCH_CAP
        {
            // Step 1: transpose W[0..k_main, :] → W_T[N, k_main] into scratch.
            let rows_i = k_main as i32;
            let cols_i = n_out as i32;
            let t_grid_x = (n_out as u32).div_ceil(32);
            let t_grid_y = (k_main as u32).div_ceil(32);
            let t_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (t_grid_x, t_grid_y, 1),
                block_dim: (32, 32, 1),
                shared_mem_bytes: 0,
            };
            let w_t_ptr = {
                use cudarc::driver::DevicePtr;
                let (ptr, _r) = kernels.transpose_scratch.device_ptr(stream);
                ptr
            };
            let mut tb = stream.launch_builder(&kernels.sgemm_transpose_f32_2d);
            tb.arg(&w_t_ptr);
            tb.arg(&w_ptr);
            tb.arg(&rows_i);
            tb.arg(&cols_i);
            unsafe { tb.launch(t_cfg) }
                .map_err(|e| format!("sgemm_transpose_f32_2d (K-tail): {:?}", e))?;

            // Step 2: Split-K NN partial — A=dY, B=W_T, output [M, k_main].
            let m_i = batch as i32;
            let k_main_i = k_main as i32;
            let k_chunks = (n_out / 32) as i32;
            let lda_dy_i = n_out as i32;
            let num_pid_m = (batch as u32).div_ceil(32);
            let num_pid_n = (k_main as u32).div_ceil(64);
            let partial_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_pid_m * num_pid_n * k_chunks as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let partial_ptr = {
                use cudarc::driver::DevicePtr;
                let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
                ptr
            };
            let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk32_partial);
            pb.arg(&partial_ptr);
            pb.arg(dy.inner());
            pb.arg(&w_t_ptr);
            pb.arg(&m_i);
            pb.arg(&k_main_i);
            pb.arg(&k_chunks);
            pb.arg(&lda_dy_i);
            unsafe { pb.launch(partial_cfg) }
                .map_err(|e| format!("sgemm_bi_nn_splitk32_partial (NT K-tail main): {:?}", e))?;

            // Step 3: reducer writes dX[:, 0..k_main] with stride n_in.
            let null_tail: u64 = 0;
            let alpha: f32 = 1.0;
            let null_bias: u64 = 0;
            let zero_i32: i32 = 0;
            let out_stride_i = n_in as i32;
            let total_main = (batch * k_main) as u32;
            let reduce_cfg = cudarc::driver::LaunchConfig {
                grid_dim: (total_main.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
            rb.arg(dx.inner_mut());
            rb.arg(&partial_ptr);
            rb.arg(&null_bias);
            rb.arg(&null_tail);
            rb.arg(&null_tail);
            rb.arg(&alpha);
            rb.arg(&m_i);
            rb.arg(&k_main_i);
            rb.arg(&k_chunks);
            rb.arg(&zero_i32);
            rb.arg(&out_stride_i); // dX row stride = n_in (K_out full)
            rb.arg(&zero_i32); // tail_cnt = 0 (tail handled by separate gemv)
            unsafe { rb.launch(reduce_cfg) }
                .map_err(|e| format!("sgemm_bi_splitk_reduce (NT K-tail main): {:?}", e))?;

            // Step 4: loop over tail columns. For each k in [0, k_tail_cnt):
            // dX[:, k_main + k] = Σ_n dY[m, n] · W[k_main + k, n]. Each call is
            // one gemv; tail_cnt ≤ 31 so total overhead is bounded. Sequential
            // (not parallel) to keep kernel launches small and deterministic.
            let w_base_ptr = w_ptr;
            let n_i = n_out as i32;
            let block = 128u32;
            let tail_cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((batch as u32).div_ceil(block), 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };
            for k in 0..k_tail_cnt {
                let k_tail_col = k_main + k;
                let w_tail_row_ptr: u64 = w_base_ptr + (k_tail_col * n_out) as u64 * 4;
                let col_idx_i = k_tail_col as i32;
                let mut gb = stream.launch_builder(&kernels.sgemm_dx_col_gemv);
                gb.arg(dx.inner_mut());
                gb.arg(dy.inner());
                gb.arg(&w_tail_row_ptr);
                gb.arg(&m_i);
                gb.arg(&n_i);
                gb.arg(&col_idx_i);
                gb.arg(&out_stride_i);
                unsafe { gb.launch(tail_cfg) }
                    .map_err(|e| format!("sgemm_bi_dx_col_gemv (NT K-tail col={}): {:?}", k, e))?;
            }
            return Ok(());
        }
    }

    // Split-K NT-via-transpose dispatch для M<128 shapes (SALE/SimbaV2 w2 bwd_dx).
    // Strategy: transpose W[K_out, N] → W_T[N, K_out], then dX = dY @ W_T via the
    // existing NN Split-K kernel. Per research 2026-04-19: 1.6-1.8× faster than
    // dedicated NT.
    //
    // A.2 — generalised к support n_out%32 != 0 by folding the N-tail (residue
    // after the largest 32-aligned prefix) into the reducer's `tail_cnt` arg.
    // The reducer (sgemm_bi.cu:2902) already supports tail folding: for each
    // (m, n) cell it appends `Σ_{k<tail_cnt} x_tail[m,k] * w_tail[k,n]` after
    // the K_CHUNKS partial reduce. For NT-via-T post-transpose the tail is along
    // the reduction axis (= original n_out), so:
    //   x_tail_ptr     = dY[:, n_main]              (stride n_out, full dY width)
    //   w_tail_ptr     = W_T[n_main, :]             (stride n_in)
    //   x_tail_stride  = n_out
    //   tail_cnt       = n_out % 32
    // For n_out%32==0 the tail is empty (tail_cnt=0) and behaviour matches the
    // pre-A.2 main path bit-exactly. For n_out%32 != 0 (e.g. production hit
    // M=36 K=128 N=796 → tail=28) the formerly-uncovered shape now lands here
    // with full custom-kernel coverage and no cuBLAS fallback.
    //
    // Envelope: M ∈ [32, 1024], K_out ∈ [64, 4096], K_out % 4 == 0,
    // N ∈ [32, 2048], n_in % 32 == 0 (K-tail bwd_dx gate at line 897 covers
    // n_in%32 != 0 separately; combined K-tail + N-tail is rare and falls
    // through к cuBLAS by design — punt unless production shows it).
    const SPLITK_NT_TRANSPOSE_CAP: usize = 1 << 22; // 4M f32 = transpose_scratch size
    let n_tail_nt = n_out % 32;
    let n_main_nt = n_out - n_tail_nt;
    let w_size_nt = n_in * n_out;
    let partial_size_nt = if n_main_nt > 0 {
        (n_main_nt / 32) * batch * n_in
    } else {
        0
    };
    // Phase C-1.5bo: same Slim NT-via-T underfill guard as K-tail variant above.
    let plain_slim_blocks_nt_main = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(64);
    let underfill_nt_main = plain_slim_blocks_nt_main < NUM_SMS;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(4)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main_nt >= 32
        && w_size_nt <= SPLITK_NT_TRANSPOSE_CAP
        && partial_size_nt <= SPLITK_SCRATCH_CAP
        && underfill_nt_main
    {
        // Step 1: transpose full W[n_in=K_out, n_out=N] → W_T[N, K_out] into
        // scratch (full width, including the tail rows W_T[n_main..n_out, :]).
        let rows_i = n_in as i32;
        let cols_i = n_out as i32;
        let t_grid_x = (n_out as u32).div_ceil(32);
        let t_grid_y = (n_in as u32).div_ceil(32);
        let t_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (t_grid_x, t_grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        let w_t_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.transpose_scratch.device_ptr(stream);
            ptr
        };
        let mut tb = stream.launch_builder(&kernels.sgemm_transpose_f32_2d);
        tb.arg(&w_t_ptr);
        tb.arg(&w_ptr);
        tb.arg(&rows_i);
        tb.arg(&cols_i);
        unsafe { tb.launch(t_cfg) }.map_err(|e| format!("sgemm_transpose_f32_2d: {:?}", e))?;

        // Step 2: NN Split-K partial on the n_main (32-aligned) prefix.
        // partial = dY[M, n_main] @ W_T[n_main, K_out], reduction over n_main.
        // lda_i = n_out (full dY row stride) — partial reads only the first
        // k_chunks*32 = n_main columns per row, leaving the tail для step 3.
        let m_i = batch as i32;
        let k_out_i = n_in as i32;
        let k_chunks = (n_main_nt / 32) as i32;

        let num_pid_m = (batch as u32).div_ceil(32);
        let num_pid_n = (n_in as u32).div_ceil(64);
        let partial_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n * k_chunks as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let partial_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
            ptr
        };
        let lda_i = n_out as i32; // dY row stride = N_full (NOT n_main)
        let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk32_partial);
        pb.arg(&partial_ptr);
        pb.arg(dy.inner());
        pb.arg(&w_t_ptr);
        pb.arg(&m_i);
        pb.arg(&k_out_i);
        pb.arg(&k_chunks);
        pb.arg(&lda_i);
        unsafe { pb.launch(partial_cfg) }
            .map_err(|e| format!("sgemm_bi_nn_splitk32_partial (NT-via-T N-tail): {:?}", e))?;

        // Step 3: reducer with N-tail fold. Computes
        //   dX[m,k] = Σ_{c<k_chunks} partial[c][m,k]               (chunk sum, ascending c)
        //          + Σ_{i<tail_cnt} dY[m, n_main+i] · W_T[n_main+i, k]   (tail, ascending i)
        // FMA single-rounding inside reducer. Total reduction order: ascending
        // n over [0, n_full) — bit-exact с CPU sgemm_nt ascending-n loop.
        let alpha: f32 = 1.0;
        let null_bias: u64 = 0;
        let total = (batch * n_in) as u32;
        let reduce_cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let zero_i32: i32 = 0;
        let tail_cnt_i = n_tail_nt as i32;
        let dy_tail_stride_i = n_out as i32; // dY row stride
        // x_tail_ptr = dY[:, n_main] (offset n_main floats into base).
        // w_tail_ptr = W_T[n_main, :] (offset n_main * n_in floats into W_T base).
        let (dy_tail_ptr, wt_tail_ptr): (u64, u64) = if n_tail_nt > 0 {
            use cudarc::driver::DevicePtr;
            let (dy_base, _r_dy) = dy.inner().device_ptr(stream);
            let dyp = dy_base + (n_main_nt as u64) * 4;
            let wtp = w_t_ptr + (n_main_nt as u64 * n_in as u64) * 4;
            (dyp, wtp)
        } else {
            (0, 0)
        };
        let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
        rb.arg(dx.inner_mut());
        rb.arg(&partial_ptr);
        rb.arg(&null_bias);
        rb.arg(&dy_tail_ptr);
        rb.arg(&wt_tail_ptr);
        rb.arg(&alpha);
        rb.arg(&m_i);
        rb.arg(&k_out_i);
        rb.arg(&k_chunks);
        rb.arg(&dy_tail_stride_i);
        rb.arg(&zero_i32); // out_col_stride default = N (= K_out = n_in)
        rb.arg(&tail_cnt_i);
        unsafe { rb.launch(reduce_cfg) }
            .map_err(|e| format!("sgemm_bi_splitk_reduce (NT-via-T N-tail): {:?}", e))?;
        return Ok(());
    }

    // Phase 6 v2 Task 5: Split-K Slim NN via transpose for fat-M bwd_dx shapes
    // (M > 1024). Mirrors the M<128 NT-via-T above but uses Slim Split-K partial
    // (BM=128 BN=64) for better arithmetic intensity on fat-M Mamba shapes.
    //
    // Transformation: dX[M, K_out] = dY[M, N] @ W^T[N, K_out]
    //   After transposing W[K_out, N] → W_T[N, K_out], becomes NN:
    //   dX[M, K_out] = dY[M, N] @ W_T[N, K_out]
    //   Kernel params: M=batch, N=K_out (n_in), K=N (n_out, reduction axis).
    //
    // Batch-invariance: K_CHUNK = compile-time constant → F = ceil(N / K_CHUNK)
    // is a pure function of N. Same N always produces same F for any batch.
    //
    // Fires AFTER M<=1024 NT-via-T gate so never steals shapes Thin-M handles.
    const SLIM_NT_K_CHUNK: u32 = 64;
    if batch > 1024
        // v6.5 Phase C-1.5av: bumped from SGEMM_SLIM_MAX=512 → SGEMM_SLIM_NT_NIN_MAX=768
        // to include critic_in=641 NT bwd_dx. Kernel handles any n_in via N-tiling,
        // so 512 cap was conservative; 768 is bit-exact safe and gives +15-25% on
        // critic backward dX. Determinism preserved (F = shape-keyed pure function).
        && (128..=SGEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= SLIM_NT_K_CHUNK as usize
        && n_out.is_multiple_of(32)
        && (n_in * n_out) <= SPLITK_NT_TRANSPOSE_CAP
    {
        // F depends only on N (reduction axis of transposed problem).
        let f_final = (n_out as u32).div_ceil(SLIM_NT_K_CHUNK);
        if f_final >= 2 && (f_final as usize) * batch * n_in <= SPLITK_SCRATCH_CAP {
            // Perf heuristic: fire only if plain Slim NT grid underfills.
            let m_tiles = (batch as u32).div_ceil(128);
            let k_out_tiles = (n_in as u32).div_ceil(64);
            let base_blocks = m_tiles * k_out_tiles;
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                let k_chunk = SLIM_NT_K_CHUNK;
                // Step 1: transpose W[n_in=K_out, n_out=N] → W_T[N, K_out] into scratch.
                let rows_i = n_in as i32;
                let cols_i = n_out as i32;
                let t_grid_x = (n_out as u32).div_ceil(32);
                let t_grid_y = (n_in as u32).div_ceil(32);
                let t_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (t_grid_x, t_grid_y, 1),
                    block_dim: (32, 32, 1),
                    shared_mem_bytes: 0,
                };
                let w_t_ptr = {
                    use cudarc::driver::DevicePtr;
                    let (ptr, _r) = kernels.transpose_scratch.device_ptr(stream);
                    ptr
                };
                let mut tb = stream.launch_builder(&kernels.sgemm_transpose_f32_2d);
                tb.arg(&w_t_ptr);
                tb.arg(&w_ptr);
                tb.arg(&rows_i);
                tb.arg(&cols_i);
                unsafe { tb.launch(t_cfg) }
                    .map_err(|e| format!("sgemm_transpose_f32_2d (slim NT): {:?}", e))?;

                // Step 2: Slim Split-K NN partial on (dY, W_T) with K_chunk split.
                let m_i = batch as i32;
                let k_out_i = n_in as i32; // NN's "N" = K_out
                let k_full_i = n_out as i32; // NN's "K" = n_out (reduction axis)
                let lda_i = n_out as i32; // dY stride = n_out
                let ldb_i = n_in as i32; // W_T stride = K_out
                let k_chunk_i = k_chunk as i32;

                let partial_ptr = {
                    use cudarc::driver::DevicePtr;
                    let (ptr, _r) = kernels.splitk_scratch.device_ptr(stream);
                    ptr
                };

                let partial_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (base_blocks, 1, f_final),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut pb = stream.launch_builder(&kernels.sgemm_nn_splitk_slim_partial);
                pb.arg(&partial_ptr);
                pb.arg(dy.inner());
                pb.arg(&w_t_ptr);
                pb.arg(&m_i);
                pb.arg(&k_out_i);
                pb.arg(&k_full_i);
                pb.arg(&lda_i);
                pb.arg(&ldb_i);
                pb.arg(&k_chunk_i);
                unsafe { pb.launch(partial_cfg) }
                    .map_err(|e| format!("sgemm_bi_nn_splitk_slim_partial (slim NT): {:?}", e))?;

                // Step 3: reducer writes dX (beta=0, no bias, alpha=1).
                let alpha: f32 = 1.0;
                let null_bias: u64 = 0;
                let null_tail: u64 = 0;
                let zero_i32_nt: i32 = 0;
                let f_i = f_final as i32;
                let total = (batch * n_in) as u32;
                let reduce_cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (total.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut rb = stream.launch_builder(&kernels.sgemm_splitk_reduce);
                rb.arg(dx.inner_mut());
                rb.arg(&partial_ptr);
                rb.arg(&null_bias);
                rb.arg(&null_tail);
                rb.arg(&null_tail);
                rb.arg(&alpha);
                rb.arg(&m_i);
                rb.arg(&k_out_i);
                rb.arg(&f_i);
                rb.arg(&zero_i32_nt);
                rb.arg(&zero_i32_nt);
                rb.arg(&zero_i32_nt);
                unsafe { rb.launch(reduce_cfg) }
                    .map_err(|e| format!("sgemm_bi_splitk_reduce (slim NT): {:?}", e))?;
                return Ok(());
            }
        }
    }

    // ===== Gap-fill: thin-batch wide-N shapes not caught by specialized branches =====
    // Closes dispatcher gap at (batch ∈ [32..128), N >= 128) that:
    //   - Narrow NT (line ~932) caps at N=127
    //   - Small-batch wide-N (line ~967) caps at batch < 32
    //   - Split-K NT-via-T requires N % 32 == 0 (n_out=194 with %32=2 falls)
    //   - Big NT requires batch >= 128
    // Order: AFTER all splitk attempts (so it never steals their coverage),
    // BEFORE big-NT. Re-uses `sgemm_nt_narrow` kernel (BM=64, BN=32 along K_out,
    // N as reduction axis with `nIdx in [0,N) by NBK=16` — no upper bound on N).
    //
    // Determinism: per-output ascending-N FMA chain — bit-identical to CPU
    // mirror `narrow_nt_sgemm_nt` regardless of tile grid. Same kernel as the
    // small-batch-<32 branch above, so byte-identical FMA path.
    //
    // Perf: ~50% tile fill at boundary (batch padded to BM=64) — acceptable
    // for a gap-fill vs cuBLAS panic / non-determinism.
    if (32..128).contains(&batch) && n_in >= 1 && n_out >= 128 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        let num_pid_m = (batch as u32).div_ceil(64);
        let num_pid_n = (n_in as u32).div_ceil(32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_pid_m * num_pid_n, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&kernels.sgemm_nt_narrow);
        builder.arg(dx.inner_mut());
        builder.arg(dy.inner());
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        unsafe { builder.launch(cfg) }
            .map_err(|e| format!("sgemm_bi_nt_narrow (gap-fill mid-batch wide-N): {:?}", e))?;
        return Ok(());
    }

    // Custom: dX[M,K] = dY[M,N] @ W^T[N,K]
    // Envelope: M ≥ 128, K_out ≥ 1. Kernel has scalar N-fallback for non-%4 N,
    // scalar K-fallback for non-%4 K_out.
    // Phase 2.4: dropped `n_in >= 128` — covers Mamba-1 dt_proj backward_dx (K_out=8).
    if batch >= SGEMM_CUSTOM_MIN && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_i = n_in as i32;
        let alpha: f32 = 1.0;
        // NT output leading dim = n_in (K_out); M-aware fan-out by batch.
        let (func, bn) = dispatch_slim_or_big(
            kernels,
            batch,
            n_in, // NT's "N" in dispatcher sense is K_out
            &kernels.sgemm_nt_slim,
            &kernels.sgemm_nt,
        );
        let slim = bn == 64;
        // Opt1: Big uses 256 threads/block; Slim stays 128.
        let threads = if slim { 128u32 } else { 256u32 };
        // T1 v2: Big NT uses dynamic smem for 2-stage cp.async (34 KB).
        let smem_bytes: u32 = if slim { 0 } else { 34 * 1024 };
        // 2026-05-13 — data-parallel launch (no persistent-CTA cap). See
        // gpu_sgemm_forward note and sgemm_bi.cu for the kernel-side unwrap.
        let total_tiles = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(bn);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let mut builder = stream.launch_builder(func);
        builder.arg(dx.inner_mut());
        builder.arg(dy.inner());
        builder.arg(&w_ptr);
        builder.arg(&alpha);
        builder.arg(&m_i);
        builder.arg(&n_i);
        builder.arg(&k_i);
        unsafe { builder.launch(cfg) }.map_err(|e| {
            format!(
                "sgemm_bi_nt{} backward_dx: {:?}",
                if slim { "_slim" } else { "" },
                e
            )
        })?;
        return Ok(());
    }

    // T2.11: zero-cuBLAS contract — see gpu_sgemm_forward.
    panic!(
        "gpu_sgemm_backward_dx: cuBLAS fallback hit (shape M={batch} K={n_in} N={n_out}). \
         The zero-cuBLAS contract requires every shape to route through a custom \
         kernel — add a dispatcher branch in this function for this shape."
    );
}

// ============================================================================
// Typed (bf16/f16) dispatch — Phase 11 stage 2 buckets.
// ============================================================================
// Same bucket geometry and launch configs as the f32 dispatcher above; the
// typed kernels are bit-identical to "upcast inputs to f32, run the f32
// kernel". Buckets not yet covered (Big/Slim/split-K — stage 3) return Err:
// callers must not silently fall back to non-deterministic cuBLAS.

use super::blas::TypedPtr;
use super::dtype::WeightDtype;

// ---------------------------------------------------------------------------
// f32-cascade routing predicates (stage 3). Each returns true iff the f32
// dispatcher would run the BIG kernel (BN=128, 2-stage 33 KB smem) for this
// shape — i.e. NO earlier bucket in the cascade claims it AND the final
// slim/big split picks Big. The typed dispatch uses these so a native typed
// Big kernel fires exactly where the f32 reference runs the same FMA chain;
// any drift between a predicate and the real cascade shows up as a bit
// mismatch in tests/sgemm_bi_typed_parity.rs.
// ---------------------------------------------------------------------------

/// NN forward: mirrors `sgemm_bi_forward` (gemv, ultra-thin, narrow tiers,
/// split-K thin-M K-tail/main, split-K slim, gap-fill, then big/slim).
pub(super) fn nn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    if n_out == 1 {
        return false; // gemv (or panic tail) — never Big
    }
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        return false; // ultra-thin
    }
    if (2..=127).contains(&n_out) {
        return false; // narrow tiers
    }
    let plain_slim_blocks = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(64);
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-K thin-M K-tail
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32 && (k_main / 32) * batch * n_out <= SPLITK_SCRATCH_CAP {
            return false;
        }
    }
    // split-K thin-M main
    if (32..=1024).contains(&batch)
        && (64..=2048).contains(&n_out)
        && n_out.is_multiple_of(4)
        && n_in >= 32
        && n_in.is_multiple_of(32)
        && (n_in / 32) * batch * n_out <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return false;
    }
    // split-K slim
    if batch > 1024
        && (128..=SGEMM_SLIM_MAX).contains(&n_out)
        && n_in >= 64
        && n_in.is_multiple_of(32)
    {
        let f_final = (n_in as u32).div_ceil(64);
        if f_final >= 6 && (f_final as usize) * batch * n_out <= SPLITK_SCRATCH_CAP {
            let base_blocks = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(64);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if batch < 128 {
        return false; // gap-fill territory
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// TN dW: mirrors `sgemm_bi_backward_dw` (gemv, narrow, split-M, big/slim
/// keyed on output rows = `n_in`).
pub(super) fn tn_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    if n_out == 1 || (2..=127).contains(&n_out) {
        return false; // gemv / narrow
    }
    if splitm_tn_partition(batch, n_in, n_out).is_some() {
        return false;
    }
    if !(n_in >= 1 && n_out >= SGEMM_CUSTOM_MIN) {
        return false;
    }
    let slim = n_out <= SGEMM_SLIM_MAX || (n_in < SGEMM_M_SLIM_FORCE && n_out >= SGEMM_CUSTOM_MIN);
    !slim
}

/// NT dX: mirrors `sgemm_bi_backward_dx` (narrow, col-gemv, gemv, split-N
/// K-tail/main, split-N slim, gap-fill, big/slim keyed on (`batch`, `n_in`)).
pub(super) fn nt_routes_to_big(batch: usize, n_in: usize, n_out: usize) -> bool {
    if (2..=127).contains(&n_out) {
        return false; // NT narrow (small reduction N)
    }
    if batch < 32 && n_out >= 128 {
        return false; // dx_col_gemv
    }
    if n_out == 1 {
        return false; // NT gemv
    }
    const SPLITK_NT_TRANSPOSE_CAP: usize = 1 << 22;
    let plain_slim_blocks = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(64);
    let underfill = plain_slim_blocks < NUM_SMS;
    // split-N K-tail
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in >= 33
        && !n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_out.is_multiple_of(32)
        && underfill
    {
        let k_main = n_in - n_in % 32;
        if k_main >= 32
            && k_main * n_out <= SPLITK_NT_TRANSPOSE_CAP
            && (n_out / 32) * batch * k_main <= SPLITK_SCRATCH_CAP
        {
            return false;
        }
    }
    // split-N main
    let n_main = n_out - n_out % 32;
    if (32..=1024).contains(&batch)
        && (64..=4096).contains(&n_in)
        && n_in.is_multiple_of(4)
        && n_in.is_multiple_of(32)
        && (32..=2048).contains(&n_out)
        && n_main >= 32
        && n_in * n_out <= SPLITK_NT_TRANSPOSE_CAP
        && (n_main / 32) * batch * n_in <= SPLITK_SCRATCH_CAP
        && underfill
    {
        return false;
    }
    // split-N slim
    if batch > 1024
        && (128..=SGEMM_SLIM_NT_NIN_MAX).contains(&n_in)
        && n_out >= 64
        && n_out.is_multiple_of(32)
        && n_in * n_out <= SPLITK_NT_TRANSPOSE_CAP
    {
        let f_final = (n_out as u32).div_ceil(64);
        if f_final >= 2 && (f_final as usize) * batch * n_in <= SPLITK_SCRATCH_CAP {
            let base_blocks = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(64);
            if base_blocks > 0 && base_blocks < 3 * NUM_SMS {
                return false;
            }
        }
    }
    if (32..128).contains(&batch) {
        return false; // gap-fill NT
    }
    if !(batch >= SGEMM_CUSTOM_MIN && n_in >= 1) {
        return false;
    }
    let slim = n_in <= SGEMM_SLIM_MAX || (batch < SGEMM_M_SLIM_FORCE && n_in >= SGEMM_CUSTOM_MIN);
    !slim
}

fn require_half(dt: WeightDtype, what: &str) -> Result<(), String> {
    if dt == WeightDtype::F32 {
        return Err(format!(
            "sgemm_bi typed dispatch: {what} is f32 — use the f32 entry points"
        ));
    }
    Ok(())
}

/// Tensor-core NN forward (stage 5, `bi_tensor_cores` tier):
/// `Y = X @ W + bias` via mma.sync.m16n8k16 with f32 accumulation.
/// SEPARATE numeric contract from the scalar triad (TC reduction tree, not
/// the ascending-K FMA chain) — deterministic and batch-invariant across
/// ALL M (each element's full K-reduction lives in one warp, independent of
/// grid shape). Covers M >= 128 && N >= 128 && K >= 1; Err otherwise.
pub fn sgemm_bi_forward_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(y.dtype, "output")?;
    if x.dtype != y.dtype || w.dtype != y.dtype {
        return Err("sgemm_bi_forward_tc: mixed dtypes not supported".into());
    }
    if !(batch >= 128 && n_out >= 128 && n_in >= 1) {
        return Err(format!(
            "sgemm_bi_forward_tc: shape M={batch} K={n_in} N={n_out} below the TC tile gate"
        ));
    }
    let dt = y.dtype;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m_i = batch as i32;
    let n_i = n_out as i32;
    let k_i = n_in as i32;
    let total_tiles = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(128);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kernels.sgemm_nn_tc_typed.get(dt));
    b.arg(&y.ptr);
    b.arg(&x.ptr);
    b.arg(&w.ptr);
    b.arg(&bias_ptr);
    b.arg(&alpha);
    b.arg(&beta);
    b.arg(&m_i);
    b.arg(&n_i);
    b.arg(&k_i);
    b.arg(&k_i);
    b.arg(&n_i);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_tc: {e:?}"))?;
    Ok(())
}

/// Tensor-core TN dW (stage 5): `dW[K,N] += X^T @ dY` via mma.sync with f32
/// accumulate straight into the f32 master gradient. Same TC contract as
/// [`sgemm_bi_forward_tc`]. Covers K_out >= 128 && N >= 128.
pub fn sgemm_bi_backward_dw_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr,
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(dy.dtype, "dY")?;
    if dy.dtype != x_saved.dtype {
        return Err("sgemm_bi_backward_dw_tc: mixed dtypes not supported".into());
    }
    if !(n_in >= 128 && n_out >= 128 && batch >= 1) {
        return Err(format!(
            "sgemm_bi_backward_dw_tc: shape M={batch} K={n_in} N={n_out} below the TC tile gate"
        ));
    }
    let dt = dy.dtype;
    let alpha: f32 = 1.0;
    let m_red_i = batch as i32;
    let k_out_i = n_in as i32;
    let n_i = n_out as i32;
    let total_tiles = (n_in as u32).div_ceil(128) * (n_out as u32).div_ceil(128);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kernels.sgemm_tn_tc_typed.get(dt));
    b.arg(&dw_ptr);
    b.arg(&x_saved.ptr);
    b.arg(&dy.ptr);
    b.arg(&alpha);
    b.arg(&m_red_i);
    b.arg(&k_out_i);
    b.arg(&n_i);
    unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_tn_tc: {e:?}"))?;
    Ok(())
}

/// Tensor-core NT dX (stage 5): `dX[M,K] = dY @ W^T` via mma.sync, typed RNE
/// overwrite. Same TC contract as [`sgemm_bi_forward_tc`]. Covers
/// M >= 128 && K_out >= 128.
pub fn sgemm_bi_backward_dx_tc(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(dx.dtype, "dX")?;
    if dx.dtype != dy.dtype || dy.dtype != w.dtype {
        return Err("sgemm_bi_backward_dx_tc: mixed dtypes not supported".into());
    }
    if !(batch >= 128 && n_in >= 128 && n_out >= 1) {
        return Err(format!(
            "sgemm_bi_backward_dx_tc: shape M={batch} K={n_in} N={n_out} below the TC tile gate"
        ));
    }
    let dt = dx.dtype;
    let alpha: f32 = 1.0;
    let m_i = batch as i32;
    let n_i = n_out as i32;
    let k_out_i = n_in as i32;
    let total_tiles = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(128);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (total_tiles, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(kernels.sgemm_nt_tc_typed.get(dt));
    b.arg(&dx.ptr);
    b.arg(&dy.ptr);
    b.arg(&w.ptr);
    b.arg(&alpha);
    b.arg(&m_i);
    b.arg(&n_i);
    b.arg(&k_out_i);
    unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nt_tc: {e:?}"))?;
    Ok(())
}

/// Typed NN forward: `Y = X @ W + bias` (bias f32, fused into the kernel).
pub fn sgemm_bi_forward_typed(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    y: TypedPtr,
    x: TypedPtr,
    w: TypedPtr,
    bias_ptr: CUptr, // f32, 0 = none
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(y.dtype, "output")?;
    if x.dtype != y.dtype || w.dtype != y.dtype {
        return Err("sgemm_bi_forward_typed: mixed dtypes not supported".into());
    }
    let dt = y.dtype;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m_i = batch as i32;
    let n_i = n_out as i32;
    let k_i = n_in as i32;

    // GEMV N=1.
    if n_out == 1 && batch >= 1 && n_in >= 32 {
        let lda_i = n_in as i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch as u32).div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nn_gemv_typed.get(dt));
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&lda_i);
        b.arg(&ldy_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_gemv typed: {e:?}"))?;
        return Ok(());
    }

    // Ultra-thin M (1..32).
    if (1..32).contains(&batch) && (32..=2048).contains(&n_in) && n_out >= 32 {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n_out as u32).div_ceil(32), batch as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n_in * std::mem::size_of::<f32>()) as u32,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nn_ultra_thin_typed.get(dt));
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_ultra_thin typed: {e:?}"))?;
        return Ok(());
    }

    // Narrow N (2..=127): small tile for batch <= 64, big-narrow otherwise.
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let post_op: i32 = 0;
        let small = batch <= 64;
        let (grid, block, func) = if small {
            (
                (batch as u32).div_ceil(16) * (n_out as u32).div_ceil(16),
                64u32,
                kernels.sgemm_nn_narrow_small_typed.get(dt),
            )
        } else {
            (
                (batch as u32).div_ceil(64) * (n_out as u32).div_ceil(32),
                128u32,
                kernels.sgemm_nn_narrow_typed.get(dt),
            )
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(func);
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        b.arg(&post_op);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_narrow typed: {e:?}"))?;
        return Ok(());
    }

    // Big NN (stage 3): native typed twin of `sgemm_bi_nn`, fired exactly
    // where the f32 cascade would run Big (predicate-mirrored gates).
    if nn_routes_to_big(batch, n_in, n_out) {
        let total_tiles = (batch as u32).div_ceil(128) * (n_out as u32).div_ceil(128);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nn_big_typed.get(dt));
        b.arg(&y.ptr);
        b.arg(&x.ptr);
        b.arg(&w.ptr);
        b.arg(&bias_ptr);
        b.arg(&alpha);
        b.arg(&beta);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_i);
        b.arg(&k_i);
        b.arg(&n_i);
        b.arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nn_big typed: {e:?}"))?;
        return Ok(());
    }

    Err(format!(
        "sgemm_bi_forward_typed: Big/Slim buckets not yet implemented (stage 3) — \
         shape M={batch} K={n_in} N={n_out}. Disable the batch-invariant flag for \
         this configuration."
    ))
}

/// Typed TN dW: `dW[K_out=n_in, n_out] += X^T @ dY` into the f32 master grad.
pub fn sgemm_bi_backward_dw_typed(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dw_ptr: CUptr, // f32 master, accumulated
    dy: TypedPtr,
    x_saved: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(dy.dtype, "dY")?;
    if x_saved.dtype != dy.dtype {
        return Err("sgemm_bi_backward_dw_typed: mixed dtypes not supported".into());
    }
    let dt = dy.dtype;
    let alpha: f32 = 1.0;

    // GEMV N=1.
    if n_out == 1 && n_in >= 4 && batch >= 32 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let lda_i = n_in as i32;
        let ldy_i: i32 = 1;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n_in as u32).div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(kernels.sgemm_tn_gemv_typed.get(dt));
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&lda_i);
        b.arg(&ldy_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_tn_gemv typed: {e:?}"))?;
        return Ok(());
    }

    // Narrow N (2..=127).
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let m_red_i = batch as i32;
        let k_out_i = n_in as i32;
        let n_i = n_out as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (n_in as u32).div_ceil(64) * (n_out as u32).div_ceil(32),
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(kernels.sgemm_tn_narrow_typed.get(dt));
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_red_i);
        b.arg(&k_out_i);
        b.arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_tn_narrow typed: {e:?}"))?;
        return Ok(());
    }

    // Big TN (stage 3): native typed twin of `sgemm_bi_tn`. dW stays f32 +=.
    if tn_routes_to_big(batch, n_in, n_out) {
        let alpha: f32 = 1.0;
        let m_red_i = batch as i32;
        let k_out_i = n_in as i32;
        let n_i = n_out as i32;
        let total_tiles = (n_in as u32).div_ceil(128) * (n_out as u32).div_ceil(128);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let mut b = stream.launch_builder(kernels.sgemm_tn_big_typed.get(dt));
        b.arg(&dw_ptr);
        b.arg(&x_saved.ptr);
        b.arg(&dy.ptr);
        b.arg(&alpha);
        b.arg(&m_red_i);
        b.arg(&k_out_i);
        b.arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_tn_big typed: {e:?}"))?;
        return Ok(());
    }

    Err(format!(
        "sgemm_bi_backward_dw_typed: split-M/Slim buckets are upcast-fallback territory — \
         shape M={batch} K={n_in} N={n_out}."
    ))
}

/// Typed NT dX: `dX[batch, n_in] = dY[batch, n_out] @ W^T` (overwrite).
pub fn sgemm_bi_backward_dx_typed(
    stream: &Arc<cudarc::driver::CudaStream>,
    kernels: &GpuKernels,
    dx: TypedPtr,
    dy: TypedPtr,
    w: TypedPtr,
    dims: (usize, usize, usize),
) -> Result<(), String> {
    let (batch, n_in, n_out) = dims;
    require_half(dx.dtype, "dX")?;
    if dy.dtype != dx.dtype || w.dtype != dx.dtype {
        return Err("sgemm_bi_backward_dx_typed: mixed dtypes not supported".into());
    }
    let dt = dx.dtype;
    let alpha: f32 = 1.0;

    // GEMV N=1 (outer product).
    if n_out == 1 && batch >= 1 && n_in >= 1 {
        let m_i = batch as i32;
        let k_i = n_in as i32;
        let ldx_i = n_in as i32;
        let ldy_i: i32 = 1;
        let total = (batch * n_in) as u32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nt_gemv_typed.get(dt));
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&k_i);
        b.arg(&ldx_i);
        b.arg(&ldy_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nt_gemv typed: {e:?}"))?;
        return Ok(());
    }

    // Narrow reduction N (2..=127).
    if (2..=127).contains(&n_out) && batch >= 1 && n_in >= 1 {
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_out_i = n_in as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (
                (batch as u32).div_ceil(64) * (n_in as u32).div_ceil(32),
                1,
                1,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nt_narrow_typed.get(dt));
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_out_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nt_narrow typed: {e:?}"))?;
        return Ok(());
    }

    // Big NT (stage 3): native typed twin of `sgemm_bi_nt` (typed dX overwrite).
    if nt_routes_to_big(batch, n_in, n_out) {
        let alpha: f32 = 1.0;
        let m_i = batch as i32;
        let n_i = n_out as i32;
        let k_out_i = n_in as i32;
        let total_tiles = (batch as u32).div_ceil(128) * (n_in as u32).div_ceil(128);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (total_tiles, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 34 * 1024,
        };
        let mut b = stream.launch_builder(kernels.sgemm_nt_big_typed.get(dt));
        b.arg(&dx.ptr);
        b.arg(&dy.ptr);
        b.arg(&w.ptr);
        b.arg(&alpha);
        b.arg(&m_i);
        b.arg(&n_i);
        b.arg(&k_out_i);
        unsafe { b.launch(cfg) }.map_err(|e| format!("sgemm_bi_nt_big typed: {e:?}"))?;
        return Ok(());
    }

    Err(format!(
        "sgemm_bi_backward_dx_typed: split-N/Slim buckets are upcast-fallback territory — \
         shape M={batch} K={n_in} N={n_out}."
    ))
}
