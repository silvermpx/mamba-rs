// Batch-invariant bf16/f16/f32 GEMM — the single FIXED-TILE family
// (BiGemmFamily::Fixed). Forward-only NN; the triad in gemm_bi_triad.cu
// carries the backward layouts.
//
// Problem: cuBLAS `cublasGemmEx` selects different algorithms per M
// (split-K, tile shape, reduction order). `Y = X @ W` at M=1 vs M=20
// produces sub-ULP differences that amplify through 24 SSM layers
// (observed KL ≈ 0.03, occasional top-1 flip on adversarial prompts).
//
// This kernel is batch-invariant by construction: fixed 64x64x32 tile,
// NO split-K, fixed K-reduction order. The K-reduction tree for
// `C[i, j]` depends ONLY on `A[i, :]` and `B[:, j]`, never on other
// rows of A. Therefore `C[i, j]` is bit-identical whether A has 1, 5,
// or 1024 rows.
//
// Inner GEMM uses Tensor Cores via nvcuda::wmma (m16n16k16 fragments,
// f32 accumulator). The MMA instruction itself is deterministic at
// the hardware level (MMA-Sim arXiv:2511.10909 — 1M random inputs
// bit-identical between simulator and hardware). Non-determinism in
// cuBLAS comes from heuristic algo/Split-K selection, NOT from MMA;
// fixing the tile + Split-K=1 + f32 accumulator is sufficient for
// batch invariance even with Tensor Cores enabled.
//
// Recipe matches vLLM `batch_invariant.bmm_kernel_persistent` and
// Thinking Machines Lab `batch_invariant_ops` (both Triton; the inner
// `tl.dot` lowers to `mma.sync` in PTX). Here transliterated to plain
// CUDA via the WMMA C++ API — mamba-rs has no Python/Triton dep.
//
//   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32
//   GROUP_M = 8     (L2 swizzle)
//   SPLIT_K = 1     (critical — split-K is the root cause)
//   8 warps/CTA arranged 4 (warp_M) × 2 (warp_N)
//   Per warp: 16M × 32N = 1 frag-M × 2 frag-N (m16n16k16 each)
//   f32 accumulator fragments throughout
//
// Semantics (unchanged from CUDA-core version):
//   A: [M, K]  row-major, element type T_IO  (bf16 or f16)
//   B: [K, N]  row-major, element type T_IO
//   C: [M, N]  row-major, element type T_OUT (bf16 / f16 / f32)
//   bias: nullable [N] f32
//   C = alpha * (A @ B) + beta * C + bias
//
// Launch (unchanged — host dispatcher in src/mamba_ssm/gpu/blas.rs):
//   grid  = ((M/BLOCK_M) * (N/BLOCK_N), 1, 1)  flat — swizzled in-kernel
//   block = (256, 1, 1)
//   smem  = 0 dynamic (all buffers are static __shared__; the TC tile
//           holds 24 KB static - smem_a 4K + smem_b 4K + smem_acc 16K -
//           and the f32 FFMA tile 16 KB). The dynamic K-buffer belongs
//           to matvec_bi_* alone.

#include "_typed_prelude.cuh"
#include <mma.h>
#include <cuda_pipeline.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define GROUP_M 8
#define THREADS 256
#define WARPS_PER_CTA 8     // THREADS / 32
#define WARPS_M 4           // 4 warps along M; 4*16 = 64 = BLOCK_M
#define WARPS_N 2           // 2 warps along N; 2*32 = 64 = BLOCK_N
#define FRAG_M 16
#define FRAG_N 16
#define FRAG_K 16
#define WARP_FRAGS_N 2      // each warp owns 2 N-fragments (covers 32 N cols)
#define K_TILES 2           // BLOCK_K / FRAG_K

using namespace nvcuda;

// --- typed zero helpers (used for OOB padding) --------------------------
__device__ __forceinline__ __nv_bfloat16 zero_bf16() {
    return __float2bfloat16(0.0f);
}
__device__ __forceinline__ __half zero_f16() {
    return __float2half(0.0f);
}
__device__ __forceinline__ float zero_f32() { return 0.0f; }

// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32→f32 instantiation we keep the original CUDA-core inner loop.
// f32 inference was never the regression source (cuBLAS f32 path was also
// CUDA cores) so this path is unchanged in performance vs prior commit.
// --- f32 path: emulate Tensor Core via per-element FMA ------------------
// Tensor Cores on Ada do NOT accept f32 inputs (only bf16/f16/tf32). For
// the f32->f32 instantiation we keep the CUDA-core inner loop - the same
// hardware path cuBLAS takes for f32.
//
// The 64x64x32 tile with 256 threads and 4x4 outputs each is a measured
// optimum for narrow-N inference projections; wider tiles, fewer threads
// and a transposed-A layout were all measured slower at those shapes.
// The launcher's block/thread constants MUST equal the ones here - a
// launch that disagrees fills part of the tile and returns plausible
// garbage. `gemm_bi_fixed_correctness` is the gate.
// ---------------------------------------------------------------------------
// GBF safety layer (adopted from the audited fix set; every item is
// bit-preserving - the packed store uses the same per-element RNE, the
// cp.async source clamp changes address formation only, and the base-
// alignment gates route misaligned operands to the scalar stage that
// produces identical smem bytes):
//   - a 16-byte cp.async source pointer is FORMED only when bytes > 0
//     (out-of-object address formation is UB even unread);
//   - the fast stage requires 16B-aligned operand BASES, not just
//     8-element strides (a 2-byte-aligned typed subview with an even
//     stride otherwise stages wrong bytes silently);
//   - the packed pair store fires only on a 4-byte-aligned destination
//     (PTX faults or silently masks misaligned 32-bit stores).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ bool gbf_aligned16(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 15ull) == 0ull;
}
static __device__ __forceinline__ bool gbf_aligned4(const void* p) {
    return (reinterpret_cast<unsigned long long>(p) & 3ull) == 0ull;
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __nv_bfloat16* dst, float v0, float v1) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
}
static __device__ __forceinline__ void gbf_store_pair_rne(
    __half* dst, float v0, float v1) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(v0, v1);
}
