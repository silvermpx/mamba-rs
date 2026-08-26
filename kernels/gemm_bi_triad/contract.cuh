// Batch-invariant deterministic GEMM — the multi-tile triad.
//
// The SGEMM in the file name is historical BLAS notation (S = single
// precision) and no longer describes the coverage: this file carries
// f32, bf16 and f16, on CUDA cores and on Tensor Cores. It is named for
// its STRUCTURE elsewhere in the API — the triad, i.e. the family that
// carries all three operand layouts and therefore the only one that can
// serve a backward. (See BiGemmFamily::Triad.)
//
// Based on siboehm's warptiling kernel (93.7% cuBLAS on A6000).
// Adapted for NVRTC compilation (no templates, no includes).
//
// Dtypes: the f32 kernels are the base contract; the bf16/f16 variants
// (typed section at the end of this file) share the same kernel
// structure with typed I/O and f32 accumulation throughout — a typed
// kernel is bit-identical to upcasting its inputs and running the f32
// kernel. The Tensor-Core variants further down are a SEPARATE numeric
// contract (mma.sync accumulation, not the scalar __fmaf_rn chain):
// deterministic and batch-invariant, but not bit-equal to the scalar
// triad. dW and bias stay f32 in every dtype (master-gradient
// invariant).
//
// Three operand layouts for training:
//   NN (forward):     C[M,N]  = alpha * A[M,K] @ B[K,N] + beta*C + bias
//   TN (backward dW): C[K,N] += alpha * A^T[K,M] @ B[M,N]
//   NT (backward dX): C[M,K]  = alpha * A[M,N] @ B^T[N,K]
//
// Architecture:
//   BM=128, BN=128, BK=16, 256 threads (8 warps)
//   Warp tile: WM=64, WN=32, arranged 2x4 (WMITER=2, WNITER=1 over 8 warps)
//   Thread tile: TM=8, TN=8
//   Per-thread output: 64 elements (16 rows x 4 cols × WMITER=2 = 64)
//   float4 coalesced global loads, A transposed in smem
//   SGB_GROUP_M per-arch L2 swizzle (8 sm_80, 16 sm_89+), deterministic K-reduction
//
// Source: github.com/siboehm/SGEMM_CUDA (kernel 10, warptiling)

// ============================================================================
// Typed (bf16/f16) variants — the sync-load buckets.
// ============================================================================
// Typed-triad contract:
//   - X / W / Y / dY / dX are T_ACT (typed I/O); loads upcast via to_f at the
//     read site, EXACTLY one RNE downcast (FROM_F) at the final store.
//   - dW and bias stay f32 (master gradients / f32 bias) — never rounded.
//   - All accumulation and the epilogue (alpha*acc + bias + beta*C) stay f32
//     with the same ascending-K __fmaf_rn chains and fixed reduce trees as
//     the f32 kernels: a typed kernel is bit-identical to "upcast inputs to
//     f32, run the f32 kernel" (bf16/f16 products are exact in f32).
//   - to_f / from_f_* come from _typed_prelude.cuh (inlined first in the
//     NVRTC blob; conversions are RNE, no FTZ — see kernels.rs flags).

// ============================================================================
// Tensor-core deterministic NN forward (bi_tensor_cores tier).
// ============================================================================
// mma.sync.aligned.m16n8k16 with f32 accumulators. SEPARATE numeric contract
// from the scalar triad (TC reduction tree, not the ascending-K FMA chain) —
// fully deterministic (fixed K order, fixed fragment/tile assignment, no
// atomics, no split-K) and batch-invariant across ALL M: each output
// element's entire K-reduction lives in one warp, independent of gridDim/M.
//
// cp.async implementation:
//   - As[m][k] (row-major) AND Bs[k][n] (row-major, global layout) are both
//     16B-chunk contiguous -> 2-stage cp.async pipeline with 4-operand
//     zero-fill for tails (bit-exact vs scalar zero stores). B fragments
//     come from ldmatrix.x2.TRANS of the k-major tile (delivers the
//     col-major k16n8 fragment without a staging transpose).
//   - Pads keep every ldmatrix row chunk in a distinct 4-bank group:
//     A row stride 72 halves (36 words ≡ 4 mod 8), B row stride 136 halves
//     (68 words ≡ 4 mod 8). Row bases are 16B-aligned (144 B / 272 B).
//   - Scalar staging fallback (uniform branch) when lda/ldb % 8 != 0.
//   - Smem (BK=64): NN 71 680 B / TN 69 632 B / NT 73 728 B — beyond the
//     48 KB static cap, so all three use dynamic smem with the
//     MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in set at module load
//     (modules.rs); launch passes the exact per-kernel byte count.
//     BK=64 halves the wait_group/__syncthreads boundary count per CTA
//     vs BK=32 (the measured per-boundary cost dominated the gap to
//     cuBLAS-TC; see internal/tc-bk64-blueprint.md).
//
// Geometry: CTA 256 threads = 8 warps as 2x4; BM=BN=128 BK=64; warp tile
// 64x32 = 4 m-frags(16) x 4 n-frags(8); bias pre-seeded into the f32
// accumulators (alpha must be 1.0 with bias); one RNE downcast at store.
//
// Fragment thread maps (PTX ISA m16n8k16, 16-bit A/B, .row.col):
//   lane L: g = L>>2, t = L&3
//   A: a0={(g,2t),(g,2t+1)} a1={(g+8,..)} a2={(g,2t+8),..} a3={(g+8,2t+8),..}
//   B: b0={(2t,g),(2t+1,g)} b1={(2t+8,g),(2t+9,g)}
//   C: c0=(g,2t) c1=(g,2t+1) c2=(g+8,2t) c3=(g+8,2t+1)
// A x4: lanes 0-7/8-15/16-23/24-31 -> (rows 0-7,k0)/(rows 8-15,k0)/
// (rows 0-7,k0+8)/(rows 8-15,k0+8). B x2.trans: lanes 0-7/8-15 -> stored
// rows (k0..k0+7)/(k0+8..k0+15) at column n0; .trans delivers M^T fragments
// = the col-major b-frags.
