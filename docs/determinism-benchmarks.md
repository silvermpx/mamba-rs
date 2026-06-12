# Deterministic GEMM Benchmarks (0.4.2)

All numbers: RTX 6000 Ada (sm_89, 142 SMs), CUDA 13.2, driver 595.45,
`--release`, `--test-threads=1`, quiet GPU. The GEMM layer is shared by
Mamba SSM and Mamba-3 SISO — these results apply to both architectures.

## Tiers and contracts

| flag | tier | contract |
|---|---|---|
| (off) | cuBLAS | f32 → TF32 tensor cores; bf16/f16 → GemmEx `COMPUTE_32F_PEDANTIC` (CUDA cores, f32 accumulate). Run-to-run stable on one machine, NOT batch-invariant, no stability across cuBLAS versions. |
| `MAMBA_RS_BATCH_INVARIANT=1` | scalar deterministic | custom fixed-reduction-order kernels (`kernels/sgemm_bi.cu`). Training bit-identical across runs on every dtype; per-bucket batch invariance (same dispatch bucket → row 0 bit-identical across M); inference decode strictly all-M invariant via `matvec_bi` (KL ≈ 1e-12). bf16/f16 outputs are bit-identical to "upcast → f32 kernel → RNE downcast". |
| + `MAMBA_RS_BI_TENSOR_CORES=1` | tensor-core deterministic | `mma.sync.m16n8k16`, f32 accumulators, no atomics/splits. OWN numeric contract (TC reduction tree ≠ scalar FMA chain) — but runs are bit-identical to each other (incl. CUDA Graph capture/replay) and the forward is STRICTLY batch-invariant across all M. Two kernel families — 128×128 tiles (256 thr, dynamic smem) and 64×64 tiles (128 thr, static smem) — that are BIT-IDENTICAL per output element (same ascending BK=64 reduction slabs, same mma chain, same tail zero-fill), so the shape-only tile routing never changes output bits. |

Accuracy cross-checks: bf16 scalar-tier training trajectory vs cuBLAS
PEDANTIC cosine 0.999999976 (5 steps); f32 vs TF32 0.999999996. TC tier
vs f32 reference on quantized inputs: bf16 cos 0.9999986, f16 0.99999998;
TC dW (f32 accumulate) cos 1.000000000.

## Training step cost (`MambaTrainer`, ms/step)

`tests/sgemm_bi_determinism.rs::bench_sgemm_bi_vs_tf32`

| model | dtype | cuBLAS baseline | scalar deterministic | + tensor cores |
|---|---|---:|---:|---:|
| d128 ×2L, B=16 T=64  | f32  | 2.053 (TF32) | 2.637 (1.28×) | — |
| d128 ×2L, B=16 T=64  | bf16 | 2.124 (PEDANTIC) | 2.540 (1.20×) | 2.199 (1.04×) |
| d128 ×2L, B=16 T=64  | f16  | 1.878 (PEDANTIC) | 2.572 (1.37×) | 2.228 (1.19×) |
| d256 ×4L, B=16 T=128 | f32  | 8.724 | 11.235 (1.29×) | — |
| d256 ×4L, B=16 T=128 | bf16 | 9.029 | 10.359 (1.15×) | 9.109 (1.01×) |
| d256 ×4L, B=16 T=128 | f16  | 7.737 | 10.386 (1.34×) | 9.146 (1.18×) |
| d768 ×4L, B=8 T=256  | f32  | 24.090 | 32.218 (1.34×) | — |
| d768 ×4L, B=8 T=256  | bf16 | 25.706 | 28.462 (1.11×) | **21.650 (0.84×)** |
| d768 ×4L, B=8 T=256  | f16  | 24.328 | 28.460 (1.17×) | **21.693 (0.89×)** |
| d1536 ×2L, B=4 T=256 | f32  | 14.123 (TF32) | 21.576 (1.53×) | — |
| d1536 ×2L, B=4 T=256 | bf16 | 17.772 | 19.406 (1.09×) | **12.486 (0.70×)** |
| d1536 ×2L, B=4 T=256 | f16  | 16.761 | 19.740 (1.18×) | **12.818 (0.76×)** |

Ratios are vs the cuBLAS baseline of the same dtype. Bold = deterministic
training FASTER than cuBLAS. Since 0.4.2 (Tile64 family + BK=64 staging)
the TC tier is at-or-near parity even on the smallest models (d128 bf16
1.04×, d256 bf16 1.01×) and 16–30 % faster than cuBLAS from d768 up; at
d1536 the bf16 TC step (12.49 ms) also beats the f32 TF32 baseline
(14.12 ms). The remaining f16 small-model gap (1.18–1.19×) is the
cuBLAS-f16-PEDANTIC baseline being unusually fast at tiny sizes, plus
non-GEMM kernels dominating those steps.

## Tensor-core tier — GEMM level (bf16, µs)

`tests/sgemm_bi_tc.rs::bench_tc_vs_scalar_paths`, vs the scalar
deterministic tier on the same shape (0.4.2: BK=64 staging):

| shape (M, K, N) | fwd scalar → TC | dW scalar → TC | dX scalar → TC |
|---|---:|---:|---:|
| 2048, 768, 3072 | 293.2 → 84.1 (**3.49×**) | 400.2 → 101.5 (**3.95×**) | 424.6 → 83.6 (**5.08×**) |
| 4096, 1536, 3072 | 1131.8 → 352.8 (3.21×) | 1450.4 → 311.8 (4.65×) | 1212.7 → 301.5 (4.02×) |
| 2048, 768, 512 | 112.7 → 17.6 (**6.40×**) | 138.4 → 24.8 (5.59×) | 93.6 → 26.7 (3.51×) |

84.1 µs at M2048 K768 N3072 ≈ 115 TFLOPS bf16 (~32 % of Ada dense
peak); the M4096 forward reaches ~144 TFLOPS in an isolated sweep
(`step0` instrumentation: 83.7 µs / 267.8 µs on the two shapes). BK=64 staging (0.4.2)
halves the per-CTA barrier/wait_group boundaries vs the 0.4.1 BK=32
kernels and bought +8–11 % on top of the 0.4.1 numbers; deeper
pipelining was measured FLAT and 2-CTA/SM occupancy is register-blocked
(166 regs vs the 128 ceiling), so the remaining gap to cuBLAS-TC class
(~210–230 TFLOPS) needs fragment-reuse restructuring, not staging depth.

## Tile64 family — small/narrow shapes (bf16, µs)

`tests/sgemm_bi_tc.rs::bench_tc64_vs_tc128_small_shapes`. The 64×64-tile
twins (0.4.2) quadruple the CTA count on grids that underfill the GPU at
128×128, and cover the 64..127 output-dim band the 128 gate excluded:

| shape (M, K, N) | op | Tile128 | Tile64 | scalar bi |
|---|---|---:|---:|---:|
| 1024, 128, 512 (d128 in_proj) | fwd | 7.6 | **5.5** | 17.9 |
| | dW | 27.7 | **12.6** | 82.1 |
| | dX | 15.3 | **6.5** | 19.0 |
| 1024, 256, 128 (d128 out_proj) | fwd | 10.2 | **4.6** | 12.9 |
| | dW | 27.7 | **12.5** | 43.5 |
| | dX | 7.3 | **3.9** | 14.7 |
| 2048, 256, 1024 (d256 in_proj) | fwd | 18.0 | **16.6** | 49.9 |
| | dW | 50.8 | **22.0** | 101.7 |
| | dX | 27.0 | **13.1** | 64.5 |

Dispatch (`tc_pick_tile`): 128-tiles when the grid has ≥ 72 CTAs,
64-tiles otherwise and for the 64..127 band. Legal under the strict
all-M invariance contract because the two families are bit-identical
per output element (`tc64_and_tc128_bit_identical`). Narrow projections
of every model size (x_proj N=80, dt_proj K≤96) ride tensor cores via
Tile64 — that is why even d768/d1536 steps improved in 0.4.2.

## Scalar tier — typed Big / upcast-fallback cost (bf16, µs)

`tests/sgemm_bi_typed_parity.rs::bench_upcast_fallback_tax`. Big-routed
shapes run the native typed kernel; split-K/Slim shapes run "upcast →
f32 kernel → RNE downcast" (bit-identical by contract):

| shape (M, K, N) | route | f32 kernel | typed | overhead |
|---|---|---:|---:|---:|
| 2048, 768, 3072 | native Big | 245.7 | 294.3 | 19.8 % |
| 4096, 1536, 3072 | native Big | 917.4 | 1328.5 | 44.8 % |
| 2048, 768, 512  | fallback (Slim) | 77.2 | 88.3 | 14.4 % |
| 256, 384, 512   | fallback (split-K) | 13.4 | 20.0 | 49.4 % |

The native Big path matches the fallback's speed while eliminating the
f32 upcast scratch (~0.5 GB at 2.8b mixed) and 3 extra launches per
GEMM. With the TC tier on, none of this is on the bf16/f16 hot path.

## Reproducing

```sh
# training step, all 3 dtypes × {cuBLAS, scalar bi, +TC}:
cargo test --features cuda --release --test sgemm_bi_determinism \
  bench_sgemm_bi_vs_tf32 -- --ignored --nocapture --test-threads=1

# TC vs scalar at GEMM level (fwd/dW/dX):
cargo test --features cuda --release --test sgemm_bi_tc \
  bench_tc_vs_scalar_paths -- --ignored --nocapture --test-threads=1

# Tile64 vs Tile128 on small/narrow shapes:
cargo test --features cuda --release --test sgemm_bi_tc \
  bench_tc64_vs_tc128_small_shapes -- --ignored --nocapture --test-threads=1

# typed upcast-fallback tax:
cargo test --features cuda --release --test sgemm_bi_typed_parity \
  bench_upcast_fallback_tax -- --ignored --nocapture --test-threads=1
```

Contract tests (non-ignored, run in the default suite): bit-identity of
training across runs (`sgemm_bi_determinism.rs`), typed bit-parity vs the
f32 triad incl. a 60-shape dispatch-gate boundary sweep
(`sgemm_bi_typed_parity.rs`), TC determinism / strict all-M invariance /
accuracy / cross-tile bit-identity / gate boundary sweep / launch-reality
geometry (`sgemm_bi_tc.rs`), cross-batch inference parity
(`hf_batch_parity.rs`, `extreme_edge_coverage.rs`).
