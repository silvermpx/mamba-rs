# Deterministic GEMM Benchmarks (0.4.1)

All numbers: RTX 6000 Ada (sm_89, 142 SMs), CUDA 13.2, driver 595.45,
`--release`, `--test-threads=1`, quiet GPU. The GEMM layer is shared by
Mamba SSM and Mamba-3 SISO — these results apply to both architectures.

## Tiers and contracts

| flag | tier | contract |
|---|---|---|
| (off) | cuBLAS | f32 → TF32 tensor cores; bf16/f16 → GemmEx `COMPUTE_32F_PEDANTIC` (CUDA cores, f32 accumulate). Run-to-run stable on one machine, NOT batch-invariant, no stability across cuBLAS versions. |
| `MAMBA_RS_BATCH_INVARIANT=1` | scalar deterministic | custom fixed-reduction-order kernels (`kernels/sgemm_bi.cu`). Training bit-identical across runs on every dtype; per-bucket batch invariance (same dispatch bucket → row 0 bit-identical across M); inference decode strictly all-M invariant via `matvec_bi` (KL ≈ 1e-12). bf16/f16 outputs are bit-identical to "upcast → f32 kernel → RNE downcast". |
| + `MAMBA_RS_BI_TENSOR_CORES=1` | tensor-core deterministic | `mma.sync.m16n8k16`, f32 accumulators, no atomics/splits. OWN numeric contract (TC reduction tree ≠ scalar FMA chain) — but runs are bit-identical to each other (incl. CUDA Graph capture/replay) and the forward is STRICTLY batch-invariant across all M. |

Accuracy cross-checks: bf16 scalar-tier training trajectory vs cuBLAS
PEDANTIC cosine 0.999999976 (5 steps); f32 vs TF32 0.999999996. TC tier
vs f32 reference on quantized inputs: bf16 cos 0.9999986, f16 0.99999998;
TC dW (f32 accumulate) cos 1.000000000.

## Training step cost (`MambaTrainer`, ms/step)

`tests/sgemm_bi_determinism.rs::bench_sgemm_bi_vs_tf32`

| model | dtype | cuBLAS baseline | scalar deterministic | + tensor cores |
|---|---|---:|---:|---:|
| d128 ×2L, B=16 T=64  | f32  | 2.060 (TF32) | 2.644 (1.28×) | — |
| d128 ×2L, B=16 T=64  | bf16 | 2.131 (PEDANTIC) | 2.549 (1.20×) | 2.345 (1.10×) |
| d128 ×2L, B=16 T=64  | f16  | 1.881 (PEDANTIC) | 2.580 (1.37×) | 2.372 (1.26×) |
| d256 ×4L, B=16 T=128 | f32  | 8.754 | 11.278 (1.29×) | — |
| d256 ×4L, B=16 T=128 | bf16 | 9.067 | 10.407 (1.15×) | 9.561 (1.05×) |
| d256 ×4L, B=16 T=128 | f16  | 7.775 | 10.431 (1.34×) | 9.589 (1.23×) |
| d768 ×4L, B=8 T=256  | f32  | 24.238 | 32.363 (1.34×) | — |
| d768 ×4L, B=8 T=256  | bf16 | 25.890 | 28.490 (1.10×) | **22.736 (0.88×)** |
| d768 ×4L, B=8 T=256  | f16  | 24.625 | 28.624 (1.16×) | **22.754 (0.92×)** |
| d1536 ×2L, B=4 T=256 | f32  | 14.152 (TF32) | 21.727 (1.54×) | — |
| d1536 ×2L, B=4 T=256 | bf16 | 17.951 | 19.537 (1.09×) | **13.599 (0.76×)** |
| d1536 ×2L, B=4 T=256 | f16  | 16.815 | 19.845 (1.18×) | **13.931 (0.83×)** |

Ratios are vs the cuBLAS baseline of the same dtype. Bold = deterministic
training FASTER than cuBLAS. The TC tier beats the scalar tier end-to-end
at every measured size (0.92× even at d128) but crosses below the cuBLAS
baseline only from ~d768 up — 128×128 TC tiles underfill on tiny GEMMs.
At d1536 the bf16 TC step (13.56 ms) is also faster than the f32 TF32
baseline (14.12 ms).

## Tensor-core tier — GEMM level (bf16, µs)

`tests/sgemm_bi_tc.rs::bench_tc_vs_scalar_paths`, vs the scalar
deterministic tier on the same shape:

| shape (M, K, N) | fwd scalar → TC | dW scalar → TC | dX scalar → TC |
|---|---:|---:|---:|
| 2048, 768, 3072 | 278.0 → 90.5 (**3.07×**) | 408.2 → 117.9 (**3.46×**) | 623.2 → 92.8 (**6.72×**) |
| 4096, 1536, 3072 | 1175.4 → 416.0 (2.83×) | 1239.5 → 388.5 (3.19×) | 1834.3 → 300.0 (6.11×) |
| 2048, 768, 512 | 101.1 → 30.0 (3.37×) | 127.2 → 62.1 (2.05×) | 114.8 → 26.7 (4.31×) |

90.5 µs at M2048 K768 N3072 ≈ 106 TFLOPS bf16 (~29 % of Ada dense peak,
within ~2× of cuBLAS-TC class). The dX win is largest because the scalar
NT path is the heaviest (W transpose through scratch + split-N reducer);
the TC kernel does it in one pass. Staging history: a naive prototype
(sync staging, manual fragment loads) reached only 1.1–1.3×; the jump to
3× came from 2-stage `cp.async` + `ldmatrix`(.trans) — a 3-stage pipeline
was also tried and measured FLAT, so the remaining ~2× to cuBLAS parity
needs BK=64 + occupancy work, not deeper pipelining.

## Scalar tier — typed Big / upcast-fallback cost (bf16, µs)

`tests/sgemm_bi_typed_parity.rs::bench_upcast_fallback_tax`, re-measured
in 0.4.1 with the native typed Big kernels actually executing (in 0.4.0
they compiled with wrong tile constants and silently fell back — see
CHANGELOG). Big-routed shapes now run the native kernel; split-K/Slim
shapes still run "upcast → f32 kernel → RNE downcast". The delta vs the
bare f32 kernel is the sync-staging cost (native) or the cast cost
(fallback) — empirically the two are equal within noise:

| shape (M, K, N) | route | f32 kernel | typed | overhead |
|---|---|---:|---:|---:|
| 2048, 768, 3072 | native Big | 244.8 | 293.8 | 20.0 % |
| 4096, 1536, 3072 | native Big | 1025.8 | 1394.3 | 35.9 % |
| 2048, 768, 512  | fallback (Slim) | 77.4 | 88.3 | 14.1 % |
| 256, 384, 512   | fallback (split-K) | 12.6 | 19.1 | 51.2 % |

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

# typed upcast-fallback tax:
cargo test --features cuda --release --test sgemm_bi_typed_parity \
  bench_upcast_fallback_tax -- --ignored --nocapture --test-threads=1
```

Contract tests (non-ignored, run in the default suite): bit-identity of
training across runs (`sgemm_bi_determinism.rs`), typed bit-parity vs the
f32 triad incl. a 60-shape dispatch-gate boundary sweep
(`sgemm_bi_typed_parity.rs`), TC determinism / strict all-M invariance /
accuracy (`sgemm_bi_tc.rs`), cross-batch inference parity
(`hf_batch_parity.rs`, `extreme_edge_coverage.rs`).
