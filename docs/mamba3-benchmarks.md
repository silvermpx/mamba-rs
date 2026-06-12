# Mamba-3 SISO Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada
Generation (48 GB), CUDA 13.2, Driver 595.45. Measured on mamba-rs 0.4.2.

> **Note**: all numbers below are against synthetic weights via
> `Mamba3Weights::init` — no public Mamba-3 SISO checkpoints exist yet
> (checked: HuggingFace `state-spaces` hosts only Mamba-1/Mamba-2).
> For end-to-end LLM inference benchmarks against production weights,
> see [mamba1-benchmarks.md](mamba1-benchmarks.md).

## GPU Inference (T=1 step, default config: d_model=128, 4 layers, nheads=16, headdim=16)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 138 us   | **87 us**  |
| B=4   | 152 us   | 103 us     |
| B=16  | 157 us   | 108 us     |
| B=64  | 167 us   | 122 us     |

CUDA Graph eliminates kernel launch overhead (~51 us saved per step).

## GPU synthetic LLM throughput (100-token greedy generation)

| dtype | eager | CUDA Graph |
|-------|------:|-----------:|
| f32  | 3 935 tok/s | **5 778 tok/s** |
| bf16 | 4 162 tok/s | **5 824 tok/s** |
| f16  | 4 170 tok/s | **5 837 tok/s** |

Synthetic default config (tiny model — numbers measure the step
pipeline, not a real LLM). From `bench_bf16_vs_f32::bench_m3_bf16_vs_f32_synthetic`.

## GPU Training (default config, B=1, T=32)

| | Time |
|---|---|
| Forward | 642 us |
| Backward | 1 141 us |
| Forward + Backward | 1 784 us |

Backward 1 534 → 1 141 µs vs 0.4.1: the training GEMMs ride the
tensor-core deterministic tier's new Tile64/BK=64 kernels
([determinism-benchmarks.md](determinism-benchmarks.md)).

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | nheads | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 16     | 12.3    |
| default| 128     | 4      | 16     | 64.5    |
| medium | 256     | 4      | 32     | 268     |
| large  | 512     | 6      | 64     | 1 973   |

## CPU Training (B=1, T=32, per layer)

| Config | d_model | layers | Forward | Backward | Total |
|--------|---------|--------|---------|----------|-------|
| small  | 64      | 2      | 205 us  | 869 us   | 1 074 us |
| default| 128     | 4      | 506 us  | 3 129 us | 3 635 us |
| medium | 256     | 4      | 1 646 us | 11 363 us | 13 009 us |
| large  | 512     | 6      | 7 291 us | 58 610 us | 65 901 us |

## CPU Parallel Training — RL workload pattern (small, 4 layers, T=32, 48 threads)

| Batch | fwd+bwd | steps/sec |
|------:|---------|----------:|
|   1   | 14.8 ms |  67.4 |
|   8   | 21.1 ms |  47.5 |
|  16   | 25.9 ms |  38.6 |
|  32   | 42.5 ms |  23.5 |
|  64   | 80.2 ms |  12.5 |
| 128   | 132.8 ms |  7.5 |

Linear scaling to B=64; larger batches approach memory-bandwidth limits.

## Mamba-3 vs Mamba SSM Comparison (default config, synthetic)

|   | Mamba-3 | Mamba SSM | Notes |
|---|--------:|--------:|-------|
| CPU Inference B=1 | **64.5 us** | 86.8 us | M3 faster — no conv1d, BLAS matvec |
| GPU Inference B=1 (Graph) | 87 us | 79 us | Similar (M3 has 4 layers vs M1 3) |
| GPU Training Fwd+Bwd | 1 784 us | 1 653 us | Near-parity since 0.4.2 (was 2 169 vs 1 640) |
| CPU Training Fwd+Bwd | 3 635 us | 14 859 us | M3 **4.1× faster** — no conv1d backward |

## Key Differences from Mamba SSM

- **No conv1d** — removed entirely (simpler and much faster CPU training).
- **Input-dependent A matrix** (per-head, clamped via `a_floor`).
- **Trapezoidal integration** (`alpha`, `beta`, `gamma` with learned `lambda`).
- **RoPE** per-head angle accumulation in `[0, 2π)`.
- **Multi-head B/C** with per-group BCNorm.
- **4 persistent recurrent states** (SSM + K + V + angle) vs 2 in Mamba SSM
  (conv_state + ssm_state).
- Implemented via **47 CUDA kernels** across 5 `.cu` files.

## Optimizations

- SIMD SSM recurrence via `pulp` (CPU inference + training).
- BLAS matvec for `in_proj` / `out_proj` (CPU inference).
- CUDA Graph capture for GPU inference (~1.6× speedup).
- Flat weight buffer + `WeightSlice` for CUDA Graph safety.
- Zero heap allocations per inference step.
- `disable_event_tracking()` for CUDA Graph capture stability.
- **Rayon-parallel backward** with thread-local gradient accumulators
  and tree-reduce (see `src/mamba3_siso/cpu/parallel.rs`).
  Scales to ~4× on 8-core Mac, ~5× on 48-thread Xeon.

Reproduce:
```
cargo test --release --test m3_gpu_benchmark -- --ignored --nocapture
cargo test --release --test m3_cpu_benchmark -- --ignored --nocapture
cargo test --release --features "cuda hf" --test bench_bf16_vs_f32 \
    bench_m3_bf16_vs_f32_synthetic -- --ignored --nocapture
cargo test --release --features "cuda hf" --test rl_llm_bench \
    rl_ -- --ignored --nocapture
```

## Deterministic training GEMM (0.4.2)

The Mamba-3 trainer shares the deterministic GEMM layer with Mamba SSM:
both opt-in tiers (`MAMBA_RS_BATCH_INVARIANT`, `MAMBA_RS_BI_TENSOR_CORES`)
apply, with the same contracts and CUDA-Graph guards
(`presize_bi_upcast_scratch_for_train_m3`). Measurement tables:
[determinism-benchmarks.md](determinism-benchmarks.md).
