# Mamba-3 SISO Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada Generation (48 GB), CUDA 12.8, Driver 595.45.

> **Note**: all numbers below are against synthetic weights via
> `Mamba3Weights::init` — no public Mamba-3 SISO checkpoints exist yet.
> For end-to-end LLM inference benchmarks against production weights,
> see [mamba1-benchmarks.md](mamba1-benchmarks.md).

## GPU Inference (T=1 step, default config: d_model=128, 4 layers, nheads=16, headdim=16)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 138 us   | **86 us**  |
| B=4   | 153 us   | 102 us     |
| B=16  | 158 us   | 108 us     |
| B=64  | 167 us   | 123 us     |

CUDA Graph eliminates kernel launch overhead (~52 us saved per step).

## GPU synthetic LLM throughput (d_model=256, 8 layers, headdim=16, synthetic weights)

Graph-captured generation, 100 tokens, greedy:

| dtype | tok/s | ms/tok |
|-------|------:|-------:|
| f32  | — (baseline) | — |
| bf16 | — | — |
| f16  | — | — |

(Populated by `cargo test --release --features "cuda hf" --test bench_bf16_vs_f32 \
bench_m3_bf16_vs_f32_synthetic -- --ignored --nocapture`.)

## GPU Training (default config, B=1, T=32)

| | Time |
|---|---|
| Forward | 635 us |
| Backward | 1 534 us |
| Forward + Backward | 2 169 us |

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | nheads | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 16     | 12.2    |
| default| 128     | 4      | 16     | 64.8    |
| medium | 256     | 4      | 32     | 270     |
| large  | 512     | 6      | 64     | 2 249   |

## CPU Training (B=1, T=32, per layer)

| Config | d_model | layers | Forward | Backward | Total |
|--------|---------|--------|---------|----------|-------|
| small  | 64      | 2      | 205 us  | 810 us   | 1 015 us |
| default| 128     | 4      | 525 us  | 3 085 us | 3 609 us |
| medium | 256     | 4      | 1 649 us | 11 218 us | 12 866 us |
| large  | 512     | 6      | 7 234 us | 60 011 us | 67 244 us |

## CPU Parallel Training — RL workload pattern (small, 4 layers, T=32, 48 threads)

| Batch | fwd+bwd | steps/sec |
|------:|---------|----------:|
|   1   | 18.7 ms |  53.5 |
|   8   | 26.2 ms |  38.2 |
|  16   | 36.1 ms |  27.7 |
|  32   | 52.5 ms |  19.0 |
|  64   | 99.1 ms |  10.1 |
| 128   | 156.5 ms |  6.4 |

Linear scaling to B=64; larger batches approach memory-bandwidth limits.

## Mamba-3 vs Mamba-1 Comparison (default config, synthetic)

|   | Mamba-3 | Mamba-1 | Notes |
|---|--------:|--------:|-------|
| CPU Inference B=1 | **64.8 us** | 83.6 us | M3 faster — no conv1d, BLAS matvec |
| GPU Inference B=1 (Graph) | 86 us | 79 us | Similar (M3 has 4 layers vs M1 3) |
| GPU Training Fwd+Bwd | 2 169 us | 1 640 us | M3 slower (4L vs 3L + RoPE + BCNorm) |
| CPU Training Fwd+Bwd | 3 609 us | 15 874 us | M3 **4.4× faster** — no conv1d backward |

## Key Differences from Mamba-1

- **No conv1d** — removed entirely (simpler and much faster CPU training).
- **Input-dependent A matrix** (per-head, clamped via `a_floor`).
- **Trapezoidal integration** (`alpha`, `beta`, `gamma` with learned `lambda`).
- **RoPE** per-head angle accumulation in `[0, 2π)`.
- **Multi-head B/C** with per-group BCNorm.
- **4 persistent recurrent states** (SSM + K + V + angle) vs 2 in Mamba-1
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
cargo test --release --features "cuda hf" --test rl_llm_bench \
    rl_ -- --ignored --nocapture
```
