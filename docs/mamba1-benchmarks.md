# Mamba SSM Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada
Generation (48 GB), CUDA 13.2, Driver 595.45. Measured on mamba-rs 0.4.2.

All decode numbers below are produced by **the batch-invariant matvec kernel**
(`kernels/gemm_batch_invariant.cu`) — pure Rust + NVRTC, no Python or
Triton dependency. The kernel guarantees bit-identical per-row output
across batch sizes (KL ≈ 1e-11) and reaches ~86% of cuBLAS gemv
throughput on M=1 decode while running ~6 percentage points faster
than vLLM / Thinking Machines Lab's `batch_invariant` Triton kernel.
Parallel prefill rides the deterministic GEMM tiers
([determinism-benchmarks.md](determinism-benchmarks.md)).

## LLM Inference — state-spaces/mamba-*-hf (end-to-end, graph-captured)

Production checkpoints loaded from HuggingFace, greedy decode, 100-token generation.

### Throughput (tokens/sec)

| Model | f32 | bf16 | f16 | bf16 vs f32 |
|-------|-----|------|-----|-------------|
| mamba-130m-hf | 725 | **1 029** | 1 028 | +42% |
| mamba-370m-hf | 304 | **435** | 435 | +43% |
| mamba-1.4b-hf | 116 | **205** | 203 | +77% |
| mamba-2.8b-hf |  61 | **111** | 110 | +82% |

bf16/f16 weight VRAM footprint is exactly half of f32. Throughput gain
grows with model size as the workload becomes increasingly bandwidth-bound.

130m row from `bench_m1_bf16_vs_f32_130m` (100-token steady-state run);
larger sizes from `llm_dtype_throughput_all_sizes` (50-token sweep —
shorter window, a few percent below steady-state).

### Prefill vs Decode (bf16, prompt=128 tokens, +100 decoded)

| Model | Prefill 128 tok (TTFT) | Decode tok/s | ms/decoded tok |
|-------|-------------------------|--------------|----------------|
| mamba-130m-hf | 98.3 ms | **1 066** | 0.94 |
| mamba-370m-hf | 243.6 ms | **475** | 2.11 |
| mamba-1.4b-hf | 532.9 ms | **224** | 4.46 |
| mamba-2.8b-hf | 1009.2 ms | **122** | 8.19 |

(Prompt length 128 < `PREFILL_PARALLEL_THRESHOLD = 256` → per-step prefill
on the same kernel as decode, guaranteeing cross-batch bit-identity.)

### Long-context prefill (mamba-370m-hf, bf16)

| Prompt length | Prefill | µs/prompt token |
|--------------:|--------:|---------------:|
|    64 |  123.1 ms | 1923 (per-step path) |
|   256 |   31.8 ms |  124 (parallel-prefill kicks in) |
|  1024 |   87.9 ms |   86 |
|  4096 |  328.8 ms |   80 |

Linear O(T) scaling confirmed at large T (SSM structural advantage:
no O(T²) cost as context grows). The 64→256 transition is the
`PREFILL_PARALLEL_THRESHOLD` boundary — short prompts use the same
T=1 SSM kernel as decode for cross-batch determinism; longer prompts
switch to the batched parallel-prefill SSM path, whose GEMMs ride the
tensor-core deterministic tier since 0.4.2 (4096-token prefill:
843 → 329 ms vs 0.4.1, 2.6× faster).

### RL parallel-envs throughput (mamba-130m-hf, bf16)

Concurrent generation across N envs sharing one GPU. Each env produces
its own logits, with strict cross-env bit-identity (slot 0 at B=N is
identical to a standalone B=1 run on the same prompt —
`tests/gpu_batch_test.rs`).

| B (envs) | tok/s/env | tok/s total | µs/env/tok |
|---:|---:|---:|---:|
|  1 | 815 |   815 | 1227 |
|  2 | 439 |   878 | 1139 |
|  4 | 400 | 1 600 |  625 |
|  8 | 375 | 3 002 |  333 |
| 16 | 314 | 5 026 |  199 |

Per-env latency grows with B because B is a 2D grid dim
(`(N/32, M=B)`) — each env loads its own weight stream. cuBLAS would
share weights across env rows for higher aggregate throughput, but at
the cost of cross-env determinism.

### Numerical parity vs f32 (tests/gpu_bf16_parity.rs)

Greedy top-1 match over 15 tokens + KL(f32 ‖ bf16) on final logits:

| Model | bf16 match | KL |
|-------|------------|-----|
| mamba-130m-hf | 15/15 | 1.08e-3 |
| mamba-370m-hf | 15/15 | 3.55e-4 |
| mamba-1.4b-hf | 15/15 | 2.30e-5 |
| mamba-2.8b-hf | 15/15 | 6.50e-5 |

### Cross-batch bit-identity (tests/hf_batch_parity.rs, tests/extreme_edge_coverage.rs)

| Test | b=1 vs b=N KL | Status |
|------|---:|---|
| `bf16_batch_divergence_known` (adversarial; original bug KL=2.7) | top-1 match | ✓ |
| `bf16_multi_length_parity` (lengths 3, 5, 32, 63, 64, 65, 128) | ≤ 3.7e-11 | ✓ |
| `inference_extreme_batch_parity_bf16_b16` | 2.55e-12 | ✓ |
| `inference_extreme_batch_parity_bf16_b32` | 7.96e-11 | ✓ |
| `inference_extreme_batch_parity_f16_b16` | 9.83e-12 | ✓ |
| `inference_extreme_batch_parity_f32_b16` | 4.70e-11 | ✓ |
| `hf_cpu_vs_gpu_inference_bf16` (20 tokens) | 2e-6 (top-1 20/20) | ✓ |

## GPU Inference synthetic (T=1 step, default config: d_model=128, 3 layers, 366K params)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 123 us   | **79 us**  |
| B=4   | 145 us   | 99 us      |
| B=16  | 148 us   | 103 us     |
| B=64  | 156 us   | 115 us     |
| B=128 | 176 us   | 139 us     |

CUDA Graph capture saves ~45 µs/step in kernel launch overhead.

## Deterministic training GEMM (0.4.2)

Opt-in deterministic GEMM tiers (scalar `MAMBA_RS_BATCH_INVARIANT` +
tensor-core `MAMBA_RS_BI_TENSOR_CORES`): bit-identical training across
runs on f32/bf16/f16; the TC tier is at-or-near cuBLAS parity even on
d128/d256 models (1.04×/1.01× per step, bf16) and FASTER than
cuBLAS-PEDANTIC from d768 up (0.70× per step at d1536 bf16). Full
tables: [determinism-benchmarks.md](determinism-benchmarks.md).

## GPU Training (mamba-130m, B=1, T=32, graph-captured)

| dtype | per step |
|---|---|
| f32  | 28.4 ms |
| bf16 | 38.9 ms |
| f16  | 29.6 ms |

(bf16 training at B=1 T=32 is dominated by cuBLAS GemmEx PEDANTIC
matmuls on thin shapes; the deterministic TC tier — see above — is the
fast path for serious training batches.)

## RL Training (synthetic d_model=128, 3 layers, B=1, T=32, graph)

| dtype | per step |
|---|---|
| f32  | 7.8 ms |
| bf16 | 8.0 ms |
| f16  | 8.3 ms |

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | params | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 70K    | 25.0    |
| default| 128     | 3      | 366K   | 86.8    |
| medium | 256     | 4      | 1.8M   | 369.1   |
| large  | 512     | 6      | 10.4M  | 2 284.6 |

## CPU Parallel Training (default config, T=32, 48 threads)

| Batch | Forward | Backward | Total | Samples/sec |
|-------|---------|----------|-------|-------------|
| B=16  |  6 763 us |  21 045 us |  27 808 us | 575 |
| B=64  | 16 386 us |  61 014 us |  77 400 us | 827 |
| B=128 | 28 260 us |  95 136 us | 123 396 us | 1 037 |

## Summary

- End-to-end bf16/f16 gives **+42% to +82%** throughput over f32 with
  no accuracy loss (15/15 greedy match on all four HF sizes).
- **Bit-identical batch invariance** (KL ≈ 1e-11) on the same kernel
  used by both decode (M=1) and RL parallel envs (M=N) — strict cross-
  batch reproducibility without an opt-in flag.
- Long-context prefill 2.6× faster in 0.4.2 (parallel-prefill GEMMs on
  the deterministic tensor-core tier).
- CUDA Graph capture saves ~45 µs/step launch overhead.
- Linear O(T) prefill scaling (4096-token prompt = same µs/tok as 1024-token).
- Zero heap allocations per inference step — all buffers pre-allocated at
  construction time.

Reproduce:
```
cargo test --release --features "cuda hf" --test rl_llm_bench \
    llm_dtype_throughput_all_sizes -- --ignored --nocapture
cargo test --release --features "cuda hf" --test rl_llm_bench \
    llm_prefill_vs_decode_all_models -- --ignored --nocapture
cargo test --release --features "cuda hf" --test rl_llm_bench \
    llm_long_context_prefill -- --ignored --nocapture
cargo test --release --features "cuda hf" --test rl_llm_bench \
    llm_batched_step_throughput -- --ignored --nocapture
cargo test --release --features "cuda hf" --test bench_bf16_vs_f32 \
    -- --ignored --nocapture
cargo test --release --features "cuda hf" --test gpu_bf16_parity \
    test_gpu_lm_bf16_matches_f32_all_cached_models -- --ignored --nocapture
cargo test --release --features "cuda hf" --test hf_batch_parity \
    -- --ignored --nocapture --test-threads=1
cargo test --release --features "cuda hf" --test extreme_edge_coverage \
    -- --ignored --nocapture --test-threads=1
```
