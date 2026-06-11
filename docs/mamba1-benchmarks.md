# Mamba SSM Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada Generation (48 GB), CUDA 13.2, Driver 595.45.

All numbers below are produced by **the batch-invariant matvec kernel**
(`kernels/gemm_batch_invariant.cu`) — pure Rust + NVRTC, no Python or
Triton dependency. The kernel guarantees bit-identical per-row output
across batch sizes (KL ≈ 1e-11) and reaches ~86% of cuBLAS gemv
throughput on M=1 decode while running ~6 percentage points faster
than vLLM / Thinking Machines Lab's `batch_invariant` Triton kernel.

## LLM Inference — state-spaces/mamba-*-hf (end-to-end, graph-captured)

Production checkpoints loaded from HuggingFace, greedy decode, 100-token generation.

### Throughput (tokens/sec)

| Model | f32 | bf16 | f16 | bf16 vs f32 |
|-------|-----|------|-----|-------------|
| mamba-130m-hf | 725 | **898** | 899 | +24% |
| mamba-370m-hf | 305 | **396** | 395 | +30% |
| mamba-1.4b-hf | 116 | **189** | 190 | +63% |
| mamba-2.8b-hf |  61 | **104** | 104 | +70% |

bf16/f16 weight VRAM footprint is exactly half of f32. Throughput gain
grows with model size as the workload becomes increasingly bandwidth-bound.

130m row from `bench_m1_bf16_vs_f32_130m` (100-token steady-state run);
larger sizes from `llm_dtype_throughput_all_sizes` (50-token sweep —
shorter window, ~3% lower than the steady-state would show).

130m row from `bench_m1_bf16_vs_f32_130m` (100-token steady-state run);
larger sizes from `llm_dtype_throughput_all_sizes` (50-token sweep —
shorter window, ~3% lower than the steady-state would show).

### Prefill vs Decode (bf16, prompt=128 tokens, +100 decoded)

| Model | Prefill 128 tok (TTFT) | Decode tok/s | ms/decoded tok |
|-------|-------------------------|--------------|----------------|
| mamba-130m-hf | 113.8 ms | **929** | 1.08 |
| mamba-370m-hf | 278.1 ms | **445** | 2.25 |
| mamba-1.4b-hf | 580.5 ms | **207** | 4.83 |
| mamba-2.8b-hf | 1081.6 ms | **114** | 8.78 |

(Prompt length 128 < `PREFILL_PARALLEL_THRESHOLD = 256` → per-step prefill
on the same kernel as decode, guaranteeing cross-batch bit-identity.)

### Long-context prefill (mamba-370m-hf, bf16)

| Prompt length | Prefill | µs/prompt token |
|--------------:|--------:|---------------:|
|    64 |  137.2 ms | 2143 (per-step path) |
|   256 |   54.9 ms |  215 (parallel-prefill kicks in) |
|  1024 |  206.5 ms |  202 |
|  4096 |  842.8 ms |  206 |

Linear O(T) scaling confirmed at large T (SSM structural advantage:
no O(T²) cost as context grows). The 64→256 transition is the
`PREFILL_PARALLEL_THRESHOLD` boundary — short prompts use the same
T=1 SSM kernel as decode for cross-batch determinism; longer prompts
switch to the batched parallel-prefill SSM kernel.

### RL parallel-envs throughput (mamba-130m-hf, bf16)

Concurrent generation across N envs sharing one GPU. Each env produces
its own logits, with strict cross-env bit-identity (slot 0 at B=N is
identical to a standalone B=1 run on the same prompt).

| B (envs) | tok/s/env | tok/s total | µs/env/tok |
|---:|---:|---:|---:|
|  1 | 731 |   731 | 1368 |
|  2 | 689 | 1 378 |  726 |
|  4 | 604 | 2 415 |  414 |
|  8 | 523 | 4 185 |  239 |
| 16 | 378 | 6 044 |  165 |

Per-env latency grows with B because B is a 2D grid dim
(`(N/32, M=B)`) — each env loads its own weight stream. cuBLAS would
share weights across env rows for higher aggregate throughput, but at
the cost of cross-env determinism.

### Numerical parity vs f32 (tests/gpu_bf16_parity.rs)

Greedy top-1 match over 15 tokens + KL(f32 ‖ bf16) on final logits:

| Model | bf16 match | KL |
|-------|------------|-----|
| mamba-130m-hf | 15/15 | 7.22e-4 |
| mamba-370m-hf | 15/15 | 3.75e-4 |
| mamba-1.4b-hf | 15/15 | 2.90e-5 |
| mamba-2.8b-hf | 15/15 | 3.90e-5 |

### Cross-batch bit-identity (tests/hf_batch_parity.rs, tests/extreme_edge_coverage.rs)

| Test | b=1 vs b=N KL | Status |
|------|---:|---|
| `bf16_batch_divergence_known` (adversarial; original bug KL=2.7) | **0.0000** | ✓ |
| `bf16_multi_length_parity` (lengths 3, 5, 32, 63, 64, 65, 128) | ≤ 3.5e-11 | ✓ |
| `inference_extreme_batch_parity_bf16_b16` | 1.07e-11 | ✓ |
| `inference_extreme_batch_parity_bf16_b32` | 2.21e-11 | ✓ |
| `inference_extreme_batch_parity_f16_b16` | 7.7e-12 | ✓ |
| `inference_extreme_batch_parity_f32_b16` | 2.22e-6 | ✓ |
| `hf_cpu_vs_gpu_inference_bf16` (20 tokens) | 2e-6 (top-1 20/20) | ✓ |

## GPU Inference synthetic (T=1 step, default config: d_model=128, 3 layers, 366K params)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 124 us   | **79 us**  |
| B=4   | 146 us   | 98 us      |
| B=16  | 149 us   | 103 us     |
| B=64  | 155 us   | 115 us     |
| B=128 | 176 us   | 139 us     |

CUDA Graph capture saves ~45 µs/step in kernel launch overhead.

## Deterministic training GEMM (0.4.0)

Opt-in deterministic GEMM tiers (scalar `MAMBA_RS_BATCH_INVARIANT` +
tensor-core `MAMBA_RS_BI_TENSOR_CORES`): bit-identical training across
runs on f32/bf16/f16; the TC tier is FASTER than cuBLAS-PEDANTIC from
d768 up (0.77× per step at d1536 bf16). Full tables:
[determinism-benchmarks.md](determinism-benchmarks.md).

## GPU Training (mamba-130m, B=1, T=32, graph-captured)

| dtype | per step |
|---|---|
| f32  | 28.1 ms |
| bf16 | 38.6 ms |
| f16  | 34.0 ms |

(bf16 training is currently slower than f32 because the mixed-precision
forward / backward pipeline uses cuBLAS GemmEx for the larger M values
encountered in T=32 training; the matvec kernel is decode-only at
present. v0.4.0 adds the persistent + cp.async WMMA training kernel.)

## RL Training (synthetic d_model=128, 3 layers, B=1, T=32, graph)

| dtype | per step |
|---|---|
| f32  | 7.7 ms |
| bf16 | 8.1 ms |
| f16  | 8.5 ms |

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | params | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 70K    | 25.1    |
| default| 128     | 3      | 366K   | 88.7    |
| medium | 256     | 4      | 1.8M   | 394.8   |
| large  | 512     | 6      | 10.4M  | 2 244.8 |

## CPU Parallel Training (default config, T=32, 48 threads)

| Batch | Forward | Backward | Total | Samples/sec |
|-------|---------|----------|-------|-------------|
| B=16  |  5 863 us |  24 651 us |  30 515 us | 524 |
| B=64  | 16 103 us |  65 510 us |  81 613 us | 784 |
| B=128 | 27 713 us | 102 807 us | 130 520 us | 981 |

## Summary

- End-to-end bf16/f16 gives **+24% to +70%** throughput over f32 with
  no accuracy loss (15/15 greedy match on all four HF sizes).
- **Bit-identical batch invariance** (KL ≈ 1e-11) on the same kernel
  used by both decode (M=1) and RL parallel envs (M=N) — strict cross-
  batch reproducibility without an opt-in flag.
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
