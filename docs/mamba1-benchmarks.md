# Mamba SSM Benchmarks

Unless a heading names another board, the numbers were taken on an
Intel Xeon Gold 5412U (48 threads) with an NVIDIA RTX 6000 Ada Generation
(48 GB), CUDA 13.2, driver 595.45. Every table is a measurement of the
release, board and toolkit named in its heading; the tables are kept as
history and are not remeasured for each release. The decode numbers were
taken with the batch-invariant kernels of the Inference family, which on
0.6.x were an opt-in and in 0.7.0 are what the default
`GemmMode::Deterministic` selects for a model context; a row's output is
bit-identical at every batch size (cross-batch KL below 1e-10). The
current kernel comparisons against cuBLAS are in
[determinism-benchmarks.md](determinism-benchmarks.md) and the modes in
[gemm-modes.md](gemm-modes.md).

## Serving prefill — classifier page shape (0.6.4, RTX 5090, CUDA 13.0)

Shape: B=1, T=4621, d_model=384, 24 layers, f32 weights, cuBLAS+TF32
(the production serve tier); one page = one prefill + one 1.5 KB pooled
download. Bit-identical outputs across 0.6.3 -> 0.6.4
(16-cell prefill hash gate).

| lane | 0.6.3 | 0.6.4 |
|---|---|---|
| pooled prefill, graph replay | 29.34 ms/page (34.1 pages/s) | **10.83 ms/page (92.4 pages/s)** |
| pooled prefill, eager | 29.43 ms/page | 11.14 ms/page |
| full-temporal prefill, eager | 30.10 ms/page | 11.53 ms/page |

What changed: the convolution is tiled over the sequence (the previous
per-channel loop left 146 of 170 SMs idle at batch 1), the gating is fused
into the scan's store, the B and C gathers are staged in time-major
order, the scan's shared memory is sized from the run-time `d_state`, and
the RMSNorm keeps its row in registers. What remains is the parallel scan
itself plus the four forward GEMMs.

## Training step — production shape (2x RTX 5090, CUDA 13.0)

d_model 384, 24 layers, B=8, T=1300, bf16, deterministic GEMMs with
tensor cores, graph replay; the step time is for one GPU of the pair. A
kernel pass took this step from 441.4 to 131.5 ms (-70%) with run-to-run
bit determinism kept throughout (one deliberate change of bit family,
recorded in the changelog). The activation tape of the scan is gone
(-12.3 GB at this shape), so a micro-batch of 32 fits a 32 GB card.

| Stage | ms/step |
|-------|--------:|
| baseline | 441.4 |
| + conv register window, tape kills, tap-split | 261.9 |
| + slim h tape (in-backward replay) | 247.4 |
| + T-major B/C layout | ~169 |
| + small-K dW split-M | 155.4 |
| + T-tiled conv dw/db | 142.6 |
| + d-group dB/dC fold | **131.5** |

Isolated ledger at 131.5 (x24-layer ms): scan fwd 10.7 / bwd ~20 plus
fold partials, backward GEMMs 15.4 (dt_proj 3.5), conv dw 2.6 / dx
1.5, dB/dC reducer ~2.5.

## Training step — production shape, the 0.7.0 kernel pass (RTX 6000 Ada, CUDA 13.2)

Same shape as above (d_model 384, 24 layers, B=8, T=1300, graph replay),
one board, the deterministic GEMMs of 0.7.0 in both columns; the pass
touched only the kernels around them and kept every output bit-identical
to the previous release (the digest suites of the decode step, the scan
forward and backward, the convolution and whole training runs were
recorded on both trees and match). Milliseconds per step, median of four
mirrored runs:

| tree | bf16 (tensor cores) | f32 |
|------|--------------------:|----:|
| 0.7.0 before the pass | 118.1 | 227.4 |
| 0.7.0 | 115.0 | 204.7 |

Per kernel, per launch, at this shape: the fold backward 2.06 to 1.68 ms
(bf16) and 3.76 to 2.79 ms (f32), the conv backward 286 to 245 µs (one
kernel per tile, no pre-activation tape), the B/C reduction 222 to 206 µs,
the conv burn-in forward 98 to 87 µs, the column sums over the batch 165
to 215 µs down to 61 µs. The scan forward is unchanged at 0.81 ms.

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

(A 128-token prompt is below the 256-token threshold of the parallel
prefill, so it runs step by step on the decode kernels, which keeps the
cross-batch bit identity.)

### Long-context prefill (mamba-370m-hf, bf16)

| Prompt length | Prefill | µs/prompt token |
|--------------:|--------:|---------------:|
|    64 |  123.1 ms | 1923 (per-step path) |
|   256 |   31.8 ms |  124 (parallel-prefill kicks in) |
|  1024 |   87.9 ms |   86 |
|  4096 |  328.8 ms |   80 |

The cost per token is flat at large T, as expected of a state-space
model. Prompts shorter than 256 tokens run step by step on the decode
kernels, which keeps the cross-batch bit identity; longer prompts switch
to the batched parallel prefill, whose GEMMs run on the deterministic
tensor-core kernels (4096-token prefill: 843 ms on the scalar kernels,
329 ms on the tensor-core kernels).

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

Per-environment latency grows with B because each environment's row is
its own block and loads its own weight stream. cuBLAS would share the
weights across rows for higher aggregate throughput; this kernel keeps
every environment's bits independent of the batch it is in.

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

## GPU Training (mamba-130m, B=1, T=32, graph-captured)

| dtype | per step |
|---|---|
| f32  | 28.4 ms |
| bf16 | 38.9 ms |
| f16  | 29.6 ms |

(Measured on 0.6.x with the cuBLAS Pedantic lane; bf16 training at B=1
T=32 is dominated by thin-shape GEMMs.)

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

- bf16 and f16 give 42 to 82 percent more decode throughput than f32 on
  the four HuggingFace sizes; 15 of 15 greedy tokens match f32 and the
  final-logit KL is at most 1.1e-3.
- Batch invariance: a row's bits are the same at batch 1 and at batch N
  (cross-batch KL below 1e-10) on the kernels that serve both decode and
  parallel environments, the default deterministic mode in 0.7.0.
- Long prompts prefill on the deterministic tensor-core kernels: a
  4096-token prompt takes 329 ms against 843 ms on the scalar kernels.
- CUDA Graph capture saves about 45 µs of launch overhead per step.
- Prefill cost per token is flat from 1024 to 4096 tokens.
- All GPU buffers are allocated at construction; a step allocates nothing
  by design.

Reproduce:
```
cargo test --release --features "cuda hf qualification" --test rl_llm_bench \
    llm_dtype_throughput_all_sizes -- --ignored --nocapture
cargo test --release --features "cuda hf qualification" --test rl_llm_bench \
    llm_prefill_vs_decode_all_models -- --ignored --nocapture
cargo test --release --features "cuda hf qualification" --test rl_llm_bench \
    llm_long_context_prefill -- --ignored --nocapture
cargo test --release --features "cuda hf qualification" --test rl_llm_bench \
    llm_batched_step_throughput -- --ignored --nocapture
cargo test --release --features "cuda hf qualification" --test bench_bf16_vs_f32 \
    -- --ignored --nocapture
cargo test --release --features "cuda hf" --test gpu_bf16_parity \
    test_gpu_lm_bf16_matches_f32_all_cached_models -- --ignored --nocapture
cargo test --release --features "cuda hf" --test hf_batch_parity \
    -- --ignored --nocapture --test-threads=1
cargo test --release --features "cuda hf" --test extreme_edge_coverage \
    -- --ignored --nocapture --test-threads=1
```
