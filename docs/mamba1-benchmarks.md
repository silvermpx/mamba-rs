# Mamba SSM Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada Generation (48 GB), CUDA 12.8, Driver 595.45.

## LLM Inference — state-spaces/mamba-*-hf (end-to-end, graph-captured)

Production checkpoints loaded from HuggingFace, greedy decode, 100-token generation.

### Throughput (tokens/sec)

| Model | f32 | bf16 | f16 | bf16 vs f32 |
|-------|-----|------|-----|-------------|
| mamba-130m-hf | 723 | **1 039** | 1 043 | +44% |
| mamba-370m-hf | 325 | **451** | 448 | +39% |
| mamba-1.4b-hf | 125 | **217** | 218 | +74% |
| mamba-2.8b-hf |  66 | **118** | 119 | +79% |

### Prefill vs Decode (bf16, prompt=128 tokens, +100 decoded)

| Model | Prefill 128 tok (TTFT) | Decode tok/s | ms/decoded tok |
|-------|-------------------------|--------------|----------------|
| mamba-130m-hf | 11.2 ms | **1 055** | 0.95 |
| mamba-370m-hf | 24.0 ms | **485** | 2.06 |
| mamba-1.4b-hf | 39.6 ms | **227** | 4.41 |
| mamba-2.8b-hf | 70.4 ms | **123** | 8.11 |

### Long-context prefill (mamba-370m-hf, bf16)

| Prompt length | Prefill | µs/prompt token |
|--------------:|--------:|---------------:|
|    64 |  15.8 ms | 247 |
|   256 |  31.5 ms | 123 |
|  1024 | 105.8 ms | 103 |
|  4096 | 435.8 ms | 106 |

Linear O(T) scaling confirmed at large T (SSM structural advantage: no O(T²)
cost as context grows).

### Numerical parity vs f32 (tests/gpu_bf16_parity.rs)

Greedy top-1 match over 15 tokens + KL(f32 ‖ bf16) on final logits:

| Model | bf16 match | KL |
|-------|------------|-----|
| mamba-130m-hf | 15/15 | 7.22e-4 |
| mamba-370m-hf | 15/15 | 3.75e-4 |
| mamba-1.4b-hf | 15/15 | 2.90e-5 |
| mamba-2.8b-hf | 15/15 | 3.90e-5 |

## GPU Inference synthetic (T=1 step, default config: d_model=128, 3 layers, 366K params)

| Batch | No Graph | CUDA Graph |
|-------|----------|------------|
| B=1   | 124 us   | **79 us**  |
| B=4   | 147 us   | 100 us     |
| B=16  | 149 us   | 104 us     |
| B=64  | 157 us   | 116 us     |
| B=128 | 177 us   | 140 us     |

CUDA Graph eliminates kernel launch overhead (~45 us saved per step).

## GPU Training (default config, B=1, T=32)

| | Time |
|---|---|
| Forward | 589 us |
| Forward + Backward | 1 640 us |

## CPU Inference (T=1 step, B=1)

| Config | d_model | layers | params | us/step |
|--------|---------|--------|--------|---------|
| small  | 64      | 2      | 70K    | 24.9    |
| default| 128     | 3      | 366K   | 83.6    |
| medium | 256     | 4      | 1.8M   | 361     |
| large  | 512     | 6      | 10.4M  | 2 285   |

## CPU Training (B=1, T=32)

| Config | d_model | layers | Forward | Backward | Total |
|--------|---------|--------|---------|----------|-------|
| small  | 64      | 2      | 1 195 us  | 1 964 us | 3 160 us |
| default| 128     | 3      | 4 074 us  | 11 801 us | 15 874 us |
| medium | 256     | 4      | 13 345 us | 54 683 us | 68 028 us |
| large  | 512     | 6      | 67 508 us | 571 632 us | 639 140 us |

## CPU Parallel Training (default config, T=32, 48 threads)

| Batch | Forward | Backward | Total | Samples/sec |
|-------|---------|----------|-------|-------------|
| B=16  | 10 524 us | 27 509 us | 38 033 us | 421 |
| B=64  | 21 021 us | 66 050 us | 87 071 us | 735 |
| B=128 | 35 593 us | 103 177 us | 138 770 us | 922 |

## Summary

- End-to-end bf16/f16 gives **+44% to +79%** throughput over f32 with no
  accuracy loss (15/15 greedy match on all four HF sizes).
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
cargo test --release --features "cuda hf" --test gpu_bf16_parity \
    test_gpu_lm_bf16_matches_f32_all_cached_models -- --ignored --nocapture
```
