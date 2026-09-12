# Mamba-3 SISO Benchmarks

Hardware: Ada server — Intel Xeon Gold 5412U (48 threads) + NVIDIA RTX 6000 Ada
Generation (48 GB), CUDA 13.2, Driver 595.45. GPU training-step
sections were measured on 2x RTX 5090 (CUDA 13.0) as marked. Every table
is a measurement of the release named in its heading and is kept as
history; the current kernel comparisons against cuBLAS are in
[determinism-benchmarks.md](determinism-benchmarks.md).

> **Note**: all numbers below are against synthetic weights via
> `Mamba3Weights::init` — no public Mamba-3 SISO checkpoints exist yet
> (checked: HuggingFace `state-spaces` hosts only Mamba-1/Mamba-2).
> For end-to-end LLM inference benchmarks against production weights,
> see [mamba1-benchmarks.md](mamba1-benchmarks.md).


## 0.7.1 — the sequential scan

RTX 6000 Ada, CUDA 13.2. The sequential lane only; the chunked lane and
every GEMM route are unchanged and their 0.7.0 tables stand.

Forward and backward of one layer, batch 64, eight heads of 32 over a
16-wide state, timed on the same buffers in one run:

| sequence | 0.7.0 | 0.7.1 | faster by |
|---|---:|---:|---:|
| T = 48 | 709 us | 178 us | 3.98x |
| T = 390 | 6969 us | 1918 us | 3.63x |
| T = 1440 | 25813 us | 7270 us | 3.55x |

A four-layer training step at the crate's default shape (d_model 128,
16 heads of 16, state 16, batch 1, T 32), from `m3_gpu_benchmark`:

| | 0.7.0 | 0.7.1 | faster by |
|---|---:|---:|---:|
| forward | 891.1 us | 701.4 us | 1.27x |
| backward | 1704.6 us | 645.5 us | 2.64x |
| forward and backward | 2595.7 us | 1346.8 us | 1.93x |

Decode is unchanged: the decode step keeps its 0.7.0 kernel, so the T=1
figures on that page stand as measured.

Bits: the 161-key bit ledger of the two trees, recorded on this board and
toolkit, is identical key for key.


## Training step — production shape (2x RTX 5090, CUDA 13.0)

d_model 384, 24 layers, B=8, T=1300, bf16, graph replay, one GPU of the
pair: 179.4 ms per step before the 0.6.4 kernel pass, 165.2 ms after.
The chunked-scan forward now runs one head per 128-thread block with a
triangle-packed decayed tile (it used to run a 32-thread block behind
32 KB of static shared memory at 5 to 6 percent occupancy), the bias add
and the RoPE rotation are one launch, and the per-layer residual round
trip is gone. All of it is bit-identical to the previous release: the
gradient digests and every parity suite match.

## Training step — production shape, the 0.7.0 kernel pass (RTX 6000 Ada, CUDA 13.2)

Same shape (d_model 384, 24 layers, B=8, T=1300), one board, the
deterministic GEMMs of 0.7.0 in both columns; the pass kept every kernel
output bit-identical to the previous release under the digest suites (the
chunked backward's pair kernel, the decode step, the prefill and whole
training runs, recorded on both trees), and the bf16 step then moved its
weight gradients to the stream-K kernel. Milliseconds per step, median of
four mirrored runs:

| tree | f32 | bf16 |
|------|----:|-----:|
| 0.7.0 before the pass | 262.1 | 231.2 |
| 0.7.0 | 184.9 | 145.5 |

Per kernel, per launch: the pair kernel of the chunked backward
(`m3_dqkv`) 3.41 to 1.58 ms, its tiles padded to an odd stride so the
sixteen-way bank conflict is gone and the pair-matrix triangle walked
flat; the column sums over the batch that the backward launches twelve
times per layer 165 to 215 µs down to 61 µs; the chunked lanes without
their fill and restore copies. The decode step runs nine kernels per
layer instead of eleven (the step kernel computes its coefficients from
dt and A on its first lane, and the bias+rope kernel advances the rotary
angles itself): 170.0 to 160.9 µs at batch 1 with graph replay on the
d128 synthetic model, 1.06×.

## Training step — the earlier kernel pass (2x RTX 5090, CUDA 13.0)

Same shape: 636 ms per step before that pass, 179.4 ms after (-72%). The
dominant backward kernel went from 11.4 ms to 1.94 ms per launch by
staging the pair and decay matrices in shared memory, widening the
per-thread work along the sequence, and splitting the backward into
state terms, state passing and parallel chunks, the same three-phase
structure as the forward.

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

## GPU Training — multi-chunk step (0.6; B=1, T=256, 24 layers, d_model=384)

Training runs the chunked parallel scan at multi-chunk sequence lengths;
earlier editions of this table timed a T=32 sequential configuration that
is never used, and those numbers are retired. Measured through the public
trainer (`Mamba3Trainer::step`: forward, backward, AdamW and sync). The
first two columns were taken on a GPU shared with other load, so their
ratio is the claim, not their absolute values; the last column is the
clean re-measurement:

| dtype | before the 0.6 kernel pass | after (shared Ada) | idle RTX 5090 |
|-------|---------------------------:|-------------------:|--------------:|
| f32   | 424 ms/step | **126.6 ms/step** | **110.5 ms/step** |
| bf16  | 395 ms/step | **121.9 ms/step** | **112.0 ms/step** |

The last column is the release re-measure on an idle RTX 5090
(CUDA 13.0, release build) — the clean-card absolute for the same
public-trainer step.

The pass: warp-parallel decay-gradient section in the dominant
backward kernel with the entering state staged in shared memory, the
shared causal Q·K tile in the intra-chunk output kernel, a loop-swapped
chunk-state kernel, head-packed full-warp blocks, and the
chunk-parallel angle accumulation.

## GPU Prompt Prefill (0.6; T=4621, 24 layers, d_model=384, f32)

One-pass prompt window through the chunked pipeline
(`tools/qualification/m3_prefill_bench.rs`), same shared-GPU caveat for
the first rows:

| stage | ms/prefill |
|-------|-----------:|
| before the 0.6 kernel pass | 384 |
| + shared Q·K tile | 251 |
| + chunk-parallel angle accumulation | 89.6 |
| + loop-swapped chunk state, head-packed blocks | **66.5** |

Release re-measure of the final stage on an idle RTX 5090 (CUDA 13.0,
release build): **23.65 ms/prefill — 42.3 prefills/s**.

## Large d_state capacity cost (0.6; Mamba-1 fused decode step, d_model=256, 4 layers)

Measured on the Mamba SSM fused decode step; the same compile-time
state-capacity mechanism covers both architectures, and no Mamba-3 twin
of this table has been measured. Past the register budget the compiler
spills to local memory, which is correct and measurably slower:

| d_state | ms/step |
|--------:|--------:|
| 64  | 0.289 |
| 128 | 0.378 |
| 256 | 1.498 |

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

## Mamba-3 and Mamba SSM at the default synthetic config

The two default configs differ (Mamba-3 has four layers and heads, Mamba
SSM three layers and a convolution), so these are step costs of two
different tiny models on the same box, not a ranking of the architectures.

|   | Mamba-3 | Mamba SSM | Notes |
|---|--------:|--------:|-------|
| CPU Inference B=1 | 64.5 us | 86.8 us | no conv1d, BLAS matvec |
| GPU Inference B=1 (Graph) | 87 us | 79 us | 4 layers against 3 |
| CPU Training Fwd+Bwd | 3 635 us | 14 859 us | no conv1d backward |

## Key Differences from Mamba SSM

- **No conv1d** — removed entirely (simpler and much faster CPU training).
- **Input-dependent A matrix** (per-head, clamped via `a_floor`).
- **Trapezoidal integration** (`alpha`, `beta`, `gamma` with learned `lambda`).
- **RoPE** per-head angle accumulation in `[0, 2π)`.
- **Multi-head B/C** with per-group BCNorm.
- **4 persistent recurrent states** (SSM + K + V + angle) vs 2 in Mamba SSM
  (conv_state + ssm_state).
- Implemented as NVRTC-compiled CUDA kernels.

## Optimizations

- SIMD SSM recurrence via `pulp` (CPU inference + training).
- BLAS matvec for `in_proj` / `out_proj` (CPU inference).
- CUDA Graph capture for GPU inference (the eager and graph columns above).
- Flat weight buffer + `WeightSlice` for CUDA Graph safety.
- Zero heap allocations per inference step.
- `disable_event_tracking()` for CUDA Graph capture stability.
- **Rayon-parallel backward** with thread-local gradient accumulators
  and tree-reduce (see `src/mamba3_siso/cpu/parallel.rs`).
  Scales to ~4× on 8-core Mac, ~5× on 48-thread Xeon.

Reproduce:
```
cargo bench --features cuda --bench m3_gpu_benchmark
cargo bench --bench m3_cpu_benchmark
cargo test --release --features "cuda hf qualification" --test bench_bf16_vs_f32 \
    bench_m3_bf16_vs_f32_synthetic -- --ignored --nocapture
```

## Deterministic GEMM

Mamba-3 shares the GEMM layer with Mamba SSM: the trainer, the inference
prefill and the decode step all multiply through the context, so the
context's `GemmMode` and its settings apply to inference exactly as to
training, with the same graph guards. The modes are described in
[gemm-modes.md](gemm-modes.md) and the kernel measurements in
[determinism-benchmarks.md](determinism-benchmarks.md).
