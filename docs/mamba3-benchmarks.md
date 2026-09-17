# Mamba-3 SISO Benchmarks

Unless a section names another board, measurements use an Intel Xeon Gold
5412U (48 threads) and NVIDIA RTX 6000 Ada Generation (48 GB), CUDA 13.2,
driver 595.45. RTX 5090 results name that board and toolkit explicitly. Every table
is a measurement of the release named in its heading and is kept as
history; the current kernel comparisons against cuBLAS are in
[determinism-benchmarks.md](determinism-benchmarks.md).

The fixtures below use synthetic initialization, not a trained language
model. They measure the implementation at the stated shapes, not model
quality. For inference measurements using trained Mamba-1 checkpoints,
see [mamba1-benchmarks.md](mamba1-benchmarks.md).

## Training step — 0.7.0 to 0.7.1 (RTX 6000 Ada, CUDA 13.2)

Fresh measurements on September 14, 2026: released `v0.7.0`
(`e2917a47494b4a1d652f5c818c3974ec3fcdd1ca`) against assembled `0.7.1`
(`83079104fe1efa7ad5dca0a28c5b48bcd86c5b14`). RTX 6000 Ada, 142 SMs, driver
595.45.04, CUDA 13.2.51 / NVRTC 13.2, Rust 1.98.1, release build.

Both versions run the public trainer in the default `Deterministic`
GEMM mode, with the Triad family. BF16, F16 and full-precision F32 are
separate storage modes; the F32 row uses the default exact policy.
TF32-permitted training is measured in the next section; the separate
[GEMM tables](gemm-benchmarks-0.7.1-ada.md) cover kernel timings.

Shape: d_model 384, d_state 16, expand 2, 24 layers, B=8, T=1300,
input width 384. The fixtures use synthetic weights and pre-generated
inputs and output gradients. They exercise backbone training, without
tokenization, a vocabulary head or its loss. Timing includes the complete
public `step`: input handling, forward, backward, optimizer, metric
handling and synchronization.
Initialization, compilation and graph capture are outside the timers.

Each dtype ran as old, new, new, old on an otherwise idle board, with
separate source trees and kernel caches. Each table entry is the median
of the two process-average step times; parentheses show their range,
not a confidence interval. `old/new` above 1 means 0.7.1 is faster.

No timed step skipped its optimizer update because of overflow.

| storage | execution | 0.7.0 ms/step (range) | 0.7.1 ms/step (range) | old/new |
|---|---|---:|---:|---:|
| BF16 | eager | 146.60 (146.60–146.61) | 133.64 (133.63–133.66) | 1.097× |
| BF16 | graph | 145.51 (145.47–145.55) | 132.46 (132.44–132.47) | 1.099× |
| F16 | eager | 146.98 (146.85–147.11) | 133.92 (133.87–133.97) | 1.098× |
| F16 | graph | 145.90 (145.88–145.92) | 132.92 (132.92–132.92) | 1.098× |
| F32 | eager | 186.16 (186.15–186.16) | 174.86 (174.79–174.93) | 1.065× |
| F32 | graph | 185.46 (185.32–185.61) | 174.18 (174.04–174.32) | 1.065× |

Mamba-3 uses the chunked scan (`Auto` resolves to parallel here), head
dimension 16, one group, RoPE fraction 0.5, a_floor 1e-4 and output
projection normalization. The model state width is 16; the actual
compiled context capacity is 64. BF16 and F32 keep the original three
eager warmup steps and five graph warmup steps. Before each F16 timed
arm, the fixture instead requires eight consecutive zero-skipped
optimizer updates, failing if it cannot reach them within 128 attempts.
Every dtype still averages exactly five eager and five graph timed steps.

The F32 fixture has an explicit identity input-projection matrix;
BF16/F16 use the mixed trainer's identity branch with that matrix
omitted. Both releases use the same per-dtype measurement overlay,
including the corrected F16 warmup. These rows should not be interpreted
as throughput on a trained checkpoint or as a controlled comparison
between storage modes.

Reproduce with `m3_prefill_bench::m3_train_step_at_multichunk_shape`,
setting `MAMBA_RS_BENCH_B=8`, `MAMBA_RS_BENCH_T=1300`,
`MAMBA_RS_BENCH_ITERS=5`, and `MAMBA_RS_BENCH_DTYPE` to one of
`bf16`, `f16`, `f32`. Leave scan-tape, IEEE-F32 and GEMM mode/policy
overrides unset; the remaining model configuration is fixed in the
instrument.

Run the instrument as an exact ignored test:

```sh
cargo test --release --locked --features cuda,qualification \
  --test m3_prefill_bench m3_train_step_at_multichunk_shape \
  -- --exact --ignored --nocapture --test-threads=1
```

The old tree uses the same measurement fixture and logging code, with
its released library and CUDA sources unchanged. Raw logs identify the
resolved mode, scan path, context capacity and skipped-step count.

Sampled device memory includes setup and both execution modes. It is the
maximum of 200 ms NVML samples across the two processes per tree, not
an allocator high-water mark or a measurement for an individual step.

| storage | 0.7.0 peak MiB | 0.7.1 peak MiB |
|---|---:|---:|
| BF16 | 6716 | 6718 |
| F16 | 6716 | 6718 |
| F32 | 9948 | 9950 |

The assembled source passed the same-board 0.7.0 comparison for all
161 original normalized Mamba ledger keys and, separately, all 48
expanded decode cells. No changed or missing cell was observed in either
comparison. These bit checks and the whole-step timings are separate
evidence. Earlier prefill/decode tables below remain historical
measurements; the RTX 5090 training step is measured in its own section.

## Mamba-3 supplemental TF32-permitted F32 training

Separate TF32-permitted measurements compare released `v0.7.0`
(`e2917a47494b4a1d652f5c818c3974ec3fcdd1ca`) with assembled `0.7.1`
(`83079104fe1efa7ad5dca0a28c5b48bcd86c5b14`). The same F32 measurement
fixture is compiled against each version; runs use separate
kernel caches on an RTX 6000 Ada in the same CUDA 13.2 environment.

`MAMBA_RS_GEMM_MODE=deterministic` resolves the Triad family and
`MAMBA_RS_BI_F32_POLICY=tf32` resolves
`AllowDeterministicTf32`. That policy permits deterministic TF32 where a
qualified route applies; it does not force every GEMM to use TF32, and an
exact deterministic fallback remains valid.

Shape: d_model 384, 24 layers, B=8, T=1300, Auto parallel scan,
capacity-64 context; five eager and five graph timed steps per process.
Each tree has two
process observations in old, new, new, old order. Values are medians of the
two process-average timings; parentheses are the process range, not a
confidence interval. Every timed step performed its optimizer update.

| execution | v0.7.0 ms/step median (range) | 0.7.1 ms/step median (range) | old/new |
|---|---:|---:|---:|
| eager | 176.88 (176.83–176.93) | 165.56 (165.54–165.58) | 1.068× |
| graph | 175.90 (175.83–175.96) | 164.52 (164.45–164.58) | 1.069× |

Peak device memory is the maximum 200 ms NVML sample across setup, eager,
capture and graph execution in each process; it is not an allocator
high-water mark or an individual-step measurement.

| model | v0.7.0 peak MiB median (range) | 0.7.1 peak MiB median (range) |
|---|---:|---:|
| Mamba-3 | 9948.0 (9948–9948) | 9950.0 (9950–9950) |

## 0.7.1 development — sequential-scan checkpoint

RTX 6000 Ada, CUDA 13.2. These measurements cover the initial sequential
scan pass, before the later backward, typed burn-in and GEMM work. They
are not final 0.7.1 bundle measurements. Historical 0.7.0 tables below
remain measurements of that release.

Forward and backward of one layer, batch 64, eight heads of 32 over a
16-wide state, timed on the same buffers in one run:

| sequence | 0.7.0 | scan checkpoint | faster by |
|---|---:|---:|---:|
| T = 48 | 709 us | 178 us | 3.98x |
| T = 390 | 6969 us | 1918 us | 3.63x |
| T = 1440 | 25813 us | 7270 us | 3.55x |

A four-layer training step at the crate's default shape (d_model 128,
16 heads of 16, state 16, batch 1, T 32), from `m3_gpu_benchmark`:

| | 0.7.0 | scan checkpoint | faster by |
|---|---:|---:|---:|
| forward | 891.1 us | 701.4 us | 1.27x |
| backward | 1704.6 us | 645.5 us | 2.64x |
| forward and backward | 2595.7 us | 1346.8 us | 1.93x |

The initial pass left decode unchanged. Later normalization changes also
affect decode, so the T=1 tables below must not be read as fresh 0.7.1
measurements.

The final assembled source `83079104` was checked against the released
0.7.0 source on this board with CUDA 13.2: all 161 original normalized
ledger keys match, with no changed or missing keys. A separate expanded
decode corpus matches all 48 cells, including native BF16/F16 eager and
graph execution and persistent states after 16 steps. These two inventories
are separate checks, not a combined cell count.

## Training step — 0.7.0 to 0.7.1 (RTX 5090, CUDA 13.2)

Measurements on September 16, 2026: released `v0.7.0`
(`e2917a47494b4a1d652f5c818c3974ec3fcdd1ca`) against released `v0.7.1`
(`1988e4b86f44b2a7819ab7336804774041640c0f`, run from the 0.7.2 tree,
which changes no kernel or route). One RTX 5090, 170 SMs, driver
595.58.03, CUDA 13.2 / NVRTC 13.2, Rust 1.98.1, release build. The board
reports 100 % utilization while idle under this driver; a quiet card was
judged by resident memory and the compute-process list instead.

Same shape and fixtures as the Ada section above: B=8, T=1300, 24 layers,
five timed eager and five timed graph steps, default `Deterministic` mode
with the Triad family; the TF32 rows use `MAMBA_RS_BI_F32_POLICY=tf32`.
Each dtype ran as old, new, new, old with separate trees and kernel caches.
Entries are the median of the two process-average step times, parentheses
their range. `old/new` above 1 means 0.7.1 is faster.

| storage | execution | 0.7.0 ms/step (range) | 0.7.1 ms/step (range) | old/new |
|---|---|---:|---:|---:|
| BF16 | eager | 108.90 (108.60–109.20) | 108.62 (108.56–108.68) | 1.003× |
| BF16 | graph | 108.24 (107.96–108.52) | 108.43 (108.39–108.47) | 0.998× |
| F16 | eager | — | 109.11 (109.02–109.19) | — |
| F16 | graph | — | 108.81 (108.68–108.93) | — |
| F32 | eager | 113.33 (113.06–113.59) | 112.22 (111.87–112.58) | 1.010× |
| F32 | graph | 113.70 (113.48–113.92) | 113.84 (113.41–114.26) | 0.999× |
| F32, TF32 permitted | eager | 113.43 (113.27–113.59) | 112.22 (111.62–112.82) | 1.011× |
| F32, TF32 permitted | graph | 113.72 (113.53–113.92) | 113.72 (113.13–114.32) | 1.000× |

The 0.7.0 tree ran its own released instrument, which measures F32 and
BF16 in one process and has no F16 arm, so the F16 row carries 0.7.1
only and no 0.7.0 peak memory is reported: a two-model process is not a
per-dtype figure. 0.7.1 peak device memory, the maximum 200 ms NVML
sample per process: 6834 MiB for BF16 and F16, 10066 MiB for F32.

0.7.1 is neutral on this board: its Mamba-3 gains on Ada came from the
retained sm89 routes, which the dispatcher does not select here, so both
releases run the same routes. The TF32-permitted policy changes nothing
on this board at this shape either. The bit ledger of both tags on this
board matched on all 161 original Mamba keys, none moved, none missing.

## Historical measurements

The following prefill, decode and RTX 5090 tables retain their original
measurement scope. They were not retimed for the 0.7.1 training comparison.

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
