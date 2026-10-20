# The Four Families Side by Side

The four families at one size, each through its own `GpuBackbone` and
`GpuTrainer`: 4 layers of d_model 768, heads of 64 x 128 for Mamba-2 and
both forms of Mamba-3 (MIMO at rank 4), d_state 16 for Mamba as in the
released mamba-130m. A decode step runs through a captured graph at batch 1
and 16, the prompt is 4096 tokens in one pass from host rows, and a
training step is the forward, the backward and the AdamW update at batch 4
by 512 tokens. The default `Deterministic` GEMM mode serves every row.
The instrument is `benches/family_compare_bench.rs`:

```text
cargo bench --features cuda --bench family_compare_bench
```

## 0.8.0, RTX 5090

Measured on October 19, 2026 on the released 0.8.0 kernels, on an RTX 5090
(170 SMs, driver 595.84, CUDA 13.2.78 / NVRTC 13.2.78, Rust 1.99.1,
release build, AMD Ryzen 5 5600 host), the card held by the bench alone.
Each cell is the median of two runs of the bench, which agree within 0.3
percent, and each run reports the median of its timed calls.

bf16 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 134 us | 150 us | 1.47 M tokens/s | 6.4 ms |
| Mamba-2 | 146 us | 180 us | 1.18 M tokens/s | 7.8 ms |
| Mamba-3 SISO | 142 us | 268 us | 0.81 M tokens/s | 7.5 ms |
| Mamba-3 MIMO | 136 us | 191 us | 0.33 M tokens/s | 26.5 ms |

f32 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 570 us | 588 us | 0.88 M tokens/s | 9.1 ms |
| Mamba-2 | 384 us | 461 us | 0.82 M tokens/s | 9.6 ms |
| Mamba-3 SISO | 400 us | 542 us | 0.56 M tokens/s | 10.3 ms |
| Mamba-3 MIMO | 371 us | 475 us | 0.28 M tokens/s | 31.5 ms |

The families differ in what a token costs. Mamba carries 16 states a
channel and the others 128, Mamba-3 adds the trapezoid and the angles, and
a MIMO head carries R = 4 inputs and outputs, which multiplies the work of
its scan.

## 0.8.1, RTX PRO 6000 Blackwell

Measured on October 20, 2026 on the 0.8.1 kernels, on an RTX PRO 6000
Blackwell Server Edition (188 SMs, ECC on, driver 595.71.05, CUDA 13.2 /
NVRTC 13.2, Rust 1.99.1, release build, two Intel Xeon 6724P), the bench
pinned to the CPUs of the card's NUMA node and holding the card alone.
Each cell is the median of four runs, which agree within 2 percent (the
batch-16 decode within 5).

bf16 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 167 us | 201 us | 1.35 M tokens/s | 6.7 ms |
| Mamba-2 | 184 us | 225 us | 1.07 M tokens/s | 9.1 ms |
| Mamba-3 SISO | 179 us | 328 us | 0.75 M tokens/s | 8.6 ms |
| Mamba-3 MIMO | 176 us | 236 us | 0.28 M tokens/s | 29.0 ms |

f32 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 737 us | 760 us | 0.74 M tokens/s | 11.3 ms |
| Mamba-2 | 498 us | 542 us | 0.69 M tokens/s | 11.9 ms |
| Mamba-3 SISO | 517 us | 661 us | 0.49 M tokens/s | 12.7 ms |
| Mamba-3 MIMO | 483 us | 551 us | 0.24 M tokens/s | 35.6 ms |

The card's GDDR7 runs at 25 Gbit/s with ECC on, against 28 Gbit/s on the
RTX 5090, and the decode steps and the scans are bound by that bandwidth.
No TF32 cohort is frozen for 188 multiprocessors, so f32 storage takes the
common TF32 routes.

## 0.8.1, H100 PCIe

Measured on October 20, 2026 on the 0.8.1 kernels, on an H100 PCIe (114
SMs, driver 595.71.05, CUDA 13.2 / NVRTC 13.2, Rust 1.99.1, release build,
two AMD EPYC 7713), the bench pinned to the CPUs of the card's NUMA node
and holding the card alone. Each cell is the median of two runs, which
agree within 3 percent. Only bf16 storage was measured on this board.

bf16 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 534 us | 605 us | 0.98 M tokens/s | 9.5 ms |
| Mamba-2 | 439 us | 535 us | 0.72 M tokens/s | 12.4 ms |
| Mamba-3 SISO | 431 us | 733 us | 0.52 M tokens/s | 12.9 ms |
| Mamba-3 MIMO | 430 us | 659 us | 0.17 M tokens/s | 43.0 ms |

## MIMO chunk length, RTX 5090

Measured earlier on October 19 on another RTX 5090 (driver 595.71.05,
AMD EPYC 7543 host), on the kernels of that morning. Each table compares
two settings on one binary.

A MIMO chunk holds 64 / R tokens, as the reference sets it, so a chunk has
about 64 rows at any rank (see [the MIMO architecture
page](mamba3-mimo-architecture.md)). The training step above, at other
ranks and widths, with the chunk forced to 64 tokens through
`MAMBA_RS_MIMO_CHUNK` against the rank's own, on one binary, bf16 unless
named, the median of two runs a side:

| head x states, rank | chunks of 64 | chunks of 64 / R | |
|---|---:|---:|---:|
| 64 x 128, rank 1 | 14.1 and 17.6 ms | 13.9 and 14.2 ms | the same chunk, so the spread is the measurement's |
| 64 x 128, rank 2 | 20.3 ms | 18.1 ms | 1.12x |
| 64 x 128, rank 4 | 38.2 ms | 31.7 ms | 1.20x |
| 64 x 128, rank 4, f32 | 39.7 ms | 34.1 ms | 1.16x |
| 64 x 128, rank 8 | 101.1 ms | 56.8 ms | 1.78x |
| 64 x 128, rank 16 | 328.6 ms | 107.1 ms | 3.07x |
| 64 x 64, rank 4 | 25.8 ms | 18.8 ms | 1.38x |
| 128 x 128, rank 4 | 34.9 ms | 32.5 ms | 1.07x |

The training step recomputes each layer's chunk states in its backward
instead of keeping them, so the shorter chunks enlarge one layer's buffer
of chunk states, not every layer's. Both columns run the CUDA-core pair
kernels (`MAMBA_RS_PAIRS_TC=off`): at a chunk of 64 tokens and rank 2 or
more a chunk's rows do not fit the tensor-core kernel's block.

## MIMO pair terms on the tensor cores, RTX 5090

Measured on the same machine and kernels as the chunk table above.

The tf32, bf16 and f16 trainers form the backward's pair terms on the
tensor cores (see [the MIMO architecture page](mamba3-mimo-architecture.md)).
The training step of the table above at other ranks and widths, the
CUDA-core kernels (`MAMBA_RS_PAIRS_TC=off`) against the tensor cores, on
one binary, runs alternating, bf16 unless named, the median of two runs a
side. Where the two runs of a side part by more than a few percent both
are shown:

| head x states, rank | CUDA cores | tensor cores | |
|---|---:|---:|---:|
| 64 x 128, rank 1 | 14.0 ms | 13.6 ms | 1.03x |
| 64 x 128, rank 2 | 21.6 and 18.6 ms | 20.6 and 18.9 ms | about the same |
| 64 x 128, rank 4 | 33.5 ms | 27.0 ms | 1.24x |
| 64 x 128, rank 4, f16 | 29.6 ms | 26.6 ms | 1.11x |
| 64 x 128, rank 8 | 54.8 ms | 48.5 ms | 1.13x |
| 64 x 128, rank 16 | 105.5 ms | 93.4 ms | 1.13x |
| 64 x 64, rank 4 | 18.9 ms | 20.9 and 17.6 ms | about the same |
| 128 x 128, rank 4 | 32.1 and 34.2 ms | 31.6 ms | 1.05x |

On an RTX 6000 Ada at two sequences a step, where every side's two runs
agree within one percent, the tensor cores take the step at rank 4 from
21.5 to 19.1 ms in bf16 (1.13x), 22.3 to 20.1 ms in tf32 (1.11x) and 21.0
to 18.5 ms in f16 (1.14x), at rank 1 from 10.8 to 10.1 ms, at rank 2 from
13.8 to 12.4 ms, with heads of 64 x 64 from 14.8 to 12.8 ms and with heads
of 128 x 128 from 21.0 to 19.7 ms. The f32 step is the same either way.
