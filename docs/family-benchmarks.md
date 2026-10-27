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

## 0.8.2 against 0.8.1, RTX 5090

Measured on October 27, 2026 on a rented RTX 5090 (170 SMs, power limit
450 W, driver 595.58.03, CUDA 13.2.78 / NVRTC 13.2.78, Rust 1.99.1,
release build, AMD Ryzen Threadripper 9960X host). The 0.8.1 tag and the
0.8.2 tree ran in turn on the card alone, four runs each, and each cell is
the median of its four runs.

bf16 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 137 -> 137 us (1.00x) | 151 -> 152 us (1.00x) | 1.69 -> 1.69 M tokens/s (1.00x) | 6.03 -> 5.88 ms (1.02x) |
| Mamba-2 | 150 -> 151 us (1.00x) | 183 -> 183 us (1.00x) | 1.28 -> 1.28 M tokens/s (1.00x) | 7.75 -> 7.69 ms (1.01x) |
| Mamba-3 SISO | 144 -> 146 us (0.99x) | 276 -> 276 us (1.00x) | 0.85 -> 0.85 M tokens/s (1.00x) | 7.34 -> 7.34 ms (1.00x) |
| Mamba-3 MIMO | 139 -> 139 us (1.00x) | 195 -> 195 us (1.00x) | 0.32 -> 0.32 M tokens/s (1.00x) | 27.01 -> 26.98 ms (1.00x) |

f32 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 589 -> 114 us (5.15x) | 606 -> 131 us (4.64x) | 0.87 -> 0.93 M tokens/s (1.07x) | 9.12 -> 8.55 ms (1.07x) |
| Mamba-2 | 396 -> 116 us (3.40x) | 472 -> 206 us (2.29x) | 0.78 -> 0.84 M tokens/s (1.08x) | 9.93 -> 10.21 ms (0.97x) |
| Mamba-3 SISO | 412 -> 133 us (3.11x) | 558 -> 291 us (1.92x) | 0.55 -> 0.56 M tokens/s (1.02x) | 10.59 -> 10.80 ms (0.98x) |
| Mamba-3 MIMO | 384 -> 108 us (3.53x) | 486 -> 227 us (2.14x) | 0.27 -> 0.28 M tokens/s (1.03x) | 32.39 -> 32.59 ms (0.99x) |

The same f32 training steps timed by a probe over the public trainer, the
two trees in rotated order, six runs of 20 steps each, the minimum of each:

| family | 0.8.1 | 0.8.2 | old/new |
|---|---:|---:|---:|
| Mamba | 9.15 ms | 8.61 ms | 1.063x |
| Mamba-2 | 9.93 ms | 10.09 ms | 0.984x |
| Mamba-3 SISO | 10.62 ms | 10.68 ms | 0.994x |
| Mamba-3 MIMO | 32.48 ms | 32.69 ms | 0.994x |

0.8.1 summed a layer of 768 inputs over 2048 rows as one chain and split a
smaller grid by the card's number of multiprocessors, so a row's bits
changed with the batch it came in. 0.8.2 folds such a layer in pieces of
128 inputs at every batch size, and on this card a block that folds holds a
second set of registers, two blocks a multiprocessor where one chain holds
three. The input projection forward of Mamba-2 and Mamba-3 (768 inputs to
3352 and more outputs over 2048 rows) takes about 209 microseconds where
0.8.1 took 175, and that layer is most of the difference. The output
projection's three products take 8 to 12 microseconds longer each, and
the rest of the Mamba-2 step runs as fast as 0.8.1 or faster. On a larger
batch the fold fills the card in more waves and the difference closes; the
same probe at batch 8 and 16 by 512 tokens, four runs of 10 steps of each
tree in rotated order, the minimum of each, `old/new` above 1 meaning 0.8.2
is faster:

| family | batch 8 | batch 16 |
|---|---:|---:|
| Mamba | 1.084x | 1.065x |
| Mamba-2 | 0.995x | 0.998x |
| Mamba-3 SISO | 0.996x | 0.999x |
| Mamba-3 MIMO | 0.998x | 0.997x |

The run's other benches, 0.8.2 against 0.8.1, geometric mean of every
value each bench prints: the Mamba GPU benchmark 1.41x, the Mamba mixer
1.92x, Mamba-3 1.38x, Mamba-2 and MIMO 1.19x, the GEMMs of a training step
alone 1.07x, the Mamba-3 scan kernels 1.01x. Two values are slower: the
CPU decode step, 50 to 54 microseconds, and the bf16 Mamba-2 prompt of 4096
tokens at batch 1, 0.97x.

## 0.8.2 against 0.8.1, A100 SXM4 40 GB

Measured on October 27, 2026 on a rented A100 SXM4 40 GB (108 SMs, ECC on,
driver 595.91.07, CUDA 13.2.78 / NVRTC 13.2.78, Rust 1.99.1, release
build, AMD EPYC 7K62 host shared with other tenants at a load average of 25
to 35). The two trees ran in turn, six runs each, and each cell is the
median of its six runs.

bf16 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 |
|---|---:|---:|---:|
| Mamba | 343 -> 344 us (1.00x) | 388 -> 389 us (1.00x) | 0.66 -> 0.66 M tokens/s (1.00x) |
| Mamba-2 | 367 -> 369 us (0.99x) | 465 -> 466 us (1.00x) | 0.45 -> 0.45 M tokens/s (1.00x) |
| Mamba-3 SISO | 342 -> 343 us (1.00x) | 755 -> 755 us (1.00x) | 0.37 -> 0.37 M tokens/s (1.00x) |
| Mamba-3 MIMO | 351 -> 352 us (1.00x) | 652 -> 653 us (1.00x) | 0.13 -> 0.12 M tokens/s (0.91x) |

f32 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 |
|---|---:|---:|---:|
| Mamba | 1281 -> 336 us (3.81x) | 1337 -> 387 us (3.46x) | 0.26 -> 0.26 M tokens/s (1.01x) |
| Mamba-2 | 1101 -> 338 us (3.26x) | 1199 -> 444 us (2.70x) | 0.24 -> 0.23 M tokens/s (0.99x) |
| Mamba-3 SISO | 1132 -> 370 us (3.06x) | 1387 -> 795 us (1.74x) | 0.20 -> 0.23 M tokens/s (1.17x) |
| Mamba-3 MIMO | 1075 -> 327 us (3.28x) | 1383 -> 644 us (2.15x) | 0.11 -> 0.11 M tokens/s (1.00x) |

The MIMO bf16 prompt ran either near 115 or near 141 thousand tokens a
second on both trees, run by run, and the median caught more slow runs of
0.8.2; the fastest runs of the two trees are 141.8 and 140.5 thousand. The
training step of this bench varied by up to a quarter between runs of the
same tree on this host, so it comes from the probe instead, in rotated
order, six runs of 20 steps each:

| family | 0.8.1, minimum | 0.8.2, minimum | old/new | old/new, medians |
|---|---:|---:|---:|---:|
| Mamba | 24.78 ms | 23.96 ms | 1.034x | 1.011x |
| Mamba-2 | 26.87 ms | 26.72 ms | 1.006x | 1.003x |
| Mamba-3 SISO | 28.48 ms | 28.88 ms | 0.986x | 1.009x |
| Mamba-3 MIMO | 82.95 ms | 83.04 ms | 0.999x | 0.999x |

The GEMMs of a training step alone (`gemm_bi_trainer_step_bench`, exact
f32) take 3.62 ms instead of 4.17 on the 128-wide shape, 6.37 instead of
8.78 on the 256-wide one, 23.87 instead of 24.10 on the 768-wide one and
20.15 instead of 23.43 on the 1536-wide one. Over the run's other benches,
geometric mean of every value: the Mamba GPU benchmark 1.42x, the Mamba
mixer 1.80x, Mamba-3 1.42x, Mamba-2 and MIMO 1.15x, the Mamba-3 scan
kernels 1.02x. The Mamba-2 chunked prompt of `m2_mimo_gpu_bench` at batch 1
ran 3 to 6 percent slower than 0.8.1 in bf16 and f32. Profiled kernel by
kernel, a 1024-token bf16 window runs the same kernels in the same time on
both trees (2.383 and 2.382 ms), so the bf16 difference is this host's
noise. In f32 0.8.1 ran the window's products on its one-chain f32 kernel,
where 0.8.2 folds them in pieces of 128 inputs, the same at every batch
size and on every route. With the SM80 slab tile taken by the last wave
of its grid the products of that window take 2.87 ms (2.99 before it), and
the window 4.56 ms of kernel time; 0.8.1's profile, at clocks under which
the same scan kernels also ran about 7 percent faster, took 4.25.

## 0.8.2 against 0.8.1, RTX 6000 Ada

Measured on October 27, 2026 on an RTX 6000 Ada Generation (142 SMs, 300 W
power limit, driver 615.71.10, CUDA 13.2 / NVRTC 13.2, Rust 1.99.1, release
build, Intel Xeon Gold 5412U host). The two trees ran in turn, four runs
each, and each cell is the median of its four runs.

bf16 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 158 -> 157 us (1.01x) | 202 -> 194 us (1.04x) | 1.09 -> 1.09 M tokens/s (1.00x) | 8.54 -> 7.85 ms (1.09x) |
| Mamba-2 | 153 -> 152 us (1.00x) | 205 -> 205 us (1.00x) | 0.88 -> 0.89 M tokens/s (1.01x) | 10.13 -> 9.86 ms (1.03x) |
| Mamba-3 SISO | 156 -> 156 us (1.00x) | 435 -> 435 us (1.00x) | 0.60 -> 0.59 M tokens/s (0.98x) | 10.38 -> 10.20 ms (1.02x) |
| Mamba-3 MIMO | 146 -> 146 us (1.00x) | 232 -> 223 us (1.04x) | 0.24 -> 0.24 M tokens/s (1.02x) | 38.82 -> 39.01 ms (1.00x) |

f32 storage, 0.8.1 -> 0.8.2 (speedup):

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 605 -> 146 us (4.14x) | 649 -> 189 us (3.44x) | 0.57 -> 0.60 M tokens/s (1.06x) | 15.07 -> 13.89 ms (1.08x) |
| Mamba-2 | 409 -> 142 us (2.88x) | 525 -> 317 us (1.66x) | 0.46 -> 0.52 M tokens/s (1.12x) | 14.54 -> 14.85 ms (0.98x) |
| Mamba-3 SISO | 434 -> 166 us (2.61x) | 733 -> 519 us (1.41x) | 0.34 -> 0.34 M tokens/s (1.02x) | 16.95 -> 17.07 ms (0.99x) |
| Mamba-3 MIMO | 405 -> 144 us (2.81x) | 560 -> 357 us (1.57x) | 0.19 -> 0.20 M tokens/s (1.02x) | 49.25 -> 49.29 ms (1.00x) |

The batch-1 decode of this bench lands on two levels about 10 microseconds
apart from run to run, on both trees; the decode step alone, timed by a
probe in rotated order (four runs of 400 steps per family and tree), puts
the two trees within 2 percent either way, with the same kernels in the
same captured graph. The f32 training step varies by about 2 percent
between runs of one tree in this bench; the probe over the public trainer,
in rotated order, four runs each, the minimum of each, `old/new` above 1
meaning 0.8.2 is faster:

| family | batch 4 by 512 | batch 8 by 512 | batch 16 by 512 |
|---|---:|---:|---:|
| Mamba | 1.051x | 1.018x | 1.022x |
| Mamba-2 | 0.994x | 1.008x | 1.017x |
| Mamba-3 SISO | 1.002x | 0.995x | 1.014x |
| Mamba-3 MIMO | 0.991x | 0.998x | 1.005x |

The medians of the same runs agree within 1.6 percent (Mamba-2 at batch 4:
1.016x). Under f32 training this card runs at its 300 W limit, where the
clock sets the pace and a kernel that moves more bytes per multiply-add
slows the whole step. SM89 keeps a weight gradient that the shape cuts
into row chunks on the portable 128 x 128 partial, as 0.8.1 did: a direct
64 x 64 partial of the same partition writes the same planes in about half
the time alone on the card, but moves twice the L2 bytes and one and a
half times the shared words per multiply-add, and in the step the card
then clocked lower at the same power. With it the step ran 3 to 4 percent
slower at batch 8 and 16 (Mamba and Mamba-2) and no faster at batch 4.

The Mamba-2 chunked prompt of the probe (the minimum of four runs) takes
0.995 to 1.005 of 0.8.1's time in bf16 and runs 1.004x, 1.025x and 1.076x
faster in f32 at batch 1 by 1024, batch 1 by 4096 and batch 8 by 1024
tokens. A process that builds a model from cached kernels starts in about
2.6 seconds here, where 0.8.1 took 2.8: each GEMM module is loaded once and
the check of its kernels' parameters reads that module: 12 module loads,
where earlier 0.8.2 builds loaded a copy of the module for each check, 30
loads in all, and started in about 3.0 seconds. The run's other benches,
0.8.2 against 0.8.1, geometric mean of every value each bench prints: the
Mamba GPU benchmark 1.43x, the Mamba mixer 1.74x, Mamba-3 1.52x, Mamba-2 and
MIMO 1.12x, the GEMMs of a training step alone 1.06x, the Mamba-3 scan
kernels 0.99x (the same kernels on both trees).

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
