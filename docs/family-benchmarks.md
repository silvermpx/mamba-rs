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

Measured on October 19, 2026 on an RTX 5090 (170 SMs, driver 595.71.05,
CUDA 13.2.78 / NVRTC 13.2.78, Rust 1.99.1, release build, AMD EPYC 7543
host), the card held by the bench alone. Each cell is the median of two
runs of the bench, and each run reports the median of its timed calls.

bf16 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 142 us | 158 us | 1.47 M tokens/s | 6.9 ms |
| Mamba-2 | 156 us | 190 us | 1.14 M tokens/s | 8.2 ms |
| Mamba-3 SISO | 151 us | 284 us | 0.81 M tokens/s | 7.5 ms |
| Mamba-3 MIMO | 147 us | 205 us | 0.31 M tokens/s | 30.2 ms |

f32 storage:

| family | decode, batch 1 | decode, batch 16 | prompt of 4096 | training step |
|---|---:|---:|---:|---:|
| Mamba | 595 us | 628 us | 0.90 M tokens/s | 9.2 ms |
| Mamba-2 | 401 us | 479 us | 0.80 M tokens/s | 9.7 ms |
| Mamba-3 SISO | 417 us | 565 us | 0.54 M tokens/s | 10.8 ms |
| Mamba-3 MIMO | 390 us | 493 us | 0.28 M tokens/s | 33.5 ms |

The families differ in what a token costs. Mamba carries 16 states a
channel and the others 128, Mamba-3 adds the trapezoid and the angles, and
a MIMO head carries R = 4 inputs and outputs, which multiplies the work of
its scan.

## MIMO chunk length, RTX 5090

A MIMO chunk holds 64 / R tokens, as the reference sets it, so a chunk has
about 64 rows at any rank (see [the MIMO architecture
page](mamba3-mimo-architecture.md)). The training step above, at other
ranks and widths, with the chunk forced to 64 tokens through
`MAMBA_RS_MIMO_CHUNK` against the rank's own, on one binary, bf16 unless
named, the median of two runs a side, measured with the table above:

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
of chunk states, not every layer's.
