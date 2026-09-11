# GEMM benchmarks

This page holds the measured performance of the deterministic GEMM kernels
that mamba-rs 0.7.0 runs by default. Every number was taken on real
hardware with the protocol described below, and every table names the GPU,
the CUDA toolkit, the precision and the cuBLAS setting it compares against.
The modes themselves are explained in [GEMM modes](gemm-modes.md).

Two GPUs were measured:

| GPU | architecture | SMs | driver | CUDA toolkit |
|---|---|---:|---|---|
| NVIDIA RTX 6000 Ada Generation | SM89 (Ada Lovelace) | 142 | 595.45.04 | 13.2 (nvcc 13.2.51) |
| NVIDIA GeForce RTX 5090 | CC 12.0 (Blackwell) | 170 | 595.84 | 13.2 (nvcc 13.2.78) |

All kernel timings on this page come from source revision `732c1146`, the
last commit that touched a kernel before the release. The old-versus-new
tables were taken on the RTX 6000 Ada against the unchanged 0.6.9 tree
(commit `d8f2efbe`).

## How to read the numbers

- **Speedup** means cuBLAS time divided by mamba-rs time. Above 1.0 the
  deterministic kernel is faster than cuBLAS; below 1.0 it is slower. In
  the old-versus-new tables the speedup is 0.6.9 time divided by 0.7.0
  time, so above 1.0 means 0.7.0 is faster.
- **Comparators are chosen by precision contract.** Exact f32 kernels are
  compared with cuBLAS Pedantic (`CUBLAS_COMPUTE_32F_PEDANTIC`), which
  performs the same full-f32 arithmetic. The same kernels are also shown
  against cuBLAS Fast TF32 (`CUBLAS_COMPUTE_32F_FAST_TF32`) so the cost of
  keeping full precision is visible, but that is a comparison across two
  precisions, not a like-for-like one. Deterministic TF32 kernels are
  compared with cuBLAS Fast TF32. bf16 and f16 kernels are compared with
  cuBLAS Fast (`CUBLAS_COMPUTE_32F`), which runs the native half-precision
  tensor-core kernels with f32 accumulation; TF32 does not apply to half
  inputs. cuBLAS Pedantic for half inputs is the slower, more careful
  accumulation and appears in the training-step tables.
- **Eager and graph** are two measurements: eager times individual kernel
  launches, graph times the replay of a captured CUDA graph holding the same
  launches. On the RTX 5090 the event timer has a 2.048 µs grid, so graph
  timings of the smallest shapes sit on that grid and their ratios are step
  counts rather than resolved differences; those rows are marked.
- **These are kernel timings, not model timings.** A training step or a
  decode step spends most of its time outside the GEMMs. The whole-model
  comparison of 0.6.9 against 0.7.0 is in its own section below.
- **Bias.** In the Inference family tables the cuBLAS arm includes the
  separate bias-broadcast kernel cuBLAS needs, because that is what a model
  pays; the deterministic kernels fuse the bias. The Triad tables have no
  bias.

## Summary

### Inference family (serving), five shapes, bias off and on

Geometric mean over the ten shape and bias combinations.

| input → output | compared with | Ada eager | Ada graph | RTX 5090 eager | RTX 5090 graph |
|---|---|---:|---:|---:|---:|
| BF16 → BF16 | cuBLAS Fast | 1.187× | 1.185× | 1.243× | 1.235× |
| F16 → F16 | cuBLAS Fast | 1.167× | 1.165× | 1.234× | 1.238× |
| BF16 → F32 | cuBLAS Fast | 0.828× | 0.825× | 1.288× | 1.276× |
| F16 → F32 | cuBLAS Fast | 0.815× | 0.812× | 1.278× | 1.270× |
| F32 → F32, deterministic TF32 | cuBLAS Fast TF32 | 0.900× | 0.900× | 1.132× | 1.126× |
| F32 → F32, exact | cuBLAS Pedantic | 1.004× | 1.002× | 1.083× | 1.074× |
| F32 → F32, exact | cuBLAS Fast TF32 | 0.462× | 0.459× | 0.772× | 0.776× |

On the RTX 5090 every one of the ten cells wins in each of the four
half-precision rows. On Ada the native half rows win eight of ten; the two
losses are the two largest shapes with bias off. The exact-f32 row against
Pedantic on Ada is parity, not a win.

### Triad family (training), NN, TN and NT products

Geometric mean over the cells of each row; the wins column counts cells
where the deterministic kernel was faster.

| precision | compared with | cells | Ada eager | Ada graph | Ada wins | RTX 5090 eager | RTX 5090 graph | RTX 5090 wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | cuBLAS Fast | 15 | 0.890× | 0.869× | 7 | 0.689× | 0.929× | 6 |
| F16 | cuBLAS Fast | 15 | 0.881× | 0.851× | 6 | 0.690× | 0.929× | 6 |
| F32, deterministic TF32 | cuBLAS Fast TF32 | 21 | 0.718× | 0.709× | 0 | 1.051× | 0.939× | 18 |
| F32, exact | cuBLAS Pedantic | 15 | 0.775× | 0.772× | 4 | 1.019× | 0.919× | 11 |
| F32, exact | cuBLAS Fast TF32 | 15 | 0.422× | 0.422× | 0 | 0.793× | 0.698× | 0 |

The family averages hide a split by shape. On the large training shapes
(d768, the 4096-row deep shape and the 4621-row classifier page) the half
rows are above parity on both boards: Ada BF16 1.086×, F16 1.098×, RTX 5090
1.060× for both. The six small d128 shapes pull the averages down; on those
shapes a launch costs more than the arithmetic, and the RTX 5090's eager
launches are especially expensive (about 21 µs against 4 to 12 µs from a
graph). The Triad does not beat cuBLAS Fast as a whole on either board.

## New kernels against the 0.6.9 kernels

Measured on the RTX 6000 Ada with one program compiled against both trees.
For every cell the program times three arms in the same process with CUDA
events, 21 windows of calibrated launch counts, the arms rotating inside
each window in mirrored order: the tree's deterministic kernel, cuBLAS in
the tree's fast setting and cuBLAS in the tree's pedantic setting. Each
tree runs as its own process, in mirrored blocks (old, new, new, old);
the table shows the median of the four runs of each tree. Before timing,
every arm's output is compared with the pedantic arm's (relative L2 below
5e-2 in every cell, below 1e-4 for the exact-f32 kernels). Deterministic
means the tree's own kernels with `batch_invariant` on; half precision
runs with tensor cores on both trees, f32 on the exact kernels. The
speedup is 0.6.9 time divided by 0.7.0 time.

The 0.6.9 tree has no deterministic TF32, so there is no old kernel to put
beside the new TF32 kernels; those are compared with cuBLAS Fast TF32 in
the Triad tables further down.

Geometric mean of the speedup by family, precision and shape class:

| family | precision | large shapes | small shapes (d128, underfill) |
|---|---|---:|---:|
| Triad (training) | BF16 | 1.29× (1.02 to 1.58) | 1.08× |
| Triad (training) | F16 | 1.25× (1.02 to 1.44) | 1.08× |
| Triad (training) | F32 exact | 1.40× (0.90 to 2.26) | 1.25× (1.02 to 2.98) |
| Inference (serving) | BF16 | 1.33× (1.27 to 1.43) | |
| Inference (serving) | F16 | 1.33× (1.27 to 1.42) | |
| Inference (serving) | F32 exact | 1.39× (1.19 to 1.47) | |

The largest single gains are the exact-f32 weight-gradient kernels on the
d128 shapes (2.0× and 3.0×) and the exact-f32 input-gradient kernel on
the d768 in_proj shape (2.26×). Two cells did not move or moved the wrong
way: the deep 4096-row exact-f32 forward is 10 percent slower than the
0.6.9 kernel (the new automatic selection has no measured kernel for that
shape and the scalar kernel serves), and the deep-shape half kernels are
within 2 percent of the old ones. The small d128 half kernels are the same
portable kernels in both trees.

The cuBLAS arms are the control: on the calls that are identical in both
trees (half precision, `COMPUTE_32F`) the new process measured 0 to 12
percent slower than the old one on the same board, which is run-to-run
drift on a power-capped GPU; dividing each tree's deterministic time by
its own cuBLAS Fast time moves the averages above by at most 0.06 for
half precision and 0.14 for f32, in the direction of the new kernels. The
f32 and Pedantic arms differ between the trees in the cuBLAS call they
make (the 0.6.9 tree calls `sgemm` under the handle's math mode, the
release tree calls `GemmEx` with an explicit compute type), so they are
context, not a control.

| family | shape | M × K × N | op | precision | 0.6.9 kernel µs | 0.7.0 kernel µs | 0.6.9 / 0.7.0 | cuBLAS Fast µs | cuBLAS Pedantic µs | 0.7.0 vs Fast | 0.7.0 vs Pedantic |
|---|---|---|:--:|---|---:|---:|---:|---:|---:|---:|---:|
| Triad | d128 out_proj | 1024 × 256 × 128 | NN | BF16 | 6.58 | 6.59 | 1.00× | 5.38 | 14.63 | 0.82× | 2.22× |
| Triad | d128 out_proj | 1024 × 256 × 128 | NT | BF16 | 5.34 | 5.35 | 1.00× | 4.66 | 11.67 | 0.87× | 2.18× |
| Triad | d128 out_proj | 1024 × 256 × 128 | TN | BF16 | 18.12 | 18.18 | 1.00× | 7.20 | 50.48 | 0.40× | 2.78× |
| Triad | underfill | 256 × 512 × 384 | NN | BF16 | 10.31 | 6.62 | 1.56× | 7.35 | 25.94 | 1.11× | 3.92× |
| Triad | underfill | 256 × 512 × 384 | NT | BF16 | 7.84 | 7.84 | 1.00× | 6.53 | 27.39 | 0.83× | 3.49× |
| Triad | underfill | 256 × 512 × 384 | TN | BF16 | 8.86 | 8.92 | 0.99× | 5.53 | 16.97 | 0.62× | 1.90× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NN | BF16 | 7.39 | 5.76 | 1.28× | 4.81 | 10.44 | 0.84× | 1.81× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NT | BF16 | 9.31 | 9.32 | 1.00× | 7.64 | 35.15 | 0.82× | 3.77× |
| Triad | d128 in_proj | 1024 × 128 × 512 | TN | BF16 | 18.17 | 18.35 | 0.99× | 7.91 | 50.46 | 0.43× | 2.75× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NN | BF16 | 55.55 | 37.95 | 1.46× | 47.42 | 188.76 | 1.25× | 4.97× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NT | BF16 | 60.30 | 41.48 | 1.45× | 39.62 | 206.62 | 0.96× | 4.98× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | TN | BF16 | 74.59 | 55.20 | 1.35× | 50.31 | 213.37 | 0.91× | 3.87× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NN | BF16 | 83.63 | 67.09 | 1.25× | 76.54 | 289.41 | 1.14× | 4.31× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NT | BF16 | 74.75 | 51.89 | 1.44× | 60.92 | 318.86 | 1.17× | 6.14× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | TN | BF16 | 81.36 | 77.67 | 1.05× | 54.77 | 334.08 | 0.71× | 4.30× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NN | BF16 | 96.83 | 72.23 | 1.34× | 82.49 | 395.74 | 1.14× | 5.48× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NT | BF16 | 104.20 | 71.36 | 1.46× | 77.02 | 361.20 | 1.08× | 5.06× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | TN | BF16 | 145.22 | 91.79 | 1.58× | 92.72 | 393.26 | 1.01× | 4.28× |
| Triad | large deep | 4096 × 3072 × 1536 | NN | BF16 | 385.14 | 378.78 | 1.02× | 264.67 | 1634.88 | 0.70× | 4.32× |
| Triad | large deep | 4096 × 3072 × 1536 | NT | BF16 | 419.33 | 408.34 | 1.03× | 268.32 | 1588.74 | 0.66× | 3.89× |
| Triad | large deep | 4096 × 3072 × 1536 | TN | BF16 | 466.08 | 392.93 | 1.19× | 329.65 | 1518.26 | 0.84× | 3.86× |
| Triad | d128 out_proj | 1024 × 256 × 128 | NN | F16 | 6.54 | 6.63 | 0.99× | 4.79 | 9.39 | 0.72× | 1.42× |
| Triad | d128 out_proj | 1024 × 256 × 128 | NT | F16 | 5.36 | 5.34 | 1.00× | 4.49 | 7.25 | 0.84× | 1.36× |
| Triad | d128 out_proj | 1024 × 256 × 128 | TN | F16 | 18.05 | 18.23 | 0.99× | 7.39 | 9.89 | 0.41× | 0.54× |
| Triad | underfill | 256 × 512 × 384 | NN | F16 | 10.30 | 6.61 | 1.56× | 5.73 | 11.56 | 0.87× | 1.75× |
| Triad | underfill | 256 × 512 × 384 | NT | F16 | 7.82 | 7.83 | 1.00× | 5.30 | 11.33 | 0.68× | 1.45× |
| Triad | underfill | 256 × 512 × 384 | TN | F16 | 8.88 | 8.87 | 1.00× | 5.57 | 10.69 | 0.63× | 1.20× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NN | F16 | 7.38 | 5.72 | 1.29× | 5.31 | 9.53 | 0.93× | 1.67× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NT | F16 | 9.40 | 9.40 | 1.00× | 5.85 | 12.64 | 0.62× | 1.34× |
| Triad | d128 in_proj | 1024 × 128 × 512 | TN | F16 | 18.15 | 18.36 | 0.99× | 7.87 | 12.24 | 0.43× | 0.67× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NN | F16 | 58.50 | 41.63 | 1.41× | 48.56 | 165.33 | 1.17× | 3.97× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NT | F16 | 62.42 | 43.40 | 1.44× | 41.48 | 177.80 | 0.96× | 4.10× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | TN | F16 | 74.63 | 58.26 | 1.28× | 51.39 | 157.58 | 0.88× | 2.70× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NN | F16 | 96.13 | 76.28 | 1.26× | 83.41 | 253.54 | 1.09× | 3.32× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NT | F16 | 79.65 | 58.51 | 1.36× | 99.26 | 254.27 | 1.70× | 4.35× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | TN | F16 | 93.82 | 90.10 | 1.04× | 59.13 | 234.39 | 0.66× | 2.60× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NN | F16 | 112.52 | 84.23 | 1.34× | 76.55 | 359.08 | 0.91× | 4.26× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NT | F16 | 118.57 | 85.78 | 1.38× | 77.27 | 333.93 | 0.90× | 3.89× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | TN | F16 | 142.35 | 104.04 | 1.37× | 101.55 | 353.94 | 0.98× | 3.40× |
| Triad | large deep | 4096 × 3072 × 1536 | NN | F16 | 389.36 | 383.41 | 1.02× | 269.13 | 1236.03 | 0.70× | 3.22× |
| Triad | large deep | 4096 × 3072 × 1536 | NT | F16 | 466.09 | 446.35 | 1.04× | 296.18 | 1413.50 | 0.66× | 3.17× |
| Triad | large deep | 4096 × 3072 × 1536 | TN | F16 | 548.05 | 445.72 | 1.23× | 313.72 | 1395.52 | 0.70× | 3.13× |
| Triad | d128 out_proj | 1024 × 256 × 128 | NN | F32 exact | 10.68 | 10.13 | 1.05× | 6.76 | 8.69 | 0.67× | 0.86× |
| Triad | d128 out_proj | 1024 × 256 × 128 | NT | F32 exact | 13.27 | 12.84 | 1.03× | 5.02 | 8.17 | 0.39× | 0.64× |
| Triad | d128 out_proj | 1024 × 256 × 128 | TN | F32 exact | 54.17 | 27.53 | 1.97× | 9.75 | 10.49 | 0.35× | 0.38× |
| Triad | underfill | 256 × 512 × 384 | NN | F32 exact | 13.07 | 12.67 | 1.03× | 9.98 | 12.19 | 0.79× | 0.96× |
| Triad | underfill | 256 × 512 × 384 | NT | F32 exact | 16.00 | 15.69 | 1.02× | 7.99 | 11.76 | 0.51× | 0.75× |
| Triad | underfill | 256 × 512 × 384 | TN | F32 exact | 83.79 | 80.66 | 1.04× | 7.83 | 9.54 | 0.10× | 0.12× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NN | F32 exact | 16.46 | 15.80 | 1.04× | 5.66 | 9.95 | 0.36× | 0.63× |
| Triad | d128 in_proj | 1024 × 128 × 512 | NT | F32 exact | 17.88 | 17.17 | 1.04× | 9.30 | 13.73 | 0.54× | 0.80× |
| Triad | d128 in_proj | 1024 × 128 × 512 | TN | F32 exact | 105.63 | 35.41 | 2.98× | 9.98 | 12.69 | 0.28× | 0.36× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NN | F32 exact | 227.55 | 166.72 | 1.36× | 77.20 | 173.85 | 0.46× | 1.04× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | NT | F32 exact | 369.26 | 181.79 | 2.03× | 86.03 | 189.65 | 0.47× | 1.04× |
| Triad | d768 out_proj | 2048 × 1536 × 768 | TN | F32 exact | 303.73 | 200.17 | 1.52× | 86.13 | 143.73 | 0.43× | 0.72× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NN | F32 exact | 320.72 | 248.44 | 1.29× | 132.70 | 267.06 | 0.53× | 1.07× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | NT | F32 exact | 499.64 | 349.18 | 1.43× | 89.32 | 283.07 | 0.26× | 0.81× |
| Triad | classifier page in_proj | 4621 × 384 × 1928 | TN | F32 exact | 400.47 | 287.24 | 1.39× | 122.05 | 233.84 | 0.42× | 0.81× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NN | F32 exact | 350.68 | 300.03 | 1.17× | 123.26 | 357.63 | 0.41× | 1.19× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | NT | F32 exact | 826.43 | 365.30 | 2.26× | 158.01 | 368.56 | 0.43× | 1.01× |
| Triad | d768 in_proj | 2048 × 768 × 3072 | TN | F32 exact | 530.94 | 326.97 | 1.62× | 137.93 | 277.40 | 0.42× | 0.85× |
| Triad | large deep | 4096 × 3072 × 1536 | NN | F32 exact | 1482.29 | 1644.94 | 0.90× | 427.66 | 1078.08 | 0.26× | 0.66× |
| Triad | large deep | 4096 × 3072 × 1536 | NT | F32 exact | 2859.45 | 2352.26 | 1.22× | 463.22 | 1223.87 | 0.20× | 0.52× |
| Triad | large deep | 4096 × 3072 × 1536 | TN | F32 exact | 1518.21 | 1424.00 | 1.07× | 498.81 | 974.51 | 0.35× | 0.68× |
| Inference | page in_proj (A) | 4621 × 384 × 1928 | NN | BF16 | 87.24 | 68.01 | 1.28× | 75.98 | 278.97 | 1.12× | 4.10× |
| Inference | page narrow (C) | 4621 × 1928 × 384 | NN | BF16 | 69.62 | 54.67 | 1.27× | 76.35 | 298.13 | 1.40× | 5.45× |
| Inference | batched wide (D) | 2048 × 768 × 2304 | NN | BF16 | 89.23 | 65.97 | 1.35× | 60.26 | 278.22 | 0.91× | 4.22× |
| Inference | batched narrow (E) | 2048 × 2304 × 768 | NN | BF16 | 78.56 | 60.65 | 1.30× | 67.80 | 280.68 | 1.12× | 4.63× |
| Inference | page wide (B) | 4621 × 768 × 2304 | NN | BF16 | 189.76 | 132.93 | 1.43× | 109.89 | 762.18 | 0.83× | 5.73× |
| Inference | page in_proj (A) | 4621 × 384 × 1928 | NN | F16 | 100.31 | 76.63 | 1.31× | 81.11 | 246.47 | 1.06× | 3.22× |
| Inference | page narrow (C) | 4621 × 1928 × 384 | NN | F16 | 78.52 | 61.94 | 1.27× | 80.76 | 235.03 | 1.30× | 3.79× |
| Inference | batched wide (D) | 2048 × 768 × 2304 | NN | F16 | 98.58 | 70.93 | 1.39× | 69.09 | 265.75 | 0.97× | 3.75× |
| Inference | batched narrow (E) | 2048 × 2304 × 768 | NN | F16 | 85.94 | 67.88 | 1.27× | 58.99 | 236.83 | 0.87× | 3.49× |
| Inference | page wide (B) | 4621 × 768 × 2304 | NN | F16 | 217.35 | 153.04 | 1.42× | 131.43 | 659.34 | 0.86× | 4.31× |
| Inference | page in_proj (A) | 4621 × 384 × 1928 | NN | F32 exact | 346.05 | 240.49 | 1.44× | 131.81 | 267.54 | 0.55× | 1.11× |
| Inference | page narrow (C) | 4621 × 1928 × 384 | NN | F32 exact | 407.00 | 340.66 | 1.19× | 94.13 | 230.09 | 0.28× | 0.68× |
| Inference | batched wide (D) | 2048 × 768 × 2304 | NN | F32 exact | 303.10 | 216.95 | 1.40× | 103.72 | 253.20 | 0.48× | 1.17× |
| Inference | batched narrow (E) | 2048 × 2304 × 768 | NN | F32 exact | 361.35 | 249.24 | 1.45× | 113.82 | 254.73 | 0.46× | 1.02× |
| Inference | page wide (B) | 4621 × 768 × 2304 | NN | F32 exact | 888.70 | 604.26 | 1.47× | 275.58 | 626.88 | 0.46× | 1.04× |


## Whole model: 0.6.9 against 0.7.0

Measured on the RTX 6000 Ada with the 0.6.9 tree (`d8f2efbe`) and the
release tree built side by side from clean checkouts, the same program and
settings in both, in mirrored blocks of separate processes (old, new, new,
old) with private kernel caches and an idle check before every process.

### Training step

`MambaTrainer::step` with graph replay, three warmups and twenty timed
steps per arm, median of the four runs of each tree, milliseconds per step.
The deterministic arms are like for like (the same precision contract in
both trees); the cuBLAS arms are the same library called from both trees
and agree to within 1 to 2 percent, which is the control for the
comparison.

| model | precision | 0.6.9 deterministic | 0.7.0 deterministic | speedup | cuBLAS Fast | cuBLAS Pedantic |
|---|---|---:|---:|---:|---:|---:|
| d128, 2 layers, B=16, T=64 | f32 | 3.03 | 2.47 | 1.23× | 2.12 | 2.19 |
| d128, 2 layers, B=16, T=64 | bf16, tensor cores | 2.35 | 1.94 | 1.21× | 1.85 | 2.35 |
| d128, 2 layers, B=16, T=64 | f16, tensor cores | 2.39 | 2.00 | 1.19× | 1.90 | 1.99 |
| d256, 4 layers, B=16, T=128 | f32 | 12.46 | 10.99 | 1.13× | 8.85 | 9.38 |
| d256, 4 layers, B=16, T=128 | bf16, tensor cores | 9.55 | 7.63 | 1.25× | 7.18 | 9.39 |
| d256, 4 layers, B=16, T=128 | f16, tensor cores | 9.71 | 7.76 | 1.25× | 7.29 | 7.89 |
| d768, 4 layers, B=8, T=256 | f32 | 34.84 | 24.56 | 1.42× | 16.89 | 20.52 |
| d768, 4 layers, B=8, T=256 | bf16, tensor cores | 22.19 | 13.60 | 1.63× | 13.37 | 19.94 |
| d768, 4 layers, B=8, T=256 | f16, tensor cores | 22.32 | 13.85 | 1.61× | 13.64 | 17.72 |
| d1536, 2 layers, B=4, T=256 | f32 | 24.33 | 19.12 | 1.27× | 11.27 | 15.02 |
| d1536, 2 layers, B=4, T=256 | bf16, tensor cores | 13.06 | 9.52 | 1.37× | 8.96 | 14.58 |
| d1536, 2 layers, B=4, T=256 | f16, tensor cores | 13.39 | 9.93 | 1.35× | 9.32 | 12.95 |

The whole-step gain is the GEMM kernels, the Mamba kernel pass and the
two route changes of this release together: the stream-K weight gradient
on every deep reduction, on by default in the tensor-core tier, and the
parallel scan route from 65 steps on (the changelog's Performance section
has the parts). The cuBLAS arms moved with them, since they share every
kernel but the products. One setting stays off by default: with
`MAMBA_RS_BI_F32_POLICY=tf32` the d768 f32 step takes the deterministic
TF32 kernels (the only shape of the four with a measured one; the others
keep the exact kernels and the same time).

### Inference step

`GpuMambaBackbone::step` on the default synthetic model (d_model 128,
3 layers, 366 K parameters), batches 1 to 128, twenty warmups, 2000 to
5000 eager and 5000 to 10000 graph steps per batch, median of the four
alternating runs of each tree, microseconds per step. Both trees run the
inference family with `batch_invariant` on (`Fixed` in 0.6.9, `Inference`
in 0.7.0); bf16 uses an identity input projection on both sides. The
outputs of the two trees are bit-identical at every batch, step and path,
and the CPU reference agrees (cosine 1.000000 in f32, at least 0.99998 in
bf16).

| storage | batch | path | 0.6.9 µs/step | 0.7.0 µs/step | speedup |
|---|---:|---|---:|---:|---:|
| f32 | 1 | eager | 238.5 | 195.9 | 1.22× |
| f32 | 1 | graph | 199.6 | 170.7 | 1.17× |
| f32 | 4 | eager | 245.9 | 201.3 | 1.22× |
| f32 | 4 | graph | 206.1 | 176.3 | 1.17× |
| f32 | 16 | eager | 246.7 | 208.1 | 1.19× |
| f32 | 16 | graph | 207.8 | 183.2 | 1.13× |
| f32 | 64 | eager | 271.0 | 216.0 | 1.25× |
| f32 | 64 | graph | 232.6 | 192.5 | 1.21× |
| f32 | 128 | eager | 296.8 | 240.6 | 1.23× |
| f32 | 128 | graph | 259.4 | 218.7 | 1.19× |
| bf16 | 1 | eager | 124.1 | 113.7 | 1.09× |
| bf16 | 1 | graph | 93.9 | 83.3 | 1.13× |
| bf16 | 4 | eager | 128.8 | 110.9 | 1.16× |
| bf16 | 4 | graph | 97.8 | 84.6 | 1.16× |
| bf16 | 16 | eager | 132.5 | 116.2 | 1.14× |
| bf16 | 16 | graph | 102.1 | 87.8 | 1.16× |
| bf16 | 64 | eager | 146.4 | 126.4 | 1.16× |
| bf16 | 64 | graph | 117.8 | 105.3 | 1.12× |
| bf16 | 128 | eager | 197.5 | 156.2 | 1.26× |
| bf16 | 128 | graph | 167.5 | 133.6 | 1.25× |

The same step on the training family with exact f32 kernels, the one
route both trees share exactly (the digests are identical here too):

| batch | path | 0.6.9 µs/step | 0.7.0 µs/step | speedup |
|---:|---|---:|---:|---:|
| 1 | eager | 141.6 | 105.2 | 1.35× |
| 1 | graph | 102.8 | 79.0 | 1.30× |
| 4 | eager | 146.4 | 106.3 | 1.38× |
| 4 | graph | 106.7 | 80.9 | 1.32× |
| 16 | eager | 150.9 | 108.9 | 1.39× |
| 16 | graph | 112.0 | 85.4 | 1.31× |
| 64 | eager | 208.3 | 159.6 | 1.31× |
| 64 | graph | 164.9 | 132.1 | 1.25× |
| 128 | eager | 279.5 | 228.2 | 1.22× |
| 128 | graph | 237.2 | 201.8 | 1.18× |

A model this small spends its step on kernel launches rather than on
arithmetic, so the new GEMM kernels alone left both tables where 0.6.9
had them (within 3 percent from batch 1 to 64). The gains above come from
the Mamba kernel pass, which fused the decode step into seven kernels per
layer instead of eleven on both families; the bit identity of the outputs
was checked on both trees. The first measurement of this step on the
release tree was 45 µs per graph replay slower than 0.6.9 at every batch;
that was host-side work in the replay validation (a launch set digest
rebuilt on every replay), fixed in 0.7.0 before release.



## How the numbers were taken

The kernel comparisons against cuBLAS use one harness
(`tools/qualification/gemm_bi_performance_matrix.rs`) on an idle GPU. For
each cell the deterministic kernel and the cuBLAS kernel are timed with
CUDA events in alternating windows, 21 windows in each of two mirrored
orders, so 42 paired ratios per cell and path. The number reported is the
inverse of the lower median of those 42 ratios; the microsecond columns are
the lower medians of the two arms and are printed for scale. A median of
paired ratios is not the ratio of the two medians, so the columns do not
divide exactly into each other. Each arm's launch count per window is
calibrated so a window lasts a few milliseconds. The deterministic arm is
the production automatic selection with no forced tile; the kernel name in
each row is the symbol that actually ran.

The old-versus-new kernel tables use one adapter compiled against both
trees (`internal/perf/old-versus-new-20260910/adapter/`), timing the
deterministic route, cuBLAS Fast and cuBLAS Pedantic in the same process
with the same event protocol, in mirrored blocks of separate processes
(old, new, new, old). The whole-model tables use the same block layout with
one benchmark-only program per tree.

The raw records, the verification scripts and the run metadata are kept
under `internal/perf/final-auto-benchmarks-20260910/` and
`internal/perf/old-versus-new-20260910/` in the repository.

## Per-kernel tables: Inference family

The five serving shapes are the projections of a vision classifier page
(4621 rows) and two batched training-class projections (2048 rows). Kernel
names keep the `fixed` prefix of the family's old name; the family is
selected as `inference`. Columns: mamba-rs and cuBLAS microseconds per
launch, and the speedup, for the eager and the graph path.

### RTX 6000 Ada

#### BF16 in, BF16 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_fixed_sm89_tc128_pipeline_v1_bf16` | 64.16 | 74.90 | 1.167× | 63.75 | 74.53 | 1.169× |
| 4621 × 384 × 1928 | on | `nn_fixed_sm89_tc128_pipeline_v1_bf16` | 65.40 | 108.67 | 1.662× | 65.00 | 107.96 | 1.661× |
| 4621 × 768 × 2304 | off | `nn_fixed_sm89_tc128_s3_v1_bf16` | 126.12 | 103.83 | 0.822× | 126.22 | 103.48 | 0.821× |
| 4621 × 768 × 2304 | on | `nn_fixed_sm89_tc128_swizzle_v1_bf16` | 123.08 | 124.51 | 1.012× | 123.50 | 124.34 | 1.007× |
| 4621 × 1928 × 384 | off | `nn_fixed_sm89_tc128_pipeline_v1_bf16` | 54.05 | 74.69 | 1.382× | 53.88 | 75.01 | 1.393× |
| 4621 × 1928 × 384 | on | `nn_fixed_sm89_tc128_pipeline_v1_bf16` | 54.35 | 84.10 | 1.547× | 54.32 | 85.60 | 1.576× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm89_tc128_swizzle_v1_bf16` | 65.62 | 60.40 | 0.921× | 65.06 | 59.12 | 0.909× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm89_tc128_swizzle_v1_bf16` | 66.41 | 80.15 | 1.207× | 65.67 | 78.66 | 1.198× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm89_tc128_swizzle_v1_bf16` | 60.61 | 68.76 | 1.135× | 60.58 | 68.69 | 1.134× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm89_tc128_swizzle_v1_bf16` | 60.87 | 77.46 | 1.273× | 60.70 | 76.38 | 1.259× |

Geometric mean over the ten cells: eager 1.187×, graph 1.185× (mamba-rs faster in 8 of 10 cells eager, 8 of 10 graph).

#### F16 in, F16 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_fixed_sm89_tc128_pipeline_v1_f16` | 66.83 | 75.20 | 1.125× | 66.97 | 75.02 | 1.120× |
| 4621 × 384 × 1928 | on | `nn_fixed_sm89_tc128_pipeline_v1_f16` | 65.40 | 108.35 | 1.657× | 65.00 | 107.60 | 1.655× |
| 4621 × 768 × 2304 | off | `nn_fixed_sm89_tc128_s3_v1_f16` | 139.02 | 119.87 | 0.862× | 138.79 | 119.30 | 0.861× |
| 4621 × 768 × 2304 | on | `nn_fixed_sm89_tc128_swizzle_v1_f16` | 134.47 | 140.97 | 1.048× | 135.34 | 140.13 | 1.040× |
| 4621 × 1928 × 384 | off | `nn_fixed_sm89_tc128_pipeline_v1_f16` | 56.88 | 75.93 | 1.335× | 57.22 | 76.64 | 1.343× |
| 4621 × 1928 × 384 | on | `nn_fixed_sm89_tc128_pipeline_v1_f16` | 54.97 | 84.28 | 1.536× | 54.57 | 85.38 | 1.565× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm89_m64n64_bk64_s3_v1_f16` | 72.14 | 70.96 | 0.979× | 73.02 | 70.47 | 0.966× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm89_tc128_swizzle_v1_f16` | 66.41 | 80.20 | 1.208× | 65.64 | 78.64 | 1.198× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm89_m128n64_bk64_s2_v1_f16` | 60.70 | 62.38 | 1.028× | 60.54 | 62.19 | 1.027× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm89_tc128_pipeline_v1_f16` | 61.88 | 69.23 | 1.119× | 61.72 | 68.51 | 1.110× |

Geometric mean over the ten cells: eager 1.167×, graph 1.165× (mamba-rs faster in 8 of 10 cells eager, 8 of 10 graph).

#### BF16 in, F32 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_tc64_f32out_bf16` | 93.27 | 78.77 | 0.847× | 93.36 | 79.21 | 0.848× |
| 4621 × 384 × 1928 | on | `nn_tc64_f32out_bf16` | 86.92 | 111.57 | 1.284× | 87.06 | 111.35 | 1.279× |
| 4621 × 768 × 2304 | off | `nn_tc128_f32out_bf16` | 227.58 | 114.64 | 0.505× | 229.20 | 115.21 | 0.503× |
| 4621 × 768 × 2304 | on | `nn_tc128_f32out_bf16` | 195.24 | 132.66 | 0.674× | 194.52 | 131.79 | 0.678× |
| 4621 × 1928 × 384 | off | `nn_tc128_f32out_bf16` | 77.41 | 76.19 | 0.984× | 77.24 | 75.67 | 0.979× |
| 4621 × 1928 × 384 | on | `nn_tc128_f32out_bf16` | 77.59 | 86.16 | 1.110× | 77.45 | 86.78 | 1.121× |
| 2048 × 768 × 2304 | off | `nn_tc64_f32out_bf16` | 100.49 | 62.69 | 0.624× | 101.90 | 61.94 | 0.610× |
| 2048 × 768 × 2304 | on | `nn_tc64_f32out_bf16` | 98.44 | 80.62 | 0.819× | 98.22 | 79.41 | 0.808× |
| 2048 × 2304 × 768 | off | `nn_tc128_f32out_bf16` | 85.13 | 68.73 | 0.808× | 84.91 | 68.69 | 0.809× |
| 2048 × 2304 × 768 | on | `nn_tc128_f32out_bf16` | 85.25 | 77.71 | 0.912× | 85.08 | 76.89 | 0.904× |

Geometric mean over the ten cells: eager 0.828×, graph 0.825× (mamba-rs faster in 2 of 10 cells eager, 2 of 10 graph).

#### F16 in, F32 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_tc64_f32out_f16` | 102.72 | 85.12 | 0.828× | 103.68 | 86.26 | 0.832× |
| 4621 × 384 × 1928 | on | `nn_tc64_f32out_f16` | 93.67 | 117.18 | 1.253× | 93.88 | 117.65 | 1.253× |
| 4621 × 768 × 2304 | off | `nn_tc128_f32out_f16` | 245.48 | 131.45 | 0.537× | 245.31 | 131.37 | 0.535× |
| 4621 × 768 × 2304 | on | `nn_tc128_f32out_f16` | 211.39 | 148.33 | 0.702× | 211.09 | 147.85 | 0.699× |
| 4621 × 1928 × 384 | off | `nn_tc128_f32out_f16` | 77.43 | 76.23 | 0.985× | 77.23 | 75.61 | 0.979× |
| 4621 × 1928 × 384 | on | `nn_tc128_f32out_f16` | 77.60 | 86.17 | 1.111× | 77.43 | 86.83 | 1.122× |
| 2048 × 768 × 2304 | off | `nn_tc64_f32out_f16` | 108.18 | 67.86 | 0.629× | 110.25 | 68.20 | 0.619× |
| 2048 × 768 × 2304 | on | `nn_tc64_f32out_f16` | 102.60 | 82.07 | 0.800× | 103.54 | 81.22 | 0.786× |
| 2048 × 2304 × 768 | off | `nn_tc128_f32out_f16` | 85.11 | 62.62 | 0.736× | 84.89 | 62.14 | 0.732× |
| 2048 × 2304 × 768 | on | `nn_tc128_f32out_f16` | 85.25 | 69.69 | 0.817× | 85.04 | 69.01 | 0.812× |

Geometric mean over the ten cells: eager 0.815×, graph 0.812× (mamba-rs faster in 2 of 10 cells eager, 2 of 10 graph).

#### F32 in, F32 out, deterministic TF32, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 111.40 | 103.00 | 0.924× | 110.41 | 102.59 | 0.929× |
| 4621 × 384 × 1928 | on | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 113.87 | 131.21 | 1.152× | 113.18 | 130.13 | 1.150× |
| 4621 × 768 × 2304 | off | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 234.10 | 213.24 | 0.911× | 234.39 | 214.11 | 0.913× |
| 4621 × 768 × 2304 | on | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 234.87 | 245.60 | 1.045× | 234.70 | 244.20 | 1.046× |
| 4621 × 1928 × 384 | off | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 106.42 | 81.14 | 0.762× | 107.86 | 82.09 | 0.761× |
| 4621 × 1928 × 384 | on | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 105.18 | 88.12 | 0.838× | 105.74 | 87.87 | 0.831× |
| 2048 × 768 × 2304 | off | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 125.39 | 99.37 | 0.793× | 123.15 | 98.75 | 0.802× |
| 2048 × 768 × 2304 | on | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 126.69 | 116.37 | 0.919× | 125.14 | 114.92 | 0.918× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3` | 106.62 | 95.75 | 0.898× | 108.13 | 97.02 | 0.897× |
| 2048 × 2304 × 768 | on | `nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3` | 119.56 | 98.74 | 0.826× | 119.18 | 97.79 | 0.821× |

Geometric mean over the ten cells: eager 0.900×, graph 0.900× (mamba-rs faster in 2 of 10 cells eager, 2 of 10 graph).

#### F32 in, F32 out, exact, against cuBLAS Pedantic (COMPUTE_32F_PEDANTIC)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 285.58 | 315.33 | 1.105× | 288.87 | 318.85 | 1.104× |
| 4621 × 384 × 1928 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 265.95 | 321.67 | 1.210× | 268.59 | 324.39 | 1.208× |
| 4621 × 768 × 2304 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 744.74 | 755.57 | 1.015× | 747.08 | 755.42 | 1.013× |
| 4621 × 768 × 2304 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 686.55 | 734.79 | 1.070× | 686.52 | 732.89 | 1.068× |
| 4621 × 1928 × 384 | off | `f32_f32_s2` | 361.40 | 239.15 | 0.668× | 373.96 | 251.72 | 0.667× |
| 4621 × 1928 × 384 | on | `f32_f32_s2` | 359.15 | 246.78 | 0.687× | 379.02 | 259.84 | 0.686× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 232.40 | 278.85 | 1.191× | 233.98 | 278.15 | 1.188× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 233.04 | 293.55 | 1.259× | 234.73 | 295.19 | 1.257× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 284.44 | 287.29 | 1.011× | 300.43 | 304.30 | 1.008× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 281.83 | 292.14 | 1.036× | 298.78 | 308.52 | 1.033× |

Geometric mean over the ten cells: eager 1.004×, graph 1.002× (mamba-rs faster in 8 of 10 cells eager, 8 of 10 graph).

#### F32 in, F32 out, exact, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 234.06 | 128.59 | 0.549× | 226.94 | 122.38 | 0.547× |
| 4621 × 384 × 1928 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 225.97 | 151.88 | 0.670× | 228.78 | 151.55 | 0.667× |
| 4621 × 768 × 2304 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 567.17 | 256.00 | 0.451× | 560.77 | 251.85 | 0.450× |
| 4621 × 768 × 2304 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 557.29 | 287.99 | 0.517× | 557.18 | 286.78 | 0.514× |
| 4621 × 1928 × 384 | off | `f32_f32_s2` | 309.06 | 85.37 | 0.276× | 314.37 | 86.49 | 0.275× |
| 4621 × 1928 × 384 | on | `f32_f32_s2` | 301.90 | 91.45 | 0.303× | 307.06 | 92.00 | 0.300× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 236.35 | 115.98 | 0.491× | 238.31 | 115.98 | 0.486× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 232.93 | 131.49 | 0.565× | 232.59 | 130.10 | 0.559× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm89_f32_n64_copyplan_v1` | 238.93 | 108.50 | 0.454× | 243.27 | 110.01 | 0.453× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm89_f32_n64_copyplan_v1` | 237.35 | 115.46 | 0.486× | 237.21 | 114.47 | 0.483× |

Geometric mean over the ten cells: eager 0.462×, graph 0.459× (mamba-rs faster in 0 of 10 cells eager, 0 of 10 graph).

### RTX 5090

#### BF16 in, BF16 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_128x64_bk32_s3_bf16` | 37.84 | 43.08 | 1.139× | 38.93 | 44.92 | 1.154× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_128x64_bk32_s3_bf16` | 37.99 | 61.96 | 1.631× | 38.93 | 63.48 | 1.631× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_128x128_bk32_s3_bf16` | 74.03 | 77.09 | 1.041× | 75.76 | 77.88 | 1.028× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_128x128_bk32_s3_bf16` | 74.25 | 99.24 | 1.337× | 75.77 | 100.35 | 1.324× |
| 4621 × 1928 × 384 | off | `nn_sm120_tma_64x64_bk64_s2_bf16` | 40.19 | 45.98 | 1.144× | 41.01 | 47.12 | 1.149× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_64x64_bk64_s2_bf16` | 40.23 | 51.18 | 1.272× | 41.01 | 51.26 | 1.250× |
| 2048 × 768 × 2304 | off | `nn_sm120_tma_64x64_bk64_s2_bf16` | 36.91 | 39.69 | 1.075× | 38.76 | 40.96 | 1.057× |
| 2048 × 768 × 2304 | on | `nn_sm120_tma_64x64_bk64_s2_bf16` | 37.01 | 50.86 | 1.374× | 38.87 | 51.24 | 1.318× |
| 2048 × 2304 × 768 | off | `nn_sm120_tma_64x64_bk64_s2_bf16` | 44.16 | 53.37 | 1.209× | 45.09 | 55.29 | 1.226× |
| 2048 × 2304 × 768 | on | `nn_sm120_tma_64x64_bk64_s2_bf16` | 44.40 | 58.13 | 1.310× | 45.09 | 59.39 | 1.317× |

Geometric mean over the ten cells: eager 1.243×, graph 1.235× (mamba-rs faster in 10 of 10 cells eager, 10 of 10 graph).

#### F16 in, F16 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_128x64_bk32_s3_f16` | 38.30 | 43.43 | 1.133× | 38.94 | 45.05 | 1.157× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_128x64_bk32_s3_f16` | 38.43 | 62.04 | 1.615× | 38.99 | 63.48 | 1.628× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_128x128_bk32_s3_f16` | 74.82 | 78.09 | 1.044× | 75.83 | 79.86 | 1.053× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_128x128_bk32_s3_f16` | 75.03 | 100.08 | 1.334× | 75.80 | 100.42 | 1.325× |
| 4621 × 1928 × 384 | off | `nn_sm120_tma_64x64_bk64_s2_f16` | 40.31 | 45.89 | 1.139× | 41.06 | 47.10 | 1.147× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_64x64_bk64_s2_f16` | 40.32 | 51.28 | 1.272× | 41.04 | 51.53 | 1.256× |
| 2048 × 768 × 2304 | off | `nn_sm120_tma_128x128_bk32_s3_f16` | 38.40 | 39.65 | 1.033× | 38.95 | 40.96 | 1.052× |
| 2048 × 768 × 2304 | on | `nn_sm120_tma_64x64_bk64_s2_f16` | 37.52 | 51.12 | 1.362× | 38.93 | 51.32 | 1.318× |
| 2048 × 2304 × 768 | off | `nn_sm120_tma_64x64_bk64_s2_f16` | 44.20 | 53.23 | 1.205× | 45.09 | 55.16 | 1.223× |
| 2048 × 2304 × 768 | on | `nn_sm120_tma_64x64_bk64_s2_f16` | 44.45 | 58.22 | 1.310× | 45.11 | 59.39 | 1.317× |

Geometric mean over the ten cells: eager 1.234×, graph 1.238× (mamba-rs faster in 10 of 10 cells eager, 10 of 10 graph).

#### BF16 in, F32 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_128x64_bk32_s3_f32out_bf16` | 38.85 | 47.43 | 1.221× | 39.68 | 49.15 | 1.239× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_128x64_bk32_s3_f32out_bf16` | 39.01 | 66.59 | 1.707× | 40.06 | 67.60 | 1.687× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_128x128_bk32_s3_f32out_bf16` | 75.68 | 82.52 | 1.090× | 76.92 | 83.96 | 1.092× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_128x128_bk32_s3_f32out_bf16` | 75.88 | 105.14 | 1.386× | 77.44 | 106.49 | 1.375× |
| 4621 × 1928 × 384 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 40.19 | 46.78 | 1.164× | 41.02 | 47.15 | 1.149× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 40.69 | 52.05 | 1.279× | 41.68 | 53.24 | 1.278× |
| 2048 × 768 × 2304 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 37.03 | 42.27 | 1.141× | 38.90 | 43.04 | 1.106× |
| 2048 × 768 × 2304 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 37.09 | 53.32 | 1.437× | 38.92 | 54.14 | 1.391× |
| 2048 × 2304 × 768 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 44.15 | 54.30 | 1.230× | 45.08 | 55.33 | 1.227× |
| 2048 × 2304 × 768 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_bf16` | 44.46 | 59.21 | 1.331× | 45.10 | 59.43 | 1.318× |

Geometric mean over the ten cells: eager 1.288×, graph 1.276× (mamba-rs faster in 10 of 10 cells eager, 10 of 10 graph).

#### F16 in, F32 out, against cuBLAS Fast (COMPUTE_32F)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_128x64_bk32_s3_f32out_f16` | 39.43 | 47.57 | 1.205× | 40.88 | 49.15 | 1.202× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_128x64_bk32_s3_f32out_f16` | 39.62 | 66.65 | 1.683× | 40.95 | 67.62 | 1.651× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_128x128_bk32_s3_f32out_f16` | 77.05 | 82.88 | 1.079× | 77.80 | 83.99 | 1.080× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_128x128_bk32_s3_f32out_f16` | 77.07 | 105.45 | 1.369× | 77.82 | 106.49 | 1.369× |
| 4621 × 1928 × 384 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 40.29 | 47.00 | 1.166× | 41.06 | 48.07 | 1.171× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 40.79 | 52.25 | 1.281× | 41.98 | 53.26 | 1.269× |
| 2048 × 768 × 2304 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 37.69 | 42.37 | 1.124× | 38.92 | 43.04 | 1.106× |
| 2048 × 768 × 2304 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 37.69 | 53.37 | 1.416× | 38.93 | 54.54 | 1.402× |
| 2048 × 2304 × 768 | off | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 44.22 | 54.34 | 1.229× | 45.08 | 55.33 | 1.227× |
| 2048 × 2304 × 768 | on | `nn_sm120_tma_64x64_bk64_s2_f32out_f16` | 44.55 | 59.26 | 1.331× | 45.16 | 59.51 | 1.319× |

Geometric mean over the ten cells: eager 1.278×, graph 1.270× (mamba-rs faster in 10 of 10 cells eager, 10 of 10 graph).

#### F32 in, F32 out, deterministic TF32, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_tf32_v1_m128n64_bk32_s2` | 71.95 | 80.97 | 1.125× | 73.62 | 81.95 | 1.113× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_tf32_v1_m128n64_bk32_s2` | 72.02 | 100.20 | 1.391× | 73.68 | 100.46 | 1.365× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_tf32_v1_m128n64_bk32_s2` | 160.85 | 153.09 | 0.956× | 160.98 | 153.72 | 0.958× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_tf32_v1_m128n64_bk32_s2` | 160.02 | 177.64 | 1.110× | 160.25 | 178.30 | 1.115× |
| 4621 × 1928 × 384 | off | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2` | 79.40 | 89.80 | 1.131× | 80.05 | 90.18 | 1.127× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2` | 79.54 | 94.95 | 1.194× | 80.21 | 96.25 | 1.200× |
| 2048 × 768 × 2304 | off | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store` | 75.63 | 75.25 | 0.995× | 76.41 | 75.83 | 0.993× |
| 2048 × 768 × 2304 | on | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store` | 75.79 | 85.78 | 1.132× | 76.66 | 86.08 | 1.124× |
| 2048 × 2304 × 768 | off | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2` | 92.17 | 105.39 | 1.144× | 93.74 | 106.52 | 1.136× |
| 2048 × 2304 × 768 | on | `nn_sm120_tma_tf32_v1_m64n64_bk32_s2` | 92.65 | 110.70 | 1.195× | 94.20 | 110.90 | 1.177× |

Geometric mean over the ten cells: eager 1.132×, graph 1.126× (mamba-rs faster in 8 of 10 cells eager, 8 of 10 graph).

#### F32 in, F32 out, exact, against cuBLAS Pedantic (COMPUTE_32F_PEDANTIC)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_fma_v1_m64n128_bk16_s2` | 113.83 | 139.17 | 1.221× | 114.04 | 133.24 | 1.170× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n64_t256_bk16_s2` | 114.71 | 157.70 | 1.376× | 116.30 | 157.97 | 1.375× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 267.00 | 306.15 | 1.146× | 257.32 | 291.88 | 1.135× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2` | 266.70 | 324.53 | 1.213× | 260.54 | 316.63 | 1.207× |
| 4621 × 1928 × 384 | off | `f32_f32_s2` | 150.07 | 128.29 | 0.857× | 150.51 | 127.25 | 0.845× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n96_bk16_s2` | 144.55 | 129.68 | 0.895× | 145.03 | 129.41 | 0.893× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm120_f32_n64_copyplan_v1` | 114.65 | 125.37 | 1.126× | 121.69 | 137.41 | 1.122× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm120_f32_n64_copyplan_v1` | 120.13 | 149.21 | 1.239× | 119.83 | 148.37 | 1.237× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm120_f32_n64_copyplan_v1` | 136.03 | 125.23 | 0.921× | 137.26 | 125.15 | 0.912× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm120_f32_n64_copyplan_v1` | 136.57 | 131.49 | 0.965× | 137.26 | 132.84 | 0.968× |

Geometric mean over the ten cells: eager 1.083×, graph 1.074× (mamba-rs faster in 6 of 10 cells eager, 6 of 10 graph).

#### F32 in, F32 out, exact, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| M × K × N | bias | mamba-rs kernel | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---:|---:|---:|---:|---:|---:|
| 4621 × 384 × 1928 | off | `nn_sm120_tma_fma_v1_m64n128_bk16_s2` | 109.71 | 81.25 | 0.743× | 110.22 | 82.01 | 0.744× |
| 4621 × 384 × 1928 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n64_t256_bk16_s2` | 111.97 | 100.41 | 0.897× | 112.59 | 101.47 | 0.901× |
| 4621 × 768 × 2304 | off | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 225.58 | 162.73 | 0.716× | 225.85 | 167.94 | 0.744× |
| 4621 × 768 × 2304 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2` | 230.43 | 195.27 | 0.842× | 230.57 | 192.72 | 0.836× |
| 4621 × 1928 × 384 | off | `f32_f32_s2` | 132.84 | 90.01 | 0.678× | 133.54 | 90.87 | 0.681× |
| 4621 × 1928 × 384 | on | `nn_sm120_tma_fma_v1_fixed_postbias_m128n96_bk16_s2` | 128.26 | 95.65 | 0.743× | 129.07 | 96.27 | 0.746× |
| 2048 × 768 × 2304 | off | `nn_fixed_sm120_f32_n64_copyplan_v1` | 111.01 | 76.03 | 0.683× | 112.49 | 77.91 | 0.693× |
| 2048 × 768 × 2304 | on | `nn_fixed_sm120_f32_n64_copyplan_v1` | 111.45 | 87.15 | 0.783× | 112.30 | 86.81 | 0.782× |
| 2048 × 2304 × 768 | off | `nn_fixed_sm120_f32_n64_copyplan_v1` | 129.75 | 105.99 | 0.817× | 130.99 | 106.56 | 0.814× |
| 2048 × 2304 × 768 | on | `nn_fixed_sm120_f32_n64_copyplan_v1` | 129.80 | 110.83 | 0.854× | 131.00 | 111.75 | 0.853× |

Geometric mean over the ten cells: eager 0.772×, graph 0.776× (mamba-rs faster in 0 of 10 cells eager, 0 of 10 graph).

## Per-kernel tables: Triad family

Shapes: `d128` and `d768` are the input and output projections of models
with those hidden sizes, `underfill` is a shape too small to fill the GPU,
`large deep` is a 4096-row product with a 3072-long reduction, and
`classifier page in_proj` is the 4621-row page projection. NN is the
forward product, TN the weight gradient, NT the input gradient. A kernel
chain joined with `+` is a route of more than one launch (a transpose or a
split reduction followed by its reducer).

### RTX 6000 Ada

#### BF16 in; BF16 out for NN and NT, F32 out for TN, against cuBLAS Fast (COMPUTE_32F)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_tc64_bf16` | 5.680 | 4.834 | 0.851× | 5.338 | 4.857 | 0.910× |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_tc64_bf16` | 9.342 | 7.679 | 0.822× | 9.230 | 7.218 | 0.782× |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_tc64_bf16` | 18.427 | 8.018 | 0.435× | 18.203 | 7.453 | 0.409× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_tc64_bf16` | 6.693 | 5.444 | 0.813× | 6.515 | 5.054 | 0.776× |
| d128 out_proj | NT | 1024 × 256 × 128 | `nt_tc64_bf16` | 5.464 | 4.691 | 0.859× | 5.221 | 4.249 | 0.814× |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_tc64_bf16` | 18.429 | 7.194 | 0.390× | 18.212 | 6.528 | 0.358× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm89_m128n128_bk64_s3_v1_bf16` | 62.270 | 77.295 | 1.241× | 61.773 | 76.377 | 1.237× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm89_m128n128_bk64_s3_bxor_v1_bf16` | 70.982 | 77.890 | 1.098× | 70.671 | 77.616 | 1.098× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_bf16` | 86.950 | 91.972 | 1.058× | 86.687 | 88.519 | 1.021× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm89_m128n128_bk64_s3_v1_bf16` | 38.299 | 49.052 | 1.280× | 37.822 | 47.833 | 1.265× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm89_m96n128_bk64_s3_v1_bf16` | 36.689 | 40.043 | 1.092× | 36.301 | 39.131 | 1.078× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_bf16` | 55.272 | 50.697 | 0.917× | 55.216 | 49.356 | 0.894× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm89_m128n128_bk64_s3_v1_bf16` | 59.923 | 76.179 | 1.281× | 59.209 | 74.752 | 1.262× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm89_m128n128_bk64_s3_bxor_v1_bf16` | 48.464 | 61.007 | 1.259× | 48.399 | 60.330 | 1.246× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_v1_bf16` | 77.583 | 54.659 | 0.705× | 76.643 | 53.433 | 0.697× |

Geometric mean over the 15 cells: eager 0.890×, graph 0.869× (mamba-rs faster in 7 of 15 cells eager, 7 of 15 graph).

#### F16 in; F16 out for NN and NT, F32 out for TN, against cuBLAS Fast (COMPUTE_32F)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_tc64_f16` | 5.680 | 5.385 | 0.948× | 5.341 | 5.101 | 0.955× |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_tc64_f16` | 9.349 | 5.899 | 0.631× | 9.237 | 5.255 | 0.569× |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_tc64_f16` | 18.431 | 7.983 | 0.433× | 18.119 | 7.477 | 0.413× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_tc64_f16` | 6.601 | 4.756 | 0.723× | 6.511 | 4.174 | 0.641× |
| d128 out_proj | NT | 1024 × 256 × 128 | `nt_tc64_f16` | 5.456 | 4.581 | 0.840× | 5.232 | 4.053 | 0.775× |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_tc64_f16` | 18.262 | 7.518 | 0.412× | 18.013 | 6.581 | 0.365× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm89_m128n128_bk64_s3_v1_f16` | 62.270 | 59.291 | 0.953× | 61.786 | 58.786 | 0.952× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm89_m128n128_bk64_s3_bxor_v1_f16` | 70.997 | 75.695 | 1.067× | 70.700 | 77.063 | 1.090× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_v1_f16` | 86.753 | 91.286 | 1.052× | 86.727 | 88.612 | 1.022× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm89_m128n128_bk64_s3_v1_f16` | 38.028 | 47.585 | 1.251× | 37.725 | 47.249 | 1.253× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm89_m96n128_bk64_s3_v1_f16` | 36.683 | 40.186 | 1.096× | 36.316 | 39.200 | 1.079× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm89_m64n64_bk64_s2_compact_bxor_v1_f16` | 56.249 | 50.717 | 0.902× | 56.191 | 49.331 | 0.878× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm89_m128n128_bk64_s3_v1_f16` | 59.742 | 74.815 | 1.252× | 59.194 | 74.135 | 1.252× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm89_m128n128_bk64_s3_bxor_v1_f16` | 48.466 | 96.338 | 1.988× | 48.419 | 95.406 | 1.971× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_v1_f16` | 77.566 | 54.615 | 0.704× | 76.595 | 53.454 | 0.698× |

Geometric mean over the 15 cells: eager 0.881×, graph 0.851× (mamba-rs faster in 6 of 15 cells eager, 6 of 15 graph).

#### F32 in, F32 out, deterministic TF32, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| underfill | NN | 256 × 512 × 384 | `nn_sm80_mma_tf32_splitk2_v1_m16n32_bk32_s4` | 11.169 | 10.019 | 0.897× | 10.742 | 9.758 | 0.908× |
| underfill | NT | 256 × 512 × 384 | `nt_sm80_mma_tf32_v1_m16n32_bk32_s4` | 10.129 | 7.993 | 0.789× | 10.132 | 7.682 | 0.758× |
| underfill | TN | 256 × 512 × 384 | `tn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 10.468 | 7.828 | 0.748× | 10.268 | 7.837 | 0.763× |
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_sm80_mma_tf32_v1_m64n64_bk32_s2` | 10.010 | 5.746 | 0.574× | 9.738 | 5.558 | 0.571× |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3` | 12.094 | 9.377 | 0.775× | 11.869 | 9.059 | 0.763× |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 19.094 | 9.998 | 0.524× | 19.092 | 9.382 | 0.491× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 7.661 | 6.820 | 0.890× | 7.561 | 6.511 | 0.860× |
| d128 out_proj | NT | 1024 × 256 × 128 | `nt_sm80_mma_tf32_v1_m16n32_bk32_s4` | 7.894 | 5.141 | 0.651× | 7.808 | 4.969 | 0.638× |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 19.038 | 9.821 | 0.516× | 18.829 | 8.953 | 0.475× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm80_mma_tf32_v1_m128n128_bk32_s3` | 123.649 | 101.701 | 0.823× | 121.881 | 100.987 | 0.829× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3_v1` | 186.780 | 117.156 | 0.627× | 186.520 | 116.807 | 0.626× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm89_tf32_pre_rna_transpose_32x32_v1 + tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1` | 175.386 | 135.216 | 0.771× | 172.985 | 132.109 | 0.764× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3_v1` | 64.184 | 64.124 | 0.999× | 63.604 | 62.879 | 0.989× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2` | 93.860 | 66.813 | 0.712× | 93.590 | 66.277 | 0.708× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm89_tf32_pre_rna_transpose_32x32_v1 + tn_sm89_tf32_pre_rna_m128n96_bk32_s3_v1` | 94.189 | 82.818 | 0.879× | 93.487 | 82.333 | 0.881× |
| large deep | NN | 4096 × 3072 × 1536 | `nn_sm80_mma_tf32_v1_m128n128_bk32_s3` | 490.325 | 390.071 | 0.796× | 493.568 | 392.670 | 0.796× |
| large deep | NT | 4096 × 3072 × 1536 | `nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2` | 658.631 | 424.521 | 0.645× | 661.618 | 426.496 | 0.645× |
| large deep | TN | 4096 × 3072 × 1536 | `tn_sm80_mma_tf32_v1_m64n64_bk32_s2` | 896.512 | 474.950 | 0.530× | 895.488 | 470.723 | 0.526× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct_v1` | 104.277 | 103.248 | 0.990× | 102.758 | 103.215 | 1.005× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2` | 144.683 | 76.944 | 0.532× | 144.323 | 76.594 | 0.531× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm89_tf32_pre_rna_transpose_32x32_v1 + tn_sm89_tf32_pre_rna_m64n96_bk32_s2_v1` | 140.317 | 100.243 | 0.714× | 138.917 | 99.350 | 0.715× |

Geometric mean over the 21 cells: eager 0.718×, graph 0.709× (mamba-rs faster in 0 of 21 cells eager, 1 of 21 graph).

#### F32 in, F32 out, exact, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_splitk32_partial + splitk_reduce` | 15.721 | 5.745 | 0.366× | 15.056 | 5.573 | 0.370× |
| d128 in_proj | NT | 1024 × 128 × 512 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 17.271 | 9.377 | 0.543× | 15.744 | 9.059 | 0.575× |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm89_f32_d128_in_m16n16_f64fold_v1` | 35.271 | 10.008 | 0.284× | 35.260 | 9.381 | 0.266× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_splitk32_partial + splitk_reduce` | 10.220 | 6.818 | 0.667× | 9.517 | 6.494 | 0.682× |
| d128 out_proj | NT | 1024 × 256 × 128 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 13.028 | 5.147 | 0.395× | 11.521 | 4.963 | 0.431× |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm89_f32_d128_out_m8n16_f64fold_v1` | 27.545 | 9.868 | 0.358× | 27.474 | 8.967 | 0.326× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 289.402 | 118.443 | 0.409× | 289.353 | 117.737 | 0.407× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 283.971 | 122.110 | 0.430× | 282.993 | 121.761 | 0.430× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `transpose_f32_32x16_d768_v1 + tn_sm89_f32_n64_dual_chunk_fused_finalize_v1` | 309.489 | 134.189 | 0.433× | 307.977 | 131.986 | 0.429× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 136.739 | 63.888 | 0.467× | 137.728 | 64.102 | 0.466× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 139.840 | 66.848 | 0.478× | 138.951 | 66.317 | 0.477× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm89_f32_m64n64_bk16_s2_d768_out_raw_v1 + splitm_reduce` | 189.023 | 82.898 | 0.439× | 188.148 | 82.332 | 0.438× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 195.812 | 109.568 | 0.557× | 198.371 | 110.547 | 0.555× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 299.008 | 76.912 | 0.257× | 298.044 | 76.567 | 0.257× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm89_f32_m64n64_bk16_s2_prism_raw_v1 + splitm_reduce` | 235.306 | 100.230 | 0.426× | 234.484 | 99.338 | 0.424× |

Geometric mean over the 15 cells: eager 0.422×, graph 0.422× (mamba-rs faster in 0 of 15 cells eager, 0 of 15 graph).

#### F32 in, F32 out, exact, against cuBLAS Pedantic (COMPUTE_32F_PEDANTIC)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_splitk32_partial + splitk_reduce` | 15.719 | 10.031 | 0.638× | 15.058 | 9.421 | 0.626× |
| d128 in_proj | NT | 1024 × 128 × 512 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 17.258 | 13.831 | 0.802× | 15.744 | 13.458 | 0.855× |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm89_f32_d128_in_m16n16_f64fold_v1` | 35.264 | 12.759 | 0.362× | 35.249 | 11.349 | 0.322× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_splitk32_partial + splitk_reduce` | 10.222 | 8.728 | 0.854× | 9.513 | 8.337 | 0.876× |
| d128 out_proj | NT | 1024 × 256 × 128 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 13.023 | 8.215 | 0.631× | 11.517 | 7.671 | 0.666× |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm89_f32_d128_out_m8n16_f64fold_v1` | 27.527 | 10.531 | 0.383× | 27.475 | 9.840 | 0.358× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 271.116 | 329.416 | 1.206× | 270.775 | 325.928 | 1.204× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 354.304 | 353.792 | 0.992× | 352.202 | 349.920 | 0.993× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `transpose_f32_32x16_d768_v1 + tn_sm89_f32_n64_dual_chunk_fused_finalize_v1` | 309.609 | 269.339 | 0.871× | 308.211 | 267.244 | 0.867× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 165.894 | 171.794 | 1.036× | 166.580 | 172.459 | 1.033× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 171.662 | 177.217 | 1.033× | 170.918 | 177.508 | 1.034× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm89_f32_m64n64_bk16_s2_d768_out_raw_v1 + splitm_reduce` | 188.947 | 134.740 | 0.713× | 188.151 | 133.743 | 0.711× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_fixed_sm89_f32_n64_copyplan_v1` | 255.621 | 282.368 | 1.104× | 253.915 | 281.899 | 1.103× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `transpose_f32_32x16_d768_v1 + nn_fixed_sm89_f32_n64_copyplan_v1` | 301.538 | 245.858 | 0.815× | 300.514 | 244.834 | 0.815× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm89_f32_m64n64_bk16_s2_prism_raw_v1 + splitm_reduce` | 235.398 | 189.975 | 0.807× | 234.223 | 187.834 | 0.802× |

Geometric mean over the 15 cells: eager 0.775×, graph 0.772× (mamba-rs faster in 4 of 15 cells eager, 4 of 15 graph).

### RTX 5090

#### BF16 in; BF16 out for NN and NT, F32 out for TN, against cuBLAS Fast (COMPUTE_32F)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 21.169 | 7.576 | 0.354× | 4.096 | 4.095 | 1.000× ‡ |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_sm120_tma_64x64_bk64_s3_bf16` | 21.613 | 6.578 | 0.303× | 6.148 | 4.100 | 0.667× ‡ |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm120_tma_64x64_bk32_s3_bf16` | 21.425 | 10.094 | 0.471× | 12.302 | 6.145 | 0.500× ‡ |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 21.287 | 6.606 | 0.310× | 4.098 | 4.098 | 1.000× ‡ |
| d128 out_proj | NT | 1024 × 256 × 128 | `nt_sm120_tma_64x64_bk32_s2_bf16` | 21.639 | 6.640 | 0.305× | 4.095 | 4.094 | 1.000× ‡ |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm120_tma_64x64_bk64_s3_bf16` | 21.586 | 10.143 | 0.467× | 10.252 | 6.143 | 0.599× ‡ |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm120_tma_128x64_bk32_s2_bf16` | 47.262 | 50.641 | 1.072× | 48.063 | 51.308 | 1.067× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm120_tma_64x64_bk64_s2_bf16` | 58.715 | 57.136 | 0.972× | 59.432 | 55.714 | 0.938× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm120_tma_64x128_bk32_s3_bf16` | 49.253 | 50.906 | 1.033× | 49.288 | 51.332 | 1.041× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 30.528 | 36.366 | 1.191× | 30.796 | 36.893 | 1.198× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm120_tma_64x64_bk64_s2_bf16` | 26.642 | 37.368 | 1.402× | 27.311 | 38.910 | 1.425× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm120_tma_64x128_bk32_s3_bf16` | 26.465 | 26.695 | 1.009× | 26.692 | 27.292 | 1.033× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm120_tma_128x64_bk64_s2_bf16` | 38.074 | 42.927 | 1.127× | 38.939 | 43.910 | 1.128× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm120_tma_128x128_bk64_s3_bf16` | 47.160 | 46.795 | 0.992× | 47.573 | 47.140 | 0.991× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm120_tma_64x128_bk32_s3_bf16` | 54.786 | 45.794 | 0.836× | 55.494 | 45.170 | 0.814× |

Geometric mean over the 15 cells: eager 0.689×, graph 0.929× (mamba-rs faster in 6 of 15 cells eager, 6 of 15 graph).

‡ Both sides sit on the 2.048 µs event-timer grid of this board in graph mode, so the ratio is a step count, not a resolved difference.

#### F16 in; F16 out for NN and NT, F32 out for TN, against cuBLAS Fast (COMPUTE_32F)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_sm120_tma_64x64_bk64_s2_f16` | 21.707 | 7.663 | 0.352× | 4.095 | 4.095 | 1.000× ‡ |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_sm120_tma_64x64_bk64_s3_f16` | 21.590 | 6.559 | 0.301× | 6.148 | 4.098 | 0.667× ‡ |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm120_tma_64x64_bk32_s3_f16` | 21.445 | 10.272 | 0.479× | 12.303 | 6.146 | 0.499× ‡ |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_sm120_tma_64x64_bk64_s2_f16` | 21.305 | 6.555 | 0.305× | 4.099 | 4.094 | 0.999× ‡ |
| d128 out_proj | NT | 1024 × 256 × 128 | `nt_sm120_tma_64x64_bk32_s2_f16` | 21.507 | 6.531 | 0.303× | 4.095 | 4.094 | 1.000× ‡ |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm120_tma_64x64_bk64_s3_f16` | 21.546 | 10.355 | 0.479× | 10.256 | 6.136 | 0.598× ‡ |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm120_tma_128x64_bk32_s2_f16` | 47.268 | 50.479 | 1.068× | 48.157 | 51.255 | 1.064× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm120_tma_64x64_bk64_s2_f16` | 58.693 | 57.124 | 0.973× | 59.430 | 55.696 | 0.937× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm120_tma_64x128_bk32_s3_f16` | 49.207 | 50.885 | 1.034× | 49.280 | 51.331 | 1.042× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm120_tma_64x64_bk64_s2_f16` | 30.627 | 36.253 | 1.184× | 30.915 | 36.894 | 1.194× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm120_tma_64x64_bk64_s2_f16` | 26.611 | 37.502 | 1.409× | 27.216 | 38.910 | 1.430× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm120_tma_64x128_bk32_s3_f16` | 26.284 | 26.700 | 1.015× | 26.687 | 26.868 | 1.007× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm120_tma_128x64_bk64_s2_f16` | 38.145 | 43.176 | 1.132× | 38.937 | 45.044 | 1.157× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm120_tma_128x128_bk64_s3_f16` | 47.122 | 46.568 | 0.988× | 47.445 | 47.140 | 0.994× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm120_tma_64x128_bk32_s3_f16` | 54.744 | 45.770 | 0.836× | 55.482 | 45.160 | 0.814× |

Geometric mean over the 15 cells: eager 0.690×, graph 0.929× (mamba-rs faster in 6 of 15 cells eager, 6 of 15 graph).

‡ Both sides sit on the 2.048 µs event-timer grid of this board in graph mode, so the ratio is a step count, not a resolved difference.

#### F32 in, F32 out, deterministic TF32, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| underfill | NN | 256 × 512 × 384 | `nn_splitk32_partial + splitk_reduce` | 6.858 | 7.299 | 1.068× | 8.194 | 8.198 | 1.000× ‡ |
| underfill | NT | 256 × 512 × 384 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 9.908 | 6.605 | 0.672× | 10.010 | 7.336 | 0.734× |
| underfill | TN | 256 × 512 × 384 | `tn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 6.314 | 6.498 | 1.028× | 8.195 | 6.149 | 0.750× ‡ |
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_sm80_mma_tf32_v1_m64n64_bk32_s3` | 5.078 | 6.637 | 1.317× | 6.148 | 4.100 | 0.667× ‡ |
| d128 in_proj | NT | 1024 × 128 × 512 | `nt_sm80_mma_tf32_v1_m16n16_bk32_s4` | 7.535 | 7.471 | 1.021× | 8.199 | 8.199 | 1.000× ‡ |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_sm80_mma_tf32_v1_m16n32_bk32_s4` | 9.692 | 10.149 | 1.047× | 10.255 | 8.194 | 0.799× ‡ |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_splitk32_partial + splitk_reduce` | 8.893 | 9.075 | 1.016× | 6.148 | 6.144 | 0.999× ‡ |
| d128 out_proj | NT | 1024 × 256 × 128 | `transpose_f32_32x16_d768_v1 + nn_m64n64_bk16_s2_v1` | 6.955 | 6.595 | 1.009× | 8.192 | 4.098 | 0.500× ‡ |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_sm80_mma_tf32_v1_m16n16_bk32_s4` | 9.381 | 10.177 | 1.085× | 10.250 | 8.087 | 0.789× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2` | 100.509 | 112.437 | 1.119× | 102.046 | 112.811 | 1.106× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2` | 122.704 | 139.153 | 1.134× | 123.468 | 139.398 | 1.129× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk` | 96.627 | 96.737 | 1.001× | 97.171 | 96.514 | 0.993× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2` | 65.453 | 71.243 | 1.089× | 66.559 | 71.734 | 1.078× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2` | 54.574 | 74.739 | 1.370× | 55.363 | 75.801 | 1.369× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk` | 53.750 | 50.351 | 0.938× | 55.389 | 51.304 | 0.926× |
| large deep | NN | 4096 × 3072 × 1536 | `nn_sm120_tma_mma_tf32_v1_m64n128_bk32_s2` | 375.934 | 416.947 | 1.109× | 376.919 | 416.953 | 1.107× |
| large deep | NT | 4096 × 3072 × 1536 | `nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2` | 372.254 | 359.440 | 0.966× | 372.967 | 359.859 | 0.965× |
| large deep | TN | 4096 × 3072 × 1536 | `tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk` | 361.627 | 373.829 | 1.034× | 360.965 | 368.569 | 1.022× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2` | 75.470 | 80.530 | 1.067× | 76.012 | 81.946 | 1.078× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm120_tma_mma_tf32_v1_m64n64_bk32_s2` | 79.571 | 89.381 | 1.123× | 80.173 | 90.174 | 1.125× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3_pair_streamk` | 74.608 | 78.367 | 1.051× | 76.080 | 78.021 | 1.026× |

Geometric mean over the 21 cells: eager 1.051×, graph 0.939× (mamba-rs faster in 18 of 21 cells eager, 10 of 21 graph).

‡ Both sides sit on the 2.048 µs event-timer grid of this board in graph mode, so the ratio is a step count, not a resolved difference.

#### F32 in, F32 out, exact, against cuBLAS Fast TF32 (COMPUTE_32F_FAST_TF32)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_splitk32_partial + splitk_reduce` | 8.650 | 7.304 | 0.845× | 10.242 | 4.101 | 0.400× ‡ |
| d128 in_proj | NT | 1024 × 128 × 512 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 9.987 | 7.445 | 0.749× | 10.240 | 8.199 | 0.801× ‡ |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_splitm_partial_aligned + splitm_reduce` | 26.467 | 10.096 | 0.382× | 27.041 | 8.192 | 0.303× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_splitk32_partial + splitk_reduce` | 8.516 | 8.279 | 0.974× | 6.148 | 6.144 | 0.999× ‡ |
| d128 out_proj | NT | 1024 × 256 × 128 | `transpose_f32_32x16_d768_v1 + nn_m64n64_bk16_s2_v1` | 6.842 | 7.548 | 0.998× | 8.192 | 4.098 | 0.500× ‡ |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_splitm_partial_aligned + splitm_reduce` | 15.132 | 10.150 | 0.671× | 16.390 | 8.056 | 0.492× |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 132.546 | 117.514 | 0.888× | 131.971 | 113.452 | 0.861× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec` | 146.018 | 139.406 | 0.955× | 146.880 | 139.691 | 0.952× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm120_tma_fma_v1_m64n128_bk16_s2` | 135.619 | 96.774 | 0.714× | 135.971 | 96.496 | 0.710× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 75.355 | 71.335 | 0.946× | 76.089 | 71.734 | 0.943× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm120_tma_fma_v1_m64n128_bk16_s2` | 78.653 | 74.877 | 0.952× | 79.708 | 75.830 | 0.951× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm120_tma_fma_v1_m128n64_bk16_s2` | 73.896 | 50.232 | 0.680× | 74.265 | 51.298 | 0.691× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm120_tma_fma_v1_m64n128_bk16_s2` | 105.687 | 80.541 | 0.762× | 106.574 | 81.985 | 0.770× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec` | 102.425 | 89.777 | 0.877× | 102.759 | 90.189 | 0.878× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm120_tma_fma_v1_m128n64_bk16_s2` | 98.350 | 78.345 | 0.797× | 99.010 | 78.040 | 0.788× |

Geometric mean over the 15 cells: eager 0.793×, graph 0.698× (mamba-rs faster in 0 of 15 cells eager, 0 of 15 graph).

‡ Both sides sit on the 2.048 µs event-timer grid of this board in graph mode, so the ratio is a step count, not a resolved difference.

#### F32 in, F32 out, exact, against cuBLAS Pedantic (COMPUTE_32F_PEDANTIC)

| shape | op | M × K × N | mamba-rs kernel chain | mamba-rs eager µs | cuBLAS eager µs | eager speedup | mamba-rs graph µs | cuBLAS graph µs | graph speedup |
|---|:--:|---|---|---:|---:|---:|---:|---:|---:|
| d128 in_proj | NN | 1024 × 128 × 512 | `nn_splitk32_partial + splitk_reduce` | 8.631 | 6.507 | 0.759× | 10.240 | 6.149 | 0.601× ‡ |
| d128 in_proj | NT | 1024 × 128 × 512 | `transpose_f32_2d + nn_splitk32_partial + splitk_reduce` | 9.830 | 9.926 | 1.010× | 10.239 | 10.239 | 1.000× ‡ |
| d128 in_proj | TN | 1024 × 128 × 512 | `tn_splitm_partial_aligned + splitm_reduce` | 26.440 | 10.033 | 0.380× | 27.466 | 8.193 | 0.298× |
| d128 out_proj | NN | 1024 × 256 × 128 | `nn_splitk32_partial + splitk_reduce` | 6.891 | 9.930 | 1.449× | 6.148 | 6.149 | 1.000× ‡ |
| d128 out_proj | NT | 1024 × 256 × 128 | `transpose_f32_32x16_d768_v1 + nn_m64n64_bk16_s2_v1` | 6.671 | 6.513 | 0.983× | 8.192 | 8.192 | 1.000× ‡ |
| d128 out_proj | TN | 1024 × 256 × 128 | `tn_splitm_partial_aligned + splitm_reduce` | 15.100 | 10.140 | 0.672× | 16.390 | 6.147 | 0.375× ‡ |
| d768 in_proj | NN | 2048 × 768 × 3072 | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 144.968 | 188.558 | 1.305× | 146.883 | 189.295 | 1.288× |
| d768 in_proj | NT | 2048 × 768 × 3072 | `nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec` | 154.637 | 193.583 | 1.248× | 151.848 | 186.873 | 1.236× |
| d768 in_proj | TN | 2048 × 768 × 3072 | `tn_sm120_tma_fma_v1_m64n128_bk16_s2` | 135.677 | 137.771 | 1.016× | 136.059 | 136.082 | 1.000× |
| d768 out_proj | NN | 2048 × 1536 × 768 | `nn_sm120_tma_fma_v1_m128n64_bk16_s2` | 77.596 | 90.325 | 1.158× | 79.350 | 90.019 | 1.143× |
| d768 out_proj | NT | 2048 × 1536 × 768 | `nt_sm120_tma_fma_v1_m64n128_bk16_s2` | 80.102 | 106.466 | 1.339× | 81.076 | 105.697 | 1.314× |
| d768 out_proj | TN | 2048 × 1536 × 768 | `tn_sm120_tma_fma_v1_m128n64_bk16_s2` | 73.941 | 78.753 | 1.065× | 74.288 | 77.502 | 1.044× |
| classifier page in_proj | NN | 4621 × 384 × 1928 | `nn_sm120_tma_fma_v1_m64n128_bk16_s2` | 109.613 | 131.581 | 1.221× | 112.089 | 135.282 | 1.207× |
| classifier page in_proj | NT | 4621 × 384 × 1928 | `nt_sm120_tma_fma_v1_m128n64_bk16_s2_kvec` | 110.765 | 147.232 | 1.328× | 106.567 | 137.413 | 1.306× |
| classifier page in_proj | TN | 4621 × 384 × 1928 | `tn_sm120_tma_fma_v1_m128n64_bk16_s2` | 98.274 | 106.253 | 1.081× | 98.996 | 104.865 | 1.059× |

Geometric mean over the 15 cells: eager 1.019×, graph 0.919× (mamba-rs faster in 11 of 15 cells eager, 10 of 15 graph).

‡ Both sides sit on the 2.048 µs event-timer grid of this board in graph mode, so the ratio is a step count, not a resolved difference.

## Reproducing

The cuBLAS comparison harness needs the `qualification` feature and an
idle GPU:

```sh
cargo test --release --features "cuda hf qualification" \
    --test gemm_bi_performance_matrix -- --ignored --nocapture
```

The training-step comparison of the three modes runs as a bench:

```sh
cargo bench --features cuda --bench gemm_bi_trainer_step_bench
```

The old-versus-new runs need the 0.6.9 tree checked out beside the release
tree; the adapters and the runner scripts are in
`internal/perf/old-versus-new-20260910/adapter/`.
