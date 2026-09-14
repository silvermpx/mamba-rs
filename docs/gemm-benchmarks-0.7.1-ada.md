# GEMM benchmarks — 0.7.1, RTX 6000 Ada

Measured on September 14, 2026, with the assembled deterministic Inference
and Triad dispatchers. The baseline is the released 0.7.0 source, not a
development checkpoint. These are GEMM timings; whole training steps are
reported separately in [Mamba-1](mamba1-benchmarks.md) and
[Mamba-3](mamba3-benchmarks.md).

## Measurement scope

| item | value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, SM89, 142 SMs |
| measured GPU UUID | `GPU-d1edd7be-e88d-aed6-047d-622163306f0e` |
| driver / toolkit | 595.45.04 / CUDA 13.2.51, NVRTC 13.2 |
| released source | `e2917a47494b4a1d652f5c818c3974ec3fcdd1ca` (`v0.7.0`) |
| assembled source | `83079104fe1efa7ad5dca0a28c5b48bcd86c5b14` |
| host compiler | Rust 1.98.1, release profile |
| GEMM mode | `Deterministic`; no forced candidate selection |
| cuBLAS workspace | 32 MiB |
| capacities | 64 for the direct version comparison; 16 in a separate companion pass |

The 0.7.0 harness uses a capacity-64 context. Each version's capacity-64
suite ran twice in the order old, new, new, old. Capacity 16 ran once on
0.7.1. A cell uses 21 calibrated CUDA-event windows in each of two mirrored
within-process arm orders. The table takes the median of the windows,
then the median of the two orders, then the median of the process results.

Row summaries are geometric means of per-cell ratios. Capacities, eager
execution and graph replay are never pooled.

The GPU was reserved for timing, with no resident model process. Quiet
checks bracket the measured pairs in the current harness and the Triad
baseline; the native 0.7.0 Inference logger lacks those per-pair fields.
Every process has a separate idle preflight. Old and new sources use
separate kernel caches; compilation and setup are outside the event windows.
Source and executable hashes are checked before and after the campaign.

**Ratios above 1 mean faster.** `old/new` is 0.7.0 deterministic time
divided by 0.7.1 deterministic time. `cuBLAS/new` compares the current
deterministic path with the cuBLAS arm measured beside it.

BF16/F16 Fast uses `CUBLAS_COMPUTE_32F` with native half inputs and F32
accumulation. TF32 uses `CUBLAS_COMPUTE_32F_FAST_TF32`. Exact F32 is shown
twice: against `CUBLAS_COMPUTE_32F_PEDANTIC` and against Fast TF32. The
latter compares different arithmetic precision, not equivalent modes.
Inference bias-on timings include cuBLAS's separate bias launch; the
deterministic arm fuses it. Triad has no bias.

No new RTX 5090 timing is claimed here. Its earlier measurements remain
in the [0.7.0 tables](determinism-benchmarks.md). The newly qualified Ada
routes do not imply the same gains on another architecture.

## Summary at capacity 64

| family | precision / comparator | path | cells | old/new | cuBLAS/new | vendor control old/new |
|---|---|---|---:|---:|---:|---:|
| inference | BF16 → BF16 / Fast | eager | 10 | 1.001× | 1.187× | 1.001× |
| inference | BF16 → BF16 / Fast | graph | 10 | 1.002× | 1.185× | 1.002× |
| inference | BF16 → F32 / Fast | eager | 10 | 1.216× | 1.001× | 1.005× |
| inference | BF16 → F32 / Fast | graph | 10 | 1.216× | 0.999× | 1.005× |
| inference | F16 → F16 / Fast | eager | 10 | 1.033× | 1.171× | 1.015× |
| inference | F16 → F16 / Fast | graph | 10 | 1.041× | 1.170× | 1.020× |
| inference | F16 → F32 / Fast | eager | 10 | 1.222× | 0.976× | 1.018× |
| inference | F16 → F32 / Fast | graph | 10 | 1.216× | 0.972× | 1.015× |
| inference | exact F32 / Pedantic | eager | 10 | 1.096× | 1.028× | 1.070× |
| inference | exact F32 / Pedantic | graph | 10 | 1.117× | 1.026× | 1.092× |
| inference | exact F32 / Fast TF32 | eager | 10 | 1.102× | 0.473× | 1.072× |
| inference | exact F32 / Fast TF32 | graph | 10 | 1.119× | 0.472× | 1.086× |
| inference | TF32 / Fast TF32 | eager | 10 | 1.032× | 0.901× | 1.029× |
| inference | TF32 / Fast TF32 | graph | 10 | 1.035× | 0.901× | 1.031× |
| triad | BF16 / Fast | eager | 15 | 1.027× | 0.909× | 1.003× |
| triad | BF16 / Fast | graph | 15 | 1.027× | 0.891× | 1.003× |
| triad | F16 / Fast | eager | 15 | 1.027× | 0.896× | 1.000× |
| triad | F16 / Fast | graph | 15 | 1.027× | 0.874× | 0.999× |
| triad | exact F32 / Fast TF32 | eager | 15 | 1.000× | 0.423× | 1.000× |
| triad | exact F32 / Fast TF32 | graph | 15 | 1.000× | 0.422× | 1.001× |
| triad | exact F32 / Pedantic | eager | 15 | 1.000× | 0.776× | 0.999× |
| triad | exact F32 / Pedantic | graph | 15 | 1.000× | 0.773× | 1.000× |
| triad | TF32 / Fast TF32 | eager | 21 | 1.005× | 0.803× | 1.001× |
| triad | TF32 / Fast TF32 | graph | 21 | 1.004× | 0.795× | 1.001× |

The vendor-control column shows run-to-run change in the cuBLAS arm.
Small changes near 1 should not be treated as established kernel gains.

## Shapes

Dimensions are M × K × N of the forward NN product. Triad also measures
the corresponding input gradient (NT) and weight gradient (TN).

| family | case | M × K × N |
|---|---|---|
| Inference | hot_a | 4621 × 384 × 1928 |
| Inference | hot_b | 4621 × 768 × 2304 |
| Inference | hot_c | 4621 × 1928 × 384 |
| Inference | hot_d | 2048 × 768 × 2304 |
| Inference | hot_e | 2048 × 2304 × 768 |
| Triad | d128_in_proj | 1024 × 128 × 512 |
| Triad | d128_out_proj | 1024 × 256 × 128 |
| Triad | d768_in_proj | 2048 × 768 × 3072 |
| Triad | d768_out_proj | 2048 × 1536 × 768 |
| Triad | prism_in_proj | 4621 × 384 × 1928 |
| Triad | underfill | 256 × 512 × 384 |
| Triad | large_deep | 4096 × 3072 × 1536 |

Inference has 70 shape/bias/comparator views. Triad has 81 comparator
views over 66 distinct deterministic cases: 15 each for BF16, F16 and
exact F32, and 21 for TF32. Exact F32's second comparator does not add
another deterministic case. Both eager and graph are measured for every
view. The exact-F32 underfill/deep supplement is separate from these means.

## Full matrix, capacity 64

### inference — BF16 → BF16 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 64.16 | 64.17 | 1.000× | 74.90 | 1.167× |
| hot_a / bias off | graph | 63.76 | 63.76 | 1.000× | 74.57 | 1.169× |
| hot_a / bias on | eager | 65.41 | 65.42 | 1.000× | 108.70 | 1.662× |
| hot_a / bias on | graph | 65.00 | 65.01 | 1.000× | 107.97 | 1.661× |
| hot_b / bias off | eager | 128.68 | 127.89 | 1.006× | 105.17 | 0.822× |
| hot_b / bias off | graph | 129.03 | 128.30 | 1.006× | 105.51 | 0.822× |
| hot_b / bias on | eager | 125.61 | 124.46 | 1.009× | 125.63 | 1.009× |
| hot_b / bias on | graph | 126.39 | 124.65 | 1.014× | 125.09 | 1.004× |
| hot_c / bias off | eager | 54.05 | 54.06 | 1.000× | 74.75 | 1.383× |
| hot_c / bias off | graph | 53.88 | 53.88 | 1.000× | 75.02 | 1.392× |
| hot_c / bias on | eager | 54.36 | 54.37 | 1.000× | 84.22 | 1.549× |
| hot_c / bias on | graph | 54.32 | 54.34 | 1.000× | 85.57 | 1.575× |
| hot_d / bias off | eager | 65.62 | 65.64 | 1.000× | 60.42 | 0.920× |
| hot_d / bias off | graph | 65.06 | 65.05 | 1.000× | 59.15 | 0.909× |
| hot_d / bias on | eager | 66.41 | 66.42 | 1.000× | 80.19 | 1.207× |
| hot_d / bias on | graph | 65.67 | 65.67 | 1.000× | 78.71 | 1.198× |
| hot_e / bias off | eager | 60.61 | 60.62 | 1.000× | 68.80 | 1.135× |
| hot_e / bias off | graph | 60.58 | 60.59 | 1.000× | 68.72 | 1.134× |
| hot_e / bias on | eager | 60.87 | 60.88 | 1.000× | 77.48 | 1.273× |
| hot_e / bias on | graph | 60.69 | 60.70 | 1.000× | 76.40 | 1.259× |

### inference — BF16 → F32 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 96.75 | 89.35 | 1.083× | 76.93 | 0.861× |
| hot_a / bias off | graph | 96.07 | 88.99 | 1.080× | 76.90 | 0.864× |
| hot_a / bias on | eager | 90.29 | 86.02 | 1.050× | 110.94 | 1.290× |
| hot_a / bias on | graph | 88.42 | 85.70 | 1.032× | 110.25 | 1.286× |
| hot_b / bias off | eager | 229.88 | 146.83 | 1.566× | 123.26 | 0.839× |
| hot_b / bias off | graph | 228.84 | 146.19 | 1.565× | 123.41 | 0.844× |
| hot_b / bias on | eager | 199.87 | 126.71 | 1.577× | 139.58 | 1.102× |
| hot_b / bias on | graph | 200.56 | 126.10 | 1.591× | 139.14 | 1.103× |
| hot_c / bias off | eager | 77.46 | 49.60 | 1.562× | 76.25 | 1.537× |
| hot_c / bias off | graph | 77.32 | 49.46 | 1.563× | 75.71 | 1.531× |
| hot_c / bias on | eager | 77.66 | 50.46 | 1.539× | 86.20 | 1.708× |
| hot_c / bias on | graph | 77.51 | 50.14 | 1.546× | 86.78 | 1.731× |
| hot_d / bias off | eager | 102.91 | 98.88 | 1.041× | 61.68 | 0.624× |
| hot_d / bias off | graph | 104.39 | 99.26 | 1.052× | 60.71 | 0.612× |
| hot_d / bias on | eager | 98.91 | 98.48 | 1.004× | 80.61 | 0.819× |
| hot_d / bias on | graph | 98.81 | 98.25 | 1.006× | 79.39 | 0.808× |
| hot_e / bias off | eager | 85.16 | 85.13 | 1.000× | 68.76 | 0.808× |
| hot_e / bias off | graph | 84.94 | 84.92 | 1.000× | 68.71 | 0.809× |
| hot_e / bias on | eager | 85.30 | 85.26 | 1.000× | 77.73 | 0.912× |
| hot_e / bias on | graph | 85.13 | 85.10 | 1.000× | 76.90 | 0.904× |

### inference — F16 → F16 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 68.66 | 66.62 | 1.031× | 75.13 | 1.128× |
| hot_a / bias off | graph | 69.28 | 66.59 | 1.040× | 74.81 | 1.124× |
| hot_a / bias on | eager | 65.40 | 65.41 | 1.000× | 108.40 | 1.657× |
| hot_a / bias on | graph | 65.00 | 65.00 | 1.000× | 107.63 | 1.656× |
| hot_b / bias off | eager | 141.60 | 136.40 | 1.038× | 118.03 | 0.865× |
| hot_b / bias off | graph | 142.24 | 137.39 | 1.035× | 118.25 | 0.861× |
| hot_b / bias on | eager | 137.79 | 133.97 | 1.029× | 140.12 | 1.046× |
| hot_b / bias on | graph | 139.09 | 133.21 | 1.044× | 139.52 | 1.047× |
| hot_c / bias off | eager | 58.46 | 55.64 | 1.051× | 75.34 | 1.354× |
| hot_c / bias off | graph | 59.10 | 55.42 | 1.066× | 75.76 | 1.367× |
| hot_c / bias on | eager | 57.41 | 54.37 | 1.056× | 84.18 | 1.548× |
| hot_c / bias on | graph | 56.32 | 54.29 | 1.037× | 85.43 | 1.574× |
| hot_d / bias off | eager | 73.70 | 69.51 | 1.060× | 68.14 | 0.980× |
| hot_d / bias off | graph | 77.59 | 71.35 | 1.087× | 68.82 | 0.964× |
| hot_d / bias on | eager | 66.40 | 66.42 | 1.000× | 80.16 | 1.207× |
| hot_d / bias on | graph | 65.65 | 65.66 | 1.000× | 78.62 | 1.197× |
| hot_e / bias off | eager | 64.10 | 59.78 | 1.072× | 61.94 | 1.036× |
| hot_e / bias off | graph | 65.01 | 59.17 | 1.099× | 61.47 | 1.039× |
| hot_e / bias on | eager | 61.99 | 61.88 | 1.002× | 69.25 | 1.119× |
| hot_e / bias on | graph | 62.30 | 61.74 | 1.009× | 68.54 | 1.110× |

### inference — F16 → F32 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 105.86 | 95.32 | 1.111× | 80.22 | 0.842× |
| hot_a / bias off | graph | 103.88 | 94.79 | 1.096× | 80.07 | 0.845× |
| hot_a / bias on | eager | 95.13 | 88.44 | 1.076× | 112.85 | 1.276× |
| hot_a / bias on | graph | 94.53 | 88.21 | 1.072× | 112.25 | 1.273× |
| hot_b / bias off | eager | 249.84 | 154.84 | 1.614× | 135.20 | 0.873× |
| hot_b / bias off | graph | 246.35 | 156.85 | 1.571× | 136.41 | 0.870× |
| hot_b / bias on | eager | 214.04 | 135.20 | 1.583× | 154.22 | 1.141× |
| hot_b / bias on | graph | 210.34 | 134.73 | 1.561× | 153.52 | 1.139× |
| hot_c / bias off | eager | 77.41 | 53.67 | 1.442× | 77.56 | 1.445× |
| hot_c / bias off | graph | 77.23 | 53.97 | 1.431× | 77.51 | 1.436× |
| hot_c / bias on | eager | 77.59 | 52.06 | 1.491× | 86.63 | 1.664× |
| hot_c / bias on | graph | 77.43 | 51.61 | 1.500× | 87.25 | 1.691× |
| hot_d / bias off | eager | 111.92 | 104.31 | 1.073× | 64.82 | 0.621× |
| hot_d / bias off | graph | 112.99 | 105.02 | 1.076× | 64.38 | 0.613× |
| hot_d / bias on | eager | 104.28 | 99.06 | 1.053× | 80.83 | 0.816× |
| hot_d / bias on | graph | 105.57 | 99.68 | 1.059× | 79.89 | 0.801× |
| hot_e / bias off | eager | 85.12 | 85.13 | 1.000× | 62.62 | 0.736× |
| hot_e / bias off | graph | 84.90 | 84.90 | 1.000× | 62.15 | 0.732× |
| hot_e / bias on | eager | 85.26 | 85.26 | 1.000× | 69.65 | 0.817× |
| hot_e / bias on | graph | 85.05 | 85.06 | 1.000× | 69.02 | 0.811× |

### inference — exact F32 / Pedantic

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 291.94 | 270.54 | 1.079× | 299.50 | 1.107× |
| hot_a / bias off | graph | 290.08 | 270.33 | 1.073× | 298.94 | 1.106× |
| hot_a / bias on | eager | 271.25 | 251.50 | 1.079× | 303.92 | 1.208× |
| hot_a / bias on | graph | 271.09 | 252.72 | 1.073× | 305.24 | 1.208× |
| hot_b / bias off | eager | 758.78 | 698.09 | 1.087× | 711.09 | 1.019× |
| hot_b / bias off | graph | 758.64 | 704.47 | 1.077× | 714.34 | 1.014× |
| hot_b / bias on | eager | 684.35 | 642.00 | 1.066× | 687.91 | 1.072× |
| hot_b / bias on | graph | 691.30 | 643.69 | 1.074× | 686.17 | 1.066× |
| hot_c / bias off | eager | 365.34 | 283.82 | 1.287× | 233.50 | 0.823× |
| hot_c / bias off | graph | 381.94 | 284.75 | 1.341× | 234.86 | 0.825× |
| hot_c / bias on | eager | 365.63 | 340.29 | 1.074× | 232.39 | 0.683× |
| hot_c / bias on | graph | 385.79 | 340.02 | 1.135× | 232.25 | 0.683× |
| hot_d / bias off | eager | 236.46 | 221.01 | 1.070× | 263.89 | 1.194× |
| hot_d / bias off | graph | 238.32 | 219.82 | 1.084× | 262.65 | 1.195× |
| hot_d / bias on | eager | 235.61 | 218.38 | 1.079× | 277.26 | 1.270× |
| hot_d / bias on | graph | 237.98 | 219.08 | 1.086× | 277.60 | 1.267× |
| hot_e / bias off | eager | 287.13 | 266.47 | 1.078× | 270.17 | 1.014× |
| hot_e / bias off | graph | 300.75 | 268.49 | 1.120× | 271.12 | 1.010× |
| hot_e / bias on | eager | 287.54 | 266.83 | 1.078× | 277.66 | 1.041× |
| hot_e / bias on | graph | 299.80 | 265.96 | 1.127× | 275.74 | 1.037× |

### inference — exact F32 / Fast TF32

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 232.23 | 211.82 | 1.096× | 117.01 | 0.552× |
| hot_a / bias off | graph | 231.03 | 212.51 | 1.087× | 117.01 | 0.551× |
| hot_a / bias on | eager | 231.03 | 214.65 | 1.076× | 145.42 | 0.677× |
| hot_a / bias on | graph | 230.82 | 213.73 | 1.080× | 144.64 | 0.677× |
| hot_b / bias off | eager | 570.62 | 529.62 | 1.077× | 239.90 | 0.453× |
| hot_b / bias off | graph | 572.41 | 523.90 | 1.093× | 236.11 | 0.451× |
| hot_b / bias on | eager | 566.91 | 529.45 | 1.071× | 272.87 | 0.515× |
| hot_b / bias on | graph | 562.05 | 530.96 | 1.059× | 273.27 | 0.515× |
| hot_c / bias off | eager | 312.58 | 241.47 | 1.294× | 82.19 | 0.340× |
| hot_c / bias off | graph | 325.71 | 241.91 | 1.346× | 82.01 | 0.339× |
| hot_c / bias on | eager | 307.93 | 283.23 | 1.087× | 86.53 | 0.306× |
| hot_c / bias on | graph | 318.78 | 284.61 | 1.120× | 86.15 | 0.303× |
| hot_d / bias off | eager | 237.71 | 219.34 | 1.084× | 107.92 | 0.492× |
| hot_d / bias off | graph | 237.88 | 219.28 | 1.085× | 107.84 | 0.492× |
| hot_d / bias on | eager | 234.60 | 218.68 | 1.073× | 124.52 | 0.569× |
| hot_d / bias on | graph | 237.08 | 215.71 | 1.099× | 122.22 | 0.567× |
| hot_e / bias off | eager | 240.55 | 222.65 | 1.080× | 101.31 | 0.455× |
| hot_e / bias off | graph | 250.39 | 222.73 | 1.124× | 101.04 | 0.454× |
| hot_e / bias on | eager | 240.66 | 218.82 | 1.100× | 106.95 | 0.489× |
| hot_e / bias on | graph | 245.58 | 219.39 | 1.119× | 106.47 | 0.485× |

### inference — TF32 / Fast TF32

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| hot_a / bias off | eager | 111.85 | 111.29 | 1.005× | 103.01 | 0.926× |
| hot_a / bias off | graph | 110.42 | 110.33 | 1.001× | 102.58 | 0.930× |
| hot_a / bias on | eager | 113.89 | 113.77 | 1.001× | 131.19 | 1.153× |
| hot_a / bias on | graph | 113.20 | 113.09 | 1.001× | 130.09 | 1.150× |
| hot_b / bias off | eager | 237.87 | 222.55 | 1.069× | 203.40 | 0.914× |
| hot_b / bias off | graph | 237.87 | 221.43 | 1.074× | 202.96 | 0.917× |
| hot_b / bias on | eager | 238.65 | 222.30 | 1.074× | 234.19 | 1.053× |
| hot_b / bias on | graph | 238.65 | 222.89 | 1.071× | 234.76 | 1.053× |
| hot_c / bias off | eager | 108.46 | 102.12 | 1.062× | 77.89 | 0.763× |
| hot_c / bias off | graph | 109.27 | 102.36 | 1.067× | 78.02 | 0.762× |
| hot_c / bias on | eager | 106.72 | 102.62 | 1.040× | 86.13 | 0.839× |
| hot_c / bias on | graph | 108.97 | 102.32 | 1.065× | 85.18 | 0.833× |
| hot_d / bias off | eager | 125.34 | 125.34 | 1.000× | 99.28 | 0.792× |
| hot_d / bias off | graph | 123.13 | 123.14 | 1.000× | 98.65 | 0.801× |
| hot_d / bias on | eager | 126.64 | 126.67 | 1.000× | 116.15 | 0.917× |
| hot_d / bias on | graph | 125.16 | 125.21 | 1.000× | 114.73 | 0.916× |
| hot_e / bias off | eager | 108.73 | 101.49 | 1.071× | 91.13 | 0.898× |
| hot_e / bias off | graph | 109.08 | 101.15 | 1.078× | 90.75 | 0.897× |
| hot_e / bias on | eager | 119.52 | 119.52 | 1.000× | 98.64 | 0.825× |
| hot_e / bias on | graph | 119.16 | 119.17 | 1.000× | 97.71 | 0.820× |

### triad — BF16 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| nn / d128_in_proj | eager | 5.63 | 5.63 | 1.000× | 4.80 | 0.852× |
| nn / d128_in_proj | graph | 5.31 | 5.31 | 1.000× | 4.82 | 0.908× |
| nn / d128_out_proj | eager | 6.68 | 6.68 | 1.000× | 5.44 | 0.814× |
| nn / d128_out_proj | graph | 6.42 | 6.42 | 1.000× | 4.96 | 0.772× |
| nn / d768_in_proj | eager | 62.25 | 62.23 | 1.000× | 76.31 | 1.226× |
| nn / d768_in_proj | graph | 61.78 | 61.78 | 1.000× | 75.65 | 1.224× |
| nn / d768_out_proj | eager | 37.95 | 37.95 | 1.000× | 47.42 | 1.249× |
| nn / d768_out_proj | graph | 37.76 | 37.75 | 1.000× | 47.16 | 1.250× |
| nn / prism_in_proj | eager | 59.67 | 59.68 | 1.000× | 73.53 | 1.232× |
| nn / prism_in_proj | graph | 59.12 | 59.13 | 1.000× | 73.14 | 1.237× |
| nt / d128_in_proj | eager | 9.37 | 9.37 | 1.000× | 7.64 | 0.816× |
| nt / d128_in_proj | graph | 9.16 | 9.16 | 1.000× | 7.15 | 0.781× |
| nt / d128_out_proj | eager | 5.39 | 5.39 | 1.000× | 4.71 | 0.872× |
| nt / d128_out_proj | graph | 5.18 | 5.18 | 1.000× | 4.28 | 0.827× |
| nt / d768_in_proj | eager | 71.07 | 70.98 | 1.001× | 77.70 | 1.095× |
| nt / d768_in_proj | graph | 70.74 | 70.66 | 1.001× | 77.13 | 1.092× |
| nt / d768_out_proj | eager | 36.50 | 36.55 | 0.999× | 38.87 | 1.063× |
| nt / d768_out_proj | graph | 36.18 | 36.22 | 0.999× | 39.20 | 1.082× |
| nt / prism_in_proj | eager | 48.65 | 48.65 | 1.000× | 62.97 | 1.294× |
| nt / prism_in_proj | graph | 48.68 | 48.68 | 1.000× | 62.28 | 1.280× |
| tn / d128_in_proj | eager | 18.34 | 18.34 | 1.000× | 7.92 | 0.432× |
| tn / d128_in_proj | graph | 18.21 | 18.20 | 1.000× | 7.45 | 0.409× |
| tn / d128_out_proj | eager | 18.26 | 12.27 | 1.487× | 7.15 | 0.583× |
| tn / d128_out_proj | graph | 18.03 | 12.13 | 1.487× | 6.48 | 0.534× |
| tn / d768_in_proj | eager | 86.79 | 86.78 | 1.000× | 90.25 | 1.040× |
| tn / d768_in_proj | graph | 86.56 | 86.61 | 0.999× | 87.71 | 1.013× |
| tn / d768_out_proj | eager | 55.26 | 55.23 | 1.001× | 50.68 | 0.918× |
| tn / d768_out_proj | graph | 55.09 | 55.13 | 0.999× | 49.29 | 0.894× |
| tn / prism_in_proj | eager | 77.60 | 77.59 | 1.000× | 54.88 | 0.707× |
| tn / prism_in_proj | graph | 76.96 | 77.02 | 0.999× | 53.46 | 0.694× |

### triad — F16 / Fast

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| nn / d128_in_proj | eager | 5.63 | 5.63 | 1.000× | 5.35 | 0.951× |
| nn / d128_in_proj | graph | 5.31 | 5.31 | 1.000× | 5.06 | 0.953× |
| nn / d128_out_proj | eager | 6.68 | 6.68 | 1.000× | 4.72 | 0.706× |
| nn / d128_out_proj | graph | 6.42 | 6.42 | 1.000× | 4.25 | 0.663× |
| nn / d768_in_proj | eager | 62.24 | 62.23 | 1.000× | 59.24 | 0.952× |
| nn / d768_in_proj | graph | 61.77 | 61.79 | 1.000× | 58.81 | 0.952× |
| nn / d768_out_proj | eager | 37.96 | 37.99 | 0.999× | 47.09 | 1.240× |
| nn / d768_out_proj | graph | 37.76 | 37.76 | 1.000× | 46.79 | 1.239× |
| nn / prism_in_proj | eager | 59.77 | 59.79 | 1.000× | 73.35 | 1.227× |
| nn / prism_in_proj | graph | 59.21 | 59.21 | 1.000× | 72.83 | 1.230× |
| nt / d128_in_proj | eager | 9.37 | 9.37 | 1.000× | 5.82 | 0.622× |
| nt / d128_in_proj | graph | 9.16 | 9.16 | 1.000× | 5.27 | 0.576× |
| nt / d128_out_proj | eager | 5.39 | 5.39 | 1.000× | 4.50 | 0.834× |
| nt / d128_out_proj | graph | 5.18 | 5.18 | 1.000× | 4.02 | 0.777× |
| nt / d768_in_proj | eager | 70.93 | 71.01 | 0.999× | 72.85 | 1.026× |
| nt / d768_in_proj | graph | 70.70 | 70.74 | 0.999× | 75.99 | 1.074× |
| nt / d768_out_proj | eager | 36.51 | 36.55 | 0.999× | 39.84 | 1.090× |
| nt / d768_out_proj | graph | 36.18 | 36.22 | 0.999× | 39.89 | 1.101× |
| nt / prism_in_proj | eager | 48.66 | 48.66 | 1.000× | 96.56 | 1.984× |
| nt / prism_in_proj | graph | 48.68 | 48.68 | 1.000× | 95.79 | 1.968× |
| tn / d128_in_proj | eager | 18.34 | 18.34 | 1.000× | 7.87 | 0.429× |
| tn / d128_in_proj | graph | 18.11 | 18.11 | 1.000× | 7.47 | 0.413× |
| tn / d128_out_proj | eager | 18.26 | 12.27 | 1.488× | 7.44 | 0.607× |
| tn / d128_out_proj | graph | 18.03 | 12.00 | 1.502× | 6.58 | 0.548× |
| tn / d768_in_proj | eager | 86.65 | 86.68 | 1.000× | 89.74 | 1.035× |
| tn / d768_in_proj | graph | 86.64 | 86.66 | 1.000× | 86.79 | 1.002× |
| tn / d768_out_proj | eager | 56.20 | 56.14 | 1.001× | 50.67 | 0.903× |
| tn / d768_out_proj | graph | 56.03 | 56.09 | 0.999× | 49.35 | 0.880× |
| tn / prism_in_proj | eager | 77.58 | 77.67 | 0.999× | 54.91 | 0.707× |
| tn / prism_in_proj | graph | 76.95 | 77.06 | 0.999× | 53.47 | 0.694× |

### triad — exact F32 / Fast TF32

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| nn / d128_in_proj | eager | 15.72 | 15.71 | 1.000× | 5.67 | 0.361× |
| nn / d128_in_proj | graph | 15.04 | 15.04 | 1.000× | 5.53 | 0.368× |
| nn / d128_out_proj | eager | 10.19 | 10.19 | 1.000× | 6.82 | 0.669× |
| nn / d128_out_proj | graph | 9.50 | 9.50 | 1.000× | 6.58 | 0.693× |
| nn / d768_in_proj | eager | 291.60 | 293.92 | 0.992× | 119.98 | 0.408× |
| nn / d768_in_proj | graph | 294.97 | 294.71 | 1.001× | 119.38 | 0.405× |
| nn / d768_out_proj | eager | 137.80 | 138.03 | 0.998× | 64.35 | 0.466× |
| nn / d768_out_proj | graph | 138.36 | 138.36 | 1.000× | 64.34 | 0.465× |
| nn / prism_in_proj | eager | 200.05 | 200.17 | 0.999× | 111.03 | 0.555× |
| nn / prism_in_proj | graph | 201.38 | 200.36 | 1.005× | 111.11 | 0.555× |
| nt / d128_in_proj | eager | 17.16 | 17.16 | 1.000× | 9.42 | 0.549× |
| nt / d128_in_proj | graph | 15.69 | 15.69 | 1.000× | 9.06 | 0.577× |
| nt / d128_out_proj | eager | 12.83 | 12.83 | 1.000× | 5.14 | 0.401× |
| nt / d128_out_proj | graph | 11.35 | 11.35 | 1.000× | 4.97 | 0.438× |
| nt / d768_in_proj | eager | 285.87 | 285.13 | 1.003× | 122.77 | 0.431× |
| nt / d768_in_proj | graph | 284.99 | 285.44 | 0.998× | 122.86 | 0.430× |
| nt / d768_out_proj | eager | 140.09 | 139.96 | 1.001× | 66.97 | 0.478× |
| nt / d768_out_proj | graph | 139.19 | 139.59 | 0.997× | 66.62 | 0.477× |
| nt / prism_in_proj | eager | 299.01 | 298.99 | 1.000× | 76.97 | 0.257× |
| nt / prism_in_proj | graph | 297.98 | 298.00 | 1.000× | 76.48 | 0.257× |
| tn / d128_in_proj | eager | 35.24 | 35.24 | 1.000× | 10.09 | 0.286× |
| tn / d128_in_proj | graph | 35.27 | 35.26 | 1.000× | 9.34 | 0.265× |
| tn / d128_out_proj | eager | 27.50 | 27.50 | 1.000× | 9.78 | 0.356× |
| tn / d128_out_proj | graph | 27.54 | 27.54 | 1.000× | 8.95 | 0.325× |
| tn / d768_in_proj | eager | 309.35 | 309.37 | 1.000× | 133.81 | 0.433× |
| tn / d768_in_proj | graph | 308.00 | 307.97 | 1.000× | 131.98 | 0.429× |
| tn / d768_out_proj | eager | 188.83 | 188.78 | 1.000× | 82.60 | 0.438× |
| tn / d768_out_proj | graph | 188.04 | 188.03 | 1.000× | 82.29 | 0.438× |
| tn / prism_in_proj | eager | 235.04 | 235.05 | 1.000× | 100.29 | 0.427× |
| tn / prism_in_proj | graph | 234.06 | 234.05 | 1.000× | 99.28 | 0.424× |

### triad — exact F32 / Pedantic

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| nn / d128_in_proj | eager | 15.71 | 15.71 | 1.000× | 10.00 | 0.636× |
| nn / d128_in_proj | graph | 15.04 | 15.04 | 1.000× | 9.36 | 0.623× |
| nn / d128_out_proj | eager | 10.18 | 10.18 | 1.000× | 8.70 | 0.854× |
| nn / d128_out_proj | graph | 9.49 | 9.50 | 1.000× | 8.43 | 0.888× |
| nn / d768_in_proj | eager | 276.60 | 276.67 | 1.000× | 333.81 | 1.207× |
| nn / d768_in_proj | graph | 276.66 | 276.43 | 1.001× | 333.30 | 1.206× |
| nn / d768_out_proj | eager | 167.71 | 167.77 | 1.000× | 173.56 | 1.035× |
| nn / d768_out_proj | graph | 168.14 | 168.33 | 0.999× | 174.04 | 1.034× |
| nn / prism_in_proj | eager | 255.10 | 254.99 | 1.000× | 281.63 | 1.104× |
| nn / prism_in_proj | graph | 256.26 | 256.03 | 1.001× | 283.29 | 1.106× |
| nt / d128_in_proj | eager | 17.15 | 17.15 | 1.000× | 13.75 | 0.802× |
| nt / d128_in_proj | graph | 15.69 | 15.69 | 1.000× | 13.36 | 0.851× |
| nt / d128_out_proj | eager | 12.83 | 12.83 | 1.000× | 8.19 | 0.638× |
| nt / d128_out_proj | graph | 11.35 | 11.35 | 1.000× | 7.79 | 0.687× |
| nt / d768_in_proj | eager | 359.74 | 359.09 | 1.002× | 358.61 | 0.999× |
| nt / d768_in_proj | graph | 361.05 | 358.80 | 1.006× | 355.70 | 0.991× |
| nt / d768_out_proj | eager | 172.98 | 172.42 | 1.003× | 178.14 | 1.033× |
| nt / d768_out_proj | graph | 172.72 | 171.81 | 1.005× | 178.04 | 1.036× |
| nt / prism_in_proj | eager | 308.43 | 312.26 | 0.988× | 253.30 | 0.811× |
| nt / prism_in_proj | graph | 306.81 | 310.80 | 0.987× | 252.46 | 0.812× |
| tn / d128_in_proj | eager | 35.24 | 35.24 | 1.000× | 12.70 | 0.360× |
| tn / d128_in_proj | graph | 35.26 | 35.26 | 1.000× | 11.31 | 0.321× |
| tn / d128_out_proj | eager | 27.50 | 27.50 | 1.000× | 10.55 | 0.383× |
| tn / d128_out_proj | graph | 27.54 | 27.54 | 1.000× | 9.83 | 0.357× |
| tn / d768_in_proj | eager | 309.38 | 309.41 | 1.000× | 268.72 | 0.868× |
| tn / d768_in_proj | graph | 307.97 | 308.07 | 1.000× | 266.95 | 0.867× |
| tn / d768_out_proj | eager | 188.86 | 188.81 | 1.000× | 134.48 | 0.712× |
| tn / d768_out_proj | graph | 188.03 | 188.06 | 1.000× | 133.77 | 0.711× |
| tn / prism_in_proj | eager | 235.11 | 235.17 | 1.000× | 189.60 | 0.806× |
| tn / prism_in_proj | graph | 234.12 | 234.13 | 1.000× | 187.87 | 0.802× |

### triad — TF32 / Fast TF32

| case | path | 0.7.0 µs | 0.7.1 µs | old/new | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|---:|---:|
| nn / d128_in_proj | eager | 9.95 | 9.95 | 1.000× | 5.66 | 0.569× |
| nn / d128_in_proj | graph | 9.74 | 9.74 | 1.000× | 5.51 | 0.566× |
| nn / d128_out_proj | eager | 7.65 | 7.65 | 1.000× | 6.81 | 0.890× |
| nn / d128_out_proj | graph | 7.56 | 7.57 | 0.999× | 6.58 | 0.869× |
| nn / d768_in_proj | eager | 123.65 | 124.57 | 0.993× | 102.34 | 0.822× |
| nn / d768_in_proj | graph | 123.61 | 123.57 | 1.000× | 102.15 | 0.827× |
| nn / d768_out_proj | eager | 63.70 | 63.70 | 1.000× | 62.86 | 0.987× |
| nn / d768_out_proj | graph | 63.31 | 63.32 | 1.000× | 62.56 | 0.988× |
| nn / large_deep | eager | 512.79 | 507.43 | 1.011× | 403.26 | 0.795× |
| nn / large_deep | graph | 510.59 | 506.39 | 1.008× | 402.74 | 0.795× |
| nn / prism_in_proj | eager | 104.17 | 104.17 | 1.000× | 103.07 | 0.989× |
| nn / prism_in_proj | graph | 102.44 | 102.45 | 1.000× | 102.70 | 1.002× |
| nn / underfill | eager | 11.14 | 11.00 | 1.013× | 9.98 | 0.908× |
| nn / underfill | graph | 10.72 | 10.73 | 0.999× | 9.70 | 0.905× |
| nt / d128_in_proj | eager | 12.16 | 12.16 | 1.000× | 9.42 | 0.775× |
| nt / d128_in_proj | graph | 11.89 | 11.89 | 1.000× | 9.07 | 0.763× |
| nt / d128_out_proj | eager | 7.90 | 7.90 | 1.000× | 5.15 | 0.652× |
| nt / d128_out_proj | graph | 7.83 | 7.83 | 1.000× | 4.96 | 0.634× |
| nt / d768_in_proj | eager | 115.34 | 115.40 | 1.000× | 117.12 | 1.015× |
| nt / d768_in_proj | graph | 114.88 | 114.96 | 0.999× | 116.74 | 1.015× |
| nt / d768_out_proj | eager | 64.94 | 64.98 | 0.999× | 66.78 | 1.028× |
| nt / d768_out_proj | graph | 63.30 | 63.31 | 1.000× | 66.26 | 1.047× |
| nt / large_deep | eager | 553.39 | 549.05 | 1.008× | 425.94 | 0.776× |
| nt / large_deep | graph | 552.96 | 550.49 | 1.004× | 426.72 | 0.775× |
| nt / prism_in_proj | eager | 144.66 | 134.54 | 1.075× | 76.98 | 0.572× |
| nt / prism_in_proj | graph | 144.40 | 134.32 | 1.075× | 76.46 | 0.569× |
| nt / underfill | eager | 10.13 | 10.13 | 1.000× | 7.99 | 0.788× |
| nt / underfill | graph | 10.14 | 10.14 | 1.000× | 7.64 | 0.753× |
| tn / d128_in_proj | eager | 16.62 | 16.63 | 1.000× | 10.09 | 0.607× |
| tn / d128_in_proj | graph | 16.45 | 16.45 | 1.000× | 9.36 | 0.569× |
| tn / d128_out_proj | eager | 10.22 | 10.22 | 1.000× | 9.85 | 0.963× |
| tn / d128_out_proj | graph | 9.96 | 9.96 | 1.000× | 8.98 | 0.901× |
| tn / d768_in_proj | eager | 159.17 | 159.26 | 0.999× | 133.99 | 0.841× |
| tn / d768_in_proj | graph | 157.67 | 157.71 | 1.000× | 132.06 | 0.837× |
| tn / d768_out_proj | eager | 94.10 | 94.10 | 1.000× | 82.60 | 0.878× |
| tn / d768_out_proj | graph | 93.41 | 93.44 | 1.000× | 82.28 | 0.881× |
| tn / large_deep | eager | 602.19 | 601.88 | 1.001× | 475.01 | 0.789× |
| tn / large_deep | graph | 600.14 | 600.06 | 1.000× | 471.36 | 0.786× |
| tn / prism_in_proj | eager | 140.58 | 140.43 | 1.001× | 100.33 | 0.714× |
| tn / prism_in_proj | graph | 139.11 | 139.03 | 1.001× | 99.28 | 0.714× |
| tn / underfill | eager | 10.43 | 10.43 | 1.000× | 7.82 | 0.750× |
| tn / underfill | graph | 10.22 | 10.22 | 1.001× | 7.82 | 0.766× |


## Full matrix, capacity 16

Capacity 16 has no matched old-source run; no release speedup is inferred for it.

### inference — BF16 → BF16 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 64.36 | 75.45 | 1.172× |
| hot_a / bias off | graph | 63.96 | 75.10 | 1.174× |
| hot_a / bias on | eager | 65.60 | 109.45 | 1.669× |
| hot_a / bias on | graph | 65.23 | 108.70 | 1.666× |
| hot_b / bias off | eager | 122.47 | 100.85 | 0.823× |
| hot_b / bias off | graph | 123.07 | 100.97 | 0.820× |
| hot_b / bias on | eager | 118.89 | 121.49 | 1.022× |
| hot_b / bias on | graph | 119.84 | 121.85 | 1.017× |
| hot_c / bias off | eager | 54.19 | 75.10 | 1.386× |
| hot_c / bias off | graph | 54.03 | 75.54 | 1.398× |
| hot_c / bias on | eager | 54.51 | 84.86 | 1.557× |
| hot_c / bias on | graph | 54.49 | 86.14 | 1.581× |
| hot_d / bias off | eager | 65.69 | 60.79 | 0.925× |
| hot_d / bias off | graph | 65.10 | 59.47 | 0.913× |
| hot_d / bias on | eager | 66.46 | 80.41 | 1.210× |
| hot_d / bias on | graph | 65.71 | 79.30 | 1.207× |
| hot_e / bias off | eager | 60.62 | 69.31 | 1.143× |
| hot_e / bias off | graph | 60.60 | 69.25 | 1.143× |
| hot_e / bias on | eager | 60.89 | 77.81 | 1.278× |
| hot_e / bias on | graph | 60.69 | 76.40 | 1.259× |

### inference — BF16 → F32 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 88.14 | 76.51 | 0.868× |
| hot_a / bias off | graph | 88.16 | 76.30 | 0.865× |
| hot_a / bias on | eager | 86.02 | 110.93 | 1.290× |
| hot_a / bias on | graph | 85.67 | 110.25 | 1.287× |
| hot_b / bias off | eager | 144.70 | 121.03 | 0.836× |
| hot_b / bias off | graph | 143.61 | 120.45 | 0.839× |
| hot_b / bias on | eager | 124.78 | 137.80 | 1.104× |
| hot_b / bias on | graph | 123.18 | 136.13 | 1.105× |
| hot_c / bias off | eager | 49.60 | 76.20 | 1.536× |
| hot_c / bias off | graph | 49.26 | 75.63 | 1.535× |
| hot_c / bias on | eager | 50.46 | 86.18 | 1.708× |
| hot_c / bias on | graph | 50.14 | 86.82 | 1.732× |
| hot_d / bias off | eager | 98.43 | 61.40 | 0.624× |
| hot_d / bias off | graph | 98.67 | 60.37 | 0.612× |
| hot_d / bias on | eager | 98.45 | 80.60 | 0.819× |
| hot_d / bias on | graph | 98.25 | 79.45 | 0.809× |
| hot_e / bias off | eager | 85.20 | 68.93 | 0.809× |
| hot_e / bias off | graph | 84.92 | 68.71 | 0.809× |
| hot_e / bias on | eager | 85.27 | 77.76 | 0.912× |
| hot_e / bias on | graph | 85.10 | 76.91 | 0.904× |

### inference — F16 → F16 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 64.88 | 74.56 | 1.149× |
| hot_a / bias off | graph | 64.97 | 74.26 | 1.143× |
| hot_a / bias on | eager | 65.40 | 108.38 | 1.657× |
| hot_a / bias on | graph | 65.00 | 107.61 | 1.655× |
| hot_b / bias off | eager | 133.32 | 115.80 | 0.869× |
| hot_b / bias off | graph | 134.23 | 115.48 | 0.860× |
| hot_b / bias on | eager | 130.84 | 138.55 | 1.059× |
| hot_b / bias on | graph | 129.65 | 136.87 | 1.056× |
| hot_c / bias off | eager | 54.43 | 74.82 | 1.375× |
| hot_c / bias off | graph | 54.32 | 75.22 | 1.385× |
| hot_c / bias on | eager | 54.36 | 84.09 | 1.547× |
| hot_c / bias on | graph | 54.28 | 85.38 | 1.573× |
| hot_d / bias off | eager | 67.91 | 67.26 | 0.990× |
| hot_d / bias off | graph | 70.08 | 67.38 | 0.961× |
| hot_d / bias on | eager | 66.41 | 80.18 | 1.207× |
| hot_d / bias on | graph | 65.66 | 78.62 | 1.197× |
| hot_e / bias off | eager | 59.78 | 61.97 | 1.037× |
| hot_e / bias off | graph | 59.17 | 61.45 | 1.039× |
| hot_e / bias on | eager | 61.87 | 69.23 | 1.119× |
| hot_e / bias on | graph | 61.75 | 68.54 | 1.110× |

### inference — F16 → F32 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 96.24 | 80.45 | 0.836× |
| hot_a / bias off | graph | 94.12 | 79.13 | 0.841× |
| hot_a / bias on | eager | 87.80 | 112.25 | 1.278× |
| hot_a / bias on | graph | 86.92 | 111.23 | 1.280× |
| hot_b / bias off | eager | 154.88 | 134.66 | 0.869× |
| hot_b / bias off | graph | 154.28 | 134.04 | 0.869× |
| hot_b / bias on | eager | 132.88 | 151.52 | 1.140× |
| hot_b / bias on | graph | 132.53 | 151.52 | 1.143× |
| hot_c / bias off | eager | 52.91 | 77.33 | 1.462× |
| hot_c / bias off | graph | 53.37 | 77.26 | 1.448× |
| hot_c / bias on | eager | 51.04 | 86.36 | 1.692× |
| hot_c / bias on | graph | 50.90 | 87.04 | 1.710× |
| hot_d / bias off | eager | 102.28 | 63.56 | 0.621× |
| hot_d / bias off | graph | 104.13 | 63.60 | 0.611× |
| hot_d / bias on | eager | 98.50 | 80.63 | 0.819× |
| hot_d / bias on | graph | 98.74 | 79.58 | 0.806× |
| hot_e / bias off | eager | 85.13 | 62.59 | 0.735× |
| hot_e / bias off | graph | 84.90 | 62.14 | 0.732× |
| hot_e / bias on | eager | 85.26 | 69.72 | 0.818× |
| hot_e / bias on | graph | 85.05 | 69.00 | 0.811× |

### inference — exact F32 / Pedantic

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 266.48 | 294.31 | 1.104× |
| hot_a / bias off | graph | 269.46 | 297.42 | 1.104× |
| hot_a / bias on | eager | 251.03 | 303.73 | 1.210× |
| hot_a / bias on | graph | 252.05 | 304.43 | 1.208× |
| hot_b / bias off | eager | 694.74 | 708.61 | 1.020× |
| hot_b / bias off | graph | 697.86 | 706.96 | 1.013× |
| hot_b / bias on | eager | 638.76 | 683.70 | 1.070× |
| hot_b / bias on | graph | 638.63 | 684.87 | 1.072× |
| hot_c / bias off | eager | 285.11 | 233.79 | 0.820× |
| hot_c / bias off | graph | 284.09 | 232.39 | 0.818× |
| hot_c / bias on | eager | 337.26 | 230.31 | 0.683× |
| hot_c / bias on | graph | 341.21 | 233.69 | 0.685× |
| hot_d / bias off | eager | 219.52 | 262.58 | 1.196× |
| hot_d / bias off | graph | 219.74 | 262.92 | 1.197× |
| hot_d / bias on | eager | 218.30 | 277.22 | 1.270× |
| hot_d / bias on | graph | 216.18 | 275.15 | 1.273× |
| hot_e / bias off | eager | 267.72 | 271.25 | 1.013× |
| hot_e / bias off | graph | 267.31 | 269.74 | 1.009× |
| hot_e / bias on | eager | 266.08 | 276.25 | 1.038× |
| hot_e / bias on | graph | 265.89 | 275.63 | 1.037× |

### inference — exact F32 / Fast TF32

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 211.78 | 116.89 | 0.552× |
| hot_a / bias off | graph | 210.62 | 116.48 | 0.553× |
| hot_a / bias on | eager | 212.71 | 144.48 | 0.679× |
| hot_a / bias on | graph | 211.82 | 142.13 | 0.671× |
| hot_b / bias off | eager | 532.81 | 239.69 | 0.450× |
| hot_b / bias off | graph | 528.34 | 238.42 | 0.451× |
| hot_b / bias on | eager | 520.41 | 270.00 | 0.519× |
| hot_b / bias on | graph | 527.83 | 272.63 | 0.517× |
| hot_c / bias off | eager | 239.43 | 82.08 | 0.343× |
| hot_c / bias off | graph | 240.80 | 81.75 | 0.339× |
| hot_c / bias on | eager | 281.96 | 86.14 | 0.306× |
| hot_c / bias on | graph | 283.93 | 85.83 | 0.302× |
| hot_d / bias off | eager | 217.52 | 107.72 | 0.495× |
| hot_d / bias off | graph | 217.29 | 106.35 | 0.489× |
| hot_d / bias on | eager | 215.45 | 123.35 | 0.573× |
| hot_d / bias on | graph | 217.06 | 122.81 | 0.566× |
| hot_e / bias off | eager | 223.64 | 101.74 | 0.455× |
| hot_e / bias off | graph | 221.53 | 100.58 | 0.454× |
| hot_e / bias on | eager | 218.15 | 106.80 | 0.490× |
| hot_e / bias on | graph | 216.02 | 104.92 | 0.486× |

### inference — TF32 / Fast TF32

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| hot_a / bias off | eager | 111.29 | 103.01 | 0.926× |
| hot_a / bias off | graph | 110.31 | 102.59 | 0.930× |
| hot_a / bias on | eager | 113.79 | 131.19 | 1.153× |
| hot_a / bias on | graph | 113.08 | 130.09 | 1.150× |
| hot_b / bias off | eager | 220.16 | 201.00 | 0.913× |
| hot_b / bias off | graph | 219.46 | 201.01 | 0.916× |
| hot_b / bias on | eager | 219.52 | 232.09 | 1.057× |
| hot_b / bias on | graph | 219.12 | 231.54 | 1.057× |
| hot_c / bias off | eager | 102.10 | 77.87 | 0.763× |
| hot_c / bias off | graph | 101.91 | 77.53 | 0.761× |
| hot_c / bias on | eager | 102.63 | 86.13 | 0.839× |
| hot_c / bias on | graph | 102.32 | 85.20 | 0.833× |
| hot_d / bias off | eager | 125.32 | 99.29 | 0.792× |
| hot_d / bias off | graph | 123.14 | 98.64 | 0.801× |
| hot_d / bias on | eager | 126.67 | 116.14 | 0.917× |
| hot_d / bias on | graph | 125.17 | 114.72 | 0.917× |
| hot_e / bias off | eager | 101.47 | 91.11 | 0.898× |
| hot_e / bias off | graph | 101.15 | 90.74 | 0.897× |
| hot_e / bias on | eager | 119.54 | 98.65 | 0.825× |
| hot_e / bias on | graph | 119.17 | 97.72 | 0.820× |

### triad — BF16 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| nn / d128_in_proj | eager | 5.63 | 4.80 | 0.853× |
| nn / d128_in_proj | graph | 5.31 | 4.82 | 0.907× |
| nn / d128_out_proj | eager | 6.68 | 5.44 | 0.814× |
| nn / d128_out_proj | graph | 6.42 | 4.96 | 0.772× |
| nn / d768_in_proj | eager | 62.24 | 76.30 | 1.226× |
| nn / d768_in_proj | graph | 61.77 | 75.67 | 1.225× |
| nn / d768_out_proj | eager | 37.95 | 47.89 | 1.262× |
| nn / d768_out_proj | graph | 37.76 | 47.72 | 1.264× |
| nn / prism_in_proj | eager | 59.66 | 73.54 | 1.233× |
| nn / prism_in_proj | graph | 59.13 | 73.16 | 1.237× |
| nt / d128_in_proj | eager | 9.37 | 7.64 | 0.816× |
| nt / d128_in_proj | graph | 9.16 | 7.15 | 0.781× |
| nt / d128_out_proj | eager | 5.39 | 4.71 | 0.872× |
| nt / d128_out_proj | graph | 5.18 | 4.28 | 0.827× |
| nt / d768_in_proj | eager | 71.02 | 77.10 | 1.086× |
| nt / d768_in_proj | graph | 70.68 | 77.00 | 1.089× |
| nt / d768_out_proj | eager | 36.50 | 39.39 | 1.079× |
| nt / d768_out_proj | graph | 36.18 | 39.44 | 1.090× |
| nt / prism_in_proj | eager | 48.65 | 63.00 | 1.295× |
| nt / prism_in_proj | graph | 48.67 | 62.30 | 1.280× |
| tn / d128_in_proj | eager | 18.34 | 7.93 | 0.432× |
| tn / d128_in_proj | graph | 18.20 | 7.45 | 0.409× |
| tn / d128_out_proj | eager | 12.27 | 7.14 | 0.582× |
| tn / d128_out_proj | graph | 12.14 | 6.48 | 0.534× |
| tn / d768_in_proj | eager | 86.74 | 90.05 | 1.038× |
| tn / d768_in_proj | graph | 86.57 | 87.64 | 1.012× |
| tn / d768_out_proj | eager | 55.31 | 50.81 | 0.919× |
| tn / d768_out_proj | graph | 55.09 | 49.27 | 0.894× |
| tn / prism_in_proj | eager | 77.73 | 55.01 | 0.708× |
| tn / prism_in_proj | graph | 77.06 | 53.46 | 0.694× |

### triad — F16 / Fast

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| nn / d128_in_proj | eager | 5.63 | 5.35 | 0.951× |
| nn / d128_in_proj | graph | 5.31 | 5.06 | 0.953× |
| nn / d128_out_proj | eager | 6.68 | 4.72 | 0.705× |
| nn / d128_out_proj | graph | 6.42 | 4.25 | 0.663× |
| nn / d768_in_proj | eager | 62.25 | 59.26 | 0.952× |
| nn / d768_in_proj | graph | 61.77 | 58.81 | 0.952× |
| nn / d768_out_proj | eager | 37.96 | 47.64 | 1.255× |
| nn / d768_out_proj | graph | 37.76 | 47.36 | 1.254× |
| nn / prism_in_proj | eager | 59.78 | 73.33 | 1.227× |
| nn / prism_in_proj | graph | 59.22 | 72.82 | 1.230× |
| nt / d128_in_proj | eager | 9.37 | 5.83 | 0.622× |
| nt / d128_in_proj | graph | 9.16 | 5.27 | 0.576× |
| nt / d128_out_proj | eager | 5.39 | 4.50 | 0.835× |
| nt / d128_out_proj | graph | 5.18 | 4.02 | 0.777× |
| nt / d768_in_proj | eager | 70.95 | 73.15 | 1.031× |
| nt / d768_in_proj | graph | 70.70 | 76.94 | 1.088× |
| nt / d768_out_proj | eager | 36.52 | 40.84 | 1.118× |
| nt / d768_out_proj | graph | 36.19 | 40.30 | 1.114× |
| nt / prism_in_proj | eager | 48.66 | 96.58 | 1.985× |
| nt / prism_in_proj | graph | 48.68 | 95.75 | 1.967× |
| tn / d128_in_proj | eager | 18.34 | 7.88 | 0.430× |
| tn / d128_in_proj | graph | 18.11 | 7.47 | 0.413× |
| tn / d128_out_proj | eager | 12.27 | 7.44 | 0.606× |
| tn / d128_out_proj | graph | 12.00 | 6.58 | 0.548× |
| tn / d768_in_proj | eager | 86.75 | 89.99 | 1.037× |
| tn / d768_in_proj | graph | 86.68 | 86.82 | 1.002× |
| tn / d768_out_proj | eager | 56.25 | 50.82 | 0.903× |
| tn / d768_out_proj | graph | 56.06 | 49.27 | 0.879× |
| tn / prism_in_proj | eager | 77.60 | 54.90 | 0.707× |
| tn / prism_in_proj | graph | 76.80 | 53.48 | 0.696× |

### triad — exact F32 / Fast TF32

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| nn / d128_in_proj | eager | 15.71 | 5.67 | 0.361× |
| nn / d128_in_proj | graph | 15.04 | 5.53 | 0.367× |
| nn / d128_out_proj | eager | 10.18 | 6.81 | 0.669× |
| nn / d128_out_proj | graph | 9.50 | 6.58 | 0.693× |
| nn / d768_in_proj | eager | 293.69 | 119.97 | 0.408× |
| nn / d768_in_proj | graph | 292.33 | 118.61 | 0.406× |
| nn / d768_out_proj | eager | 137.55 | 64.14 | 0.466× |
| nn / d768_out_proj | graph | 137.73 | 63.97 | 0.464× |
| nn / prism_in_proj | eager | 200.51 | 110.61 | 0.552× |
| nn / prism_in_proj | graph | 202.84 | 112.22 | 0.553× |
| nt / d128_in_proj | eager | 17.16 | 9.42 | 0.549× |
| nt / d128_in_proj | graph | 15.69 | 9.06 | 0.577× |
| nt / d128_out_proj | eager | 12.83 | 5.15 | 0.401× |
| nt / d128_out_proj | graph | 11.35 | 4.97 | 0.438× |
| nt / d768_in_proj | eager | 287.56 | 123.07 | 0.428× |
| nt / d768_in_proj | graph | 286.64 | 123.38 | 0.430× |
| nt / d768_out_proj | eager | 140.78 | 67.36 | 0.478× |
| nt / d768_out_proj | graph | 138.96 | 66.39 | 0.478× |
| nt / prism_in_proj | eager | 299.04 | 76.98 | 0.257× |
| nt / prism_in_proj | graph | 298.04 | 76.47 | 0.257× |
| tn / d128_in_proj | eager | 35.24 | 10.10 | 0.287× |
| tn / d128_in_proj | graph | 35.26 | 9.35 | 0.265× |
| tn / d128_out_proj | eager | 27.50 | 9.77 | 0.355× |
| tn / d128_out_proj | graph | 27.54 | 8.95 | 0.325× |
| tn / d768_in_proj | eager | 309.31 | 133.76 | 0.432× |
| tn / d768_in_proj | graph | 307.95 | 131.99 | 0.429× |
| tn / d768_out_proj | eager | 189.07 | 82.79 | 0.438× |
| tn / d768_out_proj | graph | 188.20 | 82.30 | 0.437× |
| tn / prism_in_proj | eager | 235.03 | 100.22 | 0.426× |
| tn / prism_in_proj | graph | 234.07 | 99.29 | 0.424× |

### triad — exact F32 / Pedantic

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| nn / d128_in_proj | eager | 15.71 | 10.00 | 0.636× |
| nn / d128_in_proj | graph | 15.04 | 9.36 | 0.623× |
| nn / d128_out_proj | eager | 10.18 | 8.70 | 0.854× |
| nn / d128_out_proj | graph | 9.50 | 8.44 | 0.888× |
| nn / d768_in_proj | eager | 275.55 | 333.64 | 1.211× |
| nn / d768_in_proj | graph | 278.52 | 335.09 | 1.203× |
| nn / d768_out_proj | eager | 166.15 | 171.88 | 1.035× |
| nn / d768_out_proj | graph | 166.99 | 172.97 | 1.036× |
| nn / prism_in_proj | eager | 255.73 | 282.42 | 1.104× |
| nn / prism_in_proj | graph | 257.32 | 285.16 | 1.108× |
| nt / d128_in_proj | eager | 17.15 | 13.75 | 0.802× |
| nt / d128_in_proj | graph | 15.69 | 13.36 | 0.851× |
| nt / d128_out_proj | eager | 12.83 | 8.19 | 0.638× |
| nt / d128_out_proj | graph | 11.34 | 7.79 | 0.687× |
| nt / d768_in_proj | eager | 357.56 | 358.22 | 1.002× |
| nt / d768_in_proj | graph | 357.97 | 357.49 | 0.999× |
| nt / d768_out_proj | eager | 170.07 | 176.17 | 1.036× |
| nt / d768_out_proj | graph | 170.54 | 176.67 | 1.036× |
| nt / prism_in_proj | eager | 310.42 | 251.57 | 0.810× |
| nt / prism_in_proj | graph | 307.83 | 248.85 | 0.808× |
| tn / d128_in_proj | eager | 35.24 | 12.70 | 0.360× |
| tn / d128_in_proj | graph | 35.26 | 11.31 | 0.321× |
| tn / d128_out_proj | eager | 27.51 | 10.55 | 0.383× |
| tn / d128_out_proj | graph | 27.54 | 9.83 | 0.357× |
| tn / d768_in_proj | eager | 309.40 | 268.87 | 0.869× |
| tn / d768_in_proj | graph | 307.89 | 266.85 | 0.867× |
| tn / d768_out_proj | eager | 189.19 | 135.05 | 0.714× |
| tn / d768_out_proj | graph | 188.15 | 133.80 | 0.711× |
| tn / prism_in_proj | eager | 235.14 | 189.51 | 0.806× |
| tn / prism_in_proj | graph | 234.11 | 187.84 | 0.802× |

### triad — TF32 / Fast TF32

| case | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---|---|---:|---:|---:|
| nn / d128_in_proj | eager | 9.95 | 5.67 | 0.570× |
| nn / d128_in_proj | graph | 9.74 | 5.51 | 0.566× |
| nn / d128_out_proj | eager | 7.65 | 6.81 | 0.889× |
| nn / d128_out_proj | graph | 7.55 | 6.58 | 0.872× |
| nn / d768_in_proj | eager | 124.13 | 102.04 | 0.822× |
| nn / d768_in_proj | graph | 123.86 | 102.35 | 0.826× |
| nn / d768_out_proj | eager | 63.68 | 62.84 | 0.987× |
| nn / d768_out_proj | graph | 63.32 | 62.53 | 0.988× |
| nn / large_deep | eager | 513.67 | 408.31 | 0.795× |
| nn / large_deep | graph | 512.47 | 407.86 | 0.796× |
| nn / prism_in_proj | eager | 104.17 | 103.06 | 0.989× |
| nn / prism_in_proj | graph | 102.44 | 102.70 | 1.003× |
| nn / underfill | eager | 11.00 | 9.99 | 0.908× |
| nn / underfill | graph | 10.73 | 9.71 | 0.905× |
| nt / d128_in_proj | eager | 12.16 | 9.42 | 0.775× |
| nt / d128_in_proj | graph | 11.89 | 9.07 | 0.763× |
| nt / d128_out_proj | eager | 7.90 | 5.15 | 0.652× |
| nt / d128_out_proj | graph | 7.83 | 4.97 | 0.635× |
| nt / d768_in_proj | eager | 115.40 | 117.16 | 1.015× |
| nt / d768_in_proj | graph | 114.94 | 116.74 | 1.016× |
| nt / d768_out_proj | eager | 64.95 | 66.81 | 1.029× |
| nt / d768_out_proj | graph | 63.30 | 66.27 | 1.047× |
| nt / large_deep | eager | 554.46 | 429.45 | 0.775× |
| nt / large_deep | graph | 553.15 | 428.73 | 0.775× |
| nt / prism_in_proj | eager | 134.56 | 76.98 | 0.572× |
| nt / prism_in_proj | graph | 134.34 | 76.47 | 0.569× |
| nt / underfill | eager | 10.13 | 7.98 | 0.788× |
| nt / underfill | graph | 10.14 | 7.64 | 0.753× |
| tn / d128_in_proj | eager | 16.64 | 10.13 | 0.609× |
| tn / d128_in_proj | graph | 16.45 | 9.36 | 0.569× |
| tn / d128_out_proj | eager | 10.22 | 9.85 | 0.964× |
| tn / d128_out_proj | graph | 9.96 | 8.98 | 0.902× |
| tn / d768_in_proj | eager | 159.10 | 133.64 | 0.840× |
| tn / d768_in_proj | graph | 157.71 | 132.06 | 0.837× |
| tn / d768_out_proj | eager | 94.17 | 82.76 | 0.879× |
| tn / d768_out_proj | graph | 93.41 | 82.28 | 0.881× |
| tn / large_deep | eager | 601.94 | 475.41 | 0.790× |
| tn / large_deep | graph | 600.17 | 471.50 | 0.786× |
| tn / prism_in_proj | eager | 140.32 | 100.18 | 0.714× |
| tn / prism_in_proj | graph | 138.98 | 99.27 | 0.714× |
| tn / underfill | eager | 10.43 | 7.82 | 0.750× |
| tn / underfill | graph | 10.22 | 7.82 | 0.765× |

## Exact-F32 supplement

Four cases outside the exact-F32 release average, measured once per capacity.

| capacity | case | comparator | path | 0.7.1 µs | cuBLAS µs | cuBLAS/new |
|---:|---|---|---|---:|---:|---:|
| 16 | nn / large_deep | exact F32 / Fast TF32 | eager | 1216.84 | 452.52 | 0.372× |
| 16 | nn / large_deep | exact F32 / Fast TF32 | graph | 1220.86 | 452.30 | 0.370× |
| 16 | nt / large_deep | exact F32 / Fast TF32 | eager | 1315.24 | 475.94 | 0.362× |
| 16 | nt / large_deep | exact F32 / Fast TF32 | graph | 1325.11 | 478.33 | 0.361× |
| 16 | tn / large_deep | exact F32 / Fast TF32 | eager | 1339.52 | 475.37 | 0.355× |
| 16 | tn / large_deep | exact F32 / Fast TF32 | graph | 1336.68 | 471.07 | 0.352× |
| 16 | tn / underfill | exact F32 / Fast TF32 | eager | 21.61 | 7.85 | 0.363× |
| 16 | tn / underfill | exact F32 / Fast TF32 | graph | 21.65 | 7.78 | 0.360× |
| 16 | nn / large_deep | exact F32 / Pedantic | eager | 1207.38 | 1145.42 | 0.949× |
| 16 | nn / large_deep | exact F32 / Pedantic | graph | 1207.12 | 1144.15 | 0.948× |
| 16 | nt / large_deep | exact F32 / Pedantic | eager | 1488.19 | 1495.96 | 1.005× |
| 16 | nt / large_deep | exact F32 / Pedantic | graph | 1496.73 | 1504.64 | 1.005× |
| 16 | tn / large_deep | exact F32 / Pedantic | eager | 1340.54 | 940.01 | 0.701× |
| 16 | tn / large_deep | exact F32 / Pedantic | graph | 1338.86 | 934.07 | 0.698× |
| 16 | tn / underfill | exact F32 / Pedantic | eager | 21.62 | 9.61 | 0.445× |
| 16 | tn / underfill | exact F32 / Pedantic | graph | 21.64 | 9.11 | 0.421× |
| 64 | nn / large_deep | exact F32 / Fast TF32 | eager | 1234.18 | 457.94 | 0.371× |
| 64 | nn / large_deep | exact F32 / Fast TF32 | graph | 1234.19 | 457.14 | 0.370× |
| 64 | nt / large_deep | exact F32 / Fast TF32 | eager | 1316.01 | 476.49 | 0.362× |
| 64 | nt / large_deep | exact F32 / Fast TF32 | graph | 1323.35 | 480.07 | 0.363× |
| 64 | tn / large_deep | exact F32 / Fast TF32 | eager | 1338.60 | 474.49 | 0.354× |
| 64 | tn / large_deep | exact F32 / Fast TF32 | graph | 1336.06 | 471.04 | 0.353× |
| 64 | tn / underfill | exact F32 / Fast TF32 | eager | 21.62 | 7.84 | 0.363× |
| 64 | tn / underfill | exact F32 / Fast TF32 | graph | 21.65 | 7.79 | 0.360× |
| 64 | nn / large_deep | exact F32 / Pedantic | eager | 1214.70 | 1155.33 | 0.951× |
| 64 | nn / large_deep | exact F32 / Pedantic | graph | 1225.81 | 1166.75 | 0.952× |
| 64 | nt / large_deep | exact F32 / Pedantic | eager | 1506.99 | 1515.01 | 1.005× |
| 64 | nt / large_deep | exact F32 / Pedantic | graph | 1506.13 | 1514.24 | 1.005× |
| 64 | tn / large_deep | exact F32 / Pedantic | eager | 1340.40 | 939.03 | 0.701× |
| 64 | tn / large_deep | exact F32 / Pedantic | graph | 1338.26 | 933.65 | 0.698× |
| 64 | tn / underfill | exact F32 / Pedantic | eager | 21.61 | 9.61 | 0.445× |
| 64 | tn / underfill | exact F32 / Pedantic | graph | 21.65 | 9.11 | 0.421× |

## Reproducing the matrix

Use a separate clean source tree and a fresh private kernel cache for each
version, with the toolkit named above. First select the idle CC 8.9 Ada GPU
to measure; do not copy the UUID in this page unless it names your device:

```sh
nvidia-smi --query-gpu=uuid,name,compute_cap --format=csv
export ADA_GPU_UUID='<UUID of the selected idle CC 8.9 Ada GPU>'
export CUDA_VISIBLE_DEVICES="$ADA_GPU_UUID"
```

In each source checkout, build both instruments before timing:

```sh
cargo test --release --locked --features cuda,qualification \
  --test gemm_bi_inference_performance --test gemm_bi_performance_matrix \
  --no-run
```

The exact ignored tests are
`fixed_ada_production_auto_paired_precision_cublas` in
`gemm_bi_inference_performance` and
`gemm_bi_production_auto_paired_cublas_release_matrix` in
`gemm_bi_performance_matrix`. Run with
`--exact --ignored --nocapture --test-threads=1`.

Start from the full clean selection below. Kernel caches must live under a
trusted user-owned home/cache parent: use a physical absolute path with no
symlink components and no group- or world-writable ancestor, and keep the
new leaf at mode 0700. A general temporary directory is not suitable because
the loader rejects caches with writable ancestors. The cache shown is for
the current checkout; create a separate one in the v0.7.0 checkout.
`git rev-parse HEAD` binds every record to the source actually being run.

```sh
unset MAMBA_RS_GEMM_MODE MAMBA_RS_BATCH_INVARIANT MAMBA_RS_FAST_GEMM \
  MAMBA_RS_BI_TENSOR_CORES MAMBA_RS_BI_F32_POLICY MAMBA_RS_BI_HALF_POLICY \
  MAMBA_RS_BI_GEMM_FAMILY MAMBA_RS_ARCH_RUNG MAMBA_RS_BENCH_IEEE_F32 \
  MAMBA_FIXED_VENDOR_TILES GEMM_BI_QUAL_PATH_ORDER GEMM_BI_QUAL_CELL_IDS
unset GEMM_BI_QUAL_STATE_CAP GEMM_BI_QUAL_VARIANT \
  GEMM_BI_FINAL_AUTO_JSONL GEMM_BI_FINAL_AUTO_GIT_SHA

test "$(git rev-parse HEAD)" = '83079104fe1efa7ad5dca0a28c5b48bcd86c5b14'
CACHE_PARENT="$(cd "$HOME" && pwd -P)"
export MAMBA_RS_KERNEL_CACHE="$(mktemp -d "$CACHE_PARENT/.mamba-release071-cache.XXXXXX")"
chmod 700 "$MAMBA_RS_KERNEL_CACHE"
export GEMM_BI_FINAL_AUTO_GIT_SHA="$(git rev-parse HEAD)"
export GEMM_BI_QUAL_VARIANT=release071-full-current-cap64
export NVIDIA_TF32_OVERRIDE=1
export MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9
export MAMBA_FIXED_VENDOR_PATHS=eager,graph
export MAMBA_FIXED_ADA_WINDOWS=21 GEMM_BI_QUAL_WINDOWS=21
export MAMBA_FIXED_ADA_ROWS=bf16,f16,bf16_f32,f16_f32,tf32,f32_exact,f32_exact_fast
export MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e
export MAMBA_FIXED_ADA_BIAS=0,1
export GEMM_BI_QUAL_STATE_CAP=64
unset GEMM_BI_QUAL_CELL_IDS

TRIAD_OUT_DIR="$(mktemp -d "$PWD/release071-triad.XXXXXX")"
export GEMM_BI_FINAL_AUTO_JSONL="$TRIAD_OUT_DIR/evidence.jsonl"
test ! -e "$GEMM_BI_FINAL_AUTO_JSONL"
```

Before every additional Triad invocation, create another `TRIAD_OUT_DIR`
with `mktemp -d` and point `GEMM_BI_FINAL_AUTO_JSONL` at its nonexistent
`evidence.jsonl`; the sink uses create-new semantics. In the native v0.7.0
checkout, require `git rev-parse HEAD` to equal the released SHA shown
above, create a separate private cache, bind that SHA, and leave capacity
unset because its default is already 64:

```sh
test "$(git rev-parse HEAD)" = 'e2917a47494b4a1d652f5c818c3974ec3fcdd1ca'
CACHE_PARENT="$(cd "$HOME" && pwd -P)"
export MAMBA_RS_KERNEL_CACHE="$(mktemp -d "$CACHE_PARENT/.mamba-release070-cache.XXXXXX")"
chmod 700 "$MAMBA_RS_KERNEL_CACHE"
export GEMM_BI_FINAL_AUTO_GIT_SHA="$(git rev-parse HEAD)"
export GEMM_BI_QUAL_VARIANT=release071-full-old-cap64
unset GEMM_BI_QUAL_STATE_CAP
```

Repeat the current suite with capacity 16 for the companion table. A
complete Inference run emits 280 paired records; a complete Triad run emits
324 plus its completion record. Do not publish a filtered run as the full
matrix. Run both capacity-64 suites twice per tree in old, new, new, old order.

The separate supplement uses
`gemm_bi_production_auto_paired_cublas_outlier_matrix`, filtered to
`f32_policy_exact/tn/underfill/contiguous` and the NN/NT/TN
`f32_policy_exact/*/large_deep/contiguous` cells. It emits 32 paired
records plus completion per capacity. The asterisk here describes
three explicit cell IDs, not a filter wildcard.

Numerical acceptance is separate from timing. On this Ada board the
combined Inference/Triad corpus matches released 0.7.0 output words for
all 99 cases under CUDA 12.8, 13.0 and 13.2, at both capacities: 594
case/toolkit/capacity combinations per version. Finite and exceptional
inputs are checked over six independent runs, along with eager/graph
execution and physical launch identities. This does not assert bit
identity across different GPUs or toolkits.
