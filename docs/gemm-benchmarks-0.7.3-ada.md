# GEMM benchmarks — 0.7.3, RTX 6000 Ada

Measured on September 18, 2026 with the 0.7.3 tree: one run of the
kernel-level adapter that timed 0.7.1 on this board, against two saved
records of the same cells. These are GEMM timings on the board the routes
were measured on; no other board is timed here, and the bits of every
route are the 0.7.1 bits (bit ledger 263 of 263 unchanged).

## Measurement scope

| item | value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, SM89, 142 SMs |
| measured GPU UUID | `GPU-d1edd7be-e88d-aed6-047d-622163306f0e` |
| driver / toolkit | 595.45.04 / CUDA 13.2.51, NVRTC 13.2 |
| host compiler | Rust 1.98.1, release profile |
| GEMM mode | `Deterministic`; the automatic route, no forced candidate |
| storage precisions | exact F32, TF32, BF16, F16 |
| path | eager; every cell times three arms in one process |

Each cell times the deterministic route, cuBLAS in the tree's fast
setting and cuBLAS in the tree's pedantic setting in windows of calibrated
launch counts, 21 windows, the arm order mirrored from window to window;
the number is the lower median per arm. The row's `kernel` is the symbol
the route recorded for the launch.

Two saved records are compared, and they are not the same instrument:

- **adapter 0.7.1** - this adapter, run on September 11 on the 0.7.1
  candidate tree, median of its four runs. Same harness, same operands,
  same windows: the like-for-like comparison. It has no TF32 rows.
- **page 0.7.1** - the eager column of the [0.7.1 page](gemm-benchmarks-0.7.1-ada.md),
  taken by the qualification matrix harness on September 14. A different
  instrument on a different day; `vendor page/new` divides its cuBLAS
  time by this run's cuBLAS time on the same cell, so a row where the two
  harnesses disagree on cuBLAS shows it there before the deterministic
  ratio is read.

**Ratios above 1 mean faster.** Row summaries are geometric means of
per-cell ratios over the cells the respective record carries. A single
cell of one run moves by a few percent from run to run on this stand;
a ratio inside that band is not a change.

## Summary

| family / precision / comparator | cells vs adapter 0.7.1 | adapter 0.7.1/new | cells vs page | page 0.7.1/new | vendor control |
|---|---:|---:|---:|---:|---:|
| triad — exact F32 / Fast TF32 | 21 | 1.163× | 15 | 1.013× | 0.939× |
| triad — exact F32 / Pedantic | 21 | 1.163× | 15 | 1.073× | 0.987× |
| triad — TF32 / Fast TF32 | 0 | — | 21 | 0.992× | 0.941× |
| triad — BF16 / Fast | 21 | 1.165× | 15 | 1.174× | 0.997× |
| triad — F16 / Fast | 21 | 1.159× | 15 | 1.091× | 0.952× |
| inference — exact F32 / Fast TF32 | 5 | 1.043× | 5 | 0.917× | 0.907× |
| inference — exact F32 / Pedantic | 5 | 1.043× | 5 | 1.091× | 1.077× |
| inference — TF32 / Fast TF32 | 0 | — | 5 | 0.938× | 0.886× |
| inference — BF16 / Fast | 5 | 0.999× | 5 | 0.984× | 0.986× |
| inference — F16 / Fast | 5 | 0.978× | 5 | 0.900× | 0.949× |

## Per-cell tables

### triad — exact F32 / Fast TF32

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nn / d128_in_proj | `nn_splitk32_partial+splitk_reduce` | 15.75 | 15.80 | 1.003× | 15.71 | 0.997× | 5.69 | 0.361× | 0.996× |
| nn / d128_out_proj | `nn_splitk32_partial+splitk_reduce` | 10.30 | 10.10 | 0.981× | 10.19 | 0.990× | 6.82 | 0.662× | 1.001× |
| nn / d768_in_proj | `nn_sm89_f32_n64_copyplan` | 297.27 | 296.50 | 0.997× | 293.92 | 0.989× | 118.46 | 0.399× | 1.013× |
| nn / d768_out_proj | `nn_sm89_f32_n64_copyplan` | 160.80 | 162.95 | 1.013× | 138.03 | 0.858× | 74.00 | 0.460× | 0.870× |
| nn / large_deep | `nn_sm89_f32_n64_copyplan` | 1268.48 | 1310.48 | 1.033× | — | —× | 481.46 | 0.380× | —× |
| nn / prism_in_proj | `nn_sm89_f32_n64_copyplan` | 239.10 | 241.36 | 1.009× | 200.17 | 0.837× | 128.71 | 0.538× | 0.863× |
| nn / underfill | `nn_splitk32_partial+splitk_reduce` | 12.76 | 12.67 | 0.993× | — | —× | 10.04 | 0.787× | —× |
| tn / d128_in_proj | `tn_sm89_f32_d128_in_m16n16_g8_s2_cg` | 19.61 | 35.40 | 1.805× | 35.24 | 1.797× | 10.04 | 0.512× | 1.005× |
| tn / d128_out_proj | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 13.20 | 27.53 | 2.085× | 27.50 | 2.083× | 9.82 | 0.744× | 0.996× |
| tn / d768_in_proj | `transpose_f32_32x16_d768+tn_sm89_f32_n64_dual_chunk_fused_finalize` | 321.08 | 320.63 | 0.999× | 309.37 | 0.964× | 135.31 | 0.421× | 0.989× |
| tn / d768_out_proj | `tn_sm89_f32_m64n64_bk16_s2_d768_out_raw+splitm_reduce` | 192.56 | 195.53 | 1.015× | 188.78 | 0.980× | 82.35 | 0.428× | 1.003× |
| tn / large_deep | `tn_aligned` | 1371.65 | 1395.22 | 1.017× | — | —× | 484.86 | 0.353× | —× |
| tn / prism_in_proj | `tn_sm89_f32_m64n64_bk16_s2_prism_raw+splitm_reduce` | 281.90 | 281.15 | 0.997× | 235.05 | 0.834× | 119.91 | 0.425× | 0.836× |
| tn / underfill | `tn_m16n16_bk16_s2_splitm16` | 21.59 | 80.66 | 3.736× | — | —× | 7.82 | 0.362× | —× |
| nt / d128_in_proj | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 17.20 | 17.17 | 0.998× | 17.16 | 0.998× | 9.33 | 0.543× | 1.009× |
| nt / d128_out_proj | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 12.89 | 12.84 | 0.996× | 12.83 | 0.995× | 5.04 | 0.391× | 1.019× |
| nt / d768_in_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 346.61 | 361.29 | 1.042× | 285.13 | 0.823× | 149.47 | 0.431× | 0.821× |
| nt / d768_out_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 173.46 | 175.41 | 1.011× | 139.96 | 0.807× | 82.72 | 0.477× | 0.810× |
| nt / large_deep | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 1564.16 | 2335.23 | 1.493× | — | —× | 572.32 | 0.366× | —× |
| nt / prism_in_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 335.71 | 339.76 | 1.012× | 298.99 | 0.891× | 85.80 | 0.256× | 0.897× |
| nt / underfill | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 15.59 | 15.69 | 1.007× | — | —× | 8.00 | 0.513× | —× |
### triad — exact F32 / Pedantic

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nn / d128_in_proj | `nn_splitk32_partial+splitk_reduce` | 15.75 | 15.80 | 1.003× | 15.71 | 0.997× | 9.98 | 0.634× | 1.002× |
| nn / d128_out_proj | `nn_splitk32_partial+splitk_reduce` | 10.30 | 10.10 | 0.981× | 10.18 | 0.989× | 8.69 | 0.844× | 1.001× |
| nn / d768_in_proj | `nn_sm89_f32_n64_copyplan` | 297.27 | 296.50 | 0.997× | 276.67 | 0.931× | 346.19 | 1.165× | 0.964× |
| nn / d768_out_proj | `nn_sm89_f32_n64_copyplan` | 160.80 | 162.95 | 1.013× | 167.77 | 1.043× | 166.50 | 1.035× | 1.042× |
| nn / large_deep | `nn_sm89_f32_n64_copyplan` | 1268.48 | 1310.48 | 1.033× | — | —× | 1254.91 | 0.989× | —× |
| nn / prism_in_proj | `nn_sm89_f32_n64_copyplan` | 239.10 | 241.36 | 1.009× | 254.99 | 1.066× | 255.94 | 1.070× | 1.100× |
| nn / underfill | `nn_splitk32_partial+splitk_reduce` | 12.76 | 12.67 | 0.993× | — | —× | 12.23 | 0.959× | —× |
| tn / d128_in_proj | `tn_sm89_f32_d128_in_m16n16_g8_s2_cg` | 19.61 | 35.40 | 1.805× | 35.24 | 1.797× | 12.73 | 0.649× | 0.997× |
| tn / d128_out_proj | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 13.20 | 27.53 | 2.085× | 27.50 | 2.083× | 10.47 | 0.793× | 1.008× |
| tn / d768_in_proj | `transpose_f32_32x16_d768+tn_sm89_f32_n64_dual_chunk_fused_finalize` | 321.08 | 320.63 | 0.999× | 309.41 | 0.964× | 272.59 | 0.849× | 0.986× |
| tn / d768_out_proj | `tn_sm89_f32_m64n64_bk16_s2_d768_out_raw+splitm_reduce` | 192.56 | 195.53 | 1.015× | 188.81 | 0.981× | 138.13 | 0.717× | 0.974× |
| tn / large_deep | `tn_aligned` | 1371.65 | 1395.22 | 1.017× | — | —× | 948.74 | 0.692× | —× |
| tn / prism_in_proj | `tn_sm89_f32_m64n64_bk16_s2_prism_raw+splitm_reduce` | 281.90 | 281.15 | 0.997× | 235.17 | 0.834× | 224.26 | 0.796× | 0.845× |
| tn / underfill | `tn_m16n16_bk16_s2_splitm16` | 21.59 | 80.66 | 3.736× | — | —× | 9.54 | 0.442× | —× |
| nt / d128_in_proj | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 17.20 | 17.17 | 0.998× | 17.15 | 0.997× | 13.82 | 0.804× | 0.995× |
| nt / d128_out_proj | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 12.89 | 12.84 | 0.996× | 12.83 | 0.995× | 8.20 | 0.636× | 0.998× |
| nt / d768_in_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 346.61 | 361.29 | 1.042× | 359.09 | 1.036× | 355.88 | 1.027× | 1.008× |
| nt / d768_out_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 173.46 | 175.41 | 1.011× | 172.42 | 0.994× | 182.97 | 1.055× | 0.974× |
| nt / large_deep | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 1564.16 | 2335.23 | 1.493× | — | —× | 1587.84 | 1.015× | —× |
| nt / prism_in_proj | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 335.71 | 339.76 | 1.012× | 312.26 | 0.930× | 273.09 | 0.813× | 0.928× |
| nt / underfill | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 15.59 | 15.69 | 1.007× | — | —× | 11.77 | 0.755× | —× |
### triad — TF32 / Fast TF32

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nn / d128_in_proj | `nn_sm80_mma_tf32_m64n64_bk32_s2` | 9.89 | — | —× | 9.95 | 1.006× | 5.67 | 0.574× | 0.998× |
| nn / d128_out_proj | `nn_sm80_mma_tf32_m16n32_bk32_s4` | 7.62 | — | —× | 7.65 | 1.004× | 6.80 | 0.893× | 1.001× |
| nn / d768_in_proj | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 129.79 | — | —× | 124.57 | 0.960× | 106.49 | 0.820× | 0.961× |
| nn / d768_out_proj | `nn_sm89_tf32_addhalf_m128n96_bk32_s3` | 71.47 | — | —× | 63.70 | 0.891× | 71.45 | 1.000× | 0.880× |
| nn / large_deep | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 550.11 | — | —× | 507.43 | 0.922× | 433.34 | 0.788× | 0.931× |
| nn / prism_in_proj | `nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct` | 119.24 | — | —× | 104.17 | 0.874× | 117.00 | 0.981× | 0.881× |
| nn / underfill | `nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4` | 11.01 | — | —× | 11.00 | 0.999× | 10.04 | 0.911× | 0.994× |
| tn / d128_in_proj | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4` | 16.67 | — | —× | 16.63 | 0.997× | 10.02 | 0.601× | 1.007× |
| tn / d128_out_proj | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3` | 10.15 | — | —× | 10.22 | 1.006× | 9.79 | 0.964× | 1.006× |
| tn / d768_in_proj | `tn_sm89_tf32_pre_rna_m96n192_w3x4_bk32_s2` | 143.91 | — | —× | 159.26 | 1.107× | 133.21 | 0.926× | 1.006× |
| tn / d768_out_proj | `tn_sm89_tf32_pre_rna_m96n96_bk32_s3` | 82.30 | — | —× | 94.10 | 1.143× | 82.36 | 1.001× | 1.003× |
| tn / large_deep | `tn_sm89_tf32_m192n192_w3x4_bk32_s2` | 390.77 | — | —× | 601.88 | 1.540× | 515.40 | 1.319× | 0.922× |
| tn / prism_in_proj | `tn_sm89_tf32_pre_rna_m64n96_bk32_s2` | 149.86 | — | —× | 140.43 | 0.937× | 111.10 | 0.741× | 0.903× |
| tn / underfill | `tn_sm80_mma_tf32_m16n32_bk32_s4` | 10.35 | — | —× | 10.43 | 1.008× | 7.81 | 0.755× | 1.001× |
| nt / d128_in_proj | `nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3` | 12.03 | — | —× | 12.16 | 1.011× | 9.33 | 0.776× | 1.009× |
| nt / d128_out_proj | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 7.91 | — | —× | 7.90 | 0.999× | 5.04 | 0.638× | 1.022× |
| nt / d768_in_proj | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 147.88 | — | —× | 115.40 | 0.780× | 151.88 | 1.027× | 0.771× |
| nt / d768_out_proj | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 73.90 | — | —× | 64.98 | 0.879× | 76.23 | 1.032× | 0.876× |
| nt / large_deep | `nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2` | 667.52 | — | —× | 549.05 | 0.823× | 542.91 | 0.813× | 0.785× |
| nt / prism_in_proj | `nt_sm89_tf32_rna_m144n96_w3x4_bk32_s2` | 116.76 | — | —× | 134.54 | 1.152× | 88.81 | 0.761× | 0.867× |
| nt / underfill | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 10.12 | — | —× | 10.13 | 1.001× | 8.00 | 0.790× | 0.999× |
### triad — BF16 / Fast

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nn / d128_in_proj | `nn_tc64_bf16` | 5.77 | 5.76 | 0.998× | 5.63 | 0.976× | 4.80 | 0.833× | 0.999× |
| nn / d128_out_proj | `nn_sm89_m16n64_bk64_s4_bf16` | 4.61 | 6.59 | 1.431× | 6.68 | 1.450× | 5.40 | 1.173× | 1.007× |
| nn / d768_in_proj | `nn_sm89_m128n128_bk64_s3_bf16` | 70.80 | 70.64 | 0.998× | 62.23 | 0.879× | 82.43 | 1.164× | 0.926× |
| nn / d768_out_proj | `nn_sm89_m128n128_bk64_s3_bf16` | 37.93 | 37.95 | 1.001× | 37.95 | 1.001× | 48.14 | 1.269× | 0.985× |
| nn / large_deep | `nn_tc_bf16` | 370.77 | 372.96 | 1.006× | — | —× | 258.17 | 0.696× | —× |
| nn / prism_in_proj | `nn_sm89_m128n128_bk64_s3_bf16` | 62.98 | 64.52 | 1.024× | 59.68 | 0.948× | 75.15 | 1.193× | 0.978× |
| nn / underfill | `nn_tc16_bf16` | 6.62 | 6.62 | 1.000× | — | —× | 7.29 | 1.102× | —× |
| tn / d128_in_proj | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16` | 6.90 | 18.35 | 2.660× | 18.34 | 2.658× | 7.90 | 1.145× | 1.003× |
| tn / d128_out_proj | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16` | 5.89 | 18.18 | 3.084× | 12.27 | 2.082× | 7.17 | 1.216× | 0.997× |
| tn / d768_in_proj | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_bf16` | 89.76 | 89.06 | 0.992× | 86.78 | 0.967× | 89.27 | 0.995× | 1.011× |
| tn / d768_out_proj | `tn_sm89_relay_m64n64_bk64_s3_bf16` | 48.97 | 55.21 | 1.127× | 55.23 | 1.128× | 50.42 | 1.030× | 1.005× |
| tn / large_deep | `tn_tc64_streamk_bf16` | 347.42 | 347.29 | 1.000× | — | —× | 333.90 | 0.961× | —× |
| tn / prism_in_proj | `tn_sm89_m64n64_bk64_s2_compact_bxor_bf16` | 77.70 | 77.62 | 0.999× | 77.59 | 0.999× | 54.58 | 0.702× | 1.006× |
| tn / underfill | `tn_tc64_bf16` | 8.87 | 8.92 | 1.006× | — | —× | 5.52 | 0.622× | —× |
| nt / d128_in_proj | `nt_sm89_m16n64_bk64_s4_bf16` | 6.08 | 9.32 | 1.533× | 9.37 | 1.541× | 7.62 | 1.253× | 1.003× |
| nt / d128_out_proj | `nt_sm89_m16n64_bk64_s4_bf16` | 4.54 | 5.34 | 1.175× | 5.39 | 1.187× | 4.66 | 1.026× | 1.012× |
| nt / d768_in_proj | `nt_sm89_m128n128_bk64_s3_bxor_bf16` | 71.09 | 71.07 | 1.000× | 70.98 | 0.998× | 78.38 | 1.103× | 0.991× |
| nt / d768_out_proj | `nt_sm89_m96n128_bk64_s3_bf16` | 41.03 | 40.87 | 0.996× | 36.55 | 0.891× | 39.36 | 0.959× | 0.988× |
| nt / large_deep | `nt_tc_bf16` | 397.00 | 400.19 | 1.008× | — | —× | 260.27 | 0.656× | —× |
| nt / prism_in_proj | `nt_sm89_m128n128_bk64_s3_bxor_bf16` | 50.61 | 50.89 | 1.006× | 48.65 | 0.961× | 60.26 | 1.191× | 1.045× |
| nt / underfill | `nt_tc64_bf16` | 7.82 | 7.84 | 1.003× | — | —× | 6.55 | 0.837× | —× |
### triad — F16 / Fast

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nn / d128_in_proj | `nn_tc64_f16` | 5.82 | 5.72 | 0.983× | 5.63 | 0.968× | 5.32 | 0.915× | 1.005× |
| nn / d128_out_proj | `nn_sm89_m16n64_bk64_s4_f16` | 4.67 | 6.63 | 1.418× | 6.68 | 1.430× | 4.74 | 1.015× | 0.996× |
| nn / d768_in_proj | `nn_sm89_m128n128_bk64_s3_f16` | 82.14 | 82.78 | 1.008× | 62.23 | 0.758× | 74.52 | 0.907× | 0.795× |
| nn / d768_out_proj | `nn_sm89_m128n128_bk64_s3_f16` | 38.86 | 40.39 | 1.039× | 37.99 | 0.978× | 47.38 | 1.219× | 0.994× |
| nn / large_deep | `nn_tc_f16` | 378.38 | 376.40 | 0.995× | — | —× | 265.27 | 0.701× | —× |
| nn / prism_in_proj | `nn_sm89_m128n128_bk64_s3_f16` | 74.01 | 74.28 | 1.004× | 59.79 | 0.808× | 82.36 | 1.113× | 0.891× |
| nn / underfill | `nn_tc16_f16` | 6.62 | 6.60 | 0.998× | — | —× | 5.75 | 0.869× | —× |
| tn / d128_in_proj | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_f16` | 6.93 | 18.37 | 2.650× | 18.34 | 2.646× | 7.98 | 1.151× | 0.986× |
| tn / d128_out_proj | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_f16` | 5.89 | 18.23 | 3.092× | 12.27 | 2.081× | 7.50 | 1.272× | 0.993× |
| tn / d768_in_proj | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_f16` | 102.49 | 102.44 | 1.000× | 86.68 | 0.846× | 99.97 | 0.975× | 0.898× |
| tn / d768_out_proj | `tn_sm89_relay_m64n64_bk64_s3_f16` | 56.38 | 57.83 | 1.026× | 56.14 | 0.996× | 55.41 | 0.983× | 0.914× |
| tn / large_deep | `tn_tc64_streamk_f16` | 392.98 | 394.52 | 1.004× | — | —× | 320.82 | 0.816× | —× |
| tn / prism_in_proj | `tn_sm89_m64n64_bk64_s2_compact_bxor_f16` | 88.49 | 88.15 | 0.996× | 77.67 | 0.878× | 58.72 | 0.664× | 0.935× |
| tn / underfill | `tn_tc64_f16` | 8.94 | 8.87 | 0.992× | — | —× | 5.55 | 0.621× | —× |
| nt / d128_in_proj | `nt_sm89_m16n64_bk64_s4_f16` | 6.06 | 9.40 | 1.551× | 9.37 | 1.546× | 5.88 | 0.971× | 0.989× |
| nt / d128_out_proj | `nt_sm89_m16n64_bk64_s4_f16` | 4.57 | 5.34 | 1.168× | 5.39 | 1.178× | 4.50 | 0.984× | 1.000× |
| nt / d768_in_proj | `nt_sm89_m128n128_bk64_s3_bxor_f16` | 83.67 | 84.02 | 1.004× | 71.01 | 0.849× | 76.41 | 0.913× | 0.953× |
| nt / d768_out_proj | `nt_sm89_m96n128_bk64_s3_f16` | 43.33 | 42.95 | 0.991× | 36.55 | 0.844× | 41.29 | 0.953× | 0.965× |
| nt / large_deep | `nt_tc_f16` | 432.35 | 440.82 | 1.020× | — | —× | 285.81 | 0.661× | —× |
| nt / prism_in_proj | `nt_sm89_m128n128_bk64_s3_bxor_f16` | 58.06 | 57.62 | 0.992× | 48.66 | 0.838× | 96.77 | 1.667× | 0.998× |
| nt / underfill | `nt_tc64_f16` | 7.88 | 7.83 | 0.995× | — | —× | 5.34 | 0.678× | —× |
### inference — exact F32 / Fast TF32

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hot_a | `nn_sm89_f32_n64_copyplan` | 236.69 | 240.44 | 1.016× | 211.82 | 0.895× | 129.57 | 0.547× | 0.903× |
| hot_b | `nn_sm89_f32_n64_copyplan` | 616.45 | 605.10 | 0.982× | 529.62 | 0.859× | 279.02 | 0.453× | 0.860× |
| hot_c | `nn_sm89_m112n128_bk32_s3_f32` | 260.53 | 329.91 | 1.266× | 241.47 | 0.927× | 95.55 | 0.367× | 0.860× |
| hot_d | `nn_sm89_f32_n64_copyplan` | 215.63 | 213.81 | 0.992× | 219.34 | 1.017× | 106.01 | 0.492× | 1.018× |
| hot_e | `nn_sm89_f32_n64_copyplan` | 248.73 | 244.71 | 0.984× | 222.65 | 0.895× | 111.96 | 0.450× | 0.905× |
### inference — exact F32 / Pedantic

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hot_a | `nn_sm89_f32_n64_copyplan` | 236.69 | 240.44 | 1.016× | 270.54 | 1.143× | 261.35 | 1.104× | 1.146× |
| hot_b | `nn_sm89_f32_n64_copyplan` | 616.45 | 605.10 | 0.982× | 698.09 | 1.132× | 627.33 | 1.018× | 1.134× |
| hot_c | `nn_sm89_m112n128_bk32_s3_f32` | 260.53 | 329.91 | 1.266× | 283.82 | 1.089× | 233.67 | 0.897× | 0.999× |
| hot_d | `nn_sm89_f32_n64_copyplan` | 215.63 | 213.81 | 0.992× | 221.01 | 1.025× | 256.24 | 1.188× | 1.030× |
| hot_e | `nn_sm89_f32_n64_copyplan` | 248.73 | 244.71 | 0.984× | 266.47 | 1.071× | 248.99 | 1.001× | 1.085× |
### inference — TF32 / Fast TF32

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hot_a | `nn_rna_wide_tf32_m128n128_bk32_s3` | 120.75 | — | —× | 111.29 | 0.922× | 111.17 | 0.921× | 0.927× |
| hot_b | `nn_rna_wide_tf32_m128n128_bk32_s3` | 283.06 | — | —× | 222.55 | 0.786× | 258.82 | 0.914× | 0.786× |
| hot_c | `nn_rna_wide_tf32_m128n128_bk32_s3` | 120.73 | — | —× | 102.12 | 0.846× | 90.88 | 0.753× | 0.857× |
| hot_d | `nn_sm89_m64n288_bk16_s2_tf32` | 93.18 | — | —× | 125.34 | 1.345× | 99.66 | 1.070× | 0.996× |
| hot_e | `nn_sm89_rna_tf32_m128n96_bk32_s3` | 115.32 | — | —× | 101.49 | 0.880× | 104.08 | 0.903× | 0.876× |
### inference — BF16 / Fast

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hot_a | `nn_sm89_tc128_pipeline_bf16` | 67.42 | 67.36 | 0.999× | 64.17 | 0.952× | 75.91 | 1.126× | 0.987× |
| hot_b | `nn_sm89_tc128_s3_bf16` | 132.91 | 129.51 | 0.974× | 127.89 | 0.962× | 109.33 | 0.823× | 0.962× |
| hot_c | `nn_sm89_tc128_pipeline_bf16` | 54.52 | 54.67 | 1.003× | 54.06 | 0.991× | 75.21 | 1.379× | 0.994× |
| hot_d | `nn_sm89_m128n144_bk32_s2_vec_bf16` | 65.88 | 65.70 | 0.997× | 65.64 | 0.996× | 61.47 | 0.933× | 0.983× |
| hot_e | `nn_sm89_m128n96_bk64_s2_vec_bf16` | 59.35 | 60.65 | 1.022× | 60.62 | 1.021× | 68.29 | 1.151× | 1.007× |
### inference — F16 / Fast

| case | kernel | new µs | adapter 0.7.1 µs | adapter 0.7.1/new | page 0.7.1 µs | page 0.7.1/new | cuBLAS µs | cuBLAS/new | vendor page/new |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hot_a | `nn_sm89_tc128_pipeline_f16` | 76.68 | 74.23 | 0.968× | 66.62 | 0.869× | 81.15 | 1.058× | 0.926× |
| hot_b | `nn_sm89_tc128_s3_f16` | 151.30 | 149.60 | 0.989× | 136.40 | 0.902× | 130.10 | 0.860× | 0.907× |
| hot_c | `nn_sm89_tc128_pipeline_f16` | 61.93 | 61.67 | 0.996× | 55.64 | 0.898× | 80.95 | 1.307× | 0.931× |
| hot_d | `nn_sm89_m64n64_bk64_s3_f16` | 72.38 | 71.62 | 0.989× | 69.51 | 0.960× | 70.44 | 0.973× | 0.967× |
| hot_e | `nn_sm89_m128n96_bk64_s2_vec_f16` | 68.52 | 64.99 | 0.949× | 59.78 | 0.872× | 60.73 | 0.886× | 1.020× |
