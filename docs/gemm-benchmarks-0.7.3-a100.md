# GEMM benchmarks - 0.7.3, A100 SXM4 40 GB

Measured on September 18, 2026 with the final 0.7.3 tree, one run of the
kernel-level adapter on a rented board. The bits of every route are the
0.7.2 bits on this board (bit ledger 263 of 263 unchanged, acceptance
capture 89 of 89 identical), and the training-step rows are on the
[Mamba-1](mamba1-benchmarks.md) and [Mamba-3](mamba3-benchmarks.md) pages.

## Measurement scope

| item | value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, SM80 (CC 8.0), 108 SMs |
| measured GPU UUID | `GPU-1842567e-ae39-1f51-cce1-aa2d42583d91` |
| driver / toolkit | 595.58.03 / CUDA 13.2, NVRTC 13.2.78 |
| host compiler | Rust 1.98.1, release profile |
| GEMM mode | `Deterministic`; the automatic route, no forced candidate |
| storage precisions | exact F32, TF32, BF16, F16 |
| path | eager; every cell times three arms in one process |

Each cell times the deterministic route, cuBLAS in the tree's fast
setting (TF32 handle math) and cuBLAS in the tree's pedantic setting in
windows of calibrated launch counts, 21 windows, the arm order mirrored
from window to window; the number is the lower median per arm. The row's
`route` is the symbol the dispatcher recorded for the launch. This board
has no kernels of its own: every route is the common tier, admitted at
first use by the bit proof against the portable reference, or the scalar
tier where no candidate was admitted.

`fast/det` and `pedantic/det` divide the vendor time by ours on this
board: above 1.0 the deterministic route is faster than that vendor
setting. There is no earlier per-cell record of this board; the 0.7.2
tree on it had no route this adapter can time.

Two common routes do not serve here, and the context says so once:
`nt_sm89_m96n128_bk64_s3_bf16` is declined before its proof because its
96x128 tile takes more waves on 108 multiprocessors than the reference
tile it would replace, and `tn_sm89_tf32_m192n192_w3x4_bk32_s2` is
declined by the proof itself: its output words differ from the reference
route of the same numeric contract on this board, which is exactly the
case the first-use proof exists for.


Eager microseconds, p50 over 21 windows, one run, kernel cache on. `det` is the deterministic route the dispatcher chose on this board; `fast` is cuBLAS with TF32 handle math and `pedantic` is cuBLAS pedantic on the same board, the vendor controls. `fast/det` and `pedantic/det` divide the vendor time by ours: above 1.0 the deterministic route is faster.

## triad - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_tc64_bf16` | 9.74 | 8.17 | 26.19 | 0.839 | 2.690 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm89_m16n64_bk64_s4_bf16` | 6.76 | 7.43 | 21.79 | 1.100 | 3.224 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_m128n128_bk64_s3_bf16` | 73.96 | 48.86 | 690.56 | 0.661 | 9.337 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_m128n128_bk64_s3_bf16` | 36.11 | 25.95 | 389.73 | 0.718 | 10.792 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_tc_bf16` | 368.73 | 159.62 | 2730.88 | 0.433 | 7.406 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_m128n128_bk64_s3_bf16` | 69.09 | 45.01 | 520.45 | 0.651 | 7.532 |
| underfill | nn | 256 x 512 x 384 | `nn_tc64_bf16` | 11.41 | 7.45 | 39.69 | 0.653 | 3.478 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm89_m16n64_bk64_s4_bf16` | 9.32 | 7.60 | 46.30 | 0.815 | 4.970 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm89_m16n64_bk64_s4_bf16` | 6.24 | 7.53 | 15.19 | 1.206 | 2.432 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm89_m128n128_bk64_s3_bxor_bf16` | 70.46 | 44.64 | 745.60 | 0.634 | 10.582 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_tc_bf16` | 55.05 | 26.15 | 356.17 | 0.475 | 6.469 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_tc_bf16` | 379.53 | 173.06 | 2509.70 | 0.456 | 6.613 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm89_m128n128_bk64_s3_bxor_bf16` | 88.95 | 54.22 | 579.58 | 0.609 | 6.516 |
| underfill | nt | 256 x 512 x 384 | `nt_tc64_bf16` | 9.21 | 7.58 | 36.11 | 0.822 | 3.919 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16` | 9.95 | 9.12 | 69.02 | 0.916 | 6.936 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16` | 7.15 | 11.49 | 69.04 | 1.608 | 9.659 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_bf16` | 114.31 | 60.76 | 683.52 | 0.532 | 5.980 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_relay_m64n64_bk64_s3_bf16` | 68.63 | 32.50 | 347.60 | 0.474 | 5.065 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_tc64_streamk_bf16` | 462.96 | 219.48 | 2497.66 | 0.474 | 5.395 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_bf16` | 98.64 | 46.82 | 553.98 | 0.475 | 5.616 |
| underfill | tn | 256 x 512 x 384 | `tn_tc64_bf16` | 9.88 | 7.69 | 22.60 | 0.778 | 2.287 |

Geometric mean fast/det over 21 cells: 0.685.

## triad - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_tc64_f16` | 9.54 | 9.11 | 28.27 | 0.954 | 2.963 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm89_m16n64_bk64_s4_f16` | 6.67 | 8.34 | 12.03 | 1.250 | 1.804 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_m128n128_bk64_s3_f16` | 73.67 | 60.86 | 549.89 | 0.826 | 7.464 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_m128n128_bk64_s3_f16` | 35.90 | 36.44 | 283.14 | 1.015 | 7.888 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_tc_f16` | 380.65 | 164.91 | 2242.94 | 0.433 | 5.892 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_m128n128_bk64_s3_f16` | 68.12 | 50.87 | 412.16 | 0.747 | 6.050 |
| underfill | nn | 256 x 512 x 384 | `nn_tc64_f16` | 11.39 | 9.26 | 15.79 | 0.813 | 1.386 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm89_m16n64_bk64_s4_f16` | 9.25 | 8.38 | 19.98 | 0.907 | 2.161 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm89_m16n64_bk64_s4_f16` | 6.07 | 8.84 | 11.93 | 1.456 | 1.964 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm89_m128n128_bk64_s3_bxor_f16` | 71.32 | 53.02 | 609.66 | 0.743 | 8.548 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_tc_f16` | 55.75 | 30.25 | 302.00 | 0.543 | 5.417 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_tc_f16` | 383.63 | 181.38 | 2188.03 | 0.473 | 5.704 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm89_m128n128_bk64_s3_bxor_f16` | 89.32 | 55.69 | 426.67 | 0.624 | 4.777 |
| underfill | nt | 256 x 512 x 384 | `nt_tc64_f16` | 9.19 | 14.06 | 16.43 | 1.530 | 1.788 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_f16` | 9.89 | 9.02 | 18.56 | 0.912 | 1.877 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_f16` | 7.01 | 11.15 | 15.18 | 1.590 | 2.166 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_f16` | 119.60 | 54.78 | 556.67 | 0.458 | 4.654 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_relay_m64n64_bk64_s3_f16` | 70.25 | 32.56 | 278.60 | 0.464 | 3.966 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_tc64_streamk_f16` | 468.65 | 178.84 | 2145.28 | 0.382 | 4.578 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_f16` | 100.50 | 56.70 | 447.26 | 0.564 | 4.450 |
| underfill | tn | 256 x 512 x 384 | `tn_tc64_f16` | 9.84 | 8.48 | 12.79 | 0.862 | 1.300 |

Geometric mean fast/det over 21 cells: 0.765.

## triad - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_splitk32_partial+splitk_reduce` | 24.35 | 10.95 | 18.87 | 0.450 | 0.775 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_splitk32_partial+splitk_reduce` | 15.38 | 10.65 | 15.75 | 0.693 | 1.024 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_f32_n64_copyplan` | 594.05 | 95.90 | 578.05 | 0.161 | 0.973 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_f32_n64_copyplan` | 321.77 | 47.35 | 281.97 | 0.147 | 0.876 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm89_f32_n64_copyplan` | 2331.14 | 319.88 | 2264.96 | 0.137 | 0.972 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_f32_n64_copyplan` | 422.19 | 85.48 | 419.43 | 0.202 | 0.993 |
| underfill | nn | 256 x 512 x 384 | `nn_splitk32_partial+splitk_reduce` | 19.29 | 13.72 | 15.00 | 0.712 | 0.778 |
| d128_in_proj | nt | 1024 x 128 x 512 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 25.69 | 12.13 | 20.30 | 0.472 | 0.790 |
| d128_out_proj | nt | 1024 x 256 x 128 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 18.57 | 8.61 | 12.50 | 0.464 | 0.673 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 663.30 | 88.13 | 580.61 | 0.133 | 0.875 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 318.70 | 48.53 | 300.69 | 0.152 | 0.943 |
| large_deep | nt | 4096 x 3072 x 1536 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 2278.27 | 337.58 | 2241.28 | 0.148 | 0.984 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 574.72 | 110.76 | 438.84 | 0.193 | 0.764 |
| underfill | nt | 256 x 512 x 384 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 22.78 | 8.54 | 16.87 | 0.375 | 0.740 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_splitm_partial_aligned+splitm_reduce` | 55.15 | 11.78 | 21.32 | 0.214 | 0.386 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 18.90 | 11.46 | 15.04 | 0.607 | 0.796 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `transpose_f32_32x16_d768+tn_sm89_f32_n64_dual_chunk_fused_finalize` | 647.42 | 106.15 | 580.74 | 0.164 | 0.897 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_splitm_partial_aligned+splitm_reduce` | 339.12 | 62.38 | 274.50 | 0.184 | 0.809 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_aligned` | 2445.57 | 345.60 | 2156.42 | 0.141 | 0.882 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_splitm_partial_aligned+splitm_reduce` | 625.54 | 104.10 | 452.61 | 0.166 | 0.724 |
| underfill | tn | 256 x 512 x 384 | `tn_m16n16_bk16_s2_splitm16` | 26.46 | 8.74 | 13.53 | 0.330 | 0.511 |

Geometric mean fast/det over 21 cells: 0.249.

## triad - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_sm80_mma_tf32_m64n64_bk32_s2` | 16.19 | 10.98 | 23.98 | 0.678 | 1.481 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm80_mma_tf32_m16n32_bk32_s4` | 10.49 | 10.58 | 15.73 | 1.009 | 1.500 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 166.67 | 96.38 | 586.24 | 0.578 | 3.517 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3` | 126.94 | 47.12 | 281.89 | 0.371 | 2.221 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 604.03 | 326.18 | 2338.94 | 0.540 | 3.872 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct` | 142.41 | 85.45 | 419.94 | 0.600 | 2.949 |
| underfill | nn | 256 x 512 x 384 | `nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4` | 16.03 | 13.69 | 15.11 | 0.854 | 0.942 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3` | 15.99 | 12.08 | 20.37 | 0.755 | 1.273 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 8.98 | 8.53 | 12.60 | 0.950 | 1.404 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 253.53 | 87.95 | 580.74 | 0.347 | 2.291 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 100.98 | 49.45 | 305.23 | 0.490 | 3.023 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2` | 555.90 | 337.66 | 2249.86 | 0.607 | 4.047 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm89_tf32_rna_m144n96_w3x4_bk32_s2` | 216.87 | 108.88 | 436.34 | 0.502 | 2.012 |
| underfill | nt | 256 x 512 x 384 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 14.10 | 8.56 | 16.98 | 0.608 | 1.204 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4` | 20.68 | 11.77 | 21.57 | 0.569 | 1.043 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3` | 16.04 | 11.36 | 15.18 | 0.708 | 0.947 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm89_tf32_pre_rna_m96n192_w3x4_bk32_s2` | 307.20 | 106.39 | 573.95 | 0.346 | 1.868 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_tf32_pre_rna_m96n96_bk32_s3` | 138.91 | 62.55 | 275.32 | 0.450 | 1.982 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm80_mma_tf32_m64n64_bk32_s2` | 1012.22 | 356.95 | 2242.30 | 0.353 | 2.215 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm89_tf32_pre_rna_m64n96_bk32_s2` | 241.91 | 105.03 | 457.05 | 0.434 | 1.889 |
| underfill | tn | 256 x 512 x 384 | `tn_sm80_mma_tf32_m16n32_bk32_s4` | 12.71 | 8.61 | 13.91 | 0.677 | 1.094 |

Geometric mean fast/det over 21 cells: 0.564.

## inference - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_tc128_pipeline_bf16` | 91.75 | 54.67 | 628.48 | 0.596 | 6.850 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_tc128_s3_bf16` | 129.76 | 81.76 | 1154.69 | 0.630 | 8.899 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_tc128_pipeline_bf16` | 93.78 | 53.98 | 575.62 | 0.576 | 6.138 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_m128n144_bk32_s2_vec_bf16` | 87.33 | 47.64 | 505.60 | 0.546 | 5.790 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_m128n96_bk64_s2_vec_bf16` | 105.12 | 34.56 | 575.36 | 0.329 | 5.473 |

Geometric mean fast/det over 5 cells: 0.522.

## inference - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_tc128_pipeline_f16` | 91.56 | 54.63 | 417.02 | 0.597 | 4.554 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_tc128_s3_f16` | 133.02 | 83.44 | 960.13 | 0.627 | 7.218 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_tc128_pipeline_f16` | 94.71 | 49.42 | 442.71 | 0.522 | 4.674 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_tc128_swizzle_f16` | 64.81 | 39.83 | 412.16 | 0.615 | 6.360 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_m128n96_bk64_s2_vec_f16` | 106.04 | 42.55 | 417.18 | 0.401 | 3.934 |

Geometric mean fast/det over 5 cells: 0.545.

## inference - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_f32_n64_copyplan` | 513.28 | 102.40 | 509.18 | 0.200 | 0.992 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_f32_n64_copyplan` | 987.65 | 164.37 | 961.92 | 0.166 | 0.974 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_m112n128_bk32_s3_f32` | 733.44 | 108.24 | 422.19 | 0.148 | 0.576 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_f32_n64_copyplan` | 439.19 | 73.82 | 428.65 | 0.168 | 0.976 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_f32_n64_copyplan` | 484.35 | 68.13 | 419.74 | 0.141 | 0.867 |

Geometric mean fast/det over 5 cells: 0.163.

## inference - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 175.42 | 102.45 | 509.70 | 0.584 | 2.906 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 310.13 | 165.31 | 968.70 | 0.533 | 3.124 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 202.96 | 108.34 | 422.80 | 0.534 | 2.083 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_m64n288_bk16_s2_tf32` | 142.94 | 74.39 | 434.38 | 0.520 | 3.039 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_rna_tf32_m128n96_bk32_s3` | 207.77 | 67.48 | 413.18 | 0.325 | 1.989 |

Geometric mean fast/det over 5 cells: 0.489.

