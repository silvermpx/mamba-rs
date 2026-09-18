# GEMM benchmarks - 0.7.3, RTX 5090

Measured on September 18, 2026 with the final 0.7.3 tree, one run of the
kernel-level adapter on a rented board. The bits of every route are the
0.7.2 bits on this board (bit ledger 263 of 263 unchanged, acceptance
capture 89 of 89 identical), and the training-step rows are on the
[Mamba-1](mamba1-benchmarks.md) and [Mamba-3](mamba3-benchmarks.md) pages.

## Measurement scope

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, SM120 (CC 12.0), 170 SMs |
| measured GPU UUID | `GPU-c563a487-19d0-b99b-2910-293badfa6a25` |
| driver / toolkit | 610.43.02 / CUDA 13.2, NVRTC 13.2.78 |
| host compiler | Rust 1.98.1, release profile |
| GEMM mode | `Deterministic`; the automatic route, no forced candidate |
| storage precisions | exact F32, TF32, BF16, F16 |
| path | eager; every cell times three arms in one process |

Each cell times the deterministic route, cuBLAS in the tree's fast
setting (TF32 handle math) and cuBLAS in the tree's pedantic setting in
windows of calibrated launch counts, 21 windows, the arm order mirrored
from window to window; the number is the lower median per arm. The row's
`route` is the symbol the dispatcher recorded for the launch: the board's
own SM120 kernels on the shapes its table names, the common tier
elsewhere, the scalar tier where no candidate was admitted.

`fast/det` and `pedantic/det` divide the vendor time by ours on this
board: above 1.0 the deterministic route is faster than that vendor
setting. There is no earlier per-cell record of this board to compare
against; the 0.7.2 tree on it had no route this adapter can time.

What this board leaves out of the common tier, said aloud at context
creation: ptxas for `compute_120` allocates more registers than the Ada
gates of eleven common kernels allow, so the six half tiles of the
`nn_sm89_m128n128_bk64_s3` family, the TF32 joint tiles
`nn_sm89_tf32_addhalf_m128n96_bk32_s3` (both variants),
`nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3`, `tn_sm89_tf32_pre_rna_m128n96_bk32_s3`,
`tn_sm89_tf32_pre_rna_m64n64_bk32_s3` and the `tn_sm89_tf32_pre_rna_transpose_32x32`
helper, and the TF32 finalist `nt_sm89_mma_tf32_compact8_m128n64_bk32_s2`
are excluded here while their siblings serve; the inference overlay is
not composed on CC 12.x at all. Each of these is a candidate for the
board's own tuning pass, where the gate would be measured for this
target rather than inherited from the Ada.


Eager microseconds, p50 over 21 windows, one run, kernel cache on. `det` is the deterministic route the dispatcher chose on this board; `fast` is cuBLAS with TF32 handle math and `pedantic` is cuBLAS pedantic on the same board, the vendor controls. `fast/det` and `pedantic/det` divide the vendor time by ours: above 1.0 the deterministic route is faster.

## triad - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.78 | 4.31 | 6.87 | 0.291 | 0.465 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.79 | 4.00 | 9.81 | 0.270 | 0.663 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm120_tma_128x64_bk32_s2_bf16` | 48.63 | 53.03 | 224.28 | 1.091 | 4.612 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 30.86 | 36.67 | 120.62 | 1.188 | 3.909 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm120_tma_128x64_bk32_s2_bf16` | 187.17 | 189.97 | 902.80 | 1.015 | 4.823 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm120_tma_128x64_bk64_s2_bf16` | 38.56 | 43.35 | 159.76 | 1.124 | 4.143 |
| underfill | nn | 256 x 512 x 384 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.83 | 4.25 | 17.71 | 0.287 | 1.195 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm120_tma_64x64_bk64_s3_bf16` | 14.88 | 3.94 | 24.36 | 0.265 | 1.637 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm120_tma_64x64_bk32_s2_bf16` | 14.91 | 3.95 | 7.84 | 0.265 | 0.525 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm120_tma_64x64_bk64_s2_bf16` | 59.45 | 57.44 | 234.95 | 0.966 | 3.952 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm120_tma_64x64_bk64_s2_bf16` | 26.91 | 37.73 | 119.18 | 1.402 | 4.429 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm120_tma_128x128_bk32_s3_bf16` | 178.69 | 185.49 | 818.85 | 1.038 | 4.583 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm120_tma_128x128_bk64_s3_bf16` | 47.46 | 47.02 | 154.53 | 0.991 | 3.256 |
| underfill | nt | 256 x 512 x 384 | `nt_sm120_tma_64x64_bk64_s2_bf16` | 14.92 | 3.96 | 18.94 | 0.265 | 1.270 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm120_tma_64x64_bk64_s3_streamk_bf16` | 14.98 | 6.00 | 32.24 | 0.401 | 2.152 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm120_tma_64x64_bk64_s3_streamk_bf16` | 15.05 | 5.97 | 32.22 | 0.397 | 2.141 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm120_tma_64x128_bk32_s3_bf16` | 49.24 | 50.74 | 192.44 | 1.030 | 3.908 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm120_tma_64x128_bk32_s3_bf16` | 26.21 | 26.72 | 101.23 | 1.019 | 3.862 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm120_tma_128x128_bk32_s3_bf16` | 190.33 | 193.55 | 698.89 | 1.017 | 3.672 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm120_tma_64x64_bk64_s3_streamk_bf16` | 50.73 | 46.93 | 214.25 | 0.925 | 4.223 |
| underfill | tn | 256 x 512 x 384 | `tn_sm120_tma_64x64_bk32_s3_bf16` | 14.96 | 4.03 | 10.49 | 0.269 | 0.701 |

Geometric mean fast/det over 21 cells: 0.615.

## triad - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_sm120_tma_64x64_bk64_s2_f16` | 14.79 | 4.51 | 9.38 | 0.305 | 0.634 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm120_tma_64x64_bk64_s2_f16` | 14.79 | 4.28 | 15.29 | 0.289 | 1.033 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm120_tma_128x64_bk32_s2_f16` | 49.56 | 54.37 | 248.57 | 1.097 | 5.015 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm120_tma_64x64_bk64_s2_f16` | 30.66 | 36.50 | 130.26 | 1.190 | 4.248 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm120_tma_128x64_bk32_s2_f16` | 193.78 | 198.21 | 992.24 | 1.023 | 5.121 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm120_tma_128x64_bk64_s2_f16` | 38.94 | 43.60 | 178.37 | 1.120 | 4.581 |
| underfill | nn | 256 x 512 x 384 | `nn_sm120_tma_64x64_bk64_s2_f16` | 14.84 | 4.27 | 29.05 | 0.288 | 1.957 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm120_tma_64x64_bk64_s3_f16` | 14.88 | 4.24 | 24.36 | 0.285 | 1.637 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm120_tma_64x64_bk32_s2_f16` | 14.89 | 4.06 | 7.83 | 0.273 | 0.526 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm120_tma_64x64_bk64_s2_f16` | 60.98 | 57.44 | 236.50 | 0.942 | 3.878 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm120_tma_64x64_bk64_s2_f16` | 27.35 | 37.80 | 119.73 | 1.382 | 4.378 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm120_tma_128x128_bk32_s3_f16` | 181.21 | 191.40 | 843.12 | 1.056 | 4.653 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm120_tma_128x128_bk64_s3_f16` | 47.56 | 46.94 | 155.97 | 0.987 | 3.279 |
| underfill | nt | 256 x 512 x 384 | `nt_sm120_tma_64x64_bk64_s2_f16` | 14.92 | 3.99 | 18.92 | 0.268 | 1.268 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm120_tma_64x64_bk64_s3_streamk_f16` | 14.94 | 5.92 | 32.26 | 0.396 | 2.160 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm120_tma_64x64_bk64_s3_streamk_f16` | 15.06 | 5.90 | 32.22 | 0.392 | 2.140 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm120_tma_64x128_bk32_s3_f16` | 49.54 | 50.85 | 195.66 | 1.026 | 3.949 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm120_tma_64x128_bk32_s3_f16` | 26.23 | 26.73 | 103.15 | 1.019 | 3.932 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm120_tma_128x128_bk32_s3_f16` | 190.42 | 196.67 | 719.14 | 1.033 | 3.777 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm120_tma_64x64_bk64_s3_streamk_f16` | 50.73 | 47.04 | 214.28 | 0.927 | 4.224 |
| underfill | tn | 256 x 512 x 384 | `tn_sm120_tma_64x64_bk32_s3_f16` | 14.93 | 3.93 | 10.48 | 0.263 | 0.702 |

Geometric mean fast/det over 21 cells: 0.620.

## triad - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_splitk32_partial+splitk_reduce` | 8.53 | 4.47 | 5.28 | 0.524 | 0.619 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_splitk32_partial+splitk_reduce` | 5.15 | 4.48 | 6.15 | 0.870 | 1.193 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm120_tma_fma_m128n64_bk16_s2` | 145.10 | 114.52 | 172.18 | 0.789 | 1.187 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm120_tma_fma_m128n64_bk16_s2` | 81.56 | 72.12 | 88.15 | 0.884 | 1.081 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm120_tma_fma_m128n64_bk16_s2` | 588.26 | 436.94 | 569.90 | 0.743 | 0.969 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm120_tma_fma_m64n128_bk16_s2` | 115.46 | 81.57 | 131.92 | 0.706 | 1.143 |
| underfill | nn | 256 x 512 x 384 | `nn_splitk32_partial+splitk_reduce` | 6.53 | 7.26 | 6.44 | 1.111 | 0.985 |
| d128_in_proj | nt | 1024 x 128 x 512 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 8.61 | 7.41 | 7.70 | 0.860 | 0.894 |
| d128_out_proj | nt | 1024 x 256 x 128 | `transpose_f32_32x16_d768+nn_m64n64_bk16_s2` | 6.26 | 3.91 | 6.05 | 0.625 | 0.966 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm120_tma_fma_m128n64_bk16_s2_kvec` | 160.47 | 141.15 | 185.57 | 0.880 | 1.156 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm120_tma_fma_m64n128_bk16_s2` | 86.06 | 76.68 | 106.95 | 0.891 | 1.243 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm120_tma_fma_m128n64_bk16_s2_kvec` | 679.04 | 376.66 | 741.00 | 0.555 | 1.091 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm120_tma_fma_m128n64_bk16_s2_kvec` | 113.36 | 90.71 | 140.54 | 0.800 | 1.240 |
| underfill | nt | 256 x 512 x 384 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 8.07 | 5.94 | 8.47 | 0.736 | 1.050 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm89_f32_d128_in_m16n16_g8_s2_cg` | 10.81 | 5.99 | 6.17 | 0.554 | 0.571 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 7.54 | 5.96 | 5.67 | 0.791 | 0.752 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm120_tma_fma_m64n128_bk16_s2` | 148.28 | 98.29 | 148.34 | 0.663 | 1.000 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm120_tma_fma_m128n64_bk16_s2` | 79.58 | 51.17 | 86.65 | 0.643 | 1.089 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm120_tma_fma_m64n128_bk16_s2` | 578.10 | 383.09 | 590.78 | 0.663 | 1.022 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm120_tma_fma_m128n64_bk16_s2` | 108.04 | 81.42 | 118.74 | 0.754 | 1.099 |
| underfill | tn | 256 x 512 x 384 | `tn_m16n16_bk16_s2_splitm16` | 14.44 | 5.35 | 6.82 | 0.371 | 0.472 |

Geometric mean fast/det over 21 cells: 0.715.

## triad - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_sm80_mma_tf32_m64n64_bk32_s3` | 5.08 | 4.48 | 5.28 | 0.883 | 1.039 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm80_mma_tf32_m16n32_bk32_s4` | 3.87 | 4.54 | 6.21 | 1.173 | 1.606 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm120_tma_mma_tf32_m64n128_bk32_s2` | 104.05 | 114.63 | 172.10 | 1.102 | 1.654 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm120_tma_mma_tf32_m64n64_bk32_s2` | 66.40 | 71.95 | 88.10 | 1.084 | 1.327 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm120_tma_mma_tf32_m64n128_bk32_s2` | 391.80 | 425.89 | 564.34 | 1.087 | 1.440 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm120_tma_mma_tf32_m64n64_bk32_s2` | 79.86 | 81.68 | 131.65 | 1.023 | 1.648 |
| underfill | nn | 256 x 512 x 384 | `nn_splitk32_partial+splitk_reduce` | 6.52 | 7.28 | 6.43 | 1.115 | 0.987 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm80_mma_tf32_m16n16_bk32_s4` | 7.58 | 7.39 | 7.70 | 0.975 | 1.016 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 4.18 | 3.94 | 6.04 | 0.942 | 1.446 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm120_tma_mma_tf32_m64n64_bk32_s2` | 126.30 | 140.70 | 184.49 | 1.114 | 1.461 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm120_tma_mma_tf32_m64n64_bk32_s2` | 56.81 | 76.52 | 106.98 | 1.347 | 1.883 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm120_tma_mma_tf32_m64n128_bk32_s2` | 406.31 | 370.90 | 728.34 | 0.913 | 1.793 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm120_tma_mma_tf32_m64n64_bk32_s2` | 83.15 | 90.47 | 139.97 | 1.088 | 1.683 |
| underfill | nt | 256 x 512 x 384 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 5.39 | 5.93 | 8.47 | 1.101 | 1.572 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm80_mma_tf32_m16n32_bk32_s4` | 9.93 | 5.99 | 6.17 | 0.603 | 0.622 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm80_mma_tf32_m16n16_bk32_s4` | 9.64 | 5.96 | 5.71 | 0.619 | 0.592 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk` | 99.05 | 97.20 | 147.68 | 0.981 | 1.491 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk` | 54.84 | 51.05 | 86.28 | 0.931 | 1.573 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk` | 377.12 | 373.15 | 566.36 | 0.989 | 1.502 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm120_tma_mma_tf32_m64n128_bk32_s3_pair_streamk` | 76.72 | 78.51 | 116.89 | 1.023 | 1.524 |
| underfill | tn | 256 x 512 x 384 | `tn_sm80_mma_tf32_m16n32_bk32_s4` | 6.23 | 5.35 | 6.81 | 0.859 | 1.094 |

Geometric mean fast/det over 21 cells: 0.982.

## inference - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm120_tma_128x64_bk32_s3_bf16` | 38.54 | 43.78 | 160.38 | 1.136 | 4.162 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm120_tma_128x128_bk32_s3_bf16` | 73.54 | 77.85 | 348.95 | 1.059 | 4.745 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 40.35 | 46.24 | 157.37 | 1.146 | 3.900 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 38.29 | 40.03 | 155.74 | 1.045 | 4.067 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm120_tma_64x64_bk64_s2_bf16` | 44.47 | 53.52 | 179.41 | 1.203 | 4.034 |

Geometric mean fast/det over 5 cells: 1.116.

## inference - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm120_tma_128x64_bk32_s3_f16` | 39.10 | 43.87 | 177.88 | 1.122 | 4.550 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm120_tma_128x128_bk32_s3_f16` | 74.22 | 78.03 | 391.20 | 1.051 | 5.271 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm120_tma_64x64_bk64_s2_f16` | 40.98 | 46.12 | 169.96 | 1.126 | 4.148 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm120_tma_128x128_bk32_s3_f16` | 38.32 | 39.83 | 172.32 | 1.039 | 4.497 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm120_tma_64x64_bk64_s2_f16` | 44.47 | 53.36 | 193.51 | 1.200 | 4.352 |

Geometric mean fast/det over 5 cells: 1.106.

## inference - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm120_tma_fma_m64n128_bk16_s2` | 115.39 | 82.22 | 131.62 | 0.713 | 1.141 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm120_tma_fma_m128n64_bk16_s2` | 253.55 | 158.93 | 255.69 | 0.627 | 1.008 |
| hot_c | nn | 4621 x 1928 x 384 | `f32_f32_s2` | 138.97 | 90.20 | 122.46 | 0.649 | 0.881 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm120_f32_n64_copyplan` | 121.15 | 76.32 | 120.93 | 0.630 | 0.998 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm120_f32_n64_copyplan` | 136.08 | 106.02 | 122.08 | 0.779 | 0.897 |

Geometric mean fast/det over 5 cells: 0.677.

## inference - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm120_tma_tf32_m128n64_bk32_s2` | 74.02 | 81.44 | 131.41 | 1.100 | 1.775 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm120_tma_tf32_m128n64_bk32_s2` | 165.36 | 154.35 | 258.83 | 0.933 | 1.565 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm120_tma_tf32_m64n64_bk32_s2` | 81.72 | 90.08 | 122.30 | 1.102 | 1.497 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm120_tma_tf32_m64n64_bk32_s2_pair_store` | 78.43 | 75.90 | 120.66 | 0.968 | 1.538 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm120_tma_tf32_m64n64_bk32_s2` | 94.64 | 105.93 | 121.83 | 1.119 | 1.287 |

Geometric mean fast/det over 5 cells: 1.042.

## The board's own tier against the common tier

The same adapter was run on a copy of the tree whose loader leaves the
SM120 artifact set out, so every triad cell is served by the common tier
as it would be on a board without kernels of its own. Eager, one run each,
same board, same day.

Small shapes, where the common tier is faster eager:

| cell | dtype | op | own route | own us | common route | common us | own/common |
|---|---|---|---|---:|---:|---:|---:|
| d128_out_proj (1024 x 256 x 128) | bf16 | nt | `nt_sm120_tma_64x64_bk32_s2_bf16` | 14.91 | `nt_sm89_m16n64_bk64_s4_bf16` | 2.54 | 5.87 |
| d128_out_proj | bf16 | nn | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.79 | `nn_sm89_m16n64_bk64_s4_bf16` | 2.67 | 5.54 |
| d128_out_proj | bf16 | tn | `tn_sm120_tma_64x64_bk64_s3_streamk_bf16` | 15.05 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16` | 3.39 | 4.44 |
| d128_in_proj (1024 x 128 x 512) | bf16 | nn | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.78 | `nn_tc64_bf16` | 3.39 | 4.35 |
| d128_in_proj | bf16 | nt | `nt_sm120_tma_64x64_bk64_s3_bf16` | 14.88 | `nt_sm89_m16n64_bk64_s4_bf16` | 4.02 | 3.70 |
| d128_in_proj | bf16 | tn | `tn_sm120_tma_64x64_bk64_s3_streamk_bf16` | 14.98 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16` | 4.09 | 3.67 |
| underfill (256 x 512 x 384) | bf16 | nn | `nn_sm120_tma_64x64_bk64_s2_bf16` | 14.83 | `nn_tc16_bf16` | 3.23 | 4.59 |
| underfill | bf16 | nt | `nt_sm120_tma_64x64_bk64_s2_bf16` | 14.92 | `nt_tc64_bf16` | 6.71 | 2.23 |
| underfill | bf16 | tn | `tn_sm120_tma_64x64_bk32_s3_bf16` | 14.96 | `tn_tc64_bf16` | 7.00 | 2.14 |
| d128_out_proj | tf32 | tn | `tn_sm80_mma_tf32_m16n16_bk32_s4` | 9.64 | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 7.53 | 1.28 |

The f16 rows repeat the bf16 rows within a few percent. Every SM120 route
on these shapes sits on the same floor, 14.8 to 15.1 us whatever the
shape, which is the mark of a per-launch cost on the host rather than of
the kernel. On the large shapes the board's own tier wins eager by 1.3x
to 2.3x (bf16/f16 0.53 to 0.76 own/common, TF32 0.43 to 0.87, exact f32
0.67 to 0.99), nineteen of sixty-three cells fall to the common tier and
all nineteen are small.

Whether that floor reaches a training step is the question the graph
path answers, since a step replays a captured graph and pays no
per-launch host cost. The whole Mamba-1 step, 24 layers, batch 8, 1300
tokens, own tier against the same common-only copy:

| d_model | precision | own ms/step | common-only ms/step | own/common |
|---:|---|---:|---:|---:|
| 128 | BF16 | 33.70 | 39.58 | 0.851 |
| 128 | F32, TF32 | 45.19 | 45.08 | 1.002 |
| 384 | BF16 | 68.38 | 77.27 | 0.885 |
| 384 | F32, TF32 | 109.49 | 114.32 | 0.958 |

In the graph the board's own tier wins at d_model 128 as well, so the
table of this release keeps the SM120 tier on every shape it names. The
eager floor stays on the list for the board's tuning pass: it is a cost
of preparing the launch, not of running it, and a program that drives
small GEMMs eagerly pays it today.

