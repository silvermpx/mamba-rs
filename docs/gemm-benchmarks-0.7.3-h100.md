# GEMM benchmarks - 0.7.3, H100 PCIe

Measured on September 18, 2026 with the final 0.7.3 tree, one run of the
kernel-level adapter on a rented board, CUDA 13.2. On this toolkit the
board runs the common tier alone: the SM90a WGMMA module loads on CUDA
13.3 and newer (ptxas before 13.3 could drop the register moves after
`wgmma.wait_group 1`, CUDA 13.3 release notes), so every route here is a
common kernel admitted at first use by the bit proof, or the scalar tier
where no candidate was admitted. The training-step rows are on the
[Mamba-1](mamba1-benchmarks.md) and [Mamba-3](mamba3-benchmarks.md) pages.

## Measurement scope

| item | value |
|---|---|
| GPU | NVIDIA H100 PCIe, SM90 (CC 9.0), 114 SMs (NVIDIA's published count for this part) |
| measured GPU UUID | `GPU-508ace01-a2b3-bec9-5fe1-855752e6b08c` |
| driver / toolkit | 610.57.04 / CUDA 13.2, NVRTC 13.2.78 |
| host compiler | Rust 1.98.1, release profile |
| GEMM mode | `Deterministic`; the automatic route, no forced candidate |
| storage precisions | exact F32, TF32, BF16, F16 |
| path | eager; every cell times three arms in one process |

Each cell times the deterministic route, cuBLAS in the tree's fast
setting (TF32 handle math) and cuBLAS in the tree's pedantic setting in
windows of calibrated launch counts, 21 windows, the arm order mirrored
from window to window; the number is the lower median per arm. The row's
`route` is the symbol the dispatcher recorded for the launch.

`fast/det` and `pedantic/det` divide the vendor time by ours on this
board: above 1.0 the deterministic route is faster than that vendor
setting. cuBLAS on Hopper runs WGMMA kernels; the common tier does not,
which is what these ratios measure. There is no earlier record of this
board.

What the board leaves out of the common tier, said once at context
creation: ptxas 13.2 for `sm_90a` allocates `nt_sm89_m128n128_bk64_s3_bxor_f16`
and five siblings above their Ada register gates, puts 104 bytes of local
memory under `tn_sm89_tf32_m192n192_w3x4_bk32_s2`, and takes
`tn_sm89_tf32_pre_rna_m64n64_bk32_s3` and the `tn_sm89_tf32_pre_rna_transpose_32x32`
helper above their gates; their siblings serve.


Eager microseconds, p50 over 21 windows, one run, kernel cache on. `det` is the deterministic route the dispatcher chose on this board; `fast` is cuBLAS with TF32 handle math and `pedantic` is cuBLAS pedantic on the same board, the vendor controls. `fast/det` and `pedantic/det` divide the vendor time by ours: above 1.0 the deterministic route is faster.

## triad - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_tc64_bf16` | 6.29 | 4.70 | 14.07 | 0.746 | 2.236 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm89_m16n64_bk64_s4_bf16` | 5.97 | 4.71 | 15.25 | 0.788 | 2.554 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_m128n128_bk64_s3_bf16` | 68.95 | 24.22 | 425.11 | 0.351 | 6.165 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_m128n128_bk64_s3_bf16` | 32.98 | 12.45 | 239.20 | 0.378 | 7.253 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_tc_bf16` | 324.56 | 82.00 | 1680.56 | 0.253 | 5.178 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_m128n128_bk64_s3_bf16` | 58.45 | 24.23 | 299.32 | 0.415 | 5.121 |
| underfill | nn | 256 x 512 x 384 | `nn_tc64_bf16` | 22.49 | 62.57 | 45.24 | 2.782 | 2.012 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm89_m16n64_bk64_s4_bf16` | 8.53 | 4.70 | 37.29 | 0.551 | 4.372 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm89_m16n64_bk64_s4_bf16` | 5.12 | 5.90 | 12.52 | 1.154 | 2.447 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_tc_bf16` | 78.77 | 21.95 | 480.66 | 0.279 | 6.102 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_tc_bf16` | 44.94 | 13.31 | 210.25 | 0.296 | 4.679 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_tc_bf16` | 295.62 | 84.37 | 1599.40 | 0.285 | 5.410 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_tc_bf16` | 55.10 | 18.35 | 311.29 | 0.333 | 5.649 |
| underfill | nt | 256 x 512 x 384 | `nt_tc64_bf16` | 8.13 | 5.47 | 29.51 | 0.673 | 3.629 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_bf16` | 7.75 | 5.13 | 48.50 | 0.662 | 6.254 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_bf16` | 6.57 | 5.09 | 48.00 | 0.775 | 7.303 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_bf16` | 98.26 | 62.10 | 439.12 | 0.632 | 4.469 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_relay_m64n64_bk64_s3_bf16` | 62.32 | 13.36 | 219.36 | 0.214 | 3.520 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_tc64_streamk_bf16` | 360.52 | 95.91 | 1624.78 | 0.266 | 4.507 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_bf16` | 86.25 | 21.28 | 353.87 | 0.247 | 4.103 |
| underfill | tn | 256 x 512 x 384 | `tn_tc64_bf16` | 9.29 | 10.74 | 17.41 | 1.155 | 1.873 |

Geometric mean fast/det over 21 cells: 0.496.

## triad - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_tc64_f16` | 6.53 | 10.54 | 16.27 | 1.612 | 2.490 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm89_m16n64_bk64_s4_f16` | 6.04 | 6.09 | 22.01 | 1.009 | 3.644 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_m128n128_bk64_s3_f16` | 71.01 | 25.21 | 453.29 | 0.355 | 6.384 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_m128n128_bk64_s3_f16` | 32.77 | 12.49 | 250.80 | 0.381 | 7.652 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_tc_f16` | 334.82 | 89.82 | 1792.86 | 0.268 | 5.355 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_m128n128_bk64_s3_f16` | 60.07 | 24.59 | 318.89 | 0.409 | 5.309 |
| underfill | nn | 256 x 512 x 384 | `nn_tc64_f16` | 10.03 | 5.14 | 41.26 | 0.513 | 4.115 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm89_m16n64_bk64_s4_f16` | 8.65 | 4.75 | 37.66 | 0.549 | 4.355 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm89_m16n64_bk64_s4_f16` | 5.21 | 4.90 | 12.72 | 0.940 | 2.441 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_tc_f16` | 79.99 | 22.26 | 483.84 | 0.278 | 6.049 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_tc_f16` | 45.14 | 13.44 | 210.52 | 0.298 | 4.664 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_tc_f16` | 298.53 | 90.20 | 1621.83 | 0.302 | 5.433 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_tc_f16` | 56.92 | 18.33 | 321.68 | 0.322 | 5.652 |
| underfill | nt | 256 x 512 x 384 | `nt_tc64_f16` | 17.41 | 62.08 | 44.72 | 3.566 | 2.569 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm89_half_d128_in_m32n16_bk64_s4_cg_f16` | 7.94 | 5.25 | 55.44 | 0.661 | 6.980 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_half_d128_out_m32n16_bk64_s4_cg_f16` | 24.44 | 61.07 | 56.43 | 2.499 | 2.309 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm89_m64n64_bk64_s2_regpipe_vec2_f16` | 101.58 | 24.23 | 451.13 | 0.239 | 4.441 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_relay_m64n64_bk64_s3_f16` | 62.23 | 13.55 | 218.98 | 0.218 | 3.519 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_tc64_streamk_f16` | 361.45 | 98.12 | 1637.30 | 0.271 | 4.530 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm89_m64n64_bk64_s2_compact_bxor_f16` | 89.26 | 61.88 | 388.40 | 0.693 | 4.352 |
| underfill | tn | 256 x 512 x 384 | `tn_tc64_f16` | 17.61 | 61.62 | 44.47 | 3.500 | 2.526 |

Geometric mean fast/det over 21 cells: 0.579.

## triad - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_splitk32_partial+splitk_reduce` | 17.06 | 6.45 | 11.10 | 0.378 | 0.651 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_splitk32_partial+splitk_reduce` | 11.59 | 13.15 | 25.61 | 1.135 | 2.210 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm89_f32_n64_copyplan` | 330.90 | 59.15 | 333.87 | 0.179 | 1.009 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_f32_n64_copyplan` | 192.54 | 28.53 | 167.40 | 0.148 | 0.869 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm89_f32_n64_copyplan` | 1317.16 | 168.93 | 1173.54 | 0.128 | 0.891 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_f32_n64_copyplan` | 248.43 | 49.80 | 246.45 | 0.200 | 0.992 |
| underfill | nn | 256 x 512 x 384 | `nn_splitk32_partial+splitk_reduce` | 13.64 | 9.93 | 18.38 | 0.728 | 1.347 |
| d128_in_proj | nt | 1024 x 128 x 512 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 18.10 | 8.68 | 14.42 | 0.479 | 0.796 |
| d128_out_proj | nt | 1024 x 256 x 128 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 14.96 | 6.69 | 11.19 | 0.448 | 0.748 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 419.30 | 46.09 | 308.73 | 0.110 | 0.736 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 168.05 | 26.70 | 154.19 | 0.159 | 0.918 |
| large_deep | nt | 4096 x 3072 x 1536 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 1277.43 | 167.31 | 1287.61 | 0.131 | 1.008 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `transpose_f32_32x16_d768+nn_sm89_f32_n64_copyplan` | 360.12 | 47.70 | 214.55 | 0.132 | 0.596 |
| underfill | nt | 256 x 512 x 384 | `transpose_f32_2d+nn_splitk32_partial+splitk_reduce` | 16.69 | 7.56 | 18.68 | 0.453 | 1.119 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_splitm_partial_aligned+splitm_reduce` | 43.84 | 9.97 | 13.52 | 0.227 | 0.308 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm89_f32_d128_out_m16n16_g8_s2_cg` | 14.63 | 9.05 | 10.22 | 0.619 | 0.698 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `transpose_f32_32x16_d768+tn_sm89_f32_n64_dual_chunk_fused_finalize` | 364.46 | 88.62 | 275.19 | 0.243 | 0.755 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm89_f32_m64n64_bk16_s2_d768_out_raw+splitm_reduce` | 216.82 | 56.70 | 153.45 | 0.262 | 0.708 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_aligned` | 1317.12 | 250.61 | 1103.28 | 0.190 | 0.838 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_splitm_partial_aligned+splitm_reduce` | 341.24 | 76.89 | 224.57 | 0.225 | 0.658 |
| underfill | tn | 256 x 512 x 384 | `tn_m16n16_bk16_s2_splitm16` | 19.48 | 46.45 | 60.54 | 2.385 | 3.108 |

Geometric mean fast/det over 21 cells: 0.292.

## triad - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| d128_in_proj | nn | 1024 x 128 x 512 | `nn_sm80_mma_tf32_m64n64_bk32_s2` | 12.17 | 6.43 | 11.04 | 0.528 | 0.907 |
| d128_out_proj | nn | 1024 x 256 x 128 | `nn_sm80_mma_tf32_m16n32_bk32_s4` | 21.60 | 55.40 | 70.28 | 2.564 | 3.253 |
| d768_in_proj | nn | 2048 x 768 x 3072 | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 133.72 | 60.19 | 333.99 | 0.450 | 2.498 |
| d768_out_proj | nn | 2048 x 1536 x 768 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3` | 101.28 | 28.67 | 168.43 | 0.283 | 1.663 |
| large_deep | nn | 4096 x 3072 x 1536 | `nn_sm80_mma_tf32_m128n128_bk32_s3` | 501.58 | 169.51 | 1189.48 | 0.338 | 2.371 |
| prism_in_proj | nn | 4621 x 384 x 1928 | `nn_sm89_tf32_addhalf_m128n96_bk32_s3_direct` | 106.45 | 49.51 | 245.16 | 0.465 | 2.303 |
| underfill | nn | 256 x 512 x 384 | `nn_sm80_mma_tf32_splitk2_m16n32_bk32_s4` | 12.28 | 9.67 | 17.74 | 0.788 | 1.445 |
| d128_in_proj | nt | 1024 x 128 x 512 | `nt_sm80_mma_tf32_splitk4_m16n32_bk32_s3` | 56.88 | 47.61 | 84.77 | 0.837 | 1.490 |
| d128_out_proj | nt | 1024 x 256 x 128 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 9.28 | 5.03 | 8.49 | 0.542 | 0.914 |
| d768_in_proj | nt | 2048 x 768 x 3072 | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 192.99 | 46.38 | 313.16 | 0.240 | 1.623 |
| d768_out_proj | nt | 2048 x 1536 x 768 | `nt_sm89_tf32_a_ldmatrix_m128n96_bk32_s3` | 66.92 | 88.37 | 154.85 | 1.321 | 2.314 |
| large_deep | nt | 4096 x 3072 x 1536 | `nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2` | 485.74 | 168.01 | 1331.00 | 0.346 | 2.740 |
| prism_in_proj | nt | 4621 x 384 x 1928 | `nt_sm89_tf32_rna_m144n96_w3x4_bk32_s2` | 201.97 | 46.23 | 210.37 | 0.229 | 1.042 |
| underfill | nt | 256 x 512 x 384 | `nt_sm80_mma_tf32_m16n32_bk32_s4` | 20.75 | 47.24 | 60.83 | 2.276 | 2.931 |
| d128_in_proj | tn | 1024 x 128 x 512 | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s4` | 56.81 | 61.86 | 61.12 | 1.089 | 1.076 |
| d128_out_proj | tn | 1024 x 256 x 128 | `tn_sm80_mma_tf32_splitk8_m32n32_bk32_s3` | 56.47 | 61.68 | 61.16 | 1.092 | 1.083 |
| d768_in_proj | tn | 2048 x 768 x 3072 | `tn_sm80_mma_tf32_m64n64_bk32_s2` | 331.42 | 88.90 | 275.72 | 0.268 | 0.832 |
| d768_out_proj | tn | 2048 x 1536 x 768 | `tn_sm80_mma_tf32_m64n64_bk32_s2` | 179.01 | 49.98 | 152.55 | 0.279 | 0.852 |
| large_deep | tn | 4096 x 3072 x 1536 | `tn_sm80_mma_tf32_m64n64_bk32_s2` | 1119.18 | 251.13 | 1103.95 | 0.224 | 0.986 |
| prism_in_proj | tn | 4621 x 384 x 1928 | `tn_sm80_mma_tf32_m128n64_bk32_s3` | 288.38 | 76.95 | 224.67 | 0.267 | 0.779 |
| underfill | tn | 256 x 512 x 384 | `tn_sm80_mma_tf32_m16n32_bk32_s4` | 21.45 | 46.88 | 61.07 | 2.186 | 2.848 |

Geometric mean fast/det over 21 cells: 0.566.

## inference - bf16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_tc128_pipeline_bf16` | 61.56 | 23.46 | 299.40 | 0.381 | 4.863 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_tc128_s3_bf16` | 105.66 | 37.46 | 706.23 | 0.355 | 6.684 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_tc128_pipeline_bf16` | 48.40 | 17.08 | 305.74 | 0.353 | 6.317 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_m128n144_bk32_s2_vec_bf16` | 64.85 | 19.00 | 329.62 | 0.293 | 5.083 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_m128n96_bk64_s2_vec_bf16` | 84.02 | 16.52 | 355.09 | 0.197 | 4.226 |

Geometric mean fast/det over 5 cells: 0.307.

## inference - f16

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_tc128_pipeline_f16` | 62.62 | 24.08 | 316.33 | 0.384 | 5.051 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_tc128_s3_f16` | 108.55 | 69.96 | 741.39 | 0.645 | 6.830 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_tc128_pipeline_f16` | 47.93 | 17.91 | 335.32 | 0.374 | 6.996 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_tc128_swizzle_f16` | 56.66 | 19.57 | 351.66 | 0.345 | 6.206 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_m128n96_bk64_s2_vec_f16` | 85.04 | 17.02 | 380.41 | 0.200 | 4.473 |

Geometric mean fast/det over 5 cells: 0.364.

## inference - f32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_sm89_f32_n64_copyplan` | 248.67 | 49.51 | 247.22 | 0.199 | 0.994 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_sm89_f32_n64_copyplan` | 583.42 | 82.91 | 503.34 | 0.142 | 0.863 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_sm89_m112n128_bk32_s3_f32` | 425.94 | 59.76 | 198.05 | 0.140 | 0.465 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_f32_n64_copyplan` | 256.38 | 45.75 | 258.66 | 0.178 | 1.009 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_f32_n64_copyplan` | 302.14 | 40.95 | 252.74 | 0.136 | 0.836 |

Geometric mean fast/det over 5 cells: 0.157.

## inference - tf32

| cell | op | shape m x k x n | route | det us | fast us | pedantic us | fast/det | pedantic/det |
|---|---|---|---|---:|---:|---:|---:|---:|
| hot_a | nn | 4621 x 384 x 1928 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 122.76 | 49.86 | 249.86 | 0.406 | 2.035 |
| hot_b | nn | 4621 x 768 x 2304 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 228.52 | 81.31 | 494.35 | 0.356 | 2.163 |
| hot_c | nn | 4621 x 1928 x 384 | `nn_rna_wide_tf32_m128n128_bk32_s3` | 94.09 | 59.86 | 216.04 | 0.636 | 2.296 |
| hot_d | nn | 2048 x 768 x 2304 | `nn_sm89_m64n288_bk16_s2_tf32` | 127.34 | 45.89 | 259.39 | 0.360 | 2.037 |
| hot_e | nn | 2048 x 2304 x 768 | `nn_sm89_rna_tf32_m128n96_bk32_s3` | 178.70 | 40.60 | 253.99 | 0.227 | 1.421 |

Geometric mean fast/det over 5 cells: 0.376.
