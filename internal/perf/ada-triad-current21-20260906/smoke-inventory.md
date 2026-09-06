# Ada Triad production route smoke

Source: committed `318b3fbd09c8fe0bfac7eaabefdec2bba457f212`; CUDA 13.2.51; CC 8.9; RTX 6000 Ada. One window per path, 60 cells × eager/graph = 120 records; all eager/graph identities passed. These smoke timings are not performance evidence.

| Route | Op | Shape | Actual physical symbols |
|---|---|---|---|
| f32_policy_exact | nn | d128_in_proj | `gemm_bi_nn_splitk32_partial` + `gemm_bi_splitk_reduce` |
| f32_policy_allow_tf32 | nn | d128_in_proj | `gemm_bi_nn_sm80_mma_tf32_v1_m64n64_bk32_s2` |
| bf16_policy_tc | nn | d128_in_proj | `gemm_bi_nn_tc64_bf16` |
| f16_policy_tc | nn | d128_in_proj | `gemm_bi_nn_tc64_f16` |
| f32_policy_exact | tn | d128_in_proj | `gemm_bi_tn_splitm_partial_aligned` + `gemm_bi_splitm_reduce` |
| f32_policy_allow_tf32 | tn | d128_in_proj | `gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4` |
| bf16_policy_tc | tn | d128_in_proj | `gemm_bi_tn_tc64_bf16` |
| f16_policy_tc | tn | d128_in_proj | `gemm_bi_tn_tc64_f16` |
| f32_policy_exact | nt | d128_in_proj | `gemm_bi_transpose_f32_2d` + `gemm_bi_nn_splitk32_partial` + `gemm_bi_splitk_reduce` |
| f32_policy_allow_tf32 | nt | d128_in_proj | `gemm_bi_nt_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s3` |
| bf16_policy_tc | nt | d128_in_proj | `gemm_bi_nt_tc64_bf16` |
| f16_policy_tc | nt | d128_in_proj | `gemm_bi_nt_tc64_f16` |
| f32_policy_exact | nn | d128_out_proj | `gemm_bi_nn_splitk32_partial` + `gemm_bi_splitk_reduce` |
| f32_policy_allow_tf32 | nn | d128_out_proj | `gemm_bi_nn_sm80_mma_tf32_v1_m16n32_bk32_s4` |
| bf16_policy_tc | nn | d128_out_proj | `gemm_bi_nn_tc64_bf16` |
| f16_policy_tc | nn | d128_out_proj | `gemm_bi_nn_tc64_f16` |
| f32_policy_exact | tn | d128_out_proj | `gemm_bi_tn_splitm_partial_aligned` + `gemm_bi_splitm_reduce` |
| f32_policy_allow_tf32 | tn | d128_out_proj | `gemm_bi_tn_sm80_mma_tf32_v1_m16n32_bk32_s4` |
| bf16_policy_tc | tn | d128_out_proj | `gemm_bi_tn_tc64_bf16` |
| f16_policy_tc | tn | d128_out_proj | `gemm_bi_tn_tc64_f16` |
| f32_policy_exact | nt | d128_out_proj | `gemm_bi_transpose_f32_2d` + `gemm_bi_nn_splitk32_partial` + `gemm_bi_splitk_reduce` |
| f32_policy_allow_tf32 | nt | d128_out_proj | `gemm_bi_nt_sm80_mma_tf32_v1_m16n32_bk32_s4` |
| bf16_policy_tc | nt | d128_out_proj | `gemm_bi_nt_tc64_bf16` |
| f16_policy_tc | nt | d128_out_proj | `gemm_bi_nt_tc64_f16` |
| f32_policy_exact | nn | d768_in_proj | `gemm_bi_nn` |
| f32_policy_allow_tf32 | nn | d768_in_proj | `gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3` |
| bf16_policy_tc | nn | d768_in_proj | `gemm_bi_nn_tc_bf16` |
| f16_policy_tc | nn | d768_in_proj | `gemm_bi_nn_tc_f16` |
| f32_policy_exact | tn | d768_in_proj | `gemm_bi_tn_splitm_partial_aligned` + `gemm_bi_splitm_reduce` |
| f32_policy_allow_tf32 | tn | d768_in_proj | `gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2` |
| bf16_policy_tc | tn | d768_in_proj | `gemm_bi_tn_tc64_bf16` |
| f16_policy_tc | tn | d768_in_proj | `gemm_bi_tn_tc64_f16` |
| f32_policy_exact | nt | d768_in_proj | `gemm_bi_nt` |
| f32_policy_allow_tf32 | nt | d768_in_proj | `gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3` |
| bf16_policy_tc | nt | d768_in_proj | `gemm_bi_nt_tc_bf16` |
| f16_policy_tc | nt | d768_in_proj | `gemm_bi_nt_tc_f16` |
| f32_policy_exact | nn | d768_out_proj | `gemm_bi_nn` |
| f32_policy_allow_tf32 | nn | d768_out_proj | `gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3` |
| bf16_policy_tc | nn | d768_out_proj | `gemm_bi_nn_tc_bf16` |
| f16_policy_tc | nn | d768_out_proj | `gemm_bi_nn_tc_f16` |
| f32_policy_exact | tn | d768_out_proj | `gemm_bi_tn_splitm_partial_aligned` + `gemm_bi_splitm_reduce` |
| f32_policy_allow_tf32 | tn | d768_out_proj | `gemm_bi_tn_sm80_mma_tf32_v1_m64n64_bk32_s2` |
| bf16_policy_tc | tn | d768_out_proj | `gemm_bi_tn_tc64_bf16` |
| f16_policy_tc | tn | d768_out_proj | `gemm_bi_tn_tc64_f16` |
| f32_policy_exact | nt | d768_out_proj | `gemm_bi_nt` |
| f32_policy_allow_tf32 | nt | d768_out_proj | `gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3` |
| bf16_policy_tc | nt | d768_out_proj | `gemm_bi_nt_tc64_bf16` |
| f16_policy_tc | nt | d768_out_proj | `gemm_bi_nt_tc64_f16` |
| f32_policy_exact | nn | prism_in_proj | `gemm_bi_nn` |
| f32_policy_allow_tf32 | nn | prism_in_proj | `gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3` |
| bf16_policy_tc | nn | prism_in_proj | `gemm_bi_nn_tc_bf16` |
| f16_policy_tc | nn | prism_in_proj | `gemm_bi_nn_tc_f16` |
| f32_policy_exact | tn | prism_in_proj | `gemm_bi_tn_splitm_partial_aligned` + `gemm_bi_splitm_reduce` |
| f32_policy_allow_tf32 | tn | prism_in_proj | `gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3` |
| bf16_policy_tc | tn | prism_in_proj | `gemm_bi_tn_tc64_bf16` |
| f16_policy_tc | tn | prism_in_proj | `gemm_bi_tn_tc64_f16` |
| f32_policy_exact | nt | prism_in_proj | `gemm_bi_nt_slim` |
| f32_policy_allow_tf32 | nt | prism_in_proj | `gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3` |
| bf16_policy_tc | nt | prism_in_proj | `gemm_bi_nt_tc_bf16` |
| f16_policy_tc | nt | prism_in_proj | `gemm_bi_nt_tc_f16` |

