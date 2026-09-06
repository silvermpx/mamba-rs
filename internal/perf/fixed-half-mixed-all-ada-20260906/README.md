# Ada Fixed half/mixed complete production-candidate screen

2026-09-06. Immutable C-promotion source and binary, later committed as
`26b44c4d`; source/binary hashes are in preflight.log. NVRTC 13.2, 142-SM Ada.
Default/warm caches; this is a 21-window screen, not a new cold-cache 101
promotion. No production changes were made from these numbers.

## Coverage and result

The 22 candidate specifications cover 7 BF16, 7 F16, 4 BF16->F32 and 4 F16->F32
physical tiles. Shapes A-E, both bias states, eager/graph and both timing
orders give 880 planned timing cohorts. 800 timing records passed; 20 Legacy+bias
candidate/cell attempts failed cross-AUTO bit identity before timing, each
excluding 4 timing cohorts. All recorded raw-storage/AUTO/repeat/vendor-repeat
bits and all 400 applicable graph replay flags pass. See
`legacy-bias-rejections.md`; these are not 20 failed AUTO cells.

Across 200 successful candidate/cell/bias groups, no non-incumbent tile beats
current AUTO at both paired p50 and p95 in all 4 timing cohorts. Thus this complete
shipped-candidate screen identifies no missed compatible AUTO promotion.
It does not include unintegrated standalone RNA or half-swizzle experiments.

## Current AUTO versus native half cuBLAS

The comparator is explicitly `CUBLAS_COMPUTE_32F` for half inputs, with the
same output dtype and timed bias epilogue. It is not the F32 FAST_TF32
comparator and not PEDANTIC. Every entry below is the conservative result
over all AUTO measurements in the candidate comparisons, not only the
forced copy of the same physical AUTO tile.

| Input -> output | Robust wins / 10 shape-bias cases | Winning cases | Worst paired p95 AUTO/vendor |
| --- | ---: | --- | ---: |
| BF16 -> BF16 | 7 | A0, A1, C0, C1, D1, E0, E1 | 1.605802 |
| F16 -> F16 | 4 | A1, C0, C1, D1 | 1.496841 |
| BF16 -> F32 | 2 | A1, C1 | 2.029817 |
| F16 -> F32 | 1 | A1 | 2.025610 |

Robust means paired p50 AND p95 < 1 in every applicable comparison. Different
forced-copy measurements of the same tile can produce less conservative
counts; do not substitute them for actual AUTO. Main independently recomputed
these 14/40 wins from each record's matched AUTO/vendor sample arrays, using
the harness's nearest-index quantile rule `round((count-1)*fraction)`.

`summary.json` preserves all 200 candidate group maxima, 40 AUTO cell maxima,
20 rejections and source/artifact metadata. Top remaining work: mixed-output
B/D/E and homogeneous-half B/D0. This is kernel optimization work, not a
missed promotion among these 22 already shipped candidate specifications.

Shapes are `(M,K,N)`: A=(4621,384,1928), B=(4621,768,2304),
C=(4621,1928,384), D=(2048,768,2304), E=(2048,2304,768).

Command: existing release `gemm_bi_fixed_performance` binary with
`MAMBA_FIXED_ADA_VENDOR=1 MAMBA_FIXED_VENDOR_EXACT_CC=8.9`
`MAMBA_FIXED_ADA_ROWS=bf16,f16,bf16_f32,f16_f32`
`MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_c,hot_d,hot_e`
`MAMBA_FIXED_ADA_BIAS=0,1 MAMBA_FIXED_VENDOR_PATHS=eager,graph`
`MAMBA_FIXED_ADA_WINDOWS=21`, with `MAMBA_FIXED_VENDOR_TILES` unset;
test `fixed_ada_forced_rungs_paired_precision_cublas`
`--ignored --exact --nocapture --test-threads=1`. CUDA_HOME and library/PATH
pointed explicitly at /usr/local/cuda-13.2. Runtime 328.59 s, exit 0.
