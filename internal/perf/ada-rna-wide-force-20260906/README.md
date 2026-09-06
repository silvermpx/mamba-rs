# Ada Fixed explicit-RNA wide qualification

This directory preserves the force-only integration and qualification of
`Tf32RnaM128N128S3`. It does not by itself assert AUTO promotion or a finished
inference performance matrix. The fragment is Fixed-owned and composed only
for `sm_89`; the existing add-based Triad wide symbol remains distinct.

## Physical and numerical contract

The new export is
`gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3`, block 256, tile 128x128,
BK32/S3, 98,304 bytes of opt-in dynamic shared memory. Live NVRTC/Driver
admission reports 153 registers, zero static shared/local memory/spills, and
at least one resident block. The five-argument ABI is four pointers followed
by the 32-byte `{alpha,beta,m,k,n,lda,ldb,ldc}` bundle. The implementation uses
explicit `cvt.rna.tf32.f32`, bias-seeded accumulation and ascending k8 MMA.

The force inventory and real Driver graph verify the exact export, parameters
and launch geometry. Even `PATHS=eager` must capture an untimed identity graph
and pass these checks; this was added after a review finding and observed
RED/GREEN regression. No timing can qualify a different bit-compatible
physical function merely because it returned the expected enum name.

Raw bits are compared to all five ordinary Fixed TF32 rungs, with finite and
exceptional inputs, both bias states, prefixes/subviews, C4 output, repeated
eager and poisoned graph replay. See the integration report and final test
logs for the exact covered shapes and remaining gates. Comparisons across
different precision modes are numerical error checks, not raw bit equality.

## Production 21-window screen

`screen21-rna-all5-both-paths.log` contains 40 passing timing records:
A-E x bias 0/1 x eager/graph x two timing orders. No candidate was rejected;
all raw-storage/AUTO/repeat/vendor-repeat bits and all applicable graph replay
bits pass. Main independently checked these flags, the actual RNA symbol,
the artifact identity and the raw log SHA256.

Each number below is the worst paired p95 ratio over both paths and both
orders for that shape/bias. A ratio below one favors RNA.

| Case | RNA / current AUTO | RNA / cuBLAS FAST |
| --- | ---: | ---: |
| A0 | 0.840548 | 1.101121 |
| A1 | 0.854200 | 0.881080 |
| B0 | 0.730872 | 1.114531 |
| B1 | 0.735926 | 0.970577 |
| C0 | 0.621345 | 1.313748 |
| C1 | 0.621768 | 1.203567 |
| D0 | 0.898113 | 1.263077 |
| D1 | 0.901641 | 1.094039 |
| E0 | 0.871150 | 1.307663 |
| E1 | 0.873651 | 1.221414 |

All ten cases also win against current AUTO at paired p50. Only A1 and B1
win both p50 and p95 against FAST in every cohort. These are preliminary
21-window results; the 101-window confirmation below supersedes them for
promotion admission. Validation through the actual promoted AUTO route is
still a separate required gate.

Shapes `(M,K,N)`: A=(4621,384,1928), B=(4621,768,2304),
C=(4621,1928,384), D=(2048,768,2304), E=(2048,2304,768).
FAST means explicit `CUBLAS_COMPUTE_32F_FAST_TF32`, including the required
bias broadcast inside vendor timing. PEDANTIC is only the numerical reference.

## Evidence interpretation

The final host build also passed 40/40 unique 101-window records with no
rejections, all raw/repeat/graph bit gates true, and the exact RNA symbol and
artifact throughout. Main independently checked the record matrix and log
SHA256 `4983d29fa241538f170e087372c70baa92858033a7b075c7e53945088ba6d910`.

| Case | Worst RNA / AUTO p95 | Worst RNA / FAST p95 |
| --- | ---: | ---: |
| A0 | 0.849957 | 1.114369 |
| A1 | 0.860678 | 0.887512 |
| B0 | 0.732515 | 1.114562 |
| B1 | 0.736051 | 0.976818 |
| C0 | 0.627831 | 1.324942 |
| C1 | 0.628870 | 1.212045 |
| D0 | 0.903182 | 1.274882 |
| D1 | 0.907060 | 1.124559 |
| E0 | 0.891869 | 1.365570 |
| E1 | 0.888943 | 1.271124 |

All ten cases win against AUTO at both paired p50 and p95 in all cohorts;
A1 and B1 likewise win against FAST. The final 101-window binary SHA256 is
`db3dbafb1b1ccf3c0ea422d2860e8a7789e1eca9b9c14f51480711f18ac5dde1`.
It adds the reviewed RNA-only empty-output no-op; CUDA source/artifact is
unchanged from the screen. Final correctness includes cross-rung comparisons
at every prefix, actual hot-A boundaries 4620/4621/4622, incumbent underfill
boundaries 6016/6017/6018, K0/all-five-rung/repeat/graph/guard checks and null
empty operands with unused unrepresentable dimensions. See final logs and
`integration-report.md` for exact hashes and commands.

The frozen screen binary SHA256 is
`2d7721f42aa08dccef786a0dead4e9242eb7c54d607ad2e494fecb35f1745982`.
The Fixed source digest is
`7ccad9938c4ee7cb0080d62b7053e082537fcb2ad184288ffc59a513cf526301`,
and the loaded artifact digest is
`c6dc1ee707ceceb512b58f226f8b5288097162c5257c41ef89a2e235fda1945f`.

The header-manifest digest changes because its source analysis includes
conditional/macro/call information from the appended fragment, even without
a new include. It is not evidence of a toolkit upgrade. The NVRTC library
domain remains unchanged. Tests compare composed Triad and CC12 source bytes;
they do not claim unmeasured cross-build artifact identity.

Historical filenames are not verdicts. In particular,
`green-rna-eager-a01-w21.log` records a rejected `CELLS=A` operator typo,
not a passing GPU run. The corrected eager-only run is
`green-rna-eager-hot-a01-w21-v2.log`. Intermediate `v1` and RED logs are
retained for provenance. PTX dumps remain available in the ignored local and
remote evidence directories, but large compiler artifacts are not committed.
