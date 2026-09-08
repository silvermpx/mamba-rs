# BF16 NN d768-in: aligned S3 beats cuBLAS Fast

Ada / CUDA13.2, 2026-09-08, shape (M,K,N)=(2048,768,3072).
Existing Fixed S3 body, test-only Triad wrapper; no dispatcher admission.

| Path | Candidate us | Current TC128 us | Fast us | Paired candidate/Fast p50 |
| --- | ---: | ---: | ---: | ---: |
| Eager | 62.59–63.08 | 91.09–91.56 | 80.62–81.23 | .7750–.7761 |
| Graph | 61.93–62.46 | 90.78–91.07 | 78.73–79.32 | .7860–.7863 |

Candidate/Fast p95 <= .788520 in all four strata; 21–23% lower time.
Candidate/current paired p50 .6821–.6874. Current is the forced TC128
reference, NOT proof of public AUTO selection. All timed logical pointers
are 256B aligned; eager/graph exact current bits and guards pass.
This replaces the weaker N64 BF16 d768-in finalist: prefer S3, not a new
N64 production holder. Together with the four previously retained aligned
S3 d768-out/Prism cells, five half NN cells now have short-screen Fast wins.

Exact test `ada_half_nn_fixed_s3_aligned_bf16_d768_in_confirmation_once7`
PASS; 2 resources / 8 screens / 1 decision. Once7, ABBA/BAAB, eager/graph,
20 GEMMs per observation. Root independently replayed all 56 brackets /
224 observations and quantiles. No full gates or cross-toolkit admission.

Measured typed source SHA256
`e77e1961d03d7f26d19ffd1a5fe6514127ead50871768f44ad4eab6d70d160b6`.
[Raw](evidence/once7-cuda132/test.log), SHA256
`aed0f7c9033ebdbf5d0fb1c58a2e6ea688b135c72a63081bf08be37f7a6e4330`.
