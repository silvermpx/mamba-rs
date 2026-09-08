# Half NT compact BK64/S2: stop both d768-out cells

Ada / CUDA13.2, 2026-09-08. Shared-memory XOR compaction only; preserve
BK64/S2, MMA order and typed RNE epilogue. This batch has TWO cells, not six.

| Dtype | Candidate/Fast eager p50 | Candidate/Fast graph p50 |
| --- | ---: | ---: |
| F16 | 1.281–1.282 | 1.343–1.348 |
| BF16 | 1.335–1.336 | 1.359–1.365 |

Both valid losses, no retry/integration. TC64 supplies exact-bit reference
only: despite the test name, this batch has NO paired candidate/current timing.
NCU motivated lower shared-memory occupancy cost, not a bank-conflict claim.
Candidate shared 32768 bytes permits 3 CTA/SM; this alone does not beat Fast.
256B-aligned guarded buffers, repeat eager/graph bits and Fast own bits pass.

Exact `ada_half_nt_d768_out_compact_bk64_s2_vs_current_and_fast_discovery_once7`
PASS: 2 resources / 16 bits / 8 screens / 2 decisions. Root replayed all
56 brackets / 224 observations and quantiles. No broad gates.
Measured main `c2db48361dc8ee90d511bf7dff04e1d416f5b24ab9045b6ab1e40bb37f2e665e`
(retained in subsequent e77e snapshot); helper
`637dc875d69a9fd08282f2035d97f4254529f651810058880e65d0b3355c4b56`.
[Raw](evidence/once7-cuda132/test.log), SHA256
`2dd58be6ed290354f9570c69e837224ab70af85ad91142506200c0c3b46e4cba`.
