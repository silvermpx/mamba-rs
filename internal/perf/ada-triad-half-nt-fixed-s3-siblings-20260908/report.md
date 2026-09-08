# Four new half NT Fast-winning cells

RTX6000Ada / CUDA13.2, 2026-09-08. Reuse the exact Fixed-S3 B-XOR source
previously screened on d768-out; no CUDA source change and no d768-out rerun.

| Cell | Dtype | Candidate/Fast p50 range | Worst paired p95 | Decision |
| --- | --- | ---: | ---: | --- |
| d768-in | F16 | .880228–.892602 | .894937 | Advance |
| d768-in | BF16 | .895531–.927009 | .938543 | Advance |
| Prism | F16 | .499980–.509241 | .509820 | Advance |
| Prism | BF16 | .778067–.801779 | .804258 | Advance |

Each row covers eager/graph × ABBA/BAAB, independently paired against explicit
native-half cuBLAS Fast (`CUBLAS_COMPUTE_32F`, default tensor-op algorithm).
Every current-TC64 comparison also advances. Candidate output matches TC64
bits; Fast repeats against its own finite/nonzero oracle, not the TC64 bits.

Exact test PASS:
`ada_half_nt_fixed_s3_bxor_d768_in_and_prism_vs_current_and_fast_discovery_once7`.
2 resources /32 bit /32 screen /8 decision records. Root independently
replayed224 paired brackets /896 observations and both quantiles.20 GEMMs
per observation,7 windows, actual256B-aligned pointers, output/input guard
checks and resets outside the event interval. Native CPU shape/grid test
passed1/1 and the unchanged source helper passed5/5.

Candidate resources:167 registers,0 local/static bytes,98,304 dynamic shared,
256 threads, occupancy1. Grids96 for d768-in `(2048,768,3072)` and111 for Prism
`(4621,384,1928)`. The latter exercises the M and reduction tails on its actual
target. This is not full exceptional-value/all-tail/batch-invariance or
multi-toolkit qualification. Public AUTO admission is still pending.

Main SHA256: `dbbcfaf887ef3dd823f0803e2b8e91570771943b53a2afd7d870093d60da7a93`.
Helper SHA256: `2c41646460a23f92f43417b08b4389bc306386b26d521282c904881b2e1c2fc0`.
Binary SHA256: `1903e69d093b96524807625f0a6d439a438f2b93a16ed101a695db2f7f1ea913`.
[Raw samples](evidence/cuda132/run1/test.log), SHA256
`9d6cb191b890f662aad239843f5669ec3bb5bdb0e8ff8baca3482a0648e3c654`.
Exact measured sources are archived in `evidence/source/`. Command receipts
confirm no competing application in PRE/RELEASE/DRAIN and unchanged private
cache artifacts. No CUDA12.8/13.0 or new RTX5090 performance is inferred.
