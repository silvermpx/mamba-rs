# TF32 TN small register pipeline: no retained winner

Ada / CUDA13.2, one build, two exact once7 processes. Candidate specializes
only TN M16N32/BK32/S4 register-fragment prefetch; staging, barriers, RNA,
ascending K8 MMA association and epilogue are unchanged.

| Cell | Candidate/AUTO p50 | Worst AUTO p95 | Candidate/Fast graph p50 | Decision |
| --- | ---: | ---: | ---: | --- |
| d128-in | .9020–.9771 | 1.0435 | 1.6643–1.7547 | Stop |
| d128-out | .9249–.9956 | 1.0107 | 1.8340–1.8689 | Stop |

Strict target/tail/alpha/exception/K0 bit, input and guard preflights passed.
Both actual AUTO and candidate are M16N32/S4, grid128/64 respectively for
the two cells, block128/dynamic32768. Candidate has63 registers,0 local/static
bytes, occupancy3. Some lower medians do not pass the all-strata p95 rule;
no winner, no unchanged retry and no production change.

Exact tests:
`cuda_suite::ada_tf32_tn_d128_in_small_regpipe_discovery_once7` and
`cuda_suite::ada_tf32_tn_d128_out_small_regpipe_discovery_once7`.
Each has1 resource,8 paired timing records and1 decision. Root replayed112
brackets /448 observations and quantiles across both tests. Timing uses1
GEMM/observation,256B-aligned pointers, eager/graph ABBA/BAAB with explicit
cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32`. Private cache stable; no competing
application in PRE/RELEASE/DRAIN. Native helper4/4 and combined53/53 passed.

Main SHA256: `88e238cf7a44d0e964f3fc963606ff65cfa8d00688c8e86fbde5b1d7de875861`.
Helper SHA256: `bfc9052dfdca733db3f7d1620af8f27ee8aee3d15ab444055f166900ae62ab87`.
Binary SHA256: `dc1e97c6c288a8fbc9e94fa9e36f5cb3b2c52d682169e5bef70c8c6e44975521`.
[In raw](evidence/cuda132/run1/test.log), SHA256
`a018d47653a5bf6a0468d0efa4ffb790fb8c8faad789e58b42c7470aa7c22230`.
[Out raw](evidence/cuda132/run2/test.log), SHA256
`99362d1e4378c6bdd12a44b49ae30406cf6d889705d0b217046c0b4d37ee19ad`.
Exact source snapshots are under `evidence/source/`; no full toolkit gate.
