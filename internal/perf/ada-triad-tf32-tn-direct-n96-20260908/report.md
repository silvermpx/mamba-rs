# Ada TF32 TN direct-N96 discovery

d768-in passes focused bits and resources but loses cuBLAS Fast. Removing
the separate raw-A transpose does not establish a new best over the retained
transpose+N96 pipeline: that pipeline was not paired in this screen. Keep
the earlier shortlist and do not repeat this unchanged candidate.

| Comparator | Eager candidate/comparator p50 | Graph p50 | Worst p95 |
| --- | ---: | ---: | ---: |
| Actual AUTO | .7234–.7235 | .7316–.7347 | .7572 |
| Explicit TF32 Fast | 1.0177–1.0247 | 1.3539–1.3654 | 1.4024 |

The direct candidate is roughly27% below old actual AUTO, not a Fast win.
Graph candidate time is about202–207 microseconds versus Fast147–154.
The complete vendor graph has two nodes; both nodes are timed, without
imposing the candidate's one-kernel ABI on cuBLAS internals.

RTX6000Ada / CUDA13.2 only. Exact ignored test
`cuda_suite::ada_tf32_tn_d768_in_direct_rna_n96_discovery_once7` passed with
target/current/AUTO bits, forced-family tail alpha1/-0.75, exceptional values,
guards and K0 checks. The direct-only tail is `(129,68,36)`: lda68/ldb36
satisfy16B copies while retaining output/reduction tile tails. Legacy variants
keep `(129,65,36)`; K0 remains before the nonzero-reduction alignment guard.

Candidate: grid192/block256, 158 registers, zero local/static shared bytes,
86,016 dynamic shared bytes, occupancy1. Actual AUTO is the64/S2 route with
grid576/block128/shared36,864. No arithmetic association, RNA conversion or
TN epilogue change was admitted.

Root replayed56 brackets /224 observations using explicit observation arms
and nearest-rank quantiles;8 timing screens,1 resource and1 decision record.
One GEMM per observation, once7 eager/graph ABBA/BAAB,256B-aligned logical
pointers. Native helper tests7/7 passed. Receipts show stable private cache,
quiet PRE/DRAIN and no competing application; immediate RELEASE shows9%
GPU utilization after the test. Raw evidence: `evidence/cuda132/`.

Exact measured source is preserved in the commit containing this report:

- Main: `36126907bdb1a9cd98e9009bed0c7e7a2adec7a026be2b4b5e33b350db069251`
- Direct helper: `a153aeb2e00b6ec0ced05862639a964370ce30986012bc8086ddc56d57ca2be6`
- Parent transpose helper: `f969041b0b149dcfef4da65465df6ee47397299124c6658e56b4fa3a3a77773d`
- Composed CUDA: `c49ad6b30e45a4b36b3bbcfadcced40cd54fd97e60e596b94520d49e011d34bd`
- Binary: `a7bda4ad3e93d701499bf2db4a0b1b144a71d0bfa6612cf717a3c1e68eb751ee`
- Raw log: `cd869092aa949d0f36aaca26c37be48010f2aa58b961f49e506f8eeb585b811c`

No production/Fixed/SM120 change or full/other-toolkit qualification.
