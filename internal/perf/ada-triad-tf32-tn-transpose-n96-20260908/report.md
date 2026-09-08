# Ada TF32 TN transpose + RNA-N96 screen

CUDA 13.2 ran the exact ignored test
`cuda_suite::ada_tf32_tn_d768_in_transpose_rna_n96_discovery_once7` and
passed. The frozen main/helper SHA-256 values were `c009c10f8500a6f9e24018a0bcf93eb7037268a752a818ac5b649e99b0a975ef`
and `f969041b0b149dcfef4da65465df6ee47397299124c6658e56b4fa3a3a77773d`;
the binary SHA-256 was `ffc0c291abeb67cd6c5f056d64560f83076293923202faab83acd6f6f610b214`.

The transpose used 4,224 static shared bytes and 0 local bytes. The N96 GEMM
used 128 registers, 0 local/static shared bytes, 86,016 dynamic shared bytes,
256 threads, and occupancy 1. Its strict target, tail, exceptional, K0,
input/scratch, guard, eager, and graph checks passed before timing.

The full two-node transpose + GEMM pipeline was materially faster than actual
AUTO: p50 ratios were 0.6998--0.7064 and p95 ratios were 0.7073--0.7187,
roughly 29--30% less time. Candidate observations were about 192--203 us,
versus roughly 272--288 us for AUTO. The observed AUTO was M64N64/S2 on grid
576; the candidate GEMM used grid 192.

It did not beat Fast. Eager p50 ratios were 1.0104 and 1.0592, with p95
1.0403 and 1.0847. Graph p50 ratios were 1.3440--1.3451. The result therefore
retains against actual AUTO but is a valid `stop_no_retry` against Fast.

The exact log is `evidence/cuda132/run1/test.log` (SHA-256
`3cd94f8868195fd091aaf27ac8176ee43f67852c376428032bea3e756cb6a309`).
Root independently replayed all 56 timing brackets / 224 observations,
quantiles, arm mappings, decisions, cache stability, and quiet-process data.
Native helper tests4/4 pass; the frozen combined native harness passed48/48.
Timing uses1 logical GEMM per observation, explicit256B pointer checks and
independent eager/graph ABBA/BAAB once7. Resets and readbacks are outside
the event interval; both transform and GEMM are inside it. The transform
workspace is checked word-for-word including zero padding in preflight.
Exact source snapshots are retained in `evidence/source/`. No public
dispatcher admission, full toolkit matrix, or new5090 result is claimed.
