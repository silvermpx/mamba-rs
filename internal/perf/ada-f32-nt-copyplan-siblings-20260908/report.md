# Exact F32 NT sibling CopyPlan reuse — 2026-09-08

Two new measured improvements over actual public AUTO on RTX6000Ada/CUDA13.2.
Both are retained for joint integration, not yet dispatched and not Fast wins.
The candidate times the whole unchanged production transpose32x16 + Fixed
CopyPlan pipeline, with exact F32 arithmetic and alpha1/beta+0/no bias.

| Cell | Candidate us | Actual AUTO us | Candidate/AUTO p50 | Worst p95 | Candidate/Fast p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| d768-in (2048,768,3072) | 276–277 | about640 | .4317–.4331 | .4334 | 2.360–2.366 |
| Prism (4621,384,1928) | 301–302 | about346 | .8700–.8717 | .8722 | 3.915–3.920 |

Time reduction is56.7–56.8% and12.8–13.0% respectively, across eager/graph and
both paired orders. Fast is explicit `CUBLAS_COMPUTE_32F_FAST_TF32`, not
Pedantic. Its different numerical results are checked for self-repeat, not
used as the exact F32 oracle. No claim of beating Fast or of a silicon limit.

## Identity and correctness

- Public AUTO is `gemm_bi_nt`, grid96/block256/shared33,376 for d768-in;
  `gemm_bi_nt_slim`, grid222/block128/shared0 for Prism. Identities were queried
  from the real public-entrypoint graphs, not inferred from source availability.
- Candidate uses `gemm_bi_transpose_f32_32x16_d768_v1` then
  `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1`: grids(96,24)+384 for d768-in and
  (61,12)+438 for Prism. Both graph nodes are included in every observation.
- Fixed:135 registers,32,768 static shared,zero local/dynamic,block128,occ3.
  Transpose:18 registers,4,224 static shared,zero local/dynamic,block512,occ3.
- Full target output equals forced generic exact NT and actual public AUTO;
  two eager and two graph repetitions. Tail67x68x36, exceptional values and
  zero reduction also pass. Transpose preserves all B storage bits; input and
  output/scratch guards are checked. All timed pointers are256B aligned.
- Root replayed all112 brackets/448 event observations and nearest-rank
  p50/p95 using explicit `observation_arms`. Total:2 resources,56 bit records,
  14 graph records,16 screens,4 decisions. Twenty GEMMs per observation.
- Private artifact cache unchanged. Pre/drain quiet; release had no compute
  apps but5% residual utilization, not labelled quiet. Exact one test passed.

## Frozen evidence

The unchanged measured test is committed with this report:
`tests/gemm_bi_scalar_nt_copyplan_siblings_discovery.rs`, SHA256
`589aff760bf672f72f8d35b2aa8c0df60c725ca74569fc17308a7c8d8d99cbd1`.
Native geometry/ABI/ratio TDD was RED then GREEN3/3; focused CUDA release build
and test passed. No production CUDA source changed.

Binary: `ec76a1d7cc30a8a8a49b6cbb5bb64a4e200c9e7e65fd4f4dd02284d62493ab3a`.
[Raw log](evidence/cuda132/run1/test.log):
`287d58db6323f9f7d5ec345ee83441d1bbb5dfbb9fb6700d5608f6c2f48c72ca`.
Exact test: `cuda_suite::ada_f32_nt_copyplan_in_prism_discovery_once7`.
An initial shell list-parser typo was repaired before GPU work; the authoritative
exact-list check counted one test. No valid measurement was rerun for that typo.

CUDA12.8/13.0, cold-cache, dispatcher and broader admission remain part of the
joint finalist batch. Preserve SM120 and the older d768-out finalist unchanged.
