# Ada F32 TN d128-in direct fold: short discovery, 2026-09-08

Target `(M,K,N)=(1024,128,512)`, CUDA13.2 / CC8.9 / 142 SMs only.
Test-only adaptation of the existing underfill direct kernel; no production
CUDA or RTX5090 route changes. Both candidates eliminate the 16 MiB partial
buffer and second reduction launch. They preserve each ascending 16-FMA FP32
chunk and the ascending 64-chunk FP64 fold, alpha scaling, final FP32 cast,
and addition to the same nonzero initial C.

One exact test passed in 7.31 seconds:
`cuda_tournament::ada_d128_direct_fold_two_arm_once7`.
Candidate eager/graph repeats and every timed output match the actual public
SplitM64 route bit-for-bit. Candidate buffers have two-sided guards; public
inputs are checked unchanged. Public C uses the pre-existing unguarded fixture;
this remains a discovery limitation, not a production qualification claim.

| Candidate | candidate/AUTO median range | candidate/Fast median range |
| --- | ---: | ---: |
| M16N16, 64 threads | 0.3813–0.4644 | 2.5625–3.0086 |
| M8N32, 64 threads | 0.4300–0.5012 | 2.5789–3.6658 |

Ratios below 1 are faster. Ranges cover eager/graph × ABBA/BAAB, seven paired
brackets per stratum. M16N16 is retained for the finalist batch: about 54–62%
less observed time than public AUTO, but **still substantially behind Fast**.
Fast is explicit F32-storage `cublasGemmEx` / `COMPUTE_32F_FAST_TF32`, TN,
alpha=beta=1. It has its own repeat-bit oracle, not our exact F32 contract.

These event measurements contain one logical GEMM per observation, including
launch-path overhead; graph samples visibly vary. They establish a large
discovery gain, not a precise kernel-only speed ceiling. A targeted profile
is the next step before choosing another tile hypothesis.

Resources: M16N16 105 registers / 4096 dynamic shared bytes; M8N32 107 / 5120.
Both have zero local/static shared bytes, 256 CTAs and reported occupancy 8.
Only ~1.8 CTAs per SM are available globally, motivating the occupancy check.

Main SHA `8491c861ff67d4027d2953a9b9233906d1cd4c92b5f579deee14496f88bbea6c`;
adapter SHA `06e866e6d19999fc90fd85cdb6c6f47b817676b235569af5aed17e0cca80309a`.
Binary SHA `0060f788d46e8a5bcf251cba047a6a6ed86e129198df062206d1b5ccc0e607fd`.
Raw SHA `10e419ab0a59662278d915d3fab01f1b59e966e30859cd954db662ac16e03e72`,
at `evidence/final/once7-cuda132/test.log`; command and telemetry are adjacent.
Root independently recomputed all 112 brackets / 448 timed observations.
Private cache was unchanged and post-run drain quiet. Initial compiler E0308
is preserved under `evidence/build1/`; the checked u32 launch-size repair built
successfully in `evidence/final/build2.log`.

No once21, all-toolkit, full-suite or production promotion run is claimed.
