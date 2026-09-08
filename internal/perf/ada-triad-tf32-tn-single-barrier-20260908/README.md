# TF32 TN Prism single barrier: small AUTO gain, still loses Fast

Ada / CUDA13.2, 2026-09-08, (M,K,N)=(4621,384,1928).
Test-only M128N64/BK32/S3 variant removes the redundant bottom CTA barrier
for this specialization; keeps wait_group and the next top CTA barrier.
Copies, conversion, MMA order and epilogue remain unchanged. Source reverse
equality and ring-lifetime model pass; native main43/helper3 tests pass.

| Path | Paired candidate/AUTO p50 | Paired candidate/Fast p50 |
| --- | ---: | ---: |
| Eager | .9657–.9673 | 1.862–1.874 |
| Graph | .9847–.9853 | 2.314–2.323 |

1.5–3.4% below actual AUTO, but a valid STOP against Fast, not a Fast winner.
Retain evidence only; no retry/integration. Guard64 F32 keeps 256B alignment;
strict target/tail/exception/K0 and repeat eager/graph bits pass before timing.
Grid93, block256, dynamic79872, local0, occupancy1. Candidate times one GEMM;
cuBLAS times its complete nonempty graph without inspecting vendor ABI.

Exact `cuda_suite::ada_tf32_tn_prism_single_barrier_s3_discovery_once7` PASS:
1 resource / 4 AUTO screens / 4 Fast screens / 1 decision. Root independently
replayed 56 brackets / 224 observations and all quantiles. One GEMM per
observation with C+A+B reset before events; do not conflate with half20-op
screens. No sanitizers/full gates/cross-toolkit qualification at this stage.

Main `2587c6b634505710835c3822d642b2e53fa24c8e58ce975cc7f2df81b7cf4570`;
helper `0c6e772d1d58c51011f94ea93a0fbad3986409b05462ba860e8b48d8167b574a`.
[Raw](evidence/once7-cuda132/test.log), SHA256
`bd936573d8c8cae5bc81dfaade9926071382655e62813f3b50f2ae2fc0cadbff`.
