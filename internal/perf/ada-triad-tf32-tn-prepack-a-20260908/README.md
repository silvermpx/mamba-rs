# Ada TF32 TN: A-only prepack rejected

2026-09-08, CUDA13.2 / RTX6000 Ada. Packs A once with exact RNA conversion,
then preserves the existing GEMM math; BOTH pack and GEMM are timed in eager
and a two-node candidate graph. Packed scratch is reseeded before every
observation. Logical pointers256-byte aligned. No production changes.

| TN cell | Candidate/actual AUTO p50 | Candidate/Fast p50 | Decision |
| --- | ---: | ---: | --- |
| d768-in | 1.126–1.145 | 1.805–2.182 | Stop |
| Prism | .998–1.016 | 1.880–2.384 | Stop |

Ranges cover eager/graph ×ABBA/BAAB, once7. Prism is not a win: graph loses
and eager p95 exceeds1. Candidate144 registers/local0/shared79872/256threads,
occupancy1. Finite/exceptional/tail/alpha/K0 bit checks against true AUTO pass.
Fast is explicit cuBLAS Fast with its own eager/graph bits; vendor graph is
timed in full, without imposing our node count or private kernel-node ABI.

The first valid aligned d768-in cell completed before the two-cell wrapper
rejected its own retained CUDA context at the second PRE gate. Preserve that
valid cell; do not rerun it. A separate Prism-only exact test then PASSed.
Earlier identity/graph/alignment harness failures are retained as failures,
not candidate verdicts. Root independently replayed both valid cells:
112 brackets/448 observations and32 p50/p95 values, PASS. Native tests39PASS.

- Final main `476a1c1f7b880bce5047c6a5b53c0fc38a0e2f5452dc1865fb3363352f9ee357`.
  Helper `26f8a84619c7c13c3506f58d09f8e2f6d8b8a130f8e86c1b5f26c801c5325cc2`.
- d768-in [raw](evidence/once7-cuda132-align-repair/test.log), SHA256
  `aabeae4ce019c6d0e9b75000fa9dd85b5c12d20ec7f16512898a61991d3afa44`.
- Prism [raw](evidence/once7-cuda132-prism-only/test.log), SHA256
  `b29edac463d4559c490e09c71a3f5694fd2bdec3b2f661ee073f9f73b794905c`;
  [exact command](evidence/once7-cuda132-prism-only/command.json).

Both mechanisms stop here; no tile resweep or rerun of these valid losers.
