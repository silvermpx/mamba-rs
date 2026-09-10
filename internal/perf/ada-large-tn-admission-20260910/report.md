# Ada large exact-F32 TN admission

Date: 2026-09-10. Source baseline: `1cbd572c` on
`codex/gemm-bi-triad-sm80`. GPU: RTX 6000 Ada, CC8.9, 142 SMs.
This report records assembly of three retained kernels, not new discovery.

## Pre-admission evidence

All three existing production forced routes pass the expanded qualification
against the actual prior AUTO (`TriadScalar` aligned SplitM partial + reducer).
The CUDA bodies and their module composition are unchanged:

- Owner SHA256: `cdcb768216699f41553e73492a32d92717c62889a4a329ca1990360b361541c7`.
- Composed source: `b83eea55e9cced220366c8160934340f8e4f4a59f007d632dbbcbb24dd2a503c`.
- Pre-admission selector harness SHA256:
  `a32b1f5e75e21bc0893ec17d80f753221e695bcdd3d322f0abca6ced3fe3e7bd`;
  raw helper: `5deff79e9d80325aa654cc9db64cb2e3dcbae187c3e836a38f10f792306b62f5`.
- CUDA12.8: 1 passed, 0 failed, 37.61 s; raw log in `cuda128/pre/`.
- CUDA13.0: 1 passed, 0 failed, 38.66 s; raw log in `cuda130/pre/`.
- CUDA13.2: reuse the valid expanded B2 run, 1 passed, 0 failed, 40.96 s,
  at `../ada-sm89-exact-f32-b2-20260909/b2-cuda132-k0-fixed-run.log`.
  Its SHA256 is `0ca5ff3beb9b53d151c0935cb3dc29578e00c00c551730e265dc59fd5b884ecf`.

The lower-toolkit runs use the unchanged historical test name
`live::sm89_exact_f32_large_tn_forced_correctness_and_actual_auto_admission`,
with `MAMBA_SM89_EXACT_F32_EXPECT_AUTO=0`. The final harness renames this to
`live::sm89_exact_f32_large_tn_pre_admission_forced_vs_prior_actual_auto` and
provides a separate post-admission test; the original evidence is not rewritten.

Coverage: module identity/ABI/resources, forced eager and prepared physical
routes, full/tail/exceptional/non-unit-alpha/K0 raw oracles, scratch/transpose
words, repeated accumulation, exact timed-window outputs, immutable inputs,
two-sided guards, and 256-byte-aligned active allocations. Performance uses
once3 followed by once7, ABBA and BAAB ordering, eager and whole CUDA Graph.
Both lower-toolkit GPU preflight files contain five idle/free-VRAM samples.

Independent root replay reconstructed 24 screens, 120 brackets and 480 positive
observations per lower toolkit. Maximum numerical reconstruction difference is
1.12e-16; every p50/p95 passes the strict candidate/prior-AUTO <0.99 gate.
Details are in `pre-admission-replay.json`. Candidate/prior-AUTO once7 p50:

| Route | CUDA12.8 eager | CUDA12.8 graph | CUDA13.0 eager | CUDA13.0 graph |
|---|---:|---:|---:|---:|
| d768-in, dual-chunk fused | 0.7595–0.7598 | 0.7563–0.7582 | 0.7591–0.7595 | 0.7578–0.7581 |
| d768-out, direct BK16 | 0.7683–0.7719 | 0.7694–0.7700 | 0.7707–0.7724 | 0.7694–0.7695 |
| Prism, direct BK16 | 0.7797–0.7798 | 0.7771–0.7774 | 0.7805–0.7808 | 0.7777–0.7779 |

Ranges span the two ordering strata, not a confidence interval. These are
roughly 22–24% lower times than prior AUTO. No fresh cuBLAS Fast/Pedantic arm
was run; these results do not establish a vendor win.

## Admission and final checks

The three exact known compiler/artifact cohorts are enabled and final selector
and actual-AUTO qualification pass. The first new CUDA-gated selector
build found a missing `CudaTarget` import in the test module; its log is retained
under `cuda132/red-initial-compile/`. No GPU run or performance result was
affected. This compile failure is not the intended behavioral RED.

After adding the import, the exact selector regression produced the intended
RED: 0 passed, 1 failed; actual `TnSplitM { m_chunk:1024, chunks:2 }` differed
from required `TnD768InSm89DualChunkQualified`. The CUDA-gated post-admission
harness compiled successfully in 39.34 s. Evidence: `cuda132/red/`.

Final source freeze: dispatcher SHA256
`d2cf6ded1c743f316451b0417acc536220884954f8a5f3262f54faa746ccfd14`,
selector harness `a1ef5626f4c0fbf5b8351a1ab5b07bde30f7874ad01646429b05b60d08e1c222`.
The raw helper and CUDA owner remain unchanged. The post-admission test
`live::sm89_exact_f32_large_tn_post_admission_actual_auto_exactness` requires
`MAMBA_SM89_EXACT_F32_EXPECT_AUTO=1` and checks actual AUTO against the
independent raw oracle; it performs no self-comparison timing.

The first admitted selector run passed all nine positive toolkit/route pairs,
but its negative probe failed because increasing K without refreshing lda
constructed an invalid matrix. Independent review identified the same missing
coherent-neighbor coverage. The corrected test rebuilds contiguous shapes for
each dimension neighbor and for the wrong-op NT probe; separate stride tests
remain. This is test-only; production admission and CUDA arithmetic are
unchanged. The failed selector log is preserved in `cuda132/green-malformed-probe/`.

Final results:

| Check | Result |
|---|---|
| CUDA-gated selector tests | 2 passed; all nine positive identity/route pairs plus fail-closed probes |
| CUDA12.8 actual AUTO | 3 routes, eager/graph identity and bits, guards and immutable inputs; 1 passed in 5.63 s |
| CUDA13.0 actual AUTO | Same coverage; 1 passed in 5.93 s |
| CUDA13.2 actual AUTO | Same coverage; 1 passed in 5.95 s |
| Native library / qualification harness | 84 / 4 passed (implementer runs) |
| Independent source review | Spec and task quality approved after malformed-probe fix |

Root independently read all nine post-admission JSON receipts and confirmed
every exactness, repeat, input, guard and admitted-AUTO flag. Logs, exact source
hashes and five-sample preflight are in each `cuda*/post/` directory.
Only exact CC8.9/142-SM, loaded symbols and the three known compiler/artifact
identities are admitted. Other devices and unsupported identities retain their
existing selection; this is not a new RTX 5090 performance result.

The large-TN assembly task is complete. The separate two-cell d128 module/routes
and the combined final Inference/Triad release checks remain unfinished.
