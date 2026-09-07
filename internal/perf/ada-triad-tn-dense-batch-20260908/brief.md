# Three-cell TN dense-stage discovery batch

Base0e1d0ce4, same branch/worktree. User requested broad Triad performance
work: TF32 source work here, independent F32 and BF16/F16 audits in parallel.
No production change or admission, and no compact4/8 retry.

One new mechanism: bypass per-copy bounds/clamping/address-safety decisions
on full TN128x64/BK32/S3 stages. Retain padded79872-byte layout,256threads,
original four compute warps, RNA conversions, ascending MMA order, barriers,
commit/wait sequence and old-C FMA. Partial stages and unaligned operands
retain the unchanged generic path. Actual shader header is test-only.

Three logical MKN targets share one candidate source/symbol:

| Cell | MKN | Grid |
|---|---|---:|
| d768 in | 2048,768,3072 | 288 |
| d768 out | 2048,1536,768 | 144 |
| Prism | 4621,384,1928 | 93 |

Reference correction before any run: the recorded actual AUTO for both d768
TN cells is M64N64/S2 (grid576/288,128threads,36864shared), not M128N64/S3.
Root independently checked the exact physical records in
`../ada-triad-state-20260907/run-cuda132/auto60.log`; the harness must assert
the same identity on the live run. Prism AUTO is M128N64/S3. Therefore a
d768 improvement cannot be attributed solely to removing predicates: those
comparisons also change tile/stages versus actual AUTO. All remain direct RNA
with unchanged per-output reduction order; actual raw-bit tests still gate.

Hypothesis comes from existing NCU integer/predicate counts and prior NT
dense-copy wins, not proof of a removable entire Fast gap. Original compute
and pipeline bodies are preserved. Primary background:
https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
and existing `../ada-triad-two-cell-ncu-20260907/report.md`.

Root owns tests/support/triad_tn_dense_source.rs and
tests/gemm_bi_tf32_tn_dense_copy.cuh; GPU owner owns only the existing
discovery harness and this directory's runner/raw/report. Independent reviewer
is read-only. No competing GPU executor, new branch, push or deletion.

One build/list, then exact candidate bit/resource/once7 tests on CUDA13.2.
Resource gate0local/static,79872dynamic,256threads,occupancy>=1; record registers.
Retain full-mantissa target/tail/alpha eager2+graph2, finite K0 oracle,
immutable inputs/guards. Each beta1 timing observation resets C+A+B before
one GEMM/event. Seven brackets in eager/graph xABBA/BAAB; retain only when
both p50 and p95 candidate/AUTO<.99 in all four strata. Capture actual AUTO
identity/grid; decline mismatches instead of silently using another reference.
PRE/raw RELEASE/separate quiet DRAIN; exact source/binary bindings. Compare
survivors to Fast and qualify on all toolkits in a later batch, not per tweak.

Root native composer: two genuine behavioral RED failures, then2/2GREEN.
The actual-header host replay uses CUDA-instruction shims, so it proves copy
addresses/coverage, not CUDA code generation or performance. It verifies
40608 full stages,525 tail decisions,62373888 unique/aligned16-byte copies,
plus negative reduction/alignment/stride guards across all three shapes.
Run from worktree root: `clang++ -std=c++17 -O2 -I . internal/perf/ada-triad-tn-dense-batch-20260908/native-copy-replay.cpp -o <temporary-binary>`.
Source composer30d7eefd, header0a947ce3, composedCUDA36b18377.

User's autonomous fast-search instruction authorizes this bounded batch
without a new approval pause. Safe isolated measurement may overlap read-only
review; acceptance requires review. No weakened numerical contract or inferred
cross-GPU speedup. Preserve all losing results for later deliberate cleanup.
