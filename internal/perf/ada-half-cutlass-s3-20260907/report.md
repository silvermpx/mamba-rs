# Task 5B — Ada half S3 result

## Decision

**RETAIN BOTH DTYPES: CUDA 13.2 exact-B0 production improvement confirmed by fresh paired21 and paired101.** No production integration was performed in this task. Native-half cuBLAS remains faster. BF16's final p95 margin over production is only 0.36%; retain the measured improvement while treating that small tail margin explicitly in subsequent qualification.

Only the final harness and evidence under `internal/perf/ada-half-cutlass-s3-20260907/final/` determine this result. Earlier timing files at the evidence-directory top level are **excluded incomplete-gate attempts**, even though their own ratios were favorable: review found that their graphs began from already-correct eager output and captured arguments were not decoded. The candidate kernel did not change during the harness correction.

## Final raw-recomputed results

Exact shape `(M,K,N)=(4621,768,2304)`, bias false, alpha1, beta0; SM89 RTX 6000 Ada, CUDA runtime13020/NVCC13.2.51. Ratios use paired CUDA-event measurements, not ratios of pooled arm medians.

| dtype | windows | candidate / production p50 | p95 | candidate / native-half cuBLAS p50 | p95 | own gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 21 | 0.965950168044 | 0.994138802505 | 1.263909922796 | 1.284298294062 | PASS |
| F16 | 21 | 0.964494892399 | 0.988817385023 | 1.192499985178 | 1.221832813535 | PASS |
| BF16 | 101 | 0.964968888968 | 0.996412833347 | 1.267528694145 | 1.290857051084 | PASS |
| F16 | 101 | 0.964326220305 | 0.981133357944 | 1.187129654181 | 1.221770514572 | PASS |

Paired101 production/cuBLAS controls: BF16 p50/p95 = 1.325109892201 / 1.372023142637; F16 = 1.254395345602 / 1.285125359294. The S3 improvement relative to production is 3.50%/0.36% at BF16 p50/p95 and 3.57%/1.89% at F16 p50/p95.

Every cell used an exact-UUID idle/no-apps wrapper, 128 eager warmups per arm, 20 graph operations per sample, alternating ABBA/BAAB, and alternating comparison ordering. Each 21 cell contains 252 unique samples and 63 pairs; each 101 cell contains 1,212 unique samples and 303 pairs. The analyzer checks key ranges and uniqueness, shape, arm/order identity, positive finite timings, emitted pairs against sums of raw arm observations, and emitted median/nearest-rank p95 against independently recomputed quantiles.

The timed vendor uses homogeneous native-half inputs/output, `CUBLAS_COMPUTE_32F`, `CUBLAS_DEFAULT_MATH`, `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, and atomics disallowed. PEDANTIC exists only in the untimed F32 numerical reference.

## Scope, source identity and implementation

Worktree: `internal/worktrees/gemm-bi-triad-sm80`; branch `codex/gemm-bi-triad-sm80`. Production source baseline is `b508805e81ed0f3edb70dacbc0c626d3342d747b`; documentation HEAD observed was `9c74e1bfb37b3f25dd0de4419d9eda99004e40ff`. The production baseline diff for the compiled prelude/common/kernel/layout files is empty.

The candidate derives from validated E1. It preserves CTA128x128/BK64, eight 64x32 warps, block256, packed/XOR per-stage layout, 16-byte .cg copies, fragment double buffering, four ascending k16 issues, FP32 bias seed and accumulator order, alpha/beta arithmetic, explicit RNE output and output ownership. No split-K, atomics, work counter, copy-plan redesign or arithmetic shortcut was introduced.

S3 allocates 98,304 bytes: three 16,384-byte A slots followed by three 16,384-byte B slots. The 69,632-byte vector epilogue aliases this storage only after all reads retire. Scalar/misaligned staging remains the existing two-slot path with the corrected B base.

Aligned prologue separately commits up to two real slabs. K0 does not copy, commit or load; K64 uses initial wait0; two-or-more slabs use initial wait1. While consuming tile t, tile t+2 is copied into its three-slot ring position with slices 1/1/2 across issues 0/1/2. Only a real refill commits. A transition with refill uses wait1, otherwise wait0; the barrier and next issue0 load precede current issue3 MMA. Read/write indices use bounded increments. Last tile performs no refill/transition.

Official design reference remains the prior practices note E2 and CUTLASS revision `59e3a3338d516ca6ce0e073af8da65289678a35c`, `include/cutlass/gemm/threadblock/mma_multistage.h`, lines 360–604.

Final SHA-256:

- Candidate `candidate.cu`: `a90f504d6d2d0363908fd00d230263164105b41c6b5a6085a83ae130f2ce4222`.
- Harness `benchmark-final.cu`: `6548e67ea619588d5d5a92b91fcbf07282d13c6689036960438986902d6edf71`.
- Final binary `final/benchmark`: `1c1e050f8cc2827194ab1111fca12dc7653404a12a3166aaed9793993c2e67f1`.

Local and remote hashes match. Both final candidate SASS sections are identical to the initial S3 build; harness fixes did not alter generated candidate instructions.

## Host and physical gates

The schedule model checks 0..97 slabs, actual three-slot residency, ordered pending-group completion, four exact slices and one commit per real slab, no phantom slab, no premature read/overwrite, ascending tile-major MMA issues, next issue0 before current issue3, and empty pending writes before epilogue alias.

The initial deliberately incorrect wait1 schedule fails at one slab. A separate mutant with a correct prologue but wrong final wait1 fails at two slabs, event 19. The correct model passes the entire range. The production constexpr layout helpers were reused across all three slots: 6,144 copied chunks, 147,456 fragment halves and 2,304 conflict-free ldmatrix groups,98,304-byte input ring.

CUDA13.2 same-binary production/candidate physical results:

| property | production BF16/F16 | S3 BF16/F16 |
| --- | ---: | ---: |
| registers/thread | 182 | 178 |
| stack/spill stores/spill loads/local/static shared | all0 | all0 |
| dynamic shared | 69,632 | 98,304 |
| active CTA/SM | 1 | 1 |
| exact requested grid/block | 666/256 | 666/256 |

The amended register gate is architectural <=255 with zero stack/local/spills and unchanged one-CTA residency; production+8 is diagnostic, not a hard rejection. No register cap was forced.

Generated SASS contains 3 commits, 2 wait0 sites, 2 wait1 sites, 24 LDGSTS and 128 HMMA. The SASS analyzer finds commit, conditional wait0/wait1, barrier, next issue0 LDSM, then final-current HMMA. Initial K>64 branch selects wait1; the transition's real-refill predicate selects wait1 and its drain branch selects wait0. These counts are observations, not E1's fixed static census admission rule. The SASS analyzer's valid/missing-wait1/missing-drain/stack mutants passed their expected outcomes.

## Correctness, poisoned graph and captured parameters

Final BF16 correctness-only run passed 285 validation records, including 126 biastrue records, and emitted zero timing samples. Each of the four final timing cells repeated the same 285-record corpus before warmups. It covers K0/tails; aligned K64/128/192/256 prologue/drain; aligned longer slabs and ring wrap; odd strides; independent A/B/C misalignment; row/prefix/subviews; guard regions, inactive rows and padding; unchanged A/B/Cold/bias; ordinary and exceptional values; bias-seeded arithmetic; nonunit alpha; beta 0.5 reset semantics; eager repeats and graph repeats.

Every independent eager repeat is freshly poisoned. Before each graph replay the harness uploads the bitwise complement of every golden half output, preserving 0xff guards, reads it back and verifies every logical output differs. After replay, production and candidate must independently match golden raw bits and all guards/inputs must pass. Existing captured reset-old nodes preserve the beta!=0 semantics. Skipping **only the candidate graph** via the negative-test environment flag causes a candidate BIT PARITY failure at K0; `final/graph-overwrite-red.log` records that expected failure.

Requested graphs contain 20 nodes at grid666/block256/shared98,304 and exact candidate function pointers. The source declares five arguments; the harness decodes captured C/A/B/bias pointers and the complete 32-byte scalar parameter struct with memcpy and checks their expected values. Records correctly call this `source_abi_args=5`, `captured_arguments_checked=true`, rather than a measured driver argument census.

The final raw-data analyzer accepts biastrue correctness rows only with all genuine passed flags; the requested timed row remains biasfalse. Eighteen valid/adversarial fixtures passed, including absent poisoning, unchecked arguments, wrong requested bias, missing/duplicate/out-of-range samples, wrong pair/comparison order, zero time, forged ratios/quantiles, wrong shared/ABI/residency, spills, and consistently rebuilt production/mixed-p95 losses.

## Evidence map and excluded attempts

All paths below are relative to `internal/perf/ada-half-cutlass-s3-20260907/`.

- `candidate.cu`, `candidate-vs-e1.diff`: candidate and exact E1 delta.
- `schedule_model.h`, `schedule_test.cpp`, `schedule-red.log`, `schedule-green-final.log`, `schedule-mutant.log`, `schedule-mutant-drain.log`: host schedule proof.
- `layout-test.cpp`, `layout-green.log`: actual production layout across S3.
- `benchmark-final.cu`, `run-final-cell.sh`, `final/benchmark`: authoritative final harness, idle wrapper and binary.
- `final/compile-cuda132.log`, `final/resource-usage.txt`, `final/benchmark.sass`, `final/sass-proof-v2.json`, `final/kernel-sass-identity.log`: authoritative physical evidence.
- `final/correctness-bf16-cuda132.log`, `final/correctness-proof-v2.json`, `final/graph-overwrite-red.log`: final correctness and candidate-only no-op negative control.
- `final/paired{21,101}-{bf16,f16}-cuda132.log` and matching analyzer logs: four authoritative cells; `final/data-integrity-summary.json` summarizes record counts and graph identities.
- `analyze-final.rb`, `test-analyze-final.rb`, `final/analyzer-tests-v2.log`; `analyze_sass.rb`, `test-analyze-sass.rb`, `final/sass-analyzer-tests.log`: analyzers and adversarial tests.
- `final/local-source-binary.log`, `final/remote-source-binary-toolkit.log`, `final/production-baseline-diff.log`, `SHA256SUMS`: rooted source/binary integrity.

Top-level initial paired logs, `benchmark.cu`, and the initial binary are excluded from admission because their graph overwrite gate was incomplete. `benchmark-final-v1.cu`, `final/benchmark-v1`, and `*-v1.log` preserve the intermediate poisoned-graph harness before captured-parameter validation; that intermediate harness produced no admitted timing. The initial missing-prelude compile error is retained as diagnostic evidence; the explicit prelude sync fixed the build recipe without changing source. No evidence was deleted.

## Handoff and lane release

No commits, staging, branches, production edits, cross-toolkit qualification, or integration were performed by this worker. Root owns subsequent review, cross-toolkit qualification and dispatcher integration.

Final telemetry at **2026-09-07T01:40:54Z**: exact UUID `GPU-d1edd7be-e88d-aed6-047d-622163306f0e`, 0% GPU utilization, 0% memory utilization, 42 C, P5, no compute applications; final binary hash unchanged. Ada was explicitly released to root immediately. Remaining packaging was host-only.
