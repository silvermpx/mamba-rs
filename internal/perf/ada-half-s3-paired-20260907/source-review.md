# Task6B frozen source review

## Spec compliance

**Issues found.** The literal experiment, mirrored timing, physical argument inspection, numerical/storage gates, exact timing cohorts, and own-win arithmetic match the brief. Evidence exit/attempt closure needs correction (Important I1). Two explicit preflight requirements are incomplete (Minor M1/M2).

**Pending evidence, not source defects:** the controller reports all three final builds passed focused 3 tests and the nonignored suite (51 passed/64 ignored), all three functional smokes closed 8 configurations/96 observations/24 pairs, and all three screens and fresh eligible 101 runs have completed; raw mirroring/analysis is underway. I did not rerun those checks or inspect their runtime outputs. Final six-cell admission, the screen-to-confirm eligibility linkage, rooted archive/manifest, failure history, and exact-UUID idle/no-apps release remain controller verification items. This report grants no runtime admission.

## Strengths

- `tests/gemm_bi_fixed_performance.rs:44`, `:94`, `:595`, `:643`: both starting parities reverse comparison traversal and the four-position ABBA/BAAB bracket; eager uses 20 real public launches and graph uses one 20-operation replay; emitted chronology and B/A sums follow the requested protocol.
- `tests/gemm_bi_fixed_performance.rs:185`, `:221`, `:468`, `:488`: AUTO binds the actual public return to revision-42 Swizzle, candidate uses public forced S3, Fast requests native-half compute-32F/tensor-op with explicitly queried math/pointer/atomics modes, and PEDANTIC F32 is used only as the untimed numerical reference.
- `tests/gemm_bi_fixed_performance.rs:249`, `:447`, `:512`, `:562`: guarded aligned C allocations, actual captured pointer/bundle decoding for every custom one-op/20-op node, complement upload/readback, repeated independent graph/eager overwrites, and the empty candidate graph negative are substantive functional checks.
- `internal/perf/ada-half-s3-paired-20260907/analyze.py:94`, `:105`, `:136`: exact cohorts and chronological sample/pair/summary reconstruction are independent of the Rust scheduler. Each dtype uses all four path/parity strata; genuine losses and mixed p95 results remain valid and cannot advance. `test_validation.py:58` checks literal win/loss/mixed expected quantiles rather than merely reusing the analyzer's arithmetic.

## Findings

### Important I1 — bind exit and telemetry evidence to this attempt and parse exact closure records

`internal/perf/ada-half-s3-paired-20260907/analyze.py:147` and `:160`; corresponding emitted evidence at `run.py:55`, `:120`, `:124`.

`verify_run` accepts any supplied SSH text containing each of the three success substrings once. It never parses wrapper/SSH exit values, associates `RUN_RESULT` with the loaded `result.json`, or associates the SSH PRE/POST records with the loaded telemetry files. For example, a log containing one success marker of each kind followed by `WRAPPER_EXIT=1` and `OUTER_SSH_EXIT=255` still satisfies the predicate. An unrelated successful smoke SSH log also satisfies it when supplied for a different run. `run.closure(result['test_exit'], result['post_exit'])` supplies the default zero wrapper/SSH values, so it does not repair this gap.

This fails the explicit full wrapper/outer-SSH exit closure requirement and permits accidental evidence mixing to pass the source/binary/run checks. The existing tests at `test_validation.py:135` validate `run.closure` in isolation; they do not exercise these malformed or mismatched `verify_run` inputs.

Parse anchored, exact marker records, require one complete ordered closure with zero actual exits and no conflicting/failure markers, and bind its command/result/telemetry to this attempt (the existing emitted `RUN_RESULT`, PRE/POST JSON, command and command-exit records provide the necessary material). Check the telemetry phase labels too. Add host-only negative cases for conflicting/nonzero/duplicate markers, another attempt's SSH log, mismatched result/telemetry, and swapped PRE/POST records. Existing GPU timings need not be recollected if their original logs satisfy the corrected checks.

### Minor M1 — legacy-control rejection omits existing control families

`tests/gemm_bi_fixed_performance.rs:360`.

The allow/reject loop only rejects other `MAMBA_FIXED_ADA_*`, `MAMBA_FIXED_VENDOR_*`, and `NVIDIA_TF32_OVERRIDE`. Existing `MAMBA_FIXED_AUTO_VENDOR_ROW`, `MAMBA_FIXED_AUTO_VENDOR_CELL`, `MAMBA_FIXED_AUTO_VENDOR_BIAS` (consumed by the historical helper at `:10278`) and `MAMBA_FIXED_HALF_TILE_CANDIDATE` (`:5336`) are accepted. The brief explicitly requires rejecting stale legacy row/cell/bias/tile/path controls.

These variables are unused by this new literal harness, so their presence does **not** currently filter, mute, or redirect a timed cohort. This is a preflight contract gap, not evidence that the measured cells are wrong. Extend the rejection set to the existing legacy controls and cover it with a host-level parser/control-list test.

### Minor M2 — input immutability is first asserted after timing

`tests/gemm_bi_fixed_performance.rs:569`, `:586`, `:620`.

Initial A/B bytes are saved at `:444`, but the first comparison against them occurs after the complete first configuration's timed windows. The pre-timing output/graph checks do not assert input immutability. Thus a correctness replay that alters A or B can reach warmup/timing before the eventual immutable-input assertion rejects the run. The brief/audit require pre/post input gates and failure before warmup on functional-gate errors.

Check both input snapshots after the pre-timing correctness replays and before warmup in each configuration; keep the current postchecks. The present postcheck prevents such a run from becoming accepted evidence, which limits the severity.

## Review boundary and checks

- Read the complete Task6B brief, harness audit, interim report, and all 1,204 lines of frozen package `ada-half-s3-paired-task6b-source-review-v1.md`, in successive bounded sections. Applied requesting-code-review and SDD task-review discipline; no subagents.
- Frozen base is `0892986d6382fc00f5012e3f58b750bdb37383fb`. Read-only SHA256 checks confirmed the working source and all three utilities match the package: Rust `f43a2a22dea7435716ff0c99d721e1d255ad9311f18e9f0b53a15280eafb582b`; run `ce0de6882fa45835ba81b90d02a6cc085ed32226e2462b72260ce9cacd0fd713`; analyze `5427291e7315c9244ce045fc13d4582f183de189b3c4cb25d8782e1ebd3a1eaa`; validation tests `a012349c20fc1583de0405d153eba6b49a8e00fc834c542ae69653c20fdde520`.
- Focused reused-helper checks addressed named risks only: public AUTO versus surrogate (`:9900`, `:9916`); reference storage/compute/beta (`:10321`); event normalization/boundaries (`:10377`); finite numerical tolerance and dtype row order (`:11332`, `:11426`); exact physical ABI/bundle checks (`:12077`); synchronized full-allocation readback (`:12318`); quantile convention (`:793`); preflight scope (`:7879`). A targeted root build-input-path check found no `build.rs` or `.cargo/config.toml` requiring addition to the wrapper's manifest. A targeted legacy-control search substantiated M1.
- No SSH, GPU, builds, test reruns, source edits, staging, commits, branches, or broad project crawl. Only this report was authored.
- Bounded follow-up: reviewed the complete test-only delta from the frozen `test_validation.py` to SHA256 `29da83368a39fb7558a19f1344f38f56895e0b16e023e74923022249b291c6f5`, as requested by the controller. It extends the independent fixtures to both 21 and 101, uses the correct literal median/p95 indices (10/19 and 50/95), and checks `advance101` versus `admission` separately for true win/loss/mixed outcomes. No new finding. The earlier test references above use frozen-package line numbers; Rust/run/analyze are unchanged.

## Task quality

**Needs fixes for I1.** The measurement and numerical/physical checks are strong; acceptance must bind every exit layer to the same run before the evidence can be trusted as an independently reproducible record.

Preserving existing clean-environment timing evidence across a control-list-only source fix is reasonable: the new checks execute before GPU creation/warmup and cannot change the arms, graphs, timed intervals, or ratio calculation for already accepted environments. Preserve the original measured source/binary binding and raw logs; qualify the revised source separately and never relabel old timings with a new binary hash. A pre-warmup input check likewise leaves timing code unchanged, but its additional functional assertion needs covering qualification. No source review can establish a fresh binary's exact performance equivalence; the final report should clearly state which source/binary produced the actual performance record.
