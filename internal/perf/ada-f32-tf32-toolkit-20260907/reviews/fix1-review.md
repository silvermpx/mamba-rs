# Task7 fix1 scoped re-review

## Review scope and basis

This is the requested fix-only re-review of
`ada-f32-tf32-toolkit-task7-fix1-review-package.diff` against the three
accepted Important findings. I did not repeat the original six-file review or
rerun a build, GPU workflow, Cargo command, or test suite. The six current
source hashes and formatted composite hash match the controller-supplied
final5b identities; the fix report itself matches SHA-256
`1ebf760b3ce46087928991d3b8e5082ea4be287a2d5051f7542e87d27eeb80f0`.
Final5b build/smoke closure remains an evidence-stage matter and is not part of
this source verdict.

## Finding verdicts

1. **ADDRESSED — numeric ABI 5 and schedule 8 are rejected and bound.** The
   live `CompilerIdentity` values now enter `prelaunch_gate`, which rejects
   anything other than numeric ABI 5 / schedule 8 before `single_term_controls`
   and before `Case::new` performs per-literal allocation
   (`tests/support/fixed_sm89_toolkit_admission.rs:1235`, `:1306`, `:1369`).
   Both actual values are emitted in the unique identity record (`:1313`) and
   independently required by the analyzer
   (`internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py:181`). Separate
   missing/wrong field negatives are present for both revisions
   (`internal/perf/ada-f32-tf32-toolkit-20260907/test_validation.py:125`), and
   the Rust prelaunch test separately rejects wrong numeric and wrong schedule
   values without invoking its launch closure
   (`tests/support/fixed_sm89_toolkit_admission.rs:1993`).

2. **ADDRESSED — confirm artifact identity is checked before the first
   workflow launch.** The live Fixed artifact digest is obtained and compared
   to `screen_artifact_sha` inside `prelaunch_gate`; only the gate's success
   closure can call `single_term_controls`
   (`tests/support/fixed_sm89_toolkit_admission.rs:1304`). Per-literal
   allocation and launches remain later at `:1357`. The focused negative uses
   a counted closure and proves a screen/live artifact mismatch leaves the
   count at zero (`:1993`).

3. **ADDRESSED — complete inputs and guards are checked at the timing
   boundary.** `Case::timing_boundary` checks the complete saved A/B/bias
   allocations (including their guards and common pointers) and calls
   `words()` on every output to check output guards
   (`tests/support/fixed_sm89_toolkit_admission.rs:708`, `:724`; guarded-canary
   check at `:539`). It is invoked immediately after the timed observation
   loop and before the first post-timing output read, eager replay, or graph
   replay (`:1514`, `:1527`). The later `case.inputs` check is retained at
   `:1547`. The sequencing regression covers separate A, B, and bias
   mutations and proves a restoring replay is not invoked before rejection
   (`:2016`).

## New breakage in fix diff

- **Minor:** `NUMERIC_ABI_REVISION` and `SCHEDULE_REVISION` were newly imported
  into `tests/gemm_bi_fixed_performance.rs:31` but are unused. The supplied
  focused GREEN log records the corresponding compiler warning
  (`internal/perf/ada-f32-tf32-toolkit-20260907/cuda128-fix1-green/host.log`).
  Runtime behavior is not affected: the gate reads the actual compiler fields
  and compares them to the required literal values. This warning is cleanup,
  not a screen21 blocker.

No new Critical or Important breakage was found in the fix-only diff.

## Out-of-scope observations

Final5b matching build/smoke artifacts, analyzer closure, and release telemetry
were still being produced during this review. No screen21 or confirm101 result
is reviewed or implied here.

## Verdict

**All three findings are addressed, with no new Critical or Important
breakage.** The single new Minor unused-import warning is nonblocking. This
accepts the corrected source gate before screen21; it does not accept final
performance admission or any AUTO promotion.
