# Task8 source review — exact toolkit AUTO44 and post stage

Date: 2026-09-07

## Verdicts

- **Spec compliance: CHANGES REQUIRED.** The production predicate, epoch
  transition, and post-stage Rust mechanics match the bounded Task8 design,
  but the frozen runner cannot complete the required all-toolkit functional
  gate and the frozen analyzer cannot accept the binding that the same runner
  writes.
- **Task quality: CHANGES REQUIRED.** I found no Critical defect and no defect
  in the narrow production route expansion, but the two Important integration
  defects below block source-gate acceptance and all later evidence closure.
- **Recommendation: REJECT THIS SOURCE PACKAGE PENDING ONE SCOPED FIX.** Repair
  the binding interface and select the Fixed RNA functional tests, then submit
  only that fix for re-review before any smoke1/post101 timing.

## Findings

### Critical

None.

### Important

1. **The analyzer rejects the real binding emitted by the Task8 runner.** The
   build operation writes the four executable digests only under the
   `binaries` map (`internal/perf/ada-exact-toolkit-auto-20260907/run.py:250-278`).
   A run correctly records the selected performance executable digest in
   `result.json` (`run.py:355-366`, `:393-403`), but
   `analyze.py:158-166` compares that value with
   `binding.get("binary_sha")`. The emitted binding has no top-level
   `binary_sha`, so the real `analyze.py RUN_DIR BINDING` path always compares
   the valid identity/result digest with `None` and raises `binary binding
   differs`. The five-test validator misses this interface failure because it
   constructs a synthetic binding containing a top-level `binary_sha`
   (`test_validation.py:130-148`) instead of exercising the builder's actual
   schema. Align the builder/analyzer contract and add a regression using the
   real `binaries`-map shape (including proof that the measured performance
   binary, rather than one of the other three executables, is bound).

2. **The all-three-toolkit functional lane runs Triad-only cohort assertions
   in place of the required Fixed RNA retained gates.** The functional list
   chooses `tf32_cohort_binds_on_this_board` and
   `sm89_tf32_bias_cohort_serves_the_qualified_wide_epilogues`
   (`run.py:295-344`). Both tests explicitly select
   `BiGemmFamily::Triad` (`tests/gemm_bi_tf32_cohort_binding.rs:20-25`,
   `:87-94`) and require the Triad SM80 TF32 body. That historical cohort is
   CUDA 13.2-only; CUDA 12.8/13.0 intentionally fall back, so these commands
   cannot be an all-three retained Fixed gate and should fail on the two newly
   admitted toolkits without exposing a production regression. The matching
   Fixed tests are
   `fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph`
   (`tests/gemm_bi_fixed_correctness.rs:124-126`) and
   `fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits`
   (`:583-599`); both explicitly accept only 12.8/13.0/13.2 and exercise the
   Fixed RNA route. Select and bind the appropriate Fixed correctness binary
   for these checks while retaining the Triad cohort's 13.2-only meaning in
   its proper host/qualified context. Do not widen the Triad production route
   to make the current functional list pass.

### Minor

None.

## Verified bounded requirements

- The frozen source diff has SHA-256
  `4532035215b307f0e3a742b6274fb5a1a7cdf7bc2475aba62fd583fc02823d95`;
  the immutable checkpoint report has SHA-256
  `5c2bf338a0bf49a3f68ef73391f041919ebed05ae17c261958e354793d8f43f8`.
  All nine Rust and four Python file hashes match that checkpoint. The
  independently recomputed NUL-framed four-file source digest is
  `2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285`.
  A scoped `git diff --check` on the 13 files is clean.
- `fixed_sm89_exact_n64_auto_eligible` adds exactly the finite known NVRTC
  versions 12.8 and 13.0 beside retained 13.2 while preserving the existing
  CopyPlan holder, exact-policy, homogeneous-F32, device, shape, pointer, and
  alignment guards. The positive matrix covers the 24 supported
  toolkit/literal rows and keeps the required device/version/holder/policy/
  dtype/alignment/adjacent/thin negatives.
- The only runtime epoch change is tuning revision 43 to 44. Current fixtures
  move to 44, captured historical values remain historical, a captured-43
  identity is rejected against current 44, and current acceptance plus the
  Fixed-artifact mutation rejection remain.
- The existing exact AUTO prefix/view test is generalized to the exact finite
  version set 12.8/13.0/13.2 without dropping its Legacy reference, offset
  views, special values, or eager/graph bit checks. Half, TF32/RNA, and SM120
  production routing are unchanged by the package.
- The historical Task7 ignored entry
  `fixed_ada_forced_rungs_paired_precision_cublas` remains present. Adding the
  distinct thin wrapper
  `fixed_ada_exact_post_auto_paired_precision_cublas` is appropriate: it
  dispatches only the new explicit post mode and does not alter the retained
  Task7 entry.
- Post mode uses schema `MambaBiFixedAdaExactPostAutoV1`, revision 44, exact
  eight-literal controls, toolkit 12.8/13.0, and smoke1/post101 windows. Arm 0
  is explicit forced Legacy, arm 1 is public AUTO with the returned
  `F32Sm89N64CopyPlan` enum checked on every call and across calls, and arm 2
  remains Fast. Physical graph records preserve this actual ordering and map
  Legacy/AUTO/Fast to `physical[0]/[1]/[2]` consistently.
- Numeric ABI 5, schedule revision 8, actual compiler/toolkit/target, source,
  binary, and Fixed artifact checks occur before the first single-term or
  per-literal kernel workflow. The post identity carries the frozen Task7
  promotion basis without requiring the old Task7 source or binary to equal
  the new revision-44 build.
- The Task8 analyzer's in-memory adapter maps Legacy/AUTO/Fast and the three
  post directions consistently to the frozen Task7 arithmetic/physical
  validator. The frozen Task7 analyzer is an explicit support-tool dependency;
  no Task7 raw data or artifact is rewritten. Apart from the binding-shape
  defect above, the post validator retains record cardinality, chronology,
  quantile, physical, telemetry, and attempt-closure validation.
- The timing path retains full input/output guard checks at the timing
  boundary before any restoring replay, complement poison/readback,
  one-/twenty-operation graph inspection, exact-bit controls, alternating
  chronology, both paths and start parities, and the required
  `AUTO/Legacy` p50-and-p95 owner rule.

## Cannot verify in this source gate

- Matching GPU functionals, smoke1, and post101 were pending or deliberately
  held during this review. I did not run or accept any GPU, SSH, Cargo, build,
  or timing result. Future runtime evidence cannot cure either deterministic
  source-tooling defect above; the functional lane and analyzer contract must
  first be corrected and source-gated.
- I did not independently prove external Ada exclusivity or remote cache and
  binary provenance. Those are final-evidence matters after an accepted source
  and matching binaries.

## Review basis

This review is limited to the binding Task8 brief, the immutable checkpoint,
and the one-pass frozen 13-file source package against base
`b78aebf466b72428eaaaf2654a7e671949b62d20`. Unchanged source was inspected
only for the concrete runner/analyzer and functional-test interfaces named
above. I did not repeat Task7 review, broaden into the earlier Triad audit, or
change implementation, index, branch, or HEAD.
