# Ada RNA toolkit AUTO static review

## Scope

- Base and current `HEAD`: `c35646367b30acff74136520f1ad6e9cbb376b9e`.
- Candidate: six uncommitted source/test files, exactly 71 insertions and 31
  deletions.
- Review mode: read-only except for this ignored note. No build, test, SSH,
  GPU, process-control, index, tracked-file, or `HEAD` write was performed.
- Governing documents read: the full Ada RNA toolkit AUTO plan and the full
  Fixed RNA production integration spec. The committed toolkit-census and
  full-toolkit-qualification READMEs/reports were also inspected.

## Code against plan

No static plan-compliance issues found.

- `fixed_sm89_rna_wide_auto_eligible` changes only the finite runtime-NVRTC
  predicate from 13.2 to exactly 12.8, 13.0, or 13.2. The loaded-holder,
  known-library, CC8.9/142-SM, deterministic-TF32 policy, homogeneous-F32,
  non-null C/A/B16, bias4, and exact A-E shape gates are unchanged.
- The selector regression admits each of the three explicit versions and
  rejects 12.7, 12.9, 13.1, 13.3, and 14.0 while retaining all independent
  holder/library/policy/device/precision/alignment/shape declines.
- The global routing epoch is 41. Historical revision-39 and revision-38
  replay regressions remain, and a revision-40 replay regression checks that
  compiler, artifacts, numeric contracts, and schedule identity stay equal
  while replay is rejected.
- Every changed current-epoch pin is an epoch assertion/name. This includes the
  half test's `(NUMERIC_ABI_REVISION, TUNING_TABLE_REVISION,
  SCHEDULE_REVISION)` tuple and the `F32_TF32_TUNING_REVISION` assertion,
  because that constant is a direct alias of `TUNING_TABLE_REVISION`. Existing
  literal 40 values belonging to ABI sizes, offsets/extents, digests,
  pointer/stride boundaries, or unrelated test tuples remain untouched.
- Both actual-AUTO paths require a known runtime NVRTC and one of the three
  qualified versions. The shared six-family force/AUTO corpus, five-rung bit
  comparisons, 448 marker groups, special values, guards, and graph checks are
  otherwise unchanged.
- The misaligned A/B section uses an explicit old-AUTO tile for both its
  control launch and selection assertion: hot C expects M128S2 on 12.8/13.0
  and M64S2 on 13.2; A, B, D, and E expect M64S2 on all three.
- The diff contains no kernel, composer, loader, contract, ABI, symbol,
  artifact-digest, invocation-digest, schedule, old-picker, other-precision,
  other-architecture, or Triad-selector edit. The committed 101-log hashes in
  the updated guard comment match the referenced files.
- `git diff --check` is clean. The intended selector, revision-40, and
  pre-promotion hot-A RED failures are present in the synchronized ignored
  evidence directory. The amended final source manifest hashes to
  `21f828115c983114175615bf6b6c044c461b811d3cc11aad787697dff3f7a574`.

## Code quality

No code-quality issues found. The finite `matches!` predicates are direct and
fail closed, the toolkit-specific fallback is localized to the existing test
section, and the epoch regression distinguishes route invalidation from
compiler/artifact/numeric/schedule changes.

## Verification-helper review

No remaining helper issues found after amendment. Three initial Important
fail-closed gaps were corrected during review:

- The timing helper now emits diagnostic JSON and then exits nonzero unless
  all ten internal p50 and p95 promotions win.
- It now requires the exact harness test identity and the planned per-toolkit
  invocation shape: one complete 40-record log on 13.2, or the exact 32-record
  A/B/D/E and 8-record C partitions on 12.8/13.0. Its aggregate uniqueness
  check makes the two planned partitions occur exactly once.
- The corpus helper now has explicit `final` and `legacy` modes. Final mode
  requires the exact two AUTO wrapper identities and literal two-test success;
  legacy mode requires only the historical all-cells wrapper rather than an
  arbitrary caller-supplied count.

The timing helper otherwise uses the required nearest-index quantile, computes
samplewise `AUTO / forced-old` and `AUTO / FAST` in the correct direction,
takes the worst p50/p95 over all four path/order cohorts, validates the exact
40-key union, and binds compiler/source/invocation/artifact/header/NVRTC-library
identity to the committed per-toolkit baseline. Its amended graph, tolerance,
suite-result, and raw reported-ratio checks are internally consistent. The
corpus helper's expected 448-key construction, duplicate rejection, and
`actual_auto=true` requirement are also correct.

Independent local self-checks passed for the committed 13.2 epoch-40 timing
baseline, final 12.8/13.0 corpus logs, and the historical legacy corpus log.
The helpers rejected a wrong epoch, duplicate timing input, and a force-only
corpus log as intended.

## Runtime status

No runtime-evidence issues found. The synchronized final evidence establishes:

- All three final library runs report 634 passed, 0 failed, and 46 ignored;
  all three performance-static runs report 43 passed, 0 failed, and 62
  ignored. The retained initial 12.8 library log's two failures are the two
  stale current-epoch fixtures subsequently corrected and rerun, not waived
  failures.
- The exact two actual-AUTO wrappers pass on each toolkit with all 448 unique
  expected corpus groups. CUDA 13.2 additionally passes the two retained Ada
  Triad cohort tests and the C-prefix test. Main's independent CUDA 13.0
  actual-AUTO rerun also passes 2/2 and 448/448 groups; its log SHA256 is
  `b2eb479da2e6b24cb8b79fe90b813b2a908fd6af4178541c8e4af06537f56dbc`.
- The five planned quiet post-AUTO101 logs contain exactly 120 unique accepted
  epoch-41 records: 32 A/B/D/E plus 8 C on 12.8, the same partition on 13.0,
  and one 40-record run on 13.2. Every toolkit has 10/10 prior-route p50 and
  p95 wins. FAST wins remain only A1 on 12.8/13.0 and A1/B1 on 13.2, as
  reported; the other 9/9/8 cells remain explicit gaps. Physical AUTO and old
  graph symbols, block/shared/grid geometry, old-tile mapping, bits,
  tolerances, FAST mode, timed bias, source/artifact identity, and raw
  samplewise ratios all pass the amended independent verifier.
- Every timing preflight and in-run preflight records 0% GPU and memory
  utilization, 90 MiB use, P5, and 1800 MHz. The final source manifest contains
  168 inputs and hashes to
  `21f828115c983114175615bf6b6c044c461b811d3cc11aad787697dff3f7a574`.
  Main's independent remote check reports all 168 source hashes OK, the six
  live correctness/performance binaries, and all nine cache key/blob hashes
  matching the preceding toolkit qualification.

The tracked archive README, integration report, handoff addendum, and
retirement-ledger entry are consistent with those raw results and keep the
scope limitations explicit. Its `raw/` copy is byte-for-byte identical to the
39 worker artifacts; `raw/SHA256SUMS` hashes to
`3173d5830bb4272ed088abe37c76e102635c581cefe701f2c1bcd8baa39acf38`, and
all 39 entries verify locally. No source, helper, runtime-evidence, archive,
handoff, or retirement-ledger issue was found. This reviewer did not rebuild,
rerun GPU work, or claim coverage beyond the archived qualification.
