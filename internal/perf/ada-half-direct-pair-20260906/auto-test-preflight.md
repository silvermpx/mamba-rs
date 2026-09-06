# Task 3 Ada homogeneous-half AUTO test preflight

## Status and boundary

This is a read-only test-assumption map, not an AUTO recommendation. The final
CUDA 12.8/13.0 direct-101 winner matrix is still an input to the implementation
brief and must not be guessed. I inspected only the named performance harnesses,
the SM89 pipeline integration test, and `ada-half-auto-preflight.md`. No source,
analyzer, archive, GPU, or remote state changed.

Two test assumptions require bounded changes for revision 42. The existing
three-arm forced-rung workflow is already route-dynamic and needs no structural
rewrite. No current symbol named `fixed_auto_vendor_expected_native_tile` exists;
the nearest helper is `fixed_auto_vendor_expected_half_tile` at
`tests/gemm_bi_fixed_performance.rs:8830`, whose only caller is inside the
CC12-only AUTO/vendor census (`:9297-9307,9438-9450`). It should remain an SM120
expectation helper rather than being broadened for Ada.

## One literal revision-42 expectation table

Add a small test-only function in each integration-test crate that needs an
independent expected route, rather than calling the production selector. A useful
contract is:

```text
expected_ada_half_auto_v42(nvrtc, dtype, (m,k,n), has_bias)
    -> Tc128Sm89Pipeline | Tc128Sm89Swizzle
```

It must fail closed on an unknown toolkit, non-homogeneous/non-half dtype, or a
shape outside hot A-E. Populate CUDA 12.8/13.0 only from the final independently
reviewed direct-101 matrix, including any dtype/bias differences. The already
known CUDA 13.2 rows are independent of bias:

| dtype | A | B | C | D | E |
|---|---|---|---|---|---|
| BF16 | pipeline | swizzle | pipeline | swizzle | swizzle |
| F16 | pipeline | swizzle | pipeline | swizzle | pipeline |

Use literal table tests, including both bias values and all three toolkit
branches. Do not derive expected values from `fixed_sm89_half_pipeline_auto_eligible`
or its replacement. The table represents the preferred route when both
qualified holders are live; availability fallback belongs in production selector
unit tests, described below.

## Required callsite changes

### 1. Direct forced-pair V1 harness

`fixed_ada_half_forced_direct_pair` currently computes one `expected_auto` from
toolkit alone—Tc128 on CUDA 12.8/13.0 and pipeline on 13.2
(`tests/gemm_bi_fixed_performance.rs:13498-13502`). That becomes false for every
promoted 12.8/13.0 cell and for the listed 13.2 swizzle cells.

Smallest change:

- remove the outer toolkit-only constant;
- inside the row/cell/bias loop, after `input_dtype`, `shape`, and `has_bias` are
  known, obtain the literal revision-42 expected tile;
- bind the tile returned by `launch_fixed_auto_vendor_custom` to
  `actual_auto`, assert `actual_auto == expected`, and emit `actual_auto` in
  `actual_auto_tile` rather than emitting an unobserved expectation
  (`:13581-13586,13826,13855`);
- retain all existing raw equality, PEDANTIC, poison, graph, and two-arm timing
  checks unchanged. Both candidate graphs are already captured and physically
  validated (`:13619-13690`), so selection changes do not require timing or ABI
  rewrites.

The V1 schema already has a truthful `actual_auto_tile` field and uses the global
`TUNING_TABLE_REVISION` on records and completion (`:13819-13841,13880-13884`).
It will therefore emit revision 42 without a numeric replacement in this file.
Do not edit the frozen revision-41 direct-pair analyzer or old controls/logs to
accept new-head output. Task 3's post-AUTO proof uses the three-arm workflow and
a new revision-42 analyzer; any incidental revision-42 V1 run must be treated as
new evidence, never appended to the frozen pre-AUTO archive.

### 2. Actual SM89 AUTO prefix/view/graph corpus

The AUTO test currently restricts `forced=None` to NVRTC 13.2 and then asserts
only the boolean equivalence `picked == pipeline` iff `m == hot_m &&
output_offset == 8` (`tests/gemm_bi_fixed_sm89_pipeline.rs:78-97,179-185`). It
both rejects the new 12.8/13.0 scope and expects pipeline on every eligible
13.2 dtype/cell/bias.

Smallest functional adaptation in
`fixed_sm89_half_hot_cell_prefix_view_graph_bits`:

- for the AUTO invocation, accept exactly known NVRTC 12.8/13.0/13.2, assert
  `nvrtc_library_known`, revision 42, CC8.9/142 SM, and both independently
  qualified pipeline/swizzle holders live with no rejection;
- for the exact admitted case (`m == hot_m && output_offset == 8` in this
  corpus), assert `picked` equals the literal matrix entry for
  toolkit/dtype/hot shape/bias;
- for every prefix/non-hot-M or misaligned-output case, assert that `picked` is
  neither pipeline nor swizzle. Leave the exact portable-rung choice to the
  unchanged ordinary picker; these non-hot shapes need not all choose Tc128;
- keep the forced-swizzle branch (`forced=Some(SWIZZLE_CANDIDATE)`) unchanged;
  keep its exceptional inputs, row/output views, guards, graph ABI/pointers/bundle,
  two poisoned replays, and A/B immutability checks (`:98-225,382-483`).

The exact candidate-exclusion assertion is an important missing production-scope guard in
the current test: after swizzle becomes selectable, the boolean at lines 181-185
would accept an accidental swizzle selection for any ineligible shape or
misaligned C pointer because `picked == CANDIDATE` would still be false. An
eligible-case `assert_eq!(picked, expected_tile)` plus an ineligible-case
`assert!(!matches!(picked, CANDIDATE | SWIZZLE_CANDIDATE))` closes that hole
without freezing the unrelated portable-rung ladder. The historical test name
may be retained to avoid unrelated runner churn; at minimum its ignore text at
lines 78-80 must no longer claim CUDA 13.2/pipeline-only admission.

### 3. Three-arm forced-rung post-AUTO harness

No route-selection source change is required in
`fixed_ada_forced_rungs_paired_precision_cublas`. It binds the actual AUTO return
to `selected`, asserts that route stays stable, emits it as `auto_tile`, and
already captures an exact physical graph whenever AUTO is pipeline or swizzle
(`tests/gemm_bi_fixed_performance.rs:12889-12923,12986-13034,13107-13124,
13323-13351`). The forced inventory already contains Tc128, pipeline, and
swizzle on Ada (`:11170-11237`). Thus the revision-42 runner can select only:

- forced Tc128 on CUDA 12.8/13.0; and
- forced pipeline on CUDA 13.2,

while preserving both orders, both paths, raw arrays, AUTO/forced/vendor timing,
bits, reference, and graph checks. On unchanged 13.2 pipeline cells the AUTO and
forced route being identical is expected and must not be mislabeled as a new
internal win.

The harness emits global tuning revision on each record (`:13323-13346`) but
does not independently know the qualified AUTO matrix. A new post-AUTO analyzer
must compare every emitted `auto_tile` to the external reviewed revision-42
matrix and reject missing/foreign cells before computing AUTO/old and AUTO/vendor
paired ratios. Preserve the old V2 analyzer and all revision-41 evidence
byte-for-byte.

## Production selector test obligation exposed by this audit

The GPU AUTO corpus proves the preferred matrix only with both holders live. The
new host-side selector tests must separately exercise availability without
deriving expectations from the selector under test:

- on 12.8/13.0, preferred qualified route present selects it; if absent, the
  other independently qualified route is used; if neither is live, selection
  falls through to the unchanged ordinary ladder;
- on a 13.2 swizzle-preferred B/D/E cell, missing swizzle falls back to the live
  qualified pipeline;
- on a 13.2 pipeline-retained A/C or F16-E cell, missing pipeline must **not**
  infer swizzle admission; it falls through to the old ordinary path;
- a rejected swizzle holder must not suppress a live pipeline, and unknown
  library/toolkit, wrong CC/SM, mixed/F32 storage, non-hot shape, bad A/B/C
  alignment, or bad bias alignment must not select either new AUTO route.

This is the missing availability/negative guard around the new two-holder
selection. Existing forced-route capability tests should remain unchanged: an
AUTO rejection is not a forced-route deletion.

## Revision and regression boundary

No literal `41` occurs in the affected integration test logic. Raising the
global revision to 42 automatically changes the performance records; add only
the explicit revision-42 assertion to the actual AUTO corpus that is tied to the
new table. Do not globally replace historic revision values or relax old parser
controls. After implementation, the focused regression set is the revised
all-three-toolkit actual-AUTO corpus, the unchanged forced pipeline/swizzle
corpora, the three-arm post-AUTO 80-record-per-toolkit runs, and the already
required revision-sensitive RNA/Triad identity cohorts. No mixed-output or
non-Ada AUTO expectation is implied by these test changes.
