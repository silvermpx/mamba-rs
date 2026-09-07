# Ada Triad current-AUTO two-cell physical diagnosis

User authority: continue autonomously after the assembled Fixed finalist batch,
in the same worktree/branch, preserve deterministic arithmetic and use bounded
experiments. Fixed AUTO45 is committed as `b221b71e`; its raw evidence stays
immutable. The follow-up holder documentation changes no runtime behavior.

## Scope and ownership

- Sole SSH/CUDA/GPU owner: `implement_ada_f32_tf32_toolkits`.
- Root: source/contract inspection, decisions, evidence replay and commits.
- `audit_triad_split8_reachability`: independent read-only result review.
- No production/test/CUDA edits, new branches, cleanup, clock changes, full
  precision/toolkit sweeps or candidate admission in this diagnostic task.
- Preserve the old unbuilt Triad discovery WIP and its helper locally; do not
  sync or build them for this task. No new comparator framework is required.

## Existing entrypoint and exact contracts

Build only `gemm_bi_performance_matrix` in release with matching CUDA13.2,
using the existing runner environment and source/binary/cache bindings. Keep
the Fixed phase2 target/cache/evidence frozen. A separate profile target and
cache namespace are permitted; this is not a separate git branch.

Run the existing ignored exact test `gemm_bi_deterministic_performance_matrix`
with `GEMM_BI_QUAL_WINDOWS=1`, `GEMM_BI_QUAL_PATH_ORDER=ab`,
`GEMM_BI_QUAL_VARIANT=ada-triad-ncu`. Select one exact cell per invocation via
`GEMM_BI_QUAL_CELL_IDS`; unknown/empty selections must fail, never fall back
to a full matrix. Require one executed Rust test, not merely exit0.

| Cell ID suffix after `f32_policy_allow_tf32/` | Logical M,K,N | Output | Expected actual AUTO symbol | Grid/block | Dynamic shared |
| --- | --- | --- | --- | --- | ---: |
|`tn/prism_in_proj/contiguous`|4621,384,1928|384x1928|`gemm_bi_tn_sm80_mma_tf32_v1_m128n64_bk32_s3`|93/256|79872|
|`nt/d768_in_proj/contiguous`|2048,768,3072|2048x768|`gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3`|192/256|82944|

Both use the public AllowDeterministicTf32 policy and the existing direct
RegisterCvtRnaTf32F32V1 family on qualified CC8.9/142SM/CUDA13.2. Alpha1,
no bias; TN beta1, NT beta0. The public logical tuple must not be transposed
twice. Assert actual physical request/route/compiler/artifact identity from
the emitted matrix evidence; reject exact-scalar fallback or wrong geometry.

The matrix independently samples eager and graph and does not restore C
before every TN beta1 operation. Its one-window unprofiled records establish
invocation/physical identity only, not paired Fast admission, training-workflow
latency or a numerical qualification. Do not present their ratio to historical
vendor timings as a win. No vendor or split8 run is authorized here.

## Acquisition

1. Bind the exact source set, one built binary, matching NVRTC/CUDA library,
   GPU UUID/driver, module artifacts and NCU tool version. Preserve quiet PRE,
   same-attempt exits and RELEASE; if release is transiently busy, retain it
   and record a distinct quiet drain without rerunning valid GPU work.
2. Obtain one unprofiled one-window invocation per exact cell to confirm actual
   AUTO identity. Do not expand this into21/101 windows or run the full suite.
3. Reuse `internal/perf/ada-f32-b0-ncu-20260907/profile.py`'s NCU recipe,
   not its hardcoded old Fixed source/binary hash or evidence destination.
   One exact-symbol report per cell, `--kernel-name-base mangled`, anchored
   symbol regex, `--launch-skip 130 --launch-count 1`, kernel replay,
   `--clock-control none --cache-control none --pipeline-boost-state dynamic`,
   config off and application-only target. The ordinal is a bounded diagnostic
   selection, not a claim of an exact warmup position: calibration is adaptive.
4. Capture the existing nine sections: LaunchStats, Occupancy, SpeedOfLight,
   ComputeWorkloadAnalysis, SchedulerStats, WarpStateStats,
   MemoryWorkloadAnalysis, InstructionStats and SourceCounters, plus the eight
   existing explicit traffic metrics. Query installed definitions if a metric
   is unavailable; do not fabricate a counter or recapture a successful report
   just because a local parser/export failed.
5. Preserve the two `.ncu-rep` files and raw/details/SASS exports. Require one
   captured kernel with the expected symbol/grid/block/shared for each cell.
   Profile durations are diagnostic, never admission latency. Use the known
   CSV units+data row shape; inspect headers rather than assuming row counts.

## Decision report

Report registers, local/static/dynamic memory and resource-limited residency;
waves/SM, achieved occupancy, eligible warps, tensor-pipe activity, long/short
scoreboard and barrier stalls, actual-versus-ideal shared wavefronts, DRAM/L2
traffic, and the dominant instruction mix. Keep values and units from the
installed NCU definitions. Distinguish active-cycle counters from whole-device
utilization; low output-grid coverage alone is not proof of a bottleneck.

TN has93 output CTAs and NT192 before any measured residency limit; the Ada
has142SM. Healthy active tensor issue with a thin grid motivates an underfill
hypothesis. Poor active issue with measured stalls motivates one concrete
preserve-association staging/scheduling hypothesis. Old six direct tile losses
are already archived; do not resweep them. Existing split8 has a different
reduction association and cannot silently replace the user's numerical family.

Finish with one bounded next experiment, justified by actual counters, or an
explicit statement that the profile does not isolate a mechanism. Keep code
unchanged until root records that next design. Root and reviewer replay raw
evidence, then root commits the diagnostic report/evidence with human identity.
