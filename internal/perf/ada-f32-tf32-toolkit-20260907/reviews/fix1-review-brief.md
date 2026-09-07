# Task7 fix1 scoped re-review

Worktree /Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80; HEAD still e1d503c78694936b46ee79ca6b1a320a494ce7f5.
Read Task7 brief, incumbent ruling, fix1 brief and fix1 report in this directory.
Previous review: ada-f32-tf32-toolkit-task7-source-review.md.
Fix-only diff: ada-f32-tf32-toolkit-task7-fix1-review-package.diff.
Do not repeat the original full six-file source review. Read this fix diff once;
verdict each finding below and only new breakage introduced by this fix.

Root preserved byte-identical final4 originals under task7-final4-freeze/.
The parent diff contains only new numeric/schedule imports; its16 previously
reviewed admission lines are unchanged. Runtime/support/analyzer/tests are
final5b. Earlier final5 build hashes in fix1 report are pre-format history;
final5b matching build/smoke evidence is being produced by the sole implementer
and will update the report. These pending GPU facts are not a source-code
finding or a reason to rerun tests. No screen21 has started.

Root current final5b hashes:
- performance317cb44c60926bca32da6b3f7828187be7ab61b4173cf6b07297a86961698419
- supportcb20805c079b200f1bb77f02912c16fb56113d26cf3204c04d046cc872cee73f
- runner24275ce7ed506336f64297a0eb29afaefe12d366ca4ed47860a0a16031dbd1aa
- analyzere8bebda48987a699392f7c8bdbdf8f74b61349a2d3aebc20a36bf1e222026182
- remotecb91f07492a9e9b661d98e6be58d7b71f48511eae082ffbeca133d50e7bfdf0e
- tests43afdd008451cd89195200e687b3fe8142072969fbb701b78d5f3c33e5ab4608

Both Rust formatting checks PASS. Focused fix RED/GREEN evidence is in report;
no suite repetition. Read-only on source/index/HEAD, no GPU/SSH/Cargo/build,
branches/worktrees/staging/commit/push/deletion or subagents. Write only
ada-f32-tf32-toolkit-task7-fix1-review.md via apply_patch. Out-of-scope
observations outside the fix are nonblocking and do not extend this loop.

## Findings under verification (verbatim)

1. **Numeric ABI revision 5 and schedule revision 8 are neither rejected nor
   bound into evidence.** The binding brief requires
   tuning43/numeric5/schedule8 and says any wrong revision must reject. The
   runtime checks only `TUNING_TABLE_REVISION == 43`
   (`tests/support/fixed_sm89_toolkit_admission.rs:1233`) and emits only the
   tuning revision in its identity (`tests/support/fixed_sm89_toolkit_admission.rs:1283`).
   The independent analyzer likewise checks only tuning43
   (`internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py:191`), even though
   `CompilerIdentity` exposes `numeric_abi_revision` and `schedule_revision`
   (`src/mamba_ssm/gpu/kernel_identity.rs:687`). Consequently, a freshly built
   and self-consistently source/binary-bound package with either revision
   changed could still produce records that the runtime and analyzer admit.
   Reject compiler numeric/schedule revisions other than 5/8 before case
   allocation or timing, emit both values, independently validate them, and
   add separate negative tests for each field.

2. **A confirm attempt launches GPU workflows before validating the stored
   screen's compiled-artifact identity.** `single_term_controls` launches the
   incumbent and candidate at
   `tests/support/fixed_sm89_toolkit_admission.rs:1268`; only afterward does the
   code obtain and compare `fixed_artifact_digest` with the screen binding at
   lines 1269-1278. The brief requires the exact source/binary/artifact/toolkit
   binding before launch. Move the artifact identity comparison ahead of all
   correctness/workflow launches. Add a mismatched-screen-artifact negative
   that proves rejection occurs before the first launch, not merely before the
   measured hot-case loop.

3. **The required post-timing input/guard comparison is delayed until after
   additional eager and graph launches.** After the timed observation loop,
   the code first reads outputs and executes new eager, one-graph, and
   twenty-graph poison replays (`tests/support/fixed_sm89_toolkit_admission.rs:1491`),
   then checks saved A/B/bias and their guards only at line 1505. The brief
   requires complete saved inputs and guards to be compared immediately after
   each timed configuration. Intervening launches leave a masking window in
   which a timed corruption could be overwritten or otherwise transformed
   before evidence is checked. Call `case.inputs(&ctx)` immediately after the
   timed loop and before any further launch/readback workflow; retaining the
   later check is useful. Add sequencing negatives for A, B, and bias mutations
   introduced by the timed path, including a case where a later replay would
   restore the saved value.



## Output

Finding Verdicts: each ADDRESSED or NOT ADDRESSED, with file:line.
New Breakage in Fix Diff: Critical/Important/Minor, or none.
Out-of-Scope Observations: separate and nonblocking, or none.
Verdict: All findings addressed, no new Critical/Important breakage, or list
open findings. This gate covers corrected source before screen21, not final
performance admission or AUTO promotion.
