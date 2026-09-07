# Task7 corrected final4 source review

## Review scope and basis

This is the first task-scoped source gate for the corrected final4 package. I
reviewed the complete six-file frozen diff against
`e1d503c78694936b46ee79ca6b1a320a494ce7f5`, the amended Task7 brief, the
binding incumbent ruling, and the current implementer report (SHA-256
`2de5bbf67e0cff663658fb9790a65ba14342a44f45ba217ac398cc29d686b301`).
I also inspected only the two authorized unchanged interfaces: public
`fixed_forward` TF32 precedence and the reused launch, graph, timing, and
numeric helpers.

The six reviewed source hashes match the report's corrected final4 table. HEAD
remains the stated base. The tracked handoff edit is root-owned documentation
and is outside this implementation package.

This is a source gate before screen21, not final performance qualification.
The four final4 smoke1 attempts are reported and locally mirrored as passing
functional, closure, and analyzer validation, but I did not rerun any build,
host, GPU, or package-wide suite. No screen21 or confirm101 attempt has begun.

## Spec compliance

- **Scope and entry behavior:** Compliant. The change is test/evidence tooling
  only, retains `fixed_ada_forced_rungs_paired_precision_cublas`, dispatches the
  narrow mode only for an exact `MAMBA_FIXED_ADA_TOOLKIT_ADMISSION=1`, and
  preserves the historical path when that variable is absent
  (`tests/gemm_bi_fixed_performance.rs:13783`).
- **Strict controls and literal closure:** Compliant. The parser rejects stale
  controls, binds the family/candidate/toolkit/stage/window/source/binary, uses
  a single ordered `cell:bias` control, requires all literals for smoke1 and
  screen21, and permits a nonempty confirm subset
  (`tests/support/fixed_sm89_toolkit_admission.rs:36`).
- **Corrected incumbent and physical contracts:** Compliant. Exact F32 uses
  Legacy. TF32 uses public AUTO with `Tf32RnaM128N128S3`; RNA and M64 retain
  their distinct 32-byte and 24-byte final bundles, and all one/20 custom graph
  nodes are inspected (`tests/support/fixed_sm89_toolkit_admission.rs:416`,
  `tests/support/fixed_sm89_toolkit_admission.rs:768`). Focused inspection
  confirms production AUTO evaluates RNA-wide eligibility before the ordinary
  TF32 picker (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:3510`).
- **Measurement and admission arithmetic:** Compliant. The implementation uses
  the required three B/A comparisons, parity-dependent traversal and ABBA/BAAB
  observations, 20-operation event windows, four path/parity strata, direct
  p50/p95 computation, and an own-win rule independent of Fast
  (`tests/support/fixed_sm89_toolkit_admission.rs:191`,
  `tests/support/fixed_sm89_toolkit_admission.rs:1478`). The analyzer recomputes
  raw chronology, ratios, summaries, decisions, and exact record counts
  (`internal/perf/ada-f32-tf32-toolkit-20260907/analyze.py:235`).
- **Physical, numeric, and closure gates:** Substantially implemented: guarded
  storage, complement poison/readback, eager/graph repeat bits, finite/order
  controls, one/20 custom ABI inspection, opaque Fast inventory, telemetry,
  create-new artifacts, and inner/wrapper/outer closure are present. The three
  Important findings below prevent full compliance.
- **Final performance evidence:** Pending by design. The code implements the
  screen21-to-fresh101 subset gate, but no 21/101 result or performance winner
  is accepted by this review.

## Strengths

- The correction does not evade the installed RNA route: full hot-C operands
  remain aligned, AUTO calls public `fixed_forward`, and the candidate remains
  ordinary M64S2.
- Known custom kernels are validated through actual symbols, launch geometry,
  complete Driver ABI, captured pointers and bundles, and terminal-argument
  rejection. The opaque cuBLAS arm is treated separately and keeps the timed
  bias-broadcast plus beta-1 GEMM workflow.
- JSONL is create-new and incrementally flushed, with a digest over all
  preceding records. Partial/failed attempts therefore remain inspectable and
  cannot satisfy exact completion.
- The runner has one toolkit environment constructor and rechecks frozen source
  inputs, executable, tools, libraries, private cache mode, PRE/POST telemetry,
  and copied raw hashes. Confirm recomputes the complete stored screen rather
  than trusting only a supplied digest.

## Critical findings

None.

## Important findings

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

## Minor findings

None.

## Evidence outside this source gate

- The locally mirrored final4 smoke artifacts support functional readiness but
  do not replace screen21/confirm101 performance evidence.
- Exact Task6A historical artifact provenance, the final rooted manifest,
  complete 20-literal 21/101 matrices, release telemetry, and any winner or
  promotion recommendation remain evidence-gate work for root. They are not
  inferable from this six-file source review.

## Task quality verdict

**Needs fixes.** The corrected RNA incumbent behavior and the central
measurement algorithm are implemented well, but the three Important contract
gaps above must be corrected and source-reviewed before screen21 starts. This
verdict makes no final performance or selector-promotion claim.
