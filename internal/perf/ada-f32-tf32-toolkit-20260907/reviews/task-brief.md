# Task7 — qualify existing exact F32 and TF32 candidates on CUDA12.8/13.0

Controller correction2026-09-07: read the adjacent
`ada-f32-tf32-toolkit-task7-incumbent-ruling.md` with this brief. The original
audit missed RNA-wide precedence: current aligned TF32 AUTO is RNA-wide,
not M128S2. The correction below and that ruling supersede the audit's
TF32-incumbent claims. Keep original audit/raw failures unchanged.

Read this first. This is the next bounded measurement task under the existing
handoff and `docs/performance-playbook.md` section9, not a new branch or kernel
design. Start only after root commits reviewed Task6C and transfers the sole
Ada lane. Root supplies that actual commit at dispatch. Required source audit:
`ada-f32-tf32-toolkit-harness-audit.md` in this directory, SHA256
3d0a70aa7b10eb8cbbb94c577093041ae13ee172415f2788320f15129e7dfcc5.
It explains existing helpers and exact physical contracts; do not repeat its
loader inventory. Its immutablebase42 is historical: this task binds current43.

## Goal and ownership

Produce a complete per-literal winner matrix for existing CopyPlan and M64S2
against actual incumbent AUTO and independently against cuBLAS Fast. No source
kernel, loader, selector, compiler option, numeric/schedule revision change.
No new benchmark entry: retain the existing ignored
`fixed_ada_forced_rungs_paired_precision_cublas` as the measurement entry and add
an explicit narrow qualification mode. Ordinary historical mode remains intact.

Worktree `/Users/silvermpx/IdeaProjects/mamba-rs/internal/worktrees/gemm-bi-triad-sm80`,
branch `codex/gemm-bi-triad-sm80`. Own `tests/gemm_bi_fixed_performance.rs`, a
focused support module `tests/support/fixed_sm89_toolkit_admission.rs` if needed,
new evidence `internal/perf/ada-f32-tf32-toolkit-20260907/`, and the report beside
this brief `ada-f32-tf32-toolkit-task7-report.md`. Root owns all staging, commits,
handoff and ledger. No new branch/worktree/push/deletion/signals/subagents.
Use apply_patch. No Mac cargo/CUDA; matching builds on Ada only. Direct rustfmt
is allowed and must finish before frozen-source final build/screening.

Reuse directly applicable production launch, force registry, row definitions,
bias-inclusive vendor launch, graph inspection, strict filters and tested
timing/poison helpers. Factor only the shared mechanics needed; do not copy an
entire old benchmark or broadly restructure the performance file. In particular,
the prior half pre42/post43 schemas and their entry behavior must not change.

## Literal inventory and denominator

Exact device UUID GPU-d1edd7be-e88d-aed6-047d-622163306f0e, RTX6000Ada,
CC8.9/142SM, known NVRTC library and compiler targetsm_89. Independently use:

| Toolkit | Directory | Cargo features |
| --- | --- | --- |
|12.8|/usr/local/cuda-12.8|cuda,cudarc/cuda-12080|
|13.0|/usr/local/cuda-13.0|cuda,cudarc/cuda-13000|

Each uses isolated source/target/private0700 cache with recorded source, compiler,
library, executable and cache identities. Tuning43/numeric5/schedule8; wrong
version/device/library/revision must reject before allocation or timing.
Build, holder/cold/warm, functional and timed commands must all consume the
same per-toolkit environment constructor (CUDA_HOME/CUDA_PATH/PATH/LD_LIBRARY_PATH,
matching cargo features/target/cache). Do not maintain a second handwritten
functional-loop environment. Task6C caught that omission when all three holder
invocations printed NVRTC13.2; their success flags were not toolkit evidence.

| Family | Literal shapes M,K,N | Bias | Actual AUTO | Candidate |
| --- | --- | --- | --- | --- |
|f32_exact_fast|A4621,384,1928;B4621,768,2304;D2048,768,2304;E2048,2304,768|0,1 each|Legacy|F32Sm89N64CopyPlan|
|tf32|C4621,1928,384|0,1|Tf32RnaM128N128S3|Tf32M64S2|

Homogeneous F32 storage; contiguous ldaK, ldbN, ldcN; alpha1/beta0. Exact policy
ExactScalarFmaV1, TF32 policyAllowDeterministicTf32V1. Custom outputs must match
incumbent raw bits and repeats under their policy. Both timed vendor arms use
CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_DEFAULT_MATH and the recorded native
algorithm request. Explicit host pointer mode/atomics-not-allowed are queried
and recorded. Require NVIDIA_TF32_OVERRIDE absent. PEDANTIC F32 is a separate
untimed numerical reference only: exact custom tolerance0.0002, TF32 custom
0.0025, Fast0.0025. Bias broadcast and beta1 GEMM belong inside every timed
vendor logical operation for biastrue, including its capturedgraph.

## Narrow mode and host test cycle

- [ ] Add explicit enable MAMBA_FIXED_ADA_TOOLKIT_ADMISSION=1 in the existing
  entry. A present malformed value rejects. Existing MAMBA_FIXED_ADA_VENDOR=1
  remains mandatory. Existing row/tile/path controls must identify one family,
  its exact candidate, and eager,graph; no defaults or unrelated inventory.
- [ ] Add explicit toolkit, tuning revision, stage, source/executable bindings,
  and a strict literal cell:bias list for the qualification mode. Canonical
  pairs are hot_a:0 etc; reject duplicate/unknown/empty pairs. Use one truth
  source for this list: reject simultaneous old independent cell/bias filters,
  whose Cartesian product cannot express a general eligible101 subset. Record
  the exact chosen control names and commands in the report. Reject stale
  half experiment, other vendor-row, force-candidate and TF32 override controls.
- [ ] Stages are smoke1, screen21, confirm101. Smoke/screen requires all eight
  exact literals or both TF32 literals for its family. Confirm requires the
  explicit nonempty subset admitted by the stored same-source21 screen.
  Reject a stage/window mismatch, absent controls, wrong AUTO incumbent, or
  partial smoke/screen rather than emitting success with some records.
- [ ] Before implementation run meaningful host RED for the new strict mode,
  exact inventory/closure and mirrored raw arithmetic. Preserve original
  command output and failures. Then implement the smallest mode/helper changes
  and run focused plus nonignored performance tests with matching features.
- [ ] Synthetic negatives must reject missing/duplicate/foreign literals or
  raw records, malformed stage/revision/source/binary/device/toolkit/ABI,
  wrong arm/position/traversal/parity or captured args, wrong vendor modes,
  nonpositive/nonfinite times, forged summaries/completion and failed exits.
  Separately prove a genuinely recomputed loss or mixed-p95 result is valid
  evidence with admissionfalse, not malformed data. Eligible siblings survive
  valid literal losses; no sibling survives an invalid shared run/stage.

## Physical and bit gate before timing

- [ ] Guard A/B/bias/C allocations; retain every input byte before any arm.
  Compare complete saved inputs and guards after pre-timing checks before
  warmup and after each timed configuration. Add changedA/B/bias gate negatives.
- [ ] Actual AUTO calls public fixed_forward and must return the exact above
  incumbent. Candidate uses public fixed_forward_with_tile. Require unchanged
  per-arm enum on every eager launch and actual graph functions, not labels.
- [ ] Inspect one-operation graphs and every node of each twenty-operation
  graph: node kinds/counts, actual symbols, grid/block/shared, full Driver ABI,
  terminal parameter rejection, actual captured C/A/B/bias pointers and bundles.
  Exact pointer/bundle/terminal ABI assertions apply to the known custom
  AUTO/candidate symbols below. For opaque cuBLAS kernels, preserve actual
  graph node inventory/geometry, requested and queried vendor modes, complete
  timed bias workflow and numerical/repeat/overwrite gates; do not invent an
  undocumented private cuBLAS ABI. Reuse the existing vendor graph helper.
  Exact CopyPlan compact5args/eightwords/block128/static32768/dynamic0 and
  Legacy12args use contracts in the audit and exact support module. TF32
  candidateM64 compact5args/24-byte bundle is
  [(0,8),(8,8),(16,8),(24,8),(32,24)] with sixth absent and
  words[4621,1928,384,1928,384,384], grid438/block128/shared32768. ActualRNA AUTO
  has5args ending(32,32), words[1.0bits,0,4621,1928,384,1928,384,384], grid111,
  block256/shared98304, sixth absent. Validate both bias pointer states and
  keep the C16 hot operand class; no bypass of RNA. See the ruling for reuse
  of the existing RNA graph contract and valid one-active-term probe geometry.
- [ ] Each correctness replay uploads the bitwise complement of expected
  output, verifies its GPU readback and every storage word differs, then
  executes the intended workflow and requires exact expected bits plus guards.
  Include repeated eager/graph, single-term/bias-orientation and exact finite/
  ordering controls. Capture a real empty graph and demonstrate that it fails
  the same overwrite gate. Fast has its own expected repeatbits, not custombits.
  Never put poison/reference/readbacks inside timed workflows.

## Single measurement algorithm and admission

Arms are actualAUTO, candidate, Fast. Three comparisons use B/A:

```text
0: A=AUTO, B=candidate -> candidate/AUTO
1: A=Fast, B=AUTO      -> AUTO/Fast
2: A=Fast, B=candidate -> candidate/Fast
parity = (window + start_parity) % 2
comparison traversal = [0,1,2] if parity0 else [2,1,0]
observations = [A,B,B,A] if parity0 else [B,A,A,B]
ratio = sum(two B times) / sum(two A times)
```

Both eager/graph and startparity0/1 for every literal. 128 eager warmups per arm;
warm each graph. CUDA events surround20 public eager operations or one captured
20-operation workflow; reportus/op. Preallocate collection capacity, and no
printing/readbacks/device allocation/compile between warmup and full windows.
Emit raw chronological observations, bracket indices/pairs and p50/p95 with
round((W-1)*fraction). Do not invert a percentile to reverse its direction.

Full screen/toolkit is exact32configurations + TF328configurations, separate
family artifacts. Each config has12Wraw/3Wpairs/3summaries. Exact requested-key
closure and preceding JSONLdigest are mandatory, with no rejected or omitted
configs. Source/exe/artifact, exactcommand/env, telemetry, actual test/wrapper/
outerSSHexit and copied rawhashes are bound to the same attempt. Reuse the
reviewed same-attempt validator mechanics; never mutate historical raw records.

- [ ] Freeze source and one matchingbinary/toolkit, report source-stable for
  root source review. Run complete one-window functionalsmoke for bothfamilies
  on eachtoolkit, then one21screen per family/toolkit. Independent analyzer
  must validate physical/bit/identity/exit closure and recompute raw ratios.
- [ ] A literal advances only when candidate/AUTO p50 ANDp95<1 in allfour
  path/start strata. Run one fresh101 only on explicit eligible literals using
  the same frozen source, binary and compiled-artifact binding as21. No rebuild
  between matching21/101, no screen repeats until lucky, no pooling samples.
- [ ] Report every valid loss/tie/mixed constituent. Ownwin is independent
  of Fast: a remaining candidate/Fast deficit does not veto a robust ownwin.
  Any functional/physical/numeric/identity/guard/immutability/telemetry/exit or
  exact-completion failure invalidates the affected run/stage; preserve reason
  and raw data before a justified rerun.

## Final closure and next boundary

PRE requires exactUUID0%GPU/0%memory/noapps. POST requires sameidentity/noapps;
residual utilization alone is not a new failure. Finalreleasequiet/noapps.
Only the assigned owner uses Ada; no overlap with root or reviewers.

Source remains test-only, so unchanged approved Task6A production qualification
is reusable only after exact compiled source/invocation/artifact identity check
onbothtoolkits. Verify relevant liveholder and captureidentity on this actual
source/binary. Keep current43 routing labels separate from historical42 artifact
qualification. No requalification of unrelated unchanged CUDA architectures.

Write report with all commands/exits/REDGREEN/failurehistory, complete20literal
toolkit matrix, full21/101constituents, source/exe/cache/compiler identities,
rawlogs and rooted SHA256manifest. Preserve source/binary/cache archives and
do not delete losers. Root independently reviews source/evidence and commits.
Only after reviewed winners exist does root authorize a separate literal AUTO
promotion task with newtuningepoch, retained-route tests and actualpostAUTO101.
This task does not widen any selector or claim allinference/Fastvictory.
