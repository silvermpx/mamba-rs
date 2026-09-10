# v0.7.0 assembly and release checklist

Updated 2026-09-10. Worktree: `internal/worktrees/gemm-bi-triad-sm80`;
branch: `codex/gemm-bi-triad-sm80`. This is the active execution checklist;
the detailed historical evidence remains in `internal/perf/` and the handoff.

## Binding release decisions

- `Deterministic` is the release default: custom Inference/Triad only, no
  hidden cuBLAS fallback. `CublasFast` and `CublasPedantic` are explicit opt-in
  vendor modes. F32/TF32/F16/BF16 are numerical policies/data types, not extra
  backend modes. Owner explicitly reconfirmed this on2026-09-10; do not ask
  again or preserve the old hybrid default.
- Select the fastest qualified retained implementation for each covered cell.
  Beating cuBLAS Fast is not a prerequisite for replacing a slower AUTO route.
- Preserve deterministic numerical contracts, eager/graph bits, shape/stride
  and memory safety, with no hidden cuBLAS fallback in Deterministic mode.
- Assemble retained candidates before the combined final qualification batch.
  Use focused checks during wiring; run broad release gates after assembly.
- Test the same final source on Ada and the newly rented RTX 5090 in parallel,
  with one timed process per GPU and an idle/free-VRAM preflight on each board.
- Supported-toolkit qualification includes CUDA 12.8, 13.0 and 13.2. Keep
  per-board, per-toolkit, per-precision and eager/graph results distinct.
- Commit verified phases locally without AI/coauthor trailers. Merge into
  `main` after the release gate passes; publication requires separate approval.

## Retained-kernel assembly

- [x] Existing inference improvements committed before Triad assembly; preserve
  both Ada and RTX 5090 routes and their recorded snapshots.
- [x] Ada half NN/NT Fast-winning batch: six exports, eleven cells; actual AUTO
  validated on all three toolkits (`a74d9624` records the completed batch).
- [x] Exact-F32 NN large and NT d768-out CopyPlan routes assembled previously.
- [x] Exact-F32 NT d768-in and Prism route wiring (`fadf5250`).
- [x] NT pre-admission raw evidence archived and independently replayed
  (`36d81318`); post-admission CUDA13.2 passes.
- [x] Correct NT CUDA12.8/13.0 Scalar header provenance and pass actual AUTO.
  The first post-admission 13.0 check found Fixed headers in the Scalar rows.
  Both corrected toolkits now pass actual AUTO eager/graph, bits, guards and
  once21. The independent native receipt regression covers all three toolkits.
- [x] Three large exact-F32 TN source bodies and optional module
  (`04defc80`, `0184db89`); all three toolkit module checks pass (`36d81318`).
- [x] Large exact-F32 TN eager/prepared wiring (`f76f81f4`); CUDA13.2 focused
  smoke passes for all three routes against actual prior AUTO, with exact
  eager/graph bits and guards. Cohorts were left empty at that checkpoint.
- [x] Large exact-F32 TN combined qualification and admission: all three known
  CUDA12.8/13.0/13.2 cohorts enabled; actual AUTO passes all three routes on
  every toolkit, with exact eager/graph bits, guards and immutable inputs.
  Lower-toolkit pre-admission once3/once7 shows ~22–24% lower time than prior
  AUTO. Report: `internal/perf/ada-large-tn-admission-20260910/report.md`.
- [x] Large exact-F32 TN expanded CUDA13.2 check: all three production forced
  routes pass full/tail/exceptional/K0/non-unit raw oracles, repeated bits,
  eager/prepared identity, guards and paired once3/once7 versus actual AUTO.
  Raw replay matches all 24 screens/480 observations. This valid receipt was
  reused for the completed supported-toolkit phase; no fresh Fast arm was run.
- [x] Exact-F32 TN d128-in/out source-only promotion (`7d390fa7`), with
  normalized retained-body parity and exactly two exports; 13 native checks pass.
- [x] Exact-F32 TN d128-in/out module, routes and focused qualification.
  Both routes are admitted on CUDA12.8/13.0/13.2; exact finite/exceptional
  bits, guards, repeated eager/graph and one-node/no-scratch AUTO pass all3.
  Paired pre-admission p50 is roughly0.364–0.401 of priorAUTO for in and
  0.495–0.502 for out. Source review approved after fail-closed regressions.
  Evidence: `internal/perf/ada-d128-assembly-20260910/report.md`.
- [x] Four-cell TF32 joint source frozen (`560169d5`).
- [x] TF32 joint standalone composer, two frozen primitives and four typed
  specs; 26 native source-contract checks pass at that historical checkpoint.
- [x] Retained-winner audit recovers TF32 NN d768-out baseline N96, missing
  from the four-cell plan; joint source now has five exports. Add the required
  frozen alignment helper. All five-body/source contracts pass 28/28.
- [x] Original five-cell TF32 joint module, NN/TN routes, transform/scratch
  ownership and toolkit-specific admission (`efb38c38`, `08736549`, `fdda603b`).
- [x] Integrate the two later TF32 additions: NT d768-in A-ldmatrix N96/S3 and
  TN Prism pre-RNA M64N96/S2. Faithful source promotion, all three compiler
  identities and six-cell actual AUTO pass. CUDA13.2 paired production timing
  retains ~7.1% and ~19.3–19.4% lower time against the prior selected bodies.
  Lower toolkits retain their measured routes; no new Fast-win claim.
- [x] Final seven-export TF32 joint module passes loading/ABI/resources on
  CUDA12.8/13.0/13.2, local0, no exclusions. Six-cell post-admission AUTO,
  eager/graph bits and guards pass on all three. Evidence:
  `internal/perf/ada-final-assembly-20260910/report.md`.
- [x] Remaining half improvements beyond the eleven Fast-winning cells:
  F16 NN d768-in and half TN compact/regpipe/vec2 are assembled. Actual AUTO
  passes all18 cells/ten symbols on all three toolkits. Lower-toolkit checks
  prove correctness and selection, not fresh half performance comparisons.
- [x] Missing half actual-AUTO comparison: seven rows pass CUDA13.2 eager/graph
  exact oracles, physical manifests, guards and paired once7 p50/p95 gates.
  Both d768-in tournaments retain vec2; compact remains selected for F16 out
  and both Prism rows. Production exports/admission and post-AUTO now pass.
- [x] Current RTX5090 TF32 retained routes admitted on CUDA12.8/13.0/13.2:
  23 measured keys per toolkit, including six portable routes. True G10 stays
  exact because its candidate lost on all three toolkits; G11 remains live.
  Actual AUTO checks all24 cases on each toolkit, with identical full-output
  digest records across toolkits and eager/repeat/graph bit equality.
  Independent review and final test-only corrections pass host20 plus a
  repeated live24 check. Evidence: `internal/perf/sm120-current-cohorts-20260910/`.
- [ ] Reconcile every retained-winner report with production dispatch and the
  qualification harness. No candidate disappears because it loses to Fast.
- [x] Independent retained-winner census finds no further omitted Triad winner
  beyond the explicit large-TN, d128, five-cell TF32 and seven-cell half sets.
  The existing TF32 NT finalist already selects its stage-sliced body for
  d768-out/large-deep and its A-only-ldmatrix body for d768-in/Prism. The
  separate inference F16 NN d768-in N96/S3 experiment lacks an actual-AUTO
  comparison; keep it labelled as unqualified, not a missing Triad admission.
- [x] Reconcile all66 actual Ada paired-Triad physical routes against retained
  assembly decisions: zero omitted accepted winners, foreign symbols or
  unexplained fallthroughs. Review: `internal/perf/final-auto-benchmarks-20260910/ada-triad-reachability-review.md`.
- [x] Bounded Inference lower-toolkit audit: no already-qualified omitted
  winner identified. Ada half, exact-F32 CopyPlan and TF32 RNA-wide selectors
  explicitly cover12.8/13.0/13.2; narrower13.2 overrides have separate evidence
  and preserve lower-toolkit incumbents. The fresh23-key SM120 TF32 admission
  above belongs to Triad, not to the separate Inference selector.
- [ ] Deferred optimization candidate, not a blocker for this assembly: SM120
  Inference TF32 hot-B M128S2 on CUDA12.8/13.0. Older-source screen21 records
  are favorable but do not supply current-source confirmation/post-AUTO proof.
  Keep it in the follow-up inventory; do not widen a13.2 gate by assumption.
  Evidence: `internal/perf/sm120-fixed-full-census-20260906/`.
- [ ] Preserve or justify every route from the saved RTX 5090 inference and
  Triad snapshots; investigate any newly observed route or speed regression.
  The 66-cell assembly smoke passes132 eager/graph rows;65 physical routes
  match the saved snapshot. The remaining TN underfill change is the retained
  M16N32/S4 winner, not an Ada route replacing a 5090 winner.

## Combined validation on Ada and RTX 5090

- [ ] Record GPU/driver/toolkit/compiler identities and exact final source SHA.
- [ ] Inference: F32, deterministic TF32, F16, BF16 and mixed half-to-F32
  cases, bias/no-bias, guards, tails, repeated bits and actual physical routes.
- [ ] Triad: NN/TN/NT, every supported precision and covered shape, exact
  selector identity, eager and prepared/captured graph correctness.
- [x] Paired eager and whole-graph performance against cuBLAS Fast; also
  cuBLAS Pedantic for exact F32, with numerical contracts clearly labelled.
  Full paired acquisition is complete on Ada and RTX5090/CUDA13.2:
  Inference280 records per board; Triad324 records per board. Raw statistics
  and completion metadata replay independently. Results and exact measured
  source identities are in `internal/perf/final-auto-benchmarks-20260910/`.
  Current assembly preserves the measured CUDA13.2 routes: only lower-toolkit
  SM120 TF32 admission and removal of G10 (absent from the66-cell benchmark)
  changed in production. Keep actual measured732c1146 source identities, not
  a later commit label. The tables do not claim that all cases beat cuBLAS Fast.
  Subsequent API/rename/cleanup changes still require an affected-route audit.
- [ ] Real model integration: Mamba and Mamba-3 inference graph replay,
  training graph parity/safety, and trainer full-step replay. The split
  `forward`/`backward_step` API is intentionally eager for external losses.
- [x] Positive deterministic-mode decode graph replay on Ada and RTX5090:
  Mamba/Mamba-3 × F32/BF16, explicit complete policy, graph presence, finite
  output after poison, then rejection of route drift. Tests-first RED and
  both GREEN runs archived in `internal/perf/positive-inference-graph-20260910/`.
  This is not whole-model eager/graph bit equivalence or a throughput claim.
- [ ] Confirm architecture/toolkit portability and fallback coverage for the
  advertised SM80–SM120 families, distinguishing compile-only from live tests.
  The bounded source audit finds production portable half/exact-F32 floors
  and no disconnected accepted winner in the traced paths. It does not qualify
  performance on untested cards. Triad TF32 remains evidence-gated on SM80/86
  and the empty-cohort SM90a/SM100 rows; guarded native half heuristics on
  SM90a/SM100 are not measured winners. Audit:
  `internal/perf/final-auto-benchmarks-20260910/architecture-release-wiring-audit.md`.

Relevant existing integration tests include `inference_graph_route`,
`training_graph_parity`, `f32_training_graph_parity`,
`m3_training_graph_parity`, `m3_training_graph_safety`, `trainer_split`,
`trainer_split_m3`, `inference_f16_smoke` and `gpu_inference_prefill_parity`.
Select their actual supported cases and prerequisites before executing them.

## API, cleanup and release preparation

- [x] Rename `gemm_bi_fixed` / Fixed to `gemm_bi_inference` / Inference in a
  separate mechanical phase after the kernels are assembled.
  Preserve frozen CUDA bytes, virtual compiler filenames and artifact/module
  identities. Changing them can invalidate measured cohorts; audit first and
  requalify affected artifacts if necessary, never clear admission as a workaround.
  Source commit64d2888a; real composer identities,102 focused CUDA-host checks,
  84 non-CUDA tests and five live Ada checks pass. All60 CUDA/header files are
  unchanged;22 move into the Inference directory. Independent spec/quality
  review approved. Evidence: `internal/perf/inference-rename-20260910/`.
- [ ] Public modes: `Deterministic`, `CublasFast`, `CublasPedantic`; verify
  defaults, setters/builders, captured-policy invalidation and documentation.
  The initial API preflight found a hybrid default policy (F32 TF32 math,
  half PEDANTIC), not a clean three-mode default. Several LM-head/M3 projection
  helpers bypass context dispatch and call cuBLAS directly. Close these before
  advertising a model-wide no-cuBLAS Deterministic mode; kernel assembly alone
  does not prove model-wide dispatch. Preserve mixed half-input/F32-output
  precision rather than adding a silent half-output round-trip.
  Source map: `internal/release-mode-surface-audit-20260910.md`.
  Context foundation committed as0dcfe167: canonical GemmMode, deterministic
  constructor default, fallible capture-aware transitions, env resolution,
  actual vendor compute mapping and IDE Rustdoc. Ada focused tests/real handle
  checks, capture rejection, host identity, Rustdoc/doctests and84 non-CUDA
  tests pass; independent Task901 spec/quality review approved. High-level constructors and
  direct model dispatch seams still block completion of this release item.
  Evidence: `internal/perf/gemm-mode-api-20260910/report.md`.
  F32 routing committed as5ba6c5e2: context-routed raw NN/tied NT and M3
  projection/head dispatch. Ada routing7/7 and prepared-control54/54 pass;
  six existing GPU-ignored unit cases are not included as passes. CUDA+HF
  compilation and Rustdoc pass. Independent spec/quality review approved
  without findings. Half-input tied F32 heads are now committed21a3f0e5:
  two exact input casts plus existing ExactScalar NT directly into F32 logits,
  with pre-capture M1/M3 scratch reservation. Ada12/12 actual tests, CUDA-only
  and CUDA+HF compilation, Rustdoc and independent review pass. Evidence:
  `internal/perf/gemm-tied-f32-output-20260910/report.md`.
  Inference terminal inventory is committed7d890e6c with empty-architecture
  guard fix086716e3:85 exact terminal symbols, conditional observation,
  actual function/ABI/storage binding and existing bridge forwarding. Focused
  Ada packet has29 host+8 actual GPU passes; the review fix separately passes
  three host guards and the aligned no-op GPU case. Independent review and
  scoped fix review accepted. Evidence:
  `internal/perf/inference-route-inventory-20260910/report.md`.
  Complete M1/M3 decode/head manifests and scoped no-vendor acceptance are
  complete; prefill/training integration remains separate follow-up work.
  Task905 source is committed at96640d62, with review fix a651b133: focused Ada verification passes
  15actual GPU groups (48backbone/24head cases) and14host tests. The runtime
  packet found and fixed missing native-half TC terminal records; all three
  affected model/route cases then passed. CUDA+HF/CUDA-only compilation and
  Rustdoc passed; legacy warning cleanup remains. The review fix additionally
  passes all eight M1 stale-permit cases, the M3 lifecycle test and exact owner
  contract. Independent review and scoped fix review are accepted.
  Constructor/API and remaining prefill/training integration
  are next, not covered by this model-decode completion claim.
  Evidence: `internal/perf/model-gemm-guards-20260910/report.md`.
  Evidence: `internal/perf/gemm-context-routing-20260910/report.md`.
- [ ] Document the public API in Rustdoc alongside implementation, not only
  in README: IDE hover/completion must explain each mode, defaults, arguments,
  return values, errors, numeric/determinism scope and graph restrictions.
  Include practical linked examples and verify Rustdoc links and doctests
  with the appropriate feature/toolkit lane. Owner explicitly requested this
  as part of API work on 2026-09-10.
  Use plain, concise technical language: what the API does and how to use it.
  No marketing, filler, "magic" or AI-style boilerplate.
  Pre-API Rustdoc baseline on Ada/CUDA13.2 succeeds with broken links denied
  (2.26s), but has one existing public-to-private link warning at
  gemm_bi_triad/contract.rs:2834 (`tf32_route_specs_for`). Fix that reference in
  the documentation phase. Receipt: `internal/perf/gemm-mode-api-20260910/doc-baseline/`.
- [ ] Review the two old Codex-owned Split8 WIP files; deliberately retain,
  redesign or retire them rather than staging them as performance evidence.
- [ ] Keep useful regression and reproducible qualification tests. Archive
  useful discovery tools; remove proven duplicate/dead artifacts only after
  retaining their source and conclusions. Audit published crate contents.
  Repository-wide formatting still flags older discovery files; scoped
  formatting of this assembly block is green. Resolve the former in cleanup.
  `cargo package --list --allow-dirty` preflight includes4438 internal files,
  two tracked SDD reports, and local agent/fleet configuration. Add explicit
  package exclusions for non-shipping evidence/tooling; preserve repository
  evidence. The read-only usefulness audit now classifies253 prior tracked
  targets and97 support helpers, with mandatory extraction of independent
  source fixtures before archiving candidate stands. Current Cargo metadata
  has254 test targets including the now-retained Inference route inventory.
  This is an audit, not completed cleanup; new API tests must join the final
  census. See `internal/release-test-layout-audit-20260910.md`.
- [ ] Benchmark unchanged monolithic `main` versus final new inference/Triad
  with one immutable harness and identical settings. Use reproducible measured
  deltas in the changelog; do not multiply unrelated discovery ratios.
- [ ] Refresh README, docs, API docstrings, examples and benchmark tables;
  remove obsolete comparisons, unsupported claims and filler.
  Plain-language audit anchors: Cargo.toml package description still promises
  an opt-in tier that beats cuBLAS; README:422 labels vendor F32 inherently
  nondeterministic; device.rs:231 promises approximately8x; blas.rs:3380
  retains an obsolete throughput/dispatch narrative. Replace these with the
  implemented contracts and explicitly scoped current measurements. Line
  numbers are audit-time anchors, not frozen references.
  In particular replace stale SM120 half18-cell descriptions (current source
  has60 tiled/12 stream-K entries), do not extend CC12.0 evidence toCC12.1,
  and distinguish measured overrides from guarded SM90a/SM100 heuristics.
  SM120 half also interpolates within a guarded nearest-entry band; do not
  describe either all nearby shapes as independently measured or every
  unlisted shape as a portable fallback. Correct corresponding diagnostics.
- [ ] Bump Cargo/manifests/lockfile and changelog consistently to 0.7.0.
- [ ] Complete release tests and independent final review, then merge `main`.
- [ ] Stop before publication until the owner explicitly approves the release.

## Current GPU lanes

Ada remains available via `ssh ada`. The latest supplied RTX5090 endpoint,
`ssh -p 18481 root@61.32.91.194`, refused the latest SSH check on2026-09-10.
Saved5090 qualification remains evidence for its recorded source, not a fresh
runtime pass for subsequent API changes. Earlier rental endpoints are historical.
The 5090 CUDA13.2 production snapshot is recorded in
`internal/perf/sm120-triad-head-5090-20260909/report.md`; it is not a final
validation of subsequent WIP. Recheck utilization and free VRAM before each run.
