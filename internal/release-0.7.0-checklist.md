# v0.7.0 assembly and release checklist

Updated 2026-09-10. Worktree: `internal/worktrees/gemm-bi-triad-sm80`;
branch: `codex/gemm-bi-triad-sm80`. This is the active execution checklist;
the detailed historical evidence remains in `internal/perf/` and the handoff.

## Binding release decisions

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
- [ ] Exact-F32 TN d128-in/out module, routes and focused qualification.
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
- [ ] Reconcile every retained-winner report with production dispatch and the
  qualification harness. No candidate disappears because it loses to Fast.
- [x] Independent retained-winner census finds no further omitted Triad winner
  beyond the explicit large-TN, d128, five-cell TF32 and seven-cell half sets.
  The existing TF32 NT finalist already selects its stage-sliced body for
  d768-out/large-deep and its A-only-ldmatrix body for d768-in/Prism. The
  separate inference F16 NN d768-in N96/S3 experiment lacks an actual-AUTO
  comparison; keep it labelled as unqualified, not a missing Triad admission.
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
- [ ] Paired eager and whole-graph performance against cuBLAS Fast; also
  cuBLAS Pedantic for exact F32, with numerical contracts clearly labelled.
- [ ] Real model integration: Mamba and Mamba-3 inference graph replay,
  training graph parity/safety, and trainer full-step replay. The split
  `forward`/`backward_step` API is intentionally eager for external losses.
- [ ] Confirm architecture/toolkit portability and fallback coverage for the
  advertised SM80–SM120 families, distinguishing compile-only from live tests.

Relevant existing integration tests include `inference_graph_route`,
`training_graph_parity`, `f32_training_graph_parity`,
`m3_training_graph_parity`, `m3_training_graph_safety`, `trainer_split`,
`trainer_split_m3`, `inference_f16_smoke` and `gpu_inference_prefill_parity`.
Select their actual supported cases and prerequisites before executing them.

## API, cleanup and release preparation

- [ ] Rename `gemm_bi_fixed` / Fixed to `gemm_bi_inference` / Inference in a
  separate mechanical phase after the kernels are assembled.
- [ ] Public modes: `Deterministic`, `CublasFast`, `CublasPedantic`; verify
  defaults, setters/builders, captured-policy invalidation and documentation.
- [ ] Review the two old Codex-owned Split8 WIP files; deliberately retain,
  redesign or retire them rather than staging them as performance evidence.
- [ ] Keep useful regression and reproducible qualification tests. Archive
  useful discovery tools; remove proven duplicate/dead artifacts only after
  retaining their source and conclusions. Audit published crate contents.
  Repository-wide formatting still flags older discovery files; scoped
  formatting of this assembly block is green. Resolve the former in cleanup.
- [ ] Benchmark unchanged monolithic `main` versus final new inference/Triad
  with one immutable harness and identical settings. Use reproducible measured
  deltas in the changelog; do not multiply unrelated discovery ratios.
- [ ] Refresh README, docs, API docstrings, examples and benchmark tables;
  remove obsolete comparisons, unsupported claims and filler.
- [ ] Bump Cargo/manifests/lockfile and changelog consistently to 0.7.0.
- [ ] Complete release tests and independent final review, then merge `main`.
- [ ] Stop before publication until the owner explicitly approves the release.

## Current GPU lanes

Both hosts were reachable on 2026-09-10: Ada via `ssh ada`, RTX 5090 via
`ssh -p 18481 root@61.32.91.194`. Earlier rental endpoints are historical.
The 5090 CUDA13.2 production snapshot is recorded in
`internal/perf/sm120-triad-head-5090-20260909/report.md`; it is not a final
validation of subsequent WIP. Recheck utilization and free VRAM before each run.
