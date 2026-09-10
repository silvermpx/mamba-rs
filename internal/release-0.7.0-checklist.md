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
  Layout committed as c65c1c85: explicit targets (108 tests, 15 benches, 46
  qualification tools behind the non-default feature), lane census over the
  manifest, 96 stands plus their candidate sources and adapters archived under
  `internal/experiments/release-0.7.0-archive/` with an index, mixed stands
  split with the compact-finalist and scalar-NT receipt fixtures extracted,
  shared harness as single-purpose modules, the five shipping lint allowances
  removed by restructuring, retained callers of the deprecated setters moved
  to `set_gemm_mode` by intent, package excludes in place. Ada packet: fmt,
  four featured checks with zero rustc warnings, CI-form clippy, 274 host
  tests, library/bench compiles, one CPU bench run, NCCL check, Rustdoc with
  broken links denied, package; extracted crate holds no internal material and
  its 275 host tests pass offline. Evidence:
  `internal/perf/release-layout-20260910/report.md`.
- [~] Benchmark unchanged monolithic `main` versus the release tree on the
  RTX 6000 Ada with one adapter per lane and identical settings
  (`internal/perf/old-versus-new-20260910/`, `report.md`). Done: set A
  (inference step, exact f32, Triad lane: eager equal, graph replay 20 to 30
  percent slower on the release tree), set C (training step, all three GEMM
  settings, 2 to 17 percent faster like for like, cuBLAS arms agree between
  trees). Running: set B (inference step on the inference family, f32 and
  bf16) and set D (kernel level, every op and dtype, each tree's deterministic
  route beside its cuBLAS Fast and Pedantic arms). Open: the set A graph
  replay slowdown was traced to the per-replay launch-set digest rebuild in
  `CapturedGemmGraphPlan::with_validated_launch`; the digest check moved to
  plan construction and the replay validates routes once against the live
  identity. Re-run set A on the fixed tree to confirm before the numbers are
  published.
- [~] Refresh README, docs, API docstrings, examples and benchmark tables in
  plain language. Done in the working tree: CHANGELOG 0.7.0 rewritten as a
  performance release note (the tree's expanded 0.6.9 section was not the
  published 0.6.9 entry and was folded into 0.7.0; the published text is
  restored); new `docs/gemm-modes.md`; `docs/determinism-benchmarks.md` is the
  per-kernel benchmark page for both boards against cuBLAS Fast and Pedantic;
  README, both architecture pages, both benchmark pages, the playbook, the
  qualification runbook, the examples, the crate docs and the GPU module docs
  corrected per the documentation critique. Open: the old-versus-new sections
  of the benchmark page and the changelog take the set B and set D tables when
  the runs finish; the `gemm_bi_fixed_*` targets are renamed
  `gemm_bi_inference_*`.
- [x] Bump Cargo/manifests/lockfile and changelog consistently to 0.7.0.
- [x] Re-verify that every Inference and Triad kernel is wired (run on Ada
  after the documentation batch). Green: `coverage_gaps`, `gemm_bi_tf32_selector`,
  `gemm_inference_route_inventory`, `gemm_context_routing`,
  `gemm_bi_contract_census`, the SM89 TF32 joint selector. Fixed on the way:
  the sm_89 compile gate's expected TF32 export set, the invariance matrix's
  scalar-tier arm (tensor cores are on by default since 0.7.0), and the half
  qualification harness, which since `96640d62` saw the native half branch
  record its route and refused it as "unexpectedly recorded F32 routes" (the
  three ignored library tests and `gemm_bi_sm89_half_selector_qualification`
  failed on that alone). The exact-f32 selector tools need
  `MAMBA_SM89_EXACT_F32_EXPECT_AUTO=1` / `MAMBA_SM89_EXACT_F32_D128_EXPECT_AUTO=1`
  for their post-admission arms (the pre-admission arms describe the state
  before admission and are historical); the TF32 selector qualification needs
  a fresh `MAMBA_RS_TF32_SELECTOR_JSONL`. A symbol inventory (every kernel in
  `kernels/` against every name a loader asks for) found 19 handles nothing
  launched and 41 kernels behind them; deleted, except the vectorized
  elementwise multiply and softplus copy twins, which are now wired at their
  f32 sites. The four `sm120_scalar_nt_*` library tests are RTX 5090 only and
  fail on Ada by design.
- [x] The seven-argument contract test (`gemm_bi_tf32_contract`, gate lane)
  was red before the kernel pass: `GemmBiKernels::load` took fifteen
  arguments. They travel as `GemmBiModuleSet` now.
- [ ] `cargo clippy --all-targets --features cuda,hf,qualification -- -D warnings`
  still reports 16 findings the release gate never ran: three argument
  counts outside the contract's sources (`fixed_pick_tf32` 10,
  `fixed_select_sm89_half_auto_tile` 8, the `spec` const fn 12) and
  thirteen in the library's own test modules (complex tuple types, an
  8-argument test helper, index loops, an OR pattern, a late init, a
  `vec!`, items after a test module). Not release-blocking; decide whether
  to clear them or add clippy with the CUDA feature to the gate first.
- [ ] `gpu_forward_mamba_target_burnin` and `gpu_forward_mamba3_target_burnin`
  (the target-network forwards RL consumers call) have no test in the tree.
- [ ] Audit the non-GEMM kernels (sequential and chunked scans, conv, norms,
  the dispatchers) for math and performance against `reference/mamba` and the
  knowledge base, then re-measure the training and inference steps and update
  the tables.
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
