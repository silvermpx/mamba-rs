# Full kernel wiring plan — every kernel, every card class, every toolkit

## Current execution override — efficient discovery, 2026-09-08

The user reaffirmed the binding source/profile-guided search method in
`internal/agent-operational-rules.md` (first section): primary NVIDIA/CUTLASS/PTX
research, parallel independent agent work, concrete reusable candidates, focused
bit/resource checks and short paired eager/graph once7 screens. Stop unchanged
valid losers. Compare new candidates with the retained best and explicit cuBLAS
Fast; exact F32 keeps its exact arithmetic even when Fast is faster. PEDANTIC
results below are historical reference evidence, not the current speed target.
Assemble finalists across the whole Triad before joint integration and ONE
CUDA12.8/13.0/13.2 qualification batch. The older per-candidate/full-gate ordering
below does not apply to current discovery. Fixed inference and SM120 routes stay
unchanged; no fresh5090 performance claim while that rental is unavailable.

## Active dual-GPU execution amendment — 2026-09-05

The owner's latest instructions authorize concurrent Ada and RTX 5090 work.
The objective is BOTH Fixed inference and Triad NN/TN/NT on BOTH GPUs, for
BF16, FP16, TF32 and exact F32. Existing completed phases below are history,
not a reason to repeat them. The checkpoint entering this amendment is
`5b81925b`; current progress is in `internal/ada-continuation-2026-09-05.md`.

### Binding constraints

- Fixed retains cross-batch and real row-subview byte equality. Both families
  retain repeated eager and CUDA Graph byte equality, including exceptional
  values. Tolerances belong only to independent numeric reference checks.
- TF32 compares against explicit cuBLAS FAST_TF32; half compares against native
  half Tensor Core compute; exact F32 compares against PEDANTIC. Never trade
  exact F32 for TF32 to manufacture a performance win.
- Each physical GPU has one owner of timed jobs. Check other compute jobs,
  clocks and temperature before timing. Never kill another user's process.
- Ada lane: `ssh ada`, qualification source
  `/root/mamba-rs-codex-ada-qualification-20260905`, CUDA 13.2 / 12.8.
- New SM120 lane: `ssh -p 19682 root@175.155.64.241`, isolated source
  `/root/mamba-rs-codex-sm120-20260905`, initially committed `5b81925b`, CUDA
  13.2. Old proxy addresses are stale. Long runs use tmux. No port-forward is
  needed for CUDA work.
- SM120 availability now permits future measured changes, but does NOT waive
  composition/ABI/identity gates. An Ada-only change must preserve unrelated
  SM120 and Triad bytes. A deliberately changed module needs fresh evidence.
- Winning qualified kernels enter AUTO immediately in the same implementation
  phase, scoped to tested architecture/compiler/shape/dtype/epilogue. No
  indefinite force-only parking. First force-only wiring is a test stage, not
  the deliverable. A failed gate blocks promotion, not further investigation.
- Remove a beaten kernel from AUTO only where its replacement is qualified.
  Retain required fallback/other-device/other-shape uses and explicit forced
  access. Record deletion candidates before any eventual source removal in
  `internal/kernel-retirement-candidates-2026-09-05.md`.

### Work streams and acceptance

| Stream | Deliverable | Files / evidence | Immediate verification |
| --- | --- | --- | --- |
| Ada Fixed half | Production-owned five-argument pipe_vec kernel, live ABI/resource admission, forced bits, then measured AUTO cells | `kernels/gemm_bi_fixed/sm89_half_pipeline.cu`, `modules.rs`, `kernels.rs`, `gemm_bi_fixed.rs`, Fixed correctness/performance and arch compile tests | Cold NVRTC; all portable half rungs vs candidate, batch/view/tail/NaN/K0/eager/graph bytes; actual graph symbol; paired old/new/vendor |
| Ada Fixed TF32 | RNA-compatible wide route preserving old Fixed arithmetic, then winning exact-cell AUTO | `internal/experiments/fixed-rna-production-integration.md`; separate Fixed CUDA fragment and same module/host/test boundaries | All five old Fixed TF32 rungs vs RNA, including payloads; matched compiler timing, not NVCC-only results |
| SM120 baseline and Fixed | Reproduce existing production bindings/bits, profile remaining losses, qualify and route winners | `internal/experiments/sm120-restart-report.md`, `internal/perf/sm120-restart-20260905/`, Fixed selector and qualification suites | Fresh board identity; no silent architecture skips; exact actual launch census and same explicit denominators |
| Triad on both GPUs | NN/TN/NT winners in AUTO in every precision, with repeat/graph determinism | Triad dispatcher/launch/qualification and existing TN/NN experiments; separate per-GPU evidence | Fixed-order reductions only; eager and graph performance separately; no numerical atomics |
| All architecture wiring | Complete source -> composition -> PTX -> load -> forced -> AUTO candidate -> actual launch map | `internal/experiments/gemm-bi-wiring-inventory-20260905.md`, qualification/CLI/contract/module tests | SM80/86/87/89, SM90a, SM100/103/110, SM120/121 compile and route gates; physical hardware results only for cards present |

### Ordered steps inside each implementation phase

- [x] Freeze standalone Ada half screening: 20/20 cells passed all gates;
  pipe_vec beats old Tc128 in all 20, native cuBLAS in 14. This is NOT yet
  a production NVRTC/AUTO qualification.
- [x] Observe RED for missing wide TF32 qualification inventory, vector-tail
  strides and stale CLI/per-op counts. Existing CC12 ordering tests pass.
- [x] Implement the target-aware qualification inventory once, reuse it in
  CLI/per-op census; preserve CC12 order and fix wide-specific exceptional
  oracle through RED/GREEN. This is independent host qualification repair.
- [x] Observe RED for Ada Fixed half extension admission and forced launch.
- [x] Add only the winning pipe_vec bodies, with five-argument production ABI:
  `(C,A,B,bias,params32)` and params `{alpha,beta,m,n,k,lda,ldb,ldc}`. Keep old
  CUDA files unchanged. Admit only sm_89; live ABI, 71680 shared, zero local,
  register ceiling 224, 256 threads and occupancy >=1 are mandatory.
- [x] Run production NVRTC forced bits and reference gates; then paired
  eager/graph old/new/vendor 101-window final measurements on hot A-E, both
  bias forms. Include M=1/15/16/17/31/32/33/63/64/65/127/128/129/255/256/257,
  2047/2048/2049, and each exact promoted M-1/M/M+1 for cross-route bits.
- [x] Promote winning cells, assert actual AUTO graph symbol and repeat the
  cross-route prefix test. Update winner/retirement journals and tell owner.
- [ ] Complete new 5090 baseline, prioritize its worst verified Fixed losses,
  and run the same qualify -> promote -> report loop there concurrently.
- [ ] Repeat that loop for Triad NN/TN/NT on both cards; profile the largest
  verified loss with NCU, form one scheduling hypothesis, run same-build
  twins, and reject regressions. NCU durations are not benchmark evidence.
- [ ] Close wiring gaps for physical HalfPolicy AUTO Stream-K, forced
  SM90/100 Fixed admission and exact-FMA census, without asserting absent
  GPU performance. Re-run supported toolkit/target compile gates.
- [ ] Run whole-phase correctness, compiler/ABI/identity/resource gates,
  clippy and review; commit locally. No main merge or push.

Checkpoint `e72fbab4`: the first G1/Ada-half implementation phase is committed
locally, including20 old-kernel wins/12 robust vendor wins in AUTO, full19
TF32 runtime100repeat plus4sanitizers, newhalf4sanitizers, cold/warm cache,
557lib/85contract/11CLI/12comparatorCPU andall50CUDA13.2 architecture gates.
NVRTC12.8 forced-half3GPU PASS240.62s (CUDA13-built Rust binary; expected
Triad tensor-map ABI decline), separately NATIVE12.8 Rust557lib+clippyPASS.
Native12.8contract85PASS; full12.8architecture48PASS/2FAIL: existing scalarNT
16Bstack/spill and unsupported sm_110a requested by extension test. Diagnose
without weakening the no-spill gate. The broader both-card, all-family
acceptance above is NOT complete. G2 minimal d59d64b1 committed after actual
twoGPURED->GREEN, full559lib/50matrix/1censusCPU/clippy and4kernel sanitizers
PASS, independent review. Explicit half permission and TNrect graph holder
fixed; CUDA arithmetic and selectors unchanged. Expanded R4 committed225b4f68:
actual72AUTO+22forced+6reuseholders, permission/boundary/offset/lease proof,
matrixmemcheckPASS. G5 boundproduction exactFMA12-route runtime census still
missing (loader ABI already covers all12); future split1/2/3/4/5 coverage
must not be inferred from experimental or zero-output diagnostic tests.

Checkpoint738fe5ad restores three fresh-driver5090 TF32 AUTO cells after
full36+fourtools and unchanged five-cell21/101selector. ActualAUTO symbols/
graphs/bits testPASS77.16s; CUDA lib564PASS onbothboards andAda oldcohort2PASS.
Old F32 qualification facade guard census was zero; separate full36 guard
evidence is valid, facade selector/cacheprobe guard claims are not. Next
reviewed guarded facade/vendor comparator runs only on NEW snapshot, without
altering frozen timings. Ada exactN64 standalone four own wins/one PEDANTIC
win confirmed101; bounded14fixtures/allfourOWNsanitizers PASS with byte-equal
SASS. ProductioncompactABI/actualNVRTC qualification inprogress; noAUTOyet.

Ruling: the old blanket “no new kernels until every one is measured” cannot
require unavailable architectures to block paid Ada/5090 optimization. All
existing target-valid kernels remain reachable/qualifiable; the independent
wiring audit continues while measured isolated candidates are integrated.
Cost if wrong: additional candidate maintenance; no existing kernel is erased.

Ruling: no kernel is globally retired from one losing cell. A deletion needs
a documented replacement for every live use, including future hardware
qualification. Cost: temporary duplicate storage, preserving safe fallbacks.

Written 2026-09-03 (evening) after five read-only audits over the working tree
(kernel census, triad reachability, fixed-family reachability, arch/toolchain
admission, measurement evidence), each spot-checked by hand against the code
before anything below was written. Reports live in the session scratchpad
(`audit/A..E-*.md`); this file carries only what was verified.

## Decision of record (owner, 2026-09-03)

Every compiled kernel gets a production dispatcher path for its card class
(SM80/86/89, SM90a, SM100/103/110, SM120/121) and for every toolkit
(12.8 / 13.0 / 13.2). An unreachable kernel is a defect. Duplicates are
measured on the same box and the loser is removed, with the numbers in front
of the owner first. The optimisation target is cuBLAS FAST parity (exact F32
is judged against PEDANTIC — FAST is TF32 math). Measurement boards: RTX 5090
(SM120, 170 SM, driver 595.84, CUDA 13.0 + 13.2) and the RTX 6000 Ada on ada
(SM89, 142 SM, driver 595.45.04, CUDA 12.8 + 13.2). SM90a / SM100 get code and
compile gates now, numerics and perf on rented boards by the owner's word.
Release 0.7 afterwards: merge to main, rename fixed -> inference, docs.

## Verified state

Counts (census A, re-derived by B): 420 GEMM symbols in six NVRTC modules; at
most 290 exist on one board; **231 are compiled and no production selector can
name them**:

| family | dark | blocking gate |
|---|---|---|
| SM120 half TMA | 78 / 96 | `SM120_AUTO_CELLS_CC120` is an exact 8-tuple table (dispatch.rs:1695-2207); `resolve_sm120_auto_from_cells` :2245 does `find` and nothing else; CC 12.1 table empty (:2208) |
| SM120 exact F32 | 6 / 12 | measured table never uses M64N64; generic rule hard-codes one (tile,kvec) per op (:1616-1620); floor 3 tiles/SM (:1608) |
| SM120 TF32 non-FMA | 13 / 18 on the live box | no cohort cell names them (:404-1215) |
| SM80 portable TF32 | 5 / 18 | same |
| TF32 split-K | 6 / 6 | no cell constructor emits a split-K route (:387, :664, :689) |
| SM90a half + TF32 | 18 / 18 | `SM90A_AUTO_CELLS = &[]` (:1688); no `resolve_sm90a_auto`; `launch_sm90a_wgmma_forced` has no caller in src/ |
| SM100 half + TF32 | 108 / 108 | `SM100_AUTO_CELLS_CC100/_CC103 = &[]` (:1689-1690); no `_CC110` constant; no `resolve_sm100_auto` |
| `TcTile::Rect128x64` | 2 / 16 | never constructed (:5177-5179) |
| tuned scalar cells under `tf32` policy | 8 symbols | `dispatch.rs:2590` requires `policy == ExactScalarFmaV1` |

Fixed (inference) family (C): off by default (`MAMBA_RS_BATCH_INVARIANT=1` +
`MAMBA_RS_BI_GEMM_FAMILY=fixed`, blas.rs:2979); its SM120 half kernels ignore
alpha/beta (`GbfSm120HalfDirectOutput<T>::value = true`, sm120_tma.cu:478)
and are safe only because the launcher sends 1.0/0.0; two of five SM120 half
tiles reachable only through the four-way-pinned overlay; the wide-deep
comparator picks a measured loser (50 us/call at (4096,3072,1536): auto 228.2
vs m64n128 178.2, cuBLAS 181.7) and cannot return the winner at all; exact F32
is 1.51-1.98x cuBLAS on all five hot cells with three on `Legacy`.

Duplicates (A, C): 14 name-identical C symbols across Fixed and Triad
(`gemm_bi_nn_tc64_*`, `gemm_bi_nn_tc16_*` on every device; 10 SM120 half NN
names; `gemm_bi_nn_sm90a_wgmma_wg1_*` with incompatible ABIs), plus five
geometry-twin families (six of seven Fixed SM120 half tiles duplicate a
TriadSm120 NN tile exactly; five TF32 tiles; five SM120 TF32 tiles; the
SM100 tcgen c4 pair). They coexist only because each family is its own
CUmodule.

Admission (D): two arch derivations disagree (device.rs:119 vs :153) and the
version-blind `nvrtc_arch()` panics (:186); loader keys on the arch string
(kernels.rs:849/888/896 load SM120 Fixed kernels on any major > 12) while
every selector keys on `(12,0)|(12,1)`; the triad SM120 module is not even
compiled above 12.1 (kernels.rs:558); eleven literal 170-SM pins (nine are
honest evidence scope, `gemm_bi_fixed.rs:1240-1252` and `modules.rs:5383` are
defects); `--frandom-seed` exists only on NVRTC >= 12.9, so a 12.8 compile key
can never equal a 13.x key (three cohorts per board by construction); no
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` handling anywhere; every decline on the
selection path is a bare `None` and `dispatch.rs:1335` discards the one
human-readable rejection the tree builds.

Evidence (E): triad half has no dispatcher loss left (production symbol is the
census winner on all 24 cells after commit 1e3bbdb). Exact F32 loses 399 us
per call on `large_deep` across the three ops against FAST, but is at parity
with PEDANTIC on 11 of 12 cells (only tn/large_deep at 1.031). Deterministic
TF32 beats FAST on 8 of 12 cells; three `d128_out_proj` TF32 cells are
`qualified: true, admitted: false` and fall to scalar, forfeiting a measured
1.02 / 1.60 / 1.33x. The plan-of-record's 7.17x tail figure and the row-127
cliff were measured on **ada (CC 8.9)**, not on the 5090. The SM89 TF32
cohort was frozen on ada on 2026-08-28 (142 SM, 595.45.04, 13.2); whether its
source digest still matches the sm80 module is unknown until ada runs it.

## Rules for the work

1. Fail-closed admission with a **visible** decline: one `mamba-rs WARNING:`
   per process at bind time naming the first identity field that did not
   match. A wrong-SKU box must be distinguishable from a served one.
2. Evidence-scope pins stay scoped; heuristics take the live SM count; the
   loader and every selector share one CC -> family predicate.
3. Nothing is removed without a paired measurement on the same box, and the
   owner sees the table before removal.
4. Any edit to an SM120 module rotates the source digest and kills the TF32
   cohorts: such edits land together with a requalification, never alone.
5. Phase = build whole -> one gate cycle -> commit. Gates run on the 5090
   (perf + tests, 13.0 and 13.2) and on ada (tests under 12.8 and 13.2; perf
   once the serve is stopped).
6. No new kernels until every existing one is reachable and measured.
7. Comments and commit messages describe the defect, never a plan index.

## Phases

### A — visibility and portability plumbing (host code; gate on both boxes)

- A1 Report declines once per process: `dispatch.rs:1335` stop discarding the
  binding error; `launch.rs:6263-6271` use the three messages its graph twin
  already carries (:6295-6311); `modules.rs:786-819` artifact-set failures;
  `kernels.rs:558` driver-query failure; `resolve_sm120_forced` :2400-2406
  reasons; `select_sm100_candidate` :751-761. Idiom: `arch_rung_enabled`
  (gemm_bi_fixed.rs:1486-1506). Expose `specialized_tf32_rejection` on the
  production path (today read only by one tournament).
- A2 One CC -> family predicate used by the loader (kernels.rs:558, :849,
  :888, :896) and by every selector; CC 12.1 sites in the fixed family
  (:431, :440, :459, :916, :940, :1149, :1292) move to it. Policy for major
  > 12: portable path plus warning until a board exists.
- A3 Live SM count in heuristics: `gemm_bi_fixed.rs:1240-1252`; the scalar
  resource verification (`modules.rs:5376-5392`) runs on every board.
- A4 Dead guards: tuning revision frozen per cell (dispatch.rs:1254);
  portable coupling keyed on the route's module kind (:1272-1279); the
  staleness gate becomes strict (every cohort matches or is retired) and
  runs at bind time, not only under `cfg(test)`; `sm120_cohort((13,2))`
  helper (:6378) points at a live cohort.
- A5 Compile-key hygiene: drop `-DMAMBA_RS_STATE_CAP` from the triad argv
  (modules.rs:522); include paths leave the invocation digest for the header
  manifest (kernels.rs:1062-1084). Rotates the key once; lands with B10.
- A6 Env flags fail loud: `MAMBA_RS_BI_GEMM_FAMILY=fixed` without
  `MAMBA_RS_BATCH_INVARIANT` (idiom context.rs:400-407),
  `MAMBA_RS_BI_TENSOR_CORES` under fixed, `MAMBA_RS_FAST_GEMM` under
  batch-invariant, `MAMBA_RS_ARCH_RUNG` values other than `off`; the
  inference path applies `apply_env_route` (inference.rs:306).
- A7 `nvrtc_arch()` returns `Result` (device.rs:186); CC 10.1 arm
  (kernels.rs:595, sm100.cu:1); CC 11.0 Fixed rung (kernels.rs:979);
  `arch_compile_gates` compiles sm_86 / sm_87 too.
- A8 Harness cost: memoize NVRTC per (source, arch, options) in
  `arch_compile_gates`; NVML in-process quiet gate; bound-out final windows
  for candidates below the incumbent's discovery score.
- Tests that pin the old scope (about twenty host-lane tests, D section 7.5
  and 5.7) move deliberately with each change, never "fixed to pass".

### B — SM120 on the 5090: make every kernel reachable, then measure it

- B1 (refuted by hand, 2026-09-03 21:00): the `d128_out_proj` TF32 cells are
  `admitted: false` because the EAGER winner is `scalar_fma_v1` (only the
  graph winner is a TF32 kernel; `selector-152009.jsonl`); the selector
  requires a win on both paths, so declining is by design on a 1024x256x128
  cell. No change.
- B2 `dispatch.rs:2590` drop the `ExactScalarFmaV1` conjunct so the ten tuned
  scalar cells serve under the `tf32` policy too.
- B3 TF32 split-K: a cell constructor that can emit
  `MmaTf32RnaSplitK{2,4,8}V1`; deep-K census ((128,8192,128), the 10400-batch
  dW shapes) against FAST; `deep_split_k_compute_capability` (8,9) widened to
  a measured k crossover that includes (12,0).
- B4 Exact F32: forced census of all twelve kernels (the four M64N64 arms and
  two NT arms have never run) on the nine hot shapes, `large_deep`,
  `prism_out_proj`, `prism_input_proj` and the 10400 batch; the generic floor
  moves to a measured value and gains an M64N64 ladder; denominator PEDANTIC.
  Routing nn/large_deep to the TF32 route (376.96 vs 491.85 us) is a policy
  decision for the owner, not a kernel.
- B5 Triad half: a shape-class rule beside the exact table (wave arithmetic
  on the live SM count) so the 78 dark tiles have a path; forced 16-arm
  census on `prism_out_proj`, `prism_input_proj` and the 10400 batch; CC 12.1
  is served by the rule while its exact table stays empty (fail-closed).
- B6 TF32 non-FMA: the requalification covers all 18 routes on the 21 + 5
  shapes; a route that lost the tournament everywhere is a measured loser
  (kept as a candidate, removed only when dominated on every shape); a route
  that was never a candidate is a defect.
- B7 TC tier on SM120: the edge matrix (tail gates 256 / 4 CTAs,
  `large_tile_min` 128, the half-wave rule) is measured on the 5090 first,
  then the constants move to measured boundaries with an SM120-tuned arm
  (dispatch.rs:5328 has one for CC 8.9 only); `Rect128x64` gets an arm and a
  measurement.
- B8 note (2026-09-03 21:40, refuted in part): the plan-of-record's "weight 4
  where the census says 2" is not a defect: the 17 measured anchors of
  `sm120_half_deep_selector_matches_measured_170_sm_anchors` (k 1032..2304)
  are reproduced by weight 4 and several would flip under weight 2. The loss
  is an extrapolation past the fitted band (k 3072): fixed by a measured
  overlay cell (4096,3072,1536) -> M64N128Bk64S2, 228 -> 178 us; a deep-band
  census (k >= 2560) is what a rule needs.
- B8 Fixed family on SM120: comparator weight 2 and an M64N128 arm; the
  generic picker reaches all five tiles; the TF32 picker gets wave arithmetic
  instead of one tile; the three `Legacy` exact-F32 cells get SM120 routes;
  alpha/beta are asserted at launch so the epilogue trap cannot fire.
- B9 Duplicates Fixed <-> Triad: pairwise census on the same shapes (six half
  tiles, five TF32, five SM120 TF32, tc64 / tc16, sm90a wg1); the loader
  points both families at the faster body; the table goes to the owner before
  any removal.
- B10 One requalification and identity refreeze after all kernel and argv
  changes of the phase; full gate; commit.

### C — SM89 on ada: the portable ladder on its home board

- C1 Gate under CUDA 13.2 (running: lib, arch_compile_gates, tf32_contract,
  census, source gates, determinism, invariance).
- C2 The same gate under CUDA 12.8 (`CUDA_HOME=/usr/local/cuda-12.8`); the
  12.8 compile key differs by construction, so cohort declines are expected
  and must be visible after A1.
- C3 SM89 TF32 cohort freshness (source digest against the sm80 module); if
  stale, requalify on ada once the serve is stopped (selector suite, about
  thirty minutes).
- C4 Perf matrix on ada: hot shapes in half / TF32 / exact against cuBLAS
  FAST and PEDANTIC -> the SM89 target list; the tail and row-127 constants
  are measured where they were first observed.
- C5 Invariance-matrix edge tables recorded for sm_120 (they exist for sm_89
  only).

### D — SM90a / SM100 resolvers (code and compile gates now, boards later)

- D1 The 26-item port list from the SM120 resolver (B section 4.2): request
  structs, `resolve_sm90a_auto` / `resolve_sm100_auto`, cell tables (empty,
  fail-closed), `sm*_device_caps`, `sm*_kernel_resources`, observed launch
  and graph sequence, prepared-launch cache and seals, the three blas call
  sites, operand and binding validators, identity revisions (SM90a has none),
  target candidates with NVRTC arms, `SM100_AUTO_CELLS_CC110`.
- D2 TF32 cell constructors and empty cohorts for `Sm90aWgmmaTf32TmaV1` and
  `Sm100Tcgen05Tf32TmaV1`.
- D3 `arch_compile_gates` on the 5090's toolkit compiles sm_90a / sm_100a /
  sm_103a / sm_110 PTX and runs the ABI and resource validators; forced
  census tests stay ignored until a board exists.
- D4 On rented boards (owner's word): forced census -> cells -> cohorts ->
  requalification -> the same target list as B.

### E — full retest on both boxes, rental, release 0.7 (order unchanged).

## Immediate order of work

1. C1 result, then C2 (ada, 12.8).
2. A1 + A2 + A3 and B1 + B2 as one phase: host code plus two admission fixes
   with recorded evidence; gate on both boxes; commit.
3. B4 exact-F32 forced census on the 5090 (largest measured loss, 399 us per
   call on one shape) as an evidence run before any exact-F32 code moves.
4. B5 half rule + census on the product shapes; B3 split-K; B7 edge matrix.
5. B8 + B9 (fixed family and the duplicate table) with the owner's decision on
   each removal.
6. D1-D3, then A4/A5 + B10 requalification and refreeze.

## Progress log

- 2026-09-03 21:00 CEST — commit c447234: declines report once (`diagnostics::warn_once`),
  one SM120 family predicate (`device::is_sm120_family`) for loader and selectors, live
  SM count in the fixed half picker, tuned scalar cells under both f32 policies.
- 21:15 — commit 7b72c7f: 30 half cells for the five product shapes (table 60 cells, all
  qualified eager==graph on the 5090; evidence sm120-half-census-20260903b).
- 21:30 — ada (SM89, 13.2): full gate green on 1e3bbdb; included-ignored suites green except
  the other-arch hardware gates (by design). Hot shapes measured (sm89-hot-20260903/SUMMARY.md):
  TF32 cohort dead — the new report names it: "differs at: compile key"; exact f32 on the
  generic scalar family 0.96-2.25x PEDANTIC; half on SM80 tiles 1.13-1.51x cuBLAS. The SM89
  cohort held five synthetic TN cells only; the selector harness is now board-generic and a
  21-shape SM89 qualification is queued on ada.
- 21:35 — ada under CUDA 12.8: `compute_100f` unknown to nvrtc 12.8 (family targets are
  12.9+) and the SM100 tcgen SASS pairing differs on 12.8. Fix: the SM100 family is offered
  only on 12.9+ (`sm100_target_candidates_for_nvrtc`), the contract pins scoped the same way.
- 21:40 — exact-F32 screen on the product shapes (sm120-exact-wide-product-20260903/SUMMARY.md):
  nine cells beat the exact policy's current resolution by 1.21-2.07x; promoted into
  `SM120_FMA_MEASURED_CELLS_CC120_170` (18 cells) with the generic floor lowered 3 -> 1
  tiles per multiprocessor; production-path proof pending (perf matrix now carries the
  five product shapes). Four TN cells with a 10400-long reduction still lose to the scalar
  split-M route: deeper splits (8-16) are the next arm to screen.
- Fixed family: the deep wide projection (4096,3072,1536) gets a measured overlay cell
  (M64N128Bk64S2, 228 -> 178 us); the plan-of-record's "weight 4 vs 2" claim is refuted by
  the 17 measured anchors of the comparator's own test.
- 22:05 — commit: nine exact-F32 cells + generic floor 3 -> 1 (production path on the
  product shapes 0.62-0.79x PEDANTIC, evidence sm120-exact-product-20260903); perf
  matrix 22 shapes / 742 cells.
- ada under CUDA 12.8, arch_compile_gates: ptxas 12.8 spills 8 bytes in
  `gemm_bi_nn_sm120_tma_mma_tf32_v1_m80n32_bk64_s2` (rect-wide); the runtime rule
  "zero local memory or the whole module is rejected" (modules.rs
  `validate_tf32_driver_jit_local_memory_facts`) would then darken every SM120 TF32
  route on a 5090 running CUDA 12.8. Next item A9: per-symbol resource admission
  (exclude the spilling route, keep the module), and the arch gate expects that
  exclusion on 12.8.
- 23:20 — commit 5b5e9e8: nearest-cell half rule with the underfill guard (held-out
  shapes: best tile on 9 of 12 in-band cells, worst loss 1.24x), per-symbol TF32
  admission (a spilled kernel no longer rejects the module), quiet gate idle limit.
- 23:25 — SM89 TF32 cohort refrozen from the board-generic selector (21 shapes, 31 min
  on ada): 14 cells replaced, 7 added (one split-K winner via `sm89_tf32_route_cell`),
  identity from the run; production path binds TF32 on ada (sm89-hot-tf32cohort-
  20260903/SUMMARY.md): 1.68-2.64x FAST, 0.79-1.29x PEDANTIC, 1.2-2.3x faster than the
  exact route it fell to before. Gate on the 5090 pending, then commit.
- 23:45 — commit 045dddf: SM89 TF32 cohort from the 21-shape qualification (41 cells,
  split-K winner via `sm89_tf32_route_cell`); five-cell harness passes on ada.
- 23:50 — commit: `resolve_sm100_auto` / `resolve_sm90a_auto` with per-CC tables (CC 11.0
  table added), operand law and forced admission, host tests; published next to the forced
  resolvers. REMAINING for D1: `Sm100RouteIdentity::resolved_route` / SM90a twin (for the
  physical observation), an observed enqueue for SM100 (`launch_sm100_tcgen_prepared` launches
  unobserved) and SM90a, `launch_sm100_auto_observed` / `launch_sm90a_auto_observed` with a
  capture-time decline, `HalfPolicyBranchSeal::{Sm100,Sm90a}` and the qualification seal arms
  (qualification.rs:827, :2041-2060, :1885, :5425), the three blas.rs call sites after the
  SM120 attempt.
- 23:45 — deep-reduction selector cells (split_candidate 128x8192x128, batch_in_proj
  10400x384x1536, three ops) added with a cell filter env
  (`MAMBA_RS_SM120_TF32_SELECTOR_CELLS`). First run on BOTH boards: the split-K TF32 arms
  fail the selector's numeric gate at K=8192 (5090: splitk4 m16n32 s4, ada: splitk2 m16n32
  s4; candidate -0.1217/-0.1229 vs portable reference -0.1247/-0.1260, tolerance 0.0028).
  The reference is itself TF32 in another summation order and the output cancels heavily
  at that depth, so the gate's tolerance model is the suspect as much as the kernels; the
  split-K family stays out of automatic tables at deep K until that is settled. The harness
  now excludes a candidate that fails the gate (reason `bit_gate_mismatch`) instead of
  aborting the cell.
- 2026-09-04 00:15 — deep-reduction cells measured on both boards (summary in
  internal/perf/sm120-requal-deepk-20260903/SUMMARY.md): four cells enter each cohort
  (5090 cohort 595.84: nn/nt batch_in_proj, tn/nt split_candidate; SM89: the same four
  shapes). Split-K TF32: no win on 72 cells across two boards, and the numeric gate
  rejects the split arms at K=8192 (and the SM120 stream-K arm at a 10400 reduction).
  Verdict for the split-K family: candidates for removal once the gate's
  tolerance model at deep reductions is settled (B3 closed as "measured, no win").
  The selector harness now excludes a gate failure per candidate and records the
  measured cell count when filtered (`MAMBA_RS_SM120_TF32_SELECTOR_CELLS`).
- 2026-09-04 01:00 — D1 closed. The SM100 and SM90a automatic branches launch from the
  three half entry points after the SM120 attempt (a2d7a71, 4793449): a prepared-launch
  cache per family keyed like the SM120 one, an observed enqueue sealed by the qualification
  harness (`HalfPolicyBranchSeal::{Sm100,Sm90a}` with their validators), and a capture arm
  that replays the eager preparation. The SM90a prepared maps carry the managed epoch of
  every buffer the launch touches. Resolver tests follow the wave rule (d57e071). Gate on
  the 5090 (target-check): clippy, lib 527, sm120/sm100/sm90 contracts green. The identity
  test's Fixed TF32 pin now follows the compiled target (7cb115f) — on ada the PTX carries
  only the five portable entries; green on ada under 12.8.
  Full workspace runs on both boxes (started 22:24Z) show only the three failures fixed
  since (two SM120 pins from ea241db, the identity pin) — re-run after they finish.
  D2 in progress: the TF32 cohort search follows the bound module family (empty SM90a and
  SM100 cohort lists, family-named decline report).
- 2026-09-04 01:20 — D2 landed (3b5feb1): the TF32 cohort search follows the bound module
  family; SM90a and SM100 hold empty cohort lists and decline by name. A6 landed (4da099d):
  the no-op flag combinations fail at context creation (fixed family without the
  batch-invariant dispatch, fast cuBLAS compute under it, any arch-rung value but off), the
  rung site warns on an unknown value, and the inference engine reads the same flags. The
  provenance census audits the six new observed scopes (2f83f28). Gate on the 5090:
  clippy, lib 530, tf32 contract 81, sm120/sm100/sm90 contracts, inference_graph_route.
  In gate: A7 — CC 11.0 moves from the baseline sm_110 target to sm_110a so the Fixed
  tcgen05 rung is in its PTX and loads (the CC 10.x parts already run arch-specific
  targets); the compile gates add sm_86, sm_87 and sm_110a. CC 10.1 stays a target only:
  NVIDIA replaced sm_101 with sm_110 for Thor in CUDA 13, no shipping part reports it.
  `nvrtc_arch()` keeps its expect: every caller passes a capability validated by
  `GpuDevice::new`.
  ada: full workspace run restarted 23:10Z on the same tree (the earlier chain lost the
  script's executable bit and left the box idle for six minutes).
- 2026-09-04 01:35 — 5090 full workspace run (tree of 22:24Z): 2234 passed, 2 failed — the two
  SM120 pins fixed in ea241db and green in every gate since. ada full run restarted 23:10Z on
  the current tree. A7 gate on the 5090: arch_compile_gates 48 green (588 s) with sm_86, sm_87
  and sm_110a; tf32 contract and identity suites running. The selector harness now records the
  portable module identity per cell (`portable_identity`), and a one-cell run is chained on the
  5090 to capture it under driver 595.84 for the cohort coupling (A4).
- 2026-09-04 02:20 — A4 in gate on both boxes (lib 531 green on each). The cohort carries its
  tuning revision and a portable twin; the live SM120 cohort (595.84, 23 cells) couples its
  five portable cells to the portable identity recorded under 595.84
  (internal/perf/sm120-portable-identity-20260904); the three cohorts frozen against sources
  this tree no longer contains (12.8, 13.0, the first 13.2 on 595.91) leave the runtime table
  for a test-only retired record, and the freshness test is strict both ways. The selector
  harness refuses a cell filter naming no projection cell (a first run finished green with
  zero records). B9 instrument written: `fixed_vs_triad_pairwise_census` in
  gemm_bi_fixed_performance (five portable TF32 pairs, five SM120 TF32 pairs, tc64/tc16 and
  five SM120 half pairs per dtype, eight NN shapes) — runs after the gates. A5 patch prepared
  (scratchpad): the GEMM compile key drops the machine's include paths and the SSM state cap;
  lands with an identity refreeze captured by one-cell selector runs on both boards, not a
  re-measurement.
- 2026-09-04 03:15 — commits f91c10d (A7), 0e5ce29 (A4), 6cb6a33 (B9 instrument); both gates
  green (5090: clippy, lib 531, tf32 81, sm120 21, arch 48; ada 13.2: all green). ada 12.8:
  three tf32-contract pins move deliberately (the spill assert reports below 12.9 like the
  compile gates; two SASS-checker self-tests describe the >= 12.9 contract) — re-check chained.
  B9 first pass on the 5090 (internal/perf/fixed-vs-triad-20260904, 8 NN shapes, 200 launches
  per body, medians of fixed/triad time):
  | pair | fixed wins | triad wins | median fixed/triad |
  |---|---|---|---|
  | half SM120 tiles (5 tiles x 2 dtypes) | 55 | 25 | 0.997..1.002 (same body within noise) |
  | tf32 m128n64 s2 / s3 | 8 / 8 | 0 / 0 | 0.964 / 0.967 |
  | tf32 m16n32 s4 | 8 | 0 | 0.953 |
  | tf32 m64n64 s2 | 5 | 3 | 0.998 |
  | tf32 m64n64 s3 | 0 | 8 | 1.193 (Triad body faster by 19%) |
  | tf32 sm120 m128n64 s2 / s3 | 5 / 1 | 3 / 7 | 0.986 / 1.009 |
  | tf32 sm120 m64n128 s2 / s3 | 3 / 5 | 5 / 3 | 1.017 / 0.998 |
  | tf32 sm120 m64n64 s2 | 8 | 0 | 0.959 |
  | half tc64 / tc16 (2 tiles x 2 dtypes, second run) | 17 | 15 | 0.985..1.025 (same body within noise) |
  Reading: the SM120 half tiles and the tc64/tc16 rungs are one body compiled twice (no winner); the portable TF32
  bodies differ by 3-5% in the Fixed family's favour except m64n64 s3 where the Triad body is
  19% faster — that pair deserves a look at the two sources before any removal. The table goes
  to the owner; nothing is removed on it.
  A5 applied and in gate on the 5090 (identity refrozen from a one-cell capture under the new
  key: internal/perf/sm120-identity-argv-20260904; artifact digests unchanged, compile keys
  moved). The SM89 identity refreezes the same way after the ada chain.
- 2026-09-04 03:50 — B9 on ada (SM89, internal/perf/fixed-vs-triad-ada-20260904, portable pairs
  only; the tc64/tc16 rows waited on the pre-size fix): the Fixed portable TF32 bodies beat the
  Triad sm80 TF32 bodies on every shape — median fixed/triad 0.894 (m128n64 s2), 0.918 (s3),
  0.855 (m16n32 s4), 0.743 (m64n64 s2, down to 0.594), 0.803 (m64n64 s3). On the 5090 the same
  pairs sit within 3-5%. Reading: the two portable TF32 implementations differ, and on Ada the
  Fixed one is the better body by 8-26%; the Triad TF32 family on SM89 (1.7-2.6x from FAST)
  would gain by adopting it. Owner's table; nothing moves without the word.
  A5 gate on the 5090 fully green (lib 531, tf32 81, sm120 21, identity 2, census, matrix,
  arch 48); the SM89 identity capture runs on ada; the 5090 full workspace run is on the
  current tree. ada 12.8 re-check: one spill-record pin left (the m80n32 symbol's 8-byte
  ptxas 12.8 spill, reported below 12.9 like its sibling).
- 2026-09-04 04:05 — ada census second pass (pre-sized half scratch,
  internal/perf/fixed-vs-triad-ada-20260904b): tc64/tc16 pairs 0.980..1.025 (one body); the
  portable TF32 rows reproduce (0.894 / 0.920 / 0.855 / 0.744 / 0.803). SM89 identity refrozen
  from a one-cell capture under the new key (internal/perf/sm89-identity-argv-20260904, artifact
  digest unchanged); the first capture ran an old binary because rsync kept the source mtimes
  older than the box's last build — syncs now touch changed files (scratchpad/sync_box.sh).
  ada 13.2 gate on the refrozen tree: all green. New smoke `tf32_cohort_binds_on_this_board`
  (tests/gemm_bi_tf32_cohort_binding.rs) qualifies the automatic TF32 policy on five projection
  shapes and reports the serving module: the host tests cannot see whether a frozen identity
  matches the board's compiled module. Chained on both boxes; the A5 commit waits on it.
- 2026-09-04 04:40 — A5 defect caught by the 5090 full run (large_d_state 128/256 diverged):
  the Fixed module carries the SSM kernels, which read MAMBA_RS_STATE_CAP; dropping the define
  from every module dropped it from them. Corrected: only the GEMM (Triad) modules omit it;
  the Fixed module keeps its state cap. The refrozen TF32 identities are unaffected (they name
  the Triad modules, compiled without the define either way). Gates relaunched on both boxes;
  the 5090 full run restarts after its gate and binding smoke.
- 2026-09-04 06:10 — half TN stream-K: kernel written in sm120.cu (persistent grid, fixed-order
  slab fold, same 64x64 BK64 S3 body as the batch tile), `Sm120Schedule` on the physical route,
  two stream-K specs, workspace and grid in the launch path, census + timing tests. NVRTC
  compiles it. Consequence to carry: any edit of sm120.cu moves the TriadSm120 source digest,
  so the live SM120 TF32 cohort (595.84) is stale until one 27-cell selector requalification
  refreezes it — planned once, after the kernel wave of this phase, as B10 says.
- 2026-09-04 06:40 — owner: the 5090 rental ends around 05:30Z today; ada only until the morning
  top-up. 5090 queue for the remaining hours: full run (161/164) -> paired TN/NT vendor
  (batch shape added) -> stream-K census + tiled/streamk timing -> one-cell SM120 identity
  capture (refreeze after the sm120.cu edit) -> pull every evidence dir. ada afterwards: the
  portable work (TF32 bodies Fixed vs Triad on SM89, exact F32 long reductions, host gates
  12.8/13.2); SM120-specific tuning resumes when a 5090 is rented again.
- 2026-09-04 06:55 — SM89 (ada) baseline on the four projection shapes, current tree
  (internal/perf/sm89-hot-20260904, medians of ours / cuBLAS):
  | lane | vs cuBLAS FAST | vs cuBLAS PEDANTIC |
  |---|---|---|
  | half bf16/f16 TC (Triad) | 1.11-1.52x slower (one NT f16 cell 0.77x) | 0.23-0.46x (3-4x faster) |
  | TF32 policy | 1.68-2.58x slower | 0.79-1.29x |
  | exact F32 | 2.35-5.44x slower (NT worst) | 0.96-2.25x (NT 1.55-2.25x) |
  Reading: on Ada the half family is within 1.1-1.5x of FAST; TF32 and exact F32 are the
  distance to close, exact NT first. The Fixed TF32 bodies beat Triad's by 8-26% on this
  board (B9), which alone does not reach FAST. The 5090 full run on the committed tree:
  2242 passed, 0 failed.
- 2026-09-04 07:25 — stream-K half TN measured on the 5090 (internal/perf/sm120-streamk-tn-20260904):
  the census passes on every TN hot shape (CPU reference, bit-identical reruns), and against the
  best tiled route of each shape the stream-K body gives prod_input_proj 0.32, prod_out_proj 0.53,
  prism_out_proj 0.60, prism_input_proj 0.74, rect_tall 0.75, prism_in_proj 0.93, prod_in_proj 0.96,
  and loses 1.35-1.8x wherever the tile grid already covers the multiprocessors (large, d768,
  d128, underfill). The half table routes the twelve measured TN winners (both dtypes) to the
  stream-K body; the neighbour rule hands a stream-K neighbour to an uncovered shape only while
  its own grid underfills the device.
  Paired half TN/NT against cuBLASLt on the 5090 (internal/perf/sm120-paired-tn-nt-vendor-20260904,
  production auto route, ratio ours/vendor p50): TN large 0.96, large_deep 1.00, d768_out_proj
  1.00, batch_in_proj 1.41; NT large 1.02, large_deep 1.00, d768_out_proj 1.00, batch_in_proj 0.95
  (both dtypes within 0.01). The half family sits at vendor parity on the projection shapes; the
  TN batch_in_proj gap (144 tiles, 10400 deep) is the one left, and stream-K takes only 4% of it.
- 2026-09-04 07:40 — SM89 22 shapes x 3 ops against cuBLAS (internal/perf/sm89-hotall-20260904,
  medians of ours/vendor p50 over 66 cells per lane):
  | lane | vs FAST median [min..max] | vs PEDANTIC median | worst cells vs FAST |
  |---|---|---|---|
  | half bf16 TC | 1.41 [0.34..5.20] | 0.33 | tn/thin_cols_tail 5.2, nn/split_candidate 4.3, tn/batch_input_proj 4.1 |
  | half f16 TC | 1.45 [0.48..5.01] | 0.54 | tn/batch_input_proj 5.0, tn/thin_cols_tail 4.9, nt/thin_cols 3.8 |
  | TF32 policy | 1.99 [0.53..4.82] | 1.04 | nn/prism_input_proj 4.8, nn/prism_out_proj 4.7, nn/batch_out_proj 4.7 |
  | exact F32 | 3.10 [0.53..10.05] | 1.51 | tn/underfill 10.1, nn/rect_wide 9.8, tn/d128_in_proj 9.2 |
  Reading for the ada lane: the half TN batch shapes lose 4-5x on SM89 for the same reason the
  5090 did before stream-K (few tiles, deep reduction) — the sm80 half TN kernels have no split
  schedule; a portable stream-K twin over the tc64 body is the next ada kernel. The thin/tail
  cells are the grid-underfill family the scalar rules already cover for f32; the half tables
  have no such rule on SM89 yet.
- 2026-09-04 07:55 — stream-K half TN accepted on the 5090: promo gate green (fmt, clippy, lib
  532, auto-cell census 5/5 with the twelve stream-K cells qualifying eager == graph against the
  forced route). The stream-K route carries its own numeric contract (`TmaMma16F32StreamKV1`,
  resolved `MmaSyncF32StreamKFixedOrderV1`, ownership `OwnerCtaPerOutputTileStreamKFixedOrderV1`):
  bit-stable for a shape on a device, not bit-equal to the tiled ladder, so the census compares
  an automatic stream-K cell with the forced stream-K route and the forced route with the CPU
  reference. The 595.84 identity is refrozen from a one-cell capture after the sm120.cu edit
  (source, header manifest and artifact digests moved; the TF32 kernels are byte-identical —
  a full 27-cell requalification stays scheduled for the phase end). Contract, identity, matrix
  and compile-gate suites run now; ada runs its 13.2 and 12.8 gates on the same tree.
- 2026-09-04 08:00 — ada kernel queue (for the hours without a 5090): the portable half TN
  on SM89 runs the TC128 body only (sm80.cu has no tc64 TN), so a 384x384 TN output is nine
  CTAs on 142 multiprocessors — the 4-5x loss of tn/batch_input_proj and the thin/tail cells.
  The stream-K twin of the SM120 kernel over the sm80 TC128 TN body (persistent grid, fixed-order
  slab fold, own numeric contract) is the next kernel; it needs the portable module's TN split
  plumbing the SM120 side now has.
- 2026-09-04 08:15 — commit 4777396 (stream-K half TN on SM120). 5090 gate for it: clippy, lib
  532, sm120 contract 21, perf matrix 50, tf32 contract 81, identity 2, arch gates 48, auto-cell
  census 5/5. ada 13.2 gate on the same tree: twelve suites green; 12.8 gate: lib 532 and the
  TF32 contract 81 green, the rest running. Final full workspace run on the 5090 in flight on the
  commit. Sixteen commits on the branch since last evening, none pushed.
- 2026-09-04 08:25 — ada 12.8 gate on 4777396: all green (lib 532, tf32 81, determinism,
  invariance 17, matrix 50, census, arch gates 48). Both toolkits on ada and the 5090 gate agree
  on the stream-K tree; the 5090 final full run is the last check before the rental ends.
- 2026-09-04 09:10 — the final 5090 full run on 4777396 failed one test:
  tc64_backward_qualified_routes_match_the_forced_kernel — the neighbour rule handed the
  stream-K body to (2048, 48, 1536) and the automatic TN path stopped reproducing the forced
  portable kernel's bits, which is the half family's standing contract. Follow-up commit: the
  table returns to its measured tiled cells; the kernel, forced route, census and numbers stay.
  Next step for the schedule: a separately enabled numeric-contract set (like the TF32 split-K
  family) so a caller opts into the grid-dependent fold; only then the twelve cells return to
  the table. Lib 532, clippy clean, gemm_bi_tc 30/0 on the final tree.

- 2026-09-04 09:40 — the 5090 rental ended (connection refused since ~09:15Z); ada carries the
  lane. Half stream-K opt-in landed as a policy: `MAMBA_RS_BI_HALF_POLICY=tiled|streamk`
  (`HalfTriadPolicy::{TiledParityV1, AllowStreamKFixedOrderV1}`, `ctx.set_half_triad_policy`),
  a ninth numeric-contract bit `TRIAD_MMA_SYNC_STREAM_K_V1` (the set widened to u16, policy
  revision 5), the graph guard maps `MmaSyncF32StreamKFixedOrderV1` to it, and the env flag
  refuses to be a silent no-op without `MAMBA_RS_BI_TENSOR_CORES=1`. The twelve measured SM120
  stream-K cells return as their own table `SM120_STREAMK_CELLS_CC120`, consulted only under
  the permission; the tiled table keeps its own cell for every one of those shapes, so the
  default policy is byte-identical to before. A stream-K neighbour now declines (rather than
  flips to tiled) when the target grid fills the device, and the tiled table answers. ada 13.2
  gate on the tree: fmt, clippy, lib 536, kernel_identity 22, identity_cuda 2, sm120 contract,
  tc 30, determinism 5, tf32 81, arch gates 48 all green (one unit test corrected on the way:
  the "filled grid" probe had used a shape whose dW output was 48 rows, i.e. 24 tiles).
- 2026-09-04 09:50 — SM89 stream-K twin written over the sm80 tc64 TN dW body
  (`gemm_bi_tn_tc64_streamk_{bf16,f16}` in sm80.cu): persistent grid = min(SM count, units),
  the SM120 dealing formula, per-segment tc64 mainloop (ascending slabs, same mma chain), one
  partial slab per CTA in L2 with release/acquire flags, ascending-CTA fixed-order fold by the
  tile's last CTA, the ordinary accumulate epilogue. Host: `TcTile::Tile64StreamK` (forced API,
  qualification facade, graph replay pushes the workspace pointers), `HalfSchedule` on the
  half kernel identity so the route carries `MmaSyncF32StreamKFixedOrderV1` and
  `OwnerCtaPerOutputTileStreamKFixedOrderV1`, workspace from the split-K scratch and counter
  buffers with cap checks, and the SM89 automatic rule (dW, permitted half policy, tc64 grid
  under one wave, reduction >= 1024 = `Sm80TcPolicyV3::stream_k_min_reduction`, in the policy
  digest). Tests: forced census against an f64 reference with bit-stable repeats, contract and
  persistent-grid pins through the qualification facade, the SM89 automatic pick under both
  policies, and a 22-shape x 2-dtype timing census (`tn_tc64_streamk_versus_tiled_timing_census`,
  ignored) whose numbers set the rule before any adoption. The sm80.cu edit moves the SM89 TF32
  identity; it refreezes from a one-cell capture once the gate is green.
- 2026-09-04 11:05 — the first SM89 gate on ada: the stream-K kernel passed every new test
  (gemm_bi_tc 32/32: the seven-shape forced census against an f64 reference with bit-stable
  repeats, the qualification-facade contract pins, tails and unaligned staging included), and
  the tree failed exactly where an sm80.cu edit must: the SM120 cohort's portable twin is
  frozen against the compute_120 composition of sm80.cu, and that composition had moved. A
  foreign refreeze from ada is impossible by design (the compile key digests the NVRTC library
  set of the capturing box), retiring the SM120 cohort would blind the 5090's TF32 route by my
  own hand, and a new module kind touches ~90 sites. Resolution: the kernel moves to its own
  fragment (kernels/gemm_bi_triad/sm80_streamk.cu, fragment-local GEMM_BI_SK64_* macros) that
  the portable module composes for every sm80-family target except CC 12.x
  (`sm80_target_composes_streamk`); sm80.cu is byte-identical to HEAD again, so the compute_120
  composition and the SM120 twin are unchanged, while sm_89 gains the kernel and refreezes its
  own identity on ada. The composition is now target-aware (`compose_module_source_for`), the
  PTX validation pins the fragment's presence per target, the loader holds the kernel as an
  `Option`, and the compile gates compile the fragment for every sm80 target.
- 2026-09-04 10:30 — second ada gate (fragment tree): fmt, clippy green; lib 534/3 where the three
  are pins that move with the design (the policy identity digests for five device sizes gained
  the stream-K field and are repinned from the live values; the fragment boundary list and the
  stream-K symbol check now describe the fragment); gemm_bi_tc 32/1 where the one is the new
  automatic-route test, which had asked the eager route recorder for a native half launch. The
  native sm80 half path records only through a physical observer (qualification, capture), not
  the eager recorder that the F32, TF32 and SM120 paths feed, so the route is proven by its bits
  instead: the automatic launch equals the forced kernel of the schedule the policy names.
  The first census run answered nothing past the first shape: every qualification captures a
  graph and the context scratch cannot grow after a capture, so the suite is now pre-sized
  (`presize_physical_qualification_suite`) before the first cell.
- 2026-09-04 10:55 — SM89 tc64 TN dW census on ada (internal/perf/sm89-streamk-tn-20260904,
  graph replays, min of 9 windows x 20, forced Tile64 / Tile128 / Tile64StreamK, both dtypes
  within 0.01). Ratio of stream-K to the best tiled route, then best tiled and stream-K against
  cuBLAS (bf16, from sm89-hotall-20260904):
  | shape (m x k x n) | tiles64 | slabs | sk / best tiled | tiled / cuBLAS -> sk / cuBLAS |
  |---|---|---|---|---|
  | batch_input_proj 10400x384x384 | 36 | 163 | 0.34 | 4.04 -> 1.39 |
  | batch_out_proj 10400x768x384 | 72 | 163 | 0.56 | 2.07 -> 1.17 |
  | prism_out_proj 4621x768x384 | 72 | 73 | 0.64 | 2.44 -> 1.57 |
  | prism_input_proj 4621x1024x384 | 96 | 73 | 0.79 | 1.50 -> 1.18 |
  | rect_tall 4096x512x768 | 96 | 64 | 0.80 | 1.47 -> 1.18 |
  | batch_in_proj 10400x384x1536 | 144 | 163 | 0.86 | 1.61 -> 1.38 |
  | d768_out_proj 2048x1536x768 | 288 | 32 | 1.00 | 1.32 -> 1.32 |
  | prism_in_proj 4621x384x1928 | 186 | 73 | 1.12 | 1.46 -> 1.64 |
  | large / large_deep / rect_wide / d768_in_proj | 576-1152 | 8-64 | 1.22-1.36 | worse |
  | d128_in_proj / d128_out_proj | 16 / 8 | 16 | 1.22 / 1.34 | worse |
  | underfill / thin_cols / split_candidate | 48 / 32 / 4 | 4 / 8 / 128 | 1.49-1.65 | worse |
  | thin_rows (one slab), sq64, tails | - | 1-3 | 1.04-1.60 | worse |
  Reading: the persistent grid wins where the tile grid is at most about one wave (up to 144
  tiles on 142 multiprocessors) and every CTA holds a deep reduction (37 slabs and more); it
  loses wherever a CTA holds a handful of slabs (the fold of one slab per contributor is paid
  per tile) or the tiles exceed the wave by a third or more (segment restarts and folds on a
  grid the tiled kernel already fills). The rule is therefore two policy fields on
  Sm80TcPolicyV3: tiles64 x 8 <= multiprocessors x 9, and tiles64 x slabs >= 32 x
  multiprocessors; the six winners and eight losers above are pinned in the pick test.
  The SM89 TF32 identity refroze from a one-cell capture on the fragment tree
  (internal/perf/sm89-identity-streamk-20260904; compile key, artifact, source and header
  manifest moved, the TF32 kernels are byte-identical since sm80.cu is HEAD).
- 2026-09-04 11:50 — commits a90bad1 (half stream-K opt-in policy) and 2f8b02a (SM89 stream-K
  twin, census rule, refrozen SM89 identity). Gates on the committed tree (ada 13.2): fmt,
  clippy, lib 537/0, gemm_bi_tc 33/0, tf32 contract 81/0, kernel_identity 22/0, determinism 5/0,
  arch gates 48/0, performance matrix 50/0, sm120 contract 22/0, cohort binding 1/1. Known red:
  repeated_nvrtc_compiles_have_the_same_identity (kernel_identity_cuda) fails three times in a
  row on ada with two PTX images of the scalar module under one compile key (46cb0da6 in the
  cache payload, 8137b9cd in the first context's identity); a probe that builds three contexts
  in the same binary against a trusted temp cache gets 46cb0da6 every time with matching
  payloads, and four NVRTC compiles of the scalar blob in one process are byte-identical. The
  scalar module is untouched by both commits; the failure began after the SM89 identity refreeze
  and is parked as its own task, by the owner's word, so the SM89 optimisation lane continues.
- 2026-09-04 12:15 — F32 on SM89, the real lever: under the TF32 policy the hot batch and prism
  cells of the census still ran the scalar kernels (nn prism_input_proj 4.82x, batch_out_proj
  4.67x, prism_out_proj 4.66x, batch_input_proj 4.22x; tn rect_tall 4.11x, prism_out_proj 4.08x,
  batch_out_proj 4.05x; nt prism/batch 3.75-3.88x against cuBLAS) because those shapes were
  never qualified for SM89: the 43-cell table holds d768, prism_in_proj, large, large_deep, d128,
  split and batch_in_proj (sm89-requal-projection/deepk-20260903) and nothing else. Eighteen
  cells (prism_out_proj, prism_input_proj, batch_input_proj, batch_out_proj, rect_tall,
  underfill x NN/TN/NT) join the selector suite (PROJECTION_CELLS 27 -> 45) and qualify on ada
  (internal/perf/sm89-requal-hot-20260904); the winners become table rows through
  scratchpad sm89_rows_from_selector.py, the inventory pins (43 cells, gate counts
  [5, 3, 24, 11]) move with them.
- 2026-09-04 12:45 — SM89 hot requalification done (internal/perf/sm89-requal-hot-20260904, 18
  cells, 1541 s): fifteen TF32 winners join the table (NN x6 m64n64 s2 for every hot NN cell;
  NT x6 m64n64 s2 / m128n64 s3; TN rect_tall, prism_input_proj, underfill), three TN cells kept
  the scalar winner (prism_out_proj 768x384 over 4621 rows, batch_input_proj 384x384 and
  batch_out_proj 768x384 over 10400 rows): the portable TN TF32 family has no split schedule,
  so a 36-72 tile grid over a ten-thousand-row reduction cannot beat the scalar split-M kernel
  there. The table grows 43 -> 58, gate counts [5, 3, 24, 11] -> [5, 3, 33, 17]; the gate and
  a TF32-only census (22 shapes x 3 ops, both orders, cuBLAS denominators) run on ada.
- 2026-09-04 12:55 — TF32 lane on ada after the fifteen rows (internal/perf/sm89-hot-tf32-20260904
  against sm89-hotall-20260904, p50 ours/cuBLAS f32, both orders, 101 windows):
  | cell | before | after |
  |---|---|---|
  | nn batch_input_proj | 4.22 (scalar) | 2.18 (m64n64 s2) |
  | nn batch_out_proj | 4.67 (scalar) | 2.30 (m64n64 s2) |
  | nn prism_input_proj | 4.82 (scalar) | 2.39 (m64n64 s2) |
  | nn prism_out_proj | 4.66 (scalar) | 2.30 (m64n64 s2) |
  | nn underfill | 1.24 (scalar split-K) | 1.05 (tf32 splitk2 m16n32 s4) |
  | nt batch_input_proj | 3.75 (scalar) | 2.31 (m64n64 s2) |
  | nt batch_out_proj | 3.79 (scalar) | 2.05 (m128n64 s3) |
  | nt prism_input_proj | 3.87 (scalar) | 1.86 (m64n64 s2) |
  | nt prism_out_proj | 3.88 (scalar) | 2.25 (m64n64 s2) |
  | nt underfill | 1.83 (transpose + scalar) | 1.27 (m16n32 s4) |
  | tn prism_input_proj | 3.63 (scalar split-M) | 3.49 (m64n64 s3) |
  | tn rect_tall | 4.11 (scalar split-M) | 3.37 (m64n64 s3) |
  | tn prism_out_proj / batch_input_proj / batch_out_proj | 4.05-4.08 (scalar) | unchanged, scalar won |
  Every other cell is unchanged (same kernel, within 0.01). Median over 66 cells 1.98 -> 1.94:
  the lane is now limited by the TF32 kernel bodies themselves (1.7-2.6x on the covered
  cells), which is where B9 says the Fixed bodies are 8-26% faster, and by the TN family's
  missing split schedule (3.4-4.1x on the tall reductions).
- 2026-09-04 14:20 — the wide deterministic TF32 tile. Both portable TF32 bodies (Triad and
  Fixed) compute with four of their eight warps on the 128 x 64 tile, so the whole family sits
  at 32-46 TF on ada. A 128 x 128 NN tile with eight computing warps (2 x 4 warp tiles of
  64 x 32, BK 32, three cp.async stages, XOR-swizzled unpadded 32 KB stages, 123 registers,
  no spills), probed standalone on ada against the best existing tile: 10400x768x384 0.84,
  4621x384x1928 0.87, 2048x3072x768 0.72, 4096x3072x1536 0.73 (63 TF), 10400x384x384 0.97,
  2048x768x3072 0.79 - bit-identical output on every shape (same ascending k8 chain). The
  tile lands as `kernels/gemm_bi_triad/sm80_tf32_wide.cu`, a second extension fragment on the
  same non-CC-12 targets as the stream-K twin; `Tf32PortableTile::M128N128`,
  `SM80_TF32_WIDE_ROUTE_SPECS`, and `tf32_route_specs_for(module, extensions)` keep the base
  18-spec inventory for the CC 12.x composition while the loader, the PTX validators, the
  selector and the identity fixtures take the extension on the targets that compose it.
  Ceiling reading: 63 TF at 1800 MHz is the TF32 tensor peak of this board at this clock;
  cuBLAS FAST F32 reaches 109 TF on the same shape, above the TF32 peak, i.e. through the
  16-bit tensor paths. Deterministic TF32 cannot meet that lane; an f16-input F32 policy could,
  and that is a numerics decision for the owner.
- 2026-09-04 14:40 — the identity flake, root cause. `repeated_nvrtc_compiles_have_the_same_identity`
  failed 5 of 11 runs on ada (13.2): a cache trace (`MAMBA_RS_CACHE_TRACE=1`, PTX dumps under
  `MAMBA_RS_CACHE_TRACE_DIR`) caught both scalar-module images under the one compile key
  6fe8d440: 46cb0da6 and 8137b9cd, 2341064 vs 2341118 bytes, every one of the 1174 differing
  lines inside `gemm_bi_nt_slim` - a `.reg .f32 %f<738>` vs `%f<736>` declaration, the
  fragment loads merged as v4+v4 in one image and v2+v4 in the other, and the fma/mov/st chain
  renumbered behind them. NVRTC's load vectoriser decides differently from compile to compile
  for the same source; eight separate processes compiling the test blob agree, so the flip is
  a property of the composed module in a live process. The source left the decision to the
  compiler: every scalar kernel loaded its register fragments element by element
  (`regM[... + i] = As[... + i]`, 43 sites across the main, slim, narrow, split-K and macro'd
  families). They now load through `gemm_bi_scalar_load_fragment<Count>` as explicit
  float4/float2 vectors (every base is a multiple of the width: four-float pads, fragment
  widths of 8, 4 or 2), which pins the instruction selection at the source and leaves the
  FMA chain untouched. A source gate in arch_compile_gates refuses the element-wise form.
  Four traced identity runs and the wide-tile gate run on the fixed tree.
- 2026-09-04 14:50 — TN split-K lane opened. The three scalar TN cells (prism_out_proj 768x384
  over 4621 rows, batch_input_proj 384x384 and batch_out_proj 768x384 over 10400 rows, all at
  4.05-4.08x cuBLAS) had no TF32 answer because the portable split-K family was NN/NT only: the
  fused kernel's static assertion, the fast-path stage loader (row-major A only) and the fixup
  epilogue (beta folded for NN only). The family now carries four TN candidates, all eight
  partitions: m64n64 s2 (36864 B, two CTAs per SM on the 100 KB Ada carveout), m64n64 s3
  (55296 B, one), m32n32 s3 (30720 B, three) and m32n32 s4 (40960 B, two); the TN storage is
  k-major for A (32 rows of BM+8), so the m32n32 stage costs 10240 B against the NT 9216. The
  loader gains the k-major A branch of the generic stage without its bounds (the fast path runs
  on full tiles only), the fixup folds beta for TN as NN does (bias stays NN-only), and the
  contract table, the loaders' Driver-ABI census, the selector harness and the identity pins
  follow the spec table. Grid for 768x384 at m64n64: 72 tiles x 8 = 576 CTAs; at m32n32: 288 x 8.
  Scratch: 768*384*8 = 2.36M floats under the 8M cap. Gate on ada, then the three cells requal.
- 2026-09-04 15:00 — the TN split-K family moved out of sm80.cu into its own extension fragment
  (`sm80_tn_splitk.cu`, composed with the stream-K and wide-tile fragments on sm80-family
  targets only). The first cut edited the shared fused kernel, which moved the CC 12.x portable
  twin's source digest and unbound every SM120 TF32 cohort (`at_least_one_sm120_tf32_cohort_
  matches_this_tree` went red) - the same trap the wide tile hit yesterday. The fragment carries
  TN copies of the split-K stage (k-major A fast path), the pipeline and the fused kernel (beta
  folded, no bias), and the contract splits the spec table into the portable six and the
  extension four (`tf32_splitk_specs_for(extensions)`), which the Driver-ABI census, the PTX
  validator, the loader and the harness now consult by target. sm80.cu is byte-identical.
- 2026-09-04 15:15 — identity fix, second cut. Explicit float4/float2 loads pinned the
  vectorisation but moved the flip into `gemm_bi_tn_aligned` and the TN split-M kernels: with
  register numbers normalised, two compiles of one process differ by 2-6 lines, always an
  extra `ld.shared.v2.f32` at the second half of a fragment carrying `.pragma "used_bytes_mask
  255"` - the optimiser re-loading part of a fragment instead of keeping it, a register
  allocation decision it makes differently per compile (and per process, five digests over three
  runs). The fragment loads are now fixed PTX `ld.shared.v4.f32` / `v2` through inline asm
  (opaque values cannot be split or re-loaded); nvcc on the same blob: identical PTX twice, 55
  entries, zero spills, `gemm_bi_tn_aligned` and both TN split-M kernels at zero stack. NVRTC
  in-process determinism is measured by the four traced identity runs of chain2 on ada.
- 2026-09-04 16:02 — identity fixed. Four traced runs of `repeated_nvrtc_compiles_have_the_same_
  identity` on the asm-loader tree: 4/4 ok, one scalar image per run and the same image across
  runs (TriadScalar-7229de42e356), against 5-of-11 failures before and five digests over three
  runs with the float4 cut. The wide gate and both requalifications follow on the same tree.
- 2026-09-04 16:45 — SM89 TN requalification on the fragment tree (internal/perf/sm89-requal-tn-
  20260904, 6 cells, 11 windows, p50 graph us; cuBLAS = f32 fast of sm89-hotall-20260904):
  | cell | m,k,n | scalar | winner | winner us | x scalar | scalar/cuBLAS | winner/cuBLAS |
  | tn prism_out_proj | 4621,768,384 | 154.1 | splitk8 m64n64 s3 | 106.4 | 1.45 | 4.09 | 2.82 |
  | tn prism_input_proj | 4621,1024,384 | 173.5 | splitk8 m64n64 s3 | 132.6 | 1.31 | 3.65 | 2.79 |
  | tn rect_tall | 4096,512,768 | 178.8 | splitk8 m64n64 s3 | 123.2 | 1.45 | 4.10 | 2.82 |
  | tn underfill | 256,512,384 | 75.6 | m16n32 s4 (unchanged) | 10.4 | 7.24 | 9.63 | 1.33 |
  | tn batch_input_proj | 10400,384,384 | 162.5 | scalar kept | - | - | 3.70 | 3.70 |
  | tn batch_out_proj | 10400,768,384 | 279.6 | scalar kept | - | - | 4.11 | 4.11 |
  The two 10400-row cells did not time the split family: all four TN split candidates failed the
  numeric gate against the tiled portable reference by 0.6-6% on single elements (identical
  values across the four configurations, so a systematic path, not a race); the 4621- and
  4096-row cells passed the same gate. An element-level diagnostic (candidate, tiled reference
  and exact scalar, with row/column ranges) runs after the NN requalification.
- 2026-09-04 17:10 — SM89 NN requalification with the wide tile (internal/perf/sm89-requal-wide-
  20260904, 15 cells, 11 windows, p50 graph us; cuBLAS = f32 fast of sm89-hotall-20260904):
  | cell | m,k,n | scalar | winner | winner us | x scalar | scalar/cuBLAS | winner/cuBLAS |
  | nn d768_in_proj | 2048,768,3072 | 351.0 | m128n128 s3 | 184.8 | 1.90 | 3.45 | 1.81 |
  | nn d768_out_proj | 2048,1536,768 | 253.5 | m128n128 s3 | 109.2 | 2.32 | 4.03 | 1.74 |
  | nn prism_in_proj | 4621,384,1928 | 295.5 | m128n128 s3 | 150.4 | 1.97 | 2.87 | 1.46 |
  | nn large_deep | 4096,3072,1536 | 1387.3 | m128n128 s3 | 654.4 | 2.12 | 3.90 | 1.84 |
  | nn large | 2048,3072,768 | 497.4 | m128n128 s3 | 204.0 | 2.44 | 4.17 | 1.71 |
  | nn batch_in_proj | 10400,384,1536 | 483.4 | m128n128 s3 | 290.9 | 1.66 | 3.32 | 2.00 |
  | nn prism_out_proj | 4621,768,384 | 144.5 | m128n128 s3 | 64.4 | 2.25 | 4.12 | 1.83 |
  | nn prism_input_proj | 4621,1024,384 | 189.0 | m128n128 s3 | 80.0 | 2.36 | 4.24 | 1.79 |
  | nn batch_input_proj | 10400,384,384 | 145.3 | m128n128 s3 | 81.3 | 1.79 | 3.70 | 2.07 |
  | nn batch_out_proj | 10400,768,384 | 274.2 | m128n128 s3 | 128.4 | 2.14 | 4.11 | 1.92 |
  | nn d128_in_proj | 1024,128,512 | 15.1 | m64n64 s2 | 9.9 | 1.52 | 2.65 | 1.75 |
  | nn rect_tall | 4096,512,768 | 153.2 | m64n64 s2 | 86.6 | 1.77 | 3.26 | 1.84 |
  | nn d128_out_proj | 1024,256,128 | 9.6 | m16n32 s4 | 7.5 | 1.27 | 1.39 | 1.09 |
  | nn underfill | 256,512,384 | 12.0 | splitk2 m16n32 s4 | 10.7 | 1.12 | 1.19 | 1.07 |
  | nn split_candidate | 128,8192,128 | 24.6 | scalar kept | - | - | 1.93 | 1.93 |
  The wide tile takes every large NN cell at 1.9-2.4x over scalar; the remaining 1.5-2.0x to
  cuBLAS is the TF32-against-16-bit-tensor gap (f16-input policy = owner decision). The 24
  measured rows (10 wide, 3 TN split-K, the rest unchanged winners) replace or join the SM89
  table through the row applier keyed by cell comment.
- 2026-09-04 17:35 — the 10400-row TN gate failures were the reference, not the candidate.
  `tf32_candidate_mismatch_map` (new ignored diagnostic in the selector harness) on the split
  m64n64 s3 candidate, the tiled portable reference (TN m128n64 s2) and the exact scalar:
  | cell | elements | candidate/reference | candidate/exact | reference/exact | max rel err cand/exact |
  | tn batch_input_proj | 147456 | 75 | 0 | 72 | 0.00065 |
  | tn batch_out_proj | 294912 | 155 | 0 | 171 | 0.00078 |
  | tn prism_out_proj | 294912 | 0 | 0 | 0 | 0.00026 |
  The split family sums eight partials and sits within a tenth of the tolerance of the exact
  output; the tiled TF32 kernel accumulating 325 tiles in one chain drifts past the 0.25%
  tolerance on ~0.05% of the elements, and the gate compared the candidate to it. The harness
  now holds the split and stream families to the exact scalar output (the reference family
  keeps its bitwise check against the forced portable bits); tolerance unchanged. The two cells
  requalify (sm89-requal-tn2-20260904).
- 2026-09-04 17:37 — with the gate on the exact reference, the two long-reduction TN cells take
  the split (sm89-requal-tn2-20260904): batch_input_proj 10400x384x384 162.5 -> 122.3 us
  (splitk8 m64n64 s3, 1.33x scalar, 3.70 -> 2.79x cuBLAS), batch_out_proj 10400x768x384 279.6 ->
  213.0 us (1.31x, 4.11 -> 3.13x). All six TN cells of the hot set now run TF32; the identity is
  refrozen from the wide-tile record (compile key d150502dab94, artifact 61567d8c44a7). Final
  gate on the assembled tree, then the commits.
- 2026-09-04 18:22 — committed on codex/gemm-bi-triad-sm80: dce79ac (scalar fragments as fixed
  PTX vector loads, the identity test 4/4) and 9bac78e (TF32 128x128 tile, TN split-K fragment,
  extension-aware tables/census/validators/loaders, harness gate on the exact reference, 16
  measured SM89 rows, identity refrozen). Final gate on the assembled tree: fmt, clippy, lib
  538, arch 50, identity_cuda 2, contract 82, cohort 1, selector 27, tc 33, identity 22 - all
  green. Next lane: the TN cells sit at 2.8-3.1x cuBLAS on a 4-warp m64n64 split; a k-major
  TN 128x128 body (eight computing warps) under the split family is the direct analogue of
  what took NN from 3-4x to 1.5-2x.
- 2026-09-04 20:16 — wide-tile efficiency lane, measured on nn d768_out_proj / large_deep /
  batch_out_proj (us; cuBLAS FAST 63.0 / 355.5 / 66.8). ncu on the 8-warp 128x128x32 tile at
  2.4 GHz: tensor pipe 35%, 70% of scheduler cycles without an eligible warp (one CTA of eight
  warps per SM), stalls led by math_pipe_throttle, lg_throttle and wait. cuBLAS on the same
  shape: `cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4`, 8 warps of 64x64, 224
  registers, 74 KB smem, 67% tensor, 52 us against our 89 us under the profiler; the
  executed-instruction mix is 4.7 per HMMA against our 10.5 (ours: the cvt.rna.tf32 emulation
  ~6 SASS each, 24 per 16 mma, plus swizzle address math). Variants:
  | variant | d768_out | large_deep | batch_out | verdict |
  | 8 warps 64x32, loads then mma (committed) | 109.2 | 654.4 | 128.4 | baseline |
  | + ldmatrix A, fragment double-buffer, cp.async spread | 93.9 | 573.4 | 113.5 | kept (-13%) |
  | 4 warps 64x64 (128 threads, 238 regs) | 103.0 | 603.2 | 122.3 | slower: one warp per scheduler |
  | 128x256x16, 8 warps 64x64 (218 regs) | (not top 3) | 703.5 | (not top 3) | slower: tile waste on n=384/768, wave tail |
  | TF32 rounding once per stage in shared (+1 barrier) | 109.7 | 645.1 | 128.8 | slower: the pass and barrier cost more than the cvt saved |
  | + fragment offsets precomputed once per thread (172 regs) | 91.7 | 551.3 | 110.8 | kept (-2..-4%) |
  | + rna as add-half-ulp in the integer domain (164 regs, bit-identical on the gate) | 100.2 | 596.5 | 119.1 | slower than the cvt form (-9%); reverted |
  | + stage boundary (wait, barrier, next stage's first fragments) before the last mma group (173 regs) | 102.1 | 609.2 | 120.6 | slower (-10%): the barrier idles the tensor pipe when a warp reaches it with a quarter of its mma unissued; reverted |
  The instruction count is not the limiter after the pipelining: fewer instructions on the
  cvt path made the kernel slower, so the remaining gap to cuBLAS sits in the warp-tile
  geometry (32 mma per fragment set on 64x64 warps against our 16) and in synchronization.
  A source-level stall profile (ncu, per SASS line) of the kept body decides the next move. A 256x128x16 mirror of the
  128x256 tile joins the route table for the outputs whose column count wastes a 256-wide
  tile; the full NN requalification runs with all three wide routes competing.
- 2026-09-04 21:25 — full SM89 NN requalification on the pipelined 128x128 tile with the 128x256
  and 256x128 candidates competing (internal/perf/sm89-wide3-20260904, 7 windows): neither
  wide-warp tile won a cell (dropped from the tree again), the pipelined 128x128 takes eleven of
  fifteen cells; against cuBLAS FAST (TF32), morning -> now:
  | cell | m,k,n | morning | now | us now / cuBLAS |
  | nn d768_in_proj | 2048,768,3072 | 1.81 | 1.56 | 158.8 / 101.9 |
  | nn d768_out_proj | 2048,1536,768 | 1.74 | 1.46 | 91.8 / 63.0 |
  | nn prism_in_proj | 4621,384,1928 | 1.46 | 1.29 | 133.1 / 103.0 |
  | nn large_deep | 4096,3072,1536 | 1.84 | 1.55 | 551.6 / 355.5 |
  | nn large | 2048,3072,768 | 1.71 | 1.41 | 168.4 / 119.4 |
  | nn batch_in_proj | 10400,384,1536 | 2.00 | 1.81 | 263.8 / 145.8 |
  | nn prism_out_proj | 4621,768,384 | 1.83 | 1.59 | 55.7 / 35.1 |
  | nn prism_input_proj | 4621,1024,384 | 1.79 | 1.54 | 68.5 / 44.6 |
  | nn batch_input_proj | 10400,384,384 | 2.07 | 1.87 | 73.5 / 39.3 |
  | nn batch_out_proj | 10400,768,384 | 1.92 | 1.67 | 111.3 / 66.8 |
  | nn rect_tall | 4096,512,768 | 1.84 (m64n64) | 1.66 (m128n128) | 78.0 / 47.1 |
  | nn d128_in_proj / d128_out_proj / underfill / split_candidate | | 1.75 / 1.09 / 1.07 / 1.93 | unchanged | |
  Rows replaced from the record (14); gate on the assembled tree, identity refrozen after it.
- 2026-09-05 22:47 — per-SASS stall profile of the kept body: 21.5% of the samples on the
  epilogue's predicated scalar STG (64 per thread), 30% on HMMA (the pipe), ~18% on address
  and bound arithmetic. Two changes measured separately:
  | variant | d768_out | large_deep | batch_out | verdict |
  | epilogue through shared (row stride 136, float2 fragment writes, float4 row segments) | 84.7 | 540.9 | 89.6 | kept: -8% / -2% / -19% |
  | interior fast path for the stage copies (no bound clamps) | 102.4 | 603.1 | 121.8 | slower (+11%): the uniform branch per copy costs more than the clamps; dropped |
  Against cuBLAS FAST now 1.35 / 1.52 / 1.34. Full NN requalification with the staged epilogue.
- 2026-09-05 23:18 — full SM89 NN requalification with the staged epilogue (internal/perf/
  sm89-wide4-20260904, 7 windows), us / cuBLAS FAST (TF32):
  | cell | m,k,n | ours | cuBLAS | ratio | morning |
  | nn prism_in_proj | 4621,384,1928 | 114.6 | 103.0 | 1.11 | 1.46 |
  | nn batch_input_proj | 10400,384,384 | 50.2 | 39.3 | 1.28 | 2.07 |
  | nn prism_out_proj | 4621,768,384 | 45.6 | 35.1 | 1.30 | 1.83 |
  | nn rect_tall | 4096,512,768 | 61.8 | 47.1 | 1.31 | 1.84 |
  | nn prism_input_proj | 4621,1024,384 | 58.8 | 44.6 | 1.32 | 1.79 |
  | nn batch_out_proj | 10400,768,384 | 89.6 | 66.8 | 1.34 | 1.92 |
  | nn d768_out_proj | 2048,1536,768 | 85.1 | 63.0 | 1.35 | 1.74 |
  | nn large | 2048,3072,768 | 163.7 | 119.4 | 1.37 | 1.71 |
  | nn d768_in_proj | 2048,768,3072 | 139.3 | 101.9 | 1.37 | 1.81 |
  | nn batch_in_proj | 10400,384,1536 | 215.7 | 145.8 | 1.48 | 2.00 |
  | nn large_deep | 4096,3072,1536 | 534.1 | 355.5 | 1.50 | 1.84 |
  | nn d128_in_proj / d128_out_proj / underfill / split_candidate | | | | 1.75 / 1.09 / 1.07 / 1.93 | unchanged |
  Winners unchanged (the table keeps its rows); identity refrozen from this record.
- 2026-09-05 00:14 — 128x256x16 on the pipelined body (four stages, offsets, two-half staged
  epilogue, 218 regs) against the kept 128x128 (us): d768_in_proj (n = 3072, no tile waste)
  184.3 vs 135.9, large_deep 692.6 vs 542.0, batch_in_proj 238.6 vs 217.7; never wins. The
  64x64-warp body is slower than the 64x32 one even where cuBLAS's own choice is that shape,
  so a defect specific to our body, not the geometry, is left; profiled next (bank conflicts,
  stall mix, opcode counts).
- 2026-09-05 00:17 — profile of the 128x256x16 body on d768_in_proj (ncu, 2.35 GHz): 143.9 us,
  tensor 39.4%, issue slots 20.6%, 231 registers, no bank conflicts (4.6K of 5.0M shared
  wavefronts), stalls math_pipe_throttle 44K > wait 21K > selected 20K > barrier 12K. From the
  HMMA count (4.72M) and the duration the tensor pipe rate on this board is 256 TF32 FMA per
  clock per multiprocessor, one HMMA.1688 every 16 pipe cycles per sub-partition; we issue one
  per 41. Instructions per HMMA 8.4 (cuBLAS 4.7): IMAD 2.6, the cvt emulation 3.1 (FSETP,
  LOP3, SEL, IADD3, SHF, IMNMX), LDS 0.6, LDSM 0.125 (the same ldmatrix-A / LDS-B pattern as
  cuBLAS's kernel). The 128x256 route is reverted; the tree is the committed 72e0420 state.
  Decision point recorded for the owner: the remaining 1.1-1.5x on NN needs CUTLASS's exact
  scheduling of the same inner loop (their conversion is two instructions, their address work
  half of ours); the TN/NT/half lanes hold 2-5x gaps.
- 2026-09-05 10:55 — literature pass (CUTLASS multistage order: raw fragments double-buffered,
  the rounding one step after the load, right before the mma; the conversion as
  `isfinite ? bits + 0x1000 : bits`; the sm_120 ablation ladder where 64x64 warps were the
  decisive step) applied and measured (us, d768_out / large_deep / batch_out; kept body 84.7 /
  540.9 / 89.6):
  | four 64x64 warps, decoupled rounding, add-half-ulp (244 regs) | 96.2 / 586.5 / 100.8 |
  | same with immediate-offset B addressing (232 regs) | 96.1 / 585.9 / 100.7 |
  | eight 64x32 warps, decoupled rounding, add-half-ulp (164 regs) | 86.4 / 541.8 / 91.1 (noise) |
  Neither the warp geometry nor the rounding form moves the eight-warp body; the tree is back
  at 72e0420. Next: the SASS of cuBLAS's kernel itself (cuobjdump on libcublasLt) against ours.
- 2026-09-05 11:30 — SASS of both kernels compared (ours from the cubin, cuBLAS's from the
  profiler's source page). cuBLAS's loop: conversion as `FSETP` + predicated `IMAD.IADD` (a
  register holding 0x1000), interleaved one-for-one between the HMMAs; fragment LDS/LDSM as
  `[R + UR + imm]` (a per-thread base, a uniform-register stage offset, an immediate: no
  address math per load); LDGSTS predicated with precomputed predicates, one per HMMA slot;
  the barrier in the middle of the HMMA stream. Ours (578 instructions per stage of 64 HMMA):
  178 IMAD, 96 FSETP, 42 SEL, 32 LDS, 16 LDSM, 31 LEA, 31 IADD3, 28 ISETP, 16 IMNMX — the cvt
  emulation (~290) and the per-copy bound math (~120) dominate. Changes measured (us,
  d768_out / large_deep / batch_out; committed 84.7 / 540.9 / 89.6):
  | copy plan: pointers, destinations and row/column bounds once per thread; a stage = pointer advance + k clamp (158 regs) | 81.1 / 510.5 / 85.5 | kept (-4..-6%) |
  | + decoupled add-half-ulp rounding (152 regs) | 82.0 / 515.9 / 86.3 | noise; the cvt form stays |
  | + precomputed full/tail copy lengths, one uniform select per copy (155 regs) | 82.8 / 515.9 / 87.0 | noise; the k-clamp plan stays |
  Against cuBLAS FAST now 1.29 / 1.44 / 1.28. The copy-path instruction count is no longer the
  limiter either; a fresh per-instruction stall profile of the plan body decides the next move.
- 2026-09-05 13:00 — per-line stall profile of the plan body (large_deep, ncu 2.08 GHz): tensor
  63%, 2.00 warps per scheduler, 0.43 eligible; cycles per issued instruction 5.68 = math pipe
  throttle 2.05, wait 1.13, selected 1.00, short scoreboard 0.39, barrier 0.36, not selected
  0.23, mio throttle 0.20, long scoreboard 0.17. Hot loop 507 SASS per 64 HMMA (7.9 per HMMA):
  the copy path re-derived every cp.async's address and length from the thread id each stage
  (S2R and a 14-instruction chain per copy, the runtime src-size form of cp.async lowering to a
  pointer adjustment and a compare). Grid facts: large_deep is 2.7 waves of 128x128 tiles (the
  third wave idles 42 of 142 SMs), d768_out_proj is 96 tiles on 142 SMs. cuBLAS's own kernels on
  these cells (ncu): d768_out_proj and large_deep run `s1688gemm_128x128_16x5` — 128 threads,
  four warps of 64x64, BK 16, five stages, 224 registers, the same 96 / 384 tiles (its 128 / 512
  CTA grids are the CUTLASS swizzle round-up); batch_out_proj runs `128x256_16x3`. So cuBLAS's
  in-loop tensor utilization is about 86% against our 67% at equal wave quantization.
  Copy-path rewrites measured (us, d768_out / large_deep / batch_out; plan body 81.1 / 510.5 / 85.5):
  | cp.async ignore-src predicate form, whole/tail stage split (171 regs) | 102.0 / 603.3 / 106.2 | slower: the copy block became a branch with the same address chain inside |
  | clamped in-range sources, one destination per operand with slice immediates, shuffle-opaque destinations (172 regs; 7 SASS per copy slice) | 84.4 / 519.7 / 88.9 | slower |
  | + interleaved column atoms, B as 16-byte loads (160 regs) | 86.0 / 530.7 / 90.3 | slower |
  None kept. Separate runs on this power-capped board drift 3-5%, so from here every variant
  is a second route symbol measured in the SAME selector run as the kept body.
- 2026-09-05 13:00 — four-warp twin `gemm_bi_nn_sm80_mma_tf32_v1_m128n128w4_bk32_s3` (64x64 warps,
  128 threads, clean copy path, row-padded B stage with immediate-offset loads, the stage
  boundary before the last k8 step's mma, fragments loaded and rounded per atom group between
  the mma groups; 250 registers, no spills; 6.5-7.3 SASS per HMMA) measured in the same run as
  the eight-warp body: speedup over the exact scalar 2.63 / 2.45 / 2.74 against 3.04 / 2.86 /
  3.13, i.e. about 15% slower on all three cells. The first build issued six of the eight copy
  slices of a stage and failed the repeat-bits gate; fixed. Sixteen-byte B loads are not usable
  for TF32 mma B pairs (b0 and b1 come from k rows t and t+4, so a vector load never yields an
  adjacent register pair; ptxas inserted a move per HMMA). Profiling the four-warp body next:
  with one warp per scheduler the stall reasons name the exposed dependency directly.
- 2026-09-05 13:40 — the four-warp body's profile (large_deep, ncu): tensor 54%, cycles per
  issued instruction 4.65 = wait 1.70 (fixed-latency dependency), math pipe throttle 1.22,
  selected 1.00, short scoreboard 0.19, long scoreboard 0.15, barrier 0.14; cuBLAS's kernel on
  the same profile set: math pipe throttle 4.80, wait 0.85, barrier 0.48 (its warps queue on
  the tensor pipe, ours wait on dependency chains). The rounding pair (FSETP, predicated
  IMAD.IADD) is the dominant chain. Measured on sm_89 (probe kernel, mma.m16n8k8.tf32 with one
  operand set to a bit pattern): the tensor core reads only the upper 19 bits of an operand
  (0x3f800fff multiplies as 1.0; 0x7f800001 multiplies as +inf even after cvt.rna), so the
  unguarded `bits + 0x1000` reaches the mma bit-identical to cvt.rna.tf32.f32 for every finite
  value and both infinities; the only divergence is a NaN whose payload sits in bits 12-13,
  which the add keeps a NaN where cvt.rna followed by the truncation yields +inf. Contract:
  `ResolvedOperandConversion::RegisterAddHalfUlpTf32V1` (= 4); the validator accepts it for
  the portable module's wide routes, requires the mma token and rejects cvt.rna in such an
  entry. Same-run comparison (speedup over the exact scalar, d768_out / large_deep /
  batch_out): four-warp cvt 2.63 / 2.45 / 2.74 -> four-warp add 2.98 / 2.80 / 3.09 (+13%),
  eight-warp cvt 3.04 / 2.86 / 3.13; the add drops the four-warp loop to 733 static SASS per
  128 HMMA (no FSETP, no LOP3 mask). Next: the same rounding in the eight-warp body, both
  measured in one run; then the deeper five-stage BK 16 ring for the four-warp body.
- 2026-09-05 13:50 — the half-ulp add rounding in the eight-warp body (same run, us,
  d768_out / large_deep / batch_out): 78.4 / 488.4 / 82.8 against the cvt form's 81.1 / 510.5 /
  85.5 (-3.4 / -4.3 / -3.2%), kept; against cuBLAS FAST 1.24 / 1.37 / 1.24. The four-warp add
  body in the same run: 5% behind it (2.98 / 2.80 / 3.09 over the scalar against 3.15 / 2.96 /
  3.24). The four-warp body is now templated on the k depth; its BK 16 five-stage twin
  (`..._m128n128w4_bk16_s5`, 84480 B of stages, 242 registers, no spills, 391 static SASS per
  64 HMMA) joins the wide route table (`Tf32PortableTile::M128N128W4K16`,
  `Tf32PortableStages::S5`) for the next same-run comparison of all three.
- 2026-09-05 14:15 — five wide bodies in one run (speedup over the exact scalar, d768_out /
  large_deep / batch_out): eight-warp add 3.151 / 2.959 / 3.232; the same body with its
  fragment loads written one atom group at a time between the mma groups 3.151 / 2.962 /
  3.233 (ptxas emits the identical loop: this pair is the same-run noise floor, 0.1%); the
  grouped body with the stage boundary before the last k8 step's mma 3.034 / 2.838 / 3.121
  (-4%); four-warp BK 32 2.998 / 2.800 / 3.096 (-5%); four-warp BK 16 five stages 2.733 /
  2.524 / 2.846 (-13%). The eight-warp add body stays; the schedule twins and the five-stage
  twin are removed. With a 0.1% same-run floor the earlier cross-run verdicts inside the
  3-5% drift band are re-measured as twins: first the clamped-source copy path (whole
  stages copy without a length operand), `..._m128n128c_bk32_s3`.
- 2026-09-05 14:41 — clamped-copy twin in the same run (speedup over the exact scalar, d768_out /
  large_deep / batch_out): 3.088 / 2.902 / 3.178 against the base 3.172 / 2.990 / 3.259 (-2.7%),
  dropped; the base eight-warp add body (77.9 / 489.2 / 82.2 us) closes the wide-tile variant
  round. Checkpoint: full NN requalification with the four-warp twin still competing
  (internal/perf/sm89-wide5-20260905), rows, identity, gate, commit. Handoff for the next
  engineer: internal/handoff-codex-gemm-bi-triad-2026-09-05.md.
- 2026-09-05 15:25 — full SM89 NN requalification on the half-ulp add body with the four-warp
  twin competing (internal/perf/sm89-wide5-20260905, 15 cells, 7 windows, p50 graph us; cuBLAS =
  f32 fast of sm89-hotall-20260904):
  | cell | m,k,n | winner | winner us | x scalar | winner/cuBLAS (was 2026-09-04 21:25) |
  | nn d768_in_proj | 2048,768,3072 | m128n128 s3 | 125.8 | 2.80 | 1.23 (1.56) |
  | nn d768_out_proj | 2048,1536,768 | m128n128 s3 | 77.8 | 3.22 | 1.24 (1.46) |
  | nn prism_in_proj | 4621,384,1928 | m128n128 s3 | 106.0 | 2.79 | 1.03 (1.29) |
  | nn large_deep | 4096,3072,1536 | m128n128 s3 | 485.5 | 2.85 | 1.37 (1.55) |
  | nn large | 2048,3072,768 | m128n128 s3 | 149.4 | 3.29 | 1.25 (1.41) |
  | nn batch_in_proj | 10400,384,1536 | m128n128 s3 | 203.2 | 2.45 | 1.39 (1.81) |
  | nn prism_out_proj | 4621,768,384 | m128n128 s3 | 42.2 | 3.42 | 1.20 (1.59) |
  | nn prism_input_proj | 4621,1024,384 | m128n128 s3 | 54.3 | 3.48 | 1.22 (1.54) |
  | nn batch_input_proj | 10400,384,384 | m128n128 s3 | 46.4 | 3.13 | 1.18 (1.87) |
  | nn batch_out_proj | 10400,768,384 | m128n128 s3 | 82.1 | 3.34 | 1.23 (1.92) |
  | nn rect_tall | 4096,512,768 | m128n128 s3 (was m64n64 s2) | 57.0 | 2.69 | 1.21 (1.84) |
  | nn d128_in_proj / d128_out_proj / underfill / split_candidate | | m64n64 s2 / m16n32 s4 / splitk2 m16n32 s4 / scalar | 9.9 / 7.5 / 10.8 / - | | 1.74 / 1.09 / 1.08 / 1.93 (unchanged) |
  The four-warp twin won no cell and is removed with the clamped-copy twin; the 14 NN rows are
  replaced from this record (69 rows, gate counts [5, 3, 36, 25]), the SM89 identity refrozen
  from a one-cell capture of the final tree (internal/perf/sm89-identity-halfulp-20260905).
  CPU gate on ada: fmt, clippy, arch_compile_gates 50 green; GPU suites running; then the commit.
- 2026-09-05 15:50 — commit 417eafa1 (half-ulp add rounding, the same-run twin protocol, 14 NN
  rows from sm89-wide5-20260905, SM89 identity refrozen). Gate on ada 13.2 all green (fmt,
  clippy, lib 538, arch 50, identity 22 + cuda 2, contract 82, tc 33, selector units 27, cohort
  binding smoke). OPEN, found by the smoke: for (2048, 768, 3072) under the TF32 policy the
  production path served m64n64_bk32_s2, not the wide tile the table row names — every NN row
  carries `RequiresNoBiasAndVectorAlignmentEvidence`, so a request with a bias may fall past
  the wide-tile rows. To settle first (handoff section 6a). Handoff for the next engineer:
  internal/handoff-codex-gemm-bi-triad-2026-09-05.md.
