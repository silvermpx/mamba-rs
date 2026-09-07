# Ada exact-F32 / TF32 cross-toolkit harness audit

Date: 2026-09-07  
Immutable audit base: `4386e41a43f1b3b95baa829b1e7ab2b470eecf29`  
Scope: source-only, bounded to the two named comparator tests, their directly required helpers, the exact-N64 paired support module, and the stable interfaces already present in the Ada S3 paired module. The prerequisite `internal/perf/ada-half-s3-force-20260907/existing-toolkit-auto-gaps.md` was read completely first. Its loader/selector inventory is accepted and is not repeated here. No GPU, build, test, SSH, Git mutation, selector edit, or performance claim was made.

## Decision

Use `fixed_ada_forced_rungs_paired_precision_cublas` as the **single existing measurement spine**, narrowed to one row/tile/cell family per invocation. Do not add a new benchmark and do not use `fixed_ada_production_auto_paired_precision_cublas` for admission.

The forced-rung test already supplies production AUTO, a public forced tile, a separately timed explicit-compute cuBLAS arm, an untimed PEDANTIC F32 reference, eager/graph execution, bias-in-the-timed-vendor-workflow, raw output comparison, and useful compiler/artifact metadata. Its row registry already has exactly the two denominator contexts needed:

- `tf32`: F32 storage, `AllowDeterministicTf32V1`, timed `CUBLAS_COMPUTE_32F_FAST_TF32`, tolerance `0.0025`.
- `f32_exact_fast`: F32 storage, `ExactScalarFmaV1`, timed `CUBLAS_COMPUTE_32F_FAST_TF32`, exact custom tolerance `0.0002`, vendor tolerance `0.0025`; the separate reference remains `CUBLAS_COMPUTE_32F_PEDANTIC`.

Base anchors: `tests/gemm_bi_fixed_performance.rs:11426-11501` (row specifications), `:10321-10375` (explicit compute and timed bias workflow), `:13702-13728` (AUTO/vendor/reference first launches), and `:14145-14220` (emitted comparator metadata and completion), all at the immutable SHA above.

This spine is **not admission-ready as written**. It needs a narrow qualification mode or equivalently bounded amendments in the same test/helper area. The must-fix gaps are: exact toolkit and tuning-revision binding, exact expected AUTO incumbent, strict literal row/cell/bias inventory, full physical graph ABI/argument validation for the ordinary F32/TF32 tiles, guarded storage plus verified complement overwrite and a real no-op negative, input/bias pre/post immutability, true mirrored four-position ABBA/BAAB with both starting parities, exact 21/101 stage semantics, complete raw-record closure, binary/source/runner binding, and a p50 **and** p95 owner-win decision.

## Existing literal controls and the intended rows

The current strict list filter is reusable: it rejects empty, duplicate and unknown entries (`tests/gemm_bi_fixed_performance.rs:11190-11224`). The exact force registry and symbol mapping are also reusable (`:11687-11954`), including:

- `F32Sm89N64CopyPlan` -> `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1` for `f32_exact` or `f32_exact_fast` on CC8.9.
- `Tf32M64S2` -> `gemm_bi_nn_tf32_v1_m64n64_bk32_s2` for `tf32`.
- `Legacy` -> `gemm_bi_f32_f32_s2` for exact F32.
- `Tf32M128S2` -> `gemm_bi_nn_tf32_v1_m128n64_bk32_s2` for TF32.

The public forcing API is already sufficient: `fixed_forward_with_tile` directly launches the SM89 exact candidate and admits the portable TF32 tiles for homogeneous F32 operands (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:183-192,272-300`). No new kernel or force API is required.

Required current CLI selections for the two pre-selector families are:

| Family | Existing controls to set | Literal meaning |
|---|---|---|
| exact F32 | `MAMBA_FIXED_ADA_VENDOR=1`; `MAMBA_FIXED_ADA_ROWS=f32_exact_fast`; `MAMBA_FIXED_ADA_CELLS=hot_a,hot_b,hot_d,hot_e`; `MAMBA_FIXED_ADA_BIAS=0,1`; `MAMBA_FIXED_VENDOR_TILES=F32Sm89N64CopyPlan`; `MAMBA_FIXED_VENDOR_PATHS=eager,graph`; `MAMBA_FIXED_VENDOR_EXACT_CC=8.9`; `MAMBA_FIXED_ADA_WINDOWS=21` then `101` | A0/A1, B0/B1, D0/D1, E0/E1; forced CopyPlan against actual AUTO Legacy and a separate FAST_TF32 cuBLAS denominator; PEDANTIC is reference-only. |
| TF32 | `MAMBA_FIXED_ADA_VENDOR=1`; `MAMBA_FIXED_ADA_ROWS=tf32`; `MAMBA_FIXED_ADA_CELLS=hot_c`; `MAMBA_FIXED_ADA_BIAS=0,1`; `MAMBA_FIXED_VENDOR_TILES=Tf32M64S2`; `MAMBA_FIXED_VENDOR_PATHS=eager,graph`; `MAMBA_FIXED_VENDOR_EXACT_CC=8.9`; `MAMBA_FIXED_ADA_WINDOWS=21` then `101` | literal C0/C1; forced M64S2 against actual AUTO M128S2 and its own FAST_TF32 cuBLAS denominator. |

The labels and dimensions come from `FIXED_AUTO_VENDOR_EXACT_CELLS`: A `(4621,384,1928)`, B `(4621,768,2304)`, C `(4621,1928,384)`, D `(2048,768,2304)`, E `(2048,2304,768)` (`tests/gemm_bi_fixed_performance.rs:9440-9485`). Only the labels/shapes are reusable here; the table's `expected` field was authored for a different comparator context and must not decide these SM89 qualifications.

The ignored test target is `fixed_ada_forced_rungs_paired_precision_cublas` (`tests/gemm_bi_fixed_performance.rs:13562-13564`). The eventual runner can invoke that exact release test with `--ignored --exact --nocapture`; the audit does not prescribe or execute a build command.

### Missing fail-closed controls

There is currently no exact toolkit or tuning-revision input. Add one qualification-specific expected-toolkit binding (for example `MAMBA_FIXED_ADA_TOOLKIT=12.8` or `13.0`) and one expected tuning revision, and assert both against the loaded compiler identity and `TUNING_TABLE_REVISION` before allocation/timing. The base revision is 42 at `src/mamba_ssm/gpu/kernel_identity.rs:35`; records from a later AUTO43 integration must deliberately bind that later revision and must not be mixed with rev42 records.

There is also no general expected-AUTO phase control. Pre-selector qualification must fail unless actual AUTO returns `Legacy` for the exact row and `Tf32M128S2` for C0/C1. A later post-selector run must separately fail unless AUTO returns the promoted candidate. The exact support module's `MAMBA_FIXED_EXACT_N64_AUTO=legacy|candidate` and `auto_phase` logic are a usable model (`tests/support/fixed_sm89_exact_n64_admission.rs:207-230,1590-1606`), but the name/implementation is exact-only and should not be silently reused for TF32.

`NVIDIA_TF32_OVERRIDE` is only rejected when equal to `0` in both generic tests (`tests/gemm_bi_fixed_performance.rs:13297-13301,13572-13576`). Admission should require it to be absent, not accept arbitrary/nonzero stale values, so both FAST_TF32 contexts are native and reproducible.

## Harness-by-harness fitness

### `fixed_ada_forced_rungs_paired_precision_cublas`: retain and harden

What is genuinely checked now:

- **Actual AUTO return:** yes, partially. `launch_fixed_auto_vendor_custom` returns the production `FixedTile` (`tests/gemm_bi_fixed_performance.rs:9916-9930`); the first result is stored and every eager AUTO launch asserts it does not drift (`:13702-13704,13729-13735`). It does **not** assert the required incumbent value.
- **Rows/tiles/bias/paths:** strict optional row/cell/bias/tile/path filters exist (`:13580-13612`); the force specification validates row dtype and exact symbol (`:13737-13747`). Defaults are far too broad for this qualification, so all filters must be mandatory and the mode must reject unrelated/stale controls.
- **Denominators:** the row selects FAST_TF32, while a separate F32 output is launched with PEDANTIC before timing (`:13705-13728`). Bias broadcast plus beta=1 is inside each timed vendor call (`:10329-10373`). This is the required separation.
- **Graph existence and forced symbol:** with `graph` selected, AUTO/forced/vendor workflows are captured. The forced graph is required to contain the force-spec symbol (`:13802-13830`). The generic inventory records kernel names, geometry and shared bytes (`:13925-13980`).
- **Raw output/replay checks:** forced must equal AUTO bit-for-bit or it is rejected; forced repeat bits are checked (`:13882-13900`). Graph outputs are poisoned to `0xff` and replayed twice, and AUTO/forced/vendor are checked before and after timing (`:14030-14068,14123-14134`).
- **Artifact metadata:** it records compiler NVRTC version/target, SM count, Fixed source/invocation/artifact/header digests, library domain and known flag (`:13620-13643`) plus tuning revision per result (`:14145-14180`).

What is not genuinely closed:

- **Physical captured arguments:** no for these ordinary F32/TF32 routes. The exact forced symbol is asserted, but AUTO only gets a rendered inventory. The generic graph helper validates pointers/bundle/ABI only for Ada half descriptors, RNA-wide TF32, and one SM120 pair-store special case (`tests/gemm_bi_fixed_performance.rs:12154-12171,12358-12557`). It does not validate AUTO Legacy, CopyPlan, Tf32M64S2, or Tf32M128S2 argument pointers, dimensions/strides, ABI terminal, block/grid/shared contract.
- **Guards:** none around A, B, bias, or outputs.
- **Complement readback:** no. `cuMemsetD8(...,0xff)` is not derived from expected output and its poison upload is not read back before replay (`:12340-12355,14030-14050`).
- **No-op negative:** no real no-op graph is captured/launched. The comment at `:14032` describes the desired property but every real graph is then launched, so it is not a negative control.
- **Pre/post immutability:** output bits are checked, but A, B and bias are not snapshotted or checked after warmup/timing. There is no proof that all arms retain common input pointers beyond construction.
- **Timing protocol:** not mirrored pairing. Each window is either `AUTO,forced,vendor` or `vendor,forced,AUTO`, one observation per arm; forced is always in the middle (`:14079-14122`). Those are old triple AB/BA blocks, not ABBA/BAAB, and there is no starting-parity field.
- **Admission/completion:** the test accepts any window count `1..=10001`, emits aggregate arrays, merely requires `records > 0`, and ends `passed:true` even if requested cells were rejected (`:13588-13598,14212-14220`). It has no exact expected inventory and no p50/p95 owner decision.
- **Source/binary/runner closure:** Fixed NVRTC artifacts are identified, but the Rust source file, executable binary, source commit, exact command/environment, output file digest, and SSH copy/exit/host closure are not bound.

### `fixed_ada_production_auto_paired_precision_cublas`: do not use for admission

This test is useful only as historical eager AUTO/vendor smoke. It has no force arm, no graph path, no physical graph identity/arguments, no guards/poison/no-op negative, no input immutability, and no artifact digests. It stores/rechecks the AUTO return but never asserts the expected route (`tests/gemm_bi_fixed_performance.rs:13417-13428,13449-13455`). Its timing is one `custom,vendor` or `vendor,custom` observation per window (`:13474-13495`), explicitly emitted as `ab`/`ba` and eager-only (`:13507-13547`). Its exact row uses PEDANTIC as the timed denominator and has no `f32_exact_fast` row (`:13320-13334`), so it is wrong for the requested exact-F32 FAST_TF32 context. It records only CC/SM/NVRTC/target and revision (`:13361-13374,13507-13527`). No bounded amendment to this test is preferable to hardening the already three-arm forced-rung test.

### Exact-N64 paired module: reuse its gates, not its current run unchanged

`fixed_sm89_exact_n64_paired_admission` delegates to `tests/support/fixed_sm89_exact_n64_admission.rs` (`tests/gemm_bi_fixed_performance.rs:732-737`). It contains the strongest exact-F32 controls in scope:

- strict filters and exactly 21/101 windows (`tests/support/fixed_sm89_exact_n64_admission.rs:37-63`);
- candidate/AUTO/explicit Legacy/PEDANTIC/old-oracle/separate-reference arms (`:728-737,973-1002`);
- guarded A/B/bias and all six outputs, plus common-input pointer and pre/post immutability proof (`:859-907,920-1018`);
- actual AUTO allowlist/drift check and phase assertion (`:977-986,1772-1778`);
- full own-graph physical validation: node kind/count, actual symbol, ABI and terminal probe, four captured pointers, captured dimension bundle, grid/block/shared (`:1110-1257`);
- poisoned graph replay twice before and after timing, with complete raw equality (`:1467-1512,1779-1807,1841-1864`);
- exact pair-key inventory and create-new JSONL with prefix digest and synced completion (`:806-856,1608-1625,1961-1963`);
- CC8.9/142 SM/known-library/sm_89, detailed compiler/artifact/driver/revision/cublas identity, and pinned telemetry (`:1515-1538,1644-1690`).

It cannot be used unchanged for this task:

- identity hard-requires NVRTC 13.2 (`:1648-1653,1668`), excluding both target toolkits;
- its timed vendor arm and reference are both PEDANTIC (`:990-998,1881`), not FAST_TF32 with PEDANTIC reference-only;
- its poison is a fixed `0xa5a5a5a5`, not an expected-output complement with verified pre-replay readback (`:677-705,885-905`);
- its optional missing-launch red proves a skipped launch leaves poison, but it is not a captured no-op graph negative in the ordinary passing run (`:1693-1753`);
- it runs all ABBA windows as one block and all BAAB windows as another (`:1820-1855`). That is true four-position pairing, but not the requested alternating mirrored schedule with both starting parities;
- admission stores only p95 and tests `<1`; p50 is emitted but not part of the decision (`:195-205,1816-1819,1867-1898`);
- it hashes its two Rust source files and detailed loaded artifacts, but not the executable or outer command/SSH closure (`:1934-1940`).

The right reuse is to port/generalize its `Guarded`, `LaunchProof`/`own_graph`, common-input, replay, identity, exact inventory, and evidence-completion gates into the narrow forced-rung qualification mode. Do not turn this exact-only module into a second TF32 benchmark.

## Required physical contracts

The hardened spine must inspect the Driver graph and assert the actual launch, not infer it from the returned enum.

- Exact candidate: symbol `gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1`; exact support already validates the compact five-argument ABI, four pointers, eight parameter words, computed grid, `(128,1,1)` block, zero **dynamic** shared, and terminal argument rejection (`tests/support/fixed_sm89_exact_n64_admission.rs:1152-1257`). Its live resource gate separately expects 32,768 bytes static shared, zero local, at least 128 threads and three active blocks (`:182-193`).
- Exact incumbent: actual AUTO must return `Legacy` and capture `gemm_bi_f32_f32_s2`; the exact support module already validates its twelve-argument physical ABI and the same pointers/dimensions (`:1166-1237`).
- TF32 candidate: `gemm_bi_nn_tf32_v1_m64n64_bk32_s2`, grid `(438,1,1)` for C, block `(128,1,1)`, dynamic shared 32,768.
- TF32 incumbent: actual AUTO must return `Tf32M128S2` and capture `gemm_bi_nn_tf32_v1_m128n64_bk32_s2`, grid `(222,1,1)`, block `(256,1,1)`, dynamic shared 55,296.
- Both portable TF32 launches take four pointers plus the 24-byte `FixedTf32Params {m,k,n,lda,ldb,ldc}` bundle (`src/mamba_ssm/gpu/gemm_bi_fixed.rs:1633-1649,2316-2390`). Validate ABI `[(0,8),(8,8),(16,8),(24,8),(32,24)]`, terminal sixth-argument rejection, exact C/A/B/bias pointers, and words `[4621,1928,384,1928,384,384]`.

The forced-rung `single_graph_kernel_name` check remains useful but is insufficient by itself.

## Stable interfaces available from the base test file

The concurrent half-S3 pre/post harness is not an edit target. Only these stable base interfaces should be reused or generalized by the sole implementation owner:

- the mirrored three-arm scheduler alternates ABBA/BAAB by `(window + start) % 2`, reverses comparison traversal, and covers candidate/AUTO, AUTO/Fast, candidate/Fast in four positions (`tests/gemm_bi_fixed_performance.rs:43-56,88-107`);
- the output gate writes the bitwise complement of the expected bytes, reads the poison back, proves every storage word differs, launches exactly one workflow, then requires exact expected bytes and intact guards (`:143-177`);
- the passing run also captures a true empty graph and proves it fails that overwrite gate (`:522-526`);
- the stable identity pattern asserts CC8.9/142 SM, exact toolkit text, known library, an exact tuning revision, source SHA and current executable SHA, and emits the Fixed artifact digests (`:381-424`);
- the stable timing loop uses both `start_parity` values, raw chronology/traversal/comparison/position records, 128 warmups, 20 logical operations, eager and 20-operation graph windows, and post-timing input immutability (`:528-616`).

Those are interface precedents, not authority to edit or broaden the half harness.

## Bounded next-stage recipe

1. **Harden only the existing forced-rung comparator path.** Add a literal cross-toolkit qualification mode to `fixed_ada_forced_rungs_paired_precision_cublas` and directly shared helpers. Reject missing/extra row, cell, bias, tile, path, toolkit, stage, revision, source and binary bindings. Do not alter kernels, loaders or selectors.
2. **Bind one reviewed source and one binary per toolkit.** Before any timing, require CC8.9, 142 SM, `sm_89`, known NVRTC library, exact toolkit `12.8` or `13.0`, intended tuning revision, compiler/header/invocation/artifact identities, Rust source hash and current executable hash. The external runner—not the benchmark—must bind exact command/env, host/GPU identity, unique binary resolution, SSH destination/exit status if SSH is used, copied artifact hashes, log hash and nonzero-failure propagation. None of the inspected admission candidates currently closes all of that.
3. **Run the negative overwrite control first.** Capture/inspect every real workflow, verify complement upload readback and guards, then show a captured no-op graph fails without emitting a passing completion. Run single-term/bias-orientation and exact-F32 finite/order controls before timing. A failure aborts the family/toolkit stage.
4. **Pre-selector screen, separately per family and toolkit.** Use the exact CLI inventories above with 21 windows. Assert AUTO is physically Legacy for exact or M128S2 for TF32. Use the three-arm mirrored scheduler for eager and graph, both starting parities, with raw per-position and same-window pair records. Do not pool toolkits, biases, paths, parities or cells.
5. **Advance literal owners independently.** For each literal cell+bias+toolkit, candidate/AUTO must have finite positive p50 and p95 `<1.0` in every eager/graph and starting-parity stratum, with all correctness/physical/guard/immutability/telemetry gates passing. Candidate/Fast is an independent, fully recorded denominator comparison, not an owner-win gate: Fast may still lead without invalidating a robust candidate/AUTO win. A complete, otherwise valid literal whose candidate/AUTO result ties, loses, or has a mixed p50/p95 verdict is simply ineligible; eligible siblings in the same family/toolkit remain independent, and one toolkit never carries another. Missing, duplicate or unexpected records, or any failed functional, numeric-correctness, physical, identity, guard, immutability, telemetry, completion or process-exit gate, instead invalidate the affected run/stage: no literal may be admitted from incomplete or invalid evidence.
6. **Repeat only explicit eligible literals at 101.** The 21-window screen covers the full authorized inventory. Re-run at 101 only the literal cell+bias+toolkit entries whose complete path/parity screen strata passed. This must be a fresh 101 run using the **same frozen measured source, executable binary and compiled-artifact binding** as its screen; a fresh process/cache state may be useful but is not an admission requirement, and a rebuild/new artifact would instead require a new matching screen. Apply the same per-literal candidate/AUTO p50+p95 owner rule. Do not aggregate 21-window samples into 101.
7. **Inventory closure.** The full screen per toolkit has exact 8 literal bias rows x 2 paths x 2 start parities = 32 configurations and TF32 2 x 2 x 2 = 8. Each configuration must contain `12 * windows` raw observations, `3 * windows` four-position pair records, and three summaries, so a complete full screen closes 40 configurations and `120 * windows` pair records, split into separate family artifacts. An eligible101 artifact instead closes exactly the explicitly enumerated eligible literal subset times both paths and both start parities; it must not require or silently include screen losers. Completion must reject duplicates, omissions and unexpected literals and include the preceding JSONL digest.
8. **Selector work remains a later root-owned decision.** Only after an individual literal passes eligible101 may root consider adding that exact tested toolkit literal, without widening device/SM/shape/bias/policy/library/holder guards or changing sibling fallbacks. Then rebuild and run a separate post-selector actual-AUTO phase: first a functional/physical smoke asserting AUTO is the candidate and capturing its full args, then a fresh 101 confirmation comparing AUTO against the explicitly forced former incumbent and that family's independent FAST_TF32 denominator, followed by retained production-NVRTC routes. A second 21-window candidate lottery is not required by default. A pre-selector forced win is not post-selector proof.

## Admission-changing risks

- Base SHA has tuning revision 42 while Task6C/AUTO43 integration is concurrent. Mixing binaries, source hashes or records across revisions invalidates the qualification even if the relevant selector text appears unchanged.
- The exact support harness's 13.2 identity and PEDANTIC timed comparator can silently answer the wrong question if reused wholesale. The required exact row is `f32_exact_fast`; PEDANTIC is reference-only.
- Merely printing `auto_tile` or a graph symbol is not physical proof. The ordinary TF32 routes and exact AUTO need full pointer/parameter/ABI/geometry checks.
- The generic `passed:true` completion currently means only that some record ran. It cannot be treated as an admission verdict.
- Old AB/BA or separate all-ABBA/all-BAAB blocks do not satisfy both-starting-parity mirrored pairing. Recomputed ratios cannot repair missing chronology.
- A `0xff` memset without pre-replay readback, or a skipped launch red without a real no-op graph, does not close the overwrite/no-op gate.
- No result may widen a guard or claim a performance win from this audit. A candidate/AUTO loss in either p50 or p95 on any path/parity stratum preserves the current fallback for that exact literal cell+bias+toolkit only; eligible siblings remain independently admissible. A candidate/Fast deficit is recorded independently and is not by itself a selector veto.
