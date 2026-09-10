# 0.7.0 public-documentation handoff audit

Read-only audit at branch `codex/gemm-bi-triad-sm80`, HEAD
`b7397d4fd53e52c0d070a76b9e91ee3ed51fabc4`, on 2026-09-10. Public files
were unchanged during this audit. Task 905 model graph/no-vendor proof was
in progress; Task 906 additive constructors were planned, not yet dispatched.
This report does not
claim either is complete and does not propose signatures beyond the approved
constructor map.

## Replacement authority

The following short labels are used in the tables.

- **Decision** — `internal/release-0.7.0-checklist.md`: default
  `GemmMode::Deterministic`, custom Inference for model/inference contexts and
  custom Triad for trainer/generic contexts, with no hidden vendor fallback;
  `CublasFast` and `CublasPedantic` are explicit vendor opt-ins. Storage dtype,
  F32/TF32 policy, half policy and backend mode are separate concepts.
- **API** — `internal/release-mode-constructor-signatures-20260910.md`: exact
  additive constructors/accessors and their environment behavior. Existing
  no-mode high-level constructors remain environment-aware. Explicit-mode
  constructors bypass GEMM mode/precision/tensor-core/family environment
  selection, but not the separate first-use Inference architecture-rung policy.
- **Inference280** —
  `internal/perf/final-auto-benchmarks-20260910/inference-results.md`: 280
  records per board, CUDA 13.2, measured production source `732c1146...`, with
  precision, bias, eager/whole-graph and board kept distinct.
- **Triad324-Ada** and **Triad324-5090** —
  `internal/perf/final-auto-benchmarks-20260910/triad-ada-results.md` and
  `triad-sm120-results.md`: 324 records per board, CUDA 13.2, measured source
  `732c1146...`; Fast and Pedantic denominators are explicit. Neither packet
  supports a whole-Triad Fast-win claim.
- **Architecture** —
  `internal/perf/final-auto-benchmarks-20260910/architecture-release-wiring-audit.md`:
  preserve compile/live/performance distinctions. Do not extend CC 12.0
  evidence to CC 12.1 or describe guarded SM90a/SM100 heuristics as measured
  winners.
- **Vendor** — `internal/release-vendor-wording-sources-20260910.md`, backed by
  NVIDIA CUDA 13.2.1 cuBLAS documentation: Pedantic is a precision/optimization
  policy, not a universal determinism guarantee; cuBLAS has conditional
  reproducibility and is not inherently nondeterministic.
- **Layout** — `internal/release-test-layout-audit-20260910.md`: planned explicit
  R regressions, B benches, Q qualification tools and excluded E experiments.
  Its commands are recommendations to apply after the move, not commands that
  were run in that audit.

## Canonical public story

Use this as the common replacement, with detail expanded once in Rustdoc and
once in `docs/determinism-benchmarks.md`:

1. `GemmMode::Deterministic` is the 0.7.0 default. It selects only custom
   kernels: Inference for model/inference contexts and Triad for trainers and a
   generic `GpuCtx`. It must not silently execute a vendor GEMM.
2. `GemmMode::CublasFast` and `GemmMode::CublasPedantic` are explicit vendor
   modes. Fast and Pedantic state numerical permission; do not label either one
   universally deterministic/nondeterministic.
3. `WeightDtype::{F32,Bf16,F16}` selects storage. It does not select a GEMM
   mode. Exact F32, deterministic TF32, and the half tiled/stream-K choices are
   custom numerical policies inside Deterministic mode. Deterministic TF32 is
   not the vendor Fast-TF32 mode.
4. Describe repeatable bits only at the scope actually proved: selected route,
   board/toolkit, eager/graph path and test fixture. `CublasPedantic` does not
   imply cross-GPU or cross-toolkit identity. The existing positive M1/M3
   decode graph receipt proves route stability and finite replay, not whole-model
   eager/graph bit equivalence or throughput.
5. Current 0.7.0 public performance claims may use only the qualified
   Inference280 and Triad324 packets, retaining board, path, precision and
   comparator labels. They are kernel/graph event timings, not whole-model
   inference/training numbers. There is no measured old-`main` versus final
   0.7.0 delta yet.

## Per-public-file findings

Line numbers are anchors at the audited HEAD, not permanent references.

| Public file | Actual anchors and conflict | Required replacement/removal and supporting receipt |
|---|---|---|
| `README.md` | `19-27`, `28-42`, `43-81`: deterministic execution is called opt-in, cuBLAS is called the default, and legacy flags/families are presented as the primary API. `50-61` still says the SM120 half table has 18 cells and extends a negative claim to CC12.1. `82-86` presents guarded Hopper/Blackwell rungs without the compile/live/performance qualification. | Replace the opening GPU story with the three `GemmMode` values and role defaults from **Decision/API**. Keep legacy environment controls only in a compatibility subsection. Replace the 18-cell text with the current source scope (60 tiled and 12 stream-K entries), but describe only measured/admitted cells and do not infer CC12.1. Qualify architecture claims with **Architecture**. Do not publish model-wide no-vendor wording until Task 905 closes. |
| `README.md` | `16-18`, `258-265`, `279-283`, `292-323`: storage dtype is explained but mode is absent, so snippets imply dtype chooses the relevant compute behavior. `388-445` repeats the old cuBLAS-default/tier model; `422` calls cuBLAS F32 “non-deterministic.” | State that `WeightDtype` and `GemmMode` are orthogonal. After Task 906 lands, show the exact **API** entries `GpuMambaBackbone::new_with_dtype_and_mode`, `GpuMambaLM::from_hf_with_dtype_and_mode`, and `MambaTrainer::new_full_with_mode`; show `ctx().gemm_mode()` for inspection. Replace `non-deterministic` with the literal comparator, for example Fast TF32, and use **Vendor** for the reproducibility scope. |
| `README.md` | `384-461`: old whole-model and training tables are presented as current, comparators are inconsistently labelled, and `431-433`/the package-style narrative implies a general deterministic speedup. `286-288` also says “bit-identical up to ... roundoff,” which is internally inconsistent. | Label retained model tables explicitly as versioned historical measurements or move them to the detailed versioned benchmark pages. A 0.7.0 current-results paragraph should point to one canonical final-AUTO table and say it is kernel-only. Use **Inference280/Triad324-Ada/Triad324-5090**; do not derive a whole-model or old-main delta. Say either bit-identical (bits) or tolerance/KL parity, never both. |
| `README.md` | `477-508`: “fast”/“full” commands assume Cargo auto-discovers every test and that `--include-ignored` is the full qualification suite. | After the **Layout** move, describe ordinary R regressions separately from explicit Q qualification and B benches. Keep a small host/CUDA R smoke command and link the authoritative command matrix in `docs/release-qualification.md`; do not call `cargo test ... --include-ignored` the full release qualification. |
| `CHANGELOG.md` | The file starts at `0.6.9` (`3`); there is no 0.7.0 entry. `0.6.9:10-14` and older `Fixed`/flag/default statements are historical release facts, not the 0.7.0 API. | Add a 0.7.0 entry only after the relevant work is complete. Preserve old entries as history; do not retroactively rename their public APIs or tables. The new entry should separate the completed Inference rename/kernel assembly from Task 905 and Task 906, and should not claim those tasks early. Use **Decision/API** for behavior and **Inference280/Triad324** for qualified current kernel comparisons. Do not add an old-main speedup until that immutable A/B exists. |
| `Cargo.toml` | `3` is still `0.6.9`. Description `7` says determinism is opt-in and that a tensor-core tier “beats cuBLAS on LLM-sized models.” | Bump version with the lockfile in the release phase. Replace the description with factual scope: CPU/GPU Mamba-1/Mamba-3 inference/training, CUDA Graphs, storage dtypes, deterministic custom GEMM default, and explicit vendor modes. Remove the unscoped performance claim. Supported by **Decision**; the final packets do not prove a whole-model general win. |
| `docs/determinism-benchmarks.md` | `7-43` is organized around legacy flags/tiers, says cuBLAS is off/default, and repeats the obsolete 18-cell/CC12.1 boundary. `254-264` again calls cuBLAS the crate default. | Make this the canonical detailed mode/policy page: mode table first, then role family, storage dtype, custom precision policy, graph restrictions and architecture scope. Use **Decision/API/Architecture**. The page should explicitly distinguish custom deterministic TF32 from `CublasFast` Fast-TF32. |
| `docs/determinism-benchmarks.md` | `50-224`, `316-344` are older exploratory/whole-step/tile tables; several denominators are ambiguous, and `318-335` uses the retired `fixed` name. `273-314` concludes “no default flip” for a cuBLAS lane, which is obsolete under the new default. | Retain older data only in a clearly dated historical appendix with literal comparator definitions, or remove it from the current release story. Rename historical family labels only in explanatory prose (`Fixed`, now `Inference`) without rewriting commands/results. Reframe the cuBLAS probe as Fast-versus-Pedantic precision evidence, not a default decision. Use **Vendor**. |
| `docs/determinism-benchmarks.md` | No table reports the completed final AUTO packet. `226-252` reproduces timing arms inside tests that **Layout** will split into B/Q; `275`'s `cublas_compute_probe` becomes Q. | Add one compact current section sourced verbatim in meaning from **Inference280/Triad324-Ada/Triad324-5090**: source `732c1146...`, per-board and eager/graph columns, exact comparator, record counts, and the “kernel-only/no whole-Triad Fast win” limit. Update reproduction commands after target names are final: B via `cargo bench --bench ...`; Q via `--features cuda,qualification --test ...`; R contracts stay `cargo test`. |
| `docs/mamba1-architecture.md` | `91-137` defines numeric routing through old flags: cuBLAS default, `set_fast_gemm`, `set_batch_invariant`, family and tensor-core toggles; it repeats the 18-cell/CC12.1 statement and broadly says graph captures assert the full route. | Replace with the mode/policy separation from **Decision/API**, identifying M1 model contexts as Inference and trainers as Triad. Link the detailed policy/benchmark page instead of repeating table topology. Keep route-level graph rules, but gate model-wide replay/no-vendor wording on Task 905 and preserve the narrow positive-graph receipt scope. Qualify architectures with **Architecture**. |
| `docs/mamba1-benchmarks.md` | `6-14`, `16-20`, `164-183`, `210-218`: custom execution is called opt-in and cuBLAS/TF32 the shipped production default. `125-127` frames cuBLAS as sacrificing determinism. `56-58` is an external-framework comparison. | Make all existing numbers explicitly versioned historical measurements, remove the framework comparison, and replace the mode narrative with a short link to the canonical mode page. Describe the measured batch-dependent result for that fixture without generalizing cuBLAS. Use **Decision/Vendor**. Do not substitute final-AUTO kernel numbers into whole-model tables. |
| `docs/mamba1-benchmarks.md` | `224-241`: `rl_llm_bench` and `bench_bf16_vs_f32` are Q under **Layout**, while `gpu_bf16_parity`, `hf_batch_parity` and `extreme_edge_coverage` remain R. | After relocation, add `qualification` to Q invocations and use their final declared target paths/names. Keep R commands as ordinary CUDA/HF regressions. Move the detailed mode benchmarks to `docs/determinism-benchmarks.md` and cross-link them rather than repeating claims here. |
| `docs/mamba3-architecture.md` | `98-102` says `set_batch_invariant`/`set_bi_gemm_family` steer the forward and provides no 0.7.0 mode or storage distinction. | Replace with the M3 role statement: model/inference contexts use Inference, trainer contexts use Triad; default mode is Deterministic and vendor modes are explicit. Cross-link the canonical page. Do not claim complete model-wide enforcement until Task 905. `121-129` kernel-count material is unrelated to the mode release and should be retained only if separately maintained. |
| `docs/mamba3-benchmarks.md` | `181-193` repeats the old opt-in flag/family story. `13-93` presents earlier 5090 model numbers as the current production story. `139-150` uses synthetic cross-architecture “faster” headlines that are not comparable release evidence. | Replace the mode section with a short **Decision/API** summary and canonical link. Mark old model numbers by release/source or archive them; the final AUTO receipts are kernel-only. Remove or neutralize the synthetic cross-model ranking and retain architectural facts in the architecture page. |
| `docs/mamba3-benchmarks.md` | `171-179`: `m3_gpu_benchmark` and `m3_cpu_benchmark` become B; `bench_bf16_vs_f32`, `rl_llm_bench`, and referenced `m3_prefill_bench` (`83`) are Q. | Change B invocations to the final `cargo bench --bench ...` targets. Change Q invocations to include `cuda,qualification` (and `hf` where required) after manifests land. Link the canonical release-qualification command matrix. |
| `docs/performance-playbook.md` | `3-38` and `134-162` are verbose campaign anecdotes; `29-38` teaches the obsolete two-flag tier selection. `168-192` contains dated execution priority that is no longer true because kernel assembly is complete. | Keep a concise, professional measurement protocol; move campaign history and superseded priority to internal evidence. Replace flag instructions with printing/resolving `GemmMode`, family and numeric policy separately. Preserve `199-269`'s useful comparator, provenance, per-domain and fail-closed rules, updated to **Decision/Vendor/Layout** terminology. Cross-link release qualification for commands. |
| `docs/release-qualification.md` | `10-19` says `qual/lanes.toml` covers every suite; **Layout** found it incomplete and recommends explicit Cargo targets. `38-49` has no R/B/Q split, equates one serve-page A/B with the performance guard, and omits the owner's publication approval. | Rewrite around declared R/B/Q targets: ordinary smoke does not build Q; Q is explicit; B is `cargo bench`; E is excluded. Add extracted-package inspection/offline checks from **Layout**. Keep per-board/toolkit/precision/eager/graph records separate. State that the old-main versus final immutable harness is still pending and that publishing requires explicit owner approval. `acceptance_diff` (`23-36`) can remain if its example stays packaged/reachable. |
| `examples/acceptance_diff.rs` | `1-9` describes a generic evidence diff and contains no GEMM-mode claim. | No semantic mode change. If it remains the public qualification tool linked from `docs/release-qualification.md`, ensure the explicit package target includes it; otherwise relocate it with Q and update the link. **Layout** supplies the boundary. |
| `examples/cpu_prefill.rs` | `1-17` is CPU-only and does not discuss `GemmMode`. | No mode change. Retain the CPU command and storage/CPU-backend explanation; avoid importing the CUDA vendor/custom story here. |
| `examples/custom_loss.rs` | Constructor at `48-60` uses environment-aware `MambaTrainer::new_full`, so it does not teach the 0.7.0 explicit mode/storage separation. | After Task 906, use the exact `MambaTrainer::new_full_with_mode(..., WeightDtype::Bf16, GemmMode::Deterministic)` signature from **API** (or explicitly explain why the environment-aware lane is intentional). State that this is Triad and that the split remains eager. Do not imply Task 905 graph proof applies to this split. |
| `examples/gpu_inference.rs` | `7-11`, `32-45`: dtype is treated as the significant route selector; the example asserts batch-invariant behavior without selecting/inspecting mode and includes unsupported generic `~30%` and `~2-5x` claims. | After Task 906, use exact `GpuMambaBackbone::new_with_dtype_and_mode(..., WeightDtype::Bf16, GemmMode::Deterministic)` and inspect `ctx().gemm_mode()`. Explain Inference role and separate storage from mode. Remove the generic speed claims unless tied to a reproducible, versioned whole-model receipt. Do not claim model-wide no-vendor until Task 905 passes. |
| `examples/gpu_training_bf16.rs` | `7-19`, constructor `89-101`: says it exercises batch-invariant GEMM but relies on ambient `new_full`; `118` describes cuBLAS warmup even though 0.7.0 Deterministic is the default. | Use exact `MambaTrainer::new_full_with_mode(..., WeightDtype::Bf16, GemmMode::Deterministic)` from **API**, identify Triad, and describe warmup as route/artifact preparation rather than cuBLAS selection. The printed local eager/graph ratio may remain clearly example-local, not a release performance claim. |
| `examples/inference.rs` | `1-9` and body are CPU-only. | No mode change. Keep separate from CUDA `GemmMode` documentation. |
| `examples/mamba3_gpu_inference.rs` | Constructor `27-32` uses ambient F32 `new`; `27` hard-codes a brittle CUDA-kernel count; `34` makes a generic graph speed claim. | After Task 906, use exact F32 shortcut `GpuMamba3Backbone::new_with_mode(..., GemmMode::Deterministic)` and inspect the mode through the widened public `ctx()`. Explain the Inference role. Remove the kernel count and unscoped speed ratio unless independently maintained. Gate model-wide no-vendor wording on Task 905. |
| `examples/mamba3_gpu_training_bf16.rs` | Constructor `82-94` uses ambient `new_full`, so storage and backend mode are not explicit. | After Task 906, use exact `Mamba3Trainer::new_full_with_mode(..., WeightDtype::Bf16, GemmMode::Deterministic)` and identify Triad. Keep the example-local measured eager/graph output as local output, not a release claim. |
| `examples/mamba3_inference.rs` | CPU-only (`1-10`). | No mode change. |
| `examples/mamba3_training.rs` | CPU-only (`1-13`). | No mode change. |
| `examples/train_and_infer.rs` | CPU-only (`1-16`). | No mode change. |
| `examples/training.rs` | CPU-only (`1-19`); the link to the GPU example is the only indirect mode surface. | No code change. Ensure the linked GPU example becomes the canonical explicit Deterministic/Triad example after Task 906. |

## Exact API snippets that public prose may name after Task 906

Do not publish these as available before the implementation and Rustdoc gates
pass. The complete signatures and argument order remain in **API**.

- M1/M3 backbones: `new_with_mode` for the F32 shortcut and
  `new_with_dtype_and_mode` for explicit storage plus mode.
- M1/M3 inference engines and mixed engines: `new_with_mode`.
- M1 LM: `from_hf_with_mode`, `from_hf_with_dtype_and_mode`, and
  `from_hf_with_dtype_batch_and_mode`.
- M3 LM: `from_weights_with_mode` and `build_with_mode`; do not invent an
  eight-argument dtype constructor or add a required field to
  `Mamba3LmBuild`. Add/use the approved public `ctx()` inspection seam.
- M1/M3 trainers: `new_full_with_mode`; do not add an eighth argument to
  `new_with_dtype`.
- Inspection is `ctx().gemm_mode()` (and `ctx().bi_gemm_family()` where role
  evidence is useful), not duplicate model fields.

Every explicit-model constructor's Rustdoc must retain the architecture-rung
caveat from **API**: explicit mode construction bypasses the GEMM mode,
precision, tensor-core and family selectors, while `MAMBA_RS_ARCH_RUNG` remains
a separate first-use process policy for the Inference architecture rung.

## Command changes after the planned test layout

Do not update public commands until target relocation and manifest names are
final. The semantic changes are already determined:

| Current public command location | Planned lane | Required command shape |
|---|---|---|
| `docs/determinism-benchmarks.md:230-243` timing arms in `gemm_bi_determinism`, `gemm_bi_tc`, `gemm_bi_typed_parity` | Split R/B (and Q where hardware inspection remains) | R: `cargo test ... --test <retained>`; B: `cargo bench --bench <final-name>`; Q: `cargo test --features cuda,qualification --test <final-name> ...` |
| `docs/determinism-benchmarks.md:275` `cublas_compute_probe` | Q | Add `qualification` and invoke the explicitly declared Q target. |
| `docs/determinism-benchmarks.md:318` `classifier_gemm_tier_bench` | B | Use `cargo bench --bench <final-name>` after conversion. |
| `docs/mamba1-benchmarks.md:226-235` `rl_llm_bench`, `bench_bf16_vs_f32` | Q | Add `qualification` (and retain `hf`/`cuda` requirements) using final target names. |
| `docs/mamba1-benchmarks.md:236-241` parity/edge suites | R | Keep as ordinary explicit tests; do not conflate them with Q performance acquisition. |
| `docs/mamba3-benchmarks.md:173-174` CPU/GPU benchmark targets | B | Use final `cargo bench --bench ...` declarations with required features. |
| `docs/mamba3-benchmarks.md:175-178` `bench_bf16_vs_f32`, `rl_llm_bench`; `:83` `m3_prefill_bench` | Q | Add `qualification` and invoke final explicitly declared Q targets. |
| `README.md:498-508`, `docs/release-qualification.md:10-49`, `qual/run.sh` references | R/Q/B orchestration | A normal CUDA test is R smoke only. List Q rather than running it implicitly; run B through `cargo bench`; validate the extracted package directly as prescribed by **Layout**. |

## Narratives to deduplicate

- Put the full three-mode/default/environment/error/graph contract in Rustdoc;
  keep a short quick-start in `README.md` and link to it from both architecture
  pages and every example.
- Put the only current final-AUTO comparison tables in
  `docs/determinism-benchmarks.md`. README and the M1/M3 benchmark pages should
  link to those tables instead of restating selected ratios.
- Put architecture route coverage and SM120/SM90a/SM100 limits in one section
  of the deterministic benchmark page, sourced from **Architecture**. The README
  and M1 architecture page currently repeat and drift from it.
- Put executable R/B/Q/package commands in `docs/release-qualification.md`.
  Benchmark pages should link there and retain only a command specific to a
  table when it is stable after relocation.
- Keep the concise measurement method in `docs/performance-playbook.md`; move
  dated campaign chronology and internal task priority out of public docs.

## Release-documentation gates

- Task 905 must close model graph/no-vendor coverage before public model-wide
  Deterministic claims are enabled.
- Task 906 implementation, Rustdoc links/doctests and constructor propagation
  tests must pass before new signatures appear as available examples.
- Audit later API/cleanup changes against the measured route/source dependency;
  keep final-AUTO records labelled with measured source `732c1146...`.
- Do not publish a 5090 remeasurement or an old-main delta that was not run.
- After the test move, validate the extracted crate, declared target inventory,
  one R host/CUDA lane, one Q build/run and one B invocation as required by
  **Layout**. Publication still requires explicit owner approval.
