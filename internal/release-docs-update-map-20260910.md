# Release 0.7.0 documentation update map

Date: 2026-09-10
Scope: shipping README, `docs/`, examples, public Rustdoc surfaces, changelog,
and Cargo package metadata. This is a preparation checklist, not a rewrite.

## Canonical sources and evidence

- Mode names and semantics: `src/mamba_ssm/gpu/gemm_mode.rs:3-65`.
  `GemmMode::Deterministic` is the default custom-GEMM mode;
  `CublasFast` and `CublasPedantic` are explicit vendor modes.
- Construction, environment parsing, capture-aware transition and deprecated
  compatibility methods: `src/mamba_ssm/gpu/context.rs:87-145`,
  `619-714`, `1078-1235`, `1770-1790`.
- Rename: the public/source family is `gemm_bi_inference` and
  `BiGemmFamily::Inference`; the legacy selector text `fixed` remains an input
  alias. Frozen virtual compiler filenames and artifact identities that still
  contain `gemm_bi_fixed` are compatibility/evidence identities and must not
  be mechanically renamed.
- Final production AUTO comparisons:
  `internal/perf/final-auto-benchmarks-20260910/inference-results.md`,
  `triad-ada-results.md`, and `triad-sm120-results.md`. All three pin measured
  source `732c11462dd9a47376e6d15b36040ca69bde3e34`; they are kernel/graph event
  measurements, not whole-model throughput or measurements of later source.
- Architecture/admission limitations:
  `internal/perf/final-auto-benchmarks-20260910/architecture-release-wiring-audit.md`.

## File and section checklist

| File / section | Stale or missing content | 0.7.0 update and evidence boundary |
|---|---|---|
| `Cargo.toml` `[package]` | Version remains `0.6.9`. Description says determinism is opt-in and that the tensor-core tier beats cuBLAS on LLM-sized models. | Bump with the release manifests. Replace the performance promise with neutral implemented capabilities: deterministic-by-default custom GPU GEMMs, explicit cuBLAS Fast/Pedantic modes, graph capture, and supported dtypes. The final receipts do not justify a blanket cuBLAS-win sentence. |
| `README.md` Features, lines 19-27 | Says deterministic inference/training is opt-in and cuBLAS is default; documents only `MAMBA_RS_BATCH_INVARIANT` / `set_batch_invariant`. | Lead with canonical `GemmMode`: `Deterministic` default, `CublasFast` and `CublasPedantic` opt-in. Document `MAMBA_RS_GEMM_MODE=deterministic\|cublas-fast\|cublas-pedantic`, `GpuCtx::new_with_mode`, and fallible `set_gemm_mode`. Move old flags/methods to a clearly deprecated migration note. Do not claim all high-level model helpers honor the mode until the checklist’s direct-cuBLAS seams are closed. |
| `README.md` family/tier descriptions, lines 29-85 | Family rename is mostly present, but mode and tensor-core policies are still described as flags layered over an opt-in deterministic switch. “Fastest qualified typed route” is globally phrased. SM120 is described as exactly 18 cells; every non-cell and CC12.1 is said to decline; stream-K is generalized to CC12.x. | Retain `Triad` versus `Inference`, note legacy `fixed` only as selector compatibility, and describe F32/TF32/half settings as policies inside `Deterministic`, not modes. Scope “qualified” to exact device/toolkit/shape cohorts. Replace the 18-cell statement with the current 60 tiled + 12 stream-K entries and explain guarded nearest-cell interpolation: exact entries are measured, interpolated shapes are admitted by policy but are not independently measured. Keep CC12.1 separate and do not extend CC12.0 evidence. Call SM90a/SM100 native half selection guarded heuristic coverage, not measured fastest routes. |
| `README.md` typed routing, lines 146-164 | Does not introduce the three canonical modes at the low-level entry point. | Add a short mode example and link to `GemmMode`/`GpuCtx` Rustdoc. Preserve the warning that forced SM120 APIs are qualification hooks, not AUTO admission. |
| `README.md` GPU quick start, lines 251-269 | Dependency example says `0.6`; example gives no mode/default explanation and includes an unscoped `~2×` graph comment. | Change dependency to `0.7`. State that the constructor’s context defaults to `Deterministic` only after that constructor is verified to propagate the canonical mode. Show explicit vendor opt-in through a supported high-level API if one ships. Remove or precisely source the graph speedup; final AUTO receipts do not measure whole-model decode. |
| `README.md` Performance / family choice, lines 384-460 | Table labels cuBLAS as crate default; instructions use deprecated setters. It calls cuBLAS F32 “non-deterministic,” gives old single-Ada family/training tables, and makes broad fastest/parity claims. | Relabel old tables as historical 0.6.x measurements or move them behind the benchmark docs. Replace setup snippets with `GemmMode`. “Not batch-invariant across algorithms/stacks” is supportable; “non-deterministic” is not the precise mode distinction. Add a compact final dual-GPU kernel/graph table only with the exact values below and its comparator/numeric-contract qualifiers. Do not derive a model or trainer speedup from those kernel timings. |
| `README.md` Documentation index, lines 510-519 | Links do not identify a 0.7 mode migration/API reference or the final dual-GPU measurement scope. | Add links to canonical mode Rustdoc and refreshed determinism/benchmark sections. Keep internal receipt paths out of the polished user index unless the project intentionally ships them. |
| `docs/determinism-benchmarks.md` Tiers and contracts, lines 7-48 | `(off) = cuBLAS` and deterministic-as-opt-in are obsolete. It repeats the 18-cell/every-non-cell fallback model. | Reframe the table around the three `GemmMode` variants, then list family, TF32, tensor-core and stream-K as deterministic sub-policies. Correct SM120 exact-cell versus nearest-cell interpolation semantics and CC12.0 scope. Keep historic tables dated/device-specific rather than presenting them as current defaults. |
| `docs/determinism-benchmarks.md` run determinism and cuBLAS probe, lines 254-315 | Calls cuBLAS the crate default and concludes “no default flip”; older API labels conflict with 0.7. | Preserve the measurements as historical probe evidence, but say the release default is now `Deterministic`. Map old “cuBLAS lane” labels to `CublasFast`/`CublasPedantic` where the recorded compute type proves which one. Do not rewrite old measured values. |
| `docs/determinism-benchmarks.md` family comparison, line 316 onward | Test/table terminology still says `fixed`; old single-shape results do not represent the renamed final production selector. | Rename prose to Inference, retaining an exact old test name only when needed for reproducibility. Label the table historical and do not combine it with final AUTO aggregates. |
| `docs/mamba1-architecture.md` Numeric routes, lines 91-125 | Says default cuBLAS; uses `set_fast_gemm` and `set_batch_invariant`; repeats 18-cell and every-outside-table fallback claims. | Replace the top-level selection description with `GemmMode`, retain the sub-policy hierarchy, explain capture rejection on mode changes, and correct SM120 interpolation/CC12.1 wording. |
| `docs/mamba3-architecture.md` around line 99 | Describes dispatcher behavior through `set_batch_invariant`. | Use canonical modes and state only the context-carrying paths proven to route through them. Add a caveat or defer final wording until remaining projection/LM-head direct-cuBLAS seams are resolved. |
| `docs/mamba1-benchmarks.md` intro and deterministic sections (lines 1-16, 164-219) | Uses opt-in/deprecated selector language and mixes several historical release measurements. | Mark each table with release/source/device/toolkit and mode. Replace API instructions with canonical names. Do not merge historical step timings with final GEMM-only comparisons. |
| `docs/mamba3-benchmarks.md` Deterministic GEMM, lines 181-195 | Uses the old environment flags and implies all inference paths share the dispatcher. | Replace with canonical modes only after high-level bypasses are closed; otherwise state the exact context-carrying scope. Link to final kernel receipts as component measurements, not Mamba-3 throughput. |
| `docs/performance-playbook.md` examples and process rules | Historical command examples use legacy environment flags. | Keep reproducible historical commands verbatim when they describe old evidence, but add a 0.7 note mapping them to `MAMBA_RS_GEMM_MODE`. Preserve the rule that admission is device/toolkit/dtype/op/cell-specific. |
| `docs/release-qualification.md` | Correctly limits CI/GPU claims but does not name the new public modes or the final dual-GPU receipt set. | Add a small 0.7 release-evidence pointer and require mode, comparator, dtype/output dtype, device, driver, toolkit, source hash, eager/graph path and aggregation method in every performance table. |
| `CHANGELOG.md` top | No 0.7.0 section; current top is 0.6.9 and contains now-stale 18-cell wording. | Add a 0.7.0 section covering canonical modes/default, deprecated compatibility setters/envs, `gemm_bi_inference` rename, retained-route assembly, graph validation and scoped dual-GPU results. Leave older release history intact; correct the 0.6.9 18-cell wording only if that section describes unreleased/current behavior rather than historical 0.6.9 behavior. Explicitly say no whole-model or all-GPU speedup is claimed. |
| `examples/gpu_inference.rs` module docs/comments | Says batch-invariant GEMM without explaining it is now the default and claims `~30%` dtype / `~2-5×` graph speedups without a cited current receipt. | State the default mode, show explicit mode choice only through an API the backbone exposes, and remove or scope the speed comments to their original benchmark. The final AUTO receipts do not substantiate them. |
| `examples/mamba3_gpu_inference.rs` module docs/comments | No mode example; unscoped `~1.6x` graph claim. | Add canonical-mode guidance after high-level mode plumbing is complete; remove or source the speed claim. Do not imply the positive graph tests are throughput evidence—they establish successful capture/replay and route-drift rejection only. |
| `examples/gpu_training_bf16.rs` and `examples/mamba3_gpu_training_bf16.rs` prologues | Describe batch-invariant behavior/graph speed but do not name the default `GemmMode`; likely retain old conceptual tier wording. | State `Deterministic` default, explicitly opt into a vendor mode only where demonstrated, and scope graph timing to what the example itself measures. Avoid promising model-wide no-cuBLAS until all helpers carry context. |
| Other GPU examples (`train_and_infer.rs`, CUDA portions of training/inference examples) | No discoverable canonical-mode example in the bounded search. | Add one concise linked example rather than duplicating mode prose everywhere; each CUDA example should at least state which mode its constructor uses. CPU-only examples need no GEMM-mode material. |
| `src/mamba_ssm/gpu/gemm_mode.rs` and `context.rs` public Rustdoc | Canonical foundation is substantially present, including constructor/setter examples and deprecations. Checklist reports one broken public-to-private link elsewhere. | Treat these as the terminology source. Complete argument/error/graph-transition documentation and fix the known private-link warning at `gemm_bi_triad/contract.rs:2834`; run the documented Rustdoc/doctest lane later. |
| `src/mamba_ssm/gpu/blas.rs` public-facing docs/comments | Module heading still presents the file only as cuBLAS wrappers; lines 359/428 call cuBLAS “non-deterministic”; lines 3458/3487 retain “fixed-tile” terminology. | Describe the dispatch boundary and canonical modes precisely. Rename user-facing family prose to Inference. Preserve `Fixed` only where it is an internal ABI/type/artifact identity. |
| `src/mamba_ssm/gpu/gemm_bi_inference.rs` module/public item docs | Top-level rename is correct, but some public comments still say “former”/“Fixed” ABI (`:437-438`, `:1895-1928`). | Use Inference terminology for user-facing concepts; explicitly mark any retained `Fixed` identifier as frozen internal identity rather than presenting it as a selectable family. |
| Crate/docs.rs landing surface (`src/lib.rs`, `src/mamba_ssm/gpu/mod.rs`) | `GemmMode` is re-exported, but the bounded inventory found no concise landing-page mode matrix/link. | Add a short canonical three-mode summary and link to `GpuCtx::{new_with_mode,set_gemm_mode}`. Avoid duplicating benchmark claims in Rustdoc. |

## Final measurement tables available for documentation

These are descriptive medians. Speedup is cuBLAS/AUTO; above 1 favors AUTO.
They may be reproduced in user docs only with the exact device, CUDA 13.2,
measured source SHA, eager/whole-graph path, comparator and “kernel/graph event,
not whole model” qualification.

### Inference AUTO, five shapes × bias off/on

| Input → output / comparator | Ada eager / graph | RTX 5090 eager / graph |
|---|---:|---:|
| BF16→BF16 / Fast 32F | 1.187× / 1.185× | 1.243× / 1.235× |
| F16→F16 / Fast 32F | 1.167× / 1.165× | 1.234× / 1.238× |
| BF16→F32 / Fast 32F | 0.828× / 0.825× | 1.288× / 1.276× |
| F16→F32 / Fast 32F | 0.815× / 0.812× | 1.278× / 1.270× |
| deterministic TF32→F32 / Fast TF32 | 0.900× / 0.900× | 1.132× / 1.126× |
| exact F32→F32 / Fast TF32 | 0.462× / 0.459× | 0.772× / 0.776× |
| exact F32→F32 / Pedantic F32 | 1.004× / 1.002× | 1.083× / 1.074× |

### Triad AUTO, NN/TN/NT cells

| AUTO precision / comparator | Ada eager / graph | RTX 5090 eager / graph |
|---|---:|---:|
| BF16 / Fast 32F | 0.890× / 0.869× | 0.689× / 0.929× |
| F16 / Fast 32F | 0.881× / 0.851× | 0.690× / 0.929× |
| deterministic TF32 / Fast TF32 | 0.718× / 0.709× | 1.051× / 0.939× |
| exact F32 / Fast TF32 | 0.422× / 0.422× | 0.793× / 0.698× |
| exact F32 / Pedantic F32 | 0.775× / 0.772× | 1.019× / 0.919× |

Required adjacent caveats: different numeric contracts must not be conflated;
win counts/aggregates are descriptive rather than admission tests; Ada and RTX
5090 results remain separate; TF32 RTX-5090 eager aggregate does not imply its
graph aggregate wins; neither Triad table supports a whole-Triad Fast win.

## Omissions that require new data or implementation closure

1. No unchanged-monolithic-`main` versus final 0.7 end-to-end measurement
   exists. Do not publish a release-wide speedup until that checklist item has
   one immutable harness and identical settings.
2. Final receipts cover RTX 6000 Ada and RTX 5090 on CUDA 13.2. Lower-toolkit
   route/binding qualification is separate; it is not a fresh full performance
   table. No all-GPU performance claim is available.
3. The positive Mamba/Mamba-3 graph tests prove four deterministic decode cases
   replay and reject route drift. They do not prove whole-model eager/graph bit
   equivalence or throughput.
4. The checklist still identifies direct cuBLAS LM-head/Mamba-3 projection
   seams. Until closed and tested, “Deterministic means no cuBLAS for the whole
   model” is not a shippable high-level claim, even though the low-level GEMM
   boundary rejects vendor dispatch in that mode.
5. CUDA Graph speedup comments in examples and older whole-model tables need
   their original source/device/toolkit or new measurements; the final AUTO
   event receipts cannot be substituted.
6. SM80/86/90a/SM100 and CC12.1 have portable/guarded code coverage of varying
   strength, not dual-GPU measured performance qualification. Preserve this
   distinction in support matrices and release prose.
