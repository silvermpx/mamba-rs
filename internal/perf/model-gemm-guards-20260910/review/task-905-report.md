# Task 905 — Model GEMM manifests and guarded replay

Status: compile-ready source frozen; root-owned compilation and GREEN runtime
verification pending. No commit created by implementer. CUDA source, selectors,
compiler/admission contracts, public constructors/defaults and trainer APIs are
unchanged. Fresh 5090 execution remains pending; no 5090 receipts were changed.

## Implementation

All six M1 and four M3 decode entries now record their existing private body
exactly once on eager execution and use the shared validated launch seam on
replay. Input uploads and output downloads remain outside the body recorder.
The corresponding eager permit is cleared before a new eager upload/validation
attempt, and published after the body succeeds. Architecture-rung preparation
runs before opening the recorder and only when the path has nonzero GEMM work.

All five capture methods take/consume their matching eager permit before
presizing or capture, use `capture_into_graph_with_gemm_plan`, require the
deterministic plan for actual nonzero GEMM work, and install graph/plan only
after successful validation. The F32/legacy work predicate is
`batch != 0 && (!identity_proj || n_layers != 0)`; native uses
`batch != 0 && n_layers != 0`.

M1 legacy mixed and native mixed retain independent eager permit cells. Their
single graph and plan slot has a private `MixedGraphPath::{Legacy,Native}` tag.
Both replay entries per path reject a mismatching tag before scratch-pointer
assertions, and the shared launcher rechecks the tag. Existing logical route,
state/scratch pointer, staging pointer, presizing/freeze, context resource
ownership and drop-time stream synchronization/graph destruction are retained.

`with_validated_gemm_graph_launch` checks context health, deterministic plan
presence, and the existing plan's complete ordered bindings before invoking its
callback. The vendor None-plan branch still checks health. A tiny crate-private
`cfg(test)` context poison setter enables actual health rejection tests without
a public testing API.

The old `require_f32_triad_graph_plan` remains, with its original behavior,
solely for five out-of-scope prefill/trainer calls (M1 prefill/training, M3
prefill/pooled-prefill/training). Deleting it would break their imports, while
redirecting their logical-F32 booleans to the new has-work contract would change
their behavior. No Task905 decode entry uses it.

Root explicitly confirmed this boundary: Task905 proves decode and heads, not
all prefill/training acceptance. Root then added the affected existing host
contract file as a ninth scoped file. Its stale M3/legacy exclusions are replaced
by the actual five decode capture/ten entry map; unchanged prefill/training
assertions remain. The graph-launch scanner admits exactly the old and new
guard names and checks zero-argument launch calls through local receivers, with
positive, bypass, and prefixed-impostor fixtures.

`blas.rs::vendor_gemm_test` is entirely `cfg(test)`: one thread-local scoped
count/deny guard, a pre-FFI boundary function, explicit nested-guard rejection
without changing outer state, and Drop cleanup after error/unwind. A non-Send
marker keeps the owning guard on its original thread. All nine actual vendor
result calls are hooked immediately before FFI: four SGEMM and five GemmEx,
including standalone no-context compatibility functions. Handle creation is
not counted. No environment switch, production hook, or global mutable count.

Touched capture documentation now states the consumed eager prerequisite,
fixed buffers, mode/family restrictions, error conditions and GEMM-only scope.
Head fixtures preserve the existing tied-half arithmetic and scratch reservation.

## TDD evidence

RED source consisted only of two added colocated real tests; no production
extraction or fake context was necessary. Root reported CUDA+HF build success
(54.61 s) and observed both intended actual Ada failures with unchanged frozen
source/build manifests:

- `mamba3_siso::gpu::inference::model_gemm_manifest_tests::m3_capture_requires_successful_public_eager_manifest`
  returned `Ok(())` from actual capture despite only private-body warming and no
  successful public eager step. Exit 101; 1 failed, 0 ignored; 4.71 s.
- `mamba_ssm::gpu::graph_capture::model_gemm_guard_tests::deterministic_inference_half_work_rejects_empty_graph_plan`
  returned `Ok(())` from the old empty-plan guard. Its actual BF16 Inference
  GEMM output assertion `[32.0; 16]` passed first. Exit 101; 1 failed,
  0 ignored; 4.48 s.

Root's invocation contract is the CUDA+HF library test runner with each exact
name and `--ignored --exact`; the implementer did not run Cargo, CUDA or GPU
commands. Exact root shell commands and durable runtime receipt locations are
owned by root and should be attached to this report when available.

GREEN: pending root-owned build and focused runtime batch. No runtime success
is claimed in this source-freeze report.

## Exact focused test groups

All following CUDA library tests are ignored by default and require
`--ignored --exact` with the full path. Head groups additionally require `hf`.

`mamba_ssm::gpu::inference::model_gemm_manifest_tests::`

- `m1_model_manifests_replay_all_paths_without_vendor_gemm`
- `m1_failed_steps_clear_permits_and_mixed_graphs_reject_wrong_path`

`mamba3_siso::gpu::inference::model_gemm_manifest_tests::`

- `m3_capture_requires_successful_public_eager_manifest`
- `m3_model_manifests_replay_all_paths_without_vendor_gemm`
- `m3_failed_steps_clear_permits_and_failed_capture_keeps_installed_plan`
- `m3_explicit_vendor_graph_without_custom_plan_replays`

`mamba_ssm::gpu::graph_capture::model_gemm_guard_tests::`

- `deterministic_inference_half_work_rejects_empty_graph_plan`
- `model_graph_guard_zero_vendor_and_poison_controls`

`mamba_ssm::gpu::blas::vendor_gemm_test::`

- `vendor_gemm_counter_observes_both_modes_and_denies_before_ffi`
- `vendor_gemm_guard_restores_on_error_unwind_and_rejects_nesting`
  (host-only test, not ignored; still compiled under the library's CUDA feature)

`module::gpu_lm::tied_head_capture_tests::`

- `m1_tied_and_untied_heads_all_storage_without_vendor_gemm`
- `tied_bf16_m1_capture_reserves_head_scratch_before_freeze` (existing, fixture reused)

`module::gpu_lm3::tied_head_capture_tests::`

- `m3_tied_and_untied_heads_all_storage_without_vendor_gemm`
- `tied_f16_m3_capture_reserves_head_scratch_before_freeze` (existing, fixture reused)

Integration binary `inference_graph_route`:

- `decode_graphs_reject_complete_route_drift` (existing Triad control)
- `inference_decode_graphs_reject_complete_route_drift` (same owned public-path control, Inference)

Host integration binary `gemm_bi_tf32_contract`:

- `exact_graph_inventory_wires_decode_and_existing_prefill_training_holders`
  (renamed/migrated obsolete Triad-only holder contract)
- `graph_launch_scanner_rejects_a_bypass_beside_a_guarded_call`

The core matrices have 30 M1 cases and 18 M3 cases: B1/B3, two layers,
Inference, Triad TC-off and Triad TC-on, F32/BF16/F16. M1 F32 and legacy mixed
exercise distinct-width nonidentity input projections; native mixed is identity
only. M3 F32 is nonidentity and native mixed is identity. Every case uses both
eager entries and both replay entries, resets recurrence on the same owned
inputs/buffers for bit comparisons, and executes under vendor deny. Independent
literal per-layer shape sequences generate expected projection groups. Direct
Inference expects exactly 9/8 M1 and 5/4 M3 routes (with/without input projection),
while scalar decompositions are validated as ordered projection groups with
actual physical routes. Triad TC-off half cases assert actual matvec symbols.

The shared actual-plan mutation check covers missing middle projection,
reordered projections, symbol, module, storage, argument digest and map digest,
plus live family and mode drift, always with a callback count that stays zero
after rejection. Other tests cover missing and failed-upload stale permits for
all ten entries, private warm body without public permit, first cold public
preparation, changed capture buffers, failed capture retaining an installed
plan, scratch growth and address checks, M1 path mismatch in both directions,
poisoned deterministic plans and poisoned vendor None plans, zero-work positive
guard controls, and actual explicit vendor graph capture/replay without a custom
plan. The 24 head cases cover both families, all three dtypes, tied/untied, eager
head execution and backbone graph replay followed by the same head, under deny.
The vendor positive control executes both modes and all three storage dtypes,
checks real results and one observed call, then verifies denial leaves poisoned
output unchanged before FFI.

## Source-only verification and self-review

- Scoped `rustfmt --edition 2024` on the ten changed Rust files: pass.
- `git diff --check`: pass.
- Vendor source census: nine result calls, each immediately preceded by the
  same cfg(test) boundary hook. Runtime controls remain the primary proof.
- No Cargo, GPU, network, staging or commit command executed by implementer.
- Root-owned internal handoff/checklist metadata changes were not edited.
- Existing state/resource anchors, recurrences, conversion arithmetic, kernel
  selectors and CUDA/compiler sources were left unchanged.
- No automatic edits after compile-ready freeze; waiting for root findings.

The obsolete source-text contract was migrated under root's explicit scope
extension while the original eight files stayed frozen. Runtime model and
vendor controls remain the acceptance evidence; the source census is supporting
evidence only. Both host groups await root execution.

## Frozen files (SHA256)

```text
100fbbb0239c73d6ee5de6819579ddf972840c4eeee1bf5b6d1ed530e2a9ad5f  src/mamba_ssm/gpu/graph_capture.rs
69c53b2482489423a017a28d534b98d8cde4a13e6757a20b444e20850de6310e  src/mamba_ssm/gpu/inference.rs
4017f1096db49382f0766ffef3fb2281138e3f8a33c313ac6f4b9418494c62d3  src/mamba3_siso/gpu/inference.rs
bcb5b3c393f4fca6e54a993e6c729c538d53ee64c96163f79e7073ee72f94729  src/mamba_ssm/gpu/blas.rs
0f32c23b99dc812700a1bed0979d9856374b8be1fd89548b7b873947cc9137b9  src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
9e6eaf005668b3896f45e638f7df6ae62340884b2962afb7a7250ccf45acd433  src/mamba_ssm/gpu/context.rs
f3297b971c9c0de2bc775d4b9a5d3acdee9f7bcca3bd4c1c96e3cb332bd7f28c  src/module/gpu_lm.rs
c273a8ee67405a07e016be275d000504aba4bc6adfcbd5e0d85e729366c1a879  src/module/gpu_lm3.rs
9d6a0777942a95e6dd5738b5619b26c9157fcc3c035a37dd78b9dae27421b570  tests/inference_graph_route.rs
e12519d450910daf8cf16beab09d8008e2a91b58e52c14ccbc85cde9044e15b9  tests/gemm_bi_tf32_contract.rs
```

## Host parser fix round 1

Root reported CUDA+HF library and both integration binaries built successfully
in 64 s. The first two host tests passed; the new holder-map test failed before
GPU execution because the old substring parser matched the earlier
`Mamba3GpuInferenceMixedScratch` declaration when asked for
`Mamba3GpuInferenceMixed`. Root receipt:
`/root/mamba-model-gemm-guards-final.7dt0nG/evidence-model-gemm-green/exact_graph_inventory_wires_decode_and_existing_prefill_training_holders.log`.

Only the ninth contract file changed. Its new `struct_scope_for_type` matches
the `struct` token and complete following type token and requires exactly one
declaration. The three owner-map loops use this parser. Added
`struct_scope_parser_distinguishes_exact_type_from_prefix_collision` exercises
Scratch/suffix/Other collisions, a masked comment, exact-body selection and
rejection when only a prefix match exists. This directly covers the observed
parser defect; production code remains unchanged.

Root should run the new parser regression, the corrected holder-map test, and
the graph-launch scanner negative fixtures. The implementer ran only scoped
rustfmt and `git diff --check`, both passing. New ninth-file SHA256 is reflected
above; the original eight hashes are unchanged. Root is running the 13 actual
GPU groups against the already-built unchanged eight-file binary separately.

## Native-half terminal fix round 2

Root's first actual GPU packet passed its first four tests, then the M1 matrix
failed at Triad TC-on B1 BF16 native. Its only observed projection group was
`[(1,64,18)]`, instead of the independently expected four groups repeated over
two layers. Root receipt:
`/root/mamba-model-gemm-guards-final.7dt0nG/evidence-model-gemm-gpu/mamba_ssm::gpu::inference::model_gemm_manifest_tests::m1_model_manifests_replay_all_paths_without_vendor_gemm.log`.
Root subsequently reported the matching M3 TC-on B1 BF16 native failure:
`captured no resolved GEMM route`. These actual failures are the RED evidence
for the terminal fix; model assertions were not relaxed or changed.

Systematic tracing found that homogeneous half TC-on projections with N>=32
reach `gemm_bi_forward_typed` -> `gemm_bi_forward_tc_observed` ->
`enqueue_half_gemm` with `NoPhysicalObserver`. That terminal resolved identities
only under `O::ENABLED` and never published a context route. The N=18 M1
x-projection instead reaches Task904's already-instrumented matvec terminal,
which explains the exact observed partial inventory.

Root explicitly extended scope to `gemm_bi_triad/launch.rs` and the targeted
existing `blas.rs` test. The existing `HalfLaunchEnvironment` now carries an
optional real context: observed context entries store Some(ctx), public
stream/kernels-only entries store None. The existing context-aware forced
`gemm_bi_forward_tc_with_tile_shape` was corrected to preserve its ctx as well.
All 16 terminal calls pass the binding (13 environment calls, three direct
SM89 calls). The old two-usize environment size assertion is retained and
truthfully updated to three usize.

The existing enqueue helper now conditionally resolves, validates and publishes
the actual context route before enqueue whenever its context recorder is active,
independently of physical observation. It reuses the existing exact symbol,
module, artifact, compiler, operation, shape/stride and launch metadata. A small
argument-identity closure in the existing digest machinery preserves physical
allocation digests byte-for-byte while the GEMM-only context record binds actual
pointer values and spans in its own versioned digest domain. It does not claim
allocation liveness. Physical `if O::ENABLED` behavior remains unchanged, and
there is no context route synthesis at the model/projection level.

The second helper argument is now `(kernels, Option<&GpuCtx>)`, retaining its
arity and config argument index 3. Consequently the existing exact topology
census needs no signature adjustment. No CUDA, numerical contract, selector,
admission, compiler or public API behavior was changed. Context-aware native
NN/TN/NT paths using this terminal gain the missing GEMM records, but this does
not establish complete training/prefill model acceptance.

Added exact CUDA test:
`mamba_ssm::gpu::blas::matvec_inventory_cuda_tests::triad_native_half_context_inventory_records_all_projection_terminals`
(`--ignored --exact`). Four literal M1 projection shapes, B1/B3 and BF16/F16
exercise the actual public typed path with NoPhysicalObserver and require four
ordered routes plus correct numerical output. The first projection additionally
compares physical observer nodes with and without an active context recorder,
requires unchanged allocation digests, one context route without duplication,
and rejection before enqueue when the context recorder has zero capacity
(output sentinel remains unchanged). The whole fixture runs under vendor deny.

Fix-round source-only checks: scoped rustfmt and `git diff --check` pass.
`launch.rs` diff is 74 additions/33 deletions. `blas.rs` receives only the new
targeted test in this round; model and head test files remain frozen. Updated
hashes appear above. Root owns the targeted GREEN build/runtime and the affected
physical observation topology/ownership host tests; no Cargo/GPU commands were
run by the implementer. No further source edits after this fix freeze.

## Host provenance census fix round 3

Root's half-fix CUDA+HF build passed in 57.15 seconds; CUDA-only also passed.
Six affected owner/parser/topology host tests passed. The remaining provenance
host test then failed on unchanged kernel_identity.rs with uncovered counts
ObserverConstructor=1 and ObservationResolver=1. Source tracing identified the
existing cfg(test) inference_test_support::observer helper and the existing
Inference argument resolution inside PhysicalLaunchObservation::resolve.
The same old exact census predated the reviewed BLAS matvec terminals, Inference
terminals/test fixtures, and pre-RNA observation helpers and payloads.

Root authorized a ninth-file-only exact census repair. No whole module was
authorized: every production function and test function has its own count;
test helpers retain exact adjacent cfg(test) checks. The repaired census includes
the two BLAS terminal submitters, three BLAS test helpers, all 17 existing
Inference submitters, its observed inventory run_case helper, the two pre-RNA
helpers, and the owner-private resolver/test constructor. With further explicit
root approval, the same classifier now also recognizes the existing Inference
and input-transform semantic constructors, including aliases: one Inference
production constructor and five test functions (eight literal constructor and
resolver calls), and one transform constructor in each pre-RNA helper. Exact
opaque payload and constructor assertions now include those existing fields.

Existing provenance/forwarder/cfg/count negative fixtures remain. Added the
exact test physical_904_census_keeps_test_authorities_gated_private_and_exact:
missing test gates, widened module/factory visibility, a duplicate factory,
widened resolver visibility, duplicate argument resolution, widened transform
payload, and an incorrect Inference payload must reject. The existing provenance
test also gains direct and aliased unauthorized-use fixtures for both newly
recognized constructor names.

Root separately found the colocated HalfLaunchEnvironment runtime host assertion
still expected two pointers although its compile-time assertion had correctly
changed to three. Only that cfg(test) expectation changed, preserving the
NoPhysicalObserver ZST assertion. No production source changed in this round.

Root reports all five half-fix GPU groups PASS: direct half projection regression
4.74 seconds, full M1 matrix 128.19 seconds, full M3 matrix 81.22 seconds,
Triad route-drift 18.83 seconds, and the six-case physical-half fixture (elapsed
not supplied). These are actual Ada results from root, not locally executed
checks. No 5090 runtime claim or saved receipt change is made.

Final host-only frozen tests for root:
- physical_trace_provenance_has_no_crate_visible_mint_or_src_bypass
- physical_owner_rejects_forwarders_and_visible_function_pointers
- physical_904_census_keeps_test_authorities_gated_private_and_exact
- mamba_ssm::gpu::gemm_bi_triad::launch::half_physical_trace_tests::no_physical_observer_is_zero_sized_and_never_records

Scoped rustfmt and git diff --check passed. Latest two hashes appear above;
all other source hashes remain unchanged. Final host runtime and Rustdoc
acceptance remain root-owned and pending at this freeze. No further source
changes without root findings.

## Host trace-literal classifier fix round 4

Root's final host packet compiled successfully, and Rustdoc passed in 2.52
seconds. Five of eight host tests passed; three reached the same stale blanket
check: `mask.contains("RecordedPhysicalTrace {")` misclassified the existing
BLAS private test function's return annotation as construction of a trace literal.
Root located the exact false positive at record_tied_half_f32_trace and
authorized only the ninth-file classifier correction.

The classifier now uses source tokens and the existing function-body parser.
It distinguishes the opening brace of a returning function from an actual trace
literal, including a literal inside that very function. The existing new census
regression now includes unqualified and qualified return-signature positives and
forged literal negatives with ordinary, newline, qualified and comment-separated
spelling. No module exception was added. All other ownership checks remain.

Latest ninth-file hash is recorded above. Scoped rustfmt and git diff --check
pass; all production/GPU files are unchanged. Root reruns only these affected
host tests, with no further GPU or Rustdoc run required:
- physical_trace_provenance_has_no_crate_visible_mint_or_src_bypass
- physical_904_census_keeps_test_authorities_gated_private_and_exact
- physical_owner_rejects_private_method_authorities

Root packet: /root/mamba-model-gemm-final-host.4izdkP/evidence-final-host.
Final host acceptance remains pending at this freeze.

## Independent review fix round 1 — test-only RED freeze

Review against accepted base 96640d62 found one Important permit-lifecycle
exception: entries cleared eager permits only when no graph was installed.
M1 mixed can retain the opposite path's permit after capture. A wrong-path
step or invalid upload therefore left stale admission for that path's recapture.
The old regression incorrectly expected that recapture to succeed after failure.

Following review reception and TDD skills, only the existing M1 CUDA regression
was changed before requesting root's actual RED run. Exact test:
mamba_ssm::gpu::inference::model_gemm_manifest_tests::m1_failed_steps_clear_permits_and_mixed_graphs_reject_wrong_path.
Eight owned real fixtures cover both attempted paths, both entry forms, and
wrong-path versus invalid-upload failure with an opposite-path graph installed.
Each requires the relevant permit to be absent and recapture to return the eager
requirement error; the installed graph, path, routes and scratch binding must
survive unchanged and remain replayable. The independent positive recapture
control now has no intervening step attempt. Existing cold-upload, scratch
growth/address, pointer-manifest mismatch and poison controls remain.

Only cfg(test) changes are frozen in src/mamba_ssm/gpu/inference.rs:
4d23c182b1ca59205db20252e75409827705747c720b7f17bfaf281db7ba3037.
Expected first RED assertion: failed legacy attempt retained its eager permit
with native graph installed. Scoped rustfmt and git diff --check pass. No
production source change, Cargo/GPU run or Git mutation was made. Root's RED
observation and production authorization remain pending.

### Review round 1 — intended RED observed, minimal GREEN frozen

Root's CUDA+HF library build passed in 59.05 seconds. The expanded actual Ada
test exited 101 with one failed and zero ignored in 9.15 seconds, at its intended
first Legacy/non-GPU/wrong-path assertion: failed legacy attempt retained its
eager permit with native graph installed. Runner exit and frozen source checks
were clean. Root then authorized the production fix.

Exactly ten production statements now clear the matching eager permit
unconditionally before input upload: six M1 entries and four M3 entries.
Each change removes only the surrounding graph.is_none condition; capture,
installed plan, path association, scratch/resource bindings and launch handling
are untouched. The RED test body is unchanged. The existing exact source owner
assertion accepts unconditional clear-before-upload, so no contract edit was
needed. No arithmetic, dispatch, CUDA or graph framework changes were made.

Final two-file hashes appear in the source table above. Scoped rustfmt and
git diff --check pass; no Cargo/GPU command or commit was run by the implementer.
Root-owned focused GREEN acceptance is pending for:
- mamba_ssm::gpu::inference::model_gemm_manifest_tests::m1_failed_steps_clear_permits_and_mixed_graphs_reject_wrong_path
- mamba3_siso::gpu::inference::model_gemm_manifest_tests::m3_failed_steps_clear_permits_and_failed_capture_keeps_installed_plan
- exact_graph_inventory_wires_decode_and_existing_prefill_training_holders

Earlier full matrix acceptance remains unchanged; no full matrix rerun is
requested for this review fix. Source is frozen pending root's results and
scoped re-review.
