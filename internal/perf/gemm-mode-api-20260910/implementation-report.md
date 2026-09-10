# Task 901 implementation report

## Scope

Implement the canonical three-mode GPU context policy described by
`task-901-brief.md` and `gemm-mode-api-design.md`. This slice owns the public
mode API, context-local mode/env resolution, reversible cuBLAS math changes,
context-aware vendor compute selection, the unusable-context guard, policy
identity revision, and direct production migrations in the named GPU files. It
does not propagate modes through model constructors or close the separately
tracked no-context cuBLAS call sites.

## TDD checkpoint

The frozen RED integration test is `tests/gemm_mode_api.rs`, SHA-256
`0124f97cb89b30d321d46a61f89d2cf7df89b1068e8a4775621166eb2bf02599`.
Root observed the expected missing-public-API compile failure on Ada before
GREEN production edits were authorized.

## Implementation plan

- [x] Add the public `GemmMode` type and private pure environment/adapter/math
  transition logic, with unit tests for strict parsing, legacy resolution, all
  adapter transitions, and failure-atomic rollback.
- [x] Replace the three independently writable mode cells in `GpuCtx` with one
  canonical cell; add explicit/default/environment constructors, validation,
  reversible handle mutation, and plain technical Rustdoc.
- [x] Route context-aware GemmEx compute through the mode, preserve no-context
  dtype-only probes, and apply one central unusable-context check to supported
  GEMM recording/capture/replay boundaries.
- [x] Version the vendor numeric contract and dispatch-policy identity without
  changing custom kernel/compiler/tuning/schedule identities.
- [x] Migrate active production uses of deprecated mode adapters whose intent is
  unambiguous; leave benchmark/test migrations to the explicitly separate
  follow-on work except for the frozen compatibility regression.
- [x] Parse/format-check the touched files without invoking a build, record source hashes,
  and hand the frozen source to root for CUDA compilation, tests, Rustdoc, GPU
  probes, review, indexing, and commit.

## Implementation and evidence

### Public mode and context authority

- Added the public, default-Deterministic `GemmMode` enum with the three exact
  canonical spellings, re-exported from both `gpu::context` and `gpu`.
- Replaced the independent batch-invariant, fast-compute, and vendor-TF32 cells
  with one `Cell<GemmMode>`. Compatibility getters and `GemmPolicy` fields are
  derived from it. Family, tensor-core permission, f32 policy, and half policy
  remain separate and survive mode round trips.
- Added `new_with_mode`, `new_with_state_cap_and_mode`, `set_gemm_mode`, and
  `gemm_mode`. Ordinary constructors are deterministic and nonambient. Fresh
  custom defaults are Triad, tensor-core permission true, exact scalar f32,
  and tiled half parity.
- Added the crate-visible central guard with the downstream routing name
  `GpuCtx::ensure_gemm_usable(&self) -> Result<(), String>`.

### Environment and transition behavior

- Canonical and legacy environment values are captured once, resolved with
  presence-sensitive conflict rules, and validated before an environment
  constructor starts GPU setup. Non-Unicode canonical values fail. Vendor
  modes reject deterministic controls by presence. The retained `fixed` family
  alias still maps to Inference. Tensor-core permission no longer selects
  stream-K implicitly.
- cuBLAS math changes query the actual handle, update it once, verify or restore
  after a failed foreign update, and publish the mode cell only after success.
  An unverified rollback stores a stable unusable-context diagnostic. A usable
  same-mode request returns before capture/recorder validation or handle work.
- Different-mode changes inspect both the live CUDA stream-capture state and
  route-recorder state before mutation. Completed captures do not permanently
  invalidate the context; the existing route comparison remains authoritative.
- Deprecated setters retain their signatures and implement the approved
  adapter table through the canonical fallible setter. The intentional legacy
  API regression has one narrow `#[expect(deprecated)]`.

### Dispatch, identity, and production migration

- Context-aware GemmEx maps every F32/F16/BF16 operation to COMPUTE_32F in
  CublasFast and COMPUTE_32F_PEDANTIC in CublasPedantic. Deterministic reaching
  those vendor boundaries returns an error. Standalone no-context helpers keep
  their dtype-only compute behavior.
- Supported context-aware BLAS entries, route-recording starts, shared graph
  capture, captured route replay, and physical replay now consume the central
  unusable-context guard.
- Retained `CUBLAS_POLICY_V1`; added `CUBLAS_POLICY_V2` at bit 9 for active
  vendor routes. Bumped `POLICY_REVISION` to 6 and the dispatch digest domain to
  v5, with a framed `vendor-policy-revision=2` field. Added a regression that
  reconstructs the prior v4 framing and proves the current digest differs.
- Migrated both production qualification policy leases to the exact canonical
  mode. Their setup and drop ordering preserves dormant custom policy, restores
  Fast versus Pedantic exactly, and releases the active token on setup failure.

### Rustdoc examples

The three `no_run` examples are in `src/mamba_ssm/gpu/context.rs`:

- line 625: deterministic `GpuCtx::new` and `gemm_mode` result;
- line 645: explicit `GpuCtx::new_with_mode(CublasPedantic)`;
- line 1077: fallible Fast-to-Deterministic `set_gemm_mode` changes.

The surrounding Rustdoc describes defaults, arguments/results, environment
precedence and conflicts, capture/recorder restrictions, failure handling,
custom-versus-vendor TF32, and dormant policy preservation.

### Focused test inventory for root

Pure transaction unit-test filters:

- `mamba_ssm::gpu::gemm_mode::tests::math_transition_publishes_only_after_successful_update`
- `mamba_ssm::gpu::gemm_mode::tests::math_transition_stops_on_initial_query_failure`
- `mamba_ssm::gpu::gemm_mode::tests::failed_math_update_with_unchanged_handle_is_recoverable`
- `mamba_ssm::gpu::gemm_mode::tests::failed_math_update_restores_a_changed_handle`
- `mamba_ssm::gpu::gemm_mode::tests::unverified_math_rollback_marks_the_transition_unusable`

Pure mode/environment/identity filters:

- `mamba_ssm::gpu::gemm_mode::tests::legacy_adapter_table_is_complete`
- `mamba_ssm::gpu::gemm_mode::tests::vendor_compute_mapping_is_literal_for_all_modes`
- `mamba_ssm::gpu::context::tests::gemm_mode_environment_resolves_canonical_and_legacy_tables`
- `mamba_ssm::gpu::context::tests::gemm_mode_environment_rejects_conflicts_and_vendor_custom_controls`
- `mamba_ssm::gpu::context::tests::gemm_mode_environment_rejects_non_unicode_canonical_value`
- `mamba_ssm::gpu::context::tests::deterministic_environment_defaults_and_validates_custom_policy`
- integration target `kernel_identity`, filter
  `vendor_policy_v2_changes_the_dispatch_identity`
- integration target `kernel_identity`, filter
  `backend_contract_sets_match_reachable_dispatch_trees`
- integration target `kernel_identity`, filter
  `half_policy_opens_the_stream_k_contract_only_inside_the_tensor_core_tier`

Public non-device integration filters in target `gemm_mode_api`:

- `gemm_mode_names_and_default`
- `gpu_module_reexports_the_canonical_gemm_mode`

Ignored live context/handle tests in target `gemm_mode_api`:

- `gpu_context_constructors_use_explicit_modes_and_deterministic_defaults`
- `gpu_context_supports_all_nine_canonical_mode_transitions`
- `legacy_mode_adapters_preserve_the_documented_call_order`
- `vendor_mode_round_trip_preserves_custom_deterministic_policy`

New ignored capture/compute live tests:

- target `gemm_mode_live`: `gpu_mode_change_rejects_capture_but_allows_same_mode_noop`
- target `gemm_mode_live`: `gpu_mode_change_is_rejected_during_route_recording`
- lib filter:
  `mamba_ssm::gpu::blas::physical_graph_tests::context_aware_vendor_compute_maps_all_dtypes`

### Local checks and root-owned evidence

No local Cargo, CUDA compilation, GPU execution, doctest, or Rustdoc command was
run. Direct `rustfmt --emit stdout` parsed every touched Rust file without
writing it. Direct `rustfmt --check` passed for the three new/frozen test/source
files, and `git diff --check` passed. Root owns all requested build, Ada, doc,
listing, warning, snapshot, index, and commit evidence.

Root's RED evidence: Ada `cargo test --locked --release --features cuda --test
gemm_mode_api --no-run` exited 101 only for the absent public mode API (14
E0432/E0425/E0599 diagnostics); source-after integrity passed. Root also
reported the pre-GREEN CUDA Rustdoc baseline passed with one pre-existing
private-link warning at `contract.rs:2834`.

Root's initial GREEN all-targets compile reached `tests/gemm_mode_live.rs` and
stopped on E0277 because `Result::expect_err` required `CudaGraph: Debug`. The
test now uses the repository's explicit `match` pattern; no production source
changed in that correction. The preserved log is
`internal/perf/gemm-mode-api-20260910/green-initial/all-targets-check.log`.

Review then identified a qualification-guard error-path risk. The pure
tests-only checkpoint adds
`qualification_policy_guard_preserves_mode_setup_error`,
which requires a failed guard entry to return its original error, attempt the
mode transition once, and restore every custom field. Its containing source
hash before the production repair is
`97f69936e4c8ef08f3965aa87d57163f1a13ba588d2cd1a22aaef8213542497a`.
Root observed the exact focused test fail with a two-versus-one transition call
count, exit 101, and source-after integrity intact. The production repair now
captures the old fields as plain locals and constructs the Drop guard only
after the fallible mode transition succeeds. The failure branch restores the
custom fields and returns the original error without arming Drop.

### Frozen source hashes

```text
079dd42be168b0134399aa3f2f23a4521d91dd17e29347669c68d0658376d009  src/mamba_ssm/gpu/gemm_mode.rs
e6cf8297bf741738f003e0717196395eedad5af2e4b9efabeb529e7915a72bb4  src/mamba_ssm/gpu/context.rs
10c99488aa8b96e0b6f8ec4eef81de270a7b841bee640a0e9db5ffa96a785b34  src/mamba_ssm/gpu/device.rs
2e4976eb27d2b166abc1fe897cec975127b48a2f7ca261bbd2c587fb769c8671  src/mamba_ssm/gpu/blas.rs
ebe64487d1a44c3e94b268b130c99da454a3c2b09cb4dc2108f9b22cb4de2a72  src/mamba_ssm/gpu/kernel_identity.rs
6fd6b62144c7046623a69e9665cf4f3a8449eca386767694d127295d370d9b5e  src/mamba_ssm/gpu/mod.rs
b609dbdae1f543043d1523586a880efb99e4b2d25d168e5eef50546520e91b3d  src/mamba_ssm/gpu/graph_capture.rs
e0103ff13e4c48c7449c74b68d24c0ebee19e1e81fad2dc7d5a6c292ee5cdcf3  src/mamba_ssm/gpu/gemm_bi_triad/qualification.rs
0124f97cb89b30d321d46a61f89d2cf7df89b1068e8a4775621166eb2bf02599  tests/gemm_mode_api.rs
cdb4e2e9f25152bb89e388e382a469452d9595797ef778c10931ae9ed2c8121b  tests/gemm_mode_live.rs
9d645e1777707234e31a763d087d85352ccb2fe78073e9f1529e34d2f3eca781  tests/kernel_identity.rs
```

### Explicitly deferred release work

This slice does not claim model-wide deterministic no-cuBLAS closure. The
brief's next dependencies remain: high-level M1/M3 inference and trainer mode
constructors, context-routed borrowed F32/raw-pointer GEMMs, typed tied-head
F32-output composition, direct M3 projection routing and physical manifests,
benchmark/test baseline migration, full release documentation, and packaging.
The worktree also contains root-owned changes to the release checklist and
handoff task list; they are not part of the source hash set above.

## Final root verification

Root's `green-fix1` Ada packet completed with runner and source/fixture
integrity checks passing. Results were:

- CUDA release all-targets check: pass;
- six public API/default/constructor/transition/adapter tests: pass;
- two live capture/recording rejection tests: pass;
- pure mode, environment, and transaction tests: pass;
- real-context vendor compute mapping for F32/F16/BF16: pass;
- all 24 kernel-identity tests: pass;
- CUDA Rustdoc with broken intra-doc links denied: pass;
- CUDA-feature doctests: 14 pass, 3 existing ignored;
- non-CUDA library suite: 84 pass.

The separate `lease-red` packet observed the qualification regression fail on
two mode-set calls instead of one. The `lease-green` packet passed the unchanged
regression and both qualification guard tests, with runner/source integrity
passing, and the final-source CUDA all-targets check passed. Root's consolidated
evidence is `internal/perf/gemm-mode-api-20260910/report.md`.

Root committed the verified source as
`0dcfe167cf10c0ab68c9630b4b0191fa7bffbfd9` (`Add canonical deterministic and
cuBLAS GEMM modes`). Independent task review remains root-owned.
