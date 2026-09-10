# HALF architecture/context-inventory audit

Snapshot: branch `codex/gemm-bi-triad-sm80`, HEAD `b7397d4fd53e`; source was inspected read-only while the Task 905 owner had uncommitted changes in `blas.rs` and `gemm_bi_triad/launch.rs`.

## Result

No additional Task-905-class missing context-route gap is present in the SM90a, SM100, or SM120 HALF automatic branches, nor in the exact-F32 fallback wrappers reached by the typed NN/TN/NT BLAS paths. Those routes already record through `GpuCtx` when the public entry point supplies `NoPhysicalObserver`, and enabling the physical observer does not add a second entry to the context recorder.

The portable/SM89 `enqueue_half_gemm` defect is therefore the bounded source fix. Do not expand it into the three specialized architecture dispatchers or the exact-F32 fallback machinery for release correctness.

## Exact typed call graph

All three public typed entry points create `NoPhysicalObserver` and then enter the observer-generic body:

- `src/mamba_ssm/gpu/blas.rs::gemm_bi_forward_typed` (`NoPhysicalObserver` at lines 762-763). Its accepted architecture calls are `launch_sm120_auto_observed` (811), `launch_sm100_auto_observed` (834), and `launch_sm90a_auto_observed` (856); its exact-F32 terminal is `record_physical_exact_scalar_f32_forward` (899).
- `src/mamba_ssm/gpu/blas.rs::gemm_bi_backward_dw_typed` (932-933). The corresponding TN calls are at 971, 994, 1016, and the exact-F32 terminal is at 1057.
- `src/mamba_ssm/gpu/blas.rs::gemm_bi_backward_dx_typed` (1086-1087). The corresponding NT calls are at 1125, 1148, 1170, and the exact-F32 terminal is at 1203.

An architecture auto function returning `Ok(None)` has not enqueued a kernel and correctly contributes no context route. Every accepted branch returns immediately, so only one of SM120, SM100, SM90a, SM89, portable TC/scalar HALF, or exact-F32 fallback owns a given typed operation.

## Specialized HALF branches

The three accepted specialized routes all follow the same topology in `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs`:

| Function | Physical enqueue | Context route | No-observer outcome | Both-recorders outcome |
|---|---:|---:|---|---|
| `launch_sm100_auto_observed` (1095-1138) | `enqueue_sm100_tcgen_prepared_observed` (1123-1135) | `ctx.record_resolved_gemm_route(resolved)` (1136) | Physical observation is ignored because `O::ENABLED == false`; the context route is still appended. | One physical node is recorded by `enqueue_with_physical_observation`, then one route is appended to the distinct context recorder. |
| `launch_sm90a_auto_observed` (1460-1508) | `enqueue_sm90a_wgmma_prepared_observed` (1493-1505) | direct context record (1506) | Same. | Same; no context-side duplicate. |
| `launch_sm120_auto_observed` (8182-8245) | `enqueue_sm120_tma_prepared_observed` (8230-8242) | direct context record (8243) | Same. | Same; no context-side duplicate. |

`NoPhysicalObserver::ENABLED` is `false` in `src/mamba_ssm/gpu/kernel_identity.rs` (3282-3287), and `enqueue_with_physical_observation` only resolves/records a physical node inside `if O::ENABLED` (3700-3729). It never writes the `GpuCtx` route recorder. Thus the direct `ctx.record_resolved_gemm_route` calls above are not duplicates of observer activity; the two recorders have different products (logical/pointer-bound GEMM routes versus allocation-bound physical CUDA nodes).

## Typed exact-F32 fallback

The relevant wrappers are:

- `record_physical_exact_scalar_f32_forward` (launch.rs 5958-6025)
- `record_physical_exact_scalar_f32_backward_dw` (6087-6149)
- `record_physical_exact_scalar_f32_backward_dx_with_arguments` (6249-6312), reached through `record_physical_exact_scalar_f32_backward_dx` (6234-6247)

Each wrapper selects `F32PreparedSelection::ExactScalar`. With `NoPhysicalObserver`, it calls `launch_cached_f32_triad`; with a recording observer, it calls `launch_cached_f32_triad_observed`. Both routes share the same context-recording control:

- Nonzero-reduction scalar kernels call `ScalarLaunchControl::bind`, whose sole context mutation is `self.ctx.record_resolved_gemm_route(*route)` (launch.rs 1614-1628), once for each actual scalar GEMM enqueue.
- The observed scalar controller calls that same `base.bind` once and then submits one physical observation for the same kernel; it does not call `record_resolved_gemm_route` again (`PhysicalScalarLaunchControl`, 1710-1744).
- A zero-reduction exact-scalar preparation becomes `PreparedF32Kind::ScalarZero`. The unobserved launcher records once at 7021 before its enqueue; the observed launcher records once at 7140 and separately supplies one physical observation at 7148-7152.

The typed fallback's input upcasts and NN/NT output downcast are real physical launches but not GEMM routes. Their absence from `RecordedGemmTrace` is intentional; a physical observer inventories them separately. TN has no output downcast. This does not leave a deterministic GEMM graph plan incomplete.

Although the generic prepared F32 launcher also has TF32, split-K, and SM89 pre-RNA arms, the typed HALF fallback fixes `ExactScalar`, so those arms are not reachable from the NN/TN/NT fallback calls above. They also have their own single context-record sites (directly or through `ScalarLaunchControl`) and reveal no adjacent same-class omission.

## Minimal change scope and residual risk

- Confirmed release-blocking change in this class: none beyond the Task 905 owner's portable/SM89 `enqueue_half_gemm` context propagation/recording fix.
- The specialized SM90a/SM100/SM120 functions append the context route **after** a successful driver enqueue. Consequently, a deliberately undersized fixed recorder reports capacity overflow after the CUDA enqueue, unlike the Task 905 portable helper's new pre-enqueue check. Correctly sized eager-manifest/capture flows still record and validate the route, so this is not a missing-route or existing deterministic-graph failure. Treat uniform fail-before-enqueue behavior as optional hardening/test portability, not release scope; simply moving the record earlier would trade this for stale context entries on driver failure unless rollback semantics were added.
- Prepared physical graph adapters (`prepare_sm120_auto_graph_sequence`, `prepare_sm100_auto_graph_sequence`, `prepare_sm90a_auto_graph_sequence`) construct physical packages rather than run the public context-inventory capture body. They should not gain context-recorder writes.

## Existing focused evidence

- `tests/gemm_bi_tf32_contract.rs::typed_fallback_records_scalar_routes_without_reading_f32_policy` statically seals the NN/TN/NT wrapper split between observed and unobserved cached exact-scalar launches.
- `tests/gemm_bi_tf32_contract.rs::physical_observation_uses_one_inseparable_enqueue_primitive` covers the specialized auto functions and observed F32 enqueue topology through the ownership oracle.
- `tests/gemm_bi_sm120_contract.rs::sm120_auto_bridge_is_request_based_and_capture_prepared_only` and `sm120_half_graph_package_uses_the_cached_prepared_route` cover the SM120 bridge/package boundary.
- `src/mamba_ssm/gpu/blas.rs::physical_graph_tests::physical_graph_captures_native_half_and_exact_typed_fallback` exercises physical HALF and exact-F32 graph packages for NN/TN/NT.
- The in-progress Task 905 fixture `src/mamba_ssm/gpu/blas.rs::matvec_inventory_cuda_tests::triad_native_half_context_inventory_records_all_projection_terminals` explicitly checks one context route with both recorders active and no physical-trace mutation. Its selected architecture is hardware/policy dependent, so it is direct coverage of the portable fix on the owner's target, not guaranteed SM90a/SM100/SM120 branch coverage.

No Cargo, CUDA, or GPU command was run for this audit.
