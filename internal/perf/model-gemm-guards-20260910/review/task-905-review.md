### Spec Compliance

- ❌ Issues found: the clear-before-any-step-attempt requirement is violated when M1 mixed already holds the other path's graph (`src/mamba_ssm/gpu/inference.rs:1479`, `:1519`, `:1939`, `:1982`). One Important finding below.
- ✅ The shared nonzero deterministic guard is independent of family/dtype (`src/mamba_ssm/gpu/graph_capture.rs:285`); the shared replay seam checks health before plan validation or the legal None callback (`:322`). Five capture owners and ten replay entries are migrated; the five external legacy admission calls remain separate Task906 scope (`tests/gemm_bi_tf32_contract.rs:4783`, `:4908`).
- ⚠️ Fresh SM120 hardware acceptance is not supplied and is not claimed. The remaining-owner and architecture audit documents were read as scope/evidence boundaries, not treated as fresh runtime evidence. Unchanged resource-anchor internals and terminal inventories outside this task's diff were not independently re-audited.

### Strengths

- The model fixtures use independent literal per-layer projections, actual eager/capture bodies, owned buffers, two layers, B1/B3 and both replay entries; they compare output bits after recurrence resets (`src/mamba_ssm/gpu/inference.rs:259`, `src/mamba3_siso/gpu/inference.rs:125`, shared inventory checks at `src/mamba_ssm/gpu/graph_capture.rs:358`). H2D/D2H remain outside the body recorders.
- Capture consumes its matching permit and validates before graph/plan installation; the small M1 mixed path tag explicitly guards the shared slot (`src/mamba_ssm/gpu/inference.rs:1380`, `:2032`, `:2095`; `src/mamba3_siso/gpu/inference.rs:770`, `:782`).
- The native-half fix carries context to the existing terminal independently of physical observation and preserves the physical argument-identity closure (`src/mamba_ssm/gpu/gemm_bi_triad/launch.rs:10866`, `:11431`). The direct regression checks four ordered real projections, physical-node equality, no double counting and rejection before enqueue (`src/mamba_ssm/gpu/blas.rs:4327`).
- The vendor tripwire is cfg(test), thread-local and scoped, rejects nesting without disturbing outer state, and is inserted at all nine FFI boundaries in the diff; positive controls execute both vendor modes and verify denied calls leave output unchanged (`src/mamba_ssm/gpu/blas.rs:31`, `:110`).

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- `src/mamba_ssm/gpu/inference.rs:1479` (also `:1519`, `:1939`, `:1982`): clearing the corresponding eager permit is conditional on `self.graph.is_none()`. A concrete reachable sequence is: successfully eager-run legacy and native; capture native, leaving the legacy permit; attempt legacy `step` or `step_gpu_only`, which fails its captured-path check; then call legacy `capture_graph`. The old permit survives and capture succeeds despite the intervening failed attempt. An invalid-input upload/panic with that installed graph also retains the permit. The reverse path has the same defect. This contradicts the binding requirement to clear the corresponding manifest before **any** new step attempt, including validation/upload failure. Capture still compares the actual inventory, so this is stale-permission admission, not evidence that a mismatched physical graph launches.
  The new test currently encodes the incorrect behavior: after two rejected legacy attempts it explicitly expects legacy recapture to succeed at `src/mamba_ssm/gpu/inference.rs:205`. Its accepted Ada log confirms that expectation ran successfully (`internal/perf/model-gemm-guards-20260910/initial-gpu-remaining/mamba_ssm::gpu::inference::model_gemm_manifest_tests::m1_failed_steps_clear_permits_and_mixed_graphs_reject_wrong_path.log:36`). Clear only the corresponding permit unconditionally at each entry, before upload. Apply the stated rule consistently to all ten entries. Change this regression to require the failed-attempt permit to be None and recapture to reject; retain a separate positive recapture fixture with no intervening step attempt. Cover both mixed directions, both entry forms and invalid upload with an installed opposite-path graph.

#### Minor (Nice to Have)

- Existing verification output is not pristine: `internal/perf/model-gemm-guards-20260910/final-host-first/rustdoc.log:2` reports the unchanged private-item link at `src/mamba_ssm/gpu/gemm_bi_triad/contract.rs:2834`; `internal/perf/model-gemm-guards-20260910/provenance-fix/build.log:2` retains deprecated-setter warnings. These are acknowledged pre-existing cleanup items, not additional Task905 production defects or reasons to rerun accepted GPU tests.

### Assessment

- **Task quality: Needs fixes.** The integration and physical/context separation are well supported, but the explicit permit lifecycle has a reachable exception and its new regression blesses that exception.
- **Checks:** Read the supplied base `b7397d4f` to head `96640d62` diff in sequential chunks, all ten changed files, requirements and implementer report. Read the root acceptance report and raw result lines, including corrected M1/M3 matrices (128.19s/81.22s), direct native-half (4.74s), both heads, vendor/poison controls, and final provenance fixes. Earlier failed packets remain failures; final correcting packets supply acceptance. No tests, builds, GPU/Cargo commands or Git mutations were run. No source outside the diff was inspected; only the requested review report was written.
