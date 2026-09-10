# Task 902 implementation report

## Status

GREEN source/test freeze ready for root-owned CUDA+HF compilation and Ada verification. No commits, index changes, Cargo/CUDA runs, or CUDA/header edits were made by the implementer.

## Implemented

- Added crate-private unsafe borrowed F32 NN and NT cached seams with raw-boundary Rustdoc.
- Kept one scalar/prepared dispatcher body. `ScalarInputArgument` / `ScalarOutputArgument` submit owned buffers through cudarc slice arguments (preserving event-managed legacy stream behavior) and GpuCtx raw pointers through byte-identical pointer arguments.
- Made cached buffer wrappers thin pointer adapters without duplicating hot-path allocation queries.
- Added the central context-routed F32 NN pointer seam. Deterministic mode follows Inference/Triad; vendor modes retain SGEMM and configured handle behavior. Buffer, pointer, and homogeneous-F32 typed wrappers converge on it.
- Routed deterministic tied F32 heads and the Inference-family dX fallback through cached Triad NT with dimensions `(B,Vpad,D)`.
- Routed all M3 F32 projections, typed projections, untied heads, and tied heads through `GpuCtx`; added `GpuMamba3Backbone::ctx()` and documented remaining no-context helpers as vendor-only.
- Rewrote the stale typed-forward dispatcher overview around canonical deterministic/vendor modes.
- Added real-owner CUDA regressions covering route recording, CPU NN/NT sums, repeated bits, an interior NN input pointer, guarded tied/typed outputs, NN bias, NN/NT zero reduction, both deterministic families, and null/overflow rejection.

## TDD evidence

Root ran the exact frozen Phase-1 test source SHA-256 `97e3a6e9c0ca045e43971f4415e30cad2dea10a989881325134ba0f6b2934c68` on Ada. Both tests failed only at the expected explicit error `deterministic GEMM mode reached a cuBLAS dispatch boundary`; each test process exited 101, runner/source integrity passed, completed 2026-09-10 11:37:52 UTC. Root then authorized GREEN.

GREEN execution is root-owned and pending for this freeze. Implementer checks:

- `rustfmt --edition 2024 --check` on all six changed Rust targets: exit 0.
- `git diff --check`: exit 0.
- M3 high-level census for `_blas`, `gpu_gemm_typed_raw_no_bias`, and `sgemm_no_bias`: no matches.

Root's initial CUDA+HF compile reached source compilation and stopped at E0364
because the narrow validator was re-exported with broader visibility; no GPU
tests ran. The follow-up CUDA+HF all-target compile passed in 9.54s after the
re-export received the same `pub(in crate::mamba_ssm::gpu)` visibility. That
compile identified the retained thin cached-buffer helper as test-only; it is
now gated with `#[cfg(test)]` without changing the public buffer wrappers.

## Focused CUDA tests

- `deterministic_triad_public_forward_ptr_records_nn_route`
- `deterministic_triad_tied_f32_raw_records_nt_route_and_strides`
- `f32_nn_buffer_and_interior_pointer_wrappers_match_cpu_and_repeat_for_both_families`
- `tied_f32_guarded_output_matches_cpu_and_repeats_with_nt_routes_for_both_families`
- `tied_f32_zero_reduction_zeros_only_the_guarded_output_subspan_for_both_families`
- `all_f32_typed_nn_zero_reduction_applies_bias_once_inside_output_guards`
- `tied_f32_raw_rejects_null_and_overflowing_spans_before_launch`

## Files changed

- `tests/gemm_context_routing.rs`
- `src/mamba_ssm/gpu/gemm_bi_triad/launch.rs`
- `src/mamba_ssm/gpu/gemm_bi_triad/mod.rs`
- `src/mamba_ssm/gpu/blas.rs`
- `src/mamba3_siso/gpu/inference.rs`
- `src/module/gpu_lm3.rs`

## Self-review and concerns

- Audited every NN output and NT input/output scalar branch. Sites that formerly
  submitted `inner()` / `inner_mut()` use the owned/raw adapter. Existing
  `GpuBuffer::raw_ptr` sites were already cached no-sync accesses and do not
  imply an additional wait. The distinct shifted-tail path that previously
  obtained `dy.inner().device_ptr()` retains its parent-input dependency through
  an earlier `dy` adapter submission on the same stream before the derived
  subpointer is used; no extra synchronization machinery was added.
- Triad cached hot paths do not call the new full-resource validator twice. The validator is limited to previously unvalidated Inference/vendor pointer routes.
- Exact argument order and scalar controller/prepared identity code were not changed; CUDA/header bytes were not edited.
- Root's first prepared-control run passed 53 tests with one unchanged source
  sentry failure: it used the first `#[cfg(test)]` token as a production
  boundary and expected four calls, while baseline `de8b463e` already had six
  production `physical_prepared_f32_route(prepared, ...)` calls. The test-only
  repair binds the explicit `prepared_f32_launch_tests` module boundary,
  names all six intended producer functions, and checks the three real call
  forms/counts (1 + 4 + 1) while preserving the semantic argument/resource
  assertions.
- Remaining API-audit item: public no-context vendor-only raw-pointer helpers are safe Rust functions and cannot validate arbitrary foreign allocation ownership because they have no `GpuCtx`. High-level model paths no longer call them; signature migration is outside Task 902.
- Root compilation may expose Rust lifetime/type issues in the new narrow generic adapter; no local Cargo execution was authorized.
