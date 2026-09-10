# Task 903 implementation report

## Status

GREEN implementation frozen for root-owned compilation and Ada execution.
The agent ran no Cargo, CUDA, or GPU command.

## TDD evidence

The two initial public regressions were frozen test-only at SHA-256
`7bb6ecfd9f51f134ee4267025c94d8657cb3422209bc0335b3a58a0310fea517`.
Root then observed both fail on Ada at the intended production boundary:
`context-aware GemmEx: deterministic GEMM mode reached a cuBLAS dispatch boundary`.
BF16 ran for 4.91s and F16 for 4.88s after a 41.13s build; runner exit was 0.
Evidence is retained at `/root/tied-head-red-evidence-20260910`.
Root explicitly authorized GREEN before production edits.

## Implementation

- The public tied GemmEx wrapper now checks context health first, delegates F32
  to the Task902 F32 tied wrapper, preserves canonical GemmEx in vendor modes,
  and composes deterministic BF16/F16 as two input upcasts plus ExactScalar F32
  NT directly into caller-owned logits.
- The deterministic composition validates checked dimensions, byte spans,
  alignment, allocation context/liveness, and output non-aliasing before
  scratch allocation or enqueue. Zero reduction emits no cast launches and
  uses the existing F32 epilogue route.
- The existing exact-scalar observed NT recorder now shares one generic
  borrowed-argument body. Its owned wrapper still submits real `GpuBuffer`
  arguments, preserving cudarc event-aware wait/record behavior; the raw
  wrapper uses `RawScalarArgument` and does not create fake owners.
- `presize_tied_lm_head_scratch` reserves `(B*D,Vpad*D,0)` without launching
  and is available only for HF LM builds or colocated tests.
  Both M1 and M3 tied-half LM capture wrappers invoke it before backbone
  capture freezes graph-visible scratch.
- No CUDA runtime body, global GEMM policy, half-output round trip, second NT
  dispatcher, route vocabulary, or unrelated path was changed.

## Tests for root runner

Public CUDA integration (`tests/gemm_tied_f32_output.rs`):

- `deterministic_tied_bf16_inputs_preserve_true_f32_product_bits`
- `deterministic_tied_f16_inputs_preserve_true_f32_product_bits`
- `deterministic_tied_bf16_irregular_inputs_keep_exact_f32_bits_and_guards`
- `deterministic_tied_f16_irregular_inputs_keep_exact_f32_bits_and_guards`
- `deterministic_tied_half_zero_reduction_uses_f32_epilogue_without_inputs`
- `deterministic_tied_half_rejects_null_and_overflow_before_execution`

Colocated CUDA observer/scratch tests (`blas::physical_graph_tests`):

- `tied_half_f32_observer_records_two_upcasts_all_nt_launches_and_no_downcast`
- `tied_half_f32_zero_reduction_observer_records_only_f32_epilogue`
- `tied_half_f32_scratch_freeze_reuses_reserved_pointers_and_rejects_growth`
- `tied_half_f32_rejects_mismatched_and_f32_private_inputs_before_execution`

Colocated CUDA+HF synthetic LM tests:

- `module::gpu_lm::tied_head_capture_tests::tied_bf16_m1_capture_reserves_head_scratch_before_freeze`
- `module::gpu_lm3::tied_head_capture_tests::tied_f16_m3_capture_reserves_head_scratch_before_freeze`

The LM fixtures use one-layer owned synthetic models, warm only the backbone,
reserve through the actual LM `capture_graph` wrapper, and make the first tied
head call after freeze. `Vpad*D = 8192` exceeds the tiny backbone's head-irrelevant
scratch needs, and stable scratch addresses are asserted after the head call.

## Frozen source hashes

```text
5866aeafacb4103c0eb09f2f5c14caa0e2d015558b6748a20640ae924003f25f  src/mamba_ssm/gpu/blas.rs
f097be5a1afcc934e7ad063c36416bb2eaea361a12b3fd46ddbf73a4fdd04c62  src/mamba_ssm/gpu/gemm_bi_triad/launch.rs
eb97b6598d3c108b428b35e8bbc99ab0891d2b9ba237876333cbd88594a1de6f  src/mamba_ssm/gpu/gemm_bi_triad/mod.rs
55aa3c9f11fc3f61127a05d74c7d8011f90614eb535b7f4b17d29503f1ba6939  src/module/gpu_lm.rs
655db313f7ea78c1ab808ebd4e23cee782f666a8283aff52f566e0e2de1cdc6e  src/module/gpu_lm3.rs
ed12ab6a2cdeaee69604bbeeb295e716634dfb994d3e1b1ac5cbc5cba541544d  tests/gemm_tied_f32_output.rs
```

## Local verification and self-review

- `rustfmt --edition 2024` completed cleanly for all six source/test files.
- Exact NT request is `(B,Vpad,D)` with strides `(D,D,Vpad)`, alpha 1, beta 0,
  and no bias; observer tests assert every real sublaunch and zero downcasts.
- Original half inputs and F32 output are backed by real registered owners in
  public and observer tests. Irregular tests use independent F32 `mul_add`
  oracles, asymmetric inputs, guards, and repeat-bit assertions.
- The frozen-growth test checks pointer stability and verifies an oversized
  request leaves sentinel output unchanged.
- No lint suppression, fake owning buffer, network/model download, git/index,
  commit, Cargo, CUDA, or GPU action was used by this agent.
