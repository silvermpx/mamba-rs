# Changelog

## Unreleased

### Batch-invariant matvec is now opt-in

The `matvec_bi_*` kernel dispatch, introduced as the default in 0.3.0,
is now behind an opt-in flag — cuBLAS gemv is the default path.

Enable via either:
- `ctx.set_batch_invariant(true)` on the `GpuCtx`
- `MAMBA_RS_BATCH_INVARIANT=1` environment variable at process start

When enabled, output is bit-identical across batch sizes (KL ≈ 1e-11
between `b=1` and `b=N` per slot). When disabled, cuBLAS's per-M algo
selection may produce sub-ULP differences between decode and prefill.

Also: the `matvec_bi` kernel now uses vectorized 128-bit `cp.async`
loads for the A-row staging path (sm_80+). Parity and KL guardrails
unchanged.

## 0.3.0

GPU training and bf16/f16 inference for Mamba SSM and Mamba-3 SISO,
unified trainer API, custom batch-invariant matvec.

### Trainer API

`MambaTrainer` / `Mamba3Trainer` — one entry point per arch.
`step(input, d_temporal)` does forward + backward + AdamW + (for mixed)
master→compute sync. `WeightDtype` chosen at construction; the F32/Mixed
engine split is internal.

`capture_graph()` records the whole training step into a CUDA Graph and
asserts pointer stability on every replay (catches the silent-corruption
class if any backing buffer gets reallocated between capture and replay).

`scaler_state()` / `load_scaler_state()` round-trip the f16 dynamic
loss scaler across checkpoint resume so training picks up at the
last-known scale instead of repaying the ~2000 step discovery phase.

`GpuAdamW` matches PyTorch — decoupled WD, f64 bias correction,
capturable variant reads bias factors from a 2-element device buffer.
`DynamicLossScaler` matches `torch.cuda.amp.GradScaler`; eager skips
AdamW on overflow, captured-graph runs a device-side conditional
unscale that sanitises inf/NaN to zero.

### Inference

Native bf16/f16 activation pipeline for both archs. Activations stay
in weight dtype, residual stream stays f32 (matches HF
`residual_in_fp32=True`), SSM state stays f32. `step_kernels_mixed_native`
replaces the cast-staged path. Native parallel prefill — no f32 fallback.

`GpuMambaBackbone::new_with_dtype` / `GpuMamba3Backbone::new_with_dtype`,
plus the corresponding `GpuMambaLM` / `GpuMamba3LM` LM wrappers.

### Batch-invariant matvec (the headline change)

Custom CUDA kernel in `kernels/gemm_batch_invariant.cu`. Output is
bit-identical across batch sizes (KL ≈ 1e-11), against cuBLAS's
`cublasGemmEx` which drifts to KL ≈ 1e-3 between M=1 and M=N because
it picks a different split-K / tile / algo per M.

Layout: 8 warps × 32 threads per CTA, K split across warps, per-warp
partials reduced in smem with a fixed reduction tree
`(((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7)))`. Grid is 2D
`(ceil(N/32), M)` — one CTA per `(m_row, col_chunk)`, each row
independent. The same kernel handles M=1 (decode) and M>1 (RL parallel
envs, prefill).

Parity guardrails (`tests/hf_batch_parity.rs`, `tests/extreme_edge_coverage.rs`):

- `bf16_batch_divergence_known` (adversarial prompt that hit
  KL ≈ 2.7 with cuBLAS): now **KL = 0.0000**.
- `bf16_multi_length_parity` lengths 3, 5, 32, 63, 64, 65, 128:
  all KL ≤ 3.5e-11.
- `inference_extreme_batch_parity_bf16_b{16,32}`: KL ≤ 2.2e-11.
- `hf_cpu_vs_gpu_inference_bf16`: 20/20 token match, KL = 2e-6.

### HuggingFace

`rms_norm_eps` and `layer_norm_epsilon` are read from `config.json`;
the loader warns when a checkpoint specifies a value other than the
kernel-hardcoded `1e-5` (e.g. FalconMamba uses 1e-6).

Untied `lm_head` stride bug — the GEMM wrote at `vocab_size` but the
CPU downloader sliced at `vocab_size_padded`, so batch slots beyond
the first got wrong logits on any vocab not already 64-aligned (e.g.
mamba-130m's 50 280). Padded on upload in both LM wrappers.

End-to-end bf16 inference verified on every cached
`state-spaces/mamba-*-hf` snapshot (130m / 370m / 1.4b / 2.8b):
15/15 greedy match vs f32, KL ≤ 1.6e-3 (best 4e-5 at 2.8b).

### Critical bugfixes

`a_neg` staleness in training — the per-layer `a_neg = -exp(a_log)`
buffer was computed once at trainer construction and never refreshed,
so optimizer updates to `a_log` never reached the SSM. Recomputed
after every AdamW step across the eager, captured-graph, and f16
paths. Pre-fix, gradient descent on the A-matrix was a silent no-op
for the entire training run.

f16 loss-scaler overflow corruption — the captured-graph unscale
kernel did `grads[i] *= 0.0f`, and `±Inf * 0 = NaN` poisoned the
next AdamW step. Switched to `grads[i] = overflow ? 0 : grads[i] * unscale`
so overflow is always a clean skip.

Mamba SSM mixed forward routed `WeightDtype::F32` through the legacy
`conv1d_burnin_forward`, whose argument order differs from the typed
variants — the persistent conv state and the post-conv activation
buffer got silently swapped every step. Routed through a typed-signature
f32 kernel.

RMSNorm finite-guard — on 48-layer bf16 models a transient activation
overflow produced one NaN that cascaded through every subsequent
RMSNorm via `1/NaN = NaN`. Catastrophic on mamba-1.4b-hf (0/15 vs f32
pre-fix). Added the `if (!isfinite(rms) || rms < 1e-20f) rms = 1.0f`
guard to every RMSNorm / BCNorm / RMSNormGated variant.

M1 parallel scan at d_state > 64 used a too-restrictive smem load
filter; replaced with a strided loop so every `ds` entry loads regardless
of the `ds / hd` ratio.

### Other fixes

f16 eager overflow path now skips AdamW + sync + `a_neg` recompute on
scaler-detected overflow (matches `GradScaler`). Spurious skips on
`m1_trainer_f16_production_lr_stable`: 47/50 → 3/50.

`AdamWBiasFactors` defaults to `[1.0, 1.0]` instead of `[0.0, 0.0]`,
which would have produced an all-weight-decay update if a graph were
captured before the first bias-factor write.

AdamW step counter clamps the bias-correction exponent at `2^30` — the
prior `as i32` cast went negative past `i32::MAX`. Cosmetic, free fix.

`step_kernels_mixed_native` (M1 + M3) uses `cuMemcpyDtoDAsync` on
cached raw pointers instead of `GpuBuffer::copy_from`, which goes
through a `SyncOnDrop` slice view that invalidates graph capture.
M3 `angle_dt` launch fixed for the same reason.

M3 NVRTC inlines `_typed_prelude.cuh` so the bf16 / f16 helpers are
in scope before any `DEFINE_*` macro expansion.

`GpuMambaWeights` / `GpuMambaMixedWeights` layout formula uses the
actual CPU weight lengths instead of `d_model`-sized `input_proj` —
HF checkpoints frequently have an empty `input_proj` (identity).

### API + crate plumbing

Inner trainer engines (`MambaTrainerMixed`, `MambaTrainerF32`,
`Mamba3TrainerMixed`, `Mamba3TrainerF32`) are now `pub(crate)`. Public
surface is the wrappers only — the F32/Mixed dispatch lives behind a
private enum, mirroring the existing inference `BackboneEngine` pattern.

`cuda` feature pulls `half` + `bytemuck` directly so
`cargo build --features cuda` (no `hf`) works on its own.

`WeightDtype` is re-exported from the crate root. `pub mod gpu3`
mirrors the existing `pub mod gpu` so Mamba-3 types land one import deep.
`Mamba3Weights` / `Mamba3LayerWeights` are `#[derive(Clone)]`.

All remaining `#[allow(clippy::too_many_arguments)]` removed by
bundling args into named structs (`TypedPtr`, `TiedLmDims`,
`PrefillInputs`, `Mamba3States`, `Mamba3LmBuild`, `BiGemmArgs`).

README rewritten with a bf16/f16 quickstart and an HF LLM example.

### Tests

49 test files, 287 tests pass on Ada with `cuda hf` + 60 ignored
benches. Notable:

`hf_training_convergence` — 30-step real-checkpoint training on
mamba-130m-hf for all three dtypes, monotone weight progress + valid
post-training inference. `hf_full_cycle` — load → infer → train →
re-infer. `hf_batch_parity` — cross-batch logit parity on real weights
(CPU ↔ GPU f32 20/20 exact; bf16 20/20 KL ≤ 3e-6). `extreme_edge_coverage`
— batch=16/32, 1024-token generation stability, M3 training at
T=512/1024. `stability_stress` — CUDA Graph replay determinism, training
repeatability across independent trainer instances. `cpu_gpu_train_parity`
— M1 CPU vs GPU backward parity at f32. `coverage_gaps::a_log_actually_reaches_ssm_after_training`
— regression guard for the `a_neg` staleness fix.
`backward_mixed_parity::backbone_grad_parity_multi_layer_{bf16,f16}` and
`trainer_smoke::m{1,3}_trainer_multi_layer_bf16` — multi-layer
(`n_layers = 3`) parity coverage.

### Performance summary

`state-spaces/mamba-130m-hf` on RTX 6000 Ada with CUDA Graph:

| | f32 | bf16 | f16 |
|---|---:|---:|---:|
| Decode tok/s (B=1)                  | 725   | **898**   | 899   |
| Training step (B=1, T=32) µs        | 1 640 | 1 120     | 1 110 |
| 30-step real-checkpoint convergence | 3.8 s | 2.9 s     | 3.1 s |
| Cross-batch KL (B=1 vs B=32)        | —     | **2e-11** | 2e-11 |

Weight VRAM: bf16 / f16 = 0.50 × f32.

### Notes

No public Mamba-3 SISO HuggingFace checkpoint exists yet — the M3 LM
wrapper drives synthetic weights for now. When a real M3 checkpoint
lands, the HF loader becomes a key remapper on top of the existing
safetensors path, no pipeline changes.

`m3_compute_abg` and `m3_angle_dt_fwd_*` stay f32 by design; the
RoPE angle accumulator stays f64. Both match `state-spaces/mamba/mamba3.py`.

Pure-f32 inference and pure-f32 training are byte-unchanged from 0.2.x,
regression-guarded by `test_gpu_f32_backbone_unchanged_after_mixed_refactor`
on both archs.

## 0.2.1

### Fixed

- **Mamba-3 RoPE angle accumulation precision**: upcast angle accumulator to f64 for addition and modulo wrap, then back to f32 for sin/cos. Prevents drift over long inference sequences (390+ steps). Matches upstream `mamba3.py` fix from `state-spaces/mamba`. Applied to CPU inference, CPU training forward, and all 3 GPU CUDA angle kernels (`angle_dt_fwd`, `m3_angle_dt_fwd_batch`, `m3_angle_dt_fwd_seq`).

## 0.2.0

**Mamba-3 SISO** — full implementation with CPU + GPU inference/training, CUDA Graph, 47 kernels.

### Added

- `mamba3_siso` module: complete Mamba-3 SISO (Lahoti et al., ICLR 2026)
- CPU inference with BLAS matvec + SIMD SSM recurrence (pulp)
- CPU training: 7-phase forward (F1-F7) + 8-phase BPTT backward (B1-B8)
- GPU inference: `GpuMamba3Backbone` with CUDA Graph capture (~1.6x speedup)
- GPU training: `gpu_forward_mamba3_backbone` + `gpu_backward_mamba3_backbone`
- 47 CUDA kernels across 5 .cu files (SSM, chunked scan, ops, norms, elementwise)
- `Mamba3GpuInferenceEngine` with `disable_event_tracking()` for graph capture stability
- GPU weight upload: `GpuMamba3WeightsInf::from_cpu()` (flat buffer + WeightSlice)
- GPU training weights: `GpuMamba3Weights::from_cpu()` + `GpuMamba3Grads::new()`
- Parallel batch training via Rayon with thread-local scratch + epoch zeroing
- `Mamba3Config` with full validation (headdim, d_state, ngroups, RoPE, a_floor)
- 4 persistent states (SSM + K + V + angle) per layer
- Safetensors serialization (save/load)
- 25 integration tests (9 finite-diff gradient checks, correctness, stability)
- `pulp` dependency for SIMD vectorized SSM recurrence
- `ops/norms.rs`: shared RMSNorm, BCNorm, RMSNormGated

### Fixed

- `m3_dqkv` chunked backward: Part 2 read overwritten shared memory (ssm_sm held d_state instead of SSM_States)
- CPU inference/training parity: softplus uses std `f32::ln()`, sin_cos uses `f32::sin_cos()`
- CPU backward angle reconstruction: `angle_state_init` parameter for correct RoPE gradients with burn-in
- GPU inference: kernel arg order fixes (m3_step_fwd, m3_angle_dt, m3_split)
- GPU training: chunk_size scratch buffers use dims.chunk_size() (64) not hardcoded 16

### Architecture (Mamba-3 SISO vs Mamba-1)

| Feature | Mamba-1 | Mamba-3 SISO |
|---------|---------|-------------|
| Conv1d | Yes | No |
| A matrix | Fixed | Input-dependent per-head |
| Integration | Exponential | Trapezoidal |
| RoPE | No | Per-head angles [0, 2pi) |
| B/C | Single d_state | Multi-head + BCNorm |
| D | Per-channel | Per-head |
| Parallel scan | T>128 | T>64 (chunk_size=64) |

---

## 0.1.4

Extended GPU architecture support for all modern NVIDIA GPUs.

### Added

- SM 120 support: Blackwell consumer (RTX 5090, RTX 5080, RTX 5070)
- SM 61 support: Pascal consumer (GTX 1080, GTX 1070)
- SM 60 support: Pascal datacenter (P100)
- Future-proof fallback: GPUs with compute capability > 12.x automatically use sm_120

### Changed

- `nvrtc_arch()` now covers SM 60 through SM 120 (Pascal → Blackwell, 10 years of GPUs)
- Unknown future architectures (cc > 12) fall back to latest known (sm_120) instead of ancient sm_70

---

## 0.1.3

CPU performance, GPU architecture refactor, parallel training, bug fixes.

### Added

- Cephes degree-7 polynomial `fast_exp` with NEON (AArch64) and AVX2+FMA (x86_64) SIMD
- Pre-computed `a_neg = -exp(a_log)` at weight load time (eliminates 12K+ exp() per inference step)
- Batch `da_buf` + `fast_exp_inplace` in SSM inner loop for SIMD vectorization
- Apple Accelerate framework BLAS dispatch (`accelerate` feature, macOS AMX coprocessor)
- `gemm` crate BLAS dispatch (`gemm-blas` feature, AVX2/AVX-512/NEON microkernels)
- Rayon parallel batch inference (`mamba_step_batch`) with automatic threshold
- Parallel training forward + backward with thread-local gradient accumulation
- CPU training benchmark (B=1 sequential + parallel B=16/64/128)
- GPU parallel prefix scan for long sequences (T>256): warp shuffle scan, single-pass Y accumulation, chunked processing with inter-chunk carry
- `GpuMambaTrainWeights`: per-tensor GPU weight storage for training (industry standard)
- `GpuCtx::disable_tf32()` for exact f32 parity testing

### Changed

- GPU training weights: per-tensor `GpuBuffer` allocation (`GpuMambaTrainWeights`), matching PyTorch convention
- GPU inference weights: flat buffer + `WeightSlice` views (`GpuMambaWeights`), optimized for CUDA Graph capture
- GPU gradients: flat buffer + `GradSlice` views (`GpuMambaGrads`), single memset zeros all
- Activation kernels: scalar dispatch (safe for any buffer size)
- Backward parity test: PyTorch-style `allclose(atol + rtol * |expected|)` tolerance

### Fixed

- GPU backward gradient buffer synchronization
- NVRTC compile options aligned with production configuration
- Parallel backward gradient accumulation (epoch-based lazy zeroing)
- CPU backward scratch buffer size for `mamba_input_dim < d_model` configurations
- `gemm` crate: disabled unused f16 sub-crate (ARM Grace lacks `fullfp16` NEON)

### Performance (GH200 Grace, 72 cores)

**CPU Inference (T=1, B=1):**

| Config | Before | After | Speedup |
|--------|--------|-------|---------|
| small (64, 2L) | 61 us | **21 us** | 2.9x |
| default (128, 3L) | 377 us | **76 us** | 5.0x |
| medium (256, 4L) | 2.2 ms | **261 us** | 8.4x |
| large (512, 6L) | 13.6 ms | **1,226 us** | 11.1x |

**Parallel Training (default config, T=32, 72 cores):**

| Batch | Total | Samples/sec | vs sequential |
|-------|-------|-------------|---------------|
| B=16 | 17.6 ms | 910 | 12.6x |
| B=64 | 26.0 ms | 2,458 | 34.3x |
| B=128 | 40.2 ms | **3,183** | **44.3x** |

## 0.1.2

GPU inference engine, CUDA Graph support, comprehensive test suite.

### Added

- GPU inference engine (`GpuMambaBackbone`) with step/reset API
- CUDA Graph capture — 1.4x speedup on inference (115 us vs 155 us on H100)
- 3-level modular API: `mamba_layer_step` / `mamba_block_step` / `mamba_step`
- safetensors serialization (HuggingFace compatible, cross-framework)
- Batch CPU inference (B>1)
- Sequence forward (`forward_sequence`) for T>1
- Parallel prefix scan CUDA kernels (warp shuffle, for long sequences)
- Flat weight/gradient buffers with WeightSlice/GradSlice zero-cost views
- `GpuBuffer::copy_from_raw` — graph-safe D2D copy via `cuMemcpyDtoDAsync`
- Separate `gpu_input` buffer for input_dim != d_model
- 26 correctness tests covering CPU inference, GPU parity, training backward, serialization
- Full benchmark suite: GPU/CPU inference + training on GH200 and RTX 6000 Ada

### Fixed

- softplus backward kernel dispatch correctness
- CUDA Graph stream capture isolation (graph-safe kernel dispatch with cached device pointers)
- RmsNorm backward gradient validated against PyTorch source

### Performance (default config: d_model=128, 3 layers, 366K params)

- GPU inference B=1: 155 us (GH200), 124 us (RTX 6000 Ada)
- GPU inference + CUDA Graph B=1: 115 us (GH200), 79 us (RTX 6000 Ada)
- GPU training fwd+bwd T=32: 2.3 ms (GH200), 1.7 ms (Ada) — 5.5-8.6x vs CPU
- CPU inference B=1: 377 us (Grace ARM), 348 us (Xeon)

## 0.1.1

Minor fixes.

- Remove dead GPU BLAS functions (unused GpuBuffer variants)
- Remove unused multi-GPU infrastructure (GpuTopology, detect_topology, peer access)
- Remove no-op set_blas_threads and all call sites
- Remove unused Conv1dDims.batch field
- Clean up internal comments

## 0.1.0

Initial release.

- CPU inference: zero-allocation single-step recurrent forward pass
- CPU training: batched forward + backward with BPTT through SSM state
- Burn-in support for recurrent state warming
- CUDA GPU backend with custom kernels (SSM, conv1d, RMSNorm, fused ops)
- Flat contiguous weight/gradient buffers for optimizer fusion
- Rayon-parallel batch processing
- Mamba-specific weight initialization (A_log, dt_proj from paper Section 3.5)
