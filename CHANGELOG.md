# Changelog

## 0.2.2

**End-to-end bf16/f16 inference for both Mamba-1 and Mamba-3.**

### Added

**Unified half-precision inference API**
- `GpuMambaBackbone::new_with_dtype(ordinal, weights, cfg, input_dim, batch, dtype)` for Mamba-1 and the symmetric `GpuMamba3Backbone::new_with_dtype` for Mamba-3. `WeightDtype::{F32, Bf16, F16}` chooses storage dtype at construction; compute stays f32 (CUBLAS_COMPUTE_32F, upcast-inside-kernel).
- New `GpuMamba3LM` wrapper (`src/module/gpu_lm3.rs`), mirror of `GpuMambaLM` — same `generate / generate_streaming / generate_batch / capture_graph / reset` API for M3. Construction via `from_weights_with_dtype` (no HF loader: no real Mamba-3 SISO checkpoint is public yet).
- `GpuMambaLM::last_logits(b)` accessor.
- `GpuMamba3Backbone::blas()` / `download_temporal()` / `temporal_dtype()` accessors.

**End-to-end bf16/f16 activation pipeline (Mamba-1)**
- `GpuMambaInferenceMixed::step_kernels_mixed_native` — new execution path that keeps activations in the weight dtype throughout (no cast-staging per GEMM). Residual stream stays f32 (matches HF `residual_in_fp32=True`). SSM state stays f32.
- `gpu_forward_inference_prefill_mixed` + `GpuMambaTargetMixedScratch` — native bf16/f16 parallel prefill. Replaces the legacy "f32 prefill + downcast" fallback.
- `GpuMambaBackbone::prefill_sequence_mixed` + `alloc_prefill_mixed_scratch`.

**End-to-end bf16/f16 activation pipeline (Mamba-3)**
- `Mamba3GpuInferenceMixed` engine with `step_kernels_mixed_native` + CUDA Graph capture, mirroring the M1 pipeline.
- `Mamba3GpuInferenceMixedScratch` — DtypedBuf activations, f32 residual/coefficients/stats.
- `GpuMamba3MixedWeights` bulk arena (bf16/f16 linear weights) + f32 arena (norms, biases, per-head coefficients).

**New CUDA kernels (all with f32/bf16/f16 variants via `DEFINE_*` macros)**
- `rmsnorm_forward_f32in_{bf16,f16}`: f32 residual → half post-norm (shared by M1 + M3).
- `residual_add_f32_{bf16,f16}`: f32 accumulator += half branch → f32 output (shared).
- `gather_last_timestep_{f32,bf16,f16}`: last-timestep extractor (shared).
- `m3_split_{bf16,f16}`: 8-way split + fused softplus/sigmoid (M3).
- `bcnorm_fwd_{bf16,f16}`, `bc_bias_add_{bf16,f16}`: B/C RMSNorm + bias add (M3).
- `rope_fwd_{bf16,f16}`: RoPE rotation with f32 angle_cumsum (M3).
- `silu_gate_fwd_{bf16,f16}`, `rmsnorm_gated_forward_{bf16,f16}`: M3 output gating.

**New dispatch helpers**
- `HalfKernel` struct (bf16/f16 only) for dual-dtype bridge kernels.
- `gpu_gemm_typed_forward_raw` with fully-independent A/W/C dtypes.
- `gpu_gemm_typed_raw_no_bias` — blas-only variant (no GpuCtx dependency; used by M3).
- `gpu_sgemm_tied_lm_head_blas` + `gpu_gemm_ex_tied_lm_head_blas` — blas-only tied-lm_head helpers.

**Parity & benchmarks**
- `tests/gpu_bf16_parity.rs`: Mamba-1 bf16 vs f32 on mamba-130m-hf. **20/20 greedy token match, KL(f32‖bf16) ≈ 1.0e-3.**
- `tests/m3_bf16_parity.rs`: Mamba-3 f32 vs bf16 synthetic-weight parity (monotone convergence, final cosine > 0.994).
- `tests/gpu_mamba3_lm_test.rs`: M3 LM smoke tests for both dtypes.
- `tests/bench_bf16_vs_f32.rs`: dtype × graph benchmark harness for M1 + M3.

### Performance (RTX 4090, greedy generation, 100 tokens)

| Model                       | f32 +graph | bf16 +graph | speedup |
|---                          |---         |---          |---      |
| Mamba-1 130m-hf             | 398 tok/s  | 494 tok/s   | +24%    |
| Mamba-3 synthetic (d=256×8) | 546 tok/s  | 802 tok/s   | +47%    |

Weight VRAM footprint: **bf16 / f16 = 0.50× f32** (clean 2× compression).

### Fixed

- `GpuMambaWeights` / `GpuMambaMixedWeights`: layout formula used hardcoded `d_model`-sized input_proj; HF Mamba-1 checkpoints have empty input_proj (identity projection) and tripped `debug_assert!(off == total)` in debug builds. Formula now uses the actual CPU weight lengths. No release-mode effect — pure debug guard.
- `step_kernels_mixed_native` (M1 + M3): used `GpuBuffer::copy_from` to seed the f32 residual from input; that path goes through a `SyncOnDrop`-creating CUDA slice view which invalidates CUDA Graph capture. Switched to `copy_from_raw` (cuMemcpyDtoDAsync on cached raw ptrs — capture safe).
- M3 NVRTC compile now inlines `_typed_prelude.cuh` and strips `#include` lines to match the M1 setup — the prelude's bf16/fp16 helpers must be in scope before any `DEFINE_*` macro expansion.
- Clippy hygiene: removed all `#[allow(clippy::too_many_arguments)]` by bundling parameters into named structs (`TypedPtr`, `TiedLmDims`, `PrefillInputs`, `Mamba3States`, `Mamba3LmBuild`).
- `Mamba3Weights` / `Mamba3LayerWeights`: `#[derive(Clone)]` so `GpuMamba3LM` can take `&Mamba3Weights` and internally clone+clear `input_proj` for identity_proj.

### Notes

- The RL training path (`GpuMambaTrainWeights` + training forward/backward) is bit-identical to 0.2.1. The `BackboneEngine::F32` variant never touches the new mixed code; regression-guarded by `test_gpu_f32_backbone_unchanged_after_mixed_refactor` (M1) and the M3 equivalent.
- No real Mamba-3 SISO HF checkpoint exists publicly; `GpuMamba3LM` therefore has no `from_hf` constructor. When checkpoints land, adding one is a 2-3 hour additive change (new HF loader + key remapper), no pipeline changes needed.
- `m3_compute_abg`, `m3_angle_dt_fwd*` stay f32 — they operate on coefficient scalars (dt/a/trap/α/β/γ) and the RoPE angle state is accumulated in f64. Keeping them f32 preserves parity with the state-spaces/mamba reference.

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
