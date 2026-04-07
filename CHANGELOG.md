# Changelog

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
