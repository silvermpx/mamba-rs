# mamba-rs

Mamba SSM implementation in Rust with optional CUDA GPU acceleration. Supports Mamba-1 and Mamba-3 SISO.

Full inference and training pipelines with BPTT through recurrent SSM state. Custom CUDA kernels with CUDA Graph capture for minimal-latency GPU inference.

## Features

- **Two architectures** — Mamba SSM (Gu & Dao, 2023) and Mamba-3 SISO (Lahoti et al., ICLR 2026)
- **CPU inference** — zero-allocation single-step recurrent forward pass with SIMD + BLAS
- **GPU inference** — CUDA kernels with optional CUDA Graph capture (~1.6x speedup)
- **CPU training** — full backward pass with BPTT, parallel batch training via Rayon
- **GPU training** — custom CUDA forward + backward kernels (47 for M3, 12 for M1)
- **Serialization** — safetensors format (HuggingFace compatible)
- **Standalone** — no framework dependency (no PyTorch, no Burn, no Candle)
- **f32 / bf16 / f16** — end-to-end half-precision GPU inference (since 0.2.2): 2× weight VRAM compression, +24 % tok/s on Mamba-1 130m, +47 % on Mamba-3. f32 stays the default; bf16/f16 opt-in via `WeightDtype`.

## Quick Start — Mamba SSM

```rust
use mamba_rs::{MambaConfig, MambaState, MambaStepScratch, MambaWeights, mamba_step};

let cfg = MambaConfig::default(); // d_model=128, 3 layers
let weights = MambaWeights::init(&cfg, input_dim, 42);
let mut state = MambaState::zeros(cfg.n_layers, cfg.d_inner(), cfg.d_state, cfg.d_conv);
let mut scratch = MambaStepScratch::new(&cfg);
let mut output = vec![0.0f32; cfg.d_model];

mamba_step(&input, &mut output, &weights, &mut state.layers, &mut scratch, &cfg, input_dim);
state.reset(); // episode boundary
```

## Quick Start — Mamba-3 SISO

```rust
use mamba_rs::mamba3_siso::config::Mamba3Config;
use mamba_rs::mamba3_siso::cpu::inference::{Mamba3StepScratch, mamba3_step};
use mamba_rs::mamba3_siso::state::Mamba3State;
use mamba_rs::mamba3_siso::weights::Mamba3Weights;

let cfg = Mamba3Config::default();
let weights = Mamba3Weights::init(&cfg, input_dim, 42);
let mut state = Mamba3State::zeros(&cfg);
let mut scratch = Mamba3StepScratch::new(&cfg);
let mut output = vec![0.0f32; cfg.d_model];

mamba3_step(&mut output, &input, &mut scratch, &weights, &mut state.layers, &cfg);
```

## GPU Inference (CUDA)

```toml
[dependencies]
mamba-rs = { version = "0.2", features = ["cuda"] }
```

### Mamba SSM

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;

let mut gpu = GpuMambaBackbone::new(0, &weights, cfg, input_dim, batch)?;
gpu.capture_graph()?; // optional ~2x speedup
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

### Mamba-3

```rust
use mamba_rs::mamba3_siso::gpu::inference::GpuMamba3Backbone;

let mut gpu = GpuMamba3Backbone::new(0, &weights, cfg, input_dim, batch)?;
gpu.capture_graph()?; // optional ~1.6x speedup
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

Requires NVIDIA GPU + CUDA toolkit. Kernels compiled at runtime via NVRTC.

### End-to-end bf16 / f16 inference (0.2.2+)

Choose weight and activation dtype at construction. Compute stays f32
(CUBLAS_COMPUTE_32F, upcast-inside-kernel) — no precision loss in
accumulation. Residual stream and SSM state stay f32 for stability
(matches HF `residual_in_fp32=True`). Weight VRAM is halved; typical
decode speedup +20–50 % on Ada/Hopper.

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;

// Same API; the Mixed engine + native bf16 pipeline is picked automatically
// when dtype != F32.
let mut gpu_bf16 = GpuMambaBackbone::new_with_dtype(
    0, &weights, cfg, input_dim, batch, WeightDtype::Bf16,
)?;
gpu_bf16.capture_graph()?;
gpu_bf16.step(&input, &mut output)?;
```

For LLM inference with HuggingFace-format Mamba-1 checkpoints:

```rust
use mamba_rs::module::gpu_lm::GpuMambaLM;
use mamba_rs::module::sample::SampleParams;
use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
use std::path::Path;

let mut lm = GpuMambaLM::from_hf_with_dtype(
    Path::new("./mamba-130m-hf"), 0, WeightDtype::Bf16,
)?;
lm.capture_graph()?;
let tokens = lm.generate(&[1, 2, 3, 4, 5], &SampleParams::default())?;
```

Mamba-3 mirrors the same API — `GpuMamba3Backbone::new_with_dtype`
and `GpuMamba3LM::from_weights_with_dtype` (no HF loader yet; no
public Mamba-3 SISO checkpoint exists). Validation: bf16 vs f32
greedy match 20/20 tokens on `state-spaces/mamba-130m-hf`,
KL ≈ 1e-3 on final logits.

## Weight Serialization

```rust
use mamba_rs::serialize;

// Mamba-1
serialize::save(Path::new("model.safetensors"), backbone.weights(), cfg, input_dim)?;
let (weights, cfg, input_dim) = serialize::load(Path::new("model.safetensors"))?;

// Mamba-3
use mamba_rs::mamba3_siso::serialize::{save_mamba3, load_mamba3};
save_mamba3(Path::new("m3.safetensors"), &weights, &cfg, input_dim)?;
let (weights, input_dim) = load_mamba3(Path::new("m3.safetensors"), &cfg)?;
```

## Performance (RTX 6000 Ada)

| | Mamba SSM | Mamba-3 SISO |
|---|---|---|
| GPU Inference B=1 (CUDA Graph) | **79 us** | **86 us** |
| GPU Training Fwd+Bwd (T=32) | 1,640 us | 2,169 us |
| CPU Inference B=1 | 84 us | **65 us** |
| CPU Training Fwd+Bwd (T=32) | 15,874 us | **3,609 us** |

Zero heap allocations per inference step. See detailed results:
- [Mamba SSM benchmarks](docs/mamba1-benchmarks.md)
- [Mamba-3 benchmarks](docs/mamba3-benchmarks.md)

## Documentation

- [Mamba SSM architecture](docs/mamba1-architecture.md) — pipeline, modular API, weight layout
- [Mamba-3 architecture](docs/mamba3-architecture.md) — trapezoidal SSM, RoPE, BCNorm, CUDA kernels
- [Mamba SSM benchmarks](docs/mamba1-benchmarks.md) — GPU/CPU inference + training numbers
- [Mamba-3 benchmarks](docs/mamba3-benchmarks.md) — GPU/CPU inference + training numbers

## Citation

```bibtex
@inproceedings{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{mamba3,
  title={Mamba-3: Improved Sequence Modeling using State Space Principles},
  author={Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

## License

Dual-licensed under MIT or Apache-2.0.
