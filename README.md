# mamba-rs

Mamba SSM and Mamba-3 SISO in Rust with optional CUDA GPU acceleration.
Inference and training for both, with custom CUDA kernels.

Pure Rust + CUDA. Kernels compile at runtime via NVRTC.

## Features

- **Two architectures** — Mamba SSM (Gu & Dao, 2023) and Mamba-3 SISO (Lahoti
  et al., ICLR 2026).
- **CPU + GPU** — both paths exposed, with a cross-path parity test on shared
  weights.
- **Inference + training** — full backward pass with BPTT through the
  recurrent SSM state; AdamW optimizer; CUDA Graph capture for both.
- **f32 / bf16 / f16** — a single `WeightDtype` selector at construction.
  Compute stays f32 (upcast-in-kernel, f32 accumulators) regardless of
  storage dtype.
- **Deterministic inference & training (opt-in)** — `MAMBA_RS_BATCH_INVARIANT=1`
  / `ctx.set_batch_invariant(true)` routes every GEMM through custom
  deterministic kernels (`kernels/sgemm_bi.cu`, `gemm_batch_invariant.cu`):
  inference logits are bit-identical across batch sizes (KL ≈ 1e-11), and
  f32 / bf16 / f16 training is bit-identical across runs. Default path is
  cuBLAS for maximum throughput.
- **Tensor-core deterministic tier (opt-in)** — `MAMBA_RS_BI_TENSOR_CORES=1`
  / `ctx.set_bi_tensor_cores(true)` on top of the flag above swaps the
  training GEMM triad for mma.sync tensor-core kernels: still fully
  deterministic (own numeric contract), and **faster than cuBLAS** on
  d_model ≥ 768 (0.77× of PEDANTIC per step at d1536).
- **HuggingFace loader** — safetensors, synthetic + real Mamba SSM
  checkpoints (130m / 370m / 1.4b / 2.8b validated).
- **Standalone** — no framework dependency.

## Use cases and API choice

The crate targets two workloads. Pick the entry point that matches yours.

### Reinforcement learning / small custom models

Latency-critical, typically `d_model ≤ 256`, often batch = 1 for actor
rollouts. Both CPU and GPU paths are supported; CPU is competitive at
these sizes (~85 µs/step on Ada Xeon vs 79 µs/step on RTX 6000 Ada).

- **Inference** — `mamba_step` (CPU) or `GpuMambaBackbone::step` (GPU)
- **Training** — `parallel_mamba_forward` / `parallel_mamba_backward`
  (CPU, Rayon-parallel batch) or `MambaTrainer::step` (GPU, CUDA-Graph-
  captured forward + backward + AdamW + sync)

CPU training works for model sizes where GPU overhead dominates
(`d_model ≤ 128`, `batch ≤ 8`); GPU training scales well to `batch ≥ 32`.

### Large language models

Throughput-critical, `d_model ≥ 768`, sequence-level decoding with a
HuggingFace checkpoint. GPU-only in practice — a 2.8b model on CPU is
single-digit tokens/sec regardless of implementation.

- **Inference** — `GpuMambaLM::from_hf_with_dtype` + `generate`
- **Fine-tuning** — `MambaTrainer::new_full` accepting the HF
  backbone weights (Mamba SSM only; no public Mamba-3 SISO checkpoint
  exists yet)

The CPU `MambaLM` path compiles and runs end-to-end, but exists for
CPU↔GPU parity testing (`tests/hf_batch_parity.rs`), not for production
LLM serving.

### Sharing weights across paths

All paths consume the same `MambaWeights` / `Mamba3Weights` struct.
A training run's `MambaTrainer::snapshot_master()` output loads directly
into `GpuMambaBackbone`, `GpuMambaLM`, or the CPU `MambaBackbone` without
conversion.

## Quick start (CPU)

### Mamba SSM

```rust
use mamba_rs::{MambaConfig, MambaState, MambaStepScratch, MambaWeights, mamba_step};

let cfg = MambaConfig::default();
let weights = MambaWeights::init(&cfg, input_dim, 42);
let mut state = MambaState::zeros(cfg.n_layers, cfg.d_inner(), cfg.d_state, cfg.d_conv);
let mut scratch = MambaStepScratch::new(&cfg);
let mut output = vec![0.0f32; cfg.d_model];

mamba_step(&input, &mut output, &weights, &mut state.layers, &mut scratch, &cfg, input_dim);
```

### Mamba-3

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

## Quick start (GPU inference)

```toml
[dependencies]
mamba-rs = { version = "0.4", features = ["cuda"] }
```

`GpuMambaBackbone::new_with_dtype` and the symmetric Mamba-3 constructor take
`WeightDtype::{F32, Bf16, F16}` — the rest of the API is unchanged.

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::WeightDtype;

let mut gpu = GpuMambaBackbone::new_with_dtype(0, &weights, cfg, input_dim, batch, WeightDtype::Bf16)?;
gpu.capture_graph()?; // optional; ~2× decode speedup
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

### HuggingFace LM inference

```rust
use mamba_rs::module::gpu_lm::GpuMambaLM;
use mamba_rs::module::sample::SampleParams;
use mamba_rs::WeightDtype;
use std::path::Path;

let mut lm = GpuMambaLM::from_hf_with_dtype(
    Path::new("./mamba-130m-hf"), 0, WeightDtype::Bf16,
)?;
lm.capture_graph()?;
let tokens = lm.generate(&[1, 2, 3, 4, 5], &SampleParams::default())?;
```

bf16 vs f32 on all four cached `state-spaces/mamba-*-hf` checkpoints:
15/15 greedy match, KL ≤ 1.6e-3. Batch=1 vs batch=32 on the same prompt:
KL ≈ 2e-11 (bit-identical up to f32 roundoff of the fixed reduction tree).

## Quick start (GPU training)

`MambaTrainer` / `Mamba3Trainer` wrap the full forward + backward + AdamW +
sync pipeline behind a single `.step()` call. One dispatch struct per
architecture; an internal enum selects the f32 or mixed (bf16/f16) inner
engine based on the `WeightDtype` constructor argument.

```rust
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
use mamba_rs::WeightDtype;

let session = TrainSessionCfg {
    input_dim,
    batch: 2,
    seq_len: 64,
    lr: 3e-4,
    weight_decay: 1e-2,
};
let mut trainer = MambaTrainer::new_full(
    /* gpu_ordinal */ 0,
    &cpu_weights, cfg, session,
    WeightDtype::Bf16,
)?;
trainer.capture_graph()?; // optional; one cuGraphLaunch per step after this

let metrics = trainer.step(&input, &d_temporal_upstream)?;
// metrics.step, metrics.graph_replayed, metrics.loss_scale (f16), metrics.overflow_skipped (f16)

let master = trainer.snapshot_master()?; // CPU-side MambaWeights for checkpointing
```

`Mamba3Trainer` mirrors the same API. f16 training activates the dynamic
loss scaler automatically; `metrics.loss_scale` / `metrics.overflow_skipped`
report its state each step.

## Serialization

```rust
use mamba_rs::serialize;

serialize::save(Path::new("model.safetensors"), backbone.weights(), cfg, input_dim)?;
let (weights, cfg, input_dim) = serialize::load(Path::new("model.safetensors"))?;

// Mamba-3
use mamba_rs::mamba3_siso::serialize::{save_mamba3, load_mamba3};
save_mamba3(Path::new("m3.safetensors"), &weights, &cfg, input_dim)?;
let (weights, input_dim) = load_mamba3(Path::new("m3.safetensors"), &cfg)?;
```

## Performance (RTX 6000 Ada)

### LLM throughput — mamba-130m-hf, greedy decode, CUDA Graph, RTX 6000 Ada

| dtype | cuBLAS (default) | batch-invariant matvec | Δ |
|-------|-----------------:|-----------------------:|--:|
| f32   | 727 tok/s        | 727 tok/s              | 0 % |
| bf16  | **1 046 tok/s**  | 974 tok/s              | −7 % |
| f16   | 1 047 tok/s      | 972 tok/s              | −7 % |

On f32 the two paths are equivalent — cuBLAS SGEMM and the custom
matvec kernel both run on CUDA cores with no Tensor Core path. On
bf16/f16 cuBLAS routes through Tensor Cores (TF32-style accumulation)
and wins ~7 % on per-token latency, at the cost of M=1 vs M=N
algorithm-selection drift (KL ≈ 1e-3 on adversarial prompts). The
batch-invariant path keeps `b=1` ≡ `b=N` per slot (KL ≈ 1e-11).

Enable the batch-invariant path when cross-batch bit-identity matters
(KL ≈ 1e-11 between `b=1` and `b=N` per slot): set
`MAMBA_RS_BATCH_INVARIANT=1` or call `ctx.set_batch_invariant(true)`.

### Deterministic training — cost per step (RTX 6000 Ada, `MambaTrainer`)

With the batch-invariant flag on, every training GEMM (forward, dW, dX)
runs on custom fixed-reduction-order kernels: two runs with the same
seed/inputs produce bit-identical weights, on every dtype. The optional
tensor-core tier keeps full determinism under its own numeric contract
(mma.sync f32 accumulation instead of the scalar FMA chain) and turns the
determinism overhead into a speedUP on LLM-sized models:

| model | dtype | cuBLAS baseline | deterministic (scalar) | deterministic + TC |
|---|---|---:|---:|---:|
| d768, B=8 T=256  | bf16 | 25.6 ms (PEDANTIC) | 28.8 ms (1.13×) | **22.6 ms (0.88×)** |
| d1536, B=4 T=256 | bf16 | 17.5 ms (PEDANTIC) | 19.6 ms (1.12×) | **13.6 ms (0.77×)** |
| d1536, B=4 T=256 | f32  | 14.1 ms (TF32)     | 21.6 ms (1.53×) | — |
| d128 (RL), B=16 T=64 | bf16 | 2.1 ms (PEDANTIC) | 2.5 ms (1.20×) | 2.3 ms (1.10×) |

```rust
trainer.ctx().set_batch_invariant(true);   // bit-identical runs, scalar contract
trainer.ctx().set_bi_tensor_cores(true);   // + tensor-core tier (own contract)
```

GEMM-level tensor-core speedups vs the scalar deterministic tier: forward
3.0–3.4×, dW 2.1–3.5×, dX 4.3–6.7× (bf16, M=2048-class shapes). Full
tables and contracts: [deterministic GEMM benchmarks](docs/determinism-benchmarks.md).

### Per-step latency (default config: d_model=128, 3 layers)

| | Mamba SSM | Mamba-3 SISO |
|---|---|---|
| GPU inference B=1 (CUDA Graph) | **79 µs** | **86 µs** |
| GPU training fwd+bwd (T=32)    | 1 629 µs | 2 169 µs |
| CPU inference B=1              | 88 µs    | **70 µs** |
| CPU training fwd+bwd (T=32)    | 15 874 µs | **3 609 µs** |

Detailed tables: [Mamba SSM benchmarks](docs/mamba1-benchmarks.md),
[Mamba-3 SISO benchmarks](docs/mamba3-benchmarks.md).

## Testing

52 test files, 360+ individual tests:

- Correctness: bit-parity across CPU ↔ GPU, eager ↔ CUDA Graph, f32 ↔ bf16/f16
- Gradient checks: finite-difference vs analytical on every weight tensor
- Real checkpoints: 30-step training convergence + inference on
  `state-spaces/mamba-130m-hf` for all three dtypes
- Batch invariance: KL < 1e-4 across batch sizes 1 / 4 / 16 / 32 at bf16
- Determinism: bit-identical training across runs (f32/bf16/f16, scalar
  and tensor-core tiers), typed-GEMM bit-parity vs the f32 reference
  across a 60-shape dispatch-gate boundary sweep
- Long-sequence stability: 1024-token generation + T=1024 M3 training
- CUDA Graph: replay determinism, pointer-stability assertions

Run the fast suite:

```sh
cargo test --release --features cuda
```

Full suite including HuggingFace-backed tests (needs the HF cache):

```sh
cargo test --release --features "cuda hf" -- --include-ignored
```

## Documentation

- [Mamba SSM architecture](docs/mamba1-architecture.md)
- [Mamba-3 SISO architecture](docs/mamba3-architecture.md)
- [Mamba SSM benchmarks](docs/mamba1-benchmarks.md)
- [Mamba-3 SISO benchmarks](docs/mamba3-benchmarks.md)
- [Deterministic GEMM benchmarks](docs/determinism-benchmarks.md) — tiers,
  contracts, full measurement tables (training step, tensor-core GEMM
  level, fallback tax), reproduction commands

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
