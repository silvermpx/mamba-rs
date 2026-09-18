# mamba-rs

Mamba SSM and Mamba-3 SISO in Rust, on the CPU and on NVIDIA GPUs.
Inference and training for both architectures, with the crate's own CUDA
kernels compiled at run time through NVRTC. No Python, no C++ build step,
no framework dependency; the GPU path links only the CUDA driver API and
cuBLAS.

## What's new in 0.7.x

0.7.0 made the new deterministic GEMM kernels the default, optimized the
Mamba kernels around them, and moved the two scan and weight-gradient
routes to their faster families. A training step of the 0.7.0 release
shapes ran 1.13 to 1.63 times faster than 0.6.9 on an RTX 6000 Ada.

- **Every GPU context has a `GemmMode`.** `Deterministic` (the default)
  runs the crate's own kernels and never calls cuBLAS; `CublasFast` and
  `CublasPedantic` select cuBLAS explicitly. In the deterministic mode the
  same inputs give the same bits run after run, from an eager launch and
  from a captured graph, and for serving the same bits for a row at any
  batch size. To keep the exact numbers 0.6.9 produced, construct with
  `GemmMode::CublasPedantic`.
- **New deterministic training kernels** (the Triad family: forward, weight
  gradient and input gradient) and **new deterministic serving kernels**
  (the Inference family, formerly `Fixed`), measured kernel by kernel on an
  RTX 6000 Ada and an RTX 5090 against cuBLAS Fast and cuBLAS Pedantic, and
  against the 0.6.9 kernels: on the large training and serving shapes the
  new kernels are 1.26 to 1.43 times faster on average, single kernels up
  to 3 times.
- **Deterministic TF32**, a fourth storage precision, `WeightDtype::Tf32`:
  f32 storage whose products run on the tensor cores in TF32 with one fixed
  rounding and a fixed summation order, the accumulation stays f32, and the
  bits are reproducible like the rest of the deterministic mode. 0.6.9 had
  no such precision. Stream-K weight-gradient kernels serve the deep
  reductions by default.
- **Explicit-mode constructors** beside every environment-reading one, and
  a recorded numeric route on every captured graph.
- **The measured routes reach every card.** The specialized GEMM modules
  were compiled for one target and one board; every kernel in them uses
  the SM80 instruction tier and nothing above it, so each is now built
  for the device's own target on any SM80-tier board. A board with frozen
  evidence takes a route as a measured winner; any other board admits it
  by a first-use bit proof against the reference route of the same
  numeric contract, and declines it aloud if a single output word
  differs. The bits are guaranteed everywhere; the speed is the speed of
  the board the route was measured on. One behaviour changes off the Ada:
  `WeightDtype::Tf32` now serves the portable deterministic TF32 tier
  there instead of falling back to the exact f32 kernels, on the measured
  cells and, within a factor of four on every dimension, on their
  neighbourhood.

The full list and patch-release changes are in [CHANGELOG.md](CHANGELOG.md);
the mode guide is [docs/gemm-modes.md](docs/gemm-modes.md); the numbers are in
[docs/determinism-benchmarks.md](docs/determinism-benchmarks.md).

The 0.7.x releases continue optimizing the scan, convolution and norm
kernels around the GEMMs, and the GEMM kernels toward cuBLAS Fast.
The boards without kernels of their own run the Ada-measured routes under
the first-use bit proof: the A100 and the H100 are timed that way in 0.7.3
(see the changelog's board table); the native Hopper and Blackwell paths,
each measured on its own board, are the work of the next releases.

## Features

- Mamba SSM (Gu and Dao, 2023) and Mamba-3 SISO (Lahoti et al., 2026).
- CPU and GPU paths for both, with cross-path parity tests on shared
  weights.
- Inference and training: full backward pass through the recurrent SSM
  state, AdamW, CUDA Graph capture for inference steps, prefill and
  training steps.
- f32, tf32, bf16 and f16 storage through one `WeightDtype` selector; every
  kernel accumulates in f32.
- Three GEMM modes, deterministic by default. See below.
- Bring-your-own-loss training: `trainer.forward()` returns the full
  temporal output on the host, any loss gradient computed in Rust goes into
  `trainer.backward_step()`, bit-identical to the fused `step()`.
- Full-sequence CPU prefill for both architectures, and a GPU prompt
  prefill for Mamba-3 with a captured-graph twin.
- Deterministic data-parallel training over NCCL (`dist`): one reduction
  per optimizer step in a fixed order, so the bits do not depend on the
  transport or the GPU permutation. The fixed-order path is pinned against
  an in-process oracle on one GPU; the NCCL collective path was validated
  on two RTX 5090s.
- Bit-continuous resume: optimizer state and the carried recurrence export
  and import, so a resumed run lands where the unbroken run would.
- State dimensions up to 256, sized at compile time from the config.
- HuggingFace safetensors loader for Mamba SSM checkpoints (130m, 370m,
  1.4b and 2.8b validated).
- MSRV 1.97.

## Cargo features

| feature | what it enables | when to use it |
|---|---|---|
| *(default)* | pure-Rust scalar GEMM | correctness work only; much slower than a BLAS |
| `gemm-blas` | the [`gemm`] crate's BLAS-class CPU GEMM (with rayon) | any serious CPU use |
| `accelerate` | Apple Accelerate GEMM (macOS) | macOS deployments |
| `cuda` | GPU inference and training (NVRTC-compiled kernels) | needs the CUDA toolkit |
| `hf` | safetensors and HuggingFace checkpoint loaders | LM checkpoints |
| `cli` | the `mamba-generate` binary (tokenizers and hf-hub) | text generation from the command line |
| `nccl` | data-parallel transport (pinned NCCL binding) | multi-GPU training |
| `qualification` | the hardware and toolkit instruments under `tools/qualification/` | maintainers measuring kernels on a chosen board |
| `cuda-cublaslt-qualification` | cuBLASLt in the vendor-comparison harness | maintainers only; production routing does not use cuBLASLt |

The `cuda` feature binds the toolkit through cudarc, which reads the
toolkit version from `nvcc` at build time and lists the versions it
knows: cudarc 0.19.9 knows up to CUDA 13.3. On a CUDA 13.4 box build with
`CUDARC_CUDA_VERSION=13030`: the 13.3 bindings, the 13.4 libraries loaded
at run time. The kernels themselves are compiled by the installed NVRTC,
so the targets that need 13.4 (CC 10.7) come with it.

## GEMM modes and storage precision

A GPU context has two settings. `GemmMode` decides who multiplies;
`WeightDtype` decides how the weights are stored and, with it, the
precision of every product. There is nothing else to set: inside the
deterministic mode the kernels are chosen from the two, and from whether
the context serves a model or a trainer.

| you want | use |
|---|---|
| reproducible results: the same bits run to run, eager or graph, and for serving at any batch size | `GemmMode::Deterministic` (default) |
| the fastest vendor path, TF32 permitted for f32 | `GemmMode::CublasFast` |
| the numbers 0.6.9 produced, or the vendor's most careful f32 accumulation as a reference | `GemmMode::CublasPedantic` |

| `WeightDtype` | storage | products in the deterministic mode |
|---|---|---|
| `F32` (default) | f32 | exact: one fused multiply-add per step in a fixed order, f32 throughout |
| `Tf32` | f32 | deterministic TF32: each operand rounded to TF32 once, a fixed summation order, f32 accumulation; a shape or board without a measured TF32 kernel takes the exact f32 kernel. This is not the vendor's Fast TF32; it reproduces its bits like the rest of the mode |
| `Bf16` | bf16 | tensor-core kernels with f32 accumulation |
| `F16` | f16 | tensor-core kernels with f32 accumulation; training runs the dynamic loss scaler |

Every GPU entry point has a plain constructor that reads `MAMBA_RS_GEMM_MODE`
(`deterministic`, `cublas-fast`, `cublas-pedantic`; default
`deterministic`), the one environment variable the crate reads for its
GEMMs, and a `*_with_mode` twin that takes the mode as its last argument
and ignores the environment. The storage precision is always an argument.

```rust
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
use mamba_rs::mamba_ssm::gpu::device::GpuDevice;

let device = GpuDevice::new(0)?;
let ctx = GpuCtx::new_with_mode(&device, GemmMode::Deterministic)?;
assert_eq!(ctx.gemm_mode(), GemmMode::Deterministic);
ctx.set_gemm_mode(GemmMode::CublasPedantic)?; // refused while a graph is being captured
```

Inside the deterministic mode a model context uses the Inference kernels
and a trainer the Triad kernels. What each mode guarantees, what it does
not, and the architecture coverage are in
[docs/gemm-modes.md](docs/gemm-modes.md).

## Use cases and API choice

### Reinforcement learning and small custom models

Latency-critical, typically `d_model` up to 256, often batch 1 for actor
rollouts. Both CPU and GPU paths apply; at these sizes the CPU step and the
GPU step are within a factor of two of each other, so the choice depends
on where the rest of the program lives.

- Inference: `mamba_step` (CPU) or `GpuMambaBackbone::step` (GPU)
- Training: `parallel_mamba_forward` and `parallel_mamba_backward` (CPU,
  rayon-parallel over the batch) or `MambaTrainer::step` (GPU, one
  captured graph for forward, backward, AdamW and sync)

CPU training is practical where GPU launch overhead dominates (`d_model`
up to 128, batch up to 8); GPU training scales well from batch 32.

### Large language models

Throughput-critical, `d_model` from 768, token-by-token decoding of a
HuggingFace checkpoint. GPU only in practice: a 2.8b model on the CPU
decodes at single-digit tokens per second whatever the implementation.

- Inference: `GpuMambaLM::from_hf_with_dtype` and `generate`
- Fine-tuning: `MambaTrainer::new_full` on the HuggingFace backbone
  weights (Mamba SSM only; no public Mamba-3 SISO checkpoint exists)

The CPU `MambaLM` path runs end to end but exists for CPU-versus-GPU
parity tests, not for serving.

### Sequence classification, embeddings and custom heads

Whole-sequence reads with a caller-defined loss (document classifiers,
distillation, contrastive embeddings). GPU training uses the
forward/backward split; CPU serving uses the prefill.

- Training: `MambaTrainer::forward`, a host-side loss,
  `MambaTrainer::backward_step` (see `examples/custom_loss.rs`)
- CPU serving: `MambaBackbone::forward_prefill` and
  `forward_mamba3_backbone_prefill` (see `examples/cpu_prefill.rs`), or
  `prefill_batch` and `prefill3_batch` for batches of sequences

### Sharing weights across paths

All paths consume the same `MambaWeights` or `Mamba3Weights`. A training
run's `MambaTrainer::snapshot_master()` loads directly into
`GpuMambaBackbone`, `GpuMambaLM` or the CPU `MambaBackbone` without
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

Enable `gemm-blas` (or `accelerate` on macOS) for any CPU work beyond a
correctness check.

## Quick start (GPU inference)

```toml
[dependencies]
mamba-rs = { version = "0.7", features = ["cuda"] }
```

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::WeightDtype;

let mut gpu = GpuMambaBackbone::new_with_dtype_and_mode(
    0, &weights, cfg, input_dim, batch, WeightDtype::Bf16, GemmMode::Deterministic,
)?;
gpu.capture_graph()?; // optional: one graph launch per step
gpu.step(&input, &mut output)?;
gpu.reset()?;
```

`GpuMamba3Backbone` has the same constructor. `new_with_dtype` reads
`MAMBA_RS_GEMM_MODE` instead.

### HuggingFace LM inference

```rust
use mamba_rs::module::gpu_lm::GpuMambaLM;
use mamba_rs::module::sample::SampleParams;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::WeightDtype;
use std::path::Path;

let mut lm = GpuMambaLM::from_hf_with_dtype_and_mode(
    Path::new("./mamba-130m-hf"), 0, WeightDtype::Bf16, GemmMode::Deterministic,
)?;
lm.capture_graph()?;
let tokens = lm.generate(&[1, 2, 3, 4, 5], &SampleParams::default())?;
```

bf16 against f32 on the four `state-spaces/mamba-*-hf` checkpoints: 15 of
15 greedy tokens match and the KL divergence of the final logits is at
most 1.1e-3 (`tests/gpu_bf16_parity.rs`). Batch 1 against batch 32 on the
same prompt in the deterministic mode: KL about 8e-11
(`tests/extreme_edge_coverage.rs`), the batch invariance of the Inference
kernels measured through a whole model; both tests need a local checkpoint
and run with `--ignored`.

## Quick start (GPU training)

`MambaTrainer` and `Mamba3Trainer` run forward, backward, AdamW and the
master-weight sync behind one `step()` call. The `WeightDtype` argument
selects the f32 engine (`F32`, or `Tf32` for f32 storage with deterministic
TF32 products) or the mixed bf16/f16 engine.

```rust
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::mamba_ssm::gpu::trainer::{MambaTrainer, TrainSessionCfg};
use mamba_rs::WeightDtype;

let session = TrainSessionCfg {
    input_dim,
    batch: 2,
    seq_len: 64,
    lr: 3e-4,
    weight_decay: 1e-2,
};
let mut trainer = MambaTrainer::new_full_with_mode(
    /* gpu_ordinal */ 0,
    &cpu_weights, cfg, session,
    WeightDtype::Bf16,
    GemmMode::Deterministic,
)?;
trainer.capture_graph()?; // optional; one graph launch per step after this

let metrics = trainer.step(&input, &d_temporal_upstream)?;
// metrics.step, metrics.graph_replayed, metrics.loss_scale (f16), metrics.overflow_skipped (f16)

let master = trainer.snapshot_master()?; // CPU-side MambaWeights for checkpointing
```

f16 training activates the dynamic loss scaler automatically;
`metrics.loss_scale` and `metrics.overflow_skipped` report its state.

### Custom losses: the forward/backward split

The fused `step()` needs the loss gradient up front; the split lets you
compute it from the actual forward output:

```rust
use mamba_rs::mamba_ssm::gpu::trainer::BackwardOpts;

let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
trainer.forward(&input, &mut temporal)?;          // full temporal output, f32, on the host
let d_temporal = my_loss_grad(&temporal);          // any host-side loss
let m = trainer.backward_step(
    &d_temporal,
    BackwardOpts::default().with_clip_max_norm(1.0),
)?;                                                // backward, clipping, AdamW
// gradient accumulation: .with_accumulate_only(true) on the non-applying
// micro-batches; the fused step() refuses while a window is open.
```

The split always runs eagerly, because a caller-side loss cannot live
inside a captured graph, and it is bit-identical to the fused `step()`.
`examples/custom_loss.rs` is a complete training loop.

## Quick start (CPU prefill)

```rust
use mamba_rs::inference::PrefillMode;
use mamba_rs::module::MambaBackbone;

let backbone = MambaBackbone::init(cfg, input_dim, 42);
let mut state = backbone.alloc_state();
let mut scratch = backbone.alloc_prefill_scratch(seq_len);
let mut out = vec![0.0f32; seq_len * backbone.config().d_model];

// One batched GEMM pass over the whole prompt instead of T step calls.
backbone.forward_prefill(&prompt, &mut out, &mut state, &mut scratch,
                         seq_len, PrefillMode::Parallel);
// `out` holds the post-norm output at every position; `state` is
// positioned after the prompt, so forward_step continues from it.
```

Mamba-3 has the same surface (`forward_mamba3_backbone_prefill` and
`Mamba3PrefillScratch`).

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

## Performance

Measured on an RTX 6000 Ada (SM89, driver 595.45.04) and an RTX 5090
(CC 12.0, driver 595.84 for the serving tables and 595.58.03 for the
training tables), both on CUDA 13.2. Speedups are cuBLAS time
divided by mamba-rs time; above 1.0 the deterministic kernel is faster.
Exact f32 is compared with cuBLAS Pedantic, which performs the same
arithmetic; bf16 and f16 with cuBLAS Fast on the native half-precision
tensor-core kernels; deterministic TF32 with cuBLAS Fast TF32.

Serving kernels (the Inference family, five shapes, bias off and on,
geometric mean, eager path):

| input → output | compared with | RTX 6000 Ada | RTX 5090 |
|---|---|---:|---:|
| BF16 → BF16 | cuBLAS Fast | 1.19× | 1.24× |
| F16 → F16 | cuBLAS Fast | 1.17× | 1.23× |
| BF16 → F32 | cuBLAS Fast | 0.83× | 1.29× |
| F32 stored as `Tf32` (deterministic TF32) | cuBLAS Fast TF32 | 0.90× | 1.13× |
| F32, exact | cuBLAS Pedantic | 1.00× | 1.08× |

Training kernels (the Triad family, the large shapes, geometric mean,
eager path; the small d128 shapes are launch-bound and slower than
cuBLAS on both boards):

| precision | compared with | RTX 6000 Ada | RTX 5090 |
|---|---|---:|---:|
| BF16 | cuBLAS Fast | 1.07× | 1.06× |
| F16 | cuBLAS Fast | 1.08× | 1.06× |
| F32 stored as `Tf32` (deterministic TF32) | cuBLAS Fast TF32 | 0.84× | 1.07× |
| F32, exact | cuBLAS Pedantic | 0.94× | 1.17× |

Whole training step on the RTX 6000 Ada, 0.7.0 against 0.6.9, same shapes
and settings in both trees (bf16 with tensor cores, ms per step): d128
2.35 → 1.94, d256 9.54 → 7.66, d768 22.17 → 13.60, d1536 13.07 → 9.45.
The step is dominated by the scan and the other non-GEMM kernels, so the
whole-step gain is smaller than the kernel gain.

The per-kernel tables for both boards, the old-versus-new kernel
comparison, the whole-model comparison and the measurement protocol are in
[docs/determinism-benchmarks.md](docs/determinism-benchmarks.md). The
whole-model tables of earlier releases stay on the
[Mamba SSM](docs/mamba1-benchmarks.md) and
[Mamba-3 SISO](docs/mamba3-benchmarks.md) benchmark pages with their
release, board and comparator labels.

## Testing

Every target is declared in `Cargo.toml` and has a lane in
`qual/lanes.toml`; a host test keeps the two in step.

- **Regressions** (`tests/`): bit parity within a numeric route (eager and
  graph, run and run), tolerance parity across routes (CPU and GPU, f32
  and half, scalar and tensor-core), gradient checks, source and dispatch
  contracts, batch invariance and determinism gates.
  `cargo test --release --features cuda` runs the CUDA gate;
  `cargo test --no-default-features` runs the host part on any machine.
  Tests marked `contract` contain arms that need a checkpoint or a
  specific board and run with `-- --ignored`; tests marked `record`
  write evidence and never run automatically.
- **Benches** (`benches/`): timing instruments with no verdict,
  `cargo bench --features cuda --bench <name>`, optionally followed by
  `-- <instrument>`.
- **Qualification tools** (`tools/qualification/`): hardware, toolkit and
  inventory instruments that need a specific board, built with
  `--features "cuda hf qualification"` and run by name with `-- --ignored`.

`qual/run.sh <lane>` runs or lists one lane;
[docs/release-qualification.md](docs/release-qualification.md) describes
the release order.

## Documentation

For users:

- [GEMM modes](docs/gemm-modes.md): the three modes and the four storage
  precisions, which to choose, how to set them, what is guaranteed, the
  one environment variable, architecture coverage
- [GEMM benchmarks](docs/determinism-benchmarks.md): kernel-by-kernel
  timings on both boards against cuBLAS Fast and Pedantic, the 0.6.9
  comparison, the protocol
- [Mamba SSM architecture](docs/mamba1-architecture.md) and
  [benchmarks](docs/mamba1-benchmarks.md)
- [Mamba-3 SISO architecture](docs/mamba3-architecture.md) and
  [benchmarks](docs/mamba3-benchmarks.md)
- Rustdoc: `GemmMode`, `WeightDtype`, `GpuCtx::new_with_mode`,
  `GpuCtx::set_gemm_mode` and the `*_with_mode` constructors carry the API
  contract

For contributors:

- [Performance playbook](docs/performance-playbook.md): how kernels are
  measured, changed and admitted
- [Release qualification](docs/release-qualification.md): test lanes,
  package inspection, evidence and the release order

## Roadmap

- Multi-GPU inference for models larger than one device (pipeline
  sharding), beside the data-parallel training that ships now.
- Reduced-precision tiers (fp8, int8) under the same bit discipline as
  the f32, tf32, bf16 and f16 paths.
- The Mamba-2 generation beside Mamba-1 and Mamba-3, with the same
  determinism and testing discipline.

## Citation

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
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
