# mamba-rs

The four Mamba families in Rust, on the CPU and on NVIDIA GPUs: Mamba,
Mamba-2, Mamba-3 SISO and Mamba-3 MIMO. Every family runs inference and
training on both, with the crate's own CUDA kernels compiled at run time
through NVRTC. There is no Python, no C++ build step and no framework
underneath. The GPU path links only the CUDA driver API and cuBLAS.

## What's new in 0.8

- **Mamba-2 and Mamba-3 MIMO** join Mamba and Mamba-3 SISO. Both run on
  the CPU and the GPU, for inference and training, and load the official
  checkpoints. `state-spaces/mamba2-130m` and the Mamba-3 SISO and MIMO
  187m models match float64 reference implementations of the papers'
  models on the CPU and on the GPU.
- **One API for all four.** A GPU backbone, trainer or language model is
  built the same way in every family, with one `GpuOptions` value for the
  device, the storage precision and the GEMM mode. The `GpuBackbone` and
  `GpuTrainer` traits name what they share, and `MambaGpuLM::from_hf`
  reads the family from the checkpoint. The old constructors of Mamba and
  SISO stay for this release as deprecated wrappers. The
  [changelog](CHANGELOG.md) lists every rename.
- **The Mamba kernels compile as their own module**, apart from the GEMM
  kernels and the kernels the families share. They were reworked for this
  release: the convolution walks shallow tiles with batched loads, and the
  residual add of a layer rides in the next layer's norm.

## Families

| | paper | CPU | GPU | checkpoints |
|---|---|---|---|---|
| Mamba | Gu and Dao, 2023 | inference, training | inference, training | `state-spaces/mamba-*-hf`, 130m to 2.8b |
| Mamba-2 | Dao and Gu, 2024 | inference, training | inference, training | `state-spaces/mamba2-*`, 130m checked against float64 |
| Mamba-3 SISO | Lahoti et al., 2026 | inference, training | inference, training | the official SISO checkpoints, 187m checked against float64 |
| Mamba-3 MIMO | Lahoti et al., 2026 | inference, training | inference, training | the official MIMO checkpoints, 187m checked against float64 |

## Features

- CPU and GPU paths for every family, held to each other by parity tests
  on shared weights.
- Inference: single steps, prompts in one pass, and the decode step and
  the prompt pass of one length as CUDA graphs.
- Training on the CPU and the GPU: the full backward through the
  recurrent state, AdamW, global and control-channel norm clipping, the
  skip of a spiking window and gradient accumulation. Bring your own loss:
  `forward` gives the full output on the host, and the gradient of any
  loss goes back through `backward_step`.
- f32, tf32, bf16 and f16 storage through one `WeightDtype`. Every kernel
  accumulates in f32.
- Three GEMM modes, deterministic by default. See below.
- Deterministic data-parallel training over NCCL for every family, with
  one reduction per optimizer step in a fixed order.
- Bit-continuous resume for every family: the optimizer state, the f16
  loss scale and the carried recurrence export and import, and a
  trainer-state file holds them beside the weights.
- Safetensors checkpoints for every family, a language-model checkpoint
  of any family in one file, and loaders for the official HuggingFace and
  `mamba_ssm` checkpoints.
- State dimensions up to 256.
- MSRV 1.98.1.

## Cargo features

| feature | what it enables | when to use it |
|---|---|---|
| *(default)* | pure-Rust scalar GEMM | correctness work only, much slower than a BLAS |
| `gemm-blas` | the [`gemm`] crate's BLAS-class CPU GEMM (with rayon) | any serious CPU use |
| `accelerate` | Apple Accelerate GEMM (macOS) | macOS deployments |
| `cuda` | GPU inference and training (NVRTC-compiled kernels) | needs the CUDA toolkit |
| `hf` | checkpoint loaders and the language models | text generation, fine-tuning |
| `cli` | the `mamba-generate` binary (tokenizers and hf-hub) | text generation from the command line |
| `nccl` | data-parallel transport (pinned NCCL binding) | multi-GPU training |
| `qualification` | the hardware and toolkit instruments under `tools/qualification/` | maintainers measuring kernels on a chosen board |
| `cuda-cublaslt-qualification` | cuBLASLt in the vendor-comparison harness | maintainers only, production routing never uses cuBLASLt |

The `cuda` feature binds the toolkit through cudarc, which reads the
toolkit version from `nvcc` at build time and knows the versions up to
CUDA 13.3. On a CUDA 13.4 box build with `CUDARC_CUDA_VERSION=13030`,
which takes the 13.3 bindings and loads the 13.4 libraries at run time.
The kernels themselves are compiled by the installed NVRTC, so the targets
that need 13.4 (CC 10.7) come with it.

## Quick start (CPU)

Every family has the same calls. Here is Mamba-2:

```rust
use mamba_rs::{Mamba2Backbone, Mamba2Config};

let cfg = Mamba2Config::default();
let backbone = Mamba2Backbone::init(cfg, input_dim, 42)?;
let mut state = backbone.alloc_state();
let mut scratch = backbone.alloc_scratch();

// One token at a time.
let mut output = vec![0.0f32; cfg.d_model];
backbone.forward_step(&input, &mut output, &mut state, &mut scratch);

// Or a whole window: every position's output, and the state after it.
let mut outputs = vec![0.0f32; seq_len * cfg.d_model];
backbone.forward_sequence(&prompt, &mut outputs, &mut state, &mut scratch, seq_len);
```

The others are `MambaBackbone`, `Mamba3Backbone` and `Mamba3MimoBackbone`.
Mamba-2 and MIMO check their configuration in `init` and return a
`Result`, Mamba and SISO return the backbone itself. `new(weights, cfg,
input_dim)` wraps weights loaded from a checkpoint or a trainer.

Every family also trains on the CPU, through the trainers
`MambaCpuTrainer`, `Mamba2CpuTrainer`, `Mamba3CpuTrainer` and
`Mamba3MimoCpuTrainer`. They take the GPU trainers' session, options and
metrics and have their calls (`forward`, `backward_step`, `step`, the
accumulation window, `apply_step_full`, the optimizer and trainer state),
and the `CpuTrainer` trait and `AnyCpuTrainer` name what they share. The
sequences of a batch run side by side and their gradients sum in a fixed
number of chunks, so a run gives the same bits on any thread pool.

```rust
use mamba_rs::{BackwardOpts, Mamba2CpuTrainer, TrainSessionCfg};

let session = TrainSessionCfg::new(input_dim, batch, seq_len);
let mut trainer = Mamba2CpuTrainer::new(&weights, cfg, session)?;
let mut out = vec![0.0f32; batch * seq_len * cfg.d_model];
trainer.forward(&input, &mut out)?;
trainer.backward_step(&my_loss_grad(&out), BackwardOpts::default().with_clip_max_norm(1.0))?;
let master = trainer.snapshot_master();
```

Enable `gemm-blas`, or `accelerate` on macOS, for any CPU work beyond a
correctness check.

## Quick start (GPU)

```toml
[dependencies]
mamba-rs = { version = "0.8", features = ["cuda"] }
```

```rust
use mamba_rs::{GpuOptions, Mamba3MimoGpuBackbone, WeightDtype};

let opts = GpuOptions::default().dtype(WeightDtype::Bf16);
let mut gpu = Mamba3MimoGpuBackbone::new(&weights, cfg, input_dim, batch, opts)?;

gpu.capture_prefill_graph(seq_len)?; // optional: a prompt of this length is one launch
gpu.capture_graph()?;              // optional: each step is then one graph launch
gpu.prefill(&prompt, seq_len)?;    // a prompt in one pass
gpu.step(&input, &mut output)?;    // then token by token
gpu.reset()?;                      // a new sequence
```

`GpuOptions::default()` is device 0, f32 storage and the GEMM mode the
`MAMBA_RS_GEMM_MODE` variable names (deterministic when it is unset).
`.device(n)`, `.dtype(d)` and `.mode(m)` change them. The backbones of the
other families take the same arguments: `MambaGpuBackbone`,
`Mamba2GpuBackbone` and `Mamba3GpuBackbone`. `step_gpu_only` leaves the
output on the device, where `temporal_ptr` points at it for a head that
runs on the GPU. A captured graph fixes the GEMM staging, so run or
capture the longest prompt before `capture_graph` when the prompts are
long. `capture_prefill_graph` keeps one prompt length; a prompt of another
length runs eagerly and drops it.

Code that takes any family uses the `GpuBackbone` trait, and
`AnyGpuBackbone` holds a backbone of a family chosen at run time.
`examples/families.rs` serves and trains all four families through the
two traits.

### Text generation

```rust
use mamba_rs::{GpuOptions, MambaGpuLM, SampleParams, WeightDtype};
use std::path::Path;

let opts = GpuOptions::default().dtype(WeightDtype::Bf16);
let mut lm = MambaGpuLM::from_hf(Path::new("./mamba2-130m"), 1, opts)?;
lm.capture_graph()?;
let tokens = lm.generate(&prompt_ids, &SampleParams::default())?;
```

`from_hf` reads the family from the checkpoint's `config.json`, so the same
line loads a Mamba, Mamba-2, SISO or MIMO model. `MambaGpuLM::new(backbone,
embed, lm_head, vocab_size)` wraps a backbone built in code, and a batch
above one generates several sequences side by side with
`generate_batch`. The CPU `MambaLM::from_hf` loads the same checkpoints.
The `mamba-generate` binary (feature `cli`) runs any of them from the
command line; with `--gpu` it takes `--dtype`, `--gemm-mode`,
`--scan-mode` and `--batch` (continuations of one prompt in one batched
run), which the CPU path, f32 and one sequence, refuses.

bf16 against f32 on the four `state-spaces/mamba-*-hf` checkpoints: 15 of
15 greedy tokens match and the KL divergence of the final logits is at
most 1.1e-3 (`tests/gpu_bf16_parity.rs`). Batch 1 against batch 32 on the
same prompt in the deterministic mode: KL about 8e-11
(`tests/extreme_edge_coverage.rs`). Both tests need a local checkpoint
and run with `--ignored`.

## Quick start (GPU training)

```rust
use mamba_rs::{BackwardOpts, GpuOptions, Mamba2GpuTrainer, TrainSessionCfg, WeightDtype};

let session = TrainSessionCfg {
    lr: 3e-4,
    weight_decay: 1e-2,
    ..TrainSessionCfg::new(input_dim, batch, seq_len)
};
let opts = GpuOptions::default().dtype(WeightDtype::Bf16);
let mut trainer = Mamba2GpuTrainer::new(&weights, cfg, session, opts)?;

// The loss gradient known up front: forward, backward and AdamW in one call.
let metrics = trainer.step(&input, &d_temporal)?;

// A loss computed from the output: forward, the loss on the host, then
// the backward and the optimizer step.
let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
trainer.forward(&input, &mut temporal)?;
let d_temporal = my_loss_grad(&temporal);
trainer.backward_step(&d_temporal, BackwardOpts::default().with_clip_max_norm(1.0))?;

let master = trainer.snapshot_master()?; // f32 weights for a checkpoint or the CPU model
```

`MambaGpuTrainer`, `Mamba3GpuTrainer` and `Mamba3MimoGpuTrainer` are built
and driven the same way, and the `GpuTrainer` trait names what they share.
`BackwardOpts` also skips a window whose gradient norm spikes
(`with_step_skip_above`) and accumulates micro-batches
(`with_accumulate_only`). f16 training runs a dynamic loss scaler whose
scale holds for a whole accumulation window; an overflow in any
micro-batch skips the window, and the metrics report the scale and the
skipped steps. Every trainer trains data-parallel (`backward_step_dist`,
or `backward_step`, `grad_arena` and `apply_step` around a reducer of the
caller's own), captures the whole step as one CUDA graph
(`capture_graph`), and exports its optimizer state and carried
recurrence for a bit-continuous resume. `AnyGpuTrainer` holds a trainer of
any family behind the `GpuTrainer` trait.
`examples/custom_loss.rs` is a complete training loop.

## Checkpoints

```rust
use mamba_rs::mamba2::serialize::{load, save};
use std::path::Path;

save(Path::new("model.safetensors"), &weights, &cfg, input_dim)?;
let (weights, cfg, input_dim) = load(Path::new("model.safetensors"))?;
```

Each family has the same two functions in its own module:
`mamba1::serialize`, `mamba2::serialize`, `mamba3_siso::serialize` and
`mamba3_mimo::serialize`. The file carries the configuration and the input
width, so `load` needs nothing else. The official checkpoints load through
`hf::load_hf` (Mamba), `hf::load_mamba2::load_hf_mamba2` and
`hf::load_mamba3::{load_hf_mamba3, load_hf_mamba3_mimo}`, or through
`from_hf` of the language models.

What a trainer needs beside its weights to resume bit for bit (the AdamW
moments and settings, the f16 loss scale and the carried recurrence)
goes to a trainer-state file of its own:

```rust
use mamba_rs::{GpuTrainer, TrainerState};
use std::path::Path;

trainer.trainer_state()?.save(Path::new("trainer.safetensors"))?;
// Later, on a trainer built from the saved weights:
trainer.load_trainer_state(&TrainerState::load(Path::new("trainer.safetensors"))?)?;
```

The CPU and GPU trainers of a family hold the state in one layout, so a
state either device saved resumes on the other.

`MambaLM::save_checkpoint` writes a language model of any family to one
file: the backbone as its family's checkpoint, with the embedding, the
untied head and the vocabulary size beside it. `MambaLM::from_checkpoint`
and `MambaGpuLM::from_checkpoint` load it.

## GEMM modes and storage precision

A GPU context has two settings. `GemmMode` decides who multiplies, and
`WeightDtype` decides how the weights are stored and with it the
precision of every product. There is nothing else to set. Inside the
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
| `Tf32` | f32 | deterministic TF32: each operand rounded to TF32 once, a fixed summation order and f32 accumulation. A shape or board without a measured TF32 kernel takes the exact f32 kernel. This is not the vendor's Fast TF32, it reproduces its bits like the rest of the mode |
| `Bf16` | bf16 | tensor-core kernels with f32 accumulation |
| `F16` | f16 | tensor-core kernels with f32 accumulation, and training runs the dynamic loss scaler |

The mode comes from `GpuOptions::mode`, or from `MAMBA_RS_GEMM_MODE`
(`deterministic`, `cublas-fast` or `cublas-pedantic`) when the options
leave it unset. That variable is the only one the crate reads for its
GEMMs. `GpuCtx::set_gemm_mode` changes the mode of a live context and is
refused while a graph is being captured.

Inside the deterministic mode a model context uses the Inference kernels
and a trainer the Triad kernels. What each mode guarantees, what it does
not, and the architecture coverage are in
[docs/gemm-modes.md](docs/gemm-modes.md).

## Performance

Measured for 0.7 on an RTX 6000 Ada (SM89, driver 595.45.04) and an RTX
5090 (CC 12.0, driver 595.84 for the serving tables and 595.58.03 for the
training tables), both on CUDA 13.2. A speedup is the cuBLAS time divided
by the mamba-rs time, so above 1.0 the deterministic kernel is faster.
Exact f32 is compared with cuBLAS Pedantic, which performs the same
arithmetic. bf16 and f16 are compared with cuBLAS Fast on the native
half-precision tensor-core kernels, and deterministic TF32 with cuBLAS Fast
TF32.

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
eager path. The small d128 shapes are bound by launches and slower than
cuBLAS on both boards):

| precision | compared with | RTX 6000 Ada | RTX 5090 |
|---|---|---:|---:|
| BF16 | cuBLAS Fast | 1.07× | 1.06× |
| F16 | cuBLAS Fast | 1.08× | 1.06× |
| F32 stored as `Tf32` (deterministic TF32) | cuBLAS Fast TF32 | 0.84× | 1.07× |
| F32, exact | cuBLAS Pedantic | 0.94× | 1.17× |

The per-kernel tables for both boards, the whole-model comparison and the
measurement protocol are in
[docs/determinism-benchmarks.md](docs/determinism-benchmarks.md). The
whole-model tables of earlier releases stay on the
[Mamba](docs/mamba1-benchmarks.md) and
[Mamba-3 SISO](docs/mamba3-benchmarks.md) benchmark pages with their
release, board and comparator labels.

## Testing

Every target is declared in `Cargo.toml` and has a lane in
`qual/lanes.toml`, and a host test keeps the two in step.

- **Regressions** (`tests/`): bit parity within a numeric route (eager
  and graph, run and run), tolerance parity across routes (CPU and GPU,
  f32 and half, scalar and tensor-core), gradient checks against finite
  differences, the official checkpoints against float64 references,
  source and dispatch contracts, batch invariance and determinism gates.
  `cargo test --release --features cuda` runs the CUDA gate, and
  `cargo test --no-default-features` runs the host part on any machine.
  Tests marked `contract` hold arms that need a checkpoint or a specific
  board and run with `-- --ignored`. Tests marked `record` write evidence
  and never run on their own.
- **Benches** (`benches/`): timing instruments with no verdict,
  `cargo bench --features cuda --bench <name>`, optionally followed by
  `-- <instrument>`.
- **Qualification tools** (`tools/qualification/`): hardware, toolkit and
  inventory instruments for a specific board, built with
  `--features "cuda hf qualification"` and run by name with `-- --ignored`.

`qual/run.sh <lane>` runs or lists one lane.
[docs/release-qualification.md](docs/release-qualification.md) describes
the release order.

## Documentation

For users:

- [GEMM modes](docs/gemm-modes.md): the three modes and the four storage
  precisions, which to choose and how to set them, what is guaranteed,
  the one environment variable, the architecture coverage
- [GEMM benchmarks](docs/determinism-benchmarks.md): kernel-by-kernel
  timings on both boards against cuBLAS Fast and Pedantic, and the
  protocol
- [Mamba architecture](docs/mamba1-architecture.md) and
  [benchmarks](docs/mamba1-benchmarks.md)
- [Mamba-2 architecture](docs/mamba2-architecture.md)
- [Mamba-3 SISO architecture](docs/mamba3-architecture.md) and
  [benchmarks](docs/mamba3-benchmarks.md)
- [Mamba-3 MIMO architecture](docs/mamba3-mimo-architecture.md)
- The rustdoc of `GpuOptions`, `GemmMode`, `WeightDtype`, the
  `GpuBackbone` and `GpuTrainer` traits and the family types, which carries
  the API contract

For contributors:

- [Performance playbook](docs/performance-playbook.md): how kernels are
  measured, changed and admitted
- [Release qualification](docs/release-qualification.md): test lanes,
  package inspection, evidence and the release order

## Roadmap

- Inference across several GPUs for models larger than one device, beside
  the data-parallel training that ships now.
- Reduced-precision tiers (fp8, int8) under the same bit discipline as the
  f32, tf32, bf16 and f16 paths.

## Citation

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning},
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

[`gemm`]: https://crates.io/crates/gemm
