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
  / `ctx.set_batch_invariant(true)` routes the forward, dW and dX GEMMs and
  the M<128 typed decode matvec through custom deterministic kernels:
  inference logits are bit-identical across batch sizes (KL ≈ 1e-11), and
  f32 / bf16 / f16 training is bit-identical across runs. Default path is
  cuBLAS for maximum throughput. Scope note: the tied LM heads and the
  no-context `*_blas` twins take no context and stay on cuBLAS regardless
  of the flag; every path that carries a `GpuCtx` — including the M3
  engine, prefill and inference alike — follows it.
- **Two batch-invariant families, selectable** — `ctx.set_bi_gemm_family()`
  / `MAMBA_RS_BI_GEMM_FAMILY=triad|fixed` picks which family serves the
  forward while the flag above is on. `Triad` (`kernels/gemm_bi_triad/`,
  default) is the multi-tile dispatcher: it carries all three operand
  layouts, so it is the only family that can serve a backward, and its
  invariance holds across every M inside one dispatch bucket. `Fixed`
  (`kernels/gemm_bi_fixed/`) is the forward serving family: a ladder of
  bit-identical tiles (16-row thin, 64, 128, and a wide 128×256
  fragment-reuse tile) with `SPLIT_K=1` everywhere, batch-invariant BY
  CONSTRUCTION — the K-reduction for `C[i,j]` reads only `A[i,:]` and
  `B[:,j]`, every rung produces the same bits per element, so tile choice
  is pure scheduling and no bucket boundary exists to cross. The family is part of the numeric route
  (`ctx.gemm_route()`) and a flip after a CUDA-graph capture is refused at
  replay like any tier flip.
- **Fast typed deterministic tier (opt-in)** — `MAMBA_RS_BI_TENSOR_CORES=1`
  / `ctx.set_bi_tensor_cores(true)` on top of the flag above selects the
  fastest qualified typed route: usually mma.sync tensor-core kernels, with
  measured scalar fallbacks where they win. On SM89, automatic BF16/F16 NN
  uses the exact scalar Split-K route only when N=128 and the scalar plan is
  `NnSplitKThinTail` with K>=511 or `NnSplitKThin` with K>=1024; forced tile
  requests and every other architecture remain unchanged. On CC12.0, automatic
  SM120 routing is deliberately narrower: an immutable 18-cell BF16/F16 table
  covers the qualified NN, TN and NT hot shapes with route-sealed BK32 or BK64
  TMA/MMA16 schedules. CC12.1 and every non-cell decline to the existing
  portable ladder. Each SM120 output has one owner, follows one fixed ascending
  K16 MMA chain, and uses no numerical atomics or Split-K reduction. It stays
  fully deterministic, freezes the selected numeric contract and physical
  schedule in the route identity, and reaches at-or-near cuBLAS parity even on
  d128/d256 models and **faster than cuBLAS** from d_model ≥ 768
  (0.70× of PEDANTIC per step at d1536 bf16). A qualified SM120 CUDA Graph
  route must first run eagerly to prepare its tensor maps and cache entry;
  capture fails closed on a missing, stale or untracked allocation epoch rather
  than allocating, retuning or silently changing routes inside capture.
- **Deterministic F32 policy** — `MAMBA_RS_BI_F32_POLICY=exact|tf32` or
  `ctx.set_f32_triad_policy(...)` controls the batch-invariant F32 route.
  `exact` is the default scalar `__fmaf_rn` contract. `tf32` permits only a
  frozen qualified deterministic TF32 route; it does not force one, and an
  unsupported or unmeasured cell remains on exact scalar FMA. TF32 has a
  different reduction contract from exact scalar FMA, but repeated eager and
  graph launches of the same frozen route are bit-identical.
- **Half-precision stream-K policy** — `MAMBA_RS_BI_HALF_POLICY=tiled|streamk`
  or `ctx.set_half_triad_policy(...)` controls the batch-invariant bf16/f16
  route on the CC 12.x boards. `tiled` is the default: every automatic half
  route reproduces the forced portable tensor-core kernel bit for bit.
  `streamk` permits only the measured stream-K cells, a persistent grid that
  folds per-CTA partials in a fixed order (a different reduction contract from
  the tiled body; repeated eager and graph launches of the same route stay
  bit-identical). It does not force one: an unmeasured shape, or a grid that
  already fills the device, stays on the tiled route. Requires
  `MAMBA_RS_BI_TENSOR_CORES=1`; the flag refuses to be a silent no-op.
- **Per-architecture tensor-core rungs** — on Hopper (`wgmma`) and
  Blackwell (`tcgen05`) the deterministic forward ladder routes to native
  per-architecture kernels, each a bit family of its own, guarded by a
  first-use numeric self-check that falls back to the portable ladder
  (and can be disabled with `MAMBA_RS_ARCH_RUNG=off`).
- **Bring-your-own-loss training split** — `trainer.forward()` returns the
  full `batch * seq_len * d_model` post-norm_f temporal output on the host;
  compute ANY loss gradient in plain Rust and feed it to
  `trainer.backward_step()` (global-norm clipping, exact gradient
  accumulation, LR schedules via `set_lr`, reference-faithful AdamW
  no-decay groups). Bit-identical to the fused `step()` — both compose the
  same eager phase bodies. Same API on `Mamba3Trainer`.
- **Full-sequence CPU prefill** — `forward_prefill` runs a whole prompt/page
  through the training forward's batched-SGEMM pipeline (no activation
  tape) instead of T per-step dispatches, then hands the recurrent state to
  the step path (prefill-then-decode). Serial mode is the deterministic
  reference; `PrefillMode::Parallel` parallelizes every phase and stays
  bit-equal. Both architectures.
- **GPU prompt prefill (Mamba-3)** — one-pass prompt window through the
  chunked pipeline, leaving all four recurrent states positioned for
  decode; continued windows apply the trapezoidal boundary fold, and a
  captured CUDA-graph twin replays bit-identically. The LM generate
  path switches to it automatically for long prompts.
- **Deterministic data-parallel training (`dist`)** — one process per
  GPU, one reduction per optimizer step over the flat gradient arena
  (the fixed-order tier implements it as a byte-only shard exchange
  around its fold kernel; the library tier as one collective).
  The default fixed-order contract (ascending logical-rank fold: bits
  independent of transport, topology, library version, and physical GPU
  permutation) is implemented twice and cross-pinned: the emulated
  oracle proves the contract in one process, and the transport-backed
  house reducer runs it live — peer addends move as pure bytes (NCCL
  send/recv/broadcast, zero library arithmetic) and every float add
  happens in the `det_sum_ranks` kernel in program-text order. The
  `nccl` feature also carries the explicit `NcclSum` tier (library
  collective; run-to-run stable on a frozen box) — live-validated on
  two RTX 5090s bit-for-bit against the oracle. The FixedOrder
  transport path is oracle-pinned on one GPU (same kernel, same slot
  layout over a loopback byte mover); its own live multi-GPU first
  light rides the next validation window.
- **Bit-continuous resume** — optimizer state (Adam moments, step,
  update hyperparameters) and the carried recurrence export/import, so
  a resumed run lands bit-for-bit where the unbroken run would.
- **Large state dimensions** — per-thread state arrays are sized at JIT
  time from the config, up to the reference implementations' own
  maximum of 256; every generation runs the same code path at any
  supported `d_state`.
- **HuggingFace loader** — safetensors, synthetic + real Mamba SSM
  checkpoints (130m / 370m / 1.4b / 2.8b validated).
- **Standalone** — no framework dependency. MSRV 1.97.

## Cargo features

| feature | what it enables | when |
|---|---|---|
| *(default)* | pure-Rust scalar GEMM | correctness work only — 5-20x slower |
| `gemm-blas` | [`gemm`] crate BLAS-class CPU GEMM (+ rayon) | ANY serious CPU use |
| `accelerate` | Apple Accelerate GEMM (macOS) | macOS deployments |
| `cuda` | GPU inference + training (NVRTC-compiled kernels) | needs the CUDA toolkit |
| `cuda-cublaslt-qualification` | adds cuBLASLt to the CUDA qualification/benchmark harness; ordinary `cuda` does not enable it | maintainer qualification only, not production route selection |
| `hf` | safetensors/HF checkpoint loaders | LM checkpoints |
| `cli` | `mamba-generate` binary (tokenizers + hf-hub) | text generation CLI |
| `nccl` | data-parallel transport (pinned NCCL binding) | multi-GPU training |

### Typed GEMM routing and low-level SM120 APIs

Application code normally reaches the deterministic GEMM engine through the
trainers/backbones. Direct GPU integrations should use
`mamba_ssm::gpu::blas::gemm_bi_forward_typed`,
`mamba_ssm::gpu::blas::gemm_bi_backward_dw_typed` and
`mamba_ssm::gpu::blas::gemm_bi_backward_dx_typed`; these entries own automatic
policy lookup and fall back without exposing tile choices to callers. The
similarly scoped `gemm_bi_triad::*_typed_native` functions are native-bucket
qualification hooks and may return `UNCOVERED`; they are not application APIs.

The exported SM120 route constants and the `resolve_sm120_forced`,
`prepare_sm120_tensor_maps`, `prepare_sm120_tma_forced`,
`launch_sm120_tma_prepared` and `validate_sm120_graph_replay` functions are
low-level qualification and census building blocks. A forced launch neither
adds a cell to production automatic dispatch nor relaxes its exact target,
shape, pointer, tensor-map, context or allocation-lifetime validation. IDE and
docs users inspecting normal CUDA code need only the `cuda` feature; enable
`cuda-cublaslt-qualification` only when building the vendor-comparison harness.

## Use cases and API choice

The crate targets two workloads. Pick the entry point that matches yours.

### Reinforcement learning / small custom models

Latency-critical, typically `d_model ≤ 256`, often batch = 1 for actor
rollouts. Both CPU and GPU paths are supported; CPU is competitive at
these sizes (~87 µs/step on Ada Xeon vs 79 µs/step on RTX 6000 Ada).

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

### Sequence classification / embeddings / custom heads

Whole-sequence reads with a caller-defined loss (document classifiers,
distillation, contrastive embedding). GPU training rides the
forward/backward split; CPU serving rides the prefill.

- **Training** — `MambaTrainer::forward` + host-side loss +
  `MambaTrainer::backward_step` (see `examples/custom_loss.rs`)
- **CPU serving** — `MambaBackbone::forward_prefill` /
  `forward_mamba3_backbone_prefill` (see `examples/cpu_prefill.rs`);
  batches of sequences via `prefill_batch` / `prefill3_batch`

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
mamba-rs = { version = "0.6", features = ["cuda"] }
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

### Custom losses: the forward/backward split

The fused `step()` needs the loss gradient up front; the split lets you
compute it from the actual forward output:

```rust
use mamba_rs::mamba_ssm::gpu::trainer::BackwardOpts;

let mut temporal = vec![0.0f32; batch * seq_len * cfg.d_model];
trainer.forward(&input, &mut temporal)?;          // full temporal readback (f32)
let d_temporal = my_loss_grad(&temporal);          // any host-side loss
let m = trainer.backward_step(
    &d_temporal,
    BackwardOpts::default().with_clip_max_norm(1.0),
)?;                                                // backward + clip + AdamW
// gradient accumulation: .with_accumulate_only(true) on the non-applying
// micro-batches (the fused step() refuses while a window is open).
```

Always eager (a caller-side loss cannot live inside a captured graph), and
bit-identical to the fused `step()` — both compose the same phase bodies.
See `examples/custom_loss.rs` for a complete training loop.

## Quick start (CPU prefill)

```rust
use mamba_rs::inference::PrefillMode;
use mamba_rs::module::MambaBackbone;

let backbone = MambaBackbone::init(cfg, input_dim, 42);
let mut state = backbone.alloc_state();
let mut scratch = backbone.alloc_prefill_scratch(seq_len);
let mut out = vec![0.0f32; seq_len * backbone.config().d_model];

// One batched-SGEMM pass over the whole prompt instead of T step dispatches.
backbone.forward_prefill(&prompt, &mut out, &mut state, &mut scratch,
                         seq_len, PrefillMode::Parallel);
// `out` holds the post-norm_f output at EVERY position (pooling-ready);
// `state` is positioned after the prompt — forward_step continues from it.
```

Enable `gemm-blas` (or `accelerate` on macOS) — the default scalar GEMM is
a correctness fallback, not a serving configuration. Mamba-3 has the same
surface (`forward_mamba3_backbone_prefill` + `Mamba3PrefillScratch`).

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
| f32   | 725 tok/s        | 686 tok/s              | −5 % |
| bf16  | **1 029 tok/s**  | 958 tok/s              | −7 % |
| f16   | 1 028 tok/s      | 958 tok/s              | −7 % |

On f32 both paths run on CUDA cores (no Tensor Core route), so the gap
is small. On bf16/f16 cuBLAS routes through Tensor Cores with f32
accumulation and wins ~7 % on per-token latency, at the cost of M=1 vs M=N
algorithm-selection drift (KL ≈ 1e-3 on adversarial prompts). The
batch-invariant path keeps `b=1` ≡ `b=N` per slot (KL ≈ 1e-11).

Enable the batch-invariant path when cross-batch bit-identity matters
(KL ≈ 1e-11 between `b=1` and `b=N` per slot): set
`MAMBA_RS_BATCH_INVARIANT=1` or call `ctx.set_batch_invariant(true)`.

### Choosing a batch-invariant family

```rust
ctx.set_batch_invariant(true);                       // deterministic forward
ctx.set_bi_gemm_family(BiGemmFamily::Fixed);         // or ::Triad (default)
```

| | `Triad` (`gemm_bi_triad/`) | `Fixed` (`gemm_bi_fixed/`) |
|---|---|---|
| layouts | NN + TN + NT | NN only |
| invariance | across M inside one dispatch bucket | by construction, no buckets |
| structure | shape-routed tiles (ultra-thin, narrow-N, GEMV, split-K, Slim/Big) | bit-identical tile ladder (thin 16, 64, 128, wide 128×256), `SPLIT_K=1` |
| dtypes | f32 / bf16 / f16, CUDA cores and Tensor Cores | f32 / bf16 / f16; Tensor Cores for bf16/f16, CUDA-core FMA tile for f32 |

A backward requires `Triad`. For a forward-only serve workload the
tensor-core ladder makes `Fixed` the fast route: at a vision-classifier
prefill shape (M = 4621 rows per page, RTX 6000 Ada) the deterministic
bf16 page runs 10.9 ms end to end — ahead of the 11.8 ms
non-deterministic cuBLAS f32 baseline and 1.8× the 20.0 ms deterministic
f32 route. Both families differ from cuBLAS f32 by the same
1.0e-4–1.8e-4 envelope, and reruns are bit-identical.

### Deterministic training — cost per step (RTX 6000 Ada, `MambaTrainer`)

With the batch-invariant flag on, every training GEMM (forward, dW, dX)
runs on custom fixed-reduction-order kernels: two runs with the same
seed/inputs produce bit-identical weights, on every dtype. The optional
tensor-core tier keeps full determinism under its own numeric contract
(mma.sync f32 accumulation instead of the scalar FMA chain) and turns the
determinism overhead into a speedup on LLM-sized models:

| model | dtype | cuBLAS baseline | deterministic (scalar) | deterministic + TC |
|---|---|---:|---:|---:|
| d768, B=8 T=256  | bf16 | 25.7 ms (PEDANTIC) | 28.5 ms (1.11×) | **21.7 ms (0.84×)** |
| d1536, B=4 T=256 | bf16 | 17.8 ms (PEDANTIC) | 19.4 ms (1.09×) | **12.5 ms (0.70×)** |
| d1536, B=4 T=256 | f32  | 14.1 ms (TF32)     | 21.6 ms (1.53×) | — |
| d128 (RL), B=16 T=64 | bf16 | 2.12 ms (PEDANTIC) | 2.54 ms (1.20×) | 2.20 ms (1.04×) |

```rust
trainer.ctx().set_batch_invariant(true);   // bit-identical runs, scalar contract
trainer.ctx().set_bi_tensor_cores(true);   // + tensor-core tier (own contract)
```

GEMM-level tensor-core speedups vs the scalar deterministic tier: forward
3.7–6.3×, dW 4.0–5.6×, dX 4.5–5.4× (bf16, M=2048-class shapes). At fat
training shapes the wide fragment-reuse tile carries the deterministic
ladder to parity with cuBLAS's tensor-core path (143.5 vs 144.9 TFLOPS
bf16 at 4096×768×3072) and +12% over the square tile just past a wave
boundary. Bit-identical tiles, shape-routed, cover everything from d128
RL models to LLM projections. Full tables and
contracts: [deterministic GEMM benchmarks](docs/determinism-benchmarks.md).

A dedicated kernel-optimization pass cut the deterministic training step 3.4×
(d_model 384, 24 layers, B=8, T=1300, bf16, tensor-core tier:
441 → 131.5 ms/step on an RTX 5090) and removed the O(T) scan tape
(−12.3 GB at that shape) — stage-by-stage table in
[Mamba SSM benchmarks](docs/mamba1-benchmarks.md) and the
[CHANGELOG](CHANGELOG.md).

### Per-step latency (default config: d_model=128, 3 layers)

| | Mamba SSM | Mamba-3 SISO |
|---|---|---|
| GPU inference B=1 (CUDA Graph) | **79 µs** | **87 µs** |
| GPU training fwd+bwd (T=32, tiny synthetic shape) | 1 653 µs | 1 784 µs |
| CPU inference B=1              | 87 µs    | **65 µs** |
| CPU training fwd+bwd (T=32)    | 14 859 µs | **3 635 µs** |

Production-scale Mamba-3 training and prefill tables (multi-chunk
sequences, 24-layer shapes) live in the detailed docs:
[Mamba SSM benchmarks](docs/mamba1-benchmarks.md),
[Mamba-3 SISO benchmarks](docs/mamba3-benchmarks.md).

## Testing

The suite combines integration tests with in-module unit tests. CI results are
the authoritative inventory; hand-maintained test totals are intentionally
omitted because architecture qualification adds and retires cells over time:

- Correctness: bit-parity WITHIN a numeric route (eager ↔ CUDA Graph,
  run ↔ run, save ↔ nosave prefill, CPU Single ↔ CPU Parallel); tolerance
  parity ACROSS routes (CPU ↔ GPU, sequential ↔ parallel scan, f32 ↔
  bf16/f16, scalar ↔ tensor-core GEMM) — different reduction orders are
  different bit families by design
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

## Roadmap

- Multi-GPU inference for models larger than one device (pipeline
  sharding), complementing the data-parallel training that ships now.
- Reduced-precision tiers (fp8 / int8) under the same bit-discipline
  as the existing f32 / bf16 / f16 paths.
- The Mamba-2 generation, living beside Mamba-1 and Mamba-3 in this
  crate with the same determinism and testing discipline.

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
