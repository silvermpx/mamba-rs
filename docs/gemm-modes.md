# GEMM modes

Every GPU context in mamba-rs multiplies matrices in one of three modes.
The mode decides who runs the matrix multiplications behind a model or a
trainer: the crate's own deterministic kernels, or cuBLAS in one of two
precision settings. This page explains the three modes, how to pick one,
how to set it, and exactly what each one promises.

## The three modes

| mode | who multiplies | what you get |
|---|---|---|
| `Deterministic` (default) | the crate's own kernels | the same bits every run, the same bits from an eager call and from a captured CUDA graph, and for inference the same bits for a row whether it is decoded alone or inside a batch. cuBLAS is never called. |
| `CublasFast` | cuBLAS with `CUBLAS_TF32_TENSOR_OP_MATH` and `CUBLAS_COMPUTE_32F` | the vendor's fast path. f32 inputs may be multiplied on TF32 tensor cores; bf16 and f16 inputs use ordinary f32 accumulation. |
| `CublasPedantic` | cuBLAS with `CUBLAS_PEDANTIC_MATH` and `CUBLAS_COMPUTE_32F_PEDANTIC` | the vendor's most accurate f32 accumulation. This is what mamba-rs 0.6.9 and earlier ran by default. |

cuBLAS gives repeatable results under the conditions NVIDIA documents
(same library, same GPU, same shapes, same stream count). What it does not
give is batch invariance: cuBLAS picks its algorithm from the shape of the
whole multiplication, so the numbers computed for one row can change when
the rows around it change. The `Deterministic` mode exists to close that
gap.

## Which mode to use

| you want | use |
|---|---|
| reproducible training runs and reproducible serving, with results that do not depend on batch composition | `Deterministic` (the default) |
| the fastest vendor path and you do not need reproducibility across batch sizes | `CublasFast` |
| the numbers a mamba-rs 0.6.9 program produced, or a precise vendor reference to compare against | `CublasPedantic` |

Speed depends on the GPU, the precision and the shape. The measured
comparison of the deterministic kernels against both cuBLAS modes, kernel by
kernel, is in [GEMM benchmarks](determinism-benchmarks.md).

## Setting the mode and the storage precision

A GPU context has two settings and nothing else. `WeightDtype` picks how
weights are stored and, with it, the precision of every product;
`GemmMode` picks who multiplies. Every route accumulates in f32. Inside the
deterministic mode the kernels are chosen from the two and from the
context's role (a model context or a trainer); there is no further knob.

| `WeightDtype` | storage | products in the deterministic mode |
|---|---|---|
| `F32` (default) | f32 | exact: one fused multiply-add per step in ascending K order, f32 throughout |
| `Tf32` | f32 | deterministic TF32 where a measured kernel exists for the shape and the board, the exact f32 kernel elsewhere. Deterministic TF32 rounds each input to TF32 once and accumulates in f32 in a fixed order; it is not the vendor's Fast TF32, and its bits are reproducible like the rest of the mode. Under `CublasFast` the vendor's TF32 already applies to every f32 product; under `CublasPedantic` the products are pedantic f32 |
| `Bf16` | bf16 | tensor-core kernels with f32 accumulation |
| `F16` | f16 | tensor-core kernels with f32 accumulation; training runs the dynamic loss scaler |

Every GPU entry point comes in two forms. The plain constructor reads the
environment variable `MAMBA_RS_GEMM_MODE` and defaults to `Deterministic`.
The `*_with_mode` twin takes the mode as its last argument and ignores the
environment entirely. The storage precision is always an argument.

| plain constructor (reads the environment) | explicit twin |
|---|---|
| `GpuCtx::new_from_env`, `GpuCtx::new_from_env_with_state_cap` | `GpuCtx::new_with_mode`, `GpuCtx::new_with_state_cap_and_mode` (`GpuCtx::new` and `new_with_state_cap` are `Deterministic` and read nothing) |
| `GpuMambaBackbone::new`, `new_with_dtype` | `GpuMambaBackbone::new_with_mode`, `new_with_dtype_and_mode` |
| `GpuMamba3Backbone::new`, `new_with_dtype` | `GpuMamba3Backbone::new_with_mode`, `new_with_dtype_and_mode` |
| `GpuMambaLM::from_hf`, `from_hf_with_dtype`, `from_hf_with_dtype_batch` | `GpuMambaLM::from_hf_with_mode`, `from_hf_with_dtype_and_mode`, `from_hf_with_dtype_batch_and_mode` |
| `GpuMamba3LM::from_weights`, `Mamba3LmBuild::build` | `GpuMamba3LM::from_weights_with_mode`, `build_with_mode` |
| `GpuMamba3LM::from_weights_with_dtype` | `Mamba3LmBuild { dtype, batch: 1, .. }.build_with_mode(mode)` |
| `MambaTrainer::new_full`, `Mamba3Trainer::new_full` | `MambaTrainer::new_full_with_mode`, `Mamba3Trainer::new_full_with_mode` |
| `MambaTrainer::new_with_dtype`, `Mamba3Trainer::new_with_dtype` | `new_full_with_mode` with `TrainSessionCfg::new(input_dim, batch, seq_len)`, which carries the same default optimizer settings |

```rust
use mamba_rs::gpu::inference::GpuMambaBackbone;
use mamba_rs::mamba_ssm::gpu::GemmMode;
use mamba_rs::WeightDtype;

let mut gpu = GpuMambaBackbone::new_with_dtype_and_mode(
    0, &weights, cfg, input_dim, batch, WeightDtype::Bf16, GemmMode::CublasFast,
)?;
assert_eq!(gpu.ctx().gemm_mode(), GemmMode::CublasFast);
```

Every model, LM and trainer exposes `ctx()` so the mode can be read back.
The mode can also be changed on a live context:

```rust
gpu.ctx().set_gemm_mode(GemmMode::Deterministic)?;
```

`set_gemm_mode` returns an error while a CUDA graph is being captured on
the context's stream or while a GEMM route is being recorded, and it
restores the previous cuBLAS setting if the change fails half way. Change
the mode before capturing a graph; a graph captured in one mode refuses to
replay in another (see [Graphs](#graphs)).

### The environment variable

The plain constructors read one variable once, at construction; the
`*_with_mode` constructors ignore it.

| variable | values | default |
|---|---|---|
| `MAMBA_RS_GEMM_MODE` | `deterministic`, `cublas-fast`, `cublas-pedantic` (case-sensitive) | `deterministic` |

Nothing else in the environment changes a GEMM route. The variables that
tuned the deterministic mode in 0.7.0 to 0.7.2 (`MAMBA_RS_BI_GEMM_FAMILY`,
`MAMBA_RS_BI_TENSOR_CORES`, `MAMBA_RS_BI_F32_POLICY`,
`MAMBA_RS_BI_HALF_POLICY`, `MAMBA_RS_ARCH_RUNG`) and the pre-0.7 selectors
(`MAMBA_RS_BATCH_INVARIANT`, `MAMBA_RS_FAST_GEMM`) are no longer read; a
program that set `MAMBA_RS_BI_F32_POLICY=tf32` stores its weights as
`WeightDtype::Tf32` instead, and the 0.6.9 numbers are
`MAMBA_RS_GEMM_MODE=cublas-pedantic`.

## Two kernel families

The deterministic mode has two families of kernels. A context uses one of
them, chosen by its role:

| family | kernels | used by | shapes | batch invariance |
|---|---|---|---|---|
| `Inference` | `kernels/gemm_bi_inference/` | model and LM contexts (`GpuMambaBackbone`, `GpuMamba3Backbone`, `GpuMambaLM`, `GpuMamba3LM` and the engines under them) | the forward product only (`Y = X W + b`) | every output element is computed from its own row of X and its own column of W in one fixed order, never split across thread blocks, so a row's result is bit-identical at any batch size |
| `Triad` | `kernels/gemm_bi_triad/` | trainers (`MambaTrainer`, `Mamba3Trainer`) and plain `GpuCtx` | forward, weight gradient and input gradient (the NN, TN and NT products) | bit-identical across every batch size that lands in the same dispatch bucket; crossing a bucket boundary changes the association deterministically |

The weight-gradient product always runs on the Triad kernels, whatever the
family, because the Inference family has no transposed products. The
family is part of the recorded numeric route, so a graph captured with one
family refuses to replay with the other.

### What the deterministic mode chooses by itself

- bf16 and f16 products run on the tensor-core kernels wherever a shape
  and board gate admits one; the scalar kernels serve the rest. The
  tensor-core kernels are their own bit family: repeated launches of one
  kernel are bit-identical, but they do not reproduce the scalar kernels'
  bits.
- A half-precision weight gradient takes the stream-K kernel whenever the
  reduction is deep enough for its persistent grid to pay (32 or more
  64-row slabs per multiprocessor); it folds per-block partial sums in a
  fixed order and is a separate bit family from the tiled kernels.
- f32 products are exact unless the weights are stored as `Tf32`, which
  permits the deterministic TF32 kernels on the shapes and boards where
  they were measured and stays exact everywhere else.
- The Hopper and Blackwell native kernels run behind a first-use
  self-check (see [Architecture coverage](#architecture-coverage)).

## What is guaranteed

The guarantees below are backed by tests in the repository; the section
after this one lists what is not claimed.

- **Run to run.** Within one process, one build and one GPU, the same
  inputs give the same bits in every mode, including both cuBLAS modes on
  a frozen machine. Tested for f32, bf16 and f16 training steps and for
  the sequential and parallel scan regimes.
- **Eager versus captured graph.** A kernel launched eagerly and the same
  kernel replayed from a captured graph produce the same bits. At the
  whole-step level this is tested bit for bit for the Mamba-1 bf16 training
  step, the Mamba-1 pooled prefill and the Mamba-3 f32 serving prefill; the
  Mamba-3 bf16 training step and the f32 training steps are tested to a
  1e-5 tolerance.
- **Batch invariance.** The Inference family gives the same bits for a row
  at every batch size; measured through a whole model, batch 1 against
  batch 32 on the same prompt gives a KL divergence of about 8e-11
  (`tests/extreme_edge_coverage.rs`), where cuBLAS gives about 1e-3. The
  Triad family gives the same bits within a dispatch bucket.
- **No hidden vendor call.** In the deterministic mode every cuBLAS entry
  point in the crate returns an error instead of running; a model context
  never reaches one.
- **Route recording.** A captured graph stores the complete numeric route
  (mode, storage precision and role, the kernels selected, compiler and
  device identity) and every replay checks it.

## What is not guaranteed

- Bits are not equal across routes: `F32` against `Tf32`, Inference
  against Triad, either against cuBLAS, and, inside one precision, the
  scalar and tensor-core kernels or the tiled and stream-K weight-gradient
  kernels, which the mode chooses by shape and board. Across routes only
  tolerance parity holds, and the parity tests state their tolerances.
- Bits are not equal across GPU architectures, drivers or CUDA toolkits.
  A frozen route is identified by its board, toolkit and compiled artifact.
- Whole-model eager and graph outputs are bit-equal only where the tests
  above say so.
- The cuBLAS modes are repeatable but not batch-invariant.

## Graphs

A graph captured through `capture_graph` on a model, LM or trainer records
the numeric route of every GEMM in it. Changing the mode after the capture
prints a warning, and the next replay fails with
`GEMM route changed since capture; re-capture before replay`. Changing the
mode during a capture is refused outright.

The split training step (`trainer.forward()` followed by
`trainer.backward_step()`) runs eagerly and pins the route between the two
calls: a mode change in between is an error, and the pending forward must
be re-run.

On an RTX 5090 (compute capability 12.0) the qualified kernels use tensor
maps that must be built outside a capture, so the first step of a model
must run eagerly before `capture_graph`; the capture fails closed
otherwise. The GPU staging buffers also cannot grow inside a capture, so
the largest shape a graph will see must run once before the capture or be
pre-sized with the `presize_*` methods on `GpuCtx`.

## Architecture coverage

The deterministic kernels compile for every architecture from SM80 up.
"Measured" below means the automatic kernel selection on that board was
timed cell by cell and the winners were frozen with the board, the toolkit
and the compiled artifact; the driver build is not part of that identity.
"Portable" means the generic kernels serve and no timing claim is made.

| GPU | bf16 and f16 | exact f32 (`F32`) | deterministic TF32 (`Tf32`) |
|---|---|---|---|
| RTX 6000 Ada (SM89) | measured on CUDA 13.2 for the large shapes; portable kernels for the small ones | measured on CUDA 13.2; scalar kernels elsewhere | measured joint kernels on CUDA 13.2; portable kernels elsewhere |
| RTX 5090 (CC 12.0) | 60 measured tiled entries and 12 stream-K entries on CUDA 13.2; nearby shapes take the nearest measured entry within a factor of 8 on each dimension | scalar kernels plus qualified TMA-fed FMA kernels | frozen retained kernels on CUDA 12.8, 13.0 and 13.2 |
| SM80, SM86, SM87 | portable | scalar | portable (Inference); exact f32 (Triad) |
| SM90 and SM90a | portable; a native WGMMA kernel is tried behind a first-use self-check | scalar | exact f32 |
| SM100 family (CC 10.0, 10.3, 11.0) | portable; a native `tcgen05` kernel is tried behind the same self-check | scalar | exact f32 |
| CC 10.1 and CC 12.1 | portable | scalar | exact f32 |

The Hopper and Blackwell native kernels are guarded by a numeric self-check
against the portable kernels at first use; if it fails, the portable
kernels serve for the rest of the process. They are not measured winners
on any board.

Toolkits outside CUDA 12.8, 13.0 and 13.2 are not rejected. A frozen kernel
whose recorded toolkit does not match the running one is simply not
selected, and the portable or scalar kernel serves instead, with a warning
printed once.

## Errors

Every constructor and setter on this surface returns `Result<_, String>`.
The messages name the variable or the call that failed and what to do
instead:

```text
MAMBA_RS_GEMM_MODE="fast" is not a recognized GEMM mode (use deterministic, cublas-fast, or cublas-pedantic)
cannot change GEMM mode while CUDA stream capture state is ...
M1 f32 inference graph replay: GEMM route changed since capture; re-capture before replay
gpu_gemm_bi_forward_grad: deterministic GEMM mode reached a cuBLAS dispatch boundary
```
