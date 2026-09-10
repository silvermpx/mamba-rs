# Monolithic-main versus Inference/Triad baseline plan

## Execution corrections after the API audit

The endpoint and shell recipe below are historical preparation, not a runnable
final packet. Before execution apply these already established constraints:

- Use the final reviewed API commit as the new endpoint, not69ed1c80 or the
  current intermediate identity commit. Main remainsd8f2efbe unchanged.
- Extract only the two inference sections into one immutable benchmark adapter;
  do not execute the original file's later training/CPU sections and merely
  discard their output.
- The common exact-F32 adapter must explicitly set batch-invariant=true,
  family=Triad, tensor-cores=false and fast=false on both actual contexts.
  Main's constructor ignores the environment, so environment settings alone
  do not prove this baseline configuration.
- That common Triad-family lane measures the old versus new Triad-backed
  end-to-end pipeline. It is not a measurement of the new Inference-family
  default. Report a separately configured final Inference-family lane if
  claiming a default-inference delta; do not relabel the Triad result.
- Use separate private0700 kernel/Driver caches for old/new, with
  CUDA_CACHE_DISABLE=0 after initial compilation. The older cold-cache recipe
  below must not force every context to repeat cold JIT work.
- Verify fixed-step output correctness and same-arm repeatability before
  pairing process windows; distinguish those checks from any cross-version
  exact-bit claim. Retain source hashes, selected mode/family/precision,
  transfer/synchronization scope and raw ABBA windows.

No new benchmark source, GPU run or performance result is created by these
corrections. The actual adapter and final endpoint are still pending API closure.

Source-only plan, 2026-09-10. No GPU or remote command was run and no timing is
claimed. Current committed endpoints are:

- unchanged `main`: `d8f2efbeaecc04a53cec9890097f2913cd3f09e6`;
- current Triad worktree `HEAD`: `69ed1c80d2ce855555626df8a1c5450c836cf25e`.

The Triad worktree is dirty while TF32 integration is in progress. A real run
must replace the second SHA with the final committed integration SHA and record
it in every result. Do not time the uncommitted working tree.

## Smallest common harness

Reuse `tests/m1_gpu_benchmark.rs::m1_gpu_benchmark`. Its Git blob is identical
at both endpoints (`59edf8bd26d78e04fc4e7e9436387f545e3ed48a`; file SHA-256
`5ec73ac29bf4c135bb730aef18266d9cb3db4463512337b7225226a57d6b7bc0`).
`src/config.rs` is also unchanged. The two inference sections therefore use the
same seed-42 `MambaConfig::default()` (`d_model=128`, `d_inner=256`,
`d_state=16`, three layers), batches `1,4,16,64,128`, 20 warmups, and the same
eager/graph iteration counts. One step contains these F32 NN GEMMs:

- once: `(B,128,128)` with bias;
- per layer: `(B,128,512)`, `(B,256,40)`, `(B,8,256)` with bias, and
  `(B,256,128)`; hence each timed step has 13 GEMMs plus the identical
  non-GEMM inference pipeline and H2D/D2H/synchronization.

This is an end-to-end inference latency smoke baseline, not a kernel admission
benchmark. Parse only the two `GPU Inference` sections; the same test continues
with training and CPU measurements after them.

## Required benchmark-only adapter

Running the existing file unchanged with `MAMBA_RS_BATCH_INVARIANT=1` is not a
valid comparison. At `main`, `GpuMambaInference::new` calls
`GpuCtx::new_with_state_cap`, which deliberately ignores the environment and
defaults to cuBLAS. The new inference code calls
`new_from_env_with_state_cap`. Both endpoints, however, expose the same public
`GpuMambaBackbone::ctx()`, `set_batch_invariant`, and `set_bi_gemm_family`
interfaces.

Apply the same benchmark-only edit to both clean source trees (do not alter GEMM
or inference production code): import `BiGemmFamily`, then immediately after
each of the two `GpuMambaBackbone::new(...)` calls in the eager and graph loops
add:

```rust
bb.ctx().set_batch_invariant(true);
bb.ctx().set_bi_gemm_family(BiGemmFamily::Triad);
assert!(bb.ctx().batch_invariant());
assert_eq!(bb.ctx().bi_gemm_family(), BiGemmFamily::Triad);
```

This is the minimum source adaptation (one import plus four common-API lines at
each construction site). It leaves `main`'s monolithic
`src/mamba_ssm/gpu/gemm_bi_triad.rs` / `kernels/gemm_bi_triad.cu` untouched and
selects the new modular Inference/Triad through the same API.

Use **exact F32** for the apples-to-apples baseline. `main`'s monolith explicitly
has full-F32 accumulation and has no `F32TriadPolicy`/deterministic-TF32 API;
the new side's default is `ExactScalarFmaV1`. Thus a new-side
`MAMBA_RS_BI_F32_POLICY=tf32` result cannot be called a speedup over main: it is
a different precision/numerical contract. It may be reported later as a
separate, clearly labelled throughput lane.

Other API differences do not require harness changes: the low-level new
`GpuMambaInference::capture_graph` is `unsafe` and requires a successful eager
manifest, but the common safe `GpuMambaBackbone::capture_graph()` wrapper does
that eager step and owns the buffers on both endpoints.

## Exact build/run recipe, once per board

Use two already-existing clean checkouts, one at each recorded SHA. Keep target
and NVRTC caches separate so neither binary/artifact can leak across arms.
CUDA 13.2 is the common first lane used by the existing Ada/SM120 scripts.
Substitute the board's exact GPU UUID and checkout paths:

```bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-EXACT-UUID
export CUDA_CACHE_DISABLE=1
unset NVIDIA_TF32_OVERRIDE

# Pin the common route; setters in the adapter are the main-side authority.
export MAMBA_RS_BATCH_INVARIANT=1
export MAMBA_RS_BI_GEMM_FAMILY=triad
export MAMBA_RS_BI_TENSOR_CORES=0
export MAMBA_RS_FAST_GEMM=0
export MAMBA_RS_BI_F32_POLICY=exact
unset MAMBA_RS_BI_HALF_POLICY MAMBA_RS_ARCH_RUNG

# In the main checkout: cd /absolute/path/to/main-checkout
export CARGO_TARGET_DIR=/root/target-m1-monolith-main-cuda132
export MAMBA_RS_KERNEL_CACHE=/root/cache-m1-monolith-main-cuda132
mkdir -p "$MAMBA_RS_KERNEL_CACHE" && chmod 700 "$MAMBA_RS_KERNEL_CACHE"
cargo test --release --locked --features cuda --test m1_gpu_benchmark \
  m1_gpu_benchmark -- --exact --ignored --nocapture --test-threads=1 \
  |& tee main-m1-inference-cuda132.log

# In the final Triad checkout: cd /absolute/path/to/final-triad-checkout
export CARGO_TARGET_DIR=/root/target-m1-triad-final-cuda132
export MAMBA_RS_KERNEL_CACHE=/root/cache-m1-triad-final-cuda132
mkdir -p "$MAMBA_RS_KERNEL_CACHE" && chmod 700 "$MAMBA_RS_KERNEL_CACHE"
cargo test --release --locked --features cuda --test m1_gpu_benchmark \
  m1_gpu_benchmark -- --exact --ignored --nocapture --test-threads=1 \
  |& tee triad-m1-inference-cuda132.log
```

Before each arm, record `git rev-parse HEAD`, `git status --short`,
`sha256sum tests/m1_gpu_benchmark.rs`, `nvcc --version`, `rustc --version`,
`cargo --version`, and
`nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv`.
Require compute capability 12.0 for RTX 5090 and 8.9 for the RTX 6000 Ada
board, one visible GPU, no other compute process, and zero utilization before
each arm.

For a minimally credible comparison, build both once, then run process-level
arms in `main,triad,triad,main` order and repeat that block. Keep all raw logs;
compute per `(board,path,B)` ratios as `triad_us/main_us` (below 1 wins), using
paired block ratios rather than ratios of unrelated historical medians. Do not
pool RTX 5090 and Ada, eager and graph, or different batches. If this smoke
result motivates a performance claim, replace the single averages with a
dedicated same-process ABBA/BAAB harness and raw-window quantiles.

## Interpretation boundary

- The harness is identical but the surrounding inference implementation is
  intentionally not: the new side adds route manifests, prepared resources,
  graph-plan validation, and modular/architecture-specific dispatch. The result
  is the aggregate effect of new Inference/Triad, not a pure CUDA-kernel delta.
- Eager includes CPU upload/download and a synchronization per step; graph also
  leaves transfers outside capture. That is appropriate for end-to-end API
  latency but can hide GEMM-level changes.
- The benchmark advances recurrent state during warmup and timing identically
  in both arms. It does not print an output digest. Before treating timing as
  meaningful, run a fixed-step bit/digest check (the existing
  `tests/decode_digest.rs` is the reusable pattern, but its low-level graph call
  needs the known main/new safe-versus-unsafe compatibility shim).
- Existing `internal/perf` receipts use different shapes, routing, pairing, or
  denominator semantics. They are useful context only and must not be combined
  with these new logs to state a speedup.
