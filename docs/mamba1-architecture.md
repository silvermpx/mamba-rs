# Mamba SSM Architecture

Reference: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (ICLR 2024).

## Pipeline

```
    input [B, T, input_dim]
        |
    input_proj (linear + bias)
        |
        v
    +--------- x N layers ---------+
    |                               |
    |   residual                    |
    |      |                        |
    |   RmsNorm                     |
    |      |                        |
    |   in_proj ----+---- gate      |
    |      |             |          |
    |   conv1d           |          |
    |      |             |          |
    |   SiLU          SiLU          |
    |      |             |          |
    |   x_proj           |          |
    |    / | \           |          |
    |  dt  B  C          |          |
    |   |                |          |
    |  dt_proj           |          |
    |   |                |          |
    |  softplus          |          |
    |   |                |          |
    |  SSM recurrence    |          |
    |  h = A*h + B*x     |          |
    |  y = C*h + D*x     |          |
    |      |             |          |
    |      +--- gate * --+          |
    |            |                  |
    |        out_proj               |
    |            |                  |
    |      + residual               |
    |                               |
    +-------------------------------+

    norm_f (RmsNorm)
        |
    output [B, T, d_model]
```

## Modular API

Three levels matching the original architecture:

```rust
// Level 1: Pure mixer — no norm, no residual (like Mamba class in mamba_simple.py)
mamba_layer_step(input, output, layer_weights, state, scratch, cfg);

// Level 2: Block — pre-norm + mixer + residual (like Block class in block.py)
mamba_block_step(hidden, layer_weights, state, scratch, cfg);

// Level 3: Full backbone — input_proj + N blocks + norm_f
mamba_step(input, output, weights, states, scratch, cfg, input_dim);

// Full-sequence variant of level 3: one batched-SGEMM pass over
// T positions instead of T step dispatches; state carries in AND out so
// mamba_step continues from it (prefill-then-decode).
forward_mamba_backbone_prefill(out, input, weights, state, scratch, dims);
```

## Recurrent State

2 persistent states per layer:
- `conv_state`: `[(d_conv - 1) * d_inner]` — conv1d history (the training
  pipeline uses a `d_conv`-wide shift register; the prefill widens on
  entry and writes the last `d_conv - 1` entries back on exit)
- `ssm_state`: `[d_inner, d_state]` — SSM hidden state

## Weight Layout

| Weight | Shape | Bias |
|--------|-------|------|
| in_proj | [d_model, 2*d_inner] | No |
| conv1d | [d_inner, d_conv] | Yes |
| x_proj | [d_inner, dt_rank + 2*d_state] | No |
| dt_proj | [dt_rank, d_inner] | Yes |
| A_log | [d_inner, d_state] | — |
| D | [d_inner] | — |
| out_proj | [d_inner, d_model] | No |
| norm | [d_model] | — |

## Numeric routes (scan + GEMM), and how one is selected

A "numeric route" is the pair (scan implementation, GEMM tier). Bits are
guaranteed stable WITHIN a route (run-to-run, eager vs captured graph,
save vs nosave prefill); ACROSS routes only tolerance parity holds —
different reduction orders are different bit families, permanently.

Scan: `ScanMode::{Sequential, Parallel, Auto}` on `MambaConfig`;
`use_parallel(T, d_state)` is the single dispatch predicate (Auto routes
parallel above T=256; `d_state > 64` always forces parallel because the
sequential kernels cap per-thread state at 64). At the classifier shape
(T=4621) the parallel scan is both ~5x faster and numerically preferable
(~220x shorter rounding chains).

GEMM tiers, per `GpuCtx` flags:
- default: cuBLAS (TF32 for f32 sgemm, PEDANTIC f32-accumulate for typed);
- `set_fast_gemm(true)`: typed GEMMs use non-PEDANTIC `CUBLAS_COMPUTE_32F`
  (tensor-core cuBLAS kernels; opt-in, unmeasured — see changelog);
- `set_batch_invariant(true)`: forward/dW/dX and the typed decode matvec on
  custom fixed-order kernels (deterministic, batch-invariant);
- `set_bi_gemm_family(..)`: which family serves the forward under that flag
  — `Triad` (`gemm_bi_triad/`, default; all three layouts, per-bucket
  invariance) or `Fixed` (`gemm_bi_fixed/`; forward-only, a bit-identical
  tile ladder with `SPLIT_K=1`, invariant by construction);
- + `set_bi_tensor_cores(true)`: permission to use the separately identified
  deterministic `mma.sync` contract. CC12.0 automatic dispatch is sealed to
  the qualified 18-cell BF16/F16 NN/TN/NT table, whose physical routes include
  both BK32 and BK64 schedules. CC12.1 and shapes outside that table decline
  the SM120 route and continue through the portable deterministic ladder;
- `set_f32_triad_policy(..)`: exact scalar F32 is the default. The TF32 policy
  permits a frozen, separately identified deterministic TF32 route and falls
  back to exact scalar `__fmaf_rn` when no such route is qualified.

The deterministic custom routes assign each output to one owner and reduce K
in a fixed ascending order. Numerical atomics and dynamic Split-K reductions
are not part of the contract. The public typed forward, dW, and dX calls are
the normal integration surface. The low-level forced SM120 resolver, tensor-map
preparation, launch, and replay-validation calls exist for qualification and
route census; forcing one does not make it eligible for automatic dispatch.

Graph captures snapshot the full route (`ctx.gemm_route()`, including policy,
physical schedule, compiler, artifact, and device identity) and replays assert
it; the split forward/backward cycle refuses a mid-cycle flip. A qualified
SM120 route must be prepared once in eager execution. Capture fails closed if
its tensor-map cache entry is missing, stale for the current managed-allocation
epoch, or backed by an untracked allocation; unsupported routes keep using the
existing fallback rather than silently changing the numeric contract.
Checkpoint provenance:
`serialize` carries `scan_mode` + `rms_norm_eps` in the checkpoint.
