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

// Full-sequence variant of level 3 (0.5.0): one batched-SGEMM pass over
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

GEMM tiers, per `GpuCtx` flags (see CLAUDE.md for coverage boundaries):
- default: cuBLAS (TF32 for f32 sgemm, PEDANTIC f32-accumulate for typed);
- `set_fast_gemm(true)`: typed GEMMs use non-PEDANTIC `CUBLAS_COMPUTE_32F`
  (tensor-core cuBLAS kernels; opt-in, unmeasured — see changelog);
- `set_batch_invariant(true)`: training triads + typed decode matvec on
  custom fixed-order kernels (deterministic, batch-invariant);
- + `set_bi_tensor_cores(true)`: the mma.sync tier of the same contract.

Graph captures snapshot the flags and replays assert them; the split
forward/backward cycle refuses a mid-cycle flip. Checkpoint provenance:
`serialize` carries `scan_mode` + `rms_norm_eps` from 0.5.2.
