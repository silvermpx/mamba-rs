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

The block norms (`norm`, the gated MLP's `norm2` when `d_intermediate` is
set, and `norm_f`) are RMSNorm, or LayerNorm with a bias for a model built
with `rms_norm = false` (`MambaConfig::block_norm`). Both take
`rms_norm_eps`; FalconMamba's mixer norms stay RMSNorm either way.

A backbone built with `final_norm = false` leaves `norm_f` out: it is a
segment of a stack that continues in the caller's own layers, and its
output is the f32 residual stream after its last block in every
precision. A segment built with an identity input carries that stream on.

## Modular API

Three levels matching the original architecture:

```rust
// Level 1: Pure mixer — no norm, no residual (like Mamba class in mamba_simple.py)
mamba_layer_step(input, output, layer_weights, state, scratch, cfg);

// Level 2: Block — pre-norm + mixer + residual, then the gated MLP branch
// for a model with one (like Block class in block.py)
mamba_block_step(hidden, (layer_weights, mlp), state, scratch, cfg);

// Level 3: Full backbone — input_proj + N blocks + norm_f
mamba_step(input, output, weights, states, scratch, cfg, input_dim);

// Full-sequence variant of level 3: one batched-SGEMM pass over
// T positions instead of T step dispatches; state carries in AND out so
// mamba_step continues from it (prefill-then-decode).
forward_mamba_backbone_prefill(out, input, weights, state, scratch, dims);
```

## Recurrent State

Two states per layer persist from token to token:

- `conv_state`, `[(d_conv - 1) * d_inner]`: the conv1d history. The
  training pipeline uses a shift register `d_conv` wide, and the prefill
  widens on entry and writes the last `d_conv - 1` entries back on exit.
- `ssm_state`, `[d_inner, d_state]`: the SSM hidden state.

## Weight Layout

| Weight | Shape | Bias |
|--------|-------|------|
| in_proj | [d_model, 2*d_inner] | with `proj_bias` |
| conv1d | [d_inner, d_conv] | unless `conv_bias` is false |
| x_proj | [d_inner, dt_rank + 2*d_state] | No |
| dt_proj | [dt_rank, d_inner] | Yes |
| A_log | [d_inner, d_state] | n/a |
| D | [d_inner] | n/a |
| out_proj | [d_inner, d_model] | with `proj_bias` |
| norm | [d_model] | n/a |
| norm2 (MLP) | [d_model] | n/a |
| fc1 (MLP) | [d_model, 2*hidden] | No |
| fc2 (MLP) | [hidden, d_model] | No |

`dt_rank` is `ceil(d_model / 16)` unless the config gives one. The three
options are the reference's `dt_rank`, `bias` and `conv_bias` (in the HF
format `time_step_rank`, `use_bias` and `use_conv_bias`); an absent bias
is an empty tensor, so a model without one holds and saves nothing for it.

A config with `d_intermediate > 0` puts the reference `GatedMLP` after the
mixer in every layer: `x += fc2(value * silu(gate))` with
`[value | gate] = fc1(norm2(x))`, the hidden width `d_intermediate` rounded
up to a multiple of 128 (the `Block` of `mamba_ssm/modules/block.py` with
the MLP of `modules/mlp.py`). The official Mamba-1 checkpoints have none;
the same MLP serves Mamba-2 and Mamba-3.

## Numeric routes: the scan and the GEMM

A numeric route is the pair of scan implementation and GEMM mode. Bits are
stable within a route: run to run, eager against a captured graph, and
between the saving and the non-saving prefill. Across routes only
tolerance parity holds, because different reduction orders are different
bit families.

The scan is chosen by `ScanMode::{Sequential, Parallel, Auto}` on
`MambaConfig`. `Auto` uses the parallel scan above T = 64, and any
`d_state` above 64 uses it at every length. That rule dates from a time
when the sequential kernels were capped at 64 and stays because it is part
of the numeric identity of existing runs. At long sequences the parallel
scan is also the faster path.

The GEMM is chosen by the context's `GemmMode`. In the default
`Deterministic` mode a model context uses the Inference kernels and a
trainer the Triad kernels. `CublasFast` and `CublasPedantic` select cuBLAS.
The modes, the storage precisions, the environment variable and the
graph rules are described in
[gemm-modes.md](gemm-modes.md), and the measurements in
[determinism-benchmarks.md](determinism-benchmarks.md).

## Checkpoint provenance

`serialize` stores `scan_mode` and `rms_norm_eps` in the checkpoint, so a
loaded model reproduces the route it was trained with, and `rms_norm =
false` with the norms' biases for a LayerNorm model.
