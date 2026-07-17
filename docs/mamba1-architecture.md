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
