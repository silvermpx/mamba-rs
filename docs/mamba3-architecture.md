# Mamba-3 SISO Architecture

Reference: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles* (ICLR 2026, arXiv 2603.15569).

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
    |   in_proj (8-way split)       |
    |   /  |  |  |  |  |  |  \     |
    |  z   x  B  C  dt  A  λ  θ    |
    |      |  |  |  |   |  |  |    |
    |      | BCNorm    softplus     |
    |      |  |  |  |   |  sig     |
    |      | bias  bias clamp      |
    |      |  |  |  |              |
    |      | RoPE(θ) on B,C        |
    |      |  |  |                 |
    |      | Trapezoidal SSM       |
    |      | h = α*h + β*Bx_prev  |
    |      |        + γ*Bx_cur    |
    |      | y = C·h + D*x        |
    |      |                       |
    |      +--- y * SiLU(z) ---+   |
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

## Key Differences from Mamba SSM

| Feature | Mamba SSM | Mamba-3 SISO |
|---------|---------|-------------|
| Conv1d | Yes | No |
| A matrix | Fixed (A_log) | Input-dependent per-head |
| Integration | Exponential (2 terms) | Trapezoidal (3 terms) |
| RoPE | No | Per-head angles [0, 2π) |
| B/C projection | Single d_state | Multi-head + BCNorm |
| D parameter | Per-channel | Per-head |
| in_proj split | 2-way (x + gate) | 8-way (z, x, B, C, dt, A, λ, θ) |

## Trapezoidal Integration

```
α = exp(A · dt)                    # decay
β = α · dt · (1 - σ(λ))           # previous contribution
γ = σ(λ) · dt                      # current contribution
h_new = α·h + β·(B_prev ⊗ x_prev) + γ·(B_cur ⊗ x_cur)
y = C · h_new + D · x
```

Where `σ(λ) = sigmoid(λ_raw)` is a learned mixing parameter. When `σ(λ) = 0.5`, this recovers the classical trapezoidal rule.

## RoPE Angle Accumulation

Per-head angles accumulate over time:
```
θ[h,a] += tanh(θ_raw[a]) · π · dt[h]
θ[h,a] = θ[h,a] mod 2π
```

Applied as 2D rotation pairs to B and C before SSM recurrence.

## Recurrent State

4 persistent states per layer:
- `ssm_state`: `[nheads, headdim, d_state]` — SSM hidden state
- `k_state`: `[nheads, d_state]` — previous K (post-RoPE B)
- `v_state`: `[nheads, headdim]` — previous x
- `angle_state`: `[nheads, num_rope_angles]` — cumulative RoPE angles

## Weight Layout

| Weight | Shape | Bias |
|--------|-------|------|
| in_proj | [d_model, in_proj_dim] | No |
| dt_bias | [nheads] | — |
| b_norm_weight | [d_state] | — |
| c_norm_weight | [d_state] | — |
| b_bias | [nheads * d_state] | — |
| c_bias | [nheads * d_state] | — |
| D | [nheads] | — |
| norm_gate_weight | [d_inner] | — |
| out_proj | [d_inner, d_model] | No |
| norm | [d_model] | — |

Where `in_proj_dim = 2·d_inner + 2·ngroups·d_state + 3·nheads + num_rope_angles`.

## CUDA Kernels (47 total)

| File | Kernels | Purpose |
|------|---------|---------|
| mamba3_ssd.cu | 5 | Sequential SSM forward/backward |
| mamba3_ops.cu | 19 | Split, BCNorm, RoPE, ABG, gating |
| mamba3_chunked.cu | 15 | Chunked parallel scan (T>64) |
| norms.cu | 3 | RMSNorm forward/backward |
| elementwise.cu | 5 | Residual, fill, gather, vec ops |
