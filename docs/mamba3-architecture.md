# Mamba-3 SISO Architecture

Reference: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles* (ICLR 2026, arXiv 2603.15569).
The MIMO variant is described in [mamba3-mimo-architecture.md](mamba3-mimo-architecture.md).

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
    |      | BCNorm    sp / -ht     |
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

Activations in the split:

- `sp` on dt: `softplus(dd_dt + dt_bias)`.
- `-ht` on A: `-heavy_tail(dd_A)` clamped at `-a_floor`, where
  `heavy_tail` is state-spaces/mamba's `heavy_tail_activation`, `1 + x`
  for `x >= 0` and `1 / (1 - x)` for `x < 0`.
- `sig` on the trapezoid gate λ: the sigmoid.

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

Four states per layer persist from token to token:

- `ssm_state`, `[nheads, headdim, d_state]`: the SSM hidden state
- `k_state`, `[nheads, d_state]`: the previous K (B after RoPE)
- `v_state`, `[nheads, headdim]`: the previous x
- `angle_state`, `[nheads, num_rope_angles]`: the accumulated RoPE angles

All four carry through `forward_mamba3_backbone_prefill` (the
full-sequence batched-SGEMM CPU forward, no activation tape) exactly as
through `mamba3_step`, so prefill-then-decode is seamless.

GEMM routes: the Mamba-3 trainer, the inference prefill and the decode
step all multiply through the context, so the context's `GemmMode`
applies to all of them: `Deterministic` by default (a model context uses
the Inference kernels, a trainer the Triad kernels) or one of the two
cuBLAS modes. The tied half-input LM heads keep f32 logits. The prefill's
pooled output is per sample (`[B * d_model]`), each sample summed over its
own rows, so batching a prefill does not change a sample's bits. The modes
are described in [gemm-modes.md](gemm-modes.md).

The official SISO checkpoints load through
`hf::load_mamba3::load_hf_mamba3`. `tests/hf_m3_checkpoint_parity.rs`
holds the 187m model on the CPU and on the GPU to a float64 transcription
of the reference (`tools/oracles/m3_oracle.py`).

## Weight Layout

| Weight | Shape | Bias |
|--------|-------|------|
| in_proj | [d_model, in_proj_dim] | No |
| dt_bias | [nheads] | n/a |
| b_norm_weight | [d_state] | n/a |
| c_norm_weight | [d_state] | n/a |
| b_bias | [nheads * d_state] | n/a |
| c_bias | [nheads * d_state] | n/a |
| D | [nheads] | n/a |
| norm_gate_weight | [d_inner] | n/a |
| out_proj | [d_inner, d_model] | No |
| norm | [d_model] | n/a |

Where `in_proj_dim = 2·d_inner + 2·ngroups·d_state + 3·nheads + num_rope_angles`.

The block norms (`norm`, the MLP's `norm2`, `norm_f`) take `rms_norm_eps`,
the reference's `norm_epsilon`; the B/C norms and the gated output norm take
`ssm_norm_eps`, which the reference fixes at 1e-5.

## CUDA kernels

| File | Kernels | Purpose |
|------|---------|---------|
| mamba3_siso.cu | 5 | Sequential SSM forward/backward, the decode step |
| mamba3_ops.cu | 19 | Split, BCNorm, RoPE, ABG, gating |
| mamba3_chunked.cu | 16 | Chunked parallel scan (T>64) |
| norms.cu | 3 | RMSNorm forward/backward |
| elementwise.cu | 12 | Residual, fill, last-step gather, column sums, vec ops |
| gemm_bi_inference/casts.cu | 4 | Casts between f32 and bf16/f16 |

Counted by kernel family. The typed variants (f32, bf16, f16) of a family
share one row.

## Chunked scan and carried state (contract)

The chunked parallel scan carries state in one direction only. The GPU
prompt prefill carries all four recurrent states from one window into the
next through the trapezoidal boundary fold, so a long prompt can be run in
several windows and decoded from the result. The training path does not:
every training window starts from a zero state, and the state a window
writes back is not consumed by the next one. Consequences a consumer must
respect:

- `reset_state()` before a chunked training window is a no-op by design.
  The window never sees a nonzero incoming state either way.
- Packing several documents into one row is not masked, on this path or in
  the reference implementation: the scan would carry state across the
  document boundary inside a window. Train one document (or one padded
  page) per row, as the shipped trainers do.
- Carrying state across training windows (truncated BPTT) is not
  available: the backward of the boundary fold is not implemented.
