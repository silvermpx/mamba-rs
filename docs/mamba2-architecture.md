# Mamba-2 architecture

Reference: Dao and Gu, *Transformers are SSMs: Generalized Models and
Efficient Algorithms Through Structured State Space Duality* (ICML 2024,
arXiv 2405.21060), and the reference module `mamba_ssm/modules/mamba2.py`
of state-spaces/mamba.

## Pipeline

```
    input [B, T, input_dim]
        |
    input_proj (linear + bias, or none when input_dim = d_model)
        |
        v
    +------------- x N layers -------------+
    |                                       |
    |   residual                            |
    |      |                                |
    |   RmsNorm                             |
    |      |                                |
    |   in_proj, split [z0 | x0 | z | xBC | dt]
    |      |                                |
    |   conv1d + SiLU over xBC              |
    |      |                                |
    |   split x | B | C                     |
    |      |                                |
    |   SSD scan per head                   |
    |   s = s dA + B (x dt)                 |
    |   y = C . s + D x                     |
    |      |                                |
    |   gated RMSNorm with z                |
    |   (or y silu(z) without the norm)     |
    |      |                                |
    |   [ silu(z0) x0 | y ]                 |
    |      |                                |
    |   out_proj                            |
    |      |                                |
    |   + residual                          |
    |      |                                |
    |   gated MLP (when d_intermediate > 0) |
    |                                       |
    +---------------------------------------+

    norm_f (RmsNorm)
        |
    output [B, T, d_model]
```

`d_inner = expand * d_model`. The SSM runs on `d_ssm` of those channels
(all of them when `d_ssm` is 0), in `nheads = d_ssm / headdim` heads. The
remaining `d_mlp = d_inner - d_ssm` channels form a small gated MLP inside
the mixer, `silu(z0) * x0`, whose output sits in front of `y` at the input
of `out_proj`. B and C come in `ngroups` groups, and every head reads the
group it belongs to.

## The scan

For each head, with `A = -exp(A_log)`:

```
dt = clamp(softplus(dt_raw + dt_bias), dt_min, dt_max)
dA = exp(dt * A)
s[p][n] = s[p][n] * dA + B[n] * (x[p] * dt)
y[p]    = sum_n s[p][n] * C[n] + D * x[p]
```

`s` is the head's state, `headdim x d_state`. D is one value per head, or
one per channel when `d_has_hdim` is set. The reference clamps `dt` only in
its chunked forward. This crate clamps in every path, so a decode and a
prefill of the same tokens give the same model.

## The chunked scan

On the GPU a prompt longer than 64 tokens runs the state space dual in
chunks of 64 tokens, and so does every training window, unless the
configuration asks for `ScanMode::Sequential`. For chunk `c`
of head `h` in group `g`, with `a_t = dt_t A` and the running sum
`Acs_t = a_0 + ... + a_t` restarted every chunk:

```
CB[t][s]  = sum_n C_t[n] B_s[n]                         (s <= t)
W[t][s]   = CB[t][s] * exp(min(Acs_t - Acs_s, 0)) * dt_s
xw[t][p]  = x_t[p] * exp(min(Acs_last - Acs_t, 0)) * dt_t
S[p][n]   = sum_t B_t[n] * xw[t][p]
h_c       = exp(Acs_last) * h_(c-1) + S
y_t[p]    = (sum_n C_t[n] h_(c-1)[p][n]) * exp(Acs_t) + sum_(s <= t) W[t][s] x_s[p] + D x_t[p]
```

Every value is one thread's chain in that order, with explicit
round-to-nearest intrinsics, so the tiling of the work never moves a bit.
The decode step runs the same formulas at one token, so it continues a
prefill of whole chunks exactly. `ScanMode::Sequential` runs the decode
step's recurrence over the window instead, at any length.

## Recurrent state

Per layer and sequence:

- the conv state, `[conv_dim, d_conv]`: the last `d_conv` inputs of the
  conv, the newest last
- the SSM state, `[nheads, headdim, d_state]`, f32 in every storage
  precision

A prompt run through `prefill` or `forward_sequence` leaves both where a
token-by-token run would, and the next step continues from them.

## Weights

| weight | shape | when |
|---|---|---|
| `norm` | `[d_model]` | always |
| `in_proj` | `[d_model, in_proj_dim]` | always, with a bias when `proj_bias` |
| `conv1d` | `[conv_dim, d_conv]` | always, with a bias when `conv_bias` |
| `dt_bias` | `[nheads]` | always |
| `A_log` | `[nheads]` | always |
| `D` | `[nheads]`, or `[d_ssm]` with `d_has_hdim` | always |
| `gate_norm` | `[d_ssm]` | when `rmsnorm` |
| `out_proj` | `[d_inner, d_model]` | always, with a bias when `proj_bias` |

`conv_dim = d_ssm + 2 * ngroups * d_state` and
`in_proj_dim = 2 * d_mlp + d_ssm + conv_dim + nheads`. Projections are
stored `[in, out]`, as everywhere in this crate. A model with
`d_intermediate > 0` adds a gated MLP after every mixer, with its own norm.

## Configuration

| field | meaning |
|---|---|
| `d_ssm` | channels the SSM runs on, 0 for all of `d_inner` |
| `ngroups` | B and C groups, also the groups of the gated norm |
| `d_has_hdim` | D per channel instead of per head |
| `rmsnorm` | the gated RMSNorm before `out_proj`. Without it the output is `y * silu(z)` |
| `norm_before_gate` | normalise `y` before the gate instead of after it |
| `dt_limit` | the clamp of `dt` after the softplus |
| `proj_bias`, `conv_bias` | the biases of the projections and of the conv |
| `block_norm` | the block norms (`norm`, `norm2`, `norm_f`): RMSNorm, or LayerNorm with a bias for a model built with `rms_norm = false` |
| `final_norm` | whether the backbone ends in `norm_f`; `false` builds a segment of a stack, whose output is its f32 residual stream |
| `rms_norm_eps`, `gated_norm_eps` | the epsilon of the block norms and of the gated norm, which the reference fixes at 1e-5 |
| `scan_mode` | the GPU scan over a window: `Auto`, `Sequential` or `Parallel` |

## Training

The GPU trainer runs the chunked scan by default and the sequential one
with `ScanMode::Sequential`. The sequential backward replays the
recurrence from checkpoints written every 16 tokens, so its memory does
not grow with a state per token. Both run in f32, tf32, bf16 and f16, with
f32 master weights and the dynamic loss scaler in f16. The CPU trains
through `forward_mamba2_train` and `backward_mamba2_train`.

A packing (`Varlen`, see `src/varlen.rs`) trains several sequences in
one row, as the reference does with `seq_idx`. A sequence's first token
starts from the empty state and an empty conv history; the chunk's
cumulative decay restarts there, the chunk state takes only the chunk's
last sequence, the state passes into a chunk only when no sequence starts
in the one before, and a token reads only the earlier tokens of its own
sequence. The backward cuts every carry at a start the same way.

## CUDA kernels

| file | purpose |
|---|---|
| `mamba2_ops.cu` | the decode step, the sequential scan over a window, the conv, the gates and the gated norm |
| `mamba2_ssd.cu` | the chunked scan: the chunk cumsum, the CB products, the chunk states, the state passing and the chunk scan, as plain kernels and as register tiles |
| `mamba2_bwd.cu` | the training backward of all of the above, and the sequential backward with its replay |
| `gated_mlp.cu` | the gate of the MLP after the mixer |

The norms, element-wise kernels, AdamW, the loss scaler, the gradient
clip and every GEMM come from the modules the families share, so the GEMM
modes apply to Mamba-2 as to the other families (see
[gemm-modes.md](gemm-modes.md)).

## Checkpoints

`hf::load_mamba2::load_hf_mamba2` reads the official checkpoints in the
`mamba_ssm` format (`config.json` with `ssm_cfg.layer = "Mamba2"` and
`pytorch_model.bin`). `tests/hf_m2_checkpoint_parity.rs` holds
`state-spaces/mamba2-130m` on the CPU and on the GPU to a float64
transcription of the reference step (`tools/oracles/m2_oracle.py`).
`mamba2::serialize` saves and loads this crate's own safetensors
checkpoints.
