# Mamba-3 MIMO architecture

Reference: Lahoti et al., *Mamba-3: Improved Sequence Modeling using State
Space Principles* (ICLR 2026, arXiv 2603.15569), and the reference module
`mamba_ssm/modules/mamba3.py` of state-spaces/mamba with `is_mimo=True`.
The SISO model is described in [mamba3-architecture.md](mamba3-architecture.md).

## What MIMO changes

A SISO head feeds its state one key and one value per token and reads one
output back. A MIMO head keeps one state of the same size but feeds it
`mimo_rank` (R) key and value pairs per token and reads R outputs back. The
values and the gate come from the token's x and z through per-head
projections of rank R (`mimo_x`, `mimo_z`), and `mimo_o` mixes the R
outputs into one. The state update does more arithmetic per byte of state
it reads, which is the point of the design on hardware that is bound by
memory.

MIMO also rotates its keys and queries in half-split pairs `(a, a + N/2)`,
where SISO rotates adjacent pairs, so a rank-1 MIMO model is not a SISO
model.

## Pipeline

```
    input [B, T, input_dim]
        |
    input_proj (linear + bias, or none when input_dim = d_model)
        |
        v
    +------------------- x N layers -------------------+
    |                                                   |
    |   residual                                        |
    |      |                                            |
    |   RmsNorm                                         |
    |      |                                            |
    |   in_proj, split                                  |
    |   [z | x | B | C | dd_dt | dd_A | trap | angles]  |
    |      |                                            |
    |   B and C RmsNorm per (rank, group) row           |
    |      |                                            |
    |   per head:                                       |
    |     discretise, advance the angles               |
    |     R keys and queries: bias, half-split RoPE     |
    |     u_r = mimo_x_r * x                            |
    |     state rows, y_r += D u_r                      |
    |     o = sum_r mimo_o_r * y_r * silu(mimo_z_r * z) |
    |      |                                            |
    |   out_proj                                        |
    |      |                                            |
    |   + residual                                      |
    |      |                                            |
    |   gated MLP (when d_intermediate > 0)             |
    |                                                   |
    +---------------------------------------------------+

    norm_f (RmsNorm)
        |
    output [B, T, d_model]
```

## The step of one head

The discretisation is SISO's trapezoid:

```
A      = min(-heavy_tail(dd_A), -a_floor)
dt     = softplus(dd_dt + dt_bias)
alpha  = exp(A * dt)
lambda = sigmoid(trap)
gamma  = lambda * dt
beta   = alpha * dt * sigmoid(-trap)
```

`sigmoid(-trap)` stands for `1 - lambda`, as in the reference's chunked
kernel, because it stays exact where `1 - sigmoid` would cancel. The
angles advance by `tanh(angles_raw) * pi * dt`, summed in f64 and wrapped
into `[0, 2 pi)`.

For each channel `p` of the head and each state entry `n`:

```
s[p][n] = alpha * s[p][n] + sum_r (gamma * u_r[p] * kc_r[n] + beta * up_r[p] * kp_r[n])
y_r[p]  = sum_n s[p][n] * q_r[n] + D * u_r[p]
o[p]    = sum_r mimo_o[r][p] * y_r[p] * silu(mimo_z[r][p] * z[p])
```

`kc_r` and `q_r` are this token's keys and queries (the group's
normalised B and C rows plus the head's bias, rotated), `kp_r` and `up_r`
the previous token's keys and values, kept in the state. With
`is_outproj_norm` every rank's readout row is normalised over the head's
channels before the gate.

## Recurrent state

Per layer, sequence and head:

- the SSM state, `[headdim, d_state]`
- the previous token's keys, `[R, d_state]`
- the previous token's raw input, `[headdim]`
- the angles, `[num_rope_angles]`

All f32 in every storage precision. A prompt run through `prefill` or
`forward_sequence` leaves all four where a token-by-token run would.

## Weights

| weight | shape |
|---|---|
| `norm` | `[d_model]` |
| `in_proj` | `[d_model, in_proj_dim]` |
| `dt_bias` | `[nheads]` |
| `b_norm`, `c_norm` | `[d_state]` |
| `b_bias`, `c_bias` | `[nheads, R, d_state]` |
| `D` | `[nheads]` |
| `mimo_x`, `mimo_z`, `mimo_o` | `[nheads, R, headdim]` |
| `norm_gate` | `[d_inner]`, read only with `is_outproj_norm` |
| `out_proj` | `[d_inner, d_model]` |

`in_proj_dim = 2 * d_inner + 2 * R * ngroups * d_state + 3 * nheads +
num_rope_angles`. Projections are stored `[in, out]`. A model with
`d_intermediate > 0` adds a gated MLP after every mixer.

## On the GPU

The decode step and the sequential prefill are one kernel: one block per
batch row and head walks the window token by token, so a decode step is a
window of one and continues a prefill bit for bit. The state rows live in
shared memory; a head whose rows do not fit there runs its channels in
tiles over several blocks, with the same bits. The released ranks (1, 2, 4
and 8) are compiled with their rank as a constant, which unrolls the
per-rank arrays into registers. Any other rank up to 16 runs the same
arithmetic in the same order.

A prompt longer than 64 tokens runs the chunked scan (chunks of 64), and
so does every training window unless the configuration asks for
`ScanMode::Sequential`. The trainer runs in f32, tf32, bf16 and f16 with
f32 master weights, chunked or sequential. With `is_outproj_norm` either
scan also writes its readouts, and a kernel of its own normalises and
gates them; in training its backward runs ahead of the scan's.

| file | purpose |
|---|---|
| `mamba3_mimo.cu` | the B and C row norms, and the scan of a window token by token (decode is the window of one) |
| `mamba3_mimo_chunked.cu` | the chunked scan: the chunk prep, the chunk states, the state passing, the chunk scan and the writeback of the state |
| `mamba3_mimo_bwd.cu` | the backward of the sequential path |
| `mamba3_mimo_chunked_bwd.cu` | the backward of the chunked path |
| `gated_mlp.cu` | the gate of the MLP after the mixer |

The GEMMs, norms, element-wise kernels, AdamW, the loss scaler and the
gradient clip come from the modules the families share.

## Checkpoints

`hf::load_mamba3::load_hf_mamba3_mimo` reads the official MIMO
checkpoints. `tests/hf_m3_mimo_checkpoint_parity.rs` holds the 187m model
on the CPU and on the GPU to a float64 transcription of the reference
(`tools/oracles/m3_oracle.py`). `mamba3_mimo::serialize` saves and loads
this crate's own safetensors checkpoints.
