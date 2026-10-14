//! O2+O3+O4+O5 optimized Mamba forward and backward passes using flat buffers.
//!
//! Four optimization layers applied to the Mamba layer:
//!
//! - **O5**: Uses [`MambaLayerFlat`] (single contiguous `[T*stride]` buffer) instead of
//!   `Vec<Vec<f32>>` (18 heap allocations per timestep).
//! - **O3**: Pre-computed `a_neg[idx] = -exp(a_log[idx])` passed in, avoiding
//!   `d_inner * d_state` transcendentals per timestep.
//! - **O4**: Fused conv1d depthwise dot product + SiLU into a single loop
//!   (eliminates separate SiLU pass over `d_inner` elements).
//! - **O2**: Batched SGEMM for in_proj/out_proj across all T timesteps.
//!
//! Input/output is `&mut [f32]` flat `[T * d_model]` instead of `&mut [Vec<f32>]`.
//! The `pre_norm` field is removed (was duplicate of `residual`).
//!
//! ## Module structure
//!
//! - Batched SGEMM: `forward_mamba_layer_batched` / `backward_mamba_layer_batched`
//! - Full backbone: `forward_mamba_backbone_batched` / `backward_mamba_backbone_batched`

use super::flat::{MambaBackboneFlat, MambaLayerFlat};
use super::scratch::PhaseScratch;
use super::weights::{TrainMambaLayerWeights, TrainMambaWeights};
use crate::mlp::cpu::{RowsGemm, gated_mlp_rows, mlp_norm_row};
use crate::ops::blas::sgemm_forward;
use crate::ops::dims::{MambaDims, MambaRecurrentState};
use crate::ops::fast_math::{fast_exp_inplace, fast_exp_scalar};
use crate::ops::norms::{bcdt_rms_norm, layer_norm_inplace, layer_norm_row};
use crate::ops::rounding::{store, stored};

/// One step of a channel's selective scan: the decays `exp(delta * A)` into
/// `da` and the state `h` advanced in place, `h = da * h + delta * u * B`.
/// The backward replays a segment's states through it, so they are the
/// forward's bits.
#[inline(always)]
pub(crate) fn scan_advance(
    h: &mut [f32],
    da: &mut [f32],
    (delta_d, u_d): (f32, f32),
    (a_neg, b): (&[f32], &[f32]),
) {
    let delta_u_d = delta_d * u_d;
    for (dn, &a) in da.iter_mut().zip(a_neg) {
        *dn = delta_d * a;
    }
    fast_exp_inplace(da);
    for ((hn, &dn), &bn) in h.iter_mut().zip(da.iter()).zip(b) {
        *hn = dn * *hn + delta_u_d * bn;
    }
}

/// O2 optimized single-layer Mamba forward pass: 7-phase pipeline with batched SGEMM.
///
/// Restructures the per-timestep forward into phases where the two biggest SGEMM
/// calls (in_proj and out_proj) are batched across ALL timesteps in a single call.
/// The SSM core (phase F4) remains sequential since SSM recurrence is inherently
/// sequential.
///
/// Amortises SGEMM dispatch overhead across `T` timesteps.
///
/// # 7 Phases
///
/// - **F1**: Batch RmsNorm (loop over T)
/// - **F2**: Batch in_proj — ONE SGEMM call `[T * d_model] -> [T * 2*d_inner]`
/// - **F3**: Split x/gate + batch SiLU (loop over T)
/// - **F4**: Sequential SSM core (per-timestep: conv1d, x_proj, dt_proj, SSM, gating)
/// - **F5**: Batch out_proj — ONE SGEMM call `[T * d_inner] -> [T * d_model]`
/// - **F6**: Residual add (all T)
///
/// In a reduced precision (the scratch's) the values between the phases
/// are stored ones, as the GPU trainer keeps them: the normalized rows,
/// every projection's output, the gate after its SiLU, the conv output
/// after its SiLU, delta after the softplus, the scan output and the gated
/// rows. TF32 rounds every GEMM operand instead. The residual stream, the
/// conv and SSM states, the norms' statistics and the decay factors stay
/// f32.
///
/// # Arguments
///
/// - `temporal_flat`: `[T * d_model]` — input on entry, output on exit (in-place).
/// - `acts`: flat activation storage for this layer.
/// - `layer_w`: this layer's weights.
/// - `conv_state`: `[d_inner * d_conv]` — persistent conv1d shift register state.
/// - `ssm_state`: `[d_inner * d_state]` — persistent SSM hidden state.
/// - `a_neg`: `[d_inner * d_state]` — pre-computed `-exp(a_log[idx])`.
/// - `scratch`: per-phase scratch buffers, allocated once and reused.
/// - `dims`: collected Mamba dimensions.
pub fn forward_mamba_layer_batched(
    temporal_flat: &mut [f32],
    acts: &mut MambaLayerFlat,
    layer_w: &TrainMambaLayerWeights,
    state: &mut MambaRecurrentState<'_>,
    scratch: &mut PhaseScratch,
    dims: &MambaDims,
) {
    let conv_state = &mut *state.conv;
    let ssm_state = &mut *state.ssm;
    let a_neg = state.a_neg;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let seq_len = dims.seq_len;
    let dtype = scratch.dtype;
    let lane = (sgemm_forward as RowsGemm, dtype);

    // ===================================================================
    // Phase F1: the batch's block norm (loop over T)
    // ===================================================================
    for t in 0..seq_len {
        let off = t * dm;
        let src = &temporal_flat[off..off + dm];

        // Save residual
        acts.residual_mut(t).copy_from_slice(src);

        // Write post_norm into scratch buffer for batched in_proj
        let pn = &mut scratch.post_norm_flat[off..off + dm];
        if layer_w.norm_bias.is_empty() {
            let mut sum_sq = 0.0_f32;
            for &v in &src[..dm] {
                sum_sq += v * v;
            }
            let mean_sq = sum_sq / dm as f32;
            let rms = (mean_sq + dims.rms_norm_eps).sqrt();
            acts.set_rms_val(t, rms);
            let inv_rms = 1.0 / rms;
            for d in 0..dm {
                pn[d] = src[d] * inv_rms * layer_w.norm_weight[d];
            }
        } else {
            let stats = layer_norm_row(
                pn,
                src,
                (&layer_w.norm_weight, &layer_w.norm_bias),
                dims.rms_norm_eps,
            );
            acts.set_norm_stats(t, stats);
        }
        store(dtype, pn);

        // Also save post_norm into acts (needed for backward)
        acts.post_norm_mut(t).copy_from_slice(pn);
    }

    // ===================================================================
    // Phase F2: Batch in_proj -- ONE SGEMM call
    //           [T * d_model] -> [T * 2*d_inner]
    // ===================================================================
    scratch.gemm.forward(
        lane,
        (&mut scratch.proj_flat, &scratch.post_norm_flat),
        (&layer_w.in_proj_w, super::bias(&layer_w.in_proj_b)),
        (seq_len, dm, 2 * di),
    );

    // ===================================================================
    // Phase F3: Split x/gate + batch SiLU (loop over T)
    // ===================================================================
    for t in 0..seq_len {
        let proj_off = t * 2 * di;

        // x_branch = proj[..d_inner]
        acts.x_branch_mut(t)
            .copy_from_slice(&scratch.proj_flat[proj_off..proj_off + di]);

        // gate_pre_silu = proj[d_inner..2*d_inner]
        let gate = &scratch.proj_flat[proj_off + di..proj_off + 2 * di];
        acts.gate_pre_silu_mut(t).copy_from_slice(gate);

        // SiLU(gate) -> scratch.gate_silu_flat
        let gs_off = t * di;
        for (d, &g) in gate.iter().enumerate().take(di) {
            let sig = 1.0 / (1.0 + fast_exp_scalar(-g));
            scratch.gate_silu_flat[gs_off + d] = g * sig;
        }
        store(dtype, &mut scratch.gate_silu_flat[gs_off..gs_off + di]);
        acts.gate_post_silu_mut(t)
            .copy_from_slice(&scratch.gate_silu_flat[gs_off..gs_off + di]);
    }

    // ===================================================================
    // Phase F4: conv1d (sequential state) -> BATCHED x_proj/dt_proj ->
    //           sequential SSM recurrence -> gating
    //
    // The depthwise conv depends only on x_branch (fully known after F3),
    // so conv + SiLU runs for all T first, then x_proj and dt_proj become
    // ONE SGEMM each over [T, ...] rows — exactly mirroring how backward
    // B4/B5 already batch these projections. The old code issued 2*T
    // matvecs, re-streaming both weight matrices T times.
    // ===================================================================
    let b_offset = dt_rank;
    let c_offset = dt_rank + ds;

    // -- F4a: conv1d shift + fused SiLU for ALL timesteps --
    // The register the window starts from; the backward rebuilds every
    // later one from it and the x_branch rows.
    acts.conv_init.copy_from_slice(&conv_state[..di * dc]);
    for t in 0..seq_len {
        {
            let x_branch = acts.x_branch(t);
            for (d, &xb) in x_branch.iter().enumerate().take(di) {
                let base = d * dc;
                for k in 0..dc - 1 {
                    conv_state[base + k] = conv_state[base + k + 1];
                }
                conv_state[base + dc - 1] = xb;
            }
        }

        // Fused depthwise conv1d dot product + SiLU. u is written both to
        // acts (backward reads it) and to the contiguous u_flat rows the
        // batched x_proj SGEMM consumes.
        {
            let step_base = t * acts.offsets.step_stride;
            let pc_start = step_base + acts.offsets.post_conv;
            let u_start = step_base + acts.offsets.u;
            let uf = t * di;
            for d in 0..di {
                let base = d * dc;
                let mut val = layer_w.conv1d_bias.get(d).copied().unwrap_or(0.0);
                for k in 0..dc {
                    val += conv_state[base + k] * layer_w.conv1d_weight[base + k];
                }
                acts.data[pc_start + d] = val;
                let u = stored(dtype, val / (1.0 + fast_exp_scalar(-val)));
                acts.data[u_start + d] = u;
                scratch.u_flat[uf + d] = u;
            }
        }
    }

    // -- F4b: x_proj as ONE SGEMM over all T rows --
    scratch.gemm.forward(
        lane,
        (
            &mut scratch.xdbl_flat[..seq_len * xdbl_dim],
            &scratch.u_flat[..seq_len * di],
        ),
        (&layer_w.x_proj_w, None),
        (seq_len, di, xdbl_dim),
    );
    for t in 0..seq_len {
        let row = &mut scratch.xdbl_flat[t * xdbl_dim..(t + 1) * xdbl_dim];
        // FalconMamba normalizes the row's dt, B and C parts and keeps the
        // scales for the backward.
        if let Some(eps) = dims.bcdt_rms_eps {
            let r = bcdt_rms_norm(row, (dt_rank, ds), eps);
            store(dtype, row);
            acts.bcdt_rstd_mut(t).copy_from_slice(&r);
        }
        acts.xdbl_mut(t).copy_from_slice(row);
        scratch.dt_in_flat[t * dt_rank..(t + 1) * dt_rank].copy_from_slice(&row[..dt_rank]);
    }

    // -- F4c: dt_proj as ONE SGEMM + softplus --
    scratch.gemm.forward(
        lane,
        (
            &mut scratch.delta_raw_flat[..seq_len * di],
            &scratch.dt_in_flat[..seq_len * dt_rank],
        ),
        (&layer_w.dt_proj_w, Some(&layer_w.dt_proj_b)),
        (seq_len, dt_rank, di),
    );
    for t in 0..seq_len {
        let step_base = t * acts.offsets.step_stride;
        let dr_start = step_base + acts.offsets.delta_raw;
        let d_start = step_base + acts.offsets.delta;
        let df = t * di;
        for d in 0..di {
            let raw = scratch.delta_raw_flat[df + d];
            acts.data[dr_start + d] = raw;
            acts.data[d_start + d] = if raw > 20.0 {
                raw
            } else {
                stored(dtype, fast_exp_scalar(raw).ln_1p())
            };
        }
    }

    let stride = acts.stride;
    for t in 0..seq_len {
        // -- F4d: SSM recurrence, the state checkpointed every `stride` steps --
        if t % stride == 0 {
            acts.ssm_ckpt_mut(t / stride)
                .copy_from_slice(&ssm_state[..di * ds]);
        }

        {
            let step_base = t * acts.offsets.step_stride;
            let delta_start = step_base + acts.offsets.delta;
            let u_start = step_base + acts.offsets.u;
            let xdbl_start = step_base + acts.offsets.xdbl;
            let y_start = step_base + acts.offsets.y;

            for d in 0..di {
                let delta_d = acts.data[delta_start + d];
                let u_d = acts.data[u_start + d];
                let a_base = d * ds;

                // `da[n] = exp(delta * a_neg[n])` across the d_state lane
                // via SIMD `fast_exp_inplace` (NEON 4-wide, AVX2 8-wide),
                // the bits of the per-element `fast_exp_scalar`.
                let h = &mut ssm_state[a_base..a_base + ds];
                let xdbl = &acts.data[xdbl_start..xdbl_start + xdbl_dim];
                scan_advance(
                    h,
                    &mut scratch.da_buf[..ds],
                    (delta_d, u_d),
                    (&a_neg[a_base..a_base + ds], &xdbl[b_offset..b_offset + ds]),
                );
                let mut y_d = 0.0_f32;
                for (&h_n, &c_n) in h.iter().zip(&xdbl[c_offset..c_offset + ds]) {
                    y_d += h_n * c_n;
                }

                y_d += layer_w.d_param[d] * u_d;
                acts.data[y_start + d] = stored(dtype, y_d);
            }
        }

        // -- F4e: gating: gated = y * gate_post_silu --
        {
            let step_base = t * acts.offsets.step_stride;
            let y_start = step_base + acts.offsets.y;
            let gpost_start = step_base + acts.offsets.gate_post_silu;
            let gated_start = step_base + acts.offsets.gated;
            let gs_off = t * di;
            for d in 0..di {
                scratch.gated_flat[gs_off + d] =
                    stored(dtype, acts.data[y_start + d] * acts.data[gpost_start + d]);
                acts.data[gated_start + d] = scratch.gated_flat[gs_off + d];
            }
        }
    }

    // ===================================================================
    // Phase F5: Batch out_proj -- ONE SGEMM call
    //           [T * d_inner] -> [T * d_model]
    // ===================================================================
    scratch.gemm.forward(
        lane,
        (&mut scratch.out_flat, &scratch.gated_flat),
        (&layer_w.out_proj_w, super::bias(&layer_w.out_proj_b)),
        (seq_len, di, dm),
    );

    // ===================================================================
    // Phase F6: Residual add (all T)
    // ===================================================================
    for t in 0..seq_len {
        let off = t * dm;
        let residual = acts.residual(t);
        for d in 0..dm {
            temporal_flat[off + d] = residual[d] + scratch.out_flat[off + d];
        }
    }
}

// ---------------------------------------------------------------------------
// Full Mamba backbone — O2 batched SGEMM
// ---------------------------------------------------------------------------

/// O2 optimized full Mamba backbone forward pass with batched SGEMM.
///
/// Batches the input projection into a single `sgemm_forward(T, mamba_input_dim,
/// d_model)` call, and calls [`forward_mamba_layer_batched`] for each layer.
///
/// # Arguments
///
/// - `temporal_flat`: `[T * d_model]` — written by input projection, then
///   modified in-place by each Mamba layer.
/// - `acts`: flat activation storage for the entire backbone.
/// - `mamba_w`: all Mamba weights (input_proj + layers).
/// - `mamba_input_flat`: `[T * mamba_input_dim]` — input sequence.
/// - `conv_states`: `[n_layers * d_inner * d_conv]` — persistent conv states.
/// - `ssm_states`: `[n_layers * d_inner * d_state]` — persistent SSM states.
/// - `a_neg_all`: `[n_layers * d_inner * d_state]` — pre-computed `-exp(a_log)`.
/// - `scratch`: per-phase scratch buffers (allocated once, reused).
/// - `dims`: collected Mamba dimensions.
pub fn forward_mamba_backbone_batched(
    temporal_flat: &mut [f32],
    acts: &mut MambaBackboneFlat,
    mamba_w: &TrainMambaWeights,
    mamba_input_flat: &[f32],
    state: &mut MambaRecurrentState<'_>,
    scratch: &mut PhaseScratch,
    dims: &MambaDims,
) {
    super::assert_dims_fit(dims, mamba_w);
    let conv_states = &mut *state.conv;
    let ssm_states = &mut *state.ssm;
    let a_neg_all = state.a_neg;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dc = dims.d_conv;
    let mid = dims.mamba_input_dim;
    let seq_len = dims.seq_len;
    let n_layers = dims.n_layers;

    // --- Batched input projection: [T * mid] -> [T * dm] ---
    // A model without one (the official checkpoints, whose embedding feeds
    // the first norm) takes its input as the rows themselves.
    if mamba_w.input_proj_w.is_empty() {
        temporal_flat[..seq_len * dm].copy_from_slice(&mamba_input_flat[..seq_len * dm]);
    } else {
        // Save inputs for backward. A half lane casts them to its storage,
        // the GEMMs read them and its output so stored; the residual
        // stream starts from that output.
        let inputs = &mut acts.input_proj_inputs[..seq_len * mid];
        inputs.copy_from_slice(&mamba_input_flat[..seq_len * mid]);
        store(scratch.dtype, inputs);
        scratch.gemm.forward(
            (sgemm_forward as RowsGemm, scratch.dtype),
            (&mut temporal_flat[..seq_len * dm], inputs),
            (&mamba_w.input_proj_w, Some(&mamba_w.input_proj_b)),
            (seq_len, mid, dm),
        );
    }

    // --- Mamba layers (batched) ---
    let conv_per_layer = di * dc;
    let ssm_per_layer = di * ds;
    let a_neg_per_layer = di * ds;

    for layer_idx in 0..n_layers {
        let conv_start = layer_idx * conv_per_layer;
        let ssm_start = layer_idx * ssm_per_layer;
        let a_neg_start = layer_idx * a_neg_per_layer;

        forward_mamba_layer_batched(
            temporal_flat,
            &mut acts.layers[layer_idx],
            &mamba_w.layers[layer_idx],
            &mut MambaRecurrentState {
                conv: &mut conv_states[conv_start..conv_start + conv_per_layer],
                ssm: &mut ssm_states[ssm_start..ssm_start + ssm_per_layer],
                a_neg: &a_neg_all[a_neg_start..a_neg_start + a_neg_per_layer],
            },
            scratch,
            dims,
        );
        // The gated MLP after the mixer, its input rows kept for the
        // backward.
        if let Some(m) = mamba_w.mlp.get(layer_idx) {
            let rows = &mut temporal_flat[..seq_len * dm];
            acts.mlp_in[layer_idx].copy_from_slice(rows);
            scratch.mlp.dtype = scratch.dtype;
            gated_mlp_rows(
                rows,
                seq_len,
                m,
                &mut scratch.mlp,
                sgemm_forward,
                |dst, src| {
                    for (d_row, s_row) in dst.chunks_exact_mut(dm).zip(src.chunks_exact(dm)) {
                        mlp_norm_row(d_row, s_row, m, dims.rms_norm_eps);
                    }
                },
            );
        }
    }

    // The final norm (norm_f) after all Mamba layers.
    // Save pre-norm input for backward, then apply norm in-place.
    acts.norm_f_input[..seq_len * dm].copy_from_slice(&temporal_flat[..seq_len * dm]);
    if !mamba_w.norm_f_bias.is_empty() {
        let p = (&mamba_w.norm_f_weight[..], &mamba_w.norm_f_bias[..]);
        for (row, st) in temporal_flat[..seq_len * dm]
            .chunks_exact_mut(dm)
            .zip(acts.norm_f_rms.as_chunks_mut::<2>().0)
        {
            *st = layer_norm_inplace(row, p, dims.rms_norm_eps);
            store(scratch.dtype, row);
        }
        return;
    }
    for t in 0..seq_len {
        let off = t * dm;
        let mean_sq: f32 = temporal_flat[off..off + dm]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            / dm as f32;
        let rms = (mean_sq + dims.rms_norm_eps).sqrt();
        acts.norm_f_rms[t] = rms;
        let inv_rms = 1.0 / rms;
        for d in 0..dm {
            temporal_flat[off + d] = temporal_flat[off + d] * inv_rms * mamba_w.norm_f_weight[d];
        }
        store(scratch.dtype, &mut temporal_flat[off..off + dm]);
    }
}
