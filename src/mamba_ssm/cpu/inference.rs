//! Single-step recurrent Mamba inference (T=1).
//!
//! Three levels of API:
//! - [`mamba_layer_step`]: Pure mixer (no norm, no residual). Matches Tri Dao's `Mamba` class.
//! - [`mamba_block_step`]: Pre-norm + mixer + residual. Matches Tri Dao's `Block` class.
//! - [`mamba_step`]: Full backbone: input_proj + N blocks + norm_f.

use crate::config::MambaConfig;
use crate::ops::fast_math::{RMS_NORM_EPS, fast_exp_inplace, fast_exp_scalar};
use crate::state::MambaLayerState;
use crate::weights::{MambaLayerWeights, MambaWeights};

// ---------------------------------------------------------------------------
// Scratch buffers
// ---------------------------------------------------------------------------

/// Scratch buffers for a single Mamba layer step (pure mixer, no norm).
pub struct MambaLayerScratch {
    /// In-proj output (x + gate concatenated) `[2 * d_inner]`.
    pub proj: Vec<f32>,
    /// x-branch after split `[d_inner]`.
    pub x: Vec<f32>,
    /// Gate branch after SiLU activation `[d_inner]`.
    pub gate_silu: Vec<f32>,
    /// x_proj output (delta_raw, B, C concatenated) `[dt_rank + 2*d_state]`.
    pub xdbl: Vec<f32>,
    /// Discretized delta after softplus `[d_inner]`.
    pub delta: Vec<f32>,
    /// SSM output `[d_inner]`.
    pub y: Vec<f32>,
    /// Batch buffer for SSM `da = exp(delta * a_neg)` SIMD vectorization `[d_state]`.
    pub da_buf: Vec<f32>,
}

impl MambaLayerScratch {
    /// Allocate scratch buffers for a single layer.
    pub fn new(cfg: &MambaConfig) -> Self {
        let d_inner = cfg.d_inner();
        Self {
            proj: vec![0.0; 2 * d_inner],
            x: vec![0.0; d_inner],
            gate_silu: vec![0.0; d_inner],
            xdbl: vec![0.0; cfg.xdbl_dim()],
            delta: vec![0.0; d_inner],
            y: vec![0.0; d_inner],
            da_buf: vec![0.0; cfg.d_state],
        }
    }
}

/// Scratch buffers for full backbone step (layers + norm + residual).
pub struct MambaStepScratch {
    /// Per-layer scratch (reused across layers).
    pub layer: MambaLayerScratch,
    /// RMSNorm intermediate `[d_model]`.
    pub norm_buf: Vec<f32>,
    /// Saved residual for skip connection `[d_model]`.
    pub residual: Vec<f32>,
}

impl MambaStepScratch {
    /// Allocate scratch buffers sized for the given Mamba configuration.
    pub fn new(cfg: &MambaConfig) -> Self {
        Self {
            layer: MambaLayerScratch::new(cfg),
            norm_buf: vec![0.0; cfg.d_model],
            residual: vec![0.0; cfg.d_model],
        }
    }
}

// ---------------------------------------------------------------------------
// Level 1: Pure Mamba layer (mixer only — no norm, no residual)
// ---------------------------------------------------------------------------

/// Pure Mamba layer step (T=1): in_proj → conv1d → SiLU → SSM → gate → out_proj.
///
/// Matches Tri Dao's `Mamba` class in `mamba_simple.py`.
/// No RMSNorm, no residual connection — those belong to the Block wrapper.
///
/// # Arguments
/// - `input`: `[d_model]` — pre-normed input (NOT raw hidden state)
/// - `output`: `[d_model]` — layer output (before residual add)
/// - `lw`: layer weights
/// - `state`: persistent conv + SSM state (mutated)
/// - `scratch`: pre-allocated scratch buffers
/// - `cfg`: Mamba config
pub fn mamba_layer_step(
    input: &[f32],
    output: &mut [f32],
    lw: &MambaLayerWeights,
    state: &mut MambaLayerState,
    scratch: &mut MambaLayerScratch,
    cfg: &MambaConfig,
) {
    let d_model = cfg.d_model;
    let d_inner = cfg.d_inner();
    let d_state = cfg.d_state;
    let d_conv = cfg.d_conv;
    let dt_rank = cfg.dt_rank();
    let xdbl_dim = cfg.xdbl_dim();

    // in_proj: [d_model] -> [2*d_inner] (no bias)
    matvec_no_bias(
        &mut scratch.proj[..2 * d_inner],
        &input[..d_model],
        &lw.in_proj_w,
        d_model,
        2 * d_inner,
    );

    // Split x + gate SiLU
    scratch.x[..d_inner].copy_from_slice(&scratch.proj[..d_inner]);
    for d in 0..d_inner {
        let v = scratch.proj[d_inner + d];
        scratch.gate_silu[d] = v * sigmoid(v);
    }

    // Conv1d shift register (dot BEFORE shift)
    let hist = d_conv - 1;
    for d in 0..d_inner {
        let state_off = d * hist;
        let mut val = 0.0_f32;
        for k in 0..hist {
            val += state.conv_state[state_off + k] * lw.conv1d_weight[d * d_conv + k];
        }
        val += scratch.x[d] * lw.conv1d_weight[d * d_conv + hist];
        val += lw.conv1d_bias[d];

        // Shift state left + insert new
        for k in 0..hist.saturating_sub(1) {
            state.conv_state[state_off + k] = state.conv_state[state_off + k + 1];
        }
        if hist > 0 {
            state.conv_state[state_off + hist - 1] = scratch.x[d];
        }

        scratch.x[d] = val * sigmoid(val); // fused conv1d + SiLU
    }

    // x_proj: u [d_inner] -> xdbl [dt_rank + 2*d_state]
    matvec_no_bias(
        &mut scratch.xdbl[..xdbl_dim],
        &scratch.x[..d_inner],
        &lw.x_proj_w,
        d_inner,
        xdbl_dim,
    );

    // dt_proj: delta_raw [dt_rank] -> delta [d_inner] + softplus
    matvec_with_bias(
        &mut scratch.delta[..d_inner],
        &scratch.xdbl[..dt_rank],
        &lw.dt_proj_w,
        Some(&lw.dt_proj_b),
        dt_rank,
        d_inner,
    );
    for d in 0..d_inner {
        let raw = scratch.delta[d];
        scratch.delta[d] = if raw > 20.0 {
            raw
        } else {
            (1.0 + fast_exp_scalar(raw)).ln()
        };
    }

    // SSM recurrence (SIMD-friendly: batch exp per channel via da_buf)
    let b_offset = dt_rank;
    let c_offset = dt_rank + d_state;
    for d in 0..d_inner {
        let delta_d = scratch.delta[d];
        let u_d = scratch.x[d];
        let delta_u_d = delta_d * u_d;

        // Fill da_buf with delta * a_neg for SIMD batch exp
        let a_base = d * d_state;
        for n in 0..d_state {
            scratch.da_buf[n] = delta_d * lw.a_neg[a_base + n];
        }
        fast_exp_inplace(&mut scratch.da_buf[..d_state]);

        // Apply SSM update with precomputed da values
        let mut y_d = 0.0_f32;
        for n in 0..d_state {
            let b_n = scratch.xdbl[b_offset + n];
            let c_n = scratch.xdbl[c_offset + n];
            let h_idx = a_base + n;
            state.ssm_state[h_idx] = scratch.da_buf[n] * state.ssm_state[h_idx] + delta_u_d * b_n;
            y_d += state.ssm_state[h_idx] * c_n;
        }

        y_d += lw.d_param[d] * u_d;
        scratch.y[d] = y_d;
    }

    // Gating: y * SiLU(gate)
    for d in 0..d_inner {
        scratch.y[d] *= scratch.gate_silu[d];
    }

    // out_proj: gated [d_inner] -> output [d_model]
    matvec_no_bias(
        &mut output[..d_model],
        &scratch.y[..d_inner],
        &lw.out_proj_w,
        d_inner,
        d_model,
    );
}

// ---------------------------------------------------------------------------
// Level 2: Mamba block (pre-norm + layer + residual)
// ---------------------------------------------------------------------------

/// Mamba block step (T=1): RMSNorm → MambaLayer → residual add.
///
/// Matches Tri Dao's `Block` class in `block.py`.
/// `hidden` is modified in-place: normed → passed to layer → output + residual.
///
/// # Arguments
/// - `hidden`: `[d_model]` — hidden state (modified in-place)
/// - `lw`: layer weights (includes `norm_weight`)
/// - `state`: persistent conv + SSM state
/// - `scratch`: pre-allocated scratch (includes layer scratch + norm_buf + residual)
/// - `cfg`: Mamba config
pub fn mamba_block_step(
    hidden: &mut [f32],
    lw: &MambaLayerWeights,
    state: &mut MambaLayerState,
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    let d_model = cfg.d_model;

    // Save residual
    scratch.residual[..d_model].copy_from_slice(&hidden[..d_model]);

    // RMSNorm
    let mean_sq: f32 = hidden[..d_model].iter().map(|v| v * v).sum::<f32>() / d_model as f32;
    let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
    for (nb, (&h, &nw)) in scratch.norm_buf[..d_model]
        .iter_mut()
        .zip(hidden[..d_model].iter().zip(lw.norm_weight.iter()))
    {
        *nb = h * inv_rms * nw;
    }

    // Pure mixer
    mamba_layer_step(
        &scratch.norm_buf[..d_model],
        &mut hidden[..d_model],
        lw,
        state,
        &mut scratch.layer,
        cfg,
    );

    // Residual connection
    for (h, &r) in hidden[..d_model]
        .iter_mut()
        .zip(scratch.residual[..d_model].iter())
    {
        *h += r;
    }
}

// ---------------------------------------------------------------------------
// Level 3: Full backbone (input_proj + N blocks + norm_f)
// ---------------------------------------------------------------------------

/// Full Mamba backbone step (T=1): input_proj → N blocks → norm_f.
///
/// # Arguments
/// - `input`: `[input_dim]`
/// - `output`: `[d_model]`
/// - `weights`: all backbone weights
/// - `state`: per-layer persistent state
/// - `scratch`: pre-allocated scratch
/// - `cfg`: Mamba config
/// - `input_dim`: external input dimension
pub fn mamba_step(
    input: &[f32],
    output: &mut [f32],
    weights: &MambaWeights,
    state: &mut [MambaLayerState],
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
    input_dim: usize,
) {
    let d_model = cfg.d_model;

    // Input projection
    matvec_with_bias(
        &mut output[..d_model],
        input,
        &weights.input_proj_w,
        Some(&weights.input_proj_b),
        input_dim,
        d_model,
    );

    // N blocks
    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        mamba_block_step(output, lw, &mut state[layer_idx], scratch, cfg);
    }

    // Final RmsNorm (norm_f)
    let mean_sq: f32 = output[..d_model].iter().map(|v| v * v).sum::<f32>() / d_model as f32;
    let inv_rms = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
    for (o, &nfw) in output[..d_model]
        .iter_mut()
        .zip(weights.norm_f_weight[..d_model].iter())
    {
        *o *= inv_rms * nfw;
    }
}

/// Batched T=1 inference: processes B independent samples through the same weights.
pub fn mamba_step_batch(
    inputs: &[f32],
    outputs: &mut [f32],
    weights: &MambaWeights,
    states: &mut [crate::state::MambaState],
    scratches: &mut [MambaStepScratch],
    cfg: &MambaConfig,
    input_dim: usize,
) {
    let batch = states.len();
    let d_model = cfg.d_model;
    assert_eq!(
        inputs.len(),
        batch * input_dim,
        "inputs size mismatch: expected {}, got {}",
        batch * input_dim,
        inputs.len()
    );
    assert_eq!(
        outputs.len(),
        batch * d_model,
        "outputs size mismatch: expected {}, got {}",
        batch * d_model,
        outputs.len()
    );
    assert_eq!(
        scratches.len(),
        batch,
        "scratches count mismatch: expected {batch}, got {}",
        scratches.len()
    );

    if batch >= rayon::current_num_threads().max(2) {
        // Parallel: each sample is independent at T=1 (no cross-sample dependencies)
        use rayon::prelude::*;

        // Split output into per-sample chunks for safe parallel write
        let out_chunks: Vec<&mut [f32]> = outputs.chunks_mut(d_model).collect();
        out_chunks
            .into_par_iter()
            .zip(states.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .enumerate()
            .for_each(|(b, ((out, state), scratch))| {
                let inp = &inputs[b * input_dim..(b + 1) * input_dim];
                mamba_step(
                    inp,
                    out,
                    weights,
                    &mut state.layers,
                    scratch,
                    cfg,
                    input_dim,
                );
            });
    } else {
        // Sequential: rayon overhead > compute for small batches
        for b in 0..batch {
            let inp = &inputs[b * input_dim..(b + 1) * input_dim];
            let out = &mut outputs[b * d_model..(b + 1) * d_model];
            mamba_step(
                inp,
                out,
                weights,
                &mut states[b].layers,
                &mut scratches[b],
                cfg,
                input_dim,
            );
        }
    }
}

// --- BLAS dispatch (delegates to ops/blas.rs for platform-specific SGEMM) ---

use crate::ops::blas::matvec_forward;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp_scalar(-x))
}

#[inline]
fn matvec_with_bias(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n_in: usize,
    n_out: usize,
) {
    matvec_forward(out, x, w, bias, n_in, n_out);
}

#[inline]
fn matvec_no_bias(out: &mut [f32], x: &[f32], w: &[f32], n_in: usize, n_out: usize) {
    matvec_forward(out, x, w, None, n_in, n_out);
}
