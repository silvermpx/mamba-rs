//! Single-step recurrent Mamba inference (T=1).
//!
//! Three levels of API:
//! - [`mamba_layer_step`]: Pure mixer (no norm, no residual). Matches Tri Dao's `Mamba` class.
//! - [`mamba_block_step`]: Pre-norm + mixer + residual, then the gated MLP
//!   branch for a model with one. Matches Tri Dao's `Block` class.
//! - [`mamba_step`]: Full backbone: input_proj + N blocks + norm_f.
//!
//! [`mamba_step_packed`] runs the same step with the projections packed in
//! panels ([`MambaPackedLinears`]), the same bits as the row-major product
//! without a BLAS, faster on the weights a decode streams from memory.

use super::bias;
use crate::mamba1::config::MambaConfig;
use crate::mamba1::state::MambaLayerState;
use crate::mamba1::weights::{MambaLayerWeights, MambaWeights};
use crate::mlp::GatedMlpLayerWeights;
use crate::mlp::cpu::{GatedMlpPacked, GatedMlpStepScratch, gated_mlp_step, gated_mlp_step_packed};
use crate::ops::fast_math::{fast_exp_inplace, fast_exp_scalar};
use crate::ops::lane::{LaneGemm, input_rows};
use crate::ops::norms::bcdt_rms_norm;
use crate::ops::packed::{PackedLinear, linear_row as linear};
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

// ---------------------------------------------------------------------------
// Packed projections
// ---------------------------------------------------------------------------

/// One layer's projections packed for decode.
#[derive(Clone, Debug)]
pub struct MambaLayerPacked {
    pub in_proj: PackedLinear,
    pub x_proj: PackedLinear,
    pub dt_proj: PackedLinear,
    pub out_proj: PackedLinear,
}

/// Every projection of a backbone packed for decode: a copy of the weights'
/// projections, so it must be packed again after the weights change.
#[derive(Clone, Debug)]
pub struct MambaPackedLinears {
    pub layers: Vec<MambaLayerPacked>,
    /// The input projection, absent for a backbone without one.
    pub input_proj: Option<PackedLinear>,
    /// One per layer with a gated MLP.
    pub mlp: Vec<GatedMlpPacked>,
}

impl MambaPackedLinears {
    /// Pack every projection of `weights`.
    pub fn pack(weights: &MambaWeights, cfg: &MambaConfig, input_dim: usize) -> Self {
        let (dm, di) = (cfg.d_model, cfg.d_inner());
        let layers = weights
            .layers
            .iter()
            .map(|lw| MambaLayerPacked {
                in_proj: PackedLinear::pack(&lw.in_proj_w, dm, 2 * di),
                x_proj: PackedLinear::pack(&lw.x_proj_w, di, cfg.xdbl_dim()),
                dt_proj: PackedLinear::pack(&lw.dt_proj_w, cfg.dt_rank(), di),
                out_proj: PackedLinear::pack(&lw.out_proj_w, di, dm),
            })
            .collect();
        let input_proj = (!weights.input_proj_w.is_empty())
            .then(|| PackedLinear::pack(&weights.input_proj_w, input_dim, dm));
        Self {
            layers,
            input_proj,
            mlp: weights.mlp.iter().map(GatedMlpPacked::pack).collect(),
        }
    }
}

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
    /// The precision of the step (see `CpuOptions`).
    pub(crate) dtype: WeightDtype,
    gemm: LaneGemm,
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
            dtype: WeightDtype::F32,
            gemm: LaneGemm::default(),
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
    /// The input row as the input projection multiplies it in a reduced
    /// precision.
    input: Vec<f32>,
    /// The gated MLP's buffers (zero-sized when the model has none).
    pub mlp: GatedMlpStepScratch,
}

impl MambaStepScratch {
    /// Allocate scratch buffers sized for the given Mamba configuration.
    pub fn new(cfg: &MambaConfig) -> Self {
        Self {
            layer: MambaLayerScratch::new(cfg),
            norm_buf: vec![0.0; cfg.d_model],
            residual: vec![0.0; cfg.d_model],
            input: Vec::new(),
            mlp: GatedMlpStepScratch::new(cfg.d_model, cfg.mlp_hidden()),
        }
    }

    /// Run the steps in `dtype`: the weights must hold the bulk matrices
    /// rounded as that precision keeps them.
    pub(crate) fn set_dtype(&mut self, dtype: WeightDtype) {
        self.layer.dtype = dtype;
        self.mlp.dtype = dtype;
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
    layer_step(input, output, (lw, None), state, scratch, cfg);
}

/// [`mamba_layer_step`] with the projections packed when `lp` holds them.
fn layer_step(
    input: &[f32],
    output: &mut [f32],
    (lw, lp): (&MambaLayerWeights, Option<&MambaLayerPacked>),
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
    let dtype = scratch.dtype;

    // in_proj: [d_model] -> [2*d_inner]
    scratch.gemm.row(
        dtype,
        (&mut scratch.proj[..2 * d_inner], &input[..d_model]),
        (&lw.in_proj_w, lp.map(|p| &p.in_proj)),
        bias(&lw.in_proj_b),
    );

    // Split x + gate SiLU
    scratch.x[..d_inner].copy_from_slice(&scratch.proj[..d_inner]);
    for d in 0..d_inner {
        let v = scratch.proj[d_inner + d];
        scratch.gate_silu[d] = v * sigmoid(v);
    }
    store(dtype, &mut scratch.gate_silu[..d_inner]);

    // Conv1d shift register (dot BEFORE shift)
    let hist = d_conv - 1;
    for d in 0..d_inner {
        let state_off = d * hist;
        let mut val = 0.0_f32;
        for k in 0..hist {
            val += state.conv_state[state_off + k] * lw.conv1d_weight[d * d_conv + k];
        }
        val += scratch.x[d] * lw.conv1d_weight[d * d_conv + hist];
        val += lw.conv1d_bias.get(d).copied().unwrap_or(0.0);

        // Shift state left + insert new
        for k in 0..hist.saturating_sub(1) {
            state.conv_state[state_off + k] = state.conv_state[state_off + k + 1];
        }
        if hist > 0 {
            state.conv_state[state_off + hist - 1] = scratch.x[d];
        }

        scratch.x[d] = val * sigmoid(val); // fused conv1d + SiLU
    }
    store(dtype, &mut scratch.x[..d_inner]);

    // x_proj: u [d_inner] -> xdbl [dt_rank + 2*d_state]
    scratch.gemm.row(
        dtype,
        (&mut scratch.xdbl[..xdbl_dim], &scratch.x[..d_inner]),
        (&lw.x_proj_w, lp.map(|p| &p.x_proj)),
        None,
    );
    // FalconMamba normalizes the row's dt, B and C parts; the GPU stores the
    // normalized row again.
    if let Some(eps) = cfg.bcdt_rms_eps {
        bcdt_rms_norm(&mut scratch.xdbl[..xdbl_dim], (dt_rank, d_state), eps);
        store(dtype, &mut scratch.xdbl[..xdbl_dim]);
    }

    // dt_proj: delta_raw [dt_rank] -> delta [d_inner] + softplus
    scratch.gemm.row(
        dtype,
        (&mut scratch.delta[..d_inner], &scratch.xdbl[..dt_rank]),
        (&lw.dt_proj_w, lp.map(|p| &p.dt_proj)),
        Some(&lw.dt_proj_b),
    );
    for d in 0..d_inner {
        let raw = scratch.delta[d];
        scratch.delta[d] = if raw > 20.0 {
            raw
        } else {
            fast_exp_scalar(raw).ln_1p()
        };
    }
    store(dtype, &mut scratch.delta[..d_inner]);

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
    store(dtype, &mut scratch.y[..d_inner]);

    // out_proj: gated [d_inner] -> output [d_model]
    scratch.gemm.row(
        dtype,
        (&mut output[..d_model], &scratch.y[..d_inner]),
        (&lw.out_proj_w, lp.map(|p| &p.out_proj)),
        bias(&lw.out_proj_b),
    );
}

// ---------------------------------------------------------------------------
// Level 2: Mamba block (pre-norm + layer + residual)
// ---------------------------------------------------------------------------

/// Mamba block step (T=1): RMSNorm → MambaLayer → residual add, then for a
/// model with a gated MLP its branch (norm2 → MLP → residual add).
///
/// Matches Tri Dao's `Block` class in `block.py`.
/// `hidden` is modified in-place: normed → passed to layer → output + residual.
///
/// # Arguments
/// - `hidden`: `[d_model]` — hidden state (modified in-place)
/// - `(lw, mlp)`: layer weights (includes `norm_weight`) and the layer's
///   gated MLP, `None` for a model without one
/// - `state`: persistent conv + SSM state
/// - `scratch`: pre-allocated scratch (includes layer scratch + norm_buf + residual)
/// - `cfg`: Mamba config
///
/// # Panics
///
/// When `mlp` is given for a config without a gated MLP, or missing for
/// one with it.
pub fn mamba_block_step(
    hidden: &mut [f32],
    (lw, mlp): (&MambaLayerWeights, Option<&GatedMlpLayerWeights>),
    state: &mut MambaLayerState,
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    assert_eq!(
        mlp.is_some(),
        cfg.mlp_hidden() > 0,
        "mamba_block_step takes the gated MLP exactly when the config has one"
    );
    block_step(hidden, (lw, None), state, scratch, cfg);
    if let Some(m) = mlp {
        gated_mlp_step(hidden, m, &mut scratch.mlp, cfg.rms_norm_eps);
    }
}

/// [`mamba_block_step`] with the projections packed when `lp` holds them.
fn block_step(
    hidden: &mut [f32],
    (lw, lp): (&MambaLayerWeights, Option<&MambaLayerPacked>),
    state: &mut MambaLayerState,
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    let d_model = cfg.d_model;

    // Save residual
    scratch.residual[..d_model].copy_from_slice(&hidden[..d_model]);

    // RMSNorm
    let mean_sq: f32 = hidden[..d_model].iter().map(|v| v * v).sum::<f32>() / d_model as f32;
    let inv_rms = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
    for (nb, (&h, &nw)) in scratch.norm_buf[..d_model]
        .iter_mut()
        .zip(hidden[..d_model].iter().zip(lw.norm_weight.iter()))
    {
        *nb = h * inv_rms * nw;
    }
    store(scratch.layer.dtype, &mut scratch.norm_buf[..d_model]);

    // Pure mixer
    layer_step(
        &scratch.norm_buf[..d_model],
        &mut hidden[..d_model],
        (lw, lp),
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
    linear(
        &mut output[..d_model],
        input_rows(scratch.layer.dtype, &mut scratch.input, input),
        (&weights.input_proj_w, None),
        Some(&weights.input_proj_b),
        (input_dim, d_model),
    );
    mamba_blocks_and_norm(output, (weights, None), state, scratch, cfg);
}

/// [`mamba_step`] with every projection taken from `packed` (packed from
/// these `weights` by [`MambaPackedLinears::pack`]).
///
/// # Panics
///
/// When `packed` holds another number of layers than `weights`.
pub fn mamba_step_packed(
    input: &[f32],
    output: &mut [f32],
    (weights, packed): (&MambaWeights, &MambaPackedLinears),
    state: &mut [MambaLayerState],
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
    input_dim: usize,
) {
    let d_model = cfg.d_model;
    linear(
        &mut output[..d_model],
        input_rows(scratch.layer.dtype, &mut scratch.input, input),
        (&weights.input_proj_w, packed.input_proj.as_ref()),
        Some(&weights.input_proj_b),
        (input_dim, d_model),
    );
    mamba_blocks_and_norm(output, (weights, Some(packed)), state, scratch, cfg);
}

/// [`mamba_step_no_proj`] with every projection taken from `packed`.
///
/// # Panics
///
/// When `packed` holds another number of layers than `weights`.
pub fn mamba_step_no_proj_packed(
    input: &[f32],
    output: &mut [f32],
    (weights, packed): (&MambaWeights, &MambaPackedLinears),
    state: &mut [MambaLayerState],
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    output[..cfg.d_model].copy_from_slice(&input[..cfg.d_model]);
    mamba_blocks_and_norm(output, (weights, Some(packed)), state, scratch, cfg);
}

/// Like [`mamba_step`] but skips the input projection (identity: input copied to output).
/// Used for HF LM models where the embedding output is already `[d_model]`.
pub fn mamba_step_no_proj(
    input: &[f32],
    output: &mut [f32],
    weights: &MambaWeights,
    state: &mut [MambaLayerState],
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    output[..cfg.d_model].copy_from_slice(&input[..cfg.d_model]);
    mamba_blocks_and_norm(output, (weights, None), state, scratch, cfg);
}

#[inline]
fn mamba_blocks_and_norm(
    output: &mut [f32],
    (weights, packed): (&MambaWeights, Option<&MambaPackedLinears>),
    state: &mut [MambaLayerState],
    scratch: &mut MambaStepScratch,
    cfg: &MambaConfig,
) {
    let d_model = cfg.d_model;
    if let Some(p) = packed {
        assert!(
            p.layers.len() == weights.layers.len() && p.mlp.len() == weights.mlp.len(),
            "packed projections of another backbone"
        );
    }
    // Each layer: the mixer branch, then the MLP branch when the model has
    // one.
    for (layer_idx, lw) in weights.layers.iter().enumerate() {
        let lp = packed.map(|p| &p.layers[layer_idx]);
        block_step(output, (lw, lp), &mut state[layer_idx], scratch, cfg);
        if let Some(m) = weights.mlp.get(layer_idx) {
            let eps = cfg.rms_norm_eps;
            match packed {
                Some(p) => {
                    gated_mlp_step_packed(output, (m, &p.mlp[layer_idx]), &mut scratch.mlp, eps)
                }
                None => gated_mlp_step(output, m, &mut scratch.mlp, eps),
            }
        }
    }
    let mean_sq: f32 = output[..d_model].iter().map(|v| v * v).sum::<f32>() / d_model as f32;
    let inv_rms = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
    for (o, &nfw) in output[..d_model]
        .iter_mut()
        .zip(weights.norm_f_weight[..d_model].iter())
    {
        *o *= inv_rms * nfw;
    }
    store(scratch.layer.dtype, &mut output[..d_model]);
}

/// Batched T=1 inference: processes B independent samples through the same
/// weights. A model without an input projection takes each sample's input
/// as its rows.
pub fn mamba_step_batch(
    inputs: &[f32],
    outputs: &mut [f32],
    weights: &MambaWeights,
    states: &mut [crate::mamba1::state::MambaState],
    scratches: &mut [MambaStepScratch],
    cfg: &MambaConfig,
    input_dim: usize,
) {
    let identity = weights.input_proj_w.is_empty();
    step_batch_by(
        (inputs, outputs),
        (states, scratches),
        (cfg.d_model, input_dim),
        |inp, out, state, scratch| {
            if identity {
                mamba_step_no_proj(inp, out, weights, &mut state.layers, scratch, cfg);
            } else {
                mamba_step(
                    inp,
                    out,
                    weights,
                    &mut state.layers,
                    scratch,
                    cfg,
                    input_dim,
                );
            }
        },
    );
}

/// B independent samples through `step`, one T=1 step each: in parallel
/// when the batch fills the pool, one after another otherwise.
pub(crate) fn step_batch_by(
    (inputs, outputs): (&[f32], &mut [f32]),
    (states, scratches): (
        &mut [crate::mamba1::state::MambaState],
        &mut [MambaStepScratch],
    ),
    (d_model, input_dim): (usize, usize),
    step: impl Fn(&[f32], &mut [f32], &mut crate::mamba1::state::MambaState, &mut MambaStepScratch)
    + Sync,
) {
    let batch = states.len();
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

        // par_chunks_mut iterates per-sample output slices directly —
        // no per-step Vec of pointers (zero-allocation hot path).
        outputs
            .par_chunks_mut(d_model)
            .zip(states.par_iter_mut())
            .zip(scratches.par_iter_mut())
            .enumerate()
            .for_each(|(b, ((out, state), scratch))| {
                step(
                    &inputs[b * input_dim..(b + 1) * input_dim],
                    out,
                    state,
                    scratch,
                );
            });
    } else {
        // Sequential: rayon overhead > compute for small batches
        for (b, ((out, state), scratch)) in outputs
            .chunks_mut(d_model)
            .zip(states.iter_mut())
            .zip(scratches.iter_mut())
            .enumerate()
        {
            step(
                &inputs[b * input_dim..(b + 1) * input_dim],
                out,
                state,
                scratch,
            );
        }
    }
}

// --- BLAS dispatch (delegates to ops/blas.rs for platform-specific SGEMM) ---

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + fast_exp_scalar(-x))
}
