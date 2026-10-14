//! Mamba-2 decode: one token through the backbone, mutating the state.
//!
//! ```text
//! temporal -> RMSNorm -> in_proj -> split [z0 | x0 | z | xBC | dt]
//!   -> conv over the last d_conv inputs + SiLU -> x | B | C
//!   -> per head: dt, dA, scan along the head's channels, + D x
//!   -> [silu(z0) x0 | gated RMSNorm(y, z)] -> out_proj + residual
//!   [-> gated MLP] -> ... -> final RMSNorm
//! ```

use pulp::{Arch, Simd, WithSimd};

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::{Mamba2LayerWeights, Mamba2Weights};
use super::core::{block_norm_row, conv_hist_push, conv_silu_token, gate_token, ssm_token};
use crate::mlp::cpu::{GatedMlpPacked, GatedMlpStepScratch, gated_mlp_step, gated_mlp_step_packed};
use crate::ops::lane::LaneGemm;
use crate::ops::packed::PackedLinear;
use crate::ops::rounding::store;
use crate::precision::WeightDtype;

/// One layer's projections packed for decode.
#[derive(Clone, Debug)]
pub struct Mamba2LayerPacked {
    pub in_proj: PackedLinear,
    pub out_proj: PackedLinear,
}

/// Every projection of a Mamba-2 backbone packed for decode: a copy of the
/// weights' projections, so it must be packed again after they change.
#[derive(Clone, Debug)]
pub struct Mamba2PackedLinears {
    pub layers: Vec<Mamba2LayerPacked>,
    /// One per layer with an MLP.
    pub mlp: Vec<GatedMlpPacked>,
    /// The input projection, absent for a backbone without one.
    pub input_proj: Option<PackedLinear>,
}

impl Mamba2PackedLinears {
    /// Pack every projection of `weights`.
    pub fn pack(weights: &Mamba2Weights, cfg: &Mamba2Config, input_dim: usize) -> Self {
        let (dm, di) = (cfg.d_model, cfg.d_inner());
        Self {
            layers: weights
                .layers
                .iter()
                .map(|lw| Mamba2LayerPacked {
                    in_proj: PackedLinear::pack(&lw.in_proj_w, dm, cfg.in_proj_out_dim()),
                    out_proj: PackedLinear::pack(&lw.out_proj_w, di, dm),
                })
                .collect(),
            mlp: weights.mlp.iter().map(GatedMlpPacked::pack).collect(),
            input_proj: (!weights.input_proj_w.is_empty())
                .then(|| PackedLinear::pack(&weights.input_proj_w, input_dim, dm)),
        }
    }
}

/// Reusable buffers for [`mamba2_step`].
pub struct Mamba2StepScratch {
    residual: Vec<f32>,
    norm_buf: Vec<f32>,
    proj: Vec<f32>,
    xbc: Vec<f32>,
    y: Vec<f32>,
    gated: Vec<f32>,
    mlp: GatedMlpStepScratch,
    /// The precision of the step (see `CpuOptions`).
    dtype: WeightDtype,
    gemm: LaneGemm,
}

impl Mamba2StepScratch {
    pub fn new(cfg: &Mamba2Config) -> Self {
        let dm = cfg.d_model;
        Self {
            residual: vec![0.0; dm],
            norm_buf: vec![0.0; dm],
            proj: vec![0.0; cfg.in_proj_out_dim()],
            xbc: vec![0.0; cfg.conv_dim()],
            y: vec![0.0; cfg.ssm_dim()],
            gated: vec![0.0; cfg.d_inner()],
            mlp: GatedMlpStepScratch::new(dm, cfg.mlp_hidden()),
            dtype: WeightDtype::F32,
            gemm: LaneGemm::default(),
        }
    }

    /// Run the steps in `dtype`: the weights must hold the bulk matrices
    /// rounded as that precision keeps them.
    pub(crate) fn set_dtype(&mut self, dtype: WeightDtype) {
        self.dtype = dtype;
        self.mlp.dtype = dtype;
    }
}

fn bias(b: &[f32]) -> Option<&[f32]> {
    if b.is_empty() { None } else { Some(b) }
}

/// The input projection's precision: f32 in the half lanes, as the GPU
/// engine keeps it, TF32 operands in the TF32 lane.
fn input_proj_dtype(dtype: WeightDtype) -> WeightDtype {
    if dtype == WeightDtype::Tf32 {
        dtype
    } else {
        WeightDtype::F32
    }
}

/// One token through one Mamba-2 mixer; `temporal` (`[d_model]`) is both
/// the input and the output, residual included.
pub fn mamba2_layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    lw: &Mamba2LayerWeights,
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    layer_step(temporal, scratch, (lw, None), state, cfg);
}

/// [`mamba2_layer_step`] with the projections packed when `lp` holds them.
/// The step runs under pulp's dispatch, so its scalar fused multiply-adds
/// (norm, conv, dt, gate, all inlined) compile to the FMA instruction on an
/// x86 built without it instead of a call to the library's fmaf; a fused
/// multiply-add rounds once either way, so the bits are the same.
fn layer_step(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    (lw, lp): (&Mamba2LayerWeights, Option<&Mamba2LayerPacked>),
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    struct Step<'a> {
        temporal: &'a mut [f32],
        scratch: &'a mut Mamba2StepScratch,
        lw: &'a Mamba2LayerWeights,
        lp: Option<&'a Mamba2LayerPacked>,
        state: &'a mut Mamba2LayerState,
        cfg: &'a Mamba2Config,
    }
    impl WithSimd for Step<'_> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) {
            layer_step_body(
                self.temporal,
                self.scratch,
                (self.lw, self.lp),
                self.state,
                self.cfg,
            );
        }
    }
    Arch::new().dispatch(Step {
        temporal,
        scratch,
        lw,
        lp,
        state,
        cfg,
    });
}

#[inline(always)]
fn layer_step_body(
    temporal: &mut [f32],
    scratch: &mut Mamba2StepScratch,
    (lw, lp): (&Mamba2LayerWeights, Option<&Mamba2LayerPacked>),
    state: &mut Mamba2LayerState,
    cfg: &Mamba2Config,
) {
    let dm = cfg.d_model;
    let (d_mlp, d_ssm, cd) = (cfg.d_mlp(), cfg.ssm_dim(), cfg.conv_dim());
    let dtype = scratch.dtype;
    scratch.residual.copy_from_slice(&temporal[..dm]);
    block_norm_row(
        &mut scratch.norm_buf,
        &temporal[..dm],
        (&lw.norm_weight, &lw.norm_bias),
        cfg.rms_norm_eps,
    );
    store(dtype, &mut scratch.norm_buf);
    scratch.gemm.row(
        dtype,
        (&mut scratch.proj, &scratch.norm_buf),
        (&lw.in_proj_w, lp.map(|p| &p.in_proj)),
        bias(&lw.in_proj_b),
    );
    let xbc_raw = &scratch.proj[2 * d_mlp + d_ssm..2 * d_mlp + d_ssm + cd];
    let dt_raw = &scratch.proj[2 * d_mlp + d_ssm + cd..];
    conv_silu_token(&mut scratch.xbc, xbc_raw, &state.conv_hist, lw);
    conv_hist_push(&mut state.conv_hist, xbc_raw);
    store(dtype, &mut scratch.xbc);
    ssm_token(
        (&mut state.ssm, &mut scratch.y),
        &scratch.xbc,
        dt_raw,
        lw,
        cfg,
    );
    gate_token(&mut scratch.gated, &scratch.y, &scratch.proj, lw, cfg);
    store(dtype, &mut scratch.gated);
    scratch.gemm.row(
        dtype,
        (&mut temporal[..dm], &scratch.gated),
        (&lw.out_proj_w, lp.map(|p| &p.out_proj)),
        bias(&lw.out_proj_b),
    );
    for (t, &r) in temporal[..dm].iter_mut().zip(&scratch.residual) {
        *t += r;
    }
}

/// One token through the whole Mamba-2 backbone: input projection, every
/// layer (mixer, then the MLP when the model has one), final norm.
/// `input` is `[input_dim]`; `temporal` (`[d_model]`) receives the output.
///
/// # Panics
///
/// For a bidirectional model, which reads whole sequences and has no
/// decode step.
pub fn mamba2_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    weights: &Mamba2Weights,
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    backbone_step(temporal, input, scratch, (weights, None), states, cfg);
}

/// [`mamba2_step`] with every projection taken from `packed` (packed from
/// these `weights` by [`Mamba2PackedLinears::pack`]).
///
/// # Panics
///
/// When `packed` holds another number of layers or MLPs than `weights`,
/// and for a bidirectional model.
pub fn mamba2_step_packed(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    (weights, packed): (&Mamba2Weights, &Mamba2PackedLinears),
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    assert!(
        packed.layers.len() == weights.layers.len() && packed.mlp.len() == weights.mlp.len(),
        "packed projections of another backbone"
    );
    backbone_step(
        temporal,
        input,
        scratch,
        (weights, Some(packed)),
        states,
        cfg,
    );
}

fn backbone_step(
    temporal: &mut [f32],
    input: &[f32],
    scratch: &mut Mamba2StepScratch,
    (weights, packed): (&Mamba2Weights, Option<&Mamba2PackedLinears>),
    states: &mut [Mamba2LayerState],
    cfg: &Mamba2Config,
) {
    assert!(
        !cfg.bidirectional,
        "a bidirectional model reads whole sequences and has no decode step"
    );
    let dm = cfg.d_model;
    if weights.input_proj_w.is_empty() {
        temporal[..dm].copy_from_slice(&input[..dm]);
    } else {
        scratch.gemm.row(
            input_proj_dtype(scratch.dtype),
            (&mut temporal[..dm], input),
            (
                &weights.input_proj_w,
                packed.and_then(|p| p.input_proj.as_ref()),
            ),
            bias(&weights.input_proj_b),
        );
    }
    for (l, (lw, st)) in weights.layers.iter().zip(states.iter_mut()).enumerate() {
        layer_step(
            temporal,
            scratch,
            (lw, packed.map(|p| &p.layers[l])),
            st,
            cfg,
        );
        if let Some(mlp) = weights.mlp.get(l) {
            let t = &mut temporal[..dm];
            match packed {
                Some(p) => {
                    gated_mlp_step_packed(t, (mlp, &p.mlp[l]), &mut scratch.mlp, cfg.rms_norm_eps)
                }
                None => gated_mlp_step(t, mlp, &mut scratch.mlp, cfg.rms_norm_eps),
            }
        }
    }
    // A segment without the final norm hands out the residual stream.
    if weights.norm_f_weight.is_empty() {
        return;
    }
    scratch.norm_buf.copy_from_slice(&temporal[..dm]);
    block_norm_row(
        &mut temporal[..dm],
        &scratch.norm_buf,
        (&weights.norm_f_weight, &weights.norm_f_bias),
        cfg.rms_norm_eps,
    );
}
