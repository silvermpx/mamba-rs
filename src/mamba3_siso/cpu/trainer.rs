//! The Mamba-3 SISO host trainer: the shared CPU trainer over this
//! family's layer forward and backward ([`forward_mamba3_layer_batched`],
//! [`backward_mamba3_layer_batched`]), composed as the GPU backbone
//! composes them: the input projection, every layer followed by its gated
//! MLP when the model has one, the final RMSNorm, and the mirrored backward.

use super::backward::{backward_mamba3_layer_batched, backward_mamba3_layer_in};
use super::dims::Mamba3Dims;
use super::flat::Mamba3LayerFlat;
use super::forward::{Mamba3LayerStateMut, forward_mamba3_layer_in};
use super::scratch::Mamba3Scratch;
use super::weights::TrainMamba3LayerWeights;
use crate::bidi::{PairGrad, mlps_after, opens_pair, reads_reversed, reverse_rows};
use crate::config::ModelFamily;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::weights::{Mamba3LayerWeights, Mamba3Weights};
use crate::mlp::GatedMlpLayerWeights;
use crate::mlp::cpu::RowsGemm;
use crate::mlp::cpu::{
    GatedMlpBackwardScratch, GatedMlpRowsScratch, gated_mlp_rows, gated_mlp_rows_backward,
    mlp_norm_row,
};
use crate::module::cpu_trainer::{Control, CpuCore, CpuFamily, TensorRole, cpu_trainer_type};
use crate::ops::blas::sgemm_forward;
use crate::ops::lane::round_matrix;
use crate::ops::norms::{layer_norm_inplace, layer_norm_row_backward_in_place, rms_norm_inplace};
use crate::ops::rounding::store;
use crate::optim::session::TrainSessionCfg;
use crate::optim::tensors::tensor_list;
use crate::precision::{CpuOptions, WeightDtype};

/// The trainable weights of a SISO model: the layers in the training
/// layout, the gated MLPs of a model that has them.
#[derive(Clone)]
pub(crate) struct Mamba3CpuWeights {
    input_proj_w: Vec<f32>,
    input_proj_b: Vec<f32>,
    layers: Vec<TrainMamba3LayerWeights>,
    mlp: Vec<GatedMlpLayerWeights>,
    norm_f_weight: Vec<f32>,
    norm_f_bias: Vec<f32>,
}

tensor_list!(
    Mamba3CpuWeights,
    pre: [input_proj_w, input_proj_b],
    layers: layers [
        norm_weight,
        norm_bias,
        in_proj_w,
        dt_bias,
        b_norm_weight,
        c_norm_weight,
        b_bias,
        c_bias,
        d_param,
        norm_gate_weight,
        out_proj_w
    ],
    mlp: mlp [norm_weight, norm_bias, fc1_w, fc2_w],
    post: [norm_f_weight, norm_f_bias],
);

pub(crate) struct Mamba3Cpu {
    cfg: Mamba3Config,
    dims: Mamba3Dims,
    input_dim: usize,
    dtype: WeightDtype,
}

/// One sequence's carried states, the tape its backward reads, scratch.
pub(crate) struct Mamba3CpuSeq {
    acts: Vec<Mamba3LayerFlat>,
    ssm: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    angle: Vec<f32>,
    phase: Mamba3Scratch,
    /// The window's input rows, for the input projection's backward.
    input: Vec<f32>,
    /// Each MLP's input rows.
    mlp_in: Vec<Vec<f32>>,
    /// The rows before the final norm and their statistics: the inverse
    /// RMS, or a LayerNorm's std and mean.
    norm_f_in: Vec<f32>,
    norm_f_inv: Vec<f32>,
    /// A bidirectional pair's input with every sequence reversed in the
    /// forward, d loss / d (its output) in the backward; empty for a model
    /// read one way.
    pair: Vec<f32>,
    mlp: GatedMlpRowsScratch,
    mlp_bwd: GatedMlpBackwardScratch,
}

impl Mamba3Cpu {
    /// One layer's share of the four states: `[ssm, k, v, angle]` lengths.
    fn per_layer(&self) -> [usize; 4] {
        let d = &self.dims;
        let (nh, hd, ds) = (d.nheads, d.headdim, d.d_state);
        [
            nh * hd * ds,
            nh * ds,
            nh * hd,
            nh * d.num_rope_angles.max(1),
        ]
    }
}

impl CpuFamily for Mamba3Cpu {
    type Weights = Mamba3CpuWeights;
    type Seq = Mamba3CpuSeq;
    const FAMILY: ModelFamily = ModelFamily::Mamba3;
    const RECURRENCE: &'static [&'static str] = &["ssm", "k", "v", "angle"];

    fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The GPU trainer's mirror: the input projection, in_proj, out_proj
    /// and the MLP's projections.
    fn round_weights(&self, w: &mut Mamba3CpuWeights) {
        round_matrix(self.dtype, &mut w.input_proj_w);
        for lw in &mut w.layers {
            round_matrix(self.dtype, &mut lw.in_proj_w);
            round_matrix(self.dtype, &mut lw.out_proj_w);
        }
        for m in &mut w.mlp {
            round_matrix(self.dtype, &mut m.fc1_w);
            round_matrix(self.dtype, &mut m.fc2_w);
        }
    }

    fn new_seq(&self, w: &Mamba3CpuWeights, seq_len: usize) -> Result<Mamba3CpuSeq, String> {
        let (dm, nl) = (self.cfg.d_model, self.cfg.n_layers);
        let [ssm, k, v, angle] = self.per_layer().map(|n| vec![0.0; nl * n]);
        let mut phase = Mamba3Scratch::zeros(&self.dims);
        let mut mlp = GatedMlpRowsScratch::default();
        let mut mlp_bwd = GatedMlpBackwardScratch::default();
        (phase.dtype, mlp.dtype, mlp_bwd.dtype) = (self.dtype, self.dtype, self.dtype);
        Ok(Mamba3CpuSeq {
            acts: (0..nl).map(|_| Mamba3LayerFlat::zeros(self.dims)).collect(),
            ssm,
            k,
            v,
            angle,
            phase,
            input: vec![0.0; seq_len * self.input_dim],
            mlp_in: w.mlp.iter().map(|_| vec![0.0; seq_len * dm]).collect(),
            norm_f_in: vec![0.0; seq_len * dm],
            norm_f_inv: vec![0.0; seq_len * self.cfg.block_norm.stats_per_row()],
            pair: vec![
                0.0;
                if self.cfg.bidirectional {
                    seq_len * dm
                } else {
                    0
                }
            ],
            mlp,
            mlp_bwd,
        })
    }

    fn reset_seq(&self, seq: &mut Mamba3CpuSeq) {
        for s in [&mut seq.ssm, &mut seq.k, &mut seq.v, &mut seq.angle] {
            s.fill(0.0);
        }
    }

    fn set_varlen(&self, seq: &mut Mamba3CpuSeq, ids: Option<&[i32]>) -> Result<(), String> {
        seq.phase.varlen = ids.map(crate::varlen::RowStarts::from_ids);
        Ok(())
    }

    fn forward(
        &self,
        w: &Mamba3CpuWeights,
        seq: &mut Mamba3CpuSeq,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        let (dm, t_len, eps) = (self.cfg.d_model, self.dims.seq_len, self.cfg.rms_norm_eps);
        let bidi = self.cfg.bidirectional;
        // A bidirectional model reads whole sequences: every window starts
        // from the empty state.
        if bidi {
            self.reset_seq(seq);
        }
        let rows = &mut seq.norm_f_in;
        if w.input_proj_w.is_empty() {
            seq.input.copy_from_slice(input);
            rows.copy_from_slice(&input[..t_len * dm]);
        } else {
            // A half lane casts the rows to its storage, the backward's
            // product reading them so stored, and keeps the projection's
            // output stored; the residual stream starts from that output.
            seq.input.copy_from_slice(input);
            store(self.dtype, &mut seq.input);
            seq.phase.gemm.forward(
                (sgemm_forward as RowsGemm, self.dtype),
                (&mut rows[..], &seq.input),
                (&w.input_proj_w, Some(&w.input_proj_b)),
                (t_len, self.input_dim, dm),
            );
        }
        let [ssm_l, k_l, v_l, a_l] = self.per_layer();
        for (l, lw) in w.layers.iter().enumerate() {
            if opens_pair(bidi, l) {
                seq.pair.copy_from_slice(rows);
                reverse_rows(&mut seq.pair, dm, seq.phase.varlen.as_ref());
            }
            forward_mamba3_layer_in(
                rows,
                &mut seq.acts[l],
                lw,
                Mamba3LayerStateMut {
                    ssm: &mut seq.ssm[l * ssm_l..(l + 1) * ssm_l],
                    k: &mut seq.k[l * k_l..(l + 1) * k_l],
                    v: &mut seq.v[l * v_l..(l + 1) * v_l],
                    angle: &mut seq.angle[l * a_l..(l + 1) * a_l],
                },
                &mut seq.phase,
                &self.dims,
                reads_reversed(bidi, l).then_some(&seq.pair[..]),
            );
            // A pair's MLPs follow both its mixers.
            for m in mlps_after(bidi, l) {
                let Some(mw) = w.mlp.get(m) else { continue };
                seq.mlp_in[m].copy_from_slice(rows);
                gated_mlp_rows(rows, t_len, mw, &mut seq.mlp, sgemm_forward, |dst, src| {
                    for (d_row, s_row) in dst.chunks_exact_mut(dm).zip(src.chunks_exact(dm)) {
                        mlp_norm_row(d_row, s_row, mw, eps);
                    }
                });
            }
        }
        // The final norm in place on a copy, as the inference prefill
        // applies it, keeping its statistics for the backward.
        out[..t_len * dm].copy_from_slice(rows);
        if w.norm_f_weight.is_empty() {
            // A segment without the final norm hands out the residual
            // stream.
        } else if w.norm_f_bias.is_empty() {
            for (row, inv) in out.chunks_exact_mut(dm).zip(seq.norm_f_inv.iter_mut()) {
                *inv = rms_norm_inplace(row, &w.norm_f_weight, eps);
            }
        } else {
            let p = (&w.norm_f_weight[..], &w.norm_f_bias[..]);
            for (row, st) in out
                .chunks_exact_mut(dm)
                .zip(seq.norm_f_inv.as_chunks_mut::<2>().0)
            {
                *st = layer_norm_inplace(row, p, eps);
            }
        }
        Ok(())
    }

    fn backward(
        &self,
        w: &Mamba3CpuWeights,
        seq: &mut Mamba3CpuSeq,
        grads: &mut Mamba3CpuWeights,
        d_out: &mut [f32],
        d_input: &mut [f32],
    ) -> Result<(), String> {
        let (dm, t_len, eps) = (self.cfg.d_model, self.dims.seq_len, self.cfg.rms_norm_eps);
        // The final norm: a LayerNorm's backward, or for RMSNorm
        // dw += dy x_hat, dx = inv (dy w - x_hat mean(dy w x_hat)).
        if w.norm_f_weight.is_empty() {
            // Without the norm the output's gradient is the residual
            // stream's.
        } else if w.norm_f_bias.is_empty() {
            for ((d, x), &inv) in d_out
                .chunks_exact_mut(dm)
                .zip(seq.norm_f_in.chunks_exact(dm))
                .zip(&seq.norm_f_inv)
            {
                let mut c = 0.0f32;
                for ((g, &xv), (&dv, &wv)) in grads
                    .norm_f_weight
                    .iter_mut()
                    .zip(x)
                    .zip(d.iter().zip(&w.norm_f_weight))
                {
                    let xh = xv * inv;
                    *g += dv * xh;
                    c += (dv * wv) * xh;
                }
                let c = c / dm as f32;
                for ((dv, &xv), &wv) in d.iter_mut().zip(x).zip(&w.norm_f_weight) {
                    *dv = inv * (*dv * wv - (xv * inv) * c);
                }
            }
        } else {
            for ((d, x), st) in d_out
                .chunks_exact_mut(dm)
                .zip(seq.norm_f_in.chunks_exact(dm))
                .zip(seq.norm_f_inv.as_chunks::<2>().0)
            {
                layer_norm_row_backward_in_place(
                    d,
                    (&mut grads.norm_f_weight, &mut grads.norm_f_bias),
                    x,
                    &w.norm_f_weight,
                    *st,
                );
            }
        }
        seq.phase.zero_all();
        let mlp_backward =
            |l: usize, d_out: &mut [f32], grads: &mut Mamba3CpuWeights, seq: &mut Mamba3CpuSeq| {
                if let Some(m) = w.mlp.get(l) {
                    gated_mlp_rows_backward(
                        d_out,
                        &seq.mlp_in[l],
                        t_len,
                        (m, &mut grads.mlp[l]),
                        &mut seq.mlp_bwd,
                        eps,
                    );
                }
            };
        if self.cfg.bidirectional {
            // A pair from its MLPs down: both mixers take d loss / d (the
            // pair's output), the second reversed, and each norm's gradient
            // joins the stream's.
            for l in (0..w.layers.len()).step_by(2).rev() {
                mlp_backward(l + 1, d_out, grads, seq);
                mlp_backward(l, d_out, grads, seq);
                seq.pair.copy_from_slice(&d_out[..t_len * dm]);
                reverse_rows(&mut seq.pair, dm, seq.phase.varlen.as_ref());
                for (layer, reversed) in [(l + 1, true), (l, false)] {
                    if !reversed {
                        reverse_rows(&mut seq.pair, dm, seq.phase.varlen.as_ref());
                    }
                    backward_mamba3_layer_in(
                        d_out,
                        &seq.acts[layer],
                        (&w.layers[layer], &mut grads.layers[layer]),
                        &mut seq.phase,
                        (
                            &self.dims,
                            None,
                            Some(PairGrad {
                                d_out: &seq.pair,
                                reversed,
                            }),
                        ),
                    );
                }
            }
        } else {
            for l in (0..w.layers.len()).rev() {
                mlp_backward(l, d_out, grads, seq);
                backward_mamba3_layer_batched(
                    d_out,
                    &seq.acts[l],
                    &w.layers[l],
                    &mut grads.layers[l],
                    &mut seq.phase,
                    &self.dims,
                    None,
                );
            }
        }
        if w.input_proj_w.is_empty() {
            let n = d_input.len();
            d_input.copy_from_slice(&d_out[..n]);
        } else {
            // A half lane casts the gradient to its storage once, for the
            // weight's product and the bias's sum alike.
            let d_rows = &mut d_out[..t_len * dm];
            store(self.dtype, d_rows);
            seq.phase.gemm.backward(
                self.dtype,
                (
                    d_input,
                    &mut grads.input_proj_w,
                    Some(&mut grads.input_proj_b),
                ),
                (d_rows, &seq.input, &w.input_proj_w),
                (t_len, self.input_dim, dm),
            );
        }
        Ok(())
    }

    fn zeros_like(w: &Mamba3CpuWeights) -> Mamba3CpuWeights {
        let mut g = w.clone();
        crate::optim::tensors::TensorList::zero(&mut g);
        g
    }

    /// The matrices decay, every norm scale and bias does not, as on the
    /// GPU; the control channels are the dt/A/trapezoid/angle columns of
    /// in_proj and dt_bias.
    fn roles(&self, w: &Mamba3CpuWeights) -> Vec<TensorRole> {
        let ip = self.cfg.in_proj_out_dim();
        let ctrl = 3 * self.cfg.nheads() + self.cfg.num_rope_angles();
        let role = |decays, control| TensorRole { decays, control };
        let mut roles = vec![role(true, Control::None), role(false, Control::None)];
        for _ in &w.layers {
            roles.extend([
                role(false, Control::None),
                role(false, Control::None),
                role(
                    true,
                    Control::Columns {
                        row_len: ip,
                        cols: ip - ctrl..ip,
                    },
                ),
                role(false, Control::All),
                role(false, Control::None),
                role(false, Control::None),
                role(false, Control::None),
                role(false, Control::None),
                role(false, Control::None),
                role(false, Control::None),
                role(true, Control::None),
            ]);
        }
        for _ in &w.mlp {
            roles.extend([
                role(false, Control::None),
                role(false, Control::None),
                role(true, Control::None),
                role(true, Control::None),
            ]);
        }
        roles.extend([role(false, Control::None), role(false, Control::None)]);
        roles
    }

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    fn recurrence_len(&self, i: usize) -> usize {
        self.per_layer()[i.min(3)]
    }

    /// The host keeps the four states as the GPU does, layer after layer.
    fn export_recurrence(&self, seq: &Mamba3CpuSeq, i: usize, l: usize, out: &mut Vec<f32>) {
        let n = self.recurrence_len(i);
        let all = match i {
            0 => &seq.ssm,
            1 => &seq.k,
            2 => &seq.v,
            _ => &seq.angle,
        };
        out.extend_from_slice(&all[l * n..(l + 1) * n]);
    }

    fn import_recurrence(&self, seq: &mut Mamba3CpuSeq, i: usize, l: usize, values: &[f32]) {
        let n = self.recurrence_len(i);
        let all = match i {
            0 => &mut seq.ssm,
            1 => &mut seq.k,
            2 => &mut seq.v,
            _ => &mut seq.angle,
        };
        all[l * n..(l + 1) * n].copy_from_slice(values);
    }
}

cpu_trainer_type!(
    /// A Mamba-3 SISO model trained on the CPU: the batch's sequences side
    /// by side, the accumulation window, the clips, the spike skip and
    /// AdamW, with the surface of [`crate::module::CpuTrainer`]. The
    /// gradients of the batch sum in a fixed number of chunks reduced in a
    /// fixed tree, so the bits do not depend on the thread pool.
    Mamba3CpuTrainer,
    Mamba3Cpu
);

impl Mamba3CpuTrainer {
    /// A trainer of `weights` (copied as the f32 master) for the session's
    /// batch and window, with its AdamW settings.
    pub fn new(
        weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        Self::new_with(weights, cfg, session, CpuOptions::default())
    }

    /// [`Mamba3CpuTrainer::new`] in the precision `opts` names: f32
    /// masters, the projections and the values between the kernels
    /// rounded as the GPU trainer of that precision rounds them, and in
    /// f16 the dynamic loss scaler.
    pub fn new_with(
        weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        cfg.validate()?;
        weights.validate(&cfg, session.input_dim)?;
        let train = Mamba3CpuWeights {
            input_proj_w: weights.input_proj_w.clone(),
            input_proj_b: weights.input_proj_b.clone(),
            layers: weights
                .layers
                .iter()
                .map(|lw| TrainMamba3LayerWeights {
                    norm_weight: lw.norm_weight.clone(),
                    norm_bias: lw.norm_bias.clone(),
                    in_proj_w: lw.in_proj_w.clone(),
                    dt_bias: lw.dt_bias.clone(),
                    b_norm_weight: lw.b_norm_weight.clone(),
                    c_norm_weight: lw.c_norm_weight.clone(),
                    b_bias: lw.b_bias.clone(),
                    c_bias: lw.c_bias.clone(),
                    d_param: lw.d_param.clone(),
                    norm_gate_weight: lw.norm_gate_weight.clone(),
                    out_proj_w: lw.out_proj_w.clone(),
                })
                .collect(),
            mlp: weights.mlp.clone(),
            norm_f_weight: weights.norm_f_weight.clone(),
            norm_f_bias: weights.norm_f_bias.clone(),
        };
        let fam = Mamba3Cpu {
            cfg,
            dims: Mamba3Dims::from_config(&cfg, session.seq_len),
            input_dim: session.input_dim,
            dtype: opts.dtype,
        };
        Ok(Self {
            core: CpuCore::new(fam, train, session, cfg.d_model)?,
        })
    }

    /// The master weights, for a checkpoint or the inference model.
    pub fn snapshot_master(&self) -> Mamba3Weights {
        let w = &self.core.weights;
        Mamba3Weights {
            input_proj_w: w.input_proj_w.clone(),
            input_proj_b: w.input_proj_b.clone(),
            layers: w
                .layers
                .iter()
                .map(|lw| Mamba3LayerWeights {
                    norm_weight: lw.norm_weight.clone(),
                    norm_bias: lw.norm_bias.clone(),
                    in_proj_w: lw.in_proj_w.clone(),
                    dt_bias: lw.dt_bias.clone(),
                    b_norm_weight: lw.b_norm_weight.clone(),
                    c_norm_weight: lw.c_norm_weight.clone(),
                    b_bias: lw.b_bias.clone(),
                    c_bias: lw.c_bias.clone(),
                    d_param: lw.d_param.clone(),
                    norm_gate_weight: lw.norm_gate_weight.clone(),
                    out_proj_w: lw.out_proj_w.clone(),
                })
                .collect(),
            norm_f_weight: w.norm_f_weight.clone(),
            norm_f_bias: w.norm_f_bias.clone(),
            mlp: w.mlp.clone(),
        }
    }
}
