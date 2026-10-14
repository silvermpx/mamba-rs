//! The Mamba-2 host trainer: the shared CPU trainer over this family's
//! window forward and backward ([`forward_mamba2_train`],
//! [`backward_mamba2_train`]).

use super::super::config::Mamba2Config;
use super::super::state::Mamba2LayerState;
use super::super::weights::Mamba2Weights;
use super::train::{Mamba2Tape, Mamba2TrainScratch, backward_mamba2_train, forward_mamba2_train};
use crate::config::ModelFamily;
use crate::module::cpu_trainer::{Control, CpuCore, CpuFamily, TensorRole, cpu_trainer_type};
use crate::optim::session::TrainSessionCfg;
use crate::optim::tensors::tensor_list;
use crate::precision::{CpuOptions, WeightDtype};

/// Positions between the tape's state checkpoints: the backward replays a
/// head's recurrence over at most this many steps from a stored state.
const TAPE_STRIDE: usize = 16;

tensor_list!(
    Mamba2Weights,
    pre: [input_proj_w, input_proj_b],
    layers: layers [
        norm_weight,
        norm_bias,
        in_proj_w,
        in_proj_b,
        conv_w,
        conv_b,
        dt_bias,
        a_log,
        d_param,
        gate_norm_weight,
        out_proj_w,
        out_proj_b
    ],
    mlp: mlp [norm_weight, norm_bias, fc1_w, fc2_w],
    post: [norm_f_weight, norm_f_bias],
);

pub(crate) struct Mamba2Cpu {
    cfg: Mamba2Config,
    input_dim: usize,
    dtype: WeightDtype,
}

/// One sequence's carried states, tape and scratch.
pub(crate) struct Mamba2CpuSeq {
    states: Vec<Mamba2LayerState>,
    tape: Mamba2Tape,
    scratch: Mamba2TrainScratch,
}

impl CpuFamily for Mamba2Cpu {
    type Weights = Mamba2Weights;
    type Seq = Mamba2CpuSeq;
    const FAMILY: ModelFamily = ModelFamily::Mamba2;
    const RECURRENCE: &'static [&'static str] = &["conv", "ssm"];

    fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    /// The GPU trainer's half mirror: in_proj, out_proj and the MLP's two
    /// projections; TF32 rounds the input projection's operands too.
    fn round_weights(&self, w: &mut Mamba2Weights) {
        w.round_bulk(self.dtype);
    }

    fn new_seq(&self, _w: &Mamba2Weights, seq_len: usize) -> Result<Mamba2CpuSeq, String> {
        Ok(Mamba2CpuSeq {
            states: (0..self.cfg.n_layers)
                .map(|_| Mamba2LayerState::zeros(&self.cfg))
                .collect(),
            tape: Mamba2Tape::new(&self.cfg, seq_len, self.input_dim, TAPE_STRIDE)
                .with_dtype(self.dtype),
            scratch: Mamba2TrainScratch::default(),
        })
    }

    fn reset_seq(&self, seq: &mut Mamba2CpuSeq) {
        seq.states.iter_mut().for_each(Mamba2LayerState::reset);
    }

    fn forward(
        &self,
        w: &Mamba2Weights,
        seq: &mut Mamba2CpuSeq,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        forward_mamba2_train(
            out,
            input,
            w,
            &mut seq.states,
            &mut seq.tape,
            (&self.cfg, &mut seq.scratch),
        );
        Ok(())
    }

    fn backward(
        &self,
        w: &Mamba2Weights,
        seq: &mut Mamba2CpuSeq,
        grads: &mut Mamba2Weights,
        d_out: &mut [f32],
        d_input: &mut [f32],
    ) -> Result<(), String> {
        backward_mamba2_train(
            d_out,
            d_input,
            (w, grads),
            &seq.tape,
            (&self.cfg, &mut seq.scratch),
        );
        Ok(())
    }

    fn zeros_like(w: &Mamba2Weights) -> Mamba2Weights {
        w.zeros_like()
    }

    /// The matrices decay; the norms' scales and biases do not. The control
    /// channels are the dt columns of in_proj (with their bias entries where
    /// the model has the bias), dt_bias and A_log, as on the GPU.
    fn roles(&self, w: &Mamba2Weights) -> Vec<TensorRole> {
        let (ip, nh) = (self.cfg.in_proj_out_dim(), self.cfg.nheads());
        let dt = Control::Columns {
            row_len: ip,
            cols: ip - nh..ip,
        };
        let plain = |decays| TensorRole {
            decays,
            control: Control::None,
        };
        let mut roles = vec![plain(true), plain(false)];
        for _ in &w.layers {
            roles.extend([
                plain(false),
                plain(false),
                TensorRole {
                    decays: true,
                    control: dt.clone(),
                },
                TensorRole {
                    decays: false,
                    control: dt.clone(),
                },
                plain(true),
                plain(false),
                TensorRole {
                    decays: false,
                    control: Control::All,
                },
                TensorRole {
                    decays: false,
                    control: Control::All,
                },
                plain(false),
                plain(false),
                plain(true),
                plain(false),
            ]);
        }
        for _ in &w.mlp {
            roles.extend([plain(false), plain(false), plain(true), plain(true)]);
        }
        roles.extend([plain(false), plain(false)]);
        roles
    }

    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }

    /// The GPU's conv state is the reference's window `[conv_dim, d_conv]`,
    /// newest input last, and its SSM state `[nheads, headdim, d_state]`.
    fn recurrence_len(&self, i: usize) -> usize {
        let c = &self.cfg;
        match i {
            0 => c.conv_dim() * c.d_conv,
            _ => c.nheads() * c.d_state * c.headdim,
        }
    }

    /// The host keeps the last `d_conv - 1` inputs `[d_conv - 1, conv_dim]`
    /// and the SSM state `[nheads, d_state, headdim]`; both turn into the
    /// GPU's layout. The window's first column, the oldest input, is read
    /// by no later step and goes out zero, as the GPU trainer exports it.
    fn export_recurrence(&self, seq: &Mamba2CpuSeq, i: usize, l: usize, out: &mut Vec<f32>) {
        let (c, st) = (&self.cfg, &seq.states[l]);
        if i == 0 {
            let (cd, k) = (c.conv_dim(), c.d_conv);
            for ch in 0..cd {
                out.push(0.0);
                out.extend((0..k - 1).map(|t| st.conv_hist[t * cd + ch]));
            }
        } else {
            let (n, p) = (c.d_state, c.headdim);
            for head in st.ssm.chunks_exact(n * p) {
                for pi in 0..p {
                    out.extend((0..n).map(|ni| head[ni * p + pi]));
                }
            }
        }
    }

    fn import_recurrence(&self, seq: &mut Mamba2CpuSeq, i: usize, l: usize, values: &[f32]) {
        let (c, st) = (&self.cfg, &mut seq.states[l]);
        if i == 0 {
            let (cd, k) = (c.conv_dim(), c.d_conv);
            for (ch, window) in values.chunks_exact(k).enumerate() {
                for (t, &v) in window[1..].iter().enumerate() {
                    st.conv_hist[t * cd + ch] = v;
                }
            }
        } else {
            let (n, p) = (c.d_state, c.headdim);
            for (head, src) in st
                .ssm
                .chunks_exact_mut(n * p)
                .zip(values.chunks_exact(n * p))
            {
                for (pi, row) in src.chunks_exact(n).enumerate() {
                    for (ni, &v) in row.iter().enumerate() {
                        head[ni * p + pi] = v;
                    }
                }
            }
        }
    }
}

cpu_trainer_type!(
    /// A Mamba-2 model trained on the CPU: the batch's sequences side by
    /// side, the accumulation window, the clips, the spike skip and AdamW,
    /// with the surface of [`crate::module::CpuTrainer`]. The gradients of
    /// the batch sum in a fixed number of chunks reduced in a fixed tree,
    /// so the bits do not depend on the thread pool.
    Mamba2CpuTrainer,
    Mamba2Cpu
);

impl Mamba2CpuTrainer {
    /// A trainer of `weights` (copied as the f32 master) for the session's
    /// batch and window, with its AdamW settings.
    pub fn new(
        weights: &Mamba2Weights,
        cfg: Mamba2Config,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        Self::new_with(weights, cfg, session, CpuOptions::default())
    }

    /// [`Mamba2CpuTrainer::new`] in the precision `opts` names: f32
    /// masters, the bulk matrices and the values between the kernels
    /// rounded as the GPU trainer of that precision rounds them, and in
    /// f16 the dynamic loss scaler.
    pub fn new_with(
        weights: &Mamba2Weights,
        cfg: Mamba2Config,
        session: TrainSessionCfg,
        opts: CpuOptions,
    ) -> Result<Self, String> {
        cfg.validate()?;
        weights.validate(&cfg, session.input_dim)?;
        let fam = Mamba2Cpu {
            cfg,
            input_dim: session.input_dim,
            dtype: opts.dtype,
        };
        Ok(Self {
            core: CpuCore::new(fam, weights.clone(), session, cfg.d_model)?,
        })
    }

    /// The master weights, for a checkpoint or the inference model.
    pub fn snapshot_master(&self) -> Mamba2Weights {
        self.core.weights.clone()
    }
}
