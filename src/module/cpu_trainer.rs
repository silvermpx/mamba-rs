//! The host trainers of every family: the batch, the accumulation window,
//! the clips, the spike skip and the AdamW of a GPU trainer, on the CPU
//! over the family's own forward and backward.
//!
//! A family gives one sequence's forward (with the tape its backward
//! needs) and that backward; [`CpuCore`] runs the batch's sequences side by
//! side, sums their gradients in a fixed number of chunks reduced in a
//! fixed tree (so the bits do not depend on the thread pool), keeps the
//! window, clips, skips and steps. In a reduced precision it keeps f32
//! masters and the weights the family computes with rounded from them after
//! every step, and in f16 it scales the window's gradient with the dynamic
//! loss scaler, as the GPU trainers do. [`CpuTrainer`] names the surface
//! the four families share, and [`AnyCpuTrainer`] holds one chosen at run
//! time.

use std::ops::Range;

use rayon::prelude::*;

use crate::config::ModelFamily;
use crate::optim::adamw::{AdamWStateBlob, HostAdamW};
use crate::optim::loss_scaler::DynamicLossScaler;
use crate::optim::session::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};
pub(crate) use crate::optim::tensors::TensorList;
use crate::optim::train_state::TrainerState;
use crate::precision::WeightDtype;
use crate::varlen::Varlen;

/// The elements of a tensor in the control channels: none, all of them,
/// or the columns `cols` of every `row_len`-wide row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Control {
    None,
    All,
    Columns { row_len: usize, cols: Range<usize> },
}

impl Control {
    /// Visit the control elements of `t` in order.
    fn for_each(&self, t: &mut [f32], mut f: impl FnMut(&mut f32)) {
        match self {
            Control::None => {}
            Control::All => t.iter_mut().for_each(f),
            Control::Columns { row_len, cols } => {
                if *row_len == 0 {
                    return;
                }
                for row in t.chunks_exact_mut(*row_len) {
                    row[cols.clone()].iter_mut().for_each(&mut f);
                }
            }
        }
    }
}

/// How AdamW and the clips treat one tensor.
#[derive(Clone, Debug)]
pub(crate) struct TensorRole {
    /// Whether the tensor decays under the reference's no-decay groups
    /// (the matrices do; A, D, biases and norm scales do not).
    pub(crate) decays: bool,
    pub(crate) control: Control,
}

/// One family's part of a host trainer.
pub(crate) trait CpuFamily: Send + Sync {
    /// The master weights, also the shape of a gradient.
    type Weights: TensorList + Clone + Send + Sync;
    /// One sequence's carried recurrence, tape and scratch.
    type Seq: Send;
    const FAMILY: ModelFamily;
    /// The names of the carried recurrence's tensors, as the trainer
    /// state stores them.
    const RECURRENCE: &'static [&'static str];

    /// The precision the family computes in.
    fn dtype(&self) -> WeightDtype;
    /// Round in place the tensors of `w` the family's GPU trainer holds in
    /// the storage of [`CpuFamily::dtype`] (bf16, f16) or multiplies in
    /// TF32: its bulk matrices. The others keep the masters' values.
    fn round_weights(&self, w: &mut Self::Weights);

    fn new_seq(&self, w: &Self::Weights, seq_len: usize) -> Result<Self::Seq, String>;
    /// Zero the sequence's carried recurrence.
    fn reset_seq(&self, seq: &mut Self::Seq);
    /// The sequence ids of this row's windows from now on (see
    /// [`crate::varlen`]); `None` runs every window as one sequence. A
    /// family the reference gives no varlen refuses ids.
    fn set_varlen(&self, _seq: &mut Self::Seq, ids: Option<&[i32]>) -> Result<(), String> {
        match ids {
            None => Ok(()),
            Some(_) => Err(format!(
                "{:?} has no varlen: the reference packs only Mamba-2 and Mamba-3",
                Self::FAMILY
            )),
        }
    }
    /// One sequence's forward over its window: `input` `[seq_len,
    /// input_dim]`, `out` `[seq_len, d_model]`, the recurrence carried.
    fn forward(
        &self,
        w: &Self::Weights,
        seq: &mut Self::Seq,
        input: &[f32],
        out: &mut [f32],
    ) -> Result<(), String>;
    /// The backward of the sequence's last forward: `d_out` (consumed as
    /// scratch) to the weight gradients added into `grads` and the input's
    /// gradient written to `d_input`.
    fn backward(
        &self,
        w: &Self::Weights,
        seq: &mut Self::Seq,
        grads: &mut Self::Weights,
        d_out: &mut [f32],
        d_input: &mut [f32],
    ) -> Result<(), String>;
    /// A gradient container: the shapes of `w`, every element zero.
    fn zeros_like(w: &Self::Weights) -> Self::Weights;
    /// Each tensor's role, in the list's order.
    fn roles(&self, w: &Self::Weights) -> Vec<TensorRole>;
    /// Refresh what the family derives from the weights (Mamba-1's
    /// `-exp(a_log)`) after an optimizer step or a load.
    fn after_update(&mut self, _w: &Self::Weights) {}
    /// The layers, each holding a slice of every recurrence tensor.
    fn n_layers(&self) -> usize;
    /// The length of one sequence's slice of the recurrence tensor `i` in
    /// one layer, as the family's GPU trainer lays it out.
    fn recurrence_len(&self, i: usize) -> usize;
    /// Append one sequence's slice of the recurrence tensor `i` in layer
    /// `l` to `out`, in the GPU trainer's layout.
    fn export_recurrence(&self, seq: &Self::Seq, i: usize, l: usize, out: &mut Vec<f32>);
    /// Load one sequence's slice of the recurrence tensor `i` in layer `l`
    /// from `values`, laid out as [`CpuFamily::export_recurrence`] gives
    /// it.
    fn import_recurrence(&self, seq: &mut Self::Seq, i: usize, l: usize, values: &[f32]);
}

/// What every host trainer runs, over a family's sequences.
pub(crate) struct CpuCore<F: CpuFamily> {
    pub(crate) fam: F,
    /// The f32 masters AdamW updates.
    pub(crate) weights: F::Weights,
    /// The weights the family computes with in a reduced precision: the
    /// masters with the bulk matrices rounded, refreshed after every
    /// optimizer step as the GPU's AdamW writes its compute copy. `None` in
    /// f32, where they are the masters.
    compute: Option<F::Weights>,
    /// Scales the window's gradient in f16; disabled in the other
    /// precisions.
    scaler: DynamicLossScaler,
    /// The open window's gradient sum.
    grads: F::Weights,
    /// One gradient per chunk of the batch, reduced in a fixed tree.
    slots: Vec<F::Weights>,
    seqs: Vec<F::Seq>,
    roles: Vec<TensorRole>,
    adam: HostAdamW,
    batch: usize,
    seq_len: usize,
    input_dim: usize,
    d_model: usize,
    /// A forward whose backward has not run.
    forwarded: bool,
    /// The options of the fused [`CpuCore::step`].
    step_opts: BackwardOpts,
    /// The gradient holds micro-batches the optimizer has not taken.
    window_open: bool,
    /// The upstream gradient, which the backward consumes.
    d_buf: Vec<f32>,
    d_input_buf: Vec<f32>,
    out_buf: Vec<f32>,
}

impl<F: CpuFamily> CpuCore<F> {
    pub(crate) fn new(
        fam: F,
        weights: F::Weights,
        session: TrainSessionCfg,
        d_model: usize,
    ) -> Result<Self, String> {
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        if batch == 0 || seq_len == 0 {
            return Err("a trainer needs at least one sequence of one step".into());
        }
        let seqs = (0..batch)
            .map(|_| fam.new_seq(&weights, seq_len))
            .collect::<Result<Vec<_>, String>>()?;
        let n_chunks = crate::ops::BATCH_GRAD_CHUNKS.min(batch);
        let roles = fam.roles(&weights);
        if roles.len() != weights.n_tensors() {
            return Err("every tensor needs a role".into());
        }
        let n_params = weights.n_elements();
        let rows = batch * seq_len;
        let dtype = fam.dtype();
        let compute = (dtype != WeightDtype::F32).then(|| {
            let mut c = weights.clone();
            fam.round_weights(&mut c);
            c
        });
        let scaler = if dtype == WeightDtype::F16 {
            DynamicLossScaler::new()
        } else {
            DynamicLossScaler::disabled()
        };
        Ok(Self {
            compute,
            scaler,
            grads: F::zeros_like(&weights),
            slots: (0..n_chunks).map(|_| F::zeros_like(&weights)).collect(),
            seqs,
            roles,
            adam: HostAdamW::new(n_params, lr, weight_decay),
            batch,
            seq_len,
            input_dim,
            d_model,
            forwarded: false,
            step_opts: BackwardOpts::default(),
            window_open: false,
            d_buf: vec![0.0; rows * d_model],
            d_input_buf: vec![0.0; rows * input_dim],
            out_buf: vec![0.0; rows * d_model],
            fam,
            weights,
        })
    }

    pub(crate) fn batch(&self) -> usize {
        self.batch
    }
    pub(crate) fn seq_len(&self) -> usize {
        self.seq_len
    }
    pub(crate) fn input_dim(&self) -> usize {
        self.input_dim
    }
    pub(crate) fn d_model(&self) -> usize {
        self.d_model
    }
    pub(crate) fn lr(&self) -> f32 {
        self.adam.lr
    }
    pub(crate) fn set_lr(&mut self, lr: f32) {
        self.adam.lr = lr;
    }
    pub(crate) fn dtype(&self) -> WeightDtype {
        self.fam.dtype()
    }

    /// The f16 loss scale and its growth counter; `None` in the other
    /// precisions.
    pub(crate) fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.enabled().then(|| self.scaler.state())
    }

    /// Restore the f16 loss scale and its growth counter; no effect in the
    /// other precisions.
    pub(crate) fn load_scaler_state(
        &mut self,
        scale: f32,
        growth_tracker: u32,
    ) -> Result<(), String> {
        if self.scaler.enabled() {
            self.scaler.load_state(scale, growth_tracker)?;
        }
        Ok(())
    }

    /// Round the compute copy afresh from the masters.
    fn refresh_compute(&mut self) {
        if let Some(c) = self.compute.as_mut() {
            c.clone_from(&self.weights);
            self.fam.round_weights(c);
        }
    }

    fn check_len(&self, what: &str, got: usize, width: usize) -> Result<(), String> {
        let want = self.batch * self.seq_len * width;
        if got != want {
            return Err(format!(
                "{what}: {got} values for batch {} x seq_len {} x {width}",
                self.batch, self.seq_len
            ));
        }
        Ok(())
    }

    /// The forward over a window of every sequence, `input` `[batch,
    /// seq_len, input_dim]`, `out` `[batch, seq_len, d_model]`; the
    /// recurrence carries from the last window.
    pub(crate) fn forward(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        self.check_len("forward input", input.len(), self.input_dim)?;
        self.check_len("forward output", out.len(), self.d_model)?;
        self.forwarded = false;
        let (per_in, per_out) = (self.seq_len * self.input_dim, self.seq_len * self.d_model);
        let (fam, w) = (&self.fam, self.compute.as_ref().unwrap_or(&self.weights));
        self.seqs
            .par_iter_mut()
            .zip(input.par_chunks(per_in))
            .zip(out.par_chunks_mut(per_out))
            .try_for_each(|((seq, x), y)| fam.forward(w, seq, x, y))?;
        self.forwarded = true;
        Ok(())
    }

    /// The backward of the last forward into the window, every chunk's
    /// sequences in order into the chunk's slot, the slots reduced in a
    /// fixed tree. `d_input`, when given, receives the input's gradient.
    fn backward_into_window(
        &mut self,
        d_temporal: &[f32],
        d_input: Option<&mut [f32]>,
    ) -> Result<(), String> {
        if !self.forwarded {
            return Err("backward needs a completed forward of this window".into());
        }
        self.check_len("backward d_temporal", d_temporal.len(), self.d_model)?;
        if let Some(d) = d_input.as_deref() {
            self.check_len("backward d_input", d.len(), self.input_dim)?;
        }
        self.forwarded = false;
        self.d_buf.copy_from_slice(d_temporal);
        // f16: the window's gradient at the loss scale, every micro-batch
        // at the same one.
        let scale = self.scaler.enabled().then(|| self.scaler.scale());
        if let Some(s) = scale {
            self.d_buf.iter_mut().for_each(|x| *x *= s);
        }
        let n_chunks = self.slots.len();
        let (per_out, per_in) = (self.seq_len * self.d_model, self.seq_len * self.input_dim);
        // The chunks own fixed ranges of the batch, a function of the batch
        // alone.
        let mut parts = Vec::with_capacity(n_chunks);
        let (mut seqs, mut d, mut d_in, mut slots) = (
            &mut self.seqs[..],
            &mut self.d_buf[..],
            &mut self.d_input_buf[..],
            &mut self.slots[..],
        );
        for c in 0..n_chunks {
            let n = (c + 1) * self.batch / n_chunks - c * self.batch / n_chunks;
            let (s_head, s_rest) = seqs.split_at_mut(n);
            let (d_head, d_rest) = d.split_at_mut(n * per_out);
            let (i_head, i_rest) = d_in.split_at_mut(n * per_in);
            let (slot, slot_rest) = slots.split_at_mut(1);
            parts.push((s_head, d_head, i_head, &mut slot[0]));
            (seqs, d, d_in, slots) = (s_rest, d_rest, i_rest, slot_rest);
        }
        let (fam, w) = (&self.fam, self.compute.as_ref().unwrap_or(&self.weights));
        parts
            .into_par_iter()
            .try_for_each(|(seqs, d, d_in, slot)| {
                slot.zero();
                for ((seq, d), d_in) in seqs
                    .iter_mut()
                    .zip(d.chunks_mut(per_out))
                    .zip(d_in.chunks_mut(per_in))
                {
                    fam.backward(w, seq, slot, d, d_in)?;
                }
                Ok::<(), String>(())
            })?;
        // The fixed stride-half tree: pairing is a function of the chunk
        // count alone, so the summation order is too.
        let mut len = n_chunks;
        while len > 1 {
            let half = len / 2;
            let (left, right) = self.slots[..len].split_at_mut(half);
            left.par_iter_mut()
                .zip(right[..half].par_iter())
                .for_each(|(a, b)| a.add(b));
            if len % 2 == 1 {
                self.slots.swap(half, len - 1);
                len = half + 1;
            } else {
                len = half;
            }
        }
        if self.window_open {
            self.grads.add(&self.slots[0]);
        } else {
            std::mem::swap(&mut self.grads, &mut self.slots[0]);
        }
        self.window_open = true;
        if let Some(d) = d_input {
            d.copy_from_slice(&self.d_input_buf);
            if let Some(s) = scale {
                // A half lane stores the input projection's input gradient
                // in f16, and no weight gradient reads it: an overflow there
                // would reach the caller while the step went on. It marks
                // the window instead, whose overflow check then skips it and
                // backs the scale off, as the GPU trainers do.
                if d.iter().any(|x| !x.is_finite())
                    && let Some(i) =
                        (0..self.grads.n_tensors()).find(|&i| !self.grads.tensor(i).is_empty())
                {
                    self.grads.tensor_mut(i)[0] = f32::NAN;
                }
                let inv = 1.0 / s;
                d.iter_mut().for_each(|x| *x *= inv);
            }
        }
        Ok(())
    }

    /// The backward of the last forward for `d_temporal`, then the tail
    /// `opts` asks for (see [`BackwardOpts`]); `accumulate_only` adds the
    /// gradient to the window and waits.
    pub(crate) fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: Option<&mut [f32]>,
    ) -> Result<BackwardMetrics, String> {
        if opts.accumulate_only
            && (opts.clip_max_norm.is_some() || opts.control_clip_max_norm.is_some())
        {
            return Err(
                "a clip belongs on the applying call: the norm is defined over the whole window"
                    .into(),
            );
        }
        self.backward_into_window(d_temporal, d_input)?;
        if opts.accumulate_only {
            return Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm: None,
                loss_scale: self.scaler_state().map(|(s, _)| s),
                overflow_skipped: None,
            });
        }
        self.apply_tail(
            opts.clip_max_norm,
            opts.step_skip_above,
            opts.control_clip_max_norm,
        )
    }

    /// The forward of `input` and the backward of `d_temporal` with the
    /// step options (see [`Self::set_step_opts`]).
    pub(crate) fn step(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<StepMetrics, String> {
        let mut out = std::mem::take(&mut self.out_buf);
        let ran = self.forward(input, &mut out);
        self.out_buf = out;
        ran?;
        let m = self.backward_step(d_temporal, self.step_opts, None)?;
        Ok(StepMetrics::from_backward(m, false))
    }

    /// Set the options of the fused [`Self::step`]: the control clip, the
    /// global clip and the spike skip. Accumulation belongs to
    /// `backward_step` and is refused here.
    pub(crate) fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        self.step_opts = opts;
        Ok(())
    }

    pub(crate) fn step_opts(&self) -> BackwardOpts {
        self.step_opts
    }

    /// One micro-batch of an accumulation window: the forward of `input`
    /// and the backward of `d_temporal`, the gradient added to the window.
    pub(crate) fn accumulate(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<BackwardMetrics, String> {
        let mut out = std::mem::take(&mut self.out_buf);
        let ran = self.forward(input, &mut out);
        self.out_buf = out;
        ran?;
        self.backward_step(
            d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
            None,
        )
    }

    /// The open window's tail with the step options.
    pub(crate) fn apply_window(&mut self) -> Result<StepMetrics, String> {
        let o = self.step_opts;
        let m =
            self.apply_step_full(o.clip_max_norm, o.step_skip_above, o.control_clip_max_norm)?;
        Ok(StepMetrics::from_backward(m, false))
    }

    /// Only the optimizer tail over the open window: the control clip,
    /// the global clip, the spike skip and the AdamW step.
    pub(crate) fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        if !self.window_open {
            return Err("no accumulation window is open".into());
        }
        self.apply_tail(clip_max_norm, step_skip_above, control_clip_max_norm)
    }

    /// The norm of the selected gradient elements, summed in f64 in the
    /// list's order, and the clip coefficient for `max_norm`:
    /// `max_norm / (norm + 1e-6)` below one, else exactly one.
    fn norm_and_coef(&mut self, control_only: bool, max_norm: f32) -> Result<(f32, f32), String> {
        let mut sum = 0.0f64;
        for i in 0..self.grads.n_tensors() {
            let g = self.grads.tensor_mut(i);
            if control_only {
                self.roles[i]
                    .control
                    .for_each(g, |x| sum += f64::from(*x) * f64::from(*x));
            } else {
                sum += g.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
            }
        }
        let norm = sum.sqrt();
        if !norm.is_finite() {
            return Err(format!("the gradient norm is not finite ({norm})"));
        }
        let coef = f64::from(max_norm) / (norm + 1e-6);
        Ok((norm as f32, if coef < 1.0 { coef as f32 } else { 1.0 }))
    }

    fn apply_tail(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        // f16: a window whose gradient overflowed is skipped whole and the
        // scale backs off; a clean one is unscaled before the clips.
        let loss_scale = self.scaler_state().map(|(s, _)| s);
        if let Some(s) = loss_scale {
            let overflow = (0..self.grads.n_tensors())
                .any(|i| self.grads.tensor(i).iter().any(|x| !x.is_finite()));
            self.scaler.update(overflow);
            if overflow {
                self.window_open = false;
                return Ok(BackwardMetrics {
                    step: self.adam.step,
                    optimizer_stepped: false,
                    grad_norm: None,
                    loss_scale,
                    overflow_skipped: Some(true),
                });
            }
            let inv = 1.0 / s;
            for i in 0..self.grads.n_tensors() {
                self.grads.tensor_mut(i).iter_mut().for_each(|x| *x *= inv);
            }
        }
        let overflow_skipped = loss_scale.map(|_| false);
        // The control channels first, so a window whose step-size and
        // decay gradients grow with its length cannot rescale the whole
        // gradient through the global clip below.
        if let Some(c) = control_clip_max_norm {
            let (_, coef) = self.norm_and_coef(true, c)?;
            for i in 0..self.grads.n_tensors() {
                let g = self.grads.tensor_mut(i);
                self.roles[i].control.for_each(g, |x| *x *= coef);
            }
        }
        let limit = clip_max_norm.or(step_skip_above.map(|_| f32::INFINITY));
        let grad_norm = match limit {
            Some(c) => {
                let (norm, coef) = self.norm_and_coef(false, c)?;
                for i in 0..self.grads.n_tensors() {
                    self.grads.tensor_mut(i).iter_mut().for_each(|x| *x *= coef);
                }
                Some(norm)
            }
            None => None,
        };
        self.window_open = false;
        // A window whose norm before the clip passes the threshold is
        // dropped whole: no AdamW step.
        if let (Some(norm), Some(limit)) = (grad_norm, step_skip_above)
            && norm > limit
        {
            return Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm,
                loss_scale,
                overflow_skipped,
            });
        }
        let roles = &self.roles;
        self.adam
            .step(&mut self.weights, &self.grads, |i| roles[i].decays)?;
        self.fam.after_update(&self.weights);
        self.refresh_compute();
        Ok(BackwardMetrics {
            step: self.adam.step,
            optimizer_stepped: true,
            grad_norm,
            loss_scale,
            overflow_skipped,
        })
    }

    /// Zero every sequence's carried recurrence, the start of new
    /// sequences.
    pub(crate) fn reset_state(&mut self) {
        let fam = &self.fam;
        self.seqs.iter_mut().for_each(|seq| fam.reset_seq(seq));
    }

    /// The sequences packed into the rows of the next windows (see
    /// [`crate::varlen`]); `None` runs each row as one sequence.
    pub(crate) fn set_varlen(&mut self, packing: Option<&Varlen>) -> Result<(), String> {
        if let Some(v) = packing {
            v.check_shape(self.batch, self.seq_len)?;
        }
        let fam = &self.fam;
        for (b, seq) in self.seqs.iter_mut().enumerate() {
            fam.set_varlen(seq, packing.map(|v| v.row(b)))?;
        }
        Ok(())
    }

    /// The optimizer state for a bit-continuous resume.
    pub(crate) fn optimizer_state(&self) -> AdamWStateBlob {
        self.adam.export_state()
    }

    pub(crate) fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
        self.adam.import_state(blob)
    }

    /// The carried recurrence, tensor by tensor as the family names it,
    /// each laid out layer by layer and within a layer sequence by
    /// sequence, as the GPU trainers keep it: a state either device saved
    /// loads into the other.
    pub(crate) fn recurrent_state(&self) -> Vec<(String, Vec<f32>)> {
        let layers = self.fam.n_layers();
        F::RECURRENCE
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let mut values =
                    Vec::with_capacity(layers * self.batch * self.fam.recurrence_len(i));
                for l in 0..layers {
                    for seq in &self.seqs {
                        self.fam.export_recurrence(seq, i, l, &mut values);
                    }
                }
                ((*name).to_string(), values)
            })
            .collect()
    }

    /// An error unless `values` holds the batch's recurrence tensor `i`.
    fn check_recurrence(&self, i: usize, values: &[f32]) -> Result<(), String> {
        let per = self.fam.recurrence_len(i) * self.fam.n_layers();
        if values.len() != per * self.batch {
            return Err(format!(
                "recurrence '{}': {} values for {} sequences of {per}",
                F::RECURRENCE[i],
                values.len(),
                self.batch
            ));
        }
        Ok(())
    }

    /// Load the recurrence tensor `i` from `values`, laid out as
    /// [`Self::recurrent_state`] gives it and checked by
    /// [`Self::check_recurrence`].
    fn load_recurrence(&mut self, i: usize, values: &[f32]) {
        let per = self.fam.recurrence_len(i);
        if per == 0 {
            return;
        }
        let fam = &self.fam;
        let mut chunks = values.chunks_exact(per);
        for l in 0..fam.n_layers() {
            for seq in &mut self.seqs {
                if let Some(chunk) = chunks.next() {
                    fam.import_recurrence(seq, i, l, chunk);
                }
            }
        }
    }

    /// Everything beside the weights a bit-continuous resume needs.
    pub(crate) fn trainer_state(&self) -> TrainerState {
        TrainerState {
            family: F::FAMILY,
            optimizer: self.optimizer_state(),
            scaler: self.scaler_state(),
            recurrent: self.recurrent_state(),
        }
    }

    /// Load a state a trainer of this family gave; another family's, or a
    /// recurrence of another shape, is refused before anything changes.
    pub(crate) fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        state.check_family(F::FAMILY)?;
        let tensors = F::RECURRENCE
            .iter()
            .map(|name| state.recurrent_tensor(name))
            .collect::<Result<Vec<_>, String>>()?;
        for (i, values) in tensors.iter().enumerate() {
            self.check_recurrence(i, values)?;
        }
        if let Some((scale, tracker)) = state.scaler {
            DynamicLossScaler::check_scale(scale)?;
            self.load_scaler_state(scale, tracker)?;
        }
        self.adam.import_state(&state.optimizer)?;
        for (i, values) in tensors.iter().enumerate() {
            self.load_recurrence(i, values);
        }
        Ok(())
    }

    /// The open window's gradient, flat in the list's order, for a reducer
    /// of the caller's own.
    pub(crate) fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
        if !self.window_open {
            return Err("no accumulation window is open".into());
        }
        let mut out = Vec::with_capacity(self.grads.n_elements());
        for i in 0..self.grads.n_tensors() {
            out.extend_from_slice(self.grads.tensor(i));
        }
        Ok(out)
    }

    /// Overwrite the open window's gradient with `values`, a reduced arena
    /// of the same length.
    pub(crate) fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
        if !self.window_open {
            return Err("no accumulation window is open".into());
        }
        if values.len() != self.grads.n_elements() {
            return Err(format!(
                "upload_grad_arena: {} values for an arena of {}",
                values.len(),
                self.grads.n_elements()
            ));
        }
        let mut at = 0;
        for i in 0..self.grads.n_tensors() {
            let t = self.grads.tensor_mut(i);
            let len = t.len();
            t.copy_from_slice(&values[at..at + len]);
            at += len;
        }
        Ok(())
    }
}

/// The surface every family's host trainer shares, as `GpuTrainer` does on
/// the GPU (feature `cuda`): the same calls with the same meaning, so a
/// loop written against one reads like a loop against the other.
pub trait CpuTrainer {
    /// The forward over a window of every sequence: `input` `[batch,
    /// seq_len, input_dim]`, `temporal_out` `[batch, seq_len, d_model]`
    /// every position's final-norm output. The recurrence carries from the
    /// last window.
    fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String>;

    /// The backward of the last forward for `d_temporal` (`[batch, seq_len,
    /// d_model]`), then the optimizer tail `opts` asks for: the control
    /// clip, the global clip, the spike skip and the AdamW step; with
    /// `accumulate_only` the gradient joins the window and the optimizer
    /// waits.
    fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String>;

    /// [`CpuTrainer::backward_step`] that also writes d loss / d the
    /// window's input into `d_input` (`[batch, seq_len, input_dim]`).
    fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String>;

    /// The forward of `input` and the backward of `d_temporal` with the
    /// step options.
    fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String>;

    /// Set the options [`CpuTrainer::step`] applies: the control clip, the
    /// global clip and the spike skip, as on the GPU. Accumulation is
    /// refused: the fused step applies its window.
    fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String>;

    /// The options [`CpuTrainer::step`] applies.
    fn step_opts(&self) -> BackwardOpts;

    /// One micro-batch of an accumulation window in one call: the forward
    /// of `input` and the backward of `d_temporal`, the gradient added to
    /// the window (the first call opens it); AdamW waits for
    /// [`CpuTrainer::apply_window`].
    fn accumulate(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<BackwardMetrics, String>;

    /// Apply the open accumulation window with the step options, as an
    /// applying `backward_step` with those options closes a window.
    fn apply_window(&mut self) -> Result<StepMetrics, String>;

    /// Only the optimizer tail over the open window; after accumulating
    /// calls it gives the bits of an applying backward.
    fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String>;

    /// The open window's gradient, flat, for a reducer of the caller's own.
    fn download_grad_arena(&self) -> Result<Vec<f32>, String>;

    /// Overwrite the open window's gradient with a reduced arena of the
    /// same length.
    fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String>;

    /// Zero every sequence's carried recurrence.
    fn reset_state(&mut self) -> Result<(), String>;

    /// Pack several sequences of different lengths into the rows of the
    /// next windows (see [`crate::varlen`]): no state or gradient crosses
    /// from one into the next. `None` runs each row as one sequence. The
    /// packing holds until replaced; Mamba-1 has none in the reference and
    /// refuses one.
    fn set_varlen(&mut self, packing: Option<&Varlen>) -> Result<(), String>;

    fn set_lr(&mut self, lr: f32) -> Result<(), String>;
    fn lr(&self) -> f32;
    fn batch(&self) -> usize;
    fn seq_len(&self) -> usize;
    fn input_dim(&self) -> usize;
    fn d_model(&self) -> usize;

    /// The precision the trainer computes in (see
    /// [`CpuOptions`](crate::CpuOptions)).
    fn dtype(&self) -> crate::WeightDtype;

    /// The f16 loss scale and its growth counter; `None` in the other
    /// precisions.
    fn scaler_state(&self) -> Option<(f32, u32)>;

    /// Restore the f16 loss scale and its growth counter; no effect in the
    /// other precisions. A scale that is not finite and positive is an
    /// error.
    fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String>;

    /// The AdamW state for a bit-continuous resume.
    fn optimizer_state(&self) -> Result<AdamWStateBlob, String>;

    /// Load an optimizer state; one of another parameter count is an
    /// error.
    fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String>;

    /// Everything beside the weights a bit-continuous resume needs: the
    /// optimizer state and the carried recurrence.
    fn trainer_state(&self) -> Result<TrainerState, String>;

    /// Load a [`CpuTrainer::trainer_state`] of a trainer of the same family
    /// and shape.
    fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String>;
}

/// A family's public host trainer: the shared surface over [`CpuCore`].
macro_rules! cpu_trainer_type {
    ($(#[$doc:meta])* $name:ident, $fam:ty) => {
        $(#[$doc])*
        pub struct $name {
            core: $crate::module::cpu_trainer::CpuCore<$fam>,
        }

        impl $name {
            /// See [`CpuTrainer::forward`]($crate::module::CpuTrainer::forward).
            pub fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
                self.core.forward(input, temporal_out)
            }

            /// See [`CpuTrainer::backward_step`]($crate::module::CpuTrainer::backward_step).
            pub fn backward_step(
                &mut self,
                d_temporal: &[f32],
                opts: $crate::optim::BackwardOpts,
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                self.core.backward_step(d_temporal, opts, None)
            }

            /// See [`CpuTrainer::backward_step_input_grad`]($crate::module::CpuTrainer::backward_step_input_grad).
            pub fn backward_step_input_grad(
                &mut self,
                d_temporal: &[f32],
                opts: $crate::optim::BackwardOpts,
                d_input: &mut [f32],
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                self.core.backward_step(d_temporal, opts, Some(d_input))
            }

            /// See [`CpuTrainer::step`]($crate::module::CpuTrainer::step).
            pub fn step(
                &mut self,
                input: &[f32],
                d_temporal: &[f32],
            ) -> Result<$crate::optim::StepMetrics, String> {
                self.core.step(input, d_temporal)
            }

            /// See [`CpuTrainer::set_step_opts`]($crate::module::CpuTrainer::set_step_opts).
            pub fn set_step_opts(&mut self, opts: $crate::optim::BackwardOpts) -> Result<(), String> {
                self.core.set_step_opts(opts)
            }

            /// See [`CpuTrainer::step_opts`]($crate::module::CpuTrainer::step_opts).
            pub fn step_opts(&self) -> $crate::optim::BackwardOpts {
                self.core.step_opts()
            }

            /// See [`CpuTrainer::accumulate`]($crate::module::CpuTrainer::accumulate).
            pub fn accumulate(
                &mut self,
                input: &[f32],
                d_temporal: &[f32],
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                self.core.accumulate(input, d_temporal)
            }

            /// See [`CpuTrainer::apply_window`]($crate::module::CpuTrainer::apply_window).
            pub fn apply_window(&mut self) -> Result<$crate::optim::StepMetrics, String> {
                self.core.apply_window()
            }

            /// See [`CpuTrainer::apply_step_full`]($crate::module::CpuTrainer::apply_step_full).
            pub fn apply_step_full(
                &mut self,
                clip_max_norm: Option<f32>,
                step_skip_above: Option<f32>,
                control_clip_max_norm: Option<f32>,
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                self.core
                    .apply_step_full(clip_max_norm, step_skip_above, control_clip_max_norm)
            }

            /// The optimizer tail with the global clip `clip_max_norm` alone.
            pub fn apply_step(
                &mut self,
                clip_max_norm: Option<f32>,
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                self.core.apply_step_full(clip_max_norm, None, None)
            }

            /// See [`CpuTrainer::download_grad_arena`]($crate::module::CpuTrainer::download_grad_arena).
            pub fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
                self.core.download_grad_arena()
            }

            /// See [`CpuTrainer::upload_grad_arena`]($crate::module::CpuTrainer::upload_grad_arena).
            pub fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
                self.core.upload_grad_arena(values)
            }

            /// Zero every sequence's carried recurrence.
            pub fn reset_state(&mut self) -> Result<(), String> {
                self.core.reset_state();
                Ok(())
            }

            /// See [`CpuTrainer::set_varlen`]($crate::module::CpuTrainer::set_varlen).
            pub fn set_varlen(
                &mut self,
                packing: Option<&$crate::varlen::Varlen>,
            ) -> Result<(), String> {
                self.core.set_varlen(packing)
            }

            /// The AdamW learning rate of the next steps.
            pub fn set_lr(&mut self, lr: f32) -> Result<(), String> {
                self.core.set_lr(lr);
                Ok(())
            }

            pub fn lr(&self) -> f32 {
                self.core.lr()
            }

            pub fn batch(&self) -> usize {
                self.core.batch()
            }

            pub fn seq_len(&self) -> usize {
                self.core.seq_len()
            }

            pub fn input_dim(&self) -> usize {
                self.core.input_dim()
            }

            pub fn d_model(&self) -> usize {
                self.core.d_model()
            }

            /// See [`CpuTrainer::dtype`]($crate::module::CpuTrainer::dtype).
            pub fn dtype(&self) -> $crate::WeightDtype {
                self.core.dtype()
            }

            /// See [`CpuTrainer::scaler_state`]($crate::module::CpuTrainer::scaler_state).
            pub fn scaler_state(&self) -> Option<(f32, u32)> {
                self.core.scaler_state()
            }

            /// See [`CpuTrainer::load_scaler_state`]($crate::module::CpuTrainer::load_scaler_state).
            pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
                $crate::optim::DynamicLossScaler::check_scale(scale)?;
                self.core.load_scaler_state(scale, growth_tracker)
            }

            /// See [`CpuTrainer::optimizer_state`]($crate::module::CpuTrainer::optimizer_state).
            pub fn optimizer_state(&self) -> Result<$crate::optim::AdamWStateBlob, String> {
                Ok(self.core.optimizer_state())
            }

            /// See [`CpuTrainer::load_optimizer_state`]($crate::module::CpuTrainer::load_optimizer_state).
            pub fn load_optimizer_state(
                &mut self,
                blob: &$crate::optim::AdamWStateBlob,
            ) -> Result<(), String> {
                self.core.load_optimizer_state(blob)
            }

            /// See [`CpuTrainer::trainer_state`]($crate::module::CpuTrainer::trainer_state).
            pub fn trainer_state(&self) -> Result<$crate::optim::TrainerState, String> {
                Ok(self.core.trainer_state())
            }

            /// See [`CpuTrainer::load_trainer_state`]($crate::module::CpuTrainer::load_trainer_state).
            pub fn load_trainer_state(
                &mut self,
                state: &$crate::optim::TrainerState,
            ) -> Result<(), String> {
                self.core.load_trainer_state(state)
            }
        }

        impl $crate::module::CpuTrainer for $name {
            fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
                <$name>::forward(self, input, temporal_out)
            }
            fn backward_step(
                &mut self,
                d_temporal: &[f32],
                opts: $crate::optim::BackwardOpts,
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                <$name>::backward_step(self, d_temporal, opts)
            }
            fn backward_step_input_grad(
                &mut self,
                d_temporal: &[f32],
                opts: $crate::optim::BackwardOpts,
                d_input: &mut [f32],
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                <$name>::backward_step_input_grad(self, d_temporal, opts, d_input)
            }
            fn step(
                &mut self,
                input: &[f32],
                d_temporal: &[f32],
            ) -> Result<$crate::optim::StepMetrics, String> {
                <$name>::step(self, input, d_temporal)
            }
            fn set_step_opts(&mut self, opts: $crate::optim::BackwardOpts) -> Result<(), String> {
                <$name>::set_step_opts(self, opts)
            }
            fn step_opts(&self) -> $crate::optim::BackwardOpts {
                <$name>::step_opts(self)
            }
            fn accumulate(
                &mut self,
                input: &[f32],
                d_temporal: &[f32],
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                <$name>::accumulate(self, input, d_temporal)
            }
            fn apply_window(&mut self) -> Result<$crate::optim::StepMetrics, String> {
                <$name>::apply_window(self)
            }
            fn apply_step_full(
                &mut self,
                clip_max_norm: Option<f32>,
                step_skip_above: Option<f32>,
                control_clip_max_norm: Option<f32>,
            ) -> Result<$crate::optim::BackwardMetrics, String> {
                <$name>::apply_step_full(self, clip_max_norm, step_skip_above, control_clip_max_norm)
            }
            fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
                <$name>::download_grad_arena(self)
            }
            fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
                <$name>::upload_grad_arena(self, values)
            }
            fn reset_state(&mut self) -> Result<(), String> {
                <$name>::reset_state(self)
            }
            fn set_varlen(
                &mut self,
                packing: Option<&$crate::varlen::Varlen>,
            ) -> Result<(), String> {
                <$name>::set_varlen(self, packing)
            }
            fn set_lr(&mut self, lr: f32) -> Result<(), String> {
                <$name>::set_lr(self, lr)
            }
            fn lr(&self) -> f32 {
                <$name>::lr(self)
            }
            fn batch(&self) -> usize {
                <$name>::batch(self)
            }
            fn seq_len(&self) -> usize {
                <$name>::seq_len(self)
            }
            fn input_dim(&self) -> usize {
                <$name>::input_dim(self)
            }
            fn d_model(&self) -> usize {
                <$name>::d_model(self)
            }
            fn dtype(&self) -> $crate::WeightDtype {
                <$name>::dtype(self)
            }
            fn scaler_state(&self) -> Option<(f32, u32)> {
                <$name>::scaler_state(self)
            }
            fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
                <$name>::load_scaler_state(self, scale, growth_tracker)
            }
            fn optimizer_state(&self) -> Result<$crate::optim::AdamWStateBlob, String> {
                <$name>::optimizer_state(self)
            }
            fn load_optimizer_state(
                &mut self,
                blob: &$crate::optim::AdamWStateBlob,
            ) -> Result<(), String> {
                <$name>::load_optimizer_state(self, blob)
            }
            fn trainer_state(&self) -> Result<$crate::optim::TrainerState, String> {
                <$name>::trainer_state(self)
            }
            fn load_trainer_state(&mut self, state: &$crate::optim::TrainerState) -> Result<(), String> {
                <$name>::load_trainer_state(self, state)
            }
        }
    };
}
pub(crate) use cpu_trainer_type;

/// A host trainer of any family, chosen at run time, behind the shared
/// surface.
pub enum AnyCpuTrainer {
    M1(Box<crate::mamba1::cpu::trainer::MambaCpuTrainer>),
    M2(Box<crate::mamba2::cpu::trainer::Mamba2CpuTrainer>),
    M3(Box<crate::mamba3_siso::cpu::trainer::Mamba3CpuTrainer>),
    M3Mimo(Box<crate::mamba3_mimo::cpu::trainer::Mamba3MimoCpuTrainer>),
}

impl AnyCpuTrainer {
    /// The family of the trainer held.
    pub fn family(&self) -> ModelFamily {
        match self {
            Self::M1(_) => ModelFamily::Mamba1,
            Self::M2(_) => ModelFamily::Mamba2,
            Self::M3(_) => ModelFamily::Mamba3,
            Self::M3Mimo(_) => ModelFamily::Mamba3Mimo,
        }
    }

    fn inner(&self) -> &dyn CpuTrainer {
        match self {
            Self::M1(t) => t.as_ref(),
            Self::M2(t) => t.as_ref(),
            Self::M3(t) => t.as_ref(),
            Self::M3Mimo(t) => t.as_ref(),
        }
    }

    fn inner_mut(&mut self) -> &mut dyn CpuTrainer {
        match self {
            Self::M1(t) => t.as_mut(),
            Self::M2(t) => t.as_mut(),
            Self::M3(t) => t.as_mut(),
            Self::M3Mimo(t) => t.as_mut(),
        }
    }
}

impl From<crate::mamba1::cpu::trainer::MambaCpuTrainer> for AnyCpuTrainer {
    fn from(t: crate::mamba1::cpu::trainer::MambaCpuTrainer) -> Self {
        Self::M1(Box::new(t))
    }
}

impl From<crate::mamba2::cpu::trainer::Mamba2CpuTrainer> for AnyCpuTrainer {
    fn from(t: crate::mamba2::cpu::trainer::Mamba2CpuTrainer) -> Self {
        Self::M2(Box::new(t))
    }
}

impl From<crate::mamba3_siso::cpu::trainer::Mamba3CpuTrainer> for AnyCpuTrainer {
    fn from(t: crate::mamba3_siso::cpu::trainer::Mamba3CpuTrainer) -> Self {
        Self::M3(Box::new(t))
    }
}

impl From<crate::mamba3_mimo::cpu::trainer::Mamba3MimoCpuTrainer> for AnyCpuTrainer {
    fn from(t: crate::mamba3_mimo::cpu::trainer::Mamba3MimoCpuTrainer) -> Self {
        Self::M3Mimo(Box::new(t))
    }
}

impl CpuTrainer for AnyCpuTrainer {
    fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
        self.inner_mut().forward(input, temporal_out)
    }
    fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut().backward_step(d_temporal, opts)
    }
    fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut()
            .backward_step_input_grad(d_temporal, opts, d_input)
    }
    fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        self.inner_mut().step(input, d_temporal)
    }
    fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        self.inner_mut().set_step_opts(opts)
    }
    fn step_opts(&self) -> BackwardOpts {
        self.inner().step_opts()
    }
    fn accumulate(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<BackwardMetrics, String> {
        self.inner_mut().accumulate(input, d_temporal)
    }
    fn apply_window(&mut self) -> Result<StepMetrics, String> {
        self.inner_mut().apply_window()
    }
    fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        self.inner_mut()
            .apply_step_full(clip_max_norm, step_skip_above, control_clip_max_norm)
    }
    fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
        self.inner().download_grad_arena()
    }
    fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
        self.inner_mut().upload_grad_arena(values)
    }
    fn reset_state(&mut self) -> Result<(), String> {
        self.inner_mut().reset_state()
    }
    fn set_varlen(&mut self, packing: Option<&Varlen>) -> Result<(), String> {
        self.inner_mut().set_varlen(packing)
    }
    fn set_lr(&mut self, lr: f32) -> Result<(), String> {
        self.inner_mut().set_lr(lr)
    }
    fn lr(&self) -> f32 {
        self.inner().lr()
    }
    fn batch(&self) -> usize {
        self.inner().batch()
    }
    fn seq_len(&self) -> usize {
        self.inner().seq_len()
    }
    fn input_dim(&self) -> usize {
        self.inner().input_dim()
    }
    fn d_model(&self) -> usize {
        self.inner().d_model()
    }
    fn dtype(&self) -> crate::WeightDtype {
        self.inner().dtype()
    }
    fn scaler_state(&self) -> Option<(f32, u32)> {
        self.inner().scaler_state()
    }
    fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        self.inner_mut().load_scaler_state(scale, growth_tracker)
    }
    fn optimizer_state(&self) -> Result<AdamWStateBlob, String> {
        self.inner().optimizer_state()
    }
    fn load_optimizer_state(&mut self, blob: &AdamWStateBlob) -> Result<(), String> {
        self.inner_mut().load_optimizer_state(blob)
    }
    fn trainer_state(&self) -> Result<TrainerState, String> {
        self.inner().trainer_state()
    }
    fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        self.inner_mut().load_trainer_state(state)
    }
}
