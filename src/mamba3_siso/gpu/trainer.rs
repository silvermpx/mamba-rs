//! High-level Mamba-3 training API.
//!
//! M3 analogue of [`crate::mamba_ssm::gpu::trainer::MambaTrainer`]. Owns
//! mixed-precision weights + grads + Adam + acts + scratch + state + graph.
//!
//! Supports f32, bf16, and f16 (with dynamic loss scaler), including CUDA
//! Graph capture for all three. See the M1 trainer for the
//! precision-support rationale.

use cudarc::driver::PushKernelArg;

use crate::mamba_ssm::gpu::adamw::{
    AdamWBiasFactors, AdamWMultiPlan, GpuAdamW, build_multi_plan, m3_specs, m3_specs_mixed,
    step_multi,
};
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::grad_clip::{
    GradRegionGeom, alloc_partials, clip_grads_device, clip_region_device, scale_grads,
};
use crate::mamba_ssm::gpu::graph_capture::capture_into_graph;
use crate::mamba_ssm::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba_ssm::gpu::trainer::StepMetrics;
pub use crate::mamba_ssm::gpu::trainer::{BackwardMetrics, BackwardOpts, TrainSessionCfg};
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::gpu::backward::gpu_backward_mamba3_backbone;
use crate::mamba3_siso::gpu::backward_mixed::gpu_backward_mamba3_backbone_mixed;
use crate::mamba3_siso::gpu::forward::gpu_forward_mamba3_backbone;
use crate::mamba3_siso::gpu::forward_mixed::{
    GpuMamba3BackboneMixedActs, GpuMamba3MixedScratch, gpu_forward_mamba3_backbone_mixed,
};
use crate::mamba3_siso::gpu::kernels::Mamba3Kernels;
use crate::mamba3_siso::gpu::state::{
    GpuMamba3BackboneActs, GpuMamba3Dims, GpuMamba3Scratch, GpuMamba3StateBufs, M3Exec,
};
use crate::mamba3_siso::gpu::training_graph::{
    GpuMamba3F32TrainingStepGraph, GpuMamba3TrainingStepGraph, Mamba3F32Capture, Mamba3F32Replay,
    Mamba3MixedCapture, Mamba3MixedReplay,
};
use crate::mamba3_siso::gpu::weights::{GpuMamba3Grads, GpuMamba3Weights};
use crate::mamba3_siso::gpu::weights_mixed_train::GpuMamba3TrainMixedWeights;
use crate::mamba3_siso::weights::Mamba3Weights;

/// CPU-side snapshot of the four carried Mamba-3 recurrences (SSM, K,
/// V, RoPE angle) for TBPTT-style window handoff on the sequential
/// path and checkpointed resume. The chunked training path runs
/// stateless windows and zeroes these each forward, so the blob is
/// meaningful for sequential-scan runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Mamba3RecurrentStateBlob {
    pub ssm: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub angle: Vec<f32>,
}

/// Internal precision-dispatch enum (mirrors `inference::M3BackboneEngine`).
enum Trainer3Inner {
    F32(Box<Mamba3TrainerF32>),
    Mixed(Box<Mamba3TrainerMixed>),
}

/// High-level Mamba-3 training wrapper. Same shape as
/// [`crate::mamba3_siso::gpu::inference::GpuMamba3Backbone`]: one public
/// struct, one method per operation, dtype dispatch happens internally.
pub struct Mamba3Trainer {
    inner: Trainer3Inner,
}

impl Mamba3Trainer {
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_full(
            gpu_ordinal,
            cpu_weights,
            cfg,
            TrainSessionCfg {
                input_dim,
                batch,
                seq_len,
                lr: 1e-3,
                weight_decay: 1e-2,
            },
            dtype,
        )
    }

    pub fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        // hoisted here so BOTH branches get it -
        // the f32 branch never validated, un-guarding the shfl width and
        // n_angles preconditions its kernels rely on.
        cfg.validate()?;
        crate::mamba_ssm::gpu::launch::validate_kernel_arg_capacity(
            session.batch,
            session.seq_len,
            cfg.d_inner(),
            cfg.d_state,
        )?;
        let inner = match dtype {
            WeightDtype::F32 => {
                // The f32 backbone runs the input projection GEMM
                // unconditionally; an empty weight is a zero-size device
                // allocation whose null pointer only surfaces later as
                // CUDA_ERROR_ILLEGAL_ADDRESS inside a GEMM kernel. The
                // empty-means-identity convention belongs to the mixed
                // trainer — refuse it here while the cause is nameable.
                if cpu_weights.input_proj_w.is_empty() {
                    return Err("f32 M3 trainer requires an explicit input projection: \
                         pass an identity matrix for a pass-through (the \
                         empty-input_proj identity convention is mixed-only)"
                        .into());
                }
                Trainer3Inner::F32(Box::new(Mamba3TrainerF32::new_full(
                    gpu_ordinal,
                    cpu_weights,
                    cfg,
                    session,
                )?))
            }
            WeightDtype::Bf16 | WeightDtype::F16 => Trainer3Inner::Mixed(Box::new(
                Mamba3TrainerMixed::new_full(gpu_ordinal, cpu_weights, cfg, session, dtype)?,
            )),
        };
        Ok(Self { inner })
    }

    /// Weight storage dtype the trainer was constructed with.
    pub fn dtype(&self) -> WeightDtype {
        match &self.inner {
            Trainer3Inner::F32(_) => WeightDtype::F32,
            Trainer3Inner::Mixed(t) => t.dtype,
        }
    }

    /// Batch dimension fixed at construction.
    pub fn batch(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.batch,
            Trainer3Inner::Mixed(t) => t.dims.batch,
        }
    }

    /// Sequence length fixed at construction.
    pub fn seq_len(&self) -> usize {
        match &self.inner {
            Trainer3Inner::F32(t) => t.dims.seq_len,
            Trainer3Inner::Mixed(t) => t.dims.seq_len,
        }
    }

    /// CUDA context the trainer runs on.
    pub fn ctx(&self) -> &GpuCtx {
        match &self.inner {
            Trainer3Inner::F32(t) => &t.ctx,
            Trainer3Inner::Mixed(t) => &t.ctx,
        }
    }

    /// `true` once [`Self::capture_graph`] has been called.
    pub fn has_graph(&self) -> bool {
        match &self.inner {
            Trainer3Inner::F32(t) => t.graph.is_some(),
            Trainer3Inner::Mixed(t) => t.has_graph(),
        }
    }

    /// Set the AdamW learning rate for subsequent EAGER steps. Errs while a
    /// captured graph exists — the lr is baked by value into the captured
    /// AdamW kernel; a field write would silently not apply under replay.
    /// Mirrors `MambaTrainer::set_lr`.
    pub fn set_lr(&mut self, lr: f32) -> Result<(), String> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(format!("set_lr: invalid learning rate {lr}"));
        }
        // lr rides the 3-element device bias buffer ({bc1, bc2, lr})
        // and is re-uploaded before every step — a schedule now works
        // under a captured graph (the old per-tensor kernels baked lr
        // by value at capture; the fused kernel reads the buffer).
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.adam.lr = lr,
            Trainer3Inner::Mixed(t) => t.adam.lr = lr,
        }
        Ok(())
    }

    /// Current AdamW learning rate.
    pub fn lr(&self) -> f32 {
        match &self.inner {
            Trainer3Inner::F32(t) => t.adam.lr,
            Trainer3Inner::Mixed(t) => t.adam.lr,
        }
    }

    /// Download the full optimizer state (Adam moments, step counter,
    /// update hyperparameters) for a bit-continuous resume. Checkpoints
    /// that carry only weights silently re-warm Adam from zero on load —
    /// the resumed run then provably diverges from the unbroken one.
    pub fn optimizer_state(&self) -> Result<crate::mamba_ssm::gpu::adamw::AdamWStateBlob, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.adam.export_state(&t.ctx.stream),
            Trainer3Inner::Mixed(t) => t.adam.export_state(&t.ctx.stream),
        }
    }

    /// Upload a previously exported optimizer state. Errs while a
    /// captured graph exists — the decay coefficient and no-decay
    /// grouping the blob adopts are baked by value into the captured
    /// AdamW launches, so the load would silently not apply under
    /// replay. Drop the graph first, load, then re-capture. The learning
    /// rate is not part of the blob; re-apply the schedule afterwards.
    pub fn load_optimizer_state(
        &mut self,
        blob: &crate::mamba_ssm::gpu::adamw::AdamWStateBlob,
    ) -> Result<(), String> {
        if self.has_graph() {
            return Err(
                "load_optimizer_state under a captured graph: the blob's decay \
                 hyperparameters are baked by value into the captured AdamW launches — \
                 drop_graph() first, then load, then re-capture"
                    .into(),
            );
        }
        match &mut self.inner {
            Trainer3Inner::F32(t) => {
                t.adam.import_state(&t.ctx.stream, blob)?;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m3_specs(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
                Ok(())
            }
            Trainer3Inner::Mixed(t) => {
                t.adam.import_state(&t.ctx.stream, blob)?;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m3_specs_mixed(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
                Ok(())
            }
        }
    }

    /// Data-parallel applying backward: local backward with the window
    /// left open, cross-rank SUM of the flat gradient arena, the mean
    /// scale, then the optimizer tail. With a single-process world this
    /// is byte-identical to a plain `backward_step`. Micro-batch
    /// (accumulate-only) calls stay purely local — the reduction
    /// happens once per optimizer step, on the window-closing call.
    pub fn backward_step_dist(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        dist: &crate::dist::DistContext,
    ) -> Result<BackwardMetrics, String> {
        let world = dist.world_size();
        if world == 1 || opts.accumulate_only {
            return self.backward_step(d_temporal, opts);
        }
        let clip = opts.clip_max_norm;
        if self.dtype() == WeightDtype::F16 {
            return Err("f16 multi-GPU training is not supported yet: the split \
                 accumulate/reduce/apply path cannot unscale the f16 \
                 loss-scaled gradients around the cross-rank reduce — \
                 use bf16 or f32 for data-parallel training"
                .into());
        }

        let m = self.backward_step(
            d_temporal,
            BackwardOpts::default().with_accumulate_only(true),
        )?;
        debug_assert!(!m.optimizer_stepped);
        let stream = self.ctx().stream.clone();
        dist.all_reduce_grad_sum(self.grad_arena(), &stream)?;
        // Mean = sum then multiply: for power-of-two worlds the scale
        // only moves exponents (exact); other sizes still get one
        // deterministic per-element rounding.
        if self.grad_arena().len() > i32::MAX as usize {
            return Err(format!(
                "gradient arena has {} elements — beyond the i32 kernel ABI \
                 of scale_grads; the mean scale would silently truncate",
                self.grad_arena().len()
            ));
        }
        let inv_w = 1.0f32 / world as f32;
        match &mut self.inner {
            Trainer3Inner::F32(t) => scale_grads(&t.ctx, &mut t.grads.flat, inv_w)?,
            Trainer3Inner::Mixed(t) => scale_grads(&t.ctx, &mut t.grads.flat, inv_w)?,
        }
        self.apply_step_full(clip, opts.step_skip_above, opts.control_clip_max_norm)
    }

    /// Borrow the flat f32 gradient arena — the single contiguous buffer
    /// every parameter's gradient accumulates into. A distributed
    /// reducer sums this buffer across ranks between the window-closing
    /// `backward_step(accumulate_only = true)` and [`Self::apply_step`];
    /// single-process training never needs it.
    ///
    /// Contract: mutate the CONTENTS only (upload / in-place collective).
    /// The per-tensor gradient views cache this allocation's device
    /// pointer, so replacing or reallocating the buffer itself would
    /// leave them dangling.
    pub fn grad_arena(&mut self) -> &mut GpuBuffer {
        match &mut self.inner {
            Trainer3Inner::F32(t) => &mut t.grads.flat,
            Trainer3Inner::Mixed(t) => &mut t.grads.flat,
        }
    }

    /// Run ONLY the optimizer tail (optional clip, AdamW, window close)
    /// over the already-accumulated gradient. Together with
    /// `backward_step(accumulate_only = true)` this splits the applying
    /// backward in two, so a gradient reducer can run in between; the
    /// pair is bit-identical to a single applying `backward_step`.
    /// Requires an open accumulation window. Not defined for f16 — the
    /// loss-scaler protocol owns that tail end to end.
    pub fn apply_step(&mut self, clip_max_norm: Option<f32>) -> Result<BackwardMetrics, String> {
        self.apply_step_with(clip_max_norm, None)
    }

    /// [`Self::apply_step`] with the window-discarding spike skip: when
    /// `step_skip_above` is `Some(t)` and the pre-clip norm exceeds `t`,
    /// the optimizer step is skipped and the window discarded (see
    /// [`BackwardOpts::step_skip_above`]).
    pub fn apply_step_with(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        self.apply_step_full(clip_max_norm, step_skip_above, None)
    }

    /// [`Self::apply_step_with`] plus the Mamba-3 control-channel clip
    /// (see [`BackwardOpts::control_clip_max_norm`]).
    pub fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        if matches!(self.dtype(), WeightDtype::F16) {
            return Err(
                "apply_step is not defined for f16: the loss-scaler protocol (overflow \
                 check, conditional unscale, scaler update, step rollback) owns the \
                 optimizer tail — use the applying backward_step directly"
                    .into(),
            );
        }
        match &mut self.inner {
            Trainer3Inner::F32(t) => {
                if !t.grads_dirty {
                    return Err("apply_step without an open accumulation window — run \
                         backward_step(accumulate_only = true) first"
                        .into());
                }
                t.apply_step_inner(clip_max_norm, step_skip_above, control_clip_max_norm)
            }
            Trainer3Inner::Mixed(t) => {
                if !t.grads_dirty {
                    return Err("apply_step without an open accumulation window — run \
                         backward_step(accumulate_only = true) first"
                        .into());
                }
                t.apply_step_inner(clip_max_norm, step_skip_above, control_clip_max_norm)
            }
        }
    }

    /// Toggle the reference-faithful AdamW no-decay parameter groups
    /// (dt bias / `d_param` / every norm scale get `weight_decay = 0`).
    /// Default OFF preserves the historical behavior bit-for-bit. Errs
    /// while a captured graph exists (the decay coefficient is baked by
    /// value into the captured per-tensor launches). Mirrors
    /// `MambaTrainer::set_reference_no_decay`.
    pub fn set_reference_no_decay(&mut self, on: bool) -> Result<(), String> {
        if self.has_graph() {
            return Err(
                "set_reference_no_decay under a captured graph: the decay coefficient \
                 is baked by value into the captured AdamW launches — drop_graph() \
                 first, then toggle, then re-capture"
                    .into(),
            );
        }
        match &mut self.inner {
            Trainer3Inner::F32(t) => {
                t.adam.reference_no_decay = on;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m3_specs(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
            }
            Trainer3Inner::Mixed(t) => {
                t.adam.reference_no_decay = on;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m3_specs_mixed(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
            }
        }
        Ok(())
    }

    /// Drop any captured step graph so `drop_graph -> set_lr ->
    /// capture_graph` is expressible. Mirrors `MambaTrainer::drop_graph`.
    pub fn drop_graph(&mut self) {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.graph = None,
            Trainer3Inner::Mixed(t) => {
                t.graph = None;
                t.graph_f16 = None;
            }
        }
    }

    /// Reset the recurrent SSM, K, V, and angle states to zero.
    pub fn reset_state(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.reset_state(),
            Trainer3Inner::Mixed(t) => t.reset_state(),
        }
    }

    /// Download the four carried recurrences (SSM, K, V, angle) — the
    /// sequential-path TBPTT handoff counterpart of
    /// [`Self::optimizer_state`].
    pub fn recurrent_state(&self) -> Result<Mamba3RecurrentStateBlob, String> {
        let (stream, ssm, k, v, angle) = match &self.inner {
            Trainer3Inner::F32(t) => (
                &t.ctx.stream,
                &t.ssm_states,
                &t.k_states,
                &t.v_states,
                &t.angle_states,
            ),
            Trainer3Inner::Mixed(t) => (
                &t.ctx.stream,
                &t.ssm_states,
                &t.k_states,
                &t.v_states,
                &t.angle_states,
            ),
        };
        Ok(Mamba3RecurrentStateBlob {
            ssm: ssm.to_cpu(stream)?,
            k: k.to_cpu(stream)?,
            v: v.to_cpu(stream)?,
            angle: angle.to_cpu(stream)?,
        })
    }

    /// Upload a previously exported recurrence. Errs on any length
    /// mismatch — the blob belongs to a different shape.
    pub fn load_recurrent_state(&mut self, blob: &Mamba3RecurrentStateBlob) -> Result<(), String> {
        let (stream, ssm, k, v, angle) = match &mut self.inner {
            Trainer3Inner::F32(t) => (
                t.ctx.stream.clone(),
                &mut t.ssm_states,
                &mut t.k_states,
                &mut t.v_states,
                &mut t.angle_states,
            ),
            Trainer3Inner::Mixed(t) => (
                t.ctx.stream.clone(),
                &mut t.ssm_states,
                &mut t.k_states,
                &mut t.v_states,
                &mut t.angle_states,
            ),
        };
        if blob.ssm.len() != ssm.len()
            || blob.k.len() != k.len()
            || blob.v.len() != v.len()
            || blob.angle.len() != angle.len()
        {
            return Err(format!(
                "Mamba3 recurrent state mismatch: blob {}/{}/{}/{} vs state                  {}/{}/{}/{} — the blob belongs to a different shape",
                blob.ssm.len(),
                blob.k.len(),
                blob.v.len(),
                blob.angle.len(),
                ssm.len(),
                k.len(),
                v.len(),
                angle.len()
            ));
        }
        ssm.upload(&stream, &blob.ssm)?;
        k.upload(&stream, &blob.k)?;
        v.upload(&stream, &blob.v)?;
        angle.upload(&stream, &blob.angle)?;
        Ok(())
    }

    /// Record the full training step into a CUDA Graph. Run at least one
    /// warmup [`Self::step`] first. See [`Self::capture_graph`]
    /// for the pointer-stability contract — same rules apply here.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.capture_graph(),
            Trainer3Inner::Mixed(t) => t.capture_graph(),
        }
    }

    /// Run one training step on `(input, d_temporal)`. Shapes match
    /// [`Self::step`]. Returns `StepMetrics`.
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.step(input, d_temporal),
            Trainer3Inner::Mixed(t) => t.step(input, d_temporal),
        }
    }

    /// Eager forward half of the split step — M3 mirror of
    /// [`crate::mamba_ssm::gpu::trainer::MambaTrainer::forward`]: runs the
    /// training forward and writes the FULL `batch * seq_len * d_model`
    /// POST-norm_f temporal output into caller-owned `temporal_out` (f32 on
    /// all dtypes). Stream-synchronized on return. Advances the recurrent
    /// state; always eager (see the M1 doc for the graph rationale — the
    /// fused [`Self::step`] composes the same eager bodies).
    pub fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.forward_split(input, temporal_out),
            Trainer3Inner::Mixed(t) => t.forward_split(input, temporal_out),
        }
    }

    /// Backward + optimizer half of the split step — M3 mirror of
    /// [`crate::mamba_ssm::gpu::trainer::MambaTrainer::backward_step`].
    /// `d_temporal` is the gradient w.r.t. the IMMEDIATELY PRECEDING
    /// [`Self::forward`]'s temporal output. Errs when no forward is pending.
    pub fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.backward_split(d_temporal, opts),
            Trainer3Inner::Mixed(t) => t.backward_split(d_temporal, opts),
        }
    }

    /// Download f32 master weights for checkpointing.
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.snapshot_master(),
            Trainer3Inner::Mixed(t) => t.snapshot_master(),
        }
    }

    /// Serialize the dynamic loss scaler state for checkpoint resume.
    /// Returns `Some((scale, growth_tracker))` only for f16 training where
    /// the scaler is active; `None` for bf16 / f32. Mirrors
    /// [`Self::scaler_state`].
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        match &self.inner {
            Trainer3Inner::F32(_) => None,
            Trainer3Inner::Mixed(t) => t.scaler_state(),
        }
    }

    /// Restore the dynamic loss scaler state saved via [`Self::scaler_state`].
    /// No-op for non-f16 trainers.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Trainer3Inner::Mixed(ref mut t) = self.inner {
            t.load_scaler_state(scale, growth_tracker);
        }
    }
}

/// The CONTROL region of the M3 grad arena: the dd_dt/dd_A/trap/angle
/// column stripe of every layer's in_proj gradient plus dt_bias — the
/// only gradients whose magnitude scales with the sequence length. The
/// stripe length is derived as `ip - (3*nh + na)` so B/C sizing changes
/// cannot desync it from the split order.
fn m3_control_region_geom(cfg: &Mamba3Config, input_dim: usize) -> GradRegionGeom {
    let dm = cfg.d_model;
    let di = cfg.d_inner();
    let ds = cfg.d_state;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();
    let na = cfg.num_rope_angles();
    let ctrl_len = 3 * nh + na;
    let ctrl0 = ip - ctrl_len;
    let per_layer = dm + dm * ip + nh + ds + ds + nh * ds + nh * ds + nh + di + di * dm;
    let layer0 = input_dim * dm + dm;
    GradRegionGeom {
        n_layers: cfg.n_layers as i32,
        layer_stride: per_layer as i32,
        a_off: (layer0 + dm + ctrl0) as i32,
        a_rows: dm as i32,
        a_cols: ctrl_len as i32,
        a_row_stride: ip as i32,
        b_off: (layer0 + dm + dm * ip) as i32,
        b_len: nh as i32,
    }
}

pub(crate) struct Mamba3TrainerMixed {
    ctx: GpuCtx,
    m3k: Mamba3Kernels,
    cfg: Mamba3Config,
    dims: GpuMamba3Dims,
    dtype: WeightDtype,

    pub weights: GpuMamba3TrainMixedWeights,
    pub grads: GpuMamba3Grads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    multi_plan: AdamWMultiPlan,

    acts: GpuMamba3BackboneMixedActs,
    f32_scratch: GpuMamba3Scratch,
    mixed_scratch: GpuMamba3MixedScratch,

    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    ssm_states: GpuBuffer,
    k_states: GpuBuffer,
    v_states: GpuBuffer,
    angle_states: GpuBuffer,

    graph: Option<GpuMamba3TrainingStepGraph>,

    /// True between a `forward_split` and the `backward_split` consuming its
    /// saved activations (the split-API staleness interlock).
    split_forward_pending: bool,
    /// True while the grad arena holds accumulated (un-applied) gradients
    /// from `accumulate_only` backward calls — the next backward must NOT
    /// zero the arena.
    grads_dirty: bool,
    /// Fixed-size f64 partials for the deterministic global grad norm.
    clip_partials: GpuByteBuffer,
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,

    // f16 AMP loss scaler (None for bf16). See M1 trainer for the protocol.
    scaler: Option<DynamicLossScaler>,
    overflow_flag: Option<OverflowFlag>,
    d_temporal_scaled: Option<GpuBuffer>,
    /// f16 CUDA Graph (M3 analogue of M1's `graph_f16`).
    graph_f16: Option<cudarc::driver::CudaGraph>,
    /// 1-element device buffer of `1/loss_scale`.
    unscale_factor: Option<UnscaleFactor>,
    /// Pointer-stability snapshots for the f16 graph.
    captured_f16_bias_ptr: u64,
    captured_f16_unscale_ptr: u64,
    captured_f16_overflow_ptr: u64,
    captured_f16_grads_ptr: u64,
    captured_f16_dt_scaled_ptr: u64,
    captured_f16_half_staging_ptr: u64,
    captured_f16_bi_upcast_ptrs: [u64; 3],
    captured_f16_gemm_route: crate::mamba_ssm::gpu::context::GemmRoute,
}

impl Mamba3TrainerMixed {
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        assert!(
            matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16),
            "Mamba3TrainerMixed accepts Bf16 or F16; got {dtype:?}"
        );

        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            arch,
            crate::mamba_ssm::gpu::kernels::state_capacity(cfg.d_state)?,
        )?;

        let weights =
            GpuMamba3TrainMixedWeights::from_cpu(&ctx.stream, cpu_weights, &cfg, input_dim, dtype)?;

        let dims = GpuMamba3Dims {
            batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            rms_norm_eps: cfg.rms_norm_eps,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: cfg.train_use_parallel_scan(),
        };
        dims.validate_index_budget()?;

        let acts = GpuMamba3BackboneMixedActs::new(
            &ctx.stream,
            &cfg,
            batch,
            seq_len,
            input_dim,
            dtype,
            cfg.train_use_parallel_scan(),
        )?;
        let f32_scratch = GpuMamba3Scratch::new(&ctx.stream, &dims)?;
        let mixed_scratch = GpuMamba3MixedScratch::new(&ctx.stream, &cfg, batch, seq_len, dtype)?;

        let bt = batch * seq_len;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;

        let temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, bt * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let ssm_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds)?;
        let k_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds)?;
        let v_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd)?;
        let angle_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na)?;
        let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;
        let multi_plan = build_multi_plan(
            &ctx.stream,
            &adam,
            grads.flat.cached_ptr(),
            &m3_specs_mixed(&weights, &grads),
            adam.reference_no_decay,
            adam.weight_decay,
        )?;

        let (scaler, overflow_flag, d_temporal_scaled, unscale_factor) =
            if matches!(dtype, WeightDtype::F16) {
                let s = DynamicLossScaler::new();
                let f = OverflowFlag::new(&ctx.stream)?;
                let scaled = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
                let u = UnscaleFactor::new(&ctx.stream)?;
                (Some(s), Some(f), Some(scaled), Some(u))
            } else {
                (None, None, None, None)
            };

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        let clip_partials = alloc_partials(&ctx.stream)?;
        let clip_scratch = GpuBuffer::zeros(&ctx.stream, 2)?;
        let initial_gemm_route = ctx.gemm_route();

        Ok(Self {
            ctx,
            m3k,
            cfg,
            dims,
            dtype,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            f32_scratch,
            mixed_scratch,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            graph: None,
            split_forward_pending: false,
            grads_dirty: false,
            clip_partials,
            clip_scratch,
            scaler,
            overflow_flag,
            d_temporal_scaled,
            graph_f16: None,
            unscale_factor,
            captured_f16_bias_ptr: 0,
            captured_f16_unscale_ptr: 0,
            captured_f16_overflow_ptr: 0,
            captured_f16_grads_ptr: 0,
            captured_f16_dt_scaled_ptr: 0,
            captured_f16_half_staging_ptr: 0,
            captured_f16_bi_upcast_ptrs: [0; 3],
            captured_f16_gemm_route: initial_gemm_route,
        })
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some() || self.graph_f16.is_some()
    }

    /// Reset SSM / K / V / angle states to zero (keeps weights untouched).
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    /// Capture the training-step CUDA Graph. Call once after at least one
    /// warmup [`Self::step`].
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if matches!(self.dtype, WeightDtype::F16) {
            return self.capture_graph_f16();
        }
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;

        let g = GpuMamba3TrainingStepGraph::capture(
            &M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            },
            &self.cfg,
            Mamba3MixedCapture {
                train_w: &mut self.weights,
                adam: &self.adam,
                bias: &self.bias,
                multi_plan: &self.multi_plan,
                grads: &mut self.grads,
                acts: &mut self.acts,
                f32_scratch: &mut self.f32_scratch,
                mixed_scratch: &mut self.mixed_scratch,
                temporal_f32: &mut self.temporal,
                mamba_input: &self.mamba_input,
                d_temporal: &mut self.d_temporal,
                states: GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
            },
        )?;
        self.graph = Some(g);
        Ok(())
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );
        if self.grads_dirty {
            return Err(
                "step(): an accumulate_only backward window is open — close it with \
                 backward_step(accumulate_only=false); the fused step zeroes the grad \
                 arena and would silently discard the accumulated gradients"
                    .into(),
            );
        }

        if matches!(self.dtype, WeightDtype::F16) {
            return self.step_f16(input, d_temporal);
        }

        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;

        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;

        let replayed = if let Some(ref g) = self.graph {
            g.replay(
                &self.ctx,
                &Mamba3MixedReplay {
                    train_w: &self.weights,
                    adam: &self.adam,
                    bias: &self.bias,
                    grads: &self.grads,
                    temporal_f32: &self.temporal,
                    mamba_input: &self.mamba_input,
                    d_temporal: &self.d_temporal,
                    ssm_states: &self.ssm_states,
                    k_states: &self.k_states,
                    v_states: &self.v_states,
                    angle_states: &self.angle_states,
                },
            )?;
            true
        } else {
            self.step_eager()?;
            false
        };

        Ok(StepMetrics::plain(step, replayed))
    }

    /// f16 step. See M1 [`crate::mamba_ssm::gpu::trainer::MambaTrainerMixed::step_f16`]
    /// for the full protocol; this is the M3 mirror.
    fn step_f16(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
        self.mamba_input.upload(&self.ctx.stream, input)?;
        // Scale d_temporal on-device — the old path built a scaled Vec<f32>
        // on the host every step (B*T*d_model alloc + traversal).
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        dt_scaled.upload(&self.ctx.stream, d_temporal)?;
        {
            let n = d_temporal.len() as i32;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.scale_grads_f32);
            builder.arg(dt_scaled.inner_mut());
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(crate::mamba_ssm::gpu::launch::grid_1d(d_temporal.len())) }
                .map_err(|e| format!("scale d_temporal (m3 f16): {e:?}"))?;
        }

        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        let prev_step = self.adam.step;
        let (next_step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;

        self.overflow_flag
            .as_mut()
            .expect("f16 overflow flag")
            .zero(&self.ctx.stream)?;

        let (step, overflow, replayed) = if let Some(ref g) = self.graph_f16 {
            assert_eq!(
                self.bias.ptr(),
                self.captured_f16_bias_ptr,
                "M3 f16 graph replay: bias pointer changed since capture"
            );
            assert_eq!(
                self.unscale_factor.as_ref().unwrap().ptr(),
                self.captured_f16_unscale_ptr,
                "M3 f16 graph replay: unscale_factor pointer changed since capture"
            );
            assert_eq!(
                self.overflow_flag
                    .as_ref()
                    .unwrap()
                    .stable_ptr(&self.ctx.stream),
                self.captured_f16_overflow_ptr,
                "M3 f16 graph replay: overflow_flag pointer changed since capture"
            );
            assert_eq!(
                self.grads.flat.cached_ptr(),
                self.captured_f16_grads_ptr,
                "M3 f16 graph replay: grads.flat pointer changed since capture"
            );
            assert_eq!(
                self.d_temporal_scaled.as_ref().unwrap().cached_ptr(),
                self.captured_f16_dt_scaled_ptr,
                "M3 f16 graph replay: d_temporal_scaled pointer changed since capture"
            );
            assert_eq!(
                self.ctx.half_staging_ptr(),
                self.captured_f16_half_staging_ptr,
                "M3 f16 graph replay: half_staging pointer changed since capture \
                 (lazy grow after capture - re-capture or pre-size the larger shape)"
            );
            assert_eq!(
                self.ctx.bi_upcast_scratch_ptrs(),
                self.captured_f16_bi_upcast_ptrs,
                "M3 f16 graph replay: bi_upcast_scratch pointer changed since capture \
                 (a larger typed triad GEMM grew the scratch - re-capture or pre-size the larger shape)"
            );
            assert_eq!(
                self.ctx.gemm_route(),
                self.captured_f16_gemm_route,
                "M3 f16 graph replay: GEMM route changed since capture \
                 (batch_invariant, bi_tensor_cores, fast_gemm, bi_gemm_family) - \
                 the captured kernels cannot follow a route change; re-capture instead"
            );

            g.launch()
                .map_err(|e| format!("M3 f16 graph launch: {e:?}"))?;
            let overflow = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .read(&self.ctx.stream)?
                != 0;
            self.scaler.as_mut().expect("f16 scaler").update(overflow);
            (next_step, overflow, true)
        } else {
            // Same op sequence as the captured graph body (composed from
            // the shared eager phase bodies).
            self.grads.zero(&self.ctx.stream)?;
            self.eager_forward()?;
            self.eager_backward(true)?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &self.grads.flat,
            )?;
            let unscale = self.unscale_factor.as_ref().unwrap();
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                unscale,
            )?;
            self.eager_optimize()?;
            let overflow = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .read(&self.ctx.stream)?
                != 0;
            self.scaler.as_mut().expect("f16 scaler").update(overflow);
            (next_step, overflow, false)
        };

        let final_step = if overflow {
            self.adam.step = prev_step;
            prev_step
        } else {
            step
        };

        Ok(StepMetrics {
            step: final_step,
            graph_replayed: replayed,
            loss_scale: Some(scale),
            overflow_skipped: Some(overflow),
        })
    }

    /// Capture the M3 f16 training step (M3 mirror of the M1 capture).
    fn capture_graph_f16(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;
        let init_unscale = 1.0 / self.scaler.as_ref().expect("f16 scaler").scale();
        self.unscale_factor
            .as_mut()
            .expect("unscale buf")
            .write(&self.ctx.stream, init_unscale)?;
        self.overflow_flag
            .as_mut()
            .expect("overflow flag")
            .zero(&self.ctx.stream)?;
        let dummy = vec![0.0f32; self.d_temporal.len()];
        self.d_temporal_scaled
            .as_mut()
            .expect("dt_scaled")
            .upload(&self.ctx.stream, &dummy)?;

        self.ctx.presize_half_staging_for_train_m3(
            &self.cfg,
            self.dims.batch,
            self.dims.seq_len,
            self.dtype,
        )?;
        self.ctx.presize_bi_upcast_scratch_for_train_m3(
            &self.cfg,
            self.dims.batch,
            self.dims.seq_len,
            self.dims.mamba_input_dim,
            self.dtype,
        )?;

        // Snapshot every device buffer baked into the captured kernels.
        let snap_bias = self.bias.ptr();
        let snap_unscale = self.unscale_factor.as_ref().unwrap().ptr();
        let snap_overflow = self
            .overflow_flag
            .as_ref()
            .unwrap()
            .stable_ptr(&self.ctx.stream);
        let snap_grads = self.grads.flat.cached_ptr();
        let snap_dt_scaled = self.d_temporal_scaled.as_ref().unwrap().cached_ptr();
        let snap_half_staging = self.ctx.half_staging_ptr();
        let snap_bi_upcast = self.ctx.bi_upcast_scratch_ptrs();
        let snap_gemm_route = self.ctx.gemm_route();

        // Capture body: mirrors the eager f16 path 1:1 (composed from the
        // shared eager phase bodies) so numerics match.
        let stream = self.ctx.stream.clone();
        let g = capture_into_graph(&stream, || {
            self.grads.zero(&self.ctx.stream)?;
            self.eager_forward()?;
            self.eager_backward(true)?;
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &self.grads.flat,
            )?;
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                self.unscale_factor.as_ref().unwrap(),
            )?;
            self.eager_optimize()?;
            Ok(())
        })?;
        self.graph_f16 = Some(g);
        self.captured_f16_bias_ptr = snap_bias;
        self.captured_f16_unscale_ptr = snap_unscale;
        self.captured_f16_overflow_ptr = snap_overflow;
        self.captured_f16_grads_ptr = snap_grads;
        self.captured_f16_dt_scaled_ptr = snap_dt_scaled;
        self.captured_f16_half_staging_ptr = snap_half_staging;
        self.captured_f16_bi_upcast_ptrs = snap_bi_upcast;
        self.captured_f16_gemm_route = snap_gemm_route;
        self.ctx.note_graph_capture();
        Ok(())
    }

    /// Eager forward body: run the mixed training forward, writing the f32
    /// post-norm_f output into `self.temporal`. One of the three shared
    /// phase bodies the fused eager step, the f16 paths, and the
    /// forward/backward split all compose (M1 seam mirror).
    fn eager_forward(&mut self) -> Result<(), String> {
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        gpu_forward_mamba3_backbone_mixed(
            &exec,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            GpuMamba3StateBufs {
                ssm: &mut self.ssm_states,
                k: &mut self.k_states,
                v: &mut self.v_states,
                angle: &mut self.angle_states,
            },
            &mut self.mixed_scratch,
        )
    }

    /// Eager backward body: accumulate gradients into the grad arena
    /// (beta=1.0 — zeroing is the caller's responsibility). `scaled` selects
    /// the f16 loss-scaled `d_temporal_scaled` buffer over the plain
    /// `d_temporal` upload buffer.
    fn eager_backward(&mut self, scaled: bool) -> Result<(), String> {
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        let d_temporal = if scaled {
            self.d_temporal_scaled
                .as_mut()
                .expect("eager_backward(scaled): f16 d_temporal_scaled missing")
        } else {
            &mut self.d_temporal
        };
        gpu_backward_mamba3_backbone_mixed(
            &exec,
            d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.f32_scratch,
            &mut self.mixed_scratch,
        )
    }

    /// Eager optimizer tail: AdamW on the f32 master weights (capturable
    /// kernel so graph and eager numerics stay bit-identical), then the
    /// master → compute sync.
    fn eager_optimize(&mut self) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.m3k.adamw_step_multi.get(self.dtype),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
        )?;
        // Bulk typed shadows ride the fused kernel; the walk covers only
        // the f32-stays-f32 tensors.
        self.weights.sync_master_to_compute(&self.ctx)
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        self.eager_forward()?;
        self.eager_backward(false)?;
        self.eager_optimize()
    }

    /// Split forward (see [`Mamba3Trainer::forward`]). The M3 mixed forward
    /// already writes f32 temporal — no upcast staging needed (unlike M1).
    pub(crate) fn forward_split(
        &mut self,
        input: &[f32],
        temporal_out: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(
            input.len(),
            self.mamba_input.len(),
            "input shape mismatch: expected batch*seq_len*input_dim={}, got {}",
            self.mamba_input.len(),
            input.len(),
        );
        assert_eq!(
            temporal_out.len(),
            self.temporal.len(),
            "temporal_out shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.temporal.len(),
            temporal_out.len(),
        );
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.eager_forward()?;
        // Sync AFTER the download enqueue — the D2H is async by
        // contract; only pageable-memory luck made the old order work.
        self.temporal.download(&self.ctx.stream, temporal_out)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("forward_split sync: {e:?}"))?;
        self.split_forward_pending = true;
        Ok(())
    }

    /// Split backward + optimizer (see [`Mamba3Trainer::backward_step`]).
    pub(crate) fn backward_split(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        if !self.split_forward_pending {
            return Err(
                "backward_step() without a pending forward() — the saved activations \
                 are stale or missing; call forward() first"
                    .into(),
            );
        }
        if opts.clip_max_norm.is_some() && opts.accumulate_only {
            return Err(
                "clip_max_norm + accumulate_only is unsupported: the global norm is only \
                 defined over the COMPLETE accumulated gradient — request the clip on the \
                 final (applying) backward_step"
                    .into(),
            );
        }
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.d_temporal.len(),
            d_temporal.len(),
        );

        if matches!(self.dtype, WeightDtype::F16) {
            if opts.accumulate_only {
                return Err(
                    "f16 + accumulate_only is unsupported: the loss-scale freeze window \
                     across micro-batches has no defined semantics"
                        .into(),
                );
            }
            let m = self.backward_split_f16(d_temporal, opts.clip_max_norm)?;
            self.split_forward_pending = false;
            return Ok(m);
        }

        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
        }
        self.eager_backward(false)?;
        self.split_forward_pending = false;

        if opts.accumulate_only {
            self.grads_dirty = true;
            Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm: None,
                loss_scale: None,
                overflow_skipped: None,
            })
        } else {
            self.apply_step_inner(
                opts.clip_max_norm,
                opts.step_skip_above,
                opts.control_clip_max_norm,
            )
        }
    }

    /// Optimizer-only tail of the applying backward: optional clip, bias
    /// factors, fused AdamW over the accumulated gradient, window close.
    /// A separate seam so a gradient reducer can run between the last
    /// backward and the weight update.
    fn apply_step_inner(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        // Control-channel clip FIRST: the T-scaled dt/A/trap/angle
        // gradients get their own bound so a resonant page cannot
        // rescale the whole arena through the global clip below.
        if let Some(cb) = control_clip_max_norm {
            let geom = m3_control_region_geom(&self.cfg, self.dims.mamba_input_dim);
            clip_region_device(
                &self.ctx,
                &mut self.grads.flat,
                &mut self.clip_partials,
                &mut self.clip_scratch,
                geom,
                cb,
            )?;
        }
        let grad_norm = match clip_max_norm {
            Some(c) => Some(self.apply_clip(c)?),
            None => None,
        };
        // Spike skip: a window whose PRE-clip norm exceeds the threshold
        // is discarded whole - no Adam advance, no weight update, arena
        // re-zeroes on the next backward. One pathological window then
        // costs one update instead of a poisoned optimizer state.
        if let (Some(n), Some(thr)) = (grad_norm, step_skip_above)
            && n > thr
        {
            self.grads_dirty = false;
            return Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm,
                loss_scale: None,
                overflow_skipped: None,
            });
        }
        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        self.eager_optimize()?;
        self.grads_dirty = false;
        Ok(BackwardMetrics {
            step,
            optimizer_stepped: true,
            grad_norm,
            loss_scale: None,
            overflow_skipped: None,
        })
    }

    /// f16 split backward: GradScaler protocol minus the forward — scale,
    /// backward, overflow check, conditional unscale + clip + optimize,
    /// scaler update, step rollback. Clip norm is computed AFTER the
    /// unscale (the unscale-then-norm-then-clip ordering law). M1 mirror.
    fn backward_split_f16(
        &mut self,
        d_temporal: &[f32],
        clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
        {
            let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
            dt_scaled.upload(&self.ctx.stream, d_temporal)?;
            let n = d_temporal.len() as i32;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.scale_grads_f32);
            builder.arg(dt_scaled.inner_mut());
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(crate::mamba_ssm::gpu::launch::grid_1d(d_temporal.len())) }
                .map_err(|e| format!("scale d_temporal (m3 f16 split): {e:?}"))?;
        }
        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        let prev_step = self.adam.step;
        let (next_step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        self.overflow_flag
            .as_mut()
            .expect("f16 overflow flag")
            .zero(&self.ctx.stream)?;

        self.grads.zero(&self.ctx.stream)?;
        self.eager_backward(true)?;
        check_inf_nan_gpu(
            &self.ctx,
            &self.ctx.kernels,
            self.overflow_flag.as_mut().unwrap(),
            &self.grads.flat,
        )?;
        let overflow = self
            .overflow_flag
            .as_ref()
            .unwrap()
            .read(&self.ctx.stream)?
            != 0;
        let mut grad_norm = None;
        if !overflow {
            let unscale = self.unscale_factor.as_ref().expect("unscale buf");
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                unscale,
            )?;
            if let Some(c) = clip_max_norm {
                grad_norm = Some(self.apply_clip(c)?);
            }
            self.eager_optimize()?;
        }
        self.scaler.as_mut().expect("f16 scaler").update(overflow);
        let final_step = if overflow {
            self.adam.step = prev_step;
            prev_step
        } else {
            next_step
        };
        Ok(BackwardMetrics {
            step: final_step,
            optimizer_stepped: !overflow,
            grad_norm,
            loss_scale: Some(scale),
            overflow_skipped: Some(overflow),
        })
    }

    /// Compute the deterministic global grad norm, apply the clip
    /// coefficient when needed, and return the PRE-clip norm.
    fn apply_clip(&mut self, max_norm: f32) -> Result<f32, String> {
        // Device-side fold: norm, coefficient and scaling are enqueued
        // back to back, so the host blocks once with the whole sequence
        // already in flight instead of draining the stream between the
        // norm and the scale. Bit-identical to the host path.
        let norm = clip_grads_device(
            &self.ctx,
            &mut self.grads.flat,
            &mut self.clip_partials,
            &mut self.clip_scratch,
            max_norm,
        )?;
        if !norm.is_finite() {
            return Err(format!(
                "clip_max_norm: non-finite global grad norm ({norm})"
            ));
        }
        Ok(norm)
    }

    /// Download master weights to CPU for checkpointing.
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let m = &self.weights.master;
        let input_dim = self.mamba_input.len() / (self.dims.batch * self.dims.seq_len);
        let mut out = Mamba3Weights::zeros(&self.cfg, input_dim);
        out.input_proj_w = m.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = m.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &m.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_bias = g.dt_bias.to_cpu(&self.ctx.stream)?;
            lw.b_norm_weight = g.b_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.c_norm_weight = g.c_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.b_bias = g.b_bias.to_cpu(&self.ctx.stream)?;
            lw.c_bias = g.c_bias.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.norm_gate_weight = g.norm_gate_weight.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
        }
        out.norm_f_weight = m.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }

    /// Serialize dynamic loss scaler state. `None` when scaler is disabled
    /// (bf16). Mirrors `MambaTrainerMixed::scaler_state`.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(|s| s.state())
    }

    /// Restore scaler state from a prior `scaler_state()`. No-op when the
    /// scaler is disabled (bf16). Also rewrites the on-device unscale
    /// factor so the next f16 step uses the restored scale.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) {
        if let Some(ref mut s) = self.scaler {
            s.load_state(scale, growth_tracker);
            if let Some(ref mut uf) = self.unscale_factor {
                let unscale = 1.0 / s.scale();
                let _ = uf.write(&self.ctx.stream, unscale);
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 M3 training wrapper.
// ════════════════════════════════════════════════════════════════════════

pub(crate) struct Mamba3TrainerF32 {
    pub ctx: GpuCtx,
    pub m3k: Mamba3Kernels,
    pub cfg: Mamba3Config,
    pub dims: GpuMamba3Dims,
    pub weights: GpuMamba3Weights,
    pub grads: GpuMamba3Grads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    multi_plan: AdamWMultiPlan,
    acts: GpuMamba3BackboneActs,
    scratch: GpuMamba3Scratch,
    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    ssm_states: GpuBuffer,
    k_states: GpuBuffer,
    v_states: GpuBuffer,
    angle_states: GpuBuffer,
    graph: Option<GpuMamba3F32TrainingStepGraph>,
    /// True between a `forward_split` and the `backward_split` consuming its
    /// saved activations (the split-API staleness interlock).
    split_forward_pending: bool,
    /// True while the grad arena holds accumulated (un-applied) gradients
    /// from `accumulate_only` backward calls.
    grads_dirty: bool,
    /// Fixed-size f64 partials for the deterministic global grad norm.
    clip_partials: GpuByteBuffer,
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,
}

impl Mamba3TrainerF32 {
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
    ) -> Result<Self, String> {
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        let device = GpuDevice::new(gpu_ordinal)?;
        let ctx = GpuCtx::new(&device)?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            arch,
            crate::mamba_ssm::gpu::kernels::state_capacity(cfg.d_state)?,
        )?;

        let weights = GpuMamba3Weights::from_cpu(&ctx.stream, cpu_weights, &cfg, input_dim)?;

        let dims = GpuMamba3Dims {
            batch,
            d_model: cfg.d_model,
            d_inner: cfg.d_inner(),
            d_state: cfg.d_state,
            nheads: cfg.nheads(),
            headdim: cfg.headdim,
            ngroups: cfg.ngroups,
            in_proj_dim: cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers: cfg.n_layers,
            n_angles: cfg.num_rope_angles(),
            a_floor: cfg.a_floor,
            rms_norm_eps: cfg.rms_norm_eps,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: cfg.train_use_parallel_scan(),
        };
        dims.validate_index_budget()?;

        let acts = GpuMamba3BackboneActs::new(&ctx.stream, &dims)?;
        let scratch = GpuMamba3Scratch::new(&ctx.stream, &dims)?;

        let bt = batch * seq_len;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;

        let temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, bt * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, bt * cfg.d_model)?;
        let ssm_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd * ds)?;
        let k_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * ds)?;
        let v_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * hd)?;
        let angle_states = GpuBuffer::zeros(&ctx.stream, batch * nl * nh * na)?;
        let grads = GpuMamba3Grads::new(&ctx.stream, &cfg, input_dim)?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;
        let multi_plan = build_multi_plan(
            &ctx.stream,
            &adam,
            grads.flat.cached_ptr(),
            &m3_specs(&weights, &grads),
            adam.reference_no_decay,
            adam.weight_decay,
        )?;

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        let clip_partials = alloc_partials(&ctx.stream)?;
        let clip_scratch = GpuBuffer::zeros(&ctx.stream, 2)?;

        Ok(Self {
            ctx,
            m3k,
            cfg,
            dims,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            graph: None,
            split_forward_pending: false,
            grads_dirty: false,
            clip_partials,
            clip_scratch,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;
        let g = GpuMamba3F32TrainingStepGraph::capture(
            &M3Exec {
                ctx: &self.ctx,
                kernels: &self.m3k,
                dims: &self.dims,
            },
            Mamba3F32Capture {
                weights: &mut self.weights,
                adam: &self.adam,
                bias: &self.bias,
                multi_plan: &self.multi_plan,
                grads: &mut self.grads,
                acts: &mut self.acts,
                scratch: &mut self.scratch,
                temporal: &mut self.temporal,
                mamba_input: &self.mamba_input,
                d_temporal: &mut self.d_temporal,
                states: GpuMamba3StateBufs {
                    ssm: &mut self.ssm_states,
                    k: &mut self.k_states,
                    v: &mut self.v_states,
                    angle: &mut self.angle_states,
                },
            },
        )?;
        self.graph = Some(g);
        Ok(())
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );
        if self.grads_dirty {
            return Err(
                "step(): an accumulate_only backward window is open — close it with \
                 backward_step(accumulate_only=false); the fused step zeroes the grad \
                 arena and would silently discard the accumulated gradients"
                    .into(),
            );
        }
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        let replayed = if let Some(ref g) = self.graph {
            g.replay(
                &self.ctx,
                &Mamba3F32Replay {
                    weights: &self.weights,
                    adam: &self.adam,
                    bias: &self.bias,
                    grads: &self.grads,
                    temporal: &self.temporal,
                    mamba_input: &self.mamba_input,
                    d_temporal: &self.d_temporal,
                    ssm_states: &self.ssm_states,
                    k_states: &self.k_states,
                    v_states: &self.v_states,
                    angle_states: &self.angle_states,
                },
            )?;
            true
        } else {
            self.step_eager()?;
            false
        };
        Ok(StepMetrics::plain(step, replayed))
    }

    /// Eager forward body (M1 seam mirror — the split API and the fused
    /// eager step compose exactly these bodies).
    fn eager_forward(&mut self) -> Result<(), String> {
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        gpu_forward_mamba3_backbone(
            &exec,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            GpuMamba3StateBufs {
                ssm: &mut self.ssm_states,
                k: &mut self.k_states,
                v: &mut self.v_states,
                angle: &mut self.angle_states,
            },
            &mut self.scratch,
        )
    }

    /// Eager backward body: accumulate gradients into the grad arena
    /// (beta=1.0 — zeroing is the caller's responsibility).
    fn eager_backward(&mut self) -> Result<(), String> {
        let exec = M3Exec {
            ctx: &self.ctx,
            kernels: &self.m3k,
            dims: &self.dims,
        };
        gpu_backward_mamba3_backbone(
            &exec,
            &mut self.d_temporal,
            &self.acts,
            &self.weights,
            &self.grads,
            &mut self.scratch,
        )
    }

    /// Eager optimizer tail: AdamW over the grad arena.
    fn eager_optimize(&mut self) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.m3k
                .adamw_step_multi
                .get(crate::mamba_ssm::gpu::dtype::WeightDtype::F32),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
        )
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.grads.zero(&self.ctx.stream)?;
        self.eager_forward()?;
        self.eager_backward()?;
        self.eager_optimize()
    }

    /// Split forward (see [`Mamba3Trainer::forward`]).
    pub(crate) fn forward_split(
        &mut self,
        input: &[f32],
        temporal_out: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(
            input.len(),
            self.mamba_input.len(),
            "input shape mismatch: expected batch*seq_len*input_dim={}, got {}",
            self.mamba_input.len(),
            input.len(),
        );
        assert_eq!(
            temporal_out.len(),
            self.temporal.len(),
            "temporal_out shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.temporal.len(),
            temporal_out.len(),
        );
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.eager_forward()?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("forward_split sync: {e:?}"))?;
        self.temporal.download(&self.ctx.stream, temporal_out)?;
        self.split_forward_pending = true;
        Ok(())
    }

    /// Split backward + optimizer (see [`Mamba3Trainer::backward_step`]).
    pub(crate) fn backward_split(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        if !self.split_forward_pending {
            return Err(
                "backward_step() without a pending forward() — the saved activations \
                 are stale or missing; call forward() first"
                    .into(),
            );
        }
        if opts.clip_max_norm.is_some() && opts.accumulate_only {
            return Err(
                "clip_max_norm + accumulate_only is unsupported: the global norm is only \
                 defined over the COMPLETE accumulated gradient — request the clip on the \
                 final (applying) backward_step"
                    .into(),
            );
        }
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.d_temporal.len(),
            d_temporal.len(),
        );
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
        }
        self.eager_backward()?;
        self.split_forward_pending = false;

        if opts.accumulate_only {
            self.grads_dirty = true;
            Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm: None,
                loss_scale: None,
                overflow_skipped: None,
            })
        } else {
            self.apply_step_inner(
                opts.clip_max_norm,
                opts.step_skip_above,
                opts.control_clip_max_norm,
            )
        }
    }

    /// Optimizer-only tail of the applying backward: optional clip, bias
    /// factors, fused AdamW over the accumulated gradient, window close.
    /// A separate seam so a gradient reducer can run between the last
    /// backward and the weight update.
    fn apply_step_inner(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        // Control-channel clip FIRST: the T-scaled dt/A/trap/angle
        // gradients get their own bound so a resonant page cannot
        // rescale the whole arena through the global clip below.
        if let Some(cb) = control_clip_max_norm {
            let geom = m3_control_region_geom(&self.cfg, self.dims.mamba_input_dim);
            clip_region_device(
                &self.ctx,
                &mut self.grads.flat,
                &mut self.clip_partials,
                &mut self.clip_scratch,
                geom,
                cb,
            )?;
        }
        let grad_norm = match clip_max_norm {
            Some(c) => Some(self.apply_clip(c)?),
            None => None,
        };
        // Spike skip: a window whose PRE-clip norm exceeds the threshold
        // is discarded whole - no Adam advance, no weight update, arena
        // re-zeroes on the next backward. One pathological window then
        // costs one update instead of a poisoned optimizer state.
        if let (Some(n), Some(thr)) = (grad_norm, step_skip_above)
            && n > thr
        {
            self.grads_dirty = false;
            return Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm,
                loss_scale: None,
                overflow_skipped: None,
            });
        }
        let (step, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        self.eager_optimize()?;
        self.grads_dirty = false;
        Ok(BackwardMetrics {
            step,
            optimizer_stepped: true,
            grad_norm,
            loss_scale: None,
            overflow_skipped: None,
        })
    }

    /// Compute the deterministic global grad norm, apply the clip
    /// coefficient when needed, and return the PRE-clip norm.
    fn apply_clip(&mut self, max_norm: f32) -> Result<f32, String> {
        // Device-side fold: norm, coefficient and scaling are enqueued
        // back to back, so the host blocks once with the whole sequence
        // already in flight instead of draining the stream between the
        // norm and the scale. Bit-identical to the host path.
        let norm = clip_grads_device(
            &self.ctx,
            &mut self.grads.flat,
            &mut self.clip_partials,
            &mut self.clip_scratch,
            max_norm,
        )?;
        if !norm.is_finite() {
            return Err(format!(
                "clip_max_norm: non-finite global grad norm ({norm})"
            ));
        }
        Ok(norm)
    }

    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let w = &self.weights;
        let input_dim = self.mamba_input.len() / (self.dims.batch * self.dims.seq_len);
        let mut out = Mamba3Weights::zeros(&self.cfg, input_dim);
        out.input_proj_w = w.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = w.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &w.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_bias = g.dt_bias.to_cpu(&self.ctx.stream)?;
            lw.b_norm_weight = g.b_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.c_norm_weight = g.c_norm_weight.to_cpu(&self.ctx.stream)?;
            lw.b_bias = g.b_bias.to_cpu(&self.ctx.stream)?;
            lw.c_bias = g.c_bias.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.norm_gate_weight = g.norm_gate_weight.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
        }
        out.norm_f_weight = w.norm_f_weight.to_cpu(&self.ctx.stream)?;
        Ok(out)
    }
}
