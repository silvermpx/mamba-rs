//! High-level Mamba-3 training API.
//!
//! M3 analogue of [`crate::mamba1::gpu::trainer::MambaGpuTrainer`]. Owns
//! mixed-precision weights + grads + Adam + acts + scratch + state + graph.
//!
//! Supports f32, bf16, and f16 (with dynamic loss scaler), including CUDA
//! Graph capture for all three. See the M1 trainer for the
//! precision-support rationale.

use crate::config::ModelFamily;
use crate::gpu::grad_window::check_backward_opts;
use crate::gpu::options::GpuOptions;
use crate::gpu::train_state::TrainerState;
use cudarc::driver::PushKernelArg;

use crate::gpu::adamw::{
    AdamWBiasFactors, AdamWMultiPlan, GpuAdamW, build_multi_plan, m3_specs, m3_specs_mixed,
    step_multi,
};
use crate::gpu::blas::{TypedPtr, gpu_gemm_ex_backward_dx_typed};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer};
use crate::gpu::context::{GemmMode, GemmRole, GpuCtx, gemm_mode_or_env};
use crate::gpu::device::GpuDevice;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::gpu::grad_clip::{
    CaptureTail, GradRegionGeom, StepTail, alloc_partials, clip_grads_device, clip_region_device,
    norm_limit, read_tail, scale_grads,
};
use crate::gpu::graph_capture::{
    TrainBody, capture_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::gpu::kernel_identity::{CapturedGemmGraphPlan, PreparedGemmCaptureManifest};
use crate::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::gpu::train::StepMetrics;
pub use crate::gpu::train::{BackwardMetrics, BackwardOpts, TrainSessionCfg};
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
use crate::matrix::{Path, Request};

/// CPU-side snapshot of the four carried Mamba-3 recurrences (SSM, K,
/// V, RoPE angle), layer after layer and within a layer sequence after
/// sequence, for a checkpointed resume. Every training window continues
/// from them and leaves them after its last step, on both scan paths.
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
/// [`crate::mamba3_siso::gpu::inference::Mamba3GpuBackbone`]: one public
/// struct, one method per operation, dtype dispatch happens internally.
pub struct Mamba3GpuTrainer {
    inner: Trainer3Inner,
    /// Width of an input row, from the session.
    input_dim: usize,
}

impl Mamba3GpuTrainer {
    /// Build a trainer for the session `session` names (shape and AdamW
    /// settings, see [`TrainSessionCfg::new`]) from `cpu_weights`, on the
    /// device, in the storage precision and in the GEMM mode `opts` names
    /// ([`GpuOptions::default`] is device 0, f32 and the mode
    /// `MAMBA_RS_GEMM_MODE` names). In the deterministic mode the trainer
    /// runs the Triad kernels; [`Self::ctx`] shows the route a captured graph
    /// binds. Shape, state-capacity, CUDA, upload or allocation errors are
    /// returned.
    pub fn new(
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        opts: GpuOptions,
    ) -> Result<Self, String> {
        let mut cfg = cfg;
        cfg.scan_mode = opts.scan_mode_or(cfg.scan_mode);
        Self::new_full_inner(
            opts.device,
            cpu_weights,
            cfg,
            session,
            opts.dtype,
            opts.mode,
        )
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `new` with `TrainSessionCfg::new(input_dim, batch, seq_len)` and \
                `GpuOptions::default().device(gpu_ordinal).dtype(dtype)`"
    )]
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let session = TrainSessionCfg::new(input_dim, batch, seq_len);
        let opts = GpuOptions::default().device(gpu_ordinal).dtype(dtype);
        Self::new(cpu_weights, cfg, session, opts)
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `new` with `GpuOptions::default().device(gpu_ordinal).dtype(dtype)`"
    )]
    pub fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let opts = GpuOptions::default().device(gpu_ordinal).dtype(dtype);
        Self::new(cpu_weights, cfg, session, opts)
    }

    #[deprecated(
        since = "0.8.0",
        note = "use `new` with `GpuOptions::default().device(gpu_ordinal).dtype(dtype).mode(mode)`"
    )]
    pub fn new_full_with_mode(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
        mode: GemmMode,
    ) -> Result<Self, String> {
        let opts = GpuOptions::default()
            .device(gpu_ordinal)
            .dtype(dtype)
            .mode(mode);
        Self::new(cpu_weights, cfg, session, opts)
    }

    fn new_full_inner(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        // Hoisted here so both branches get the model envelope, and
        // weights of the config's shapes: a model without an input
        // projection takes rows d_model wide.
        cfg.validate()?;
        cpu_weights.validate(&cfg, session.input_dim)?;
        let request = Request {
            path: Path::Train,
            batch: session.batch,
            seq_len: session.seq_len,
            scan: cfg.scan_mode,
            precision: dtype,
            gemm: gemm_mode_or_env(mode)?,
        };
        crate::mamba3_siso::route::admit(&cfg, session.input_dim, &request)?;
        let inner = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => Trainer3Inner::F32(Box::new(
                Mamba3TrainerF32::new_full(gpu_ordinal, cpu_weights, cfg, session, mode, dtype)?,
            )),
            WeightDtype::Bf16 | WeightDtype::F16 => Trainer3Inner::Mixed(Box::new(
                Mamba3TrainerMixed::new_full(gpu_ordinal, cpu_weights, cfg, session, dtype, mode)?,
            )),
        };
        Ok(Self {
            inner,
            input_dim: session.input_dim,
        })
    }

    /// Weight storage dtype the trainer was constructed with.
    pub fn dtype(&self) -> WeightDtype {
        match &self.inner {
            Trainer3Inner::F32(t) => t.ctx.f32_storage_dtype(),
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

    /// The model's configuration.
    pub fn config(&self) -> &Mamba3Config {
        match &self.inner {
            Trainer3Inner::F32(t) => &t.cfg,
            Trainer3Inner::Mixed(t) => &t.cfg,
        }
    }

    /// Width of an input row.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Width of the model, the rows `forward` writes.
    pub fn d_model(&self) -> usize {
        self.config().d_model
    }

    /// Access the trainer's GPU execution context.
    ///
    /// Storage is reported separately by [`Self::dtype`]; inspect execution
    /// with [`GpuCtx::gemm_mode`] and [`GpuCtx::gemm_route`]. A captured
    /// graph retains this complete route.
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
    /// Mirrors `MambaGpuTrainer::set_lr`.
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
    pub fn optimizer_state(&self) -> Result<crate::gpu::adamw::AdamWStateBlob, String> {
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
        blob: &crate::gpu::adamw::AdamWStateBlob,
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
    ///
    /// In f16 the ranks reduce their loss-scaled gradients, so an overflow
    /// on any rank reaches every rank's arena as inf or NaN: every rank
    /// skips the same window and keeps the same loss scale, given the same
    /// scale to start from.
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
    /// The allocation itself is intentionally read-only through this API:
    /// per-tensor views and CUDA graphs retain its device address.
    ///
    /// ```compile_fail
    /// use mamba_rs::gpu::buffers::GpuBuffer;
    /// use mamba_rs::mamba3_siso::gpu::trainer::Mamba3GpuTrainer;
    /// fn replace_arena(trainer: &mut Mamba3GpuTrainer, replacement: GpuBuffer) {
    ///     let _ = std::mem::replace(trainer.grad_arena(), replacement);
    /// }
    /// ```
    pub fn grad_arena(&self) -> &GpuBuffer {
        match &self.inner {
            Trainer3Inner::F32(t) => &t.grads.flat,
            Trainer3Inner::Mixed(t) => &t.grads.flat,
        }
    }

    /// The gradient arena downloaded to the host: the open window's sum
    /// between an accumulating call and the applying one.
    pub fn download_grad_arena(&self) -> Result<Vec<f32>, String> {
        self.grad_arena().to_cpu(&self.ctx().stream)
    }

    /// Replace the gradient contents without replacing their allocation.
    pub fn upload_grad_arena(&mut self, values: &[f32]) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.grads.flat.upload(&t.ctx.stream, values),
            Trainer3Inner::Mixed(t) => t.grads.flat.upload(&t.ctx.stream, values),
        }
    }

    /// Run ONLY the optimizer tail (optional clip, AdamW, window close)
    /// over the already-accumulated gradient. Together with
    /// `backward_step(accumulate_only = true)` this splits the applying
    /// backward in two, so a gradient reducer can run in between; the
    /// pair is bit-identical to a single applying `backward_step`.
    /// Requires an open accumulation window. In f16 the tail is the
    /// loss-scaler protocol over the window's sum: the overflow check, the
    /// conditional unscale, the clips, the step and the scaler update.
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
                if t.dtype == WeightDtype::F16 {
                    t.f16_apply_tail(clip_max_norm, step_skip_above, control_clip_max_norm)
                } else {
                    t.apply_step_inner(clip_max_norm, step_skip_above, control_clip_max_norm)
                }
            }
        }
    }

    /// Toggle the AdamW no-decay parameter groups (dt bias / `d_param` /
    /// every norm scale get `weight_decay = 0`; the reference marks the dt
    /// bias and `D`, the norm scales follow the usual practice). Default
    /// ON since 0.7.4; OFF reproduces the numbers of 0.7.3 and earlier. Errs
    /// while a captured graph exists (the decay coefficient is baked by
    /// value into the captured per-tensor launches). Mirrors
    /// `MambaGpuTrainer::set_reference_no_decay`.
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
    /// capture_graph` is expressible. Mirrors `MambaGpuTrainer::drop_graph`.
    pub fn drop_graph(&mut self) {
        match &mut self.inner {
            Trainer3Inner::F32(t) => {
                let _ = t.ctx.stream.synchronize();
                t.drop_graphs();
            }
            Trainer3Inner::Mixed(t) => {
                let _ = t.ctx.stream.synchronize();
                t.drop_graphs();
            }
        }
    }

    /// Set the options of the fused [`Self::step`]: the control-channel
    /// clip, the global clip and the spike skip, which the eager step and
    /// a captured graph both apply. Accumulation belongs to
    /// [`Self::backward_step`] and is refused here; a captured graph baked
    /// the options in, so other options need [`Self::drop_graph`] first.
    pub fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.set_step_opts(opts),
            Trainer3Inner::Mixed(t) => t.set_step_opts(opts),
        }
    }

    /// The options of the fused [`Self::step`].
    pub fn step_opts(&self) -> BackwardOpts {
        match &self.inner {
            Trainer3Inner::F32(t) => t.step_opts,
            Trainer3Inner::Mixed(t) => t.step_opts,
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
    /// window-handoff counterpart of [`Self::optimizer_state`].
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
        self.load_recurrent_parts([&blob.ssm, &blob.k, &blob.v, &blob.angle])
    }

    /// The (ssm, k, v, angle) recurrence from borrowed values.
    fn load_recurrent_parts(&mut self, parts: [&[f32]; 4]) -> Result<(), String> {
        let (stream, bufs) = match &mut self.inner {
            Trainer3Inner::F32(t) => (
                t.ctx.stream.clone(),
                [
                    &mut t.ssm_states,
                    &mut t.k_states,
                    &mut t.v_states,
                    &mut t.angle_states,
                ],
            ),
            Trainer3Inner::Mixed(t) => (
                t.ctx.stream.clone(),
                [
                    &mut t.ssm_states,
                    &mut t.k_states,
                    &mut t.v_states,
                    &mut t.angle_states,
                ],
            ),
        };
        if parts.iter().zip(&bufs).any(|(p, b)| p.len() != b.len()) {
            let lens: Vec<(usize, usize)> = parts
                .iter()
                .zip(&bufs)
                .map(|(p, b)| (p.len(), b.len()))
                .collect();
            return Err(format!(
                "Mamba3 recurrent state mismatch: the blob's (ssm, k, v, angle) lengths \
                 against this trainer's are {lens:?}; the blob belongs to a different shape"
            ));
        }
        for (buf, values) in bufs.into_iter().zip(parts) {
            buf.upload(&stream, values)?;
        }
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

    /// Run one training step on `(input, d_temporal)`: `input` is
    /// `[batch, seq_len, input_dim]`, `d_temporal` `[batch, seq_len,
    /// d_model]`. Returns `StepMetrics`. With an accumulation window open
    /// the step adds its micro-batch to the window and applies it, as
    /// [`Self::accumulate`] then [`Self::apply_window`] would.
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.step(input, d_temporal),
            Trainer3Inner::Mixed(t) => t.step(input, d_temporal),
        }
    }

    /// One micro-batch of an accumulation window in one call: the forward
    /// of `input` and the backward of `d_temporal`, as [`Self::step`] takes
    /// them, the gradient added to the window, which the first call opens;
    /// AdamW waits for [`Self::apply_window`]. In f16 every micro-batch
    /// takes its gradient at the window's loss scale. With captured graphs
    /// (see [`Self::capture_graph`]) the call replays one, with the eager
    /// bits.
    pub fn accumulate(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<BackwardMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.accumulate(input, d_temporal),
            Trainer3Inner::Mixed(t) => t.accumulate(input, d_temporal),
        }
    }

    /// Apply the open accumulation window with the step options (see
    /// [`Self::set_step_opts`]): in f16 the overflow check over the
    /// window's gradient and the unscale, then the control clip, the global
    /// clip, the spike skip and AdamW, as an applying [`Self::backward_step`]
    /// with those options closes a window, with its bits. With captured
    /// graphs the call replays one.
    pub fn apply_window(&mut self) -> Result<StepMetrics, String> {
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.apply_window(),
            Trainer3Inner::Mixed(t) => t.apply_window(),
        }
    }

    /// Eager forward half of the split step — M3 mirror of
    /// [`crate::mamba1::gpu::trainer::MambaGpuTrainer::forward`]: runs the
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
    /// [`crate::mamba1::gpu::trainer::MambaGpuTrainer::backward_step`].
    /// `d_temporal` is the gradient w.r.t. the IMMEDIATELY PRECEDING
    /// [`Self::forward`]'s temporal output. Errs when no forward is pending.
    pub fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        check_backward_opts(&opts, "Mamba-3 trainer")?;
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.backward_split(d_temporal, opts, None),
            Trainer3Inner::Mixed(t) => t.backward_split(d_temporal, opts, None),
        }
    }

    /// [`Self::backward_step`] that also writes d loss / d the window's
    /// input into `d_input` (`[batch, seq_len, input_dim]`), for a model
    /// that trains something upstream of this backbone. In f16 the
    /// gradient comes back unscaled. With an input projection it costs one
    /// more GEMM.
    pub fn backward_step_input_grad(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: &mut [f32],
    ) -> Result<BackwardMetrics, String> {
        check_backward_opts(&opts, "Mamba-3 trainer")?;
        let want = self.batch() * self.seq_len() * self.input_dim();
        if d_input.len() != want {
            return Err(format!(
                "backward_step_input_grad: d_input holds {} values; the window's input has \
                 {want}",
                d_input.len()
            ));
        }
        match &mut self.inner {
            Trainer3Inner::F32(t) => t.backward_split(d_temporal, opts, Some(d_input)),
            Trainer3Inner::Mixed(t) => t.backward_split(d_temporal, opts, Some(d_input)),
        }
    }

    /// Everything this trainer needs beside its weights to resume bit for
    /// bit: the optimizer state, the f16 loss scale and the carried
    /// recurrence. [`TrainerState::save`] writes it to a file.
    pub fn trainer_state(&self) -> Result<TrainerState, String> {
        let rec = self.recurrent_state()?;
        Ok(TrainerState {
            family: ModelFamily::Mamba3,
            optimizer: self.optimizer_state()?,
            scaler: self.scaler_state(),
            recurrent: vec![
                ("ssm".into(), rec.ssm),
                ("k".into(), rec.k),
                ("v".into(), rec.v),
                ("angle".into(), rec.angle),
            ],
        })
    }

    /// Load a state a trainer of this family gave with
    /// [`Self::trainer_state`]. The learning rate is not part of it.
    pub fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        state.check_family(ModelFamily::Mamba3)?;
        let parts = [
            state.recurrent_tensor("ssm")?,
            state.recurrent_tensor("k")?,
            state.recurrent_tensor("v")?,
            state.recurrent_tensor("angle")?,
        ];
        self.load_optimizer_state(&state.optimizer)?;
        if let Some((scale, tracker)) = state.scaler {
            self.load_scaler_state(scale, tracker)?;
        }
        self.load_recurrent_parts(parts)
    }

    /// Download f32 master weights for checkpointing.
    pub fn snapshot_master(&self) -> Result<Mamba3Weights, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.snapshot_master(),
            Trainer3Inner::Mixed(t) => t.snapshot_master(),
        }
    }

    /// Download the most recent f32 temporal output without changing trainer state.
    pub fn last_temporal(&self) -> Result<Vec<f32>, String> {
        match &self.inner {
            Trainer3Inner::F32(t) => t.temporal.to_cpu(&t.ctx.stream),
            Trainer3Inner::Mixed(t) => t.temporal.to_cpu(&t.ctx.stream),
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
    /// No effect for non-f16 trainers; a scale that is not finite and
    /// positive is an error in every precision.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        DynamicLossScaler::check_scale(scale)?;
        if let Trainer3Inner::Mixed(ref mut t) = self.inner {
            t.load_scaler_state(scale, growth_tracker)?;
        }
        Ok(())
    }
}

/// The CONTROL region of the M3 grad arena: the dd_dt/dd_A/trap/angle
/// column stripe of every layer's in_proj gradient plus dt_bias — the
/// only gradients whose magnitude scales with the sequence length. The
/// stripe length is derived as `ip - (3*nh + na)` so B/C sizing changes
/// cannot desync it from the split order.
/// An error unless a host slice holds the window's `want` values.
fn check_window_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        return Err(format!("{what}: {got} values, the window holds {want}"));
    }
    Ok(())
}

fn m3_control_region_geom(cfg: &Mamba3Config, input_dim: usize) -> GradRegionGeom {
    let dm = cfg.d_model;
    let nh = cfg.nheads();
    let ip = cfg.in_proj_out_dim();
    let na = cfg.num_rope_angles();
    let ctrl_len = 3 * nh + na;
    let ctrl0 = ip - ctrl_len;
    // The first layer's in_proj gradient, after the input projection's and
    // the layer's block norm scale and bias.
    let in_proj0 = input_dim * dm + dm + dm + cfg.block_norm.bias_len(dm);
    GradRegionGeom {
        n_layers: cfg.n_layers as i32,
        layer_stride: GpuMamba3Grads::layer_len(cfg) as i64,
        a_off: (in_proj0 + ctrl0) as i64,
        a_rows: dm as i32,
        a_cols: ctrl_len as i32,
        a_row_stride: ip as i64,
        b_off: (in_proj0 + dm * ip) as i64,
        b_len: nh as i32,
        c_off: 0,
        c_len: 0,
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
    /// The accumulating micro-batch and the window's apply, captured
    /// beside the step.
    graph_acc: Option<GpuMamba3TrainingStepGraph>,
    graph_apply: Option<GpuMamba3TrainingStepGraph>,
    prepared_gemm_manifest: Option<PreparedGemmCaptureManifest>,

    /// Route that produced the saved split-forward activations.
    split_forward_route: Option<crate::gpu::context::GemmRoute>,
    /// True while the grad arena holds accumulated (un-applied) gradients
    /// from `accumulate_only` backward calls — the next backward must NOT
    /// zero the arena.
    grads_dirty: bool,
    /// Fixed-size f64 partials for the deterministic global grad norm.
    clip_partials: GpuByteBuffer,
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,
    /// The options of the fused step, eager or captured.
    step_opts: BackwardOpts,
    /// The skip flag a measured step tail writes (see [`StepTail`]).
    skip_flag: OverflowFlag,

    // f16 AMP loss scaler (None for bf16). See M1 trainer for the protocol.
    scaler: Option<DynamicLossScaler>,
    overflow_flag: Option<OverflowFlag>,
    d_temporal_scaled: Option<GpuBuffer>,
    /// The input projection's dX in the compute precision, allocated by the
    /// first backward that asks for the input gradient.
    input_grad_half: Option<DtypedBuf>,
    /// f16 CUDA Graph (M3 analogue of M1's `graph_f16`).
    graph_f16: Option<cudarc::driver::CudaGraph>,
    /// The f16 accumulating micro-batch and the window's apply, captured
    /// beside the step, on the same buffers.
    graph_f16_acc: Option<cudarc::driver::CudaGraph>,
    graph_f16_apply: Option<cudarc::driver::CudaGraph>,
    prepared_f16_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    captured_f16_gemm_plan: Option<CapturedGemmGraphPlan>,
    captured_f16_acc_plan: Option<CapturedGemmGraphPlan>,
    has_gemm_work: bool,
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
    captured_f16_gemm_route: crate::gpu::context::GemmRoute,
}

impl Drop for Mamba3TrainerMixed {
    fn drop(&mut self) {
        let _ = self.ctx.stream.synchronize();
        self.drop_graphs();
    }
}

impl Mamba3TrainerMixed {
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        dtype: WeightDtype,
        mode: Option<GemmMode>,
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
        let role = GemmRole::triad(dtype);
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(&device, 64, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(&device, 64, role)?,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            arch,
            crate::gpu::kernels::state_capacity(cfg.d_state)?,
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
            ssm_norm_eps: cfg.ssm_norm_eps,
            block_norm: cfg.block_norm,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: cfg.train_use_parallel_scan(),
        };
        dims.check_int_args()?;

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
        let skip_flag = OverflowFlag::new(&ctx.stream)?;
        let initial_gemm_route = ctx.gemm_route();
        let has_gemm_work =
            dims.bt() != 0 && (weights.compute.input_proj_w.len_elems() != 0 || dims.n_layers != 0);

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
            graph_acc: None,
            graph_apply: None,
            prepared_gemm_manifest: None,
            split_forward_route: None,
            grads_dirty: false,
            clip_partials,
            clip_scratch,
            step_opts: BackwardOpts::default(),
            skip_flag,
            scaler,
            overflow_flag,
            d_temporal_scaled,
            input_grad_half: None,
            graph_f16: None,
            graph_f16_acc: None,
            graph_f16_apply: None,
            prepared_f16_gemm_manifest: None,
            captured_f16_gemm_plan: None,
            captured_f16_acc_plan: None,
            has_gemm_work,
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

    /// Drop every captured body; the stream must be idle.
    fn drop_graphs(&mut self) {
        drop(self.graph.take());
        drop(self.graph_acc.take());
        drop(self.graph_apply.take());
        drop(self.graph_f16.take());
        drop(self.graph_f16_acc.take());
        drop(self.graph_f16_apply.take());
    }

    /// The step options' tail (see [`StepTail`]).
    fn step_tail(&self) -> StepTail {
        StepTail::of(
            &self.step_opts,
            m3_control_region_geom(&self.cfg, self.dims.mamba_input_dim),
        )
    }

    /// Set the options of the fused step (see
    /// [`Mamba3GpuTrainer::set_step_opts`]).
    pub(crate) fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        if self.has_graph() && opts != self.step_opts {
            return Err(
                "Mamba-3 set_step_opts: the captured graph baked the step options in; \
                 drop_graph first"
                    .into(),
            );
        }
        self.step_opts = opts;
        Ok(())
    }

    /// Enqueue the step options' tail on the gradient; the flag the AdamW
    /// after it skips on (`overflow` when it measures nothing).
    fn enqueue_step_tail(
        &mut self,
        overflow: cudarc::driver::sys::CUdeviceptr,
    ) -> Result<cudarc::driver::sys::CUdeviceptr, String> {
        let tail = self.step_tail();
        let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
        tail.enqueue(
            &self.ctx,
            &mut self.grads.flat,
            (&mut self.clip_partials, &mut self.clip_scratch),
            (skip, overflow),
        )
    }

    /// After a fused step: whether its AdamW skipped and the pre-clip norm
    /// when the tail measured one. A norm that is not finite and that no
    /// overflow explains is an error, with the step counted before it given
    /// back.
    fn step_verdict(&mut self, overflowed: bool) -> Result<(bool, Option<f32>), String> {
        if !self.step_tail().measures() {
            return Ok((overflowed, None));
        }
        match read_tail(&self.ctx, &self.skip_flag, &self.clip_scratch, overflowed) {
            Ok((skipped, norm)) => Ok((skipped, Some(norm))),
            Err(e) => {
                self.adam.step -= 1;
                Err(e)
            }
        }
    }

    /// Reset SSM / K / V / angle states to zero (keeps weights untouched).
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.split_forward_route = None;
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    fn presize_prepared_gemm_scratch(&self) -> Result<(), String> {
        self.ctx.presize_bi_upcast_scratch_for_train_m3(
            &self.cfg,
            self.dims.batch,
            self.dims.seq_len,
            self.dims.mamba_input_dim,
            self.dtype,
        )
    }

    /// Capture the training-step CUDA Graph. Call once after at least one
    /// warmup [`Self::step`].
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if matches!(self.dtype, WeightDtype::F16) {
            return self.capture_graph_f16();
        }
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;
        let manifest = self.prepared_gemm_manifest.ok_or_else(|| {
            "M3 bf16 training graph capture requires one successful eager step".to_string()
        })?;
        self.graph = Some(self.capture_mixed_body(&manifest, TrainBody::Step)?);
        self.graph_acc = Some(self.capture_mixed_body(&manifest, TrainBody::Accumulate)?);
        // The apply runs no GEMM: it is held to an empty inventory, so a
        // GEMM that found its way into it would fail the capture.
        let no_gemm = self.ctx.record_eager_gemm_manifest(|| Ok(()))?;
        self.graph_apply = Some(self.capture_mixed_body(&no_gemm, TrainBody::Apply)?);
        Ok(())
    }

    /// Capture one bf16 training body (see [`TrainBody`]).
    fn capture_mixed_body(
        &mut self,
        manifest: &PreparedGemmCaptureManifest,
        body: TrainBody,
    ) -> Result<GpuMamba3TrainingStepGraph, String> {
        let tail = self.step_tail();
        let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
        // The trainer owns every captured allocation and drops the graphs first.
        unsafe {
            GpuMamba3TrainingStepGraph::capture(
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
                    tail: Some(CaptureTail {
                        tail,
                        partials: &mut self.clip_partials,
                        coef: &mut self.clip_scratch,
                        skip,
                    }),
                },
                manifest,
                body,
            )
        }
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        // Every fused attempt abandons a pending split tape, even when it is rejected.
        self.split_forward_route = None;
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );
        // A step with a window open adds its micro-batch to the window and
        // applies it, as the split calls would close the window.
        if self.grads_dirty {
            self.accumulate(input, d_temporal)?;
            return self.apply_window();
        }
        if matches!(self.dtype, WeightDtype::F16) {
            return self.step_f16(input, d_temporal);
        }

        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;

        // The step counts before the replay, as the eager step counts it
        // before its AdamW; a step the tail skipped gives it back.
        let prev_step = self.adam.step;
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;

        let replayed = if let Some(ref g) = self.graph {
            g.replay(&self.ctx, &self.mixed_replay())?;
            true
        } else {
            self.step_eager()?;
            false
        };
        self.finish_step(prev_step, None, replayed)
    }

    fn eager_f16_forward_backward(&mut self) -> Result<(), String> {
        self.presize_prepared_gemm_scratch()?;
        let has_gemm_work = self.has_gemm_work;
        if has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let Self {
            ctx,
            m3k,
            dims,
            weights,
            grads,
            acts,
            f32_scratch,
            mixed_scratch,
            temporal,
            mamba_input,
            d_temporal_scaled,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            prepared_f16_gemm_manifest,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let d_temporal = d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                &exec,
                temporal,
                acts,
                weights,
                mamba_input,
                GpuMamba3StateBufs {
                    ssm: ssm_states,
                    k: k_states,
                    v: v_states,
                    angle: angle_states,
                },
                mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &exec,
                d_temporal,
                acts,
                weights,
                grads,
                f32_scratch,
                mixed_scratch,
            )
        })?;
        *prepared_f16_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// The f16 lane's upstream gradient: `d_temporal` uploaded into the
    /// scaled buffer and multiplied by the loss scale on the device.
    fn upload_scaled_d(&mut self, d_temporal: &[f32], scale: f32) -> Result<(), String> {
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        dt_scaled.upload(&self.ctx.stream, d_temporal)?;
        let n = d_temporal.len() as i64;
        let mut builder = self
            .ctx
            .stream
            .launch_builder(&self.ctx.kernels.scale_grads_f32);
        builder.arg(dt_scaled.inner_mut());
        builder.arg(&scale);
        builder.arg(&n);
        unsafe { builder.launch(crate::gpu::launch::grid_1d_wide(d_temporal.len())) }
            .map(|_| ())
            .map_err(|e| format!("scale d_temporal (m3 f16): {e:?}"))
    }

    /// What every f16 replay checks first: the device buffers baked into
    /// the captured kernels at the same pointers, and the GEMM route.
    fn check_f16_replay(&self) -> Result<(), String> {
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
        if self.ctx.gemm_route() != self.captured_f16_gemm_route {
            return Err("M3 f16 graph replay: GEMM route changed since capture \
                     (policy, compiler, artifact, or device identity); re-capture instead"
                .into());
        }
        Ok(())
    }

    /// The bf16 graphs' replay buffers.
    fn mixed_replay(&self) -> Mamba3MixedReplay<'_> {
        Mamba3MixedReplay {
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
        }
    }

    /// The verdict of an update after it ran: in f16 the overflow, then
    /// the step options' tail, the loss scaler, and the step counter back
    /// at `prev_step` when AdamW skipped (an overflow, a spike, or an error
    /// for a norm that is not finite).
    fn finish_step(
        &mut self,
        prev_step: u64,
        scale: Option<f32>,
        replayed: bool,
    ) -> Result<StepMetrics, String> {
        let overflow = match (scale, self.overflow_flag.as_ref()) {
            (Some(_), Some(flag)) => flag.read(&self.ctx.stream)? != 0,
            _ => false,
        };
        let (skipped, grad_norm) = self.step_verdict(overflow)?;
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.update(overflow);
        }
        if skipped {
            self.adam.step = prev_step;
        }
        Ok(StepMetrics {
            step: self.adam.step,
            graph_replayed: replayed,
            loss_scale: scale,
            overflow_skipped: scale.map(|_| overflow),
            optimizer_stepped: !skipped,
            grad_norm,
        })
    }

    /// One micro-batch of an accumulation window (see
    /// [`Mamba3GpuTrainer::accumulate`]).
    pub(crate) fn accumulate(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<BackwardMetrics, String> {
        self.split_forward_route = None;
        check_window_len("accumulate input", input.len(), self.mamba_input.len())?;
        check_window_len(
            "accumulate d_temporal",
            d_temporal.len(),
            self.d_temporal.len(),
        )?;
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
            self.grads_dirty = true;
        }
        self.mamba_input.upload(&self.ctx.stream, input)?;
        match scale {
            Some(scale) => self.upload_scaled_d(d_temporal, scale)?,
            None => self.d_temporal.upload(&self.ctx.stream, d_temporal)?,
        }
        match (scale, &self.graph_acc, &self.graph_f16_acc) {
            (None, Some(g), _) => g.replay(&self.ctx, &self.mixed_replay())?,
            (Some(_), _, Some(g)) => {
                self.check_f16_replay()?;
                with_validated_gemm_graph_launch(
                    &self.ctx,
                    self.has_gemm_work,
                    self.captured_f16_acc_plan.as_ref(),
                    "M3 f16 accumulate graph replay",
                    || {
                        g.launch()
                            .map_err(|e| format!("M3 f16 accumulate graph launch: {e:?}"))
                    },
                )?;
            }
            _ => self.eager_accumulate(scale.is_some())?,
        }
        Ok(BackwardMetrics {
            step: self.adam.step,
            optimizer_stepped: false,
            grad_norm: None,
            loss_scale: scale,
            overflow_skipped: None,
        })
    }

    /// One accumulating micro-batch eagerly: the forward and the backward
    /// into the open window, the GEMM inventory recorded for a capture.
    fn eager_accumulate(&mut self, scaled: bool) -> Result<(), String> {
        self.presize_prepared_gemm_scratch()?;
        if self.has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let Self {
            ctx,
            m3k,
            dims,
            weights,
            grads,
            acts,
            f32_scratch,
            mixed_scratch,
            temporal,
            mamba_input,
            d_temporal,
            d_temporal_scaled,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            prepared_gemm_manifest,
            prepared_f16_gemm_manifest,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let d = match (scaled, d_temporal_scaled.as_mut()) {
            (true, Some(d)) => d,
            (true, None) => {
                return Err("M3 f16 accumulate: the scaled gradient buffer is missing".into());
            }
            (false, _) => d_temporal,
        };
        let manifest = ctx.record_eager_gemm_manifest(|| {
            gpu_forward_mamba3_backbone_mixed(
                &exec,
                temporal,
                acts,
                weights,
                mamba_input,
                GpuMamba3StateBufs {
                    ssm: ssm_states,
                    k: k_states,
                    v: v_states,
                    angle: angle_states,
                },
                mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &exec,
                d,
                acts,
                weights,
                grads,
                f32_scratch,
                mixed_scratch,
            )
        })?;
        if scaled {
            *prepared_f16_gemm_manifest = Some(manifest);
        } else {
            *prepared_gemm_manifest = Some(manifest);
        }
        Ok(())
    }

    /// The open window's apply (see [`Mamba3GpuTrainer::apply_window`]).
    pub(crate) fn apply_window(&mut self) -> Result<StepMetrics, String> {
        if !self.grads_dirty {
            return Err("apply_window: no accumulation window is open".into());
        }
        self.split_forward_route = None;
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        if let (Some(u), Some(scale)) = (self.unscale_factor.as_mut(), scale) {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        if let Some(flag) = self.overflow_flag.as_mut() {
            flag.zero(&self.ctx.stream)?;
        }
        // The step counts before the update, as the fused step counts it;
        // a step the overflow or the tail skipped gives it back.
        let prev_step = self.adam.step;
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        let replayed = match (scale, &self.graph_apply, &self.graph_f16_apply) {
            (None, Some(g), _) => {
                g.replay(&self.ctx, &self.mixed_replay())?;
                true
            }
            (Some(_), _, Some(g)) => {
                self.check_f16_replay()?;
                with_validated_gemm_graph_launch(
                    &self.ctx,
                    false,
                    None,
                    "M3 f16 apply graph replay",
                    || {
                        g.launch()
                            .map_err(|e| format!("M3 f16 apply graph launch: {e:?}"))
                    },
                )?;
                true
            }
            _ => {
                self.eager_apply(scale.is_some())?;
                false
            }
        };
        self.grads_dirty = false;
        self.finish_step(prev_step, scale, replayed)
    }

    /// The window's apply eagerly, the captured body's sequence: in f16 the
    /// overflow check and the unscale that skips on it, then the step
    /// options' tail and an AdamW that skips on the tail's flag.
    fn eager_apply(&mut self, f16: bool) -> Result<(), String> {
        let mut overflow = 0;
        if f16 {
            check_inf_nan_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &self.grads.flat,
            )?;
            let unscale = self.unscale_factor.as_ref().expect("unscale buf");
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                unscale,
            )?;
            overflow = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .stable_ptr(&self.ctx.stream);
        }
        let skip = self.enqueue_step_tail(overflow)?;
        self.eager_optimize(skip)
    }

    /// f16 step. See M1 [`crate::mamba1::gpu::trainer::MambaTrainerMixed::step_f16`]
    /// for the full protocol; this is the M3 mirror.
    fn step_f16(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.upload_scaled_d(d_temporal, scale)?;

        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        let prev_step = self.adam.step;
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;

        self.overflow_flag
            .as_mut()
            .expect("f16 overflow flag")
            .zero(&self.ctx.stream)?;

        let replayed = if let Some(ref g) = self.graph_f16 {
            self.check_f16_replay()?;

            with_validated_gemm_graph_launch(
                &self.ctx,
                self.has_gemm_work,
                self.captured_f16_gemm_plan.as_ref(),
                "M3 f16 training graph replay",
                || {
                    g.launch()
                        .map_err(|e| format!("M3 f16 graph launch: {e:?}"))
                },
            )?;
            true
        } else {
            // The captured body's sequence, from the shared eager phase
            // bodies: the overflow check, the unscale that skips on it, the
            // step options' tail and an AdamW that skips on the flag the
            // tail leaves. On an overflow or a spike the parameters, both
            // moments and the step counter stay where the previous step
            // left them, which is what GradScaler does; the M1 twin runs
            // the same sequence.
            self.eager_f16_forward_backward()?;
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
            let overflow_ptr = self
                .overflow_flag
                .as_ref()
                .unwrap()
                .stable_ptr(&self.ctx.stream);
            let skip = self.enqueue_step_tail(overflow_ptr)?;
            self.eager_optimize(skip)?;
            false
        };
        self.finish_step(prev_step, Some(scale), replayed)
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
        self.presize_prepared_gemm_scratch()?;
        let manifest = self.prepared_f16_gemm_manifest.ok_or_else(|| {
            "M3 f16 training graph capture requires one successful eager step".to_string()
        })?;
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
        let tail = self.step_tail();
        let skip_ptr = self.skip_flag.stable_ptr(&self.ctx.stream);

        // Capture bodies: the whole step, an accumulating micro-batch and the
        // window's apply, each the eager sequence 1:1 (composed from the
        // shared eager phase bodies) so numerics match.
        self.ctx.freeze_graph_scratch();
        let tail = (tail, skip_ptr, snap_overflow);
        let (g, captured_f16_gemm_plan) =
            self.capture_f16_body(&manifest, TrainBody::Step, tail)?;
        let (acc, acc_plan) = self.capture_f16_body(&manifest, TrainBody::Accumulate, tail)?;
        // The apply runs no GEMM: it is held to an empty inventory, so a
        // GEMM that found its way into it would fail the capture.
        let no_gemm = self.ctx.record_eager_gemm_manifest(|| Ok(()))?;
        let (apply, _) = self.capture_f16_body(&no_gemm, TrainBody::Apply, tail)?;
        for (plan, label) in [
            (&captured_f16_gemm_plan, "M3 f16 training graph capture"),
            (&acc_plan, "M3 f16 accumulate graph capture"),
        ] {
            require_deterministic_gemm_graph_plan(
                &self.ctx,
                self.has_gemm_work,
                plan.as_ref(),
                label,
            )?;
        }
        self.graph_f16_acc = Some(acc);
        self.graph_f16_apply = Some(apply);
        self.captured_f16_acc_plan = acc_plan;
        self.graph_f16 = Some(g);
        self.captured_f16_gemm_plan = captured_f16_gemm_plan;
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

    /// Capture one f16 training body (see [`TrainBody`]): the whole step
    /// (the arena zeroed, the forward, the backward on the scaled gradient,
    /// the overflow check, the unscale that skips on it, the step options'
    /// tail, an AdamW that skips on the tail's flag and the master sync),
    /// an accumulating micro-batch (the forward and the backward into the
    /// open window) or the window's apply (the step's sequence from the
    /// overflow check on).
    fn capture_f16_body(
        &mut self,
        manifest: &PreparedGemmCaptureManifest,
        body: TrainBody,
        (tail, skip_ptr, overflow_ptr): (StepTail, u64, u64),
    ) -> Result<(cudarc::driver::CudaGraph, Option<CapturedGemmGraphPlan>), String> {
        let Self {
            ctx,
            m3k,
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
            d_temporal_scaled,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            overflow_flag,
            unscale_factor,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let d_temporal = d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        unsafe {
            capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, manifest, || {
                if body.passes() {
                    if body == TrainBody::Step {
                        grads.zero(&ctx.stream)?;
                    }
                    gpu_forward_mamba3_backbone_mixed(
                        &exec,
                        temporal,
                        acts,
                        weights,
                        mamba_input,
                        GpuMamba3StateBufs {
                            ssm: ssm_states,
                            k: k_states,
                            v: v_states,
                            angle: angle_states,
                        },
                        mixed_scratch,
                    )?;
                    gpu_backward_mamba3_backbone_mixed(
                        &exec,
                        d_temporal,
                        acts,
                        weights,
                        grads,
                        f32_scratch,
                        mixed_scratch,
                    )?;
                }
                if !body.updates() {
                    return Ok(());
                }
                check_inf_nan_gpu(
                    ctx,
                    &ctx.kernels,
                    overflow_flag.as_mut().unwrap(),
                    &grads.flat,
                )?;
                scale_grads_skip_gpu(
                    ctx,
                    &ctx.kernels,
                    overflow_flag.as_mut().unwrap(),
                    &mut grads.flat,
                    unscale_factor.as_ref().unwrap(),
                )?;
                let skip = tail.enqueue(
                    ctx,
                    &mut grads.flat,
                    (&mut *clip_partials, &mut *clip_scratch),
                    (skip_ptr, overflow_ptr),
                )?;
                step_multi(
                    ctx,
                    m3k.adamw_step_multi.get(*dtype),
                    multi_plan,
                    adam,
                    bias.ptr(),
                    skip,
                )?;
                weights.sync_master_to_compute(ctx)
            })
        }
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
    fn eager_optimize(&mut self, skip: cudarc::driver::sys::CUdeviceptr) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.m3k.adamw_step_multi.get(self.dtype),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
            skip,
        )?;
        // Bulk typed shadows ride the fused kernel; the walk covers only
        // the f32-stays-f32 tensors.
        self.weights.sync_master_to_compute(&self.ctx)
    }

    fn step_eager(&mut self) -> Result<(), String> {
        self.presize_prepared_gemm_scratch()?;
        let has_gemm_work = self.has_gemm_work;
        if has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let tail = self.step_tail();
        let skip_ptr = self.skip_flag.stable_ptr(&self.ctx.stream);
        let Self {
            ctx,
            m3k,
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
            prepared_gemm_manifest,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone_mixed(
                &exec,
                temporal,
                acts,
                weights,
                mamba_input,
                GpuMamba3StateBufs {
                    ssm: ssm_states,
                    k: k_states,
                    v: v_states,
                    angle: angle_states,
                },
                mixed_scratch,
            )?;
            gpu_backward_mamba3_backbone_mixed(
                &exec,
                d_temporal,
                acts,
                weights,
                grads,
                f32_scratch,
                mixed_scratch,
            )?;
            let skip = tail.enqueue(
                ctx,
                &mut grads.flat,
                (&mut *clip_partials, &mut *clip_scratch),
                (skip_ptr, 0),
            )?;
            step_multi(
                ctx,
                m3k.adamw_step_multi.get(*dtype),
                multi_plan,
                adam,
                bias.ptr(),
                skip,
            )?;
            weights.sync_master_to_compute(ctx)
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Split forward (see [`Mamba3GpuTrainer::forward`]). The M3 mixed forward
    /// already writes f32 temporal — no upcast staging needed (unlike M1).
    pub(crate) fn forward_split(
        &mut self,
        input: &[f32],
        temporal_out: &mut [f32],
    ) -> Result<(), String> {
        self.split_forward_route = None;
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
        self.split_forward_route = Some(self.ctx.gemm_route());
        Ok(())
    }

    /// Split backward + optimizer (see [`Mamba3GpuTrainer::backward_step`]),
    /// and the input gradient into `d_input` when one is asked for.
    pub(crate) fn backward_split(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: Option<&mut [f32]>,
    ) -> Result<BackwardMetrics, String> {
        let Some(forward_route) = self.split_forward_route.take() else {
            return Err(
                "backward_step() without a pending forward() — the saved activations \
                 are stale or missing; call forward() first"
                    .into(),
            );
        };
        if self.ctx.gemm_route() != forward_route {
            return Err(
                "M3 split backward: GEMM route changed since forward; run forward again".into(),
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
            return self.backward_split_f16(d_temporal, opts, d_input);
        }

        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
        }
        self.eager_backward(false)?;
        if let Some(d_input) = d_input {
            self.download_input_grad(d_input)?;
        }

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
        let grad_norm = match norm_limit(clip_max_norm, step_skip_above) {
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
        self.eager_optimize(0)?;
        self.grads_dirty = false;
        Ok(BackwardMetrics {
            step,
            optimizer_stepped: true,
            grad_norm,
            loss_scale: None,
            overflow_skipped: None,
        })
    }

    /// f16 split backward under the GradScaler protocol. The loss scale
    /// holds for a whole accumulation window: the scaler moves only on the
    /// call that closes it, so every micro-batch adds gradients of the same
    /// scale into the arena. An accumulating call scales its d_temporal and
    /// adds its gradients; the closing call adds its own and runs the scaler
    /// tail over the window's sum, where an overflow in any micro-batch
    /// survives as inf or NaN and skips the whole window.
    fn backward_split_f16(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: Option<&mut [f32]>,
    ) -> Result<BackwardMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
        {
            let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
            dt_scaled.upload(&self.ctx.stream, d_temporal)?;
            let n = d_temporal.len() as i64;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.scale_grads_f32);
            builder.arg(dt_scaled.inner_mut());
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(crate::gpu::launch::grid_1d_wide(d_temporal.len())) }
                .map_err(|e| format!("scale d_temporal (m3 f16 split): {e:?}"))?;
        }
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
        }
        self.eager_backward(true)?;
        if let Some(d_input) = d_input {
            self.download_input_grad(d_input)?;
        }
        if opts.accumulate_only {
            self.grads_dirty = true;
            return Ok(BackwardMetrics {
                step: self.adam.step,
                optimizer_stepped: false,
                grad_norm: None,
                loss_scale: Some(scale),
                overflow_skipped: None,
            });
        }
        self.f16_apply_tail(
            opts.clip_max_norm,
            opts.step_skip_above,
            opts.control_clip_max_norm,
        )
    }

    /// d loss / d the window's input after the eager backward: the
    /// backbone input's gradient without an input projection, else the
    /// projection's dX, from the cast dY the backward leaves in the
    /// projection's saved outputs. Unscaled in f16.
    fn download_input_grad(&mut self, d_input: &mut [f32]) -> Result<(), String> {
        let stream = self.ctx.stream.clone();
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        let dtype = self.dtype;
        if self.weights.compute.input_proj_w.len_elems() == 0 {
            let src = match (&self.d_temporal_scaled, scale) {
                (Some(scaled), Some(_)) => scaled,
                _ => &self.d_temporal,
            };
            src.download(&stream, d_input)?;
        } else {
            let (bt, mid, dm) = (self.dims.bt(), self.dims.mamba_input_dim, self.dims.d_model);
            if self.input_grad_half.is_none() {
                self.input_grad_half = Some(DtypedBuf::zeros(&stream, bt * mid, dtype)?);
            }
            let Some(dx) = &self.input_grad_half else {
                return Err("the input gradient's half buffer is missing".into());
            };
            gpu_gemm_ex_backward_dx_typed(
                &self.ctx,
                TypedPtr {
                    ptr: dx.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: self.acts.input_proj_outputs.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: self.weights.compute.input_proj_w.ptr(),
                    dtype,
                },
                bt,
                mid,
                dm,
            )?;
            let cast = match dtype {
                WeightDtype::Bf16 => &self.ctx.kernels.cast_bf16_to_f32,
                _ => &self.ctx.kernels.cast_f16_to_f32,
            };
            crate::gpu::launch::launch_cast(
                &stream,
                cast,
                (self.f32_scratch.d_input_proj_dx.cached_ptr(), 4),
                (dx.cached_ptr(), dtype.size_bytes()),
                bt * mid,
                format_args!("input gradient cast"),
            )?;
            self.f32_scratch
                .d_input_proj_dx
                .download(&stream, d_input)?;
        }
        stream
            .synchronize()
            .map_err(|e| format!("input gradient download: {e:?}"))?;
        if let Some(sc) = scale {
            let inv = 1.0 / sc;
            for v in d_input.iter_mut() {
                *v *= inv;
            }
        }
        Ok(())
    }

    /// The GradScaler tail over the gradient sum of the open window: the
    /// overflow check, the conditional unscale, the control and global
    /// clips, the optimizer step, the scaler update and, on a skipped step,
    /// the step rollback. Closes the window. The clip norm is taken after
    /// the unscale, per the unscale-then-norm-then-clip ordering law.
    fn f16_apply_tail(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();
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
        self.grads_dirty = false;
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
        let mut skipped = false;
        if !overflow {
            let unscale = self.unscale_factor.as_ref().expect("unscale buf");
            scale_grads_skip_gpu(
                &self.ctx,
                &self.ctx.kernels,
                self.overflow_flag.as_mut().unwrap(),
                &mut self.grads.flat,
                unscale,
            )?;
            // Control-channel clip first, then the global one: the same
            // order the f32 and bf16 lanes use in `apply_step_inner`.
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
            if let Some(c) = norm_limit(clip_max_norm, step_skip_above) {
                grad_norm = Some(self.apply_clip(c)?);
            }
            // Spike skip: a window whose PRE-clip norm exceeds the threshold
            // is discarded whole, exactly as in the other lanes.
            skipped = matches!((grad_norm, step_skip_above), (Some(n), Some(thr)) if n > thr);
            if !skipped {
                self.eager_optimize(0)?;
            }
        }
        self.scaler.as_mut().expect("f16 scaler").update(overflow);
        let stepped = !overflow && !skipped;
        let final_step = if stepped {
            next_step
        } else {
            self.adam.step = prev_step;
            prev_step
        };
        Ok(BackwardMetrics {
            step: final_step,
            optimizer_stepped: stepped,
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
            if let Some(b) = &g.norm_bias {
                lw.norm_bias = b.to_cpu(&self.ctx.stream)?;
            }
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
        if let Some(b) = &m.norm_f_bias {
            out.norm_f_bias = b.to_cpu(&self.ctx.stream)?;
        }
        for (mo, mg) in out.mlp.iter_mut().zip(&m.mlp) {
            mo.norm_weight = mg.norm_weight.to_cpu(&self.ctx.stream)?;
            if let Some(b) = &mg.norm_bias {
                mo.norm_bias = b.to_cpu(&self.ctx.stream)?;
            }
            mo.fc1_w = mg.fc1_w.to_cpu(&self.ctx.stream)?;
            mo.fc2_w = mg.fc2_w.to_cpu(&self.ctx.stream)?;
        }
        Ok(out)
    }

    /// Serialize dynamic loss scaler state. `None` when scaler is disabled
    /// (bf16). Mirrors `MambaTrainerMixed::scaler_state`.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(|s| s.state())
    }

    /// Restore scaler state from a prior `scaler_state()`. No effect when
    /// the scaler is disabled (bf16). Also rewrites the on-device unscale
    /// factor so the next f16 step uses the restored scale.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        if let Some(ref mut s) = self.scaler {
            s.load_state(scale, growth_tracker)?;
            if let Some(ref mut uf) = self.unscale_factor {
                uf.write(&self.ctx.stream, 1.0 / s.scale())?;
            }
        }
        Ok(())
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
    /// The accumulating micro-batch and the window's apply, captured
    /// beside the step.
    graph_acc: Option<GpuMamba3F32TrainingStepGraph>,
    graph_apply: Option<GpuMamba3F32TrainingStepGraph>,
    prepared_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    has_gemm_work: bool,
    /// Route that produced the saved split-forward activations.
    split_forward_route: Option<crate::gpu::context::GemmRoute>,
    /// True while the grad arena holds accumulated (un-applied) gradients
    /// from `accumulate_only` backward calls.
    grads_dirty: bool,
    /// Fixed-size f64 partials for the deterministic global grad norm.
    clip_partials: GpuByteBuffer,
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,
    /// The options of the fused step, eager or captured.
    step_opts: BackwardOpts,
    /// The skip flag a measured step tail writes (see [`StepTail`]).
    skip_flag: OverflowFlag,
}

impl Drop for Mamba3TrainerF32 {
    fn drop(&mut self) {
        let _ = self.ctx.stream.synchronize();
        self.drop_graphs();
    }
}

impl Mamba3TrainerF32 {
    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        session: TrainSessionCfg,
        mode: Option<GemmMode>,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        let device = GpuDevice::new(gpu_ordinal)?;
        let role = GemmRole::triad(dtype);
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(&device, 64, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(&device, 64, role)?,
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let m3k = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            arch,
            crate::gpu::kernels::state_capacity(cfg.d_state)?,
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
            ssm_norm_eps: cfg.ssm_norm_eps,
            block_norm: cfg.block_norm,
            is_outproj_norm: cfg.is_outproj_norm,
            use_parallel_scan: cfg.train_use_parallel_scan(),
        };
        dims.check_int_args()?;

        let hidden = cfg.mlp_hidden();
        let acts =
            GpuMamba3BackboneActs::new(&ctx.stream, &dims)?.with_mlp(&ctx.stream, &dims, hidden)?;
        let scratch =
            GpuMamba3Scratch::new(&ctx.stream, &dims)?.with_mlp(&ctx.stream, &dims, hidden)?;

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
        let skip_flag = OverflowFlag::new(&ctx.stream)?;

        let has_gemm_work = dims.batch != 0 && dims.seq_len != 0;
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
            graph_acc: None,
            graph_apply: None,
            prepared_gemm_manifest: None,
            has_gemm_work,
            split_forward_route: None,
            grads_dirty: false,
            clip_partials,
            clip_scratch,
            step_opts: BackwardOpts::default(),
            skip_flag,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.split_forward_route = None;
        self.ssm_states.zero(&self.ctx.stream)?;
        self.k_states.zero(&self.ctx.stream)?;
        self.v_states.zero(&self.ctx.stream)?;
        self.angle_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Drop every captured body; the stream must be idle.
    fn drop_graphs(&mut self) {
        drop(self.graph.take());
        drop(self.graph_acc.take());
        drop(self.graph_apply.take());
    }

    /// The step options' tail (see [`StepTail`]).
    fn step_tail(&self) -> StepTail {
        StepTail::of(
            &self.step_opts,
            m3_control_region_geom(&self.cfg, self.dims.mamba_input_dim),
        )
    }

    /// Set the options of the fused step (see
    /// [`Mamba3GpuTrainer::set_step_opts`]).
    pub(crate) fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        if self.has_graph() && opts != self.step_opts {
            return Err(
                "Mamba-3 set_step_opts: the captured graph baked the step options in; \
                 drop_graph first"
                    .into(),
            );
        }
        self.step_opts = opts;
        Ok(())
    }

    /// After a fused step: whether its AdamW skipped and the pre-clip norm
    /// when the tail measured one. A norm that is not finite and that no
    /// overflow explains is an error, with the step counted before it given
    /// back.
    fn step_verdict(&mut self, overflowed: bool) -> Result<(bool, Option<f32>), String> {
        if !self.step_tail().measures() {
            return Ok((overflowed, None));
        }
        match read_tail(&self.ctx, &self.skip_flag, &self.clip_scratch, overflowed) {
            Ok((skipped, norm)) => Ok((skipped, Some(norm))),
            Err(e) => {
                self.adam.step -= 1;
                Err(e)
            }
        }
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;
        let manifest = self.prepared_gemm_manifest.ok_or_else(|| {
            "M3 f32 training graph capture requires one successful eager step".to_string()
        })?;
        self.graph = Some(self.capture_f32_body(&manifest, TrainBody::Step)?);
        self.graph_acc = Some(self.capture_f32_body(&manifest, TrainBody::Accumulate)?);
        // The apply runs no GEMM: it is held to an empty inventory, so a
        // GEMM that found its way into it would fail the capture.
        let no_gemm = self.ctx.record_eager_gemm_manifest(|| Ok(()))?;
        self.graph_apply = Some(self.capture_f32_body(&no_gemm, TrainBody::Apply)?);
        Ok(())
    }

    /// Capture one f32 training body (see [`TrainBody`]).
    fn capture_f32_body(
        &mut self,
        manifest: &PreparedGemmCaptureManifest,
        body: TrainBody,
    ) -> Result<GpuMamba3F32TrainingStepGraph, String> {
        let tail = self.step_tail();
        let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
        // The trainer owns every captured allocation and drops the graphs first.
        unsafe {
            GpuMamba3F32TrainingStepGraph::capture(
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
                    tail: Some(CaptureTail {
                        tail,
                        partials: &mut self.clip_partials,
                        coef: &mut self.clip_scratch,
                        skip,
                    }),
                },
                manifest,
                body,
            )
        }
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        // Every fused attempt abandons a pending split tape, even when it is rejected.
        self.split_forward_route = None;
        assert_eq!(input.len(), self.mamba_input.len(), "input shape mismatch");
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch"
        );
        // A step with a window open adds its micro-batch to the window and
        // applies it, as the split calls would close the window.
        if self.grads_dirty {
            self.accumulate(input, d_temporal)?;
            return self.apply_window();
        }
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        // The step counts before the replay, as the eager step counts it
        // before its AdamW; a step the tail skipped gives it back.
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        let replayed = if let Some(ref g) = self.graph {
            g.replay(&self.ctx, &self.f32_replay())?;
            true
        } else {
            self.step_eager()?;
            false
        };
        self.finish_step(replayed)
    }

    /// The f32 graphs' replay buffers.
    fn f32_replay(&self) -> Mamba3F32Replay<'_> {
        Mamba3F32Replay {
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
        }
    }

    /// The verdict of an update after it ran: the step options' tail, and
    /// the step counted before the update given back when AdamW skipped.
    fn finish_step(&mut self, replayed: bool) -> Result<StepMetrics, String> {
        let (skipped, grad_norm) = self.step_verdict(false)?;
        if skipped {
            self.adam.step -= 1;
        }
        Ok(StepMetrics {
            optimizer_stepped: !skipped,
            grad_norm,
            ..StepMetrics::plain(self.adam.step, replayed)
        })
    }

    /// One micro-batch of an accumulation window (see
    /// [`Mamba3GpuTrainer::accumulate`]).
    pub(crate) fn accumulate(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<BackwardMetrics, String> {
        self.split_forward_route = None;
        check_window_len("accumulate input", input.len(), self.mamba_input.len())?;
        check_window_len(
            "accumulate d_temporal",
            d_temporal.len(),
            self.d_temporal.len(),
        )?;
        if !self.grads_dirty {
            self.grads.zero(&self.ctx.stream)?;
            self.grads_dirty = true;
        }
        self.mamba_input.upload(&self.ctx.stream, input)?;
        self.d_temporal.upload(&self.ctx.stream, d_temporal)?;
        if let Some(g) = &self.graph_acc {
            g.replay(&self.ctx, &self.f32_replay())?;
        } else {
            self.eager_accumulate()?;
        }
        Ok(BackwardMetrics {
            step: self.adam.step,
            optimizer_stepped: false,
            grad_norm: None,
            loss_scale: None,
            overflow_skipped: None,
        })
    }

    /// One accumulating micro-batch eagerly: the forward and the backward
    /// into the open window, the GEMM inventory recorded for a capture.
    fn eager_accumulate(&mut self) -> Result<(), String> {
        if self.has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let Self {
            ctx,
            m3k,
            dims,
            weights,
            grads,
            acts,
            scratch,
            temporal,
            mamba_input,
            d_temporal,
            ssm_states,
            k_states,
            v_states,
            angle_states,
            prepared_gemm_manifest,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let manifest = ctx.record_eager_gemm_manifest(|| {
            gpu_forward_mamba3_backbone(
                &exec,
                temporal,
                acts,
                weights,
                mamba_input,
                GpuMamba3StateBufs {
                    ssm: ssm_states,
                    k: k_states,
                    v: v_states,
                    angle: angle_states,
                },
                scratch,
            )?;
            gpu_backward_mamba3_backbone(&exec, d_temporal, acts, weights, grads, scratch)
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// The open window's apply (see [`Mamba3GpuTrainer::apply_window`]).
    pub(crate) fn apply_window(&mut self) -> Result<StepMetrics, String> {
        if !self.grads_dirty {
            return Err("apply_window: no accumulation window is open".into());
        }
        self.split_forward_route = None;
        // The step counts before the update, as the fused step counts it;
        // a step the tail skipped gives it back.
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;
        let replayed = if let Some(g) = &self.graph_apply {
            g.replay(&self.ctx, &self.f32_replay())?;
            true
        } else {
            let tail = self.step_tail();
            let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
            let skip = tail.enqueue(
                &self.ctx,
                &mut self.grads.flat,
                (&mut self.clip_partials, &mut self.clip_scratch),
                (skip, 0),
            )?;
            self.eager_optimize(skip)?;
            false
        };
        self.grads_dirty = false;
        self.finish_step(replayed)
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
    fn eager_optimize(&mut self, skip: cudarc::driver::sys::CUdeviceptr) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.m3k
                .adamw_step_multi
                .get(crate::gpu::dtype::WeightDtype::F32),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
            skip,
        )
    }

    fn step_eager(&mut self) -> Result<(), String> {
        let has_gemm_work = self.has_gemm_work;
        if has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let tail = self.step_tail();
        let skip_ptr = self.skip_flag.stable_ptr(&self.ctx.stream);
        let Self {
            ctx,
            m3k,
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
            prepared_gemm_manifest,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let exec = M3Exec {
            ctx,
            kernels: m3k,
            dims,
        };
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba3_backbone(
                &exec,
                temporal,
                acts,
                weights,
                mamba_input,
                GpuMamba3StateBufs {
                    ssm: ssm_states,
                    k: k_states,
                    v: v_states,
                    angle: angle_states,
                },
                scratch,
            )?;
            gpu_backward_mamba3_backbone(&exec, d_temporal, acts, weights, grads, scratch)?;
            let skip = tail.enqueue(
                ctx,
                &mut grads.flat,
                (&mut *clip_partials, &mut *clip_scratch),
                (skip_ptr, 0),
            )?;
            step_multi(
                ctx,
                m3k.adamw_step_multi.get(WeightDtype::F32),
                multi_plan,
                adam,
                bias.ptr(),
                skip,
            )
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Split forward (see [`Mamba3GpuTrainer::forward`]).
    pub(crate) fn forward_split(
        &mut self,
        input: &[f32],
        temporal_out: &mut [f32],
    ) -> Result<(), String> {
        self.split_forward_route = None;
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
        self.split_forward_route = Some(self.ctx.gemm_route());
        Ok(())
    }

    /// Split backward + optimizer (see [`Mamba3GpuTrainer::backward_step`]),
    /// and the input gradient into `d_input` when one is asked for.
    pub(crate) fn backward_split(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
        d_input: Option<&mut [f32]>,
    ) -> Result<BackwardMetrics, String> {
        let Some(forward_route) = self.split_forward_route.take() else {
            return Err(
                "backward_step() without a pending forward() — the saved activations \
                 are stale or missing; call forward() first"
                    .into(),
            );
        };
        if self.ctx.gemm_route() != forward_route {
            return Err(
                "M3 split backward: GEMM route changed since forward; run forward again".into(),
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
        if let Some(d_input) = d_input {
            // The backward leaves the input projection's dX in the scratch,
            // or the backbone input's gradient in d_temporal without one.
            let src = if self.weights.input_proj_w.is_empty() {
                &self.d_temporal
            } else {
                &self.scratch.d_input_proj_dx
            };
            src.download(&self.ctx.stream, d_input)?;
            self.ctx
                .stream
                .synchronize()
                .map_err(|e| format!("input gradient download: {e:?}"))?;
        }

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
        let grad_norm = match norm_limit(clip_max_norm, step_skip_above) {
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
        self.eager_optimize(0)?;
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
            if let Some(b) = &g.norm_bias {
                lw.norm_bias = b.to_cpu(&self.ctx.stream)?;
            }
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
        if let Some(b) = &w.norm_f_bias {
            out.norm_f_bias = b.to_cpu(&self.ctx.stream)?;
        }
        for (mo, mg) in out.mlp.iter_mut().zip(&w.mlp) {
            mo.norm_weight = mg.norm_weight.to_cpu(&self.ctx.stream)?;
            if let Some(b) = &mg.norm_bias {
                mo.norm_bias = b.to_cpu(&self.ctx.stream)?;
            }
            mo.fc1_w = mg.fc1_w.to_cpu(&self.ctx.stream)?;
            mo.fc2_w = mg.fc2_w.to_cpu(&self.ctx.stream)?;
        }
        Ok(out)
    }
}

// The names these types carried before 0.8, when each family named its GPU
// types its own way.
#[deprecated(since = "0.8.0", note = "renamed to `Mamba3GpuTrainer`")]
pub type Mamba3Trainer = Mamba3GpuTrainer;
