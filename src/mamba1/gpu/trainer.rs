//! High-level Mamba SSM training API.
//!
//! Mirrors the shape of [`super::inference::MambaGpuBackbone`] but on the
//! training side. A single [`MambaGpuTrainer`] owns EVERY piece of state
//! needed to run a full training step:
//!
//!   * mixed-precision weights (master + compute shadow)
//!   * gradient arena
//!   * AdamW optimizer state + bias-factor device buffer
//!   * saved activations (`acts`) + scratch
//!   * recurrent state buffers
//!   * pre-allocated input / d_temporal upload buffers
//!   * the CUDA Graph holder (lazily captured)
//!
//! ## Usage
//! ```ignore
//! let session = TrainSessionCfg::new(input_dim, 8, 128); // 8 sequences of 128 tokens
//! let opts = GpuOptions::default().dtype(WeightDtype::Bf16);
//! let mut trainer = MambaGpuTrainer::new(&cpu_weights, cfg, session, opts)?;
//!
//! // First few steps run eagerly so cuBLAS / lazy CUDA resources settle.
//! for _ in 0..3 {
//!     trainer.step(&input, &d_temporal)?;
//! }
//! trainer.capture_graph()?;  // subsequent steps are graph-accelerated
//!
//! for _ in 0..n_steps {
//!     trainer.step(&input, &d_temporal)?;
//! }
//! ```
//!
//! ## Precision support
//! - `WeightDtype::Bf16`: full graph capture, sync_master_to_compute, AMP-style
//!   master weights in f32.
//! - `WeightDtype::F32`: full graph capture, no compute shadow (weights stay
//!   in f32 throughout). See `MambaTrainerF32`.
//! - `WeightDtype::Tf32`: the f32 trainer with deterministic TF32 GEMM
//!   products where a measured kernel exists; storage and checkpoints are
//!   those of f32.
//! - `WeightDtype::F16`: supported via the [`DynamicLossScaler`] + a
//!   device-side `scale_grads_skip_f32` kernel that conditionally zeros
//!   the grad arena on overflow, and an AdamW that skips on the overflow
//!   flag. The eager f16 step runs the captured body's sequence, so both
//!   skip an overflowed step as PyTorch `torch.cuda.amp.GradScaler` does.

use crate::config::ModelFamily;
use crate::gpu::grad_window::check_backward_opts;
use crate::gpu::options::GpuOptions;
use crate::gpu::train_state::TrainerState;
use cudarc::driver::PushKernelArg;

use crate::gpu::adamw::{
    AdamWBiasFactors, AdamWMultiPlan, GpuAdamW, build_multi_plan, m1_specs, m1_specs_mixed,
    step_multi,
};
use crate::gpu::blas::{TypedPtr, gpu_gemm_ex_backward_dx_typed};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer, GpuByteBuffer, GradSlice};
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
use crate::gpu::launch::{grid_1d_wide, launch_cast};
use crate::mamba1::config::MambaConfig;
use crate::mamba1::gpu::backward::gpu_backward_mamba_backbone;
use crate::mamba1::gpu::backward_mixed::gpu_backward_mamba_backbone_mixed;
use crate::mamba1::gpu::forward::{
    GpuMambaBackboneActs, GpuMambaDims, GpuMambaScratch, GpuRecurrentState,
    gpu_forward_mamba_backbone,
};
use crate::mamba1::gpu::forward_mixed::{
    GpuMambaBackboneMixedActs, GpuMambaMixedTrainScratch, gpu_forward_mamba_backbone_train_mixed,
};
use crate::mamba1::gpu::weights::GpuMambaTrainLayerWeights;
use crate::matrix::{Path, Request};

/// Recompute `a_neg = -exp(a_log)` from the current master weights after
/// AdamW has updated `a_log`. Writes to BOTH `a_neg_all` (consumed by the
/// backward kernels) and `state.a_neg_all` (consumed by the forward SSM
/// recurrence).
///
/// Must be called after every optimizer step — without it, forward and
/// backward read stale `a_neg` values from trainer construction time and
/// the `d_a_log` gradient never reaches the recurrence (silent no-op on
/// the A-matrix learning).
fn recompute_a_neg_all(
    ctx: &GpuCtx,
    master_layers: &[GpuMambaTrainLayerWeights],
    a_neg_all: &crate::gpu::buffers::GpuBuffer,
    state_a_neg_all: &crate::gpu::buffers::GpuBuffer,
    d_inner: usize,
    d_state: usize,
) -> Result<(), String> {
    let per_layer = d_inner * d_state;
    if per_layer == 0 {
        return Ok(());
    }
    let n = per_layer as i64;
    for (li, mw) in master_layers.iter().enumerate() {
        let src = mw.a_log.cached_ptr();
        // Both a_neg mirrors (backward-side + forward-side state) from
        // ONE kernel — same exp value stored twice, half the launches.
        let dst_a = a_neg_all.inner_at(li * per_layer);
        let dst_s = state_a_neg_all.inner_at(li * per_layer);
        let mut b1 = ctx.stream.launch_builder(&ctx.kernels.exp_negate2);
        b1.arg(&dst_a);
        b1.arg(&dst_s);
        b1.arg(&src);
        b1.arg(&n);
        unsafe { b1.launch(grid_1d_wide(per_layer)) }
            .map_err(|e| format!("exp_negate2 a_neg mirrors L{li}: {e:?}"))?;
    }
    Ok(())
}

use crate::gpu::loss_scaler::{
    DynamicLossScaler, OverflowFlag, UnscaleFactor, check_inf_nan_gpu, scale_grads_skip_gpu,
};
use crate::mamba1::gpu::training_graph::{
    GpuMambaF32TrainingStepGraph, GpuMambaTrainingStepGraph, MambaF32Capture, MambaF32Replay,
    MambaMixedCapture, MambaMixedReplay,
};
use crate::mamba1::gpu::weights::{GpuMambaGrads, GpuMambaTrainWeights};
use crate::mamba1::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;
use crate::mamba1::weights::MambaWeights;

pub use crate::gpu::train::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};

/// Internal precision-dispatch enum. Hidden behind [`MambaGpuTrainer`] so the
/// public API is a single struct with a single set of method names — caller
/// never matches `F32`/`Mixed` directly. Mirrors `inference::BackboneEngine`.
enum TrainerInner {
    F32(Box<MambaTrainerF32>),
    Mixed(Box<MambaTrainerMixed>),
}

/// High-level Mamba SSM training wrapper. Same shape as
/// [`super::inference::MambaGpuBackbone`]: one public struct, one method
/// per operation, dtype dispatch happens internally on the private enum.
pub struct MambaGpuTrainer {
    inner: TrainerInner,
    /// Width of an input row, from the session.
    input_dim: usize,
}

impl MambaGpuTrainer {
    /// Build a trainer for the session `session` names (shape and AdamW
    /// settings, see [`TrainSessionCfg::new`]) from `cpu_weights`, on the
    /// device, in the storage precision and in the GEMM mode `opts` names
    /// ([`GpuOptions::default`] is device 0, f32 and the mode
    /// `MAMBA_RS_GEMM_MODE` names). In the deterministic mode the trainer
    /// runs the Triad kernels; [`Self::ctx`] shows the route a captured graph
    /// binds. Shape, state-capacity, CUDA, upload or allocation errors are
    /// returned.
    pub fn new(
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        session: TrainSessionCfg,
        dtype: WeightDtype,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        // The kernels assume the dimensions the config validator checks
        // (a d_inner divisible by four above all); an unchecked config used
        // to reach the launchers and pick inconsistent kernel pairs. The
        // weights must have the config's shapes, and a model without an
        // input projection rows d_model wide.
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
        crate::mamba1::route::admit(&cfg, session.input_dim, &request)?;
        let inner = match dtype {
            WeightDtype::F32 | WeightDtype::Tf32 => TrainerInner::F32(Box::new(
                MambaTrainerF32::new_full(gpu_ordinal, cpu_weights, cfg, session, mode, dtype)?,
            )),
            WeightDtype::Bf16 | WeightDtype::F16 => TrainerInner::Mixed(Box::new(
                MambaTrainerMixed::new_full(gpu_ordinal, cpu_weights, cfg, session, dtype, mode)?,
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
            TrainerInner::F32(t) => t.ctx.f32_storage_dtype(),
            TrainerInner::Mixed(t) => t.dtype,
        }
    }

    /// Batch dimension fixed at construction; CUDA Graph capture binds
    /// device pointers for this exact `batch * seq_len` shape.
    pub fn batch(&self) -> usize {
        match &self.inner {
            TrainerInner::F32(t) => t.batch,
            TrainerInner::Mixed(t) => t.batch,
        }
    }

    /// Sequence length fixed at construction. See [`Self::batch`].
    pub fn seq_len(&self) -> usize {
        match &self.inner {
            TrainerInner::F32(t) => t.seq_len,
            TrainerInner::Mixed(t) => t.seq_len,
        }
    }

    /// The model's configuration.
    pub fn config(&self) -> &MambaConfig {
        match &self.inner {
            TrainerInner::F32(t) => &t.cfg,
            TrainerInner::Mixed(t) => &t.cfg,
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
            TrainerInner::F32(t) => &t.ctx,
            TrainerInner::Mixed(t) => &t.ctx,
        }
    }

    /// `true` once [`Self::capture_graph`] has been called and the
    /// captured graph is ready for replay on subsequent [`Self::step`]s.
    pub fn has_graph(&self) -> bool {
        match &self.inner {
            TrainerInner::F32(t) => t.graph.is_some(),
            TrainerInner::Mixed(t) => t.has_graph(),
        }
    }

    /// Set the AdamW learning rate for subsequent EAGER steps. Errs while a
    /// captured graph exists: the lr is baked BY VALUE into the captured
    /// AdamW kernel at capture time, so a bare field write would be a
    /// silent no-op under replay — the exact silent-wrong class this
    /// crate's history punishes. Call [`Self::drop_graph`] first, then
    /// `set_lr`, then re-[`Self::capture_graph`] if graph stepping is
    /// still wanted.
    pub fn set_lr(&mut self, lr: f32) -> Result<(), String> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(format!("set_lr: invalid learning rate {lr}"));
        }
        // lr rides the 3-element device bias buffer ({bc1, bc2, lr})
        // and is re-uploaded before every step — a schedule now works
        // under a captured graph (the old per-tensor kernels baked lr
        // by value at capture; the fused kernel reads the buffer).
        match &mut self.inner {
            TrainerInner::F32(t) => t.adam.lr = lr,
            TrainerInner::Mixed(t) => t.adam.lr = lr,
        }
        Ok(())
    }

    /// Current AdamW learning rate.
    pub fn lr(&self) -> f32 {
        match &self.inner {
            TrainerInner::F32(t) => t.adam.lr,
            TrainerInner::Mixed(t) => t.adam.lr,
        }
    }

    /// Download the full optimizer state (Adam moments, step counter,
    /// update hyperparameters) for a bit-continuous resume. Checkpoints
    /// that carry only weights silently re-warm Adam from zero on load —
    /// the resumed run then provably diverges from the unbroken one.
    pub fn optimizer_state(&self) -> Result<crate::gpu::adamw::AdamWStateBlob, String> {
        match &self.inner {
            TrainerInner::F32(t) => t.adam.export_state(&t.ctx.stream),
            TrainerInner::Mixed(t) => t.adam.export_state(&t.ctx.stream),
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
            TrainerInner::F32(t) => {
                t.adam.import_state(&t.ctx.stream, blob)?;
                // The blob adopts wd/no-decay wholesale — the chunk table
                // bakes both, so it must follow.
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m1_specs(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
                Ok(())
            }
            TrainerInner::Mixed(t) => {
                t.adam.import_state(&t.ctx.stream, blob)?;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m1_specs_mixed(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
                Ok(())
            }
        }
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
    /// use mamba_rs::mamba1::gpu::{buffers::GpuBuffer, trainer::MambaGpuTrainer};
    /// fn replace_arena(trainer: &mut MambaGpuTrainer, replacement: GpuBuffer) {
    ///     let _ = std::mem::replace(trainer.grad_arena(), replacement);
    /// }
    /// ```
    pub fn grad_arena(&self) -> &GpuBuffer {
        match &self.inner {
            TrainerInner::F32(t) => &t.grads.flat,
            TrainerInner::Mixed(t) => &t.grads.flat,
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
            TrainerInner::F32(t) => t.grads.flat.upload(&t.ctx.stream, values),
            TrainerInner::Mixed(t) => t.grads.flat.upload(&t.ctx.stream, values),
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

    /// [`Self::apply_step_with`] plus the control-channel clip (see
    /// [`BackwardOpts::control_clip_max_norm`]).
    pub fn apply_step_full(
        &mut self,
        clip_max_norm: Option<f32>,
        step_skip_above: Option<f32>,
        control_clip_max_norm: Option<f32>,
    ) -> Result<BackwardMetrics, String> {
        match &mut self.inner {
            TrainerInner::F32(t) => {
                if !t.grads_dirty {
                    return Err("apply_step without an open accumulation window — run \
                         backward_step(accumulate_only = true) first"
                        .into());
                }
                t.apply_step_inner(clip_max_norm, step_skip_above, control_clip_max_norm)
            }
            TrainerInner::Mixed(t) => {
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
            TrainerInner::F32(t) => scale_grads(&t.ctx, &mut t.grads.flat, inv_w)?,
            TrainerInner::Mixed(t) => scale_grads(&t.ctx, &mut t.grads.flat, inv_w)?,
        }
        self.apply_step_full(clip, opts.step_skip_above, opts.control_clip_max_norm)
    }

    /// Everything this trainer needs beside its weights to resume bit for
    /// bit: the optimizer state, the f16 loss scale and the carried
    /// recurrence. [`TrainerState::save`] writes it to a file.
    pub fn trainer_state(&self) -> Result<TrainerState, String> {
        let rec = self.recurrent_state()?;
        Ok(TrainerState {
            family: ModelFamily::Mamba1,
            optimizer: self.optimizer_state()?,
            scaler: self.scaler_state(),
            recurrent: vec![
                ("conv".into(), rec.conv_states),
                ("ssm".into(), rec.ssm_states),
            ],
        })
    }

    /// Load a state a trainer of this family gave with
    /// [`Self::trainer_state`]. The learning rate is not part of it.
    pub fn load_trainer_state(&mut self, state: &TrainerState) -> Result<(), String> {
        state.check_family(ModelFamily::Mamba1)?;
        let conv = state.recurrent_tensor("conv")?;
        let ssm = state.recurrent_tensor("ssm")?;
        self.load_optimizer_state(&state.optimizer)?;
        if let Some((scale, tracker)) = state.scaler {
            self.load_scaler_state(scale, tracker)?;
        }
        self.load_recurrent_parts(conv, ssm)
    }

    /// Download the carried recurrence (conv + SSM state) — the TBPTT
    /// window handoff and checkpointed-resume counterpart of
    /// [`Self::optimizer_state`].
    pub fn recurrent_state(
        &self,
    ) -> Result<crate::mamba1::gpu::forward::RecurrentStateBlob, String> {
        match &self.inner {
            TrainerInner::F32(t) => t.state.export_state(&t.ctx.stream),
            TrainerInner::Mixed(t) => t.state.export_state(&t.ctx.stream),
        }
    }

    /// Upload a previously exported recurrence.
    pub fn load_recurrent_state(
        &mut self,
        blob: &crate::mamba1::gpu::forward::RecurrentStateBlob,
    ) -> Result<(), String> {
        self.load_recurrent_parts(&blob.conv_states, &blob.ssm_states)
    }

    fn load_recurrent_parts(&mut self, conv: &[f32], ssm: &[f32]) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.state.import_parts(&t.ctx.stream, conv, ssm),
            TrainerInner::Mixed(t) => t.state.import_parts(&t.ctx.stream, conv, ssm),
        }
    }

    /// Toggle the AdamW no-decay parameter groups (`a_log` / `d_param` /
    /// `dt_proj_b` / RMSNorm scales get `weight_decay = 0`; the reference
    /// marks `A_log` and `D`, the bias and the norm scales follow the usual
    /// practice; decaying `a_log` pulls every decay rate toward A = -1 over
    /// long runs). Default ON since 0.7.4; OFF reproduces the
    /// decay-everything numbers of 0.7.3 and earlier. Errs while a captured graph exists: the decay
    /// coefficient is baked by value into the captured per-tensor AdamW
    /// launches.
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
            TrainerInner::F32(t) => {
                t.adam.reference_no_decay = on;
                // The per-tensor decay rides the chunk table — rebuild it.
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m1_specs(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
            }
            TrainerInner::Mixed(t) => {
                t.adam.reference_no_decay = on;
                t.multi_plan = build_multi_plan(
                    &t.ctx.stream,
                    &t.adam,
                    t.grads.flat.cached_ptr(),
                    &m1_specs_mixed(&t.weights, &t.grads),
                    t.adam.reference_no_decay,
                    t.adam.weight_decay,
                )?;
            }
        }
        Ok(())
    }

    /// Drop any captured step graph so `drop_graph -> set_lr ->
    /// capture_graph` is expressible. Subsequent [`Self::step`]s run
    /// eagerly until re-captured.
    pub fn drop_graph(&mut self) {
        match &mut self.inner {
            TrainerInner::F32(t) => {
                let _ = t.ctx.stream.synchronize();
                t.drop_graphs();
            }
            TrainerInner::Mixed(t) => {
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
            TrainerInner::F32(t) => t.set_step_opts(opts),
            TrainerInner::Mixed(t) => t.set_step_opts(opts),
        }
    }

    /// The options of the fused [`Self::step`].
    pub fn step_opts(&self) -> BackwardOpts {
        match &self.inner {
            TrainerInner::F32(t) => t.step_opts,
            TrainerInner::Mixed(t) => t.step_opts,
        }
    }

    /// Reset the recurrent SSM + conv states to zero. Call between
    /// independent training sequences (e.g. on episode boundary in RL
    /// or document boundary in LM).
    pub fn reset_state(&mut self) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.reset_state(),
            TrainerInner::Mixed(t) => t.reset_state(),
        }
    }

    /// Record the full training step (forward + backward + AdamW + sync)
    /// into a CUDA Graph. Run at least one warmup [`Self::step`] before
    /// capturing so cuBLAS has settled on its kernel selection. After
    /// capture, every weight / gradient / optimizer pointer is asserted
    /// stable on each replay; reallocating any of them invalidates the
    /// graph and the next [`Self::step`] will return an error.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.capture_graph(),
            TrainerInner::Mixed(t) => t.capture_graph(),
        }
    }

    /// Run one training step on `(input, d_temporal)`. `input` must have
    /// length `batch * seq_len * input_dim`; `d_temporal` must have
    /// length `batch * seq_len * d_model` (gradient w.r.t. the FULL
    /// post-norm_f temporal output sequence, not just the last position).
    /// Returns [`StepMetrics`] with overflow / replay flags. With an
    /// accumulation window open the step adds its micro-batch to the window
    /// and applies it, as [`Self::accumulate`] then [`Self::apply_window`]
    /// would.
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.step(input, d_temporal),
            TrainerInner::Mixed(t) => t.step(input, d_temporal),
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
            TrainerInner::F32(t) => t.accumulate(input, d_temporal),
            TrainerInner::Mixed(t) => t.accumulate(input, d_temporal),
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
            TrainerInner::F32(t) => t.apply_window(),
            TrainerInner::Mixed(t) => t.apply_window(),
        }
    }

    /// Eager forward half of the split step: runs the training forward and
    /// writes the FULL `batch * seq_len * d_model` POST-norm_f temporal
    /// output into caller-owned `temporal_out` — f32 on ALL dtypes
    /// (bf16/f16 outputs are upcast on-device before download).
    /// Stream-synchronized on return, so `temporal_out` is valid host data.
    ///
    /// Advances the recurrent conv/SSM state — call [`Self::reset_state`]
    /// between independent sequences. Always runs the eager path, even when
    /// a step graph is captured: the split exists so a caller-side loss can
    /// run between forward and backward, which a captured whole-step graph
    /// cannot express. The fused [`Self::step`] remains the
    /// graph-capturable fast path and is byte-identical in numerics (both
    /// compose the same eager bodies).
    pub fn forward(&mut self, input: &[f32], temporal_out: &mut [f32]) -> Result<(), String> {
        match &mut self.inner {
            TrainerInner::F32(t) => t.forward_split(input, temporal_out),
            TrainerInner::Mixed(t) => t.forward_split(input, temporal_out),
        }
    }

    /// Backward + optimizer half of the split step. `d_temporal` is the
    /// gradient w.r.t. the IMMEDIATELY PRECEDING [`Self::forward`]'s
    /// temporal output (`batch * seq_len * d_model`, f32 — exactly the
    /// tensor `temporal_out` held). Errs when no forward is pending (the
    /// saved activations would be stale or missing). Always eager.
    pub fn backward_step(
        &mut self,
        d_temporal: &[f32],
        opts: BackwardOpts,
    ) -> Result<BackwardMetrics, String> {
        check_backward_opts(&opts, "Mamba trainer")?;
        match &mut self.inner {
            TrainerInner::F32(t) => t.backward_split(d_temporal, opts, None),
            TrainerInner::Mixed(t) => t.backward_split(d_temporal, opts, None),
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
        check_backward_opts(&opts, "Mamba trainer")?;
        let want = self.batch() * self.seq_len() * self.input_dim();
        if d_input.len() != want {
            return Err(format!(
                "backward_step_input_grad: d_input holds {} values; the window's input has \
                 {want}",
                d_input.len()
            ));
        }
        match &mut self.inner {
            TrainerInner::F32(t) => t.backward_split(d_temporal, opts, Some(d_input)),
            TrainerInner::Mixed(t) => t.backward_split(d_temporal, opts, Some(d_input)),
        }
    }

    /// Download the f32 master weights to CPU for checkpointing. Always
    /// f32 regardless of the compute dtype — mixed-precision training
    /// keeps a separate master copy that the optimizer updates.
    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        match &self.inner {
            TrainerInner::F32(t) => t.snapshot_master(),
            TrainerInner::Mixed(t) => t.snapshot_master(),
        }
    }

    /// Download the SSM `a_neg_all` buffer. Test / debug only — see the
    /// same-named method on the inner trainer for rationale.
    #[doc(hidden)]
    pub fn debug_a_neg_all(&self) -> Result<Vec<f32>, String> {
        match &self.inner {
            TrainerInner::F32(t) => t.debug_a_neg_all(),
            TrainerInner::Mixed(t) => t.debug_a_neg_all(),
        }
    }

    /// Serialize the dynamic loss scaler state for checkpoint resume.
    /// Returns `Some((scale, growth_tracker))` only for f16 training where
    /// the scaler is active; `None` for bf16 / f32 (scaler is disabled).
    ///
    /// Paired with [`Self::load_scaler_state`]. Saving this alongside the
    /// master weights and restoring on resume avoids re-paying the ~2000
    /// steps of scale discovery and the overflow-spiral risk of restarting
    /// at `init_scale = 65536` when training had converged to a lower
    /// stable scale.
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        match &self.inner {
            TrainerInner::F32(_) => None,
            TrainerInner::Mixed(t) => t.scaler_state(),
        }
    }

    /// Restore the dynamic loss scaler state saved via [`Self::scaler_state`].
    /// No effect for non-f16 trainers; a scale that is not finite and
    /// positive is an error in every precision.
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        DynamicLossScaler::check_scale(scale)?;
        if let TrainerInner::Mixed(ref mut t) = self.inner {
            t.load_scaler_state(scale, growth_tracker)?;
        }
        Ok(())
    }
}

/// The CONTROL region of the M1 grad arena: the step-size columns of every
/// layer's x_proj gradient, then its dt_proj weight and bias and A_log,
/// which lie one after another. These are the gradients of the step size
/// and the decay, whose magnitude grows with the sequence length as the
/// Mamba-3 control channels' does. The offsets come from the arena's own
/// views, so they follow its layout with or without an input projection.
/// An error unless a host slice holds the window's `want` values.
fn check_window_len(what: &str, got: usize, want: usize) -> Result<(), String> {
    if got != want {
        return Err(format!("{what}: {got} values, the window holds {want}"));
    }
    Ok(())
}

fn m1_control_region_geom(grads: &GpuMambaGrads, cfg: &MambaConfig) -> GradRegionGeom {
    let base = grads.flat.cached_ptr();
    let at = |s: &GradSlice| ((s.ptr() - base) / std::mem::size_of::<f32>() as u64) as i64;
    let l0 = &grads.layers[0];
    debug_assert_eq!(
        at(&l0.dt_proj_b),
        at(&l0.dt_proj_w) + l0.dt_proj_w.len() as i64
    );
    debug_assert_eq!(at(&l0.a_log), at(&l0.dt_proj_b) + l0.dt_proj_b.len() as i64);
    GradRegionGeom {
        n_layers: cfg.n_layers as i32,
        layer_stride: grads
            .layers
            .get(1)
            .map_or(0, |l1| at(&l1.norm_weight) - at(&l0.norm_weight)),
        a_off: at(&l0.x_proj_w),
        a_rows: cfg.d_inner() as i32,
        a_cols: cfg.dt_rank() as i32,
        a_row_stride: cfg.xdbl_dim() as i64,
        b_off: at(&l0.dt_proj_w),
        b_len: (l0.dt_proj_w.len() + l0.dt_proj_b.len() + l0.a_log.len()) as i32,
        c_off: 0,
        c_len: 0,
    }
}

/// bf16 mixed-precision training inner (master f32 + compute bf16 shadow +
/// sync_master_to_compute each step).
pub(crate) struct MambaTrainerMixed {
    ctx: GpuCtx,
    cfg: MambaConfig,
    batch: usize,
    seq_len: usize,
    dtype: WeightDtype,

    // Weights + optimizer state.
    pub weights: GpuMambaTrainMixedWeights,
    pub grads: GpuMambaGrads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    multi_plan: AdamWMultiPlan,

    // Activations + scratch (forward saves → backward reads).
    acts: GpuMambaBackboneMixedActs,
    scratch: GpuMambaMixedTrainScratch,

    // Recurrent state + per-training standalone a_neg_all.
    state: GpuRecurrentState,
    a_neg_all: GpuBuffer,

    // Upload buffers (stable pointers — reused every step).
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,

    /// Dedicated f32 staging for the split forward's readback: the typed
    /// post-norm_f output (`scratch.temporal_typed`) is clobbered by the
    /// backward's B0 pass, and `DtypedBuf::download_f32` heap-allocates a
    /// full-size Vec per call — so `forward_split` upcasts into this
    /// pre-allocated buffer and downloads from it.
    temporal_f32: GpuBuffer,
    /// Route that produced the saved split-forward activations.
    split_forward_route: Option<crate::gpu::context::GemmRoute>,
    /// True while an `accumulate_only` backward window is open: the next
    /// backward must NOT zero the arena, and the fused `step()` must refuse
    /// to run (its body zeroes the arena and would silently discard the
    /// accumulated gradients).
    grads_dirty: bool,
    /// Device partials for the deterministic grad-clip norm (512 f64).
    clip_partials: GpuByteBuffer,
    /// Host mirror of the partials — pre-allocated (zero-alloc hot path).
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,
    /// The options of the fused step, eager or captured.
    step_opts: BackwardOpts,
    /// The skip flag a measured step tail writes (see [`StepTail`]).
    skip_flag: OverflowFlag,

    // Lazily-populated CUDA Graph. None → eager path; Some → replayed.
    // Always None for f16 (loss-scaler overflow check requires CPU readback,
    // which breaks graph capture).
    graph: Option<GpuMambaTrainingStepGraph>,
    /// The accumulating micro-batch and the window's apply, captured
    /// beside the step.
    graph_acc: Option<GpuMambaTrainingStepGraph>,
    graph_apply: Option<GpuMambaTrainingStepGraph>,
    prepared_gemm_manifest: Option<PreparedGemmCaptureManifest>,

    // f16 AMP loss scaler — populated for `WeightDtype::F16`, None otherwise.
    // When present, every step scales d_temporal by `scaler.scale()` before
    // backward, then checks the grad arena for inf/nan and either unscales
    // and runs AdamW or skips the step and backs off the scale.
    scaler: Option<DynamicLossScaler>,
    overflow_flag: Option<OverflowFlag>,
    /// Persistent device buffer for the scaled d_temporal (kept here so its
    /// pointer is stable across steps).
    d_temporal_scaled: Option<GpuBuffer>,
    /// The input projection's dX in the compute precision, allocated by the
    /// first backward that asks for the input gradient.
    input_grad_half: Option<DtypedBuf>,
    /// f16 CUDA Graph. Captured body: forward + backward (with
    /// scaled d_temporal) + check_inf_nan + scale_grads_skip + AdamW + sync.
    /// CPU writes the next-step `1/loss_scale` into [`Self::unscale_factor`]
    /// before each replay; the captured `scale_grads_skip` kernel reads it
    /// via a stable device pointer baked at capture time.
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
    /// Pointer-stability snapshots for the f16 graph. The three device
    /// buffers below are baked into the captured kernels; if any of them
    /// is reallocated between capture and replay, the graph silently reads
    /// freed memory. Asserted on every replay.
    captured_f16_bias_ptr: u64,
    captured_f16_unscale_ptr: u64,
    captured_f16_overflow_ptr: u64,
    captured_f16_grads_ptr: u64,
    captured_f16_dt_scaled_ptr: u64,
    // the f16 graph presizes these two
    // scratches but never asserted them at replay - the one graph without
    // the guard the bf16 graph already has. Plus the flag snapshot.
    captured_f16_half_staging_ptr: u64,
    captured_f16_bi_upcast_ptrs: [u64; 3],
    captured_f16_gemm_flags: crate::gpu::context::GemmRoute,
    // Pinned host staging for the per-step H2D uploads. The pin
    // turns the copies into true async DMA; the guard event serializes
    // staging-buffer reuse against the previous step's in-flight copy
    // (rewriting a pinned source mid-DMA is silent wrong input).
    pin_input: crate::gpu::buffers::PinnedHostBuf,
    pin_dtemp: crate::gpu::buffers::PinnedHostBuf,
    upload_guard: cudarc::driver::CudaEvent,
}

impl Drop for MambaTrainerMixed {
    fn drop(&mut self) {
        let _ = self.ctx.stream.synchronize();
        self.drop_graphs();
    }
}

impl MambaTrainerMixed {
    /// Upload through the pinned stage. Waits out the previous
    /// step's DMA (guard event) before rewriting the staging buffer,
    /// then records the guard after the enqueue. `which`: 0 = input into
    /// mamba_input, 1 = d_temporal into d_temporal.
    fn staged_upload(&mut self, which: usize, data: &[f32]) -> Result<(), String> {
        self.upload_guard
            .synchronize()
            .map_err(|e| format!("upload guard sync: {e:?}"))?;
        let (pin, dst) = match which {
            0 => (&mut self.pin_input, &mut self.mamba_input),
            1 => (&mut self.pin_dtemp, &mut self.d_temporal),
            // 2 = f16 lane: d_temporal into the pre-scale staging buffer.
            _ => (
                &mut self.pin_dtemp,
                self.d_temporal_scaled.as_mut().expect("f16 dt_scaled"),
            ),
        };
        pin.as_mut_slice()[..data.len()].copy_from_slice(data);
        dst.upload(&self.ctx.stream, &pin.as_slice()[..data.len()])?;
        self.upload_guard
            .record(&self.ctx.stream)
            .map_err(|e| format!("upload guard record: {e:?}"))
    }

    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        session: TrainSessionCfg,
        dtype: WeightDtype,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        let TrainSessionCfg {
            input_dim,
            batch,
            seq_len,
            lr,
            weight_decay,
        } = session;
        assert!(
            matches!(dtype, WeightDtype::Bf16 | WeightDtype::F16),
            "MambaTrainerMixed accepts Bf16 or F16; got {dtype:?}"
        );

        let device = GpuDevice::new(gpu_ordinal)?;
        let state_cap = crate::gpu::kernels::state_capacity(cfg.d_state)?;
        let role = GemmRole::triad(dtype);
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(&device, state_cap, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(&device, state_cap, role)?,
        };

        let weights = GpuMambaTrainMixedWeights::from_cpu(&ctx.stream, cpu_weights, &cfg, dtype)?;

        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let n_layers = cfg.n_layers;

        let dims = GpuMambaDims {
            batch,
            d_model: cfg.d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers,
            scan_mode: cfg.scan_mode,
            rms_norm_eps: cfg.rms_norm_eps,
            block_norm: cfg.block_norm,
            bcdt_rms_eps: cfg.bcdt_rms_eps,
        };
        dims.check_for(&ctx)?;

        let hidden = cfg.mlp_hidden();
        let acts = GpuMambaBackboneMixedActs::new(&ctx.stream, &dims, dtype)?.with_mlp(
            &ctx.stream,
            &dims,
            hidden,
        )?;
        let scratch = GpuMambaMixedTrainScratch::new(&ctx.stream, &dims, dtype)?
            .with_mlp(&ctx.stream, hidden)?;

        // Seed recurrent state: conv/ssm zero; a_neg is recomputed from
        // the uploaded a_log by the SAME GPU kernel the post-step refresh
        // uses. Deriving it on the CPU here (libm exp vs the device expf)
        // differs by ULPs, so a trainer rebuilt from a checkpoint would
        // start from slightly different a_neg values than the unbroken
        // run it resumes — breaking bit-continuity for the first window.
        let a_neg_all = GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?;

        // conv/ssm states are per-sample: forward indexes layers with a
        // batch * d_inner * d_conv (resp. d_state) per-layer stride.
        let state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * batch * d_inner * d_conv)?,
            ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * batch * d_inner * d_state)?,
            a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
        };
        recompute_a_neg_all(
            &ctx,
            &weights.master.layers,
            &a_neg_all,
            &state.a_neg_all,
            d_inner,
            d_state,
        )?;

        let mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let temporal_f32 = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let grads = GpuMambaGrads::new_sized(
            &ctx.stream,
            &cfg,
            input_dim,
            !cpu_weights.input_proj_w.is_empty(),
        )?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;
        // Fused-AdamW chunk table: static for the trainer's lifetime;
        // rebuilt only when the decay grouping changes (the wd rides the
        // table per tensor).
        let multi_plan = build_multi_plan(
            &ctx.stream,
            &adam,
            grads.flat.cached_ptr(),
            &m1_specs_mixed(&weights, &grads),
            adam.reference_no_decay,
            adam.weight_decay,
        )?;

        // f16 needs the dynamic loss scaler + a separate scratch buffer for
        // the scaled d_temporal (so the original caller-provided values stay
        // untouched). bf16 has the same dynamic range as f32 and skips both.
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

        let pin_input = crate::gpu::buffers::PinnedHostBuf::zeroed(batch * seq_len * input_dim)?;
        let pin_dtemp = crate::gpu::buffers::PinnedHostBuf::zeroed(batch * seq_len * cfg.d_model)?;
        let upload_guard = ctx
            .stream
            .context()
            .new_event(None)
            .map_err(|e| format!("upload guard event: {e:?}"))?;
        let initial_gemm_route = ctx.gemm_route();
        let has_gemm_work = batch != 0
            && seq_len != 0
            && (weights.compute.input_proj_w.len_elems() != 0 || cfg.n_layers != 0);
        Ok(Self {
            ctx,
            cfg,
            batch,
            seq_len,
            dtype,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal,
            temporal_f32,
            split_forward_route: None,
            grads_dirty: false,
            clip_partials,
            clip_scratch,
            step_opts: BackwardOpts::default(),
            skip_flag,
            graph: None,
            graph_acc: None,
            graph_apply: None,
            prepared_gemm_manifest: None,
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
            // Sentinel zeros — overwritten in capture_graph_f16; never used
            // before the graph is captured (gated by `if graph_f16.is_some()`).
            captured_f16_bias_ptr: 0,
            captured_f16_unscale_ptr: 0,
            captured_f16_overflow_ptr: 0,
            captured_f16_grads_ptr: 0,
            captured_f16_dt_scaled_ptr: 0,
            captured_f16_half_staging_ptr: 0,
            captured_f16_bi_upcast_ptrs: [0; 3],
            captured_f16_gemm_flags: initial_gemm_route,
            pin_input,
            pin_dtemp,
            upload_guard,
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
            m1_control_region_geom(&self.grads, &self.cfg),
        )
    }

    /// Set the options of the fused step (see
    /// [`MambaGpuTrainer::set_step_opts`]).
    pub(crate) fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        if self.has_graph() && opts != self.step_opts {
            return Err(
                "Mamba-1 set_step_opts: the captured graph baked the step options in; \
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

    fn presize_prepared_gemm_scratch(&self) -> Result<(), String> {
        let input_dim = self.mamba_input.len() / (self.batch * self.seq_len);
        self.ctx.presize_bi_upcast_scratch_for_train_with_input(
            &self.cfg,
            self.batch,
            self.seq_len,
            input_dim,
            self.dtype,
        )
    }

    /// Reset recurrent state (conv_states + ssm_states) to zero. Keeps
    /// `a_neg_all` populated — it's a fixed function of the current
    /// weights and must survive resets.
    pub fn reset_state(&mut self) -> Result<(), String> {
        self.split_forward_route = None;
        self.state.conv_states.zero(&self.ctx.stream)?;
        self.state.ssm_states.zero(&self.ctx.stream)?;
        Ok(())
    }

    /// Download the current `a_neg_all` buffer used by the SSM backward
    /// kernel. Exposed for regression tests verifying that `a_neg` is
    /// refreshed from the updated `a_log` after AdamW; it once was not, and
    /// the optimizer trained an A matrix the kernels never read.
    #[doc(hidden)]
    pub fn debug_a_neg_all(&self) -> Result<Vec<f32>, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("debug_a_neg_all sync: {e:?}"))?;
        self.a_neg_all.to_cpu(&self.ctx.stream)
    }

    /// Serialize dynamic loss scaler state. `None` when scaler is disabled
    /// (bf16 / f32). See [`crate::gpu::loss_scaler::DynamicLossScaler::state`].
    pub fn scaler_state(&self) -> Option<(f32, u32)> {
        self.scaler.as_ref().map(|s| s.state())
    }

    /// Restore scaler state from a prior `scaler_state()`. No effect if the
    /// scaler is disabled (bf16 / f32 trainer).
    pub fn load_scaler_state(&mut self, scale: f32, growth_tracker: u32) -> Result<(), String> {
        if let Some(ref mut s) = self.scaler {
            s.load_state(scale, growth_tracker)?;
            // Keep the on-device `unscale_factor` consistent with the
            // restored CPU state so the very next f16 step uses the right
            // unscale multiplier. Without this the first post-load step
            // would unscale with the old (init_scale-derived) value.
            if let Some(ref mut uf) = self.unscale_factor {
                uf.write(&self.ctx.stream, 1.0 / s.scale())?;
            }
        }
        Ok(())
    }

    /// Capture the training-step CUDA Graph. Call once after at least one
    /// warmup [`Self::step`] so cuBLAS has selected its kernels and lazy
    /// resources have settled.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if matches!(self.dtype, WeightDtype::F16) {
            return self.capture_graph_f16();
        }
        // Make sure the bias buffer holds something finite — capture_into_graph
        // will record the AdamW kernel reading from it. Real values are
        // overwritten per step by `step()`.
        self.bias.write(&self.ctx.stream, 1.0, 1.0, self.adam.lr)?;
        let manifest = self.prepared_gemm_manifest.ok_or_else(|| {
            "M1 mixed training graph capture requires one successful eager step".to_string()
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
    ) -> Result<GpuMambaTrainingStepGraph, String> {
        let tail = self.step_tail();
        let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
        // The trainer owns every captured allocation and drops the graphs first.
        unsafe {
            GpuMambaTrainingStepGraph::capture(
                &self.ctx,
                &self.cfg,
                MambaMixedCapture {
                    train_w: &mut self.weights,
                    adam: &self.adam,
                    bias: &self.bias,
                    multi_plan: &self.multi_plan,
                    grads: &mut self.grads,
                    acts: &mut self.acts,
                    scratch: &mut self.scratch,
                    a_neg_all: &self.a_neg_all,
                    mamba_input: &self.mamba_input,
                    d_temporal: &mut self.d_temporal,
                    state: &mut self.state,
                    tail: Some(CaptureTail {
                        tail,
                        partials: &mut self.clip_partials,
                        coef: &mut self.clip_scratch,
                        skip,
                    }),
                },
                self.batch,
                self.seq_len,
                manifest,
                body,
            )
        }
    }

    /// Run one training step. For bf16 this is the existing
    /// forward+backward+AdamW+sync path (graph-accelerated when captured).
    /// For f16 the path runs eager only and goes through the dynamic loss
    /// scaler (scale d_temporal → backward → check overflow → unscale +
    /// step OR skip + back off).
    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        // Every fused attempt abandons a pending split tape, even when it is rejected.
        self.split_forward_route = None;
        assert_eq!(
            input.len(),
            self.mamba_input.len(),
            "input shape mismatch: expected {} got {}",
            self.mamba_input.len(),
            input.len()
        );
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected {} got {}",
            self.d_temporal.len(),
            d_temporal.len()
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

        // bf16 path: existing graph / eager dispatch. The step counts
        // before the replay, as the eager step counts it before its AdamW;
        // a step the tail skipped gives it back.
        self.staged_upload(0, input)?;
        self.staged_upload(1, d_temporal)?;
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

    /// Split forward (see [`MambaGpuTrainer::forward`]): eager forward, upcast
    /// the typed post-norm_f output into the dedicated f32 staging buffer,
    /// sync, download into `temporal_out`.
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
            self.temporal_f32.len(),
            "temporal_out shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.temporal_f32.len(),
            temporal_out.len(),
        );
        self.staged_upload(0, input)?;
        self.eager_forward()?;
        {
            let typed = &self.scratch.temporal_typed;
            let kernel = match typed.dtype() {
                WeightDtype::Bf16 => &self.ctx.kernels.cast_bf16_to_f32,
                WeightDtype::F16 => &self.ctx.kernels.cast_f16_to_f32,
                WeightDtype::F32 | WeightDtype::Tf32 => {
                    return Err("forward_split: unexpected f32 temporal_typed in Mixed".into());
                }
            };
            launch_cast(
                &self.ctx.stream,
                kernel,
                (self.temporal_f32.cached_ptr(), 4),
                (typed.cached_ptr(), typed.dtype().size_bytes()),
                self.temporal_f32.len(),
                format_args!("forward_split: temporal upcast"),
            )?;
        }
        // Sync AFTER the async D2H enqueue: cuMemcpyDtoHAsync into
        // pageable memory happens to block in the driver, but the
        // ordering contract must not lean on that — a pinned host
        // buffer here would read stale bytes with the old order.
        self.temporal_f32.download(&self.ctx.stream, temporal_out)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("forward_split sync: {e:?}"))?;
        self.split_forward_route = Some(self.ctx.gemm_route());
        Ok(())
    }

    /// Split backward + optimizer (see [`MambaGpuTrainer::backward_step`]),
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
        if forward_route != self.ctx.gemm_route() {
            return Err(format!(
                "backward_step(): GEMM route changed since forward() \
                 ({forward_route:?} -> {:?}); re-run forward()",
                self.ctx.gemm_route()
            ));
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

        self.staged_upload(1, d_temporal)?;
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
        // The control channels first, so a window whose step-size and
        // decay gradients grow with its length cannot rescale the whole
        // arena through the global clip below.
        if let Some(cb) = control_clip_max_norm {
            let geom = m1_control_region_geom(&self.grads, &self.cfg);
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
        self.staged_upload(2, d_temporal)?;
        {
            let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
            let n = d_temporal.len() as i64;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.ctx.kernels.scale_grads_f32);
            builder.arg(dt_scaled.inner_mut());
            builder.arg(&scale);
            builder.arg(&n);
            unsafe { builder.launch(crate::gpu::launch::grid_1d_wide(d_temporal.len())) }
                .map_err(|e| format!("scale d_temporal (f16 split): {e:?}"))?;
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
            let (bt, dm) = (self.batch * self.seq_len, self.cfg.d_model);
            let mid = d_input.len() / bt;
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
            launch_cast(
                &stream,
                cast,
                (self.scratch.d_input_proj_dx.cached_ptr(), 4),
                (dx.cached_ptr(), dtype.size_bytes()),
                bt * mid,
                format_args!("input gradient cast"),
            )?;
            self.scratch.d_input_proj_dx.download(&stream, d_input)?;
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
                let geom = m1_control_region_geom(&self.grads, &self.cfg);
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
            // is discarded whole, as in the f32 and bf16 lanes.
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

    fn eager_f16_forward_backward(&mut self) -> Result<(), String> {
        self.presize_prepared_gemm_scratch()?;
        let has_gemm_work = self.has_gemm_work;
        if has_gemm_work {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        let Self {
            ctx,
            weights,
            grads,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal_scaled,
            prepared_f16_gemm_manifest,
            ..
        } = self;
        let d_temporal = d_temporal_scaled.as_mut().expect("f16 d_temporal_scaled");
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba_backbone_train_mixed(
                ctx,
                acts,
                weights,
                mamba_input,
                state,
                scratch,
            )?;
            gpu_backward_mamba_backbone_mixed(
                ctx,
                d_temporal,
                grads,
                acts,
                &weights.compute,
                a_neg_all,
                scratch,
            )
        })?;
        *prepared_f16_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// The f16 lane's upstream gradient: `d_temporal` staged into the
    /// scaled buffer and multiplied by the loss scale on the device.
    fn upload_scaled_d(&mut self, d_temporal: &[f32], scale: f32) -> Result<(), String> {
        self.staged_upload(2, d_temporal)?;
        let dt_scaled = self.d_temporal_scaled.as_mut().expect("f16 dt_scaled");
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
            .map_err(|e| format!("scale d_temporal (f16): {e:?}"))
    }

    /// Pointer-stability invariant of the f16 graphs: every device buffer
    /// baked into the captured kernels must have the same pointer at
    /// replay time.
    fn assert_f16_pointers(&self) {
        // Pointer-stability invariant — every device buffer baked into
        // the captured kernels MUST have the same pointer at replay time.
        assert_eq!(
            self.bias.ptr(),
            self.captured_f16_bias_ptr,
            "f16 graph replay: bias pointer changed since capture"
        );
        assert_eq!(
            self.unscale_factor.as_ref().unwrap().ptr(),
            self.captured_f16_unscale_ptr,
            "f16 graph replay: unscale_factor pointer changed since capture"
        );
        assert_eq!(
            self.overflow_flag
                .as_ref()
                .unwrap()
                .stable_ptr(&self.ctx.stream),
            self.captured_f16_overflow_ptr,
            "f16 graph replay: overflow_flag pointer changed since capture"
        );
        assert_eq!(
            self.grads.flat.cached_ptr(),
            self.captured_f16_grads_ptr,
            "f16 graph replay: grads.flat pointer changed since capture"
        );
        assert_eq!(
            self.d_temporal_scaled.as_ref().unwrap().cached_ptr(),
            self.captured_f16_dt_scaled_ptr,
            "f16 graph replay: d_temporal_scaled pointer changed since capture"
        );
        assert_eq!(
            self.ctx.half_staging_ptr(),
            self.captured_f16_half_staging_ptr,
            "f16 graph replay: half_staging pointer changed since capture \
                 (lazy grow after capture - re-capture or presize larger)"
        );
        assert_eq!(
            self.ctx.bi_upcast_scratch_ptrs(),
            self.captured_f16_bi_upcast_ptrs,
            "f16 graph replay: bi_upcast_scratch pointer changed since capture"
        );
        assert_eq!(
            self.ctx.gemm_route(),
            self.captured_f16_gemm_flags,
            "f16 graph replay: GEMM route changed since capture; re-capture"
        );

        // Graph replay: forward + backward + check_inf_nan +
        // scale_grads_skip + AdamW + sync all run as one cuGraphLaunch.
        // grads.zero is included in the captured body.
    }

    /// The bf16 graphs' replay buffers.
    fn mixed_replay(&self) -> MambaMixedReplay<'_> {
        MambaMixedReplay {
            train_w: &self.weights,
            adam: &self.adam,
            bias: &self.bias,
            grads: &self.grads,
            a_neg_all: &self.a_neg_all,
            mamba_input: &self.mamba_input,
            d_temporal: &self.d_temporal,
            state: &self.state,
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
    /// [`MambaGpuTrainer::accumulate`]).
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
        self.staged_upload(0, input)?;
        match scale {
            Some(scale) => self.upload_scaled_d(d_temporal, scale)?,
            None => self.staged_upload(1, d_temporal)?,
        }
        match (scale, &self.graph_acc, &self.graph_f16_acc) {
            (None, Some(g), _) => g.replay(&self.ctx, &self.mixed_replay())?,
            (Some(_), _, Some(g)) => {
                self.assert_f16_pointers();
                with_validated_gemm_graph_launch(
                    &self.ctx,
                    self.has_gemm_work,
                    self.captured_f16_acc_plan.as_ref(),
                    "M1 f16 accumulate graph replay",
                    || {
                        g.launch()
                            .map_err(|e| format!("f16 accumulate graph launch: {e:?}"))
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
            weights,
            grads,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal,
            d_temporal_scaled,
            prepared_gemm_manifest,
            prepared_f16_gemm_manifest,
            ..
        } = self;
        let d = match (scaled, d_temporal_scaled.as_mut()) {
            (true, Some(d)) => d,
            (true, None) => {
                return Err("f16 accumulate: the scaled gradient buffer is missing".into());
            }
            (false, _) => d_temporal,
        };
        let manifest = ctx.record_eager_gemm_manifest(|| {
            gpu_forward_mamba_backbone_train_mixed(
                ctx,
                acts,
                weights,
                mamba_input,
                state,
                scratch,
            )?;
            gpu_backward_mamba_backbone_mixed(
                ctx,
                d,
                grads,
                acts,
                &weights.compute,
                a_neg_all,
                scratch,
            )
        })?;
        if scaled {
            *prepared_f16_gemm_manifest = Some(manifest);
        } else {
            *prepared_gemm_manifest = Some(manifest);
        }
        Ok(())
    }

    /// The open window's apply (see [`MambaGpuTrainer::apply_window`]).
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
                self.assert_f16_pointers();
                with_validated_gemm_graph_launch(
                    &self.ctx,
                    false,
                    None,
                    "M1 f16 apply graph replay",
                    || {
                        g.launch()
                            .map_err(|e| format!("f16 apply graph launch: {e:?}"))
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

    /// f16 step (eager, no graph). Mirrors PyTorch GradScaler protocol:
    ///   1. Upload d_temporal scaled by `scaler.scale()`
    ///   2. forward + backward → grads (also scaled)
    ///   3. check_inf_nan over the grad arena, CPU readback of the flag
    ///   4. clean: unscale grads (`*= 1/scale`), AdamW + sync, scaler.update(false)
    ///      overflow: skip AdamW + sync, scaler.update(true) → scale halves
    fn step_f16(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        let scale = self.scaler.as_ref().expect("f16 scaler").scale();

        // Upload input + d_temporal (always, both eager and graph paths),
        // then scale d_temporal on-device: the old path built a scaled
        // Vec<f32> on the host every step (B*T*d_model alloc + traversal).
        self.staged_upload(0, input)?;
        self.upload_scaled_d(d_temporal, scale)?;

        // Update the unscale_factor device buffer (= 1/scale) for the
        // graph-captured `scale_grads_skip` kernel. CPU writes async H2D;
        // stream serialization ensures the captured kernel reads the
        // up-to-date value.
        if let Some(ref mut u) = self.unscale_factor {
            u.write(&self.ctx.stream, 1.0 / scale)?;
        }
        // Pre-bump Adam step counter + write bias factors. Conservatively
        // assume the optimizer WILL run (graph always launches AdamW; eager
        // skips on overflow). On eager-overflow we restore step below.
        let prev_step = self.adam.step;
        let (_, bc1, bc2) = self.adam.advance();
        self.bias.write(&self.ctx.stream, bc1, bc2, self.adam.lr)?;

        // Zero the overflow flag (drops borrow immediately so we can re-borrow below).
        self.overflow_flag
            .as_mut()
            .expect("f16 overflow flag")
            .zero(&self.ctx.stream)?;

        let replayed = if let Some(ref g) = self.graph_f16 {
            self.assert_f16_pointers();

            with_validated_gemm_graph_launch(
                &self.ctx,
                self.has_gemm_work,
                self.captured_f16_gemm_plan.as_ref(),
                "M1 f16 training graph replay",
                || {
                    g.launch()
                        .map_err(|error| format!("f16 graph launch: {error:?}"))
                },
            )?;
            true
        } else {
            // The eager step runs the captured body's sequence: the
            // overflow check, the unscale that skips on it, the step
            // options' tail and an AdamW that skips on the flag the tail
            // leaves, so an overflow or a spike leaves the weights and the
            // moments as the graph leaves them, as torch.cuda.amp's
            // GradScaler skips its step.
            self.eager_f16_forward_backward()?;
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

    /// Capture the f16 training step into a CUDA Graph.
    fn capture_graph_f16(&mut self) -> Result<(), String> {
        // Make sure bias + unscale_factor + overflow_flag have valid initial
        // values so the captured kernels record reads against stable
        // pointers (the values are overwritten per replay).
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
        // Dummy upload so captured pointers reference initialized memory.
        let dummy = vec![0.0f32; self.d_temporal.len()];
        self.d_temporal_scaled
            .as_mut()
            .expect("dt_scaled")
            .upload(&self.ctx.stream, &dummy)?;

        // Pre-size the half-staging buffer (defensive — typed forward
        // doesn't currently use it, but match the bf16 graph for parity).
        self.ctx
            .presize_half_staging_for_train(&self.cfg, self.batch, self.seq_len, self.dtype)?;
        // Same for the bi upcast scratch (input_dim-aware): under the
        // batch-invariant flag the captured body's typed GEMMs route through
        // with_bi_upcast_scratch — a lazy grow inside capture is illegal.
        self.presize_prepared_gemm_scratch()?;
        let manifest = self.prepared_f16_gemm_manifest.ok_or_else(|| {
            "M1 f16 training graph capture requires one successful eager step".to_string()
        })?;
        // Snapshot every device pointer the captured kernels reference, so
        // step_f16 can assert pointer stability on each replay; the f16
        // graph once lacked these guards.
        let snap_bias = self.bias.ptr();
        let snap_unscale = self.unscale_factor.as_ref().unwrap().ptr();
        let snap_overflow = self
            .overflow_flag
            .as_ref()
            .unwrap()
            .stable_ptr(&self.ctx.stream);
        let snap_grads = self.grads.flat.cached_ptr();
        let snap_dt_scaled = self.d_temporal_scaled.as_ref().unwrap().cached_ptr();
        let tail = self.step_tail();
        let skip_ptr = self.skip_flag.stable_ptr(&self.ctx.stream);

        // Capture bodies: the whole step, an accumulating micro-batch and the
        // window's apply, each the eager sequence 1:1 so numerics match.
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
            (&captured_f16_gemm_plan, "M1 f16 training graph capture"),
            (&acc_plan, "M1 f16 accumulate graph capture"),
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
        self.captured_f16_half_staging_ptr = self.ctx.half_staging_ptr();
        self.captured_f16_bi_upcast_ptrs = self.ctx.bi_upcast_scratch_ptrs();
        self.ctx.note_graph_capture();
        self.captured_f16_gemm_flags = self.ctx.gemm_route();
        Ok(())
    }

    /// Capture one f16 training body (see [`TrainBody`]): the whole step
    /// (the arena zeroed, the forward, the backward on the scaled
    /// gradient, the overflow check, the unscale that skips on it, the step
    /// options' tail, an AdamW that skips on the tail's flag, the master
    /// sync and the a_neg refresh), an accumulating micro-batch (the
    /// forward and the backward into the open window) or the window's
    /// apply (the step's sequence from the overflow check on).
    fn capture_f16_body(
        &mut self,
        manifest: &PreparedGemmCaptureManifest,
        body: TrainBody,
        (tail, skip_ptr, overflow_ptr): (StepTail, u64, u64),
    ) -> Result<(cudarc::driver::CudaGraph, Option<CapturedGemmGraphPlan>), String> {
        let Self {
            ctx,
            cfg,
            dtype,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal_scaled,
            overflow_flag,
            unscale_factor,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let d_temporal = d_temporal_scaled.as_mut().expect("f16 dt_scaled");
        unsafe {
            capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, manifest, || {
                if body.passes() {
                    if body == TrainBody::Step {
                        grads.zero(&ctx.stream)?;
                    }
                    gpu_forward_mamba_backbone_train_mixed(
                        ctx,
                        acts,
                        weights,
                        mamba_input,
                        state,
                        scratch,
                    )?;
                    gpu_backward_mamba_backbone_mixed(
                        ctx,
                        d_temporal,
                        grads,
                        acts,
                        &weights.compute,
                        a_neg_all,
                        scratch,
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
                // The step options' tail, then AdamW, which skips on the
                // flag the tail leaves (the overflow flag when it measures
                // nothing): branching mid-graph is unsupported, and
                // scale_grads_skip has already sanitized the arena on
                // overflow. The optimizer tail also recomputes a_neg after
                // AdamW so each replay sees the updated A-matrix (same
                // rationale as the eager and bf16-graph paths).
                let skip = tail.enqueue(
                    ctx,
                    &mut grads.flat,
                    (&mut *clip_partials, &mut *clip_scratch),
                    (skip_ptr, overflow_ptr),
                )?;
                step_multi(
                    ctx,
                    ctx.kernels.adamw_step_multi.get(*dtype),
                    multi_plan,
                    adam,
                    bias.ptr(),
                    skip,
                )?;
                weights.sync_master_to_compute(ctx)?;
                recompute_a_neg_all(
                    ctx,
                    &weights.master.layers,
                    a_neg_all,
                    &state.a_neg_all,
                    cfg.d_inner(),
                    cfg.d_state,
                )
            })
        }
    }

    /// Eager forward body: run the mixed training forward (saves the typed
    /// activations backward reads). One of the three shared phase bodies the
    /// fused eager step, the f16 eager step, and the f16 capture body all
    /// compose; the forward/backward split API builds on exactly these seams.
    fn eager_forward(&mut self) -> Result<(), String> {
        gpu_forward_mamba_backbone_train_mixed(
            &self.ctx,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )
    }

    /// Eager backward body: accumulate gradients into the grad arena
    /// (beta=1.0 — zeroing is the caller's responsibility). `scaled` selects
    /// the f16 loss-scaled `d_temporal_scaled` buffer over the plain
    /// `d_temporal` upload buffer.
    fn eager_backward(&mut self, scaled: bool) -> Result<(), String> {
        let d_temporal = if scaled {
            self.d_temporal_scaled
                .as_mut()
                .expect("eager_backward(scaled): f16 d_temporal_scaled missing")
        } else {
            &mut self.d_temporal
        };
        gpu_backward_mamba_backbone_mixed(
            &self.ctx,
            d_temporal,
            &self.grads,
            &self.acts,
            &self.weights.compute,
            &self.a_neg_all,
            &mut self.scratch,
        )
    }

    /// Eager optimizer tail: AdamW on the f32 master weights (capturable
    /// kernel so graph and eager numerics stay bit-identical), master →
    /// compute sync, then the mandatory `a_neg = -exp(a_log)` refresh
    /// (without it the SSM reads a stale A-matrix and the a_log gradient is
    /// a silent no-op — see `recompute_a_neg_all`).
    fn eager_optimize(&mut self, skip: cudarc::driver::sys::CUdeviceptr) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.ctx.kernels.adamw_step_multi.get(self.dtype),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
            skip,
        )?;
        // Bulk typed shadows now ride the fused kernel; the walk below
        // covers only the f32-stays-f32 tensors.
        self.weights.sync_master_to_compute(&self.ctx)?;
        recompute_a_neg_all(
            &self.ctx,
            &self.weights.master.layers,
            &self.a_neg_all,
            &self.state.a_neg_all,
            self.cfg.d_inner(),
            self.cfg.d_state,
        )
    }

    /// Eager fallback (used before [`Self::capture_graph`] is called and
    /// shared as the body of capture). Mirrors the exact op sequence the
    /// captured graph records.
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
            cfg,
            dtype,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            state,
            a_neg_all,
            mamba_input,
            d_temporal,
            prepared_gemm_manifest,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba_backbone_train_mixed(
                ctx,
                acts,
                weights,
                mamba_input,
                state,
                scratch,
            )?;
            gpu_backward_mamba_backbone_mixed(
                ctx,
                d_temporal,
                grads,
                acts,
                &weights.compute,
                a_neg_all,
                scratch,
            )?;
            let skip = tail.enqueue(
                ctx,
                &mut grads.flat,
                (&mut *clip_partials, &mut *clip_scratch),
                (skip_ptr, 0),
            )?;
            step_multi(
                ctx,
                ctx.kernels.adamw_step_multi.get(*dtype),
                multi_plan,
                adam,
                bias.ptr(),
                skip,
            )?;
            weights.sync_master_to_compute(ctx)?;
            recompute_a_neg_all(
                ctx,
                &weights.master.layers,
                a_neg_all,
                &state.a_neg_all,
                cfg.d_inner(),
                cfg.d_state,
            )
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Download the master weights to a CPU-side `MambaWeights` for
    /// checkpointing. Includes a stream sync.
    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let master = &self.weights.master;
        let mut out = MambaWeights::zeros(
            &self.cfg,
            self.mamba_input.len() / (self.batch * self.seq_len),
        );
        out.input_proj_w = master.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = master.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &master.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.norm_bias = g.norm_bias.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.in_proj_b = g.in_proj_b.to_cpu(&self.ctx.stream)?;
            lw.conv1d_weight = g.conv1d_weight.to_cpu(&self.ctx.stream)?;
            lw.conv1d_bias = g.conv1d_bias.to_cpu(&self.ctx.stream)?;
            lw.x_proj_w = g.x_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_w = g.dt_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_b = g.dt_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_log = g.a_log.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
            lw.out_proj_b = g.out_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_neg = lw.a_log.iter().map(|v| -v.exp()).collect();
        }
        out.norm_f_weight = master.norm_f_weight.to_cpu(&self.ctx.stream)?;
        out.norm_f_bias = master.norm_f_bias.to_cpu(&self.ctx.stream)?;
        for (m, g) in out.mlp.iter_mut().zip(&master.mlp) {
            m.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            m.norm_bias = g.norm_bias.to_cpu(&self.ctx.stream)?;
            m.fc1_w = g.fc1_w.to_cpu(&self.ctx.stream)?;
            m.fc2_w = g.fc2_w.to_cpu(&self.ctx.stream)?;
        }
        Ok(out)
    }
}

// ════════════════════════════════════════════════════════════════════════
// f32 training wrapper (no master/compute split, no half_staging).
// ════════════════════════════════════════════════════════════════════════

/// f32 training inner. Weights stay in f32 throughout — no compute shadow,
/// no master→compute sync step in the training loop.
pub(crate) struct MambaTrainerF32 {
    pub ctx: GpuCtx,
    pub cfg: MambaConfig,
    pub batch: usize,
    pub seq_len: usize,
    pub weights: GpuMambaTrainWeights,
    pub grads: GpuMambaGrads,
    pub adam: GpuAdamW,
    bias: AdamWBiasFactors,
    multi_plan: AdamWMultiPlan,
    acts: GpuMambaBackboneActs,
    scratch: GpuMambaScratch,
    state: GpuRecurrentState,
    a_neg_all: GpuBuffer,
    temporal: GpuBuffer,
    mamba_input: GpuBuffer,
    d_temporal: GpuBuffer,
    graph: Option<GpuMambaF32TrainingStepGraph>,
    /// The accumulating micro-batch and the window's apply, captured
    /// beside the step.
    graph_acc: Option<GpuMambaF32TrainingStepGraph>,
    graph_apply: Option<GpuMambaF32TrainingStepGraph>,
    prepared_gemm_manifest: Option<PreparedGemmCaptureManifest>,
    has_gemm_work: bool,
    /// Route that produced the saved split-forward activations.
    split_forward_route: Option<crate::gpu::context::GemmRoute>,
    /// True while an `accumulate_only` backward window is open (see the
    /// same-named field on `MambaTrainerMixed`).
    grads_dirty: bool,
    /// Device partials for the deterministic grad-clip norm (512 f64).
    clip_partials: GpuByteBuffer,
    /// Host mirror of the partials — pre-allocated (zero-alloc hot path).
    /// `[coef, norm]` - the device-side clip fold's output.
    clip_scratch: GpuBuffer,
    /// The options of the fused step, eager or captured.
    step_opts: BackwardOpts,
    /// The skip flag a measured step tail writes (see [`StepTail`]).
    skip_flag: OverflowFlag,
    // Pinned host staging + reuse guard (see the twin fields on
    // MambaTrainerMixed for the interlock rationale).
    pin_input: crate::gpu::buffers::PinnedHostBuf,
    pin_dtemp: crate::gpu::buffers::PinnedHostBuf,
    upload_guard: cudarc::driver::CudaEvent,
}

impl Drop for MambaTrainerF32 {
    fn drop(&mut self) {
        let _ = self.ctx.stream.synchronize();
        self.drop_graphs();
    }
}

impl MambaTrainerF32 {
    /// Upload through the pinned stage. Waits out the previous
    /// step's DMA (guard event) before rewriting the staging buffer,
    /// then records the guard after the enqueue. `which`: 0 = input into
    /// mamba_input, 1 = d_temporal into d_temporal.
    fn staged_upload(&mut self, which: usize, data: &[f32]) -> Result<(), String> {
        self.upload_guard
            .synchronize()
            .map_err(|e| format!("upload guard sync: {e:?}"))?;
        let (pin, dst) = if which == 0 {
            (&mut self.pin_input, &mut self.mamba_input)
        } else {
            (&mut self.pin_dtemp, &mut self.d_temporal)
        };
        pin.as_mut_slice()[..data.len()].copy_from_slice(data);
        dst.upload(&self.ctx.stream, &pin.as_slice()[..data.len()])?;
        self.upload_guard
            .record(&self.ctx.stream)
            .map_err(|e| format!("upload guard record: {e:?}"))
    }

    fn new_full(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
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
        let state_cap = crate::gpu::kernels::state_capacity(cfg.d_state)?;
        let role = GemmRole::triad(dtype);
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_role(&device, state_cap, mode, role)?,
            None => GpuCtx::new_from_env_with_state_cap_and_role(&device, state_cap, role)?,
        };

        let weights = GpuMambaTrainWeights::from_cpu(&ctx.stream, cpu_weights)?;

        let d_inner = cfg.d_inner();
        let d_state = cfg.d_state;
        let d_conv = cfg.d_conv;
        let n_layers = cfg.n_layers;

        let dims = GpuMambaDims {
            batch,
            d_model: cfg.d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank: cfg.dt_rank(),
            xdbl_dim: cfg.xdbl_dim(),
            seq_len,
            mamba_input_dim: input_dim,
            n_layers,
            scan_mode: cfg.scan_mode,
            rms_norm_eps: cfg.rms_norm_eps,
            block_norm: cfg.block_norm,
            bcdt_rms_eps: cfg.bcdt_rms_eps,
        };
        dims.check_for(&ctx)?;

        let hidden = cfg.mlp_hidden();
        let acts =
            GpuMambaBackboneActs::new(&ctx.stream, &dims)?.with_mlp(&ctx.stream, &dims, hidden)?;
        let scratch = GpuMambaScratch::new(&ctx.stream, &dims)?.with_mlp(&ctx.stream, hidden)?;

        // a_neg is recomputed from the uploaded a_log by the SAME GPU
        // kernel the post-step refresh uses — a CPU-side exp here differs
        // by ULPs from the device expf and breaks bit-continuous resume
        // (see the mixed constructor's twin comment).
        let a_neg_all = GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?;

        // conv/ssm states are per-sample: forward indexes layers with a
        // batch * d_inner * d_conv (resp. d_state) per-layer stride.
        let state = GpuRecurrentState {
            conv_states: GpuBuffer::zeros(&ctx.stream, n_layers * batch * d_inner * d_conv)?,
            ssm_states: GpuBuffer::zeros(&ctx.stream, n_layers * batch * d_inner * d_state)?,
            a_neg_all: GpuBuffer::zeros(&ctx.stream, n_layers * d_inner * d_state)?,
        };
        recompute_a_neg_all(
            &ctx,
            &weights.layers,
            &a_neg_all,
            &state.a_neg_all,
            d_inner,
            d_state,
        )?;

        let temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let mamba_input = GpuBuffer::zeros(&ctx.stream, batch * seq_len * input_dim)?;
        let d_temporal = GpuBuffer::zeros(&ctx.stream, batch * seq_len * cfg.d_model)?;
        let grads = GpuMambaGrads::new_sized(
            &ctx.stream,
            &cfg,
            input_dim,
            !cpu_weights.input_proj_w.is_empty(),
        )?;

        let adam = GpuAdamW::new(&ctx.stream, grads.flat.len())?
            .with_lr(lr)
            .with_weight_decay(weight_decay);
        let bias = AdamWBiasFactors::new(&ctx.stream)?;
        let multi_plan = build_multi_plan(
            &ctx.stream,
            &adam,
            grads.flat.cached_ptr(),
            &m1_specs(&weights, &grads),
            adam.reference_no_decay,
            adam.weight_decay,
        )?;

        ctx.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;

        let clip_partials = alloc_partials(&ctx.stream)?;
        let clip_scratch = GpuBuffer::zeros(&ctx.stream, 2)?;
        let skip_flag = OverflowFlag::new(&ctx.stream)?;

        let pin_input = crate::gpu::buffers::PinnedHostBuf::zeroed(batch * seq_len * input_dim)?;
        let pin_dtemp = crate::gpu::buffers::PinnedHostBuf::zeroed(batch * seq_len * cfg.d_model)?;
        let upload_guard = ctx
            .stream
            .context()
            .new_event(None)
            .map_err(|e| format!("upload guard event: {e:?}"))?;
        let has_gemm_work = batch != 0 && seq_len != 0;
        Ok(Self {
            ctx,
            cfg,
            batch,
            seq_len,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            state,
            a_neg_all,
            temporal,
            mamba_input,
            d_temporal,
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
            pin_input,
            pin_dtemp,
            upload_guard,
        })
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.split_forward_route = None;
        self.state.conv_states.zero(&self.ctx.stream)?;
        self.state.ssm_states.zero(&self.ctx.stream)?;
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
            m1_control_region_geom(&self.grads, &self.cfg),
        )
    }

    /// Set the options of the fused step (see
    /// [`MambaGpuTrainer::set_step_opts`]).
    pub(crate) fn set_step_opts(&mut self, opts: BackwardOpts) -> Result<(), String> {
        if opts.accumulate_only {
            return Err("the fused step applies its window: accumulate with backward_step".into());
        }
        if self.has_graph() && opts != self.step_opts {
            return Err(
                "Mamba-1 set_step_opts: the captured graph baked the step options in; \
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
            "M1 f32 training graph capture requires one successful eager step".to_string()
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
    ) -> Result<GpuMambaF32TrainingStepGraph, String> {
        let tail = self.step_tail();
        let skip = self.skip_flag.stable_ptr(&self.ctx.stream);
        // The trainer owns every captured allocation and drops the graphs first.
        unsafe {
            GpuMambaF32TrainingStepGraph::capture(
                &self.ctx,
                &self.cfg,
                MambaF32Capture {
                    weights: &mut self.weights,
                    adam: &self.adam,
                    bias: &self.bias,
                    multi_plan: &self.multi_plan,
                    grads: &mut self.grads,
                    acts: &mut self.acts,
                    scratch: &mut self.scratch,
                    a_neg_all: &self.a_neg_all,
                    temporal: &mut self.temporal,
                    mamba_input: &self.mamba_input,
                    d_temporal: &mut self.d_temporal,
                    state: &mut self.state,
                    tail: Some(CaptureTail {
                        tail,
                        partials: &mut self.clip_partials,
                        coef: &mut self.clip_scratch,
                        skip,
                    }),
                },
                self.batch,
                self.seq_len,
                manifest,
                body,
            )
        }
    }

    /// Download the SSM `a_neg_all` buffer (f32 trainer variant). Used by
    /// the regression test verifying the post-AdamW recompute is applied.
    #[doc(hidden)]
    pub fn debug_a_neg_all(&self) -> Result<Vec<f32>, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("debug_a_neg_all sync: {e:?}"))?;
        self.a_neg_all.to_cpu(&self.ctx.stream)
    }

    pub fn step(&mut self, input: &[f32], d_temporal: &[f32]) -> Result<StepMetrics, String> {
        // Every fused attempt abandons a pending split tape, even when it is rejected.
        self.split_forward_route = None;
        assert_eq!(
            input.len(),
            self.mamba_input.len(),
            "input shape mismatch: expected batch*seq_len*input_dim={}, got {}",
            self.mamba_input.len(),
            input.len(),
        );
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.d_temporal.len(),
            d_temporal.len(),
        );
        // A step with a window open adds its micro-batch to the window and
        // applies it, as the split calls would close the window.
        if self.grads_dirty {
            self.accumulate(input, d_temporal)?;
            return self.apply_window();
        }
        self.staged_upload(0, input)?;
        self.staged_upload(1, d_temporal)?;

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
    fn f32_replay(&self) -> MambaF32Replay<'_> {
        MambaF32Replay {
            weights: &self.weights,
            adam: &self.adam,
            bias: &self.bias,
            grads: &self.grads,
            temporal: &self.temporal,
            a_neg_all: &self.a_neg_all,
            mamba_input: &self.mamba_input,
            d_temporal: &self.d_temporal,
            state: &self.state,
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
    /// [`MambaGpuTrainer::accumulate`]).
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
        self.staged_upload(0, input)?;
        self.staged_upload(1, d_temporal)?;
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
            weights,
            grads,
            acts,
            scratch,
            state,
            a_neg_all,
            temporal,
            mamba_input,
            d_temporal,
            prepared_gemm_manifest,
            ..
        } = self;
        let manifest = ctx.record_eager_gemm_manifest(|| {
            gpu_forward_mamba_backbone(ctx, temporal, acts, weights, mamba_input, state, scratch)?;
            gpu_backward_mamba_backbone(ctx, d_temporal, grads, acts, weights, a_neg_all, scratch)
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// The open window's apply (see [`MambaGpuTrainer::apply_window`]).
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

    /// Split forward (see [`MambaGpuTrainer::forward`]): eager forward into the
    /// existing `self.temporal` device buffer, sync, download into
    /// `temporal_out`. Zero new device allocation — the buffer already
    /// exists and was previously write-only.
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
        self.staged_upload(0, input)?;
        self.eager_forward()?;
        // Sync AFTER the download enqueue — see the twin comment in
        // the mixed split path.
        self.temporal.download(&self.ctx.stream, temporal_out)?;
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("forward_split sync: {e:?}"))?;
        self.split_forward_route = Some(self.ctx.gemm_route());
        Ok(())
    }

    /// Split backward + optimizer (see [`MambaGpuTrainer::backward_step`]),
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
        if forward_route != self.ctx.gemm_route() {
            return Err(format!(
                "backward_step(): GEMM route changed since forward() \
                 ({forward_route:?} -> {:?}); re-run forward()",
                self.ctx.gemm_route()
            ));
        }
        assert_eq!(
            d_temporal.len(),
            self.d_temporal.len(),
            "d_temporal shape mismatch: expected batch*seq_len*d_model={}, got {}",
            self.d_temporal.len(),
            d_temporal.len(),
        );
        self.staged_upload(1, d_temporal)?;
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
        // The control channels first, so a window whose step-size and
        // decay gradients grow with its length cannot rescale the whole
        // arena through the global clip below.
        if let Some(cb) = control_clip_max_norm {
            let geom = m1_control_region_geom(&self.grads, &self.cfg);
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

    /// Eager forward body: run the training forward, writing the post-norm_f
    /// output into `self.temporal`. One of the three shared phase bodies the
    /// fused eager step composes; the forward/backward split API builds on
    /// exactly these seams.
    fn eager_forward(&mut self) -> Result<(), String> {
        gpu_forward_mamba_backbone(
            &self.ctx,
            &mut self.temporal,
            &mut self.acts,
            &self.weights,
            &self.mamba_input,
            &mut self.state,
            &mut self.scratch,
        )
    }

    /// Eager backward body: accumulate gradients from `self.d_temporal`
    /// through the saved activations into the grad arena (beta=1.0 —
    /// zeroing is the caller's responsibility).
    fn eager_backward(&mut self) -> Result<(), String> {
        gpu_backward_mamba_backbone(
            &self.ctx,
            &mut self.d_temporal,
            &self.grads,
            &self.acts,
            &self.weights,
            &self.a_neg_all,
            &mut self.scratch,
        )
    }

    /// Eager optimizer tail: AdamW over the grad arena, then the mandatory
    /// `a_neg = -exp(a_log)` refresh (without it the SSM reads a stale
    /// A-matrix and the a_log gradient is a silent no-op — see
    /// `recompute_a_neg_all`).
    fn eager_optimize(&mut self, skip: cudarc::driver::sys::CUdeviceptr) -> Result<(), String> {
        step_multi(
            &self.ctx,
            self.ctx.kernels.adamw_step_multi.get(WeightDtype::F32),
            &self.multi_plan,
            &self.adam,
            self.bias.ptr(),
            skip,
        )?;
        recompute_a_neg_all(
            &self.ctx,
            &self.weights.layers,
            &self.a_neg_all,
            &self.state.a_neg_all,
            self.cfg.d_inner(),
            self.cfg.d_state,
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
            cfg,
            weights,
            grads,
            adam,
            bias,
            multi_plan,
            acts,
            scratch,
            state,
            a_neg_all,
            temporal,
            mamba_input,
            d_temporal,
            prepared_gemm_manifest,
            clip_partials,
            clip_scratch,
            ..
        } = self;
        let manifest = ctx.record_eager_gemm_manifest(|| {
            grads.zero(&ctx.stream)?;
            gpu_forward_mamba_backbone(ctx, temporal, acts, weights, mamba_input, state, scratch)?;
            gpu_backward_mamba_backbone(ctx, d_temporal, grads, acts, weights, a_neg_all, scratch)?;
            let skip = tail.enqueue(
                ctx,
                &mut grads.flat,
                (&mut *clip_partials, &mut *clip_scratch),
                (skip_ptr, 0),
            )?;
            step_multi(
                ctx,
                ctx.kernels.adamw_step_multi.get(WeightDtype::F32),
                multi_plan,
                adam,
                bias.ptr(),
                skip,
            )?;
            recompute_a_neg_all(
                ctx,
                &weights.layers,
                a_neg_all,
                &state.a_neg_all,
                cfg.d_inner(),
                cfg.d_state,
            )
        })?;
        *prepared_gemm_manifest = Some(manifest);
        Ok(())
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

    pub fn snapshot_master(&self) -> Result<MambaWeights, String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-snapshot sync: {e:?}"))?;
        let w = &self.weights;
        let input_dim = self.mamba_input.len() / (self.batch * self.seq_len);
        let mut out = MambaWeights::zeros(&self.cfg, input_dim);
        out.input_proj_w = w.input_proj_w.to_cpu(&self.ctx.stream)?;
        out.input_proj_b = w.input_proj_b.to_cpu(&self.ctx.stream)?;
        for (i, lw) in out.layers.iter_mut().enumerate() {
            let g = &w.layers[i];
            lw.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            lw.norm_bias = g.norm_bias.to_cpu(&self.ctx.stream)?;
            lw.in_proj_w = g.in_proj_w.to_cpu(&self.ctx.stream)?;
            lw.in_proj_b = g.in_proj_b.to_cpu(&self.ctx.stream)?;
            lw.conv1d_weight = g.conv1d_weight.to_cpu(&self.ctx.stream)?;
            lw.conv1d_bias = g.conv1d_bias.to_cpu(&self.ctx.stream)?;
            lw.x_proj_w = g.x_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_w = g.dt_proj_w.to_cpu(&self.ctx.stream)?;
            lw.dt_proj_b = g.dt_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_log = g.a_log.to_cpu(&self.ctx.stream)?;
            lw.d_param = g.d_param.to_cpu(&self.ctx.stream)?;
            lw.out_proj_w = g.out_proj_w.to_cpu(&self.ctx.stream)?;
            lw.out_proj_b = g.out_proj_b.to_cpu(&self.ctx.stream)?;
            lw.a_neg = lw.a_log.iter().map(|v| -v.exp()).collect();
        }
        out.norm_f_weight = w.norm_f_weight.to_cpu(&self.ctx.stream)?;
        out.norm_f_bias = w.norm_f_bias.to_cpu(&self.ctx.stream)?;
        for (m, g) in out.mlp.iter_mut().zip(&w.mlp) {
            m.norm_weight = g.norm_weight.to_cpu(&self.ctx.stream)?;
            m.norm_bias = g.norm_bias.to_cpu(&self.ctx.stream)?;
            m.fc1_w = g.fc1_w.to_cpu(&self.ctx.stream)?;
            m.fc2_w = g.fc2_w.to_cpu(&self.ctx.stream)?;
        }
        Ok(out)
    }
}

// The names these types carried before 0.8, when each family named its GPU
// types its own way.
#[deprecated(since = "0.8.0", note = "renamed to `MambaGpuTrainer`")]
pub type MambaTrainer = MambaGpuTrainer;
