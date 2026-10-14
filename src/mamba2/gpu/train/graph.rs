//! The Mamba-2 training step captured as CUDA graphs: the whole step (the
//! forward, the backward, in f16 the overflow check and the unscale, the
//! step options' tail, the fused AdamW and the half copies' refresh), an
//! accumulating micro-batch (the forward, the backward and the add into the
//! window's sum) and the window's apply (the sum into the arena, in f16 the
//! overflow check, the tail and AdamW): the kernels the eager calls launch,
//! in the order they launch them. Before a replay the host uploads the
//! window's input and gradient, the AdamW step's bias corrections and
//! learning rate, and in f16 the loss scale; after it, it reads the flags
//! the captured AdamW obeyed on the device.

use cudarc::driver::CudaGraph;

use super::Mamba2GpuTrainer;
use crate::gpu::adamw::AdamWPlanKey;
use crate::gpu::context::GemmRoute;
use crate::gpu::grad_clip::read_tail;
use crate::gpu::graph_capture::{
    capture_owner_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::gpu::kernel_identity::CapturedGemmGraphPlan;
use crate::gpu::loss_scaler::DynamicLossScaler;
use crate::gpu::train::StepMetrics;

/// The captured training bodies and what their replays check.
pub(super) struct TrainGraph {
    /// The whole step.
    graph: CudaGraph,
    /// An accumulating micro-batch.
    accumulate: CudaGraph,
    /// The window's apply, which runs no GEMM.
    apply: CudaGraph,
    gemm_route: GemmRoute,
    gemm_plan: Option<CapturedGemmGraphPlan>,
    accumulate_plan: Option<CapturedGemmGraphPlan>,
    half_staging: cudarc::driver::sys::CUdeviceptr,
    bi_upcast: [cudarc::driver::sys::CUdeviceptr; 3],
    /// The AdamW chunk table's key, and the betas and epsilon the captured
    /// launch took by value.
    plan_key: AdamWPlanKey,
    adam_consts: [u32; 3],
    /// The packing's ids the captured kernels read, 0 for none.
    varlen: u64,
}

fn adam_consts(t: &Mamba2GpuTrainer) -> [u32; 3] {
    [
        t.adamw.beta1.to_bits(),
        t.adamw.beta2.to_bits(),
        t.adamw.eps.to_bits(),
    ]
}

impl Mamba2GpuTrainer {
    /// Whether a step runs any GEMM: the input projection or a layer.
    pub(super) fn has_gemm_work(&self) -> bool {
        self.batch * self.seq_len != 0
            && (self.layout.input_proj_w.len != 0 || self.cfg.n_layers != 0)
    }

    /// Capture the training step as CUDA graphs; later [`Self::step`],
    /// [`Self::accumulate`] and [`Self::apply_window`] calls replay them,
    /// with the bits of the eager calls. Needs one eager `step` or
    /// `accumulate` first, whose GEMM inventory the capture takes. The step
    /// options (see [`Self::set_step_opts`]) are baked in. The replay
    /// refuses a changed GEMM route, moved GEMM staging, or AdamW settings
    /// the capture baked in (the betas, epsilon, weight decay, no-decay
    /// groups and moment buffers): [`Self::drop_graph`] and capture again
    /// after changing those. The learning rate lives on the device and may
    /// change between replays. A step with an open accumulation window
    /// runs eagerly, since it folds the window in. After capture the
    /// context's GEMM staging can no longer grow.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        if self.graph.is_some() {
            return Err("a training graph is captured already: drop_graph first".into());
        }
        let manifest = self.eager_gemm_manifest.ok_or_else(|| {
            "Mamba-2 training graph capture needs one successful eager step first".to_string()
        })?;
        self.ensure_adamw_plan()?;
        self.window.ensure_sum(&self.ctx, self.grads.len())?;
        // The captured kernels read these buffers; every replay writes
        // them first, so any finite values do here.
        let stream = self.ctx.stream.clone();
        self.bias.write(&stream, 1.0, 1.0, self.adamw.lr)?;
        if let Some(scale) = self.scaler.as_ref().map(DynamicLossScaler::scale) {
            self.unscale.write(&stream, 1.0 / scale)?;
            self.overflow.zero(&stream)?;
        }
        self.ctx.presize_bi_scratch()?;
        let gemm_route = self.ctx.gemm_route();
        let has_gemm_work = self.has_gemm_work();
        self.ctx.freeze_graph_scratch();
        let (graph, gemm_plan) = unsafe {
            capture_owner_into_graph_with_gemm_plan(
                self,
                |t| &t.ctx,
                manifest.route_capacity,
                &manifest,
                |t| {
                    t.forward_device()?;
                    t.backward_device()?;
                    let skip = t.tail_device()?;
                    t.adamw_device(skip)
                },
            )
        }?;
        require_deterministic_gemm_graph_plan(
            &self.ctx,
            has_gemm_work,
            gemm_plan.as_ref(),
            "Mamba-2 training graph capture",
        )?;
        let (accumulate, accumulate_plan) = unsafe {
            capture_owner_into_graph_with_gemm_plan(
                self,
                |t| &t.ctx,
                manifest.route_capacity,
                &manifest,
                |t| {
                    t.forward_device()?;
                    t.backward_device()?;
                    t.accumulate_device()
                },
            )
        }?;
        require_deterministic_gemm_graph_plan(
            &self.ctx,
            has_gemm_work,
            accumulate_plan.as_ref(),
            "Mamba-2 accumulate graph capture",
        )?;
        // The apply runs no GEMM: it is held to an empty inventory, so a
        // GEMM that found its way into it would fail the capture.
        let no_gemm = self.ctx.record_eager_gemm_manifest(|| Ok(()))?;
        let (apply, _) = unsafe {
            capture_owner_into_graph_with_gemm_plan(
                self,
                |t| &t.ctx,
                no_gemm.route_capacity,
                &no_gemm,
                |t| t.apply_device(),
            )
        }?;
        self.graph = Some(TrainGraph {
            graph,
            accumulate,
            apply,
            gemm_route,
            gemm_plan,
            accumulate_plan,
            half_staging: self.ctx.half_staging_ptr(),
            bi_upcast: self.ctx.bi_upcast_scratch_ptrs(),
            plan_key: AdamWPlanKey::of(&self.adamw),
            adam_consts: adam_consts(self),
            varlen: self.varlen.ptr(),
        });
        self.ctx.note_graph_capture();
        Ok(())
    }

    /// Whether [`Self::step`], [`Self::accumulate`] and
    /// [`Self::apply_window`] replay captured graphs.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Drop the captured graphs: the next calls run eagerly and may be
    /// captured again. A replay still in flight finishes first.
    pub fn drop_graph(&mut self) {
        let _ = self.ctx.stream.synchronize();
        self.graph = None;
    }

    /// What every replay checks first: a captured graph, the GEMM route,
    /// the AdamW settings and the GEMM staging the capture baked in.
    fn check_replay(&self, label: &str) -> Result<&TrainGraph, String> {
        let Some(g) = &self.graph else {
            return Err(format!("{label}: no graph is captured"));
        };
        if g.gemm_route != self.ctx.gemm_route() {
            return Err(format!("{label}: the GEMM route changed since capture"));
        }
        if g.plan_key != AdamWPlanKey::of(&self.adamw) || g.adam_consts != adam_consts(self) {
            return Err(format!(
                "{label}: the AdamW settings changed since capture; drop_graph and \
                 capture again"
            ));
        }
        if g.varlen != self.varlen.ptr() {
            return Err(format!(
                "{label}: the packing was turned on or off since capture; drop_graph and \
                 capture again"
            ));
        }
        self.ctx
            .ensure_graph_scratch_ptrs(g.half_staging, g.bi_upcast, label)?;
        Ok(g)
    }

    /// One accumulating micro-batch through the captured graph, on a window
    /// the caller opened.
    pub(super) fn accumulate_graph(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<(), String> {
        const LABEL: &str = "Mamba-2 accumulate graph replay";
        let stream = self.ctx.stream.clone();
        self.input.upload(&stream, input)?;
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.upload_d_out(d_temporal, scale)?;
        let g = self.check_replay(LABEL)?;
        with_validated_gemm_graph_launch(
            &self.ctx,
            self.has_gemm_work(),
            g.accumulate_plan.as_ref(),
            LABEL,
            || g.accumulate.launch().map_err(|e| format!("{LABEL}: {e:?}")),
        )
    }

    /// The open window's apply through the captured graph, its AdamW step
    /// counted by the caller.
    pub(super) fn apply_graph(&self) -> Result<(), String> {
        const LABEL: &str = "Mamba-2 apply graph replay";
        let g = self.check_replay(LABEL)?;
        with_validated_gemm_graph_launch(&self.ctx, false, None, LABEL, || {
            g.apply.launch().map_err(|e| format!("{LABEL}: {e:?}"))
        })
    }

    /// One step through the captured graph.
    pub(super) fn step_graph(
        &mut self,
        input: &[f32],
        d_temporal: &[f32],
    ) -> Result<StepMetrics, String> {
        const LABEL: &str = "Mamba-2 training graph replay";
        self.check_replay(LABEL)?;
        let stream = self.ctx.stream.clone();
        self.forwarded = false;
        self.fresh_grads = false;
        self.input.upload(&stream, input)?;
        let scale = self.scaler.as_ref().map(DynamicLossScaler::scale);
        self.upload_d_out(d_temporal, scale)?;
        // The step counts before the replay, as the eager step counts it
        // before its AdamW; a step the overflow skipped gives it back.
        let (_, bias_c1, bias_c2) = self.adamw.advance();
        self.bias.write(&stream, bias_c1, bias_c2, self.adamw.lr)?;
        let Some(g) = &self.graph else {
            return Err(format!("{LABEL}: no graph is captured"));
        };
        with_validated_gemm_graph_launch(
            &self.ctx,
            self.has_gemm_work(),
            g.gemm_plan.as_ref(),
            LABEL,
            || g.graph.launch().map_err(|e| format!("{LABEL}: {e:?}")),
        )?;
        self.sync(LABEL)?;
        self.overflowed = scale.is_some() && self.overflow.read(&stream)? != 0;
        // A measured tail decided the skip on the device: an overflow, a
        // spike, or a norm that is not finite, which errs as the eager tail
        // errs. The step counted before the replay goes back with it.
        let (skipped, grad_norm) = if self.tail_measures() {
            match read_tail(
                &self.ctx,
                &self.skip_flag,
                self.window.clip_coef(),
                self.overflowed,
            ) {
                Ok((skipped, norm)) => (skipped, Some(norm)),
                Err(e) => {
                    self.adamw.step -= 1;
                    return Err(e);
                }
            }
        } else {
            (self.overflowed, None)
        };
        if let Some(scaler) = self.scaler.as_mut() {
            scaler.update(self.overflowed);
        }
        if skipped {
            self.adamw.step -= 1;
        }
        self.forwarded = true;
        Ok(StepMetrics {
            step: self.adamw.step,
            graph_replayed: true,
            loss_scale: scale,
            overflow_skipped: scale.map(|_| self.overflowed),
            optimizer_stepped: !skipped,
            grad_norm,
        })
    }
}
