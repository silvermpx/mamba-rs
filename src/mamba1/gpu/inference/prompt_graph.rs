//! A Mamba-1 prompt captured as one CUDA graph: the input projection when
//! the model has one, then every layer's prefill and the final norm, over
//! one prompt length. The prompt's buffers hold the graph, so it lives as
//! long as they do; the host uploads the rows before a replay, outside the
//! graph.

use std::rc::Rc;

use cudarc::driver::CudaGraph;

use super::{
    BackboneEngine, BackbonePrompt, BackboneScratch, MambaGpuBackbone, MambaGpuState,
    PromptScratch, project_prompt,
};
use crate::gpu::buffers::DeviceKeep;
use crate::gpu::context::{GemmRoute, GpuCtx, GpuCtxResources};
use crate::gpu::graph_capture::{
    capture_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::gpu::kernel_identity::CapturedGemmGraphPlan;
use crate::mamba1::gpu::prefill::{
    PrefillInputs, gpu_forward_inference_prefill, gpu_forward_inference_prefill_mixed,
};

/// A captured prompt and what its replay checks. It holds the context's
/// resources, which hold the kernels, so they outlive the graph.
pub(super) struct PromptGraph {
    graph: CudaGraph,
    gemm_route: GemmRoute,
    gemm_plan: Option<CapturedGemmGraphPlan>,
    has_gemm_work: bool,
    /// The state's conv and SSM buffers and the rows the prompt leaves.
    buffers: [u64; 3],
    half_staging: u64,
    bi_upcast: [u64; 3],
    guards_graph_scratch: bool,
    ctx: Rc<GpuCtxResources>,
}

impl Drop for PromptGraph {
    fn drop(&mut self) {
        // A replay may still be in flight: the graph goes once it finished.
        let _ = self.ctx.stream.synchronize();
    }
}

impl BackboneEngine {
    fn ctx(&self) -> &GpuCtx {
        match self {
            BackboneEngine::F32(e) => &e.ctx,
            BackboneEngine::Mixed(e) => e.ctx(),
        }
    }
}

fn buffers(state: &MambaGpuState, scratch: &BackboneScratch) -> [u64; 3] {
    [
        state.conv.cached_ptr(),
        state.ssm.cached_ptr(),
        scratch.temporal_ptr(true),
    ]
}

/// The prompt's kernels on the uploaded rows: the input projection when the
/// model has one, then the prefill of the engine's precision. No host
/// transfer, so a capture records exactly this.
fn prompt_kernels(
    engine: &BackboneEngine,
    scratch: &mut BackboneScratch,
    state: &mut MambaGpuState,
    prompt: &mut BackbonePrompt,
    shape: (usize, usize, usize),
) -> Result<(), String> {
    let BackbonePrompt {
        input: rows_in,
        projected,
        scratch: prompt_scratch,
        seq_len,
        ..
    } = prompt;
    match (engine, scratch, prompt_scratch) {
        (BackboneEngine::F32(e), BackboneScratch::F32(sc), PromptScratch::F32(pf)) => {
            let ip_out = project_prompt(
                &e.ctx,
                (&e.weights, e.identity_proj),
                rows_in,
                projected,
                shape,
            )?;
            gpu_forward_inference_prefill(
                &e.ctx,
                &mut sc.temporal,
                PrefillInputs {
                    ip_out_flat: ip_out,
                    weights: &e.weights,
                    a_neg_all: &e.a_neg_all,
                },
                state,
                pf,
            )
        }
        (BackboneEngine::Mixed(e), BackboneScratch::Mixed(sc), PromptScratch::Mixed(pf)) => {
            let ip_out = project_prompt(
                e.ctx(),
                (e.weights_mixed_ref(), e.engine_ref().identity_proj),
                rows_in,
                projected,
                shape,
            )?;
            gpu_forward_inference_prefill_mixed(
                e.ctx(),
                &sc.temporal,
                PrefillInputs {
                    ip_out_flat: ip_out,
                    weights: e.weights_mixed_ref(),
                    a_neg_all: e.a_neg_all_ref(),
                },
                state,
                pf,
            )?;
            if e.config().final_norm {
                return Ok(());
            }
            // A segment without the final norm hands out its f32 residual
            // stream: the prompt's last position becomes the decode rows'.
            let (rows, d_model) = (shape.0, shape.2);
            let batch = rows / *seq_len;
            last_rows_f32(
                e.ctx(),
                (&sc.residual, &pf.residual),
                (batch, *seq_len, d_model),
            )
        }
        _ => Err("engine/scratch dtype mismatch (internal invariant)".to_string()),
    }
}

/// The last position of every sequence of `src` (`[batch, seq_len,
/// d_model]` f32) into `dst` (`[batch, d_model]`).
fn last_rows_f32(
    ctx: &GpuCtx,
    (dst, src): (
        &crate::gpu::buffers::GpuBuffer,
        &crate::gpu::buffers::GpuBuffer,
    ),
    (batch, seq_len, d_model): (usize, usize, usize),
) -> Result<(), String> {
    use cudarc::driver::PushKernelArg;
    let (b_i, t_i, d_i) = (batch as i32, seq_len as i32, d_model as i32);
    let (dst_ptr, src_ptr) = (dst.cached_ptr(), src.cached_ptr());
    let mut bld = ctx.stream.launch_builder(&ctx.kernels.gather_last_timestep);
    bld.arg(&dst_ptr);
    bld.arg(&src_ptr);
    bld.arg(&b_i);
    bld.arg(&t_i);
    bld.arg(&d_i);
    unsafe { bld.launch(crate::gpu::launch::grid_1d_cover(batch * d_model)?) }
        .map(|_| ())
        .map_err(|e| format!("last residual rows: {e:?}"))
}

fn replay(
    g: &PromptGraph,
    ctx: &GpuCtx,
    state: &MambaGpuState,
    scratch: &BackboneScratch,
) -> Result<(), String> {
    const LABEL: &str = "Mamba-1 prompt graph replay";
    if g.gemm_route != ctx.gemm_route() {
        return Err(format!("{LABEL}: the GEMM route changed since capture"));
    }
    if g.buffers != buffers(state, scratch) {
        return Err(format!(
            "{LABEL}: the graph runs on the state and rows it was captured with"
        ));
    }
    if g.guards_graph_scratch {
        ctx.ensure_graph_scratch_ptrs(g.half_staging, g.bi_upcast, LABEL)?;
    }
    with_validated_gemm_graph_launch(ctx, g.has_gemm_work, g.gemm_plan.as_ref(), LABEL, || {
        g.graph.launch().map_err(|e| format!("{LABEL}: {e:?}"))
    })
}

impl MambaGpuBackbone {
    /// The uploaded prompt's kernels: its graph when the buffers hold one,
    /// else an eager run, whose GEMM inventory a capture takes.
    pub(super) fn run_prompt_kernels(
        &mut self,
        prompt: &mut BackbonePrompt,
        shape: (usize, usize, usize),
    ) -> Result<(), String> {
        if let Some(g) = &prompt.graph {
            return replay(g, self.engine.ctx(), &self.state, &self.scratch);
        }
        if let BackboneEngine::Mixed(e) = &self.engine {
            e.prepare_prompt_scratch(prompt.seq_len)?;
        }
        let Self {
            engine,
            state,
            scratch,
            ..
        } = self;
        prompt.eager_gemm_manifest = None;
        let manifest = engine
            .ctx()
            .record_eager_gemm_manifest(|| prompt_kernels(engine, scratch, state, prompt, shape))?;
        prompt.eager_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Capture the prompt pass of `seq_len` tokens per sequence as one CUDA
    /// graph, which later [`Self::prefill`] calls of that length replay with
    /// the eager bits. The capture leaves the state and the rows where the
    /// caller left them: its warm-up prompt on zero input is undone. A
    /// prompt of another length drops the graph with the buffers it runs
    /// on. Capture the longest prompt before the decode graph: afterwards
    /// the GEMM staging can no longer grow.
    pub fn capture_prefill_graph(&mut self, seq_len: usize) -> Result<(), String> {
        let (input_dim, d_model) = (self.input_dim(), self.d_model());
        let rows = self.batch() * seq_len;
        if rows == 0 {
            return Err("a prompt graph needs at least one row".into());
        }
        let kept = self.prompt.take().filter(|p| p.seq_len == seq_len);
        let mut prompt = match kept {
            Some(prompt) => prompt,
            None => self.alloc_prompt(seq_len)?,
        };
        let result = self.capture_prompt(&mut prompt, (rows, input_dim, d_model));
        self.prompt = Some(prompt);
        result
    }

    fn capture_prompt(
        &mut self,
        prompt: &mut BackbonePrompt,
        shape: (usize, usize, usize),
    ) -> Result<(), String> {
        prompt.graph = None;
        let stream = self.stream().clone();
        let f = std::mem::size_of::<f32>();
        let keep = DeviceKeep::take(
            &stream,
            &[
                (self.state.conv.cached_ptr(), self.state.conv.len() * f),
                (self.state.ssm.cached_ptr(), self.state.ssm.len() * f),
                (
                    self.temporal_ptr(),
                    self.batch() * shape.2 * self.temporal_dtype().size_bytes(),
                ),
            ],
        )?;
        let zeros = vec![0.0f32; shape.0 * shape.1];
        self.run_prompt(prompt, &zeros, shape)?;
        keep.restore(&stream)?;
        let manifest = prompt.eager_gemm_manifest.take().ok_or_else(|| {
            "Mamba-1 prompt graph capture needs an eager run of the prompt first".to_string()
        })?;
        let has_gemm_work = prompt.projected.is_some() || self.config().n_layers != 0;
        let guards_graph_scratch = matches!(self.engine, BackboneEngine::Mixed(_));
        let Self {
            engine,
            state,
            scratch,
            ..
        } = self;
        let ctx = engine.ctx();
        ctx.presize_bi_scratch()?;
        let gemm_route = ctx.gemm_route();
        if guards_graph_scratch {
            ctx.freeze_graph_scratch();
        }
        let buffers = buffers(state, scratch);
        let (graph, gemm_plan) = unsafe {
            capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, &manifest, || {
                prompt_kernels(engine, scratch, state, prompt, shape)
            })
        }?;
        require_deterministic_gemm_graph_plan(
            ctx,
            has_gemm_work,
            gemm_plan.as_ref(),
            "Mamba-1 prompt graph capture",
        )?;
        prompt.graph = Some(PromptGraph {
            graph,
            gemm_route,
            gemm_plan,
            has_gemm_work,
            buffers,
            half_staging: ctx.half_staging_ptr(),
            bi_upcast: ctx.bi_upcast_scratch_ptrs(),
            guards_graph_scratch,
            ctx: ctx.resource_anchor(),
        });
        ctx.note_graph_capture();
        Ok(())
    }

    /// Whether prompts of the kept length replay a captured graph.
    pub fn has_prefill_graph(&self) -> bool {
        self.prompt.as_ref().is_some_and(|p| p.graph.is_some())
    }

    /// Drop the prompt graph, once a replay in flight finished: prompts run
    /// eagerly until the next capture.
    pub fn clear_prefill_graph(&mut self) {
        if let Some(p) = self.prompt.as_mut() {
            p.graph = None;
        }
    }
}
