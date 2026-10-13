//! A Mamba-2 prompt window captured as one CUDA graph: the input
//! projection, every layer's norm, projections, conv, scan and gated
//! output, and the final norm, over one window length. The window's
//! buffers hold the graph, so it lives as long as they do; the host uploads
//! the prompt before a replay and reads the rows after it, outside the
//! graph.

use std::rc::Rc;

use cudarc::driver::CudaGraph;

use super::{Mamba2GpuEngine, Mamba2GpuPrefill, Mamba2GpuState};
use crate::gpu::context::{GemmRoute, GpuCtxResources};
use crate::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::gpu::graph_capture::{
    capture_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::gpu::kernel_identity::CapturedGemmGraphPlan;
use crate::gpu::kernels::CudaModuleAnchors;

/// A captured window and what its replay checks. It holds the kernels'
/// module and the context's resources, so they outlive the graph.
pub(super) struct PrefillGraph {
    graph: CudaGraph,
    gemm_route: GemmRoute,
    gemm_plan: Option<CapturedGemmGraphPlan>,
    /// The state's conv and SSM buffers the graph reads and writes.
    state: [u64; 2],
    half_staging: u64,
    bi_upcast: [u64; 3],
    _modules: CudaModuleAnchors,
    ctx: Rc<GpuCtxResources>,
}

impl Drop for PrefillGraph {
    fn drop(&mut self) {
        // A replay may still be in flight: the graph goes once it finished.
        let _ = self.ctx.stream.synchronize();
    }
}

fn state_ptrs(state: &Mamba2GpuState) -> [u64; 2] {
    [state.conv.cached_ptr(), state.ssm.cached_ptr()]
}

impl Mamba2GpuEngine {
    /// The window's kernels on the uploaded prompt: the window's graph
    /// when it holds one, else an eager run, whose GEMM inventory a
    /// capture takes.
    pub(super) fn run_prefill(
        &self,
        state: &mut Mamba2GpuState,
        pf: &mut Mamba2GpuPrefill,
    ) -> Result<(), String> {
        if let Some(g) = &pf.graph {
            return self.replay_prefill(g, state);
        }
        if self.has_gemm_work() {
            prepare_inference_arch_rung(&self.ctx)?;
        }
        self.presize_gemm_scratch(self.batch * pf.seq_len)?;
        pf.eager_gemm_manifest = None;
        let manifest = self
            .ctx
            .record_eager_gemm_manifest(|| self.prefill_kernels(state, pf))?;
        pf.eager_gemm_manifest = Some(manifest);
        Ok(())
    }

    /// Capture `pf`'s window as one CUDA graph over `state`; later
    /// prefills of the window on that state replay it with the bits of the
    /// eager run. Takes the GEMM inventory of the eager prefill of the
    /// window that must precede it. The replay refuses another state, a
    /// changed GEMM route or moved GEMM staging, and
    /// [`Mamba2GpuPrefill::clear_graph`] drops the graph. After a capture
    /// the context's GEMM staging can no longer grow: capture the longest
    /// window first.
    ///
    /// # Safety
    ///
    /// `state` must stay allocated, and the engine's weights, kernels and
    /// context unchanged, for as long as the graph replays.
    pub unsafe fn capture_prefill_graph(
        &self,
        state: &mut Mamba2GpuState,
        pf: &mut Mamba2GpuPrefill,
    ) -> Result<(), String> {
        self.check_state(state)?;
        if pf.graph.is_some() {
            return Err("the window holds a graph already: clear_graph first".into());
        }
        let manifest = pf.eager_gemm_manifest.take().ok_or_else(|| {
            "Mamba-2 prefill graph capture needs an eager prefill of the window first".to_string()
        })?;
        self.ctx.presize_bi_scratch()?;
        let gemm_route = self.ctx.gemm_route();
        self.ctx.freeze_graph_scratch();
        let (graph, gemm_plan) = unsafe {
            capture_into_graph_with_gemm_plan(&self.ctx, manifest.route_capacity, &manifest, || {
                self.prefill_kernels(state, pf)
            })
        }?;
        require_deterministic_gemm_graph_plan(
            &self.ctx,
            self.has_gemm_work(),
            gemm_plan.as_ref(),
            "Mamba-2 prefill graph capture",
        )?;
        pf.graph = Some(PrefillGraph {
            graph,
            gemm_route,
            gemm_plan,
            state: state_ptrs(state),
            half_staging: self.ctx.half_staging_ptr(),
            bi_upcast: self.ctx.bi_upcast_scratch_ptrs(),
            _modules: self.kernels.module_anchors(),
            ctx: self.ctx.resource_anchor(),
        });
        self.ctx.note_graph_capture();
        Ok(())
    }

    fn replay_prefill(&self, g: &PrefillGraph, state: &Mamba2GpuState) -> Result<(), String> {
        const LABEL: &str = "Mamba-2 prefill graph replay";
        if g.gemm_route != self.ctx.gemm_route() {
            return Err(format!("{LABEL}: the GEMM route changed since capture"));
        }
        if g.state != state_ptrs(state) {
            return Err(format!(
                "{LABEL}: the graph runs on the state it was captured with"
            ));
        }
        self.ctx
            .ensure_graph_scratch_ptrs(g.half_staging, g.bi_upcast, LABEL)?;
        with_validated_gemm_graph_launch(
            &self.ctx,
            self.has_gemm_work(),
            g.gemm_plan.as_ref(),
            LABEL,
            || g.graph.launch().map_err(|e| format!("{LABEL}: {e:?}")),
        )
    }
}
