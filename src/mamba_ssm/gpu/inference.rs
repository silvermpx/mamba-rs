//! GPU Mamba inference engine (T=1 step with optional CUDA Graphs).
//!
//! Provides zero-copy persistent state on GPU, pre-allocated scratch buffers,
//! and optional CUDA Graph capture for minimal kernel launch overhead.
//!
//! All 12 existing CUDA kernels are reused — zero new kernel code needed.

use super::blas::{gpu_gemm_forward_dispatch, gpu_sgemm_forward_raw};
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::device::GpuDevice;
use super::dtype::WeightDtype;
use super::launch::{grid_1d, grid_norm};
use super::weights::{
    GpuMambaMixedWeights, GpuMambaWeights, MambaLayerWeightsView, MambaWeightsView,
};
use crate::config::MambaConfig;
use crate::weights::MambaWeights;
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// GPU Inference State
// ---------------------------------------------------------------------------

/// Persistent GPU Mamba state for T=1 inference.
///
/// Lives in VRAM across steps. Reset on episode/sequence boundaries.
///
/// Layout:
/// - conv: `[n_layers][batch * d_inner * d_conv]` (layer-major, matches GPU training)
/// - ssm:  `[n_layers][batch * d_inner * d_state]`
///
/// Note: GPU conv uses `d_conv` slots (full window), while CPU inference uses
/// `d_conv - 1` (history only). The CUDA kernel `conv1d_step_forward` handles
/// shift-register semantics with the full `d_conv` layout.
pub struct GpuInferenceState {
    pub conv: GpuBuffer,
    pub ssm: GpuBuffer,
    batch: usize,
    d_inner: usize,
    d_conv: usize,
    d_state: usize,
}

impl GpuInferenceState {
    /// Allocate zeroed inference state.
    pub fn zeros(
        stream: &Arc<cudarc::driver::CudaStream>,
        batch: usize,
        cfg: &MambaConfig,
    ) -> Result<Self, String> {
        let di = cfg.d_inner();
        let conv_len = cfg.n_layers * batch * di * cfg.d_conv;
        let ssm_len = cfg.n_layers * batch * di * cfg.d_state;
        Ok(Self {
            conv: GpuBuffer::zeros(stream, conv_len)?,
            ssm: GpuBuffer::zeros(stream, ssm_len)?,
            batch,
            d_inner: di,
            d_conv: cfg.d_conv,
            d_state: cfg.d_state,
        })
    }

    /// Reset all state to zero (episode boundary).
    pub fn reset(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        self.conv.zero(stream)?;
        self.ssm.zero(stream)
    }

    /// Per-layer conv state offset in floats.
    pub fn conv_offset(&self, layer: usize) -> usize {
        layer * self.batch * self.d_inner * self.d_conv
    }

    /// Per-layer SSM state offset in floats.
    pub fn ssm_offset(&self, layer: usize) -> usize {
        layer * self.batch * self.d_inner * self.d_state
    }

    /// Number of batch samples.
    pub fn batch(&self) -> usize {
        self.batch
    }
}

// ---------------------------------------------------------------------------
// GPU Inference Scratch
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for GPU inference (reused every step).
///
/// All GPU buffers are sized for a fixed batch size. Host staging buffers
/// handle H2D/D2H transfers outside CUDA Graph capture.
pub struct GpuInferenceScratch {
    // GPU scratch (reused per step)
    pub gpu_input: GpuBuffer,
    pub temporal: GpuBuffer,
    pub residual: GpuBuffer,
    pub proj: GpuBuffer,
    pub x_branch: GpuBuffer,
    pub gate_silu: GpuBuffer,
    pub u: GpuBuffer,
    pub xdbl: GpuBuffer,
    pub dt_gather: GpuBuffer,
    pub delta: GpuBuffer,
    pub b_buf: GpuBuffer,
    pub c_buf: GpuBuffer,
    pub y: GpuBuffer,
    pub rms_buf: GpuBuffer,
}

impl GpuInferenceScratch {
    /// Allocate scratch buffers for the given batch size and config.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        batch: usize,
        cfg: &MambaConfig,
        input_dim: usize,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dt_rank = cfg.dt_rank();
        let xdbl_dim = cfg.xdbl_dim();
        Ok(Self {
            gpu_input: GpuBuffer::zeros(stream, batch * input_dim)?,
            temporal: GpuBuffer::zeros(stream, batch * dm)?,
            residual: GpuBuffer::zeros(stream, batch * dm)?,
            proj: GpuBuffer::zeros(stream, batch * 2 * di)?,
            x_branch: GpuBuffer::zeros(stream, batch * di)?,
            gate_silu: GpuBuffer::zeros(stream, batch * di)?,
            u: GpuBuffer::zeros(stream, batch * di)?,
            xdbl: GpuBuffer::zeros(stream, batch * xdbl_dim)?,
            dt_gather: GpuBuffer::zeros(stream, batch * dt_rank)?,
            delta: GpuBuffer::zeros(stream, batch * di)?,
            b_buf: GpuBuffer::zeros(stream, batch * ds)?,
            c_buf: GpuBuffer::zeros(stream, batch * ds)?,
            y: GpuBuffer::zeros(stream, batch * di)?,
            rms_buf: GpuBuffer::zeros(stream, batch)?,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU Inference Engine
// ---------------------------------------------------------------------------

/// GPU Mamba inference engine — owns kernels, weights, cuBLAS.
///
/// Lifecycle:
/// 1. `GpuMambaInference::new()` — compile kernels, upload weights
/// 2. Allocate state + scratch via `alloc_state()` / `alloc_scratch()`
/// 3. Call `step()` each timestep
/// 4. Optionally call `capture_graph()` for ~2-5x speedup
/// 5. Call `state.reset()` on episode boundaries
pub struct GpuMambaInference {
    ctx: GpuCtx,
    weights: GpuMambaWeights,
    a_neg_all: GpuBuffer,
    cfg: MambaConfig,
    input_dim: usize,
    batch: usize,
    /// When true (HF Mamba with no input_proj), skip input projection and copy
    /// `gpu_input` → `temporal` directly. Mirrors CPU `mamba_step_no_proj`.
    identity_proj: bool,
    graph: Option<cudarc::driver::CudaGraph>,
    /// Raw pointers captured during graph capture for runtime validation.
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
}

impl GpuMambaInference {
    /// Create inference engine: compile kernels, upload weights.
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let ctx = GpuCtx::new(device)?;

        let weights = GpuMambaWeights::from_cpu(&ctx.stream, cpu_weights, &cfg)?;

        // Precompute a_neg = -exp(a_log) for all layers
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let total_aneg = cfg.n_layers * di * ds;
        let a_neg_all = GpuBuffer::zeros(&ctx.stream, total_aneg)?;
        for (layer_idx, lw) in weights.layers.iter().enumerate() {
            let offset = layer_idx * di * ds;
            let dst_ptr = a_neg_all.raw_ptr_at(&ctx.stream, offset);
            let src_ptr = lw.a_log.ptr();
            let n_i = (di * ds) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.exp_negate);
            builder.arg(&dst_ptr);
            builder.arg(&src_ptr);
            builder.arg(&n_i);
            unsafe { builder.launch(grid_1d(di * ds)) }
                .map_err(|e| format!("exp_negate layer {layer_idx}: {e:?}"))?;
        }

        let identity_proj = cpu_weights.input_proj_w.is_empty();

        Ok(Self {
            ctx,
            weights,
            a_neg_all,
            cfg,
            input_dim,
            batch,
            identity_proj,
            graph: None,
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
        })
    }

    /// Capture CUDA Graph for the inference step.
    ///
    /// After capture, `step()` replays the graph instead of launching kernels
    /// individually, reducing launch overhead from ~50us to ~5us per step.
    ///
    /// Call after at least one warmup `step()` to stabilize kernel launches.
    /// H2D/D2H transfers remain outside the graph.
    pub fn capture_graph(
        &mut self,
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-capture sync: {e:?}"))?;

        self.ctx
            .stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| format!("begin_capture: {e:?}"))?;

        let capture_result = self.step_kernels(state, scratch);
        if capture_result.is_err() {
            // Must end capture to restore stream to normal mode, otherwise
            // the stream is left permanently in capture mode and all
            // subsequent operations will silently fail.
            let _ = self.ctx.stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return capture_result.map(|_| ());
        }

        let graph = self.ctx.stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| format!("end_capture: {e:?}"))?;

        self.graph = graph;
        self.captured_state_ptr = state.conv.cached_ptr();
        self.captured_scratch_ptr = scratch.gpu_input.cached_ptr();
        Ok(())
    }

    /// Whether a CUDA Graph has been captured.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Allocate zeroed inference state for this engine's batch size.
    pub fn alloc_state(&self) -> Result<GpuInferenceState, String> {
        GpuInferenceState::zeros(&self.ctx.stream, self.batch, &self.cfg)
    }

    /// Allocate scratch buffers for this engine.
    pub fn alloc_scratch(&self) -> Result<GpuInferenceScratch, String> {
        GpuInferenceScratch::new(&self.ctx.stream, self.batch, &self.cfg, self.input_dim)
    }

    /// Run one inference step: input → output.
    ///
    /// `input`: `[batch * input_dim]` on CPU.
    /// `output`: `[batch * d_model]` on CPU.
    /// State is mutated in-place (conv + SSM updated).
    ///
    /// # CUDA Graph mode
    /// When a graph is captured, all GPU pointers are baked in at capture time.
    /// The `state` and `scratch` parameters MUST be the same objects used during
    /// capture — passing different buffers silently operates on the original ones.
    /// Use [`GpuMambaBackbone`] which owns state/scratch and guarantees this.
    pub fn step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        // H2D: upload raw input (outside graph)
        scratch.gpu_input.upload(&self.ctx.stream, input)?;

        // Run GPU kernel pipeline (graph replay or individual launches)
        if let Some(ref g) = self.graph {
            assert_eq!(
                state.conv.cached_ptr(),
                self.captured_state_ptr,
                "CUDA Graph replay requires the same state buffers used during capture"
            );
            assert_eq!(
                scratch.gpu_input.cached_ptr(),
                self.captured_scratch_ptr,
                "CUDA Graph replay requires the same scratch buffers used during capture"
            );
            g.launch().map_err(|e| format!("graph launch: {e:?}"))?;
        } else {
            self.step_kernels(state, scratch)?;
        }

        // Sync: ensure all GPU work completes before D2H download.
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        scratch.temporal.download(&self.ctx.stream, output)?;
        Ok(())
    }

    /// Run backbone step without D2H download. Returns GPU temporal pointer.
    /// Use for chaining with lm_head SGEMM on GPU without round-trip.
    pub fn step_gpu_only(
        &self,
        input: &[f32],
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        scratch.gpu_input.upload(&self.ctx.stream, input)?;
        if let Some(ref g) = self.graph {
            assert_eq!(state.conv.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            g.launch().map_err(|e| format!("graph launch: {e:?}"))?;
        } else {
            self.step_kernels(state, scratch)?;
        }
        Ok(())
    }

    /// Get the temporal output buffer (on GPU). Valid after `step_gpu_only`.
    pub fn temporal_buffer<'a>(&self, scratch: &'a GpuInferenceScratch) -> &'a GpuBuffer {
        &scratch.temporal
    }

    /// Launch the full T=1 forward pipeline on GPU.
    ///
    /// Pipeline per layer:
    /// ```text
    /// RmsNorm → in_proj SGEMM → split_gate_silu → conv1d_step → silu →
    /// x_proj SGEMM → gather_cols(dt) → dt_proj SGEMM → softplus →
    /// gather_bc → ssm_step → elementwise_mul → out_proj SGEMM → residual_add
    /// ```
    fn step_kernels(
        &self,
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        // F32 path uses the typed-kernel dispatch via generic. This should be
        // identical to the legacy f32-only kernels because TypedKernel.get(F32)
        // returns the _f32 suffix variant which is the same PTX as legacy.
        self.step_kernels_generic(&self.weights, state, scratch)
    }

    /// Generic step pipeline — works with any weight view (f32 or bf16/f16).
    /// Bulk weights dispatch to SGEMM (f32) or cublasGemmEx (bf16/f16).
    /// Always-f32 weights read directly from their pointers.
    pub(super) fn step_kernels_generic<W: MambaWeightsView>(
        &self,
        weights: &W,
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        let b = self.batch;
        let cfg = &self.cfg;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let dt_rank = cfg.dt_rank();
        let xdbl_dim = cfg.xdbl_dim();
        let d_conv = cfg.d_conv;
        let k = &self.ctx.kernels;

        // Input projection: [B, input_dim] → [B, d_model]
        // HF Mamba models have no input_proj (embedding is already d_model).
        // Identity branch mirrors CPU `mamba_step_no_proj`.
        if self.identity_proj {
            debug_assert_eq!(
                self.input_dim, dm,
                "identity_proj requires input_dim == d_model"
            );
            scratch
                .temporal
                .copy_from_raw(&scratch.gpu_input, &self.ctx.stream)?;
        } else {
            let (ipw_ptr, ipw_dtype) = weights.input_proj_w();
            gpu_gemm_forward_dispatch(
                &self.ctx,
                &mut scratch.temporal,
                &scratch.gpu_input,
                ipw_ptr,
                ipw_dtype,
                Some(weights.input_proj_b()),
                (b, self.input_dim, dm),
            )?;
        }

        let f32_sz = std::mem::size_of::<f32>() as u64;

        for layer_idx in 0..weights.n_layers() {
            let lw = weights.layer(layer_idx);
            let conv_ptr = state.conv.cached_ptr() + (state.conv_offset(layer_idx) as u64) * f32_sz;
            let ssm_ptr = state.ssm.cached_ptr() + (state.ssm_offset(layer_idx) as u64) * f32_sz;
            let aneg_ptr = self.a_neg_all.cached_ptr() + (layer_idx * di * ds) as u64 * f32_sz;

            // F1: Save residual + RmsNorm
            scratch
                .residual
                .copy_from_raw(&scratch.temporal, &self.ctx.stream)?;
            {
                let b_i = b as i32;
                let dm_i = dm as i32;
                let eps: f32 = 1e-5;
                let mut bld = self.ctx.stream.launch_builder(&k.rmsnorm_fwd);
                let t_ptr = scratch.temporal.cached_ptr();
                let rms_ptr = scratch.rms_buf.cached_ptr();
                let res_ptr = scratch.residual.cached_ptr();
                bld.arg(&t_ptr); // output overwrites temporal
                bld.arg(&rms_ptr);
                bld.arg(&res_ptr); // input = saved residual
                let nw = lw.norm_weight();
                bld.arg(&nw);
                bld.arg(&b_i);
                bld.arg(&dm_i);
                bld.arg(&eps);
                unsafe { bld.launch(grid_norm(b, dm)) }
                    .map_err(|e| format!("rmsnorm_fwd L{layer_idx}: {e:?}"))?;
            }

            // F2: in_proj SGEMM [B, d_model] → [B, 2*d_inner]
            let (ipw, ipw_dt) = lw.in_proj_w();
            gpu_gemm_forward_dispatch(
                &self.ctx,
                &mut scratch.proj,
                &scratch.temporal,
                ipw,
                ipw_dt,
                None,
                (b, dm, 2 * di),
            )?;

            // F3: split x + gate SiLU
            {
                let b_i = b as i32;
                let di_i = di as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.split_gate_silu);
                let xb_ptr = scratch.x_branch.cached_ptr();
                bld.arg(&xb_ptr);
                let g_ptr = scratch.gate_silu.cached_ptr();
                let p_ptr = scratch.proj.cached_ptr();
                bld.arg(&g_ptr); // gate_pre
                bld.arg(&g_ptr); // gate_post (SiLU'd)
                bld.arg(&p_ptr);
                bld.arg(&b_i);
                bld.arg(&di_i);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("split_gate_silu L{layer_idx}: {e:?}"))?;
            }

            // F4: conv1d_step (no SiLU) → u buffer
            {
                let b_i = b as i32;
                let di_i = di as i32;
                let dc_i = d_conv as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.conv1d_step_fwd);
                let u_ptr = scratch.u.cached_ptr();
                let xb_ptr2 = scratch.x_branch.cached_ptr();
                bld.arg(&u_ptr);
                bld.arg(&conv_ptr); // state mutated in-place
                bld.arg(&xb_ptr2);
                let cw = lw.conv1d_weight();
                let cb = lw.conv1d_bias();
                bld.arg(&cw);
                bld.arg(&cb);
                bld.arg(&b_i);
                bld.arg(&di_i);
                bld.arg(&dc_i);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("conv1d_step L{layer_idx}: {e:?}"))?;
            }

            // F4b: SiLU on conv output (conv1d_step does NOT fuse SiLU)
            {
                let n = (b * di) as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.silu_fwd);
                let u_silu_ptr = scratch.u.cached_ptr();
                bld.arg(&u_silu_ptr); // in-place
                bld.arg(&n);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("silu_fwd L{layer_idx}: {e:?}"))?;
            }

            // F5: x_proj GEMM [B, d_inner] → [B, xdbl_dim]
            let (xpw, xpw_dt) = lw.x_proj_w();
            gpu_gemm_forward_dispatch(
                &self.ctx,
                &mut scratch.xdbl,
                &scratch.u,
                xpw,
                xpw_dt,
                None,
                (b, di, xdbl_dim),
            )?;

            // F6: gather dt from xdbl
            {
                let b_i = b as i32;
                let xdbl_i = xdbl_dim as i32;
                let dt_i = dt_rank as i32;
                let offset: i32 = 0;
                let mut bld = self.ctx.stream.launch_builder(&k.gather_cols);
                let dtg_ptr = scratch.dt_gather.cached_ptr();
                let xdbl_ptr = scratch.xdbl.cached_ptr();
                bld.arg(&dtg_ptr);
                bld.arg(&xdbl_ptr);
                bld.arg(&b_i);
                bld.arg(&xdbl_i);
                bld.arg(&dt_i);
                bld.arg(&offset);
                unsafe { bld.launch(grid_1d(b * dt_rank)) }
                    .map_err(|e| format!("gather_cols dt L{layer_idx}: {e:?}"))?;
            }

            // F7: dt_proj GEMM + softplus
            let (dpw, dpw_dt) = lw.dt_proj_w();
            gpu_gemm_forward_dispatch(
                &self.ctx,
                &mut scratch.delta,
                &scratch.dt_gather,
                dpw,
                dpw_dt,
                Some(lw.dt_proj_b()),
                (b, dt_rank, di),
            )?;
            {
                let n = (b * di) as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.softplus_copy);
                let d_ptr = scratch.delta.cached_ptr();
                bld.arg(&d_ptr); // dst
                bld.arg(&d_ptr); // src (in-place ok)
                bld.arg(&n);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("softplus L{layer_idx}: {e:?}"))?;
            }

            // F8: gather B, C from xdbl
            {
                let b_i = b as i32;
                let xdbl_i = xdbl_dim as i32;
                let ds_i = ds as i32;
                let b_off = dt_rank as i32;
                let c_off = (dt_rank + ds) as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.gather_bc_cols);
                let bb_ptr = scratch.b_buf.cached_ptr();
                let cb_ptr = scratch.c_buf.cached_ptr();
                let xdbl_bc_ptr = scratch.xdbl.cached_ptr();
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&xdbl_bc_ptr);
                bld.arg(&b_i);
                bld.arg(&xdbl_i);
                bld.arg(&ds_i);
                bld.arg(&b_off);
                bld.arg(&c_off);
                unsafe { bld.launch(grid_1d(b * ds)) }
                    .map_err(|e| format!("gather_bc L{layer_idx}: {e:?}"))?;
            }

            // F9: SSM step (mutates ssm_state)
            {
                let b_i = b as i32;
                let di_i = di as i32;
                let ds_i = ds as i32;
                let dp = lw.d_param();
                let mut bld = self.ctx.stream.launch_builder(&k.ssm_step_fwd);
                let y_ssm_ptr = scratch.y.cached_ptr();
                let delta_ssm_ptr = scratch.delta.cached_ptr();
                let u_ssm_ptr = scratch.u.cached_ptr();
                let b_ssm_ptr = scratch.b_buf.cached_ptr();
                let c_ssm_ptr = scratch.c_buf.cached_ptr();
                bld.arg(&ssm_ptr);
                bld.arg(&y_ssm_ptr);
                bld.arg(&delta_ssm_ptr);
                bld.arg(&u_ssm_ptr);
                bld.arg(&b_ssm_ptr);
                bld.arg(&c_ssm_ptr);
                bld.arg(&aneg_ptr);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_step L{layer_idx}: {e:?}"))?;
            }

            // F10: gating y * gate_silu
            {
                let n = (b * di) as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.elementwise_mul);
                let y_ptr = scratch.y.cached_ptr();
                let gs_ptr = scratch.gate_silu.cached_ptr();
                bld.arg(&y_ptr);
                bld.arg(&y_ptr);
                bld.arg(&gs_ptr);
                bld.arg(&n);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("gating L{layer_idx}: {e:?}"))?;
            }

            // F11: out_proj GEMM [B, d_inner] → [B, d_model]
            let (opw, opw_dt) = lw.out_proj_w();
            gpu_gemm_forward_dispatch(
                &self.ctx,
                &mut scratch.temporal,
                &scratch.y,
                opw,
                opw_dt,
                None,
                (b, di, dm),
            )?;

            // F12: residual add
            {
                let n = (b * dm) as i32;
                let mut bld = self.ctx.stream.launch_builder(&k.residual_add);
                let t_ptr = scratch.temporal.cached_ptr();
                let r_ptr = scratch.residual.cached_ptr();
                bld.arg(&t_ptr);
                bld.arg(&r_ptr);
                bld.arg(&t_ptr); // temporal += residual
                bld.arg(&n);
                unsafe { bld.launch(grid_1d(b * dm)) }
                    .map_err(|e| format!("residual L{layer_idx}: {e:?}"))?;
            }
        }

        // Final RmsNorm (norm_f)
        {
            let b_i = b as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let mut bld = self.ctx.stream.launch_builder(&k.rmsnorm_fwd);
            let t_ptr = scratch.temporal.cached_ptr();
            let rms_ptr = scratch.rms_buf.cached_ptr();
            bld.arg(&t_ptr);
            bld.arg(&rms_ptr);
            bld.arg(&t_ptr);
            let nfw = weights.norm_f_weight();
            bld.arg(&nfw);
            bld.arg(&b_i);
            bld.arg(&dm_i);
            bld.arg(&eps);
            unsafe { bld.launch(grid_norm(b, dm)) }.map_err(|e| format!("norm_f: {e:?}"))?;
        }

        Ok(())
    }

    /// Config reference.
    pub fn config(&self) -> &MambaConfig {
        &self.cfg
    }

    /// Batch size.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Access the GPU context.
    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision inference engine (bf16/f16 weight storage, f32 compute).
// ---------------------------------------------------------------------------

/// GPU Mamba inference with mixed-precision weights (bf16 or f16).
///
/// Bulk linear weights (in_proj, x_proj, dt_proj, out_proj, input_proj) are
/// stored in bf16/f16 to halve VRAM and memory bandwidth. All other tensors
/// (norms, biases, a_log, D, conv1d_weight) stay f32 for numerical stability.
/// Compute is f32 (CUBLAS_COMPUTE_32F) regardless of weight dtype.
pub struct GpuMambaInferenceMixed {
    engine: GpuMambaInference, // owns ctx + (possibly unused) f32 weights
    mixed_weights: GpuMambaMixedWeights,
    a_neg_all: GpuBuffer,
    graph: Option<cudarc::driver::CudaGraph>,
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
}

impl GpuMambaInferenceMixed {
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        cfg.validate()?;
        // Create f32 engine first (builds ctx, kernels, a_neg_all via CPU upload path).
        // We'll then discard its `weights` flat buffer and replace with mixed arena.
        let engine = GpuMambaInference::new(device, cpu_weights, cfg, input_dim, batch)?;
        let mixed_weights = GpuMambaMixedWeights::from_cpu(
            &engine.ctx.stream,
            cpu_weights,
            &cfg,
            bulk_dtype,
        )?;

        // Precompute a_neg into a separate arena (same as engine but from mixed weights'
        // f32 a_log — they match since a_log is f32 in both storages).
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let total_aneg = cfg.n_layers * di * ds;
        let a_neg_all = GpuBuffer::zeros(&engine.ctx.stream, total_aneg)?;
        for (layer_idx, lw) in mixed_weights.layers.iter().enumerate() {
            let offset = layer_idx * di * ds;
            let dst_ptr = a_neg_all.raw_ptr_at(&engine.ctx.stream, offset);
            let src_ptr = lw.a_log.ptr();
            let n_i = (di * ds) as i32;
            let mut builder = engine
                .ctx
                .stream
                .launch_builder(&engine.ctx.kernels.exp_negate);
            builder.arg(&dst_ptr);
            builder.arg(&src_ptr);
            builder.arg(&n_i);
            unsafe { builder.launch(grid_1d(di * ds)) }
                .map_err(|e| format!("exp_negate mixed L{layer_idx}: {e:?}"))?;
        }

        Ok(Self {
            engine,
            mixed_weights,
            a_neg_all,
            graph: None,
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
        })
    }

    pub fn step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        scratch.gpu_input.upload(&self.engine.ctx.stream, input)?;
        if let Some(ref g) = self.graph {
            assert_eq!(state.conv.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            g.launch().map_err(|e| format!("graph launch mixed: {e:?}"))?;
        } else {
            self.step_kernels_mixed(state, scratch)?;
        }
        self.engine
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        scratch.temporal.download(&self.engine.ctx.stream, output)?;
        Ok(())
    }

    pub fn step_gpu_only(
        &self,
        input: &[f32],
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        scratch.gpu_input.upload(&self.engine.ctx.stream, input)?;
        if let Some(ref g) = self.graph {
            assert_eq!(state.conv.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            g.launch().map_err(|e| format!("graph launch mixed: {e:?}"))?;
        } else {
            self.step_kernels_mixed(state, scratch)?;
        }
        Ok(())
    }

    fn step_kernels_mixed(
        &self,
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        // The generic pipeline already reads a_neg via a separate pointer;
        // for mixed we override that by temporarily pointing to our own a_neg_all.
        // Simpler: call step_kernels_generic but substitute a_neg_all lookup.
        // Since a_neg_all lives on self (not on weights), and step_kernels_generic
        // reads `self.a_neg_all`, we need to duplicate the loop or extract the
        // a_neg source. Easiest: temporarily swap the engine's a_neg_all.
        // NOTE: a_neg is derived from mixed_weights' f32 a_log — identical result.
        // So engine.a_neg_all is already correct (computed from same CPU a_log).
        self.engine
            .step_kernels_generic(&self.mixed_weights, state, scratch)
    }

    pub fn capture_graph(
        &mut self,
        state: &mut GpuInferenceState,
        scratch: &mut GpuInferenceScratch,
    ) -> Result<(), String> {
        self.engine
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("pre-capture sync: {e:?}"))?;
        self.engine
            .ctx
            .stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| format!("begin_capture mixed: {e:?}"))?;
        let captured_state_ptr = state.conv.cached_ptr();
        let captured_scratch_ptr = scratch.gpu_input.cached_ptr();
        if let Err(e) = self.step_kernels_mixed(state, scratch) {
            let _ = self.engine.ctx.stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(format!("capture body mixed failed: {e}"));
        }
        let graph = self
            .engine
            .ctx
            .stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| format!("end_capture mixed: {e:?}"))?
            .ok_or("no graph captured (mixed)")?;
        self.graph = Some(graph);
        self.captured_state_ptr = captured_state_ptr;
        self.captured_scratch_ptr = captured_scratch_ptr;
        Ok(())
    }

    pub fn alloc_state(&self) -> Result<GpuInferenceState, String> {
        self.engine.alloc_state()
    }

    pub fn alloc_scratch(&self) -> Result<GpuInferenceScratch, String> {
        self.engine.alloc_scratch()
    }

    pub fn config(&self) -> &MambaConfig {
        &self.engine.cfg
    }

    pub fn batch(&self) -> usize {
        self.engine.batch
    }

    pub fn ctx(&self) -> &GpuCtx {
        &self.engine.ctx
    }

    pub fn stream(&self) -> &Arc<cudarc::driver::CudaStream> {
        &self.engine.ctx.stream
    }

    pub fn bulk_dtype(&self) -> WeightDtype {
        self.mixed_weights.bulk_dtype
    }
}

// Mark a_neg_all used to silence warning (only used during new()).
#[allow(dead_code)]
impl GpuMambaInferenceMixed {
    fn _a_neg_keepalive(&self) -> &GpuBuffer {
        &self.a_neg_all
    }
}

// ---------------------------------------------------------------------------
// High-level GPU Backbone
// ---------------------------------------------------------------------------

/// High-level GPU Mamba backbone — owns engine + state + scratch.
///
/// Simple API for inference: `step()`, `reset()`, `capture_graph()`.
///
/// ```rust,no_run
/// use mamba_rs::MambaConfig;
/// use mamba_rs::gpu::inference::GpuMambaBackbone;
///
/// let cfg = MambaConfig::default();
/// let weights = mamba_rs::MambaWeights::init(&cfg, 128, 42);
/// let mut bb = GpuMambaBackbone::new(0, &weights, cfg, 128, 1).unwrap();
/// bb.capture_graph().unwrap(); // optional ~2-5x speedup
///
/// let input = vec![0.1f32; 128];
/// let mut output = vec![0.0f32; 128];
/// bb.step(&input, &mut output).unwrap();
/// bb.reset().unwrap();
/// ```
pub struct GpuMambaBackbone {
    engine: GpuMambaInference,
    state: GpuInferenceState,
    scratch: GpuInferenceScratch,
}

impl GpuMambaBackbone {
    /// Create a GPU backbone from CPU weights.
    ///
    /// - `gpu_ordinal`: GPU device index (0 = first GPU)
    /// - `batch`: number of parallel inference streams
    pub fn new(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let engine = GpuMambaInference::new(&device, cpu_weights, cfg, input_dim, batch)?;
        let state = engine.alloc_state()?;
        let scratch = engine.alloc_scratch()?;
        Ok(Self {
            engine,
            state,
            scratch,
        })
    }

    /// Run one inference step.
    ///
    /// `input`: `[batch * input_dim]` on CPU.
    /// `output`: `[batch * d_model]` on CPU.
    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.engine
            .step(input, output, &mut self.state, &mut self.scratch)
    }

    /// Reset recurrent state (episode/sequence boundary).
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.ctx.stream)
    }

    /// Capture CUDA Graph for faster inference.
    /// Call after at least one warmup `step()`.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        // Warmup step (dummy input, result discarded)
        let input = vec![0.0f32; self.engine.batch * self.engine.input_dim];
        let mut output = vec![0.0f32; self.engine.batch * self.engine.cfg.d_model];
        self.engine
            .step(&input, &mut output, &mut self.state, &mut self.scratch)?;
        self.state.reset(&self.engine.ctx.stream)?;

        // Capture
        self.engine
            .capture_graph(&mut self.state, &mut self.scratch)
    }

    /// Config reference.
    pub fn config(&self) -> &MambaConfig {
        self.engine.config()
    }

    /// Batch size.
    pub fn batch(&self) -> usize {
        self.engine.batch()
    }

    /// Whether CUDA Graph is captured.
    pub fn has_graph(&self) -> bool {
        self.engine.has_graph()
    }

    /// Access the GPU compute context (for external SGEMM calls like lm_head).
    pub fn ctx(&self) -> &GpuCtx {
        &self.engine.ctx
    }

    /// Access the GPU stream.
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        &self.engine.ctx.stream
    }

    /// Run backbone step, keep output on GPU (no D2H). For chaining with lm_head.
    pub fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        self.engine
            .step_gpu_only(input, &mut self.state, &mut self.scratch)
    }

    /// GPU temporal buffer pointer (valid after `step_gpu_only`).
    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.scratch.temporal.cached_ptr()
    }

    /// Download temporal from GPU to CPU.
    pub fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
        self.engine
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        self.scratch
            .temporal
            .download(&self.engine.ctx.stream, output)
    }
}

// ---------------------------------------------------------------------------
// High-level mixed-precision GPU Backbone
// ---------------------------------------------------------------------------

/// High-level mixed-precision GPU Mamba backbone.
/// Weights stored in bf16/f16, compute in f32 (CUBLAS_COMPUTE_32F).
pub struct GpuMambaBackboneMixed {
    engine: GpuMambaInferenceMixed,
    state: GpuInferenceState,
    scratch: GpuInferenceScratch,
}

impl GpuMambaBackboneMixed {
    pub fn new(
        gpu_ordinal: usize,
        cpu_weights: &MambaWeights,
        cfg: MambaConfig,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let engine = GpuMambaInferenceMixed::new(
            &device,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            bulk_dtype,
        )?;
        let state = engine.alloc_state()?;
        let scratch = engine.alloc_scratch()?;
        Ok(Self {
            engine,
            state,
            scratch,
        })
    }

    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.engine.step(input, output, &mut self.state, &mut self.scratch)
    }

    pub fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        self.engine
            .step_gpu_only(input, &mut self.state, &mut self.scratch)
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.engine.ctx.stream)
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        let input = vec![0.0f32; self.engine.engine.batch * self.engine.engine.input_dim];
        let mut output = vec![0.0f32; self.engine.engine.batch * self.engine.engine.cfg.d_model];
        self.engine.step(&input, &mut output, &mut self.state, &mut self.scratch)?;
        self.state.reset(&self.engine.engine.ctx.stream)?;
        self.engine.capture_graph(&mut self.state, &mut self.scratch)
    }

    pub fn config(&self) -> &MambaConfig {
        self.engine.config()
    }

    pub fn ctx(&self) -> &GpuCtx {
        self.engine.ctx()
    }

    pub fn stream(&self) -> &Arc<cudarc::driver::CudaStream> {
        self.engine.stream()
    }

    pub fn bulk_dtype(&self) -> WeightDtype {
        self.engine.bulk_dtype()
    }

    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.scratch.temporal.cached_ptr()
    }

    pub fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
        self.engine
            .engine
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        self.scratch
            .temporal
            .download(&self.engine.engine.ctx.stream, output)
    }
}
