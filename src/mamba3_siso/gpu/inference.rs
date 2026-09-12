//! Mamba-3 SISO GPU inference (T=1 step + CUDA Graph).
//!
//! 10-phase forward per layer:
//!   F1: RMSNorm → F2: in_proj GEMM → F3: m3_split (8-way + fused)
//!   F4: BCNorm + bias + RoPE, the angles advanced inside
//!   F6: m3_step_fwd (trapezoidal SSM, its coefficients computed inside) → F7: output gating
//!   F8: out_proj GEMM → F9: residual add
//! Final: F10: norm_f RMSNorm
//!
//! Weight format: flat buffer + WeightSlice (CUDA Graph safe).
//! State: 4 persistent buffers (SSM + K + V + angle) per layer.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use super::kernels::{Mamba3Kernels, bcnorm_fwd_bc_cfg};
use super::weights::GpuMamba3WeightsInf;
use crate::mamba_ssm::gpu::blas::gpu_gemm_f32_forward_ptrs;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode, GpuCtx};
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba_ssm::gpu::gemm_bi_inference::prepare_inference_arch_rung;
use crate::mamba_ssm::gpu::graph_capture::{
    capture_into_graph_with_gemm_plan, require_deterministic_gemm_graph_plan,
    with_validated_gemm_graph_launch,
};
use crate::mamba_ssm::gpu::kernel_identity::{CapturedGemmGraphPlan, PreparedGemmCaptureManifest};
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::weights::Mamba3Weights;
use std::{cell::Cell, sync::Arc};

type Stream = Arc<cudarc::driver::CudaStream>;

/// Persistent recurrent state for GPU Mamba-3 inference (all layers).
pub struct Mamba3GpuInferenceState {
    /// SSM hidden state: `[n_layers * batch * nh * hd * ds]`.
    pub ssm_state: GpuBuffer,
    /// K state (previous B post-RoPE): `[n_layers * batch * nh * ds]`.
    pub k_state: GpuBuffer,
    /// V state (previous x): `[n_layers * batch * nh * hd]`.
    pub v_state: GpuBuffer,
    /// RoPE angle state: `[n_layers * batch * nh * n_angles]`.
    pub angle_state: GpuBuffer,
    pub batch: usize,
    pub n_layers: usize,
    pub nheads: usize,
    pub headdim: usize,
    pub d_state: usize,
    pub n_angles: usize,
}

impl Mamba3GpuInferenceState {
    /// Allocate zeroed state for all layers.
    pub fn zeros(stream: &Stream, batch: usize, cfg: &Mamba3Config) -> Result<Self, String> {
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ds = cfg.d_state;
        let na = cfg.num_rope_angles().max(1);
        let nl = cfg.n_layers;
        Ok(Self {
            ssm_state: GpuBuffer::zeros(stream, nl * batch * nh * hd * ds)?,
            k_state: GpuBuffer::zeros(stream, nl * batch * nh * ds)?,
            v_state: GpuBuffer::zeros(stream, nl * batch * nh * hd)?,
            angle_state: GpuBuffer::zeros(stream, nl * batch * nh * na)?,
            batch,
            n_layers: nl,
            nheads: nh,
            headdim: hd,
            d_state: ds,
            n_angles: na,
        })
    }

    /// Reset all state to zero (new sequence boundary).
    pub fn reset(&mut self, stream: &Stream) -> Result<(), String> {
        self.ssm_state.zero(stream)?;
        self.k_state.zero(stream)?;
        self.v_state.zero(stream)?;
        self.angle_state.zero(stream)
    }

    /// Per-layer SSM state size.
    pub fn ssm_per_layer(&self) -> usize {
        self.batch * self.nheads * self.headdim * self.d_state
    }
    pub fn k_per_layer(&self) -> usize {
        self.batch * self.nheads * self.d_state
    }
    pub fn v_per_layer(&self) -> usize {
        self.batch * self.nheads * self.headdim
    }
    pub fn angle_per_layer(&self) -> usize {
        self.batch * self.nheads * self.n_angles
    }
}

/// Scratch buffers for T=1 inference (minimal, reused every step).
pub struct Mamba3GpuInferenceScratch {
    pub gpu_input: GpuBuffer,     // [batch * input_dim] — H2D landing
    pub temporal: GpuBuffer,      // [batch * d_model] — working buffer
    pub residual: GpuBuffer,      // [batch * d_model] — saved for skip connection
    pub proj: GpuBuffer,          // [batch * in_proj_dim]
    pub z: GpuBuffer,             // [batch * d_inner]
    pub x: GpuBuffer,             // [batch * d_inner]
    pub b_raw: GpuBuffer,         // [batch * ng * ds]
    pub c_raw: GpuBuffer,         // [batch * ng * ds]
    pub b_normed: GpuBuffer,      // [batch * ng * ds]
    pub c_normed: GpuBuffer,      // [batch * ng * ds]
    pub b_rms: GpuBuffer,         // [batch * ng]
    pub c_rms: GpuBuffer,         // [batch * ng]
    pub b_biased: GpuBuffer,      // [batch * nh * ds]
    pub c_biased: GpuBuffer,      // [batch * nh * ds]
    pub k_cur: GpuBuffer,         // [batch * nh * ds] — post-RoPE B
    pub q_cur: GpuBuffer,         // [batch * nh * ds] — post-RoPE C
    pub dd_dt_raw: GpuBuffer,     // [batch * nh]
    pub dd_a_raw: GpuBuffer,      // [batch * nh]
    pub trap_raw: GpuBuffer,      // [batch * nh]
    pub dt: GpuBuffer,            // [batch * nh]
    pub a_val: GpuBuffer,         // [batch * nh]
    pub trap: GpuBuffer,          // [batch * nh]
    pub angles_raw: GpuBuffer,    // [batch * n_angles]
    pub y: GpuBuffer,             // [batch * d_inner]
    pub gated: GpuBuffer,         // [batch * d_inner]
    pub post_norm: GpuBuffer,     // [batch * d_model] — rmsnorm output (avoids in-place aliasing)
    pub rms_buf: GpuBuffer,       // [batch]
    pub gated_rms_buf: GpuBuffer, // [batch * nheads] — rstd for rmsnorm_gated
}

impl Mamba3GpuInferenceScratch {
    pub fn zeros(
        stream: &Stream,
        batch: usize,
        cfg: &Mamba3Config,
        input_dim: usize,
    ) -> Result<Self, String> {
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ng = cfg.ngroups;
        let ip = cfg.in_proj_out_dim();
        let na = cfg.num_rope_angles().max(1);
        Ok(Self {
            gpu_input: GpuBuffer::zeros(stream, batch * input_dim)?,
            temporal: GpuBuffer::zeros(stream, batch * dm)?,
            residual: GpuBuffer::zeros(stream, batch * dm)?,
            proj: GpuBuffer::zeros(stream, batch * ip)?,
            z: GpuBuffer::zeros(stream, batch * di)?,
            x: GpuBuffer::zeros(stream, batch * di)?,
            b_raw: GpuBuffer::zeros(stream, batch * ng * ds)?,
            c_raw: GpuBuffer::zeros(stream, batch * ng * ds)?,
            b_normed: GpuBuffer::zeros(stream, batch * ng * ds)?,
            c_normed: GpuBuffer::zeros(stream, batch * ng * ds)?,
            b_rms: GpuBuffer::zeros(stream, batch * ng)?,
            c_rms: GpuBuffer::zeros(stream, batch * ng)?,
            b_biased: GpuBuffer::zeros(stream, batch * nh * ds)?,
            c_biased: GpuBuffer::zeros(stream, batch * nh * ds)?,
            k_cur: GpuBuffer::zeros(stream, batch * nh * ds)?,
            q_cur: GpuBuffer::zeros(stream, batch * nh * ds)?,
            dd_dt_raw: GpuBuffer::zeros(stream, batch * nh)?,
            dd_a_raw: GpuBuffer::zeros(stream, batch * nh)?,
            trap_raw: GpuBuffer::zeros(stream, batch * nh)?,
            dt: GpuBuffer::zeros(stream, batch * nh)?,
            a_val: GpuBuffer::zeros(stream, batch * nh)?,
            trap: GpuBuffer::zeros(stream, batch * nh)?,
            angles_raw: GpuBuffer::zeros(stream, batch * na)?,
            y: GpuBuffer::zeros(stream, batch * di)?,
            gated: GpuBuffer::zeros(stream, batch * di)?,
            post_norm: GpuBuffer::zeros(stream, batch * dm)?,
            rms_buf: GpuBuffer::zeros(stream, batch)?,
            gated_rms_buf: GpuBuffer::zeros(stream, batch * nh)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Mixed-precision scratch for end-to-end bf16/f16 Mamba-3 inference.
// Activation-linear tensors are DtypedBuf (bf16/f16). Scalars per-head
// (dt, a_val, trap, alpha, beta, gamma, angles, rms stats) and the residual
// stream stay f32 — see mamba3_siso/cpu/inference.rs for the parity contract.
// ---------------------------------------------------------------------------

use crate::mamba_ssm::gpu::buffers::DtypedBuf;
use crate::mamba_ssm::gpu::dtype::WeightDtype;

pub struct Mamba3GpuInferenceMixedScratch {
    pub gpu_input: GpuBuffer, // f32 — CPU upload staging (seeds f32 residual)
    /// f32 `[batch * d_model]` landing for the typed prefill's final
    /// hidden (downcast into `temporal` after the window) — a dedicated
    /// buffer so non-identity input projections (input_dim != d_model)
    /// prefill too.
    pub prefill_hidden: GpuBuffer,
    pub temporal: DtypedBuf, // bf16/f16 — post-norm branch + final lm_head input
    pub residual: GpuBuffer, // f32 — cross-layer accumulator (HF residual_in_fp32)
    pub proj: DtypedBuf,
    pub z: DtypedBuf,
    pub x: DtypedBuf,
    pub b_raw: DtypedBuf,
    pub c_raw: DtypedBuf,
    pub b_normed: DtypedBuf,
    pub c_normed: DtypedBuf,
    pub b_rms: GpuBuffer, // f32 stats
    pub c_rms: GpuBuffer, // f32 stats
    pub b_biased: DtypedBuf,
    pub c_biased: DtypedBuf,
    pub k_cur: DtypedBuf, // bf16 — post-RoPE B fed to ssm_step typed
    pub q_cur: DtypedBuf, // bf16 — post-RoPE C fed to ssm_step typed
    // Backward-save tensors — allocated but unused in inference.
    pub dd_dt_raw: GpuBuffer,
    pub dd_a_raw: GpuBuffer,
    pub trap_raw: GpuBuffer,
    // Recurrence coefficients — stay f32 for numerical stability.
    pub dt: GpuBuffer,
    pub a_val: GpuBuffer,
    pub trap: GpuBuffer,
    pub angles_raw: GpuBuffer, // f32 (tanh/PI·dt products accumulate in f64)
    pub y: DtypedBuf,
    pub gated: DtypedBuf,
    pub post_norm: DtypedBuf,
    pub rms_buf: GpuBuffer,
    pub gated_rms_buf: GpuBuffer,
    pub dtype: WeightDtype,
}

impl Mamba3GpuInferenceMixedScratch {
    pub fn zeros(
        stream: &Stream,
        batch: usize,
        cfg: &Mamba3Config,
        input_dim: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        if matches!(dtype, WeightDtype::F32) {
            return Err("Mamba3GpuInferenceMixedScratch requires bf16 or f16 dtype".to_string());
        }
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let ng = cfg.ngroups;
        let ip = cfg.in_proj_out_dim();
        let na = cfg.num_rope_angles().max(1);
        Ok(Self {
            gpu_input: GpuBuffer::zeros(stream, batch * input_dim)?,
            prefill_hidden: GpuBuffer::zeros(stream, batch * dm)?,
            temporal: DtypedBuf::zeros(stream, batch * dm, dtype)?,
            residual: GpuBuffer::zeros(stream, batch * dm)?,
            proj: DtypedBuf::zeros(stream, batch * ip, dtype)?,
            z: DtypedBuf::zeros(stream, batch * di, dtype)?,
            x: DtypedBuf::zeros(stream, batch * di, dtype)?,
            b_raw: DtypedBuf::zeros(stream, batch * ng * ds, dtype)?,
            c_raw: DtypedBuf::zeros(stream, batch * ng * ds, dtype)?,
            b_normed: DtypedBuf::zeros(stream, batch * ng * ds, dtype)?,
            c_normed: DtypedBuf::zeros(stream, batch * ng * ds, dtype)?,
            b_rms: GpuBuffer::zeros(stream, batch * ng)?,
            c_rms: GpuBuffer::zeros(stream, batch * ng)?,
            b_biased: DtypedBuf::zeros(stream, batch * nh * ds, dtype)?,
            c_biased: DtypedBuf::zeros(stream, batch * nh * ds, dtype)?,
            k_cur: DtypedBuf::zeros(stream, batch * nh * ds, dtype)?,
            q_cur: DtypedBuf::zeros(stream, batch * nh * ds, dtype)?,
            dd_dt_raw: GpuBuffer::zeros(stream, batch * nh)?,
            dd_a_raw: GpuBuffer::zeros(stream, batch * nh)?,
            trap_raw: GpuBuffer::zeros(stream, batch * nh)?,
            dt: GpuBuffer::zeros(stream, batch * nh)?,
            a_val: GpuBuffer::zeros(stream, batch * nh)?,
            trap: GpuBuffer::zeros(stream, batch * nh)?,
            angles_raw: GpuBuffer::zeros(stream, batch * na)?,
            y: DtypedBuf::zeros(stream, batch * di, dtype)?,
            gated: DtypedBuf::zeros(stream, batch * di, dtype)?,
            post_norm: DtypedBuf::zeros(stream, batch * dm, dtype)?,
            rms_buf: GpuBuffer::zeros(stream, batch)?,
            gated_rms_buf: GpuBuffer::zeros(stream, batch * nh)?,
            dtype,
        })
    }
}

/// Mamba-3 SISO GPU inference engine.
///
/// Holds compiled kernels, weights (flat buffer), cuBLAS handle.
/// Supports CUDA Graph capture with a validated GEMM-only inventory.
///
/// Usage:
/// 1. `Mamba3GpuInferenceEngine::new()` — compile kernels, upload weights
/// 2. Allocate state + scratch via `alloc_state()` / `alloc_scratch()`
/// 3. Call `step()` each timestep
/// 4. Optionally call `capture_graph()` for faster inference
/// 5. Call `state.reset()` on episode boundaries
pub struct Mamba3GpuInferenceEngine {
    pub kernels: Mamba3Kernels,
    pub weights: GpuMamba3WeightsInf,
    /// Full CUDA execution context (stream + cuBLAS with its Graph-safe
    /// workspace and GEMM route). One context for step, prefill and the
    /// lm-head GEMMs alike. The GEMM mode selected on this context governs
    /// every vendor call made through it;
    /// deterministic mode and its family/policy settings route the engine's
    /// projections and downstream lm-head through the same context-aware
    /// GEMM boundaries as Mamba-1.
    pub ctx: GpuCtx,
    pub cfg: Mamba3Config,
    pub batch: usize,
    pub input_dim: usize,
    /// HF M3 models have no input_proj — skip GEMM, copy input → temporal.
    pub identity_proj: bool,
    graph: Option<cudarc::driver::CudaGraph>,
    captured_gemm_route: Option<crate::mamba_ssm::gpu::context::GemmRoute>,
    captured_gemm_plan: Option<CapturedGemmGraphPlan>,
    eager_gemm_manifest: Cell<Option<PreparedGemmCaptureManifest>>,
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
}

impl Drop for Mamba3GpuInferenceEngine {
    fn drop(&mut self) {
        let _ = self.ctx.stream.synchronize();
        drop(self.graph.take());
    }
}

impl Mamba3GpuInferenceEngine {
    /// Create an f32 M3 inference engine using the GEMM environment.
    ///
    /// Missing selectors use Deterministic + Inference. M3 now parses the same
    /// strict GEMM environment as M1; use [`Self::new_with_mode`] to bypass it.
    /// Inspect the route through [`Self::ctx`] before graph capture.
    ///
    /// # Errors
    ///
    /// Returns configuration, GEMM-environment, M3 state-cap, CUDA, upload, or
    /// allocation failures. `MAMBA_RS_ARCH_RUNG` is a separate first-use policy.
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        Self::new_inner(device, cpu_weights, cfg, input_dim, batch, None)
    }

    /// Create an M3 inference engine with an explicit GEMM execution mode.
    ///
    /// The f32 storage choice is independent of `mode`. GEMM mode, custom
    /// precision/tensor-core controls, and family selectors in the environment
    /// are ignored; the context stores [`BiGemmFamily::Inference`].
    /// `MAMBA_RS_ARCH_RUNG` remains a separate first-use process policy and is
    /// not captured by this constructor. Configuration, M3 state-cap
    /// compilation, upload, allocation, and vendor setup failures are returned.
    pub fn new_with_mode(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_inner(device, cpu_weights, cfg, input_dim, batch, Some(mode))
    }

    fn new_inner(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        // GpuCtx disables cudarc's per-slice event tracking itself (the
        // CUDA-Graph capture prerequisite) and owns the cuBLAS handle plus
        // its Graph-safe workspace.
        let ctx = match mode {
            Some(mode) => GpuCtx::new_with_state_cap_mode_and_family(
                device,
                64,
                mode,
                BiGemmFamily::Inference,
            )?,
            None => {
                GpuCtx::new_from_env_with_state_cap_and_family(device, 64, BiGemmFamily::Inference)?
            }
        };
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3Kernels::compile_with_state_cap(
            device.context(),
            arch,
            crate::mamba_ssm::gpu::kernels::state_capacity(cfg.d_state)?,
        )?;
        let weights = GpuMamba3WeightsInf::from_cpu(&ctx.stream, cpu_weights, input_dim)?;
        let identity_proj = cpu_weights.input_proj_w.is_empty();

        Ok(Self {
            kernels,
            weights,
            ctx,
            cfg,
            batch,
            input_dim,
            identity_proj,
            graph: None,
            captured_gemm_route: None,
            captured_gemm_plan: None,
            eager_gemm_manifest: Cell::new(None),
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
        })
    }

    /// Capture CUDA Graph for the inference step.
    ///
    /// After capture, both step entries replay the fixed-buffer graph.
    ///
    /// Requires and consumes a successful eager step on these buffers. GEMM
    /// mode/family changes require new eager preparation and capture. Missing
    /// or mismatching GEMM inventories are errors. H2D/D2H transfers remain
    /// outside the body and graph; the manifest covers GEMMs only.
    ///
    /// # Safety
    ///
    /// `state`, `scratch`, and their views must remain unchanged until the
    /// graph is cleared and all replays complete. The engine context, stream,
    /// cuBLAS workspace, both module sets, functions, and weights must stay fixed.
    pub unsafe fn capture_graph(
        &mut self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        let manifest = self.eager_gemm_manifest.take().ok_or_else(|| {
            "M3 f32 inference graph capture requires a successful eager step".to_string()
        })?;
        self.ctx.presize_bi_scratch()?;
        let snap_state = state.ssm_state.cached_ptr();
        let snap_scratch = scratch.gpu_input.cached_ptr();
        let snap_gemm_route = self.ctx.gemm_route();
        let (graph, captured_gemm_plan) = unsafe {
            capture_into_graph_with_gemm_plan(&self.ctx, manifest.route_capacity, &manifest, || {
                self.step_kernels(state, scratch)
            })
        }?;
        require_deterministic_gemm_graph_plan(
            &self.ctx,
            self.has_gemm_work(),
            captured_gemm_plan.as_ref(),
            "M3 f32 inference graph capture",
        )?;
        self.graph = Some(graph);
        self.captured_gemm_route = Some(snap_gemm_route);
        self.captured_gemm_plan = captured_gemm_plan;
        self.captured_state_ptr = snap_state;
        self.captured_scratch_ptr = snap_scratch;
        self.ctx.note_graph_capture();
        Ok(())
    }

    fn has_gemm_work(&self) -> bool {
        self.batch != 0 && (!self.identity_proj || self.cfg.n_layers != 0)
    }

    fn launch_captured_graph(&self) -> Result<(), String> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| "M3 graph is not captured".to_string())?;
        with_validated_gemm_graph_launch(
            &self.ctx,
            self.has_gemm_work(),
            self.captured_gemm_plan.as_ref(),
            "M3 f32 inference graph replay",
            || {
                graph
                    .launch()
                    .map_err(|error| format!("M3 graph launch: {error:?}"))
            },
        )
    }

    /// Whether a CUDA Graph has been captured.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Allocate zeroed inference state.
    pub fn alloc_state(&self) -> Result<Mamba3GpuInferenceState, String> {
        Mamba3GpuInferenceState::zeros(&self.ctx.stream, self.batch, &self.cfg)
    }

    /// Allocate scratch buffers.
    pub fn alloc_scratch(&self) -> Result<Mamba3GpuInferenceScratch, String> {
        Mamba3GpuInferenceScratch::zeros(&self.ctx.stream, self.batch, &self.cfg, self.input_dim)
    }

    /// Access the context that owns this engine's GEMM mode and family.
    ///
    /// Storage remains f32; inspect execution with [`GpuCtx::gemm_mode`] and
    /// [`GpuCtx::bi_gemm_family`] before graph capture.
    pub fn ctx(&self) -> &GpuCtx {
        &self.ctx
    }

    /// Launch dimensions for a prompt window of `seq_len` on this engine.
    fn prefill_dims(&self, seq_len: usize) -> super::state::GpuMamba3Dims {
        super::state::GpuMamba3Dims {
            batch: self.batch,
            d_model: self.cfg.d_model,
            d_inner: self.cfg.d_inner(),
            d_state: self.cfg.d_state,
            nheads: self.cfg.nheads(),
            headdim: self.cfg.headdim,
            ngroups: self.cfg.ngroups,
            in_proj_dim: self.cfg.in_proj_out_dim(),
            seq_len,
            mamba_input_dim: self.input_dim,
            n_layers: self.cfg.n_layers,
            n_angles: self.cfg.num_rope_angles(),
            a_floor: self.cfg.a_floor,
            is_outproj_norm: self.cfg.is_outproj_norm,
            rms_norm_eps: self.cfg.rms_norm_eps,
            use_parallel_scan: true,
        }
    }

    /// Allocate a prefill executor for a FIXED prompt length (scratch is
    /// shaped by `seq_len`; reuse it for windows of the same length).
    pub fn alloc_prefill(&self, seq_len: usize) -> Result<super::prefill::Mamba3Prefill, String> {
        super::prefill::Mamba3Prefill::new(&self.ctx.stream, &self.prefill_dims(seq_len))
    }

    /// One-pass prompt window straight into the persistent decode state:
    /// after this returns, `state` sits after the window's last token and
    /// `step()` continues from it. `last_hidden` (`[batch * d_model]`)
    /// receives the final post-norm hidden state. `carry_state = false`
    /// zeroes the state first (stateless window); `true` continues across
    /// the seam with the trapezoidal boundary fold.
    pub fn prefill_sequence(
        &self,
        prefill: &mut super::prefill::Mamba3Prefill,
        mamba_input: &GpuBuffer,
        seq_len: usize,
        state: &mut Mamba3GpuInferenceState,
        carry_state: bool,
        last_hidden: &mut GpuBuffer,
    ) -> Result<(), String> {
        let dims = self.prefill_dims(seq_len);
        prefill.run(
            &super::prefill::Mamba3PrefillRun {
                ctx: &self.ctx,
                kernels: &self.kernels,
                dims: &dims,
                weights: &self.weights,
                mamba_input,
                identity_proj: self.identity_proj,
                carry_state,
            },
            super::state::GpuMamba3StateBufs {
                ssm: &mut state.ssm_state,
                k: &mut state.k_state,
                v: &mut state.v_state,
                angle: &mut state.angle_state,
            },
            last_hidden,
        )
    }

    /// Config reference.
    pub fn config(&self) -> &Mamba3Config {
        &self.cfg
    }

    /// Batch size.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// GPU-only kernel pipeline: input_proj + all layers + norm_f.
    /// No H2D/D2H — safe for CUDA Graph capture.
    fn step_kernels(
        &self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        use cudarc::driver::PushKernelArg;

        let b = self.batch;
        let dm = self.cfg.d_model;
        let b_i = b as i32;
        let dm_i = dm as i32;

        if self.identity_proj {
            // HF M3 models have no input_proj — embedding is already d_model.
            // Copy gpu_input → temporal directly (mirrors CPU mamba3_step no-proj).
            debug_assert_eq!(self.input_dim, dm);
            scratch
                .temporal
                .copy_from_raw(&scratch.gpu_input, &self.ctx.stream)?;
        } else {
            // Input projection SGEMM
            unsafe {
                gpu_gemm_f32_forward_ptrs(
                    &self.ctx,
                    scratch.temporal.cached_ptr(),
                    scratch.gpu_input.cached_ptr(),
                    self.weights.input_proj_w.ptr(),
                    None,
                    (b, self.input_dim, dm),
                )
            }?;
            // Add bias
            {
                let n = b * dm;
                let n_i = n as i32;
                let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
                let mut builder = self
                    .ctx
                    .stream
                    .launch_builder(&self.kernels.vec_add_inplace);
                builder.arg(scratch.temporal.inner());
                builder.arg(self.weights.input_proj_b.inner());
                builder.arg(&n_i);
                unsafe { builder.launch(grid) }.map_err(|e| format!("input_proj bias: {e:?}"))?;
            }
        }

        // Process each layer
        for layer_idx in 0..self.cfg.n_layers {
            self.step_layer_kernels(layer_idx, state, scratch)?;
        }

        // Final RMSNorm
        {
            let bytes = b * dm * std::mem::size_of::<f32>();
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    scratch.post_norm.cached_ptr(),
                    scratch.temporal.cached_ptr(),
                    bytes,
                    self.ctx.stream.cu_stream(),
                );
            }
        }
        {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
            let eps: f32 = self.cfg.rms_norm_eps;
            let mut builder = self.ctx.stream.launch_builder(&self.kernels.rmsnorm_fwd);
            builder.arg(scratch.temporal.inner());
            builder.arg(scratch.rms_buf.inner());
            builder.arg(scratch.post_norm.inner());
            builder.arg(self.weights.norm_f_weight.inner());
            builder.arg(&b_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid) }.map_err(|e| format!("norm_f: {e:?}"))?;
        }

        Ok(())
    }

    /// Launch the 10-phase T=1 forward for one layer.
    ///
    /// Phases: F1(RMSNorm) → F2(in_proj GEMM) → F3(m3_split) → F4(BCNorm+bias+RoPE)
    ///       → F5(alpha/beta/gamma) → F6(m3_step_fwd) → F7(gating) → F8(out_proj GEMM)
    ///       → F9(residual add)
    ///
    /// Called inside CUDA Graph capture or directly.
    pub fn step_layer_kernels(
        &self,
        layer_idx: usize,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        use cudarc::driver::PushKernelArg;

        let b = self.batch;
        let dm = self.cfg.d_model;
        let di = self.cfg.d_inner();
        let ds = self.cfg.d_state;
        let nh = self.cfg.nheads();
        let hd = self.cfg.headdim;
        let ng = self.cfg.ngroups;
        let ip = self.cfg.in_proj_out_dim();
        let na = self.cfg.num_rope_angles();
        let a_floor = self.cfg.a_floor;
        let lw = &self.weights.layers[layer_idx];

        // Pre-compute i32 locals to avoid temporaries in builder.arg()
        let b_i = b as i32;
        let dm_i = dm as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ng_i = ng as i32;
        let na_i = na as i32;

        let ssm_off = layer_idx * state.ssm_per_layer();
        let k_off = layer_idx * state.k_per_layer();
        let v_off = layer_idx * state.v_per_layer();
        let a_off = layer_idx * state.angle_per_layer();

        // F1: RMSNorm reads the residual stream in place - the old
        // pre-norm D2D save is gone: F8 now writes post_norm (dead by
        // then) and F9 adds it into temporal, which still holds the
        // pre-norm value. IEEE addition of the same two f32 values is
        // commutative, so the swapped operand roles round identically.
        {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
            let eps: f32 = self.cfg.rms_norm_eps;
            let nw_ptr = lw.norm_weight.ptr();
            let mut builder = self.ctx.stream.launch_builder(&self.kernels.rmsnorm_fwd);
            builder.arg(scratch.post_norm.inner());
            builder.arg(scratch.rms_buf.inner());
            builder.arg(scratch.temporal.inner());
            builder.arg(&nw_ptr);
            builder.arg(&b_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F1 rmsnorm: {e:?}"))?;
        }

        // F2: in_proj SGEMM [batch, d_model] → [batch, in_proj_dim]
        unsafe {
            gpu_gemm_f32_forward_ptrs(
                &self.ctx,
                scratch.proj.cached_ptr(),
                scratch.post_norm.cached_ptr(),
                lw.in_proj_w.ptr(),
                None,
                (b, dm, ip),
            )
        }?;

        // F3: m3_split (8-way + fused softplus/sigmoid)
        {
            let n = b * ip;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.ctx.stream.launch_builder(&self.kernels.m3_split);
            builder.arg(scratch.z.inner());
            builder.arg(scratch.x.inner());
            builder.arg(scratch.b_raw.inner());
            builder.arg(scratch.c_raw.inner());
            builder.arg(scratch.dt.inner());
            builder.arg(scratch.a_val.inner());
            builder.arg(scratch.trap.inner());
            builder.arg(scratch.angles_raw.inner());
            builder.arg(scratch.dd_dt_raw.inner());
            builder.arg(scratch.dd_a_raw.inner());
            builder.arg(scratch.trap_raw.inner());
            builder.arg(scratch.proj.inner());
            builder.arg(lw.dt_bias.inner());
            builder.arg(&a_floor);
            // CUDA signature: int N, int di, int ng, int ds, int nh, int n_angles
            builder.arg(&b_i);
            builder.arg(&di_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F3 m3_split: {e:?}"))?;
        }

        // F4a: BCNorm forward, B and C in one launch (grid.y selects the
        // tensor; per-element math identical to two sequential calls).
        {
            let grid = bcnorm_fwd_bc_cfg(b * ng, ds);
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.kernels.bcnorm_fwd_bc_f32);
            builder.arg(scratch.b_normed.inner());
            builder.arg(scratch.c_normed.inner());
            builder.arg(scratch.b_rms.inner());
            builder.arg(scratch.c_rms.inner());
            builder.arg(scratch.b_raw.inner());
            builder.arg(scratch.c_raw.inner());
            builder.arg(lw.b_norm_weight.inner());
            builder.arg(lw.c_norm_weight.inner());
            builder.arg(&b_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            let eps_g5: f32 = self.cfg.rms_norm_eps;
            builder.arg(&eps_g5);
            let src_stride = (ng * ds) as i32;
            builder.arg(&src_stride);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4a bcnorm BC: {e:?}"))?;
        }

        // F4b-c: bias add (B + C) + RoPE in one launch, the angles advanced
        // inside it (the lane that owns an angle steps the persistent state
        // and rotates with the same value); the biased tensors materialize
        // for the state writeback consumers; n_angles == 0 passes through.
        {
            let n = b * nh * ds;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let a_ptr: cudarc::driver::sys::CUdeviceptr = if na > 0 {
                state.angle_state.inner_at(a_off)
            } else {
                0
            };
            let no_cumsum: cudarc::driver::sys::CUdeviceptr = 0;
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.kernels.m3_bias_rope_fwd);
            builder.arg(scratch.b_biased.inner());
            builder.arg(scratch.c_biased.inner());
            builder.arg(scratch.k_cur.inner());
            builder.arg(scratch.q_cur.inner());
            builder.arg(scratch.b_normed.inner());
            builder.arg(scratch.c_normed.inner());
            builder.arg(lw.b_bias.inner());
            builder.arg(lw.c_bias.inner());
            builder.arg(&no_cumsum);
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            builder.arg(&na_i);
            builder.arg(&a_ptr);
            builder.arg(scratch.angles_raw.inner());
            builder.arg(scratch.dt.inner());
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4bc bias_rope: {e:?}"))?;
        }

        // F6: m3_step_fwd (trapezoidal SSM recurrence)
        {
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: (b as u32, nh as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let ssm_ptr = state.ssm_state.inner_at(ssm_off);
            let k_ptr = state.k_state.inner_at(k_off);
            let v_ptr = state.v_state.inner_at(v_off);
            let mut builder = self.ctx.stream.launch_builder(&self.kernels.m3_step_fwd);
            // CUDA signature: ssm_state, k_state, v_state, y, ...
            builder.arg(&ssm_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(scratch.y.inner());
            builder.arg(scratch.x.inner());
            builder.arg(scratch.k_cur.inner());
            builder.arg(scratch.q_cur.inner());
            // The step computes its own coefficients from dt, a_val and trap.
            builder.arg(scratch.dt.inner());
            builder.arg(scratch.a_val.inner());
            builder.arg(scratch.trap.inner());
            builder.arg(lw.d_param.inner());
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F6 m3_step_fwd: {e:?}"))?;
        }

        // F7: Output gating
        if self.cfg.is_outproj_norm {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, di);
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.kernels.rmsnorm_gated_fwd);
            builder.arg(scratch.gated.inner());
            builder.arg(scratch.gated_rms_buf.inner()); // rms_vals (rstd per group)
            builder.arg(scratch.y.inner());
            builder.arg(scratch.z.inner());
            builder.arg(lw.norm_gate_weight.inner());
            builder.arg(&b_i);
            builder.arg(&di_i);
            builder.arg(&hd_i);
            let eps_g5: f32 = self.cfg.rms_norm_eps;
            builder.arg(&eps_g5);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F7 rmsnorm_gated: {e:?}"))?;
        } else {
            let n = b * di;
            let n_i = n as i32;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.ctx.stream.launch_builder(&self.kernels.silu_gate_fwd);
            builder.arg(scratch.gated.inner());
            builder.arg(scratch.y.inner());
            builder.arg(scratch.z.inner());
            builder.arg(&n_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F7 silu_gate: {e:?}"))?;
        }

        // F8: out_proj SGEMM [batch, d_inner] → [batch, d_model] - lands
        // in post_norm (dead since F2 consumed it), keeping the residual
        // stream in temporal.
        unsafe {
            gpu_gemm_f32_forward_ptrs(
                &self.ctx,
                scratch.post_norm.cached_ptr(),
                scratch.gated.cached_ptr(),
                lw.out_proj_w.ptr(),
                None,
                (b, di, dm),
            )
        }?;

        // F9: Residual add - temporal (pre-norm stream) += block output.
        {
            let n = b * dm;
            let n_i = n as i32;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self
                .ctx
                .stream
                .launch_builder(&self.kernels.vec_add_inplace);
            builder.arg(scratch.temporal.inner());
            builder.arg(scratch.post_norm.inner());
            builder.arg(&n_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F9 residual: {e:?}"))?;
        }

        Ok(())
    }

    /// Run one inference step: input → output.
    ///
    /// `input`: `[batch * input_dim]` on CPU.
    /// `output`: `[batch * d_model]` on CPU.
    ///
    /// When a CUDA Graph is captured, replays the graph instead of launching
    /// kernels individually. H2D/D2H transfers remain outside the graph.
    pub fn step(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        // H2D: upload input (outside graph)
        self.eager_gemm_manifest.set(None);
        scratch.gpu_input.upload(&self.ctx.stream, input)?;

        // GPU kernel pipeline (graph replay or individual launches)
        if self.graph.is_some() {
            if self.captured_gemm_route != Some(self.ctx.gemm_route()) {
                return Err("M3 inference graph replay: GEMM route changed since capture".into());
            }
            assert_eq!(
                state.ssm_state.cached_ptr(),
                self.captured_state_ptr,
                "CUDA Graph replay requires the same state buffers used during capture"
            );
            assert_eq!(
                scratch.gpu_input.cached_ptr(),
                self.captured_scratch_ptr,
                "CUDA Graph replay requires the same scratch buffers used during capture"
            );
            self.launch_captured_graph()?;
        } else {
            if self.has_gemm_work() {
                prepare_inference_arch_rung(&self.ctx)?;
            }
            let manifest = self
                .ctx
                .record_eager_gemm_manifest(|| self.step_kernels(state, scratch))?;
            self.eager_gemm_manifest.set(Some(manifest));
        }

        // Sync + D2H download
        self.ctx
            .stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let cpu_out = scratch.temporal.to_cpu(&self.ctx.stream)?;
        output[..cpu_out.len()].copy_from_slice(&cpu_out);

        Ok(())
    }

    /// Run a step and keep `scratch.temporal` on GPU (no D2H download).
    /// Mirrors `GpuMambaInference::step_gpu_only` on the M1 side — the LM
    /// wrapper chains `step_gpu_only → lm_head_gemm` without going through
    /// CPU for the hidden state.
    pub fn step_gpu_only(
        &self,
        input: &[f32],
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        self.eager_gemm_manifest.set(None);
        scratch.gpu_input.upload(&self.ctx.stream, input)?;
        if self.graph.is_some() {
            if self.captured_gemm_route != Some(self.ctx.gemm_route()) {
                return Err("M3 inference graph replay: GEMM route changed since capture".into());
            }
            assert_eq!(state.ssm_state.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            self.launch_captured_graph()?;
        } else {
            if self.has_gemm_work() {
                prepare_inference_arch_rung(&self.ctx)?;
            }
            let manifest = self
                .ctx
                .record_eager_gemm_manifest(|| self.step_kernels(state, scratch))?;
            self.eager_gemm_manifest.set(Some(manifest));
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Mamba3GpuInferenceMixed — end-to-end bf16/f16 inference engine.
//
// Wraps the f32 engine (for ctx/kernels/cublas/cfg) plus mixed-dtype
// weights. All activations run in bf16/f16 through the layer; recurrence
// coefficients (dt/a_val/trap/alpha/beta/gamma), RoPE angles and the
// residual stream stay f32 for numerical stability.
// ═══════════════════════════════════════════════════════════════════

use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use crate::mamba3_siso::gpu::weights::GpuMamba3MixedWeights;

pub struct Mamba3GpuInferenceMixed {
    engine: Mamba3GpuInferenceEngine, // owns ctx + kernels + blas + (unused f32 weights)
    mixed_weights: GpuMamba3MixedWeights,
    graph: Option<cudarc::driver::CudaGraph>,
    captured_gemm_route: Option<crate::mamba_ssm::gpu::context::GemmRoute>,
    captured_gemm_plan: Option<CapturedGemmGraphPlan>,
    eager_gemm_manifest: Cell<Option<PreparedGemmCaptureManifest>>,
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
    captured_half_staging_ptr: u64,
    captured_bi_upcast_ptrs: [u64; 3],
}

impl Drop for Mamba3GpuInferenceMixed {
    fn drop(&mut self) {
        let _ = self.engine.ctx.stream.synchronize();
        drop(self.graph.take());
    }
}

impl Mamba3GpuInferenceMixed {
    fn has_gemm_work(&self) -> bool {
        self.engine.batch != 0 && self.engine.cfg.n_layers != 0
    }

    fn launch_captured_graph(&self) -> Result<(), String> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| "M3 mixed graph is not captured".to_string())?;
        with_validated_gemm_graph_launch(
            &self.engine.ctx,
            self.has_gemm_work(),
            self.captured_gemm_plan.as_ref(),
            "M3 mixed inference graph replay",
            || {
                graph
                    .launch()
                    .map_err(|error| format!("M3 graph launch mixed: {error:?}"))
            },
        )
    }

    fn ensure_graph_scratch(&self) -> Result<(), String> {
        self.engine.ctx.ensure_graph_scratch_ptrs(
            self.captured_half_staging_ptr,
            self.captured_bi_upcast_ptrs,
            "M3 mixed inference graph replay",
        )
    }

    /// CUDA execution context shared by the retained f32 engine and mixed path.
    ///
    /// Storage precision is available separately through [`Self::bulk_dtype`];
    /// inspect GEMM execution with [`GpuCtx::gemm_mode`] and the dormant or
    /// active deterministic family with [`GpuCtx::bi_gemm_family`]. Graph
    /// capture retains this complete route.
    pub fn ctx(&self) -> &GpuCtx {
        &self.engine.ctx
    }

    /// Create a bf16/f16-storage M3 engine using the GEMM environment.
    ///
    /// `bulk_dtype` controls storage, not GEMM mode. Missing selectors use
    /// Deterministic + Inference; invalid or conflicting selectors are errors.
    /// Use [`Self::new_with_mode`] for an explicit mode and [`Self::ctx`] to
    /// inspect the route that graph capture binds. `MAMBA_RS_ARCH_RUNG` is a
    /// separate first-use Inference policy. Configuration, dtype, M3 state-cap,
    /// upload, and allocation failures are returned.
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_inner(device, cpu_weights, cfg, input_dim, batch, bulk_dtype, None)
    }

    /// Create an M3 mixed-storage engine with an explicit GEMM mode.
    ///
    /// `bulk_dtype` selects bf16/f16 storage and `mode` independently selects
    /// GEMM execution. GEMM mode, custom precision/tensor-core controls, and
    /// family selectors in the environment are ignored. The retained f32
    /// engine owns the one Inference-role context; `MAMBA_RS_ARCH_RUNG` remains
    /// a separate first-use process policy and is not captured by this
    /// constructor. Errors match [`Self::new`].
    pub fn new_with_mode(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_inner(
            device,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            bulk_dtype,
            Some(mode),
        )
    }

    fn new_inner(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        // Reuse the f32 engine constructor to compile kernels + upload f32
        // weights (needed because the mixed path still consumes f32 biases
        // and per-head coefficients via the f32 engine's weight-agnostic
        // pointer views).
        let engine = match mode {
            Some(mode) => Mamba3GpuInferenceEngine::new_with_mode(
                device,
                cpu_weights,
                cfg,
                input_dim,
                batch,
                mode,
            )?,
            None => Mamba3GpuInferenceEngine::new(device, cpu_weights, cfg, input_dim, batch)?,
        };
        let mixed_weights =
            GpuMamba3MixedWeights::from_cpu(&engine.ctx.stream, cpu_weights, bulk_dtype)?;
        Ok(Self {
            engine,
            mixed_weights,
            graph: None,
            captured_gemm_route: None,
            captured_gemm_plan: None,
            eager_gemm_manifest: Cell::new(None),
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
            captured_half_staging_ptr: 0,
            captured_bi_upcast_ptrs: [0; 3],
        })
    }

    pub fn alloc_state(&self) -> Result<Mamba3GpuInferenceState, String> {
        self.engine.alloc_state()
    }

    pub fn alloc_mixed_scratch(&self) -> Result<Mamba3GpuInferenceMixedScratch, String> {
        Mamba3GpuInferenceMixedScratch::zeros(
            &self.engine.ctx.stream,
            self.engine.batch,
            &self.engine.cfg,
            self.engine.input_dim,
            self.mixed_weights.bulk_dtype,
        )
    }

    pub fn ctx_stream(&self) -> &Stream {
        &self.engine.ctx.stream
    }

    pub fn bulk_dtype(&self) -> WeightDtype {
        self.mixed_weights.bulk_dtype
    }

    pub fn engine_ref(&self) -> &Mamba3GpuInferenceEngine {
        &self.engine
    }

    /// The typed weights container (for prefill runs / graph guards).
    pub fn mixed_weights(&self) -> &GpuMamba3MixedWeights {
        &self.mixed_weights
    }

    /// Allocate a TYPED one-pass prompt executor for a fixed prompt
    /// length: the prefill runs the trainer's bf16/f16 kernel chain
    /// against `mixed_weights`.
    pub fn alloc_prefill(&self, seq_len: usize) -> Result<super::prefill::Mamba3Prefill, String> {
        super::prefill::Mamba3Prefill::new_with_dtype(
            &self.engine.ctx.stream,
            &self.engine.prefill_dims(seq_len),
            self.mixed_weights.bulk_dtype,
        )
    }

    /// One-pass typed prompt window straight into the persistent decode
    /// state (which is f32 in both pipelines). `last_hidden`
    /// (`[batch * d_model]`, f32) receives the final post-norm hidden.
    pub fn prefill_sequence(
        &self,
        prefill: &mut super::prefill::Mamba3Prefill,
        mamba_input: &GpuBuffer,
        seq_len: usize,
        state: &mut Mamba3GpuInferenceState,
        carry_state: bool,
        last_hidden: &mut GpuBuffer,
    ) -> Result<(), String> {
        let dims = self.engine.prefill_dims(seq_len);
        prefill.run(
            &super::prefill::Mamba3PrefillRun {
                ctx: &self.engine.ctx,
                kernels: &self.engine.kernels,
                dims: &dims,
                weights: &self.mixed_weights,
                mamba_input,
                identity_proj: self.engine.identity_proj,
                carry_state,
            },
            super::state::GpuMamba3StateBufs {
                ssm: &mut state.ssm_state,
                k: &mut state.k_state,
                v: &mut state.v_state,
                angle: &mut state.angle_state,
            },
            last_hidden,
        )
    }

    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// End-to-end bf16/f16 T=1 Mamba-3 pipeline.
    ///
    /// Requires identity_proj (LLM use case) — non-identity input projection
    /// in mixed mode is not supported here because it would need a mixed-
    /// dtype GEMM writing directly into the f32 residual buffer.
    pub(super) fn step_kernels_mixed_native(
        &self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        use crate::mamba_ssm::gpu::launch::grid_1d;
        use cudarc::driver::PushKernelArg;

        let engine = &self.engine;
        assert!(
            engine.identity_proj,
            "Mamba3 step_kernels_mixed_native requires identity_proj=true (LLM path)"
        );
        assert_eq!(
            scratch.dtype, self.mixed_weights.bulk_dtype,
            "Mamba3 mixed scratch dtype must match mixed weights bulk_dtype"
        );
        let dt = scratch.dtype;
        let b = engine.batch;
        let cfg = &engine.cfg;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ds = cfg.d_state;
        let nh = cfg.nheads();
        let hd = cfg.headdim;
        let ng = cfg.ngroups;
        let ip = cfg.in_proj_out_dim();
        let na = cfg.num_rope_angles();
        let a_floor = cfg.a_floor;
        let k = &engine.kernels;
        let w = &self.mixed_weights;

        let b_i = b as i32;
        let dm_i = dm as i32;
        let di_i = di as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ng_i = ng as i32;
        let na_i = na as i32;

        // Seed f32 residual with f32 gpu_input (identity_proj). copy_from_raw
        // is CUDA Graph safe (cuMemcpyDtoDAsync on raw ptrs, no SyncOnDrop).
        scratch
            .residual
            .copy_from_raw(&scratch.gpu_input, &engine.ctx.stream)?;

        let f32_sz = std::mem::size_of::<f32>() as u64;

        for layer_idx in 0..w.layers.len() {
            let lw = &w.layers[layer_idx];
            let ssm_off = layer_idx * state.ssm_per_layer();
            let k_off = layer_idx * state.k_per_layer();
            let v_off = layer_idx * state.v_per_layer();
            let a_off = layer_idx * state.angle_per_layer();
            let _ = f32_sz; // reserved for future offset math

            // F1: rmsnorm f32in → half post_norm.
            {
                let eps: f32 = engine.cfg.rms_norm_eps;
                let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
                let pn_ptr = scratch.post_norm.cached_ptr();
                let rms_ptr = scratch.rms_buf.cached_ptr();
                let res_ptr = scratch.residual.cached_ptr();
                let nw = lw.norm_weight.ptr();
                bld.arg(&pn_ptr);
                bld.arg(&rms_ptr);
                bld.arg(&res_ptr);
                bld.arg(&nw);
                bld.arg(&b_i);
                bld.arg(&dm_i);
                bld.arg(&eps);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F1 rmsnorm: {e:?}"))?;
            }

            // F2: in_proj GEMM typed (bf16 × bf16 → bf16).
            gpu_gemm_typed_forward_raw(
                &engine.ctx,
                TypedPtr {
                    ptr: scratch.proj.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: scratch.post_norm.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: lw.in_proj_w.ptr(),
                    dtype: lw.in_proj_w.dtype(),
                },
                None,
                (b, dm, ip),
            )?;

            // F3: m3_split typed — splits bf16 proj, writes bf16 activations + f32 coefficients.
            {
                let n = b * ip;
                let grid = grid_1d(n);
                let mut bld = engine.ctx.stream.launch_builder(k.m3_split_typed.get(dt));
                let z_ptr = scratch.z.cached_ptr();
                let x_ptr = scratch.x.cached_ptr();
                let br_ptr = scratch.b_raw.cached_ptr();
                let cr_ptr = scratch.c_raw.cached_ptr();
                let dt_ptr = scratch.dt.cached_ptr();
                let av_ptr = scratch.a_val.cached_ptr();
                let tp_ptr = scratch.trap.cached_ptr();
                let ang_ptr = scratch.angles_raw.cached_ptr();
                let dd_dt_ptr = scratch.dd_dt_raw.cached_ptr();
                let dd_a_ptr = scratch.dd_a_raw.cached_ptr();
                let tr_ptr = scratch.trap_raw.cached_ptr();
                let proj_ptr = scratch.proj.cached_ptr();
                let dtb_ptr = lw.dt_bias.ptr();
                bld.arg(&z_ptr);
                bld.arg(&x_ptr);
                bld.arg(&br_ptr);
                bld.arg(&cr_ptr);
                bld.arg(&dt_ptr);
                bld.arg(&av_ptr);
                bld.arg(&tp_ptr);
                bld.arg(&ang_ptr);
                bld.arg(&dd_dt_ptr);
                bld.arg(&dd_a_ptr);
                bld.arg(&tr_ptr);
                bld.arg(&proj_ptr);
                bld.arg(&dtb_ptr);
                bld.arg(&a_floor);
                bld.arg(&b_i);
                bld.arg(&di_i);
                bld.arg(&ng_i);
                bld.arg(&ds_i);
                bld.arg(&nh_i);
                bld.arg(&na_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F3 split: {e:?}"))?;
            }

            // F4a: bcnorm typed — fused B+C in single launch (gridDim.y=2
            // selects the B or C path).
            {
                let grid = bcnorm_fwd_bc_cfg(b * ng, ds);
                let bn_ptr = scratch.b_normed.cached_ptr();
                let cn_ptr = scratch.c_normed.cached_ptr();
                let br_ptr = scratch.b_rms.cached_ptr();
                let cr_ptr = scratch.c_rms.cached_ptr();
                let bs_ptr = scratch.b_raw.cached_ptr();
                let cs_ptr = scratch.c_raw.cached_ptr();
                let bw_ptr = lw.b_norm_weight.ptr();
                let cw_ptr = lw.c_norm_weight.ptr();
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.bcnorm_fwd_bc_typed.get(dt));
                bld.arg(&bn_ptr);
                bld.arg(&cn_ptr);
                bld.arg(&br_ptr);
                bld.arg(&cr_ptr);
                bld.arg(&bs_ptr);
                bld.arg(&cs_ptr);
                bld.arg(&bw_ptr);
                bld.arg(&cw_ptr);
                bld.arg(&b_i);
                bld.arg(&ng_i);
                bld.arg(&ds_i);
                let eps_g5: f32 = engine.cfg.rms_norm_eps;
                bld.arg(&eps_g5);
                let src_stride = ng_i * ds_i;
                bld.arg(&src_stride);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F4a bcnorm B+C: {e:?}"))?;
            }

            // F4b-c: typed bias add (B + C) + RoPE in one launch, the angles
            // advanced inside it (f64 accumulation on the owning lane, as
            // the standalone advance did); the biased stores round through
            // FROM_F and the rotation consumes the round-tripped values;
            // n_angles == 0 passes through. Cached raw pointers only: an
            // .inner() guard would invalidate graph capture.
            {
                let n = b * nh * ds;
                let grid = grid_1d(n);
                let bb_ptr = scratch.b_biased.cached_ptr();
                let cb_ptr = scratch.c_biased.cached_ptr();
                let kc_ptr = scratch.k_cur.cached_ptr();
                let qc_ptr = scratch.q_cur.cached_ptr();
                let bn_ptr = scratch.b_normed.cached_ptr();
                let cn_ptr = scratch.c_normed.cached_ptr();
                let bbi_ptr = lw.b_bias.ptr();
                let cbi_ptr = lw.c_bias.ptr();
                let no_cumsum: cudarc::driver::sys::CUdeviceptr = 0;
                let a_ptr: cudarc::driver::sys::CUdeviceptr = if na > 0 {
                    state.angle_state.inner_at(a_off)
                } else {
                    0
                };
                let ar_ptr = scratch.angles_raw.cached_ptr();
                let dt_ptr = scratch.dt.cached_ptr();
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.m3_bias_rope_fwd_typed.get(dt));
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&kc_ptr);
                bld.arg(&qc_ptr);
                bld.arg(&bn_ptr);
                bld.arg(&cn_ptr);
                bld.arg(&bbi_ptr);
                bld.arg(&cbi_ptr);
                bld.arg(&no_cumsum);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&ds_i);
                bld.arg(&na_i);
                bld.arg(&a_ptr);
                bld.arg(&ar_ptr);
                bld.arg(&dt_ptr);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F4bc bias_rope: {e:?}"))?;
            }

            // F6: m3_step_fwd typed — f32 state, bf16 x/k_cur/q_cur/y, f32 α/β/γ/D.
            {
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (b as u32, nh as u32, 1),
                    block_dim: (hd as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let ssm_ptr = state.ssm_state.inner_at(ssm_off);
                let kst_ptr = state.k_state.inner_at(k_off);
                let vst_ptr = state.v_state.inner_at(v_off);
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.m3_step_fwd_typed.get(dt));
                let y_ptr = scratch.y.cached_ptr();
                let x_ptr = scratch.x.cached_ptr();
                let kc_ptr = scratch.k_cur.cached_ptr();
                let qc_ptr = scratch.q_cur.cached_ptr();
                bld.arg(&ssm_ptr);
                bld.arg(&kst_ptr);
                bld.arg(&vst_ptr);
                bld.arg(&y_ptr);
                bld.arg(&x_ptr);
                bld.arg(&kc_ptr);
                bld.arg(&qc_ptr);
                // The step computes its own coefficients from dt, a_val and trap.
                let dt_ptr = scratch.dt.cached_ptr();
                let av_ptr = scratch.a_val.cached_ptr();
                let tr_ptr = scratch.trap.cached_ptr();
                let dp_ptr = lw.d_param.ptr();
                bld.arg(&dt_ptr);
                bld.arg(&av_ptr);
                bld.arg(&tr_ptr);
                bld.arg(&dp_ptr);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F6 step_fwd: {e:?}"))?;
            }

            // F7: output gating (typed).
            if cfg.is_outproj_norm {
                let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, di);
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.rmsnorm_gated_fwd_typed.get(dt));
                let gated_ptr = scratch.gated.cached_ptr();
                let gr_ptr = scratch.gated_rms_buf.cached_ptr();
                let y_ptr = scratch.y.cached_ptr();
                let z_ptr = scratch.z.cached_ptr();
                let nw_ptr = lw.norm_gate_weight.ptr();
                bld.arg(&gated_ptr);
                bld.arg(&gr_ptr);
                bld.arg(&y_ptr);
                bld.arg(&z_ptr);
                bld.arg(&nw_ptr);
                bld.arg(&b_i);
                bld.arg(&di_i);
                bld.arg(&hd_i);
                let eps_g5: f32 = engine.cfg.rms_norm_eps;
                bld.arg(&eps_g5);
                bld.arg(&di_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F7 rmsnorm_gated: {e:?}"))?;
            } else {
                let n = b * di;
                let n_i = n as i32;
                let grid = grid_1d(n);
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.silu_gate_fwd_typed.get(dt));
                let gated_ptr = scratch.gated.cached_ptr();
                let y_ptr = scratch.y.cached_ptr();
                let z_ptr = scratch.z.cached_ptr();
                bld.arg(&gated_ptr);
                bld.arg(&y_ptr);
                bld.arg(&z_ptr);
                bld.arg(&n_i);
                bld.arg(&di_i);
                bld.arg(&di_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F7 silu_gate: {e:?}"))?;
            }

            // F8: out_proj GEMM typed.
            gpu_gemm_typed_forward_raw(
                &engine.ctx,
                TypedPtr {
                    ptr: scratch.temporal.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: scratch.gated.cached_ptr(),
                    dtype: dt,
                },
                TypedPtr {
                    ptr: lw.out_proj_w.ptr(),
                    dtype: lw.out_proj_w.dtype(),
                },
                None,
                (b, di, dm),
            )?;

            // F9: residual_add_f32_typed — residual (f32) += temporal (half), stays f32.
            {
                let n = (b * dm) as i32;
                let grid = grid_1d(b * dm);
                let mut bld = engine
                    .ctx
                    .stream
                    .launch_builder(k.residual_add_f32_typed.get(dt));
                let r_ptr = scratch.residual.cached_ptr();
                let t_ptr = scratch.temporal.cached_ptr();
                bld.arg(&r_ptr);
                bld.arg(&r_ptr);
                bld.arg(&t_ptr);
                bld.arg(&n);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F9 residual_add_f32: {e:?}"))?;
            }
        }

        // Final norm_f: residual_f32 → temporal (half).
        {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
            let eps: f32 = engine.cfg.rms_norm_eps;
            let mut bld = engine
                .ctx
                .stream
                .launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
            let t_ptr = scratch.temporal.cached_ptr();
            let rms_ptr = scratch.rms_buf.cached_ptr();
            let res_ptr = scratch.residual.cached_ptr();
            let nfw = w.norm_f_weight.ptr();
            bld.arg(&t_ptr);
            bld.arg(&rms_ptr);
            bld.arg(&res_ptr);
            bld.arg(&nfw);
            bld.arg(&b_i);
            bld.arg(&dm_i);
            bld.arg(&eps);
            unsafe { bld.launch(grid) }.map_err(|e| format!("M3 norm_f: {e:?}"))?;
        }

        Ok(())
    }

    pub fn step_mixed_native(
        &self,
        input: &[f32],
        output: &mut [f32],
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        self.eager_gemm_manifest.set(None);
        scratch.gpu_input.upload(&self.engine.ctx.stream, input)?;
        if self.graph.is_some() {
            if self.captured_gemm_route != Some(self.engine.ctx.gemm_route()) {
                return Err(
                    "M3 mixed inference graph replay: GEMM route changed since capture".into(),
                );
            }
            self.ensure_graph_scratch()?;
            assert_eq!(state.ssm_state.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            self.launch_captured_graph()?;
        } else {
            if self.has_gemm_work() {
                prepare_inference_arch_rung(&self.engine.ctx)?;
            }
            let manifest = self
                .engine
                .ctx
                .record_eager_gemm_manifest(|| self.step_kernels_mixed_native(state, scratch))?;
            self.eager_gemm_manifest.set(Some(manifest));
        }
        self.engine
            .ctx
            .stream
            .synchronize()
            .map_err(|e| format!("M3 sync: {e:?}"))?;
        scratch
            .temporal
            .download_f32(&self.engine.ctx.stream, output)?;
        Ok(())
    }

    pub fn step_gpu_only_mixed_native(
        &self,
        input: &[f32],
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        self.eager_gemm_manifest.set(None);
        scratch.gpu_input.upload(&self.engine.ctx.stream, input)?;
        if self.graph.is_some() {
            if self.captured_gemm_route != Some(self.engine.ctx.gemm_route()) {
                return Err(
                    "M3 mixed inference graph replay: GEMM route changed since capture".into(),
                );
            }
            self.ensure_graph_scratch()?;
            assert_eq!(state.ssm_state.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            self.launch_captured_graph()?;
            Ok(())
        } else {
            if self.has_gemm_work() {
                prepare_inference_arch_rung(&self.engine.ctx)?;
            }
            let manifest = self
                .engine
                .ctx
                .record_eager_gemm_manifest(|| self.step_kernels_mixed_native(state, scratch))?;
            self.eager_gemm_manifest.set(Some(manifest));
            Ok(())
        }
    }

    /// Capture the native half pipeline after a successful eager step on these
    /// fixed buffers. Consumes its GEMM-only manifest and rejects missing work,
    /// changed mode/family, or changed bindings. H2D/D2H stay outside the graph.
    ///
    /// # Safety
    ///
    /// `state`, `scratch`, and their views must remain unchanged until the
    /// graph is cleared and all replays complete. The engine context, stream,
    /// cuBLAS workspace, both module sets, functions, and weights must stay fixed.
    pub unsafe fn capture_graph_mixed_native(
        &mut self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        let manifest = self.eager_gemm_manifest.take().ok_or_else(|| {
            "M3 mixed inference graph capture requires a successful eager step".to_string()
        })?;
        self.engine.ctx.presize_bi_scratch()?;
        self.engine.ctx.presize_mixed_graph_scratch_m3(
            &self.engine.prefill_dims(1),
            self.mixed_weights.bulk_dtype,
        )?;
        let snap_state = state.ssm_state.cached_ptr();
        let snap_scratch = scratch.gpu_input.cached_ptr();
        let snap_half_staging = self.engine.ctx.half_staging_ptr();
        let snap_bi_upcast = self.engine.ctx.bi_upcast_scratch_ptrs();
        let snap_gemm_route = self.engine.ctx.gemm_route();
        self.engine.ctx.freeze_graph_scratch();
        let (graph, captured_gemm_plan) = unsafe {
            capture_into_graph_with_gemm_plan(
                &self.engine.ctx,
                manifest.route_capacity,
                &manifest,
                || self.step_kernels_mixed_native(state, scratch),
            )
        }?;
        require_deterministic_gemm_graph_plan(
            &self.engine.ctx,
            self.has_gemm_work(),
            captured_gemm_plan.as_ref(),
            "M3 mixed inference graph capture",
        )?;
        self.graph = Some(graph);
        self.captured_gemm_route = Some(snap_gemm_route);
        self.captured_gemm_plan = captured_gemm_plan;
        self.captured_state_ptr = snap_state;
        self.captured_scratch_ptr = snap_scratch;
        self.captured_half_staging_ptr = snap_half_staging;
        self.captured_bi_upcast_ptrs = snap_bi_upcast;
        self.engine.ctx.note_graph_capture();
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════
// GpuMamba3Backbone — high-level wrapper (owns engine + state + scratch)
// ═══════════════════════════════════════════════════════════════════

/// High-level GPU Mamba-3 backbone: owns engine, state, and scratch.
///
/// Guarantees the same buffers are used during capture and replay.
/// Simple API: `new()` → `step()` → `capture_graph()` → `reset()`.
enum M3BackboneEngine {
    F32(Box<Mamba3GpuInferenceEngine>),
    Mixed(Box<Mamba3GpuInferenceMixed>),
}

enum M3BackboneScratch {
    F32(Box<Mamba3GpuInferenceScratch>),
    Mixed(Box<Mamba3GpuInferenceMixedScratch>),
}

impl M3BackboneScratch {
    fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        match self {
            M3BackboneScratch::F32(s) => s.temporal.cached_ptr(),
            M3BackboneScratch::Mixed(s) => s.temporal.cached_ptr(),
        }
    }

    fn temporal_dtype(&self) -> WeightDtype {
        match self {
            M3BackboneScratch::F32(_) => WeightDtype::F32,
            M3BackboneScratch::Mixed(s) => s.dtype,
        }
    }
}

/// High-level Mamba-3 GPU backbone — unified API over f32 / bf16 / f16 storage.
pub struct GpuMamba3Backbone {
    engine: M3BackboneEngine,
    state: Mamba3GpuInferenceState,
    scratch: M3BackboneScratch,
}

impl GpuMamba3Backbone {
    /// Create an f32 M3 backbone using the GEMM environment.
    ///
    /// Missing mode/family values select Deterministic and the Inference
    /// family. M3 construction now resolves the same strict GEMM environment
    /// as M1. Storage remains f32; inspect the result with [`Self::ctx`].
    /// `MAMBA_RS_ARCH_RUNG` is a separate first-use Inference policy. Invalid
    /// configuration, CUDA, upload, allocation, or graph routes return errors.
    pub fn new(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        Self::new_with_dtype(
            gpu_ordinal,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            WeightDtype::F32,
        )
    }

    /// Create an f32 M3 backbone with an explicit GEMM mode.
    ///
    /// GEMM mode, custom precision/tensor-core controls, and family selectors
    /// in the environment are ignored; the stored family is Inference even for
    /// a cuBLAS mode. `MAMBA_RS_ARCH_RUNG` remains a separate first-use process
    /// policy and is not captured by this constructor. Invalid configuration,
    /// state-cap compilation, CUDA setup, upload, or allocation failures are
    /// returned. Captured graphs require an unchanged complete GEMM route.
    pub fn new_with_mode(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_with_dtype_and_mode(
            gpu_ordinal,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            WeightDtype::F32,
            mode,
        )
    }

    /// Create an M3 backbone with explicit storage dtype and env-selected GEMMs.
    ///
    /// `dtype` selects f32, bf16, or f16 storage independently of execution
    /// mode. Missing selectors use Deterministic + Inference; invalid or
    /// conflicting selectors return an error. Use
    /// [`Self::new_with_dtype_and_mode`] to bypass GEMM selectors and
    /// [`Self::ctx`] to inspect the route that graph capture binds.
    /// `MAMBA_RS_ARCH_RUNG` remains a separate first-use Inference policy;
    /// configuration, dtype, M3 state-cap, CUDA, upload, or allocation can fail.
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Self::new_with_dtype_inner(gpu_ordinal, cpu_weights, cfg, input_dim, batch, dtype, None)
    }

    /// Create an M3 backbone with explicit storage dtype and GEMM mode.
    ///
    /// `dtype` controls storage while `mode` independently controls GEMM
    /// execution. The explicit lane ignores GEMM mode, custom precision/
    /// tensor-core controls, and family selectors in the environment, and
    /// stores the Inference family. `MAMBA_RS_ARCH_RUNG` remains a separate
    /// first-use process policy and is not captured by this constructor.
    /// Existing projection, configuration, M3 state-cap, upload, allocation,
    /// and graph-route errors are preserved.
    pub fn new_with_dtype_and_mode(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_with_dtype_inner(
            gpu_ordinal,
            cpu_weights,
            cfg,
            input_dim,
            batch,
            dtype,
            Some(mode),
        )
    }

    fn new_with_dtype_inner(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
        mode: Option<GemmMode>,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let (engine, state, scratch) = match dtype {
            WeightDtype::F32 => {
                let e = match mode {
                    Some(mode) => Mamba3GpuInferenceEngine::new_with_mode(
                        &device,
                        cpu_weights,
                        cfg,
                        input_dim,
                        batch,
                        mode,
                    )?,
                    None => {
                        Mamba3GpuInferenceEngine::new(&device, cpu_weights, cfg, input_dim, batch)?
                    }
                };
                let s = e.alloc_state()?;
                let sc = M3BackboneScratch::F32(Box::new(e.alloc_scratch()?));
                (M3BackboneEngine::F32(Box::new(e)), s, sc)
            }
            WeightDtype::Bf16 | WeightDtype::F16 => {
                let e = match mode {
                    Some(mode) => Mamba3GpuInferenceMixed::new_with_mode(
                        &device,
                        cpu_weights,
                        cfg,
                        input_dim,
                        batch,
                        dtype,
                        mode,
                    )?,
                    None => Mamba3GpuInferenceMixed::new(
                        &device,
                        cpu_weights,
                        cfg,
                        input_dim,
                        batch,
                        dtype,
                    )?,
                };
                let s = e.alloc_state()?;
                let sc = M3BackboneScratch::Mixed(Box::new(e.alloc_mixed_scratch()?));
                (M3BackboneEngine::Mixed(Box::new(e)), s, sc)
            }
        };
        Ok(Self {
            engine,
            state,
            scratch,
        })
    }

    pub fn dtype(&self) -> WeightDtype {
        match &self.engine {
            M3BackboneEngine::F32(_) => WeightDtype::F32,
            M3BackboneEngine::Mixed(e) => e.bulk_dtype(),
        }
    }

    pub fn temporal_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.scratch.temporal_ptr()
    }

    pub fn temporal_dtype(&self) -> WeightDtype {
        self.scratch.temporal_dtype()
    }

    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        match (&self.engine, &mut self.scratch) {
            (M3BackboneEngine::F32(e), M3BackboneScratch::F32(sc)) => {
                e.step(input, output, &mut self.state, sc)
            }
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                e.step_mixed_native(input, output, &mut self.state, sc)
            }
            _ => Err("M3 engine/scratch dtype mismatch (internal invariant)".to_string()),
        }
    }

    pub fn step_gpu_only(&mut self, input: &[f32]) -> Result<(), String> {
        match (&self.engine, &mut self.scratch) {
            (M3BackboneEngine::F32(e), M3BackboneScratch::F32(sc)) => {
                e.step_gpu_only(input, &mut self.state, sc)
            }
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                e.step_gpu_only_mixed_native(input, &mut self.state, sc)
            }
            _ => Err("M3 engine/scratch dtype mismatch".to_string()),
        }
    }

    pub fn reset(&mut self) -> Result<(), String> {
        let stream = match &self.engine {
            M3BackboneEngine::F32(e) => e.ctx.stream.clone(),
            M3BackboneEngine::Mixed(e) => e.ctx_stream().clone(),
        };
        self.state.reset(&stream)
    }

    pub fn capture_graph(&mut self) -> Result<(), String> {
        let (input_dim, batch, d_model) = match &self.engine {
            M3BackboneEngine::F32(e) => (e.input_dim, e.batch, e.cfg.d_model),
            M3BackboneEngine::Mixed(e) => {
                let er = e.engine_ref();
                (er.input_dim, er.batch, er.cfg.d_model)
            }
        };
        let input = vec![0.0f32; batch * input_dim];
        let mut output = vec![0.0f32; batch * d_model];
        self.step(&input, &mut output)?;
        self.reset()?;
        match (&mut self.engine, &mut self.scratch) {
            (M3BackboneEngine::F32(e), M3BackboneScratch::F32(sc)) => {
                // This owner drops its graph before the owned state and scratch.
                unsafe { e.capture_graph(&mut self.state, sc) }
            }
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                // This owner drops its graph before the owned state and scratch.
                unsafe { e.capture_graph_mixed_native(&mut self.state, sc) }
            }
            _ => Err("M3 engine/scratch dtype mismatch".to_string()),
        }
    }

    pub fn config(&self) -> &Mamba3Config {
        match &self.engine {
            M3BackboneEngine::F32(e) => &e.cfg,
            M3BackboneEngine::Mixed(e) => &e.engine_ref().cfg,
        }
    }

    pub fn batch(&self) -> usize {
        match &self.engine {
            M3BackboneEngine::F32(e) => e.batch,
            M3BackboneEngine::Mixed(e) => e.engine_ref().batch,
        }
    }

    pub fn has_graph(&self) -> bool {
        match &self.engine {
            M3BackboneEngine::F32(e) => e.has_graph(),
            M3BackboneEngine::Mixed(e) => e.has_graph(),
        }
    }

    pub fn stream(&self) -> &Stream {
        match &self.engine {
            M3BackboneEngine::F32(e) => &e.ctx.stream,
            M3BackboneEngine::Mixed(e) => e.ctx_stream(),
        }
    }

    /// Whether this backbone can run the one-pass chunked prompt prefill.
    /// Both precision arms do: the mixed engine runs the TYPED prefill
    /// against its bf16/f16 weights (the trainer's kernel chain; the
    /// persistent decode states are f32 in both pipelines), then downcasts
    /// the final hidden into the typed decode temporal.
    pub fn supports_prefill(&self) -> bool {
        true
    }

    /// Allocate a one-pass prompt executor for a fixed prompt length.
    pub fn alloc_prefill(&self, seq_len: usize) -> Result<super::prefill::Mamba3Prefill, String> {
        match &self.engine {
            M3BackboneEngine::F32(e) => e.alloc_prefill(seq_len),
            M3BackboneEngine::Mixed(e) => e.alloc_prefill(seq_len),
        }
    }

    /// One-pass prompt window (`[batch * seq_len * input_dim]` on the GPU)
    /// straight into the persistent decode state. The final post-norm
    /// hidden lands in the decode temporal, so logits consumers continue
    /// exactly as after a step.
    pub fn prefill_sequence(
        &mut self,
        prefill: &mut super::prefill::Mamba3Prefill,
        mamba_input: &GpuBuffer,
        seq_len: usize,
        carry_state: bool,
    ) -> Result<(), String> {
        match (&self.engine, &mut self.scratch) {
            (M3BackboneEngine::F32(e), M3BackboneScratch::F32(sc)) => e.prefill_sequence(
                prefill,
                mamba_input,
                seq_len,
                &mut self.state,
                carry_state,
                &mut sc.temporal,
            ),
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                // The TYPED prefill against the mixed weights (bf16/f16
                // GEMMs on the batch-invariant TC route, coefficients
                // f32 - the trainer's exact chain). The persistent
                // SSM/K/V/angle states are f32 in both pipelines, so
                // decode continues from them exactly as after typed
                // steps. Non-identity input projections work too - the
                // typed prefill casts the input itself.
                e.prefill_sequence(
                    prefill,
                    mamba_input,
                    seq_len,
                    &mut self.state,
                    carry_state,
                    &mut sc.prefill_hidden,
                )?;
                // Downcast the final hidden into the typed decode temporal
                // so the logits path continues exactly as after a step.
                use cudarc::driver::PushKernelArg;
                let eng = e.engine_ref();
                let ctx = &eng.ctx;
                let n = eng.batch * eng.cfg.d_model;
                let n_i = n as i32;
                let cast = match sc.temporal.dtype() {
                    WeightDtype::Bf16 => &ctx.kernels.cast_f32_to_bf16,
                    WeightDtype::F16 => &ctx.kernels.cast_f32_to_f16,
                    WeightDtype::F32 => {
                        return Err("M3 mixed prefill: unexpected f32 scratch dtype".into());
                    }
                };
                let dst = sc.temporal.cached_ptr();
                let src = sc.prefill_hidden.cached_ptr();
                let mut bld = ctx.stream.launch_builder(cast);
                bld.arg(&dst);
                bld.arg(&src);
                bld.arg(&n_i);
                unsafe { bld.launch(crate::mamba_ssm::gpu::launch::grid_1d(n)) }
                    .map_err(|e| format!("M3 mixed prefill hidden downcast: {e:?}"))?;
                Ok(())
            }
            _ => Err("M3 prefill: engine/scratch precision arms disagree".to_string()),
        }
    }

    /// Shared execution context for downstream projections such as lm_head.
    ///
    /// The selected storage dtype is available through [`Self::dtype`]; GEMM
    /// mode and deterministic family remain properties of this one context.
    /// Inspect them with [`GpuCtx::gemm_mode`] and
    /// [`GpuCtx::bi_gemm_family`]; graph capture retains the complete route.
    pub fn ctx(&self) -> &GpuCtx {
        match &self.engine {
            M3BackboneEngine::F32(e) => &e.ctx,
            M3BackboneEngine::Mixed(e) => &e.engine_ref().ctx,
        }
    }

    /// Access the vendor handle for explicit vendor-only compatibility code.
    pub fn blas(&self) -> &cudarc::cublas::CudaBlas {
        match &self.engine {
            M3BackboneEngine::F32(e) => &e.ctx.blas,
            M3BackboneEngine::Mixed(e) => &e.engine_ref().ctx.blas,
        }
    }

    /// Download the last-computed temporal hidden state to CPU (f32).
    /// Mixed path upcasts from bf16/f16 on the fly.
    pub fn download_temporal(&self, output: &mut [f32]) -> Result<(), String> {
        self.stream()
            .synchronize()
            .map_err(|e| format!("M3 sync: {e:?}"))?;
        match &self.scratch {
            M3BackboneScratch::F32(s) => {
                let tmp = s.temporal.to_cpu(self.stream())?;
                output[..tmp.len()].copy_from_slice(&tmp);
                Ok(())
            }
            M3BackboneScratch::Mixed(s) => s.temporal.download_f32(self.stream(), output),
        }
    }
}

#[cfg(test)]
mod model_gemm_manifest_tests {
    use super::*;
    use crate::mamba_ssm::gpu::blas::vendor_gemm_test::Guard;
    use crate::mamba_ssm::gpu::context::{BiGemmFamily, GemmMode};
    use crate::mamba_ssm::gpu::graph_capture::model_gemm_guard_tests::{
        assert_inventory, configure,
    };

    fn config(layers: usize) -> Mamba3Config {
        Mamba3Config {
            d_model: 32,
            d_state: 8,
            expand: 2,
            headdim: 8,
            ngroups: 1,
            n_layers: layers,
            rope_fraction: 0.5,
            a_floor: 0.0625,
            is_outproj_norm: true,
            ..Mamba3Config::default()
        }
    }

    fn weights(cfg: &Mamba3Config, input: usize, identity: bool) -> Mamba3Weights {
        let mut weights = Mamba3Weights::init(cfg, input, 0x9053);
        if identity {
            weights.input_proj_w.clear();
            weights.input_proj_b.clear();
        }
        weights
    }

    fn projections(
        cfg: &Mamba3Config,
        batch: usize,
        input: Option<usize>,
    ) -> Vec<(usize, usize, usize)> {
        let mut expected = Vec::new();
        if let Some(input) = input {
            expected.push((batch, input, cfg.d_model));
        }
        let layer = [
            (batch, cfg.d_model, cfg.in_proj_out_dim()),
            (batch, cfg.d_inner(), cfg.d_model),
        ];
        for _ in 0..cfg.n_layers {
            expected.extend(layer);
        }
        expected
    }

    fn bits(output: &[f32]) -> Vec<u32> {
        assert!(output.iter().all(|x| x.is_finite()));
        output.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn m3_default_constructor_uses_inference_family() {
        let cfg = config(1);
        let weights = weights(&cfg, cfg.d_model, true);

        let backbone = GpuMamba3Backbone::new(0, &weights, cfg, cfg.d_model, 1)
            .expect("construct M3 F32 backbone from owned identity-projection weights");

        assert_eq!(backbone.ctx().gemm_mode(), GemmMode::Deterministic);
        assert_eq!(backbone.ctx().bi_gemm_family(), BiGemmFamily::Inference);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn m3_capture_requires_successful_public_eager_manifest() {
        let device = GpuDevice::new(0).expect("CUDA device");
        let cfg = config(1);
        let mut weights = Mamba3Weights::init(&cfg, cfg.d_model, 0x9050_0003);
        weights.input_proj_w.clear();
        weights.input_proj_b.clear();
        let mut engine = Mamba3GpuInferenceEngine::new(&device, &weights, cfg, cfg.d_model, 1)
            .expect("M3 owned fixture");
        engine.ctx.set_gemm_mode(GemmMode::Deterministic).unwrap();
        engine.ctx.set_bi_gemm_family(BiGemmFamily::Inference);
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        scratch
            .gpu_input
            .upload(&engine.ctx.stream, &[0.01; 32])
            .unwrap();
        // Warm the actual production body and its caches, but never grant the
        // public eager-step permit that capture is required to consume.
        crate::mamba_ssm::gpu::gemm_bi_inference::prepare_inference_arch_rung(&engine.ctx).unwrap();
        engine.step_kernels(&mut state, &mut scratch).unwrap();
        engine.ctx.stream.synchronize().unwrap();
        let result = unsafe { engine.capture_graph(&mut state, &mut scratch) };
        // Destroy any incorrectly accepted graph while its buffers still live.
        drop(engine);
        let error = result.expect_err("capture must reject a missing successful eager manifest");
        assert!(
            error.contains("eager"),
            "unexpected capture rejection: {error}"
        );
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn m3_model_manifests_replay_all_paths_without_vendor_gemm() {
        let device = GpuDevice::new(0).unwrap();
        let cfg = config(2);
        let deny = Guard::new(true).unwrap();
        for (family, tc) in [
            (BiGemmFamily::Inference, true),
            (BiGemmFamily::Triad, false),
            (BiGemmFamily::Triad, true),
        ] {
            for batch in [1, 3] {
                eprintln!("M3 F32 {family:?} tc={tc} B{batch} nonidentity");
                let input = vec![0.01; batch * 24];
                let mut output = vec![0.0; batch * cfg.d_model];
                let mut engine = Mamba3GpuInferenceEngine::new(
                    &device,
                    &weights(&cfg, 24, false),
                    cfg,
                    24,
                    batch,
                )
                .unwrap();
                configure(&engine.ctx, family, tc);
                let mut state = engine.alloc_state().unwrap();
                let mut scratch = engine.alloc_scratch().unwrap();
                engine
                    .step(&input, &mut output, &mut state, &mut scratch)
                    .unwrap();
                let trace = engine
                    .ctx
                    .record_eager_gemm_trace(|| engine.step_kernels(&mut state, &mut scratch))
                    .unwrap();
                state.reset(&engine.ctx.stream).unwrap();
                engine
                    .step_gpu_only(&input, &mut state, &mut scratch)
                    .unwrap();
                output = scratch.temporal.to_cpu(&engine.ctx.stream).unwrap();
                let expected_bits = bits(&output);
                let manifest = engine.eager_gemm_manifest.get().unwrap();
                unsafe { engine.capture_graph(&mut state, &mut scratch) }.unwrap();
                assert!(engine.eager_gemm_manifest.get().is_none());
                assert_inventory(
                    &engine.ctx,
                    &trace,
                    manifest,
                    engine.captured_gemm_plan.as_ref().unwrap(),
                    &projections(&cfg, batch, Some(24)),
                );
                for gpu_only in [false, true] {
                    state.reset(&engine.ctx.stream).unwrap();
                    if gpu_only {
                        engine
                            .step_gpu_only(&input, &mut state, &mut scratch)
                            .unwrap();
                        output = scratch.temporal.to_cpu(&engine.ctx.stream).unwrap();
                    } else {
                        engine
                            .step(&input, &mut output, &mut state, &mut scratch)
                            .unwrap();
                    }
                    assert_eq!(bits(&output), expected_bits);
                }
                drop(engine);
                for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                    eprintln!("M3 {dtype:?} native {family:?} tc={tc} B{batch}");
                    let input = vec![0.01; batch * cfg.d_model];
                    let mut engine = Mamba3GpuInferenceMixed::new(
                        &device,
                        &weights(&cfg, cfg.d_model, true),
                        cfg,
                        cfg.d_model,
                        batch,
                        dtype,
                    )
                    .unwrap();
                    configure(&engine.engine.ctx, family, tc);
                    let mut state = engine.alloc_state().unwrap();
                    let mut scratch = engine.alloc_mixed_scratch().unwrap();
                    let ctx = &engine.engine.ctx;
                    engine
                        .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
                        .unwrap();
                    let trace = ctx
                        .record_eager_gemm_trace(|| {
                            engine.step_kernels_mixed_native(&mut state, &mut scratch)
                        })
                        .unwrap();
                    state.reset(&ctx.stream).unwrap();
                    engine
                        .step_gpu_only_mixed_native(&input, &mut state, &mut scratch)
                        .unwrap();
                    scratch
                        .temporal
                        .download_f32(&ctx.stream, &mut output)
                        .unwrap();
                    let expected_bits = bits(&output);
                    let manifest = engine.eager_gemm_manifest.get().unwrap();
                    unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }.unwrap();
                    let ctx = &engine.engine.ctx;
                    assert!(engine.eager_gemm_manifest.get().is_none());
                    assert_inventory(
                        ctx,
                        &trace,
                        manifest,
                        engine.captured_gemm_plan.as_ref().unwrap(),
                        &projections(&cfg, batch, None),
                    );
                    if family == BiGemmFamily::Triad && !tc {
                        assert!(trace.routes().iter().all(|r| r.symbol.contains("matvec")));
                    }
                    for gpu_only in [false, true] {
                        state.reset(&ctx.stream).unwrap();
                        if gpu_only {
                            engine
                                .step_gpu_only_mixed_native(&input, &mut state, &mut scratch)
                                .unwrap();
                            scratch
                                .temporal
                                .download_f32(&ctx.stream, &mut output)
                                .unwrap();
                        } else {
                            engine
                                .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
                                .unwrap();
                        }
                        assert_eq!(bits(&output), expected_bits);
                    }
                    drop(engine);
                }
            }
        }
        assert_eq!(deny.calls(), 0);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn m3_failed_steps_clear_permits_and_failed_capture_keeps_installed_plan() {
        let device = GpuDevice::new(0).unwrap();
        let cfg = config(2);
        let weights = weights(&cfg, cfg.d_model, true);
        let input = vec![0.01; cfg.d_model];
        let mut output = vec![0.0; cfg.d_model];
        let mut engine =
            Mamba3GpuInferenceEngine::new(&device, &weights, cfg, cfg.d_model, 1).unwrap();
        configure(&engine.ctx, BiGemmFamily::Inference, true);
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_scratch().unwrap();
        for gpu_only in [false, true] {
            engine
                .step(&input, &mut output, &mut state, &mut scratch)
                .unwrap();
            assert!(engine.eager_gemm_manifest.get().is_some());
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if gpu_only {
                    engine.step_gpu_only(&[], &mut state, &mut scratch)
                } else {
                    engine.step(&[], &mut output, &mut state, &mut scratch)
                }
            }));
            assert!(failed.is_err());
            assert!(engine.eager_gemm_manifest.get().is_none());
            assert!(
                unsafe { engine.capture_graph(&mut state, &mut scratch) }
                    .unwrap_err()
                    .contains("eager")
            );
        }
        engine
            .step(&input, &mut output, &mut state, &mut scratch)
            .unwrap();
        let trace = engine
            .ctx
            .record_eager_gemm_trace(|| engine.step_kernels(&mut state, &mut scratch))
            .unwrap();
        unsafe { engine.capture_graph(&mut state, &mut scratch) }.unwrap();
        let installed = engine.captured_gemm_plan.as_ref().unwrap().launches;
        let mut missing = trace.routes().to_vec();
        missing.remove(1);
        let wrong = crate::mamba_ssm::gpu::kernel_identity::RecordedGemmTrace::from_routes(
            engine.ctx.gemm_route(),
            missing,
        )
        .unwrap()
        .manifest();
        engine.eager_gemm_manifest.set(Some(wrong));
        assert!(unsafe { engine.capture_graph(&mut state, &mut scratch) }.is_err());
        assert!(engine.eager_gemm_manifest.get().is_none());
        assert_eq!(
            engine.captured_gemm_plan.as_ref().unwrap().launches,
            installed
        );
        assert!(engine.has_graph());
        engine
            .step(&input, &mut output, &mut state, &mut scratch)
            .unwrap();
        drop(engine);

        let mut engine =
            Mamba3GpuInferenceMixed::new(&device, &weights, cfg, cfg.d_model, 1, WeightDtype::F16)
                .unwrap();
        configure(&engine.engine.ctx, BiGemmFamily::Inference, true);
        let mut state = engine.alloc_state().unwrap();
        let mut scratch = engine.alloc_mixed_scratch().unwrap();
        assert!(
            unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }
                .unwrap_err()
                .contains("eager")
        );
        for gpu_only in [false, true] {
            engine
                .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
                .unwrap();
            assert!(engine.eager_gemm_manifest.get().is_some());
            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if gpu_only {
                    engine.step_gpu_only_mixed_native(&[], &mut state, &mut scratch)
                } else {
                    engine.step_mixed_native(&[], &mut output, &mut state, &mut scratch)
                }
            }));
            assert!(failed.is_err());
            assert!(engine.eager_gemm_manifest.get().is_none());
            assert!(
                unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }
                    .unwrap_err()
                    .contains("eager")
            );
        }
        engine
            .step_gpu_only_mixed_native(&input, &mut state, &mut scratch)
            .unwrap();
        unsafe { engine.capture_graph_mixed_native(&mut state, &mut scratch) }.unwrap();
        assert!(
            engine
                .engine
                .ctx
                .ensure_half_staging(usize::MAX)
                .unwrap_err()
                .contains("cannot grow")
        );
        let original = engine.captured_bi_upcast_ptrs;
        engine.captured_bi_upcast_ptrs[0] ^= 16;
        assert!(
            engine
                .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
                .unwrap_err()
                .contains("staging scratch changed")
        );
        engine.captured_bi_upcast_ptrs = original;
        engine
            .step_mixed_native(&input, &mut output, &mut state, &mut scratch)
            .unwrap();
        drop(engine);
    }

    #[test]
    #[ignore = "needs a CUDA device"]
    fn m3_explicit_vendor_graph_without_custom_plan_replays() {
        let device = GpuDevice::new(0).unwrap();
        let cfg = config(1);
        let weights = weights(&cfg, cfg.d_model, true);
        for mode in [GemmMode::CublasFast, GemmMode::CublasPedantic] {
            let mut engine =
                Mamba3GpuInferenceEngine::new(&device, &weights, cfg, cfg.d_model, 1).unwrap();
            engine.ctx.set_gemm_mode(mode).unwrap();
            let mut state = engine.alloc_state().unwrap();
            let mut scratch = engine.alloc_scratch().unwrap();
            let input = vec![0.01; cfg.d_model];
            let mut output = vec![0.0; cfg.d_model];
            engine
                .step(&input, &mut output, &mut state, &mut scratch)
                .unwrap();
            let expected = bits(&output);
            unsafe { engine.capture_graph(&mut state, &mut scratch) }.unwrap();
            assert!(engine.captured_gemm_plan.is_none());
            state.reset(&engine.ctx.stream).unwrap();
            engine
                .step(&input, &mut output, &mut state, &mut scratch)
                .unwrap();
            assert_eq!(bits(&output), expected);
            drop(engine);
        }
    }
}
