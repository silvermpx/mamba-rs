//! Mamba-3 SISO GPU inference (T=1 step + CUDA Graph).
//!
//! 10-phase forward per layer:
//!   F1: RMSNorm → F2: in_proj SGEMM → F3: m3_split (8-way + fused)
//!   F4: BCNorm + bias + RoPE → F5: m3_compute_abg (alpha/beta/gamma)
//!   F6: m3_step_fwd (trapezoidal SSM) → F7: output gating
//!   F8: out_proj SGEMM → F9: residual add
//! Final: F10: norm_f RMSNorm
//!
//! Weight format: flat buffer + WeightSlice (CUDA Graph safe).
//! State: 4 persistent buffers (SSM + K + V + angle) per layer.
//!
//! Source: Lahoti et al., "Mamba-3", ICLR 2026.

use super::kernels::Mamba3Kernels;
use super::weights::GpuMamba3WeightsInf;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::device::GpuDevice;
use crate::mamba3_siso::config::Mamba3Config;
use crate::mamba3_siso::weights::Mamba3Weights;
use std::ffi::c_int;
use std::sync::Arc;

type Stream = Arc<cudarc::driver::CudaStream>;

/// Inline SGEMM: Y[B,N] = X[B,K] @ W[K,N], beta=0 (no bias).
/// Uses raw cuBLAS — no GpuCtx dependency.
fn sgemm_no_bias(
    blas: &cudarc::cublas::CudaBlas,
    y: &GpuBuffer,
    x: &GpuBuffer,
    w_ptr: cudarc::driver::sys::CUdeviceptr,
    batch: usize,
    n_in: usize,
    n_out: usize,
) -> Result<(), String> {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let w_raw = w_ptr as *const f32;
    let x_raw = x.cached_ptr() as *const f32;
    let y_raw = y.cached_ptr() as *mut f32;
    unsafe {
        cudarc::cublas::result::sgemm(
            *blas.handle(),
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            n_out as c_int,
            batch as c_int,
            n_in as c_int,
            &alpha as *const f32,
            w_raw,
            n_out as c_int,
            x_raw,
            n_in as c_int,
            &beta as *const f32,
            y_raw,
            n_out as c_int,
        )
        .map_err(|e| format!("cuBLAS sgemm failed: {e:?}"))?;
    }
    Ok(())
}

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
    pub angle_cumsum: GpuBuffer,  // [batch * nh * n_angles]
    pub alpha: GpuBuffer,         // [batch * nh]
    pub beta: GpuBuffer,          // [batch * nh]
    pub gamma: GpuBuffer,         // [batch * nh]
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
            angle_cumsum: GpuBuffer::zeros(stream, batch * nh * na)?,
            alpha: GpuBuffer::zeros(stream, batch * nh)?,
            beta: GpuBuffer::zeros(stream, batch * nh)?,
            gamma: GpuBuffer::zeros(stream, batch * nh)?,
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
    pub temporal: DtypedBuf,  // bf16/f16 — post-norm branch + final lm_head input
    pub residual: GpuBuffer,  // f32 — cross-layer accumulator (HF residual_in_fp32)
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
    pub angles_raw: GpuBuffer,   // f32 (tanh/PI·dt products accumulate in f64)
    pub angle_cumsum: GpuBuffer, // f32 (sincosf consumer)
    pub alpha: GpuBuffer,
    pub beta: GpuBuffer,
    pub gamma: GpuBuffer,
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
            return Err(
                "Mamba3GpuInferenceMixedScratch requires bf16 or f16 dtype".to_string(),
            );
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
            angle_cumsum: GpuBuffer::zeros(stream, batch * nh * na)?,
            alpha: GpuBuffer::zeros(stream, batch * nh)?,
            beta: GpuBuffer::zeros(stream, batch * nh)?,
            gamma: GpuBuffer::zeros(stream, batch * nh)?,
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
/// Supports CUDA Graph capture for ~2-5x launch speedup.
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
    pub stream: Stream,
    pub blas: Arc<cudarc::cublas::CudaBlas>,
    pub cfg: Mamba3Config,
    pub batch: usize,
    pub input_dim: usize,
    /// HF M3 models have no input_proj — skip GEMM, copy input → temporal.
    pub identity_proj: bool,
    graph: Option<cudarc::driver::CudaGraph>,
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
}

impl Mamba3GpuInferenceEngine {
    /// Create inference engine: compile M3 kernels, upload weights to flat GPU buffer.
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        cfg.validate();
        unsafe { device.context().disable_event_tracking() };
        let stream = device.fork_stream()?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = Mamba3Kernels::compile(device.context(), arch)?;
        let (blas, _ws) = device.create_cublas(&stream)?;
        let weights = GpuMamba3WeightsInf::from_cpu(&stream, cpu_weights, input_dim)?;
        let identity_proj = cpu_weights.input_proj_w.is_empty();

        Ok(Self {
            kernels,
            weights,
            stream,
            blas: Arc::new(blas),
            cfg,
            batch,
            input_dim,
            identity_proj,
            graph: None,
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
        })
    }

    /// Capture CUDA Graph for the inference step.
    ///
    /// After capture, `step()` replays the graph instead of launching kernels
    /// individually, reducing launch overhead from ~2ms to ~100us per step.
    ///
    /// Call after at least one warmup `step()` to stabilize kernel launches.
    /// H2D/D2H transfers remain outside the graph.
    pub fn capture_graph(
        &mut self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceScratch,
    ) -> Result<(), String> {
        self.stream
            .synchronize()
            .map_err(|e| format!("pre-capture sync: {e:?}"))?;

        self.stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| format!("begin_capture: {e:?}"))?;

        let capture_result = self.step_kernels(state, scratch);
        if capture_result.is_err() {
            let _ = self.stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return capture_result;
        }

        let graph = self.stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| format!("end_capture: {e:?}"))?;

        // Pre-upload graph resources for zero first-launch latency
        if let Some(ref g) = graph {
            g.upload().map_err(|e| format!("graph upload: {e:?}"))?;
        }

        self.graph = graph;
        self.captured_state_ptr = state.ssm_state.cached_ptr();
        self.captured_scratch_ptr = scratch.gpu_input.cached_ptr();
        Ok(())
    }

    /// Whether a CUDA Graph has been captured.
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Allocate zeroed inference state.
    pub fn alloc_state(&self) -> Result<Mamba3GpuInferenceState, String> {
        Mamba3GpuInferenceState::zeros(&self.stream, self.batch, &self.cfg)
    }

    /// Allocate scratch buffers.
    pub fn alloc_scratch(&self) -> Result<Mamba3GpuInferenceScratch, String> {
        Mamba3GpuInferenceScratch::zeros(&self.stream, self.batch, &self.cfg, self.input_dim)
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
                .copy_from_raw(&scratch.gpu_input, &self.stream)?;
        } else {
            // Input projection SGEMM
            sgemm_no_bias(
                &self.blas,
                &scratch.temporal,
                &scratch.gpu_input,
                self.weights.input_proj_w.ptr(),
                b,
                self.input_dim,
                dm,
            )?;
            // Add bias
            {
                let n = b * dm;
                let n_i = n as i32;
                let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
                let mut builder = self.stream.launch_builder(&self.kernels.vec_add_inplace);
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
                    self.stream.cu_stream(),
                );
            }
        }
        {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
            let eps: f32 = 1e-5;
            let mut builder = self.stream.launch_builder(&self.kernels.rmsnorm_fwd);
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
    /// Phases: F1(RMSNorm) → F2(in_proj SGEMM) → F3(m3_split) → F4(BCNorm+bias+RoPE)
    ///       → F5(alpha/beta/gamma) → F6(m3_step_fwd) → F7(gating) → F8(out_proj SGEMM)
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

        // Save residual before norm (avoids in-place aliasing)
        {
            let src = scratch.temporal.cached_ptr();
            let dst = scratch.residual.cached_ptr();
            let bytes = b * dm * std::mem::size_of::<f32>();
            let result = unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.stream.cu_stream())
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!("D2D copy residual: {:?}", result));
            }
        }

        // F1: RMSNorm (input=residual, output=post_norm — separate buffers)
        {
            let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
            let eps: f32 = 1e-5;
            let nw_ptr = lw.norm_weight.ptr();
            let mut builder = self.stream.launch_builder(&self.kernels.rmsnorm_fwd);
            builder.arg(scratch.post_norm.inner());
            builder.arg(scratch.rms_buf.inner());
            builder.arg(scratch.residual.inner());
            builder.arg(&nw_ptr);
            builder.arg(&b_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F1 rmsnorm: {e:?}"))?;
        }

        // F2: in_proj SGEMM [batch, d_model] → [batch, in_proj_dim]
        sgemm_no_bias(
            &self.blas,
            &scratch.proj,
            &scratch.post_norm,
            lw.in_proj_w.ptr(),
            b,
            dm,
            ip,
        )?;

        // F3: m3_split (8-way + fused softplus/sigmoid)
        {
            let n = b * ip;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.m3_split);
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

        // F4a: BCNorm forward (B)
        {
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: ((b * ng) as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: (ds * 4) as u32,
            };
            let mut builder = self.stream.launch_builder(&self.kernels.bcnorm_fwd);
            builder.arg(scratch.b_normed.inner());
            builder.arg(scratch.b_rms.inner());
            builder.arg(scratch.b_raw.inner());
            builder.arg(lw.b_norm_weight.inner());
            builder.arg(&b_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4a bcnorm B: {e:?}"))?;
        }

        // F4a: BCNorm forward (C)
        {
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: ((b * ng) as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: (ds * 4) as u32,
            };
            let mut builder = self.stream.launch_builder(&self.kernels.bcnorm_fwd);
            builder.arg(scratch.c_normed.inner());
            builder.arg(scratch.c_rms.inner());
            builder.arg(scratch.c_raw.inner());
            builder.arg(lw.c_norm_weight.inner());
            builder.arg(&b_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4a bcnorm C: {e:?}"))?;
        }

        // F4b: Bias add (B: group → head expansion)
        {
            let n = b * nh * ds;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.bc_bias_add);
            builder.arg(scratch.b_biased.inner());
            builder.arg(scratch.b_normed.inner());
            builder.arg(lw.b_bias.inner());
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4b bias B: {e:?}"))?;
        }

        // F4b: Bias add (C)
        {
            let n = b * nh * ds;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.bc_bias_add);
            builder.arg(scratch.c_biased.inner());
            builder.arg(scratch.c_normed.inner());
            builder.arg(lw.c_bias.inner());
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4b bias C: {e:?}"))?;
        }

        // F4c: Angle accumulation + RoPE
        if na > 0 {
            // angle_dt_fwd_batch: accumulates angles for all (batch, head) pairs
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: (b as u32, ((nh * na).div_ceil(256)) as u32, 1),
                block_dim: (256.min((nh * na) as u32), 1, 1),
                shared_mem_bytes: 0,
            };
            let a_ptr = state.angle_state.inner_at(a_off);
            let mut builder = self
                .stream
                .launch_builder(&self.kernels.m3_angle_dt_fwd_batch);
            // CUDA signature: angle_cumsum (output), angle_state (in/out)
            builder.arg(scratch.angle_cumsum.inner());
            builder.arg(&a_ptr);
            builder.arg(scratch.angles_raw.inner());
            builder.arg(scratch.dt.inner());
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4c angle_dt: {e:?}"))?;

            // rope_fwd: rotate B and C pairs
            let n = b * nh * ds;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.rope_fwd);
            builder.arg(scratch.k_cur.inner());
            builder.arg(scratch.q_cur.inner());
            builder.arg(scratch.b_biased.inner());
            builder.arg(scratch.c_biased.inner());
            builder.arg(scratch.angle_cumsum.inner());
            builder.arg(&b_i);
            builder.arg(&nh_i);
            builder.arg(&ds_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F4c rope: {e:?}"))?;
        } else {
            // No RoPE — copy biased directly
            {
                let bytes = b * nh * ds * std::mem::size_of::<f32>();
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        scratch.k_cur.cached_ptr(),
                        scratch.b_biased.cached_ptr(),
                        bytes,
                        self.stream.cu_stream(),
                    );
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        scratch.q_cur.cached_ptr(),
                        scratch.c_biased.cached_ptr(),
                        bytes,
                        self.stream.cu_stream(),
                    );
                }
            }
        }

        // F5: Compute alpha/beta/gamma
        {
            let n = b * nh;
            let n_i = n as i32;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.m3_compute_abg);
            builder.arg(scratch.alpha.inner());
            builder.arg(scratch.beta.inner());
            builder.arg(scratch.gamma.inner());
            builder.arg(scratch.dt.inner());
            builder.arg(scratch.a_val.inner());
            builder.arg(scratch.trap.inner());
            builder.arg(&n_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F5 compute_abg: {e:?}"))?;
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
            let mut builder = self.stream.launch_builder(&self.kernels.m3_step_fwd);
            // CUDA signature: ssm_state, k_state, v_state, y, ...
            builder.arg(&ssm_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(scratch.y.inner());
            builder.arg(scratch.x.inner());
            builder.arg(scratch.k_cur.inner());
            builder.arg(scratch.q_cur.inner());
            builder.arg(scratch.alpha.inner());
            builder.arg(scratch.beta.inner());
            builder.arg(scratch.gamma.inner());
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
            let mut builder = self.stream.launch_builder(&self.kernels.rmsnorm_gated_fwd);
            builder.arg(scratch.gated.inner());
            builder.arg(scratch.gated_rms_buf.inner()); // rms_vals (rstd per group)
            builder.arg(scratch.y.inner());
            builder.arg(scratch.z.inner());
            builder.arg(lw.norm_gate_weight.inner());
            builder.arg(&b_i);
            builder.arg(&di_i);
            builder.arg(&hd_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F7 rmsnorm_gated: {e:?}"))?;
        } else {
            let n = b * di;
            let n_i = n as i32;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.silu_gate_fwd);
            builder.arg(scratch.gated.inner());
            builder.arg(scratch.y.inner());
            builder.arg(scratch.z.inner());
            builder.arg(&n_i);
            unsafe { builder.launch(grid) }.map_err(|e| format!("F7 silu_gate: {e:?}"))?;
        }

        // F8: out_proj SGEMM [batch, d_inner] → [batch, d_model]
        sgemm_no_bias(
            &self.blas,
            &scratch.temporal,
            &scratch.gated,
            lw.out_proj_w.ptr(),
            b,
            di,
            dm,
        )?;

        // F9: Residual add
        {
            let n = b * dm;
            let n_i = n as i32;
            let grid = crate::mamba_ssm::gpu::launch::grid_1d(n);
            let mut builder = self.stream.launch_builder(&self.kernels.vec_add_inplace);
            builder.arg(scratch.temporal.inner());
            builder.arg(scratch.residual.inner());
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
        scratch.gpu_input.upload(&self.stream, input)?;

        // GPU kernel pipeline (graph replay or individual launches)
        if let Some(ref g) = self.graph {
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
            g.launch().map_err(|e| format!("graph launch: {e:?}"))?;
        } else {
            self.step_kernels(state, scratch)?;
        }

        // Sync + D2H download
        self.stream
            .synchronize()
            .map_err(|e| format!("sync: {e:?}"))?;
        let cpu_out = scratch.temporal.to_cpu(&self.stream)?;
        output[..cpu_out.len()].copy_from_slice(&cpu_out);

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

use crate::mamba3_siso::gpu::weights::GpuMamba3MixedWeights;
use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_raw_no_bias};

pub struct Mamba3GpuInferenceMixed {
    engine: Mamba3GpuInferenceEngine, // owns ctx + kernels + blas + (unused f32 weights)
    mixed_weights: GpuMamba3MixedWeights,
    graph: Option<cudarc::driver::CudaGraph>,
    captured_state_ptr: u64,
    captured_scratch_ptr: u64,
}

impl Mamba3GpuInferenceMixed {
    pub fn new(
        device: &GpuDevice,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        bulk_dtype: WeightDtype,
    ) -> Result<Self, String> {
        cfg.validate();
        // Reuse the f32 engine constructor to compile kernels + upload f32
        // weights (needed because the mixed path still consumes f32 biases
        // and per-head coefficients via the f32 engine's weight-agnostic
        // pointer views).
        let engine = Mamba3GpuInferenceEngine::new(device, cpu_weights, cfg, input_dim, batch)?;
        let mixed_weights = GpuMamba3MixedWeights::from_cpu(&engine.stream, cpu_weights, bulk_dtype)?;
        Ok(Self {
            engine,
            mixed_weights,
            graph: None,
            captured_state_ptr: 0,
            captured_scratch_ptr: 0,
        })
    }

    pub fn alloc_state(&self) -> Result<Mamba3GpuInferenceState, String> {
        self.engine.alloc_state()
    }

    pub fn alloc_mixed_scratch(&self) -> Result<Mamba3GpuInferenceMixedScratch, String> {
        Mamba3GpuInferenceMixedScratch::zeros(
            &self.engine.stream,
            self.engine.batch,
            &self.engine.cfg,
            self.engine.input_dim,
            self.mixed_weights.bulk_dtype,
        )
    }

    pub fn ctx_stream(&self) -> &Stream {
        &self.engine.stream
    }

    pub fn bulk_dtype(&self) -> WeightDtype {
        self.mixed_weights.bulk_dtype
    }

    pub fn engine_ref(&self) -> &Mamba3GpuInferenceEngine {
        &self.engine
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
            scratch.dtype,
            self.mixed_weights.bulk_dtype,
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

        // Seed f32 residual with f32 gpu_input (identity_proj).
        scratch.residual.copy_from(&scratch.gpu_input, &engine.stream)?;

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
                let eps: f32 = 1e-5;
                let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, dm);
                let mut bld = engine
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
            gpu_gemm_typed_raw_no_bias(
                &engine.blas,
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
                (b, dm, ip),
            )?;

            // F3: m3_split typed — splits bf16 proj, writes bf16 activations + f32 coefficients.
            {
                let n = b * ip;
                let grid = grid_1d(n);
                let mut bld = engine.stream.launch_builder(k.m3_split_typed.get(dt));
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

            // F4a: bcnorm typed × 2.
            for (which, src_ptr, dst_ptr, rms_ptr, nw_ptr) in [
                (
                    "B",
                    scratch.b_raw.cached_ptr(),
                    scratch.b_normed.cached_ptr(),
                    scratch.b_rms.cached_ptr(),
                    lw.b_norm_weight.ptr(),
                ),
                (
                    "C",
                    scratch.c_raw.cached_ptr(),
                    scratch.c_normed.cached_ptr(),
                    scratch.c_rms.cached_ptr(),
                    lw.c_norm_weight.ptr(),
                ),
            ] {
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: ((b * ng) as u32, 1, 1),
                    block_dim: (ds as u32, 1, 1),
                    shared_mem_bytes: (ds * 4) as u32,
                };
                let mut bld = engine.stream.launch_builder(k.bcnorm_fwd_typed.get(dt));
                bld.arg(&dst_ptr);
                bld.arg(&rms_ptr);
                bld.arg(&src_ptr);
                bld.arg(&nw_ptr);
                bld.arg(&b_i);
                bld.arg(&ng_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F4a bcnorm {which}: {e:?}"))?;
            }

            // F4b: bc_bias_add typed × 2.
            for (which, src_ptr, dst_ptr, bias_ptr) in [
                (
                    "B",
                    scratch.b_normed.cached_ptr(),
                    scratch.b_biased.cached_ptr(),
                    lw.b_bias.ptr(),
                ),
                (
                    "C",
                    scratch.c_normed.cached_ptr(),
                    scratch.c_biased.cached_ptr(),
                    lw.c_bias.ptr(),
                ),
            ] {
                let n = b * nh * ds;
                let grid = grid_1d(n);
                let mut bld = engine.stream.launch_builder(k.bc_bias_add_typed.get(dt));
                bld.arg(&dst_ptr);
                bld.arg(&src_ptr);
                bld.arg(&bias_ptr);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&ng_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F4b bias {which}: {e:?}"))?;
            }

            // F4c: Angle accumulation + RoPE.
            if na > 0 {
                // angle_dt stays f32 (angles_raw f32 + dt f32, f64 accumulator internally).
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (b as u32, ((nh * na).div_ceil(256)) as u32, 1),
                    block_dim: (256.min((nh * na) as u32), 1, 1),
                    shared_mem_bytes: 0,
                };
                let a_ptr = state.angle_state.inner_at(a_off);
                let mut bld = engine.stream.launch_builder(&k.m3_angle_dt_fwd_batch);
                bld.arg(scratch.angle_cumsum.inner());
                bld.arg(&a_ptr);
                bld.arg(scratch.angles_raw.inner());
                bld.arg(scratch.dt.inner());
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&na_i);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F4c angle_dt: {e:?}"))?;

                // rope typed: half B/C, f32 angle_cumsum.
                let n = b * nh * ds;
                let grid = grid_1d(n);
                let mut bld = engine.stream.launch_builder(k.rope_fwd_typed.get(dt));
                let kc_ptr = scratch.k_cur.cached_ptr();
                let qc_ptr = scratch.q_cur.cached_ptr();
                let bb_ptr = scratch.b_biased.cached_ptr();
                let cb_ptr = scratch.c_biased.cached_ptr();
                let ac_ptr = scratch.angle_cumsum.cached_ptr();
                bld.arg(&kc_ptr);
                bld.arg(&qc_ptr);
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&ac_ptr);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&ds_i);
                bld.arg(&na_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F4c rope: {e:?}"))?;
            } else {
                // No RoPE: copy B_biased → k_cur and C_biased → q_cur (same dtype, bytes copy).
                let bytes = b * nh * ds * dt.size_bytes();
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        scratch.k_cur.cached_ptr(),
                        scratch.b_biased.cached_ptr(),
                        bytes,
                        engine.stream.cu_stream(),
                    );
                    cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        scratch.q_cur.cached_ptr(),
                        scratch.c_biased.cached_ptr(),
                        bytes,
                        engine.stream.cu_stream(),
                    );
                }
            }

            // F5: m3_compute_abg — stays f32 (pure coefficient kernel).
            {
                let n = b * nh;
                let n_i = n as i32;
                let grid = grid_1d(n);
                let mut bld = engine.stream.launch_builder(&k.m3_compute_abg);
                bld.arg(scratch.alpha.inner());
                bld.arg(scratch.beta.inner());
                bld.arg(scratch.gamma.inner());
                bld.arg(scratch.dt.inner());
                bld.arg(scratch.a_val.inner());
                bld.arg(scratch.trap.inner());
                bld.arg(&n_i);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F5 compute_abg: {e:?}"))?;
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
                let mut bld = engine.stream.launch_builder(k.m3_step_fwd_typed.get(dt));
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
                let alpha_ptr = scratch.alpha.cached_ptr();
                let beta_ptr = scratch.beta.cached_ptr();
                let gamma_ptr = scratch.gamma.cached_ptr();
                let dp_ptr = lw.d_param.ptr();
                bld.arg(&alpha_ptr);
                bld.arg(&beta_ptr);
                bld.arg(&gamma_ptr);
                bld.arg(&dp_ptr);
                bld.arg(&b_i);
                bld.arg(&nh_i);
                bld.arg(&hd_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F6 step_fwd: {e:?}"))?;
            }

            // F7: output gating (typed).
            if cfg.is_outproj_norm {
                let grid = crate::mamba_ssm::gpu::launch::grid_norm(b, di);
                let mut bld = engine
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
                unsafe { bld.launch(grid) }
                    .map_err(|e| format!("M3 F7 rmsnorm_gated: {e:?}"))?;
            } else {
                let n = b * di;
                let n_i = n as i32;
                let grid = grid_1d(n);
                let mut bld = engine.stream.launch_builder(k.silu_gate_fwd_typed.get(dt));
                let gated_ptr = scratch.gated.cached_ptr();
                let y_ptr = scratch.y.cached_ptr();
                let z_ptr = scratch.z.cached_ptr();
                bld.arg(&gated_ptr);
                bld.arg(&y_ptr);
                bld.arg(&z_ptr);
                bld.arg(&n_i);
                unsafe { bld.launch(grid) }.map_err(|e| format!("M3 F7 silu_gate: {e:?}"))?;
            }

            // F8: out_proj GEMM typed.
            gpu_gemm_typed_raw_no_bias(
                &engine.blas,
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
                (b, di, dm),
            )?;

            // F9: residual_add_f32_typed — residual (f32) += temporal (half), stays f32.
            {
                let n = (b * dm) as i32;
                let grid = grid_1d(b * dm);
                let mut bld = engine.stream.launch_builder(k.residual_add_f32_typed.get(dt));
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
            let eps: f32 = 1e-5;
            let mut bld = engine
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
        scratch.gpu_input.upload(&self.engine.stream, input)?;
        if let Some(ref g) = self.graph {
            assert_eq!(state.ssm_state.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            g.launch()
                .map_err(|e| format!("M3 graph launch mixed: {e:?}"))?;
        } else {
            self.step_kernels_mixed_native(state, scratch)?;
        }
        self.engine
            .stream
            .synchronize()
            .map_err(|e| format!("M3 sync: {e:?}"))?;
        scratch
            .temporal
            .download_f32(&self.engine.stream, output)?;
        Ok(())
    }

    pub fn step_gpu_only_mixed_native(
        &self,
        input: &[f32],
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        scratch.gpu_input.upload(&self.engine.stream, input)?;
        if let Some(ref g) = self.graph {
            assert_eq!(state.ssm_state.cached_ptr(), self.captured_state_ptr);
            assert_eq!(scratch.gpu_input.cached_ptr(), self.captured_scratch_ptr);
            g.launch()
                .map_err(|e| format!("M3 graph launch mixed: {e:?}"))?;
            Ok(())
        } else {
            self.step_kernels_mixed_native(state, scratch)
        }
    }

    pub fn capture_graph_mixed_native(
        &mut self,
        state: &mut Mamba3GpuInferenceState,
        scratch: &mut Mamba3GpuInferenceMixedScratch,
    ) -> Result<(), String> {
        self.engine
            .stream
            .synchronize()
            .map_err(|e| format!("M3 pre-capture sync: {e:?}"))?;
        self.engine
            .stream
            .begin_capture(
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .map_err(|e| format!("M3 begin_capture mixed: {e:?}"))?;
        let captured_state_ptr = state.ssm_state.cached_ptr();
        let captured_scratch_ptr = scratch.gpu_input.cached_ptr();
        if let Err(e) = self.step_kernels_mixed_native(state, scratch) {
            let _ = self.engine.stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(format!("M3 capture body mixed failed: {e}"));
        }
        let graph = self
            .engine
            .stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| format!("M3 end_capture mixed: {e:?}"))?
            .ok_or("M3: no graph captured (mixed)")?;
        self.graph = Some(graph);
        self.captured_state_ptr = captured_state_ptr;
        self.captured_scratch_ptr = captured_scratch_ptr;
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
    F32(Mamba3GpuInferenceScratch),
    Mixed(Mamba3GpuInferenceMixedScratch),
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
    /// Create an f32 GPU backbone (equivalent to `new_with_dtype(F32)`).
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

    /// Create a Mamba-3 GPU backbone with explicit storage dtype.
    pub fn new_with_dtype(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let (engine, state, scratch) = match dtype {
            WeightDtype::F32 => {
                let e =
                    Mamba3GpuInferenceEngine::new(&device, cpu_weights, cfg, input_dim, batch)?;
                let s = e.alloc_state()?;
                let sc = M3BackboneScratch::F32(e.alloc_scratch()?);
                (M3BackboneEngine::F32(Box::new(e)), s, sc)
            }
            WeightDtype::Bf16 | WeightDtype::F16 => {
                let e = Mamba3GpuInferenceMixed::new(
                    &device, cpu_weights, cfg, input_dim, batch, dtype,
                )?;
                let s = e.alloc_state()?;
                let sc = M3BackboneScratch::Mixed(e.alloc_mixed_scratch()?);
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
                // f32 engine has no step_gpu_only yet — run full step but ignore output download.
                // For now, call the full step and discard output.
                let mut scratch_out = vec![0.0f32; e.batch * e.cfg.d_model];
                e.step(input, &mut scratch_out, &mut self.state, sc)
            }
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                e.step_gpu_only_mixed_native(input, &mut self.state, sc)
            }
            _ => Err("M3 engine/scratch dtype mismatch".to_string()),
        }
    }

    pub fn reset(&mut self) -> Result<(), String> {
        let stream = match &self.engine {
            M3BackboneEngine::F32(e) => e.stream.clone(),
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
                e.capture_graph(&mut self.state, sc)
            }
            (M3BackboneEngine::Mixed(e), M3BackboneScratch::Mixed(sc)) => {
                e.capture_graph_mixed_native(&mut self.state, sc)
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
            M3BackboneEngine::F32(e) => &e.stream,
            M3BackboneEngine::Mixed(e) => e.ctx_stream(),
        }
    }
}
