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
            scratch.temporal.copy_from_raw(&scratch.gpu_input, &self.stream)?;
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
// GpuMamba3Backbone — high-level wrapper (owns engine + state + scratch)
// ═══════════════════════════════════════════════════════════════════

/// High-level GPU Mamba-3 backbone: owns engine, state, and scratch.
///
/// Guarantees the same buffers are used during capture and replay.
/// Simple API: `new()` → `step()` → `capture_graph()` → `reset()`.
pub struct GpuMamba3Backbone {
    engine: Mamba3GpuInferenceEngine,
    state: Mamba3GpuInferenceState,
    scratch: Mamba3GpuInferenceScratch,
}

impl GpuMamba3Backbone {
    /// Create a GPU backbone from CPU weights.
    pub fn new(
        gpu_ordinal: usize,
        cpu_weights: &Mamba3Weights,
        cfg: Mamba3Config,
        input_dim: usize,
        batch: usize,
    ) -> Result<Self, String> {
        let device = GpuDevice::new(gpu_ordinal)?;
        let engine = Mamba3GpuInferenceEngine::new(&device, cpu_weights, cfg, input_dim, batch)?;
        let state = engine.alloc_state()?;
        let scratch = engine.alloc_scratch()?;
        Ok(Self {
            engine,
            state,
            scratch,
        })
    }

    /// Run one inference step.
    pub fn step(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        self.engine
            .step(input, output, &mut self.state, &mut self.scratch)
    }

    /// Reset recurrent state (episode boundary).
    pub fn reset(&mut self) -> Result<(), String> {
        self.state.reset(&self.engine.stream)
    }

    /// Capture CUDA Graph for faster inference.
    /// Runs one warmup step, resets state, then captures.
    pub fn capture_graph(&mut self) -> Result<(), String> {
        let input = vec![0.0f32; self.engine.batch * self.engine.input_dim];
        let mut output = vec![0.0f32; self.engine.batch * self.engine.cfg.d_model];
        self.engine
            .step(&input, &mut output, &mut self.state, &mut self.scratch)?;
        self.state.reset(&self.engine.stream)?;
        self.engine
            .capture_graph(&mut self.state, &mut self.scratch)
    }

    /// Config reference.
    pub fn config(&self) -> &Mamba3Config {
        &self.engine.cfg
    }

    /// Batch size.
    pub fn batch(&self) -> usize {
        self.engine.batch
    }

    /// Whether CUDA Graph is captured.
    pub fn has_graph(&self) -> bool {
        self.engine.has_graph()
    }
}
