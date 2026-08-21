//! Mamba-3 SISO GPU **prefill**: process a whole prompt window through the
//! chunked SSD pipeline in one pass, leaving the persistent recurrent state
//! (SSM + K + V + RoPE angle) positioned for token-by-token decode.
//!
//! This is the inference twin of the training forward's chunked path with
//! the backward tape removed: no per-layer activation saves, one reusable
//! scratch across layers. Two window semantics, chosen per call:
//!
//! - **Stateless window** (`carry_state = false`): all four persistent
//!   state buffers are zeroed first — the classify-serve contract, where
//!   every page is its own world.
//! - **Continued window** (`carry_state = true`): the entering SSM state is
//!   seeded from the persistent buffers WITH the trapezoidal boundary fold
//!   (`v_state (x) k_state * dt0 * (1 - trap0)`): the discretization's beta
//!   term reaches one step back across the window seam, so carrying the
//!   SSM state alone would be wrong at exactly the first position of the
//!   new window. The RoPE angle accumulator continues natively.
//!
//! Launch configurations mirror the training forward launch-for-launch —
//! that is what makes prefill-vs-training-forward parity testable at the
//! bit level on the same device.

use super::kernels::Mamba3Kernels;
use super::state::{CUptr, GpuMamba3Dims, GpuMamba3StateBufs, GpuMamba3TargetScratch};
use super::weights::GpuMamba3WeightsInf;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use cudarc::driver::PushKernelArg;
use std::sync::Arc;

/// Chunk-pipeline intermediates the prefill needs beyond the no-save layer
/// scratch: per-window buffers consumed inside F6 and released to the next
/// layer immediately (nothing is kept for a backward).
pub struct Mamba3PrefillChunkScratch {
    /// a_val * dt per (b,t,h) — the log-decay increments.
    d_alpha: GpuBuffer, // [B*T*nh]
    /// K prescaled by the per-step scale (chunk preprocess output).
    k_scaled: GpuBuffer, // [B*T*nh*ds]
    /// Per-step qk_dot lane (chunk preprocess output).
    qk_dot: GpuBuffer, // [B*T*nh]
    /// Per-step scale lane (chunk preprocess output).
    scale: GpuBuffer, // [B*T*nh]
    /// Per-step gamma lane (chunk preprocess output).
    gamma_pre: GpuBuffer, // [B*T*nh]
    /// Intra-chunk cumulative decay.
    da_cumsum: GpuBuffer, // [B*n_chunks*nh*chunk_size]
    /// Chunk contributions, then (in-place) entering states per chunk.
    chunk_states: GpuBuffer, // [B*n_chunks*nh*hd*ds]
    /// State exiting the window per (b,h).
    final_states: GpuBuffer, // [B*nh*hd*ds]
    /// Entering SSM state incl. the trapezoidal boundary fold.
    init_state: GpuBuffer,
    /// fp64 staging for the chunk-parallel angle accumulation.
    angle_chunk_sums: crate::mamba_ssm::gpu::buffers::GpuByteBuffer,
    angle_chunk_carries: crate::mamba_ssm::gpu::buffers::GpuByteBuffer,
}

impl Mamba3PrefillChunkScratch {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        let bt = dims.bt();
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ds = dims.d_state;
        let nc = dims.n_chunks();
        let cs = dims.chunk_size();
        Ok(Self {
            d_alpha: GpuBuffer::zeros(stream, bt * nh)?,
            k_scaled: GpuBuffer::zeros(stream, bt * nh * ds)?,
            qk_dot: GpuBuffer::zeros(stream, bt * nh)?,
            scale: GpuBuffer::zeros(stream, bt * nh)?,
            gamma_pre: GpuBuffer::zeros(stream, bt * nh)?,
            da_cumsum: GpuBuffer::zeros(stream, dims.batch * nc * nh * cs)?,
            chunk_states: GpuBuffer::zeros(stream, dims.batch * nc * nh * hd * ds)?,
            final_states: GpuBuffer::zeros(stream, dims.batch * nh * hd * ds)?,
            init_state: GpuBuffer::zeros(stream, dims.batch * nh * hd * ds)?,
            angle_chunk_sums: crate::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(
                stream,
                dims.batch * nc * dims.nheads * dims.n_angles.max(1) * std::mem::size_of::<f64>(),
            )?,
            angle_chunk_carries: crate::mamba_ssm::gpu::buffers::GpuByteBuffer::zeros(
                stream,
                dims.batch * nc * dims.nheads * dims.n_angles.max(1) * std::mem::size_of::<f64>(),
            )?,
        })
    }
}

/// One prefill executor for a fixed `(batch, seq_len)` shape: the no-save
/// layer scratch plus the chunk intermediates, allocated once and reused
/// across calls (zero allocation on the hot path, CUDA-Graph friendly).
pub struct Mamba3Prefill {
    pub tgt: GpuMamba3TargetScratch,
    chunk: Mamba3PrefillChunkScratch,
}

/// Everything a prefill launch needs besides the executor itself.
pub struct Mamba3PrefillRun<'a> {
    pub ctx: &'a GpuCtx,
    pub kernels: &'a Mamba3Kernels,
    pub dims: &'a GpuMamba3Dims,
    pub weights: &'a GpuMamba3WeightsInf,
    /// `[B*T*mamba_input_dim]` (or `[B*T*d_model]` with an identity proj).
    pub mamba_input: &'a GpuBuffer,
    /// Empty `input_proj_w` on the checkpoint = identity input: skip the
    /// projection GEMM and feed the input straight to the layers.
    pub identity_proj: bool,
    /// `false` = stateless window (zero all four states first);
    /// `true` = continue from the persistent state across the seam.
    pub carry_state: bool,
}

/// Async D2D region copy between raw device pointers (stream-ordered, no
/// sync — the same primitive `GpuBuffer::copy_from_raw` uses, needed here
/// for per-layer slices of the persistent state buffers).
fn dtod_region(
    dst: CUptr,
    src: CUptr,
    floats: usize,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<(), String> {
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst,
            src,
            floats * std::mem::size_of::<f32>(),
            stream.cu_stream(),
        )
    };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!(
            "prefill D2D region copy ({floats} floats): {result:?}"
        ));
    }
    Ok(())
}

impl Mamba3Prefill {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
    ) -> Result<Self, String> {
        dims.validate_index_budget()?;
        Ok(Self {
            tgt: GpuMamba3TargetScratch::new(stream, dims)?,
            chunk: Mamba3PrefillChunkScratch::new(stream, dims)?,
        })
    }

    /// Run the prompt window through every layer. On return
    /// `last_hidden` holds the FINAL-timestep post-norm hidden state
    /// `[B*d_model]` (the decode loop's next input to the lm head) and the
    /// persistent state buffers are positioned after the window's last
    /// token, ready for `step()`.
    pub fn run(
        &mut self,
        run: &Mamba3PrefillRun<'_>,
        states: GpuMamba3StateBufs<'_>,
        last_hidden: &mut GpuBuffer,
    ) -> Result<(), String> {
        let Mamba3PrefillRun {
            ctx,
            kernels: m3k,
            dims,
            weights,
            mamba_input,
            identity_proj,
            carry_state,
        } = *run;
        let tgt = &mut self.tgt;
        let ck = &mut self.chunk;
        let bt = dims.bt();
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let hd = dims.headdim;
        let ng = dims.ngroups;
        let ip = dims.in_proj_dim;
        let na = dims.n_angles;
        let na_alloc = na.max(1);
        let nc = dims.n_chunks();
        let cs = dims.chunk_size() as i32;
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let ng_i = ng as i32;
        let f32_sz = std::mem::size_of::<f32>() as u64;

        if !carry_state {
            states.ssm.zero(&ctx.stream)?;
            states.k.zero(&ctx.stream)?;
            states.v.zero(&ctx.stream)?;
            states.angle.zero(&ctx.stream)?;
        }

        // Input projection (or identity feed) into the working temporal.
        if identity_proj {
            tgt.temporal_work.copy_from_raw(mamba_input, &ctx.stream)?;
        } else {
            crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                ctx,
                &mut tgt.temporal_work,
                mamba_input,
                weights.input_proj_w.ptr(),
                Some(weights.input_proj_b.ptr()),
                (bt, dims.mamba_input_dim, dm),
            )?;
        }

        let ssm_base = states.ssm.raw_ptr(&ctx.stream);
        let k_base = states.k.raw_ptr(&ctx.stream);
        let v_base = states.v.raw_ptr(&ctx.stream);
        let a_base = states.angle.raw_ptr(&ctx.stream);

        for l in 0..dims.n_layers {
            let lw = &weights.layers[l];
            let ssm_ptr = ssm_base + (dims.batch * l * nh * hd * ds) as u64 * f32_sz;
            let k_ptr = k_base + (dims.batch * l * nh * ds) as u64 * f32_sz;
            let v_ptr = v_base + (dims.batch * l * nh * hd) as u64 * f32_sz;
            let a_ptr = a_base + (dims.batch * l * nh * na_alloc) as u64 * f32_sz;

            // Layer norm over the residual stream.
            tgt.residual
                .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
            {
                let bt_i = bt as i32;
                let dm_i = dm as i32;
                let eps: f32 = dims.rms_norm_eps;
                let nw = lw.norm_weight.ptr();
                let mut b = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
                b.arg(tgt.out_flat.inner_mut());
                b.arg(tgt.rms_discard.inner_mut());
                b.arg(tgt.residual.inner());
                b.arg(&nw);
                b.arg(&bt_i);
                b.arg(&dm_i);
                b.arg(&eps);
                unsafe { b.launch(grid_norm(bt, dm)) }
                    .map_err(|e| format!("prefill rmsnorm L{l}: {e:?}"))?;
            }
            crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                ctx,
                &mut tgt.proj_flat,
                &tgt.out_flat,
                lw.in_proj_w.ptr(),
                None,
                (bt, dm, ip),
            )?;
            // 8-way split + fused activations.
            {
                let n_i = bt as i32;
                let di_i = di as i32;
                let na_i = na as i32;
                let db = lw.dt_bias.ptr();
                let mut b = ctx.stream.launch_builder(&m3k.m3_split);
                b.arg(tgt.z.inner_mut());
                b.arg(tgt.x.inner_mut());
                b.arg(tgt.b_raw.inner_mut());
                b.arg(tgt.c_raw.inner_mut());
                b.arg(tgt.dt.inner_mut());
                b.arg(tgt.a_val.inner_mut());
                b.arg(tgt.trap.inner_mut());
                b.arg(tgt.angles_raw.inner_mut());
                b.arg(tgt.dd_dt_raw.inner_mut());
                b.arg(tgt.dd_a_raw.inner_mut());
                b.arg(tgt.trap_raw.inner_mut());
                b.arg(tgt.proj_flat.inner());
                b.arg(&db);
                b.arg(&dims.a_floor);
                b.arg(&n_i);
                b.arg(&di_i);
                b.arg(&ng_i);
                b.arg(&ds_i);
                b.arg(&nh_i);
                b.arg(&na_i);
                unsafe { b.launch(grid_1d(bt * ip)) }
                    .map_err(|e| format!("prefill m3_split L{l}: {e:?}"))?;
            }
            // BCNorm + bias for B and C.
            for (out, rms, raw, w, tag) in [
                (
                    &mut tgt.b_normed,
                    &mut tgt.b_rms,
                    &tgt.b_raw,
                    lw.b_norm_weight.ptr(),
                    "B",
                ),
                (
                    &mut tgt.c_normed,
                    &mut tgt.c_rms,
                    &tgt.c_raw,
                    lw.c_norm_weight.ptr(),
                    "C",
                ),
            ] {
                let n_i = bt as i32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((bt * ng) as u32, 1, 1),
                    block_dim: (ds as u32, 1, 1),
                    shared_mem_bytes: ds as u32 * 4,
                };
                let eps: f32 = dims.rms_norm_eps;
                let mut b = ctx.stream.launch_builder(&m3k.bcnorm_fwd);
                b.arg(out.inner_mut());
                b.arg(rms.inner_mut());
                b.arg(raw.inner());
                b.arg(&w);
                b.arg(&n_i);
                b.arg(&ng_i);
                b.arg(&ds_i);
                b.arg(&eps);
                unsafe { b.launch(cfg) }
                    .map_err(|e| format!("prefill bcnorm {tag} L{l}: {e:?}"))?;
            }
            for (out, normed, bias, tag) in [
                (&mut tgt.b_biased, &tgt.b_normed, lw.b_bias.ptr(), "B"),
                (&mut tgt.c_biased, &tgt.c_normed, lw.c_bias.ptr(), "C"),
            ] {
                let n_i = bt as i32;
                let mut b = ctx.stream.launch_builder(&m3k.bc_bias_add);
                b.arg(out.inner_mut());
                b.arg(normed.inner());
                b.arg(&bias);
                b.arg(&n_i);
                b.arg(&nh_i);
                b.arg(&ng_i);
                b.arg(&ds_i);
                unsafe { b.launch(grid_1d(bt * nh * ds)) }
                    .map_err(|e| format!("prefill bc_bias {tag} L{l}: {e:?}"))?;
            }
            // RoPE angle accumulation continues from the persistent
            // accumulator (zeroed above for a stateless window).
            if na > 0 {
                crate::mamba3_siso::gpu::forward::gpu_angle_chunked_fwd(
                    ctx,
                    m3k,
                    &mut tgt.angle_cumsum,
                    a_ptr,
                    &tgt.angles_raw,
                    &tgt.dt,
                    &ck.angle_chunk_sums,
                    &ck.angle_chunk_carries,
                    dims.batch,
                    dims.seq_len,
                    nh,
                    na,
                )?;

                {
                    let n_i = bt as i32;
                    let na_i = na as i32;
                    let mut b = ctx.stream.launch_builder(&m3k.rope_fwd);
                    b.arg(tgt.k.inner_mut());
                    b.arg(tgt.q.inner_mut());
                    b.arg(tgt.b_biased.inner());
                    b.arg(tgt.c_biased.inner());
                    b.arg(tgt.angle_cumsum.inner());
                    b.arg(&n_i);
                    b.arg(&nh_i);
                    b.arg(&ds_i);
                    b.arg(&na_i);
                    unsafe { b.launch(grid_1d(bt * nh * ds)) }
                        .map_err(|e| format!("prefill rope L{l}: {e:?}"))?;
                }
            } else {
                tgt.k.copy_from_raw(&tgt.b_biased, &ctx.stream)?;
                tgt.q.copy_from_raw(&tgt.c_biased, &ctx.stream)?;
            }
            // Trapezoidal coefficients.
            {
                let n_total = (bt * nh) as i32;
                let mut b = ctx.stream.launch_builder(&m3k.m3_compute_abg);
                b.arg(tgt.alpha.inner_mut());
                b.arg(tgt.beta.inner_mut());
                b.arg(tgt.gamma.inner_mut());
                b.arg(tgt.dt.inner());
                b.arg(tgt.a_val.inner());
                b.arg(tgt.trap.inner());
                b.arg(&n_total);
                unsafe { b.launch(grid_1d(bt * nh)) }
                    .map_err(|e| format!("prefill abg L{l}: {e:?}"))?;
            }
            // Chunked SSD pipeline (the training forward's F6, tape-free).
            {
                let n_total = (bt * nh) as i32;
                let mut b = ctx.stream.launch_builder(&m3k.elementwise_mul);
                b.arg(ck.d_alpha.inner_mut());
                b.arg(tgt.a_val.inner());
                b.arg(tgt.dt.inner());
                b.arg(&n_total);
                unsafe { b.launch(grid_1d(bt * nh)) }
                    .map_err(|e| format!("prefill adt L{l}: {e:?}"))?;
            }
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                    block_dim: (dims.chunk_size() as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
                b.arg(ck.k_scaled.inner_mut());
                b.arg(ck.qk_dot.inner_mut());
                b.arg(ck.scale.inner_mut());
                b.arg(ck.gamma_pre.inner_mut());
                b.arg(tgt.k.inner());
                b.arg(tgt.q.inner());
                b.arg(tgt.dt.inner());
                b.arg(tgt.trap.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&ds_i);
                b.arg(&cs);
                unsafe { b.launch(cfg) }.map_err(|e| format!("prefill preprocess L{l}: {e:?}"))?;
            }
            {
                let block_x = nh.min(256) as u32;
                let grid_z = nh.div_ceil(block_x as usize) as u32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, nc as u32, grid_z),
                    block_dim: (block_x, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_da_cumsum);
                b.arg(ck.da_cumsum.inner_mut());
                b.arg(ck.d_alpha.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&cs);
                unsafe { b.launch(cfg) }.map_err(|e| format!("prefill da_cumsum L{l}: {e:?}"))?;
            }
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dims.batch * nc) as u32, nh.div_ceil(2) as u32, 1),
                    block_dim: (hd as u32, 2, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd);
                b.arg(ck.chunk_states.inner_mut());
                b.arg(tgt.x.inner());
                b.arg(ck.k_scaled.inner());
                b.arg(ck.da_cumsum.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs);
                unsafe { b.launch(cfg) }.map_err(|e| format!("prefill chunk_state L{l}: {e:?}"))?;
            }
            // Entering state: continued windows seed the inter-chunk scan
            // from the persistent SSM state plus the trapezoidal boundary
            // fold; stateless windows scan from zero (null pointer).
            let init_ptr: CUptr = if carry_state {
                dtod_region(
                    ck.init_state.cached_ptr(),
                    ssm_ptr,
                    dims.batch * nh * hd * ds,
                    &ctx.stream,
                )?;
                let dim = hd * ds;
                let block_x = dim.min(256) as u32;
                let grid_z = dim.div_ceil(block_x as usize) as u32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, nh as u32, grid_z),
                    block_dim: (block_x, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_chunk_entering_state);
                b.arg(ck.init_state.inner_mut());
                b.arg(&k_ptr);
                b.arg(&v_ptr);
                b.arg(tgt.dt.inner());
                b.arg(tgt.trap.inner());
                b.arg(&b_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&t_i);
                unsafe { b.launch(cfg) }
                    .map_err(|e| format!("prefill entering_state L{l}: {e:?}"))?;
                ck.init_state.cached_ptr()
            } else {
                0
            };
            {
                let dim = hd * ds;
                let block_x = dim.min(256) as u32;
                let grid_z = dim.div_ceil(block_x as usize) as u32;
                let nc_i = nc as i32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, nh as u32, grid_z),
                    block_dim: (block_x, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_state_passing_fwd);
                b.arg(ck.chunk_states.inner_mut());
                b.arg(ck.final_states.inner_mut());
                b.arg(ck.da_cumsum.inner());
                b.arg(&init_ptr);
                b.arg(&b_i);
                b.arg(&nc_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs);
                b.arg(&t_i);
                unsafe { b.launch(cfg) }
                    .map_err(|e| format!("prefill state_passing L{l}: {e:?}"))?;
            }
            {
                let dp = lw.d_param.ptr();
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dims.batch * nc) as u32, nh.div_ceil(2) as u32, 1),
                    block_dim: (hd as u32, 2, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_chunk_scan_fwd);
                b.arg(tgt.y.inner_mut());
                b.arg(tgt.x.inner());
                b.arg(tgt.q.inner());
                b.arg(ck.k_scaled.inner());
                b.arg(ck.qk_dot.inner());
                b.arg(ck.da_cumsum.inner());
                b.arg(ck.chunk_states.inner());
                b.arg(&dp);
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                b.arg(&cs);
                unsafe { b.launch(cfg) }.map_err(|e| format!("prefill chunk_scan L{l}: {e:?}"))?;
            }
            // Exit states -> persistent buffers (SSM from the scan, final K
            // post-RoPE, final V = x at the window's last timestep).
            {
                let block_x = hd.max(ds) as u32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, nh as u32, 1),
                    block_dim: (block_x, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.m3_writeback_parallel_states);
                b.arg(&ssm_ptr);
                b.arg(&k_ptr);
                b.arg(&v_ptr);
                b.arg(ck.final_states.inner());
                b.arg(tgt.k.inner());
                b.arg(tgt.x.inner());
                b.arg(&b_i);
                b.arg(&t_i);
                b.arg(&nh_i);
                b.arg(&hd_i);
                b.arg(&ds_i);
                unsafe { b.launch(cfg) }.map_err(|e| format!("prefill writeback L{l}: {e:?}"))?;
            }
            // Output gate + out_proj + residual.
            if dims.is_outproj_norm {
                assert!(
                    di <= 1024,
                    "d_inner ({di}) exceeds rmsnorm_gated shared memory limit"
                );
                let nw = lw.norm_gate_weight.ptr();
                let bt_i = bt as i32;
                let di_i = di as i32;
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (bt as u32, 1, 1),
                    block_dim: (di as u32, 1, 1),
                    shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
                };
                let eps: f32 = dims.rms_norm_eps;
                let mut b = ctx.stream.launch_builder(&m3k.rmsnorm_gated_fwd);
                b.arg(tgt.gated.inner_mut());
                b.arg(tgt.rms_discard.inner_mut());
                b.arg(tgt.y.inner());
                b.arg(tgt.z.inner());
                b.arg(&nw);
                b.arg(&bt_i);
                b.arg(&di_i);
                b.arg(&hd_i);
                b.arg(&eps);
                unsafe { b.launch(grid) }.map_err(|e| format!("prefill gated L{l}: {e:?}"))?;
            } else {
                let n = (bt * di) as i32;
                let mut b = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
                b.arg(tgt.gated.inner_mut());
                b.arg(tgt.y.inner());
                b.arg(tgt.z.inner());
                b.arg(&n);
                unsafe { b.launch(grid_1d(bt * di)) }
                    .map_err(|e| format!("prefill silu L{l}: {e:?}"))?;
            }
            crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                ctx,
                &mut tgt.out_flat,
                &tgt.gated,
                lw.out_proj_w.ptr(),
                None,
                (bt, di, dm),
            )?;
            {
                let ne = (bt * dm) as i32;
                let mut b = ctx.stream.launch_builder(&m3k.residual_add);
                b.arg(tgt.temporal_work.inner_mut());
                b.arg(tgt.out_flat.inner());
                b.arg(tgt.residual.inner());
                b.arg(&ne);
                unsafe { b.launch(grid_1d(bt * dm)) }
                    .map_err(|e| format!("prefill residual L{l}: {e:?}"))?;
            }
        }

        // Final norm over the whole window, then keep only the last
        // timestep's hidden state for the decode loop / lm head.
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        {
            let nf = weights.norm_f_weight.ptr();
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = dims.rms_norm_eps;
            let mut b = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
            b.arg(tgt.temporal_work.inner_mut());
            b.arg(tgt.rms_discard.inner_mut());
            b.arg(tgt.residual.inner());
            b.arg(&nf);
            b.arg(&bt_i);
            b.arg(&dm_i);
            b.arg(&eps);
            unsafe { b.launch(grid_norm(bt, dm)) }.map_err(|e| format!("prefill norm_f: {e:?}"))?;
        }
        {
            let dm_i = dm as i32;
            let mut b = ctx.stream.launch_builder(&m3k.gather_last_timestep);
            b.arg(last_hidden.inner_mut());
            b.arg(tgt.temporal_work.inner());
            b.arg(&b_i);
            b.arg(&t_i);
            b.arg(&dm_i);
            let grid = grid_1d(dims.batch * dm);
            unsafe { b.launch(grid) }.map_err(|e| format!("prefill gather_last: {e:?}"))?;
        }
        Ok(())
    }
}

/// A captured CUDA graph of one prefill window over FIXED buffers: replay
/// re-runs the whole window at graph-launch cost. The GEMM-tier flags are
/// snapshotted at capture and asserted at replay — flipping the
/// batch-invariant / tensor-core routing after capture would silently
/// replay the OLD kernels, which is a numeric-route swap the crate treats
/// as a hard error.
pub struct Mamba3PrefillGraph {
    graph: cudarc::driver::CudaGraph,
    flags_at_capture: (bool, bool, bool),
    input_ptr: CUptr,
    ssm_ptr: CUptr,
}

impl Mamba3PrefillGraph {
    /// Capture the window over the given fixed input/state/output buffers.
    /// Upload fresh bytes into the SAME input buffer before each replay.
    pub fn capture(
        prefill: &mut Mamba3Prefill,
        run: &Mamba3PrefillRun<'_>,
        mut states: GpuMamba3StateBufs<'_>,
        last_hidden: &mut GpuBuffer,
    ) -> Result<Self, String> {
        let flags_at_capture = run.ctx.gemm_flags();
        let input_ptr = run.mamba_input.cached_ptr();
        let ssm_ptr = states.ssm.cached_ptr();
        let graph =
            crate::mamba_ssm::gpu::graph_capture::capture_into_graph(&run.ctx.stream, || {
                prefill.run(run, states.reborrow(), last_hidden)
            })?;
        graph
            .upload()
            .map_err(|e| format!("prefill graph upload: {e:?}"))?;
        Ok(Self {
            graph,
            flags_at_capture,
            input_ptr,
            ssm_ptr,
        })
    }

    /// Replay the captured window. `mamba_input` and the state buffers must
    /// be the SAME allocations the capture saw (the graph baked their
    /// device pointers in).
    pub fn replay(
        &self,
        ctx: &GpuCtx,
        mamba_input: &GpuBuffer,
        states: &GpuMamba3StateBufs<'_>,
    ) -> Result<(), String> {
        if ctx.gemm_flags() != self.flags_at_capture {
            return Err(format!(
                "prefill graph replay refused: GEMM-tier flags changed since capture \
                 (captured {:?}, now {:?}) — a replay would silently run the old \
                 numeric route",
                self.flags_at_capture,
                ctx.gemm_flags()
            ));
        }
        if mamba_input.cached_ptr() != self.input_ptr || states.ssm.cached_ptr() != self.ssm_ptr {
            return Err(
                "prefill graph replay refused: input/state buffers differ from the \
                 captured allocations"
                    .to_string(),
            );
        }
        self.graph
            .launch()
            .map_err(|e| format!("prefill graph launch: {e:?}"))
    }
}
