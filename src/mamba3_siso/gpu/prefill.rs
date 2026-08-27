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
use super::state::{
    CUptr, GpuMamba3Dims, GpuMamba3StateBufs, GpuMamba3TargetScratch, Mamba3PrefillTypedScratch,
};
use super::weights::Mamba3WeightsView;
use crate::mamba_ssm::gpu::blas::TypedPtr;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use cudarc::driver::PushKernelArg;
use std::rc::Rc;
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
    /// Typed activation twins for the bf16/f16 lane; `None` on the f32
    /// lane. The dtype map mirrors the trainer's mixed forward exactly -
    /// that is what makes prefill-vs-trainer parity bitwise per lane.
    typed: Option<Mamba3PrefillTypedScratch>,
    /// Compute dtype the executor was built for (`F32` = classic lane).
    dtype: WeightDtype,
    /// The shape every buffer above was sized for; `run` refuses any
    /// other dims — a longer window would silently index past the
    /// scratch allocations.
    sized_for: GpuMamba3Dims,
}

/// Output request for [`Mamba3Prefill::run_full`] - the m3 serve surface.
///
/// `full_temporal`, when present, receives the post-`norm_f` temporal for
/// ALL T positions (`[B * T * d_model]`) - the same buffer content the m3
/// trainer forward emits, bit for bit. `pooled_sum`, when present,
/// receives the per-sample COLUMN SUM over T (`[B * d_model]`): the
/// on-device colsum walks ascending t accumulating pure f32 adds from
/// 0.0, so `download / T` reproduces a CPU mean pool bit for bit while
/// the transfer drops from the full temporal to 1.5 KB per sample. At
/// `B > 1` each sample sums over its own rows in that same order, so a
/// sample's row does not depend on its batch neighbours.
pub struct Mamba3PrefillOutputs<'a> {
    /// `[B * d_model]` - final-timestep post-norm hidden state.
    pub last_hidden: &'a mut GpuBuffer,
    /// `[B * T * d_model]` - optional post-norm_f output for every position.
    pub full_temporal: Option<&'a mut GpuBuffer>,
    /// `[B * d_model]` - optional per-sample column sum of the
    /// post-norm_f temporal.
    pub pooled_sum: Option<&'a mut GpuBuffer>,
}

/// Everything a prefill launch needs besides the executor itself.
pub struct Mamba3PrefillRun<'a> {
    pub ctx: &'a GpuCtx,
    pub kernels: &'a Mamba3Kernels,
    pub dims: &'a GpuMamba3Dims,
    /// Either weights container through the one view surface; the
    /// container's `bulk_dtype()` must equal the executor's dtype.
    pub weights: &'a dyn Mamba3WeightsView,
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
        Self::new_with_dtype(stream, dims, WeightDtype::F32)
    }

    /// The dtype-selected constructor: `F32` is the classic lane
    /// (identical to [`Self::new`]); `Bf16`/`F16` add the typed
    /// activation twins and run the bulk projections through the typed
    /// batch-invariant GEMM route.
    pub fn new_with_dtype(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMamba3Dims,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        dims.validate_index_budget()?;
        let typed = match dtype {
            WeightDtype::F32 => None,
            _ => Some(Mamba3PrefillTypedScratch::new(stream, dims, dtype)?),
        };
        Ok(Self {
            tgt: GpuMamba3TargetScratch::new(stream, dims)?,
            chunk: Mamba3PrefillChunkScratch::new(stream, dims)?,
            typed,
            dtype,
            sized_for: *dims,
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
        self.run_full(
            run,
            states,
            Mamba3PrefillOutputs {
                last_hidden,
                full_temporal: None,
                pooled_sum: None,
            },
        )
    }

    /// [`Self::run`] with the all-T serve surface: identical kernel chain
    /// plus the optional full post-norm_f temporal and the on-device
    /// pooled column sum (see [`Mamba3PrefillOutputs`]).
    pub fn run_full(
        &mut self,
        run: &Mamba3PrefillRun<'_>,
        states: GpuMamba3StateBufs<'_>,
        outputs: Mamba3PrefillOutputs<'_>,
    ) -> Result<(), String> {
        let Mamba3PrefillOutputs {
            last_hidden,
            full_temporal,
            pooled_sum,
        } = outputs;
        let Mamba3PrefillRun {
            ctx,
            kernels: m3k,
            dims,
            weights,
            mamba_input,
            identity_proj,
            carry_state,
        } = *run;
        if *dims != self.sized_for {
            return Err(format!(
                "prefill executor was sized for {:?} but run was asked for {:?} —                  allocate a prefill for the shape you run",
                self.sized_for, dims
            ));
        }
        if weights.bulk_dtype() != self.dtype {
            return Err(format!(
                "prefill executor dtype {:?} != weights bulk dtype {:?} — \
                 build the executor with new_with_dtype for this container",
                self.dtype,
                weights.bulk_dtype()
            ));
        }
        let dtype = self.dtype;
        // The serve prefill has no backward: the trainer's raw-activation
        // saves (dd/trap raws, biased tensors, scale/gamma copies) take a
        // null pointer and the kernels skip those stores.
        const NULL_DEV: CUptr = 0;
        let tgt = &mut self.tgt;
        let ck = &mut self.chunk;
        let mut typed = self.typed.as_mut();
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
        let (input_proj_w, input_proj_b) = weights.input_proj();
        if identity_proj {
            tgt.temporal_work.copy_from_raw(mamba_input, &ctx.stream)?;
        } else if let Some(ts) = typed.as_deref_mut() {
            // Typed lane, mirroring the trainer's input_proj block: cast
            // the f32 input once, run the typed GEMM (bias folds in), and
            // upcast the projection into the f32 residual stream.
            let mid = dims.mamba_input_dim;
            {
                let n = (bt * mid) as i32;
                let cast = match dtype {
                    WeightDtype::Bf16 => &m3k.cast_f32_to_bf16,
                    WeightDtype::F16 => &m3k.cast_f32_to_f16,
                    WeightDtype::F32 => unreachable!("typed scratch exists only for half dtypes"),
                };
                let dst = ts.input_cast.cached_ptr();
                let src = mamba_input.cached_ptr();
                let mut b = ctx.stream.launch_builder(cast);
                b.arg(&dst);
                b.arg(&src);
                b.arg(&n);
                unsafe { b.launch(grid_1d(bt * mid)) }
                    .map_err(|e| format!("prefill input cast: {e:?}"))?;
            }
            crate::mamba_ssm::gpu::blas::gpu_gemm_typed_forward_raw(
                ctx,
                TypedPtr {
                    ptr: ts.out_flat.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: ts.input_cast.cached_ptr(),
                    dtype,
                },
                TypedPtr {
                    ptr: input_proj_w,
                    dtype,
                },
                Some(input_proj_b),
                (bt, dims.mamba_input_dim, dm),
            )?;
            {
                let n = (bt * dm) as i32;
                let cast = match dtype {
                    WeightDtype::Bf16 => &m3k.cast_bf16_to_f32,
                    WeightDtype::F16 => &m3k.cast_f16_to_f32,
                    WeightDtype::F32 => unreachable!("typed scratch exists only for half dtypes"),
                };
                let dst = tgt.temporal_work.cached_ptr();
                let src = ts.out_flat.cached_ptr();
                let mut b = ctx.stream.launch_builder(cast);
                b.arg(&dst);
                b.arg(&src);
                b.arg(&n);
                unsafe { b.launch(grid_1d(bt * dm)) }
                    .map_err(|e| format!("prefill input upcast: {e:?}"))?;
            }
        } else {
            crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                ctx,
                &mut tgt.temporal_work,
                mamba_input,
                input_proj_w,
                Some(input_proj_b),
                (bt, dims.mamba_input_dim, dm),
            )?;
        }

        let ssm_base = states.ssm.raw_ptr(&ctx.stream);
        let k_base = states.k.raw_ptr(&ctx.stream);
        let v_base = states.v.raw_ptr(&ctx.stream);
        let a_base = states.angle.raw_ptr(&ctx.stream);

        for l in 0..dims.n_layers {
            let lw = weights.layer(l);
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
                let nw = lw.norm_weight;
                if let Some(ts) = typed.as_deref_mut() {
                    let pn = ts.post_norm.cached_ptr();
                    let rms = tgt.rms_discard.cached_ptr();
                    let x = tgt.residual.cached_ptr();
                    let mut b = ctx
                        .stream
                        .launch_builder(m3k.rmsnorm_fwd_f32in_typed.get(dtype));
                    b.arg(&pn);
                    b.arg(&rms);
                    b.arg(&x);
                    b.arg(&nw);
                    b.arg(&bt_i);
                    b.arg(&dm_i);
                    b.arg(&eps);
                    unsafe { b.launch(grid_norm(bt, dm)) }
                        .map_err(|e| format!("prefill rmsnorm typed L{l}: {e:?}"))?;
                } else {
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
            }
            if let Some(ts) = typed.as_deref_mut() {
                crate::mamba_ssm::gpu::blas::gpu_gemm_typed_forward_raw(
                    ctx,
                    TypedPtr {
                        ptr: ts.proj_flat.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: ts.post_norm.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: lw.in_proj_w,
                        dtype,
                    },
                    None,
                    (bt, dm, ip),
                )?;
            } else {
                crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                    ctx,
                    &mut tgt.proj_flat,
                    &tgt.out_flat,
                    lw.in_proj_w,
                    None,
                    (bt, dm, ip),
                )?;
            }
            // 8-way split + fused activations. Typed lane: z/x/B/C typed,
            // all coefficient lanes stay f32 (the kernel signature's
            // contract, same as the trainer).
            {
                let n_i = bt as i32;
                let di_i = di as i32;
                let na_i = na as i32;
                let db = lw.dt_bias;
                if let Some(ts) = typed.as_deref_mut() {
                    let z: CUptr = 0;
                    let x = ts.x.cached_ptr();
                    let br: CUptr = 0;
                    let cr: CUptr = 0;
                    let proj = ts.proj_flat.cached_ptr();
                    let mut b = ctx.stream.launch_builder(m3k.m3_split_typed.get(dtype));
                    b.arg(&z);
                    b.arg(&x);
                    b.arg(&br);
                    b.arg(&cr);
                    b.arg(tgt.dt.inner_mut());
                    b.arg(tgt.a_val.inner_mut());
                    b.arg(tgt.trap.inner_mut());
                    b.arg(tgt.angles_raw.inner_mut());
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(&proj);
                    b.arg(&db);
                    b.arg(&dims.a_floor);
                    b.arg(&n_i);
                    b.arg(&di_i);
                    b.arg(&ng_i);
                    b.arg(&ds_i);
                    b.arg(&nh_i);
                    b.arg(&na_i);
                    unsafe { b.launch(grid_1d(bt * ip)) }
                        .map_err(|e| format!("prefill m3_split typed L{l}: {e:?}"))?;
                } else {
                    let mut b = ctx.stream.launch_builder(&m3k.m3_split);
                    b.arg(tgt.z.inner_mut());
                    b.arg(tgt.x.inner_mut());
                    b.arg(tgt.b_raw.inner_mut());
                    b.arg(tgt.c_raw.inner_mut());
                    b.arg(tgt.dt.inner_mut());
                    b.arg(tgt.a_val.inner_mut());
                    b.arg(tgt.trap.inner_mut());
                    b.arg(tgt.angles_raw.inner_mut());
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
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
            }
            // BCNorm + bias for B and C. Typed lane: the fused B+C kernel
            // (one launch, grid.y = 2), exactly the trainer's F4a/b.
            if let Some(ts) = typed.as_deref_mut() {
                let n_i = bt as i32;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((bt * ng) as u32, 2, 1),
                    block_dim: (ds as u32, 1, 1),
                    shared_mem_bytes: ds as u32 * 4,
                };
                let eps: f32 = dims.rms_norm_eps;
                let bn = ts.b_normed.cached_ptr();
                let cn = ts.c_normed.cached_ptr();
                // B and C live inside the projection at fixed column
                // offsets - read them in place instead of copying.
                let elt = dtype.size_bytes() as u64;
                let br = ts.proj_flat.cached_ptr() + (2 * di) as u64 * elt;
                let cr = ts.proj_flat.cached_ptr() + (2 * di + ng * ds) as u64 * elt;
                let bnw = lw.b_norm_weight;
                let cnw = lw.c_norm_weight;
                let mut b = ctx
                    .stream
                    .launch_builder(m3k.bcnorm_fwd_bc_typed.get(dtype));
                b.arg(&bn);
                b.arg(&cn);
                b.arg(tgt.b_rms.inner_mut());
                b.arg(tgt.c_rms.inner_mut());
                b.arg(&br);
                b.arg(&cr);
                b.arg(&bnw);
                b.arg(&cnw);
                b.arg(&n_i);
                b.arg(&ng_i);
                b.arg(&ds_i);
                b.arg(&eps);
                let src_stride = ip as i32;
                b.arg(&src_stride);
                unsafe { b.launch(cfg) }
                    .map_err(|e| format!("prefill bcnorm typed L{l}: {e:?}"))?;
            } else {
                for (out, rms, raw, w, tag) in [
                    (
                        &mut tgt.b_normed,
                        &mut tgt.b_rms,
                        &tgt.b_raw,
                        lw.b_norm_weight,
                        "B",
                    ),
                    (
                        &mut tgt.c_normed,
                        &mut tgt.c_rms,
                        &tgt.c_raw,
                        lw.c_norm_weight,
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
            }
            // Fused bias add (B + C) + RoPE: one launch, biased tensors
            // still materialize; n_angles == 0 passes through (replacing
            // the old copy branch).
            {
                let n_i = bt as i32;
                let na_i = na as i32;
                let bb_p = lw.b_bias;
                let cb_p = lw.c_bias;
                if let Some(ts) = typed.as_deref_mut() {
                    let bb: CUptr = 0;
                    let cb: CUptr = 0;
                    let k = ts.k.cached_ptr();
                    let q = ts.q.cached_ptr();
                    let bn = ts.b_normed.cached_ptr();
                    let cn = ts.c_normed.cached_ptr();
                    let mut b = ctx
                        .stream
                        .launch_builder(m3k.m3_bias_rope_fwd_typed.get(dtype));
                    b.arg(&bb);
                    b.arg(&cb);
                    b.arg(&k);
                    b.arg(&q);
                    b.arg(&bn);
                    b.arg(&cn);
                    b.arg(&bb_p);
                    b.arg(&cb_p);
                    b.arg(tgt.angle_cumsum.inner());
                    b.arg(&n_i);
                    b.arg(&nh_i);
                    b.arg(&ng_i);
                    b.arg(&ds_i);
                    b.arg(&na_i);
                    unsafe { b.launch(grid_1d(bt * nh * ds)) }
                        .map_err(|e| format!("prefill bias_rope typed L{l}: {e:?}"))?;
                } else {
                    let mut b = ctx.stream.launch_builder(&m3k.m3_bias_rope_fwd);
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(tgt.k.inner_mut());
                    b.arg(tgt.q.inner_mut());
                    b.arg(tgt.b_normed.inner());
                    b.arg(tgt.c_normed.inner());
                    b.arg(&bb_p);
                    b.arg(&cb_p);
                    b.arg(tgt.angle_cumsum.inner());
                    b.arg(&n_i);
                    b.arg(&nh_i);
                    b.arg(&ng_i);
                    b.arg(&ds_i);
                    b.arg(&na_i);
                    unsafe { b.launch(grid_1d(bt * nh * ds)) }
                        .map_err(|e| format!("prefill bias_rope L{l}: {e:?}"))?;
                }
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
            // Fused preprocess + chunk_state when the shape allows: K_scaled
            // crosses the seam through shared memory instead of L2. The
            // dA_cumsum launch moved ahead of it (it depends only on adt).
            if let Some(fcfg) =
                super::kernels::chunk_fused_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size())
            {
                if let Some(ts) = typed.as_deref_mut() {
                    let ks = ts.k_scaled.cached_ptr();
                    let kp = ts.k.cached_ptr();
                    let qp = ts.q.cached_ptr();
                    let xp = ts.x.cached_ptr();
                    let mut b = ctx
                        .stream
                        .launch_builder(m3k.m3_chunk_pre_state_fused_typed.get(dtype));
                    b.arg(&ks);
                    b.arg(ck.qk_dot.inner_mut());
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(ck.chunk_states.inner_mut());
                    b.arg(&kp);
                    b.arg(&qp);
                    b.arg(tgt.dt.inner());
                    b.arg(tgt.trap.inner());
                    b.arg(&xp);
                    b.arg(ck.da_cumsum.inner());
                    b.arg(&b_i);
                    b.arg(&t_i);
                    b.arg(&nh_i);
                    b.arg(&hd_i);
                    b.arg(&ds_i);
                    b.arg(&cs);
                    unsafe { b.launch(fcfg) }
                        .map_err(|e| format!("prefill fused pre+state typed L{l}: {e:?}"))?;
                } else {
                    let mut b = ctx
                        .stream
                        .launch_builder(&m3k.m3_chunk_pre_state_fused_typed.f32);
                    b.arg(ck.k_scaled.inner_mut());
                    b.arg(ck.qk_dot.inner_mut());
                    b.arg(&NULL_DEV);
                    b.arg(&NULL_DEV);
                    b.arg(ck.chunk_states.inner_mut());
                    b.arg(tgt.k.inner());
                    b.arg(tgt.q.inner());
                    b.arg(tgt.dt.inner());
                    b.arg(tgt.trap.inner());
                    b.arg(tgt.x.inner());
                    b.arg(ck.da_cumsum.inner());
                    b.arg(&b_i);
                    b.arg(&t_i);
                    b.arg(&nh_i);
                    b.arg(&hd_i);
                    b.arg(&ds_i);
                    b.arg(&cs);
                    unsafe { b.launch(fcfg) }
                        .map_err(|e| format!("prefill fused pre+state L{l}: {e:?}"))?;
                }
            } else {
                {
                    let cfg = cudarc::driver::LaunchConfig {
                        grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                        block_dim: (dims.chunk_size() as u32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    if let Some(ts) = typed.as_deref_mut() {
                        let ks = ts.k_scaled.cached_ptr();
                        let kp = ts.k.cached_ptr();
                        let qp = ts.q.cached_ptr();
                        let mut b = ctx
                            .stream
                            .launch_builder(m3k.m3_preprocess_chunks_typed.get(dtype));
                        b.arg(&ks);
                        b.arg(ck.qk_dot.inner_mut());
                        b.arg(&NULL_DEV);
                        b.arg(&NULL_DEV);
                        b.arg(&kp);
                        b.arg(&qp);
                        b.arg(tgt.dt.inner());
                        b.arg(tgt.trap.inner());
                        b.arg(&b_i);
                        b.arg(&t_i);
                        b.arg(&nh_i);
                        b.arg(&ds_i);
                        b.arg(&cs);
                        unsafe { b.launch(cfg) }
                            .map_err(|e| format!("prefill preprocess typed L{l}: {e:?}"))?;
                    } else {
                        let mut b = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
                        b.arg(ck.k_scaled.inner_mut());
                        b.arg(ck.qk_dot.inner_mut());
                        b.arg(&NULL_DEV);
                        b.arg(&NULL_DEV);
                        b.arg(tgt.k.inner());
                        b.arg(tgt.q.inner());
                        b.arg(tgt.dt.inner());
                        b.arg(tgt.trap.inner());
                        b.arg(&b_i);
                        b.arg(&t_i);
                        b.arg(&nh_i);
                        b.arg(&ds_i);
                        b.arg(&cs);
                        unsafe { b.launch(cfg) }
                            .map_err(|e| format!("prefill preprocess L{l}: {e:?}"))?;
                    }
                }
                {
                    let cfg = super::kernels::chunk_state_cfg(
                        dims.batch,
                        nc,
                        nh,
                        hd,
                        ds,
                        dims.chunk_size(),
                    );
                    if let Some(ts) = typed.as_deref_mut() {
                        let xp = ts.x.cached_ptr();
                        let ks = ts.k_scaled.cached_ptr();
                        let mut b = ctx
                            .stream
                            .launch_builder(m3k.m3_chunk_state_fwd_typed.get(dtype));
                        b.arg(ck.chunk_states.inner_mut());
                        b.arg(&xp);
                        b.arg(&ks);
                        b.arg(ck.da_cumsum.inner());
                        b.arg(&b_i);
                        b.arg(&t_i);
                        b.arg(&nh_i);
                        b.arg(&hd_i);
                        b.arg(&ds_i);
                        b.arg(&cs);
                        unsafe { b.launch(cfg) }
                            .map_err(|e| format!("prefill chunk_state typed L{l}: {e:?}"))?;
                    } else {
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
                        unsafe { b.launch(cfg) }
                            .map_err(|e| format!("prefill chunk_state L{l}: {e:?}"))?;
                    }
                }
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
                let dp = lw.d_param;
                let (coop, cfg) =
                    super::kernels::chunk_scan_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size());
                if let Some(ts) = typed.as_deref_mut() {
                    let kern = if coop {
                        m3k.m3_chunk_scan_fwd_coop_typed.get(dtype)
                    } else {
                        m3k.m3_chunk_scan_fwd_typed.get(dtype)
                    };
                    let yp = ts.y.cached_ptr();
                    let xp = ts.x.cached_ptr();
                    let qp = ts.q.cached_ptr();
                    let ks = ts.k_scaled.cached_ptr();
                    let mut b = ctx.stream.launch_builder(kern);
                    b.arg(&yp);
                    b.arg(&xp);
                    b.arg(&qp);
                    b.arg(&ks);
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
                    unsafe { b.launch(cfg) }
                        .map_err(|e| format!("prefill chunk_scan typed L{l}: {e:?}"))?;
                } else {
                    let kern = if coop {
                        &m3k.m3_chunk_scan_fwd_coop
                    } else {
                        &m3k.m3_chunk_scan_fwd
                    };
                    let mut b = ctx.stream.launch_builder(kern);
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
                    unsafe { b.launch(cfg) }
                        .map_err(|e| format!("prefill chunk_scan L{l}: {e:?}"))?;
                }
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
                if let Some(ts) = typed.as_deref_mut() {
                    let kp = ts.k.cached_ptr();
                    let xp = ts.x.cached_ptr();
                    let mut b = ctx
                        .stream
                        .launch_builder(m3k.m3_writeback_parallel_states_typed.get(dtype));
                    b.arg(&ssm_ptr);
                    b.arg(&k_ptr);
                    b.arg(&v_ptr);
                    b.arg(ck.final_states.inner());
                    b.arg(&kp);
                    b.arg(&xp);
                    b.arg(&b_i);
                    b.arg(&t_i);
                    b.arg(&nh_i);
                    b.arg(&hd_i);
                    b.arg(&ds_i);
                    unsafe { b.launch(cfg) }
                        .map_err(|e| format!("prefill writeback typed L{l}: {e:?}"))?;
                } else {
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
                    unsafe { b.launch(cfg) }
                        .map_err(|e| format!("prefill writeback L{l}: {e:?}"))?;
                }
            }
            // Output gate + out_proj + residual.
            if dims.is_outproj_norm {
                assert!(
                    di <= 1024,
                    "d_inner ({di}) exceeds rmsnorm_gated shared memory limit"
                );
                let nw = lw.norm_gate_weight;
                let bt_i = bt as i32;
                let di_i = di as i32;
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (bt as u32, 1, 1),
                    block_dim: (di as u32, 1, 1),
                    shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
                };
                let eps: f32 = dims.rms_norm_eps;
                if let Some(ts) = typed.as_deref_mut() {
                    let g = ts.gated.cached_ptr();
                    let y = ts.y.cached_ptr();
                    // z is the projection's first column block - read it
                    // in place with the projection's row stride.
                    let z = ts.proj_flat.cached_ptr();
                    let mut b = ctx
                        .stream
                        .launch_builder(m3k.rmsnorm_gated_fwd_typed.get(dtype));
                    b.arg(&g);
                    b.arg(tgt.rms_discard.inner_mut());
                    b.arg(&y);
                    b.arg(&z);
                    b.arg(&nw);
                    b.arg(&bt_i);
                    b.arg(&di_i);
                    b.arg(&hd_i);
                    b.arg(&eps);
                    let z_stride = ip as i32;
                    b.arg(&z_stride);
                    unsafe { b.launch(grid) }
                        .map_err(|e| format!("prefill gated typed L{l}: {e:?}"))?;
                } else {
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
                }
            } else if let Some(ts) = typed.as_deref_mut() {
                let n = (bt * di) as i32;
                let g = ts.gated.cached_ptr();
                let y = ts.y.cached_ptr();
                let z = ts.proj_flat.cached_ptr();
                let mut b = ctx
                    .stream
                    .launch_builder(m3k.silu_gate_fwd_typed.get(dtype));
                b.arg(&g);
                b.arg(&y);
                b.arg(&z);
                b.arg(&n);
                let di_i2 = di as i32;
                let z_stride = ip as i32;
                b.arg(&di_i2);
                b.arg(&z_stride);
                unsafe { b.launch(grid_1d(bt * di)) }
                    .map_err(|e| format!("prefill silu typed L{l}: {e:?}"))?;
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
            if let Some(ts) = typed.as_deref_mut() {
                crate::mamba_ssm::gpu::blas::gpu_gemm_typed_forward_raw(
                    ctx,
                    TypedPtr {
                        ptr: ts.out_flat.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: ts.gated.cached_ptr(),
                        dtype,
                    },
                    TypedPtr {
                        ptr: lw.out_proj_w,
                        dtype,
                    },
                    None,
                    (bt, di, dm),
                )?;
                // Typed branch output + f32 residual -> f32 temporal (the
                // trainer's residual_add_f32_typed; note the arg order:
                // dst, f32 residual, typed branch).
                let ne = (bt * dm) as i32;
                let dst = tgt.temporal_work.cached_ptr();
                let a = tgt.residual.cached_ptr();
                let bb = ts.out_flat.cached_ptr();
                let mut b = ctx
                    .stream
                    .launch_builder(m3k.residual_add_f32_typed.get(dtype));
                b.arg(&dst);
                b.arg(&a);
                b.arg(&bb);
                b.arg(&ne);
                unsafe { b.launch(grid_1d(bt * dm)) }
                    .map_err(|e| format!("prefill residual typed L{l}: {e:?}"))?;
            } else {
                crate::mamba_ssm::gpu::blas::gpu_sgemm_forward_raw(
                    ctx,
                    &mut tgt.out_flat,
                    &tgt.gated,
                    lw.out_proj_w,
                    None,
                    (bt, di, dm),
                )?;
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
            let nf = weights.norm_f();
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
        if let Some(full) = full_temporal {
            if full.len() != bt * dm {
                return Err(format!(
                    "m3 prefill outputs: full_temporal len {} != batch*seq_len*d_model = {}",
                    full.len(),
                    bt * dm
                ));
            }
            full.copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        }
        if let Some(pooled) = pooled_sum {
            if pooled.len() != dims.batch * dm {
                return Err(format!(
                    "m3 prefill outputs: pooled_sum len {} != batch*d_model = {}",
                    pooled.len(),
                    dims.batch * dm
                ));
            }
            let cols = dm as i32;
            if dims.batch == 1 {
                // colsum_accumulate does db[j] += sum - zero first so the
                // result is exactly the sum over T.
                pooled.zero(&ctx.stream)?;
                let rows = bt as i32;
                let mut b = ctx.stream.launch_builder(&m3k.colsum_accumulate);
                b.arg(pooled.inner_mut());
                b.arg(tgt.temporal_work.inner());
                b.arg(&rows);
                b.arg(&cols);
                unsafe { b.launch(grid_1d(dm)) }
                    .map_err(|e| format!("m3 prefill pooled colsum: {e:?}"))?;
            } else {
                // Segmented: each sample sums over its OWN seq_len rows,
                // in the same ascending-t f32 order the batch=1 kernel
                // uses, so a sample's pooled row is bit-identical whether
                // it rode alone or inside a batch.
                let segments = dims.batch as i32;
                let seg_len = dims.seq_len as i32;
                const BLOCK: u32 = 256;
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dm as u32).div_ceil(BLOCK), dims.batch as u32, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = ctx.stream.launch_builder(&m3k.colsum_segments);
                b.arg(pooled.inner_mut());
                b.arg(tgt.temporal_work.inner());
                b.arg(&segments);
                b.arg(&seg_len);
                b.arg(&cols);
                unsafe { b.launch(cfg) }
                    .map_err(|e| format!("m3 prefill pooled colsum (batched): {e:?}"))?;
            }
        }
        Ok(())
    }
}

/// A captured CUDA graph of one prefill window over fixed buffers. Replay
/// requires the complete GEMM route captured with the graph.
pub struct Mamba3PrefillGraph {
    module_identity: String,
    graph: cudarc::driver::CudaGraph,
    ctx_resources: Rc<crate::mamba_ssm::gpu::context::GpuCtxResources>,
    _m3_modules: crate::mamba_ssm::gpu::kernels::CudaModuleAnchors,
    flags_at_capture: crate::mamba_ssm::gpu::context::GemmRoute,
    captured_ctx_token: u64,
    captured_stream_token: usize,
    captured_half_staging_ptr: u64,
    captured_bi_upcast_ptrs: [u64; 3],
    input_ptr: CUptr,
    ssm_ptr: CUptr,
    k_ptr: CUptr,
    v_ptr: CUptr,
    angle_ptr: CUptr,
    last_hidden_ptr: CUptr,
    weights_arenas: (u64, u64),
    weights_dtype: WeightDtype,
}

impl Mamba3PrefillGraph {
    /// Capture the window over the given fixed input/state/output buffers.
    /// Upload fresh bytes into the SAME input buffer before each replay.
    /// # Safety
    ///
    /// The original context, stream, cuBLAS handle, both kernel registries,
    /// weights, input, state, scratch, and output allocations must remain
    /// unchanged until this holder is destroyed and every replay has completed.
    /// Pointer checks are diagnostics and do not extend any CUDA lifetime.
    pub unsafe fn capture(
        prefill: &mut Mamba3Prefill,
        run: &Mamba3PrefillRun<'_>,
        mut states: GpuMamba3StateBufs<'_>,
        last_hidden: &mut GpuBuffer,
    ) -> Result<Self, String> {
        run.ctx.presize_bi_scratch()?;
        run.ctx
            .presize_mixed_graph_scratch_m3(run.dims, run.weights.bulk_dtype())?;
        let flags_at_capture = run.ctx.gemm_route();
        let input_ptr = run.mamba_input.cached_ptr();
        let ssm_ptr = states.ssm.cached_ptr();
        let k_ptr = states.k.cached_ptr();
        let v_ptr = states.v.cached_ptr();
        let angle_ptr = states.angle.cached_ptr();
        let last_hidden_ptr = last_hidden.cached_ptr();
        let weights_arenas = run.weights.arena_identity();
        let weights_dtype = run.weights.bulk_dtype();
        let module_identity = run.kernels.module_identity.clone();
        if weights_dtype != WeightDtype::F32 {
            run.ctx.freeze_graph_scratch();
        }
        let graph = unsafe {
            crate::mamba_ssm::gpu::graph_capture::capture_into_graph(&run.ctx.stream, || {
                prefill.run(run, states.reborrow(), last_hidden)
            })
        }?;
        graph
            .upload()
            .map_err(|e| format!("prefill graph upload: {e:?}"))?;
        run.ctx.note_graph_capture();
        Ok(Self {
            graph,
            ctx_resources: run.ctx.resource_anchor(),
            _m3_modules: run.kernels.module_anchors(),
            flags_at_capture,
            captured_ctx_token: run.ctx.instance_token(),
            captured_stream_token: run.ctx.stream_token(),
            captured_half_staging_ptr: run.ctx.half_staging_ptr(),
            captured_bi_upcast_ptrs: run.ctx.bi_upcast_scratch_ptrs(),
            input_ptr,
            ssm_ptr,
            k_ptr,
            v_ptr,
            angle_ptr,
            last_hidden_ptr,
            weights_arenas,
            weights_dtype,
            module_identity,
        })
    }

    /// Replay the captured window. `mamba_input` and ALL FOUR state
    /// buffers must be the SAME allocations the capture saw (the graph
    /// baked every device pointer in; a different buffer would leave
    /// the graph writing the old allocation while the caller reads the
    /// new one).
    pub fn replay(
        &self,
        ctx: &GpuCtx,
        kernels: &Mamba3Kernels,
        weights: &dyn Mamba3WeightsView,
        mamba_input: &GpuBuffer,
        states: &GpuMamba3StateBufs<'_>,
        last_hidden: &GpuBuffer,
    ) -> Result<(), String> {
        if ctx.instance_token() != self.captured_ctx_token {
            return Err(
                "prefill graph replay refused: GpuCtx differs from capture; re-capture instead"
                    .into(),
            );
        }
        if ctx.stream_token() != self.captured_stream_token {
            return Err(
                "prefill graph replay refused: GpuCtx stream differs from capture; \
                 re-capture instead"
                    .into(),
            );
        }
        if kernels.module_identity != self.module_identity {
            return Err(
                "prefill graph replay refused: the kernels module differs from \
                 the captured compile - the graph would run stale kernels"
                    .to_string(),
            );
        }
        if weights.arena_identity() != self.weights_arenas
            || weights.bulk_dtype() != self.weights_dtype
        {
            return Err(
                "prefill graph replay refused: weights container (or its dtype) \
                 differs from the captured one - the graph would silently run \
                 the captured weights"
                    .to_string(),
            );
        }
        if ctx.gemm_route() != self.flags_at_capture {
            return Err(format!(
                "prefill graph replay refused: GEMM route changed since capture \
                 (captured {:?}, now {:?})",
                self.flags_at_capture,
                ctx.gemm_route()
            ));
        }
        if self.weights_dtype != WeightDtype::F32 {
            ctx.ensure_graph_scratch_ptrs(
                self.captured_half_staging_ptr,
                self.captured_bi_upcast_ptrs,
                "prefill graph replay",
            )?;
        }
        if mamba_input.cached_ptr() != self.input_ptr
            || states.ssm.cached_ptr() != self.ssm_ptr
            || states.k.cached_ptr() != self.k_ptr
            || states.v.cached_ptr() != self.v_ptr
            || states.angle.cached_ptr() != self.angle_ptr
            || last_hidden.cached_ptr() != self.last_hidden_ptr
        {
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

impl Drop for Mamba3PrefillGraph {
    fn drop(&mut self) {
        let _ = self.ctx_resources.stream.synchronize();
    }
}

/// A captured CUDA graph of one POOLED prefill window over fixed buffers -
/// the m3 classify serve shape: state reset (carry_state = false zeroes
/// inside the capture, so every replay starts a fresh page) + the full
/// layer chain + norm_f + the on-device pooled column sum. Mirrors the M1
/// `PrefillPooledGraph` contract: upload the next page into the SAME
/// input buffer, replay, download the 1.5 KB pooled sum, divide by T on
/// the host.
pub struct Mamba3PrefillPooledGraph {
    module_identity: String,
    graph: cudarc::driver::CudaGraph,
    ctx_resources: Rc<crate::mamba_ssm::gpu::context::GpuCtxResources>,
    _m3_modules: crate::mamba_ssm::gpu::kernels::CudaModuleAnchors,
    flags_at_capture: crate::mamba_ssm::gpu::context::GemmRoute,
    captured_ctx_token: u64,
    captured_stream_token: usize,
    captured_half_staging_ptr: u64,
    captured_bi_upcast_ptrs: [u64; 3],
    input_ptr: CUptr,
    ssm_ptr: CUptr,
    k_ptr: CUptr,
    v_ptr: CUptr,
    angle_ptr: CUptr,
    pooled_ptr: CUptr,
    weights_arenas: (u64, u64),
    weights_dtype: WeightDtype,
}

impl Mamba3PrefillPooledGraph {
    /// Capture the pooled window. `run.carry_state` must be `false` -
    /// the state reset has to live INSIDE the graph for replays to score
    /// independent pages.
    /// # Safety
    ///
    /// The original context, stream, cuBLAS handle, both kernel registries,
    /// weights, input, state, scratch, last-hidden, and pooled allocations must
    /// remain unchanged until this holder is destroyed and every replay has
    /// completed. Pointer checks are diagnostics and do not extend lifetimes.
    pub unsafe fn capture(
        prefill: &mut Mamba3Prefill,
        run: &Mamba3PrefillRun<'_>,
        mut states: GpuMamba3StateBufs<'_>,
        last_hidden: &mut GpuBuffer,
        pooled_sum: &mut GpuBuffer,
    ) -> Result<Self, String> {
        if run.carry_state {
            return Err(
                "m3 pooled prefill graph: carry_state must be false - the state \
                 reset must be captured so each replay scores a fresh page"
                    .to_string(),
            );
        }
        run.ctx.presize_bi_scratch()?;
        run.ctx
            .presize_mixed_graph_scratch_m3(run.dims, run.weights.bulk_dtype())?;
        let flags_at_capture = run.ctx.gemm_route();
        let input_ptr = run.mamba_input.cached_ptr();
        let ssm_ptr = states.ssm.cached_ptr();
        let k_ptr = states.k.cached_ptr();
        let v_ptr = states.v.cached_ptr();
        let angle_ptr = states.angle.cached_ptr();
        let pooled_ptr = pooled_sum.cached_ptr();
        let weights_arenas = run.weights.arena_identity();
        let weights_dtype = run.weights.bulk_dtype();
        let module_identity = run.kernels.module_identity.clone();
        if weights_dtype != WeightDtype::F32 {
            run.ctx.freeze_graph_scratch();
        }
        let graph = unsafe {
            crate::mamba_ssm::gpu::graph_capture::capture_into_graph(&run.ctx.stream, || {
                prefill.run_full(
                    run,
                    states.reborrow(),
                    Mamba3PrefillOutputs {
                        last_hidden,
                        full_temporal: None,
                        pooled_sum: Some(pooled_sum),
                    },
                )
            })
        }?;
        graph
            .upload()
            .map_err(|e| format!("m3 pooled prefill graph upload: {e:?}"))?;
        run.ctx.note_graph_capture();
        Ok(Self {
            graph,
            ctx_resources: run.ctx.resource_anchor(),
            _m3_modules: run.kernels.module_anchors(),
            flags_at_capture,
            captured_ctx_token: run.ctx.instance_token(),
            captured_stream_token: run.ctx.stream_token(),
            captured_half_staging_ptr: run.ctx.half_staging_ptr(),
            captured_bi_upcast_ptrs: run.ctx.bi_upcast_scratch_ptrs(),
            input_ptr,
            ssm_ptr,
            k_ptr,
            v_ptr,
            angle_ptr,
            pooled_ptr,
            weights_arenas,
            weights_dtype,
            module_identity,
        })
    }

    /// Replay one page: same fixed buffers as capture, checked.
    pub fn replay(
        &self,
        ctx: &GpuCtx,
        kernels: &Mamba3Kernels,
        weights: &dyn Mamba3WeightsView,
        mamba_input: &GpuBuffer,
        states: &GpuMamba3StateBufs<'_>,
        pooled_sum: &GpuBuffer,
    ) -> Result<(), String> {
        if ctx.instance_token() != self.captured_ctx_token {
            return Err(
                "m3 pooled prefill graph replay refused: GpuCtx differs from capture; \
                 re-capture instead"
                    .into(),
            );
        }
        if ctx.stream_token() != self.captured_stream_token {
            return Err(
                "m3 pooled prefill graph replay refused: GpuCtx stream differs from capture; \
                 re-capture instead"
                    .into(),
            );
        }
        if kernels.module_identity != self.module_identity {
            return Err(
                "m3 pooled graph replay refused: the kernels module differs \
                  from the captured compile"
                    .to_string(),
            );
        }
        if weights.arena_identity() != self.weights_arenas
            || weights.bulk_dtype() != self.weights_dtype
        {
            return Err(
                "m3 pooled prefill graph replay refused: weights container (or its \
                 dtype) differs from the captured one"
                    .to_string(),
            );
        }
        if ctx.gemm_route() != self.flags_at_capture {
            return Err(format!(
                "m3 pooled prefill graph replay refused: GEMM route changed \
                 since capture (captured {:?}, now {:?})",
                self.flags_at_capture,
                ctx.gemm_route()
            ));
        }
        if self.weights_dtype != WeightDtype::F32 {
            ctx.ensure_graph_scratch_ptrs(
                self.captured_half_staging_ptr,
                self.captured_bi_upcast_ptrs,
                "m3 pooled prefill graph replay",
            )?;
        }
        if mamba_input.cached_ptr() != self.input_ptr
            || states.ssm.cached_ptr() != self.ssm_ptr
            || states.k.cached_ptr() != self.k_ptr
            || states.v.cached_ptr() != self.v_ptr
            || states.angle.cached_ptr() != self.angle_ptr
            || pooled_sum.cached_ptr() != self.pooled_ptr
        {
            return Err(
                "m3 pooled prefill graph replay refused: input/state/pooled buffers \
                 differ from the captured allocations"
                    .to_string(),
            );
        }
        self.graph
            .launch()
            .map_err(|e| format!("m3 pooled prefill graph launch: {e:?}"))
    }
}

impl Drop for Mamba3PrefillPooledGraph {
    fn drop(&mut self) {
        let _ = self.ctx_resources.stream.synchronize();
    }
}
