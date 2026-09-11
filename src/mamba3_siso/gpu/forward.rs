//! Mamba-3 SISO GPU **forward** pass (training + target burn-in).
//!
//! Split from the former 2313-line `mamba3_gpu.rs`. 8-phase
//! pipeline:
//!   F1: RMSNorm
//!   F2: in_proj SGEMM
//!   F3: m3_split (8-way + fused activations)
//!   F4: BCNorm + bias + RoPE on B,C
//!   F5: angle_dt accumulation → trapezoidal (alpha/beta/gamma)
//!   F6: SSM recurrence (sequential or chunked parallel)
//!   F7: RMSNormGated(y, z) or y*SiLU(z)
//!   F8: out_proj SGEMM + residual
//!
//! For mixed-precision (bf16/f16) forward see [`super::forward_mixed`].

use super::kernels::Mamba3Kernels;
use super::state::{
    CHUNK_SIZE, GpuMamba3BackboneActs, GpuMamba3LayerActs, GpuMamba3Scratch, GpuMamba3StateBufs,
    GpuMamba3TargetScratch, M3Exec, Mamba3LayerPtrs,
};
use super::weights::{GpuMamba3LayerWeights, GpuMamba3Weights};
use crate::mamba_ssm::gpu::blas::gpu_gemm_bi_forward_raw;
use crate::mamba_ssm::gpu::buffers::{GpuBuffer, GpuByteBuffer};
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use cudarc::driver::PushKernelArg;

/// Logical dimensions for chunk-parallel angle accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AngleChunkedShape {
    pub batch: usize,
    pub seq_len: usize,
    pub heads: usize,
    pub angles: usize,
}

/// Buffers consumed by one chunk-parallel angle launch.
pub struct AngleChunkedFwd<'a> {
    pub angle_cumsum: &'a mut GpuBuffer,
    pub angle_state_ptr: cudarc::driver::sys::CUdeviceptr,
    pub angles_raw: &'a GpuBuffer,
    pub dt: &'a GpuBuffer,
    pub sums: &'a GpuByteBuffer,
    pub carries: &'a GpuByteBuffer,
    pub shape: AngleChunkedShape,
}

#[derive(Clone, Copy)]
struct AngleChunkedLengths {
    angle_cumsum: usize,
    angles_raw: usize,
    dt: usize,
    sums_bytes: usize,
    carries_bytes: usize,
}

#[derive(Debug)]
struct ValidatedAngleChunkedLaunch {
    batch: i32,
    seq_len: i32,
    heads: i32,
    angles: i32,
    chunk_size: i32,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    snapshot_bytes: usize,
}

fn checked_angle_product(label: &str, factors: &[usize]) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("angle {label} overflows usize"))
    })
}

fn angle_i32(label: &str, value: usize) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("angle {label} does not fit i32: {value}"))
}

fn require_angle_length(label: &str, actual: usize, required: usize) -> Result<(), String> {
    if actual < required {
        return Err(format!(
            "angle {label} is undersized: need {required}, got {actual}"
        ));
    }
    Ok(())
}

fn validate_angle_chunked_layout(
    shape: AngleChunkedShape,
    lengths: AngleChunkedLengths,
    angle_state_ptr: cudarc::driver::sys::CUdeviceptr,
) -> Result<ValidatedAngleChunkedLaunch, String> {
    let AngleChunkedShape {
        batch,
        seq_len,
        heads,
        angles,
    } = shape;
    for (label, value) in [
        ("batch", batch),
        ("sequence length", seq_len),
        ("head count", heads),
        ("angle count", angles),
    ] {
        if value == 0 {
            return Err(format!("angle {label} must be non-zero"));
        }
    }
    if angle_state_ptr == 0 {
        return Err("angle state pointer is null".into());
    }

    let padded_seq_len = seq_len
        .checked_add(CHUNK_SIZE - 1)
        .ok_or_else(|| "angle padded sequence length overflows usize".to_string())?;
    let _ = angle_i32("padded sequence length", padded_seq_len)?;
    let n_chunks = padded_seq_len / CHUNK_SIZE;
    let batch_time = checked_angle_product("batch-time extent", &[batch, seq_len])?;
    let lanes = checked_angle_product("lane extent", &[heads, angles])?;
    let output_elements = checked_angle_product("output extent", &[batch_time, lanes])?;
    let angle_elements = checked_angle_product("raw-angle extent", &[batch_time, angles])?;
    let dt_elements = checked_angle_product("dt extent", &[batch_time, heads])?;
    let sum_elements =
        checked_angle_product("chunk-sum extent", &[batch, n_chunks, heads, angles])?;
    let state_elements = checked_angle_product("state extent", &[batch, heads, angles])?;
    let sums_bytes = checked_angle_product(
        "chunk-sum byte extent",
        &[sum_elements, std::mem::size_of::<f64>()],
    )?;
    let snapshot_bytes = checked_angle_product(
        "state snapshot byte extent",
        &[state_elements, std::mem::size_of::<f32>()],
    )?;

    let batch_i = angle_i32("batch", batch)?;
    let seq_len_i = angle_i32("sequence length", seq_len)?;
    let heads_i = angle_i32("head count", heads)?;
    let angles_i = angle_i32("angle count", angles)?;
    let chunk_size_i = angle_i32("chunk size", CHUNK_SIZE)?;
    for (label, extent) in [
        ("batch-time extent", batch_time),
        ("lane extent", lanes),
        ("output extent", output_elements),
        ("raw-angle extent", angle_elements),
        ("dt extent", dt_elements),
        ("chunk-sum extent", sum_elements),
        ("state extent", state_elements),
    ] {
        let _ = angle_i32(label, extent)?;
    }

    require_angle_length("angle_cumsum", lengths.angle_cumsum, output_elements)?;
    require_angle_length("angles_raw", lengths.angles_raw, angle_elements)?;
    require_angle_length("dt", lengths.dt, dt_elements)?;
    require_angle_length("sums", lengths.sums_bytes, sums_bytes)?;
    require_angle_length("carries", lengths.carries_bytes, snapshot_bytes)?;

    let grid_x = u32::try_from(checked_angle_product("grid-x extent", &[batch, n_chunks])?)
        .map_err(|_| "angle grid-x extent does not fit u32".to_string())?;
    let grid_y = u32::try_from(lanes.div_ceil(256))
        .map_err(|_| "angle grid-y extent does not fit u32".to_string())?;
    if grid_y > u16::MAX.into() {
        return Err(format!("angle grid-y extent exceeds CUDA limit: {grid_y}"));
    }
    let block_x = u32::try_from(lanes.min(256))
        .map_err(|_| "angle block-x extent does not fit u32".to_string())?;

    Ok(ValidatedAngleChunkedLaunch {
        batch: batch_i,
        seq_len: seq_len_i,
        heads: heads_i,
        angles: angles_i,
        chunk_size: chunk_size_i,
        grid_x,
        grid_y,
        block_x,
        snapshot_bytes,
    })
}

/// Chunk-parallel angle accumulation: per-chunk fp64 delta sums, a
/// serial carry chain per (batch, head, angle) seeded from the
/// persistent angle state, then a per-chunk re-walk writing the f32
/// cumsum and the exit state. Deterministic by construction (fixed
/// association, one writer per cell, no atomics); replaces the
/// sequential single-thread-per-lane kernel whose dependent fp64 chain
/// dominates multi-chunk windows.
///
/// # Safety
///
/// Every request buffer and `request.angle_state_ptr` must belong to
/// `ctx`'s CUDA context. The state pointer must name readable and
/// writable storage for `batch * heads * angles` contiguous `f32`
/// values and must not overlap any request buffer. All allocations must
/// remain alive and must not be accessed from another stream until this
/// stream has completed the enqueued work.
pub unsafe fn gpu_angle_chunked_fwd(
    ctx: &GpuCtx,
    m3k: &Mamba3Kernels,
    request: AngleChunkedFwd<'_>,
) -> Result<(), String> {
    use cudarc::driver::PushKernelArg;
    let AngleChunkedFwd {
        angle_cumsum,
        angle_state_ptr,
        angles_raw,
        dt,
        sums,
        carries,
        shape:
            AngleChunkedShape {
                batch,
                seq_len,
                heads: nh,
                angles: na,
            },
    } = request;
    let launch = validate_angle_chunked_layout(
        AngleChunkedShape {
            batch,
            seq_len,
            heads: nh,
            angles: na,
        },
        AngleChunkedLengths {
            angle_cumsum: angle_cumsum.len(),
            angles_raw: angles_raw.len(),
            dt: dt.len(),
            sums_bytes: sums.len_bytes(),
            carries_bytes: carries.len_bytes(),
        },
        angle_state_ptr,
    )?;
    let b_i = launch.batch;
    let t_i = launch.seq_len;
    let nh_i = launch.heads;
    let na_i = launch.angles;
    let cs_i = launch.chunk_size;
    let sums_ptr = sums.cached_ptr();
    // The carries buffer is repurposed as the ENTRY-STATE SNAPSHOT: the
    // apply kernel folds the per-chunk carry chain inline (bit-identical
    // ascending walk over the same stored doubles), and it reads the
    // entering state from this copy because the last chunk's block
    // writes the exit state into angle_state concurrently.
    let snapshot_ptr = carries.cached_ptr();
    let rc = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            snapshot_ptr,
            angle_state_ptr,
            launch.snapshot_bytes,
            ctx.stream.cu_stream(),
        )
    };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("angle entry-state snapshot: {rc:?}"));
    }
    {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_angle_chunk_sums);
        bld.arg(&sums_ptr);
        bld.arg(angles_raw.inner());
        bld.arg(dt.inner());
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&nh_i);
        bld.arg(&na_i);
        bld.arg(&cs_i);
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (launch.grid_x, launch.grid_y, 1),
            block_dim: (launch.block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { bld.launch(grid) }.map_err(|e| format!("angle chunk sums: {e:?}"))?;
    }
    {
        let mut bld = ctx.stream.launch_builder(&m3k.m3_angle_chunk_apply);
        bld.arg(angle_cumsum.inner_mut());
        bld.arg(&angle_state_ptr);
        bld.arg(&snapshot_ptr);
        bld.arg(&sums_ptr);
        bld.arg(angles_raw.inner());
        bld.arg(dt.inner());
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&nh_i);
        bld.arg(&na_i);
        bld.arg(&cs_i);
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (launch.grid_x, launch.grid_y, 1),
            block_dim: (launch.block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { bld.launch(grid) }.map_err(|e| format!("angle chunk apply: {e:?}"))?;
    }
    Ok(())
}

/// Picks the kernel for `adt = a_val * dt` over `n` f32 elements: the
/// 16-byte vectorized twin when the count divides four and every operand
/// pointer is aligned, else the scalar kernel. Both multiply the same
/// elements in the same order. Returns the kernel and its launch count.
pub(super) fn adt_multiply_launch<'k>(
    m3k: &'k Mamba3Kernels,
    n: usize,
    ptrs: &[cudarc::driver::sys::CUdeviceptr],
) -> (&'k cudarc::driver::CudaFunction, usize) {
    let w = crate::mamba_ssm::gpu::launch::vec8_width(4);
    if crate::mamba_ssm::gpu::launch::vec8_ok(n, 4, ptrs) {
        (&m3k.elementwise_mul_v, n / w)
    } else {
        (&m3k.elementwise_mul, n)
    }
}

pub fn gpu_forward_mamba3_layer(
    exec: &M3Exec<'_>,
    stream_out: cudarc::driver::sys::CUdeviceptr,
    acts: &mut GpuMamba3LayerActs,
    lw: &GpuMamba3LayerWeights,
    layer_ptrs: &Mamba3LayerPtrs,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    let na = dims.n_angles;

    // F1: RMSNorm - the residual stream already lives in acts.residual:
    // the previous layer's residual_add wrote it there directly (layer 0
    // is seeded by the driver). Same next-layer-residual plumbing as the
    // M1 forward - the per-layer temporal round trip is gone.
    {
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(acts.post_norm.inner_mut());
        builder.arg(acts.rms_vals.inner_mut());
        builder.arg(acts.residual.inner());
        let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
        builder.arg(&nw_ptr);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd m3 F1: {:?}", e))?;
    }

    // F2: in_proj SGEMM
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut scratch.proj_flat,
        &acts.post_norm,
        lw.in_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, dm, ip),
    )?;

    // F3: m3_split
    {
        let n_i = bt as i32;
        let di_i = di as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let nh_i = nh as i32;
        let na_i = na as i32;
        let db_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
        let mut builder = ctx.stream.launch_builder(&m3k.m3_split);
        builder.arg(acts.z.inner_mut());
        builder.arg(acts.x.inner_mut());
        builder.arg(acts.b_raw.inner_mut());
        builder.arg(acts.c_raw.inner_mut());
        builder.arg(acts.dt.inner_mut());
        builder.arg(acts.a_val.inner_mut());
        builder.arg(acts.trap.inner_mut());
        builder.arg(acts.angles_raw.inner_mut());
        builder.arg(acts.dd_dt_raw.inner_mut());
        builder.arg(acts.dd_a_raw.inner_mut());
        builder.arg(acts.trap_raw.inner_mut());
        builder.arg(scratch.proj_flat.inner());
        builder.arg(&db_ptr);
        builder.arg(&dims.a_floor);
        builder.arg(&n_i);
        builder.arg(&di_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        builder.arg(&nh_i);
        builder.arg(&na_i);
        unsafe { builder.launch(grid_1d(bt * ip)) }.map_err(|e| format!("m3_split F3: {:?}", e))?;
    }

    // F4a/b: BCNorm of B and C in one launch (grid y picks the operand),
    // the same per-row arithmetic the two separate launches ran.
    {
        let bn_ptr = lw.b_norm_weight.raw_ptr(&ctx.stream);
        let cn_ptr = lw.c_norm_weight.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((bt * ng) as u32, 2, 1),
            block_dim: (ds as u32, 1, 1),
            shared_mem_bytes: ds as u32 * 4,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd_bc_f32);
        builder.arg(acts.b_normed.inner_mut());
        builder.arg(acts.c_normed.inner_mut());
        builder.arg(acts.b_rms.inner_mut());
        builder.arg(acts.c_rms.inner_mut());
        builder.arg(acts.b_raw.inner());
        builder.arg(acts.c_raw.inner());
        builder.arg(&bn_ptr);
        builder.arg(&cn_ptr);
        builder.arg(&n_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        let eps_g5: f32 = dims.rms_norm_eps;
        builder.arg(&eps_g5);
        // B_raw and C_raw are dense [bt, ng*ds] here (the prefill reads
        // them inside the projection and passes its row width instead).
        let src_stride = (ng * ds) as i32;
        builder.arg(&src_stride);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("bcnorm_fwd B+C F4: {:?}", e))?;
    }
    // F5: angle accumulation (chunk-parallel; see gpu_angle_chunked_fwd)
    if na > 0 {
        // SAFETY: layer state and activations are disjoint allocations on
        // this context and remain live for the enclosing forward pass.
        unsafe {
            gpu_angle_chunked_fwd(
                ctx,
                m3k,
                AngleChunkedFwd {
                    angle_cumsum: &mut acts.angle_cumsum,
                    angle_state_ptr: layer_ptrs.angle_state,
                    angles_raw: &acts.angles_raw,
                    dt: &acts.dt,
                    sums: &scratch.angle_chunk_sums,
                    carries: &scratch.angle_chunk_carries,
                    shape: AngleChunkedShape {
                        batch: bt / dims.seq_len,
                        seq_len: dims.seq_len,
                        heads: nh,
                        angles: na,
                    },
                },
            )
        }?;
    }

    // F4c-f fused: bias add (B + C) + RoPE in one launch. The biased
    // tensors still materialize (backward saves); with n_angles == 0 the
    // kernel passes through, replacing the old copy branch too.
    {
        let bb_ptr = lw.b_bias.raw_ptr(&ctx.stream);
        let cb_ptr = lw.c_bias.raw_ptr(&ctx.stream);
        let n_i = bt as i32;
        let nh_i = nh as i32;
        let ng_i = ng as i32;
        let ds_i = ds as i32;
        let na_i = na as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_bias_rope_fwd);
        builder.arg(acts.b_biased.inner_mut());
        builder.arg(acts.c_biased.inner_mut());
        builder.arg(acts.k.inner_mut());
        builder.arg(acts.q.inner_mut());
        builder.arg(acts.b_normed.inner());
        builder.arg(acts.c_normed.inner());
        builder.arg(&bb_ptr);
        builder.arg(&cb_ptr);
        builder.arg(acts.angle_cumsum.inner());
        builder.arg(&n_i);
        builder.arg(&nh_i);
        builder.arg(&ng_i);
        builder.arg(&ds_i);
        builder.arg(&na_i);
        // No angle state on this lane: the saved angles are read.
        let no_angle: cudarc::driver::sys::CUdeviceptr = 0;
        builder.arg(&no_angle);
        builder.arg(&no_angle);
        builder.arg(&no_angle);
        unsafe { builder.launch(grid_1d(bt * nh * ds)) }
            .map_err(|e| format!("m3_bias_rope_fwd F4: {:?}", e))?;
    }

    // F5b: alpha/beta/gamma. Only the sequential kernels (the burn-in
    // forward, its backward and the abg backward) read them; the chunked
    // pipeline derives its own coefficients, so the launch is skipped there.
    if !dims.use_parallel_scan {
        let n_total = (bt * nh) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.m3_compute_abg);
        builder.arg(acts.alpha.inner_mut());
        builder.arg(acts.beta.inner_mut());
        builder.arg(acts.gamma.inner_mut());
        builder.arg(acts.dt.inner());
        builder.arg(acts.a_val.inner());
        builder.arg(acts.trap.inner());
        builder.arg(&n_total);
        unsafe { builder.launch(grid_1d(bt * nh)) }
            .map_err(|e| format!("m3_compute_abg F5b: {:?}", e))?;
    }

    // F6: SSM
    if dims.use_parallel_scan {
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let t_i = dims.seq_len as i32;
        let cs = dims.chunk_size() as i32;
        let nc = dims.n_chunks();
        let b_i = dims.batch as i32;

        {
            let (kern, count) = adt_multiply_launch(
                m3k,
                bt * nh,
                &[
                    scratch.d_alpha.cached_ptr(),
                    acts.a_val.cached_ptr(),
                    acts.dt.cached_ptr(),
                ],
            );
            let n_total = count as i32;
            let mut builder = ctx.stream.launch_builder(kern);
            builder.arg(scratch.d_alpha.inner_mut());
            builder.arg(acts.a_val.inner());
            builder.arg(acts.dt.inner());
            builder.arg(&n_total);
            unsafe { builder.launch(grid_1d(count)) }
                .map_err(|e| format!("adt compute F6: {:?}", e))?;
        }
        {
            let block_x = nh.min(256) as u32;
            let grid_z = nh.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nc as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_da_cumsum);
            builder.arg(acts.da_cumsum_saved.inner_mut());
            builder.arg(scratch.d_alpha.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_dA_cumsum F6 K2: {:?}", e))?;
        }
        // K1+K3 fused (preprocess + chunk_state) when the shape allows;
        // K2 (dA_cumsum) moved ahead - it depends only on adt.
        if let Some(fcfg) =
            super::kernels::chunk_fused_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size())
        {
            let mut builder = ctx
                .stream
                .launch_builder(&m3k.m3_chunk_pre_state_fused_typed.f32);
            builder.arg(acts.k_scaled_saved.inner_mut());
            builder.arg(acts.qk_dot_saved.inner_mut());
            builder.arg(acts.scale_saved.inner_mut());
            builder.arg(acts.gamma_saved.inner_mut());
            builder.arg(acts.chunk_states_saved.inner_mut());
            builder.arg(acts.k.inner());
            builder.arg(acts.q.inner());
            builder.arg(acts.dt.inner());
            builder.arg(acts.trap.inner());
            builder.arg(acts.x.inner());
            builder.arg(acts.da_cumsum_saved.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(fcfg) }
                .map_err(|e| format!("m3 F6 fused pre+state: {:?}", e))?;
        } else {
            {
                let cfg = cudarc::driver::LaunchConfig {
                    grid_dim: ((dims.batch * nc) as u32, nh as u32, 1),
                    block_dim: (dims.chunk_size() as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut builder = ctx.stream.launch_builder(&m3k.m3_preprocess_chunks);
                builder.arg(acts.k_scaled_saved.inner_mut());
                builder.arg(acts.qk_dot_saved.inner_mut());
                builder.arg(acts.scale_saved.inner_mut());
                builder.arg(acts.gamma_saved.inner_mut());
                builder.arg(acts.k.inner());
                builder.arg(acts.q.inner());
                builder.arg(acts.dt.inner());
                builder.arg(acts.trap.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&ds_i);
                builder.arg(&cs);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("m3_preprocess_chunks F6 K1: {:?}", e))?;
            }
            {
                let cfg =
                    super::kernels::chunk_state_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size());
                let mut builder = ctx.stream.launch_builder(&m3k.m3_chunk_state_fwd);
                builder.arg(acts.chunk_states_saved.inner_mut());
                builder.arg(acts.x.inner());
                builder.arg(acts.k_scaled_saved.inner());
                builder.arg(acts.da_cumsum_saved.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&hd_i);
                builder.arg(&ds_i);
                builder.arg(&cs);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| format!("m3_chunk_state_fwd F6 K3: {:?}", e))?;
            }
        }
        {
            let dim = hd * ds;
            let block_x = dim.min(256) as u32;
            let grid_z = dim.div_ceil(block_x as usize) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, grid_z),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let nc_i = nc as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.m3_state_passing_fwd);
            builder.arg(acts.chunk_states_saved.inner_mut());
            builder.arg(scratch.final_states.inner_mut());
            builder.arg(acts.da_cumsum_saved.inner());
            // Training forward: stateless window — chunk 0 enters at zero.
            // A state-carrying prefill passes the folded entering state.
            let init_states_null: crate::mamba3_siso::gpu::state::CUptr = 0;
            builder.arg(&init_states_null);
            builder.arg(&b_i);
            builder.arg(&nc_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            builder.arg(&t_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_state_passing_fwd F6 K4: {:?}", e))?;
        }
        {
            let (coop, cfg) =
                super::kernels::chunk_scan_cfg(dims.batch, nc, nh, hd, ds, dims.chunk_size());
            let kern = if coop {
                &m3k.m3_chunk_scan_fwd_coop
            } else {
                &m3k.m3_chunk_scan_fwd
            };
            let mut builder = ctx.stream.launch_builder(kern);
            builder.arg(acts.y.inner_mut());
            builder.arg(acts.x.inner());
            builder.arg(acts.q.inner());
            builder.arg(acts.k_scaled_saved.inner());
            builder.arg(acts.qk_dot_saved.inner());
            builder.arg(acts.da_cumsum_saved.inner());
            builder.arg(acts.chunk_states_saved.inner());
            builder.arg(&dp_ptr);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            builder.arg(&cs);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_chunk_scan_fwd F6 K5: {:?}", e))?;
        }

        // The chunked kernels above wrote their saves straight into the
        // layer acts; the backward reads them from there.

        {
            let block_x = hd.max(ds) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, 1),
                block_dim: (block_x, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_writeback_parallel_states);
            builder.arg(&layer_ptrs.ssm_state);
            builder.arg(&layer_ptrs.k_state);
            builder.arg(&layer_ptrs.v_state);
            builder.arg(scratch.final_states.inner());
            builder.arg(acts.k.inner());
            builder.arg(acts.x.inner());
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_writeback_parallel_states F6: {:?}", e))?;
        }
    } else {
        let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let nh_i = nh as i32;
        let hd_i = hd as i32;
        let ds_i = ds as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (dims.batch as u32, nh as u32, 1),
            block_dim: (hd as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.m3_burnin_fwd);
        builder.arg(&layer_ptrs.ssm_state);
        builder.arg(&layer_ptrs.k_state);
        builder.arg(&layer_ptrs.v_state);
        builder.arg(acts.y.inner_mut());
        builder.arg(acts.h_saved.inner_mut());
        builder.arg(acts.k_prev_saved.inner_mut());
        builder.arg(acts.v_prev_saved.inner_mut());
        builder.arg(acts.x.inner());
        builder.arg(acts.k.inner());
        builder.arg(acts.q.inner());
        builder.arg(acts.alpha.inner());
        builder.arg(acts.beta.inner());
        builder.arg(acts.gamma.inner());
        builder.arg(&dp_ptr);
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&nh_i);
        builder.arg(&hd_i);
        builder.arg(&ds_i);
        unsafe { builder.launch(cfg) }.map_err(|e| format!("m3_burnin_fwd F6 seq: {:?}", e))?;
    }

    // F7: Output gating
    if dims.is_outproj_norm {
        assert!(
            di <= 1024,
            "d_inner ({di}) exceeds rmsnorm_gated shared memory limit (1024)"
        );
        let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let di_i = di as i32;
        let hd_i = dims.headdim as i32;
        let grid = cudarc::driver::LaunchConfig {
            grid_dim: (bt as u32, 1, 1),
            block_dim: (di as u32, 1, 1),
            shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
        };
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_fwd);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.gated_rms_vals.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&nw_ptr);
        builder.arg(&bt_i);
        builder.arg(&di_i);
        builder.arg(&hd_i);
        let eps_g5: f32 = dims.rms_norm_eps;
        builder.arg(&eps_g5);
        unsafe { builder.launch(grid) }.map_err(|e| format!("rmsnorm_gated_fwd m3 F7: {:?}", e))?;
    } else {
        let n = (bt * di) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
        builder.arg(acts.gated.inner_mut());
        builder.arg(acts.y.inner());
        builder.arg(acts.z.inner());
        builder.arg(&n);
        unsafe { builder.launch(grid_1d(bt * di)) }
            .map_err(|e| format!("silu_gate_fwd m3 F7: {:?}", e))?;
    }

    // F8: out_proj + residual
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut scratch.out_flat,
        &acts.gated,
        lw.out_proj_w.raw_ptr(&ctx.stream),
        None,
        (bt, di, dm),
    )?;
    {
        // Write the NEXT layer's residual slot (or norm_f_input for the
        // last layer) directly - raw pointer, no aliasing of the
        // borrowed acts.
        let ne = (bt * dm) as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.residual_add);
        builder.arg(&stream_out);
        builder.arg(scratch.out_flat.inner());
        builder.arg(acts.residual.inner());
        builder.arg(&ne);
        unsafe { builder.launch(grid_1d(bt * dm)) }
            .map_err(|e| format!("residual_add m3 F8: {:?}", e))?;
    }
    Ok(())
}

/// Mamba-3 SISO full backbone forward (input proj + N layers + norm_f).
///
/// State semantics: with `dims.use_parallel_scan` the window is STATELESS —
/// this TRAINING forward runs stateless windows: it passes a null entering
/// state to `m3_state_passing_fwd` (chunk 0 starts from zero) and zeroes
/// all four state buffers (including the RoPE angle accumulator) up front
/// to keep the semantics consistent. Final window states are still written
/// back for inspection. The chunked kernels themselves DO support an
/// entering state (nullable `init_states` + the `m3_chunk_entering_state`
/// trapezoidal boundary fold) — that route belongs to the state-carrying
/// prefill; within training, cross-call continuity remains the sequential
/// path's job (`use_parallel_scan = false`, `m3_burnin_fwd`).
pub fn gpu_forward_mamba3_backbone(
    exec: &M3Exec<'_>,
    temporal: &mut GpuBuffer,
    acts: &mut GpuMamba3BackboneActs,
    mamba_w: &GpuMamba3Weights,
    mamba_input: &GpuBuffer,
    states: GpuMamba3StateBufs<'_>,
    scratch: &mut GpuMamba3Scratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let na = dims.n_angles.max(1);

    if dims.use_parallel_scan {
        // This training window is STATELESS (null entering state to the
        // chunked kernels), but the angle kernel WOULD carry the persistent accumulator — a hybrid
        // that matches neither stateless-window nor full-continuity
        // semantics. Zero all four so every parallel window is cleanly
        // stateless. (Async memsets — CUDA Graph capture safe.)
        states.ssm.zero(&ctx.stream)?;
        states.k.zero(&ctx.stream)?;
        states.v.zero(&ctx.stream)?;
        states.angle.zero(&ctx.stream)?;
    }

    acts.input_proj_inputs
        .copy_from_raw(mamba_input, &ctx.stream)?;
    gpu_gemm_bi_forward_raw(
        ctx,
        temporal,
        mamba_input,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        Some(mamba_w.input_proj_b.raw_ptr(&ctx.stream)),
        (bt, dims.mamba_input_dim, dm),
    )?;
    acts.input_proj_outputs
        .copy_from_raw(temporal, &ctx.stream)?;
    // Seed the residual chain: layer 0's slot gets the input_proj output
    // once; from there every layer's residual_add writes the next slot.
    acts.layers[0]
        .residual
        .copy_from_raw(temporal, &ctx.stream)?;

    let f32_sz = std::mem::size_of::<f32>() as u64;
    let ssm_base = states.ssm.raw_ptr(&ctx.stream);
    let k_base = states.k.raw_ptr(&ctx.stream);
    let v_base = states.v.raw_ptr(&ctx.stream);
    let a_base = states.angle.raw_ptr(&ctx.stream);

    for l in 0..dims.n_layers {
        let ssm_off = dims.batch * l * nh * hd * ds;
        let k_off = dims.batch * l * nh * ds;
        let v_off = dims.batch * l * nh * hd;
        let a_off = dims.batch * l * nh * na;
        let layer_ptrs = Mamba3LayerPtrs {
            ssm_state: ssm_base + ssm_off as u64 * f32_sz,
            k_state: k_base + k_off as u64 * f32_sz,
            v_state: v_base + v_off as u64 * f32_sz,
            angle_state: a_base + a_off as u64 * f32_sz,
        };
        let stream_out = if l + 1 < dims.n_layers {
            acts.layers[l + 1].residual.raw_ptr(&ctx.stream)
        } else {
            acts.norm_f_input.raw_ptr(&ctx.stream)
        };
        gpu_forward_mamba3_layer(
            exec,
            stream_out,
            &mut acts.layers[l],
            &mamba_w.layers[l],
            &layer_ptrs,
            scratch,
        )?;
    }

    // The last layer's residual_add already wrote acts.norm_f_input.
    {
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(temporal.inner_mut());
        builder.arg(acts.norm_f_rms.inner_mut());
        builder.arg(acts.norm_f_input.inner());
        builder.arg(&nf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f m3: {:?}", e))?;
    }
    Ok(())
}

/// Mamba-3 target burn-in forward (no activations saved, used for Bellman target).
///
/// Output: `temporal` receives `[B * d_model]` — the LAST timestep's representation.
/// Internally uses `tgt.temporal_work` as `[B*T*d_model]` working buffer.
pub fn gpu_forward_mamba3_target_burnin(
    exec: &M3Exec<'_>,
    temporal: &mut GpuBuffer,
    mamba_w: &GpuMamba3Weights,
    mamba_input: &GpuBuffer,
    tgt: &mut GpuMamba3TargetScratch,
) -> Result<(), String> {
    let M3Exec {
        ctx,
        kernels: m3k,
        dims,
    } = *exec;
    let bt = dims.bt();
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let nh = dims.nheads;
    let hd = dims.headdim;
    let ng = dims.ngroups;
    let ip = dims.in_proj_dim;
    // The TRUE angle count goes to the m3_split kernel arg — the
    // main forward already does this (see the top of this file). Passing the
    // buffer-sizing `.max(1)` here shifted the 8-way split by one slot for
    // the supported n_angles == 0 config.
    let na = dims.n_angles;
    let b = dims.batch as i32;
    let t = dims.seq_len as i32;
    let f32_sz = std::mem::size_of::<f32>() as u64;

    tgt.ssm_states.zero(&ctx.stream)?;
    tgt.k_states.zero(&ctx.stream)?;
    tgt.v_states.zero(&ctx.stream)?;
    tgt.angle_states.zero(&ctx.stream)?;

    gpu_gemm_bi_forward_raw(
        ctx,
        &mut tgt.temporal_work,
        mamba_input,
        mamba_w.input_proj_w.raw_ptr(&ctx.stream),
        Some(mamba_w.input_proj_b.raw_ptr(&ctx.stream)),
        (bt, dims.mamba_input_dim, dm),
    )?;

    for l in 0..dims.n_layers {
        let lw = &mamba_w.layers[l];
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = dims.rms_norm_eps;
            let nw_ptr = lw.norm_weight.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
            builder.arg(tgt.out_flat.inner_mut());
            builder.arg(tgt.rms_discard.inner_mut());
            builder.arg(tgt.residual.inner());
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_fwd m3 tgt L{l}: {:?}", e))?;
        }
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut tgt.proj_flat,
            &tgt.out_flat,
            lw.in_proj_w.raw_ptr(&ctx.stream),
            None,
            (bt, dm, ip),
        )?;
        {
            let n_i = bt as i32;
            let di_i = di as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let nh_i = nh as i32;
            let na_i = na as i32;
            let db_ptr = lw.dt_bias.raw_ptr(&ctx.stream);
            let mut builder = ctx.stream.launch_builder(&m3k.m3_split);
            builder.arg(tgt.z.inner_mut());
            builder.arg(tgt.x.inner_mut());
            builder.arg(tgt.b_raw.inner_mut());
            builder.arg(tgt.c_raw.inner_mut());
            builder.arg(tgt.dt.inner_mut());
            builder.arg(tgt.a_val.inner_mut());
            builder.arg(tgt.trap.inner_mut());
            builder.arg(tgt.angles_raw.inner_mut());
            builder.arg(tgt.dd_dt_raw.inner_mut());
            builder.arg(tgt.dd_a_raw.inner_mut());
            builder.arg(tgt.trap_raw.inner_mut());
            builder.arg(tgt.proj_flat.inner());
            builder.arg(&db_ptr);
            builder.arg(&dims.a_floor);
            builder.arg(&n_i);
            builder.arg(&di_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            builder.arg(&nh_i);
            builder.arg(&na_i);
            unsafe { builder.launch(grid_1d(bt * ip)) }
                .map_err(|e| format!("m3_split tgt L{l}: {:?}", e))?;
        }
        {
            // BCNorm of B and C in one launch (grid y picks the operand).
            let bn_ptr = lw.b_norm_weight.raw_ptr(&ctx.stream);
            let cn_ptr = lw.c_norm_weight.raw_ptr(&ctx.stream);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: ((bt * ng) as u32, 2, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: ds as u32 * 4,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.bcnorm_fwd_bc_f32);
            builder.arg(tgt.b_normed.inner_mut());
            builder.arg(tgt.c_normed.inner_mut());
            builder.arg(tgt.b_rms.inner_mut());
            builder.arg(tgt.c_rms.inner_mut());
            builder.arg(tgt.b_raw.inner());
            builder.arg(tgt.c_raw.inner());
            builder.arg(&bn_ptr);
            builder.arg(&cn_ptr);
            let n_i = bt as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            builder.arg(&n_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            let eps_g5: f32 = dims.rms_norm_eps;
            builder.arg(&eps_g5);
            // Dense B_raw/C_raw rows, as in the training forward.
            let src_stride = (ng * ds) as i32;
            builder.arg(&src_stride);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("bcnorm_fwd B+C tgt L{l}: {:?}", e))?;
        }
        {
            let bb_ptr = lw.b_bias.raw_ptr(&ctx.stream);
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
            builder.arg(tgt.b_biased.inner_mut());
            builder.arg(tgt.b_normed.inner());
            builder.arg(&bb_ptr);
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                .map_err(|e| format!("bc_bias_add B tgt L{l}: {:?}", e))?;
        }
        {
            let cb_ptr = lw.c_bias.raw_ptr(&ctx.stream);
            let n_i = bt as i32;
            let nh_i = nh as i32;
            let ng_i = ng as i32;
            let ds_i = ds as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.bc_bias_add);
            builder.arg(tgt.c_biased.inner_mut());
            builder.arg(tgt.c_normed.inner());
            builder.arg(&cb_ptr);
            builder.arg(&n_i);
            builder.arg(&nh_i);
            builder.arg(&ng_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                .map_err(|e| format!("bc_bias_add C tgt L{l}: {:?}", e))?;
        }
        if dims.n_angles > 0 {
            {
                let a_off = dims.batch * l * nh * na;
                let angle_st = tgt.angle_states.raw_ptr(&ctx.stream) + a_off as u64 * f32_sz;
                let b_i = dims.batch as i32;
                let t_i = dims.seq_len as i32;
                let nh_i = nh as i32;
                let na_i = na as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.m3_angle_dt_fwd_seq);
                builder.arg(tgt.angle_cumsum.inner_mut());
                builder.arg(&angle_st);
                builder.arg(tgt.angles_raw.inner());
                builder.arg(tgt.dt.inner());
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&nh_i);
                builder.arg(&na_i);
                let grid = cudarc::driver::LaunchConfig {
                    grid_dim: (dims.batch as u32, (nh * na).div_ceil(256) as u32, 1),
                    block_dim: (256.min((nh * na) as u32), 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(grid) }
                    .map_err(|e| format!("angle_dt_fwd_seq tgt L{l}: {:?}", e))?;
            }
            {
                let n_i = bt as i32;
                let nh_i = nh as i32;
                let ds_i = ds as i32;
                let na_i = na as i32;
                let mut builder = ctx.stream.launch_builder(&m3k.rope_fwd);
                builder.arg(tgt.k.inner_mut());
                builder.arg(tgt.q.inner_mut());
                builder.arg(tgt.b_biased.inner());
                builder.arg(tgt.c_biased.inner());
                builder.arg(tgt.angle_cumsum.inner());
                builder.arg(&n_i);
                builder.arg(&nh_i);
                builder.arg(&ds_i);
                builder.arg(&na_i);
                unsafe { builder.launch(grid_1d(bt * nh * ds)) }
                    .map_err(|e| format!("rope_fwd tgt L{l}: {:?}", e))?;
            }
        } else {
            tgt.k.copy_from_raw(&tgt.b_biased, &ctx.stream)?;
            tgt.q.copy_from_raw(&tgt.c_biased, &ctx.stream)?;
        }
        {
            let n_total = (bt * nh) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.m3_compute_abg);
            builder.arg(tgt.alpha.inner_mut());
            builder.arg(tgt.beta.inner_mut());
            builder.arg(tgt.gamma.inner_mut());
            builder.arg(tgt.dt.inner());
            builder.arg(tgt.a_val.inner());
            builder.arg(tgt.trap.inner());
            builder.arg(&n_total);
            unsafe { builder.launch(grid_1d(bt * nh)) }
                .map_err(|e| format!("m3_compute_abg tgt L{l}: {:?}", e))?;
        }
        {
            let ssm_off = dims.batch * l * nh * hd * ds;
            let k_off = dims.batch * l * nh * ds;
            let v_off = dims.batch * l * nh * hd;
            let ssm_ptr = tgt.ssm_states.raw_ptr(&ctx.stream) + ssm_off as u64 * f32_sz;
            let k_ptr = tgt.k_states.raw_ptr(&ctx.stream) + k_off as u64 * f32_sz;
            let v_ptr = tgt.v_states.raw_ptr(&ctx.stream) + v_off as u64 * f32_sz;
            let dp_ptr = lw.d_param.raw_ptr(&ctx.stream);
            let nh_i = nh as i32;
            let hd_i = hd as i32;
            let ds_i = ds as i32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (dims.batch as u32, nh as u32, 1),
                block_dim: (hd as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.m3_burnin_fwd_nosave);
            builder.arg(&ssm_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(tgt.y.inner_mut());
            builder.arg(tgt.x.inner());
            builder.arg(tgt.k.inner());
            builder.arg(tgt.q.inner());
            builder.arg(tgt.alpha.inner());
            builder.arg(tgt.beta.inner());
            builder.arg(tgt.gamma.inner());
            builder.arg(&dp_ptr);
            builder.arg(&b);
            builder.arg(&t);
            builder.arg(&nh_i);
            builder.arg(&hd_i);
            builder.arg(&ds_i);
            unsafe { builder.launch(cfg) }
                .map_err(|e| format!("m3_burnin_fwd_nosave tgt L{l}: {:?}", e))?;
        }
        if dims.is_outproj_norm {
            assert!(
                di <= 1024,
                "d_inner ({di}) exceeds rmsnorm_gated shared memory limit"
            );
            let nw_ptr = lw.norm_gate_weight.raw_ptr(&ctx.stream);
            let bt_i = bt as i32;
            let di_i = di as i32;
            let hd_i = dims.headdim as i32;
            let grid = cudarc::driver::LaunchConfig {
                grid_dim: (bt as u32, 1, 1),
                block_dim: (di as u32, 1, 1),
                shared_mem_bytes: (di * std::mem::size_of::<f32>()) as u32,
            };
            let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_gated_fwd);
            builder.arg(tgt.gated.inner_mut());
            builder.arg(tgt.rms_discard.inner_mut());
            builder.arg(tgt.y.inner());
            builder.arg(tgt.z.inner());
            builder.arg(&nw_ptr);
            builder.arg(&bt_i);
            builder.arg(&di_i);
            builder.arg(&hd_i);
            let eps_g5: f32 = dims.rms_norm_eps;
            builder.arg(&eps_g5);
            unsafe { builder.launch(grid) }
                .map_err(|e| format!("rmsnorm_gated_fwd m3 tgt L{l}: {:?}", e))?;
        } else {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.silu_gate_fwd);
            builder.arg(tgt.gated.inner_mut());
            builder.arg(tgt.y.inner());
            builder.arg(tgt.z.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("silu_gate_fwd m3 tgt L{l}: {:?}", e))?;
        }
        gpu_gemm_bi_forward_raw(
            ctx,
            &mut tgt.out_flat,
            &tgt.gated,
            lw.out_proj_w.raw_ptr(&ctx.stream),
            None,
            (bt, di, dm),
        )?;
        {
            let ne = (bt * dm) as i32;
            let mut builder = ctx.stream.launch_builder(&m3k.residual_add);
            builder.arg(tgt.temporal_work.inner_mut());
            builder.arg(tgt.out_flat.inner());
            builder.arg(tgt.residual.inner());
            builder.arg(&ne);
            unsafe { builder.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual_add m3 tgt L{l}: {:?}", e))?;
        }
    }

    {
        tgt.residual
            .copy_from_raw(&tgt.temporal_work, &ctx.stream)?;
        let nf_ptr = mamba_w.norm_f_weight.raw_ptr(&ctx.stream);
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        let mut builder = ctx.stream.launch_builder(&m3k.rmsnorm_fwd);
        builder.arg(tgt.temporal_work.inner_mut());
        builder.arg(tgt.rms_discard.inner_mut());
        builder.arg(tgt.residual.inner());
        builder.arg(&nf_ptr);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_fwd norm_f m3 tgt: {:?}", e))?;
    }

    {
        let b_i = dims.batch as i32;
        let t_i = dims.seq_len as i32;
        let dm_i = dm as i32;
        let mut builder = ctx.stream.launch_builder(&m3k.gather_last_timestep);
        builder.arg(temporal.inner_mut());
        builder.arg(tgt.temporal_work.inner());
        builder.arg(&b_i);
        builder.arg(&t_i);
        builder.arg(&dm_i);
        unsafe { builder.launch(grid_1d(dims.batch * dm)) }
            .map_err(|e| format!("gather_last_timestep m3 tgt: {:?}", e))?;
    }

    Ok(())
}

#[cfg(test)]
mod angle_chunked_contract_tests {
    use super::{
        AngleChunkedLengths, AngleChunkedShape, CHUNK_SIZE, validate_angle_chunked_layout,
    };

    const SHAPE: AngleChunkedShape = AngleChunkedShape {
        batch: 2,
        seq_len: 65,
        heads: 3,
        angles: 4,
    };
    const LENGTHS: AngleChunkedLengths = AngleChunkedLengths {
        angle_cumsum: 1_560,
        angles_raw: 520,
        dt: 390,
        sums_bytes: 384,
        carries_bytes: 96,
    };

    #[test]
    fn rejects_zero_dimensions_and_state_pointer() {
        for shape in [
            AngleChunkedShape { batch: 0, ..SHAPE },
            AngleChunkedShape {
                seq_len: 0,
                ..SHAPE
            },
            AngleChunkedShape { heads: 0, ..SHAPE },
            AngleChunkedShape { angles: 0, ..SHAPE },
        ] {
            let err = validate_angle_chunked_layout(shape, LENGTHS, 1).unwrap_err();
            assert!(err.contains("must be non-zero"), "{err}");
        }

        let err = validate_angle_chunked_layout(SHAPE, LENGTHS, 0).unwrap_err();
        assert!(err.contains("angle state pointer is null"), "{err}");
    }

    #[test]
    fn rejects_geometry_overflow_before_launch() {
        let shape = AngleChunkedShape {
            batch: usize::MAX,
            seq_len: 2,
            heads: 1,
            angles: 1,
        };
        let err = validate_angle_chunked_layout(shape, LENGTHS, 1).unwrap_err();
        assert!(err.contains("overflows usize"), "{err}");
    }

    #[test]
    fn rejects_dimension_truncation_before_launch() {
        let shape = AngleChunkedShape {
            batch: i32::MAX as usize + 1,
            seq_len: 1,
            heads: 1,
            angles: 1,
        };
        let lengths = AngleChunkedLengths {
            angle_cumsum: shape.batch,
            angles_raw: shape.batch,
            dt: shape.batch,
            sums_bytes: shape.batch * std::mem::size_of::<f64>(),
            carries_bytes: shape.batch * std::mem::size_of::<f32>(),
        };
        let err = validate_angle_chunked_layout(shape, lengths, 1).unwrap_err();
        assert!(err.contains("does not fit i32"), "{err}");
    }

    #[test]
    fn rejects_chunk_ceiling_integer_overflow() {
        let shape = AngleChunkedShape {
            batch: 1,
            seq_len: i32::MAX as usize,
            heads: 1,
            angles: 1,
        };
        let n_chunks = shape.seq_len.div_ceil(CHUNK_SIZE);
        let lengths = AngleChunkedLengths {
            angle_cumsum: shape.seq_len,
            angles_raw: shape.seq_len,
            dt: shape.seq_len,
            sums_bytes: n_chunks * std::mem::size_of::<f64>(),
            carries_bytes: std::mem::size_of::<f32>(),
        };
        let err = validate_angle_chunked_layout(shape, lengths, 1).unwrap_err();
        assert!(err.contains("padded sequence length"), "{err}");
    }

    #[test]
    fn rejects_each_undersized_buffer() {
        for (label, lengths) in [
            (
                "angle_cumsum",
                AngleChunkedLengths {
                    angle_cumsum: LENGTHS.angle_cumsum - 1,
                    ..LENGTHS
                },
            ),
            (
                "angles_raw",
                AngleChunkedLengths {
                    angles_raw: LENGTHS.angles_raw - 1,
                    ..LENGTHS
                },
            ),
            (
                "dt",
                AngleChunkedLengths {
                    dt: LENGTHS.dt - 1,
                    ..LENGTHS
                },
            ),
            (
                "sums",
                AngleChunkedLengths {
                    sums_bytes: LENGTHS.sums_bytes - 1,
                    ..LENGTHS
                },
            ),
            (
                "carries",
                AngleChunkedLengths {
                    carries_bytes: LENGTHS.carries_bytes - 1,
                    ..LENGTHS
                },
            ),
        ] {
            let err = validate_angle_chunked_layout(SHAPE, lengths, 1).unwrap_err();
            assert!(err.contains(label), "{label}: {err}");
        }
    }
}
