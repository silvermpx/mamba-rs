//! The gated MLP after the mixer (the reference `Block` with
//! `d_intermediate`) on the device: what the training forwards save for the
//! backward, the training forward and backward in f32 and in the mixed lane,
//! the target forward's branch with nothing saved, and the branch of the
//! decode and prefill paths, which keep the block output pending for the
//! next norm.
//!
//! The branch is `x += fc2(value * silu(gate))` with
//! `[value | gate] = fc1(norm2(x))`, the arithmetic of `crate::mlp::cpu`.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use super::weights::{GpuMambaMlpGrads, GpuMambaTrainMlpWeights, MambaMlpPtrs, mlp_norm_params};
use crate::gpu::blas::{
    TypedPtr, gpu_gemm_bi_backward_dw_grad_typed, gpu_gemm_bi_backward_grad_raw,
    gpu_gemm_bi_forward_raw, gpu_gemm_ex_backward_dx_typed, gpu_gemm_forward_dispatch,
    gpu_gemm_typed_forward_raw,
};
use crate::gpu::block_norm::{self, NormParams};
use crate::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{grid_1d_cover, launch_elems};

type Stream = Arc<cudarc::driver::CudaStream>;
type DevPtr = cudarc::driver::sys::CUdeviceptr;

/// One layer's gated MLP saves in the f32 training lane.
pub struct GpuMambaMlpActs {
    /// The rows entering the MLP, the mixer's residual sum `[B*T*d_model]`.
    pub x_in: GpuBuffer,
    /// The rows after its norm `[B*T*d_model]`.
    pub post: GpuBuffer,
    /// The norm's statistics per row `[B*T*stats]`.
    pub rms: GpuBuffer,
    /// The up projection `[B*T*2*hidden]`: value columns, then gate.
    pub up: GpuBuffer,
    /// The gated activation `[B*T*hidden]`.
    pub act: GpuBuffer,
}

/// Every layer's gated MLP saves in the f32 lane, for a model with one.
pub struct GpuMambaMlpBufs {
    pub hidden: usize,
    pub layers: Vec<GpuMambaMlpActs>,
}

impl GpuMambaMlpBufs {
    /// Saves for `n_layers` MLPs `hidden` wide over `rows` rows of
    /// `d_model`, behind a norm of the kind `norm`.
    pub fn new(
        stream: &Stream,
        (rows, d_model, n_layers): (usize, usize, usize),
        (hidden, norm): (usize, crate::config::BlockNorm),
    ) -> Result<Self, String> {
        let z = |n: usize| GpuBuffer::zeros(stream, n);
        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaMlpActs {
                    x_in: z(rows * d_model)?,
                    post: z(rows * d_model)?,
                    rms: z(rows * norm.stats_per_row())?,
                    up: z(rows * 2 * hidden)?,
                    act: z(rows * hidden)?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { hidden, layers })
    }
}

/// One layer's gated MLP saves in the mixed lane: the input rows and the
/// norm's statistics in f32 as the residual stream is, the norm's rows, the
/// up projection and the activation in the lane's storage.
pub struct GpuMambaMlpMixedActs {
    pub x_in: GpuBuffer,
    pub post: DtypedBuf,
    pub rms: GpuBuffer,
    pub up: DtypedBuf,
    pub act: DtypedBuf,
}

/// Every layer's gated MLP saves in the mixed lane, for a model with one.
pub struct GpuMambaMlpMixedBufs {
    pub hidden: usize,
    pub layers: Vec<GpuMambaMlpMixedActs>,
}

impl GpuMambaMlpMixedBufs {
    /// [`GpuMambaMlpBufs::new`] with the typed saves in `dtype`.
    pub fn new(
        stream: &Stream,
        (rows, d_model, n_layers): (usize, usize, usize),
        (hidden, norm): (usize, crate::config::BlockNorm),
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaMlpMixedActs {
                    x_in: GpuBuffer::zeros(stream, rows * d_model)?,
                    post: DtypedBuf::zeros(stream, rows * d_model, dtype)?,
                    rms: GpuBuffer::zeros(stream, rows * norm.stats_per_row())?,
                    up: DtypedBuf::zeros(stream, rows * 2 * hidden, dtype)?,
                    act: DtypedBuf::zeros(stream, rows * hidden, dtype)?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { hidden, layers })
    }
}

/// The f32 backward's buffers: the gradients of the activation, the up
/// projection and the norm's rows.
pub struct GpuMambaMlpBwdBufs {
    pub d_act: GpuBuffer,
    pub d_up: GpuBuffer,
    pub d_post: GpuBuffer,
}

impl GpuMambaMlpBwdBufs {
    /// Buffers for an MLP `hidden` wide over `rows` rows of `d_model`.
    pub fn new(
        stream: &Stream,
        (rows, d_model): (usize, usize),
        hidden: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            d_act: GpuBuffer::zeros(stream, rows * hidden)?,
            d_up: GpuBuffer::zeros(stream, rows * 2 * hidden)?,
            d_post: GpuBuffer::zeros(stream, rows * d_model)?,
        })
    }
}

/// The mixed backward's buffers, in the lane's storage.
pub struct GpuMambaMlpMixedBwdBufs {
    pub d_act: DtypedBuf,
    pub d_up: DtypedBuf,
    pub d_post: DtypedBuf,
}

impl GpuMambaMlpMixedBwdBufs {
    /// [`GpuMambaMlpBwdBufs::new`] in `dtype`.
    pub fn new(
        stream: &Stream,
        (rows, d_model): (usize, usize),
        hidden: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Ok(Self {
            d_act: DtypedBuf::zeros(stream, rows * hidden, dtype)?,
            d_up: DtypedBuf::zeros(stream, rows * 2 * hidden, dtype)?,
            d_post: DtypedBuf::zeros(stream, rows * d_model, dtype)?,
        })
    }
}

/// The weights and the buffers built for them must agree on the gated MLP:
/// buffers of another width would be indexed past their end.
pub(crate) fn check_width(
    model: Option<usize>,
    built: Option<usize>,
    what: &str,
) -> Result<(), String> {
    if model != built {
        return Err(format!(
            "the gated MLP is {model:?} wide in the weights and {built:?} in the {what} \
             (build them for the model's MLP)"
        ));
    }
    Ok(())
}

/// The row count of a launch as the kernels take it.
fn ints(rows: usize, dm: usize, h: usize) -> (i32, i32, i32) {
    (rows as i32, dm as i32, h as i32)
}

/// The gate `act = value * silu(gate)` over `rows` rows of an up projection
/// `hidden` wide, in `dtype`.
fn gate_fwd(
    ctx: &GpuCtx,
    (act, up): (DevPtr, DevPtr),
    (rows, h): (usize, usize),
    dtype: WeightDtype,
) -> Result<(), String> {
    let (rows_i, _, h_i) = ints(rows, 0, h);
    let mut gate = ctx
        .stream
        .launch_builder(ctx.kernels.mlp_gate_fwd_typed.get(dtype));
    gate.arg(&act);
    gate.arg(&up);
    gate.arg(&rows_i);
    gate.arg(&h_i);
    unsafe { gate.launch(grid_1d_cover(rows * h)?) }
        .map(|_| ())
        .map_err(|e| format!("MLP gate: {e:?}"))
}

/// The gate's backward into the up projection's row, value then gate.
fn gate_bwd(
    ctx: &GpuCtx,
    (d_up, d_act, up): (DevPtr, DevPtr, DevPtr),
    (rows, h): (usize, usize),
    dtype: WeightDtype,
) -> Result<(), String> {
    let (rows_i, _, h_i) = ints(rows, 0, h);
    let mut gate = ctx
        .stream
        .launch_builder(ctx.kernels.mlp_gate_bwd.get(dtype));
    gate.arg(&d_up);
    gate.arg(&d_act);
    gate.arg(&up);
    gate.arg(&rows_i);
    gate.arg(&h_i);
    unsafe { gate.launch(grid_1d_cover(rows * h)?) }
        .map(|_| ())
        .map_err(|e| format!("MLP gate backward: {e:?}"))
}

/// The weight gradient of the norm from its per-row partials, summed over
/// the rows in a fixed order into `d_norm_w`.
fn reduce_norm_partials(
    ctx: &GpuCtx,
    (d_norm_w, partials): (DevPtr, DevPtr),
    (rows, dm): (usize, usize),
) -> Result<(), String> {
    let (rows_i, dm_i, _) = ints(rows, dm, 0);
    let block = (rows as u32).next_power_of_two().clamp(32, 256);
    let accumulate: i32 = 1;
    let mut reduce = ctx.stream.launch_builder(&ctx.kernels.reduce_sum_axis0);
    reduce.arg(&d_norm_w);
    reduce.arg(&partials);
    reduce.arg(&rows_i);
    reduce.arg(&dm_i);
    reduce.arg(&accumulate);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (dm as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    unsafe { reduce.launch(cfg) }
        .map(|_| ())
        .map_err(|e| format!("MLP norm reduce: {e:?}"))
}

/// One layer's gated MLP in the f32 training forward: the rows the mixer
/// left in `m.x_in` through the norm, the up projection, the gate and the
/// down projection, added to them into `stream_out`. The norm's rows and
/// rms, the up projection and the activation stay saved for the backward.
pub(crate) fn forward_f32(
    ctx: &GpuCtx,
    stream_out: DevPtr,
    (m, w): (&mut GpuMambaMlpActs, &GpuMambaTrainMlpWeights),
    out_flat: &mut GpuBuffer,
    (rows, dm, eps): (usize, usize, f32),
) -> Result<(), String> {
    let h = w.hidden();
    block_norm::forward(
        ctx,
        WeightDtype::F32,
        (m.post.cached_ptr(), m.rms.cached_ptr(), m.x_in.cached_ptr()),
        w.norm(),
        (rows, dm, eps),
        "MLP norm",
    )?;
    gpu_gemm_bi_forward_raw(
        ctx,
        &mut m.up,
        &m.post,
        w.fc1_w.cached_ptr(),
        None,
        (rows, dm, 2 * h),
    )?;
    gate_fwd(
        ctx,
        (m.act.cached_ptr(), m.up.cached_ptr()),
        (rows, h),
        WeightDtype::F32,
    )?;
    gpu_gemm_bi_forward_raw(
        ctx,
        out_flat,
        &m.act,
        w.fc2_w.cached_ptr(),
        None,
        (rows, h, dm),
    )?;
    launch_elems(
        &ctx.stream,
        &ctx.kernels.residual_add,
        &[stream_out, m.x_in.cached_ptr(), out_flat.cached_ptr()],
        rows * dm,
        format_args!("MLP residual"),
    )
}

/// The backward of one layer's gated MLP in the f32 lane: `d_temporal`
/// holds the gradient of the rows after the MLP and leaves holding that of
/// the rows entering it, the residual included. The down projection, the
/// gate, the up projection, then the norm, whose input gradient the norm
/// kernel adds into `d_temporal` itself.
pub(crate) fn backward_f32(
    ctx: &GpuCtx,
    d_temporal: &mut GpuBuffer,
    (m, w, g): (
        &GpuMambaMlpActs,
        &GpuMambaTrainMlpWeights,
        &GpuMambaMlpGrads,
    ),
    (bufs, partials): (&mut GpuMambaMlpBwdBufs, DevPtr),
    (rows, dm): (usize, usize),
) -> Result<(), String> {
    let h = w.hidden();
    if bufs.d_act.len() != rows * h {
        return Err(format!(
            "the scratch's gated MLP buffers hold {} activations, the model needs {rows} x {h}",
            bufs.d_act.len()
        ));
    }
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut bufs.d_act,
        (&g.fc2_w, None),
        d_temporal,
        &m.act,
        w.fc2_w.cached_ptr(),
        (rows, h, dm),
    )?;
    gate_bwd(
        ctx,
        (
            bufs.d_up.cached_ptr(),
            bufs.d_act.cached_ptr(),
            m.up.cached_ptr(),
        ),
        (rows, h),
        WeightDtype::F32,
    )?;
    gpu_gemm_bi_backward_grad_raw(
        ctx,
        &mut bufs.d_post,
        (&g.fc1_w, None),
        &bufs.d_up,
        &m.post,
        w.fc1_w.cached_ptr(),
        (rows, dm, 2 * h),
    )?;
    // Accumulate: the norm's input gradient adds into the residual's.
    block_norm::backward(
        ctx,
        WeightDtype::F32,
        (
            d_temporal.cached_ptr(),
            partials,
            bufs.d_post.cached_ptr(),
            m.x_in.cached_ptr(),
            m.rms.cached_ptr(),
        ),
        (w.norm(), g.norm_bias_ptr()),
        (rows, dm),
        (true, 0),
        "MLP norm backward",
    )?;
    reduce_norm_partials(ctx, (g.norm_weight.ptr(), partials), (rows, dm))
}

/// One layer's gated MLP in the mixed training forward: the f32 rows the
/// mixer left in `m.x_in` through the norm into the lane's storage, the
/// typed up projection, gate and down projection, and the f32 sum with the
/// rows into `stream_out`. `out_flat` takes the down projection.
pub(crate) fn forward_mixed(
    ctx: &GpuCtx,
    stream_out: DevPtr,
    (m, w): (&GpuMambaMlpMixedActs, MambaMlpPtrs),
    out_flat: DevPtr,
    (rows, dm, eps): (usize, usize, f32),
    dtype: WeightDtype,
) -> Result<(), String> {
    let h = w.hidden;
    let (x_in, post, rms, up, act) = (
        m.x_in.cached_ptr(),
        m.post.cached_ptr(),
        m.rms.cached_ptr(),
        m.up.cached_ptr(),
        m.act.cached_ptr(),
    );
    block_norm::forward(
        ctx,
        dtype,
        (post, rms, x_in),
        mlp_norm_params(&w),
        (rows, dm, eps),
        "mixed MLP norm",
    )?;
    let typed = |ptr| TypedPtr { ptr, dtype };
    let weight = |(ptr, dtype): (DevPtr, WeightDtype)| TypedPtr { ptr, dtype };
    gpu_gemm_typed_forward_raw(
        ctx,
        typed(up),
        typed(post),
        weight(w.fc1_w),
        None,
        (rows, dm, 2 * h),
    )?;
    gate_fwd(ctx, (act, up), (rows, h), dtype)?;
    gpu_gemm_typed_forward_raw(
        ctx,
        typed(out_flat),
        typed(act),
        weight(w.fc2_w),
        None,
        (rows, h, dm),
    )?;
    launch_elems(
        &ctx.stream,
        ctx.kernels.residual_add_f32_typed.get(dtype),
        &[stream_out, x_in, out_flat],
        rows * dm,
        format_args!("mixed MLP residual"),
    )
}

/// The backward of one layer's gated MLP in the mixed lane: `d_temporal`
/// holds the f32 gradient of the rows after the MLP and `d_typed` its copy
/// in the lane's storage; `d_temporal` leaves holding the f32 gradient of
/// the rows entering the MLP, the residual included, and `d_typed` its
/// typed mirror, which the mixer's backward below reads. The projections
/// and the gate run typed with f32 weight gradients as the mixer's do; the
/// norm takes the typed gradient of its rows against the f32 input rows.
pub(crate) fn backward_mixed(
    ctx: &GpuCtx,
    (d_temporal, d_typed): (&mut GpuBuffer, DevPtr),
    (m, w, g): (&GpuMambaMlpMixedActs, MambaMlpPtrs, &GpuMambaMlpGrads),
    (bufs, partials): (&GpuMambaMlpMixedBwdBufs, DevPtr),
    (rows, dm): (usize, usize),
    dtype: WeightDtype,
) -> Result<(), String> {
    let h = w.hidden;
    if bufs.d_act.len_elems() != rows * h {
        return Err(format!(
            "the mixed scratch's gated MLP buffers hold {} activations, the model needs \
             {rows} x {h}",
            bufs.d_act.len_elems()
        ));
    }
    let tp = |ptr| TypedPtr { ptr, dtype };
    let weight = |(ptr, dtype): (DevPtr, WeightDtype)| TypedPtr { ptr, dtype };
    let (d_act, d_up, d_post) = (
        bufs.d_act.cached_ptr(),
        bufs.d_up.cached_ptr(),
        bufs.d_post.cached_ptr(),
    );
    gpu_gemm_bi_backward_dw_grad_typed(
        ctx,
        &g.fc2_w,
        tp(d_typed),
        tp(m.act.cached_ptr()),
        rows,
        h,
        dm,
    )?;
    gpu_gemm_ex_backward_dx_typed(ctx, tp(d_act), tp(d_typed), weight(w.fc2_w), rows, h, dm)?;
    gate_bwd(ctx, (d_up, d_act, m.up.cached_ptr()), (rows, h), dtype)?;
    gpu_gemm_bi_backward_dw_grad_typed(
        ctx,
        &g.fc1_w,
        tp(d_up),
        tp(m.post.cached_ptr()),
        rows,
        dm,
        2 * h,
    )?;
    gpu_gemm_ex_backward_dx_typed(ctx, tp(d_post), tp(d_up), weight(w.fc1_w), rows, dm, 2 * h)?;
    // Accumulate: the norm's input gradient adds into the residual's, and
    // the sum's typed mirror is the mixer's upstream gradient.
    block_norm::backward(
        ctx,
        dtype,
        (
            d_temporal.cached_ptr(),
            partials,
            d_post,
            m.x_in.cached_ptr(),
            m.rms.cached_ptr(),
        ),
        (mlp_norm_params(&w), g.norm_bias_ptr()),
        (rows, dm),
        (true, d_typed),
        "mixed MLP norm backward",
    )?;
    reduce_norm_partials(ctx, (g.norm_weight.ptr(), partials), (rows, dm))
}

/// The up projection and the activation the branch runs in on a path with
/// nothing saved, f32, sized for the path's rows.
pub struct GpuMambaMlpBranchBufs {
    pub up: GpuBuffer,
    pub act: GpuBuffer,
    /// The rows and the MLP width the buffers were built for.
    shape: (usize, usize),
}

impl GpuMambaMlpBranchBufs {
    /// Buffers for an MLP `hidden` wide over `rows` rows.
    pub fn new(stream: &Stream, rows: usize, hidden: usize) -> Result<Self, String> {
        Ok(Self {
            up: GpuBuffer::zeros(stream, rows * 2 * hidden)?,
            act: GpuBuffer::zeros(stream, rows * hidden)?,
            shape: (rows, hidden),
        })
    }
}

/// [`GpuMambaMlpBranchBufs`] in the half storage of a typed path.
pub struct GpuMambaMlpBranchTyped {
    pub up: DtypedBuf,
    pub act: DtypedBuf,
    /// The rows and the MLP width the buffers were built for.
    shape: (usize, usize),
}

impl GpuMambaMlpBranchTyped {
    /// Buffers for an MLP `hidden` wide over `rows` rows in `dtype`.
    pub fn new(
        stream: &Stream,
        rows: usize,
        hidden: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        Ok(Self {
            up: DtypedBuf::zeros(stream, rows * 2 * hidden, dtype)?,
            act: DtypedBuf::zeros(stream, rows * hidden, dtype)?,
            shape: (rows, hidden),
        })
    }
}

/// The rows and the MLP width a path's branch buffers were built for.
pub(crate) trait BranchShape {
    fn shape(&self) -> (usize, usize);
}

impl BranchShape for GpuMambaMlpBranchBufs {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

impl BranchShape for GpuMambaMlpBranchTyped {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

/// A path's branch buffers when they were built for these rows and this
/// MLP width; any others would be indexed past their end.
pub(crate) fn branch_bufs<'a, B: BranchShape>(
    bufs: Option<&'a mut B>,
    (rows, hidden): (usize, usize),
    what: &str,
) -> Result<&'a mut B, String> {
    match bufs {
        Some(b) if b.shape() == (rows, hidden) => Ok(b),
        other => Err(format!(
            "the {what} has buffers for the gated MLP of shape {:?}, the model needs \
             ({rows}, {hidden}) (build it for the model's MLP)",
            other.map(|b| b.shape())
        )),
    }
}

/// One layer's gated MLP where the mixer's output `branch` is still to be
/// added to the f32 `stream`, as the decode and the prefill keep it: the
/// norm adds it and normalizes the sum into `branch`, and the MLP's output
/// lands in `branch`, pending in turn for the next norm or norm_f. The rows
/// are f32 here: the projections dispatch on the weights' dtype.
pub(crate) fn pending_f32(
    ctx: &GpuCtx,
    w: MambaMlpPtrs,
    (stream, branch, rms): (DevPtr, &mut GpuBuffer, DevPtr),
    bufs: &mut GpuMambaMlpBranchBufs,
    (rows, dm, eps): (usize, usize, f32),
) -> Result<(), String> {
    let h = w.hidden;
    pending_norm(
        ctx,
        (stream, branch.cached_ptr(), rms, mlp_norm_params(&w)),
        (rows, dm, eps),
        WeightDtype::F32,
    )?;
    gpu_gemm_forward_dispatch(
        ctx,
        &mut bufs.up,
        branch,
        w.fc1_w.0,
        w.fc1_w.1,
        None,
        (rows, dm, 2 * h),
    )?;
    gate_fwd(
        ctx,
        (bufs.act.cached_ptr(), bufs.up.cached_ptr()),
        (rows, h),
        WeightDtype::F32,
    )?;
    gpu_gemm_forward_dispatch(
        ctx,
        branch,
        &bufs.act,
        w.fc2_w.0,
        w.fc2_w.1,
        None,
        (rows, h, dm),
    )
}

/// [`pending_f32`] with the branch, the up projection and the activation in
/// the half storage `dtype`, the stream staying f32.
pub(crate) fn pending_typed(
    ctx: &GpuCtx,
    w: MambaMlpPtrs,
    (stream, branch, rms): (DevPtr, DevPtr, DevPtr),
    bufs: &GpuMambaMlpBranchTyped,
    (rows, dm, eps): (usize, usize, f32),
    dtype: WeightDtype,
) -> Result<(), String> {
    let h = w.hidden;
    let (up, act) = (bufs.up.cached_ptr(), bufs.act.cached_ptr());
    pending_norm(
        ctx,
        (stream, branch, rms, mlp_norm_params(&w)),
        (rows, dm, eps),
        dtype,
    )?;
    let typed = |ptr| TypedPtr { ptr, dtype };
    let weight = |(ptr, dtype): (DevPtr, WeightDtype)| TypedPtr { ptr, dtype };
    gpu_gemm_typed_forward_raw(
        ctx,
        typed(up),
        typed(branch),
        weight(w.fc1_w),
        None,
        (rows, dm, 2 * h),
    )?;
    gate_fwd(ctx, (act, up), (rows, h), dtype)?;
    gpu_gemm_typed_forward_raw(
        ctx,
        typed(branch),
        typed(act),
        weight(w.fc2_w),
        None,
        (rows, h, dm),
    )
}

/// `stream += branch`, then the norm of the sum into `branch` (the kernel
/// reads each element before it stores over it).
fn pending_norm(
    ctx: &GpuCtx,
    (stream, branch, rms, norm): (DevPtr, DevPtr, DevPtr, NormParams),
    (rows, dm, eps): (usize, usize, f32),
    dtype: WeightDtype,
) -> Result<(), String> {
    block_norm::forward_resadd(
        ctx,
        dtype,
        (branch, rms, stream, branch),
        norm,
        (rows, dm, eps),
        "MLP norm",
    )
}

/// One layer's gated MLP with nothing saved, f32, the target forward's: the
/// norm of `rows_buf` into `out`, the up projection, the gate and the down
/// projection back into `out`, then added to `rows_buf`.
pub(crate) fn target_f32(
    ctx: &GpuCtx,
    w: &GpuMambaTrainMlpWeights,
    (rows_buf, out, rms): (&mut GpuBuffer, &mut GpuBuffer, DevPtr),
    bufs: &mut GpuMambaMlpBranchBufs,
    (rows, dm, eps): (usize, usize, f32),
) -> Result<(), String> {
    let (up, act) = (&mut bufs.up, &mut bufs.act);
    let h = w.hidden();
    block_norm::forward(
        ctx,
        WeightDtype::F32,
        (out.cached_ptr(), rms, rows_buf.cached_ptr()),
        w.norm(),
        (rows, dm, eps),
        "MLP norm",
    )?;
    gpu_gemm_bi_forward_raw(ctx, up, out, w.fc1_w.cached_ptr(), None, (rows, dm, 2 * h))?;
    gate_fwd(
        ctx,
        (act.cached_ptr(), up.cached_ptr()),
        (rows, h),
        WeightDtype::F32,
    )?;
    gpu_gemm_bi_forward_raw(ctx, out, act, w.fc2_w.cached_ptr(), None, (rows, h, dm))?;
    launch_elems(
        &ctx.stream,
        &ctx.kernels.vec_add_inplace,
        &[rows_buf.cached_ptr(), out.cached_ptr()],
        rows * dm,
        format_args!("MLP residual"),
    )
}
