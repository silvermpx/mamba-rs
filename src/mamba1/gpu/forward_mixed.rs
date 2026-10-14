//! Mixed-precision (bf16/f16) training activations for Mamba SSM.
//!
//! Mirrors [`super::forward::GpuMambaBackboneActs`] structure but stores
//! activations in `DtypedBuf` (bf16/f16/f32) where appropriate. Tensors that
//! must stay f32 for numerical stability (recurrent state, reduction stats,
//! residual stream) keep their `GpuBuffer` (f32) type — matching PyTorch AMP
//! convention and state-spaces/mamba's `scan_t = float2` invariant.
//!
//! Precision rules:
//! - **f32 always**: `residual` (residual stream), `rms_vals`/`norm_f_rms`
//!   (reduction stats), `conv_states` (recurrent state), `h_saved` (BPTT
//!   carry).
//! - **typed (bf16/f16/f32)**: `post_norm`, `proj`,
//!   `u`, `xdbl`, `delta_raw`, `delta`, `y`, `gated`,
//!   `input_proj_inputs`, `input_proj_outputs`, `norm_f_input` — these are
//!   GEMM I/O or elementwise activations where bf16 storage is safe.
//!
//! Reference: `docs/training-mixed-precision.md` for the full design.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use crate::bidi::{opens_pair, reads_reversed};
use crate::gpu::bidi::{WindowRows, reverse_rows, zero_bytes};
use crate::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use crate::gpu::block_norm;
use crate::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::gpu::context::GpuCtx;
use crate::gpu::dtype::WeightDtype;
use crate::gpu::launch::{grid_1d_cover, grid_1d_wide, launch_cast, y_runs};
use crate::mamba1::gpu::forward::{GpuMambaDims, GpuRecurrentState};
use crate::mamba1::gpu::mixer_norms::launch_bcdt_norm;
use crate::mamba1::gpu::weights::{
    GpuMambaMixedWeights, MambaLayerWeightsView, MambaWeightsView, final_norm_params,
    layer_norm_params,
};
use crate::mamba1::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;

/// Mixed-precision saved activations for one Mamba layer.
pub struct GpuMambaLayerMixedActs {
    /// f32 — pre-norm input saved for residual add `[B*T*d_model]`.
    pub residual: GpuBuffer,
    /// f32 — RMS scalar per (b,t) `[B*T]`.
    pub rms_vals: GpuBuffer,
    /// typed — post-norm output `[B*T*d_model]`.
    pub post_norm: DtypedBuf,
    /// typed — the in_proj output `[B*T*2*d_inner]`, saved whole: the
    /// conv reads its x half and the gating its gate half through a row
    /// stride, and the backward reconstructs the conv windows from the x
    /// half, so nothing splits it.
    pub proj: DtypedBuf,
    /// Carry-in window only, `[B*d_inner*d_conv]`.
    pub conv_states: GpuBuffer,
    /// typed — post-SiLU conv output, SSM input `[B*T*d_inner]`.
    pub u: DtypedBuf,
    /// typed — x_proj output (dt_raw, B, C concatenated) `[B*T*xdbl_dim]`.
    pub xdbl: DtypedBuf,
    /// typed — delta before softplus `[B*T*d_inner]`.
    pub delta_raw: DtypedBuf,
    /// typed — delta after softplus `[B*T*d_inner]`.
    pub delta: DtypedBuf,
    /// f32 — hidden state saved BEFORE each step (T+1 entries)
    /// `[B*(T+1)*d_inner*d_state]`. STAYS f32 (BPTT recurrence).
    pub h_saved: GpuBuffer,
    /// typed — SSM output before gating `[B*T*d_inner]`.
    pub y: DtypedBuf,
    /// typed — gated output `y * gate_silu` `[B*T*d_inner]`.
    pub gated: DtypedBuf,
    /// f32 — FalconMamba's dt, B and C norm scales `[B*T*3]`; empty for
    /// Mamba-1.
    pub bcdt_rstd: GpuBuffer,
}

/// Full Mamba backbone mixed-precision activations.
pub struct GpuMambaBackboneMixedActs {
    /// typed — input to input_proj `[B*T*mamba_input_dim]`.
    pub input_proj_inputs: DtypedBuf,
    /// typed — output of input_proj `[B*T*d_model]`.
    pub input_proj_outputs: DtypedBuf,
    /// Per-layer activations.
    pub layers: Vec<GpuMambaLayerMixedActs>,
    /// f32 — pre-norm_f input for backward `[B*T*d_model]` (residual stream).
    pub norm_f_input: GpuBuffer,
    /// f32 — RMS values per sample for norm_f backward `[B*T]`.
    pub norm_f_rms: GpuBuffer,
    /// Element dtype of all `DtypedBuf` fields.
    pub dtype: WeightDtype,
    /// The gated MLP's saves, for a model with one (see [`Self::with_mlp`]).
    pub mlp: Option<super::mlp::GpuMambaMlpMixedBufs>,
}

impl GpuMambaBackboneMixedActs {
    /// The same activations with the gated MLP's saves, for a model whose
    /// MLP is `hidden` wide (0 leaves them out).
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        hidden: usize,
    ) -> Result<Self, String> {
        let dtype = self.dtype;
        self.mlp = (hidden > 0)
            .then(|| {
                super::mlp::GpuMambaMlpMixedBufs::new(
                    stream,
                    (dims.bt(), dims.d_model, dims.n_layers),
                    (hidden, dims.block_norm),
                    dtype,
                )
            })
            .transpose()?;
        Ok(self)
    }

    /// Allocate all activation buffers, sized for `dims`. Fires async memsets
    /// on `stream`. Caller MUST `stream.synchronize()` before any default-stream
    /// HtoD upload into one of these buffers; without the sync the upload
    /// can race the memset and lose data.
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let batch = dims.batch;
        let seq_len = dims.seq_len;
        let d_model = dims.d_model;
        let d_inner = dims.d_inner;
        let d_state = dims.d_state;
        let d_conv = dims.d_conv;
        let dt_rank = dims.dt_rank;
        let n_layers = dims.n_layers;
        let mamba_input_dim = dims.mamba_input_dim;
        let bt = batch * seq_len;
        let xdbl_dim = dt_rank + 2 * d_state;
        let stats = bt * dims.block_norm.stats_per_row();

        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaLayerMixedActs {
                    // f32 — recurrent / reduction
                    residual: GpuBuffer::zeros(stream, bt * d_model)?,
                    rms_vals: GpuBuffer::zeros(stream, stats)?,
                    conv_states: GpuBuffer::zeros(stream, batch * d_inner * d_conv)?,
                    h_saved: GpuBuffer::zeros(
                        stream,
                        if dims.scan_mode.use_parallel(seq_len, d_state)
                            && crate::gpu::launch::scan_tape_slim()
                        {
                            crate::gpu::launch::scan_tape_len(batch, seq_len, d_inner, d_state)
                        } else {
                            batch * (seq_len + 1) * d_inner * d_state
                        },
                    )?,
                    // typed — GEMM I/O / elementwise
                    post_norm: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
                    proj: DtypedBuf::zeros(stream, bt * 2 * d_inner, dtype)?,
                    u: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    xdbl: DtypedBuf::zeros(stream, bt * xdbl_dim, dtype)?,
                    delta_raw: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    delta: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    y: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    gated: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    bcdt_rstd: GpuBuffer::zeros(
                        stream,
                        if dims.bcdt_rms_eps.is_some() {
                            bt * 3
                        } else {
                            0
                        },
                    )?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let s = Self {
            input_proj_inputs: DtypedBuf::zeros(stream, bt * mamba_input_dim, dtype)?,
            input_proj_outputs: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, stats)?,
            dtype,
            mlp: None,
        };

        // Enforce the memset-before-upload invariant: every later
        // HtoD path that targets these buffers must observe the zeros first.
        stream
            .synchronize()
            .map_err(|e| format!("sync after mixed acts alloc: {e:?}"))?;

        Ok(s)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Mixed-precision training scratch (typed I/O between layer kernels).
// ════════════════════════════════════════════════════════════════════════

/// Per-step scratch for mixed-precision training. Pre-allocated once at
/// trainer construction; reused every forward+backward pass — zero alloc on
/// hot path (**required** for CUDA Graph capture: stable pointers, no resize
/// between captures). Typed buffers carry activation gradients; f32 buffers
/// hold BPTT accumulators and the residual-stream gradient.
pub struct GpuMambaMixedTrainScratch {
    pub dims: GpuMambaDims,
    pub dtype: WeightDtype,

    // ── Forward scratch ───────────────────────────────────────────────
    /// dt gather buffer [B*T * dt_rank].
    pub dt_gather: DtypedBuf,
    /// B gather buffer [B*T * d_state].
    pub b_buf: DtypedBuf,
    /// C gather buffer [B*T * d_state].
    pub c_buf: DtypedBuf,
    /// out_proj output [B*T * d_model].
    pub out_flat: DtypedBuf,
    /// A bidirectional pair's f32 rows `[B*T*d_model]`: in the forward the
    /// stream after the pair's first mixer, in the backward d loss / d (the
    /// pair's output) held aside for the first layer (one element without
    /// pairs).
    pub pair: GpuBuffer,
    /// Final lm_head feed (post norm_f) — typed [B*T * d_model].
    pub temporal_typed: DtypedBuf,

    // ── Backward scratch — typed activation grads ────────────────────
    /// out_proj dX [B*T * d_inner].
    pub d_gated: DtypedBuf,
    /// gating dY [B*T * d_inner] — feeds `ssm_backward_local_typed`.
    pub d_y: DtypedBuf,
    /// ssm_backward_local per-sample d_B buffer [B*T * d_inner * d_state].
    pub d_b_local: DtypedBuf,
    /// ssm_backward_local per-sample d_C buffer [B*T * d_inner * d_state].
    pub d_c_local: DtypedBuf,
    /// ssm_backward_local d_delta output [B*T * d_inner].
    pub d_delta: DtypedBuf,
    /// ssm_backward_local d_u output [B*T * d_inner] — later += d_u_xproj.
    pub d_u: DtypedBuf,
    /// x_proj dX [B*T * d_inner].
    pub d_u_xproj: DtypedBuf,
    /// softplus_bwd dx [B*T * d_inner].
    pub d_delta_raw: DtypedBuf,
    /// Gathered dt slice from xdbl for dt_proj dW [B*T * dt_rank].
    pub dt_xdbl_buf: DtypedBuf,
    /// dt_proj dX [B*T * dt_rank].
    pub d_dt_input: DtypedBuf,
    /// d_xdbl accumulator [B*T * xdbl_dim] — feeds x_proj dW.
    pub d_xdbl: DtypedBuf,
    /// in_proj output gradient [B*T * 2*d_inner], the conv and gating
    /// backwards writing their halves in place — feeds in_proj dW.
    pub d_proj: DtypedBuf,
    /// in_proj dX [B*T * d_model] — feeds per-layer `rmsnorm_bwd_f32in`.
    pub d_norm: DtypedBuf,

    // ── Backward scratch — f32 master grads / accumulators ────────────
    /// Reduced d_B master grad [B*T * d_state].
    pub d_b_reduced: GpuBuffer,
    /// Reduced d_C master grad [B*T * d_state].
    pub d_c_reduced: GpuBuffer,
    /// Per-sample d_D accumulator [B * d_inner].
    pub d_d_local: GpuBuffer,
    /// Per-sample d_a_log accumulator [B * d_inner * d_state].
    pub d_a_log_local: GpuBuffer,
    /// Discarded dx for backbone input_proj backward [B*T * mamba_input_dim].
    pub d_input_proj_dx: GpuBuffer,

    // ── Rule-B axis-0 reduction partials (determinism fix) ─
    /// Scratch for per-sample partials produced by Rule-B backward kernels
    /// (rmsnorm_bwd d_scale, the conv backward's tap and bias partials). Reduced via
    /// `reduce_sum_axis0` to produce deterministic cross-batch accumulators
    /// (replaces the previous atomicAdd accumulators).
    ///
    /// Sized to the max of the two consumer requirements:
    ///   - rmsnorm_bwd partials:      B*T * d_model
    ///   - conv1d_bwd partials:       B * d_inner * (d_conv + 1)
    pub axis0_partials: GpuBuffer,
    /// The conv backward's pre-activation gradients `[B*T*d_inner]` for a
    /// conv wider than the tiled kernels hold (one element otherwise).
    pub conv_dco: GpuBuffer,
    /// The gated MLP backward's buffers, for a model with one (see
    /// [`Self::with_mlp`]).
    pub mlp: Option<super::mlp::GpuMambaMlpMixedBwdBufs>,
}

impl GpuMambaMixedTrainScratch {
    /// The same scratch with the gated MLP backward's buffers, for a model
    /// whose MLP is `hidden` wide (0 leaves them out).
    pub fn with_mlp(
        mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        hidden: usize,
    ) -> Result<Self, String> {
        let (dims, dtype) = (self.dims, self.dtype);
        self.mlp = (hidden > 0)
            .then(|| {
                super::mlp::GpuMambaMlpMixedBwdBufs::new(
                    stream,
                    (dims.bt(), dims.d_model),
                    hidden,
                    dtype,
                )
            })
            .transpose()?;
        Ok(self)
    }

    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let bt = dims.batch * dims.seq_len;
        // The fold backward writes dB/dC partials at di/G rows; only the
        // non-fold routes (sequential scan, or d_inner not divisible by
        // the d-group) need the full depth. Same predicate as the launcher.
        let bc_rows = if dims.scan_mode.use_parallel(dims.seq_len, dims.d_state)
            && dims
                .d_inner
                .is_multiple_of(crate::gpu::launch::SCAN_BWD_DGROUP)
        {
            dims.d_inner / crate::gpu::launch::SCAN_BWD_DGROUP
        } else {
            dims.d_inner
        };
        let xdbl_dim = dims.dt_rank + 2 * dims.d_state;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let dm = dims.d_model;
        let b = dims.batch;
        let s = Self {
            dims: *dims,
            dtype,
            // forward
            dt_gather: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            b_buf: DtypedBuf::zeros(stream, bt * ds, dtype)?,
            c_buf: DtypedBuf::zeros(stream, bt * ds, dtype)?,
            out_flat: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            temporal_typed: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            pair: GpuBuffer::zeros(stream, if dims.bidirectional { bt * dm } else { 1 })?,
            // backward typed
            d_gated: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_y: DtypedBuf::zeros(stream, bt * di, dtype)?,
            // The fold backward writes dB/dC partials at di/G rows; only
            // the non-fold routes (d_inner not divisible by the d-group)
            // need the full depth. Same predicate as the launcher.
            d_b_local: DtypedBuf::zeros(stream, bt * bc_rows * ds, dtype)?,
            d_c_local: DtypedBuf::zeros(stream, bt * bc_rows * ds, dtype)?,
            d_delta: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_u: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_u_xproj: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_delta_raw: DtypedBuf::zeros(stream, bt * di, dtype)?,
            dt_xdbl_buf: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            d_dt_input: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            d_xdbl: DtypedBuf::zeros(stream, bt * xdbl_dim, dtype)?,
            d_proj: DtypedBuf::zeros(stream, bt * 2 * di, dtype)?,
            d_norm: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            // backward f32
            d_b_reduced: GpuBuffer::zeros(stream, bt * ds)?,
            d_c_reduced: GpuBuffer::zeros(stream, bt * ds)?,
            d_d_local: GpuBuffer::zeros(stream, b * di)?,
            d_a_log_local: GpuBuffer::zeros(
                stream,
                b * dims.seq_len.div_ceil(crate::gpu::launch::SCAN_CHUNK).max(1) * di * ds,
            )?,
            d_input_proj_dx: GpuBuffer::zeros(stream, bt * dims.mamba_input_dim)?,
            // Rule-B axis-0 partials scratch — sized to fit largest consumer.
            axis0_partials: GpuBuffer::zeros(
                stream,
                std::cmp::max(
                    bt * dm,
                    b * dims.seq_len.div_ceil(128) * di * (dims.d_conv + 1),
                ),
            )?,
            conv_dco: GpuBuffer::zeros(stream, super::conv::conv_dco_len(bt, di, dims.d_conv))?,
            mlp: None,
        };
        // Race-fix invariant (a950648): callers may immediately upload from
        // host via default-stream HtoD; sync first.
        stream
            .synchronize()
            .map_err(|e| format!("sync after mixed train scratch alloc: {e:?}"))?;
        Ok(s)
    }
}

// ════════════════════════════════════════════════════════════════════════
// Mixed-precision training forward backbone (delegates to typed kernels).
// ════════════════════════════════════════════════════════════════════════

/// Mamba SSM backbone forward in mixed precision (bf16/f16/f32 dispatched by
/// `acts.dtype`). Mirrors [`super::forward::gpu_forward_mamba_backbone`] but
/// reads compute weights from [`GpuMambaMixedWeights`] and saves typed
/// activations in [`GpuMambaBackboneMixedActs`].
///
/// Precision conventions:
/// - Residual stream stays f32 across layers (`acts.layers[L].residual`,
///   `acts.norm_f_input`). All intermediate GEMM/elementwise activations
///   are typed.
/// - Recurrent SSM and conv1d state stay f32 (BPTT stability, per
///   state-spaces/mamba `scan_t = float2`).
///
/// Usage: pass `weights.compute()` from a [`GpuMambaTrainMixedWeights`].
pub fn gpu_forward_mamba_backbone_mixed(
    ctx: &GpuCtx,
    acts: &mut GpuMambaBackboneMixedActs,
    mamba_w: &GpuMambaMixedWeights,
    mamba_input: &GpuBuffer, // f32 input embeddings
    state: &mut GpuRecurrentState,
    scratch: &mut GpuMambaMixedTrainScratch,
) -> Result<(), String> {
    assert_eq!(acts.dtype, scratch.dtype, "acts/scratch dtype mismatch");
    assert_eq!(
        acts.dtype, mamba_w.bulk_dtype,
        "acts dtype must match weights bulk_dtype"
    );

    let dims = scratch.dims;
    dims.check_for(ctx)?;
    let bt = dims.batch * dims.seq_len;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dt_rank + 2 * ds;
    let b = dims.batch;
    let t = dims.seq_len;
    let d_conv = dims.d_conv;
    let dt = acts.dtype;
    let k = &ctx.kernels;
    super::mlp::check_width(
        mamba_w.mlp.first().map(|m| m.hidden),
        acts.mlp.as_ref().map(|m| m.hidden),
        "activations",
    )?;

    // ─── Input projection (or identity-proj for HF Mamba) ───
    // For from_cpu loaded weights with zero-len input_proj_w, treat as identity:
    // residual_f32 := f32 input directly (D2D copy, no cast needed).
    // Otherwise: cast f32 input → typed (saved for backward), GEMM, then
    // residual_f32 stream is seeded by an extra residual_add at layer 0
    // (we re-cast typed input_proj_outputs back to f32 since norm_f path needs
    // the residual stream in f32 always — see AMP residual_in_fp32).
    if mamba_w.input_proj_w.len_elems() == 0 {
        // Identity-proj. Skip any save (input_proj_inputs/outputs unused in
        // backward when input_proj is identity).
        let bytes = bt * dm * 4;
        let res = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                acts.layers[0].residual.cached_ptr(),
                mamba_input.cached_ptr(),
                bytes,
                ctx.stream.cu_stream(),
            )
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("identity proj D2D failed: {res:?}"));
        }
    } else {
        // Non-identity input_proj — the trainable patch-embed path. Three
        // launches: (1) cast the f32 input to the compute dtype, SAVED in
        // acts (the backward dW consumes it); (2) typed GEMM with the f32
        // bias into the saved typed outputs; (3) upcast the outputs to seed
        // the f32 residual stream (AMP residual_in_fp32 — norm math and
        // every residual carry stay f32).
        let mid = dims.mamba_input_dim;
        {
            let cast = match dt {
                WeightDtype::Bf16 => &k.cast_f32_to_bf16,
                WeightDtype::F16 => &k.cast_f32_to_f16,
                WeightDtype::F32 | WeightDtype::Tf32 => {
                    return Err("mixed forward: unexpected f32 compute dtype".into());
                }
            };
            launch_cast(
                &ctx.stream,
                cast,
                (acts.input_proj_inputs.cached_ptr(), dt.size_bytes()),
                (mamba_input.cached_ptr(), 4),
                bt * mid,
                format_args!("input_proj input cast"),
            )?;
        }
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: acts.input_proj_outputs.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: acts.input_proj_inputs.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: mamba_w.input_proj_w.ptr(),
                dtype: dt,
            },
            Some(mamba_w.input_proj_b.ptr()),
            (bt, mid, dm),
        )?;
        {
            let cast = match dt {
                WeightDtype::Bf16 => &k.cast_bf16_to_f32,
                WeightDtype::F16 => &k.cast_f16_to_f32,
                WeightDtype::F32 | WeightDtype::Tf32 => {
                    return Err("mixed forward: unexpected f32 compute dtype".into());
                }
            };
            launch_cast(
                &ctx.stream,
                cast,
                (acts.layers[0].residual.cached_ptr(), 4),
                (acts.input_proj_outputs.cached_ptr(), dt.size_bytes()),
                bt * dm,
                format_args!("input_proj residual upcast"),
            )?;
        }
    }

    // ─── Per-layer offsets into flat state buffers ───
    let conv_per_layer = b * di * d_conv;
    let ssm_per_layer = b * di * ds;
    let a_neg_per_layer = di * ds;
    let f32_sz = std::mem::size_of::<f32>() as u64;
    let conv_base = state.conv_states.cached_ptr();
    let ssm_base = state.ssm_states.cached_ptr();
    let aneg_base = state.a_neg_all.cached_ptr();
    let bidi = dims.bidirectional;
    // A bidirectional model reads whole sequences: every window starts
    // from the empty state.
    if bidi {
        for buf in [&state.conv_states, &state.ssm_states] {
            zero_bytes(ctx, buf.cached_ptr(), buf.len() * 4)?;
        }
    }
    let pair_rows = scratch.pair.cached_ptr();
    let window = |ptr, elem_bytes| WindowRows {
        ptr,
        elem_bytes,
        batch: b,
        seq_len: t,
        width: dm,
    };

    for layer_idx in 0..dims.n_layers {
        let conv_ptr = conv_base + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = ssm_base + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let aneg_ptr = aneg_base + (layer_idx * a_neg_per_layer) as u64 * f32_sz;
        let lw = &mamba_w.layers[layer_idx];
        // A bidirectional pair: the first mixer's sum is the pair's working
        // rows; the second runs on its reversed input, the pair's input the
        // first layer's slot holds, and adds onto them, its output turned
        // back. The pair's MLPs follow both mixers, the first MLP's output
        // the second's input.
        let reversed = reads_reversed(bidi, layer_idx);
        if reversed {
            let (before, rest) = acts.layers.split_at_mut(layer_idx);
            rest[0]
                .residual
                .copy_from(&before[layer_idx - 1].residual, &ctx.stream)?;
            reverse_rows(ctx, window(rest[0].residual.cached_ptr(), 4), 0)?;
        }
        let layer_acts = &mut acts.layers[layer_idx];

        // F1: the block norm — read f32 residual, write typed post_norm.
        // Side effect: its f32 statistics saved for backward.
        block_norm::forward(
            ctx,
            dt,
            (
                layer_acts.post_norm.cached_ptr(),
                layer_acts.rms_vals.cached_ptr(),
                layer_acts.residual.cached_ptr(),
            ),
            layer_norm_params(lw),
            (bt, dm, dims.rms_norm_eps),
            "block norm mixed",
        )?;

        // F2: in_proj GEMM typed — [B*T, dm] -> [B*T, 2*di], saved whole;
        // the conv and the gating read their halves through the row stride.
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: layer_acts.proj.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: layer_acts.post_norm.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: lw.in_proj_w.ptr(),
                dtype: dt,
            },
            lw.in_proj_b(),
            (bt, dm, 2 * di),
        )?;

        // F4a: the conv with fused SiLU, typed I/O, f32 state save; x is
        // the first half of the in_proj output.
        super::conv::conv_forward(
            ctx,
            dt,
            &super::conv::ConvForward {
                u: layer_acts.u.cached_ptr(),
                u_t: (0, 0),
                state: conv_ptr,
                saved: layer_acts.conv_states.cached_ptr(),
                x: (layer_acts.proj.cached_ptr(), 2 * di),
                weight: lw.conv1d_weight.ptr(),
                bias: lw.conv1d_bias(),
            },
            (b, t, di, d_conv),
            format_args!("training L{layer_idx}"),
        )?;

        // F4b: x_proj GEMM typed, and FalconMamba's dt, B and C norms with
        // the scales kept for the backward.
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: layer_acts.xdbl.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: layer_acts.u.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: lw.x_proj_w.ptr(),
                dtype: dt,
            },
            None,
            (bt, di, xdbl_dim),
        )?;
        if let Some(eps) = dims.bcdt_rms_eps {
            launch_bcdt_norm(
                ctx,
                (
                    layer_acts.xdbl.cached_ptr(),
                    layer_acts.bcdt_rstd.cached_ptr(),
                ),
                dt,
                (bt, dt_rank, dims.d_state),
                eps,
            )?;
        }

        // F4c: gather_cols_typed (dt slice) + dt_proj GEMM typed + softplus_copy_typed.
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let dt_i = dt_rank as i32;
            let offset: i32 = 0;
            let mut bld = ctx.stream.launch_builder(k.gather_cols_typed.get(dt));
            let dg = scratch.dt_gather.cached_ptr();
            let xd = layer_acts.xdbl.cached_ptr();
            bld.arg(&dg);
            bld.arg(&xd);
            bld.arg(&bt_i);
            bld.arg(&xdbl_i);
            bld.arg(&dt_i);
            bld.arg(&offset);
            unsafe { bld.launch(grid_1d_wide(bt * dt_rank)) }
                .map_err(|e| format!("gather_cols dt typed L{layer_idx}: {e:?}"))?;
        }
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: layer_acts.delta_raw.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: scratch.dt_gather.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: lw.dt_proj_w.ptr(),
                dtype: dt,
            },
            Some(lw.dt_proj_b.ptr()),
            (bt, dt_rank, di),
        )?;
        // Softplus is fused into the scan kernels below: they read
        // delta_raw, apply softplus inline (through the exact store
        // rounding the deleted copy pass produced) and write the
        // post-softplus save the backward replays from.

        // F4d: gather_bc_cols_typed → b_buf, c_buf; ssm_burnin_forward typed.
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_off = dt_rank as i32;
            let c_off = (dt_rank + ds) as i32;
            // Parallel route gathers T-major (see the f32 twin note).
            let tmajor = dims.scan_mode.use_parallel(t, ds);
            let kernel = if tmajor {
                k.gather_bc_cols_tmajor_typed.get(dt)
            } else {
                k.gather_bc_cols_typed.get(dt)
            };
            let t_i = t as i32;
            let mut bld = ctx.stream.launch_builder(kernel);
            let bb = scratch.b_buf.cached_ptr();
            let cb = scratch.c_buf.cached_ptr();
            let xd = layer_acts.xdbl.cached_ptr();
            bld.arg(&bb);
            bld.arg(&cb);
            bld.arg(&xd);
            bld.arg(&bt_i);
            if tmajor {
                bld.arg(&t_i);
            }
            bld.arg(&xdbl_i);
            bld.arg(&ds_i);
            bld.arg(&b_off);
            bld.arg(&c_off);
            unsafe { bld.launch(grid_1d_wide(bt * ds)) }
                .map_err(|e| format!("gather_bc_cols typed L{layer_idx}: {e:?}"))?;
        }
        // SSM forward: parallel prefix scan for T > PARALLEL_SCAN_THRESHOLD
        // or ds > 64 (matches f32 path dispatch in forward.rs:544). Typed
        // variants keep scan state + h_saved + smem f32
        // per state-spaces/mamba `scan_t = float2` invariant; only
        // delta/u/B/C/y are typed.
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            if dims.scan_mode.use_parallel(t, ds) {
                let slim = crate::gpu::launch::scan_tape_slim();
                let width = crate::gpu::launch::scan_fwd_index_width(b, t, di, ds, slim);
                let kernel = k.ssm_parallel_fwd_typed.get(width).get(dt);
                for (y_first, cfg) in y_runs(crate::gpu::launch::grid_parallel_scan(b, di, ds)) {
                    let mut bld = ctx.stream.launch_builder(kernel);
                    let y = layer_acts.y.cached_ptr();
                    let hs = layer_acts.h_saved.cached_ptr();
                    let draw = layer_acts.delta_raw.cached_ptr();
                    let dsave = layer_acts.delta.cached_ptr();
                    let u = layer_acts.u.cached_ptr();
                    let bb = scratch.b_buf.cached_ptr();
                    let cb = scratch.c_buf.cached_ptr();
                    let dp = lw.d_param.ptr();
                    bld.arg(&ssm_ptr);
                    bld.arg(&y);
                    bld.arg(&hs);
                    bld.arg(&draw);
                    bld.arg(&dsave);
                    bld.arg(&u);
                    bld.arg(&bb);
                    bld.arg(&cb);
                    bld.arg(&aneg_ptr);
                    bld.arg(&dp);
                    bld.arg(&b_i);
                    bld.arg(&t_i);
                    bld.arg(&di_i);
                    bld.arg(&ds_i);
                    let slim_i: i32 = i32::from(slim);
                    bld.arg(&hs);
                    bld.arg(&slim_i);
                    bld.arg(&y_first);
                    unsafe { bld.launch(cfg) }
                        .map_err(|e| format!("ssm_parallel_fwd typed L{layer_idx}: {e:?}"))?;
                }
            } else {
                if ds > k.state_cap {
                    return Err(format!(
                        "ssm_burnin_forward_typed: d_state {ds} exceeds the compiled \
                     state capacity {}",
                        k.state_cap
                    ));
                }
                let width = crate::gpu::launch::seq_scan_index_width(b, t, di, ds, di, true);
                let kernel = k.ssm_burnin_fwd_typed.get(width).get(dt);
                let mut bld = ctx.stream.launch_builder(kernel);
                let y = layer_acts.y.cached_ptr();
                let hs = layer_acts.h_saved.cached_ptr();
                let draw = layer_acts.delta_raw.cached_ptr();
                let dsave = layer_acts.delta.cached_ptr();
                let u = layer_acts.u.cached_ptr();
                let bb = scratch.b_buf.cached_ptr();
                let cb = scratch.c_buf.cached_ptr();
                let dp = lw.d_param.ptr();
                bld.arg(&ssm_ptr);
                bld.arg(&y);
                bld.arg(&hs);
                bld.arg(&draw);
                bld.arg(&dsave);
                bld.arg(&u);
                bld.arg(&bb);
                bld.arg(&cb);
                bld.arg(&aneg_ptr);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid_1d_cover(b * di)?) }
                    .map_err(|e| format!("ssm_burnin_forward typed L{layer_idx}: {e:?}"))?;
            }
        }

        // F4e: gating — gated = y * SiLU(gate), the gate read from the
        // second half of the in_proj output through the row stride. The
        // vector twin needs its width to divide d_inner so no vector
        // straddles a row.
        {
            let g = layer_acts.gated.cached_ptr();
            let y = layer_acts.y.cached_ptr();
            let gp = layer_acts.proj.cached_ptr();
            let w = crate::gpu::launch::vec8_width(dt.size_bytes());
            let vec = di.is_multiple_of(w)
                && crate::gpu::launch::vec8_ok(bt * di, dt.size_bytes(), &[g, y, gp]);
            let (kern, count) = if vec {
                (k.gate_mul_silu_v_typed.get(dt), bt * di / w)
            } else {
                (k.gate_mul_silu_typed.get(dt), bt * di)
            };
            let n = count as i64;
            let di_i = di as i32;
            let gate_stride = (2 * di) as i32;
            let gate_off = di as i32;
            let mut bld = ctx.stream.launch_builder(kern);
            bld.arg(&g);
            bld.arg(&y);
            bld.arg(&gp);
            bld.arg(&n);
            bld.arg(&di_i);
            bld.arg(&gate_stride);
            bld.arg(&gate_off);
            unsafe { bld.launch(grid_1d_wide(count)) }
                .map_err(|e| format!("gate_mul_silu_typed L{layer_idx}: {e:?}"))?;
        }

        // F5: out_proj GEMM typed → scratch.out_flat [B*T, d_model].
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.out_flat.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: layer_acts.gated.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: lw.out_proj_w.ptr(),
                dtype: dt,
            },
            lw.out_proj_b(),
            (bt, di, dm),
        )?;

        // F6: residual_add_f32 typed — residual_f32 += out_flat_typed (f32 stays).
        // Output destination = next layer's residual buffer (or norm_f_input
        // if this is the last layer).
        let next_res_ptr = if layer_idx + 1 < dims.n_layers {
            acts.layers[layer_idx + 1].residual.cached_ptr()
        } else {
            acts.norm_f_input.cached_ptr()
        };
        // With an MLP the mixer's sum lands in the MLP's input rows, and the
        // MLP writes the layer's output.
        let mixer_out = if opens_pair(bidi, layer_idx) {
            pair_rows
        } else {
            let first_mlp = if reversed { layer_idx - 1 } else { layer_idx };
            match &acts.mlp {
                Some(m) => m.layers[first_mlp].x_in.cached_ptr(),
                None => next_res_ptr,
            }
        };
        let residual = if reversed {
            reverse_rows(
                ctx,
                window(scratch.out_flat.cached_ptr(), dt.size_bytes()),
                0,
            )?;
            pair_rows
        } else {
            acts.layers[layer_idx].residual.cached_ptr()
        };
        // dst f32, a f32, b typed.
        crate::gpu::launch::launch_elems(
            &ctx.stream,
            k.residual_add_f32_typed.get(dt),
            &[mixer_out, residual, scratch.out_flat.cached_ptr()],
            bt * dm,
            format_args!("residual_add_f32_typed L{layer_idx}"),
        )?;
        let shape = (bt, dm, dims.rms_norm_eps);
        if reversed && let (Some(m), Some(w)) = (&acts.mlp, mamba_w.mlp(layer_idx - 1)) {
            super::mlp::forward_mixed(
                ctx,
                m.layers[layer_idx].x_in.cached_ptr(),
                (&m.layers[layer_idx - 1], w),
                scratch.out_flat.cached_ptr(),
                shape,
                dt,
            )?;
        }
        if !opens_pair(bidi, layer_idx)
            && let (Some(m), Some(w)) = (&acts.mlp, mamba_w.mlp(layer_idx))
        {
            super::mlp::forward_mixed(
                ctx,
                next_res_ptr,
                (&m.layers[layer_idx], w),
                scratch.out_flat.cached_ptr(),
                shape,
                dt,
            )?;
        }
    }

    // ─── Final norm_f: residual_f32 → temporal_typed.
    block_norm::forward(
        ctx,
        dt,
        (
            scratch.temporal_typed.cached_ptr(),
            acts.norm_f_rms.cached_ptr(),
            acts.norm_f_input.cached_ptr(),
        ),
        final_norm_params(mamba_w),
        (bt, dm, dims.rms_norm_eps),
        "norm_f mixed",
    )
}

/// Convenience overload: build the bf16 mixed weights from a
/// [`GpuMambaTrainMixedWeights`] and dispatch.
pub fn gpu_forward_mamba_backbone_train_mixed(
    ctx: &GpuCtx,
    acts: &mut GpuMambaBackboneMixedActs,
    train_w: &GpuMambaTrainMixedWeights,
    mamba_input: &GpuBuffer,
    state: &mut GpuRecurrentState,
    scratch: &mut GpuMambaMixedTrainScratch,
) -> Result<(), String> {
    gpu_forward_mamba_backbone_mixed(ctx, acts, &train_w.compute, mamba_input, state, scratch)
}

// (typed↔f32 cast helpers are not needed in this path — residual stream is
// seeded directly from f32 input via D2D copy, and post-layer residual stays
// in f32 throughout. typed→f32 only matters for inference lm_head, not here.)
