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
//!   carry), `da_exp` (state-derived).
//! - **typed (bf16/f16/f32)**: `post_norm`, `gate_pre_silu`, `gate_post_silu`,
//!   `post_conv`, `u`, `xdbl`, `delta_raw`, `delta`, `y`, `gated`,
//!   `input_proj_inputs`, `input_proj_outputs`, `norm_f_input` — these are
//!   GEMM I/O or elementwise activations where bf16 storage is safe.
//!
//! Reference: `docs/training-mixed-precision.md` for the full design.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use crate::mamba_ssm::gpu::blas::{TypedPtr, gpu_gemm_typed_forward_raw};
use crate::mamba_ssm::gpu::buffers::{DtypedBuf, GpuBuffer};
use crate::mamba_ssm::gpu::context::GpuCtx;
use crate::mamba_ssm::gpu::dtype::WeightDtype;
use crate::mamba_ssm::gpu::forward::{GpuMambaDims, GpuRecurrentState};
use crate::mamba_ssm::gpu::launch::{grid_1d, grid_norm};
use crate::mamba_ssm::gpu::weights::GpuMambaMixedWeights;
use crate::mamba_ssm::gpu::weights_mixed_train::GpuMambaTrainMixedWeights;

/// Mixed-precision saved activations for one Mamba layer.
pub struct GpuMambaLayerMixedActs {
    /// f32 — pre-norm input saved for residual add `[B*T*d_model]`.
    pub residual: GpuBuffer,
    /// f32 — RMS scalar per (b,t) `[B*T]`.
    pub rms_vals: GpuBuffer,
    /// typed — post-norm output `[B*T*d_model]`.
    pub post_norm: DtypedBuf,
    /// typed — gate branch before SiLU `[B*T*d_inner]`.
    pub gate_pre_silu: DtypedBuf,
    /// typed — gate branch after SiLU `[B*T*d_inner]`.
    pub gate_post_silu: DtypedBuf,
    /// f32 — conv1d state saved per step `[B*T*d_inner*d_conv]` (recurrent).
    pub conv_states: GpuBuffer,
    /// typed — pre-SiLU conv output `[B*T*d_inner]`.
    pub post_conv: DtypedBuf,
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
    /// f32 — discretization `exp(delta * A)` `[B*T*d_inner*d_state]`.
    pub da_exp: GpuBuffer,
    /// typed — SSM output before gating `[B*T*d_inner]`.
    pub y: DtypedBuf,
    /// typed — gated output `y * gate_silu` `[B*T*d_inner]`.
    pub gated: DtypedBuf,
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
}

impl GpuMambaBackboneMixedActs {
    /// Allocate all activation buffers, sized for `dims`. Fires async memsets
    /// on `stream`. Caller MUST `stream.synchronize()` before any default-stream
    /// HtoD upload into one of these buffers (see commit a950648 for the race
    /// class this prevents).
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

        let layers = (0..n_layers)
            .map(|_| {
                Ok(GpuMambaLayerMixedActs {
                    // f32 — recurrent / reduction
                    residual: GpuBuffer::zeros(stream, bt * d_model)?,
                    rms_vals: GpuBuffer::zeros(stream, bt)?,
                    conv_states: GpuBuffer::zeros(stream, bt * d_inner * d_conv)?,
                    h_saved: GpuBuffer::zeros(stream, batch * (seq_len + 1) * d_inner * d_state)?,
                    da_exp: GpuBuffer::zeros(stream, bt * d_inner * d_state)?,
                    // typed — GEMM I/O / elementwise
                    post_norm: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
                    gate_pre_silu: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    gate_post_silu: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    post_conv: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    u: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    xdbl: DtypedBuf::zeros(stream, bt * xdbl_dim, dtype)?,
                    delta_raw: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    delta: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    y: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                    gated: DtypedBuf::zeros(stream, bt * d_inner, dtype)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let s = Self {
            input_proj_inputs: DtypedBuf::zeros(stream, bt * mamba_input_dim, dtype)?,
            input_proj_outputs: DtypedBuf::zeros(stream, bt * d_model, dtype)?,
            layers,
            norm_f_input: GpuBuffer::zeros(stream, bt * d_model)?,
            norm_f_rms: GpuBuffer::zeros(stream, bt)?,
            dtype,
        };

        // Enforce the race-fix invariant from commit a950648: every later
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
    /// in_proj output [B*T * 2*d_inner].
    pub proj_flat: DtypedBuf,
    /// split's x branch [B*T * d_inner].
    pub x_branch: DtypedBuf,
    /// dt gather buffer [B*T * dt_rank].
    pub dt_gather: DtypedBuf,
    /// B gather buffer [B*T * d_state].
    pub b_buf: DtypedBuf,
    /// C gather buffer [B*T * d_state].
    pub c_buf: DtypedBuf,
    /// out_proj output [B*T * d_model].
    pub out_flat: DtypedBuf,
    /// Final lm_head feed (post norm_f) — typed [B*T * d_model].
    pub temporal_typed: DtypedBuf,

    // ── Backward scratch — typed activation grads ────────────────────
    /// out_proj dX [B*T * d_inner].
    pub d_gated: DtypedBuf,
    /// gating dY [B*T * d_inner] — feeds `ssm_backward_local_typed`.
    pub d_y: DtypedBuf,
    /// gating d_gate_pre [B*T * d_inner] — feeds `concat_halves_typed`.
    pub d_gate: DtypedBuf,
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
    /// conv1d_burnin_bwd d_x_branch [B*T * d_inner].
    pub d_x_branch: DtypedBuf,
    /// concat(d_x_branch, d_gate) [B*T * 2*d_inner] — feeds in_proj dW.
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
    /// rmsnorm bwd dx (f32 residual stream) [B*T * d_model] — accumulates
    /// into outer `d_temporal` via `vec_add_inplace`.
    pub d_pre_norm: GpuBuffer,
    /// Discarded dx for backbone input_proj backward [B*T * mamba_input_dim].
    pub d_input_proj_dx: GpuBuffer,

    // ── Rule-B axis-0 reduction partials (Phase 2.7.4 determinism fix) ─
    /// Scratch for per-sample partials produced by Rule-B backward kernels
    /// (rmsnorm_bwd d_scale, conv1d_burnin_bwd d_weight+d_bias). Reduced via
    /// `reduce_sum_axis0` to produce deterministic cross-batch accumulators
    /// (replaces the previous atomicAdd accumulators).
    ///
    /// Sized to the max of the two consumer requirements:
    ///   - rmsnorm_bwd partials:      B*T * d_model
    ///   - conv1d_bwd partials:       B * d_inner * (d_conv + 1)
    pub axis0_partials: GpuBuffer,
}

impl GpuMambaMixedTrainScratch {
    pub fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        dims: &GpuMambaDims,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let bt = dims.batch * dims.seq_len;
        let xdbl_dim = dims.dt_rank + 2 * dims.d_state;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let dm = dims.d_model;
        let b = dims.batch;
        let s = Self {
            dims: *dims,
            dtype,
            // forward
            proj_flat: DtypedBuf::zeros(stream, bt * 2 * di, dtype)?,
            x_branch: DtypedBuf::zeros(stream, bt * di, dtype)?,
            dt_gather: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            b_buf: DtypedBuf::zeros(stream, bt * ds, dtype)?,
            c_buf: DtypedBuf::zeros(stream, bt * ds, dtype)?,
            out_flat: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            temporal_typed: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            // backward typed
            d_gated: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_y: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_gate: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_b_local: DtypedBuf::zeros(stream, bt * di * ds, dtype)?,
            d_c_local: DtypedBuf::zeros(stream, bt * di * ds, dtype)?,
            d_delta: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_u: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_u_xproj: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_delta_raw: DtypedBuf::zeros(stream, bt * di, dtype)?,
            dt_xdbl_buf: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            d_dt_input: DtypedBuf::zeros(stream, bt * dims.dt_rank, dtype)?,
            d_xdbl: DtypedBuf::zeros(stream, bt * xdbl_dim, dtype)?,
            d_x_branch: DtypedBuf::zeros(stream, bt * di, dtype)?,
            d_proj: DtypedBuf::zeros(stream, bt * 2 * di, dtype)?,
            d_norm: DtypedBuf::zeros(stream, bt * dm, dtype)?,
            // backward f32
            d_b_reduced: GpuBuffer::zeros(stream, bt * ds)?,
            d_c_reduced: GpuBuffer::zeros(stream, bt * ds)?,
            d_d_local: GpuBuffer::zeros(stream, b * di)?,
            d_a_log_local: GpuBuffer::zeros(stream, b * di * ds)?,
            d_pre_norm: GpuBuffer::zeros(stream, bt * dm)?,
            d_input_proj_dx: GpuBuffer::zeros(stream, bt * dims.mamba_input_dim)?,
            // Rule-B axis-0 partials scratch — sized to fit largest consumer.
            axis0_partials: GpuBuffer::zeros(
                stream,
                std::cmp::max(bt * dm, b * di * (dims.d_conv + 1)),
            )?,
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
        return Err(
            "non-identity input_proj for mixed training not yet implemented \
             (HF Mamba uses identity_proj — use from_hf path)"
                .to_string(),
        );
    }

    // ─── Per-layer offsets into flat state buffers ───
    let conv_per_layer = b * di * d_conv;
    let ssm_per_layer = b * di * ds;
    let a_neg_per_layer = di * ds;
    let f32_sz = std::mem::size_of::<f32>() as u64;
    let conv_base = state.conv_states.cached_ptr();
    let ssm_base = state.ssm_states.cached_ptr();
    let aneg_base = state.a_neg_all.cached_ptr();

    for layer_idx in 0..dims.n_layers {
        let conv_ptr = conv_base + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = ssm_base + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let aneg_ptr = aneg_base + (layer_idx * a_neg_per_layer) as u64 * f32_sz;
        let lw = &mamba_w.layers[layer_idx];
        let layer_acts = &mut acts.layers[layer_idx];

        // F1: rmsnorm_fwd_f32in_typed — read f32 residual, write typed post_norm.
        // Side effect: rms_vals_f32 saved for backward.
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = 1e-5;
            let mut bld = ctx.stream.launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
            let pn_ptr = layer_acts.post_norm.cached_ptr();
            let rms_ptr = layer_acts.rms_vals.cached_ptr();
            let res_ptr = layer_acts.residual.cached_ptr();
            bld.arg(&pn_ptr);
            bld.arg(&rms_ptr);
            bld.arg(&res_ptr);
            let nw = lw.norm_weight.ptr();
            bld.arg(&nw);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            bld.arg(&eps);
            unsafe { bld.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_f32in_typed L{layer_idx}: {e:?}"))?;
        }

        // F2: in_proj GEMM typed — [B*T, dm] -> [B*T, 2*di].
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.proj_flat.cached_ptr(),
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
            None,
            (bt, dm, 2 * di),
        )?;

        // F3: split_gate_silu_typed.
        {
            let bt_i = bt as i32;
            let di_i = di as i32;
            let mut bld = ctx.stream.launch_builder(k.split_gate_silu_typed.get(dt));
            let xb = scratch.x_branch.cached_ptr();
            let gp = layer_acts.gate_pre_silu.cached_ptr();
            let gs = layer_acts.gate_post_silu.cached_ptr();
            let pf = scratch.proj_flat.cached_ptr();
            bld.arg(&xb);
            bld.arg(&gp);
            bld.arg(&gs);
            bld.arg(&pf);
            bld.arg(&bt_i);
            bld.arg(&di_i);
            unsafe { bld.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("split_gate_silu_typed L{layer_idx}: {e:?}"))?;
        }

        // F4a: conv1d_burnin_forward_typed — typed I/O, f32 state save.
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let dc_i = d_conv as i32;
            // Use the TYPED-signature f32 variant here so the argument order
            // below matches the bf16/f16 kernels. The legacy `conv1d_burnin_fwd`
            // f32 kernel has `(u_out, post_conv, conv_states, state, x_branch,
            // ...)` which is a different order — previously plugging it into
            // the typed call path silently swapped `state` with `post_conv`,
            // corrupting the persistent conv state on every mixed f32 step.
            let kernel = match dt {
                WeightDtype::F32 => &k.conv1d_burnin_fwd_f32_typed,
                WeightDtype::Bf16 => &k.conv1d_burnin_fwd_bf16,
                WeightDtype::F16 => &k.conv1d_burnin_fwd_f16,
            };
            let mut bld = ctx.stream.launch_builder(kernel);
            let u = layer_acts.u.cached_ptr();
            let cs = layer_acts.conv_states.cached_ptr();
            let pc = layer_acts.post_conv.cached_ptr();
            let xb = scratch.x_branch.cached_ptr();
            bld.arg(&u);
            bld.arg(&conv_ptr); // state (f32, layer offset)
            bld.arg(&cs);
            bld.arg(&pc);
            bld.arg(&xb);
            let cw = lw.conv1d_weight.ptr();
            let cb = lw.conv1d_bias.ptr();
            bld.arg(&cw);
            bld.arg(&cb);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&dc_i);
            unsafe { bld.launch(grid_1d(b * di)) }
                .map_err(|e| format!("conv1d_burnin_typed L{layer_idx}: {e:?}"))?;
        }

        // F4b: x_proj GEMM typed.
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
            unsafe { bld.launch(grid_1d(bt * dt_rank)) }
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
        {
            let n = (bt * di) as i32;
            let mut bld = ctx.stream.launch_builder(k.softplus_copy_typed.get(dt));
            let dl = layer_acts.delta.cached_ptr();
            let dr = layer_acts.delta_raw.cached_ptr();
            bld.arg(&dl);
            bld.arg(&dr);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("softplus_copy_typed L{layer_idx}: {e:?}"))?;
        }

        // F4d: gather_bc_cols_typed → b_buf, c_buf; ssm_burnin_forward typed.
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_off = dt_rank as i32;
            let c_off = (dt_rank + ds) as i32;
            let mut bld = ctx.stream.launch_builder(k.gather_bc_cols_typed.get(dt));
            let bb = scratch.b_buf.cached_ptr();
            let cb = scratch.c_buf.cached_ptr();
            let xd = layer_acts.xdbl.cached_ptr();
            bld.arg(&bb);
            bld.arg(&cb);
            bld.arg(&xd);
            bld.arg(&bt_i);
            bld.arg(&xdbl_i);
            bld.arg(&ds_i);
            bld.arg(&b_off);
            bld.arg(&c_off);
            unsafe { bld.launch(grid_1d(bt * ds)) }
                .map_err(|e| format!("gather_bc_cols typed L{layer_idx}: {e:?}"))?;
        }
        // SSM forward: parallel prefix scan for T > PARALLEL_SCAN_THRESHOLD
        // or ds > 64 (matches f32 path dispatch in forward.rs:544). Typed
        // variants (Step 8b) keep scan state + h_saved + da_exp + smem f32
        // per state-spaces/mamba `scan_t = float2` invariant; only
        // delta/u/B/C/y are typed.
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            if dims.scan_mode.use_parallel(t, ds) {
                let kernel = k.ssm_parallel_fwd_typed.get(dt);
                let mut bld = ctx.stream.launch_builder(kernel);
                let y = layer_acts.y.cached_ptr();
                let hs = layer_acts.h_saved.cached_ptr();
                let dae = layer_acts.da_exp.cached_ptr();
                let dl = layer_acts.delta.cached_ptr();
                let u = layer_acts.u.cached_ptr();
                let bb = scratch.b_buf.cached_ptr();
                let cb = scratch.c_buf.cached_ptr();
                let dp = lw.d_param.ptr();
                bld.arg(&ssm_ptr);
                bld.arg(&y);
                bld.arg(&hs);
                bld.arg(&dae);
                bld.arg(&dl);
                bld.arg(&u);
                bld.arg(&bb);
                bld.arg(&cb);
                bld.arg(&aneg_ptr);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe {
                    bld.launch(super::launch::grid_parallel_scan_typed(
                        b,
                        di,
                        dt.size_bytes(),
                    ))
                }
                .map_err(|e| format!("ssm_parallel_fwd typed L{layer_idx}: {e:?}"))?;
            } else {
                assert!(
                    ds <= 64,
                    "ssm_burnin_forward_typed requires d_state <= 64 (got {ds})"
                );
                let kernel = match dt {
                    WeightDtype::F32 => &k.ssm_burnin_fwd,
                    WeightDtype::Bf16 => &k.ssm_burnin_fwd_bf16,
                    WeightDtype::F16 => &k.ssm_burnin_fwd_f16,
                };
                let mut bld = ctx.stream.launch_builder(kernel);
                let y = layer_acts.y.cached_ptr();
                let hs = layer_acts.h_saved.cached_ptr();
                let dae = layer_acts.da_exp.cached_ptr();
                let dl = layer_acts.delta.cached_ptr();
                let u = layer_acts.u.cached_ptr();
                let bb = scratch.b_buf.cached_ptr();
                let cb = scratch.c_buf.cached_ptr();
                let dp = lw.d_param.ptr();
                bld.arg(&ssm_ptr);
                bld.arg(&y);
                bld.arg(&hs);
                bld.arg(&dae);
                bld.arg(&dl);
                bld.arg(&u);
                bld.arg(&bb);
                bld.arg(&cb);
                bld.arg(&aneg_ptr);
                bld.arg(&dp);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_burnin_forward typed L{layer_idx}: {e:?}"))?;
            }
        }

        // F4e: gating — gated = y * gate_post_silu (elementwise_mul_typed).
        {
            let n = (bt * di) as i32;
            let mut bld = ctx.stream.launch_builder(k.elementwise_mul_typed.get(dt));
            let g = layer_acts.gated.cached_ptr();
            let y = layer_acts.y.cached_ptr();
            let gs = layer_acts.gate_post_silu.cached_ptr();
            bld.arg(&g);
            bld.arg(&y);
            bld.arg(&gs);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("elementwise_mul_typed L{layer_idx}: {e:?}"))?;
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
            None,
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
        {
            let n = (bt * dm) as i32;
            let mut bld = ctx.stream.launch_builder(k.residual_add_f32_typed.get(dt));
            let cur_res = acts.layers[layer_idx].residual.cached_ptr();
            let of = scratch.out_flat.cached_ptr();
            bld.arg(&next_res_ptr); // dst f32
            bld.arg(&cur_res); // a f32
            bld.arg(&of); // b typed
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual_add_f32_typed L{layer_idx}: {e:?}"))?;
        }
    }

    // ─── Final norm_f: residual_f32 → temporal_typed via rmsnorm_fwd_f32in_typed.
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = 1e-5;
        let mut bld = ctx.stream.launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
        let tt = scratch.temporal_typed.cached_ptr();
        let nfr = acts.norm_f_rms.cached_ptr();
        let nfi = acts.norm_f_input.cached_ptr();
        bld.arg(&tt);
        bld.arg(&nfr);
        bld.arg(&nfi);
        let nfw = mamba_w.norm_f_weight.ptr();
        bld.arg(&nfw);
        bld.arg(&bt_i);
        bld.arg(&dm_i);
        bld.arg(&eps);
        unsafe { bld.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("rmsnorm_f32in_typed norm_f: {e:?}"))?;
    }

    Ok(())
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
