//! Parallel GPU prefill for LLM prompt processing.
//!
//! Processes T tokens of a prompt in a single forward pass using the
//! `ssm_burnin_forward_nosave` / `ssm_parallel_scan_fwd_nosave` kernels,
//! producing the final hidden state in `GpuInferenceState` + last-timestep
//! temporal output. Avoids the O(T) kernel-launch overhead of step-by-step
//! prefill — 10-40× speedup for prompts with T > 128.
//!
//! Mirrors `gpu_forward_mamba_target_burnin` but uses the inference-path
//! flat `GpuMambaWeights` (not training `GpuMambaTrainWeights`).
//!
//! Serving entries (0.5.1): `gpu_forward_inference_prefill_full` exposes the
//! post-norm_f temporal for ALL T positions, and
//! `gpu_forward_inference_prefill_from_raw` additionally applies the input
//! projection internally with the exact training-forward SGEMM call — both
//! bit-identical to `gpu_forward_mamba_backbone` on the same device
//! (tests/gpu_inference_prefill_parity.rs).

use super::backward::{GpuMambaTargetMixedScratch, GpuMambaTargetScratch};
use super::blas::{
    TypedPtr, gpu_gemm_forward_dispatch, gpu_gemm_typed_forward_raw, gpu_sgemm_forward_raw,
};
use super::buffers::GpuBuffer;
use super::context::GpuCtx;
use super::dtype::WeightDtype;
use super::forward::GpuMambaDims;
use super::inference::GpuInferenceState;
use super::launch::{grid_1d, grid_norm, grid_parallel_scan};
use super::weights::{MambaLayerWeightsView, MambaWeightsView};
use cudarc::driver::PushKernelArg;

/// Prefill a batched prompt sequence through the Mamba backbone in one call.
///
/// Inputs:
/// - `target_temporal`: `[B * d_model]` — output (last timestep hidden state, f32)
/// - `ip_out_flat`: `[B * T * d_model]` — pre-embedded prompt tokens (batch × time × d_model)
/// - `weights`: flat-buffer inference weights (M1)
/// - `state`: persistent inference state (conv + SSM) — updated in-place
/// - `a_neg_all`: precomputed `-exp(a_log)` for all layers `[n_layers * d_inner * d_state]`
/// - `scratch`: batched B*T working buffers
///
/// Inputs bundle for `gpu_forward_inference_prefill`.
pub struct PrefillInputs<'a, W: MambaWeightsView> {
    pub ip_out_flat: &'a GpuBuffer,
    pub weights: &'a W,
    pub a_neg_all: &'a GpuBuffer,
}

/// Inputs bundle for `gpu_forward_inference_prefill_from_raw`: the sequence
/// BEFORE the input projection (`[B * T * mamba_input_dim]`) — the
/// classifier/regressor serving shape, where the model consumes raw
/// features instead of pre-embedded tokens.
pub struct PrefillRawInputs<'a, W: MambaWeightsView> {
    pub input_flat: &'a GpuBuffer,
    pub weights: &'a W,
    pub a_neg_all: &'a GpuBuffer,
}

/// Output request for the `_full` / `_from_raw` prefill entries.
///
/// `full_temporal`, when present, receives the post-`norm_f` temporal for
/// ALL T positions (`[B * T * d_model]`) — the official contract for
/// consumers that pool over the whole sequence (e.g. a mean-pool
/// classification head). This is the same buffer content the training
/// forward emits, bit-for-bit (pinned by tests/gpu_inference_prefill_parity.rs).
pub struct PrefillOutputs<'a> {
    /// `[B * d_model]` — last-timestep hidden state (pre-lm_head).
    pub last_temporal: &'a mut GpuBuffer,
    /// `[B * T * d_model]` — optional post-norm_f output for every position.
    pub full_temporal: Option<&'a mut GpuBuffer>,
}

/// After this call, `state` holds the recurrent state at position T, and
/// `target_temporal` holds the pre-lm_head hidden state for token T (last).
/// Follow with normal `step()` calls to continue decoding.
pub fn gpu_forward_inference_prefill<W: MambaWeightsView>(
    ctx: &GpuCtx,
    target_temporal: &mut GpuBuffer,
    inputs: PrefillInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let PrefillInputs {
        ip_out_flat,
        weights,
        a_neg_all,
    } = inputs;
    // State is the persistent inference state — do NOT zero. We build on top.
    // Working temporal: start with the input embeddings for all T timesteps.
    scratch.out_flat.copy_from(ip_out_flat, &ctx.stream)?;
    prefill_body(ctx, weights, a_neg_all, state, scratch, None)?;
    emit_outputs(
        ctx,
        scratch,
        PrefillOutputs {
            last_temporal: target_temporal,
            full_temporal: None,
        },
    )
}

/// `gpu_forward_inference_prefill` with the all-T output surface: identical
/// kernel chain, plus the optional full post-norm_f temporal out.
///
/// The `_mixed` prefill deliberately has no all-T twin yet: its temporal
/// lives in a typed (bf16/f16) scratch and no mixed consumer pools over the
/// full sequence today — the entry is added when one exists.
pub fn gpu_forward_inference_prefill_full<W: MambaWeightsView>(
    ctx: &GpuCtx,
    outputs: PrefillOutputs<'_>,
    inputs: PrefillInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let PrefillInputs {
        ip_out_flat,
        weights,
        a_neg_all,
    } = inputs;
    scratch.out_flat.copy_from(ip_out_flat, &ctx.stream)?;
    let PrefillOutputs {
        last_temporal,
        mut full_temporal,
    } = outputs;
    check_output_lens(scratch.dims, last_temporal, full_temporal.as_deref())?;
    prefill_body(
        ctx,
        weights,
        a_neg_all,
        state,
        scratch,
        full_temporal.as_deref_mut(),
    )?;
    let src = match &full_temporal {
        Some(full) => &**full,
        None => &scratch.out_flat,
    };
    gather_last(ctx, src, last_temporal, scratch.dims)
}

/// Prefill from RAW (pre-projection) input: applies `input_proj` internally
/// through the SAME `gpu_sgemm_forward_raw` call the training forward makes
/// (`gpu_forward_mamba_backbone`), so the projected bits equal training's
/// `ip_out` — then runs the shared prefill body. This is the one-call GPU
/// serving entry for classifier-style consumers.
pub fn gpu_forward_inference_prefill_from_raw<W: MambaWeightsView>(
    ctx: &GpuCtx,
    outputs: PrefillOutputs<'_>,
    inputs: PrefillRawInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let PrefillRawInputs {
        input_flat,
        weights,
        a_neg_all,
    } = inputs;
    let dims: GpuMambaDims = scratch.dims;
    let bt = dims.batch * dims.seq_len;
    let expected = bt * dims.mamba_input_dim;
    if input_flat.len() != expected {
        return Err(format!(
            "prefill_from_raw: input_flat len {} != batch*seq_len*mamba_input_dim = {expected}",
            input_flat.len()
        ));
    }
    let (ipw, ipw_dt) = weights.input_proj_w();
    if ipw_dt != WeightDtype::F32 {
        return Err(format!(
            "prefill_from_raw: input_proj weight must be f32 (got {ipw_dt:?}) — \
             the mixed prefill has no raw-input entry"
        ));
    }
    // Projected values land directly in the working temporal; the body's
    // first step snapshots it into the residual, exactly as if it had been
    // copied from a caller-provided ip_out buffer.
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.out_flat,
        input_flat,
        ipw,
        Some(weights.input_proj_b()),
        (bt, dims.mamba_input_dim, dims.d_model),
    )?;
    let PrefillOutputs {
        last_temporal,
        mut full_temporal,
    } = outputs;
    check_output_lens(scratch.dims, last_temporal, full_temporal.as_deref())?;
    prefill_body(
        ctx,
        weights,
        a_neg_all,
        state,
        scratch,
        full_temporal.as_deref_mut(),
    )?;
    let src = match &full_temporal {
        Some(full) => &**full,
        None => &scratch.out_flat,
    };
    gather_last(ctx, src, last_temporal, scratch.dims)
}

/// Raw-input prefill emitting the COLUMN SUM of the post-`norm_f` temporal
/// over all T positions (`pooled_sum[d] = sum_t temporal[t][d]`, batch=1
/// serving shape; `[B * d_model]` generally, summed within each sample's
/// T block only when B=1 — multi-batch consumers slice per sample).
///
/// Mean-pool consumers divide by T on the HOST: the on-device
/// `colsum_accumulate` walks ascending `t` accumulating pure f32 adds from
/// 0.0 — character-identical to a CPU per-column mean_pool accumulation —
/// so `download(1.5 KB) / T` reproduces the CPU pooled vector BIT-FOR-BIT
/// while skipping the full `[T * d_model]` device-to-host transfer
/// (7.1 MB -> 1.5 KB at the classifier shape).
pub fn gpu_forward_inference_prefill_pooled_sum_from_raw<W: MambaWeightsView>(
    ctx: &GpuCtx,
    pooled_sum: &mut GpuBuffer,
    inputs: PrefillRawInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
) -> Result<(), String> {
    let dims: GpuMambaDims = scratch.dims;
    if dims.batch != 1 {
        return Err(format!(
            "prefill_pooled_sum: batch {} unsupported — the column sum would mix samples; \
             pool per sample with batch=1 (the serving shape)",
            dims.batch
        ));
    }
    if pooled_sum.len() != dims.d_model {
        return Err(format!(
            "prefill_pooled_sum: pooled_sum len {} != d_model = {}",
            pooled_sum.len(),
            dims.d_model
        ));
    }
    let PrefillRawInputs {
        input_flat,
        weights,
        a_neg_all,
    } = inputs;
    let bt = dims.batch * dims.seq_len;
    let expected = bt * dims.mamba_input_dim;
    if input_flat.len() != expected {
        return Err(format!(
            "prefill_pooled_sum: input_flat len {} != batch*seq_len*mamba_input_dim = {expected}",
            input_flat.len()
        ));
    }
    let (ipw, ipw_dt) = weights.input_proj_w();
    if ipw_dt != WeightDtype::F32 {
        return Err(format!(
            "prefill_pooled_sum: input_proj weight must be f32 (got {ipw_dt:?}) — \
             the mixed prefill has no raw-input entry"
        ));
    }
    gpu_sgemm_forward_raw(
        ctx,
        &mut scratch.out_flat,
        input_flat,
        ipw,
        Some(weights.input_proj_b()),
        (bt, dims.mamba_input_dim, dims.d_model),
    )?;
    prefill_body(ctx, weights, a_neg_all, state, scratch, None)?;
    // colsum_accumulate does `db[j] += sum` — zero the target first so the
    // result is exactly the sum over T.
    pooled_sum.zero(&ctx.stream)?;
    let b_i = bt as i32;
    let n_i = dims.d_model as i32;
    let db_ptr = pooled_sum.cached_ptr();
    let dy_ptr = scratch.out_flat.cached_ptr();
    let mut builder = ctx.stream.launch_builder(&ctx.kernels.colsum_accumulate);
    builder.arg(&db_ptr);
    builder.arg(&dy_ptr);
    builder.arg(&b_i);
    builder.arg(&n_i);
    unsafe { builder.launch(grid_1d(dims.d_model)) }
        .map_err(|e| format!("prefill_pooled_sum colsum: {e:?}"))?;
    Ok(())
}

/// A CUDA Graph capture of the ENTIRE per-page pooled-prefill sequence —
/// state reset + input_proj SGEMM + layer chain + norm_f + pooled column
/// sum (~hundreds of launches at real depths collapse into one replay).
///
/// Contract:
/// - Buffers are FIXED at capture: replays read the SAME `input_flat`
///   buffer (upload the next page into it first) and write the SAME
///   `pooled_sum` buffer (download after). The state buffer is reset
///   INSIDE the graph, so every replay starts a fresh page.
/// - Bit-identity: a replay re-issues the exact captured kernel sequence
///   with the exact pointers — outputs are bit-identical to the eager
///   entry by construction (pinned by tests/gpu_pooled_prefill.rs).
/// - the GEMM-tier flags are snapshotted at capture and asserted
///   at every launch — a mid-flight tier flip cannot silently replay
///   kernels from another numeric route.
pub struct PrefillPooledGraph {
    graph: cudarc::driver::CudaGraph,
    flags_at_capture: crate::mamba_ssm::gpu::context::GemmRoute,
    // Buffer identity at capture: the graph baked these device pointers
    // in, so a launch against reallocated buffers must refuse instead of
    // silently writing the old allocations (the ABA class).
    input_ptr: cudarc::driver::sys::CUdeviceptr,
    pooled_ptr: cudarc::driver::sys::CUdeviceptr,
    module_identity: String,
}

impl PrefillPooledGraph {
    /// Capture the pooled prefill over the given fixed buffers.
    pub fn capture<W: MambaWeightsView>(
        ctx: &GpuCtx,
        pooled_sum: &mut GpuBuffer,
        inputs: PrefillRawInputs<'_, W>,
        state: &mut GpuInferenceState,
        scratch: &mut GpuMambaTargetScratch,
    ) -> Result<Self, String> {
        // Lazy BI scratch must exist BEFORE the capture (see
        // GpuCtx::presize_bi_scratch).
        ctx.presize_bi_scratch()?;
        let flags_at_capture = ctx.gemm_route();
        let PrefillRawInputs {
            input_flat,
            weights,
            a_neg_all,
        } = inputs;
        let input_ptr = input_flat.cached_ptr();
        let pooled_ptr = pooled_sum.cached_ptr();
        let module_identity = ctx.kernels.module_identity.clone();
        let graph = super::graph_capture::capture_into_graph(&ctx.stream, || {
            state.reset(&ctx.stream)?;
            gpu_forward_inference_prefill_pooled_sum_from_raw(
                ctx,
                pooled_sum,
                PrefillRawInputs {
                    input_flat,
                    weights,
                    a_neg_all,
                },
                state,
                scratch,
            )
        })?;
        ctx.note_graph_capture();
        Ok(Self {
            graph,
            flags_at_capture,
            input_ptr,
            pooled_ptr,
            module_identity,
        })
    }

    /// Replay the captured page. The caller uploads the page into the
    /// capture-time `input_flat` buffer before, and downloads the
    /// capture-time `pooled_sum` buffer after (both transfers stay OUTSIDE
    /// the graph). The buffers handed here must BE the captured
    /// allocations - the graph launches into the captured pointers, so a
    /// swapped buffer would be silently ignored.
    pub fn launch(
        &self,
        ctx: &GpuCtx,
        input_flat: &GpuBuffer,
        pooled_sum: &GpuBuffer,
    ) -> Result<(), String> {
        let now = ctx.gemm_route();
        if now != self.flags_at_capture {
            return Err(format!(
                "PrefillPooledGraph: GEMM flags changed since capture \
                 ({:?} -> {now:?}) — the captured kernels belong to the old \
                 tier (G1)",
                self.flags_at_capture
            ));
        }
        if ctx.kernels.module_identity != self.module_identity {
            return Err("PrefillPooledGraph: the kernels module differs from the \
                 captured compile - the graph would run stale kernels"
                .to_string());
        }
        if input_flat.cached_ptr() != self.input_ptr || pooled_sum.cached_ptr() != self.pooled_ptr {
            return Err("PrefillPooledGraph: input/pooled buffers differ from the \
                 captured allocations - the graph would write the old ones"
                .to_string());
        }
        self.graph
            .launch()
            .map_err(|e| format!("PrefillPooledGraph launch: {e:?}"))
    }
}

/// The shared prefill kernel chain: layer loop + final norm_f. Assumes
/// `scratch.out_flat` is seeded with the projected input for all T; leaves
/// the post-norm_f temporal for all T in `scratch.out_flat`.
fn prefill_body<W: MambaWeightsView>(
    ctx: &GpuCtx,
    weights: &W,
    a_neg_all: &GpuBuffer,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetScratch,
    final_out: Option<&mut GpuBuffer>,
) -> Result<(), String> {
    let dims: GpuMambaDims = scratch.dims;
    let seq_len = dims.seq_len;
    let b = dims.batch;
    let bt = b * seq_len;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let d_conv = dims.d_conv;
    let t = seq_len;

    let f32_sz = std::mem::size_of::<f32>() as u64;

    for layer_idx in 0..weights.n_layers() {
        let lw = weights.layer(layer_idx);

        // Per-layer state pointers into the inference state (NOT scratch.conv_states).
        // inference state layout: conv[n_layers][batch * d_inner * d_conv],
        //                         ssm[n_layers][batch * d_inner * d_state].
        let conv_per_layer = b * di * d_conv;
        let ssm_per_layer = b * di * ds;
        let conv_ptr = state.conv.cached_ptr() + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = state.ssm.cached_ptr() + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let a_neg_ptr = a_neg_all.cached_ptr() + (layer_idx * di * ds) as u64 * f32_sz;

        // F1: RmsNorm [B*T]. Ping-pong: the activations
        // STAY in out_flat; the normed values land in `residual` (which is
        // dead until F5 reuses it as the block output). The old
        // copy-then-overwrite spent a full [B*T*dm] D2D per layer
        // (25 x 7.1 MB per page at the serve shape) moving the same bytes.
        // Values are untouched — same kernel, same inputs, other buffer.
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = dims.rms_norm_eps;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
            builder.arg(scratch.residual.inner_mut());
            builder.arg(scratch.rms_discard.inner_mut());
            builder.arg(scratch.out_flat.inner());
            let nw = lw.norm_weight();
            builder.arg(&nw);
            builder.arg(&bt_i);
            builder.arg(&dm_i);
            builder.arg(&eps);
            unsafe { builder.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm prefill L{layer_idx}: {e:?}"))?;
        }

        // F2: in_proj GEMM [B*T, dm] → [B*T, 2*di] (dtype-dispatched).
        // Reads the NORMED values from `residual` (ping-pong); after this
        // call the buffer is dead until F5 reuses it as the block output.
        let (ipw, ipw_dt) = lw.in_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.proj_flat,
            &scratch.residual,
            ipw,
            ipw_dt,
            None,
            (bt, dm, 2 * di),
        )?;

        // F3 (split x + SiLU gate) is gone: the conv reads its lane
        // strided from the in_proj output, and the scan fuses the gating
        // into its y store - three kernels and two B*T*di round-trips
        // replaced by two strided reads, bit-identically.

        // F4a: conv1d burnin nosave + fused SiLU [all T, parallel B*d_inner]
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let dc_i = d_conv as i32;
            let mut builder = ctx
                .stream
                .launch_builder(&ctx.kernels.conv1d_burnin_fwd_nosave_tiled);
            builder.arg(scratch.u.inner_mut());
            builder.arg(&conv_ptr); // INFERENCE STATE conv — persistent
            builder.arg(scratch.proj_flat.inner());
            let cw = lw.conv1d_weight();
            let cb = lw.conv1d_bias();
            builder.arg(&cw);
            builder.arg(&cb);
            builder.arg(&b_i);
            builder.arg(&t_i);
            builder.arg(&di_i);
            builder.arg(&dc_i);
            let xs_i = (2 * di) as i32;
            builder.arg(&xs_i);
            // T-tiled: 3 blocks at B=1 became ~111 - the serial T=4621
            // walk was the single largest prefill stage.
            unsafe { builder.launch(super::launch::grid_conv_tiled(b, di, t)) }
                .map_err(|e| format!("conv1d_nosave_tiled prefill L{layer_idx}: {e:?}"))?;
        }

        // F4b: x_proj GEMM [B*T, di] → [B*T, xdbl_dim]
        let (xpw, xpw_dt) = lw.x_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.xdbl,
            &scratch.u,
            xpw,
            xpw_dt,
            None,
            (bt, di, xdbl_dim),
        )?;

        // F4c: gather dt + dt_proj + softplus
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let dt_i = dt_rank as i32;
            let offset: i32 = 0;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_cols);
            builder.arg(scratch.dt_gather.inner_mut());
            builder.arg(scratch.xdbl.inner());
            builder.arg(&bt_i);
            builder.arg(&xdbl_i);
            builder.arg(&dt_i);
            builder.arg(&offset);
            unsafe { builder.launch(grid_1d(bt * dt_rank)) }
                .map_err(|e| format!("gather dt prefill L{layer_idx}: {e:?}"))?;
        }
        let (dpw, dpw_dt) = lw.dt_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.delta,
            &scratch.dt_gather,
            dpw,
            dpw_dt,
            Some(lw.dt_proj_b()),
            (bt, dt_rank, di),
        )?;
        {
            let n = (bt * di) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.softplus_fwd);
            builder.arg(scratch.delta.inner_mut());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("softplus prefill L{layer_idx}: {e:?}"))?;
        }

        // F4d: gather B/C + SSM burnin nosave (sequential for T ≤ 256, parallel otherwise)
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_offset = dt_rank as i32;
            let c_offset = (dt_rank + ds) as i32;
            // Parallel route gathers T-major (matches the scan kernels).
            // The staged tile writes t-contiguous instead of scattering one
            // sector per element; identical bytes, so bit-free. Falls back
            // to the untiled kernel past the 48 KB static smem budget.
            let tmajor = dims.scan_mode.use_parallel(t, ds);
            let tile_smem = 2 * ds * (super::launch::GBC_TILE_T + 1) * std::mem::size_of::<f32>();
            let t_i = t as i32;
            if tmajor && tile_smem <= 48 * 1024 {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.gather_bc_cols_tmajor_tiled);
                builder.arg(scratch.b_gathered.inner_mut());
                builder.arg(scratch.c_gathered.inner_mut());
                builder.arg(scratch.xdbl.inner());
                builder.arg(&t_i);
                builder.arg(&xdbl_i);
                builder.arg(&ds_i);
                builder.arg(&b_offset);
                builder.arg(&c_offset);
                unsafe {
                    builder.launch(super::launch::grid_gather_bc_tiled(
                        b,
                        t,
                        ds,
                        std::mem::size_of::<f32>(),
                    ))
                }
                .map_err(|e| format!("gather_bc prefill L{layer_idx}: {e:?}"))?;
            } else {
                let kernel = if tmajor {
                    &ctx.kernels.gather_bc_cols_tmajor
                } else {
                    &ctx.kernels.gather_bc_cols
                };
                let mut builder = ctx.stream.launch_builder(kernel);
                builder.arg(scratch.b_gathered.inner_mut());
                builder.arg(scratch.c_gathered.inner_mut());
                builder.arg(scratch.xdbl.inner());
                builder.arg(&bt_i);
                if tmajor {
                    builder.arg(&t_i);
                }
                builder.arg(&xdbl_i);
                builder.arg(&ds_i);
                builder.arg(&b_offset);
                builder.arg(&c_offset);
                unsafe { builder.launch(grid_1d(bt * ds)) }
                    .map_err(|e| format!("gather_bc prefill L{layer_idx}: {e:?}"))?;
            }
        }
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            let gate_stride = (2 * di) as i32;
            if dims.scan_mode.use_parallel(t, ds) {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.ssm_parallel_fwd_nosave);
                builder.arg(&ssm_ptr);
                builder.arg(scratch.gated.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(scratch.u.inner());
                builder.arg(scratch.b_gathered.inner());
                builder.arg(scratch.c_gathered.inner());
                builder.arg(&a_neg_ptr);
                let dp = lw.d_param();
                builder.arg(&dp);
                builder.arg(scratch.proj_flat.inner());
                builder.arg(&gate_stride);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_parallel_scan(b, di, ds)) }
                    .map_err(|e| format!("ssm_parallel prefill L{layer_idx}: {e:?}"))?;
            } else {
                let mut builder = ctx
                    .stream
                    .launch_builder(&ctx.kernels.ssm_burnin_fwd_nosave);
                builder.arg(&ssm_ptr);
                builder.arg(scratch.gated.inner_mut());
                builder.arg(scratch.delta.inner());
                builder.arg(scratch.u.inner());
                builder.arg(scratch.b_gathered.inner());
                builder.arg(scratch.c_gathered.inner());
                builder.arg(&a_neg_ptr);
                let dp = lw.d_param();
                builder.arg(&dp);
                builder.arg(scratch.proj_flat.inner());
                builder.arg(&gate_stride);
                builder.arg(&b_i);
                builder.arg(&t_i);
                builder.arg(&di_i);
                builder.arg(&ds_i);
                unsafe { builder.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_nosave prefill L{layer_idx}: {e:?}"))?;
            }
        }
        // Gating now happens inside the scan's y store (fused, one
        // rounding, same SiLU) - the separate mul kernel is gone.

        // F5: out_proj GEMM [B*T, di] → [B*T, dm] — the block output lands
        // in `residual` (dead since F2 consumed the normed values).
        let (opw, opw_dt) = lw.out_proj_w();
        gpu_gemm_forward_dispatch(
            ctx,
            &mut scratch.residual,
            &scratch.gated,
            opw,
            opw_dt,
            None,
            (bt, di, dm),
        )?;

        // F6: residual add — the running activations (out_flat) absorb the
        // block output. The roles are swapped versus the old
        // copy-then-overwrite layout, but IEEE f32 addition of the same two
        // values is commutative: a + b and b + a round identically, so the
        // sum is bit-equal to the old route.
        {
            let n = (bt * dm) as i32;
            let mut builder = ctx.stream.launch_builder(&ctx.kernels.vec_add_inplace);
            builder.arg(scratch.out_flat.inner_mut());
            builder.arg(scratch.residual.inner());
            builder.arg(&n);
            unsafe { builder.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual prefill L{layer_idx}: {e:?}"))?;
        }
    }

    // Final RmsNorm (norm_f) over all B*T timesteps
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        scratch.residual.copy_from(&scratch.out_flat, &ctx.stream)?;
        let mut builder = ctx.stream.launch_builder(&ctx.kernels.rmsnorm_fwd);
        // When the caller wants the all-T temporal, norm_f writes it
        // directly instead of landing in out_flat and being copied out
        // afterwards - same kernel, same inputs, other destination buffer.
        match final_out {
            Some(full) => builder.arg(full.inner_mut()),
            None => builder.arg(scratch.out_flat.inner_mut()),
        };
        builder.arg(scratch.rms_discard.inner_mut());
        builder.arg(scratch.residual.inner());
        let nfw = weights.norm_f_weight();
        builder.arg(&nfw);
        builder.arg(&bt_i);
        builder.arg(&dm_i);
        builder.arg(&eps);
        unsafe { builder.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("norm_f prefill: {e:?}"))?;
    }

    Ok(())
}

/// Validate the output buffer sizes before the body runs, so a bad request
/// fails fast instead of after a full prefill.
fn check_output_lens(
    dims: GpuMambaDims,
    last_temporal: &GpuBuffer,
    full_temporal: Option<&GpuBuffer>,
) -> Result<(), String> {
    let (b, t, dm) = (dims.batch, dims.seq_len, dims.d_model);
    if last_temporal.len() != b * dm {
        return Err(format!(
            "prefill outputs: last_temporal len {} != batch*d_model = {}",
            last_temporal.len(),
            b * dm
        ));
    }
    if let Some(full) = full_temporal {
        if full.len() != b * t * dm {
            return Err(format!(
                "prefill outputs: full_temporal len {} != batch*seq_len*d_model = {}",
                full.len(),
                b * t * dm
            ));
        }
    }
    Ok(())
}

/// Gather the last timestep of the post-norm_f temporal into `[B * dm]`.
fn gather_last(
    ctx: &GpuCtx,
    src: &GpuBuffer,
    last_temporal: &mut GpuBuffer,
    dims: GpuMambaDims,
) -> Result<(), String> {
    let (b, t, dm) = (dims.batch, dims.seq_len, dims.d_model);
    let b_i = b as i32;
    let t_i = t as i32;
    let dm_i = dm as i32;
    let mut builder = ctx.stream.launch_builder(&ctx.kernels.gather_last_timestep);
    builder.arg(last_temporal.inner_mut());
    builder.arg(src.inner());
    builder.arg(&b_i);
    builder.arg(&t_i);
    builder.arg(&dm_i);
    unsafe { builder.launch(grid_1d(b * dm)) }
        .map_err(|e| format!("gather_last prefill: {e:?}"))?;
    Ok(())
}

/// Deliver the prefill outputs from the finished body: the last-timestep
/// gather (always) and the optional all-T copy of the post-norm_f temporal.
/// Kept for the plain (no-full) entry; the all-T entries write norm_f into
/// the caller buffer inside the body and gather from it directly.
fn emit_outputs(
    ctx: &GpuCtx,
    scratch: &GpuMambaTargetScratch,
    outputs: PrefillOutputs<'_>,
) -> Result<(), String> {
    let dims: GpuMambaDims = scratch.dims;
    let PrefillOutputs {
        last_temporal,
        full_temporal,
    } = outputs;
    check_output_lens(dims, last_temporal, full_temporal.as_deref())?;
    gather_last(ctx, &scratch.out_flat, last_temporal, dims)?;
    if let Some(full) = full_temporal {
        full.copy_from(&scratch.out_flat, &ctx.stream)?;
    }
    Ok(())
}

/// End-to-end bf16/f16 prefill — mirror of `gpu_forward_inference_prefill`
/// with half-precision activations throughout and an f32 residual stream.
///
/// Target output (`target_temporal`) is a `DtypedBuf` in the same dtype as
/// the mixed weights — downstream lm_head expects bf16/f16 directly.
///
/// Scan dispatch mirrors the f32 prefill and the mixed training forward:
/// `use_parallel(T, d_state)` routes long prompts to the typed parallel
/// nosave kernel; the short-prompt sequential route asserts the kernel's
/// `d_state <= 64` contract instead of silently returning.
pub fn gpu_forward_inference_prefill_mixed<W: MambaWeightsView>(
    ctx: &GpuCtx,
    target_temporal: &super::buffers::DtypedBuf,
    inputs: PrefillInputs<'_, W>,
    state: &mut GpuInferenceState,
    scratch: &mut GpuMambaTargetMixedScratch,
) -> Result<(), String> {
    let PrefillInputs {
        ip_out_flat,
        weights,
        a_neg_all,
    } = inputs;
    let dt = scratch.dtype;
    assert_eq!(
        target_temporal.dtype(),
        dt,
        "target_temporal dtype must match mixed scratch dtype"
    );
    let dims: GpuMambaDims = scratch.dims;
    let seq_len = dims.seq_len;
    let b = dims.batch;
    let bt = b * seq_len;
    let dm = dims.d_model;
    let di = dims.d_inner;
    let ds = dims.d_state;
    let dt_rank = dims.dt_rank;
    let xdbl_dim = dims.xdbl_dim;
    let d_conv = dims.d_conv;
    let t = seq_len;
    let k = &ctx.kernels;

    // Seed the f32 residual stream with the incoming embeddings. `ip_out_flat`
    // is already f32 (CPU embed lookup); no downcast needed to set up residual.
    scratch.residual.copy_from(ip_out_flat, &ctx.stream)?;

    let f32_sz = std::mem::size_of::<f32>() as u64;

    for layer_idx in 0..weights.n_layers() {
        let lw = weights.layer(layer_idx);
        let conv_per_layer = b * di * d_conv;
        let ssm_per_layer = b * di * ds;
        let conv_ptr = state.conv.cached_ptr() + (layer_idx * conv_per_layer) as u64 * f32_sz;
        let ssm_ptr = state.ssm.cached_ptr() + (layer_idx * ssm_per_layer) as u64 * f32_sz;
        let a_neg_ptr = a_neg_all.cached_ptr() + (layer_idx * di * ds) as u64 * f32_sz;

        // F1: rmsnorm f32in → half_out (out_flat ← residual * norm_w).
        {
            let bt_i = bt as i32;
            let dm_i = dm as i32;
            let eps: f32 = dims.rms_norm_eps;
            let mut bld = ctx.stream.launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
            let out_ptr = scratch.out_flat.cached_ptr();
            let rms_ptr = scratch.rms_discard.cached_ptr();
            let res_ptr = scratch.residual.cached_ptr();
            bld.arg(&out_ptr);
            bld.arg(&rms_ptr);
            bld.arg(&res_ptr);
            let nw = lw.norm_weight();
            bld.arg(&nw);
            bld.arg(&bt_i);
            bld.arg(&dm_i);
            bld.arg(&eps);
            unsafe { bld.launch(grid_norm(bt, dm)) }
                .map_err(|e| format!("rmsnorm_f32in prefill L{layer_idx}: {e:?}"))?;
        }

        // F2: in_proj GEMM typed bf16 → bf16 proj.
        let (ipw, ipw_dt) = lw.in_proj_w();
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.proj_flat.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: scratch.out_flat.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: ipw,
                dtype: ipw_dt,
            },
            None,
            (bt, dm, 2 * di),
        )?;

        // F3 is fused away, same as the f32 chain: the conv reads the
        // in_proj output strided, the scan gates its own y store with the
        // round-trip emulation of the replaced typed chain.

        // F4a: conv1d burnin nosave typed (with fused SiLU).
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let dc_i = d_conv as i32;
            let mut bld = ctx
                .stream
                .launch_builder(k.conv1d_burnin_nosave_tiled_typed.get(dt));
            let u_ptr = scratch.u.cached_ptr();
            let xb_ptr = scratch.proj_flat.cached_ptr();
            bld.arg(&u_ptr);
            bld.arg(&conv_ptr);
            bld.arg(&xb_ptr);
            let cw = lw.conv1d_weight();
            let cb = lw.conv1d_bias();
            bld.arg(&cw);
            bld.arg(&cb);
            bld.arg(&b_i);
            bld.arg(&t_i);
            bld.arg(&di_i);
            bld.arg(&dc_i);
            let xs_i = (2 * di) as i32;
            bld.arg(&xs_i);
            unsafe { bld.launch(super::launch::grid_conv_tiled(b, di, t)) }
                .map_err(|e| format!("conv1d_nosave_tiled prefill L{layer_idx}: {e:?}"))?;
        }

        // F4b: x_proj GEMM typed.
        let (xpw, xpw_dt) = lw.x_proj_w();
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.xdbl.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: scratch.u.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: xpw,
                dtype: xpw_dt,
            },
            None,
            (bt, di, xdbl_dim),
        )?;

        // F4c: gather_cols typed + dt_proj GEMM typed + softplus typed.
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let dt_i = dt_rank as i32;
            let offset: i32 = 0;
            let mut bld = ctx.stream.launch_builder(k.gather_cols_typed.get(dt));
            let dtg_ptr = scratch.dt_gather.cached_ptr();
            let xdbl_ptr = scratch.xdbl.cached_ptr();
            bld.arg(&dtg_ptr);
            bld.arg(&xdbl_ptr);
            bld.arg(&bt_i);
            bld.arg(&xdbl_i);
            bld.arg(&dt_i);
            bld.arg(&offset);
            unsafe { bld.launch(grid_1d(bt * dt_rank)) }
                .map_err(|e| format!("gather dt prefill L{layer_idx}: {e:?}"))?;
        }
        let (dpw, dpw_dt) = lw.dt_proj_w();
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.delta.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: scratch.dt_gather.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: dpw,
                dtype: dpw_dt,
            },
            Some(lw.dt_proj_b()),
            (bt, dt_rank, di),
        )?;
        {
            let n = (bt * di) as i32;
            let mut bld = ctx.stream.launch_builder(k.softplus_fwd_typed.get(dt));
            let d_ptr = scratch.delta.cached_ptr();
            bld.arg(&d_ptr);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * di)) }
                .map_err(|e| format!("softplus prefill L{layer_idx}: {e:?}"))?;
        }

        // F4d: gather_bc_cols typed + ssm_burnin_nosave typed.
        {
            let bt_i = bt as i32;
            let xdbl_i = xdbl_dim as i32;
            let ds_i = ds as i32;
            let b_offset = dt_rank as i32;
            let c_offset = (dt_rank + ds) as i32;
            // Parallel route gathers T-major (matches the scan kernels);
            // staged tile when it fits, same fallback rule as the f32 chain.
            let tmajor = dims.scan_mode.use_parallel(t, ds);
            let tile_smem = 2 * ds * (super::launch::GBC_TILE_T + 1) * dt.size_bytes();
            let tm_i = t as i32;
            let bb_ptr = scratch.b_gathered.cached_ptr();
            let cb_ptr = scratch.c_gathered.cached_ptr();
            let xdbl_ptr = scratch.xdbl.cached_ptr();
            if tmajor && tile_smem <= 48 * 1024 {
                let mut bld = ctx
                    .stream
                    .launch_builder(k.gather_bc_cols_tmajor_tiled_typed.get(dt));
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&xdbl_ptr);
                bld.arg(&tm_i);
                bld.arg(&xdbl_i);
                bld.arg(&ds_i);
                bld.arg(&b_offset);
                bld.arg(&c_offset);
                unsafe {
                    bld.launch(super::launch::grid_gather_bc_tiled(
                        b,
                        t,
                        ds,
                        dt.size_bytes(),
                    ))
                }
                .map_err(|e| format!("gather_bc prefill L{layer_idx}: {e:?}"))?;
            } else {
                let kernel = if tmajor {
                    k.gather_bc_cols_tmajor_typed.get(dt)
                } else {
                    k.gather_bc_cols_typed.get(dt)
                };
                let mut bld = ctx.stream.launch_builder(kernel);
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&xdbl_ptr);
                bld.arg(&bt_i);
                if tmajor {
                    bld.arg(&tm_i);
                }
                bld.arg(&xdbl_i);
                bld.arg(&ds_i);
                bld.arg(&b_offset);
                bld.arg(&c_offset);
                unsafe { bld.launch(grid_1d(bt * ds)) }
                    .map_err(|e| format!("gather_bc prefill L{layer_idx}: {e:?}"))?;
            }
        }
        {
            let b_i = b as i32;
            let t_i = t as i32;
            let di_i = di as i32;
            let ds_i = ds as i32;
            let delta_ptr = scratch.delta.cached_ptr();
            let u_ptr = scratch.u.cached_ptr();
            let bb_ptr = scratch.b_gathered.cached_ptr();
            let cb_ptr = scratch.c_gathered.cached_ptr();
            let dp = lw.d_param();
            // this dispatch used to bypass the
            // scan router on a stale "parallel scan is f32-only" premise —
            // the typed parallel nosave kernel exists and mirrors the f32
            // route above; long prompts paid O(T) for nothing, and the
            // sequential kernel's silent d_state>64 return had no guard.
            let gated_out_ptr = scratch.gated.cached_ptr();
            let proj_ptr = scratch.proj_flat.cached_ptr();
            let gate_stride = (2 * di) as i32;
            if dims.scan_mode.use_parallel(t, ds) {
                let mut bld = ctx
                    .stream
                    .launch_builder(k.ssm_parallel_fwd_nosave_typed.get(dt));
                bld.arg(&ssm_ptr);
                bld.arg(&gated_out_ptr);
                bld.arg(&delta_ptr);
                bld.arg(&u_ptr);
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&a_neg_ptr);
                bld.arg(&dp);
                bld.arg(&proj_ptr);
                bld.arg(&gate_stride);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe {
                    bld.launch(super::launch::grid_parallel_scan_typed(
                        b,
                        di,
                        dt.size_bytes(),
                        ds,
                    ))
                }
                .map_err(|e| format!("ssm_parallel_nosave prefill L{layer_idx}: {e:?}"))?;
            } else {
                assert!(
                    ds <= k.state_cap,
                    "ssm_burnin_nosave_typed: d_state {ds} exceeds the compiled state \
                     capacity {} - the kernel returns without writing y otherwise",
                    k.state_cap
                );
                let mut bld = ctx.stream.launch_builder(k.ssm_burnin_nosave_typed.get(dt));
                bld.arg(&ssm_ptr);
                bld.arg(&gated_out_ptr);
                bld.arg(&delta_ptr);
                bld.arg(&u_ptr);
                bld.arg(&bb_ptr);
                bld.arg(&cb_ptr);
                bld.arg(&a_neg_ptr);
                bld.arg(&dp);
                bld.arg(&proj_ptr);
                bld.arg(&gate_stride);
                bld.arg(&b_i);
                bld.arg(&t_i);
                bld.arg(&di_i);
                bld.arg(&ds_i);
                unsafe { bld.launch(grid_1d(b * di)) }
                    .map_err(|e| format!("ssm_nosave prefill L{layer_idx}: {e:?}"))?;
            }
        }
        // Gating is fused into the scan store (typed round-trip
        // emulation) - the separate typed mul is gone.

        // F5: out_proj GEMM typed.
        let (opw, opw_dt) = lw.out_proj_w();
        gpu_gemm_typed_forward_raw(
            ctx,
            TypedPtr {
                ptr: scratch.out_flat.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: scratch.gated.cached_ptr(),
                dtype: dt,
            },
            TypedPtr {
                ptr: opw,
                dtype: opw_dt,
            },
            None,
            (bt, di, dm),
        )?;

        // F6: residual_add_f32_typed — residual_f32 += out_flat_bf16 (stays f32).
        {
            let n = (bt * dm) as i32;
            let mut bld = ctx.stream.launch_builder(k.residual_add_f32_typed.get(dt));
            let r_ptr = scratch.residual.cached_ptr();
            let t_ptr = scratch.out_flat.cached_ptr();
            bld.arg(&r_ptr);
            bld.arg(&r_ptr);
            bld.arg(&t_ptr);
            bld.arg(&n);
            unsafe { bld.launch(grid_1d(bt * dm)) }
                .map_err(|e| format!("residual_add_f32 prefill L{layer_idx}: {e:?}"))?;
        }
    }

    // Final norm_f: residual_f32 → out_flat_bf16 (all T timesteps).
    {
        let bt_i = bt as i32;
        let dm_i = dm as i32;
        let eps: f32 = dims.rms_norm_eps;
        let mut bld = ctx.stream.launch_builder(k.rmsnorm_fwd_f32in_typed.get(dt));
        let out_ptr = scratch.out_flat.cached_ptr();
        let rms_ptr = scratch.rms_discard.cached_ptr();
        let res_ptr = scratch.residual.cached_ptr();
        bld.arg(&out_ptr);
        bld.arg(&rms_ptr);
        bld.arg(&res_ptr);
        let nfw = weights.norm_f_weight();
        bld.arg(&nfw);
        bld.arg(&bt_i);
        bld.arg(&dm_i);
        bld.arg(&eps);
        unsafe { bld.launch(grid_norm(bt, dm)) }
            .map_err(|e| format!("norm_f prefill mixed: {e:?}"))?;
    }

    // Extract last timestep → target_temporal (bf16/f16).
    {
        let b_i = b as i32;
        let t_i = t as i32;
        let dm_i = dm as i32;
        let mut bld = ctx
            .stream
            .launch_builder(k.gather_last_timestep_typed.get(dt));
        let dst_ptr = target_temporal.cached_ptr();
        let src_ptr = scratch.out_flat.cached_ptr();
        bld.arg(&dst_ptr);
        bld.arg(&src_ptr);
        bld.arg(&b_i);
        bld.arg(&t_i);
        bld.arg(&dm_i);
        unsafe { bld.launch(grid_1d(b * dm)) }
            .map_err(|e| format!("gather_last_typed prefill: {e:?}"))?;
    }

    Ok(())
}
