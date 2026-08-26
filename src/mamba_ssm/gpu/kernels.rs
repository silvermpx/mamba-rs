//! Compile and register CUDA kernels for Mamba SSM.
//!
//! Uses NVRTC to compile .cu source to native CUBIN at runtime.
//! No pre-built PTX or binaries required.

use super::dtype::WeightDtype;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use std::sync::Arc;

/// Dtype-indexed kernel holder for activation-touching kernels.
pub struct TypedKernel {
    pub f32: CudaFunction,
    pub bf16: CudaFunction,
    pub f16: CudaFunction,
}

impl TypedKernel {
    pub fn get(&self, dt: WeightDtype) -> &CudaFunction {
        match dt {
            WeightDtype::F32 => &self.f32,
            WeightDtype::Bf16 => &self.bf16,
            WeightDtype::F16 => &self.f16,
        }
    }
}

/// Mixed-precision kernel holder with only half-dtype variants (bf16/f16).
/// Used for kernels that bridge f32 and half (e.g., `rmsnorm_fwd_f32in`
/// which reads f32 residual and writes bf16/f16 output).
pub struct HalfKernel {
    pub bf16: CudaFunction,
    pub f16: CudaFunction,
}

impl HalfKernel {
    pub fn get(&self, dt: WeightDtype) -> &CudaFunction {
        match dt {
            WeightDtype::Bf16 => &self.bf16,
            WeightDtype::F16 => &self.f16,
            WeightDtype::F32 => {
                panic!("HalfKernel has no f32 variant (use the TypedKernel f32 path instead)")
            }
        }
    }
}

/// All compiled CUDA kernels needed for Mamba forward/backward.
///
/// Kernels are compiled once via NVRTC at startup. Grouped by pipeline stage.
pub struct MambaKernels {
    /// Identity of the compiled module: the NVRTC cache key (source +
    /// arch + options + toolchain). Graph guards pin it so a replay
    /// cannot silently run kernels from a different compile.
    pub module_identity: String,
    _module: Arc<CudaModule>,

    /// State-dimension capacity the kernels were compiled with (the
    /// per-thread register-array size). The engine and trainer
    /// constructors derive it from the model config, so a mismatched
    /// launch cannot be built through the public constructors; the M1
    /// launch-path asserts additionally compare against it, and the
    /// kernels carry their own capacity guards.
    pub state_cap: usize,

    // -- SSM recurrence --
    /// Single-step SSM forward (T=1 inference).
    pub ssm_step_fwd: CudaFunction,
    /// Multi-step SSM forward with activation saves for backward.
    pub ssm_burnin_fwd: CudaFunction,
    /// Multi-step SSM forward without activation saves (target network).
    pub ssm_burnin_fwd_nosave: CudaFunction,
    /// Per-(b,t,d,n) local SSM backward: dh, du, d_delta contributions.
    pub ssm_backward_local: CudaFunction,
    /// Fused dB+dC reduction `[B*T*d_state]` each, `=`-store (no memset
    /// precondition); .get(dtype) picks the input promotion variant.
    pub ssm_reduce_d_bc_typed: TypedKernel,
    /// T-major twin for the PARALLEL route's [b][n][d][t] locals (tape
    /// layout). Same ascending-d sum, same output values and layout.
    pub ssm_reduce_d_bc_tmajor_typed: TypedKernel,
    /// Reduce local SSM grads to dD `[d_inner]`.
    pub ssm_reduce_d_d: CudaFunction,
    /// Reduce local SSM grads to d_a_log `[d_inner*d_state]`.
    pub ssm_reduce_d_a_log: CudaFunction,
    /// Chunk-partial reducer for the fold backward's d_a_log slots.
    pub ssm_reduce_d_a_log_chunks: CudaFunction,

    // -- Conv1d --
    /// Single-step depthwise conv1d forward with SiLU.
    pub conv1d_step_fwd: CudaFunction,
    /// Single-step depthwise conv1d backward.
    pub conv1d_step_bwd: CudaFunction,
    /// Multi-step conv1d forward with state saves for backward.
    pub conv1d_burnin_fwd: CudaFunction,
    /// Multi-step conv1d forward without saves (target network).
    pub conv1d_burnin_fwd_nosave: CudaFunction,
    /// Nosave tiled twin (inference prefill): tile-0 seeds from the
    /// carry-in state, later tiles from the x_branch halo - bit-identical
    /// to the serial nosave walk at two orders more parallelism.
    pub conv1d_burnin_fwd_nosave_tiled: CudaFunction,
    pub conv1d_burnin_nosave_tiled_typed: TypedKernel,
    /// Multi-step conv1d backward.
    pub conv1d_burnin_bwd: CudaFunction,

    // -- Activations --
    /// SiLU forward: `x * sigmoid(x)`.
    pub silu_fwd: CudaFunction,
    /// SiLU backward: gradient through `x * sigmoid(x)`.
    pub silu_bwd: CudaFunction,
    /// Softplus forward: `ln(1 + exp(x))`.
    pub softplus_fwd: CudaFunction,
    /// Softplus backward: gradient through `ln(1 + exp(x))`.
    pub softplus_bwd: CudaFunction,

    // -- Norms --
    /// RMSNorm forward: `x * inv_rms * scale`.
    pub rmsnorm_fwd: CudaFunction,
    /// RMSNorm backward: gradients for x and scale.
    pub rmsnorm_bwd: CudaFunction,

    // -- Elementwise (Mamba-specific) --
    /// Broadcast bias `[N]` to every row of `Y[B,N]`.
    pub bias_broadcast: CudaFunction,
    /// Column-wise sum: `db[j] += sum_b(dy[b*N + j])`.
    pub colsum_accumulate: CudaFunction,
    /// Generic 2D axis-0 tree reduce: `out[d] = [out[d] +] sum_b(partials[b * dim + d])`.
    /// Stage-2 finalizer for Rule-B per-sample partials (replaces atomicAdd
    /// in backward accumulators). Deterministic across runs.
    pub reduce_sum_axis0: CudaFunction,
    /// In-place vector add: `a[i] += b[i]`.
    pub vec_add_inplace: CudaFunction,
    /// Elementwise multiply: `c[i] = a[i] * b[i]`.
    pub elementwise_mul: CudaFunction,
    /// Negate and exponentiate: `out[i] = -exp(a_log[i])`.
    pub exp_negate: CudaFunction,
    pub exp_negate2: CudaFunction,
    /// Gather columns from a wide matrix into a contiguous buffer.
    pub gather_cols: CudaFunction,
    /// Gather B and C columns from xdbl output.
    pub gather_bc_cols: CudaFunction,
    /// T-major twin ([b][n][t] outputs) for the parallel-scan route —
    /// the scan reads B/C per (d, n) lane over consecutive t.
    pub gather_bc_cols_tmajor: CudaFunction,
    /// Staged-write twin of the t-major gather: smem transpose tile, writes
    /// t-contiguous. Falls back to the untiled kernel when the tile exceeds
    /// the 48 KB static smem budget (f32 at d_state > 186).
    pub gather_bc_cols_tmajor_tiled: CudaFunction,
    /// Scatter-add columns back into a wide matrix.
    pub scatter_add_cols: CudaFunction,
    /// Split in_proj output into x_branch and gate with SiLU on gate.
    pub split_gate_silu: CudaFunction,
    /// Split in_proj output into x_branch and gate WITHOUT materializing
    /// SiLU(gate) - the training path recomputes it at both consumers.
    pub split_gate: CudaFunction,
    /// Gating forward that recomputes SiLU(gate) from the saved pre-SiLU.
    pub gate_mul_silu: CudaFunction,
    /// Backward through gating: `y = ssm_out * gate_silu`.
    pub gating_backward: CudaFunction,
    /// Concatenate two half-vectors into one (inverse of split).
    pub concat_halves: CudaFunction,
    /// Residual add: `out[i] += residual[i]`.
    pub residual_add: CudaFunction,
    /// Copy with softplus: `out[i] = ln(1 + exp(in[i]))`.
    pub softplus_copy: CudaFunction,
    /// Gather the last timestep from `[B*T*D]` into `[B*D]`.
    pub gather_last_timestep: CudaFunction,

    // -- Mixed precision casts (mixed inference only) --
    /// f32 → bf16 downcast for weight storage.
    pub cast_f32_to_bf16: CudaFunction,
    pub cast_bf16_to_f32: CudaFunction,
    pub cast_f16_to_f32: CudaFunction,
    /// f32 → f16 downcast for weight storage.
    pub cast_f32_to_f16: CudaFunction,

    // -- Typed training-forward kernels (bf16/f16 variants of burnin save) --
    /// bf16 multi-step SSM forward with typed I/O + f32 saves.
    pub ssm_burnin_fwd_bf16: CudaFunction,
    /// f16 multi-step SSM forward with typed I/O + f32 saves.
    pub ssm_burnin_fwd_f16: CudaFunction,
    /// bf16 multi-step conv1d forward with typed I/O + f32 saves.
    pub conv1d_burnin_fwd_bf16: CudaFunction,
    /// f16 multi-step conv1d forward with typed I/O + f32 saves.
    pub conv1d_burnin_fwd_f16: CudaFunction,
    /// f32 typed-signature conv1d burnin (matches the bf16/f16 argument
    /// order `(u_out, state, conv_states_saved, post_conv, x_branch, ...)`
    /// rather than the legacy f32 `(u_out, post_conv, conv_states, state,
    /// x_branch, ...)`). Used by the mixed forward `WeightDtype::F32`
    /// branch so all three dtypes share one calling convention.
    pub conv1d_burnin_fwd_f32_typed: CudaFunction,

    // -- Typed training-backward kernels --
    /// Typed dispatch (f32/bf16/f16) for `gating_backward`. dx/dy/d_y/d_gate
    /// typed; matches DEFINE_GATING_BWD macro in elementwise.cu.
    pub gating_bwd_typed: TypedKernel,
    /// Typed dispatch for `rmsnorm_backward`. dx/dy typed; d_scale stays f32
    /// (Rule-B per-sample partials + reduce). Matches DEFINE_RMSNORM_BWD macro in
    /// norms.cu, follows NVIDIA Apex layer_norm pattern.
    pub rmsnorm_bwd_typed: TypedKernel,
    /// Tiled conv1d forward (S-conv): grid (b*di blocks, T tiles); the
    /// window seeds from x_branch halo — bit-identical to the serial walk.
    pub conv1d_burnin_fwd_tiled_typed: TypedKernel,
    /// Tiled d_x half of the conv backward (anticausal FIR, carry-seeded
    /// at tile boundaries with the serial association order).
    pub conv1d_bwd_dx_tiled_typed: TypedKernel,
    /// dw/db-only half: historical descending-t accumulation, verbatim.
    pub conv1d_bwd_dw_tiled_typed: TypedKernel,
    /// Typed dispatch for `conv1d_burnin_backward`. d_x_branch/d_u/post_conv
    /// typed; conv_states stays f32 (recurrent state); d_weight/d_bias
    /// accumulate via Rule-B partials + fixed-order reduce. Matches DEFINE_CONV1D_BURNIN_BWD
    /// macro in conv1d.cu.
    pub conv1d_burnin_bwd_typed: TypedKernel,

    // -- Typed training-backward kernels (the HOTTEST kernel) --
    /// Typed dispatch (f32/bf16/f16) for `ssm_backward_local` — the BPTT
    /// recurrence backward. delta/u/B/C/dy/d_delta/d_u/d_B_local/d_C_local
    /// typed; h_saved/a_neg/D/d_D_local/d_a_log_local stay f32 (BPTT state +
    /// T-length accumulators). Matches DEFINE_SSM_BACKWARD_LOCAL_BWD macro
    /// in mamba_ssm.cu. Validated against state-spaces/mamba reference.
    pub ssm_backward_local_typed: TypedKernel,
    /// One-kernel d_xdbl assembly (dt|B|C ranges tile the row): dt source
    /// typed, B/C sources f32 reduce outputs, FROM_F(0.0f + v) stores.
    pub pack_xdbl_cols_typed: TypedKernel,

    // -- Typed inference kernels (f32/bf16/f16 variants) --
    pub silu_fwd_typed: TypedKernel,
    pub softplus_fwd_typed: TypedKernel,
    pub rmsnorm_fwd_typed: TypedKernel,
    pub bias_broadcast_typed: TypedKernel,
    pub elementwise_mul_typed: TypedKernel,
    pub residual_add_typed: TypedKernel,
    pub gather_cols_typed: TypedKernel,
    pub gather_bc_cols_typed: TypedKernel,
    /// T-major twin of the typed gather (parallel-scan route).
    pub gather_bc_cols_tmajor_typed: TypedKernel,
    /// Staged-write twin of the typed t-major gather.
    pub gather_bc_cols_tmajor_tiled_typed: TypedKernel,
    pub split_gate_silu_typed: TypedKernel,
    /// Typed split without the post-SiLU activation.
    pub split_gate_typed: TypedKernel,
    /// Typed gating forward recomputing SiLU(gate).
    pub gate_mul_silu_typed: TypedKernel,
    /// 16-byte vectorized twins of the hot elementwise kernels: one uint4
    /// per operand per thread, same per-element arithmetic in the same
    /// order. Selected by [`vec8_ok`] when the shape and every operand
    /// pointer allow it.
    pub gate_mul_silu_v_typed: TypedKernel,
    pub elementwise_mul_v_typed: TypedKernel,
    pub softplus_copy_v_typed: TypedKernel,
    pub softplus_copy_typed: TypedKernel,
    pub ssm_step_fwd_typed: TypedKernel,
    /// SSM step with fused B/C gather from xdbl. Inference-only: replaces
    /// (gather_bc_cols + ssm_step_forward) launch pair with a single kernel,
    /// also eliminates b_buf / c_buf scratch allocations.
    pub ssm_step_fwd_gather_typed: TypedKernel,
    /// SSM step with fused B/C gather AND fused gating multiplication.
    /// Replaces (gather_bc + ssm_step + elementwise_mul) triplet.
    pub ssm_step_fwd_gather_gate_typed: TypedKernel,
    pub conv1d_step_fwd_typed: TypedKernel,
    /// Conv1d step with fused SiLU on output. Inference-only: replaces the
    /// (conv1d_step + silu_fwd) launch pair with a single kernel.
    pub conv1d_step_fwd_silu_typed: TypedKernel,
    pub ssm_burnin_nosave_typed: TypedKernel,
    pub conv1d_burnin_nosave_typed: TypedKernel,
    pub silu_bwd_typed: TypedKernel,
    pub softplus_bwd_typed: TypedKernel,
    pub gather_last_timestep_typed: TypedKernel,
    /// Typed vec_add_inplace — `a[i] += b[i]` where `a` is typed (activations
    /// or typed grad accumulator) and `b` is f32 (master bias). Used in mixed
    /// backward residual-add sequences where one operand is f32.
    pub vec_add_inplace_typed: TypedKernel,
    pub vec_cast_zplus_typed: TypedKernel,
    /// Typed concat_halves — pure load/store with typed src/dst. Used by mixed
    /// backward to concat `d_x_branch` and `d_gate_pre` into `d_proj` before
    /// the in_proj dX backward.
    pub concat_halves_typed: TypedKernel,
    /// Typed scatter_add_cols — `dst[b, off+d] += src[b, d]` with typed src/dst.
    /// Used to scatter `d_delta_raw`, `d_B`, `d_C` into the combined `d_xdbl`
    /// buffer that feeds x_proj dW backward.
    pub scatter_add_cols_typed: TypedKernel,
    /// Typed bias reduction — `d_bias[i] += sum_{b,t} dy[b, t, i]` with
    /// typed `dy` and f32 `d_bias` master grad. Used by mixed dt_proj
    /// backward (dt_proj has a learned bias; dW goes via typed GemmEx,
    /// bias grad via this launch).
    pub reduce_bias_typed: TypedKernel,

    // -- Dual-dtype kernels for end-to-end bf16/f16 inference --
    /// RMSNorm: f32 residual input → half output. Keeps residual stream in
    /// f32 across layers while feeding the branch path in bf16/f16.
    pub rmsnorm_fwd_f32in_typed: HalfKernel,
    /// Residual add: f32 accumulator + half branch → f32 output. Paired with
    /// `rmsnorm_fwd_f32in_typed` to preserve `residual_in_fp32` semantics.
    pub residual_add_f32_typed: HalfKernel,
    /// RmsNorm backward: typed `dy` + f32 `x` → f32 `dx`, f32 `d_scale`.
    /// Dual-dtype twin of `rmsnorm_fwd_f32in_typed`. Used in mixed backward
    /// per-layer rmsnorm where `d_norm` arrives typed (from in_proj dX) but
    /// the residual stream `d_pre_norm` must be f32 to accumulate into the
    /// f32 outer `d_temporal`.
    pub rmsnorm_bwd_f32in_typed: HalfKernel,

    // -- Parallel scan (optional, for T>128) --
    /// Parallel prefix scan SSM forward with activation saves.
    pub ssm_parallel_fwd: CudaFunction,
    /// Parallel prefix scan SSM forward without saves (target network).
    pub ssm_parallel_fwd_nosave: CudaFunction,
    /// Typed parallel scan forward — typed delta/u/B/C/y_out,
    /// all scan state (smem_run, block scan, h, h_saved) remains f32 per
    /// `state-spaces/mamba` `scan_t = float2` invariant.
    pub ssm_parallel_fwd_typed: TypedKernel,
    /// Typed parallel scan forward nosave twin (target network / prefill).
    pub ssm_parallel_fwd_nosave_typed: TypedKernel,

    // -- M1 parallel scan BACKWARD --
    /// Parallel reverse-scan backward, mirrors state-spaces/mamba
    /// `selective_scan_bwd_kernel.cuh`. Uses h_saved (per-t fwd state save)
    /// to skip forward re-derivation. Outputs follow the existing _local
    /// convention so the existing reduction kernels work unchanged.
    /// f32 / bf16 / f16 instantiations from one DEFINE_* macro.
    pub ssm_parallel_bwd_typed: TypedKernel,
    /// d-group fold variant: each block folds SCAN_BWD_DGROUP d lanes'
    /// dB/dC terms and writes one partial row per group. Used when
    /// d_inner is divisible by the group size; f32 staging needs the
    /// MAX_DYNAMIC_SHARED opt-in (~65 KB).
    pub ssm_parallel_bwd_fold_typed: TypedKernel,

    // -- AMP loss scaler helpers --
    /// Scan an f32 grad buffer for inf/nan, atomicOr into device int.
    pub check_inf_nan_f32: CudaFunction,
    /// In-place multiply f32 grads by a scalar (unscale, clip, etc.).
    pub scale_grads_f32: CudaFunction,
    /// CUDA-Graph-capturable conditional unscale: zeros grads if the
    /// overflow flag is set, otherwise multiplies by 1/loss_scale.
    pub scale_grads_skip_f32: CudaFunction,
    /// Deterministic global-norm support: fixed-grid sum-of-squares partial
    /// reduction with f64 accumulators (kernels/grad_clip.cu). The host sums
    /// the fixed 512 partials in order; scaling reuses `scale_grads_f32`.
    pub grad_sumsq_partial_f32: CudaFunction,
    /// Single-thread fold of the 512 partials into the PRE-clip norm and
    /// the clip coefficient, on device - removes the host drain between
    /// the norm and the scaling pass.
    pub grad_clip_coef_f32: CudaFunction,
    /// `scale_grads_f32` with the factor read from device memory.
    pub scale_grads_dev_f32: CudaFunction,
    /// Region twin of `grad_sumsq_partial_f32`: fixed-grid f64 partials
    /// over a per-layer two-block region (a strided column stripe plus a
    /// contiguous vector) of the flat arena.
    pub grad_region_sumsq_partial_f32: CudaFunction,
    /// Region twin of `scale_grads_dev_f32`: scales ONLY the described
    /// region by the device-resident clip coefficient.
    pub grad_region_scale_dev_f32: CudaFunction,

    // -- AdamW optimizer --
    /// Fused AdamW step on f32 master weights + f32 optimizer state.
    pub adamw_step_f32: CudaFunction,
    /// CUDA-Graph-capturable variant: reads bias-correction factors from a
    /// 2-elem device buffer instead of scalar args.
    pub adamw_step_f32_capturable: CudaFunction,
    /// Fused multi-tensor AdamW, one variant per shadow dtype (the f32
    /// variant serves the no-shadow lane).
    pub adamw_step_multi: TypedKernel,

    // -- Batch-invariant GEMM (bf16 cross-batch determinism fix) --
    /// Batch-invariant GEMM bf16×bf16→bf16. Tensor-Core inner GEMM via
    /// `nvcuda::wmma` (m16n16k16 fragments, f32 accumulator). Fixed
    /// 64x64x32 tile, no split-K. `C[i, j]` is bit-identical regardless
    /// of batch size M of A.
    pub gemm_bi_bf16_bf16: CudaFunction,
    /// Batch-invariant GEMM f16×f16→f16. Tensor Cores via WMMA.
    pub gemm_bi_f16_f16: CudaFunction,
    /// Batch-invariant GEMM bf16×bf16→f32 (tied lm_head: f32 logits).
    pub gemm_bi_bf16_f32: CudaFunction,
    /// Batch-invariant GEMM f16×f16→f32. Tensor Cores via WMMA.
    pub gemm_bi_f16_f32: CudaFunction,
    /// Batch-invariant GEMM f32×f32→f32. CUDA-core path (Tensor Cores
    /// require fp16/bf16/tf32 inputs; tf32 would lose 13 mantissa bits).
    pub gemm_bi_f32_f32: CudaFunction,

    // -- Batch-invariant matvec (M=1 specialization) --
    /// Specialized M=1 matvec. The GEMM kernels above waste 98% of smem
    /// bandwidth at M=1 (load BLOCK_M=64 rows, only row 0 is real).
    /// Decode (single-token per step) uses this instead — one thread per
    /// output column, K-loop with register-scalar f32 accumulator, no
    /// cross-thread reductions. Trivially batch-invariant (M=1 has no
    /// batch dim) and ~5× faster than gemm_bi_* at M=1.
    pub matvec_bi_bf16_bf16: CudaFunction,
    pub matvec_bi_f16_f16: CudaFunction,
    pub matvec_bi_bf16_f32: CudaFunction,
    pub matvec_bi_f16_f32: CudaFunction,
    pub matvec_bi_f32_f32: CudaFunction,

    // -- sgemm_bi: deterministic batch-invariant f32 training SGEMM triad --
    // (kernels/gemm_bi_triad.cu, ported from SQV-RS; siboehm warptiling + shape
    // dispatcher). Used by gpu/sgemm_bi.rs when ctx.batch_invariant() is on.
    // Big NN/TN/NT use 2-stage cp.async with 33 KB dynamic smem — loader
    // opts into the sm_80+ carveout per CUfunction.
    pub sgemm_nn: CudaFunction,
    pub sgemm_tn: CudaFunction,
    pub sgemm_nt: CudaFunction,
    pub sgemm_nn_slim: CudaFunction,
    pub sgemm_tn_slim: CudaFunction,
    pub sgemm_nt_slim: CudaFunction,
    pub sgemm_nn_ultra_thin: CudaFunction,
    pub sgemm_nn_gemv: CudaFunction,
    pub sgemm_tn_gemv: CudaFunction,
    pub sgemm_nt_gemv: CudaFunction,
    pub sgemm_nn_narrow: CudaFunction,
    pub sgemm_nn_narrow_small: CudaFunction,
    pub sgemm_tn_narrow: CudaFunction,
    pub sgemm_tn_narrow_splitm_partial: CudaFunction,
    pub sgemm_nt_narrow: CudaFunction,
    pub sgemm_nn_splitk32_partial: CudaFunction,
    pub sgemm_splitk_reduce: CudaFunction,
    pub sgemm_tn_splitm_partial: CudaFunction,
    pub sgemm_splitm_reduce: CudaFunction,
    pub sgemm_nn_splitk_big_partial: CudaFunction,
    pub sgemm_nt_splitn_big_partial: CudaFunction,
    pub sgemm_nn_splitk_slim_partial: CudaFunction,
    pub sgemm_transpose_f32_2d: CudaFunction,
    pub sgemm_dx_col_gemv: CudaFunction,
    /// Split-K/Split-M partial scratch for the sgemm_bi dispatcher:
    /// 8M f32 = 32 MB. The dispatcher asserts chunk*M*N fits before launch.
    /// LAZY: only the batch-invariant tier reads it, and
    /// inference-only consumers never enable that tier — the eager alloc
    /// held 32 MB of dead VRAM on every serve boot. Access via
    /// [`MambaKernels::splitk_scratch_buf`].
    pub splitk_scratch: std::sync::OnceLock<cudarc::driver::CudaSlice<f32>>,
    /// W-transpose staging for the bwd_dx wide path: 4M f32 = 16 MB.
    /// Lazy for the same reason; access via
    /// [`MambaKernels::transpose_scratch_buf`].
    pub transpose_scratch: std::sync::OnceLock<cudarc::driver::CudaSlice<f32>>,

    // -- sgemm_bi typed (bf16/f16) variants — typed sync-load buckets.
    // X/W/Y/dY/dX typed, dW + bias f32, f32 accumulation throughout; each
    // kernel is bit-identical to "upcast inputs to f32, run the f32 twin".
    pub sgemm_nn_gemv_typed: HalfKernel,
    pub sgemm_tn_gemv_typed: HalfKernel,
    pub sgemm_nt_gemv_typed: HalfKernel,
    pub sgemm_nn_ultra_thin_typed: HalfKernel,
    pub sgemm_nn_narrow_typed: HalfKernel,
    pub sgemm_nn_narrow_small_typed: HalfKernel,
    pub sgemm_tn_narrow_typed: HalfKernel,
    pub sgemm_nt_narrow_typed: HalfKernel,
    /// Typed Big NN/TN/NT (stage 3): bf16/f16 twins of the f32 Big kernels
    /// with sync staging + f32 smem. Dynamic smem 33 KB — the 34 KB
    /// MAX_DYNAMIC_SHARED attribute is set at load like the f32 Bigs.
    pub sgemm_nn_big_typed: HalfKernel,
    /// Tensor-core NN forward (stage 5, `bi_tensor_cores` tier) — separate
    /// numeric contract (mma.sync f32 accumulate), static smem.
    pub sgemm_nn_tc_typed: HalfKernel,
    /// The fixed family's inference ladder (kernels/gemm_bi_fixed.cu,
    /// GBF namespace): bit-identical copies of the forward TC tiles,
    /// owned by the inference kernel.
    pub gemm_bi_nn_tc128_typed: HalfKernel,
    pub gemm_bi_nn_tc64_typed: HalfKernel,
    pub gemm_bi_nn_tc16_typed: HalfKernel,
    /// The Hopper wgmma rung (compiled only for sm_90a; the dispatcher
    /// never routes here until the rung is hardware-qualified - the
    /// forced census entry is its only caller).
    pub gemm_bi_nn_sm90_typed: Option<HalfKernel>,
    pub sgemm_tn_tc_typed: HalfKernel,
    pub sgemm_nt_tc_typed: HalfKernel,
    /// 64x64-tile TC twins (stage 5b): 128 threads / 4 warps per CTA,
    /// bit-identical per element to the 128-tile TC kernels (same 32-wide
    /// reduction slabs, same mma chain). Used by the dispatcher when the
    /// 128-tile grid would underfill the GPU and for shapes with an output
    /// dim in [64, 128).
    pub sgemm_nn_tc64_typed: HalfKernel,
    /// Thin16 rung (16x32x64, 4 warps, 4-stage) - the decode end of the
    /// bit-identical TC tile ladder.
    pub sgemm_nn_tc16_typed: HalfKernel,
    pub sgemm_tn_tc64_typed: HalfKernel,
    pub sgemm_nt_tc64_typed: HalfKernel,
    pub sgemm_tn_big_typed: HalfKernel,
    pub sgemm_nt_big_typed: HalfKernel,
}

/// NVRTC library version, part of the kernel-cache key (a toolkit upgrade
/// must invalidate cached PTX). (0, 0) when the query itself fails — the
/// cache then still keys on source + arch + options.
pub(crate) fn nvrtc_version() -> (i32, i32) {
    let mut major: core::ffi::c_int = 0;
    let mut minor: core::ffi::c_int = 0;
    let rc = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if rc == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        (major, minor)
    } else {
        (0, 0)
    }
}

/// Kernel-cache directory. `MAMBA_RS_KERNEL_CACHE` overrides (a path, or
/// `0`/`off` to disable); default is `$HOME/.cache/mamba-rs/kernels`,
/// falling back to the system temp dir when HOME is absent (systemd units
/// without a HOME — exactly the environment that motivated the cache).
pub(crate) fn kernel_cache_dir() -> Option<std::path::PathBuf> {
    match std::env::var("MAMBA_RS_KERNEL_CACHE") {
        Ok(v) if matches!(v.trim(), "0" | "off" | "OFF") => None,
        Ok(v) if !v.trim().is_empty() => Some(std::path::PathBuf::from(v.trim())),
        _ => {
            let base = std::env::var("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|_| {
                    std::env::var("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
                })
                .unwrap_or_else(|_| std::env::temp_dir());
            Some(base.join("mamba-rs").join("kernels"))
        }
    }
}

/// 128-bit content key as hex: two FNV-1a-64 passes with distinct offset
/// bases over the same bytes. Not cryptographic — the cache is a local,
/// non-adversarial directory and a wrong hit requires colliding BOTH
/// lanes; a corrupt entry fails loudly at module load and is deleted.
pub(crate) fn cache_key(material: &str) -> String {
    let fnv = |seed: u64| -> u64 {
        let mut h = seed;
        for b in material.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    };
    format!(
        "{:016x}{:016x}",
        fnv(0xcbf2_9ce4_8422_2325),
        fnv(0x6c62_272e_07bb_0142)
    )
}

/// Round a model's `d_state` up to the register-array capacity the
/// kernels are compiled with. The capacity is a JIT-time knob: raising
/// it grows per-thread register pressure (the compiler spills to local
/// memory past its budget — correct, slower, accepted), so it is kept
/// as tight as the model allows. The 256 ceiling is the reference
/// implementation's own range; a larger `d_state` has no upstream
/// precedent and is refused loudly rather than silently mis-run.
pub fn state_capacity(d_state: usize) -> Result<usize, String> {
    if d_state == 0 {
        return Err("d_state must be positive".into());
    }
    if d_state > 256 {
        return Err(format!(
            "d_state {d_state} exceeds the supported range (reference \
             implementations go to 256)"
        ));
    }
    // 16-granular: the cap sizes the per-thread state register arrays in
    // every scan kernel; a 64-floor at d_state=16 made them 4x oversized,
    // which is pure local-memory traffic once the runtime-bounded loops
    // force the arrays out of registers. Callers with larger
    // states still get the exact padded fit.
    Ok(d_state.div_ceil(16) * 16)
}

impl MambaKernels {
    /// Compile with the default state capacity of 64 — the common
    /// shapes' tightest register budget. Models with a larger `d_state`
    /// use [`Self::compile_with_state_cap`].
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        Self::compile_with_state_cap(ctx, arch, 64)
    }

    /// Compile all CUDA kernels from source (NVRTC), with a disk cache for
    /// the emitted PTX: a cache hit skips the NVRTC
    /// half of the boot tax entirely; the PTX->SASS half is the driver
    /// JIT's own cache (CUDA_CACHE_PATH). Key = source blob + arch +
    /// options + NVRTC version; identical PTX by construction, so a hit
    /// cannot change a single emitted instruction. Any hit-path failure
    /// deletes the entry and falls through to a real compile; a failed
    /// compile is never cached.
    ///
    /// `state_cap` sizes the per-thread state register arrays (see
    /// [`state_capacity`]); it rides the compile options and therefore
    /// the cache key.
    pub fn compile_with_state_cap(
        ctx: &Arc<CudaContext>,
        arch: &'static str,
        state_cap: usize,
    ) -> Result<Self, String> {
        // Prelude is inlined first so templated kernels can use to_f / from_f_*
        // helpers without needing NVRTC to resolve #include "_typed_prelude.cuh"
        // (NVRTC compiles a single combined source blob, no filesystem search).
        let sources = [
            include_str!("../../../kernels/_typed_prelude.cuh"),
            include_str!("../../../kernels/mamba_ssm.cu"),
            include_str!("../../../kernels/mamba_ssm_parallel.cu"),
            include_str!("../../../kernels/conv1d.cu"),
            include_str!("../../../kernels/activations.cu"),
            include_str!("../../../kernels/norms.cu"),
            include_str!("../../../kernels/elementwise.cu"),
            include_str!("../../../kernels/loss_scaler.cu"),
            include_str!("../../../kernels/grad_clip.cu"),
            include_str!("../../../kernels/adamw.cu"),
            include_str!("../../../kernels/gemm_bi_fixed/common.cuh"),
            include_str!("../../../kernels/gemm_bi_fixed/ffma.cuh"),
            include_str!("../../../kernels/gemm_bi_fixed/wmma_legacy.cuh"),
            include_str!("../../../kernels/gemm_bi_fixed/matvec.cuh"),
            include_str!("../../../kernels/gemm_bi_fixed/mma16.cuh"),
            include_str!("../../../kernels/gemm_bi_fixed/sm90_wgmma.cuh"),
            include_str!("../../../kernels/gemm_bi_triad.cu"),
        ];

        // Strip `#include "_typed_prelude.cuh"` lines (prelude is inlined above).
        let combined: String = sources
            .iter()
            .map(|s| {
                s.lines()
                    .filter(|l| !l.trim().starts_with("#include \"_typed_prelude.cuh\""))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect::<Vec<_>>()
            .join("\n");
        // No --use_fast_math: it flushes denormals to zero and replaces
        // exp/sqrt with approximate intrinsics (__expf/__rsqrtf), which
        // breaks gradient flow through SSM BPTT chains and RMSNorm.
        // --fmad=true enables fused multiply-add (safe, precise).
        // SGB_GROUP_M: L2-swizzle row-group size for the sgemm_bi tile walker.
        // sm_80/sm_86 prefer 8 (smaller L2 working set); sm_89+ (Ada 96 MB L2,
        // Hopper, Blackwell) prefer 16. Bit-exact across values — only the
        // CTA emission order changes, never a C[m,n] reduction order.
        let group_m: usize = match arch {
            "sm_80" | "sm_86" | "sm_87" => 8,
            _ => 16,
        };
        let option_strings = vec![
            "--fmad=true".to_string(),
            "--extra-device-vectorization".to_string(),
            // Device-side assert() compiles to a live trap check
            // per call site (30 in gemm_bi_triad.cu alone, several inside the
            // register-critical TC main loops). NDEBUG removes the checks;
            // asserts compute no values, so outputs are bit-identical
            // (digest-gated).
            "-DNDEBUG".to_string(),
            format!("-DSGB_GROUP_M={group_m}"),
            format!("-DMAMBA_RS_STATE_CAP={state_cap}"),
        ];
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: option_strings.clone(),
            include_paths: cuda_include_paths(),
            ..Default::default()
        };

        let (nv_major, nv_minor) = nvrtc_version();
        // include_paths participate in the key: a box with two CUDA
        // toolkits must not serve stale PTX under an unchanged key.
        let key = cache_key(&format!(
            "{combined}\u{1f}{arch}\u{1f}{option_strings:?}\u{1f}{:?}\u{1f}nvrtc{nv_major}.{nv_minor}",
            opts.include_paths
        ));
        let cache_path = kernel_cache_dir().map(|d| d.join(format!("mamba-kernels-{key}.ptx")));

        // Cache hit: load the stored PTX. A module that fails to load from
        // a cached entry (torn write, disk rot) invalidates it and falls
        // through to the real compile.
        let mut module = None;
        if let Some(path) = &cache_path
            && let Ok(src) = std::fs::read_to_string(path)
        {
            match ctx.load_module(cudarc::nvrtc::Ptx::from_src(src)) {
                Ok(m) => module = Some(m),
                Err(_) => {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        let module = match module {
            Some(m) => m,
            None => {
                let ptx = cudarc::nvrtc::compile_ptx_with_opts(combined, opts).map_err(|e| {
                    // Unescape the compiler log - a 40-error NVRTC
                    // failure as one escaped single-line blob is
                    // unreadable exactly when it matters most.
                    format!(
                        "NVRTC compile failed: {}",
                        format!("{e:?}").replace("\\n", "\n")
                    )
                })?;
                if let Some(path) = &cache_path
                    && let Some(dir) = path.parent()
                    && std::fs::create_dir_all(dir).is_ok()
                {
                    // Atomic publish: write-then-rename so a concurrent boot
                    // never reads a torn entry. Failures are non-fatal — the
                    // cache is an accelerator, never a correctness gate.
                    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
                    if std::fs::write(&tmp, ptx.to_src()).is_ok()
                        && std::fs::rename(&tmp, path).is_err()
                    {
                        // A failed rename must not leak the tmp entry.
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
                ctx.load_module(ptx)
                    .map_err(|e| format!("Module load failed: {e:?}"))?
            }
        };

        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("Kernel '{name}' not found: {e:?}"))
        };
        let load_typed = |base: &str| -> Result<TypedKernel, String> {
            Ok(TypedKernel {
                f32: get(&format!("{base}_f32"))?,
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        let load_half = |base: &str| -> Result<HalfKernel, String> {
            Ok(HalfKernel {
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        // Like `load_half`, plus the MAX_DYNAMIC_SHARED carveout for kernels
        // whose staging exceeds the 48 KB static cap: 34 KB for the typed
        // Big tiles (2-stage f32 smem = 33 KB), 74 KB for the BK=64 TC
        // family (NN 70 KB / TN 68 KB / NT 72 KB, padded bf16/f16).
        let load_half_dynsmem = |base: &str, bytes: i32| -> Result<HalfKernel, String> {
            let k = load_half(base)?;
            for f in [&k.bf16, &k.f16] {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes,
                )
                .map_err(|e| format!("set MAX_DYNAMIC_SHARED for {base}: {e:?}"))?;
            }
            Ok(k)
        };

        Ok(Self {
            module_identity: key.clone(),
            state_cap,
            // SSM
            ssm_step_fwd: get("ssm_step_forward")?,
            ssm_burnin_fwd: get("ssm_burnin_forward")?,
            ssm_burnin_fwd_nosave: get("ssm_burnin_forward_nosave")?,
            ssm_backward_local: get("ssm_backward_local")?,
            ssm_reduce_d_bc_typed: TypedKernel {
                f32: get("ssm_reduce_d_BC_f32")?,
                bf16: get("ssm_reduce_d_BC_bf16")?,
                f16: get("ssm_reduce_d_BC_f16")?,
            },
            ssm_reduce_d_bc_tmajor_typed: TypedKernel {
                f32: get("ssm_reduce_d_BC_tmajor_f32")?,
                bf16: get("ssm_reduce_d_BC_tmajor_bf16")?,
                f16: get("ssm_reduce_d_BC_tmajor_f16")?,
            },
            ssm_reduce_d_d: get("ssm_reduce_d_D")?,
            ssm_reduce_d_a_log: get("ssm_reduce_d_a_log")?,
            ssm_reduce_d_a_log_chunks: get("ssm_reduce_d_a_log_chunks")?,
            // conv1d
            conv1d_step_fwd: get("conv1d_step_forward")?,
            conv1d_step_bwd: get("conv1d_step_backward")?,
            conv1d_burnin_fwd: get("conv1d_burnin_forward")?,
            conv1d_burnin_fwd_nosave: get("conv1d_burnin_forward_nosave")?,
            conv1d_burnin_fwd_nosave_tiled: get("conv1d_burnin_forward_nosave_tiled_f32")?,
            conv1d_burnin_nosave_tiled_typed: load_typed("conv1d_burnin_forward_nosave_tiled")?,
            conv1d_burnin_bwd: get("conv1d_burnin_backward")?,
            // activations
            silu_fwd: get("silu_forward")?,
            silu_bwd: get("silu_backward")?,
            softplus_fwd: get("softplus_forward")?,
            softplus_bwd: get("softplus_backward")?,
            // norms
            rmsnorm_fwd: get("rmsnorm_forward")?,
            rmsnorm_bwd: get("rmsnorm_backward")?,
            // elementwise
            bias_broadcast: get("bias_broadcast")?,
            colsum_accumulate: get("colsum_accumulate")?,
            reduce_sum_axis0: get("reduce_sum_axis0")?,
            vec_add_inplace: get("vec_add_inplace")?,
            elementwise_mul: get("elementwise_mul")?,
            exp_negate: get("exp_negate")?,
            exp_negate2: get("exp_negate2")?,
            gather_cols: get("gather_cols")?,
            gather_bc_cols: get("gather_bc_cols")?,
            gather_bc_cols_tmajor: get("gather_bc_cols_tmajor")?,
            gather_bc_cols_tmajor_tiled: get("gather_bc_cols_tmajor_tiled")?,
            scatter_add_cols: get("scatter_add_cols")?,
            split_gate_silu: get("split_gate_silu")?,
            split_gate: get("split_gate")?,
            gate_mul_silu: get("gate_mul_silu")?,
            gating_backward: get("gating_backward")?,
            concat_halves: get("concat_halves")?,
            residual_add: get("residual_add")?,
            softplus_copy: get("softplus_copy")?,
            gather_last_timestep: get("gather_last_timestep")?,

            // mixed precision casts
            cast_f32_to_bf16: get("cast_f32_to_bf16")?,
            cast_f32_to_f16: get("cast_f32_to_f16")?,
            cast_bf16_to_f32: get("cast_bf16_to_f32")?,
            cast_f16_to_f32: get("cast_f16_to_f32")?,
            ssm_burnin_fwd_bf16: get("ssm_burnin_forward_bf16")?,
            ssm_burnin_fwd_f16: get("ssm_burnin_forward_f16")?,
            conv1d_burnin_fwd_bf16: get("conv1d_burnin_forward_bf16")?,
            conv1d_burnin_fwd_f16: get("conv1d_burnin_forward_f16")?,
            conv1d_burnin_fwd_f32_typed: get("conv1d_burnin_forward_f32")?,

            // parallel scan
            ssm_parallel_fwd: get("ssm_parallel_scan_fwd")?,
            ssm_parallel_fwd_nosave: get("ssm_parallel_scan_fwd_nosave")?,
            ssm_parallel_fwd_typed: TypedKernel {
                f32: get("ssm_parallel_scan_fwd")?,
                bf16: get("ssm_parallel_scan_fwd_bf16")?,
                f16: get("ssm_parallel_scan_fwd_f16")?,
            },
            ssm_parallel_fwd_nosave_typed: TypedKernel {
                f32: get("ssm_parallel_scan_fwd_nosave")?,
                bf16: get("ssm_parallel_scan_fwd_nosave_bf16")?,
                f16: get("ssm_parallel_scan_fwd_nosave_f16")?,
            },
            ssm_parallel_bwd_typed: TypedKernel {
                f32: get("ssm_parallel_scan_bwd_f32")?,
                bf16: get("ssm_parallel_scan_bwd_bf16")?,
                f16: get("ssm_parallel_scan_bwd_f16")?,
            },
            ssm_parallel_bwd_fold_typed: {
                let k = TypedKernel {
                    f32: get("ssm_parallel_scan_bwd_fold_f32")?,
                    bf16: get("ssm_parallel_scan_bwd_fold_bf16")?,
                    f16: get("ssm_parallel_scan_bwd_fold_f16")?,
                };
                // The f32 delta/u/dy stage is ~49 KB on top of the ~16 KB
                // f32 workspace — past the 48 KB static cap.
                for f in [&k.f32, &k.bf16, &k.f16] {
                    f.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        67_584,
                    )
                    .map_err(|e| {
                        format!("set MAX_DYNAMIC_SHARED for scan_bwd_fold: {e:?}")
                    })?;
                }
                k
            },

            // AMP loss scaler
            check_inf_nan_f32: get("check_inf_nan_f32")?,
            scale_grads_f32: get("scale_grads_f32")?,
            scale_grads_skip_f32: get("scale_grads_skip_f32")?,
            grad_sumsq_partial_f32: get("grad_sumsq_partial_f32")?,
            grad_clip_coef_f32: get("grad_clip_coef_f32")?,
            scale_grads_dev_f32: get("scale_grads_dev_f32")?,
            grad_region_sumsq_partial_f32: get("grad_region_sumsq_partial_f32")?,
            grad_region_scale_dev_f32: get("grad_region_scale_dev_f32")?,

            // AdamW
            adamw_step_f32: get("adamw_step_f32")?,
            adamw_step_f32_capturable: get("adamw_step_f32_capturable")?,
            adamw_step_multi: load_typed("adamw_step_multi")?,

            // Batch-invariant GEMM
            gemm_bi_bf16_bf16: get("gemm_bi_bf16_bf16")?,
            gemm_bi_f16_f16: get("gemm_bi_f16_f16")?,
            gemm_bi_bf16_f32: get("gemm_bi_bf16_f32")?,
            gemm_bi_f16_f32: get("gemm_bi_f16_f32")?,
            gemm_bi_f32_f32: get("gemm_bi_f32_f32")?,

            // Batch-invariant matvec (M=1 specialization)
            matvec_bi_bf16_bf16: get("matvec_bi_bf16_bf16")?,
            matvec_bi_f16_f16: get("matvec_bi_f16_f16")?,
            matvec_bi_bf16_f32: get("matvec_bi_bf16_f32")?,
            matvec_bi_f16_f32: get("matvec_bi_f16_f32")?,
            matvec_bi_f32_f32: get("matvec_bi_f32_f32")?,

            // typed inference kernels
            silu_fwd_typed: load_typed("silu_forward")?,
            softplus_fwd_typed: load_typed("softplus_forward")?,
            rmsnorm_fwd_typed: load_typed("rmsnorm_forward")?,
            bias_broadcast_typed: load_typed("bias_broadcast")?,
            elementwise_mul_typed: load_typed("elementwise_mul")?,
            residual_add_typed: load_typed("residual_add")?,
            gather_cols_typed: load_typed("gather_cols")?,
            gather_bc_cols_typed: load_typed("gather_bc_cols")?,
            gather_bc_cols_tmajor_typed: load_typed("gather_bc_cols_tmajor")?,
            gather_bc_cols_tmajor_tiled_typed: load_typed("gather_bc_cols_tmajor_tiled")?,
            split_gate_silu_typed: load_typed("split_gate_silu")?,
            split_gate_typed: load_typed("split_gate")?,
            gate_mul_silu_typed: load_typed("gate_mul_silu")?,
            gate_mul_silu_v_typed: load_typed("gate_mul_silu_v")?,
            elementwise_mul_v_typed: load_typed("elementwise_mul_v")?,
            softplus_copy_v_typed: load_typed("softplus_copy_v")?,
            softplus_copy_typed: load_typed("softplus_copy")?,
            ssm_step_fwd_typed: load_typed("ssm_step_forward")?,
            ssm_step_fwd_gather_typed: load_typed("ssm_step_forward_gather")?,
            ssm_step_fwd_gather_gate_typed: load_typed("ssm_step_forward_gather_gate")?,
            conv1d_step_fwd_typed: load_typed("conv1d_step_forward")?,
            conv1d_step_fwd_silu_typed: load_typed("conv1d_step_forward_silu")?,
            ssm_burnin_nosave_typed: load_typed("ssm_burnin_forward_nosave")?,
            conv1d_burnin_nosave_typed: load_typed("conv1d_burnin_forward_nosave")?,
            silu_bwd_typed: load_typed("silu_backward")?,
            softplus_bwd_typed: load_typed("softplus_backward")?,
            gather_last_timestep_typed: load_typed("gather_last_timestep")?,
            vec_add_inplace_typed: load_typed("vec_add_inplace")?,
            vec_cast_zplus_typed: load_typed("vec_cast_zplus")?,
            concat_halves_typed: load_typed("concat_halves")?,
            scatter_add_cols_typed: load_typed("scatter_add_cols")?,
            reduce_bias_typed: load_typed("reduce_bias")?,

            // typed training-backward kernels
            gating_bwd_typed: load_typed("gating_backward")?,
            rmsnorm_bwd_typed: load_typed("rmsnorm_backward")?,
            conv1d_burnin_bwd_typed: load_typed("conv1d_burnin_backward")?,
            conv1d_burnin_fwd_tiled_typed: load_typed("conv1d_burnin_forward_tiled")?,
            conv1d_bwd_dx_tiled_typed: load_typed("conv1d_bwd_dx_tiled")?,
            conv1d_bwd_dw_tiled_typed: load_typed("conv1d_bwd_dw_tiled")?,
            // ssm_backward_local typed + typed-input reducers
            ssm_backward_local_typed: load_typed("ssm_backward_local")?,
            pack_xdbl_cols_typed: TypedKernel {
                f32: get("pack_xdbl_cols_f32")?,
                bf16: get("pack_xdbl_cols_bf16")?,
                f16: get("pack_xdbl_cols_f16")?,
            },

            // dual-dtype (half-only)
            rmsnorm_fwd_f32in_typed: load_half("rmsnorm_forward_f32in")?,
            rmsnorm_bwd_f32in_typed: load_half("rmsnorm_backward_f32in")?,
            residual_add_f32_typed: load_half("residual_add_f32")?,

            // -- sgemm_bi triad --
            sgemm_nn: {
                let f = get("sgemm_bi_nn")?;
                // Big NN: 2-stage cp.async, 33 KB dynamic smem (> 48 KB
                // static cap). Opt into the sm_80+ carveout per function.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    34 * 1024,
                )
                .map_err(|e| format!("set MAX_DYNAMIC_SHARED for sgemm_nn: {e:?}"))?;
                f
            },
            sgemm_tn: {
                let f = get("sgemm_bi_tn")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    34 * 1024,
                )
                .map_err(|e| format!("set MAX_DYNAMIC_SHARED for sgemm_tn: {e:?}"))?;
                f
            },
            sgemm_nt: {
                let f = get("sgemm_bi_nt")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    34 * 1024,
                )
                .map_err(|e| format!("set MAX_DYNAMIC_SHARED for sgemm_nt: {e:?}"))?;
                f
            },
            sgemm_nn_slim: get("sgemm_bi_nn_slim")?,
            sgemm_tn_slim: get("sgemm_bi_tn_slim")?,
            sgemm_nt_slim: get("sgemm_bi_nt_slim")?,
            sgemm_nn_ultra_thin: get("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_gemv: get("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv: get("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv: get("sgemm_bi_nt_gemv")?,
            sgemm_nn_narrow: get("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small: get("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow: get("sgemm_bi_tn_narrow")?,
            sgemm_tn_narrow_splitm_partial: get("sgemm_bi_tn_narrow_splitm_partial")?,
            sgemm_nt_narrow: get("sgemm_bi_nt_narrow")?,
            sgemm_nn_splitk32_partial: get("sgemm_bi_nn_splitk32_partial")?,
            sgemm_splitk_reduce: get("sgemm_bi_splitk_reduce")?,
            sgemm_tn_splitm_partial: get("sgemm_bi_tn_splitm_partial")?,
            sgemm_splitm_reduce: get("sgemm_bi_splitm_reduce")?,
            sgemm_nn_splitk_big_partial: {
                let f = get("sgemm_bi_nn_splitk_big_partial")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    34 * 1024,
                )
                .map_err(|e| {
                    format!("set MAX_DYNAMIC_SHARED for sgemm_nn_splitk_big_partial: {e:?}")
                })?;
                f
            },
            sgemm_nt_splitn_big_partial: {
                let f = get("sgemm_bi_nt_splitn_big_partial")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    34 * 1024,
                )
                .map_err(|e| {
                    format!("set MAX_DYNAMIC_SHARED for sgemm_nt_splitn_big_partial: {e:?}")
                })?;
                f
            },
            sgemm_nn_splitk_slim_partial: get("sgemm_bi_nn_splitk_slim_partial")?,
            sgemm_transpose_f32_2d: get("sgemm_transpose_f32_2d")?,
            sgemm_dx_col_gemv: get("sgemm_bi_dx_col_gemv")?,
            splitk_scratch: std::sync::OnceLock::new(),
            transpose_scratch: std::sync::OnceLock::new(),
            sgemm_nn_gemv_typed: load_half("sgemm_bi_nn_gemv")?,
            sgemm_tn_gemv_typed: load_half("sgemm_bi_tn_gemv")?,
            sgemm_nt_gemv_typed: load_half("sgemm_bi_nt_gemv")?,
            sgemm_nn_ultra_thin_typed: load_half("sgemm_bi_nn_ultra_thin")?,
            sgemm_nn_narrow_typed: load_half("sgemm_bi_nn_narrow")?,
            sgemm_nn_narrow_small_typed: load_half("sgemm_bi_nn_narrow_small")?,
            sgemm_tn_narrow_typed: load_half("sgemm_bi_tn_narrow")?,
            sgemm_nt_narrow_typed: load_half("sgemm_bi_nt_narrow")?,
            sgemm_nn_big_typed: load_half_dynsmem("sgemm_bi_nn_big", 34 * 1024)?,
            sgemm_nn_tc_typed: load_half_dynsmem("sgemm_bi_nn_tc", 75_776)?,
            gemm_bi_nn_tc128_typed: load_half_dynsmem("gemm_bi_nn_tc128", 71_680)?,
            gemm_bi_nn_tc64_typed: load_half("gemm_bi_nn_tc64")?,
            gemm_bi_nn_tc16_typed: load_half("gemm_bi_nn_tc16")?,
            gemm_bi_nn_sm90_typed: if arch == "sm_90a" {
                Some(load_half_dynsmem("gemm_bi_nn_sm90a_wgmma_wg1", 49_152)?)
            } else {
                None
            },
            sgemm_tn_tc_typed: load_half_dynsmem("sgemm_bi_tn_tc", 75_776)?,
            sgemm_nt_tc_typed: load_half_dynsmem("sgemm_bi_nt_tc", 75_776)?,
            sgemm_nn_tc64_typed: load_half("sgemm_bi_nn_tc64")?,
            sgemm_nn_tc16_typed: load_half("sgemm_bi_nn_tc16")?,
            sgemm_tn_tc64_typed: load_half("sgemm_bi_tn_tc64")?,
            sgemm_nt_tc64_typed: load_half("sgemm_bi_nt_tc64")?,
            sgemm_tn_big_typed: load_half_dynsmem("sgemm_bi_tn_big", 34 * 1024)?,
            sgemm_nt_big_typed: load_half_dynsmem("sgemm_bi_nt_big", 34 * 1024)?,

            _module: module,
        })
    }

    /// The Split-K/Split-M partial scratch (8M f32 = 32 MB), allocated on
    /// first batch-invariant use. Zero-initialized like the old eager
    /// alloc; the partial kernels overwrite their region before the
    /// reduce reads it, so first-use contents were never load-bearing.
    pub fn splitk_scratch_buf(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<&cudarc::driver::CudaSlice<f32>, String> {
        if self.splitk_scratch.get().is_none() {
            let buf = stream
                .alloc_zeros::<f32>(1 << 23)
                .map_err(|e| format!("splitk_scratch alloc: {e:?}"))?;
            // A concurrent racer's set loses and drops its buffer — safe.
            let _ = self.splitk_scratch.set(buf);
        }
        self.splitk_scratch
            .get()
            .ok_or_else(|| "splitk_scratch cell empty after init".to_string())
    }

    /// The W-transpose staging scratch (4M f32 = 16 MB), allocated on
    /// first batch-invariant bwd_dx wide-path use.
    pub fn transpose_scratch_buf(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<&cudarc::driver::CudaSlice<f32>, String> {
        if self.transpose_scratch.get().is_none() {
            let buf = stream
                .alloc_zeros::<f32>(1 << 22)
                .map_err(|e| format!("transpose_scratch alloc: {e:?}"))?;
            let _ = self.transpose_scratch.set(buf);
        }
        self.transpose_scratch
            .get()
            .ok_or_else(|| "transpose_scratch cell empty after init".to_string())
    }
}

/// Discover CUDA include directory (for cuda_fp16.h, cuda_bf16.h).
/// Checks CUDA_HOME, CUDA_PATH, CUDA_ROOT, then standard install paths.
pub fn cuda_include_paths() -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(p) = std::env::var(var) {
            candidates.push(format!("{p}/include"));
        }
    }
    for std_path in [
        "/usr/local/cuda/include",
        "/usr/local/cuda-13.2/include",
        "/usr/local/cuda-12.8/include",
        "/usr/local/cuda-12.6/include",
        "/usr/local/cuda-12.4/include",
        "/usr/local/cuda-12.2/include",
        "/opt/cuda/include",
    ] {
        candidates.push(std_path.to_string());
    }
    candidates
        .into_iter()
        .filter(|p| std::path::Path::new(p).join("cuda_fp16.h").exists())
        .collect()
}
