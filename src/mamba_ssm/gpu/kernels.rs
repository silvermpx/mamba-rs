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
    _module: Arc<CudaModule>,

    // -- SSM recurrence --
    /// Single-step SSM forward (T=1 inference).
    pub ssm_step_fwd: CudaFunction,
    /// Multi-step SSM forward with activation saves for backward.
    pub ssm_burnin_fwd: CudaFunction,
    /// Multi-step SSM forward without activation saves (target network).
    pub ssm_burnin_fwd_nosave: CudaFunction,
    /// Per-(b,t,d,n) local SSM backward: dh, du, d_delta contributions.
    pub ssm_backward_local: CudaFunction,
    /// Reduce local SSM grads to dB `[B*T*d_state]`.
    pub ssm_reduce_d_b: CudaFunction,
    /// Reduce local SSM grads to dC `[B*T*d_state]`.
    pub ssm_reduce_d_c: CudaFunction,
    /// Reduce local SSM grads to dD `[d_inner]`.
    pub ssm_reduce_d_d: CudaFunction,
    /// Reduce local SSM grads to d_a_log `[d_inner*d_state]`.
    pub ssm_reduce_d_a_log: CudaFunction,

    // -- Conv1d --
    /// Single-step depthwise conv1d forward with SiLU.
    pub conv1d_step_fwd: CudaFunction,
    /// Single-step depthwise conv1d backward.
    pub conv1d_step_bwd: CudaFunction,
    /// Multi-step conv1d forward with state saves for backward.
    pub conv1d_burnin_fwd: CudaFunction,
    /// Multi-step conv1d forward without saves (target network).
    pub conv1d_burnin_fwd_nosave: CudaFunction,
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
    /// Gather columns from a wide matrix into a contiguous buffer.
    pub gather_cols: CudaFunction,
    /// Gather B and C columns from xdbl output.
    pub gather_bc_cols: CudaFunction,
    /// Scatter-add columns back into a wide matrix.
    pub scatter_add_cols: CudaFunction,
    /// Split in_proj output into x_branch and gate with SiLU on gate.
    pub split_gate_silu: CudaFunction,
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

    // -- Typed training-backward kernels (Step 4a) --
    /// Typed dispatch (f32/bf16/f16) for `gating_backward`. dx/dy/d_y/d_gate
    /// typed; matches DEFINE_GATING_BWD macro in elementwise.cu.
    pub gating_bwd_typed: TypedKernel,
    /// Typed dispatch for `rmsnorm_backward`. dx/dy typed; d_scale stays f32
    /// (Rule-B per-sample partials + reduce). Matches DEFINE_RMSNORM_BWD macro in
    /// norms.cu, follows NVIDIA Apex layer_norm pattern.
    pub rmsnorm_bwd_typed: TypedKernel,
    /// Typed dispatch for `conv1d_burnin_backward`. d_x_branch/d_u/post_conv
    /// typed; conv_states stays f32 (recurrent state); d_weight/d_bias
    /// accumulate via Rule-B partials + fixed-order reduce. Matches DEFINE_CONV1D_BURNIN_BWD
    /// macro in conv1d.cu.
    pub conv1d_burnin_bwd_typed: TypedKernel,

    // -- Typed training-backward kernels (Step 4b — HOTTEST kernel) --
    /// Typed dispatch (f32/bf16/f16) for `ssm_backward_local` — the BPTT
    /// recurrence backward. delta/u/B/C/dy/d_delta/d_u/d_B_local/d_C_local
    /// typed; h_saved/a_neg/D/d_D_local/d_a_log_local stay f32 (BPTT state +
    /// T-length accumulators). Matches DEFINE_SSM_BACKWARD_LOCAL_BWD macro
    /// in mamba_ssm.cu. Validated against state-spaces/mamba reference.
    pub ssm_backward_local_typed: TypedKernel,
    /// Typed-input variant of `ssm_reduce_d_B` (output stays f32 master).
    /// Used when ssm_backward_local writes typed d_B_local. Promote each
    /// contribution to f32 in the inner sum, write f32 sum.
    pub ssm_reduce_d_b_bf16: CudaFunction,
    pub ssm_reduce_d_b_f16: CudaFunction,
    pub ssm_reduce_d_c_bf16: CudaFunction,
    pub ssm_reduce_d_c_f16: CudaFunction,

    // -- Typed inference kernels (f32/bf16/f16 variants) --
    pub silu_fwd_typed: TypedKernel,
    pub softplus_fwd_typed: TypedKernel,
    pub rmsnorm_fwd_typed: TypedKernel,
    pub bias_broadcast_typed: TypedKernel,
    pub elementwise_mul_typed: TypedKernel,
    pub residual_add_typed: TypedKernel,
    pub gather_cols_typed: TypedKernel,
    pub gather_bc_cols_typed: TypedKernel,
    pub split_gate_silu_typed: TypedKernel,
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
    /// Typed parallel scan forward (Step 8b) — typed delta/u/B/C/y_out,
    /// all scan state (smem_run, block scan, h, h_saved) remains f32 per
    /// `state-spaces/mamba` `scan_t = float2` invariant.
    pub ssm_parallel_fwd_typed: TypedKernel,
    /// Typed parallel scan forward nosave twin (target network / prefill).
    pub ssm_parallel_fwd_nosave_typed: TypedKernel,

    // -- Step 8e: M1 parallel scan BACKWARD (new) --
    /// Parallel reverse-scan backward, mirrors state-spaces/mamba
    /// `selective_scan_bwd_kernel.cuh`. Uses h_saved (per-t fwd state save)
    /// to skip forward re-derivation. Outputs follow the existing _local
    /// convention so the existing reduction kernels work unchanged.
    /// f32 / bf16 / f16 instantiations from one DEFINE_* macro.
    pub ssm_parallel_bwd_typed: TypedKernel,

    // -- AMP loss scaler helpers (Step 13) --
    /// Scan an f32 grad buffer for inf/nan, atomicOr into device int.
    pub check_inf_nan_f32: CudaFunction,
    /// In-place multiply f32 grads by a scalar (unscale, clip, etc.).
    pub scale_grads_f32: CudaFunction,
    /// CUDA-Graph-capturable conditional unscale: zeros grads if the
    /// overflow flag is set, otherwise multiplies by 1/loss_scale (Step 22).
    pub scale_grads_skip_f32: CudaFunction,

    // -- AdamW optimizer (Step 12) --
    /// Fused AdamW step on f32 master weights + f32 optimizer state.
    pub adamw_step_f32: CudaFunction,
    /// CUDA-Graph-capturable variant: reads bias-correction factors from a
    /// 2-elem device buffer instead of scalar args (Step 14).
    pub adamw_step_f32_capturable: CudaFunction,

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
    // (kernels/sgemm_bi.cu, ported from SQV-RS; siboehm warptiling + shape
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
    pub splitk_scratch: cudarc::driver::CudaSlice<f32>,
    /// W-transpose staging for the bwd_dx wide path: 4M f32 = 16 MB.
    pub transpose_scratch: cudarc::driver::CudaSlice<f32>,

    // -- sgemm_bi typed (bf16/f16) variants — Phase 11 stage 2 buckets.
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
    pub sgemm_tn_tc_typed: HalfKernel,
    pub sgemm_nt_tc_typed: HalfKernel,
    /// 64x64-tile TC twins (stage 5b): 128 threads / 4 warps per CTA,
    /// bit-identical per element to the 128-tile TC kernels (same 32-wide
    /// reduction slabs, same mma chain). Used by the dispatcher when the
    /// 128-tile grid would underfill the GPU and for shapes with an output
    /// dim in [64, 128).
    pub sgemm_nn_tc64_typed: HalfKernel,
    pub sgemm_tn_tc64_typed: HalfKernel,
    pub sgemm_nt_tc64_typed: HalfKernel,
    pub sgemm_tn_big_typed: HalfKernel,
    pub sgemm_nt_big_typed: HalfKernel,
}

impl MambaKernels {
    /// Compile all CUDA kernels from source. Takes ~100-200ms.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
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
            include_str!("../../../kernels/adamw.cu"),
            include_str!("../../../kernels/gemm_batch_invariant.cu"),
            include_str!("../../../kernels/sgemm_bi.cu"),
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
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
                format!("-DSGB_GROUP_M={group_m}"),
            ],
            include_paths: cuda_include_paths(),
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(combined, opts)
            .map_err(|e| format!("NVRTC compile failed: {e:?}"))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Module load failed: {e:?}"))?;

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
            // SSM
            ssm_step_fwd: get("ssm_step_forward")?,
            ssm_burnin_fwd: get("ssm_burnin_forward")?,
            ssm_burnin_fwd_nosave: get("ssm_burnin_forward_nosave")?,
            ssm_backward_local: get("ssm_backward_local")?,
            ssm_reduce_d_b: get("ssm_reduce_d_B")?,
            ssm_reduce_d_c: get("ssm_reduce_d_C")?,
            ssm_reduce_d_d: get("ssm_reduce_d_D")?,
            ssm_reduce_d_a_log: get("ssm_reduce_d_a_log")?,
            // conv1d
            conv1d_step_fwd: get("conv1d_step_forward")?,
            conv1d_step_bwd: get("conv1d_step_backward")?,
            conv1d_burnin_fwd: get("conv1d_burnin_forward")?,
            conv1d_burnin_fwd_nosave: get("conv1d_burnin_forward_nosave")?,
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
            gather_cols: get("gather_cols")?,
            gather_bc_cols: get("gather_bc_cols")?,
            scatter_add_cols: get("scatter_add_cols")?,
            split_gate_silu: get("split_gate_silu")?,
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

            // AMP loss scaler
            check_inf_nan_f32: get("check_inf_nan_f32")?,
            scale_grads_f32: get("scale_grads_f32")?,
            scale_grads_skip_f32: get("scale_grads_skip_f32")?,

            // AdamW
            adamw_step_f32: get("adamw_step_f32")?,
            adamw_step_f32_capturable: get("adamw_step_f32_capturable")?,

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
            split_gate_silu_typed: load_typed("split_gate_silu")?,
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
            concat_halves_typed: load_typed("concat_halves")?,
            scatter_add_cols_typed: load_typed("scatter_add_cols")?,
            reduce_bias_typed: load_typed("reduce_bias")?,

            // typed training-backward kernels (Step 4a)
            gating_bwd_typed: load_typed("gating_backward")?,
            rmsnorm_bwd_typed: load_typed("rmsnorm_backward")?,
            conv1d_burnin_bwd_typed: load_typed("conv1d_burnin_backward")?,
            // Step 4b: ssm_backward_local typed + typed-input reducers
            ssm_backward_local_typed: load_typed("ssm_backward_local")?,
            ssm_reduce_d_b_bf16: get("ssm_reduce_d_B_bf16")?,
            ssm_reduce_d_b_f16: get("ssm_reduce_d_B_f16")?,
            ssm_reduce_d_c_bf16: get("ssm_reduce_d_C_bf16")?,
            ssm_reduce_d_c_f16: get("ssm_reduce_d_C_f16")?,

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
            splitk_scratch: ctx
                .default_stream()
                .alloc_zeros::<f32>(1 << 23)
                .map_err(|e| format!("splitk_scratch alloc: {e:?}"))?,
            transpose_scratch: ctx
                .default_stream()
                .alloc_zeros::<f32>(1 << 22)
                .map_err(|e| format!("transpose_scratch alloc: {e:?}"))?,
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
            sgemm_tn_tc_typed: load_half_dynsmem("sgemm_bi_tn_tc", 75_776)?,
            sgemm_nt_tc_typed: load_half_dynsmem("sgemm_bi_nt_tc", 75_776)?,
            sgemm_nn_tc64_typed: load_half("sgemm_bi_nn_tc64")?,
            sgemm_tn_tc64_typed: load_half("sgemm_bi_tn_tc64")?,
            sgemm_nt_tc64_typed: load_half("sgemm_bi_nt_tc64")?,
            sgemm_tn_big_typed: load_half_dynsmem("sgemm_bi_tn_big", 34 * 1024)?,
            sgemm_nt_big_typed: load_half_dynsmem("sgemm_bi_nt_big", 34 * 1024)?,

            _module: module,
        })
    }
}

/// Discover CUDA include directory (for cuda_fp16.h, cuda_bf16.h).
/// Checks CUDA_HOME, CUDA_PATH, CUDA_ROOT, then standard install paths.
pub(crate) fn cuda_include_paths() -> Vec<String> {
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
