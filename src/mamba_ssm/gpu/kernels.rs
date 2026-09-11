//! Compile and register CUDA kernels for Mamba SSM.
//!
//! Uses NVRTC to compile `.cu` source to PTX. The CUDA Driver JIT loads the
//! PTX for the active device; no pre-built binary is required.

use super::dtype::WeightDtype;
use super::gemm_bi_triad::GemmBiKernels;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use std::ops::Deref;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct CudaModuleAnchors {
    _modules: Arc<[Arc<CudaModule>]>,
}

impl CudaModuleAnchors {
    pub(crate) fn new(modules: Vec<Arc<CudaModule>>) -> Self {
        Self {
            _modules: modules.into(),
        }
    }
}

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

/// Standalone deterministic TF32 NN-forward tiles owned by the inference
/// module. No training layout is compiled into this holder.
pub struct FixedTf32Kernels {
    pub m128n64_s2: CudaFunction,
    pub m128n64_s3: CudaFunction,
    pub m64n64_s2: CudaFunction,
    pub m64n64_s3: CudaFunction,
    pub m16n32_s4: CudaFunction,
}

/// SM120 TMA TF32 NN-forward tiles owned by the inference module.
pub struct FixedSm120Tf32Kernels {
    pub m128n64_s2: CudaFunction,
    pub m128n64_s3: CudaFunction,
    pub m64n128_s2: CudaFunction,
    pub m64n128_s3: CudaFunction,
    pub m64n64_s2_producer_warp: CudaFunction,
    pub m64n64_s2: CudaFunction,
    pub(crate) m64n64_s2_pair_store: CudaFunction,
}

/// Qualified SM120 TMA half-precision NN-forward tiles.
pub struct FixedSm120HalfKernels {
    pub m64n64_bk64_s2: HalfKernel,
    pub m64n128_bk64_s2: HalfKernel,
    pub m128n64_bk32_s3: HalfKernel,
    pub m128n128_bk32_s2: HalfKernel,
    pub m128n128_bk32_s3: HalfKernel,
}

/// Inference-owned exact-F32 TMA tiles with post-dot bias or no-bias epilogues,
/// matching the deterministic inference arithmetic contract.
pub struct FixedSm120FmaPostbiasKernels {
    pub m128n64: CudaFunction,
    pub m64n128: CudaFunction,
    pub m128n96: CudaFunction,
    /// Independently admitted force-only K4 twin; controls remain available on rejection.
    pub m128n64_k4: Option<CudaFunction>,
    pub m128n64_k4_rejection: Option<String>,
    /// Independently admitted eight-warp A1 AUTO route; absent means control fallback.
    pub m128n64_t256: Option<CudaFunction>,
    pub m128n64_t256_rejection: Option<String>,
    /// Independently admitted force-only no-bias eight-warp twin.
    pub nobias_m128n64_t256: Option<CudaFunction>,
    pub nobias_m128n64_t256_rejection: Option<String>,
}

/// All compiled CUDA kernels needed for Mamba forward/backward.
///
/// Kernels are compiled once via NVRTC at startup. Grouped by pipeline stage.
pub struct MambaKernels {
    _modules: CudaModuleAnchors,

    compiler_identity: super::kernel_identity::CompilerIdentity,
    triad: GemmBiKernels,

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
    /// T-major twin for the PARALLEL route's `[b][n][d][t]` locals (tape
    /// layout). Same ascending-d sum, same output values and layout.
    pub ssm_reduce_d_bc_tmajor_typed: TypedKernel,
    /// Reduce local SSM grads to dD `[d_inner]`.
    pub ssm_reduce_d_d: CudaFunction,
    /// Reduce local SSM grads to d_a_log `[d_inner*d_state]`.
    pub ssm_reduce_d_a_log: CudaFunction,
    /// Chunk-partial reducer for the fold backward's d_a_log slots.
    pub ssm_reduce_d_a_log_chunks: CudaFunction,

    // -- Conv1d --
    /// Nosave tiled twin (inference prefill): tile-0 seeds from the
    /// carry-in state, later tiles from the x_branch halo - bit-identical
    /// to the serial nosave walk at two orders more parallelism.
    pub conv1d_burnin_fwd_nosave_tiled: CudaFunction,
    pub conv1d_burnin_nosave_tiled_typed: TypedKernel,

    // -- Activations --
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
    /// Negate and exponentiate: `out[i] = -exp(a_log[i])`.
    pub exp_negate: CudaFunction,
    pub exp_negate2: CudaFunction,
    /// Gather columns from a wide matrix into a contiguous buffer.
    pub gather_cols: CudaFunction,
    /// Gather B and C columns from xdbl output.
    pub gather_bc_cols: CudaFunction,
    /// T-major twin (`[b][n][t]` outputs) for the parallel-scan route —
    /// the scan reads B/C per (d, n) lane over consecutive t.
    pub gather_bc_cols_tmajor: CudaFunction,
    /// Staged-write twin of the t-major gather: smem transpose tile, writes
    /// t-contiguous. Falls back to the untiled kernel when the tile exceeds
    /// the 48 KB static smem budget (f32 at d_state > 186).
    pub gather_bc_cols_tmajor_tiled: CudaFunction,
    /// Gating forward that recomputes SiLU(gate) from the in_proj output's
    /// gate half, read through a row stride.
    pub gate_mul_silu: CudaFunction,
    /// Backward through gating: `y = ssm_out * gate_silu`.
    pub gating_backward: CudaFunction,
    /// Residual add: `out[i] += residual[i]`.
    pub residual_add: CudaFunction,
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
    /// Tiled conv backward: d_x (anticausal FIR, carry-seeded at the tile
    /// boundaries with the serial association order), the tap partials and
    /// the bias partials in one walk, the pre-activation recomputed from
    /// the x window.
    pub conv1d_bwd_tiled_typed: TypedKernel,
    /// Typed dispatch for `conv1d_burnin_backward`, the serial reference
    /// the typed backward parity test drives; production runs the tiled
    /// kernel above. Matches DEFINE_CONV1D_BURNIN_BWD in conv1d.cu.
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
    pub softplus_fwd_typed: TypedKernel,
    /// RMSNorm that first adds a typed branch output into the f32 residual
    /// in place: the previous layer's residual add and this layer's norm in
    /// one launch on the decode routes. The f32 entry takes an f32 branch.
    pub rmsnorm_fwd_resadd_typed: TypedKernel,
    pub bias_broadcast_typed: TypedKernel,
    pub elementwise_mul_typed: TypedKernel,
    pub residual_add_typed: TypedKernel,
    pub gather_cols_typed: TypedKernel,
    pub gather_bc_cols_typed: TypedKernel,
    /// T-major twin of the typed gather (parallel-scan route).
    pub gather_bc_cols_tmajor_typed: TypedKernel,
    /// Staged-write twin of the typed t-major gather.
    pub gather_bc_cols_tmajor_tiled_typed: TypedKernel,
    /// Typed gating forward recomputing SiLU(gate).
    pub gate_mul_silu_typed: TypedKernel,
    /// 16-byte vectorized twins of the hot elementwise kernels: one uint4
    /// per operand per thread, same per-element arithmetic in the same
    /// order. Selected by `vec8_ok` when the shape and every operand
    /// pointer allow it.
    pub gate_mul_silu_v_typed: TypedKernel,
    pub elementwise_mul_v_typed: TypedKernel,
    pub softplus_copy_typed: TypedKernel,
    /// The decode SSM step with softplus, the B/C gather and the gating
    /// folded in: the decode routes' only step kernel.
    pub ssm_step_fwd_fused_typed: TypedKernel,
    /// Conv1d step with the SiLU fused into its store: the decode routes'
    /// only conv step kernel, one launch per layer.
    pub conv1d_step_fwd_silu_typed: TypedKernel,
    pub ssm_burnin_nosave_typed: TypedKernel,
    pub softplus_bwd_typed: TypedKernel,
    pub gather_last_timestep_typed: TypedKernel,
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
    /// Former exact-f32 CUDA-core route retained as the qualification oracle.
    pub gemm_bi_f32_f32: CudaFunction,
    /// Portable exact-f32 64x64 two-stage route and AUTO fallback.
    pub gemm_bi_f32_f32_s2: CudaFunction,
    /// Exact-f32 64x128 route for the measured A/B AUTO points and forced tests.
    pub gemm_bi_f32_f32_n128_s2: CudaFunction,
    /// Deterministic TF32 NN-forward tile ladder owned by Fixed/inference.
    pub gemm_bi_nn_tf32: FixedTf32Kernels,
    /// SM120 TMA counterpart of the deterministic inference TF32 ladder.
    pub gemm_bi_nn_tf32_sm120: Option<FixedSm120Tf32Kernels>,
    /// SM120 TMA BF16/F16 inference candidates.
    pub gemm_bi_nn_half_sm120: Option<FixedSm120HalfKernels>,
    /// SM120 TMA BF16/F16-input, F32-output inference candidates.
    pub gemm_bi_nn_half_sm120_f32out: Option<FixedSm120HalfKernels>,
    /// Independently admitted Ada Tc128 half pipeline; used by qualified AUTO rows.
    pub fixed_sm89_half_pipeline: Option<HalfKernel>,
    /// Why the optional Ada pipeline is absent (including non-Ada targets).
    pub fixed_sm89_half_pipeline_rejection: Option<String>,
    /// Independent Ada-only packed/XOR homogeneous-half candidate.
    pub fixed_sm89_half_swizzle: Option<HalfKernel>,
    /// Why only the swizzle holder is absent; never disables the incumbent.
    pub fixed_sm89_half_swizzle_rejection: Option<String>,
    /// Independently admitted Ada three-stage homogeneous-half pair.
    pub fixed_sm89_half_s3: Option<HalfKernel>,
    pub fixed_sm89_half_s3_rejection: Option<String>,
    /// Fixed-owned Ada RNA-wide TF32 route; retains qualified AUTO rows and fallbacks.
    pub fixed_sm89_tf32_rna_wide: Option<CudaFunction>,
    pub fixed_sm89_tf32_rna_wide_rejection: Option<String>,
    /// Fixed-owned Ada RNA M128xN96 TF32; qualified E0 AUTO since revision 45.
    pub fixed_sm89_tf32_rna_n96: Option<CudaFunction>,
    pub fixed_sm89_tf32_rna_n96_rejection: Option<String>,
    /// Independently admitted Ada F16 D route; qualified CUDA13.2 AUTO since revision 45.
    pub fixed_sm89_half_m64n64_s3_f16: Option<CudaFunction>,
    pub fixed_sm89_half_m64n64_s3_f16_rejection: Option<String>,
    /// Independently admitted Ada F16 E route; qualified CUDA13.2 AUTO since revision 45.
    pub fixed_sm89_half_m128n64_s2_f16: Option<CudaFunction>,
    pub fixed_sm89_half_m128n64_s2_f16_rejection: Option<String>,
    /// Optional Ada exact-F32 N64 copy-plan; admitted independently of incumbents.
    pub fixed_sm89_f32_n64_copyplan: Option<CudaFunction>,
    pub fixed_sm89_f32_n64_copyplan_rejection: Option<String>,
    /// Optional CC12.0 exact-F32 N64 copy-plan, separate from the Ada route.
    pub fixed_sm120_f32_n64_copyplan: Option<CudaFunction>,
    pub fixed_sm120_f32_n64_copyplan_rejection: Option<String>,
    /// Independently admitted force-only 256-thread CopyPlan twin.
    pub fixed_sm120_f32_n64_copyplan_t256: Option<CudaFunction>,
    pub fixed_sm120_f32_n64_copyplan_t256_rejection: Option<String>,
    pub fixed_sm120_f32_m128n64_copyplan_t256: Option<CudaFunction>,
    pub fixed_sm120_f32_m128n64_copyplan_t256_rejection: Option<String>,
    /// Independent compute_120 sliced exact-F32 route; no scratch.
    pub fixed_sm120_f32_n64_sliced: Option<CudaFunction>,
    pub fixed_sm120_f32_n64_sliced_rejection: Option<String>,
    /// Optional CC12.0 exact-FMA tiles with Fixed's post-dot bias order.
    pub fixed_sm120_fma_postbias: Option<FixedSm120FmaPostbiasKernels>,
    pub fixed_sm120_fma_postbias_rejection: Option<String>,

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

    /// The Inference family's inference ladder (`kernels/gemm_bi_inference/`,
    /// GBF namespace): bit-identical copies of the forward TC tiles,
    /// owned by the inference kernel.
    pub gemm_bi_nn_tc128_typed: HalfKernel,
    /// Portable ladder variants retaining the tensor-core accumulator in f32 at store.
    pub gemm_bi_nn_tc128_f32out: HalfKernel,
    pub gemm_bi_nn_tc64_f32out: HalfKernel,
    pub gemm_bi_nn_tc16_f32out: HalfKernel,
    pub gemm_bi_nn_tc64_typed: HalfKernel,
    pub gemm_bi_nn_tc16_typed: HalfKernel,
    /// Fragment-reuse 128x128 rung with 64x64 warp tiles.
    pub gemm_bi_nn_tcw64_typed: HalfKernel,
    /// Wide 128x256 sibling of the fragment-reuse rung.
    pub gemm_bi_nn_tcwn64_typed: HalfKernel,
    /// The Hopper wgmma rung (compiled only for sm_90a; the dispatcher
    /// never routes here until the rung is hardware-qualified - the
    /// forced census entry is its only caller).
    pub gemm_bi_nn_sm90_typed: Option<HalfKernel>,
    /// Datacenter-Blackwell tcgen05 rung, available only on admitted
    /// architecture-family targets.
    pub gemm_bi_nn_sm100_typed: Option<HalfKernel>,
}

impl Deref for MambaKernels {
    type Target = GemmBiKernels;

    fn deref(&self) -> &Self::Target {
        &self.triad
    }
}

/// NVRTC library version, part of the kernel-cache key. `(0, 0)` means the
/// query failed; persistent caching still requires exact runtime and builtins
/// library hashes.
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

/// Linux kernel-cache directory. `MAMBA_RS_KERNEL_CACHE` overrides with an
/// absolute path, or `0`/`off` disables it. Relative paths, symlinked
/// components, writable ancestors, and non-private final directories disable
/// persistent caching. Other platforms currently leave it disabled.
pub(crate) fn kernel_cache_dir() -> Option<std::path::PathBuf> {
    let path = match std::env::var("MAMBA_RS_KERNEL_CACHE") {
        Ok(v) if matches!(v.trim(), "0" | "off" | "OFF") => None,
        Ok(v) if !v.trim().is_empty() => Some(std::path::PathBuf::from(v.trim())),
        _ => {
            let base = std::env::var("XDG_CACHE_HOME")
                .map(std::path::PathBuf::from)
                .or_else(|_| {
                    std::env::var("HOME").map(|home| std::path::PathBuf::from(home).join(".cache"))
                })
                .ok()?;
            Some(base.join("mamba-rs").join("kernels"))
        }
    }?;
    super::kernel_identity::prepare_private_cache_dir(&path)
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
    // every scan kernel and bounds their loops, which unroll to constant
    // indices so the arrays stay in registers; a 64-floor at d_state=16
    // made them 4x oversized. Callers with larger states still get the
    // exact padded fit.
    Ok(d_state.div_ceil(16) * 16)
}

impl MambaKernels {
    /// Exact membership in loaded inference terminal holders, including the borrowed wide holder.
    pub(in crate::mamba_ssm::gpu) fn inference_terminal_function(
        &self,
        symbol: &str,
    ) -> Option<&CudaFunction> {
        match symbol {
            "gemm_bi_f32_f32_s2" => Some(&self.gemm_bi_f32_f32_s2),
            "gemm_bi_f32_f32_n128_s2" => Some(&self.gemm_bi_f32_f32_n128_s2),
            "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1" => {
                self.fixed_sm89_f32_n64_copyplan.as_ref()
            }
            "gemm_bi_nn_fixed_sm120_f32_n64_copyplan_v1" => {
                self.fixed_sm120_f32_n64_copyplan.as_ref()
            }
            "gemm_bi_nn_fixed_sm120_f32_n64_copyplan_t256_v1" => {
                self.fixed_sm120_f32_n64_copyplan_t256.as_ref()
            }
            "gemm_bi_nn_fixed_sm120_f32_n64_copyplan_m128n64_t256_v1" => {
                self.fixed_sm120_f32_m128n64_copyplan_t256.as_ref()
            }
            "gemm_bi_nn_fixed_sm120_f32_n64_sliced_v1" => self.fixed_sm120_f32_n64_sliced.as_ref(),
            "gemm_bi_bf16_bf16" => Some(&self.gemm_bi_bf16_bf16),
            "matvec_bi_bf16_bf16" => Some(&self.matvec_bi_bf16_bf16),
            "gemm_bi_bf16_f32" => Some(&self.gemm_bi_bf16_f32),
            "matvec_bi_bf16_f32" => Some(&self.matvec_bi_bf16_f32),
            "gemm_bi_nn_tc128_bf16" => Some(&self.gemm_bi_nn_tc128_typed.bf16),
            "gemm_bi_nn_tc128_f32out_bf16" => Some(&self.gemm_bi_nn_tc128_f32out.bf16),
            "gemm_bi_nn_tcw64_bf16" => Some(&self.gemm_bi_nn_tcw64_typed.bf16),
            "gemm_bi_nn_tcwn64_bf16" => Some(&self.gemm_bi_nn_tcwn64_typed.bf16),
            "gemm_bi_nn_tc64_bf16" => Some(&self.gemm_bi_nn_tc64_typed.bf16),
            "gemm_bi_nn_tc64_f32out_bf16" => Some(&self.gemm_bi_nn_tc64_f32out.bf16),
            "gemm_bi_nn_tc16_bf16" => Some(&self.gemm_bi_nn_tc16_typed.bf16),
            "gemm_bi_nn_tc16_f32out_bf16" => Some(&self.gemm_bi_nn_tc16_f32out.bf16),
            "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_bf16" => {
                Some(&self.fixed_sm89_half_pipeline.as_ref()?.bf16)
            }
            "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16" => {
                Some(&self.fixed_sm89_half_swizzle.as_ref()?.bf16)
            }
            "gemm_bi_nn_fixed_sm89_tc128_s3_v1_bf16" => {
                Some(&self.fixed_sm89_half_s3.as_ref()?.bf16)
            }
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_bf16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m64n64_bk64_s2.bf16)
            }
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_f32out_bf16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m64n64_bk64_s2
                    .bf16,
            ),
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_bf16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m64n128_bk64_s2.bf16)
            }
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_f32out_bf16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m64n128_bk64_s2
                    .bf16,
            ),
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_bf16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n64_bk32_s3.bf16)
            }
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_f32out_bf16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n64_bk32_s3
                    .bf16,
            ),
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_bf16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n128_bk32_s2.bf16)
            }
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_f32out_bf16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n128_bk32_s2
                    .bf16,
            ),
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_bf16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n128_bk32_s3.bf16)
            }
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_f32out_bf16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n128_bk32_s3
                    .bf16,
            ),
            "gemm_bi_nn_sm90a_wgmma_wg1_bf16" => Some(&self.gemm_bi_nn_sm90_typed.as_ref()?.bf16),
            "gemm_bi_nn_sm100_tcgen_c4_bf16" => Some(&self.gemm_bi_nn_sm100_typed.as_ref()?.bf16),
            "gemm_bi_f16_f16" => Some(&self.gemm_bi_f16_f16),
            "matvec_bi_f16_f16" => Some(&self.matvec_bi_f16_f16),
            "gemm_bi_f16_f32" => Some(&self.gemm_bi_f16_f32),
            "matvec_bi_f16_f32" => Some(&self.matvec_bi_f16_f32),
            "gemm_bi_nn_tc128_f16" => Some(&self.gemm_bi_nn_tc128_typed.f16),
            "gemm_bi_nn_tc128_f32out_f16" => Some(&self.gemm_bi_nn_tc128_f32out.f16),
            "gemm_bi_nn_tcw64_f16" => Some(&self.gemm_bi_nn_tcw64_typed.f16),
            "gemm_bi_nn_tcwn64_f16" => Some(&self.gemm_bi_nn_tcwn64_typed.f16),
            "gemm_bi_nn_tc64_f16" => Some(&self.gemm_bi_nn_tc64_typed.f16),
            "gemm_bi_nn_tc64_f32out_f16" => Some(&self.gemm_bi_nn_tc64_f32out.f16),
            "gemm_bi_nn_tc16_f16" => Some(&self.gemm_bi_nn_tc16_typed.f16),
            "gemm_bi_nn_tc16_f32out_f16" => Some(&self.gemm_bi_nn_tc16_f32out.f16),
            "gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_f16" => {
                Some(&self.fixed_sm89_half_pipeline.as_ref()?.f16)
            }
            "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_f16" => {
                Some(&self.fixed_sm89_half_swizzle.as_ref()?.f16)
            }
            "gemm_bi_nn_fixed_sm89_tc128_s3_v1_f16" => Some(&self.fixed_sm89_half_s3.as_ref()?.f16),
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_f16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m64n64_bk64_s2.f16)
            }
            "gemm_bi_nn_sm120_tma_64x64_bk64_s2_f32out_f16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m64n64_bk64_s2
                    .f16,
            ),
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_f16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m64n128_bk64_s2.f16)
            }
            "gemm_bi_nn_sm120_tma_64x128_bk64_s2_f32out_f16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m64n128_bk64_s2
                    .f16,
            ),
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_f16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n64_bk32_s3.f16)
            }
            "gemm_bi_nn_sm120_tma_128x64_bk32_s3_f32out_f16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n64_bk32_s3
                    .f16,
            ),
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_f16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n128_bk32_s2.f16)
            }
            "gemm_bi_nn_sm120_tma_128x128_bk32_s2_f32out_f16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n128_bk32_s2
                    .f16,
            ),
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_f16" => {
                Some(&self.gemm_bi_nn_half_sm120.as_ref()?.m128n128_bk32_s3.f16)
            }
            "gemm_bi_nn_sm120_tma_128x128_bk32_s3_f32out_f16" => Some(
                &self
                    .gemm_bi_nn_half_sm120_f32out
                    .as_ref()?
                    .m128n128_bk32_s3
                    .f16,
            ),
            "gemm_bi_nn_sm90a_wgmma_wg1_f16" => Some(&self.gemm_bi_nn_sm90_typed.as_ref()?.f16),
            "gemm_bi_nn_sm100_tcgen_c4_f16" => Some(&self.gemm_bi_nn_sm100_typed.as_ref()?.f16),
            "matvec_bi_f32_f32" => Some(&self.matvec_bi_f32_f32),
            "gemm_bi_nn_fixed_sm89_m64n64_bk64_s3_v1_f16" => {
                self.fixed_sm89_half_m64n64_s3_f16.as_ref()
            }
            "gemm_bi_nn_fixed_sm89_m128n64_bk64_s2_v1_f16" => {
                self.fixed_sm89_half_m128n64_s2_f16.as_ref()
            }
            "gemm_bi_nn_tf32_v1_m128n64_bk32_s2" => Some(&self.gemm_bi_nn_tf32.m128n64_s2),
            "gemm_bi_nn_tf32_v1_m128n64_bk32_s3" => Some(&self.gemm_bi_nn_tf32.m128n64_s3),
            "gemm_bi_nn_tf32_v1_m64n64_bk32_s2" => Some(&self.gemm_bi_nn_tf32.m64n64_s2),
            "gemm_bi_nn_tf32_v1_m64n64_bk32_s3" => Some(&self.gemm_bi_nn_tf32.m64n64_s3),
            "gemm_bi_nn_tf32_v1_m16n32_bk32_s4" => Some(&self.gemm_bi_nn_tf32.m16n32_s4),
            "gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3" => {
                self.fixed_sm89_tf32_rna_wide.as_ref()
            }
            "gemm_bi_nn_fixed_sm89_rna_tf32_v1_m128n96_bk32_s3" => {
                self.fixed_sm89_tf32_rna_n96.as_ref()
            }
            "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3" => {
                self.triad_kernels().tf32_function(symbol)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m128n64_s2)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s3" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m128n64_s3)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s2" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m64n128_s2)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s3" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m64n128_s3)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m64n64_s2_producer_warp)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m64n64_s2)
            }
            "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store" => {
                Some(&self.gemm_bi_nn_tf32_sm120.as_ref()?.m64n64_s2_pair_store)
            }
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2" => {
                Some(&self.fixed_sm120_fma_postbias.as_ref()?.m128n64)
            }
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m64n128_bk16_s2" => {
                Some(&self.fixed_sm120_fma_postbias.as_ref()?.m64n128)
            }
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n96_bk16_s2" => {
                Some(&self.fixed_sm120_fma_postbias.as_ref()?.m128n96)
            }
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_bk16_s2_k4" => {
                self.fixed_sm120_fma_postbias.as_ref()?.m128n64_k4.as_ref()
            }
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_postbias_m128n64_t256_bk16_s2" => self
                .fixed_sm120_fma_postbias
                .as_ref()?
                .m128n64_t256
                .as_ref(),
            "gemm_bi_nn_sm120_tma_fma_v1_fixed_nobias_m128n64_t256_bk16_s2" => self
                .fixed_sm120_fma_postbias
                .as_ref()?
                .nobias_m128n64_t256
                .as_ref(),
            _ => None,
        }
    }

    /// Compile with the default state capacity of 64 — the common
    /// shapes' tightest register budget. Models with a larger `d_state`
    /// use [`Self::compile_with_state_cap`].
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        Self::compile_with_state_cap(ctx, arch, 64)
    }

    /// Compile all CUDA kernels from source. Persistent PTX caching is used
    /// only when the complete header and NVRTC toolchain domains are known.
    /// An invalid entry is ignored and compilation proceeds normally.
    ///
    /// `state_cap` sizes the per-thread state register arrays (see
    /// [`state_capacity`]); it rides the compile options and therefore
    /// the cache key.
    pub fn compile_with_state_cap(
        ctx: &Arc<CudaContext>,
        arch: &'static str,
        state_cap: usize,
    ) -> Result<Self, String> {
        let device_cc = match ctx.compute_capability() {
            Ok(device_cc) => Some(device_cc),
            Err(error) => {
                static UNKNOWN_CC: std::sync::Once = std::sync::Once::new();
                super::diagnostics::warn_once(&UNKNOWN_CC, || {
                    format!(
                        "the driver did not report a compute capability ({error:?}); only the \
                         portable kernels are compiled"
                    )
                });
                None
            }
        };
        // The loader and the selectors share one family predicate, so a
        // kernel is never compiled for a board that cannot select it.
        let sm120_board = device_cc.is_some_and(|(major, minor)| {
            u32::try_from(major)
                .and_then(|major| Ok((major, u32::try_from(minor)?)))
                .is_ok_and(super::device::is_sm120_family)
        });
        if !sm120_board && matches!(arch, "compute_120" | "compute_121") {
            static UNQUALIFIED: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&UNQUALIFIED, || {
                format!(
                    "compute capability {device_cc:?} is not a qualified family; only the \
                     portable kernels serve it"
                )
            });
        }
        let sm120_artifacts = match device_cc {
            Some(device_cc) if sm120_board => {
                super::gemm_bi_triad::modules::compile_sm120_artifact_set(
                    ctx,
                    state_cap,
                    device_cc,
                    nvrtc_version(),
                )
            }
            _ => None,
        };
        let compile = |module_kind| {
            super::gemm_bi_triad::modules::compile_module(
                super::gemm_bi_triad::modules::CompileModuleRequest {
                    ctx,
                    arch,
                    state_cap,
                    module_kind,
                },
            )
        };
        let (
            fixed,
            scalar,
            sm80,
            finalist,
            finalist_rejection,
            sm89_half,
            sm89_half_rejection,
            sm89_exact_f32,
            sm89_exact_f32_rejection,
            sm89_exact_f32_d128,
            sm89_exact_f32_d128_rejection,
            sm89_tf32_joint,
            sm89_tf32_joint_rejection,
            specialized,
        ) = if let Some(artifacts) = sm120_artifacts {
            (
                artifacts.fixed,
                artifacts.scalar,
                artifacts.sm80,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                artifacts.specialized,
            )
        } else {
            let fixed = compile(super::kernel_identity::ModuleKind::Fixed)?;
            let scalar = compile(super::kernel_identity::ModuleKind::TriadScalar)?;
            let sm80 = compile(super::kernel_identity::ModuleKind::TriadSm80)?;
            let (finalist, finalist_rejection) =
                if matches!((arch, device_cc), ("sm_89", Some((8, 9)))) {
                    match compile(super::kernel_identity::ModuleKind::TriadSm89Finalist) {
                        Ok(module) => (Some(module), None),
                        Err(error) => (None, Some(error)),
                    }
                } else {
                    (None, None)
                };
            let (sm89_half, sm89_half_rejection) =
                if matches!((arch, device_cc), ("sm_89", Some((8, 9)))) {
                    match compile(super::kernel_identity::ModuleKind::TriadSm89Half) {
                        Ok(module) => (Some(module), None),
                        Err(error) => (None, Some(error)),
                    }
                } else {
                    (None, None)
                };
            let (sm89_exact_f32, sm89_exact_f32_rejection) =
                if matches!((arch, device_cc), ("sm_89", Some((8, 9)))) {
                    match compile(super::kernel_identity::ModuleKind::TriadSm89ExactF32) {
                        Ok(module) => (Some(module), None),
                        Err(error) => (None, Some(error)),
                    }
                } else {
                    (None, None)
                };
            let (sm89_exact_f32_d128, sm89_exact_f32_d128_rejection) =
                if matches!((arch, device_cc), ("sm_89", Some((8, 9)))) {
                    match compile(super::kernel_identity::ModuleKind::TriadSm89ExactF32D128) {
                        Ok(module) => (Some(module), None),
                        Err(error) => (None, Some(error)),
                    }
                } else {
                    (None, None)
                };
            let (sm89_tf32_joint, sm89_tf32_joint_rejection) =
                if matches!((arch, device_cc), ("sm_89", Some((8, 9)))) {
                    match compile(super::kernel_identity::ModuleKind::TriadSm89Tf32Joint) {
                        Ok(module) => (Some(module), None),
                        Err(error) => (None, Some(error)),
                    }
                } else {
                    (None, None)
                };
            let specialized = match (arch, device_cc) {
                ("sm_90a", Some((9, 0))) => compile(super::kernel_identity::ModuleKind::TriadSm90a)
                    .ok()
                    .and_then(|module| {
                        super::gemm_bi_triad::modules::qualify_specialized_module(module).ok()
                    }),
                ("sm_100a", Some(device_cc @ (10, 0))) | ("sm_103a", Some(device_cc @ (10, 3))) => {
                    super::gemm_bi_triad::modules::compile_sm100_optional(ctx, state_cap, device_cc)
                }
                ("sm_110a", Some(device_cc @ (11, 0))) => {
                    super::gemm_bi_triad::modules::compile_sm100_optional(ctx, state_cap, device_cc)
                }
                _ => None,
            };
            (
                fixed,
                scalar,
                sm80,
                finalist,
                finalist_rejection,
                sm89_half,
                sm89_half_rejection,
                sm89_exact_f32,
                sm89_exact_f32_rejection,
                sm89_exact_f32_d128,
                sm89_exact_f32_d128_rejection,
                sm89_tf32_joint,
                sm89_tf32_joint_rejection,
                specialized,
            )
        };
        let compiler_identity = fixed.compiler_identity;
        let (fixed_sm89_half_pipeline, fixed_sm89_half_pipeline_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_half_pipeline(ctx, &fixed);
        let (fixed_sm89_half_swizzle, fixed_sm89_half_swizzle_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_half_swizzle(ctx, &fixed);
        let (fixed_sm89_half_s3, fixed_sm89_half_s3_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_half_s3(ctx, &fixed);
        let (fixed_sm89_tf32_rna_wide, fixed_sm89_tf32_rna_wide_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_rna_wide(ctx, &fixed);
        let (fixed_sm89_tf32_rna_n96, fixed_sm89_tf32_rna_n96_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_rna_n96(ctx, &fixed);
        let (fixed_sm89_half_m64n64_s3_f16, fixed_sm89_half_m64n64_s3_f16_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_half_m64n64_s3(ctx, &fixed);
        let (fixed_sm89_half_m128n64_s2_f16, fixed_sm89_half_m128n64_s2_f16_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_half_m128n64_s2(ctx, &fixed);
        let (fixed_sm89_f32_n64_copyplan, fixed_sm89_f32_n64_copyplan_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm89_f32_n64_copyplan(ctx, &fixed);
        let (fixed_sm120_f32_n64_copyplan, fixed_sm120_f32_n64_copyplan_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm120_f32_n64_copyplan(ctx, &fixed);
        let (fixed_sm120_f32_n64_copyplan_t256, fixed_sm120_f32_n64_copyplan_t256_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm120_f32_n64_copyplan_t256(ctx, &fixed);
        let (
            fixed_sm120_f32_m128n64_copyplan_t256,
            fixed_sm120_f32_m128n64_copyplan_t256_rejection,
        ) = super::gemm_bi_triad::modules::load_fixed_sm120_f32_m128n64_copyplan_t256(ctx, &fixed);
        let (fixed_sm120_f32_n64_sliced, fixed_sm120_f32_n64_sliced_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm120_f32_n64_sliced(ctx, &fixed);
        let (fixed_sm120_fma_postbias, fixed_sm120_fma_postbias_rejection) =
            super::gemm_bi_triad::modules::load_fixed_sm120_fma_postbias(ctx, &fixed);
        let triad = GemmBiKernels::load(
            ctx,
            super::gemm_bi_triad::modules::GemmBiModuleSet {
                fixed_artifact: fixed.artifact_identity,
                scalar,
                sm80,
                finalist,
                finalist_compile_rejection: finalist_rejection,
                sm89_half,
                sm89_half_compile_rejection: sm89_half_rejection,
                sm89_exact_f32,
                sm89_exact_f32_compile_rejection: sm89_exact_f32_rejection,
                sm89_exact_f32_d128,
                sm89_exact_f32_d128_compile_rejection: sm89_exact_f32_d128_rejection,
                sm89_tf32_joint,
                sm89_tf32_joint_compile_rejection: sm89_tf32_joint_rejection,
                specialized,
            },
        )?;
        // A rejected TF32 module used to be recorded and never shown: the
        // process booted green and every TF32 request quietly ran scalar.
        if let Some(reason) = triad.portable_tf32_rejection() {
            static PORTABLE: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&PORTABLE, || {
                format!(
                    "the portable TF32 routes are not bound on this board ({reason}); the \
                     exact f32 kernels serve every TF32 request"
                )
            });
        }
        let excluded = triad.tf32_excluded_symbols();
        if !excluded.is_empty() {
            static EXCLUDED: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&EXCLUDED, || {
                format!(
                    "{} TF32 route(s) are excluded on this toolkit and decline to the exact \
                     family: {}",
                    excluded.len(),
                    excluded
                        .iter()
                        .map(|exclusion| exclusion.reason.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            });
        }
        if let Some(reason) = triad.specialized_tf32_rejection() {
            static SPECIALIZED: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&SPECIALIZED, || {
                format!(
                    "the specialized TF32 module is not bound on this board ({reason}); the \
                     portable or exact kernels serve every TF32 request"
                )
            });
        }
        if let Some(reason) = triad.finalist_tf32_rejection() {
            static FINALIST: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&FINALIST, || {
                format!(
                    "the optional Ada TF32 finalist is not bound ({reason}); the portable or exact kernels remain available"
                )
            });
        }
        if let Some(reason) = triad.sm89_half_rejection() {
            static HALF_MODULE: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&HALF_MODULE, || {
                format!(
                    "the optional Ada half-Triad module is not bound ({reason}); the existing typed kernels remain available"
                )
            });
        }
        let half_excluded = triad.sm89_half_exclusions();
        if !half_excluded.is_empty() {
            static HALF_EXCLUDED: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&HALF_EXCLUDED, || {
                format!(
                    "{} Ada half-Triad symbol(s) failed resource admission while their siblings remain bound: {}",
                    half_excluded.len(),
                    half_excluded
                        .iter()
                        .map(|exclusion| format!("{}: {}", exclusion.symbol, exclusion.reason))
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            });
        }
        if let Some(reason) = triad.sm89_exact_f32_rejection() {
            static EXACT_F32_MODULE: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&EXACT_F32_MODULE, || {
                format!(
                    "the optional Ada exact-F32 Triad module is not bound ({reason}); the existing exact kernels remain available"
                )
            });
        }
        for exclusion in triad.sm89_exact_f32_exclusions() {
            static D768_IN: std::sync::Once = std::sync::Once::new();
            static D768_OUT: std::sync::Once = std::sync::Once::new();
            static PRISM: std::sync::Once = std::sync::Once::new();
            let once = match exclusion.symbol {
                super::gemm_bi_triad::D768_IN_FUSED_SYMBOL => &D768_IN,
                super::gemm_bi_triad::D768_OUT_RAW_SYMBOL => &D768_OUT,
                super::gemm_bi_triad::PRISM_RAW_SYMBOL => &PRISM,
                _ => continue,
            };
            super::diagnostics::warn_once(once, || {
                format!(
                    "Ada exact-F32 Triad symbol {} is excluded while its siblings remain available: {}",
                    exclusion.symbol, exclusion.reason
                )
            });
        }
        if let Some(reason) = triad.sm89_exact_f32_d128_rejection() {
            static EXACT_F32_D128_MODULE: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&EXACT_F32_D128_MODULE, || {
                format!(
                    "the optional Ada exact-F32 d128 Triad module is not bound ({reason}); the SplitM64 kernels remain available"
                )
            });
        }
        for exclusion in triad.sm89_exact_f32_d128_exclusions() {
            static D128_IN: std::sync::Once = std::sync::Once::new();
            static D128_OUT: std::sync::Once = std::sync::Once::new();
            let once = match exclusion.symbol {
                super::gemm_bi_triad::D128_IN_SYMBOL => &D128_IN,
                super::gemm_bi_triad::D128_OUT_SYMBOL => &D128_OUT,
                _ => continue,
            };
            super::diagnostics::warn_once(once, || {
                format!(
                    "Ada exact-F32 d128 Triad symbol {} is excluded while its sibling remains available: {}",
                    exclusion.symbol, exclusion.reason
                )
            });
        }
        if let Some(reason) = triad.sm89_tf32_joint_rejection() {
            static TF32_JOINT_MODULE: std::sync::Once = std::sync::Once::new();
            super::diagnostics::warn_once(&TF32_JOINT_MODULE, || {
                format!(
                    "the optional Ada TF32 joint module is not bound ({reason}); the existing deterministic TF32 kernels remain available"
                )
            });
        }
        for exclusion in triad.sm89_tf32_joint_exclusions() {
            static TRANSPOSE: std::sync::Once = std::sync::Once::new();
            static TN_N96: std::sync::Once = std::sync::Once::new();
            static TN_M64N64: std::sync::Once = std::sync::Once::new();
            static TN_M64N96_S2: std::sync::Once = std::sync::Once::new();
            static NN_N96: std::sync::Once = std::sync::Once::new();
            static NN_N96_BASELINE: std::sync::Once = std::sync::Once::new();
            static NT_A_LDMATRIX_N96: std::sync::Once = std::sync::Once::new();
            let once = match exclusion.symbol {
                super::gemm_bi_triad::TN_PRE_RNA_TRANSPOSE_SYMBOL => &TRANSPOSE,
                super::gemm_bi_triad::TN_PRE_RNA_N96_SYMBOL => &TN_N96,
                super::gemm_bi_triad::TN_PRE_RNA_M64N64_SYMBOL => &TN_M64N64,
                super::gemm_bi_triad::TN_PRE_RNA_M64N96_S2_SYMBOL => &TN_M64N96_S2,
                super::gemm_bi_triad::NN_ADD_HALF_DIRECT_N96_SYMBOL => &NN_N96,
                super::gemm_bi_triad::NN_ADD_HALF_N96_SYMBOL => &NN_N96_BASELINE,
                super::gemm_bi_triad::NT_A_LDMATRIX_N96_SYMBOL => &NT_A_LDMATRIX_N96,
                _ => continue,
            };
            super::diagnostics::warn_once(once, || {
                format!(
                    "Ada TF32 joint symbol {} is excluded while its siblings remain available: {}",
                    exclusion.symbol, exclusion.reason
                )
            });
        }
        let module = fixed.module;

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
        let load_sm120_half = |suffix: &str| -> Result<FixedSm120HalfKernels, String> {
            let kernels = FixedSm120HalfKernels {
                m64n64_bk64_s2: load_half(&format!("gemm_bi_nn_sm120_tma_64x64_bk64_s2{suffix}"))?,
                m64n128_bk64_s2: load_half(&format!(
                    "gemm_bi_nn_sm120_tma_64x128_bk64_s2{suffix}"
                ))?,
                m128n64_bk32_s3: load_half(&format!(
                    "gemm_bi_nn_sm120_tma_128x64_bk32_s3{suffix}"
                ))?,
                m128n128_bk32_s2: load_half(&format!(
                    "gemm_bi_nn_sm120_tma_128x128_bk32_s2{suffix}"
                ))?,
                m128n128_bk32_s3: load_half(&format!(
                    "gemm_bi_nn_sm120_tma_128x128_bk32_s3{suffix}"
                ))?,
            };
            for (tile, bytes) in [
                (&kernels.m64n64_bk64_s2, 32_896),
                (&kernels.m64n128_bk64_s2, 49_280),
                (&kernels.m128n64_bk32_s3, 36_992),
                (&kernels.m128n128_bk32_s2, 32_896),
                (&kernels.m128n128_bk32_s3, 49_280),
            ] {
                for function in [&tile.bf16, &tile.f16] {
                    function
                        .set_attribute(
                            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes,
                        )
                        .map_err(|error| {
                            format!(
                                "set MAX_DYNAMIC_SHARED for Fixed SM120 half{suffix}: {error:?}"
                            )
                        })?;
                }
            }
            Ok(kernels)
        };

        Ok(Self {
            state_cap,
            compiler_identity,
            triad,
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
            conv1d_burnin_fwd_nosave_tiled: get("conv1d_burnin_forward_nosave_tiled_f32")?,
            conv1d_burnin_nosave_tiled_typed: load_typed("conv1d_burnin_forward_nosave_tiled")?,
            // activations
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
            exp_negate: get("exp_negate")?,
            exp_negate2: get("exp_negate2")?,
            gather_cols: get("gather_cols")?,
            gather_bc_cols: get("gather_bc_cols")?,
            gather_bc_cols_tmajor: get("gather_bc_cols_tmajor")?,
            gather_bc_cols_tmajor_tiled: get("gather_bc_cols_tmajor_tiled")?,
            gate_mul_silu: get("gate_mul_silu")?,
            gating_backward: get("gating_backward")?,
            residual_add: get("residual_add")?,
            gather_last_timestep: get("gather_last_timestep")?,

            // mixed precision casts
            cast_f32_to_bf16: get("cast_f32_to_bf16")?,
            cast_f32_to_f16: get("cast_f32_to_f16")?,
            cast_bf16_to_f32: get("cast_bf16_to_f32")?,
            cast_f16_to_f32: get("cast_f16_to_f32")?,
            ssm_burnin_fwd_bf16: get("ssm_burnin_forward_bf16")?,
            ssm_burnin_fwd_f16: get("ssm_burnin_forward_f16")?,

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
            gemm_bi_f32_f32_s2: get("gemm_bi_f32_f32_s2")?,
            gemm_bi_f32_f32_n128_s2: get("gemm_bi_f32_f32_n128_s2")?,
            fixed_sm89_half_pipeline,
            fixed_sm89_half_pipeline_rejection,
            fixed_sm89_half_swizzle,
            fixed_sm89_half_swizzle_rejection,
            fixed_sm89_half_s3,
            fixed_sm89_half_s3_rejection,
            fixed_sm89_tf32_rna_wide,
            fixed_sm89_tf32_rna_wide_rejection,
            fixed_sm89_tf32_rna_n96,
            fixed_sm89_tf32_rna_n96_rejection,
            fixed_sm89_half_m64n64_s3_f16,
            fixed_sm89_half_m64n64_s3_f16_rejection,
            fixed_sm89_half_m128n64_s2_f16,
            fixed_sm89_half_m128n64_s2_f16_rejection,
            fixed_sm89_f32_n64_copyplan,
            fixed_sm89_f32_n64_copyplan_rejection,
            fixed_sm120_f32_n64_copyplan,
            fixed_sm120_f32_n64_copyplan_rejection,
            fixed_sm120_f32_n64_copyplan_t256,
            fixed_sm120_f32_n64_copyplan_t256_rejection,
            fixed_sm120_f32_m128n64_copyplan_t256,
            fixed_sm120_f32_m128n64_copyplan_t256_rejection,
            fixed_sm120_f32_n64_sliced,
            fixed_sm120_f32_n64_sliced_rejection,
            fixed_sm120_fma_postbias,
            fixed_sm120_fma_postbias_rejection,
            gemm_bi_nn_tf32: {
                let kernels = FixedTf32Kernels {
                    m128n64_s2: get("gemm_bi_nn_tf32_v1_m128n64_bk32_s2")?,
                    m128n64_s3: get("gemm_bi_nn_tf32_v1_m128n64_bk32_s3")?,
                    m64n64_s2: get("gemm_bi_nn_tf32_v1_m64n64_bk32_s2")?,
                    m64n64_s3: get("gemm_bi_nn_tf32_v1_m64n64_bk32_s3")?,
                    m16n32_s4: get("gemm_bi_nn_tf32_v1_m16n32_bk32_s4")?,
                };
                for (function, bytes) in [
                    (&kernels.m128n64_s2, 55_296),
                    (&kernels.m128n64_s3, 82_944),
                    (&kernels.m64n64_s2, 32_768),
                    (&kernels.m64n64_s3, 55_296),
                    (&kernels.m16n32_s4, 29_696),
                ] {
                    function
                        .set_attribute(
                            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes,
                        )
                        .map_err(|error| {
                            format!("set MAX_DYNAMIC_SHARED for Fixed TF32: {error:?}")
                        })?;
                }
                kernels
            },
            gemm_bi_nn_tf32_sm120: if sm120_board
                && matches!(arch, "sm_120" | "sm_121" | "compute_120" | "compute_121")
            {
                let kernels = FixedSm120Tf32Kernels {
                    m128n64_s2: get("gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s2")?,
                    m128n64_s3: get("gemm_bi_nn_sm120_tma_tf32_v1_m128n64_bk32_s3")?,
                    m64n128_s2: get("gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s2")?,
                    m64n128_s3: get("gemm_bi_nn_sm120_tma_tf32_v1_m64n128_bk32_s3")?,
                    m64n64_s2_producer_warp: get(
                        "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_producer_warp",
                    )?,
                    m64n64_s2: get("gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2")?,
                    m64n64_s2_pair_store: get(
                        "gemm_bi_nn_sm120_tma_tf32_v1_m64n64_bk32_s2_pair_store",
                    )?,
                };
                for (function, bytes) in [
                    (&kernels.m128n64_s2, 49_280),
                    (&kernels.m128n64_s3, 73_856),
                    (&kernels.m64n128_s2, 49_280),
                    (&kernels.m64n128_s3, 73_856),
                    (&kernels.m64n64_s2_producer_warp, 32_896),
                    (&kernels.m64n64_s2, 32_896),
                    (&kernels.m64n64_s2_pair_store, 32_896),
                ] {
                    function
                        .set_attribute(
                            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            bytes,
                        )
                        .map_err(|error| {
                            format!("set MAX_DYNAMIC_SHARED for Fixed SM120 TF32: {error:?}")
                        })?;
                }
                Some(kernels)
            } else {
                None
            },
            gemm_bi_nn_half_sm120: if sm120_board
                && matches!(arch, "sm_120" | "sm_121" | "compute_120" | "compute_121")
            {
                Some(load_sm120_half("")?)
            } else {
                None
            },
            gemm_bi_nn_half_sm120_f32out: if sm120_board
                && matches!(arch, "sm_120" | "sm_121" | "compute_120" | "compute_121")
            {
                Some(load_sm120_half("_f32out")?)
            } else {
                None
            },

            // Batch-invariant matvec (M=1 specialization)
            matvec_bi_bf16_bf16: get("matvec_bi_bf16_bf16")?,
            matvec_bi_f16_f16: get("matvec_bi_f16_f16")?,
            matvec_bi_bf16_f32: get("matvec_bi_bf16_f32")?,
            matvec_bi_f16_f32: get("matvec_bi_f16_f32")?,
            matvec_bi_f32_f32: get("matvec_bi_f32_f32")?,

            // typed inference kernels
            softplus_fwd_typed: load_typed("softplus_forward")?,
            rmsnorm_fwd_resadd_typed: load_typed("rmsnorm_forward_resadd_f32in")?,
            bias_broadcast_typed: load_typed("bias_broadcast")?,
            elementwise_mul_typed: load_typed("elementwise_mul")?,
            residual_add_typed: load_typed("residual_add")?,
            gather_cols_typed: load_typed("gather_cols")?,
            gather_bc_cols_typed: load_typed("gather_bc_cols")?,
            gather_bc_cols_tmajor_typed: load_typed("gather_bc_cols_tmajor")?,
            gather_bc_cols_tmajor_tiled_typed: load_typed("gather_bc_cols_tmajor_tiled")?,
            gate_mul_silu_typed: load_typed("gate_mul_silu")?,
            gate_mul_silu_v_typed: load_typed("gate_mul_silu_v")?,
            elementwise_mul_v_typed: load_typed("elementwise_mul_v")?,
            softplus_copy_typed: load_typed("softplus_copy")?,
            ssm_step_fwd_fused_typed: load_typed("ssm_step_forward_fused")?,
            conv1d_step_fwd_silu_typed: load_typed("conv1d_step_forward_silu")?,
            ssm_burnin_nosave_typed: load_typed("ssm_burnin_forward_nosave")?,
            softplus_bwd_typed: load_typed("softplus_backward")?,
            gather_last_timestep_typed: load_typed("gather_last_timestep")?,
            vec_cast_zplus_typed: load_typed("vec_cast_zplus")?,
            concat_halves_typed: load_typed("concat_halves")?,
            scatter_add_cols_typed: load_typed("scatter_add_cols")?,
            reduce_bias_typed: load_typed("reduce_bias")?,

            // typed training-backward kernels
            gating_bwd_typed: load_typed("gating_backward")?,
            rmsnorm_bwd_typed: load_typed("rmsnorm_backward")?,
            conv1d_burnin_bwd_typed: load_typed("conv1d_burnin_backward")?,
            conv1d_burnin_fwd_tiled_typed: load_typed("conv1d_burnin_forward_tiled")?,
            conv1d_bwd_tiled_typed: load_typed("conv1d_bwd_tiled")?,
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

            gemm_bi_nn_tc128_typed: load_half_dynsmem("gemm_bi_nn_tc128", 71_680)?,
            gemm_bi_nn_tc128_f32out: load_half_dynsmem("gemm_bi_nn_tc128_f32out", 71_680)?,
            gemm_bi_nn_tc64_f32out: load_half("gemm_bi_nn_tc64_f32out")?,
            gemm_bi_nn_tc16_f32out: load_half("gemm_bi_nn_tc16_f32out")?,
            gemm_bi_nn_tc64_typed: load_half("gemm_bi_nn_tc64")?,
            gemm_bi_nn_tc16_typed: load_half("gemm_bi_nn_tc16")?,
            gemm_bi_nn_tcw64_typed: load_half_dynsmem("gemm_bi_nn_tcw64", 65_536)?,
            gemm_bi_nn_tcwn64_typed: load_half_dynsmem("gemm_bi_nn_tcwn64", 98_304)?,
            gemm_bi_nn_sm90_typed: if arch == "sm_90a" {
                Some(load_half_dynsmem("gemm_bi_nn_sm90a_wgmma_wg1", 49_152)?)
            } else {
                None
            },
            gemm_bi_nn_sm100_typed: if matches!(arch, "sm_100a" | "sm_103a" | "sm_110a") {
                Some(load_half_dynsmem("gemm_bi_nn_sm100_tcgen_c4", 65_536)?)
            } else {
                None
            },
            _modules: CudaModuleAnchors::new(vec![module]),
        })
    }

    /// Compiler invocation that produced the loaded Fixed module.
    ///
    /// Scalar and SM80 triad compiler identities are exposed through the
    /// embedded [`GemmBiKernels`] aggregate.
    pub fn compiler_identity(&self) -> super::kernel_identity::CompilerIdentity {
        self.compiler_identity
    }

    pub(crate) fn specialized_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad
            .sm120_compiler_identity()
            .or_else(|| self.triad.sm100_compiler_identity())
            .or_else(|| self.triad.sm90a_compiler_identity())
    }

    /// Ordered artifact set available to deterministic GEMM dispatch.
    pub fn artifact_set_identity(&self) -> super::kernel_identity::ArtifactSetIdentity {
        self.triad.artifact_set_identity()
    }

    pub fn f32_triad_availability(&self) -> super::gemm_bi_triad::F32TriadAvailability {
        self.triad.f32_triad_availability()
    }

    /// The reason the specialized TF32 module is not bound, when it is not.
    pub fn specialized_tf32_rejection(&self) -> Option<&str> {
        self.triad.specialized_tf32_rejection()
    }

    pub fn finalist_tf32_rejection(&self) -> Option<&str> {
        self.triad.finalist_tf32_rejection()
    }

    pub fn triad_scalar_compiler_identity(&self) -> super::kernel_identity::CompilerIdentity {
        self.triad.scalar_compiler_identity()
    }

    pub(crate) fn triad_scalar_compute_capability(&self) -> (u32, u32) {
        self.triad.compute_capability()
    }

    pub(crate) fn tc64_streamk_resident_ctas(&self) -> u32 {
        self.triad.tc64_streamk_resident_ctas()
    }

    pub(crate) fn triad_sm80_compiler_identity(&self) -> super::kernel_identity::CompilerIdentity {
        self.triad.sm80_compiler_identity()
    }

    pub(crate) fn triad_sm89_finalist_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad.sm89_finalist_compiler_identity()
    }

    pub fn triad_sm89_half_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad.sm89_half_compiler_identity()
    }

    pub fn triad_sm89_half_artifact_identity(
        &self,
    ) -> Option<super::kernel_identity::ArtifactIdentity> {
        self.triad.artifact_set_identity().sm89_half
    }

    pub fn triad_sm89_half_rejection(&self) -> Option<&str> {
        self.triad.sm89_half_rejection()
    }

    pub fn triad_sm89_half_exclusions(&self) -> Vec<(&'static str, &str)> {
        self.triad
            .sm89_half_exclusions()
            .iter()
            .map(|excluded| (excluded.symbol, excluded.reason.as_str()))
            .collect()
    }

    #[doc(hidden)]
    pub fn triad_sm89_half_function(
        &self,
        route: super::gemm_bi_triad::Sm89HalfRoute,
        dtype: super::dtype::WeightDtype,
    ) -> Option<&CudaFunction> {
        self.triad.sm89_half_function(route, dtype)
    }

    pub fn triad_sm89_exact_f32_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad.sm89_exact_f32_compiler_identity()
    }

    pub fn triad_sm89_exact_f32_artifact_identity(
        &self,
    ) -> Option<super::kernel_identity::ArtifactIdentity> {
        self.triad.artifact_set_identity().sm89_exact_f32
    }

    pub fn triad_sm89_exact_f32_rejection(&self) -> Option<&str> {
        self.triad.sm89_exact_f32_rejection()
    }

    pub fn triad_sm89_exact_f32_exclusions(&self) -> Vec<(&'static str, &str)> {
        self.triad
            .sm89_exact_f32_exclusions()
            .iter()
            .map(|excluded| (excluded.symbol, excluded.reason.as_str()))
            .collect()
    }

    #[doc(hidden)]
    pub fn triad_sm89_exact_f32_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.triad.sm89_exact_f32_function(symbol)
    }

    pub fn triad_sm89_exact_f32_d128_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad.sm89_exact_f32_d128_compiler_identity()
    }

    pub fn triad_sm89_exact_f32_d128_artifact_identity(
        &self,
    ) -> Option<super::kernel_identity::ArtifactIdentity> {
        self.triad.artifact_set_identity().sm89_exact_f32_d128
    }

    pub fn triad_sm89_exact_f32_d128_rejection(&self) -> Option<&str> {
        self.triad.sm89_exact_f32_d128_rejection()
    }

    pub fn triad_sm89_exact_f32_d128_exclusions(&self) -> Vec<(&'static str, &str)> {
        self.triad
            .sm89_exact_f32_d128_exclusions()
            .iter()
            .map(|excluded| (excluded.symbol, excluded.reason.as_str()))
            .collect()
    }

    #[doc(hidden)]
    pub fn triad_sm89_exact_f32_d128_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.triad.sm89_exact_f32_d128_function(symbol)
    }

    pub fn triad_sm89_tf32_joint_compiler_identity(
        &self,
    ) -> Option<super::kernel_identity::CompilerIdentity> {
        self.triad.sm89_tf32_joint_compiler_identity()
    }

    pub fn triad_sm89_tf32_joint_artifact_identity(
        &self,
    ) -> Option<super::kernel_identity::ArtifactIdentity> {
        self.triad.artifact_set_identity().sm89_tf32_joint
    }

    pub fn triad_sm89_tf32_joint_rejection(&self) -> Option<&str> {
        self.triad.sm89_tf32_joint_rejection()
    }

    pub fn triad_sm89_tf32_joint_exclusions(&self) -> Vec<(&'static str, &str)> {
        self.triad
            .sm89_tf32_joint_exclusions()
            .iter()
            .map(|excluded| (excluded.symbol, excluded.reason.as_str()))
            .collect()
    }

    #[doc(hidden)]
    pub fn triad_sm89_tf32_joint_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.triad.sm89_tf32_joint_function(symbol)
    }

    pub(crate) fn tf32_function(&self, symbol: &str) -> Option<&CudaFunction> {
        self.triad.tf32_function(symbol)
    }

    pub(crate) fn triad_kernels(&self) -> &GemmBiKernels {
        &self.triad
    }

    /// The Split-K/Split-M partial scratch (8M f32 = 32 MB), allocated on
    /// first batch-invariant use. Zero-initialized like the old eager
    /// alloc; the partial kernels overwrite their region before the
    /// reduce reads it, so first-use contents were never load-bearing.
    pub fn splitk_scratch_buf(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<&cudarc::driver::CudaSlice<f32>, String> {
        self.triad.splitk_scratch_buf(stream)
    }

    /// The W-transpose staging scratch (4,718,592 f32 = 18 MiB), allocated on
    /// first batch-invariant bwd_dx wide-path use.
    pub fn transpose_scratch_buf(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<&cudarc::driver::CudaSlice<f32>, String> {
        self.triad.transpose_scratch_buf(stream)
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
