//! Compile and register Mamba-3 SISO CUDA kernels.
//!
//! 47 kernels across 5 .cu files, compiled via NVRTC at runtime.
//! Separate from Mamba-1's `MambaKernels` — different pipeline, no conv1d.

use crate::mamba_ssm::gpu::kernels::{HalfKernel, TypedKernel};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use std::sync::Arc;

/// All compiled Mamba-3 SISO CUDA kernels.
pub struct Mamba3Kernels {
    _module: Arc<CudaModule>,

    // ── Sequential SSM (mamba3_ssd.cu) ──
    pub m3_step_fwd: CudaFunction,
    pub m3_burnin_fwd: CudaFunction,
    pub m3_burnin_fwd_nosave: CudaFunction,
    pub m3_backward_seq: CudaFunction,
    pub m3_reduce_d_d: CudaFunction,

    // ── Shared ops (mamba3_ops.cu) ──
    pub m3_split: CudaFunction,
    pub m3_split_bwd: CudaFunction,
    pub bcnorm_fwd: CudaFunction,
    pub bcnorm_bwd: CudaFunction,
    pub bc_bias_add: CudaFunction,
    pub bc_bias_add_bwd: CudaFunction,
    pub angle_dt_fwd: CudaFunction,
    pub m3_angle_dt_fwd_batch: CudaFunction,
    pub m3_angle_dt_fwd_seq: CudaFunction,
    pub angle_dt_bwd: CudaFunction,
    pub m3_angle_dt_bwd_seq: CudaFunction,
    pub rope_fwd: CudaFunction,
    pub rope_bwd: CudaFunction,
    pub m3_compute_abg: CudaFunction,
    pub m3_abg_bwd: CudaFunction,
    pub silu_gate_fwd: CudaFunction,
    pub silu_gate_bwd: CudaFunction,
    pub rmsnorm_gated_fwd: CudaFunction,
    pub rmsnorm_gated_bwd: CudaFunction,

    // ── Shared kernels from norms.cu + elementwise.cu (used by training pipeline) ──
    pub rmsnorm_fwd: CudaFunction,
    pub rmsnorm_bwd: CudaFunction,
    pub colsum_accumulate: CudaFunction,
    pub vec_add_inplace: CudaFunction,
    pub elementwise_mul: CudaFunction,
    pub fill_scalar: CudaFunction,
    pub residual_add: CudaFunction,
    pub gather_last_timestep: CudaFunction,

    // ── Chunked parallel scan (mamba3_chunked.cu) ──
    pub m3_preprocess_chunks: CudaFunction,
    pub m3_da_cumsum: CudaFunction,
    pub m3_chunk_state_fwd: CudaFunction,
    pub m3_state_passing_fwd: CudaFunction,
    pub m3_writeback_parallel_states: CudaFunction,
    pub m3_chunk_scan_fwd: CudaFunction,
    pub m3_chunk_scan_bwd: CudaFunction,
    pub m3_state_passing_bwd: CudaFunction,
    pub m3_chunk_state_bwd: CudaFunction,
    pub m3_cumsum_bwd: CudaFunction,
    pub m3_extract_da_cs_sum: CudaFunction,
    pub m3_dqkv: CudaFunction,
    pub m3_dqktheta: CudaFunction,
    pub m3_ddt_dtrap: CudaFunction,
    pub m3_final_grads: CudaFunction,

    // ── Typed variants for end-to-end bf16/f16 inference ──
    /// 8-way split + fused softplus/sigmoid, bf16/f16 proj and activation outputs.
    /// Coefficient outputs (dt, a_val, trap, angles, raw saves) stay f32.
    pub m3_split_typed: TypedKernel,
    /// RMSNorm on B/C per group, half I/O, f32 rms_val and weight.
    pub bcnorm_fwd_typed: TypedKernel,
    /// Fused B+C variant of bcnorm_fwd_typed (2× grid via blockIdx.y).
    pub bcnorm_fwd_bc_typed: TypedKernel,
    /// Per-head bias add, half I/O, f32 bias.
    pub bc_bias_add_typed: TypedKernel,
    /// Fused B+C variant of bc_bias_add_typed (2× grid via blockIdx.y).
    pub bc_bias_add_bc_typed: TypedKernel,
    /// RoPE rotation, half B/C, f32 angle_cumsum.
    pub rope_fwd_typed: TypedKernel,
    /// Plain SiLU gate (no norm), half I/O.
    pub silu_gate_fwd_typed: TypedKernel,
    /// RMSNorm-gated output (half I/O, f32 weight/rms_vals).
    pub rmsnorm_gated_fwd_typed: TypedKernel,
    /// M3 SSM step — shared with training, already templated in mamba3_ssd.cu.
    pub m3_step_fwd_typed: TypedKernel,
    /// M3 burnin forward (training) — sequential T-loop SSM with activation
    /// saves. Typed x/k/q/y; f32 state + alpha/beta/gamma + D + saves.
    pub m3_burnin_fwd_typed_bf16: CudaFunction,
    pub m3_burnin_fwd_typed_f16: CudaFunction,

    // -- Step 9a: typed M3 sequential backward kernels --
    /// RmsNorm over B/C groups, typed dy → typed d_B; f32 rms + weight +
    /// d_weight master-grad accumulator.
    pub bcnorm_bwd_typed: TypedKernel,
    /// Per-head bias reduction from typed d_B_biased → typed d_B_normed
    /// (expand groups backward). Bias grad handled by reduce_bias_typed.
    pub bc_bias_add_bwd_typed: TypedKernel,
    /// RoPE rotation backward, typed B/C grads + saved B/C, f32 angle saves.
    pub rope_bwd_typed: TypedKernel,
    /// 8-way split backward: assemble typed d_proj from typed d_z/d_x/
    /// d_B_raw/d_C_raw plus f32 dd_dt/dd_a/trap/angles.
    pub m3_split_bwd_typed: TypedKernel,
    /// Shared from M1: f32 residual → half post-norm (identical kernel, reused).
    pub rmsnorm_fwd_f32in_typed: HalfKernel,
    /// Shared from M1: f32 residual += half branch (stays f32).
    pub residual_add_f32_typed: HalfKernel,
    /// Shared from M1: gather last timestep of B×T×D into B×D, dtype-preserving.
    pub gather_last_timestep_typed: TypedKernel,
}

impl Mamba3Kernels {
    /// Compile all 47 Mamba-3 CUDA kernels from source. Takes ~100-200ms.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let sources = [
            // Inline the prelude first so each source file's
            // `#include "_typed_prelude.cuh"` can be safely stripped below.
            include_str!("../../../kernels/_typed_prelude.cuh"),
            include_str!("../../../kernels/mamba3_ssd.cu"),
            include_str!("../../../kernels/mamba3_ops.cu"),
            include_str!("../../../kernels/mamba3_chunked.cu"),
            // Shared kernels needed by training pipeline
            include_str!("../../../kernels/norms.cu"),
            include_str!("../../../kernels/elementwise.cu"),
        ];

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
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
            ],
            include_paths: crate::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(combined, opts)
            .map_err(|e| format!("NVRTC M3 compile failed: {e:?}"))?;

        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("M3 module load failed: {e:?}"))?;

        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("M3 kernel '{name}' not found: {e:?}"))
        };

        Ok(Self {
            // Sequential SSM
            m3_step_fwd: get("m3_step_fwd")?,
            m3_burnin_fwd: get("m3_burnin_fwd")?,
            m3_burnin_fwd_nosave: get("m3_burnin_fwd_nosave")?,
            m3_backward_seq: get("m3_backward_seq")?,
            m3_reduce_d_d: get("m3_reduce_d_D")?,

            // Shared ops
            m3_split: get("m3_split")?,
            m3_split_bwd: get("m3_split_bwd")?,
            bcnorm_fwd: get("bcnorm_fwd")?,
            bcnorm_bwd: get("bcnorm_bwd")?,
            bc_bias_add: get("bc_bias_add")?,
            bc_bias_add_bwd: get("bc_bias_add_bwd")?,
            angle_dt_fwd: get("angle_dt_fwd")?,
            m3_angle_dt_fwd_batch: get("m3_angle_dt_fwd_batch")?,
            m3_angle_dt_fwd_seq: get("m3_angle_dt_fwd_seq")?,
            angle_dt_bwd: get("angle_dt_bwd")?,
            m3_angle_dt_bwd_seq: get("m3_angle_dt_bwd_seq")?,
            rope_fwd: get("rope_fwd")?,
            rope_bwd: get("rope_bwd")?,
            m3_compute_abg: get("m3_compute_abg")?,
            m3_abg_bwd: get("m3_abg_bwd")?,
            silu_gate_fwd: get("silu_gate_fwd")?,
            silu_gate_bwd: get("silu_gate_bwd")?,
            rmsnorm_gated_fwd: get("rmsnorm_gated_forward")?,
            rmsnorm_gated_bwd: get("rmsnorm_gated_backward")?,

            // Shared (norms.cu + elementwise.cu)
            rmsnorm_fwd: get("rmsnorm_forward")?,
            rmsnorm_bwd: get("rmsnorm_backward")?,
            colsum_accumulate: get("colsum_accumulate")?,
            vec_add_inplace: get("vec_add_inplace")?,
            elementwise_mul: get("elementwise_mul")?,
            fill_scalar: get("fill_scalar")?,
            residual_add: get("residual_add")?,
            gather_last_timestep: get("gather_last_timestep")?,

            // Chunked parallel scan
            m3_preprocess_chunks: get("m3_preprocess_chunks")?,
            m3_da_cumsum: get("m3_dA_cumsum")?,
            m3_chunk_state_fwd: get("m3_chunk_state_fwd")?,
            m3_state_passing_fwd: get("m3_state_passing_fwd")?,
            m3_writeback_parallel_states: get("m3_writeback_parallel_states")?,
            m3_chunk_scan_fwd: get("m3_chunk_scan_fwd")?,
            m3_chunk_scan_bwd: get("m3_chunk_scan_bwd")?,
            m3_state_passing_bwd: get("m3_state_passing_bwd")?,
            m3_chunk_state_bwd: get("m3_chunk_state_bwd")?,
            m3_cumsum_bwd: get("m3_cumsum_bwd")?,
            m3_extract_da_cs_sum: get("m3_extract_da_cs_sum")?,
            m3_dqkv: get("m3_dqkv")?,
            m3_dqktheta: get("m3_dqktheta")?,
            m3_ddt_dtrap: get("m3_ddt_dtrap")?,
            m3_final_grads: get("m3_final_grads")?,

            // Typed variants for mixed-dtype inference
            m3_split_typed: TypedKernel {
                f32: get("m3_split")?,
                bf16: get("m3_split_bf16")?,
                f16: get("m3_split_f16")?,
            },
            bcnorm_fwd_typed: TypedKernel {
                f32: get("bcnorm_fwd")?,
                bf16: get("bcnorm_fwd_bf16")?,
                f16: get("bcnorm_fwd_f16")?,
            },
            bcnorm_fwd_bc_typed: TypedKernel {
                // f32 fused variant not yet implemented; reuse bcnorm_fwd
                // (callers select this only on bf16/f16 paths).
                f32: get("bcnorm_fwd")?,
                bf16: get("bcnorm_fwd_bc_bf16")?,
                f16: get("bcnorm_fwd_bc_f16")?,
            },
            bc_bias_add_typed: TypedKernel {
                f32: get("bc_bias_add")?,
                bf16: get("bc_bias_add_bf16")?,
                f16: get("bc_bias_add_f16")?,
            },
            bc_bias_add_bc_typed: TypedKernel {
                f32: get("bc_bias_add")?,
                bf16: get("bc_bias_add_bc_bf16")?,
                f16: get("bc_bias_add_bc_f16")?,
            },
            rope_fwd_typed: TypedKernel {
                f32: get("rope_fwd")?,
                bf16: get("rope_fwd_bf16")?,
                f16: get("rope_fwd_f16")?,
            },
            silu_gate_fwd_typed: TypedKernel {
                f32: get("silu_gate_fwd")?,
                bf16: get("silu_gate_fwd_bf16")?,
                f16: get("silu_gate_fwd_f16")?,
            },
            rmsnorm_gated_fwd_typed: TypedKernel {
                f32: get("rmsnorm_gated_forward")?,
                bf16: get("rmsnorm_gated_forward_bf16")?,
                f16: get("rmsnorm_gated_forward_f16")?,
            },
            m3_step_fwd_typed: TypedKernel {
                f32: get("m3_step_fwd")?,
                bf16: get("m3_step_fwd_bf16")?,
                f16: get("m3_step_fwd_f16")?,
            },
            m3_burnin_fwd_typed_bf16: get("m3_burnin_fwd_bf16")?,
            m3_burnin_fwd_typed_f16: get("m3_burnin_fwd_f16")?,
            bcnorm_bwd_typed: TypedKernel {
                f32: get("bcnorm_bwd")?,
                bf16: get("bcnorm_bwd_bf16")?,
                f16: get("bcnorm_bwd_f16")?,
            },
            bc_bias_add_bwd_typed: TypedKernel {
                f32: get("bc_bias_add_bwd")?,
                bf16: get("bc_bias_add_bwd_bf16")?,
                f16: get("bc_bias_add_bwd_f16")?,
            },
            rope_bwd_typed: TypedKernel {
                f32: get("rope_bwd")?,
                bf16: get("rope_bwd_bf16")?,
                f16: get("rope_bwd_f16")?,
            },
            m3_split_bwd_typed: TypedKernel {
                f32: get("m3_split_bwd")?,
                bf16: get("m3_split_bwd_bf16")?,
                f16: get("m3_split_bwd_f16")?,
            },
            rmsnorm_fwd_f32in_typed: HalfKernel {
                bf16: get("rmsnorm_forward_f32in_bf16")?,
                f16: get("rmsnorm_forward_f32in_f16")?,
            },
            residual_add_f32_typed: HalfKernel {
                bf16: get("residual_add_f32_bf16")?,
                f16: get("residual_add_f32_f16")?,
            },
            gather_last_timestep_typed: TypedKernel {
                f32: get("gather_last_timestep_f32")?,
                bf16: get("gather_last_timestep_bf16")?,
                f16: get("gather_last_timestep_f16")?,
            },

            _module: module,
        })
    }
}
