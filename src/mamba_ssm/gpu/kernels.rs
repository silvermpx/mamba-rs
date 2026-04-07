//! Compile and register CUDA kernels for Mamba SSM.
//!
//! Uses NVRTC to compile .cu source to native CUBIN at runtime.
//! No pre-built PTX or binaries required.

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use std::sync::Arc;

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

    // -- Parallel scan (optional, for T>128) --
    /// Parallel prefix scan SSM forward with activation saves.
    pub ssm_parallel_fwd: CudaFunction,
    /// Parallel prefix scan SSM forward without saves (target network).
    pub ssm_parallel_fwd_nosave: CudaFunction,
}

impl MambaKernels {
    /// Compile all CUDA kernels from source. Takes ~100-200ms.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let sources = [
            include_str!("../../../kernels/mamba_ssm.cu"),
            include_str!("../../../kernels/mamba_ssm_parallel.cu"),
            include_str!("../../../kernels/conv1d.cu"),
            include_str!("../../../kernels/activations.cu"),
            include_str!("../../../kernels/norms.cu"),
            include_str!("../../../kernels/elementwise.cu"),
        ];

        let combined = sources.join("\n");
        // No --use_fast_math: it flushes denormals to zero and replaces
        // exp/sqrt with approximate intrinsics (__expf/__rsqrtf), which
        // breaks gradient flow through SSM BPTT chains and RMSNorm.
        // --fmad=true enables fused multiply-add (safe, precise).
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec![
                "--fmad=true".to_string(),
                "--extra-device-vectorization".to_string(),
            ],
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

            // parallel scan
            ssm_parallel_fwd: get("ssm_parallel_scan_fwd")?,
            ssm_parallel_fwd_nosave: get("ssm_parallel_scan_fwd_nosave")?,

            _module: module,
        })
    }
}
