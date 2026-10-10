//! The Mamba-2 CUDA module: `kernels/mamba2_ops.cu`, `mamba2_ssd.cu` and
//! `mamba2_bwd.cu` after the typed prelude, compiled with NVRTC and cached
//! under its own module kind, so adding it moves no identity of the Fixed
//! or Mamba-3 modules. The norms, the residual adds and the optimizer come
//! from the Fixed module through the context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::mamba_ssm::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::mamba_ssm::gpu::kernel_identity as ki;
use crate::mamba_ssm::gpu::kernels::{HalfKernel, TypedKernel};

/// The compiled Mamba-2 kernels.
pub struct Mamba2Kernels {
    /// The decode step of every head, by activation dtype.
    pub step_fwd: TypedKernel,
    /// The sequential prefill: the decode step's recurrence over a window.
    pub seq_fwd: TypedKernel,
    /// The training forward's sequential scan, writing replay checkpoints.
    pub seq_fwd_ckpt: TypedKernel,
    /// The grouped gated RMSNorm.
    pub gated_norm_fwd: TypedKernel,
    /// The gate without a norm.
    pub gate_only_fwd: TypedKernel,
    /// The mixer's inner MLP gate.
    pub mlp_gate_fwd: TypedKernel,
    /// The prefill: conv over the window and its state, dt with the chunk
    /// cumsum, the CB products, the chunk states, the state passing and
    /// the chunk scan.
    pub conv_prefill: TypedKernel,
    /// The decode's conv, the prefill conv's chain at one token.
    pub conv_step: TypedKernel,
    pub conv_state: TypedKernel,
    pub dt_cumsum: TypedKernel,
    pub bmm_chunk: TypedKernel,
    pub chunk_state: TypedKernel,
    pub state_passing: CudaFunction,
    pub chunk_scan: TypedKernel,
    /// The training backward, f32.
    pub bwd: Mamba2BwdKernels,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
}

/// The backward of the window and the training helpers
/// (`kernels/mamba2_bwd.cu`); the kernels that read the stored projection
/// or conv rows come in every activation dtype.
pub struct Mamba2BwdKernels {
    pub a_neg: CudaFunction,
    pub dhloc: TypedKernel,
    pub state_passing: CudaFunction,
    pub chunk_dot: CudaFunction,
    pub dx: TypedKernel,
    pub rowsums: TypedKernel,
    pub pairs: TypedKernel,
    pub db: TypedKernel,
    pub dc: TypedKernel,
    pub combine: TypedKernel,
    pub head_grads: CudaFunction,
    pub d_parts: TypedKernel,
    pub d_reduce: CudaFunction,
    pub gated_norm: TypedKernel,
    pub gate_only: TypedKernel,
    pub mlp_gate: TypedKernel,
    pub conv_dv: TypedKernel,
    pub conv_dx: TypedKernel,
    pub conv_dw: TypedKernel,
    /// The sequential path's backward of the recurrence (replay per
    /// segment) and the group sums of its dB and dC.
    pub seq_bwd: TypedKernel,
    pub seq_bc_groups: CudaFunction,
    /// A bias gradient from bf16 or f16 rows.
    pub bias_grad: HalfKernel,
}

/// The threads of one decode-step block: one warp, one channel each. Its
/// shared state tile is `32 * (d_state + 1)` floats, 33 KB at d_state 256.
pub const STEP_THREADS: u32 = 32;

/// Launch geometry of [`Mamba2Kernels::step_fwd`].
pub fn step_cfg(batch: usize, nh: usize, hd: usize, ds: usize) -> LaunchConfig {
    let rows = (STEP_THREADS as usize).min(hd);
    LaunchConfig {
        grid_dim: (
            batch as u32,
            nh as u32,
            hd.div_ceil(STEP_THREADS as usize) as u32,
        ),
        block_dim: (STEP_THREADS, 1, 1),
        shared_mem_bytes: ((2 * ds + 4 + rows * (ds + 1)) * 4) as u32,
    }
}

/// Tokens per SSD chunk.
pub const CHUNK: usize = 64;

/// Launch geometry of the gated norm: one block per (row, group), its
/// group and per-head sums staged in shared memory.
pub fn gated_norm_cfg(rows: usize, d_ssm: usize, ng: usize, hd: usize) -> LaunchConfig {
    let gs = d_ssm / ng;
    LaunchConfig {
        grid_dim: (rows as u32, ng as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: ((gs + gs / hd) * 4) as u32,
    }
}

impl Mamba2Kernels {
    /// Compile (or load from the cache) the Mamba-2 module for `arch`.
    pub fn compile(ctx: &Arc<CudaContext>, arch: &'static str) -> Result<Self, String> {
        let ComposedModule {
            module,
            compiler_identity,
            artifact_identity,
        } = compile_composed(
            ctx,
            arch,
            &ComposedSpec {
                sources: &[
                    include_str!("../../../kernels/_typed_prelude.cuh"),
                    include_str!("../../../kernels/mamba2_ops.cu"),
                    include_str!("../../../kernels/mamba2_ssd.cu"),
                    include_str!("../../../kernels/mamba2_bwd.cu"),
                ],
                kind: ki::ModuleKind::Mamba2Combined,
                cache_prefix: "mamba2-kernels-v1-",
                seed: "2405210601",
                label: "Mamba-2",
            },
        )?;
        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("Mamba-2 kernel '{name}' not found: {e:?}"))
        };
        let typed = |base: &str| -> Result<TypedKernel, String> {
            Ok(TypedKernel {
                f32: get(base)?,
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        let kernels = Self {
            step_fwd: typed("m2_step_fwd")?,
            seq_fwd: typed("m2_seq_fwd")?,
            seq_fwd_ckpt: typed("m2_seq_fwd_ckpt")?,
            gated_norm_fwd: typed("m2_gated_norm_fwd")?,
            gate_only_fwd: typed("m2_gate_only_fwd")?,
            mlp_gate_fwd: typed("m2_mlp_gate_fwd")?,
            conv_prefill: typed("m2_conv_prefill")?,
            conv_step: typed("m2_conv_step")?,
            conv_state: typed("m2_conv_state")?,
            dt_cumsum: typed("m2_dt_cumsum")?,
            bmm_chunk: typed("m2_bmm_chunk")?,
            chunk_state: typed("m2_chunk_state")?,
            state_passing: get("m2_state_passing")?,
            chunk_scan: typed("m2_chunk_scan")?,
            bwd: Mamba2BwdKernels {
                a_neg: get("m2_a_neg")?,
                dhloc: typed("m2b_dhloc")?,
                state_passing: get("m2b_state_passing")?,
                chunk_dot: get("m2b_chunk_dot")?,
                dx: typed("m2b_dx")?,
                rowsums: typed("m2b_rowsums")?,
                pairs: typed("m2b_pairs")?,
                db: typed("m2b_db")?,
                dc: typed("m2b_dc")?,
                combine: typed("m2b_combine")?,
                head_grads: get("m2b_head_grads")?,
                d_parts: typed("m2b_d_parts")?,
                d_reduce: get("m2b_d_reduce")?,
                gated_norm: typed("m2b_gated_norm")?,
                gate_only: typed("m2b_gate_only")?,
                mlp_gate: typed("m2b_mlp_gate")?,
                conv_dv: typed("m2b_conv_dv")?,
                conv_dx: typed("m2b_conv_dx")?,
                conv_dw: typed("m2b_conv_dw")?,
                seq_bwd: typed("m2s_bwd")?,
                seq_bc_groups: get("m2s_bc_groups")?,
                bias_grad: HalfKernel {
                    bf16: get("m2b_bias_grad_bf16")?,
                    f16: get("m2b_bias_grad_f16")?,
                },
            },
            compiler_identity,
            artifact_identity,
        };
        {
            use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
            let k = &kernels.bwd.seq_bwd;
            for f in [&k.f32, &k.bf16, &k.f16] {
                // A refused opt-in leaves the default 48 KB; an oversized
                // launch then fails loudly with its own error.
                let _ = f.set_attribute(
                    FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SEQ_BWD_SMEM_MAX as i32,
                );
            }
        }
        Ok(kernels)
    }
}

/// The per-block opt-in the sequential backward may use: consumer parts
/// cap it near 99 KB.
pub const SEQ_BWD_SMEM_MAX: usize = 99 * 1024;

/// The sequential backward's shared memory for a head: the replay state
/// rows and the state-gradient rows (each padded by one), B and C, the step
/// scalars, x and dy of the token, one reduction slot per thread.
pub fn seq_bwd_smem_bytes(hd: usize, ds: usize, threads: usize) -> usize {
    (2 * hd * (ds + 1) + 2 * ds + 4 + 2 * hd + threads) * 4
}
