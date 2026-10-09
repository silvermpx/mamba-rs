//! The Mamba-2 CUDA module: `kernels/mamba2_ops.cu` after the typed
//! prelude, compiled with NVRTC and cached under its own module kind, so
//! adding it moves no identity of the Fixed or Mamba-3 modules. The norms
//! and the residual adds come from the Fixed module through the context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::mamba_ssm::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::mamba_ssm::gpu::kernel_identity as ki;
use crate::mamba_ssm::gpu::kernels::TypedKernel;

/// The compiled Mamba-2 kernels.
pub struct Mamba2Kernels {
    /// The decode step of every head, by activation dtype.
    pub step_fwd: TypedKernel,
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
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
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
        Ok(Self {
            step_fwd: typed("m2_step_fwd")?,
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
            compiler_identity,
            artifact_identity,
        })
    }
}
