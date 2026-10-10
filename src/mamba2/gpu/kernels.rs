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
use crate::mamba_ssm::gpu::launch::opt_in_dynamic_smem;

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
    /// The dynamic shared memory each opted-in kernel (the scans, the gated
    /// norms, the sequential backward) may stage: the device's per-block
    /// opt-in less the largest static stage among them.
    pub smem_optin: usize,
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

/// The scan's shared memory for a block of `rows` channels: B and C, the
/// step scalars and the rows' state tile, each row padded by one.
pub fn step_smem_bytes(rows: usize, ds: usize) -> usize {
    (2 * ds + 4 + rows * (ds + 1)) * 4
}

/// The channels per block of the decode and sequential scans: a warp when
/// its state tile fits `cap` bytes, fewer otherwise. Each thread keeps one
/// channel's state and sums its readout alone, so the block size never
/// changes a value. `None` when even one channel does not fit.
pub fn step_rows(hd: usize, ds: usize, cap: usize) -> Option<usize> {
    [32usize, 16, 8, 4, 2, 1]
        .into_iter()
        .map(|r| r.min(hd.max(1)))
        .find(|&r| step_smem_bytes(r, ds) <= cap)
}

/// Launch geometry of the decode and sequential scans with `rows` channels
/// per block (see [`step_rows`]).
pub fn step_cfg(batch: usize, nh: usize, (hd, ds): (usize, usize), rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, nh as u32, hd.div_ceil(rows) as u32),
        block_dim: (rows as u32, 1, 1),
        shared_mem_bytes: step_smem_bytes(rows, ds) as u32,
    }
}

/// Tokens per SSD chunk.
pub const CHUNK: usize = 64;

/// The gated norm's shared memory: the group and its per-head sums.
pub fn gated_norm_smem_bytes(d_ssm: usize, ng: usize, hd: usize) -> usize {
    let gs = d_ssm / ng;
    (gs + gs / hd) * 4
}

/// The gated norm backward's shared memory: the group twice and its
/// per-head sums.
pub fn gated_norm_bwd_smem_bytes(d_ssm: usize, ng: usize, hd: usize) -> usize {
    let gs = d_ssm / ng;
    (2 * gs + gs / hd) * 4
}

/// Launch geometry of the gated norm: one block per (row, group), its
/// group and per-head sums staged in shared memory.
pub fn gated_norm_cfg(rows: usize, d_ssm: usize, ng: usize, hd: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, ng as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: gated_norm_smem_bytes(d_ssm, ng, hd) as u32,
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
        let smem_optin = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|e| format!("query opt-in shared memory: {e:?}"))?
            .max(0) as usize;
        let mut kernels = Self {
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
            smem_optin,
            compiler_identity,
            artifact_identity,
        };
        kernels.smem_optin = {
            let optin = kernels.smem_optin;
            let mut budget = optin;
            for (k, name) in [
                (&kernels.step_fwd, "step"),
                (&kernels.seq_fwd, "sequential scan"),
                (&kernels.seq_fwd_ckpt, "checkpointing scan"),
                (&kernels.gated_norm_fwd, "gated norm"),
                (&kernels.bwd.gated_norm, "gated norm backward"),
                (&kernels.bwd.seq_bwd, "sequential backward"),
            ] {
                for f in [&k.f32, &k.bf16, &k.f16] {
                    budget = budget.min(opt_in_dynamic_smem(f, optin, name)?);
                }
            }
            budget
        };
        Ok(kernels)
    }
}

/// The sequential backward's shared memory for a head: the replay state
/// rows and the state-gradient rows (each padded by one), B and C, the step
/// scalars, x and dy of the token, one reduction slot per thread.
pub fn seq_bwd_smem_bytes(hd: usize, ds: usize, threads: usize) -> usize {
    (2 * hd * (ds + 1) + 2 * ds + 4 + 2 * hd + threads) * 4
}
