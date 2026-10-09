//! The Mamba-3 MIMO CUDA module: `kernels/gated_mlp.cu` and
//! `kernels/mamba3_mimo.cu` after the typed prelude, under its own module
//! kind. The norms and residual adds come from the Fixed module through the
//! context.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig};

use crate::mamba_ssm::gpu::composed::{ComposedModule, ComposedSpec, compile_composed};
use crate::mamba_ssm::gpu::kernel_identity as ki;
use crate::mamba_ssm::gpu::kernels::TypedKernel;

/// The compiled MIMO kernels.
pub struct Mamba3MimoKernels {
    /// B and C normalised per (rank, group) row.
    pub bc_norm: TypedKernel,
    /// The scan of every head over a window (decode is the window of one).
    pub scan: TypedKernel,
    /// The MLP's gate, `value * silu(gate)`.
    pub mlp_gate: TypedKernel,
    pub compiler_identity: ki::CompilerIdentity,
    pub artifact_identity: ki::ArtifactIdentity,
}

/// The per-block opt-in the scan may use: consumer parts cap it near
/// 99 KB.
pub const SCAN_SMEM_MAX: usize = 99 * 1024;

/// The scan's shared memory for a head: the state rows padded by one, the
/// previous keys, the keys and the queries, the angles' sines and cosines,
/// the step scalars.
pub fn scan_smem_bytes(hd: usize, ds: usize, rank: usize, na: usize) -> usize {
    (hd * (ds + 1) + 3 * rank * ds + 2 * na + 8) * 4
}

/// Launch geometry of the scan: one block per (batch, head), a whole
/// number of warps covering the head's channels.
pub fn scan_cfg(
    batch: usize,
    nh: usize,
    (hd, ds, rank, na): (usize, usize, usize, usize),
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (batch as u32, nh as u32, 1),
        block_dim: (hd.next_multiple_of(32) as u32, 1, 1),
        shared_mem_bytes: scan_smem_bytes(hd, ds, rank, na) as u32,
    }
}

impl Mamba3MimoKernels {
    /// Compile (or load from the cache) the MIMO module for `arch`.
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
                    include_str!("../../../kernels/gated_mlp.cu"),
                    include_str!("../../../kernels/mamba3_mimo.cu"),
                ],
                kind: ki::ModuleKind::Mamba3MimoCombined,
                cache_prefix: "mamba3-mimo-kernels-v1-",
                seed: "2603155690",
                label: "Mamba-3 MIMO",
            },
        )?;
        let get = |name: &str| -> Result<CudaFunction, String> {
            module
                .load_function(name)
                .map_err(|e| format!("MIMO kernel '{name}' not found: {e:?}"))
        };
        let typed = |base: &str| -> Result<TypedKernel, String> {
            Ok(TypedKernel {
                f32: get(base)?,
                bf16: get(&format!("{base}_bf16"))?,
                f16: get(&format!("{base}_f16"))?,
            })
        };
        let kernels = Self {
            bc_norm: typed("mimo_bc_norm")?,
            scan: typed("mimo_scan")?,
            mlp_gate: typed("mlp_gate_fwd")?,
            compiler_identity,
            artifact_identity,
        };
        {
            use cudarc::driver::sys::CUfunction_attribute_enum as FnAttr;
            for f in [&kernels.scan.f32, &kernels.scan.bf16, &kernels.scan.f16] {
                // A refused opt-in leaves the default 48 KB; an oversized
                // launch then fails loudly with its own error.
                let _ = f.set_attribute(
                    FnAttr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SCAN_SMEM_MAX as i32,
                );
            }
        }
        Ok(kernels)
    }
}
