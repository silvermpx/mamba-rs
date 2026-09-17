//! # mamba-rs
//!
//! Mamba SSM and Mamba-3 SISO in Rust with optional CUDA GPU acceleration.
//! Supports **Mamba SSM** (Gu & Dao, 2023) and **Mamba-3 SISO** (Lahoti
//! et al., 2026) on CPU and GPU, with full inference and training pipelines.
//!
//! No Python, no C++ build step and no framework dependency: the kernels
//! compile at run time through NVRTC, and the GPU path links only the CUDA
//! driver API and cuBLAS.
//!
//! ## Capabilities
//!
//! - Mamba SSM and Mamba-3 SISO architectures
//! - CPU and GPU (CUDA) paths for both
//! - Full training with BPTT through the recurrent SSM state + AdamW
//! - `WeightDtype::{F32, Tf32, Bf16, F16}` with f32 master state and
//!   accumulation; the storage precision decides the product precision
//! - CUDA Graph capture for inference and training steps
//! - Deterministic f32/tf32/bf16/f16 GEMM kernels for inference and training,
//!   the default since 0.7.0; the inference kernels are batch-invariant,
//!   the training kernels within one dispatch bucket
//! - HuggingFace safetensors loader for Mamba SSM checkpoints
//!
//! ## GEMM modes (CUDA)
//!
//! Every GPU context carries a `GemmMode` (`mamba_ssm::gpu::GemmMode`, behind
//! the `cuda` feature):
//!
//! - `Deterministic` (default): the crate's own fixed-reduction-order kernels
//!   serve every GEMM that goes through the context; cuBLAS is never called
//!   in this mode. Model contexts use the Inference family, trainers and
//!   plain contexts the Triad family.
//! - `CublasFast`: cuBLAS with TF32 permitted for f32 operands and f32
//!   accumulation for half operands.
//! - `CublasPedantic`: cuBLAS with pedantic f32 compute, the default of
//!   0.6.9 and earlier.
//!
//! Select the mode at construction (`GpuCtx::new_with_mode`, the model and
//! trainer `*_with_mode` constructors, or `MAMBA_RS_GEMM_MODE` for the
//! environment-reading constructors) or change it with
//! `GpuCtx::set_gemm_mode`, which is refused while a graph is being captured.
//! Storage precision (`WeightDtype`) and mode are the two settings; inside
//! the deterministic mode the kernels are chosen from them and from whether
//! the context serves a model or a trainer. The guide is
//! <https://github.com/silvermpx/mamba-rs/blob/main/docs/gemm-modes.md> and
//! the measurements are in
//! <https://github.com/silvermpx/mamba-rs/blob/main/docs/determinism-benchmarks.md>.
//!
//! ## Module Structure
//!
//! - [`mamba_ssm`] — Mamba SSM (CPU + GPU forward, backward, training)
//! - [`mamba3_siso`] — Mamba-3 SISO (CPU + GPU forward, backward, training)
//! - [`module`] — high-level backbone and LM wrappers, HF integration
//! - [`ops`] — shared dimensions, BLAS, norms, fast-math helpers
//! - [`dist`] — deterministic data-parallel training (one fixed-order
//!   reduction per optimizer step)
//! - [`config`], [`state`], [`weights`], [`serialize`] — Mamba SSM data types
//!
//! ## References
//!
//! - Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
//!   Spaces*, arXiv:2312.00752, 2023.
//! - Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space
//!   Principles*, ICLR 2026.

/// Crate version, for consumers stamping numeric-route provenance
/// (checkpoint sidecars, serve fingerprints) without parsing Cargo.lock.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod dist;
#[cfg(feature = "hf")]
pub mod hf;
pub mod mamba3_siso;
pub mod mamba_ssm;
pub mod module;
pub mod ops;
pub mod serialize;
pub mod state;
pub mod weights;

// Convenience re-export aliases for the Mamba SSM CPU + GPU paths.
// The canonical module paths are `mamba_ssm::cpu::*` / `mamba_ssm::gpu::*`;
// these aliases keep `mamba_rs::inference` / `train` / `gpu` short for the
// most common entrypoints.
pub mod inference {
    pub use crate::mamba_ssm::cpu::inference::*;
    pub use crate::mamba_ssm::cpu::prefill::*;
}
pub mod train {
    pub use crate::mamba_ssm::cpu::backward;
    pub use crate::mamba_ssm::cpu::backward_ops;
    pub use crate::mamba_ssm::cpu::flat;
    pub use crate::mamba_ssm::cpu::forward;
    pub use crate::mamba_ssm::cpu::parallel;
    pub use crate::mamba_ssm::cpu::scratch;
    pub use crate::mamba_ssm::cpu::target;
    pub use crate::mamba_ssm::cpu::weights;

    // Re-export shared ops that were previously in train/
    pub use crate::ops::blas;
    pub use crate::ops::fast_math;
}

#[cfg(feature = "cuda")]
pub mod gpu {
    pub use crate::mamba_ssm::gpu::*;
}

#[cfg(feature = "cuda")]
pub mod gpu3 {
    //! Convenience re-exports for the Mamba-3 SISO GPU path.
    pub use crate::mamba3_siso::gpu::*;
}

pub use config::MambaConfig;
pub use mamba_ssm::cpu::inference::{
    MambaLayerScratch, MambaStepScratch, mamba_block_step, mamba_layer_step, mamba_step,
    mamba_step_no_proj,
};
pub use module::MambaBackbone;
pub use state::{MambaLayerState, MambaState};
pub use weights::{MambaLayerWeights, MambaWeights};

// Mamba-3 SISO re-exports
pub use mamba3_siso::{
    Mamba3Config, Mamba3Dims, Mamba3LayerState, Mamba3LayerWeights, Mamba3State, Mamba3StepScratch,
    Mamba3Weights,
};

/// Convenience re-export of the storage-dtype selector used by the
/// mixed-precision GPU API (`GpuMambaBackbone::new_with_dtype`,
/// `GpuMamba3Backbone::new_with_dtype`, `GpuMambaLM::from_hf_with_dtype`).
#[cfg(feature = "cuda")]
pub use mamba_ssm::gpu::dtype::WeightDtype;
