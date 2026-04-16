//! # mamba-rs
//!
//! Mamba SSM and Mamba-3 SISO in Rust with optional CUDA GPU acceleration.
//! Supports **Mamba SSM** (Gu & Dao, 2023) and **Mamba-3 SISO** (Lahoti
//! et al., 2026) on CPU and GPU, with full inference and training pipelines.
//!
//! Standalone — no PyTorch, no Triton, no Burn, no Candle. Kernels compile
//! at runtime via NVRTC.
//!
//! ## Capabilities
//!
//! - Mamba SSM and Mamba-3 SISO architectures
//! - CPU and GPU (CUDA) paths for both
//! - Full training with BPTT through the recurrent SSM state + AdamW
//! - `WeightDtype::{F32, Bf16, F16}` — f32 compute regardless of storage
//! - CUDA Graph capture for inference and training steps
//! - Batch-invariant bf16 inference (custom GEMM kernel; logits are
//!   bit-identical across batch sizes for the same prompt)
//! - HuggingFace safetensors loader for Mamba SSM checkpoints
//!
//! ## Module Structure
//!
//! - [`mamba_ssm`] — Mamba SSM (CPU + GPU forward, backward, training)
//! - [`mamba3_siso`] — Mamba-3 SISO (CPU + GPU forward, backward, training)
//! - [`module`] — high-level backbone and LM wrappers, HF integration
//! - [`ops`] — shared dimensions, BLAS, norms, fast-math helpers
//! - [`config`], [`state`], [`weights`], [`serialize`] — Mamba SSM data types
//!
//! ## References
//!
//! - Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
//!   Spaces*, ICLR 2024.
//! - Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space
//!   Principles*, ICLR 2026.

pub mod config;
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
