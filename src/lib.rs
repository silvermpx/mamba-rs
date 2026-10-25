//! # mamba-rs
//!
//! A complete implementation of Mamba in Rust, on the CPU and on NVIDIA
//! GPUs: all three generations of the architecture, Mamba (Gu and Dao,
//! 2023), Mamba-2 (Dao and Gu, 2024) and Mamba-3 (Lahoti et al., 2026), the
//! last in both of its forms, SISO and MIMO. Every model runs inference and
//! training on both. The API calls each of the four a family
//! (`ModelFamily`).
//!
//! There is no Python, no C++ build step and no framework underneath: the
//! kernels compile at run time through NVRTC, and the GPU path links only
//! the CUDA driver API and cuBLAS.
//!
//! ## Capabilities
//!
//! - CPU and GPU paths for every family, held to each other by parity
//!   tests on shared weights
//! - Training with the full backward through the recurrent state, AdamW,
//!   global and control-channel norm clipping, the skip of a spiking
//!   window, gradient accumulation and data-parallel training over NCCL
//! - `WeightDtype::{F32, Tf32, Bf16, F16}` with f32 master weights and f32
//!   accumulation. The storage precision decides the product precision
//! - CUDA graph capture of the decode step, of the prompt pass of one
//!   length and of the training step, in every family
//! - Deterministic f32, tf32, bf16 and f16 GEMM kernels for inference and
//!   training, the default since 0.7.0. A forward or an input gradient in
//!   f32, bf16 or f16 gives a row the same bits at any batch size, each
//!   layer cut by its shape alone; TF32 products keep the measured kernels,
//!   whose bits hold within one dispatch bucket
//! - Safetensors checkpoints for every family, loaders for the official
//!   HuggingFace and `mamba_ssm` checkpoints, and a trainer-state file for
//!   a bit-continuous resume
//! - Segments to stack with layers of your own, such as attention, and
//!   packed sequences in training for Mamba-2 and Mamba-3
//!
//! ## One surface for every family
//!
//! A family's types share their shape: a CPU backbone (`MambaBackbone`,
//! `Mamba2Backbone`, `Mamba3Backbone`, `Mamba3MimoBackbone`), and with the
//! `cuda` feature a GPU backbone and a GPU trainer named after the family
//! (`Mamba2GpuBackbone`, `Mamba2GpuTrainer` and so on), each built from
//! one `GpuOptions` value for the device, the storage precision and the
//! GEMM mode. The `GpuBackbone` and `GpuTrainer` traits name what the
//! GPU types share, `AnyGpuBackbone` and `AnyGpuTrainer` hold a backbone
//! and a trainer of any family, and
//! `MambaLM` and `MambaGpuLM` (feature `hf`) load a checkpoint of any
//! family and generate text.
//!
//! ## GEMM modes (CUDA)
//!
//! Every GPU context carries a `GemmMode`:
//!
//! - `Deterministic` (default): the crate's own fixed-reduction-order
//!   kernels serve every GEMM that goes through the context, and cuBLAS is
//!   never called. Model contexts use the Inference family, trainers and
//!   plain contexts the Triad family.
//! - `CublasFast`: cuBLAS with TF32 permitted for f32 operands and f32
//!   accumulation for half operands.
//! - `CublasPedantic`: cuBLAS with pedantic f32 compute, the default of
//!   0.6.9 and earlier.
//!
//! The mode comes from `GpuOptions::mode`, or from the `MAMBA_RS_GEMM_MODE`
//! environment variable when the options leave it unset.
//! `GpuCtx::set_gemm_mode` changes it on a live context and is refused
//! while a graph is being captured. Inside the deterministic mode the
//! kernels are chosen from the mode, the storage precision and whether the
//! context serves a model or a trainer. The guide is
//! <https://github.com/silvermpx/mamba-rs/blob/main/docs/gemm-modes.md> and
//! the measurements are in
//! <https://github.com/silvermpx/mamba-rs/blob/main/docs/determinism-benchmarks.md>.
//!
//! ## Modules
//!
//! - [`mamba1`], [`mamba2`], [`mamba3_siso`], [`mamba3_mimo`]: the four
//!   families, each with its configuration, weights, state, CPU and GPU
//!   paths and serializer
//! - `gpu` (feature `cuda`): the GPU engine the families share, with the
//!   context, buffers, GEMMs, AdamW, the loss scaler and graph capture
//! - [`module`]: the backbones, the traits over them and the language
//!   models
//! - [`ops`]: shared dimensions, BLAS, norms and fast-math helpers
//! - [`dist`]: deterministic data-parallel training, one fixed-order
//!   reduction per optimizer step
//! - [`config`]: what the families share, `ScanMode`, `BlockNorm` and
//!   `ModelFamily`
//!
//! ## References
//!
//! - Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
//!   Spaces*, arXiv:2312.00752, 2023.
//! - Dao and Gu, *Transformers are SSMs: Generalized Models and Efficient
//!   Algorithms Through Structured State Space Duality*, ICML 2024.
//! - Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space
//!   Principles*, ICLR 2026.

/// Crate version, for consumers stamping numeric-route provenance
/// (checkpoint sidecars, serve fingerprints) without parsing Cargo.lock.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub(crate) mod bidi;
pub mod config;
pub mod dist;
#[doc(hidden)]
pub mod gemm_fold;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "hf")]
pub mod hf;
pub mod mamba1;
pub mod mamba2;
pub mod mamba3_mimo;
pub mod mamba3_siso;
pub mod matrix;
pub mod mlp;
pub mod module;
pub mod ops;
pub mod optim;
pub mod precision;
pub mod varlen;

pub use config::{BlockNorm, ModelFamily};

// Every family exports the same set here: its configuration, weights,
// state, CPU backbone with the scratch it steps in, and CPU trainer, and with
// `cuda` its GPU backbone and GPU trainer. The low-level step functions stay
// in the family's own module.
pub use mamba1::{MambaConfig, MambaLayerState, MambaLayerWeights, MambaState, MambaWeights};
pub use mamba2::{Mamba2Config, Mamba2LayerState, Mamba2LayerWeights, Mamba2State, Mamba2Weights};
pub use mamba3_mimo::{
    Mamba3MimoConfig, Mamba3MimoLayerState, Mamba3MimoLayerWeights, Mamba3MimoState,
    Mamba3MimoWeights,
};
pub use mamba3_siso::{
    Mamba3Config, Mamba3LayerState, Mamba3LayerWeights, Mamba3State, Mamba3Weights,
};

// The CPU backbones and the CPU language model of every family, and the
// sampling settings both language models take.
#[cfg(feature = "hf")]
pub use module::MambaLM;
pub use module::sample::SampleParams;
pub use module::{
    Mamba2Backbone, Mamba2Scratch, Mamba3Backbone, Mamba3MimoBackbone, Mamba3MimoScratch,
    Mamba3Scratch, MambaBackbone, MambaScratch,
};
// What training needs on either device: the session, options and metrics
// of a trainer, the AdamW state, the loss scaler and the trainer-state file.
pub use optim::{
    AdamWStateBlob, BackwardMetrics, BackwardOpts, DynamicLossScaler, StepMetrics, TrainSessionCfg,
    TrainerState,
};
pub use precision::{CpuOptions, GemmMode, WeightDtype};
pub use varlen::Varlen;
// The host trainers of every family and the trait over them.
pub use mamba1::cpu::trainer::MambaCpuTrainer;
pub use mamba2::cpu::trainer::Mamba2CpuTrainer;
pub use mamba3_mimo::cpu::trainer::Mamba3MimoCpuTrainer;
pub use mamba3_siso::cpu::trainer::Mamba3CpuTrainer;
pub use module::{AnyCpuTrainer, CpuTrainer};

// The GPU side of every family: the options every constructor takes, the
// backbones and the trainers with the traits over them, and the language
// model.
#[cfg(feature = "cuda")]
pub use gpu::GpuOptions;
#[cfg(feature = "cuda")]
pub use mamba1::gpu::{MambaGpuBackbone, MambaGpuTrainer};
#[cfg(feature = "cuda")]
pub use mamba2::gpu::{Mamba2GpuBackbone, Mamba2GpuTrainer};
#[cfg(feature = "cuda")]
pub use mamba3_mimo::gpu::{Mamba3MimoGpuBackbone, Mamba3MimoGpuTrainer};
#[cfg(feature = "cuda")]
pub use mamba3_siso::gpu::{Mamba3GpuBackbone, Mamba3GpuTrainer};
#[cfg(all(feature = "cuda", feature = "hf"))]
pub use module::MambaGpuLM;
#[cfg(feature = "cuda")]
pub use module::{AnyGpuBackbone, AnyGpuTrainer, GpuBackbone, GpuTrainer};
