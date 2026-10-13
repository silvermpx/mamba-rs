//! The models of every family, wrapped for use.
//!
//! - CPU backbones: [`MambaBackbone`] (Mamba-1, which also trains on the
//!   CPU), [`Mamba2Backbone`], [`Mamba3Backbone`] (Mamba-3 SISO) and
//!   [`Mamba3MimoBackbone`].
//! - `lm` (feature `hf`): `MambaLM`, text generation on the CPU for every
//!   family, loaded from a checkpoint directory or wrapped around a
//!   backbone.
//! - `gpu_backbone` (feature `cuda`): the `GpuBackbone` trait every family's
//!   GPU backbone implements, and `AnyGpuBackbone`, one chosen at run time.
//! - `gpu_trainer` (feature `cuda`): the `GpuTrainer` trait every family's
//!   GPU trainer implements, and `AnyGpuTrainer`, one chosen at run time.
//! - `gpu_lm` (features `cuda` and `hf`): `MambaGpuLM`, text generation on
//!   the GPU for every family. `gpu_lm3` keeps the Mamba-3 language model
//!   of 0.7.
//! - `sample`: token sampling (temperature, top-k, top-p, min-p,
//!   repetition penalty).

mod backbone;
pub mod backbone2;
// backbone3 and sample have no HF dependency: a default-feature build
// gets the CPU backbones and the sampling utilities.
pub mod backbone3;
pub mod backbone3_mimo;
#[cfg(feature = "cuda")]
pub mod gpu_backbone;
#[cfg(all(feature = "hf", feature = "cuda"))]
pub mod gpu_lm;
#[cfg(all(feature = "hf", feature = "cuda"))]
pub mod gpu_lm3;
#[cfg(feature = "cuda")]
pub mod gpu_trainer;
#[cfg(feature = "hf")]
pub mod lm;
pub mod sample;

pub use backbone::MambaBackbone;
pub use backbone2::Mamba2Backbone;
pub use backbone3::Mamba3Backbone;
pub use backbone3_mimo::Mamba3MimoBackbone;
#[cfg(feature = "cuda")]
pub use gpu_backbone::{AnyGpuBackbone, GpuBackbone};
#[cfg(all(feature = "hf", feature = "cuda"))]
pub use gpu_lm::MambaGpuLM;
#[cfg(feature = "cuda")]
pub use gpu_trainer::{AnyGpuTrainer, GpuTrainer};
#[cfg(feature = "hf")]
pub use lm::MambaLM;
