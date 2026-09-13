//! CUDA GPU backend for Mamba SSM.
//!
//! Every context carries a [`GemmMode`] (re-exported here), `Deterministic`
//! by default: a model context then multiplies on the Inference kernels, a
//! trainer or a plain context on the Triad kernels; the two cuBLAS modes are
//! explicit alternatives.
//!
//! - `context` — the GPU context: stream, kernels, cuBLAS handle and the
//!   GEMM route (mode, family, numeric policies)
//! - `device` — CUDA device and cuBLAS handle
//! - `buffers` — GPU memory management
//! - `kernels`, `kernel_identity` — NVRTC compilation and the frozen
//!   identity of every compiled kernel
//! - `blas` — the GEMM dispatch boundary: the deterministic families in the
//!   default mode, cuBLAS in the two vendor modes
//! - `gemm_bi_inference`, `gemm_bi_triad` — the two deterministic families
//! - `inference` — the decode step with CUDA Graph capture
//! - `prefill` — the GPU prompt prefill
//! - `forward`, `backward`, `forward_mixed`, `backward_mixed` — the f32 and
//!   the mixed-precision training passes
//! - `trainer`, `training_graph` — the training step and its captured graph
//! - `adamw`, `grad_clip`, `loss_scaler` — the optimizer, clipping and the
//!   f16 loss scaler
//! - `weights`, `weights_mixed_train`, `dtype` — device weights and the
//!   storage dtype
//! - `graph_capture`, `launch` — capture and launch helpers

pub mod adamw;
pub mod backward;
pub mod backward_mixed;
pub mod blas;
pub mod buffers;
pub mod context;
pub mod device;
pub(crate) mod diagnostics;
pub mod dtype;
mod fold_transport;
pub mod forward;
pub mod forward_mixed;
pub mod gemm_bi_inference;
pub mod gemm_bi_triad;
mod gemm_mode;
pub mod grad_clip;
pub mod graph_capture;
pub mod inference;
pub mod kernel_identity;
pub mod kernels;
pub mod launch;
pub mod loss_scaler;
pub mod prefill;
pub mod trainer;
pub mod training_graph;
pub mod weights;
pub mod weights_mixed_train;

pub use dtype::WeightDtype;
pub use gemm_mode::GemmMode;
