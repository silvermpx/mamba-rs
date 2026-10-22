//! The GPU engine every family runs on.
//!
//! Every context carries a [`GemmMode`] (re-exported here), `Deterministic`
//! by default: a model context then multiplies on the Inference kernels, a
//! trainer or a plain context on the Triad kernels. The two cuBLAS modes are
//! explicit alternatives.
//!
//! - `context`: the GPU context with its stream, kernels, cuBLAS handle
//!   and GEMM route (mode, storage precision and the context's role)
//! - `device`: CUDA device and cuBLAS handle
//! - `buffers`: GPU memory management
//! - `kernels`, `kernel_identity`: NVRTC compilation and the frozen
//!   identity of every compiled kernel; `ptx_rounding` gives every
//!   floating-point add, subtract and multiply its rounding before the
//!   driver assembles a module
//! - `blas`: the GEMM dispatch boundary, the deterministic families in the
//!   default mode and cuBLAS in the two vendor modes
//! - `gemm_bi_inference`, `gemm_bi_triad`: the two deterministic families
//! - `adamw`, `grad_clip`, `loss_scaler`: the optimizer, clipping and the
//!   f16 loss scaler
//! - `train`: the session, options and metrics every trainer shares
//! - `dtype`, `options`: the storage precision and the options every
//!   constructor takes
//! - `graph_capture`, `launch`: capture and launch helpers
//! - `block_norm`: the block norms (RMSNorm or LayerNorm) every family
//!   launches

pub mod adamw;
pub(crate) mod bidi;
pub mod blas;
pub(crate) mod block_norm;
pub mod buffers;
pub(crate) mod composed;
pub mod context;
#[cfg(feature = "cublas")]
mod cublas_pool;
pub mod device;
pub(crate) mod diagnostics;
pub mod dtype;
mod fold_transport;
pub mod gemm_bi_inference;
pub mod gemm_bi_triad;
mod gemm_mode;
pub mod grad_clip;
pub(crate) mod grad_window;
pub mod graph_capture;
pub mod kernel_identity;
pub mod kernels;
pub mod launch;
pub mod loss_scaler;
pub mod options;
pub(crate) mod ptx_rounding;
pub mod train;
pub(crate) mod varlen;
pub use crate::optim::train_state;

pub use dtype::WeightDtype;
pub use gemm_mode::GemmMode;
pub use options::GpuOptions;
pub use train::{BackwardMetrics, BackwardOpts, StepMetrics, TrainSessionCfg};
pub use train_state::TrainerState;
