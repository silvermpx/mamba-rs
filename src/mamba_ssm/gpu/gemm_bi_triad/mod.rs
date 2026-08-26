//! Deterministic batch-invariant GEMM dispatcher — the TRIAD family
//! (`BiGemmFamily::Triad`).
//!
//! Covers f32, bf16 and f16, on CUDA cores and on Tensor Cores; the
//! `sgemm` in the module and kernel names is historical BLAS notation
//! (S = single precision) that no longer describes the coverage.
//!
//! Ported from SQV-RS `sqv_uaac` (`blas_gpu.rs` + `kernels/gemm_bi_triad/`,
//! siboehm warptiling lineage). Three entry points used when
//! `ctx.batch_invariant()` is enabled:
//!
//!   - [`sgemm_bi_forward`]      NN: `Y = X @ W + bias`
//!   - [`sgemm_bi_backward_dw`]  TN: `dW += X^T @ dY` (accumulated)
//!   - [`sgemm_bi_backward_dx`]  NT: `dX = dY @ W^T`
//!
//! Dtypes: the f32 entry points are the base contract; the typed
//! (bf16/f16) entry points further down route homogeneous typed operand
//! triples through the typed kernel variants — typed I/O, f32
//! accumulation, dW/bias always f32 — bit-identical to upcasting the
//! inputs and running the f32 kernels. The `*_tc` entry points are a
//! separate numeric contract (mma.sync accumulation): deterministic and
//! batch-invariant, not bit-equal to the scalar variants.
//!
//! Every shape routes through a fixed-tile custom kernel (Big / Slim /
//! narrow / GEMV / split-K with deterministic tree reduce) — never cuBLAS.
//! Guarantees, in decreasing strength:
//!   - bit-identical across RUNS for a fixed shape (always);
//!   - bit-identical across BATCH SIZES that route to the same dispatch
//!     bucket (the per-cell K order is fixed within a bucket; crossing a
//!     bucket boundary — e.g. ultra-thin M<32 vs split-K M>=32 — changes
//!     the reduction association deterministically);
//!   - full f32 accumulation precision (no TF32 mantissa truncation).
//!
//! Public routes are expected to cover every admitted shape with a custom
//! kernel. Typed or tensor-core routes return `Err` outside their contracts;
//! an uncovered f32 shape panics because it violates the zero-cuBLAS invariant.

mod contract;
mod dispatch;
mod launch;
pub(crate) mod modules;

pub(super) use contract::GemmDims;
pub use contract::{SgemmFwdSubOperands, TcFwdOperands};
pub use dispatch::{TC64_PREFER_MAX_TILES128, TcTile};
pub use launch::*;
#[doc(hidden)]
pub use modules::GemmBiKernels;
