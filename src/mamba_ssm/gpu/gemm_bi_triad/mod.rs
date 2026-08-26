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
pub use contract::{
    CUptr, F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
    F32TriadSelection, F32TriadShape, SM80_TF32_ROUTE_SPECS, SM90A_DYNAMIC_SHARED_BYTES,
    SM90A_STAGES, SM90A_TF32_ROUTE_SPECS, SM90A_TILE, SM100_KERNEL_SPECS,
    SM100_TENSOR_MAP_REVISION, SM100_TF32_ROUTE_SPECS, SM100_TUNING_REVISION, SM120_KERNEL_SPECS,
    SM120_SCHEDULE_REVISION, SM120_TENSOR_MAP_REVISION, SM120_TF32_ROUTE_SPECS,
    SM120_TUNING_REVISION, SgemmFwdSubOperands, Sm90aForcedRoute, Sm90aLaunchOperands,
    Sm90aMapRequest, Sm90aNumericContract, Sm90aOp, Sm90aPreparedTensorMaps, Sm90aRouteIdentity,
    Sm90aShape, Sm90aTensorMap, Sm90aWarpgroupSchedule, Sm100ForcedRoute, Sm100KernelParams,
    Sm100KernelSpec, Sm100LaunchOperands, Sm100MapRequest, Sm100NumericContract, Sm100Op,
    Sm100PhysicalRoute, Sm100PreparedLaunch, Sm100PreparedTensorMaps, Sm100RouteIdentity,
    Sm100Schedule, Sm100Shape, Sm100Stages, Sm100TargetCandidate, Sm100TargetKind, Sm100TensorMap,
    Sm100Tile, Sm120Bk, Sm120ForcedRoute, Sm120KernelParams, Sm120KernelResources, Sm120KernelSpec,
    Sm120LaunchOperands, Sm120MapRequest, Sm120NumericContract, Sm120Op, Sm120PhysicalRoute,
    Sm120PreparedLaunch, Sm120PreparedTensorMaps, Sm120RouteIdentity, Sm120Shape, Sm120Stages,
    Sm120TargetCandidate, Sm120TensorMap, Sm120Tile, TF32_SCHEDULE_REVISION,
    TF32_TENSOR_MAP_REVISION, TcFwdOperands, Tf32KernelSpec, Tf32PhysicalRoute, Tf32PortableRoute,
    Tf32PortableStages, Tf32PortableTile, Tf32QualifiedModule, Tf32Sm90aRoute, Tf32Sm100Route,
    Tf32Sm120Route, Tf32Sm120Tile, tf32_kernel_spec, tf32_module_symbols, tf32_route_specs,
    validate_sm90a_map_request, validate_sm100_map_request, validate_sm120_map_request,
};
pub use dispatch::{
    SM90A_AUTO_CELLS, SM100_AUTO_CELLS_CC100, SM100_AUTO_CELLS_CC103, SM120_AUTO_CELLS_CC120,
    SM120_AUTO_CELLS_CC121, TC64_PREFER_MAX_TILES128, TcTile, resolve_f32_triad_auto,
    resolve_sm90a_forced, resolve_sm100_forced, resolve_sm120_forced, resolve_tf32_forced,
    sm100_target_candidates, sm120_target_candidates,
};
pub use launch::*;
#[doc(hidden)]
pub use modules::GemmBiKernels;
