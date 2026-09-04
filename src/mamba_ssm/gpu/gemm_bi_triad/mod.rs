//! Deterministic batch-invariant GEMM dispatcher — the TRIAD family
//! (`BiGemmFamily::Triad`).
//!
//! Covers f32, bf16 and f16, on CUDA cores and on Tensor Cores. Physical
//! exports and Triad-specific identifiers use the `gemm_bi` family name.
//!
//! A warptiled GEMM family of our own. Three entry points used when
//! `ctx.batch_invariant()` is enabled:
//!
//!   - [`gemm_bi_forward`]      NN: `Y = X @ W + bias`
//!   - [`gemm_bi_backward_dw`]  TN: `dW += X^T @ dY` (accumulated)
//!   - [`gemm_bi_backward_dx`]  NT: `dX = dY @ W^T`
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
//!   - exact scalar f32 routes keep full f32 operands and accumulation;
//!     deterministic TF32 routes use TF32 operand mantissas with f32
//!     accumulation and a fixed route/order, so they are not bit-equal to
//!     exact scalar f32.
//!
//! Public routes are expected to cover every admitted shape with a custom
//! kernel. Typed or tensor-core routes return `Err` outside their contracts;
//! an uncovered f32 shape panics because it violates the zero-cuBLAS invariant.

mod contract;
mod dispatch;
mod launch;
pub(crate) mod modules;
mod qualification;
#[cfg(test)]
mod training_pipeline_tests;

pub(super) use contract::GemmDims;
pub use contract::{
    CUptr, F32_TF32_TUNING_REVISION, F32TriadAvailability, F32TriadOperands, F32TriadRequest,
    F32TriadSelection, F32TriadShape, GemmBiFwdSubOperands, SCALAR_TN_M16N16_DYNAMIC_SHARED_BYTES,
    SCALAR_TN_M16N16_MIN_ACTIVE_BLOCKS, SCALAR_TN_M16N16_REGISTER_CAP,
    SCALAR_TN_M16N16_STATIC_SHARED_BYTES, SCALAR_TN_M16N16_THREADS,
    SCALAR_TRANSPOSE_SCRATCH_CAP_ELEMENTS, SM80_TF32_ROUTE_SPECS, SM80_TF32_WIDE_ROUTE_SPECS,
    SM90A_DYNAMIC_SHARED_BYTES, SM90A_STAGES, SM90A_TF32_ROUTE_SPECS, SM90A_TILE,
    SM100_KERNEL_SPECS, SM100_TENSOR_MAP_REVISION, SM100_TF32_ROUTE_SPECS, SM100_TUNING_REVISION,
    SM120_KERNEL_SPECS, SM120_SCHEDULE_REVISION, SM120_STREAMK_KERNEL_SPECS,
    SM120_TENSOR_MAP_REVISION, SM120_TF32_ROUTE_SPECS, SM120_TUNING_REVISION, Sm90aForcedRoute,
    Sm90aLaunchOperands, Sm90aMapRequest, Sm90aNumericContract, Sm90aOp, Sm90aPreparedTensorMaps,
    Sm90aRouteIdentity, Sm90aShape, Sm90aTensorMap, Sm90aWarpgroupSchedule, Sm100ForcedRoute,
    Sm100KernelSpec, Sm100LaunchOperands, Sm100MapRequest, Sm100NumericContract, Sm100Op,
    Sm100PhysicalRoute, Sm100PreparedLaunch, Sm100PreparedTensorMaps, Sm100RouteIdentity,
    Sm100Schedule, Sm100Shape, Sm100Stages, Sm100TargetCandidate, Sm100TargetKind, Sm100TensorMap,
    Sm100Tile, Sm120Bk, Sm120ForcedRoute, Sm120KernelResources, Sm120KernelSpec,
    Sm120LaunchOperands, Sm120MapRequest, Sm120NumericContract, Sm120Op, Sm120PhysicalRoute,
    Sm120PreparedLaunch, Sm120PreparedTensorMaps, Sm120RouteIdentity, Sm120Schedule, Sm120Shape,
    Sm120Stages, Sm120TargetCandidate, Sm120TensorMap, Sm120Tile, TF32_SCHEDULE_REVISION,
    TF32_SPLITK_EXTENSION_SPECS, TF32_TENSOR_MAP_REVISION, TcFwdOperands, Tf32KernelSpec,
    Tf32PhysicalRoute, Tf32PortableRoute, Tf32PortableStages, Tf32PortableTile,
    Tf32QualifiedModule, Tf32Sm90aRoute, Tf32Sm100Route, Tf32Sm120Route, Tf32Sm120Stages,
    Tf32Sm120Tile, portable_extensions_composed_for_cc, sm120_kernel_specs,
    tf32_extension_route_specs, tf32_kernel_spec, tf32_module_symbols, tf32_route_specs,
    tf32_route_specs_all, validate_sm90a_map_request, validate_sm100_map_request,
    validate_sm120_map_request,
};
#[cfg(test)]
pub(crate) use contract::{ZERO_REDUCTION_DIGEST_DOMAIN, ZERO_REDUCTION_MAP_REVISION};
pub use dispatch::{
    SM90A_AUTO_CELLS, SM100_AUTO_CELLS_CC100, SM100_AUTO_CELLS_CC103, SM100_AUTO_CELLS_CC110,
    SM120_AUTO_CELLS_CC120, SM120_AUTO_CELLS_CC121, SM120_STREAMK_CELLS_CC120,
    SM120_STREAMK_CELLS_CC121, Sm90aAutoRequest, Sm100AutoRequest, TcTile, resolve_f32_triad_auto,
    resolve_sm90a_auto, resolve_sm90a_forced, resolve_sm100_auto, resolve_sm100_forced,
    resolve_sm120_forced, resolve_tf32_forced, sm100_auto_cells, sm100_target_candidates,
    sm120_target_candidates,
};
pub(in crate::mamba_ssm::gpu) use dispatch::{
    Sm120AutoRequest, tc_half_policy_prefers_scalar_forward,
};
pub use launch::*;
pub(crate) use launch::{
    F32PreparedLaunchCache, Sm90aPreparedLaunchCache, Sm100PreparedLaunchCache,
    Sm120PreparedLaunchCache, launch_cached_f32_backward_dw, launch_cached_f32_backward_dx,
    launch_cached_f32_forward,
};
#[doc(hidden)]
pub use modules::GemmBiKernels;
#[doc(hidden)]
pub use qualification::{
    PhysicalPaddedNnLayout, PhysicalQualificationF32Epilogue, PhysicalQualificationOffset,
    PhysicalQualificationRequest, PhysicalQualificationRoute, QualifiedGuardValidation,
    QualifiedPhysicalLaunch, QualifiedPhysicalLaunchEvidence, QualifiedPhysicalLaunchNode,
    Tf32QualificationConfig, Tf32QualificationOutput, Tf32QualificationSuite,
    presize_physical_qualification_suite, qualify_physical_launch, run_tf32_qualification,
};
