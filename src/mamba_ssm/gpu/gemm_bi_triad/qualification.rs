//! Physical launch qualification for deterministic Triad performance cells.
//!
//! Build the complete request list, pre-size its scratch envelope, then qualify
//! and measure one request at a time on a dedicated context. A live holder owns
//! its exact graph and resources and excludes unrelated work, policy mutation,
//! or a second holder on that context.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys;
use sha2::{Digest as _, Sha256};

use super::contract::{
    Tf32PortableTile, portable_extensions_composed_for_cc, tf32_route_specs_for, tf32_splitk_spec,
};
use super::{
    F32TriadOperands, F32TriadRequest, F32TriadShape, HalfNativeBranchSeal, PreparedF32TriadLaunch,
    SM90A_STAGES, SM90A_TF32_ROUTE_SPECS, SM90A_TILE, SM100_TF32_ROUTE_SPECS,
    SM120_TF32_ROUTE_SPECS, Sm90aAutoBranchSeal, Sm100AutoBranchSeal, Sm120AutoBranchSeal,
    TcFwdOperands, TcTile, Tf32KernelSpec, Tf32PhysicalRoute, enqueue_validated_prepared_f32_triad,
    gemm_bi_backward_dw_tc_with_tile, gemm_bi_backward_dx_tc_with_tile,
    gemm_bi_forward_tc_with_tile, gemm_bi_forward_tc_with_tile_shape, launch_prepared_f32_triad,
    prepare_f32_triad_forced, tf32_kernel_spec, validate_prepared_f32_triad_for_timing,
    with_cached_f32_triad_prepared,
};
use crate::mamba_ssm::gpu::{
    blas::{
        F32PhysicalGraphPackageRequest, HalfPhysicalTraceRequest, HalfPolicyBranchSeal,
        PreparedPhysicalGraphPackage, TypedPtr, gemm_bi_backward_dw_typed,
        gemm_bi_backward_dx_typed, gemm_bi_forward_typed, gpu_gemm_bi_backward_dw_grad,
        gpu_gemm_bi_backward_dx_raw, gpu_gemm_bi_forward_raw, launch_half_production_branch,
        prepare_f32_physical_graph_package, prepare_half_physical_graph_package,
        record_half_physical_trace, record_prepared_f32_physical_trace,
    },
    buffers::{DtypedBuf, GpuBuffer, GradSlice},
    context::{BiGemmFamily, F32TriadPolicy, GpuCtx, HalfTriadPolicy},
    dtype::WeightDtype,
    graph_capture::{
        CapturedPhysicalGraph, capture_into_graph_with_gemm_plan,
        capture_into_graph_with_physical_plan,
    },
    kernel_identity::{
        FramedSha256, GemmRouteIdentity, ModuleKind, PhysicalGemmBackend, PhysicalLaunchKind,
        PolicyDtype, RecordedPhysicalTrace, ResolvedGemmOp, ResolvedGemmRoute,
        ResolvedInstructionFamily, ResolvedKernelLaunch, ResolvedNumericContract,
        ResolvedOperandConversion, ResolvedPhysicalKernelLaunch, digest_hex,
    },
};

const K_CASES: [usize; 16] = [0, 1, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 65, 97, 129, 257];
const STAGED_K_CASES: [usize; 4] = [65, 97, 129, 257];
const TAIL_CASES: [usize; 16] = [
    1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
];
const GUARDED_EXTENT: usize = 17;
const GUARD_ELEMENTS: usize = 32;
const QUALIFICATION_GUARD_ELEMENTS: usize = 32;
const QUALIFICATION_GUARD_VALUE: f32 = 7.0;
const PHYSICAL_PREFLIGHT_CAPACITY: usize = 64;
const PHYSICAL_EVIDENCE_SCOPE: &str = "eager_preflight_same_launcher";

fn active_qualification_contexts() -> &'static Mutex<BTreeSet<u64>> {
    static ACTIVE: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();
    ACTIVE.get_or_init(|| Mutex::new(BTreeSet::new()))
}

struct ActiveQualificationToken {
    context_token: u64,
    active: bool,
}

impl ActiveQualificationToken {
    fn acquire(ctx: &GpuCtx) -> Result<Self, String> {
        let context_token = ctx.instance_token();
        let mut active = active_qualification_contexts().lock().map_err(|_| {
            "physical qualification active-context registry is poisoned".to_string()
        })?;
        if !active.insert(context_token) {
            return Err(
                "GPU context already has a live physical qualification holder; use a dedicated context"
                    .into(),
            );
        }
        Ok(Self {
            context_token,
            active: true,
        })
    }

    fn release(&mut self) {
        if !self.active {
            return;
        }
        if let Ok(mut active) = active_qualification_contexts().lock() {
            active.remove(&self.context_token);
        }
        self.active = false;
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Production route qualified by the physical-launch harness.
pub enum PhysicalQualificationRoute {
    /// Production F32 policy dispatcher with the requested permission policy.
    ///
    /// `AllowDeterministicTf32V1` permits a deterministic TF32 route but may
    /// resolve to the exact scalar FMA path for an unsupported shape or device.
    F32Policy(F32TriadPolicy),
    /// Production BF16/F16 policy dispatcher.
    HalfPolicy {
        /// Logical storage dtype.
        dtype: WeightDtype,
        /// Whether the production policy enables Tensor Cores.
        tensor_cores: bool,
        /// Numeric permission installed for this AUTO request. Stream-K
        /// permission may resolve to a tiled or non-TC production fallback;
        /// it is not a forced tile or schedule.
        half_policy: HalfTriadPolicy,
    },
    /// Direct deterministic BF16/F16 Tensor Core route.
    ///
    /// `Thin16` is valid only for NN requests. Square tiles support NN, TN,
    /// and NT according to the selected module's route contract.
    HalfForced {
        /// Logical storage dtype.
        dtype: WeightDtype,
        /// Exact forced tile.
        tile: TcTile,
    },
    /// Prepared deterministic F32 Tensor Core route.
    Tf32Forced(Tf32PhysicalRoute),
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Named one-element offset used by the deterministic edge inventory.
pub enum PhysicalQualificationOffset {
    /// Shift the logical output origin by one element.
    Output,
    /// Shift the logical A operand origin by one element.
    A,
    /// Shift the logical B operand origin by one element.
    B,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Frozen direct-NN padded layout used by the deterministic edge inventory.
pub enum PhysicalPaddedNnLayout {
    /// `(lda, ldb, ldc) = (72, 136, 130)`.
    Lda72Ldb136Ldc130,
    /// `(lda, ldb, ldc) = (72, 136, 131)`.
    Lda72Ldb136Ldc131,
}

impl PhysicalPaddedNnLayout {
    const fn strides(self) -> (usize, usize, usize) {
        match self {
            Self::Lda72Ldb136Ldc130 => (72, 136, 130),
            Self::Lda72Ldb136Ldc131 => (72, 136, 131),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhysicalQualificationLayout {
    Contiguous,
    OneElementOffset(PhysicalQualificationOffset),
    DirectPaddedNn(PhysicalPaddedNnLayout),
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Immutable logical request for one physical qualification cell.
///
/// Dimensions are logical `(M, K, N)` training dimensions. NN computes
/// `C[M,N] = A[M,K] * B[K,N]`; TN computes
/// `C[K,N] = A[M,K]^T * B[M,N]`; NT computes
/// `C[M,K] = A[M,N] * B[K,N]^T`. `M` and `N` must be positive; `K = 0`
/// is a supported zero-extent logical dimension. Layouts may be contiguous,
/// offset subviews, or one of the explicitly padded NN layouts.
///
/// The facade owns every allocation used by the request. No device pointer,
/// capture capacity, policy guard, or prepared resource crosses this API.
pub struct PhysicalQualificationRequest {
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    route: PhysicalQualificationRoute,
    layout: PhysicalQualificationLayout,
    f32_epilogue: Option<PhysicalQualificationF32Epilogue>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
/// F32 epilogue retained by the physical qualification facade.
pub struct PhysicalQualificationF32Epilogue {
    alpha: f32,
    beta: f32,
    bias: bool,
}

impl PartialEq for PhysicalQualificationF32Epilogue {
    fn eq(&self, other: &Self) -> bool {
        self.alpha.to_bits() == other.alpha.to_bits()
            && self.beta.to_bits() == other.beta.to_bits()
            && self.bias == other.bias
    }
}

impl Eq for PhysicalQualificationF32Epilogue {}

impl PhysicalQualificationF32Epilogue {
    #[must_use]
    pub const fn new(alpha: f32, beta: f32, bias: bool) -> Self {
        Self { alpha, beta, bias }
    }

    const fn for_op(op: ResolvedGemmOp) -> Self {
        Self::new(
            1.0,
            if matches!(op, ResolvedGemmOp::Tn) {
                1.0
            } else {
                0.0
            },
            false,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PhysicalQualificationScratchEnvelope {
    upcast_elements: (usize, usize, usize),
    half_staging_bytes: usize,
}

fn qualification_scratch_envelope(
    requests: &[PhysicalQualificationRequest],
) -> Result<PhysicalQualificationScratchEnvelope, String> {
    let mut envelope = PhysicalQualificationScratchEnvelope::default();
    for request in requests.iter().filter(|request| {
        matches!(
            request.route,
            PhysicalQualificationRoute::HalfPolicy { .. }
                | PhysicalQualificationRoute::HalfForced { .. }
        )
    }) {
        let (m, k, n) = request.dims;
        let mk = m
            .checked_mul(k)
            .ok_or_else(|| "qualification M*K scratch extent overflows usize".to_string())?;
        let kn = k
            .checked_mul(n)
            .ok_or_else(|| "qualification K*N scratch extent overflows usize".to_string())?;
        let mn = m
            .checked_mul(n)
            .ok_or_else(|| "qualification M*N scratch extent overflows usize".to_string())?;
        let required = match request.op {
            ResolvedGemmOp::Nn => (mk, kn, mn),
            ResolvedGemmOp::Tn => (mn, mk, 0),
            ResolvedGemmOp::Nt => (mn, kn, mk),
        };
        envelope.upcast_elements.0 = envelope.upcast_elements.0.max(required.0);
        envelope.upcast_elements.1 = envelope.upcast_elements.1.max(required.1);
        envelope.upcast_elements.2 = envelope.upcast_elements.2.max(required.2);
        envelope.half_staging_bytes = envelope.half_staging_bytes.max(
            mk.max(kn)
                .max(mn)
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| "qualification half staging extent overflows usize".to_string())?,
        );
    }
    Ok(envelope)
}

#[doc(hidden)]
/// Preallocates the largest half-conversion scratch envelope in `requests`.
///
/// Call this once for the complete selected suite before the first graph
/// capture. Graph capture freezes scratch addresses; a later request that
/// needs larger scratch fails instead of replacing a captured allocation.
///
/// # Errors
///
/// Returns an error if a request violates its route or layout contract, an
/// extent overflows, or the context cannot reserve the required scratch.
pub fn presize_physical_qualification_suite(
    ctx: &GpuCtx,
    requests: &[PhysicalQualificationRequest],
) -> Result<(), String> {
    for request in requests {
        request.validate()?;
    }
    let envelope = qualification_scratch_envelope(requests)?;
    ctx.ensure_half_staging(envelope.half_staging_bytes)?;
    ctx.with_bi_upcast_scratch(envelope.upcast_elements, |_, _, _| Ok(()))
}

#[derive(Clone, Copy)]
struct PhysicalQualificationPolicy {
    bi_tensor_cores: bool,
    f32_triad_policy: F32TriadPolicy,
    half_triad_policy: HalfTriadPolicy,
}

struct PhysicalQualificationPolicyLease<'a> {
    ctx: &'a GpuCtx,
    original_batch_invariant: bool,
    original_family: BiGemmFamily,
    original_tensor_cores: bool,
    original_fast_gemm: bool,
    original_f32_policy: F32TriadPolicy,
    original_half_policy: HalfTriadPolicy,
    frozen_cublas_tf32: bool,
    qualified_route: GemmRouteIdentity,
    active: ActiveQualificationToken,
}

impl PhysicalQualificationPolicyLease<'_> {
    fn validate(&self, ctx: &GpuCtx) -> Result<(), String> {
        if !std::ptr::eq(self.ctx, ctx) {
            return Err("physical qualification policy lease belongs to another context".into());
        }
        if ctx.tf32() != self.frozen_cublas_tf32 {
            return Err("cuBLAS TF32 state changed during physical qualification".into());
        }
        self.qualified_route
            .ensure_current(ctx.gemm_route(), "physical qualification policy lease")
    }
}

impl Drop for PhysicalQualificationPolicyLease<'_> {
    fn drop(&mut self) {
        self.ctx.set_half_triad_policy(self.original_half_policy);
        self.ctx.set_f32_triad_policy(self.original_f32_policy);
        self.ctx.set_fast_gemm(self.original_fast_gemm);
        self.ctx.set_bi_tensor_cores(self.original_tensor_cores);
        self.ctx.set_bi_gemm_family(self.original_family);
        self.ctx.set_batch_invariant(self.original_batch_invariant);
        self.active.release();
    }
}

fn begin_physical_qualification_policy(
    ctx: &GpuCtx,
    policy: PhysicalQualificationPolicy,
) -> Result<PhysicalQualificationPolicyLease<'_>, String> {
    let active = ActiveQualificationToken::acquire(ctx)?;
    let original_batch_invariant = ctx.batch_invariant();
    let original_family = ctx.bi_gemm_family();
    let original_tensor_cores = ctx.bi_tensor_cores();
    let original_fast_gemm = ctx.fast_gemm();
    let original_f32_policy = ctx.f32_triad_policy();
    let original_half_policy = ctx.half_triad_policy();
    let frozen_cublas_tf32 = ctx.tf32();
    ctx.set_batch_invariant(true);
    ctx.set_bi_gemm_family(BiGemmFamily::Triad);
    ctx.set_fast_gemm(false);
    ctx.set_bi_tensor_cores(policy.bi_tensor_cores);
    ctx.set_f32_triad_policy(policy.f32_triad_policy);
    ctx.set_half_triad_policy(policy.half_triad_policy);
    Ok(PhysicalQualificationPolicyLease {
        ctx,
        original_batch_invariant,
        original_family,
        original_tensor_cores,
        original_fast_gemm,
        original_f32_policy,
        original_half_policy,
        frozen_cublas_tf32,
        qualified_route: ctx.gemm_route(),
        active,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalTimedRequestIdentity {
    op: ResolvedGemmOp,
    logical_dtype: PolicyDtype,
    dims: (usize, usize, usize),
    planned_symbol: Option<&'static str>,
    planned_tile: Option<(u32, u32)>,
    layout: PhysicalQualificationLayout,
    route: PhysicalQualificationRoute,
    alpha_bits: u32,
    beta_bits: u32,
    bias: bool,
}

impl PhysicalQualificationRoute {
    fn logical_dtype(self) -> PolicyDtype {
        match self {
            Self::F32Policy(_) | Self::Tf32Forced(_) => PolicyDtype::F32,
            Self::HalfPolicy { dtype, .. } | Self::HalfForced { dtype, .. } => match dtype {
                WeightDtype::F32 => PolicyDtype::F32,
                WeightDtype::Bf16 => PolicyDtype::Bf16,
                WeightDtype::F16 => PolicyDtype::F16,
            },
        }
    }

    fn policy(self) -> PhysicalQualificationPolicy {
        match self {
            Self::F32Policy(policy) => PhysicalQualificationPolicy {
                bi_tensor_cores: false,
                f32_triad_policy: policy,
                half_triad_policy: HalfTriadPolicy::TiledParityV1,
            },
            Self::HalfPolicy {
                tensor_cores,
                half_policy,
                ..
            } => PhysicalQualificationPolicy {
                bi_tensor_cores: tensor_cores,
                f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
                half_triad_policy: half_policy,
            },
            // The forced stream-K tile records a route that only the
            // permitting half policy admits; every other forced tile stays
            // on the tiled parity contract.
            Self::HalfForced { tile, .. } => PhysicalQualificationPolicy {
                bi_tensor_cores: true,
                f32_triad_policy: F32TriadPolicy::ExactScalarFmaV1,
                half_triad_policy: if tile == TcTile::Tile64StreamK {
                    HalfTriadPolicy::AllowStreamKFixedOrderV1
                } else {
                    HalfTriadPolicy::TiledParityV1
                },
            },
            Self::Tf32Forced(_) => PhysicalQualificationPolicy {
                bi_tensor_cores: false,
                f32_triad_policy: F32TriadPolicy::AllowDeterministicTf32V1,
                half_triad_policy: HalfTriadPolicy::TiledParityV1,
            },
        }
    }
}

impl PhysicalQualificationRequest {
    /// Creates a request with the operation-specific contiguous strides.
    ///
    /// Validation is deferred to [`presize_physical_qualification_suite`] or
    /// [`qualify_physical_launch`]. Half routes reject F32 storage, `Thin16`
    /// is NN-only, and `M` and `N` must be positive. `K = 0` is supported.
    #[must_use]
    pub fn contiguous(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        route: PhysicalQualificationRoute,
    ) -> Self {
        Self {
            op,
            dims,
            route,
            layout: PhysicalQualificationLayout::Contiguous,
            f32_epilogue: None,
        }
    }

    /// Creates a contiguous F32 request with an explicit retained epilogue.
    #[must_use]
    pub fn contiguous_f32(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        route: PhysicalQualificationRoute,
        epilogue: PhysicalQualificationF32Epilogue,
    ) -> Self {
        Self {
            op,
            dims,
            route,
            layout: PhysicalQualificationLayout::Contiguous,
            f32_epilogue: Some(epilogue),
        }
    }

    /// Creates a half-precision request with one named operand origin shifted
    /// by one element inside facade-owned guarded storage.
    ///
    /// F32 routes do not use this qualification layout. All other validation
    /// rules from [`Self::contiguous`] still apply.
    #[must_use]
    pub fn one_element_offset(
        op: ResolvedGemmOp,
        dims: (usize, usize, usize),
        route: PhysicalQualificationRoute,
        offset: PhysicalQualificationOffset,
    ) -> Self {
        Self {
            op,
            dims,
            route,
            layout: PhysicalQualificationLayout::OneElementOffset(offset),
            f32_epilogue: None,
        }
    }

    /// Creates one frozen direct forced-Tensor-Core NN padded layout.
    ///
    /// This constructor is valid only for `(M, K, N) = (129, 65, 129)`, BF16
    /// or F16 storage, a forced `Tile64` or `Tile128` route, and strides from
    /// [`PhysicalPaddedNnLayout`]. Padded TN and NT requests are unsupported.
    /// Validation is deferred to [`presize_physical_qualification_suite`] or
    /// [`qualify_physical_launch`].
    #[must_use]
    pub fn direct_padded_nn(
        dims: (usize, usize, usize),
        dtype: WeightDtype,
        tile: TcTile,
        layout: PhysicalPaddedNnLayout,
    ) -> Self {
        Self {
            op: ResolvedGemmOp::Nn,
            dims,
            route: PhysicalQualificationRoute::HalfForced { dtype, tile },
            layout: PhysicalQualificationLayout::DirectPaddedNn(layout),
            f32_epilogue: None,
        }
    }

    const fn f32_epilogue(self) -> PhysicalQualificationF32Epilogue {
        match self.f32_epilogue {
            Some(epilogue) => epilogue,
            None => PhysicalQualificationF32Epilogue::for_op(self.op),
        }
    }

    const fn element_offsets(self) -> (usize, usize, usize) {
        match self.layout {
            PhysicalQualificationLayout::OneElementOffset(PhysicalQualificationOffset::Output) => {
                (1, 0, 0)
            }
            PhysicalQualificationLayout::OneElementOffset(PhysicalQualificationOffset::A) => {
                (0, 1, 0)
            }
            PhysicalQualificationLayout::OneElementOffset(PhysicalQualificationOffset::B) => {
                (0, 0, 1)
            }
            PhysicalQualificationLayout::Contiguous
            | PhysicalQualificationLayout::DirectPaddedNn(_) => (0, 0, 0),
        }
    }

    fn physical_shape(self) -> F32TriadShape {
        let mut shape = F32TriadShape::contiguous(self.op, self.dims);
        if let PhysicalQualificationLayout::DirectPaddedNn(layout) = self.layout {
            (shape.lda, shape.ldb, shape.ldc) = layout.strides();
        }
        shape
    }

    fn validate(self) -> Result<(), String> {
        let (m, k, n) = self.dims;
        if m == 0 || n == 0 {
            return Err("physical qualification requires positive M and N".into());
        }
        checked_elements(m, k, "M*K")?;
        checked_elements(k, n, "K*N")?;
        checked_elements(m, n, "M*N")?;
        self.physical_shape().validate(self.op)?;
        if let Some(epilogue) = self.f32_epilogue {
            if !matches!(
                self.route,
                PhysicalQualificationRoute::F32Policy(_)
                    | PhysicalQualificationRoute::Tf32Forced(_)
            ) {
                return Err("F32 qualification epilogue requires an F32 route".into());
            }
            if !epilogue.alpha.is_finite() || !epilogue.beta.is_finite() {
                return Err("F32 qualification epilogue scalars must be finite".into());
            }
            if epilogue.bias && self.op != ResolvedGemmOp::Nn {
                return Err("F32 qualification bias is supported only for NN".into());
            }
        }
        if matches!(self.route, PhysicalQualificationRoute::F32Policy(_)) {
            let epilogue = self.f32_epilogue();
            let wrapper = PhysicalQualificationF32Epilogue::for_op(self.op);
            if epilogue.alpha.to_bits() != wrapper.alpha.to_bits()
                || epilogue.beta.to_bits() != wrapper.beta.to_bits()
            {
                return Err(
                    "F32 policy qualification epilogue must match the production wrapper alpha and beta"
                        .into(),
                );
            }
        }
        match self.route {
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::F32,
                ..
            }
            | PhysicalQualificationRoute::HalfForced {
                dtype: WeightDtype::F32,
                ..
            } => return Err("half qualification route requires BF16 or F16".into()),
            PhysicalQualificationRoute::HalfForced {
                tile: TcTile::Thin16,
                ..
            } if self.op != ResolvedGemmOp::Nn => {
                return Err("forced Thin16 qualification only supports NN".into());
            }
            PhysicalQualificationRoute::HalfForced {
                tile: TcTile::Rect128x64,
                ..
            } if self.op != ResolvedGemmOp::Tn => {
                return Err("forced Rect128x64 qualification only supports TN".into());
            }
            PhysicalQualificationRoute::HalfForced {
                tile: TcTile::Tile64StreamK,
                ..
            } if self.op != ResolvedGemmOp::Tn => {
                return Err("forced Tile64StreamK qualification only supports TN".into());
            }
            PhysicalQualificationRoute::HalfPolicy { dtype, .. }
            | PhysicalQualificationRoute::HalfForced { dtype, .. } => {
                for (label, elements) in [
                    ("M*K", checked_elements(m, k, "M*K")?),
                    ("K*N", checked_elements(k, n, "K*N")?),
                    ("M*N", checked_elements(m, n, "M*N")?),
                ] {
                    elements.checked_mul(dtype.size_bytes()).ok_or_else(|| {
                        format!("qualification {label} byte extent overflows usize")
                    })?;
                }
            }
            PhysicalQualificationRoute::Tf32Forced(route) => {
                if matches!(
                    route,
                    Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
                        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
                        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_)
                ) {
                    tf32_splitk_spec(self.op, route)?;
                } else {
                    tf32_kernel_spec(self.op, route)?;
                }
            }
            _ => {}
        }
        match self.layout {
            PhysicalQualificationLayout::Contiguous => Ok(()),
            PhysicalQualificationLayout::OneElementOffset(_) => match self.route {
                PhysicalQualificationRoute::HalfPolicy { .. }
                | PhysicalQualificationRoute::HalfForced { .. } => Ok(()),
                _ => Err("one-element qualification offsets require a half route".into()),
            },
            PhysicalQualificationLayout::DirectPaddedNn(_) => {
                if self.op != ResolvedGemmOp::Nn || self.dims != (129, 65, 129) {
                    return Err(
                        "direct padded qualification is frozen to NN M=129 K=65 N=129".into(),
                    );
                }
                match self.route {
                    PhysicalQualificationRoute::HalfForced {
                        dtype: WeightDtype::Bf16 | WeightDtype::F16,
                        tile: TcTile::Tile64 | TcTile::Tile128,
                    } => Ok(()),
                    _ => {
                        Err("direct padded qualification requires a forced half square tile".into())
                    }
                }
            }
        }
    }

    fn timed_identity(self) -> Result<PhysicalTimedRequestIdentity, String> {
        let (planned_symbol, planned_tile) = match self.route {
            PhysicalQualificationRoute::HalfForced { tile, .. } => {
                let tile = match tile {
                    TcTile::Tile128 => (128, 128),
                    TcTile::Tile64 => (64, 64),
                    TcTile::Thin16 => (16, 32),
                    TcTile::Rect128x64 => (128, 64),
                    TcTile::Tile64StreamK => (64, 64),
                };
                (None, Some(tile))
            }
            PhysicalQualificationRoute::Tf32Forced(route) => {
                if matches!(
                    route,
                    Tf32PhysicalRoute::MmaTf32RnaSplitK2V1(_)
                        | Tf32PhysicalRoute::MmaTf32RnaSplitK4V1(_)
                        | Tf32PhysicalRoute::MmaTf32RnaSplitK8V1(_)
                ) {
                    let spec = tf32_splitk_spec(self.op, route)?;
                    (Some(spec.symbol), Some(spec.tile))
                } else {
                    let spec = tf32_kernel_spec(self.op, route)?;
                    (Some(spec.symbol), Some(spec.tile))
                }
            }
            _ => (None, None),
        };
        Ok(PhysicalTimedRequestIdentity {
            op: self.op,
            logical_dtype: self.route.logical_dtype(),
            dims: self.dims,
            planned_symbol,
            planned_tile,
            layout: self.layout,
            route: self.route,
            alpha_bits: self.f32_epilogue().alpha.to_bits(),
            beta_bits: self.f32_epilogue().beta.to_bits(),
            bias: self.f32_epilogue().bias,
        })
    }
}

fn weight_dtype_name(dtype: WeightDtype) -> &'static [u8] {
    match dtype {
        WeightDtype::F32 => b"f32",
        WeightDtype::Bf16 => b"bf16",
        WeightDtype::F16 => b"f16",
    }
}

fn tc_tile_name(tile: TcTile) -> &'static [u8] {
    match tile {
        TcTile::Tile128 => b"tile128",
        TcTile::Tile64 => b"tile64",
        TcTile::Thin16 => b"thin16",
        TcTile::Rect128x64 => b"rect128x64",
        TcTile::Tile64StreamK => b"tile64_streamk",
    }
}

fn timed_request_digest(identity: PhysicalTimedRequestIdentity) -> [u8; 32] {
    let op = match identity.op {
        ResolvedGemmOp::Nn => b"nn".as_slice(),
        ResolvedGemmOp::Tn => b"tn".as_slice(),
        ResolvedGemmOp::Nt => b"nt".as_slice(),
    };
    let dtype = match identity.logical_dtype {
        PolicyDtype::F32 => b"f32".as_slice(),
        PolicyDtype::Bf16 => b"bf16".as_slice(),
        PolicyDtype::F16 => b"f16".as_slice(),
    };
    let mut digest = FramedSha256::new(b"gemm-bi-physical-timed-request.v1")
        .required(b"op", op)
        .required(b"dtype", dtype)
        .required(b"m", &(identity.dims.0 as u64).to_le_bytes())
        .required(b"k", &(identity.dims.1 as u64).to_le_bytes())
        .required(b"n", &(identity.dims.2 as u64).to_le_bytes());
    digest = digest
        .required(b"alpha-bits", &identity.alpha_bits.to_le_bytes())
        .required(b"beta-bits", &identity.beta_bits.to_le_bytes())
        .required(b"bias", &[u8::from(identity.bias)]);
    digest = match identity.route {
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1) => {
            digest.required(b"route", b"f32-policy-exact-scalar-fma-v1")
        }
        PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1) => {
            digest.required(b"route", b"f32-policy-allow-deterministic-tf32-v1")
        }
        PhysicalQualificationRoute::HalfPolicy {
            dtype,
            tensor_cores,
            half_policy,
        } => digest
            .required(
                b"route",
                match half_policy {
                    // Retain the byte-for-byte legacy tiled request encoding.
                    HalfTriadPolicy::TiledParityV1 => b"half-policy",
                    HalfTriadPolicy::AllowStreamKFixedOrderV1 => {
                        b"half-policy-allow-streamk-fixed-order-v1"
                    }
                },
            )
            .required(b"route-dtype", weight_dtype_name(dtype))
            .required(b"tensor-cores", &[u8::from(tensor_cores)]),
        PhysicalQualificationRoute::HalfForced { dtype, tile } => digest
            .required(b"route", b"half-forced")
            .required(b"route-dtype", weight_dtype_name(dtype))
            .required(b"forced-tile", tc_tile_name(tile)),
        PhysicalQualificationRoute::Tf32Forced(route) => super::contract::append_tf32_route_digest(
            digest.required(b"route", b"tf32-forced"),
            route,
        ),
    };
    digest = match identity.layout {
        PhysicalQualificationLayout::Contiguous => digest.required(b"layout", b"contiguous"),
        PhysicalQualificationLayout::OneElementOffset(offset) => {
            let offset = match offset {
                PhysicalQualificationOffset::Output => b"output-offset-1".as_slice(),
                PhysicalQualificationOffset::A => b"a-offset-1".as_slice(),
                PhysicalQualificationOffset::B => b"b-offset-1".as_slice(),
            };
            digest.required(b"layout", offset)
        }
        PhysicalQualificationLayout::DirectPaddedNn(layout) => {
            let strides = layout.strides();
            digest
                .required(b"layout", b"direct-padded-nn")
                .required(b"lda", &(strides.0 as u64).to_le_bytes())
                .required(b"ldb", &(strides.1 as u64).to_le_bytes())
                .required(b"ldc", &(strides.2 as u64).to_le_bytes())
        }
    };
    digest.finish()
}

#[doc(hidden)]
#[must_use = "physical launch nodes are qualification evidence"]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Read-only summary of one actual CUDA launch in enqueue order.
pub struct QualifiedPhysicalLaunchNode {
    /// Physical work category such as GEMM or dtype conversion.
    pub kind: PhysicalLaunchKind,
    /// Exact CUDA kernel symbol.
    pub symbol: &'static str,
    /// Module that owns the symbol.
    pub module_kind: ModuleKind,
    /// Logical GEMM operation.
    pub logical_op: ResolvedGemmOp,
    /// Logical request dtype.
    pub logical_dtype: PolicyDtype,
    /// Dtype consumed by this physical node.
    pub execution_dtype: PolicyDtype,
    /// Logical `(M, K, N)` dimensions.
    pub shape: (usize, usize, usize),
    /// Physical `(lda, ldb, ldc)` strides.
    pub strides: (usize, usize, usize),
    /// Kernel tile when the node has a tiled contract.
    pub tile: Option<(u32, u32)>,
    /// Numeric contract of the GEMM route, when the node is a GEMM.
    pub numeric_contract: Option<ResolvedNumericContract>,
    /// Output ownership of the GEMM route, when the node is a GEMM.
    pub ownership: Option<crate::mamba_ssm::gpu::kernel_identity::ResolvedOutputOwnership>,
    /// Grid, block, dynamic shared memory, and raw-address-free argument digest.
    ///
    /// The digest is bound to the holder's allocation sizes, subview offsets,
    /// required spans, buffer generations, and physical tensor-map geometry;
    /// it is not portable between independently allocated holders.
    pub launch: ResolvedKernelLaunch,
}

impl From<&ResolvedPhysicalKernelLaunch> for QualifiedPhysicalLaunchNode {
    fn from(node: &ResolvedPhysicalKernelLaunch) -> Self {
        Self {
            kind: node.kind(),
            symbol: node.symbol(),
            module_kind: node.module_kind(),
            logical_op: node.logical_op(),
            logical_dtype: node.logical_dtype(),
            execution_dtype: node.execution_dtype(),
            shape: node.shape(),
            strides: node.strides(),
            tile: node.tile(),
            numeric_contract: node.gemm_route().map(|route| route.numeric_contract),
            ownership: node.gemm_route().map(|route| route.ownership),
            launch: node.launch(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct HalfProductionBranch {
    seal: HalfPolicyBranchSeal,
    production_routes: Box<[ResolvedGemmRoute]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ProductionBranchSeal {
    F32(Box<[ResolvedGemmRoute]>),
    Half(HalfProductionBranch),
}

impl ProductionBranchSeal {
    fn is_empty(&self) -> bool {
        match self {
            Self::F32(routes) => routes.is_empty(),
            Self::Half(branch) => match branch.seal {
                HalfPolicyBranchSeal::Native(_)
                | HalfPolicyBranchSeal::Sm120(_)
                | HalfPolicyBranchSeal::Sm100(_)
                | HalfPolicyBranchSeal::Sm90a(_) => false,
                HalfPolicyBranchSeal::ExactF32Fallback => branch.production_routes.is_empty(),
            },
        }
    }
}

#[doc(hidden)]
#[must_use = "physical launch evidence must be validated or emitted"]
#[derive(Clone, Debug, PartialEq, Eq)]
/// Immutable eager-preflight and exact-graph physical evidence.
///
/// `single_launch_*` is `None` for every multi-launch route. Uniform fields
/// are `None` when nodes use more than one execution dtype or module owner.
pub struct QualifiedPhysicalLaunchEvidence {
    evidence_scope: &'static str,
    route_identity: GemmRouteIdentity,
    timed_request: PhysicalTimedRequestIdentity,
    nodes: Box<[QualifiedPhysicalLaunchNode]>,
    production_branch: ProductionBranchSeal,
    request_identity_digest: [u8; 32],
    launch_count: u32,
    launch_digest: [u8; 32],
    single_launch_symbol: Option<&'static str>,
    single_launch_tile: Option<(u32, u32)>,
    uniform_execution_dtype: Option<PolicyDtype>,
    uniform_module_kind: Option<ModuleKind>,
    eager_graph_equal: bool,
}

impl QualifiedPhysicalLaunchEvidence {
    /// Returns the evidence scope.
    ///
    /// `eager_preflight_same_launcher` means one real untimed production enqueue
    /// supplied the eager branch proof and the exact recorded manifest was captured.
    pub fn evidence_scope(&self) -> &'static str {
        self.evidence_scope
    }

    /// Returns normalized policy, compiler, artifact, device-capability, and revision identity.
    pub fn route_identity(&self) -> &GemmRouteIdentity {
        &self.route_identity
    }

    /// Returns actual physical nodes in enqueue order.
    pub fn nodes(&self) -> &[QualifiedPhysicalLaunchNode] {
        &self.nodes
    }

    /// Returns the exact number of actual physical nodes.
    pub fn launch_count(&self) -> u32 {
        self.launch_count
    }

    /// Returns the framed digest of the ordered physical node manifest.
    ///
    /// It excludes raw device addresses but remains bound to allocation size,
    /// subview, required-span, and buffer-generation identity for this holder.
    pub fn launch_digest(&self) -> [u8; 32] {
        self.launch_digest
    }

    /// Returns the portable framed logical request digest.
    ///
    /// This digest excludes allocations, device addresses, holder identity,
    /// capture state, and prepared-resource identity.
    pub fn request_identity_digest(&self) -> [u8; 32] {
        self.request_identity_digest
    }

    /// Returns the symbol only when the request enqueues exactly one node.
    pub fn single_launch_symbol(&self) -> Option<&'static str> {
        self.single_launch_symbol
    }

    /// Returns the tile only when the request enqueues exactly one node.
    pub fn single_launch_tile(&self) -> Option<(u32, u32)> {
        self.single_launch_tile
    }

    /// Returns the execution dtype when it is uniform across all nodes.
    pub fn uniform_execution_dtype(&self) -> Option<PolicyDtype> {
        self.uniform_execution_dtype
    }

    /// Returns the module owner when it is uniform across all nodes.
    pub fn uniform_module_kind(&self) -> Option<ModuleKind> {
        self.uniform_module_kind
    }

    /// Reports exact ordered eager-preflight and captured-graph equality.
    pub fn eager_graph_equal(&self) -> bool {
        self.eager_graph_equal
    }

    fn validate_timed_request(
        &self,
        ctx: &GpuCtx,
        request: PhysicalQualificationRequest,
    ) -> Result<(), String> {
        self.route_identity
            .ensure_current(ctx.gemm_route(), "qualified timed request")?;
        if request.timed_identity()? != self.timed_request {
            return Err("timed request identity differs from the qualified preflight".into());
        }
        let contract = PhysicalEvidenceContract::from_evidence(self, self.timed_request);
        validate_physical_evidence_contract(&contract)?;
        if self.production_branch.is_empty() {
            return Err("qualified production branch proof contains no branch node".into());
        }
        Ok(())
    }
}

struct QualifiedF32Resources {
    output: GpuBuffer,
    a: GpuBuffer,
    b: GpuBuffer,
    bias: Option<GpuBuffer>,
    prepared: Option<PreparedF32TriadLaunch>,
    binding: (u64, u64, u64, Option<u64>),
}

impl QualifiedF32Resources {
    fn validate(&self) -> Result<(), String> {
        let current = (
            self.output.cached_ptr(),
            self.a.cached_ptr(),
            self.b.cached_ptr(),
            self.bias.as_ref().map(GpuBuffer::cached_ptr),
        );
        if current == self.binding {
            Ok(())
        } else {
            Err("qualified F32 resource binding changed".into())
        }
    }
}

enum QualifiedHalfOutput {
    Typed(DtypedBuf),
    F32(GpuBuffer),
}

impl QualifiedHalfOutput {
    fn ptr(&self) -> u64 {
        match self {
            Self::Typed(buffer) => buffer.cached_ptr(),
            Self::F32(buffer) => buffer.cached_ptr(),
        }
    }

    fn element_bytes(&self) -> usize {
        match self {
            Self::Typed(buffer) => buffer.dtype().size_bytes(),
            Self::F32(_) => std::mem::size_of::<f32>(),
        }
    }

    fn to_cpu(&self, ctx: &GpuCtx) -> Result<Vec<f32>, String> {
        match self {
            Self::Typed(buffer) => {
                let mut values = vec![0.0; buffer.len_elems()];
                buffer.download_f32(&ctx.stream, &mut values)?;
                Ok(values)
            }
            Self::F32(buffer) => buffer.to_cpu(&ctx.stream),
        }
    }
}

struct QualifiedHalfResources {
    output: QualifiedHalfOutput,
    a: DtypedBuf,
    b: DtypedBuf,
    binding: (u64, u64, u64),
    storage: QualifiedHalfStorage,
}

#[derive(Clone, Copy)]
struct QualificationStorage {
    rows: usize,
    width: usize,
    stride: usize,
    offset: usize,
    len: usize,
}

impl QualificationStorage {
    fn new(
        rows: usize,
        width: usize,
        stride: usize,
        offset: usize,
        guarded: bool,
    ) -> Result<Self, String> {
        if rows == 0 || width == 0 || stride < width {
            return Err("invalid physical qualification storage geometry".into());
        }
        let span = (rows - 1)
            .checked_mul(stride)
            .and_then(|value| value.checked_add(width))
            .ok_or_else(|| "physical qualification storage span overflows usize".to_string())?;
        let len = offset
            .checked_add(span)
            .and_then(|value| {
                value.checked_add(if guarded {
                    QUALIFICATION_GUARD_ELEMENTS
                } else {
                    0
                })
            })
            .ok_or_else(|| "physical qualification guarded storage overflows usize".to_string())?;
        Ok(Self {
            rows,
            width,
            stride,
            offset,
            len,
        })
    }

    fn host_values(self) -> Vec<f32> {
        let mut values = vec![QUALIFICATION_GUARD_VALUE; self.len];
        for row in 0..self.rows {
            let start = self.offset + row * self.stride;
            values[start..start + self.width].fill(0.0);
        }
        values
    }

    fn guard_count(self) -> usize {
        self.len - self.rows * self.width
    }

    fn validate_guards(self, values: &[f32], label: &str) -> Result<usize, String> {
        if values.len() != self.len {
            return Err(format!("{label} guarded storage length changed"));
        }
        let guard = QUALIFICATION_GUARD_VALUE.to_bits();
        let mut checked = 0;
        for (index, value) in values.iter().copied().enumerate() {
            let relative = index.checked_sub(self.offset);
            let active = relative.is_some_and(|relative| {
                let row = relative / self.stride;
                let column = relative % self.stride;
                row < self.rows && column < self.width
            });
            if !active {
                if value.to_bits() != guard {
                    return Err(format!(
                        "{label} red zone changed at element {index}: got {value:?}, expected {QUALIFICATION_GUARD_VALUE:?}"
                    ));
                }
                checked += 1;
            }
        }
        if checked != self.guard_count() {
            return Err(format!("{label} red-zone census changed"));
        }
        Ok(checked)
    }
}

#[derive(Clone, Copy)]
struct QualifiedHalfStorage {
    output: QualificationStorage,
    a: QualificationStorage,
    b: QualificationStorage,
}

impl QualifiedHalfResources {
    fn validate(&self) -> Result<(), String> {
        let current = (
            self.output.ptr() + (self.storage.output.offset * self.output.element_bytes()) as u64,
            self.a.cached_ptr() + (self.storage.a.offset * self.a.dtype().size_bytes()) as u64,
            self.b.cached_ptr() + (self.storage.b.offset * self.b.dtype().size_bytes()) as u64,
        );
        if current == self.binding {
            Ok(())
        } else {
            Err("qualified half resource binding changed".into())
        }
    }

    fn validate_red_zones(&self, ctx: &GpuCtx) -> Result<usize, String> {
        let output = self.output.to_cpu(ctx)?;
        let mut a = vec![0.0; self.storage.a.len];
        self.a.download_f32(&ctx.stream, &mut a)?;
        let mut b = vec![0.0; self.storage.b.len];
        self.b.download_f32(&ctx.stream, &mut b)?;
        Ok(self.storage.output.validate_guards(&output, "output")?
            + self.storage.a.validate_guards(&a, "A")?
            + self.storage.b.validate_guards(&b, "B")?)
    }
}

enum QualifiedPhysicalResources {
    F32(Box<QualifiedF32Resources>),
    Half(Box<QualifiedHalfResources>),
}

impl QualifiedPhysicalResources {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::F32(resources) => resources.validate(),
            Self::Half(resources) => resources.validate(),
        }
    }

    fn validate_red_zones(&self, ctx: &GpuCtx) -> Result<QualifiedGuardValidation, String> {
        match self {
            Self::F32(_) => Ok(QualifiedGuardValidation {
                allocation_count: 0,
                element_count: 0,
            }),
            Self::Half(resources) => {
                let element_count = resources.validate_red_zones(ctx)?;
                Ok(QualifiedGuardValidation {
                    allocation_count: usize::from(element_count != 0) * 3,
                    element_count,
                })
            }
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Read-only result of checking facade-owned allocation red zones.
pub struct QualifiedGuardValidation {
    allocation_count: usize,
    element_count: usize,
}

impl QualifiedGuardValidation {
    /// Returns the number of allocations whose red zones were checked.
    pub fn allocation_count(self) -> usize {
        self.allocation_count
    }

    /// Returns the total number of red-zone elements checked.
    pub fn element_count(self) -> usize {
        self.element_count
    }
}

fn typed_pointer(pointer: u64, dtype: WeightDtype) -> TypedPtr {
    TypedPtr {
        ptr: pointer,
        dtype,
    }
}

fn launch_f32_policy_eager(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
    resources: &mut QualifiedF32Resources,
) -> Result<(), String> {
    let (m, k, n) = request.dims;
    match request.op {
        ResolvedGemmOp::Nn => gpu_gemm_bi_forward_raw(
            ctx,
            &mut resources.output,
            &resources.a,
            resources.b.cached_ptr(),
            resources.bias.as_ref().map(GpuBuffer::cached_ptr),
            request.dims,
        ),
        ResolvedGemmOp::Tn => gpu_gemm_bi_backward_dw_grad(
            ctx,
            &GradSlice::from_raw(
                resources.output.cached_ptr(),
                checked_elements(k, n, "K*N")?,
            ),
            &resources.b,
            &resources.a,
            m,
            k,
            n,
        ),
        ResolvedGemmOp::Nt => gpu_gemm_bi_backward_dx_raw(
            ctx,
            &mut resources.output,
            &resources.a,
            resources.b.cached_ptr(),
            m,
            k,
            n,
        ),
    }
}

fn launch_half_policy_eager(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
    resources: &QualifiedHalfResources,
    dtype: WeightDtype,
) -> Result<(), String> {
    let output = resources.binding.0;
    let a = typed_pointer(resources.binding.1, dtype);
    let b = typed_pointer(resources.binding.2, dtype);
    match request.op {
        ResolvedGemmOp::Nn => {
            gemm_bi_forward_typed(ctx, typed_pointer(output, dtype), a, b, 0, request.dims)
        }
        ResolvedGemmOp::Tn => gemm_bi_backward_dw_typed(ctx, output, b, a, request.dims),
        ResolvedGemmOp::Nt => {
            gemm_bi_backward_dx_typed(ctx, typed_pointer(output, dtype), a, b, request.dims)
        }
    }
}

fn launch_half_forced_eager(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
    resources: &QualifiedHalfResources,
    dtype: WeightDtype,
    tile: TcTile,
) -> Result<(), String> {
    let output = resources.binding.0;
    let a = typed_pointer(resources.binding.1, dtype);
    let b = typed_pointer(resources.binding.2, dtype);
    match request.op {
        ResolvedGemmOp::Nn => {
            let operands = TcFwdOperands {
                y: typed_pointer(output, dtype),
                x: a,
                w: b,
                bias_ptr: 0,
            };
            match request.layout {
                PhysicalQualificationLayout::DirectPaddedNn(_) => {
                    gemm_bi_forward_tc_with_tile_shape(
                        ctx,
                        &operands,
                        request.physical_shape(),
                        tile,
                    )
                }
                _ => gemm_bi_forward_tc_with_tile(
                    &ctx.stream,
                    &ctx.kernels,
                    &operands,
                    request.dims,
                    tile,
                ),
            }
        }
        ResolvedGemmOp::Tn => gemm_bi_backward_dw_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            output,
            b,
            a,
            request.dims,
            tile,
        ),
        ResolvedGemmOp::Nt => gemm_bi_backward_dx_tc_with_tile(
            &ctx.stream,
            &ctx.kernels,
            typed_pointer(output, dtype),
            a,
            b,
            request.dims,
            tile,
        ),
    }
}

fn synchronized_enqueue_result<T>(
    ctx: &GpuCtx,
    label: &str,
    result: Result<T, String>,
) -> Result<T, String> {
    let synchronization = ctx
        .stream
        .synchronize()
        .map_err(|error| format!("{label} synchronize failed: {error:?}"));
    match (result, synchronization) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary), Ok(())) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(primary), Err(cleanup)) => Err(format!("{primary}; {cleanup}")),
    }
}

fn eager_timing_error(ctx: &GpuCtx, primary: String) -> String {
    match ctx.stream.synchronize() {
        Ok(()) => primary,
        Err(cleanup) => format!("{primary}; cleanup synchronize failed: {cleanup:?}"),
    }
}

fn measure_production_eager(
    ctx: &GpuCtx,
    iterations: usize,
    mut launch: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    if iterations == 0 {
        return Err("timed eager iteration count must be positive".into());
    }
    let start = ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("record production eager start event: {error:?}"))?;
    let mut failure = None;
    for _ in 0..iterations {
        if let Err(error) = launch() {
            failure = Some(error);
            break;
        }
    }
    let end = match ctx
        .stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
    {
        Ok(end) => end,
        Err(error) => {
            return Err(eager_timing_error(
                ctx,
                format!("record production eager end event: {error:?}"),
            ));
        }
    };
    if let Some(error) = failure {
        return Err(eager_timing_error(
            ctx,
            format!("launch production eager path: {error}"),
        ));
    }
    let elapsed = start
        .elapsed_ms(&end)
        .map(f64::from)
        .map_err(|error| format!("measure production eager events: {error:?}"));
    elapsed.map_err(|primary| eager_timing_error(ctx, primary))
}

#[doc(hidden)]
#[must_use = "dropping the holder releases its graph, resources, and exclusive policy lease"]
/// Owns one qualified graph, every referenced allocation, and its policy lease.
///
/// A context may have only one live holder. Use a dedicated context and drop
/// the holder before qualifying another request. The graph drops before its
/// buffers, and the normalized policy is restored last. Do not enqueue
/// unrelated work or mutate policy on that context while the holder is live.
pub struct QualifiedPhysicalLaunch<'a> {
    evidence: QualifiedPhysicalLaunchEvidence,
    request: PhysicalQualificationRequest,
    graph: CapturedPhysicalGraph,
    resources: QualifiedPhysicalResources,
    policy: PhysicalQualificationPolicyLease<'a>,
}

impl QualifiedPhysicalLaunch<'_> {
    /// Returns the immutable physical evidence captured at construction.
    pub fn evidence(&self) -> &QualifiedPhysicalLaunchEvidence {
        &self.evidence
    }

    /// Verifies that `request` and the context still match the frozen launch.
    pub fn validate_timed_request(
        &self,
        ctx: &GpuCtx,
        request: PhysicalQualificationRequest,
    ) -> Result<(), String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        self.evidence.validate_timed_request(ctx, request)
    }

    /// Synchronizes the stream and validates every facade-owned red zone.
    pub fn validate_red_zones(&self, ctx: &GpuCtx) -> Result<QualifiedGuardValidation, String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize qualification red-zone check: {error:?}"))?;
        self.resources.validate_red_zones(ctx)
    }

    /// Replaces every F32 qualification operand with deterministic finite data.
    ///
    /// This is a test-only route-comparison hook. The same request and salt
    /// produce identical buffers in independent contexts, so physical TF32
    /// routes can be compared bit for bit without exposing raw allocations.
    pub fn seed_f32_operands(&mut self, ctx: &GpuCtx, salt: u64) -> Result<(), String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        let QualifiedPhysicalResources::F32(resources) = &mut self.resources else {
            return Err("F32 operand seeding requires an F32 qualification route".into());
        };
        resources.output.upload(
            &ctx.stream,
            &seeded_qualification_values(resources.output.len(), salt ^ 0x91),
        )?;
        resources.a.upload(
            &ctx.stream,
            &seeded_qualification_values(resources.a.len(), salt ^ 0x2d),
        )?;
        resources.b.upload(
            &ctx.stream,
            &seeded_qualification_values(resources.b.len(), salt ^ 0x67),
        )?;
        if let Some(bias) = &mut resources.bias {
            bias.upload(
                &ctx.stream,
                &seeded_qualification_values(bias.len(), salt ^ 0xc3),
            )?;
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize F32 qualification seeding: {error:?}"))
    }

    /// Seeds an NN route with one exactly representable active K term.
    ///
    /// This test-only hook detects operand-role and row-major binding mistakes
    /// before a performance comparison. Every expected output is one product,
    /// so a vendor comparator may use a different reduction implementation
    /// without becoming a bitwise reduction oracle.
    pub fn seed_f32_nn_single_term_probe(&mut self, ctx: &GpuCtx) -> Result<(), String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        if self.request.op != ResolvedGemmOp::Nn || self.request.dims.1 == 0 {
            return Err("single-term F32 probe requires NN with a nonzero reduction".into());
        }
        let QualifiedPhysicalResources::F32(resources) = &mut self.resources else {
            return Err("single-term F32 probe requires an F32 qualification route".into());
        };
        let (m, k, n) = self.request.dims;
        let mut a = vec![0.0; resources.a.len()];
        let mut b = vec![0.0; resources.b.len()];
        for row in 0..m {
            a[row * k] = ((row % 7) as i32 - 3) as f32 * 0.125;
        }
        for (column, value) in b.iter_mut().take(n).enumerate() {
            *value = ((column % 11) as i32 - 5) as f32 * 0.125;
        }
        resources
            .output
            .upload(&ctx.stream, &vec![0.0; resources.output.len()])?;
        resources.a.upload(&ctx.stream, &a)?;
        resources.b.upload(&ctx.stream, &b)?;
        if let Some(bias) = &mut resources.bias {
            bias.upload(&ctx.stream, &vec![0.0; bias.len()])?;
        }
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize single-term F32 probe seeding: {error:?}"))
    }

    /// Downloads the current F32 qualification output as exact IEEE words.
    pub fn f32_output_bits(&self, ctx: &GpuCtx) -> Result<Vec<u32>, String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        let QualifiedPhysicalResources::F32(resources) = &self.resources else {
            return Err("F32 output download requires an F32 qualification route".into());
        };
        Ok(resources
            .output
            .to_cpu(&ctx.stream)?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Measures exact graph replays and returns total CUDA-event window milliseconds.
    ///
    /// Each iteration is one logical replay. Validation occurs before the
    /// start event, and a successful measurement waits for the end event.
    /// Divide the returned total by `iterations` to obtain time per replay.
    ///
    /// # Errors
    ///
    /// Returns an error when `iterations` is zero, the holder no longer matches
    /// the context or its resources, event creation or recording fails, a graph
    /// replay fails, or stream synchronization fails.
    pub fn measure_graph_window_ms(&self, ctx: &GpuCtx, iterations: usize) -> Result<f64, String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        self.graph.measure_prevalidated(
            ctx,
            iterations,
            "qualified physical performance graph window",
        )
    }

    /// Measures production eager calls and returns total CUDA-event window milliseconds.
    ///
    /// Each iteration invokes the unchanged production no-op-observer path.
    /// Frozen identity validation stays outside the event interval; inherent
    /// production dispatch and cache lookup remain inside it. Divide the
    /// returned total by `iterations` to obtain time per eager call.
    ///
    /// # Errors
    ///
    /// Returns an error when `iterations` is zero, the holder no longer matches
    /// the context, request, route, policy, or resources, an eager launch fails,
    /// event creation or recording fails, or stream synchronization fails.
    pub fn measure_eager_window_ms(
        &mut self,
        ctx: &GpuCtx,
        iterations: usize,
    ) -> Result<f64, String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        self.evidence.validate_timed_request(ctx, self.request)?;
        match (self.request.route, &mut self.resources) {
            (
                PhysicalQualificationRoute::F32Policy(_),
                QualifiedPhysicalResources::F32(resources),
            ) => measure_production_eager(ctx, iterations, || {
                launch_f32_policy_eager(ctx, self.request, resources)
            }),
            (
                PhysicalQualificationRoute::Tf32Forced(_),
                QualifiedPhysicalResources::F32(resources),
            ) => {
                let prepared = resources
                    .prepared
                    .as_ref()
                    .ok_or_else(|| "forced TF32 holder has no prepared launch".to_string())?;
                measure_production_eager(ctx, iterations, || unsafe {
                    launch_prepared_f32_triad(ctx, prepared, |_| {
                        Err("forced TF32 performance route selected scalar code".into())
                    })
                })
            }
            (
                PhysicalQualificationRoute::HalfPolicy { dtype, .. },
                QualifiedPhysicalResources::Half(resources),
            ) => measure_production_eager(ctx, iterations, || {
                launch_half_policy_eager(ctx, self.request, resources, dtype)
            }),
            (
                PhysicalQualificationRoute::HalfForced { dtype, tile },
                QualifiedPhysicalResources::Half(resources),
            ) => measure_production_eager(ctx, iterations, || {
                launch_half_forced_eager(ctx, self.request, resources, dtype, tile)
            }),
            _ => Err("qualified route and retained resource kind disagree".into()),
        }
    }

    /// Measures a forced TF32 holder after validating it once before timing.
    ///
    /// This qualification-only comparator matches the trusted enqueue used by
    /// a hot production cache entry. The public validated timer remains useful
    /// for measuring validation overhead, but is not a kernel-throughput peer.
    pub fn measure_prevalidated_forced_eager_window_ms(
        &mut self,
        ctx: &GpuCtx,
        iterations: usize,
    ) -> Result<f64, String> {
        self.policy.validate(ctx)?;
        self.resources.validate()?;
        self.evidence.validate_timed_request(ctx, self.request)?;
        let (PhysicalQualificationRoute::Tf32Forced(_), QualifiedPhysicalResources::F32(resources)) =
            (self.request.route, &mut self.resources)
        else {
            return Err("prevalidated eager timing requires a forced TF32 holder".into());
        };
        let prepared = resources
            .prepared
            .as_ref()
            .ok_or_else(|| "forced TF32 holder has no prepared launch".to_string())?;
        validate_prepared_f32_triad_for_timing(ctx, prepared)?;
        measure_production_eager(ctx, iterations, || unsafe {
            enqueue_validated_prepared_f32_triad(ctx, prepared, |_| {
                Err("forced TF32 performance route selected scalar code".into())
            })
        })
    }
}

fn seeded_qualification_values(len: usize, salt: u64) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ salt;
    (0..len)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let signed = ((state.wrapping_add(index as u64) % 4093) as i32) - 2046;
            signed as f32 / 1024.0
        })
        .collect()
}

#[cfg(test)]
fn tf32_nt_qualification_reference(
    dims: (usize, usize, usize),
    salt: u64,
) -> Result<Vec<f64>, String> {
    let (m, k, n) = dims;
    let a = seeded_qualification_values(checked_elements(m, n, "M*N")?, salt ^ 0x2d);
    let b = seeded_qualification_values(checked_elements(k, n, "K*N")?, salt ^ 0x67);
    let mut expected = Vec::with_capacity(checked_elements(m, k, "M*K")?);
    for row in 0..m {
        for column in 0..k {
            let mut sum = 0.0_f64;
            for inner in 0..n {
                sum += tf32_value(a[row * n + inner]) * tf32_value(b[column * n + inner]);
            }
            expected.push(sum);
        }
    }
    Ok(expected)
}

struct PhysicalEvidenceContract<'a> {
    timed_request: PhysicalTimedRequestIdentity,
    eager_nodes: &'a [QualifiedPhysicalLaunchNode],
    graph_nodes: &'a [QualifiedPhysicalLaunchNode],
    eager_launch_count: u32,
    graph_launch_count: u32,
    eager_digest: [u8; 32],
    graph_digest: [u8; 32],
    physical_symbol: Option<&'static str>,
    tile: Option<(u32, u32)>,
}

impl<'a> PhysicalEvidenceContract<'a> {
    fn from_evidence(
        evidence: &'a QualifiedPhysicalLaunchEvidence,
        timed_request: PhysicalTimedRequestIdentity,
    ) -> Self {
        Self {
            timed_request,
            eager_nodes: &evidence.nodes,
            graph_nodes: &evidence.nodes,
            eager_launch_count: evidence.launch_count,
            graph_launch_count: evidence.launch_count,
            eager_digest: evidence.launch_digest,
            graph_digest: evidence.launch_digest,
            physical_symbol: evidence.single_launch_symbol,
            tile: evidence.single_launch_tile,
        }
    }
}

fn validate_physical_evidence_contract(
    contract: &PhysicalEvidenceContract<'_>,
) -> Result<(), String> {
    if contract.eager_nodes.is_empty() || contract.eager_launch_count == 0 {
        return Err("qualified physical evidence contains no actual launch".into());
    }
    if contract.eager_nodes != contract.graph_nodes
        || contract.eager_launch_count != contract.graph_launch_count
        || contract.eager_digest != contract.graph_digest
    {
        return Err("eager and graph physical evidence differ".into());
    }
    if usize::try_from(contract.eager_launch_count).ok() != Some(contract.eager_nodes.len()) {
        return Err("physical evidence count differs from its ordered node list".into());
    }
    if contract.eager_nodes.iter().any(|node| {
        node.logical_op != contract.timed_request.op
            || node.logical_dtype != contract.timed_request.logical_dtype
            || node.shape != contract.timed_request.dims
    }) {
        return Err("actual physical nodes differ from the timed request identity".into());
    }
    if let Some(symbol) = contract.timed_request.planned_symbol
        && !contract
            .eager_nodes
            .iter()
            .any(|node| node.symbol == symbol)
    {
        return Err("planned physical symbol was not present in the actual launch list".into());
    }
    if let Some(tile) = contract.timed_request.planned_tile
        && !contract
            .eager_nodes
            .iter()
            .any(|node| node.tile == Some(tile))
    {
        return Err("planned physical tile was not present in the actual launch list".into());
    }
    match contract.eager_nodes {
        [node] if contract.physical_symbol == Some(node.symbol) && contract.tile == node.tile => {}
        [_] => return Err("singular physical evidence does not match its only launch".into()),
        _ if contract.physical_symbol.is_none() && contract.tile.is_none() => {}
        _ => return Err("multi-launch physical evidence must not claim singular fields".into()),
    }
    Ok(())
}

fn half_policy_dtype(dtype: WeightDtype) -> Result<PolicyDtype, String> {
    match dtype {
        WeightDtype::Bf16 => Ok(PolicyDtype::Bf16),
        WeightDtype::F16 => Ok(PolicyDtype::F16),
        WeightDtype::F32 => Err("half production branch seal does not accept f32".into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NativeHalfBranchProjection {
    base: &'static str,
    op: ResolvedGemmOp,
    logical_dtype: PolicyDtype,
    execution_dtype: PolicyDtype,
    dims: (usize, usize, usize),
    strides: (usize, usize, usize),
    tile: Option<(u32, u32)>,
    bk_stages: (u32, u8),
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
}

impl NativeHalfBranchProjection {
    fn from_node(node: &ResolvedPhysicalKernelLaunch) -> Result<Self, String> {
        let route = node
            .gemm_route()
            .ok_or_else(|| "native half physical node has no GEMM route".to_string())?;
        let launch = node.launch();
        let suffix = match node.logical_dtype() {
            PolicyDtype::Bf16 => "_bf16",
            PolicyDtype::F16 => "_f16",
            PolicyDtype::F32 => {
                return Err("native half physical node unexpectedly executes f32".into());
            }
        };
        let base = node
            .symbol()
            .strip_suffix(suffix)
            .ok_or_else(|| "native half physical symbol has the wrong dtype suffix".to_string())?;
        Ok(Self {
            base,
            op: node.logical_op(),
            logical_dtype: node.logical_dtype(),
            execution_dtype: node.execution_dtype(),
            dims: node.shape(),
            strides: node.strides(),
            tile: node.tile(),
            bk_stages: (route.bk, route.stages),
            grid_dim: launch.grid_dim,
            block_dim: launch.block_dim,
            shared_mem_bytes: launch.shared_mem_bytes,
        })
    }
}

fn native_half_seal_projection(
    seal: HalfNativeBranchSeal,
) -> Result<NativeHalfBranchProjection, String> {
    let dtype = half_policy_dtype(seal.dtype)?;
    Ok(NativeHalfBranchProjection {
        base: seal.base,
        op: seal.op,
        logical_dtype: dtype,
        execution_dtype: dtype,
        dims: seal.dims,
        strides: seal.strides,
        tile: Some(seal.tile),
        bk_stages: seal.bk_stages,
        grid_dim: seal.grid_dim,
        block_dim: seal.block_dim,
        shared_mem_bytes: seal.shared_mem_bytes,
    })
}

fn validate_native_half_branch(
    seal: HalfNativeBranchSeal,
    nodes: &[ResolvedPhysicalKernelLaunch],
) -> Result<(), String> {
    let [node] = nodes else {
        return Err("native half production branch must match exactly one physical node".into());
    };
    let physical = NativeHalfBranchProjection::from_node(node)?;
    if node.kind() != PhysicalLaunchKind::Gemm || native_half_seal_projection(seal)? != physical {
        return Err(format!(
            "native half production seal differs from physical preflight: seal={seal:?}, physical={physical:?}"
        ));
    }
    Ok(())
}

fn validate_sm120_half_branch(
    seal: Sm120AutoBranchSeal,
    production_routes: &[ResolvedGemmRoute],
    eager_nodes: &[ResolvedPhysicalKernelLaunch],
    graph_nodes: &[ResolvedPhysicalKernelLaunch],
) -> Result<(), String> {
    let [expected] = production_routes else {
        return Err("SM120 half production branch must record exactly one production route".into());
    };
    let [eager] = eager_nodes else {
        return Err("SM120 half production branch must match exactly one eager node".into());
    };
    let expected_contract = match seal.route.physical.schedule {
        super::Sm120Schedule::Tiled => ResolvedNumericContract::MmaSyncF32V1,
        super::Sm120Schedule::StreamK => ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1,
    };
    let [graph] = graph_nodes else {
        return Err("SM120 half production branch must match exactly one graph node".into());
    };
    let spec = seal.route.kernel_spec()?;
    let expected_dtype = half_policy_dtype(seal.route.dtype)?;
    let expected_op = match seal.route.op {
        super::Sm120Op::Nn => ResolvedGemmOp::Nn,
        super::Sm120Op::Tn => ResolvedGemmOp::Tn,
        super::Sm120Op::Nt => ResolvedGemmOp::Nt,
    };
    let expected_shape = (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n);
    let expected_strides = (
        seal.route.shape.lda,
        seal.route.shape.ldb,
        seal.route.shape.ldc,
    );
    let expected_tile = (
        seal.route.physical.tile.output_rows(),
        seal.route.physical.tile.output_columns(),
    );
    if expected.op != expected_op
        || expected.dtype != expected_dtype
        || expected.backend != PhysicalGemmBackend::Sm120TmaMma16V1
        || expected.numeric_contract != expected_contract
        || expected.instruction_family != ResolvedInstructionFamily::MmaSync
        || expected.symbol != spec.symbol
        || expected.module_kind != ModuleKind::TriadSm120
        || expected.shape != expected_shape
        || expected.strides != expected_strides
        || expected.tile != expected_tile
        || expected.bk != seal.route.physical.bk.elements()
        || expected.stages != seal.route.physical.stages.count()
        || expected.threads != spec.threads
    {
        return Err(format!(
            "SM120 production route differs from its selected route: selected={:?}, production={expected:?}",
            seal.route
        ));
    }
    for (label, node) in [("eager", eager), ("graph", graph)] {
        let actual = node
            .gemm_route()
            .ok_or_else(|| format!("SM120 {label} node has no GEMM route"))?;
        if node.kind() != PhysicalLaunchKind::Gemm
            || node.module_kind() != ModuleKind::TriadSm120
            || node.logical_dtype() != expected_dtype
            || node.execution_dtype() != expected_dtype
            || actual.backend != PhysicalGemmBackend::Sm120TmaMma16V1
            || actual.numeric_contract != expected_contract
            || actual.instruction_family != ResolvedInstructionFamily::MmaSync
            || actual != *expected
        {
            return Err(format!(
                "SM120 {label} route differs from its exact production seal: expected={expected:?}, actual={actual:?}"
            ));
        }
    }
    if eager.gemm_route() != graph.gemm_route() || eager.launch() != graph.launch() {
        return Err("SM120 eager and graph nodes differ in their full prepared route".into());
    }
    Ok(())
}

fn validate_sm100_half_branch(
    seal: Sm100AutoBranchSeal,
    production_routes: &[ResolvedGemmRoute],
    eager_nodes: &[ResolvedPhysicalKernelLaunch],
    graph_nodes: &[ResolvedPhysicalKernelLaunch],
) -> Result<(), String> {
    let [expected] = production_routes else {
        return Err("SM100 half production branch must record exactly one production route".into());
    };
    let [eager] = eager_nodes else {
        return Err("SM100 half production branch must match exactly one eager node".into());
    };
    let [graph] = graph_nodes else {
        return Err("SM100 half production branch must match exactly one graph node".into());
    };
    let spec = seal.route.kernel_spec()?;
    let expected_dtype = half_policy_dtype(seal.route.dtype)?;
    let expected_op = match seal.route.op {
        super::Sm100Op::Nn => ResolvedGemmOp::Nn,
        super::Sm100Op::Tn => ResolvedGemmOp::Tn,
        super::Sm100Op::Nt => ResolvedGemmOp::Nt,
    };
    let expected_shape = (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n);
    let expected_strides = (
        seal.route.shape.lda,
        seal.route.shape.ldb,
        seal.route.shape.ldc,
    );
    let expected_tile = (
        seal.route.physical.tile.output_rows(),
        seal.route.physical.tile.output_columns(),
    );
    if expected.op != expected_op
        || expected.dtype != expected_dtype
        || expected.backend != PhysicalGemmBackend::Sm100Tcgen05V1
        || expected.numeric_contract != ResolvedNumericContract::Tcgen05F32V1
        || expected.instruction_family != ResolvedInstructionFamily::Tcgen05
        || expected.symbol != spec.symbol
        || expected.module_kind != ModuleKind::TriadSm100
        || expected.shape != expected_shape
        || expected.strides != expected_strides
        || expected.tile != expected_tile
        || expected.bk != spec.bk
        || expected.stages != seal.route.physical.stages.count()
        || expected.threads != spec.threads
    {
        return Err(format!(
            "SM100 production route differs from its selected route: selected={:?}, production={expected:?}",
            seal.route
        ));
    }
    for (label, node) in [("eager", eager), ("graph", graph)] {
        let actual = node
            .gemm_route()
            .ok_or_else(|| format!("SM100 {label} node has no GEMM route"))?;
        if node.kind() != PhysicalLaunchKind::Gemm
            || node.module_kind() != ModuleKind::TriadSm100
            || node.logical_dtype() != expected_dtype
            || node.execution_dtype() != expected_dtype
            || actual.backend != PhysicalGemmBackend::Sm100Tcgen05V1
            || actual.numeric_contract != ResolvedNumericContract::Tcgen05F32V1
            || actual.instruction_family != ResolvedInstructionFamily::Tcgen05
            || actual != *expected
        {
            return Err(format!(
                "SM100 {label} route differs from its exact production seal: expected={expected:?}, actual={actual:?}"
            ));
        }
    }
    if eager.gemm_route() != graph.gemm_route() || eager.launch() != graph.launch() {
        return Err("SM100 eager and graph nodes differ in their full prepared route".into());
    }
    Ok(())
}

fn validate_sm90a_half_branch(
    seal: Sm90aAutoBranchSeal,
    production_routes: &[ResolvedGemmRoute],
    eager_nodes: &[ResolvedPhysicalKernelLaunch],
    graph_nodes: &[ResolvedPhysicalKernelLaunch],
) -> Result<(), String> {
    let [expected] = production_routes else {
        return Err("SM90a half production branch must record exactly one production route".into());
    };
    let [eager] = eager_nodes else {
        return Err("SM90a half production branch must match exactly one eager node".into());
    };
    let [graph] = graph_nodes else {
        return Err("SM90a half production branch must match exactly one graph node".into());
    };
    let expected_dtype = half_policy_dtype(seal.route.dtype)?;
    let expected_op = match seal.route.op {
        super::Sm90aOp::Nn => ResolvedGemmOp::Nn,
        super::Sm90aOp::Tn => ResolvedGemmOp::Tn,
        super::Sm90aOp::Nt => ResolvedGemmOp::Nt,
    };
    if expected.op != expected_op
        || expected.dtype != expected_dtype
        || expected.backend != PhysicalGemmBackend::Sm90aWgmmaV1
        || expected.numeric_contract != ResolvedNumericContract::WgmmaF32V1
        || expected.instruction_family != ResolvedInstructionFamily::Wgmma
        || expected.symbol != seal.route.symbol()
        || expected.module_kind != ModuleKind::TriadSm90a
        || expected.shape != (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n)
        || expected.strides
            != (
                seal.route.shape.lda,
                seal.route.shape.ldb,
                seal.route.shape.ldc,
            )
        || expected.tile != (SM90A_TILE.0, SM90A_TILE.1)
        || expected.bk != SM90A_TILE.2
        || expected.stages != SM90A_STAGES
        || expected.threads != seal.route.schedule.threads()
    {
        return Err(format!(
            "SM90a production route differs from its selected route: selected={:?}, production={expected:?}",
            seal.route
        ));
    }
    for (label, node) in [("eager", eager), ("graph", graph)] {
        let actual = node
            .gemm_route()
            .ok_or_else(|| format!("SM90a {label} node has no GEMM route"))?;
        if node.kind() != PhysicalLaunchKind::Gemm
            || node.module_kind() != ModuleKind::TriadSm90a
            || node.logical_dtype() != expected_dtype
            || node.execution_dtype() != expected_dtype
            || actual.backend != PhysicalGemmBackend::Sm90aWgmmaV1
            || actual.numeric_contract != ResolvedNumericContract::WgmmaF32V1
            || actual.instruction_family != ResolvedInstructionFamily::Wgmma
            || actual != *expected
        {
            return Err(format!(
                "SM90a {label} route differs from its exact production seal: expected={expected:?}, actual={actual:?}"
            ));
        }
    }
    if eager.gemm_route() != graph.gemm_route() || eager.launch() != graph.launch() {
        return Err("SM90a eager and graph nodes differ in their full prepared route".into());
    }
    Ok(())
}

fn normalized_half_production_routes(
    seal: HalfPolicyBranchSeal,
    routes: &[ResolvedGemmRoute],
) -> Result<Box<[ResolvedGemmRoute]>, String> {
    if let HalfPolicyBranchSeal::Sm90a(seal) = seal {
        let [route] = routes else {
            return Err(format!(
                "SM90a production route recorder must contain exactly one route: actual={routes:?}"
            ));
        };
        if route.op
            != match seal.route.op {
                super::Sm90aOp::Nn => ResolvedGemmOp::Nn,
                super::Sm90aOp::Tn => ResolvedGemmOp::Tn,
                super::Sm90aOp::Nt => ResolvedGemmOp::Nt,
            }
            || route.dtype != half_policy_dtype(seal.route.dtype)?
            || route.shape != (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n)
            || route.strides
                != (
                    seal.route.shape.lda,
                    seal.route.shape.ldb,
                    seal.route.shape.ldc,
                )
            || route.tile != (SM90A_TILE.0, SM90A_TILE.1)
            || route.bk != SM90A_TILE.2
            || route.stages != SM90A_STAGES
        {
            return Err(format!(
                "SM90a production route recorder differs from its selected route: selected={:?}, actual={route:?}",
                seal.route
            ));
        }
        return Ok(routes.to_vec().into_boxed_slice());
    }
    if let HalfPolicyBranchSeal::Sm100(seal) = seal {
        let [route] = routes else {
            return Err(format!(
                "SM100 production route recorder must contain exactly one route: actual={routes:?}"
            ));
        };
        let spec = seal.route.kernel_spec()?;
        if route.op
            != match seal.route.op {
                super::Sm100Op::Nn => ResolvedGemmOp::Nn,
                super::Sm100Op::Tn => ResolvedGemmOp::Tn,
                super::Sm100Op::Nt => ResolvedGemmOp::Nt,
            }
            || route.dtype != half_policy_dtype(seal.route.dtype)?
            || route.shape != (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n)
            || route.strides
                != (
                    seal.route.shape.lda,
                    seal.route.shape.ldb,
                    seal.route.shape.ldc,
                )
            || route.tile
                != (
                    seal.route.physical.tile.output_rows(),
                    seal.route.physical.tile.output_columns(),
                )
            || route.bk != spec.bk
            || route.stages != seal.route.physical.stages.count()
        {
            return Err(format!(
                "SM100 production route recorder differs from its selected route: selected={:?}, actual={route:?}",
                seal.route
            ));
        }
        return Ok(routes.to_vec().into_boxed_slice());
    }
    if let HalfPolicyBranchSeal::Sm120(seal) = seal {
        let [route] = routes else {
            return Err(format!(
                "SM120 production route recorder must contain exactly one route: actual={routes:?}"
            ));
        };
        if route.op
            != match seal.route.op {
                super::Sm120Op::Nn => ResolvedGemmOp::Nn,
                super::Sm120Op::Tn => ResolvedGemmOp::Tn,
                super::Sm120Op::Nt => ResolvedGemmOp::Nt,
            }
            || route.dtype != half_policy_dtype(seal.route.dtype)?
            || route.shape != (seal.route.shape.m, seal.route.shape.k, seal.route.shape.n)
            || route.strides
                != (
                    seal.route.shape.lda,
                    seal.route.shape.ldb,
                    seal.route.shape.ldc,
                )
            || route.tile
                != (
                    seal.route.physical.tile.output_rows(),
                    seal.route.physical.tile.output_columns(),
                )
            || route.bk != seal.route.physical.bk.elements()
            || route.stages != seal.route.physical.stages.count()
        {
            return Err(format!(
                "SM120 production route recorder differs from its selected route: selected={:?}, actual={route:?}",
                seal.route
            ));
        }
    }
    Ok(routes.to_vec().into_boxed_slice())
}

fn validate_fallback_half_branch(
    op: ResolvedGemmOp,
    production_routes: &[ResolvedGemmRoute],
    nodes: &[ResolvedPhysicalKernelLaunch],
) -> Result<(), String> {
    if production_routes.is_empty() {
        return Err("half fallback production branch recorded no F32 route".into());
    }
    let downcast_count = usize::from(op != ResolvedGemmOp::Tn);
    let mut expected_kinds = Vec::with_capacity(2 + production_routes.len() + downcast_count);
    expected_kinds.extend([
        PhysicalLaunchKind::InputUpcast,
        PhysicalLaunchKind::InputUpcast,
    ]);
    expected_kinds.extend(std::iter::repeat_n(
        PhysicalLaunchKind::Gemm,
        production_routes.len(),
    ));
    if downcast_count != 0 {
        expected_kinds.push(PhysicalLaunchKind::OutputDowncast);
    }
    let physical_kinds = nodes
        .iter()
        .map(ResolvedPhysicalKernelLaunch::kind)
        .collect::<Vec<_>>();
    if physical_kinds != expected_kinds {
        return Err(format!(
            "half fallback physical envelope differs from the production branch: expected={expected_kinds:?}, physical={physical_kinds:?}"
        ));
    }
    let physical_routes = nodes
        .iter()
        .filter_map(ResolvedPhysicalKernelLaunch::gemm_route)
        .collect::<Vec<_>>();
    let same_routes = production_routes.len() == physical_routes.len()
        && production_routes
            .iter()
            .zip(&physical_routes)
            .all(|(production, physical)| same_fallback_physical_branch(*production, *physical));
    if production_routes.is_empty() || !same_routes {
        return Err(format!(
            "half fallback F32 route differs from physical preflight: production={production_routes:?}, physical={physical_routes:?}"
        ));
    }
    Ok(())
}

fn same_f32_physical_branch(
    mut production: ResolvedGemmRoute,
    mut physical: ResolvedGemmRoute,
) -> bool {
    production.resources_digest = [0; 32];
    physical.resources_digest = [0; 32];
    production.tensor_maps_digest = [0; 32];
    physical.tensor_maps_digest = [0; 32];
    production == physical
}

fn same_fallback_physical_branch(
    mut production: ResolvedGemmRoute,
    mut physical: ResolvedGemmRoute,
) -> bool {
    production.launch.arguments_digest = [0; 32];
    physical.launch.arguments_digest = [0; 32];
    same_f32_physical_branch(production, physical)
}

fn summarize_physical_evidence(
    trace: &RecordedPhysicalTrace,
    graph: &CapturedPhysicalGraph,
    production: ProductionBranchSeal,
    timed_request: PhysicalTimedRequestIdentity,
) -> Result<QualifiedPhysicalLaunchEvidence, String> {
    trace.validate_integrity()?;
    let eager_launches = trace.launches();
    let graph_launches = graph.launches();
    let eager_nodes = trace
        .nodes()
        .iter()
        .map(QualifiedPhysicalLaunchNode::from)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let graph_nodes = graph
        .nodes()
        .iter()
        .map(QualifiedPhysicalLaunchNode::from)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let contract = PhysicalEvidenceContract {
        timed_request,
        eager_nodes: &eager_nodes,
        graph_nodes: &graph_nodes,
        eager_launch_count: eager_launches.launch_count(),
        graph_launch_count: graph_launches.launch_count(),
        eager_digest: eager_launches.ordered_digest(),
        graph_digest: graph_launches.ordered_digest(),
        physical_symbol: eager_launches.physical_symbol(),
        tile: eager_launches.tile(),
    };
    validate_physical_evidence_contract(&contract)?;
    let physical_routes = trace
        .nodes()
        .iter()
        .filter_map(ResolvedPhysicalKernelLaunch::gemm_route)
        .collect::<Vec<_>>();
    match &production {
        ProductionBranchSeal::F32(production_routes) => {
            let same_branch = production_routes.len() == physical_routes.len()
                && production_routes
                    .iter()
                    .zip(&physical_routes)
                    .all(|(production, physical)| same_f32_physical_branch(*production, *physical));
            if production_routes.is_empty() || !same_branch {
                return Err(format!(
                    "production no-op wrapper selected a different branch than physical preflight: production={production_routes:?}, physical={physical_routes:?}"
                ));
            }
        }
        ProductionBranchSeal::Half(branch) => match branch.seal {
            HalfPolicyBranchSeal::Native(seal) => {
                if !branch.production_routes.is_empty() {
                    return Err(
                        "native half production branch unexpectedly recorded F32 routes".into(),
                    );
                }
                validate_native_half_branch(seal, trace.nodes())?;
            }
            HalfPolicyBranchSeal::Sm120(seal) => validate_sm120_half_branch(
                seal,
                &branch.production_routes,
                trace.nodes(),
                graph.nodes(),
            )?,
            HalfPolicyBranchSeal::Sm100(seal) => validate_sm100_half_branch(
                seal,
                &branch.production_routes,
                trace.nodes(),
                graph.nodes(),
            )?,
            HalfPolicyBranchSeal::Sm90a(seal) => validate_sm90a_half_branch(
                seal,
                &branch.production_routes,
                trace.nodes(),
                graph.nodes(),
            )?,
            HalfPolicyBranchSeal::ExactF32Fallback => validate_fallback_half_branch(
                timed_request.op,
                &branch.production_routes,
                trace.nodes(),
            )?,
        },
    }
    let execution_dtype = eager_nodes
        .first()
        .map(|node| node.execution_dtype)
        .filter(|dtype| {
            eager_nodes
                .iter()
                .all(|node| node.execution_dtype == *dtype)
        });
    let module_kind = eager_nodes
        .first()
        .map(|node| node.module_kind)
        .filter(|kind| eager_nodes.iter().all(|node| node.module_kind == *kind));
    Ok(QualifiedPhysicalLaunchEvidence {
        evidence_scope: PHYSICAL_EVIDENCE_SCOPE,
        route_identity: trace.context(),
        timed_request,
        nodes: eager_nodes,
        production_branch: production,
        request_identity_digest: timed_request_digest(timed_request),
        launch_count: eager_launches.launch_count(),
        launch_digest: eager_launches.ordered_digest(),
        single_launch_symbol: eager_launches.physical_symbol(),
        single_launch_tile: eager_launches.tile(),
        uniform_execution_dtype: execution_dtype,
        uniform_module_kind: module_kind,
        eager_graph_equal: true,
    })
}

fn checked_elements(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("qualification {label} extent overflows usize"))
}

fn allocate_f32_resources(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
) -> Result<QualifiedF32Resources, String> {
    let (m, k, n) = request.dims;
    let mk = checked_elements(m, k, "M*K")?;
    let kn = checked_elements(k, n, "K*N")?;
    let mn = checked_elements(m, n, "M*N")?;
    let (output_len, a_len, b_len) = match request.op {
        ResolvedGemmOp::Nn => (mn, mk, kn),
        ResolvedGemmOp::Tn => (kn, mk, mn),
        ResolvedGemmOp::Nt => (mk, mn, kn),
    };
    let output = GpuBuffer::zeros(&ctx.stream, output_len)?;
    let a = GpuBuffer::zeros(&ctx.stream, a_len)?;
    let b = GpuBuffer::zeros(&ctx.stream, b_len)?;
    let epilogue = request.f32_epilogue();
    let bias = if epilogue.bias {
        Some(GpuBuffer::zeros(&ctx.stream, n)?)
    } else {
        None
    };
    let binding = (
        output.cached_ptr(),
        a.cached_ptr(),
        b.cached_ptr(),
        bias.as_ref().map(GpuBuffer::cached_ptr),
    );
    let operands = F32TriadOperands {
        output: binding.0,
        a: binding.1,
        b: binding.2,
        bias: binding.3,
        alpha: epilogue.alpha,
        beta: epilogue.beta,
    };
    let triad_request = F32TriadRequest {
        op: request.op,
        shape: F32TriadShape::contiguous(request.op, request.dims),
    };
    let prepared = match request.route {
        PhysicalQualificationRoute::F32Policy(_) => None,
        PhysicalQualificationRoute::Tf32Forced(route) => Some(prepare_f32_triad_forced(
            ctx,
            triad_request,
            operands,
            route,
        )?),
        _ => return Err("non-F32 route requested F32 qualification resources".into()),
    };
    Ok(QualifiedF32Resources {
        output,
        a,
        b,
        bias,
        prepared,
        binding,
    })
}

fn allocate_half_resources(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
    dtype: WeightDtype,
) -> Result<QualifiedHalfResources, String> {
    if dtype == WeightDtype::F32 {
        return Err("half qualification route requires BF16 or F16".into());
    }
    let shape = request.physical_shape();
    // Qualification allocations always retain a trailing red zone. Offset and
    // padded layouts add their own leading or row-padding guards on top.
    let guarded = true;
    let (output_geometry, a_geometry, b_geometry) = match request.op {
        ResolvedGemmOp::Nn => (
            (shape.m, shape.n, shape.ldc),
            (shape.m, shape.k, shape.lda),
            (shape.k, shape.n, shape.ldb),
        ),
        ResolvedGemmOp::Tn => (
            (shape.k, shape.n, shape.ldc),
            (shape.m, shape.k, shape.lda),
            (shape.m, shape.n, shape.ldb),
        ),
        ResolvedGemmOp::Nt => (
            (shape.m, shape.k, shape.ldc),
            (shape.m, shape.n, shape.lda),
            (shape.k, shape.n, shape.ldb),
        ),
    };
    let offsets = request.element_offsets();
    let storage = QualifiedHalfStorage {
        output: QualificationStorage::new(
            output_geometry.0,
            output_geometry.1,
            output_geometry.2,
            offsets.0,
            guarded,
        )?,
        a: QualificationStorage::new(a_geometry.0, a_geometry.1, a_geometry.2, offsets.1, guarded)?,
        b: QualificationStorage::new(b_geometry.0, b_geometry.1, b_geometry.2, offsets.2, guarded)?,
    };
    let output = if request.op == ResolvedGemmOp::Tn {
        let mut buffer = GpuBuffer::zeros(&ctx.stream, storage.output.len)?;
        if guarded {
            buffer.upload(&ctx.stream, &storage.output.host_values())?;
        }
        QualifiedHalfOutput::F32(buffer)
    } else {
        let buffer = DtypedBuf::zeros(&ctx.stream, storage.output.len, dtype)?;
        if guarded {
            buffer.upload_f32(&ctx.stream, &storage.output.host_values())?;
        }
        QualifiedHalfOutput::Typed(buffer)
    };
    let a = DtypedBuf::zeros(&ctx.stream, storage.a.len, dtype)?;
    let b = DtypedBuf::zeros(&ctx.stream, storage.b.len, dtype)?;
    if guarded {
        a.upload_f32(&ctx.stream, &storage.a.host_values())?;
        b.upload_f32(&ctx.stream, &storage.b.host_values())?;
    }
    let binding = (
        output.ptr() + (storage.output.offset * output.element_bytes()) as u64,
        a.cached_ptr() + (storage.a.offset * dtype.size_bytes()) as u64,
        b.cached_ptr() + (storage.b.offset * dtype.size_bytes()) as u64,
    );
    Ok(QualifiedHalfResources {
        output,
        a,
        b,
        binding,
        storage,
    })
}

fn half_trace_request(
    request: PhysicalQualificationRequest,
    resources: &QualifiedHalfResources,
    dtype: WeightDtype,
) -> HalfPhysicalTraceRequest {
    HalfPhysicalTraceRequest {
        op: request.op,
        output: resources.binding.0,
        a: resources.binding.1,
        b: resources.binding.2,
        bias: 0,
        dtype,
        dims: request.dims,
        nn_strides: match request.layout {
            PhysicalQualificationLayout::DirectPaddedNn(layout) => Some(layout.strides()),
            _ => None,
        },
        forced_tile: match request.route {
            PhysicalQualificationRoute::HalfForced { tile, .. } => Some(tile),
            _ => None,
        },
        capacity: PHYSICAL_PREFLIGHT_CAPACITY,
    }
}

fn f32_resource_operands(
    request: PhysicalQualificationRequest,
    resources: &QualifiedF32Resources,
) -> F32TriadOperands {
    let epilogue = request.f32_epilogue();
    F32TriadOperands {
        output: resources.binding.0,
        a: resources.binding.1,
        b: resources.binding.2,
        bias: resources.binding.3,
        alpha: epilogue.alpha,
        beta: epilogue.beta,
    }
}

fn qualify_half_launch<'ctx>(
    ctx: &'ctx GpuCtx,
    request: PhysicalQualificationRequest,
    dtype: WeightDtype,
    resources: QualifiedHalfResources,
    policy: PhysicalQualificationPolicyLease<'ctx>,
) -> Result<QualifiedPhysicalLaunch<'ctx>, String> {
    let resources = QualifiedPhysicalResources::Half(Box::new(resources));
    let retained = match &resources {
        QualifiedPhysicalResources::Half(resources) => resources,
        QualifiedPhysicalResources::F32(_) => unreachable!(),
    };
    let mut physical = half_trace_request(request, retained, dtype);
    let production_seal = Cell::new(None);
    let production_result = ctx.record_eager_gemm_trace(|| {
        let seal = launch_half_production_branch(ctx, physical)?;
        production_seal.set(Some(seal));
        Ok(())
    });
    let production_trace =
        synchronized_enqueue_result(ctx, "production branch proof", production_result)?;
    let seal = production_seal
        .get()
        .ok_or_else(|| "production half enqueue returned no branch seal".to_string())?;
    let production = HalfProductionBranch {
        seal,
        production_routes: normalized_half_production_routes(seal, production_trace.routes())?,
    };
    let trace_result = unsafe { record_half_physical_trace(ctx, physical) };
    let trace = synchronized_enqueue_result(ctx, "physical eager preflight", trace_result)?;
    let manifest = trace.manifest();
    physical.capacity = manifest.launch_capacity();
    let package = prepare_half_physical_graph_package(ctx, physical, &manifest)?;
    let graph = unsafe { capture_into_graph_with_physical_plan(package) }?;
    let evidence = summarize_physical_evidence(
        &trace,
        &graph,
        ProductionBranchSeal::Half(production),
        request.timed_identity()?,
    )?;
    Ok(QualifiedPhysicalLaunch {
        evidence,
        request,
        graph,
        resources,
        policy,
    })
}

fn prepare_f32_preflight<'ctx>(
    ctx: &'ctx GpuCtx,
    resources: &mut QualifiedF32Resources,
    prepared: &PreparedF32TriadLaunch,
) -> Result<(RecordedPhysicalTrace, PreparedPhysicalGraphPackage<'ctx>), String> {
    let capacity = prepared.physical_graph_launch_count();
    let trace_result = unsafe {
        record_prepared_f32_physical_trace(
            ctx,
            F32PhysicalGraphPackageRequest {
                prepared,
                output: &mut resources.output,
                a: &resources.a,
                b: &resources.b,
                capacity,
            },
        )
    };
    let trace = synchronized_enqueue_result(ctx, "physical eager preflight", trace_result)?;
    let manifest = trace.manifest();
    let package = prepare_f32_physical_graph_package(
        ctx,
        F32PhysicalGraphPackageRequest {
            prepared,
            output: &mut resources.output,
            a: &resources.a,
            b: &resources.b,
            capacity,
        },
        &manifest,
    )?;
    Ok((trace, package))
}

fn qualify_f32_launch<'ctx>(
    ctx: &'ctx GpuCtx,
    request: PhysicalQualificationRequest,
    mut resources: QualifiedF32Resources,
    policy: PhysicalQualificationPolicyLease<'ctx>,
) -> Result<QualifiedPhysicalLaunch<'ctx>, String> {
    let production_result = ctx.record_eager_gemm_trace(|| match request.route {
        PhysicalQualificationRoute::F32Policy(_) => {
            launch_f32_policy_eager(ctx, request, &mut resources)
        }
        PhysicalQualificationRoute::Tf32Forced(_) => {
            let prepared = resources
                .prepared
                .as_ref()
                .ok_or_else(|| "forced TF32 holder has no prepared launch".to_string())?;
            unsafe {
                launch_prepared_f32_triad(ctx, prepared, |_| {
                    Err("forced TF32 branch proof selected scalar code".into())
                })
            }
        }
        _ => Err("non-F32 route requested F32 branch proof".into()),
    });
    let production =
        synchronized_enqueue_result(ctx, "production branch proof", production_result)?;
    let triad_request = F32TriadRequest {
        op: request.op,
        shape: F32TriadShape::contiguous(request.op, request.dims),
    };
    let operands = f32_resource_operands(request, &resources);
    let (trace, package) = match request.route {
        PhysicalQualificationRoute::F32Policy(_) => {
            with_cached_f32_triad_prepared(ctx, triad_request, operands, |prepared| {
                prepare_f32_preflight(ctx, &mut resources, prepared)
            })?
        }
        PhysicalQualificationRoute::Tf32Forced(_) => {
            let prepared = resources
                .prepared
                .take()
                .ok_or_else(|| "forced TF32 holder has no prepared launch".to_string())?;
            let result = prepare_f32_preflight(ctx, &mut resources, &prepared);
            resources.prepared = Some(prepared);
            result?
        }
        _ => return Err("non-F32 route requested F32 physical preflight".into()),
    };
    let graph = unsafe { capture_into_graph_with_physical_plan(package) }?;
    let evidence = summarize_physical_evidence(
        &trace,
        &graph,
        ProductionBranchSeal::F32(production.routes().to_vec().into_boxed_slice()),
        request.timed_identity()?,
    )?;
    Ok(QualifiedPhysicalLaunch {
        evidence,
        request,
        graph,
        resources: QualifiedPhysicalResources::F32(Box::new(resources)),
        policy,
    })
}

#[doc(hidden)]
/// Qualifies one production request and captures its exact graph.
///
/// Construction performs an untimed production branch proof and one real
/// physical recording preflight. The returned holder owns every allocation
/// and prepared resource referenced by its graph. Use a dedicated context:
/// only one holder may be live for a context, and the holder retains a policy
/// lease until drop. The normalized policy is restored when the holder drops.
/// F32 `AllowDeterministicTf32V1` is permission rather than a forced route and
/// may resolve to exact scalar FMA. Forced TF32 and half routes must satisfy
/// their architecture, dtype, tile, operation, and layout contracts.
///
/// # Errors
///
/// Returns an error for an invalid request, a conflicting live holder, policy
/// or device incompatibility, allocation/preparation failure, or any mismatch
/// between eager physical observation and the captured graph.
pub fn qualify_physical_launch(
    ctx: &GpuCtx,
    request: PhysicalQualificationRequest,
) -> Result<QualifiedPhysicalLaunch<'_>, String> {
    request.validate()?;
    let policy = begin_physical_qualification_policy(ctx, request.route.policy())?;
    policy.validate(ctx)?;
    match request.route {
        PhysicalQualificationRoute::F32Policy(_) | PhysicalQualificationRoute::Tf32Forced(_) => {
            let resources = allocate_f32_resources(ctx, request)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("finish F32 qualification allocations: {error:?}"))?;
            qualify_f32_launch(ctx, request, resources, policy)
        }
        PhysicalQualificationRoute::HalfPolicy { dtype, .. }
        | PhysicalQualificationRoute::HalfForced { dtype, .. } => {
            let resources = allocate_half_resources(ctx, request, dtype)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("finish half qualification allocations: {error:?}"))?;
            qualify_half_launch(ctx, request, dtype, resources, policy)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ExceptionalClass {
    Nan,
    Infinity { negative: bool },
    Zero,
    Finite(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GuardedStorage {
    rows: usize,
    width: usize,
    stride: usize,
    offset: usize,
    origin_x: usize,
    origin_y: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AdversarialCounts {
    staged_cases: usize,
    guard_poison_pairs: usize,
    exceptional_symbols: usize,
    cross_m_symbols: usize,
    cross_m_shapes: usize,
    nn_bias_beta_symbols: usize,
}

fn round_up_four(value: usize) -> Result<usize, String> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| "qualification stride padding overflows usize".to_owned())
}

fn padded_shape(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    padding: usize,
) -> Result<F32TriadShape, String> {
    let mut shape = F32TriadShape::contiguous(op, dims);
    let (a_width, b_width, output_width) = match op {
        ResolvedGemmOp::Nn => (shape.k, shape.n, shape.n),
        ResolvedGemmOp::Tn => (shape.k, shape.n, shape.n),
        ResolvedGemmOp::Nt => (shape.n, shape.n, shape.k),
    };
    shape.lda = round_up_four(
        a_width
            .checked_add(padding)
            .ok_or_else(|| "qualification A padding overflows usize".to_owned())?,
    )?;
    shape.ldb = round_up_four(
        b_width
            .checked_add(padding)
            .ok_or_else(|| "qualification B padding overflows usize".to_owned())?,
    )?;
    shape.ldc = round_up_four(
        output_width
            .checked_add(padding)
            .ok_or_else(|| "qualification output padding overflows usize".to_owned())?,
    )?;
    shape.validate(op)?;
    Ok(shape)
}

fn shape_for_spec(
    spec: &Tf32KernelSpec,
    dims: (usize, usize, usize),
) -> Result<F32TriadShape, String> {
    if matches!(
        spec.route,
        Tf32PhysicalRoute::MmaTf32RnaV1(route) if route.tile != Tf32PortableTile::M128N128
    ) {
        Ok(F32TriadShape::contiguous(spec.op, dims))
    } else {
        // The wide portable tile uses 16-byte copies, like TMA routes it
        // needs aligned physical A/B strides even for logical K/N tails.
        padded_shape(spec.op, dims, 1)
    }
}

fn guarded_storage(
    rows: usize,
    width: usize,
    stride: usize,
    origin_x: usize,
) -> Result<GuardedStorage, String> {
    if rows == 0 || width == 0 || origin_x == 0 || origin_x + width > stride {
        return Err("invalid guarded qualification storage geometry".into());
    }
    let offset = stride
        .checked_add(origin_x)
        .ok_or_else(|| "guarded qualification offset overflows usize".to_owned())?;
    let active_end = (rows - 1)
        .checked_mul(stride)
        .and_then(|tail| offset.checked_add(tail))
        .and_then(|tail| tail.checked_add(width))
        .ok_or_else(|| "guarded qualification storage span overflows usize".to_owned())?;
    let len = active_end
        .checked_add(GUARD_ELEMENTS)
        .ok_or_else(|| "guarded qualification red zone overflows usize".to_owned())?;
    Ok(GuardedStorage {
        rows,
        width,
        stride,
        offset,
        origin_x,
        origin_y: 1,
        len,
    })
}

fn exceptional_class(value: f32) -> ExceptionalClass {
    if value.is_nan() {
        return ExceptionalClass::Nan;
    }
    let converted = tf32_value(value);
    if converted.is_infinite() {
        ExceptionalClass::Infinity {
            negative: converted.is_sign_negative(),
        }
    } else if converted == 0.0 {
        ExceptionalClass::Zero
    } else {
        ExceptionalClass::Finite(converted)
    }
}

fn exceptional_class_for_spec(spec: &Tf32KernelSpec, value: f32) -> ExceptionalClass {
    if spec.operand_conversion != ResolvedOperandConversion::RegisterAddHalfUlpTf32V1 {
        return exceptional_class(value);
    }

    // This route does an unconditional wrapping integer add, then MMA
    // consumes only the upper 19 bits. Classify that operand, not the input:
    // low-payload NaNs can become infinities, and payload carry can wrap
    // across the sign bit to zero. Do not change the older route oracles.
    let converted = f32::from_bits(value.to_bits().wrapping_add(0x1000) & 0xffff_e000);
    if converted.is_nan() {
        ExceptionalClass::Nan
    } else if converted.is_infinite() {
        ExceptionalClass::Infinity {
            negative: converted.is_sign_negative(),
        }
    } else if converted == 0.0 {
        ExceptionalClass::Zero
    } else {
        ExceptionalClass::Finite(f64::from(converted))
    }
}

fn exceptional_values_for_spec(spec: &Tf32KernelSpec) -> &'static [u32] {
    if spec.operand_conversion == ResolvedOperandConversion::RegisterAddHalfUlpTf32V1 {
        return &[
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0x7f81_2345,
            0x0000_0000,
            0x8000_0001,
            0x807f_ffff,
            0xff7f_ffff,
            0x3f80_1000,
            0xbf80_1000,
            0x7f80_0001,
            0xff80_0001,
            0x7f80_1000,
            0xff80_1000,
            0x7f80_2000,
            0xff80_2000,
            0x7fff_ffff,
            0xffff_ffff,
            0x7fc0_1234,
            0xffc0_1234,
        ];
    }
    &[
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0x7f81_2345,
    ]
}

fn adversarial_counts(specs: &[&Tf32KernelSpec]) -> AdversarialCounts {
    let nn = specs
        .iter()
        .filter(|spec| spec.op == ResolvedGemmOp::Nn)
        .count();
    let independent_rows = specs
        .iter()
        .filter(|spec| matches!(spec.op, ResolvedGemmOp::Nn | ResolvedGemmOp::Nt))
        .count();
    AdversarialCounts {
        staged_cases: specs.len() * STAGED_K_CASES.len(),
        guard_poison_pairs: specs.len(),
        exceptional_symbols: specs.len(),
        cross_m_symbols: independent_rows,
        cross_m_shapes: independent_rows * 4,
        nn_bias_beta_symbols: nn,
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tf32QualificationSuite {
    Full,
    Sanitizer,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tf32QualificationConfig {
    pub exact_cc: (u32, u32),
    pub expected_routes: usize,
    pub repeat: u32,
    pub suite: Tf32QualificationSuite,
}

#[doc(hidden)]
pub struct Tf32QualificationOutput {
    pub report_json: String,
    pub artifact: Vec<u8>,
    pub driver_abi_proof: String,
}

struct HostCase {
    request: F32TriadRequest,
    a: Vec<f32>,
    b: Vec<f32>,
    bias: Option<Vec<f32>>,
    initial: Vec<f32>,
    expected: Vec<f64>,
    alpha: f32,
    beta: f32,
}

#[derive(Clone, Copy)]
struct EpilogueCase {
    alpha: f32,
    beta: f32,
    bias: bool,
}

struct DeviceCase {
    a: GpuBuffer,
    b: GpuBuffer,
    bias: Option<GpuBuffer>,
    output: GpuBuffer,
}

struct GuardedHostCase {
    request: F32TriadRequest,
    a: Vec<f32>,
    b: Vec<f32>,
    bias: Option<Vec<f32>>,
    initial: Vec<f32>,
    expected: Vec<f64>,
    a_storage: GuardedStorage,
    b_storage: GuardedStorage,
    bias_storage: Option<GuardedStorage>,
    output_storage: GuardedStorage,
    alpha: f32,
    beta: f32,
}

struct GuardedDeviceCase {
    a: GpuBuffer,
    b: GpuBuffer,
    bias: Option<GpuBuffer>,
    output: GpuBuffer,
}

impl GuardedDeviceCase {
    fn operands(&self, host: &GuardedHostCase, ctx: &GpuCtx) -> F32TriadOperands {
        F32TriadOperands {
            output: self
                .output
                .raw_ptr_at(&ctx.stream, host.output_storage.offset),
            a: self.a.raw_ptr_at(&ctx.stream, host.a_storage.offset),
            b: self.b.raw_ptr_at(&ctx.stream, host.b_storage.offset),
            bias: self
                .bias
                .as_ref()
                .zip(host.bias_storage)
                .map(|(bias, storage)| bias.raw_ptr_at(&ctx.stream, storage.offset)),
            alpha: host.alpha,
            beta: host.beta,
        }
    }
}

fn physical_layout(shape: F32TriadShape, op: ResolvedGemmOp) -> [(usize, usize, usize); 3] {
    match op {
        ResolvedGemmOp::Nn => [
            (shape.m, shape.k, shape.lda),
            (shape.k, shape.n, shape.ldb),
            (shape.m, shape.n, shape.ldc),
        ],
        ResolvedGemmOp::Tn => [
            (shape.m, shape.k, shape.lda),
            (shape.m, shape.n, shape.ldb),
            (shape.k, shape.n, shape.ldc),
        ],
        ResolvedGemmOp::Nt => [
            (shape.m, shape.n, shape.lda),
            (shape.k, shape.n, shape.ldb),
            (shape.m, shape.k, shape.ldc),
        ],
    }
}

fn copy_active_rows(destination: &mut [f32], storage: GuardedStorage, source: &[f32]) {
    for row in 0..storage.rows {
        let source_start = row * storage.stride;
        let destination_start = storage.offset + source_start;
        destination[destination_start..destination_start + storage.width]
            .copy_from_slice(&source[source_start..source_start + storage.width]);
    }
}

fn build_guarded_case(
    op: ResolvedGemmOp,
    dims: (usize, usize, usize),
    salt: usize,
    epilogue: EpilogueCase,
    poison_bits: u32,
) -> Result<GuardedHostCase, String> {
    let shape = padded_shape(op, dims, 8)?;
    let request = F32TriadRequest { op, shape };
    let logical = build_host_case(request, salt, epilogue);
    let [a_layout, b_layout, output_layout] = physical_layout(shape, op);
    let a_storage = guarded_storage(a_layout.0, a_layout.1, a_layout.2, 4)?;
    let b_storage = guarded_storage(b_layout.0, b_layout.1, b_layout.2, 8)?;
    let output_storage = guarded_storage(output_layout.0, output_layout.1, output_layout.2, 3)?;
    let poison = f32::from_bits(poison_bits);
    let output_canary = f32::from_bits(0x4b7f_a55a);
    let mut a = vec![poison; a_storage.len];
    let mut b = vec![poison; b_storage.len];
    let mut initial = vec![output_canary; output_storage.len];
    copy_active_rows(&mut a, a_storage, &logical.a);
    copy_active_rows(&mut b, b_storage, &logical.b);
    copy_active_rows(&mut initial, output_storage, &logical.initial);

    let (bias, bias_storage) = if let Some(logical_bias) = logical.bias.as_deref() {
        let stride = round_up_four(
            logical_bias
                .len()
                .checked_add(8)
                .ok_or_else(|| "qualification bias padding overflows usize".to_owned())?,
        )?;
        let storage = guarded_storage(1, logical_bias.len(), stride, 1)?;
        let mut values = vec![poison; storage.len];
        values[storage.offset..storage.offset + storage.width].copy_from_slice(logical_bias);
        (Some(values), Some(storage))
    } else {
        (None, None)
    };

    Ok(GuardedHostCase {
        request,
        a,
        b,
        bias,
        initial,
        expected: logical.expected,
        a_storage,
        b_storage,
        bias_storage,
        output_storage,
        alpha: epilogue.alpha,
        beta: epilogue.beta,
    })
}

fn upload_guarded_case(ctx: &GpuCtx, host: &GuardedHostCase) -> Result<GuardedDeviceCase, String> {
    Ok(GuardedDeviceCase {
        a: GpuBuffer::from_cpu(&ctx.stream, &host.a)?,
        b: GpuBuffer::from_cpu(&ctx.stream, &host.b)?,
        bias: host
            .bias
            .as_deref()
            .map(|values| GpuBuffer::from_cpu(&ctx.stream, values))
            .transpose()?,
        output: GpuBuffer::from_cpu(&ctx.stream, &host.initial)?,
    })
}

fn guarded_output_bits(ctx: &GpuCtx, device: &GuardedDeviceCase) -> Result<Vec<u32>, String> {
    output_bits(ctx, &device.output)
}

fn check_guarded_output(
    host: &GuardedHostCase,
    actual: &[u32],
    label: &str,
) -> Result<Vec<u32>, String> {
    if actual.len() != host.initial.len() {
        return Err(format!("{label}: guarded output length changed"));
    }
    let storage = host.output_storage;
    let mut active = Vec::with_capacity(storage.rows * storage.width);
    let mut active_mask = vec![false; actual.len()];
    for row in 0..storage.rows {
        for column in 0..storage.width {
            let backing_index = storage.offset + row * storage.stride + column;
            let logical_index = row * storage.stride + column;
            let actual_value = f64::from(f32::from_bits(actual[backing_index]));
            let expected = host.expected[logical_index];
            let tolerance = 0.0025 * (1.0 + expected.abs());
            if !actual_value.is_finite() || (actual_value - expected).abs() > tolerance {
                return Err(format!(
                    "{label}: guarded output ({row},{column}) was {actual_value}, reference {expected}, tolerance {tolerance}"
                ));
            }
            active_mask[backing_index] = true;
            active.push(actual[backing_index]);
        }
    }
    for (index, (&actual, initial)) in actual
        .iter()
        .zip(host.initial.iter().map(|value| value.to_bits()))
        .enumerate()
    {
        if !active_mask[index] && actual != initial {
            return Err(format!(
                "{label}: output red zone changed at element {index}"
            ));
        }
    }
    Ok(active)
}

impl DeviceCase {
    fn operands(&self, host: &HostCase) -> F32TriadOperands {
        F32TriadOperands {
            output: self.output.cached_ptr(),
            a: self.a.cached_ptr(),
            b: self.b.cached_ptr(),
            bias: self.bias.as_ref().map(GpuBuffer::cached_ptr),
            alpha: host.alpha,
            beta: host.beta,
        }
    }
}

struct RouteEvidence {
    symbol: &'static str,
    eager_digest: [u8; 32],
    graph_digest: [u8; 32],
    zero_digest: [u8; 32],
    boundary_digest: [u8; 32],
    eager_route_digest: [u8; 32],
    graph_route_digest: [u8; 32],
    tensor_map_digest: [u8; 32],
    staged_digest: [u8; 32],
    exceptional_digest: [u8; 32],
    cross_m_digest: [u8; 32],
    adversarial_counts: AdversarialCounts,
    eager_ms: f32,
    graph_ms: f32,
    registers: u32,
    driver_jit_local_bytes: u32,
    approved_driver_jit_local_cap: u32,
}

struct AdversarialRouteEvidence {
    staged_digest: [u8; 32],
    exceptional_digest: [u8; 32],
    cross_m_digest: [u8; 32],
    counts: AdversarialCounts,
}

fn beta_for_op(op: ResolvedGemmOp) -> f32 {
    match op {
        ResolvedGemmOp::Nn | ResolvedGemmOp::Nt => 0.0,
        ResolvedGemmOp::Tn => 1.0,
    }
}

fn primary_epilogue(op: ResolvedGemmOp) -> EpilogueCase {
    EpilogueCase {
        alpha: 1.0,
        beta: beta_for_op(op),
        bias: op == ResolvedGemmOp::Nn,
    }
}

fn adversarial_epilogue(op: ResolvedGemmOp) -> EpilogueCase {
    EpilogueCase {
        alpha: -0.75,
        beta: match op {
            ResolvedGemmOp::Nn => 0.375,
            ResolvedGemmOp::Tn => 1.0,
            ResolvedGemmOp::Nt => 0.0,
        },
        bias: false,
    }
}

fn nn_bias_beta_epilogue() -> EpilogueCase {
    EpilogueCase {
        alpha: 1.0,
        beta: -0.5,
        bias: true,
    }
}

/// TF32 route-spec corpus composed for the qualification device, in order.
/// Exact-FMA and separate portable split-K spec collections are excluded.
#[doc(hidden)]
pub fn tf32_qualification_route_specs(
    cc: (u32, u32),
) -> Result<Vec<&'static Tf32KernelSpec>, String> {
    let specialized = match cc {
        (8, 0 | 6 | 7 | 9) => &[][..],
        (9, 0) => &SM90A_TF32_ROUTE_SPECS,
        (10, 0 | 3) | (11, 0) => &SM100_TF32_ROUTE_SPECS,
        (12, 0 | 1) => &SM120_TF32_ROUTE_SPECS,
        _ => {
            return Err(format!(
                "unsupported TF32 qualification CC {}.{}",
                cc.0, cc.1
            ));
        }
    };
    // The exact-F32 SM120 routes share the module but not the TF32 numeric
    // contract; their qualification is the exact-family harness, not the
    // TF32 selector.
    Ok(tf32_route_specs_for(
        ModuleKind::TriadSm80,
        portable_extensions_composed_for_cc(cc),
    )
    .chain(specialized.iter())
    .filter(|spec| !spec.route.is_exact_fma())
    .collect())
}

fn logical_dims(
    op: ResolvedGemmOp,
    rows: usize,
    columns: usize,
    reduction: usize,
) -> (usize, usize, usize) {
    match op {
        ResolvedGemmOp::Nn => (rows, reduction, columns),
        ResolvedGemmOp::Tn => (reduction, rows, columns),
        ResolvedGemmOp::Nt => (rows, columns, reduction),
    }
}

fn input_value(index: usize, salt: usize) -> f32 {
    let mixed = index
        .wrapping_mul(1_664_525)
        .wrapping_add(salt.wrapping_mul(1_013_904_223));
    let centered = i32::try_from(mixed % 257).expect("value is below 257") - 128;
    centered as f32 * (1.0 / 256.0)
}

fn tf32_value(value: f32) -> f64 {
    let bits = value.to_bits();
    let exponent = bits & 0x7f80_0000;
    if exponent == 0x7f80_0000 {
        return value as f64;
    }
    let lsb = (bits >> 13) & 1;
    let rounded = bits.wrapping_add(0x0fff + lsb) & 0xffff_e000;
    f32::from_bits(rounded) as f64
}

fn build_host_case(request: F32TriadRequest, salt: usize, epilogue: EpilogueCase) -> HostCase {
    let op = request.op;
    let shape = request.shape;
    let (a_len, b_len) = match op {
        ResolvedGemmOp::Nn => (shape.m * shape.lda, shape.k * shape.ldb),
        ResolvedGemmOp::Tn => (shape.m * shape.lda, shape.m * shape.ldb),
        ResolvedGemmOp::Nt => (shape.m * shape.lda, shape.k * shape.ldb),
    };
    let rows = shape.output_rows(op);
    let columns = shape.output_columns(op);
    let a = (0..a_len)
        .map(|index| input_value(index, salt + 1))
        .collect::<Vec<_>>();
    let b = (0..b_len)
        .map(|index| input_value(index, salt + 7))
        .collect::<Vec<_>>();
    let bias = epilogue.bias.then(|| {
        (0..columns)
            .map(|index| input_value(index, salt + 13) * 0.25)
            .collect::<Vec<_>>()
    });
    let initial = (0..rows * shape.ldc)
        .map(|index| input_value(index, salt + 19) * 0.125)
        .collect::<Vec<_>>();
    let mut expected = initial.iter().copied().map(f64::from).collect::<Vec<_>>();

    match op {
        ResolvedGemmOp::Nn => {
            for row in 0..shape.m {
                for column in 0..shape.n {
                    let mut sum = 0.0_f64;
                    for inner in 0..shape.k {
                        sum += tf32_value(a[row * shape.lda + inner])
                            * tf32_value(b[inner * shape.ldb + column]);
                    }
                    let accumulator =
                        sum + bias.as_ref().map_or(0.0, |bias| f64::from(bias[column]));
                    expected[row * shape.ldc + column] = f64::from(epilogue.alpha) * accumulator
                        + f64::from(epilogue.beta) * f64::from(initial[row * shape.ldc + column]);
                }
            }
        }
        ResolvedGemmOp::Tn => {
            for row in 0..shape.k {
                for column in 0..shape.n {
                    let mut sum = 0.0_f64;
                    for inner in 0..shape.m {
                        sum += tf32_value(a[inner * shape.lda + row])
                            * tf32_value(b[inner * shape.ldb + column]);
                    }
                    expected[row * shape.ldc + column] += f64::from(epilogue.alpha) * sum;
                }
            }
        }
        ResolvedGemmOp::Nt => {
            for row in 0..shape.m {
                for column in 0..shape.k {
                    let mut sum = 0.0_f64;
                    for inner in 0..shape.n {
                        sum += tf32_value(a[row * shape.lda + inner])
                            * tf32_value(b[column * shape.ldb + inner]);
                    }
                    expected[row * shape.ldc + column] = f64::from(epilogue.alpha) * sum;
                }
            }
        }
    }

    HostCase {
        request,
        a,
        b,
        bias,
        initial,
        expected,
        alpha: epilogue.alpha,
        beta: epilogue.beta,
    }
}

fn upload_case(ctx: &GpuCtx, host: &HostCase) -> Result<DeviceCase, String> {
    Ok(DeviceCase {
        a: GpuBuffer::from_cpu(&ctx.stream, &host.a)?,
        b: GpuBuffer::from_cpu(&ctx.stream, &host.b)?,
        bias: host
            .bias
            .as_deref()
            .map(|values| GpuBuffer::from_cpu(&ctx.stream, values))
            .transpose()?,
        output: GpuBuffer::from_cpu(&ctx.stream, &host.initial)?,
    })
}

fn launch_forced(ctx: &GpuCtx, prepared: &PreparedF32TriadLaunch) -> Result<(), String> {
    unsafe {
        launch_prepared_f32_triad(ctx, prepared, |_| {
            Err("forced TF32 qualification unexpectedly selected scalar code".into())
        })
    }
}

fn output_bits(ctx: &GpuCtx, output: &GpuBuffer) -> Result<Vec<u32>, String> {
    Ok(output
        .to_cpu(&ctx.stream)?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn words_digest(domain: &[u8], values: &[u32]) -> [u8; 32] {
    let mut digest =
        FramedSha256::new(domain).required(b"word-count", &(values.len() as u64).to_le_bytes());
    for (index, value) in values.iter().copied().enumerate() {
        digest = digest
            .required(b"word-index", &(index as u64).to_le_bytes())
            .required(b"word", &value.to_le_bytes());
    }
    digest.finish()
}

fn append_boundary_digest_case(
    digest: FramedSha256,
    kind: &[u8],
    index: usize,
    tail: Option<usize>,
    dims: (usize, usize, usize),
    output: [u8; 32],
) -> FramedSha256 {
    let tail = tail.map(|value| (value as u64).to_le_bytes());
    digest
        .required(b"case", kind)
        .required(b"case-index", &(index as u64).to_le_bytes())
        .optional(b"tail", tail.as_ref().map(|value| value.as_slice()))
        .required(b"m", &(dims.0 as u64).to_le_bytes())
        .required(b"k", &(dims.1 as u64).to_le_bytes())
        .required(b"n", &(dims.2 as u64).to_le_bytes())
        .required(b"output", &output)
}

fn check_accuracy(actual: &[u32], expected: &[f64], label: &str) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: output length {} differs from reference {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(f32::from_bits(actual));
        let tolerance = 0.0025 * (1.0 + expected.abs());
        if !actual.is_finite() || (actual - expected).abs() > tolerance {
            return Err(format!(
                "{label}: output {index} was {actual}, reference {expected}, tolerance {tolerance}"
            ));
        }
    }
    Ok(())
}

fn timed_launches(
    ctx: &GpuCtx,
    iterations: u32,
    mut launch: impl FnMut() -> Result<(), String>,
) -> Result<f32, String> {
    let start = ctx
        .stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("create start event: {error:?}"))?;
    let end = ctx
        .stream
        .context()
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|error| format!("create end event: {error:?}"))?;
    start
        .record(&ctx.stream)
        .map_err(|error| format!("record start event: {error:?}"))?;
    for _ in 0..iterations {
        launch()?;
    }
    end.record(&ctx.stream)
        .map_err(|error| format!("record end event: {error:?}"))?;
    let elapsed = start
        .elapsed_ms(&end)
        .map_err(|error| format!("measure CUDA events: {error:?}"))?;
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return Err(format!(
            "CUDA event timing was not positive and finite: {elapsed}"
        ));
    }
    Ok(elapsed / iterations as f32)
}

fn qualify_primary_route(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    config: Tf32QualificationConfig,
    salt: usize,
) -> Result<RouteEvidence, String> {
    let dims = logical_dims(spec.op, spec.tile.0 as usize, spec.tile.1 as usize, 33);
    let request = F32TriadRequest {
        op: spec.op,
        shape: shape_for_spec(spec, dims)?,
    };
    let host = build_host_case(request, salt, primary_epilogue(spec.op));
    let mut device = upload_case(ctx, &host)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("finish route allocations: {error:?}"))?;
    let prepared = prepare_f32_triad_forced(ctx, host.request, device.operands(&host), spec.route)?;

    device.output.upload(&ctx.stream, &host.initial)?;
    let eager_trace = ctx.record_eager_gemm_trace(|| launch_forced(ctx, &prepared))?;
    let manifest = eager_trace.manifest();
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize eager {}: {error:?}", spec.symbol))?;
    let eager = output_bits(ctx, &device.output)?;
    check_accuracy(&eager, &host.expected, spec.symbol)?;

    let correctness_repeats = match config.suite {
        Tf32QualificationSuite::Full => config.repeat,
        Tf32QualificationSuite::Sanitizer => 1,
    };
    for iteration in 0..correctness_repeats {
        device.output.upload(&ctx.stream, &host.initial)?;
        launch_forced(ctx, &prepared)?;
        ctx.stream.synchronize().map_err(|error| {
            format!(
                "synchronize eager repeat {iteration} {}: {error:?}",
                spec.symbol
            )
        })?;
        let repeated = output_bits(ctx, &device.output)?;
        if repeated != eager {
            return Err(format!(
                "{} changed output bits on eager repeat {iteration}",
                spec.symbol
            ));
        }
    }

    device.output.upload(&ctx.stream, &host.initial)?;
    let (graph, plan) = unsafe {
        capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, &manifest, || {
            launch_forced(ctx, &prepared)
        })
    }?;
    let plan = plan.ok_or_else(|| format!("{} captured no GEMM route", spec.symbol))?;
    let routes = plan.routes();
    if routes.len() != 1 || routes[0].symbol != spec.symbol {
        return Err(format!(
            "{} captured an unexpected physical route inventory",
            spec.symbol
        ));
    }
    if eager_trace.routes() != routes {
        return Err(format!(
            "{} eager and captured physical routes differ",
            spec.symbol
        ));
    }
    let eager_launches = eager_trace
        .launches()
        .ok_or_else(|| format!("{} eager trace contains no physical route", spec.symbol))?;
    if eager_launches != plan.launches {
        return Err(format!(
            "{} eager and captured ordered launch identities differ",
            spec.symbol
        ));
    }

    let mut graph_bits = None;
    for iteration in 0..correctness_repeats {
        device.output.upload(&ctx.stream, &host.initial)?;
        plan.with_validated_launch(ctx, spec.symbol, || {
            graph
                .launch()
                .map_err(|error| format!("launch graph {}: {error:?}", spec.symbol))
        })?;
        ctx.stream.synchronize().map_err(|error| {
            format!(
                "synchronize graph repeat {iteration} {}: {error:?}",
                spec.symbol
            )
        })?;
        let actual = output_bits(ctx, &device.output)?;
        if actual != eager {
            return Err(format!(
                "{} eager and graph output bits differ on replay {iteration}",
                spec.symbol
            ));
        }
        graph_bits = Some(actual);
    }
    let graph_bits = graph_bits.ok_or_else(|| "graph replay count must be positive".to_owned())?;

    let warmup = match config.suite {
        Tf32QualificationSuite::Full => 50,
        Tf32QualificationSuite::Sanitizer => 1,
    };
    for _ in 0..warmup {
        launch_forced(ctx, &prepared)?;
    }
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize eager warmup: {error:?}"))?;
    let timed_iterations = correctness_repeats.max(1);
    let eager_ms = timed_launches(ctx, timed_iterations, || launch_forced(ctx, &prepared))?;
    let graph_ms = timed_launches(ctx, timed_iterations, || {
        plan.with_validated_launch(ctx, spec.symbol, || {
            graph
                .launch()
                .map_err(|error| format!("launch timed graph: {error:?}"))
        })
    })?;

    let function = ctx
        .kernels
        .tf32_function(spec.symbol)
        .ok_or_else(|| format!("loaded module is missing {}", spec.symbol))?;
    let local_bytes = u32::try_from(
        function
            .local_size_bytes()
            .map_err(|error| format!("query {} local bytes: {error:?}", spec.symbol))?,
    )
    .map_err(|_| format!("{} reports negative local memory", spec.symbol))?;
    let compiler = match spec.module_kind {
        ModuleKind::TriadSm80 => ctx.kernels.triad_sm80_compiler_identity(),
        ModuleKind::TriadSm90a | ModuleKind::TriadSm100 | ModuleKind::TriadSm120 => ctx
            .kernels
            .specialized_compiler_identity()
            .ok_or_else(|| format!("{} has no specialized compiler identity", spec.symbol))?,
        module_kind => {
            return Err(format!(
                "{} has unsupported TF32 module kind {module_kind:?}",
                spec.symbol
            ));
        }
    };
    let local_admission = super::modules::validate_tf32_driver_jit_local_memory(
        local_bytes,
        spec.module_kind,
        spec.symbol,
        compiler,
    )?;
    let registers = u32::try_from(
        function
            .num_regs()
            .map_err(|error| format!("query {} registers: {error:?}", spec.symbol))?,
    )
    .map_err(|_| format!("{} reports a negative register count", spec.symbol))?;
    let zero_digest = qualify_zero_reduction(ctx, spec, salt + 29)?;
    let adversarial = qualify_adversarial_route(ctx, spec, salt + 100_000)?;
    let boundary_digest = qualify_shape_boundaries(ctx, spec, salt + 49_000)?;

    Ok(RouteEvidence {
        symbol: spec.symbol,
        eager_digest: words_digest(b"tf32-qualification-output-bits.v2", &eager),
        graph_digest: words_digest(b"tf32-qualification-output-bits.v2", &graph_bits),
        zero_digest,
        boundary_digest,
        eager_route_digest: eager_launches.ordered_digest,
        graph_route_digest: plan.launches.ordered_digest,
        tensor_map_digest: routes[0].tensor_maps_digest,
        staged_digest: adversarial.staged_digest,
        exceptional_digest: adversarial.exceptional_digest,
        cross_m_digest: adversarial.cross_m_digest,
        adversarial_counts: adversarial.counts,
        eager_ms,
        graph_ms,
        registers,
        driver_jit_local_bytes: local_admission.observed_bytes,
        approved_driver_jit_local_cap: local_admission.approved_cap_bytes,
    })
}

fn qualify_zero_reduction(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<[u8; 32], String> {
    let dims = logical_dims(spec.op, spec.tile.0 as usize, spec.tile.1 as usize, 0);
    let shape = F32TriadShape::contiguous(spec.op, dims);
    let rows = shape.output_rows(spec.op);
    let columns = shape.output_columns(spec.op);
    let initial = (0..rows * shape.ldc)
        .map(|index| input_value(index, salt) * 0.125)
        .collect::<Vec<_>>();
    let bias = (spec.op == ResolvedGemmOp::Nn).then(|| {
        (0..columns)
            .map(|index| input_value(index, salt + 3) * 0.25)
            .collect::<Vec<_>>()
    });
    let mut output = GpuBuffer::from_cpu(&ctx.stream, &initial)?;
    let bias_gpu = bias
        .as_deref()
        .map(|values| GpuBuffer::from_cpu(&ctx.stream, values))
        .transpose()?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("finish K=0 allocations: {error:?}"))?;
    let request = F32TriadRequest { op: spec.op, shape };
    let operands = F32TriadOperands {
        output: output.cached_ptr(),
        a: 0,
        b: 0,
        bias: bias_gpu.as_ref().map(GpuBuffer::cached_ptr),
        alpha: 1.0,
        beta: beta_for_op(spec.op),
    };
    let prepared = prepare_f32_triad_forced(ctx, request, operands, spec.route)?;
    launch_forced(ctx, &prepared)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize K=0 {}: {error:?}", spec.symbol))?;
    let first = output_bits(ctx, &output)?;
    let expected = match spec.op {
        ResolvedGemmOp::Nn => {
            let bias = bias.as_ref().expect("NN K=0 bias");
            (0..rows * shape.ldc)
                .map(|index| bias[index % shape.ldc].to_bits())
                .collect::<Vec<_>>()
        }
        ResolvedGemmOp::Tn => initial.iter().copied().map(f32::to_bits).collect(),
        ResolvedGemmOp::Nt => vec![0_u32; rows * shape.ldc],
    };
    if first != expected {
        return Err(format!("{} K=0 epilogue bits differ", spec.symbol));
    }

    output.upload(&ctx.stream, &initial)?;
    let eager_trace = ctx.record_eager_gemm_trace(|| launch_forced(ctx, &prepared))?;
    let manifest = eager_trace.manifest();
    let (graph, plan) = unsafe {
        capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, &manifest, || {
            launch_forced(ctx, &prepared)
        })
    }?;
    let plan = plan.ok_or_else(|| format!("{} K=0 captured no route", spec.symbol))?;
    if plan.routes().len() != 1 || plan.routes()[0].symbol != spec.symbol {
        return Err(format!("{} K=0 captured the wrong symbol", spec.symbol));
    }
    if eager_trace.routes() != plan.routes() {
        return Err(format!(
            "{} K=0 eager and captured physical routes differ",
            spec.symbol
        ));
    }
    output.upload(&ctx.stream, &initial)?;
    plan.with_validated_launch(ctx, spec.symbol, || {
        graph
            .launch()
            .map_err(|error| format!("launch K=0 graph {}: {error:?}", spec.symbol))
    })?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize K=0 graph {}: {error:?}", spec.symbol))?;
    let replay = output_bits(ctx, &output)?;
    if replay != first {
        return Err(format!("{} K=0 eager and graph bits differ", spec.symbol));
    }
    Ok(words_digest(b"tf32-qualification-zero-output.v1", &first))
}

fn qualify_shape_boundaries(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<[u8; 32], String> {
    let mut digest = FramedSha256::new(b"tf32-qualification-boundary-corpus.v1")
        .required(b"symbol", spec.symbol.as_bytes());
    let mut observed_cases = 0_usize;
    for (index, reduction) in K_CASES.into_iter().filter(|value| *value != 0).enumerate() {
        let (rows, columns) = if STAGED_K_CASES.contains(&reduction) {
            (GUARDED_EXTENT, GUARDED_EXTENT)
        } else {
            (spec.tile.0 as usize, spec.tile.1 as usize)
        };
        let dims = logical_dims(spec.op, rows, columns, reduction);
        let output = qualify_single_eager(ctx, spec, dims, salt + index)?;
        digest = append_boundary_digest_case(digest, b"reduction", index, None, dims, output)
            .required(b"reduction", &(reduction as u64).to_le_bytes());
        observed_cases += 1;
    }
    for (index, tail) in TAIL_CASES.into_iter().enumerate() {
        let row_tail = logical_dims(spec.op, tail, spec.tile.1 as usize, 33);
        let row_output = qualify_single_eager(ctx, spec, row_tail, salt + 101 + index * 2)?;
        digest = append_boundary_digest_case(
            digest,
            b"row-tail",
            index,
            Some(tail),
            row_tail,
            row_output,
        );
        observed_cases += 1;
        let column_tail = logical_dims(spec.op, spec.tile.0 as usize, tail, 33);
        let column_output = qualify_single_eager(ctx, spec, column_tail, salt + 102 + index * 2)?;
        digest = append_boundary_digest_case(
            digest,
            b"column-tail",
            index,
            Some(tail),
            column_tail,
            column_output,
        );
        observed_cases += 1;
    }
    if observed_cases != boundary_cases_per_route() {
        return Err(format!(
            "{} observed {observed_cases} boundary cases, expected {}",
            spec.symbol,
            boundary_cases_per_route()
        ));
    }
    Ok(digest.finish())
}

fn qualify_single_eager(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    dims: (usize, usize, usize),
    salt: usize,
) -> Result<[u8; 32], String> {
    let request = F32TriadRequest {
        op: spec.op,
        shape: shape_for_spec(spec, dims)?,
    };
    let host = build_host_case(request, salt, primary_epilogue(spec.op));
    let device = upload_case(ctx, &host)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("finish boundary allocations: {error:?}"))?;
    let prepared = prepare_f32_triad_forced(ctx, host.request, device.operands(&host), spec.route)?;
    launch_forced(ctx, &prepared)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize boundary {}: {error:?}", spec.symbol))?;
    let actual = output_bits(ctx, &device.output)?;
    check_accuracy(&actual, &host.expected, spec.symbol)?;
    Ok(words_digest(
        b"tf32-qualification-boundary-output.v1",
        &actual,
    ))
}

fn run_guarded_case(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    host: &GuardedHostCase,
    label: &str,
) -> Result<Vec<u32>, String> {
    let device = upload_guarded_case(ctx, host)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("finish guarded allocations: {error:?}"))?;
    let prepared =
        prepare_f32_triad_forced(ctx, host.request, device.operands(host, ctx), spec.route)?;
    launch_forced(ctx, &prepared)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize guarded {label}: {error:?}"))?;
    let full = guarded_output_bits(ctx, &device)?;
    check_guarded_output(host, &full, label)
}

fn qualify_staged_guarded(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<([u8; 32], usize, usize), String> {
    let mut digest = FramedSha256::new(b"tf32-qualification-staged-guarded.v1")
        .required(b"symbol", spec.symbol.as_bytes());
    let mut k65_active = None;
    let mut launches = 0;
    for (index, reduction) in STAGED_K_CASES.into_iter().enumerate() {
        let dims = logical_dims(spec.op, GUARDED_EXTENT, GUARDED_EXTENT, reduction);
        let host = build_guarded_case(
            spec.op,
            dims,
            salt + index * 17,
            adversarial_epilogue(spec.op),
            0x7fc1_5a5a,
        )?;
        let active = run_guarded_case(ctx, spec, &host, spec.symbol)?;
        digest = digest
            .required(b"k", &(reduction as u64).to_le_bytes())
            .required(
                b"shape",
                &[
                    host.request.shape.m as u64,
                    host.request.shape.k as u64,
                    host.request.shape.n as u64,
                    host.request.shape.lda as u64,
                    host.request.shape.ldb as u64,
                    host.request.shape.ldc as u64,
                ]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
            )
            .required(b"alpha", &host.alpha.to_bits().to_le_bytes())
            .required(b"beta", &host.beta.to_bits().to_le_bytes())
            .required(
                b"active-output",
                &words_digest(b"tf32-qualification-staged-output.v1", &active),
            );
        launches += 1;
        if reduction == 65 {
            k65_active = Some(active);
        }
    }

    let dims = logical_dims(spec.op, GUARDED_EXTENT, GUARDED_EXTENT, 65);
    let alternate = build_guarded_case(
        spec.op,
        dims,
        salt,
        adversarial_epilogue(spec.op),
        0xca54_0000,
    )?;
    let alternate_active = run_guarded_case(ctx, spec, &alternate, spec.symbol)?;
    if k65_active.as_deref() != Some(alternate_active.as_slice()) {
        return Err(format!(
            "{} active output changed with guarded padding poison",
            spec.symbol
        ));
    }
    digest = digest.required(
        b"alternate-poison-output",
        &words_digest(
            b"tf32-qualification-guard-poison-output.v1",
            &alternate_active,
        ),
    );
    launches += 1;
    Ok((digest.finish(), launches - 1, 1))
}

fn set_guarded_value(
    values: &mut [f32],
    storage: GuardedStorage,
    row: usize,
    column: usize,
    value: f32,
) {
    values[storage.offset + row * storage.stride + column] = value;
}

fn check_guarded_red_zones(host: &GuardedHostCase, actual: &[u32]) -> Result<(), String> {
    let storage = host.output_storage;
    for (index, (&actual, initial)) in actual
        .iter()
        .zip(host.initial.iter().map(|value| value.to_bits()))
        .enumerate()
    {
        let relative = index.checked_sub(storage.offset);
        let active = relative.is_some_and(|relative| {
            let row = relative / storage.stride;
            let column = relative % storage.stride;
            row < storage.rows && column < storage.width
        });
        if !active && actual != initial {
            return Err(format!(
                "exceptional output red zone changed at element {index}"
            ));
        }
    }
    Ok(())
}

fn qualify_exceptional_values(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<[u8; 32], String> {
    let values = exceptional_values_for_spec(spec);
    let extent = values.len();
    let dims = logical_dims(spec.op, extent, extent, extent);
    let mut host = build_guarded_case(spec.op, dims, salt, primary_epilogue(spec.op), 0x7fc1_5a5a)?;
    host.bias = None;
    host.bias_storage = None;
    host.alpha = 1.0;
    host.beta = beta_for_op(spec.op);
    for (row, bits) in values.iter().copied().enumerate() {
        for column in 0..extent {
            set_guarded_value(&mut host.a, host.a_storage, row, column, 0.0);
            set_guarded_value(&mut host.b, host.b_storage, row, column, 0.0);
            set_guarded_value(&mut host.initial, host.output_storage, row, column, 0.0);
        }
        set_guarded_value(&mut host.a, host.a_storage, row, row, f32::from_bits(bits));
        set_guarded_value(&mut host.b, host.b_storage, row, row, 1.0);
    }

    let mut device = upload_guarded_case(ctx, &host)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("finish exceptional allocations: {error:?}"))?;
    let prepared =
        prepare_f32_triad_forced(ctx, host.request, device.operands(&host, ctx), spec.route)?;
    launch_forced(ctx, &prepared)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize exceptional {}: {error:?}", spec.symbol))?;
    let first = guarded_output_bits(ctx, &device)?;
    check_guarded_red_zones(&host, &first)?;

    device.output.upload(&ctx.stream, &host.initial)?;
    launch_forced(ctx, &prepared)?;
    ctx.stream
        .synchronize()
        .map_err(|error| format!("synchronize exceptional repeat {}: {error:?}", spec.symbol))?;
    let second = guarded_output_bits(ctx, &device)?;
    if first != second {
        return Err(format!(
            "{} exceptional output bits changed on repeat",
            spec.symbol
        ));
    }
    check_guarded_red_zones(&host, &second)?;

    for (index, bits) in values.iter().copied().enumerate() {
        let output_index = host.output_storage.offset + index * host.output_storage.stride + index;
        let actual = f32::from_bits(first[output_index]);
        match exceptional_class_for_spec(spec, f32::from_bits(bits)) {
            ExceptionalClass::Nan if !actual.is_nan() => {
                return Err(format!("{} exceptional NaN class changed", spec.symbol));
            }
            ExceptionalClass::Infinity { negative }
                if !actual.is_infinite() || actual.is_sign_negative() != negative =>
            {
                return Err(format!(
                    "{} exceptional infinity class changed",
                    spec.symbol
                ));
            }
            ExceptionalClass::Zero if actual != 0.0 => {
                return Err(format!("{} exceptional zero class changed", spec.symbol));
            }
            ExceptionalClass::Finite(expected) => {
                let actual = f64::from(actual);
                let tolerance = 0.0025 * (1.0 + expected.abs());
                if !actual.is_finite() || (actual - expected).abs() > tolerance {
                    return Err(format!(
                        "{} exceptional finite value was {actual}, expected {expected}",
                        spec.symbol
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(words_digest(
        b"tf32-qualification-exceptional-output.v1",
        &first,
    ))
}

fn qualify_cross_m(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<([u8; 32], usize, usize), String> {
    let mut digest = FramedSha256::new(b"tf32-qualification-cross-m.v1")
        .required(b"symbol", spec.symbol.as_bytes());
    if spec.op == ResolvedGemmOp::Tn {
        return Ok((digest.required(b"applicable", &[0]).finish(), 0, 0));
    }
    digest = digest.required(b"applicable", &[1]);
    let tile_rows =
        usize::try_from(spec.tile.0).map_err(|_| "TF32 tile rows exceed usize::MAX".to_owned())?;
    let rows = [1, tile_rows, tile_rows + 1, tile_rows * 2 + 1];
    let mut reference = None;
    for row_count in rows {
        let dims = logical_dims(spec.op, row_count, GUARDED_EXTENT, 65);
        let host = build_guarded_case(
            spec.op,
            dims,
            salt,
            adversarial_epilogue(spec.op),
            0x7fc1_5a5a,
        )?;
        let active = run_guarded_case(ctx, spec, &host, spec.symbol)?;
        let row_zero = &active[..GUARDED_EXTENT];
        if let Some(reference) = reference.as_deref()
            && reference != row_zero
        {
            return Err(format!(
                "{} row zero changed across M={row_count}",
                spec.symbol
            ));
        }
        reference = Some(row_zero.to_vec());
        digest = digest
            .required(b"m", &(row_count as u64).to_le_bytes())
            .required(
                b"row-zero",
                &words_digest(b"tf32-qualification-cross-m-row.v1", row_zero),
            );
    }
    Ok((digest.finish(), 1, rows.len()))
}

fn qualify_nn_bias_beta(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<Option<[u8; 32]>, String> {
    if spec.op != ResolvedGemmOp::Nn {
        return Ok(None);
    }
    let dims = logical_dims(spec.op, GUARDED_EXTENT, GUARDED_EXTENT, 65);
    let host = build_guarded_case(spec.op, dims, salt, nn_bias_beta_epilogue(), 0x7fc1_5a5a)?;
    let active = run_guarded_case(ctx, spec, &host, spec.symbol)?;
    Ok(Some(words_digest(
        b"tf32-qualification-nn-bias-beta-output.v1",
        &active,
    )))
}

fn qualify_adversarial_route(
    ctx: &GpuCtx,
    spec: &'static Tf32KernelSpec,
    salt: usize,
) -> Result<AdversarialRouteEvidence, String> {
    let (staged_digest, staged_cases, guard_poison_pairs) =
        qualify_staged_guarded(ctx, spec, salt)?;
    let exceptional_digest = qualify_exceptional_values(ctx, spec, salt + 10_000)?;
    let (cross_m_digest, cross_m_symbols, cross_m_shapes) =
        qualify_cross_m(ctx, spec, salt + 20_000)?;
    let nn_bias_beta = qualify_nn_bias_beta(ctx, spec, salt + 30_000)?;
    let staged_digest = if let Some(nn_bias_beta) = nn_bias_beta {
        FramedSha256::new(b"tf32-qualification-staged-plus-epilogue.v1")
            .required(b"staged", &staged_digest)
            .required(b"nn-bias-beta", &nn_bias_beta)
            .finish()
    } else {
        staged_digest
    };
    Ok(AdversarialRouteEvidence {
        staged_digest,
        exceptional_digest,
        cross_m_digest,
        counts: AdversarialCounts {
            staged_cases,
            guard_poison_pairs,
            exceptional_symbols: 1,
            cross_m_symbols,
            cross_m_shapes,
            nn_bias_beta_symbols: usize::from(nn_bias_beta.is_some()),
        },
    })
}

fn semantic_digest(
    domain: &[u8],
    evidence: &[RouteEvidence],
    select: fn(&RouteEvidence) -> [u8; 32],
) -> [u8; 32] {
    let mut digest =
        FramedSha256::new(domain).required(b"route-count", &(evidence.len() as u64).to_le_bytes());
    for route in evidence {
        digest = digest
            .required(b"symbol", route.symbol.as_bytes())
            .required(b"value", &select(route));
    }
    digest.finish()
}

fn observed_adversarial_counts(evidence: &[RouteEvidence]) -> AdversarialCounts {
    evidence
        .iter()
        .fold(AdversarialCounts::default(), |mut total, route| {
            total.staged_cases += route.adversarial_counts.staged_cases;
            total.guard_poison_pairs += route.adversarial_counts.guard_poison_pairs;
            total.exceptional_symbols += route.adversarial_counts.exceptional_symbols;
            total.cross_m_symbols += route.adversarial_counts.cross_m_symbols;
            total.cross_m_shapes += route.adversarial_counts.cross_m_shapes;
            total.nn_bias_beta_symbols += route.adversarial_counts.nn_bias_beta_symbols;
            total
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DriverJitResourceSummary {
    max_local_bytes: u32,
    exception_count: usize,
    digest: [u8; 32],
}

fn driver_jit_resource_summary(evidence: &[RouteEvidence]) -> DriverJitResourceSummary {
    let mut max_local_bytes = 0;
    let mut exception_count = 0;
    let mut digest = FramedSha256::new(b"tf32-driver-jit-local-resources.v1")
        .required(b"route-count", &(evidence.len() as u64).to_le_bytes());
    for route in evidence {
        max_local_bytes = max_local_bytes.max(route.driver_jit_local_bytes);
        exception_count += usize::from(route.driver_jit_local_bytes != 0);
        digest = digest
            .required(b"symbol", route.symbol.as_bytes())
            .required(b"route-identity", &route.eager_route_digest)
            .required(b"registers", &route.registers.to_le_bytes())
            .required(
                b"observed-local-bytes",
                &route.driver_jit_local_bytes.to_le_bytes(),
            )
            .required(
                b"approved-local-cap-bytes",
                &route.approved_driver_jit_local_cap.to_le_bytes(),
            );
    }
    DriverJitResourceSummary {
        max_local_bytes,
        exception_count,
        digest: digest.finish(),
    }
}

fn validate_driver_jit_resource_inventory(
    config: Tf32QualificationConfig,
    specs: &[&Tf32KernelSpec],
    evidence: &[RouteEvidence],
) -> Result<DriverJitResourceSummary, String> {
    if specs.len() != evidence.len() {
        return Err(format!(
            "Driver JIT resource inventory has {} specs and {} evidence rows",
            specs.len(),
            evidence.len()
        ));
    }
    let mut sm120_routes = 0;
    let mut sm120_zero_routes = 0;
    for (spec, route) in specs.iter().copied().zip(evidence) {
        if spec.symbol != route.symbol {
            return Err(format!(
                "Driver JIT resource row {} does not match route {}",
                route.symbol, spec.symbol
            ));
        }
        if route.driver_jit_local_bytes > route.approved_driver_jit_local_cap {
            return Err(format!(
                "{} reports {} Driver JIT local bytes above its approved {}-byte cap",
                route.symbol, route.driver_jit_local_bytes, route.approved_driver_jit_local_cap
            ));
        }
        if spec.module_kind == ModuleKind::TriadSm120 {
            sm120_routes += 1;
            let resources = (
                route.driver_jit_local_bytes,
                route.approved_driver_jit_local_cap,
            );
            if resources == (0, 0) {
                sm120_zero_routes += 1;
            } else {
                return Err(format!(
                    "{} has unexpected SM120 Driver JIT local-memory evidence {}/{}",
                    route.symbol, resources.0, resources.1
                ));
            }
        } else if route.driver_jit_local_bytes != 0 || route.approved_driver_jit_local_cap != 0 {
            return Err(format!(
                "non-SM120 route {} has nonzero Driver JIT local-memory evidence {}/{}",
                route.symbol, route.driver_jit_local_bytes, route.approved_driver_jit_local_cap
            ));
        }
    }
    if matches!(config.exact_cc, (12, 0 | 1)) {
        if (sm120_routes, sm120_zero_routes) != (18, 18) {
            return Err(format!(
                "SM120 Driver JIT inventory is {sm120_routes} routes with {sm120_zero_routes} zero-byte routes; expected 18/18"
            ));
        }
    } else if sm120_routes != 0 || sm120_zero_routes != 0 {
        return Err("non-SM120 qualification unexpectedly contains SM120 routes".into());
    }
    Ok(driver_jit_resource_summary(evidence))
}

fn boundary_cases_per_route() -> usize {
    K_CASES.iter().filter(|reduction| **reduction != 0).count() + 2 * TAIL_CASES.len()
}

fn build_artifact(
    config: Tf32QualificationConfig,
    evidence: &[RouteEvidence],
    artifact_set_digest: [u8; 32],
    driver_abi_digest: [u8; 32],
) -> Vec<u8> {
    let mut artifact = String::from("MambaBiTf32QualificationArtifactV5\n");
    writeln!(artifact, "cc\t{}.{}", config.exact_cc.0, config.exact_cc.1)
        .expect("String writes cannot fail");
    let (suite, actual_repeat) = match config.suite {
        Tf32QualificationSuite::Full => ("full", config.repeat),
        Tf32QualificationSuite::Sanitizer => ("sanitizer", 1),
    };
    writeln!(artifact, "suite\t{suite}").expect("String writes cannot fail");
    writeln!(artifact, "repeat\t{actual_repeat}").expect("String writes cannot fail");
    writeln!(
        artifact,
        "boundary_cases_per_route\t{}",
        boundary_cases_per_route()
    )
    .expect("String writes cannot fail");
    writeln!(
        artifact,
        "artifact_set\t{}",
        digest_hex(&artifact_set_digest)
    )
    .expect("String writes cannot fail");
    writeln!(artifact, "driver_abi\t{}", digest_hex(&driver_abi_digest))
        .expect("String writes cannot fail");
    for route in evidence {
        writeln!(
            artifact,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{}\t{}\t{}",
            route.symbol,
            digest_hex(&route.eager_digest),
            digest_hex(&route.graph_digest),
            digest_hex(&route.zero_digest),
            digest_hex(&route.boundary_digest),
            digest_hex(&route.eager_route_digest),
            digest_hex(&route.graph_route_digest),
            digest_hex(&route.tensor_map_digest),
            digest_hex(&route.staged_digest),
            digest_hex(&route.exceptional_digest),
            digest_hex(&route.cross_m_digest),
            route.eager_ms,
            route.graph_ms,
            route.registers,
            route.driver_jit_local_bytes,
            route.approved_driver_jit_local_cap,
        )
        .expect("String writes cannot fail");
    }
    artifact.into_bytes()
}

fn plain_sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

struct ReportDigests {
    artifact: [u8; 32],
    output: [u8; 32],
    zero: [u8; 32],
    boundary: [u8; 32],
    maps: [u8; 32],
    routes: [u8; 32],
    staged: [u8; 32],
    exceptional: [u8; 32],
    cross_m: [u8; 32],
    driver_abi: [u8; 32],
    driver_jit_resources: [u8; 32],
}

fn build_report(
    config: Tf32QualificationConfig,
    routes_qualified: usize,
    counts: AdversarialCounts,
    resources: DriverJitResourceSummary,
    digests: ReportDigests,
) -> String {
    let placeholder = "0".repeat(64);
    let (suite, actual_repeat, repeat_determinism) = match config.suite {
        Tf32QualificationSuite::Full => ("full", config.repeat, "pass"),
        Tf32QualificationSuite::Sanitizer => ("sanitizer", 1, "single_run"),
    };
    let mut report = format!(
        concat!(
            "{{\"schema\":\"MambaBiTf32QualificationV5\",",
            "\"exact_cc\":\"{}.{}\",",
            "\"suite\":\"{}\",\"repeat\":{},",
            "\"runtime\":\"pass\",\"primary_accuracy\":\"pass\",",
            "\"boundary_accuracy\":\"pass\",",
            "\"repeat_determinism\":\"{}\",",
            "\"graph_replay\":\"pass\",",
            "\"eager_graph_route_identity\":\"pass\",",
            "\"driver_abi\":\"pass\",\"k0_runtime\":\"pass\",",
            "\"k0_epilogue_bits\":\"pass\",",
            "\"staged_k_reuse\":\"pass\",",
            "\"strided_subviews\":\"pass\",",
            "\"red_zone_canaries\":\"pass\",",
            "\"epilogue_scalars\":\"pass\",",
            "\"exceptional_values\":\"pass\",",
            "\"cross_m_batch_invariance\":\"pass\",",
            "\"timings\":\"recorded\",",
            "\"driver_jit_local_memory\":\"pass\",",
            "\"max_driver_jit_local_memory_bytes\":{},",
            "\"driver_jit_local_memory_exception_count\":{},",
            "\"routes_qualified\":{},",
            "\"k0_symbols_qualified\":{},\"driver_abi_entries\":{},",
            "\"boundary_cases\":{},",
            "\"staged_cases\":{},\"guard_poison_pairs\":{},",
            "\"exceptional_symbols\":{},\"cross_m_symbols\":{},",
            "\"cross_m_shapes\":{},\"nn_bias_beta_symbols\":{},",
            "\"artifact_digest\":\"{}\",\"output_digest\":\"{}\",",
            "\"zero_reduction_digest\":\"{}\",",
            "\"boundary_output_digest\":\"{}\",",
            "\"tensor_map_identity_digest\":\"{}\",",
            "\"ordered_graph_route_digest\":\"{}\",",
            "\"staged_guarded_digest\":\"{}\",",
            "\"exceptional_values_digest\":\"{}\",",
            "\"cross_m_invariance_digest\":\"{}\",",
            "\"driver_abi_digest\":\"{}\",",
            "\"driver_jit_resource_digest\":\"{}\",",
            "\"report_digest\":\"{}\"}}"
        ),
        config.exact_cc.0,
        config.exact_cc.1,
        suite,
        actual_repeat,
        repeat_determinism,
        resources.max_local_bytes,
        resources.exception_count,
        routes_qualified,
        routes_qualified,
        routes_qualified,
        routes_qualified * boundary_cases_per_route(),
        counts.staged_cases,
        counts.guard_poison_pairs,
        counts.exceptional_symbols,
        counts.cross_m_symbols,
        counts.cross_m_shapes,
        counts.nn_bias_beta_symbols,
        digest_hex(&digests.artifact),
        digest_hex(&digests.output),
        digest_hex(&digests.zero),
        digest_hex(&digests.boundary),
        digest_hex(&digests.maps),
        digest_hex(&digests.routes),
        digest_hex(&digests.staged),
        digest_hex(&digests.exceptional),
        digest_hex(&digests.cross_m),
        digest_hex(&digests.driver_abi),
        digest_hex(&digests.driver_jit_resources),
        placeholder,
    );
    let report_digest = digest_hex(&plain_sha256(report.as_bytes()));
    let offset = report
        .rfind(&placeholder)
        .expect("report digest placeholder is present");
    report.replace_range(offset..offset + 64, &report_digest);
    report
}

fn driver_abi_proof(ctx: &GpuCtx, specs: &[&Tf32KernelSpec]) -> Result<String, String> {
    let mut proof = String::from("MambaBiTf32DriverAbiV2\n");
    for spec in specs {
        let abi = ctx
            .kernels
            .triad_kernels()
            .tf32_driver_abi(spec.symbol)
            .ok_or_else(|| format!("missing live Driver ABI census for {}", spec.symbol))?;
        proof.push_str(&abi.tsv_record(spec.symbol)?);
        proof.push('\n');
    }
    Ok(proof)
}

trait QualificationPolicyContext {
    fn batch_invariant(&self) -> bool;
    fn set_batch_invariant(&self, value: bool);
    fn bi_gemm_family(&self) -> BiGemmFamily;
    fn set_bi_gemm_family(&self, family: BiGemmFamily);
    fn bi_tensor_cores(&self) -> bool;
    fn set_bi_tensor_cores(&self, value: bool);
    fn f32_triad_policy(&self) -> F32TriadPolicy;
    fn set_f32_triad_policy(&self, policy: F32TriadPolicy);
}

impl QualificationPolicyContext for GpuCtx {
    fn batch_invariant(&self) -> bool {
        GpuCtx::batch_invariant(self)
    }

    fn set_batch_invariant(&self, value: bool) {
        GpuCtx::set_batch_invariant(self, value);
    }

    fn bi_gemm_family(&self) -> BiGemmFamily {
        GpuCtx::bi_gemm_family(self)
    }

    fn set_bi_gemm_family(&self, family: BiGemmFamily) {
        GpuCtx::set_bi_gemm_family(self, family);
    }

    fn bi_tensor_cores(&self) -> bool {
        GpuCtx::bi_tensor_cores(self)
    }

    fn set_bi_tensor_cores(&self, value: bool) {
        GpuCtx::set_bi_tensor_cores(self, value);
    }

    fn f32_triad_policy(&self) -> F32TriadPolicy {
        GpuCtx::f32_triad_policy(self)
    }

    fn set_f32_triad_policy(&self, policy: F32TriadPolicy) {
        GpuCtx::set_f32_triad_policy(self, policy);
    }
}

struct QualificationPolicyGuard<'a, C: QualificationPolicyContext> {
    context: &'a C,
    batch_invariant: bool,
    family: BiGemmFamily,
    tensor_cores: bool,
    f32_policy: F32TriadPolicy,
}

impl<'a, C: QualificationPolicyContext> QualificationPolicyGuard<'a, C> {
    fn enter(context: &'a C) -> Self {
        let guard = Self {
            context,
            batch_invariant: context.batch_invariant(),
            family: context.bi_gemm_family(),
            tensor_cores: context.bi_tensor_cores(),
            f32_policy: context.f32_triad_policy(),
        };
        context.set_batch_invariant(true);
        context.set_bi_gemm_family(BiGemmFamily::Triad);
        context.set_bi_tensor_cores(false);
        context.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        guard
    }
}

impl<C: QualificationPolicyContext> Drop for QualificationPolicyGuard<'_, C> {
    fn drop(&mut self) {
        self.context.set_f32_triad_policy(self.f32_policy);
        self.context.set_bi_tensor_cores(self.tensor_cores);
        self.context.set_bi_gemm_family(self.family);
        self.context.set_batch_invariant(self.batch_invariant);
    }
}

#[doc(hidden)]
pub fn run_tf32_qualification(
    ctx: &GpuCtx,
    config: Tf32QualificationConfig,
) -> Result<Tf32QualificationOutput, String> {
    let live_cc = ctx
        .stream
        .context()
        .compute_capability()
        .map_err(|error| format!("query qualification compute capability: {error:?}"))?;
    let live_cc = (
        u32::try_from(live_cc.0).map_err(|_| "negative CUDA CC major".to_owned())?,
        u32::try_from(live_cc.1).map_err(|_| "negative CUDA CC minor".to_owned())?,
    );
    if live_cc != config.exact_cc {
        return Err(format!(
            "qualification requires CC {}.{}, found {}.{}",
            config.exact_cc.0, config.exact_cc.1, live_cc.0, live_cc.1
        ));
    }
    if config.repeat == 0 {
        return Err("TF32 qualification repeat count must be positive".into());
    }
    let specs = tf32_qualification_route_specs(config.exact_cc)?;
    if specs.len() != config.expected_routes {
        return Err(format!(
            "qualification route inventory has {} entries, expected {}",
            specs.len(),
            config.expected_routes
        ));
    }

    let _policy_guard = QualificationPolicyGuard::enter(ctx);

    let mut evidence = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().copied().enumerate() {
        let route = qualify_primary_route(ctx, spec, config, 1000 + index * 1000)?;
        evidence.push(route);
    }
    ctx.stream
        .synchronize()
        .map_err(|error| format!("final qualification synchronization: {error:?}"))?;

    let artifact_set = ctx.kernels.artifact_set_identity();
    let driver_abi_proof = driver_abi_proof(ctx, &specs)?;
    let driver_abi = plain_sha256(driver_abi_proof.as_bytes());
    let artifact = build_artifact(config, &evidence, artifact_set.ordered_digest, driver_abi);
    let output = semantic_digest(b"tf32-qualification-all-output.v1", &evidence, |route| {
        route.eager_digest
    });
    let zero = semantic_digest(
        b"tf32-qualification-all-zero-reduction.v1",
        &evidence,
        |route| route.zero_digest,
    );
    let boundary = semantic_digest(
        b"tf32-qualification-all-boundary-output.v1",
        &evidence,
        |route| route.boundary_digest,
    );
    let maps = semantic_digest(
        b"tf32-qualification-all-encoded-maps.v1",
        &evidence,
        |route| route.tensor_map_digest,
    );
    let routes = semantic_digest(
        b"tf32-qualification-all-graph-routes.v1",
        &evidence,
        |route| route.graph_route_digest,
    );
    let staged = semantic_digest(
        b"tf32-qualification-all-staged-guarded.v1",
        &evidence,
        |route| route.staged_digest,
    );
    let exceptional = semantic_digest(
        b"tf32-qualification-all-exceptional-values.v1",
        &evidence,
        |route| route.exceptional_digest,
    );
    let cross_m = semantic_digest(b"tf32-qualification-all-cross-m.v1", &evidence, |route| {
        route.cross_m_digest
    });
    let counts = observed_adversarial_counts(&evidence);
    let expected_counts = adversarial_counts(&specs);
    if counts != expected_counts {
        return Err(format!(
            "qualification adversarial counts {counts:?} differ from route inventory {expected_counts:?}"
        ));
    }
    let resources = validate_driver_jit_resource_inventory(config, &specs, &evidence)?;
    let report_json = build_report(
        config,
        evidence.len(),
        counts,
        resources,
        ReportDigests {
            artifact: plain_sha256(&artifact),
            output,
            zero,
            boundary,
            maps,
            routes,
            staged,
            exceptional,
            cross_m,
            driver_abi,
            driver_jit_resources: resources.digest,
        },
    );
    Ok(Tf32QualificationOutput {
        report_json,
        artifact,
        driver_abi_proof,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba_ssm::gpu::gemm_bi_triad::{
        SM120_AUTO_CELLS_CC120, SM120_SCHEDULE_REVISION, SM120_TENSOR_MAP_REVISION,
        SM120_TUNING_REVISION, Sm120AutoBranchSeal, Sm120ForcedRoute, Sm120NumericContract,
        Sm120Op, Sm120RouteIdentity, sm120_target_candidates,
    };
    use crate::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, DeviceCaps, DeviceIdentity, DriverIdentity, NUMERIC_ABI_REVISION,
        SCHEDULE_REVISION,
    };
    use std::cell::Cell;

    use super::super::contract::{
        TF32_NT_SPLITK4_S3_SPEC, TF32_NT_SPLITK4_S4_SPEC, TF32_NT_SPLITK8_S3_SPEC,
        TF32_NT_SPLITK8_S4_SPEC, TF32_SPLITK2_SPEC, TF32_SPLITK4_SPEC, Tf32SplitKSpec,
    };

    #[test]
    fn splitk2_forced_request_freezes_the_fused_kernel_as_its_planned_route() {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (64, 833, 384),
            PhysicalQualificationRoute::Tf32Forced(TF32_SPLITK2_SPEC.route),
        );
        request.validate().unwrap();
        let identity = request.timed_identity().unwrap();
        assert_eq!(identity.planned_symbol, Some(TF32_SPLITK2_SPEC.symbol));
        assert_eq!(identity.planned_tile, Some(TF32_SPLITK2_SPEC.tile));
    }

    #[test]
    fn splitk4_forced_request_freezes_the_fused_kernel_as_its_planned_route() {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (64, 1_536, 384),
            PhysicalQualificationRoute::Tf32Forced(TF32_SPLITK4_SPEC.route),
        );
        request.validate().unwrap();
        let identity = request.timed_identity().unwrap();
        assert_eq!(identity.planned_symbol, Some(TF32_SPLITK4_SPEC.symbol));
        assert_eq!(identity.planned_tile, Some(TF32_SPLITK4_SPEC.tile));
    }

    #[test]
    fn nt_splitk_forced_requests_freeze_live_and_reverse_candidate_routes() {
        for dims in [(64, 384, 1_536), (384, 64, 1_536)] {
            for spec in [
                &TF32_NT_SPLITK4_S3_SPEC,
                &TF32_NT_SPLITK4_S4_SPEC,
                &TF32_NT_SPLITK8_S3_SPEC,
                &TF32_NT_SPLITK8_S4_SPEC,
            ] {
                let request = PhysicalQualificationRequest::contiguous(
                    ResolvedGemmOp::Nt,
                    dims,
                    PhysicalQualificationRoute::Tf32Forced(spec.route),
                );
                request.validate().unwrap();
                let identity = request.timed_identity().unwrap();
                assert_eq!(identity.planned_symbol, Some(spec.symbol));
                assert_eq!(identity.planned_tile, Some(spec.tile));
            }
        }
    }

    #[test]
    fn nt_splitk_cpu_oracle_matches_the_seed_contract_and_rejects_wrong_output() {
        let dims = (2, 3, 4);
        let salt = 0x6e74_7038;
        let expected = tf32_nt_qualification_reference(dims, salt).unwrap();
        let a = seeded_qualification_values(dims.0 * dims.2, salt ^ 0x2d);
        let b = seeded_qualification_values(dims.1 * dims.2, salt ^ 0x67);
        let row = 1;
        let column = 2;
        let manual = (0..dims.2)
            .map(|inner| {
                tf32_value(a[row * dims.2 + inner]) * tf32_value(b[column * dims.2 + inner])
            })
            .sum::<f64>();
        assert_eq!(expected[row * dims.1 + column], manual);

        let mut actual = expected
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>();
        check_accuracy(&actual, &expected, "NT split-K CPU oracle").unwrap();
        actual[row * dims.1 + column] = f32::INFINITY.to_bits();
        assert!(check_accuracy(&actual, &expected, "NT split-K CPU oracle").is_err());
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn nt_splitk_forced_routes_are_eager_graph_bit_exact_with_one_fused_node() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let cases = [
            ((64, 384, 1_536), &TF32_NT_SPLITK4_S3_SPEC),
            ((64, 384, 1_536), &TF32_NT_SPLITK4_S4_SPEC),
            ((64, 384, 1_536), &TF32_NT_SPLITK8_S3_SPEC),
            ((64, 384, 1_536), &TF32_NT_SPLITK8_S4_SPEC),
            ((384, 64, 1_536), &TF32_NT_SPLITK4_S3_SPEC),
            ((384, 64, 1_536), &TF32_NT_SPLITK4_S4_SPEC),
            ((384, 64, 1_536), &TF32_NT_SPLITK8_S3_SPEC),
            ((384, 64, 1_536), &TF32_NT_SPLITK8_S4_SPEC),
        ];
        let requests = cases.map(|(dims, spec)| {
            PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nt,
                dims,
                PhysicalQualificationRoute::Tf32Forced(spec.route),
            )
        });
        presize_physical_qualification_suite(&ctx, &requests).unwrap();

        for ((dims, spec), request) in cases.into_iter().zip(requests) {
            let mut qualified = qualify_physical_launch(&ctx, request).unwrap();
            let evidence = qualified.evidence();
            let rows = dims.0.div_ceil(spec.tile.0 as usize) as u32;
            let columns = dims.1.div_ceil(spec.tile.1 as usize) as u32;
            assert!(evidence.eager_graph_equal());
            assert_eq!(evidence.launch_count(), 1);
            assert_eq!(evidence.single_launch_symbol(), Some(spec.symbol));
            assert_eq!(evidence.single_launch_tile(), Some(spec.tile));
            assert_eq!(
                evidence.nodes()[0].launch.grid_dim,
                (columns, rows, spec.partitions)
            );
            assert_eq!(evidence.nodes()[0].launch.block_dim, (spec.threads, 1, 1));
            assert_eq!(
                evidence.nodes()[0].launch.shared_mem_bytes,
                spec.dynamic_shared_bytes
            );

            let numeric_reference = tf32_nt_qualification_reference(dims, 0x6e74_7038).unwrap();
            let mut expected = None;
            for graph in [false, true] {
                for _ in 0..3 {
                    qualified.seed_f32_operands(&ctx, 0x6e74_7038).unwrap();
                    if graph {
                        qualified.measure_graph_window_ms(&ctx, 1).unwrap();
                    } else {
                        qualified.measure_eager_window_ms(&ctx, 1).unwrap();
                    }
                    let actual = qualified.f32_output_bits(&ctx).unwrap();
                    if let Some(expected) = &expected {
                        assert_eq!(&actual, expected);
                    } else {
                        check_accuracy(&actual, &numeric_reference, spec.symbol).unwrap();
                        expected = Some(actual);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn splitk4_forced_route_is_eager_graph_bit_exact_with_one_fused_node() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nn,
            (64, 1_536, 384),
            PhysicalQualificationRoute::Tf32Forced(TF32_SPLITK4_SPEC.route),
        );
        presize_physical_qualification_suite(&ctx, &[request]).unwrap();
        let mut qualified = qualify_physical_launch(&ctx, request).unwrap();
        let evidence = qualified.evidence();
        assert!(evidence.eager_graph_equal());
        assert_eq!(evidence.launch_count(), 1);
        assert_eq!(
            evidence.single_launch_symbol(),
            Some("gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4")
        );
        assert_eq!(evidence.single_launch_tile(), Some((16, 32)));
        assert_eq!(
            evidence.nodes()[0].symbol,
            "gemm_bi_nn_sm80_mma_tf32_splitk4_v1_m16n32_bk32_s4"
        );
        assert_eq!(evidence.nodes()[0].tile, Some((16, 32)));
        assert_eq!(evidence.nodes()[0].launch.grid_dim, (12, 4, 4));
        assert_eq!(evidence.nodes()[0].launch.block_dim, (128, 1, 1));
        assert_eq!(evidence.nodes()[0].launch.shared_mem_bytes, 29_696);

        let mut expected = None;
        for graph in [false, true] {
            for _ in 0..10 {
                qualified.seed_f32_operands(&ctx, 0x5a17_4b21).unwrap();
                if graph {
                    qualified.measure_graph_window_ms(&ctx, 1).unwrap();
                } else {
                    qualified.measure_eager_window_ms(&ctx, 1).unwrap();
                }
                let actual = qualified.f32_output_bits(&ctx).unwrap();
                if let Some(expected) = &expected {
                    assert_eq!(&actual, expected);
                } else {
                    expected = Some(actual);
                }
            }
        }
    }

    fn qualify_splitk_numeric_case(
        ctx: &GpuCtx,
        dims: (usize, usize, usize),
        epilogue: EpilogueCase,
        salt: usize,
        spec: &Tf32SplitKSpec,
    ) -> Result<Vec<u32>, String> {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, dims),
        };
        let host = build_host_case(request, salt, epilogue);
        let mut device = upload_case(ctx, &host)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("finish split-K allocations: {error:?}"))?;
        let prepared =
            prepare_f32_triad_forced(ctx, host.request, device.operands(&host), spec.route)?;

        device.output.upload(&ctx.stream, &host.initial)?;
        let eager_trace = ctx.record_eager_gemm_trace(|| launch_forced(ctx, &prepared))?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("synchronize split-K eager preflight: {error:?}"))?;
        let expected_symbols = [spec.symbol];
        let eager_symbols = eager_trace
            .routes()
            .iter()
            .map(|route| route.symbol)
            .collect::<Vec<_>>();
        if eager_symbols != expected_symbols {
            return Err(format!(
                "split-K eager route order changed: {eager_symbols:?}"
            ));
        }
        let eager = output_bits(ctx, &device.output)?;
        check_accuracy(&eager, &host.expected, spec.symbol)?;
        for iteration in 0..10 {
            device.output.upload(&ctx.stream, &host.initial)?;
            launch_forced(ctx, &prepared)?;
            ctx.stream.synchronize().map_err(|error| {
                format!("synchronize split-K eager repeat {iteration}: {error:?}")
            })?;
            if output_bits(ctx, &device.output)? != eager {
                return Err(format!(
                    "split-K eager output bits changed on repeat {iteration}"
                ));
            }
        }

        device.output.upload(&ctx.stream, &host.initial)?;
        let manifest = eager_trace.manifest();
        let (graph, plan) = unsafe {
            capture_into_graph_with_gemm_plan(ctx, manifest.route_capacity, &manifest, || {
                launch_forced(ctx, &prepared)
            })
        }?;
        let plan = plan.ok_or_else(|| "split-K graph captured no GEMM route".to_string())?;
        if plan.routes() != eager_trace.routes()
            || plan.launches
                != eager_trace
                    .launches()
                    .ok_or_else(|| "split-K eager trace has no physical launches".to_string())?
        {
            return Err("split-K eager and graph physical manifests differ".into());
        }
        for iteration in 0..10 {
            device.output.upload(&ctx.stream, &host.initial)?;
            plan.with_validated_launch(ctx, spec.symbol, || {
                graph
                    .launch()
                    .map_err(|error| format!("launch split-K graph: {error:?}"))
            })?;
            ctx.stream.synchronize().map_err(|error| {
                format!("synchronize split-K graph repeat {iteration}: {error:?}")
            })?;
            if output_bits(ctx, &device.output)? != eager {
                return Err(format!(
                    "split-K eager and graph bits differ on replay {iteration}"
                ));
            }
        }
        Ok(eager)
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn splitk4_deep_reduction_shapes_pass_numeric_and_epilogue_qualification() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        for (index, dims) in [(64, 1_536, 384), (64, 833, 384)].into_iter().enumerate() {
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                adversarial_epilogue(ResolvedGemmOp::Nn),
                0x321 + index,
                &TF32_SPLITK4_SPEC,
            )
            .unwrap();
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                nn_bias_beta_epilogue(),
                0x521 + index,
                &TF32_SPLITK4_SPEC,
            )
            .unwrap();
        }
        for (index, dims) in [(9, 65, 33), (9, 17, 33)].into_iter().enumerate() {
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                adversarial_epilogue(ResolvedGemmOp::Nn),
                0x721 + index,
                &TF32_SPLITK4_SPEC,
            )
            .unwrap();
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                nn_bias_beta_epilogue(),
                0x821 + index,
                &TF32_SPLITK4_SPEC,
            )
            .unwrap();
        }
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn splitk2_nn_m64_k833_n384_m64_k1536_n384_and_tail_shapes_are_bit_exact() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_fast_gemm(false);
        ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
        for (index, dims) in [(64, 833, 384), (64, 1_536, 384), (9, 65, 33), (9, 17, 33)]
            .into_iter()
            .enumerate()
        {
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                adversarial_epilogue(ResolvedGemmOp::Nn),
                0xa21 + index,
                &TF32_SPLITK2_SPEC,
            )
            .unwrap();
            qualify_splitk_numeric_case(
                &ctx,
                dims,
                nn_bias_beta_epilogue(),
                0xb21 + index,
                &TF32_SPLITK2_SPEC,
            )
            .unwrap();
        }
    }

    struct FakePolicyContext {
        batch_invariant: Cell<bool>,
        family: Cell<BiGemmFamily>,
        tensor_cores: Cell<bool>,
        f32_policy: Cell<F32TriadPolicy>,
    }

    fn synthetic_route_with_resources(
        symbol: &'static str,
        boundary_digest: [u8; 32],
        driver_jit_local_bytes: u32,
        approved_driver_jit_local_cap: u32,
    ) -> RouteEvidence {
        RouteEvidence {
            symbol,
            eager_digest: [1; 32],
            graph_digest: [2; 32],
            zero_digest: [3; 32],
            boundary_digest,
            eager_route_digest: [5; 32],
            graph_route_digest: [6; 32],
            tensor_map_digest: [7; 32],
            staged_digest: [8; 32],
            exceptional_digest: [9; 32],
            cross_m_digest: [10; 32],
            adversarial_counts: AdversarialCounts::default(),
            eager_ms: 1.25,
            graph_ms: 1.0,
            registers: 64,
            driver_jit_local_bytes,
            approved_driver_jit_local_cap,
        }
    }

    fn synthetic_route(symbol: &'static str, boundary_digest: [u8; 32]) -> RouteEvidence {
        synthetic_route_with_resources(symbol, boundary_digest, 0, 0)
    }

    fn synthetic_report_digests(boundary: u8) -> ReportDigests {
        ReportDigests {
            artifact: [1; 32],
            output: [2; 32],
            zero: [3; 32],
            boundary: [boundary; 32],
            maps: [5; 32],
            routes: [6; 32],
            staged: [7; 32],
            exceptional: [8; 32],
            cross_m: [9; 32],
            driver_abi: [10; 32],
            driver_jit_resources: [11; 32],
        }
    }

    impl QualificationPolicyContext for FakePolicyContext {
        fn batch_invariant(&self) -> bool {
            self.batch_invariant.get()
        }

        fn set_batch_invariant(&self, value: bool) {
            self.batch_invariant.set(value);
        }

        fn bi_gemm_family(&self) -> BiGemmFamily {
            self.family.get()
        }

        fn set_bi_gemm_family(&self, family: BiGemmFamily) {
            self.family.set(family);
        }

        fn bi_tensor_cores(&self) -> bool {
            self.tensor_cores.get()
        }

        fn set_bi_tensor_cores(&self, value: bool) {
            self.tensor_cores.set(value);
        }

        fn f32_triad_policy(&self) -> F32TriadPolicy {
            self.f32_policy.get()
        }

        fn set_f32_triad_policy(&self, policy: F32TriadPolicy) {
            self.f32_policy.set(policy);
        }
    }

    fn fail_with_qualification_policy(context: &FakePolicyContext) -> Result<(), String> {
        let _guard = QualificationPolicyGuard::enter(context);
        assert!(context.batch_invariant());
        assert_eq!(context.bi_gemm_family(), BiGemmFamily::Triad);
        assert!(!context.bi_tensor_cores());
        assert_eq!(
            context.f32_triad_policy(),
            F32TriadPolicy::AllowDeterministicTf32V1
        );
        Err("injected qualification failure".into())
    }

    #[test]
    fn qualification_policy_guard_restores_every_setting_after_error() {
        let context = FakePolicyContext {
            batch_invariant: Cell::new(false),
            family: Cell::new(BiGemmFamily::Fixed),
            tensor_cores: Cell::new(true),
            f32_policy: Cell::new(F32TriadPolicy::ExactScalarFmaV1),
        };

        assert!(fail_with_qualification_policy(&context).is_err());
        assert!(!context.batch_invariant());
        assert_eq!(context.bi_gemm_family(), BiGemmFamily::Fixed);
        assert!(context.bi_tensor_cores());
        assert_eq!(context.f32_triad_policy(), F32TriadPolicy::ExactScalarFmaV1);
    }

    #[test]
    fn nn_bias_beta_case_matches_the_supported_bias_preseed_contract() {
        let request = F32TriadRequest {
            op: ResolvedGemmOp::Nn,
            shape: F32TriadShape::contiguous(ResolvedGemmOp::Nn, (1, 1, 1)),
        };
        let host = build_host_case(request, 71, nn_bias_beta_epilogue());
        let bias = f64::from(host.bias.as_ref().expect("NN bias case")[0]);
        let product = tf32_value(host.a[0]) * tf32_value(host.b[0]);
        let initial = f64::from(host.initial[0]);
        let expected = product + bias + f64::from(host.beta) * initial;

        assert_eq!(host.alpha, 1.0);
        assert_eq!(host.expected[0], expected);
    }

    #[test]
    fn route_inventory_counts_match_the_release_contract() {
        for (cc, count) in [
            ((8, 0), 19),
            ((8, 6), 19),
            ((8, 7), 19),
            ((8, 9), 19),
            ((9, 0), 25),
            ((10, 0), 55),
            ((10, 3), 55),
            ((11, 0), 55),
            ((12, 0), 36),
            ((12, 1), 36),
        ] {
            assert_eq!(
                tf32_qualification_route_specs(cc)
                    .expect("supported CC")
                    .len(),
                count
            );
        }
    }

    #[test]
    fn wide_qualification_inventory_matches_target_composition() {
        const WIDE: &str = "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3";
        for (cc, expected_wide) in [
            ((8, 0), 1),
            ((8, 6), 1),
            ((8, 7), 1),
            ((8, 9), 1),
            ((9, 0), 1),
            ((10, 0), 1),
            ((10, 3), 1),
            ((11, 0), 1),
            ((12, 0), 0),
            ((12, 1), 0),
        ] {
            let specs = tf32_qualification_route_specs(cc).expect("supported CC");
            assert_eq!(
                specs.iter().filter(|spec| spec.symbol == WIDE).count(),
                expected_wide,
                "wide qualification membership for CC {cc:?}"
            );
            let symbols = specs
                .iter()
                .map(|spec| spec.symbol)
                .collect::<BTreeSet<_>>();
            assert_eq!(symbols.len(), specs.len(), "duplicate route for CC {cc:?}");
            assert!(specs.iter().all(|spec| !spec.route.is_exact_fma()));
        }
        assert!(tf32_qualification_route_specs((8, 1)).is_err());
        assert!(tf32_qualification_route_specs((12, 2)).is_err());
    }

    #[test]
    fn wide_qualification_keeps_the_frozen_cc12_route_order() {
        let frozen = super::super::SM80_TF32_ROUTE_SPECS
            .iter()
            .chain(SM120_TF32_ROUTE_SPECS.iter())
            .filter(|spec| !spec.route.is_exact_fma())
            .map(|spec| spec.symbol)
            .collect::<Vec<_>>();
        assert_eq!(frozen.len(), 36);
        for cc in [(12, 0), (12, 1)] {
            let actual = tf32_qualification_route_specs(cc)
                .expect("frozen CC12 inventory")
                .into_iter()
                .map(|spec| spec.symbol)
                .collect::<Vec<_>>();
            assert_eq!(actual, frozen, "CC12 qualification route order for {cc:?}");
        }
    }

    #[test]
    fn sm120_pair_route_is_qualified_exactly_once() {
        const SYMBOL: &str = "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair";
        for cc in [(12, 0), (12, 1)] {
            assert_eq!(
                tf32_qualification_route_specs(cc)
                    .expect("SM120 route inventory")
                    .iter()
                    .filter(|spec| spec.symbol == SYMBOL)
                    .count(),
                1,
                "CC {}.{}",
                cc.0,
                cc.1
            );
        }
    }

    #[test]
    fn tf32_rounding_uses_round_to_nearest_even() {
        for (input, expected) in [
            (0x3f80_0fff, 0x3f80_0000),
            (0x3f80_1000, 0x3f80_0000),
            (0x3f80_1001, 0x3f80_2000),
            (0xbf80_1000, 0xbf80_0000),
        ] {
            assert_eq!(
                (tf32_value(f32::from_bits(input)) as f32).to_bits(),
                expected
            );
        }
    }

    #[test]
    fn report_digest_is_bound_to_the_exact_json_bytes() {
        let report = build_report(
            Tf32QualificationConfig {
                exact_cc: (8, 9),
                expected_routes: 15,
                repeat: 100,
                suite: Tf32QualificationSuite::Full,
            },
            15,
            AdversarialCounts {
                staged_cases: 60,
                guard_poison_pairs: 15,
                exceptional_symbols: 15,
                cross_m_symbols: 10,
                cross_m_shapes: 40,
                nn_bias_beta_symbols: 5,
            },
            DriverJitResourceSummary {
                max_local_bytes: 0,
                exception_count: 0,
                digest: [11; 32],
            },
            ReportDigests {
                artifact: [1; 32],
                output: [2; 32],
                zero: [3; 32],
                boundary: [4; 32],
                maps: [5; 32],
                routes: [6; 32],
                staged: [7; 32],
                exceptional: [8; 32],
                cross_m: [9; 32],
                driver_abi: [10; 32],
                driver_jit_resources: [11; 32],
            },
        );
        assert!(!report.contains('\n'));
        assert!(report.contains("\"schema\":\"MambaBiTf32QualificationV5\""));
        assert!(report.contains("\"boundary_cases\":705"));
        assert!(report.contains(&format!(
            "\"boundary_output_digest\":\"{}\"",
            digest_hex(&[4; 32])
        )));
        assert!(report.contains("\"staged_cases\":60"));
        let marker = "\"report_digest\":\"";
        let start = report.find(marker).expect("report digest") + marker.len();
        let mut zeroed = report.clone();
        zeroed.replace_range(start..start + 64, &"0".repeat(64));
        assert_eq!(
            &report[start..start + 64],
            digest_hex(&plain_sha256(zeroed.as_bytes()))
        );
        assert!(report.contains(&format!(
            "\"driver_abi_digest\":\"{}\"",
            digest_hex(&[10; 32])
        )));
        assert!(report.contains(&format!(
            "\"driver_jit_resource_digest\":\"{}\"",
            digest_hex(&[11; 32])
        )));
    }

    #[test]
    fn boundary_word_digest_has_a_frozen_bitwise_contract() {
        let values = [0x0000_0000, 0x8000_0000, 0x7fc0_0001];
        let digest = words_digest(b"tf32-qualification-boundary-output.v1", &values);
        assert_eq!(
            digest_hex(&digest),
            "d1a205ee98f9cdade3a51893644f42d4b5aa42d995fb9d5f2f0728eff7c2c8a0"
        );
        for changed in [
            vec![0x8000_0000, 0x0000_0000, 0x7fc0_0001],
            vec![0x0000_0000, 0x8000_0000],
            vec![0x0000_0000, 0x8000_0000, 0x7fc0_0000],
            vec![0x0000_0000, 0x8000_0000, 0x7fc0_0002],
        ] {
            assert_ne!(
                words_digest(b"tf32-qualification-boundary-output.v1", &changed),
                digest
            );
        }
        assert_ne!(words_digest(b"different-domain", &values), digest);
    }

    #[test]
    fn boundary_corpus_binds_every_case_coordinate_and_order() {
        type BoundaryDigestCase<'a> = (
            &'a [u8],
            usize,
            Option<usize>,
            (usize, usize, usize),
            [u8; 32],
        );

        fn corpus(symbol: &[u8], cases: &[BoundaryDigestCase<'_>]) -> [u8; 32] {
            let mut digest = FramedSha256::new(b"tf32-qualification-boundary-corpus.v1")
                .required(b"symbol", symbol);
            for &(kind, index, tail, dims, output) in cases {
                digest = append_boundary_digest_case(digest, kind, index, tail, dims, output);
            }
            digest.finish()
        }

        let first = (b"reduction".as_slice(), 0, None, (128, 1, 64), [1; 32]);
        let second = (b"row-tail".as_slice(), 1, Some(1), (1, 33, 64), [2; 32]);
        let base = corpus(b"route-a", &[first, second]);
        for changed in [
            corpus(b"route-b", &[first, second]),
            corpus(b"route-a", &[second, first]),
            corpus(b"route-a", &[first]),
            corpus(
                b"route-a",
                &[first, (b"column-tail", 1, Some(1), (1, 33, 64), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 2, Some(1), (1, 33, 64), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 1, Some(2), (1, 33, 64), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 1, Some(1), (2, 33, 64), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 1, Some(1), (1, 34, 64), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 1, Some(1), (1, 33, 65), [2; 32])],
            ),
            corpus(
                b"route-a",
                &[first, (b"row-tail", 1, Some(1), (1, 33, 64), [3; 32])],
            ),
        ] {
            assert_ne!(changed, base);
        }
        assert_eq!(boundary_cases_per_route(), 47);
    }

    #[test]
    fn boundary_route_digest_binds_count_order_symbol_and_value() {
        let first = synthetic_route("route-a", [4; 32]);
        let second = synthetic_route("route-b", [5; 32]);
        let base = semantic_digest(
            b"tf32-qualification-all-boundary-output.v1",
            &[first, second],
            |route| route.boundary_digest,
        );
        for changed in [
            semantic_digest(
                b"tf32-qualification-all-boundary-output.v1",
                &[synthetic_route("route-a", [4; 32])],
                |route| route.boundary_digest,
            ),
            semantic_digest(
                b"tf32-qualification-all-boundary-output.v1",
                &[
                    synthetic_route("route-b", [5; 32]),
                    synthetic_route("route-a", [4; 32]),
                ],
                |route| route.boundary_digest,
            ),
            semantic_digest(
                b"tf32-qualification-all-boundary-output.v1",
                &[
                    synthetic_route("route-c", [4; 32]),
                    synthetic_route("route-b", [5; 32]),
                ],
                |route| route.boundary_digest,
            ),
            semantic_digest(
                b"tf32-qualification-all-boundary-output.v1",
                &[
                    synthetic_route("route-a", [6; 32]),
                    synthetic_route("route-b", [5; 32]),
                ],
                |route| route.boundary_digest,
            ),
        ] {
            assert_ne!(changed, base);
        }
    }

    #[test]
    fn qualification_artifact_v5_freezes_driver_jit_resources_per_route() {
        let config = Tf32QualificationConfig {
            exact_cc: (8, 9),
            expected_routes: 1,
            repeat: 1,
            suite: Tf32QualificationSuite::Sanitizer,
        };
        let boundary = [4; 32];
        let artifact = build_artifact(
            config,
            &[synthetic_route_with_resources("route-a", boundary, 16, 16)],
            [11; 32],
            [12; 32],
        );
        let artifact = String::from_utf8(artifact).expect("UTF-8 qualification artifact");
        let row = artifact.lines().last().expect("qualification route row");
        let fields = row.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 16);
        assert_eq!(fields[0], "route-a");
        assert_eq!(fields[4], digest_hex(&boundary));
        for digest in &fields[1..=10] {
            assert_eq!(digest.len(), 64);
            assert!(
                digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            );
        }
        assert_eq!(fields[14], "16");
        assert_eq!(fields[15], "16");
    }

    #[test]
    fn sm120_driver_jit_inventory_requires_all_routes_to_be_zero() {
        let config = Tf32QualificationConfig {
            exact_cc: (12, 0),
            expected_routes: 36,
            repeat: 1,
            suite: Tf32QualificationSuite::Sanitizer,
        };
        let specs = tf32_qualification_route_specs(config.exact_cc).expect("SM120 route inventory");
        let mut evidence = specs
            .iter()
            .map(|spec| synthetic_route_with_resources(spec.symbol, [4; 32], 0, 0))
            .collect::<Vec<_>>();

        let summary = validate_driver_jit_resource_inventory(config, &specs, &evidence)
            .expect("exact SM120 Driver JIT resource inventory");
        assert_eq!(summary.max_local_bytes, 0);
        assert_eq!(summary.exception_count, 0);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.module_kind == ModuleKind::TriadSm120)
                .count(),
            18
        );

        let rejected = evidence
            .iter_mut()
            .find(|route| route.symbol.contains("_sm120_"))
            .expect("SM120 route");
        rejected.driver_jit_local_bytes = 16;
        rejected.approved_driver_jit_local_cap = 16;
        assert!(validate_driver_jit_resource_inventory(config, &specs, &evidence).is_err());
    }

    #[test]
    fn driver_jit_resource_digest_binds_route_identity_registers_observed_bytes_and_cap() {
        let base = driver_jit_resource_summary(&[synthetic_route_with_resources(
            "route-a", [4; 32], 16, 16,
        )])
        .digest;
        let mut changed_identity = synthetic_route_with_resources("route-a", [4; 32], 16, 16);
        changed_identity.eager_route_digest = [22; 32];
        let mut changed_registers = synthetic_route_with_resources("route-a", [4; 32], 16, 16);
        changed_registers.registers = 65;
        for changed in [
            driver_jit_resource_summary(&[changed_identity]).digest,
            driver_jit_resource_summary(&[changed_registers]).digest,
            driver_jit_resource_summary(&[synthetic_route_with_resources(
                "route-a", [4; 32], 0, 16,
            )])
            .digest,
            driver_jit_resource_summary(&[synthetic_route_with_resources(
                "route-a", [4; 32], 16, 32,
            )])
            .digest,
        ] {
            assert_ne!(changed, base);
        }
    }

    #[test]
    fn report_digest_changes_with_only_the_boundary_corpus() {
        let config = Tf32QualificationConfig {
            exact_cc: (8, 9),
            expected_routes: 15,
            repeat: 1,
            suite: Tf32QualificationSuite::Sanitizer,
        };
        let counts = AdversarialCounts::default();
        let resources = DriverJitResourceSummary {
            max_local_bytes: 0,
            exception_count: 0,
            digest: [12; 32],
        };
        let first = build_report(config, 15, counts, resources, synthetic_report_digests(4));
        let second = build_report(config, 15, counts, resources, synthetic_report_digests(11));
        assert_ne!(first, second);
        assert!(first.contains(&format!(
            "\"boundary_output_digest\":\"{}\"",
            digest_hex(&[4; 32])
        )));
        assert!(second.contains(&format!(
            "\"boundary_output_digest\":\"{}\"",
            digest_hex(&[11; 32])
        )));
        let marker = "\"report_digest\":\"";
        let first_digest = &first[first.find(marker).unwrap() + marker.len()..][..64];
        let second_digest = &second[second.find(marker).unwrap() + marker.len()..][..64];
        assert_ne!(first_digest, second_digest);
    }

    #[test]
    fn artifact_binds_the_exact_driver_abi_proof_digest() {
        let config = Tf32QualificationConfig {
            exact_cc: (8, 9),
            expected_routes: 15,
            repeat: 100,
            suite: Tf32QualificationSuite::Full,
        };
        let proof_digest = plain_sha256(b"canonical-driver-abi");
        let artifact = build_artifact(config, &[], [1; 32], proof_digest);
        let artifact = String::from_utf8(artifact).expect("UTF-8 qualification artifact");

        assert!(artifact.starts_with("MambaBiTf32QualificationArtifactV5\n"));
        assert!(artifact.contains("boundary_cases_per_route\t47\n"));
        assert!(artifact.contains(&format!("driver_abi\t{}", digest_hex(&proof_digest))));
        assert!(!artifact.contains(&digest_hex(&plain_sha256(b"different-driver-abi"))));
    }

    #[test]
    fn padded_shapes_keep_every_physical_row_inside_an_aligned_stride() {
        let nn = padded_shape(ResolvedGemmOp::Nn, (17, 65, 17), 8).unwrap();
        assert_eq!((nn.lda, nn.ldb, nn.ldc), (76, 28, 28));

        let tn = padded_shape(
            ResolvedGemmOp::Tn,
            logical_dims(ResolvedGemmOp::Tn, 17, 17, 65),
            8,
        )
        .unwrap();
        assert_eq!((tn.lda, tn.ldb, tn.ldc), (28, 28, 28));

        let nt = padded_shape(
            ResolvedGemmOp::Nt,
            logical_dims(ResolvedGemmOp::Nt, 17, 17, 65),
            8,
        )
        .unwrap();
        assert_eq!((nt.lda, nt.ldb, nt.ldc), (76, 76, 28));
        for shape in [nn, tn, nt] {
            assert_eq!(shape.lda % 4, 0);
            assert_eq!(shape.ldb % 4, 0);
            assert_eq!(shape.ldc % 4, 0);
        }
    }

    #[test]
    fn portable_primary_cases_stay_contiguous_but_tma_cases_are_stride_aligned() {
        let portable = tf32_qualification_route_specs((8, 9)).unwrap()[0];
        let portable_shape =
            shape_for_spec(portable, logical_dims(portable.op, 17, 17, 33)).unwrap();
        assert_eq!(portable_shape.lda, 33);

        let specialized = SM100_TF32_ROUTE_SPECS.first().unwrap();
        let specialized_shape =
            shape_for_spec(specialized, logical_dims(specialized.op, 17, 17, 33)).unwrap();
        assert_eq!(specialized_shape.lda, 36);
        assert_eq!(specialized_shape.ldb % 4, 0);
        assert_eq!(specialized_shape.ldc % 4, 0);
    }

    #[test]
    fn wide_qualification_tail_shapes_keep_vector_copy_strides_aligned() {
        let wide = &super::super::SM80_TF32_WIDE_ROUTE_SPECS[0];
        for reduction in K_CASES {
            for columns in TAIL_CASES {
                let shape = shape_for_spec(wide, (1, reduction, columns))
                    .expect("wide qualification tail layout");
                assert_eq!((shape.m, shape.k, shape.n), (1, reduction, columns));
                assert!(shape.lda >= reduction && shape.ldb >= columns);
                assert_eq!(shape.lda % 4, 0, "wide A stride at K={reduction}");
                assert_eq!(shape.ldb % 4, 0, "wide B stride at N={columns}");
                shape
                    .validate(ResolvedGemmOp::Nn)
                    .expect("valid wide shape");
            }
        }
    }

    #[test]
    fn guarded_storage_has_nonzero_xy_origin_and_a_trailing_red_zone() {
        let storage = guarded_storage(17, 65, 76, 1).unwrap();
        assert_eq!(storage.offset, 77);
        assert_eq!((storage.origin_x, storage.origin_y), (1, 1));
        assert_eq!(storage.len, 77 + 16 * 76 + 65 + 32);
        assert!(storage.offset + 16 * storage.stride + storage.width <= storage.len - 32);
    }

    #[test]
    fn guarded_tf32_inputs_meet_the_tma_global_address_alignment() {
        let host = build_guarded_case(
            ResolvedGemmOp::Nn,
            (65, 65, 65),
            11,
            adversarial_epilogue(ResolvedGemmOp::Nn),
            0x7fc1_5a5a,
        )
        .unwrap();
        assert_eq!(host.a_storage.offset * size_of::<f32>() % 16, 0);
        assert_eq!(host.b_storage.offset * size_of::<f32>() % 16, 0);
        assert_ne!(host.a_storage.origin_x, 0);
        assert_ne!(host.b_storage.origin_x, 0);
    }

    #[test]
    fn exceptional_oracle_classifies_without_freezing_nan_payloads_or_zero_signs() {
        assert_eq!(
            exceptional_class(f32::from_bits(0x7fc1_2345)),
            ExceptionalClass::Nan
        );
        assert_eq!(
            exceptional_class(f32::from_bits(0x7f81_2345)),
            ExceptionalClass::Nan
        );
        assert_eq!(
            exceptional_class(f32::NEG_INFINITY),
            ExceptionalClass::Infinity { negative: true }
        );
        assert_eq!(exceptional_class(-0.0), ExceptionalClass::Zero);
        assert_eq!(exceptional_class(f32::from_bits(1)), ExceptionalClass::Zero);
        assert_eq!(
            exceptional_class(f32::from_bits(0x007f_ffff)),
            ExceptionalClass::Finite(f32::from_bits(0x0080_0000) as f64)
        );
        assert_eq!(
            exceptional_class(f32::MAX),
            ExceptionalClass::Infinity { negative: false }
        );
    }

    #[test]
    fn wide_exceptional_oracle_observes_the_add_half_ulp_operand_classes() {
        let wide = &super::super::SM80_TF32_WIDE_ROUTE_SPECS[0];
        assert_eq!(
            wide.operand_conversion,
            ResolvedOperandConversion::RegisterAddHalfUlpTf32V1
        );
        // Literal expectations describe the bits consumed by MMA after the
        // register add and discarded low 13 bits, not the original f32 class.
        for (bits, expected) in [
            (0x7f80_0001, ExceptionalClass::Infinity { negative: false }),
            (0xff80_0001, ExceptionalClass::Infinity { negative: true }),
            (0x7f80_1000, ExceptionalClass::Nan),
            (0xff80_1000, ExceptionalClass::Nan),
            (0x7f80_2000, ExceptionalClass::Nan),
            (0xff80_2000, ExceptionalClass::Nan),
            (0x7fff_ffff, ExceptionalClass::Zero),
            (0xffff_ffff, ExceptionalClass::Zero),
            (0x7fc0_1234, ExceptionalClass::Nan),
            (0xffc0_1234, ExceptionalClass::Nan),
            (0x7f80_0000, ExceptionalClass::Infinity { negative: false }),
            (0xff80_0000, ExceptionalClass::Infinity { negative: true }),
            (0x7f7f_ffff, ExceptionalClass::Infinity { negative: false }),
            (0xff7f_ffff, ExceptionalClass::Infinity { negative: true }),
            (0x3f80_1000, ExceptionalClass::Finite(1.000_976_562_5)),
            (0xbf80_1000, ExceptionalClass::Finite(-1.000_976_562_5)),
            (0x0000_0000, ExceptionalClass::Zero),
            (0x8000_0000, ExceptionalClass::Zero),
            (0x0000_0001, ExceptionalClass::Zero),
            (0x8000_0001, ExceptionalClass::Zero),
            (
                0x007f_ffff,
                ExceptionalClass::Finite(f32::MIN_POSITIVE as f64),
            ),
            (
                0x807f_ffff,
                ExceptionalClass::Finite(-(f32::MIN_POSITIVE as f64)),
            ),
        ] {
            assert_eq!(
                exceptional_class_for_spec(wide, f32::from_bits(bits)),
                expected,
                "wide operand 0x{bits:08x}"
            );
        }
    }

    #[test]
    fn wide_exceptional_corpus_exercises_conversion_edges_on_the_bound_route() {
        let wide = &super::super::SM80_TF32_WIDE_ROUTE_SPECS[0];
        let values = exceptional_values_for_spec(wide);
        for required in [
            0x7f80_0001,
            0xff80_0001,
            0x7f80_1000,
            0xff80_1000,
            0x7f80_2000,
            0xff80_2000,
            0x7fff_ffff,
            0xffff_ffff,
            0x3f80_1000,
            0xbf80_1000,
            0x0000_0000,
            0x8000_0001,
            0x807f_ffff,
            0xff7f_ffff,
            0x7fc0_1234,
            0xffc0_1234,
        ] {
            assert!(
                values.contains(&required),
                "wide corpus omits 0x{required:08x}"
            );
        }
        assert_eq!(values.len(), 24);
        assert_eq!(values.iter().copied().collect::<BTreeSet<_>>().len(), 24);
    }

    #[test]
    fn wide_exceptional_changes_leave_existing_conversion_corpora_unchanged() {
        for spec in super::super::SM80_TF32_ROUTE_SPECS
            .iter()
            .chain(SM90A_TF32_ROUTE_SPECS.iter())
            .chain(SM100_TF32_ROUTE_SPECS.iter())
            .chain(SM120_TF32_ROUTE_SPECS.iter())
            .filter(|spec| !spec.route.is_exact_fma())
        {
            assert_eq!(
                exceptional_values_for_spec(spec),
                &[
                    0x8000_0000,
                    0x0000_0001,
                    0x007f_ffff,
                    0x7f7f_ffff,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc1_2345,
                    0x7f81_2345,
                ],
                "legacy exceptional corpus changed for {}",
                spec.symbol
            );
            for bits in exceptional_values_for_spec(spec).iter().copied() {
                assert_eq!(
                    exceptional_class_for_spec(spec, f32::from_bits(bits)),
                    exceptional_class(f32::from_bits(bits)),
                    "legacy class changed for {} operand 0x{bits:08x}",
                    spec.symbol
                );
            }
            // These distinguish the legacy oracle from the wide one even
            // though the frozen eight-value corpus does not include them.
            for (bits, expected) in [
                (0x7f80_0001, ExceptionalClass::Nan),
                (0x7fff_ffff, ExceptionalClass::Nan),
                (0x3f80_1000, ExceptionalClass::Finite(1.0)),
                (0xbf80_1000, ExceptionalClass::Finite(-1.0)),
            ] {
                assert_eq!(
                    exceptional_class_for_spec(spec, f32::from_bits(bits)),
                    expected,
                    "wide oracle leaked into {} for 0x{bits:08x}",
                    spec.symbol
                );
            }
        }
    }

    #[test]
    #[ignore = "requires exact Ada CC 8.9 and the bound portable wide TF32 kernel"]
    fn wide_tf32_qualification_numeric_smoke() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "wide smoke requires Ada");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let _policy_guard = QualificationPolicyGuard::enter(&ctx);
        let spec = tf32_qualification_route_specs(device.compute_capability)
            .expect("Ada qualification inventory")
            .into_iter()
            .find(|spec| spec.symbol == "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3")
            .expect("wide must be in the admitted qualification inventory");

        // These are the full corpus's production-bound helpers, not a
        // separately compiled kernel. No timing or all-route qualification.
        qualify_exceptional_values(&ctx, spec, 0x89_0100)
            .expect("wide exceptional classes, repeat bits and red zones");
        qualify_zero_reduction(&ctx, spec, 0x89_0200)
            .expect("wide K=0 null inputs, bias bits and eager/graph symbol identity");
        qualify_shape_boundaries(&ctx, spec, 0x89_0300)
            .expect("wide complete logical K/row/column tail corpus");
        let (_, symbols, shapes) = qualify_cross_m(&ctx, spec, 0x89_0400)
            .expect("wide guarded M=1/tile/tile+1/two-tiles+1 prefix bits");
        assert_eq!((symbols, shapes), (1, 4));
    }

    #[test]
    fn adversarial_counts_are_derived_from_the_live_route_inventory() {
        let specs = tf32_qualification_route_specs((8, 9)).unwrap();
        let counts = adversarial_counts(&specs);
        assert_eq!(counts.staged_cases, 76);
        assert_eq!(counts.guard_poison_pairs, 19);
        assert_eq!(counts.exceptional_symbols, 19);
        assert_eq!(counts.cross_m_symbols, 13);
        assert_eq!(counts.cross_m_shapes, 52);
        assert_eq!(counts.nn_bias_beta_symbols, 7);
    }

    fn physical_node(symbol: &'static str) -> QualifiedPhysicalLaunchNode {
        QualifiedPhysicalLaunchNode {
            kind: PhysicalLaunchKind::Gemm,
            symbol,
            module_kind: ModuleKind::TriadScalar,
            logical_op: ResolvedGemmOp::Nn,
            logical_dtype: PolicyDtype::F32,
            execution_dtype: PolicyDtype::F32,
            shape: (64, 64, 64),
            strides: (64, 64, 64),
            tile: Some((64, 64)),
            numeric_contract: None,
            ownership: None,
            launch: ResolvedKernelLaunch {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
                arguments_digest: [7; 32],
            },
        }
    }

    fn sm120_branch_fixture() -> (Sm120AutoBranchSeal, ResolvedPhysicalKernelLaunch) {
        let forced = SM120_AUTO_CELLS_CC120[0];
        let target = sm120_target_candidates((12, 0), (12, 8))[0];
        let compiler = CompilerIdentity {
            source_digest: [21; 32],
            invocation_digest: [22; 32],
            header_manifest_digest: [23; 32],
            target: CudaTarget::new(target.nvrtc_arch).unwrap(),
            nvrtc_version: (12, 8),
            nvrtc_library_domain: [24; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = ArtifactIdentity {
            module_kind: ModuleKind::TriadSm120,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: [25; 32],
            artifact_digest: [26; 32],
        };
        let caps = DeviceCaps {
            compute_capability: (12, 0),
            nvrtc_version: (12, 8),
            accepted_target: Some(compiler.target),
            optin_shared_bytes: 101_376,
            tensor_map_access: true,
        };
        let identity = Sm120RouteIdentity {
            numeric_contract: Sm120NumericContract::TmaMma16F32V1,
            op: forced.op,
            dtype: forced.dtype,
            physical: forced.physical,
            shape: forced.shape,
            symbol: forced.kernel_spec().unwrap().symbol,
            module_kind: ModuleKind::TriadSm120,
            target,
            artifact,
            compiler,
            device: DeviceIdentity {
                compute_capability: (12, 0),
                multiprocessor_count: 24,
                target: CudaTarget::new(target.ptx_target).unwrap(),
                driver: DriverIdentity {
                    api_version: 12_800,
                    build_sources: 1,
                    build_digest: [27; 32],
                },
            },
            device_caps: caps,
            tensor_map_revision: SM120_TENSOR_MAP_REVISION,
            tensor_maps_digest: [28; 32],
            resources_digest: [29; 32],
            tuning_revision: SM120_TUNING_REVISION,
            schedule_revision: SM120_SCHEDULE_REVISION,
        };
        let route = identity.resolved_route().unwrap();
        let node = ResolvedPhysicalKernelLaunch {
            kind: PhysicalLaunchKind::Gemm,
            symbol: route.symbol,
            module_kind: route.module_kind,
            logical_op: route.op,
            logical_dtype: route.dtype,
            execution_dtype: route.dtype,
            shape: route.shape,
            strides: route.strides,
            tile: Some(route.tile),
            launch: route.launch,
            gemm_route: Some(route),
        };
        (
            Sm120AutoBranchSeal {
                route: Sm120ForcedRoute {
                    op: Sm120Op::Nn,
                    dtype: WeightDtype::Bf16,
                    physical: identity.physical,
                    shape: identity.shape,
                },
            },
            node,
        )
    }

    #[test]
    fn sm120_branch_seal_requires_full_exact_eager_and_graph_routes() {
        let (seal, eager) = sm120_branch_fixture();
        let recorded = eager.gemm_route().unwrap();
        validate_sm120_half_branch(seal, &[recorded], &[eager], &[eager]).unwrap();
        assert!(
            normalized_half_production_routes(HalfPolicyBranchSeal::Sm120(seal), &[recorded])
                .unwrap()
                .as_ref()
                == [recorded]
        );
        let mut mismatched_record = recorded;
        mismatched_record.resources_digest[0] ^= 1;
        assert!(
            validate_sm120_half_branch(seal, &[mismatched_record], &[eager], &[eager]).is_err()
        );

        let mut f32_route = eager.gemm_route().unwrap();
        f32_route.dtype = PolicyDtype::F32;
        assert!(validate_sm120_half_branch(seal, &[f32_route], &[eager], &[eager]).is_err());

        for field in 0..3 {
            let mut graph = eager;
            let mut route = graph.gemm_route().unwrap();
            match field {
                0 => route.tensor_maps_digest[0] ^= 1,
                1 => route.resources_digest[0] ^= 1,
                _ => {
                    route.launch.arguments_digest[0] ^= 1;
                    graph.launch = route.launch;
                }
            }
            graph.gemm_route = Some(route);
            assert!(
                validate_sm120_half_branch(seal, &[recorded], &[eager], &[graph]).is_err(),
                "mutated full-route field {field}"
            );
        }
        assert!(validate_sm120_half_branch(seal, &[recorded], &[eager, eager], &[eager]).is_err());
    }

    fn physical_contract<'a>(
        eager_nodes: &'a [QualifiedPhysicalLaunchNode],
        graph_nodes: &'a [QualifiedPhysicalLaunchNode],
    ) -> PhysicalEvidenceContract<'a> {
        PhysicalEvidenceContract {
            timed_request: PhysicalTimedRequestIdentity {
                op: ResolvedGemmOp::Nn,
                logical_dtype: PolicyDtype::F32,
                dims: (64, 64, 64),
                planned_symbol: None,
                planned_tile: None,
                layout: PhysicalQualificationLayout::Contiguous,
                route: PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
                alpha_bits: 1.0_f32.to_bits(),
                beta_bits: 0.0_f32.to_bits(),
                bias: false,
            },
            eager_nodes,
            graph_nodes,
            eager_launch_count: eager_nodes.len() as u32,
            graph_launch_count: graph_nodes.len() as u32,
            eager_digest: [3; 32],
            graph_digest: [3; 32],
            physical_symbol: (eager_nodes.len() == 1).then(|| eager_nodes[0].symbol),
            tile: (eager_nodes.len() == 1)
                .then(|| eager_nodes[0].tile)
                .flatten(),
        }
    }

    #[test]
    fn physical_evidence_contract_rejects_missing_or_planned_only_routes() {
        let empty = physical_contract(&[], &[]);
        assert!(validate_physical_evidence_contract(&empty).is_err());

        let nodes = [physical_node("actual")];
        let mut planned_only = physical_contract(&nodes, &nodes);
        planned_only.timed_request.planned_symbol = Some("planned_but_not_enqueued");
        assert!(validate_physical_evidence_contract(&planned_only).is_err());
    }

    #[test]
    fn physical_evidence_contract_rejects_graph_drift_and_multi_launch_singular_fields() {
        let eager = [physical_node("first"), physical_node("second")];
        let reordered = [physical_node("second"), physical_node("first")];
        let graph_drift = physical_contract(&eager, &reordered);
        assert!(validate_physical_evidence_contract(&graph_drift).is_err());

        let mut singular = physical_contract(&eager, &eager);
        singular.physical_symbol = Some("first");
        singular.tile = Some((64, 64));
        assert!(validate_physical_evidence_contract(&singular).is_err());
    }

    #[test]
    fn half_branch_seal_rejects_production_and_physical_route_drift() {
        let seal = HalfNativeBranchSeal {
            base: "gemm_bi_nn_narrow_small",
            op: ResolvedGemmOp::Nn,
            dtype: WeightDtype::Bf16,
            dims: (64, 64, 64),
            strides: (64, 64, 64),
            tile: (16, 16),
            bk_stages: (16, 1),
            grid_dim: (16, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut physical = native_half_seal_projection(seal).expect("valid half seal");
        physical.base = "gemm_bi_nn_tc64";
        physical.tile = Some((64, 64));

        assert_ne!(native_half_seal_projection(seal).unwrap(), physical);
    }

    #[test]
    fn f32_branch_identity_rejects_argument_digest_drift() {
        let (_, node) = sm120_branch_fixture();
        let production = node.gemm_route().unwrap();
        let mut physical = production;
        physical.launch.arguments_digest[0] ^= 1;
        assert!(!same_f32_physical_branch(production, physical));

        let mut capture_projection = production;
        capture_projection.resources_digest[0] ^= 1;
        capture_projection.tensor_maps_digest[0] ^= 1;
        assert!(same_f32_physical_branch(production, capture_projection));
    }

    #[test]
    fn timed_request_digest_separates_policy_mode_and_forced_tile() {
        let dims = (256, 512, 384);
        let routes = [
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1),
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: false,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
            PhysicalQualificationRoute::HalfPolicy {
                dtype: WeightDtype::Bf16,
                tensor_cores: true,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
            PhysicalQualificationRoute::HalfForced {
                dtype: WeightDtype::Bf16,
                tile: TcTile::Tile64,
            },
            PhysicalQualificationRoute::HalfForced {
                dtype: WeightDtype::Bf16,
                tile: TcTile::Tile128,
            },
        ];
        let digests = routes
            .map(|route| {
                PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, dims, route)
                    .timed_identity()
                    .map(timed_request_digest)
                    .expect("valid timed request")
            })
            .into_iter()
            .collect::<BTreeSet<_>>();

        assert_eq!(digests.len(), routes.len());
    }

    #[test]
    fn g2_half_auto_request_preserves_permission_and_separates_identity() {
        let mut digests = BTreeSet::new();
        for op in [ResolvedGemmOp::Nn, ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
                for tensor_cores in [false, true] {
                    for half_policy in [
                        HalfTriadPolicy::TiledParityV1,
                        HalfTriadPolicy::AllowStreamKFixedOrderV1,
                    ] {
                        let route = PhysicalQualificationRoute::HalfPolicy {
                            dtype,
                            tensor_cores,
                            half_policy,
                        };
                        let policy = route.policy();
                        assert_eq!(policy.bi_tensor_cores, tensor_cores);
                        assert_eq!(policy.half_triad_policy, half_policy);
                        assert_eq!(policy.f32_triad_policy, F32TriadPolicy::ExactScalarFmaV1);
                        let request =
                            PhysicalQualificationRequest::contiguous(op, (256, 512, 384), route);
                        // Permission is valid for every op and also when TC is
                        // disabled; it never nominates or promises a schedule.
                        request.validate().expect("valid half policy permission");
                        let identity = request.timed_identity().expect("half AUTO identity");
                        assert_eq!(identity.planned_symbol, None);
                        assert_eq!(identity.planned_tile, None);
                        assert_eq!(identity.route, route);
                        let digest = timed_request_digest(identity);
                        assert_eq!(
                            digest,
                            timed_request_digest(request.timed_identity().unwrap()),
                            "identical request digest changed"
                        );
                        assert!(
                            digests.insert(digest),
                            "half AUTO request identities alias: {op:?} {route:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(digests.len(), 24);
    }

    #[test]
    fn g2_half_tiled_request_digests_preserve_the_legacy_encoding() {
        // Independently computed before adding the permission field. These
        // freeze NN (256,512,384), alpha=1, beta=0, no bias, contiguous layout.
        for (dtype, tensor_cores, expected) in [
            (
                WeightDtype::Bf16,
                false,
                "3258c17392c48db82d2772b6fa585cfb1b9920722c9c3df2d61b921bc937235a",
            ),
            (
                WeightDtype::Bf16,
                true,
                "4ec1dff5b852a30a3a35e52e8e37b2af789e51b74403dca415b60354aad707f5",
            ),
            (
                WeightDtype::F16,
                false,
                "12c69c6d6fe5f211d1056af91d02915d0c579a98f5077a8bf4464e6a1f6a3c46",
            ),
            (
                WeightDtype::F16,
                true,
                "fffc2cfcca16831e4e00eb7cf4594a378f4f59da0e223def24005f862f7051ae",
            ),
        ] {
            let request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                (256, 512, 384),
                PhysicalQualificationRoute::HalfPolicy {
                    dtype,
                    tensor_cores,
                    half_policy: HalfTriadPolicy::TiledParityV1,
                },
            );
            assert_eq!(
                digest_hex(&timed_request_digest(request.timed_identity().unwrap())),
                expected,
                "legacy tiled request changed for {dtype:?}, TC={tensor_cores}"
            );
        }
    }

    #[test]
    fn request_validation_rejects_invalid_extent_dtype_and_thin_tile() {
        let scalar = PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1);
        assert!(
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, (0, 1, 1), scalar)
                .validate()
                .is_err()
        );
        assert!(
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, (1, 1, 0), scalar)
                .validate()
                .is_err()
        );
        let half_f32 = PhysicalQualificationRoute::HalfPolicy {
            dtype: WeightDtype::F32,
            tensor_cores: false,
            half_policy: HalfTriadPolicy::TiledParityV1,
        };
        assert!(
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, (1, 0, 1), half_f32)
                .validate()
                .is_err()
        );
        let thin_tn = PhysicalQualificationRoute::HalfForced {
            dtype: WeightDtype::F16,
            tile: TcTile::Thin16,
        };
        assert!(
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Tn, (64, 64, 64), thin_tn)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn edge_layout_requests_freeze_named_offsets_and_direct_nn_strides() {
        let route = PhysicalQualificationRoute::HalfForced {
            dtype: WeightDtype::Bf16,
            tile: TcTile::Tile64,
        };
        let aligned =
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, (129, 65, 129), route);
        let a_offset = PhysicalQualificationRequest::one_element_offset(
            ResolvedGemmOp::Nn,
            (129, 65, 129),
            route,
            PhysicalQualificationOffset::A,
        );
        let b_offset = PhysicalQualificationRequest::one_element_offset(
            ResolvedGemmOp::Nn,
            (129, 65, 129),
            route,
            PhysicalQualificationOffset::B,
        );
        let output_offset = PhysicalQualificationRequest::one_element_offset(
            ResolvedGemmOp::Nn,
            (129, 65, 129),
            route,
            PhysicalQualificationOffset::Output,
        );

        for request in [aligned, a_offset, b_offset, output_offset] {
            request.validate().expect("valid named edge layout");
        }
        let identities = [aligned, a_offset, b_offset, output_offset]
            .map(|request| timed_request_digest(request.timed_identity().unwrap()));
        assert_eq!(identities.into_iter().collect::<BTreeSet<_>>().len(), 4);

        let first = PhysicalQualificationRequest::direct_padded_nn(
            (129, 65, 129),
            WeightDtype::F16,
            TcTile::Tile64,
            PhysicalPaddedNnLayout::Lda72Ldb136Ldc130,
        );
        let second = PhysicalQualificationRequest::direct_padded_nn(
            (129, 65, 129),
            WeightDtype::F16,
            TcTile::Tile64,
            PhysicalPaddedNnLayout::Lda72Ldb136Ldc131,
        );
        let first_shape = first.physical_shape();
        let second_shape = second.physical_shape();
        assert_eq!(
            (first_shape.lda, first_shape.ldb, first_shape.ldc),
            (72, 136, 130)
        );
        assert_eq!(
            (second_shape.lda, second_shape.ldb, second_shape.ldc),
            (72, 136, 131)
        );
        assert_ne!(
            timed_request_digest(first.timed_identity().unwrap()),
            timed_request_digest(second.timed_identity().unwrap())
        );
    }

    #[test]
    fn padded_tn_nt_layouts_fail_before_resource_allocation() {
        for op in [ResolvedGemmOp::Tn, ResolvedGemmOp::Nt] {
            let request = PhysicalQualificationRequest {
                op,
                dims: (129, 65, 129),
                route: PhysicalQualificationRoute::HalfForced {
                    dtype: WeightDtype::Bf16,
                    tile: TcTile::Tile128,
                },
                layout: PhysicalQualificationLayout::DirectPaddedNn(
                    PhysicalPaddedNnLayout::Lda72Ldb136Ldc130,
                ),
                f32_epilogue: None,
            };
            assert!(request.validate().is_err());
        }
    }

    #[test]
    fn f32_epilogue_request_preserves_contract_and_identity() {
        let route = PhysicalQualificationRoute::F32Policy(F32TriadPolicy::AllowDeterministicTf32V1);
        let plain =
            PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Nn, (64, 384, 1536), route);
        let biased = PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            (64, 384, 1536),
            route,
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, true),
        );

        plain.validate().unwrap();
        biased.validate().unwrap();
        assert_eq!(
            plain.f32_epilogue(),
            PhysicalQualificationF32Epilogue::for_op(ResolvedGemmOp::Nn)
        );
        assert_eq!(
            biased.f32_epilogue(),
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, true)
        );
        assert_ne!(
            timed_request_digest(plain.timed_identity().unwrap()),
            timed_request_digest(biased.timed_identity().unwrap())
        );
    }

    #[test]
    fn f32_policy_epilogue_rejects_values_the_production_wrappers_cannot_express() {
        let policy = PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1);
        for (op, epilogue) in [
            (
                ResolvedGemmOp::Nn,
                PhysicalQualificationF32Epilogue::new(0.5, 0.0, false),
            ),
            (
                ResolvedGemmOp::Nn,
                PhysicalQualificationF32Epilogue::new(1.0, 1.0, false),
            ),
            (
                ResolvedGemmOp::Nn,
                PhysicalQualificationF32Epilogue::new(1.0, -0.0, false),
            ),
            (
                ResolvedGemmOp::Tn,
                PhysicalQualificationF32Epilogue::new(0.5, 1.0, false),
            ),
            (
                ResolvedGemmOp::Tn,
                PhysicalQualificationF32Epilogue::new(1.0, 0.0, false),
            ),
            (
                ResolvedGemmOp::Nt,
                PhysicalQualificationF32Epilogue::new(0.5, 0.0, false),
            ),
            (
                ResolvedGemmOp::Nt,
                PhysicalQualificationF32Epilogue::new(1.0, 1.0, false),
            ),
        ] {
            let request =
                PhysicalQualificationRequest::contiguous_f32(op, (64, 384, 1536), policy, epilogue);
            assert!(
                request.validate().is_err(),
                "policy request admitted unsupported {op:?} epilogue {epilogue:?}"
            );
        }
    }

    #[test]
    fn f32_policy_epilogue_accepts_exact_wrapper_values_and_forced_routes_stay_general() {
        for policy in [
            F32TriadPolicy::ExactScalarFmaV1,
            F32TriadPolicy::AllowDeterministicTf32V1,
        ] {
            for (op, bias) in [
                (ResolvedGemmOp::Nn, false),
                (ResolvedGemmOp::Nn, true),
                (ResolvedGemmOp::Tn, false),
                (ResolvedGemmOp::Nt, false),
            ] {
                PhysicalQualificationRequest::contiguous_f32(
                    op,
                    (64, 384, 1536),
                    PhysicalQualificationRoute::F32Policy(policy),
                    PhysicalQualificationF32Epilogue::new(
                        1.0,
                        if op == ResolvedGemmOp::Tn { 1.0 } else { 0.0 },
                        bias,
                    ),
                )
                .validate()
                .unwrap();
            }
        }

        PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            (64, 1_536, 384),
            PhysicalQualificationRoute::Tf32Forced(TF32_SPLITK4_SPEC.route),
            PhysicalQualificationF32Epilogue::new(0.5, 0.25, false),
        )
        .validate()
        .unwrap();
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn biased_nn_f32_policy_matches_eager_graph_and_differs_from_unbiased() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let dims = (64, 384, 1_536);
        let salt = 0xb1a5_5eed;

        for policy in [
            F32TriadPolicy::ExactScalarFmaV1,
            F32TriadPolicy::AllowDeterministicTf32V1,
        ] {
            let biased_request = PhysicalQualificationRequest::contiguous_f32(
                ResolvedGemmOp::Nn,
                dims,
                PhysicalQualificationRoute::F32Policy(policy),
                PhysicalQualificationF32Epilogue::new(1.0, 0.0, true),
            );
            let mut biased = qualify_physical_launch(&ctx, biased_request).unwrap();
            biased.seed_f32_operands(&ctx, salt).unwrap();
            biased.measure_eager_window_ms(&ctx, 1).unwrap();
            let eager = biased.f32_output_bits(&ctx).unwrap();
            biased.seed_f32_operands(&ctx, salt).unwrap();
            biased.measure_graph_window_ms(&ctx, 1).unwrap();
            let graph = biased.f32_output_bits(&ctx).unwrap();
            assert_eq!(eager, graph, "{policy:?} biased eager/graph mismatch");
            drop(biased);

            let unbiased_request = PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Nn,
                dims,
                PhysicalQualificationRoute::F32Policy(policy),
            );
            let mut unbiased = qualify_physical_launch(&ctx, unbiased_request).unwrap();
            unbiased.seed_f32_operands(&ctx, salt).unwrap();
            unbiased.measure_eager_window_ms(&ctx, 1).unwrap();
            let without_bias = unbiased.f32_output_bits(&ctx).unwrap();
            assert_ne!(eager, without_bias, "{policy:?} bias had no output effect");
        }
    }

    #[test]
    fn f32_epilogue_rejects_half_routes_before_allocation() {
        let request = PhysicalQualificationRequest::contiguous_f32(
            ResolvedGemmOp::Nn,
            (64, 384, 1536),
            PhysicalQualificationRoute::HalfForced {
                dtype: WeightDtype::Bf16,
                tile: TcTile::Tile64,
            },
            PhysicalQualificationF32Epilogue::new(1.0, 0.0, true),
        );

        assert_eq!(
            request.validate().unwrap_err(),
            "F32 qualification epilogue requires an F32 route"
        );
    }

    // G2 regression-only helpers: all writes/readbacks stay inside facade-owned
    // allocations. Three nonzero reduction rows give an exact, cheap F64
    // reference while exercising the first, second and last reduction slabs.
    fn g2_check_seeded_tn_eager_graph(
        ctx: &GpuCtx,
        qualified: &mut QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        if qualified.request.op != ResolvedGemmOp::Tn {
            return Err("G2 fixture requires TN".into());
        }
        let (m, k, n) = qualified.request.dims;
        if m <= 65 {
            return Err("G2 fixture needs three distinct reduction rows".into());
        }
        let storage = match &qualified.resources {
            QualifiedPhysicalResources::Half(resources) => resources.storage,
            _ => return Err("G2 fixture requires half inputs".into()),
        };
        let mut a = storage.a.host_values();
        let mut b = storage.b.host_values();
        let active_rows = [0, 64, m - 1];
        for (term, row) in active_rows.into_iter().enumerate() {
            for column in 0..k {
                a[storage.a.offset + row * storage.a.stride + column] = match term {
                    0 => ((column % 7) as i32 - 3) as f32 * 0.125,
                    1 => ((column % 5) + 1) as f32 * 0.0625,
                    _ => ((column % 11) as i32 - 5) as f32 * 0.03125,
                };
            }
            for column in 0..n {
                b[storage.b.offset + row * storage.b.stride + column] = match term {
                    0 => ((column % 13) as i32 - 6) as f32 * 0.0625,
                    1 => ((column % 3) + 1) as f32 * 0.125,
                    _ => ((column % 7) as i32 - 3) as f32 * 0.0625,
                };
            }
        }
        let mut initial = storage.output.host_values();
        let mut expected = Vec::with_capacity(k * n);
        let mut changed = false;
        for row in 0..k {
            for column in 0..n {
                let old = 0.25 + ((row * n + column) % 17) as f32 * 0.03125;
                initial[storage.output.offset + row * storage.output.stride + column] = old;
                let mut value = f64::from(old);
                for reduction in active_rows {
                    value += f64::from(a[storage.a.offset + reduction * storage.a.stride + row])
                        * f64::from(b[storage.b.offset + reduction * storage.b.stride + column]);
                }
                // All operands/products/sums are exactly representable in F32,
                // BF16/F16 inputs included. This is not a general MMA oracle.
                let word = (value as f32).to_bits();
                changed |= word != old.to_bits();
                expected.push(word);
            }
        }
        if !changed {
            return Err("G2 fixture would not detect a no-op GEMM".into());
        }
        let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
            unreachable!();
        };
        resources.a.upload_f32(&ctx.stream, &a)?;
        resources.b.upload_f32(&ctx.stream, &b)?;

        // ABBA execution order; every TN call starts from identical nonzero C.
        // Checking guards BEFORE resetting prevents a reset from hiding damage.
        for graph_first in [false, true, true, false] {
            let guards = qualified.validate_red_zones(ctx)?;
            if guards.allocation_count() != 3
                || guards.element_count() < 3 * QUALIFICATION_GUARD_ELEMENTS
            {
                return Err("G2 fixture lost guarded allocations".into());
            }
            let QualifiedPhysicalResources::Half(resources) = &mut qualified.resources else {
                unreachable!();
            };
            let QualifiedHalfOutput::F32(output) = &mut resources.output else {
                return Err("G2 TN fixture requires F32 output".into());
            };
            // Guard values are already validated and copied unchanged.
            output.upload(&ctx.stream, &initial)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("G2 C reset synchronization: {error:?}"))?;
            if graph_first {
                qualified.measure_graph_window_ms(ctx, 1)?;
            } else {
                qualified.measure_eager_window_ms(ctx, 1)?;
            }
            let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
                unreachable!();
            };
            let values = resources.output.to_cpu(ctx)?;
            ctx.stream
                .synchronize()
                .map_err(|error| format!("G2 output readback synchronization: {error:?}"))?;
            let mut actual = Vec::with_capacity(k * n);
            for row in 0..k {
                let start = storage.output.offset + row * storage.output.stride;
                actual.extend(values[start..start + n].iter().copied().map(f32::to_bits));
            }
            if actual != expected {
                return Err(format!(
                    "G2 TN beta=1 exact three-term reference mismatch, graph={graph_first}"
                ));
            }
            let guards = qualified.validate_red_zones(ctx)?;
            if guards.allocation_count() != 3 {
                return Err("G2 post-launch guard census changed".into());
            }
            let mut a_after = vec![0.0; a.len()];
            let mut b_after = vec![0.0; b.len()];
            resources.a.download_f32(&ctx.stream, &mut a_after)?;
            resources.b.download_f32(&ctx.stream, &mut b_after)?;
            if a_after != a || b_after != b {
                return Err("G2 input values or guards changed".into());
            }
        }
        Ok(())
    }

    fn g2_check_half_tn_physical_evidence(
        qualified: &QualifiedPhysicalLaunch<'_>,
        symbol: &str,
        tile: (u32, u32),
        bk_stages: (u32, u8),
        grid: u32,
        threads: u32,
        stream_k: bool,
    ) -> Result<(), String> {
        use crate::mamba_ssm::gpu::kernel_identity::ResolvedOutputOwnership;
        let evidence = qualified.evidence();
        let [eager] = evidence.nodes() else {
            return Err("G2 expected exactly one eager physical GEMM".into());
        };
        let [captured] = qualified.graph.nodes() else {
            return Err("G2 expected exactly one captured physical GEMM".into());
        };
        if !evidence.eager_graph_equal()
            || evidence.launch_count() != 1
            || *eager != QualifiedPhysicalLaunchNode::from(captured)
        {
            return Err("G2 exact eager/captured node evidence differs".into());
        }
        let route = captured
            .gemm_route()
            .ok_or_else(|| "G2 captured node lacks GEMM contract".to_string())?;
        let contract = if stream_k {
            ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1
        } else {
            ResolvedNumericContract::MmaSyncF32V1
        };
        let ownership = if stream_k {
            ResolvedOutputOwnership::OwnerCtaPerOutputTileStreamKFixedOrderV1
        } else {
            ResolvedOutputOwnership::OneCtaPerOutputTileV1
        };
        let (m, k, n) = qualified.request.dims;
        if eager.symbol != symbol
            || eager.kind != PhysicalLaunchKind::Gemm
            || eager.module_kind != ModuleKind::TriadSm80
            || eager.logical_op != ResolvedGemmOp::Tn
            || eager.logical_dtype != qualified.request.route.logical_dtype()
            || eager.execution_dtype != qualified.request.route.logical_dtype()
            || eager.shape != (m, k, n)
            || eager.strides != (k, n, n)
            || eager.tile != Some(tile)
            || (route.bk, route.stages) != bk_stages
            || eager.numeric_contract != Some(contract)
            || eager.ownership != Some(ownership)
            || eager.launch.grid_dim != (grid, 1, 1)
            || eager.launch.block_dim != (threads, 1, 1)
            || eager.launch.shared_mem_bytes != 0
            || eager.launch.arguments_digest == [0; 32]
        {
            return Err(format!(
                "G2 expected {symbol}, tile={tile:?}, BK/stages={bk_stages:?}, grid={grid}, threads={threads}, stream_k={stream_k}; observed {eager:?}, BK/stages={:?}",
                (route.bk, route.stages)
            ));
        }
        Ok(())
    }

    fn g2_exact_ada_142_context() -> GpuCtx {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        assert_eq!(device.compute_capability, (8, 9), "G2 exact Ada fixture");
        assert_eq!(
            device.multiprocessor_count(),
            142,
            "G2 measured-board fixture"
        );
        GpuCtx::new(&device).expect("G2 actual NVRTC context")
    }

    #[test]
    #[ignore = "requires CC8.9/142SM; explicit half AUTO permission and actual eager/graph route"]
    fn g2_half_auto_streamk_permission_is_not_lost() {
        let ctx = g2_exact_ada_142_context();
        ctx.set_half_triad_policy(HalfTriadPolicy::TiledParityV1);
        // The explicit request must install permission independently of the
        // ambient context, then restore the original tiled policy on drop.
        let requests = [WeightDtype::Bf16, WeightDtype::F16].map(|dtype| {
            PhysicalQualificationRequest::contiguous(
                ResolvedGemmOp::Tn,
                (10400, 384, 384),
                PhysicalQualificationRoute::HalfPolicy {
                    dtype,
                    tensor_cores: true,
                    half_policy: HalfTriadPolicy::AllowStreamKFixedOrderV1,
                },
            )
        });
        presize_physical_qualification_suite(&ctx, &requests).expect("G2 suite scratch");
        let mut failures = Vec::new();
        for (request, symbol) in requests.into_iter().zip([
            "gemm_bi_tn_tc64_streamk_bf16",
            "gemm_bi_tn_tc64_streamk_f16",
        ]) {
            let planned = request.timed_identity().expect("G2 AUTO identity");
            assert_eq!(planned.planned_symbol, None, "AUTO must not force a symbol");
            assert_eq!(planned.planned_tile, None, "AUTO must not force a tile");
            let result = (|| -> Result<(), String> {
                let mut qualified = qualify_physical_launch(&ctx, request)?;
                qualified.validate_timed_request(&ctx, request)?;
                g2_check_seeded_tn_eager_graph(&ctx, &mut qualified)?;
                // 36 output tiles * 163 reduction slabs -> 142 persistent CTAs.
                g2_check_half_tn_physical_evidence(
                    &qualified,
                    symbol,
                    (64, 64),
                    (64, 2),
                    142,
                    128,
                    true,
                )
            })();
            if let Err(error) = result {
                failures.push(format!("{symbol}: {error}"));
            }
            assert_eq!(
                ctx.half_triad_policy(),
                HalfTriadPolicy::TiledParityV1,
                "G2 lease must restore the original tiled policy even on error"
            );
        }
        assert!(
            failures.is_empty(),
            "G2 AUTO permission/physical-route failures for both dtypes:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    #[ignore = "requires CC8.9/142SM; forced TN Rect128x64 actual eager/graph route"]
    fn g2_half_forced_rectangular_prepares_the_exact_graph() {
        let ctx = g2_exact_ada_142_context();
        ctx.set_half_triad_policy(HalfTriadPolicy::TiledParityV1);
        let requests = [WeightDtype::Bf16, WeightDtype::F16].map(|dtype| {
            PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Tn,
                (129, 131, 100),
                PhysicalQualificationRoute::HalfForced {
                    dtype,
                    tile: TcTile::Rect128x64,
                },
                PhysicalQualificationOffset::Output,
            )
        });
        presize_physical_qualification_suite(&ctx, &requests).expect("G2 suite scratch");
        let mut failures = Vec::new();
        for (request, symbol) in requests
            .into_iter()
            .zip(["gemm_bi_tn_tc128x64_bf16", "gemm_bi_tn_tc128x64_f16"])
        {
            let result = (|| -> Result<(), String> {
                let mut qualified = qualify_physical_launch(&ctx, request)?;
                qualified.validate_timed_request(&ctx, request)?;
                g2_check_seeded_tn_eager_graph(&ctx, &mut qualified)?;
                // 131x100 output: two row tiles * two column tiles.
                g2_check_half_tn_physical_evidence(
                    &qualified,
                    symbol,
                    (128, 64),
                    (32, 3),
                    4,
                    256,
                    false,
                )
            })();
            if let Err(error) = result {
                failures.push(format!("{symbol}: {error}"));
            }
            assert_eq!(
                ctx.half_triad_policy(),
                HalfTriadPolicy::TiledParityV1,
                "G2 rectangular error must restore the original tiled policy"
            );
        }
        assert!(
            failures.is_empty(),
            "G2 rectangular graph qualification failures for both dtypes:\n{}",
            failures.join("\n")
        );
    }

    // R4 tests below extend already-working production behavior. They are not
    // general half-GEMM accuracy qualification: the inherited three-term corpus
    // has an independent exact reference for EVERY output, but only three
    // nonzero reduction rows. No cross-schedule floating-point parity is claimed.
    fn g2_other_half_policy(policy: HalfTriadPolicy) -> HalfTriadPolicy {
        match policy {
            HalfTriadPolicy::TiledParityV1 => HalfTriadPolicy::AllowStreamKFixedOrderV1,
            HalfTriadPolicy::AllowStreamKFixedOrderV1 => HalfTriadPolicy::TiledParityV1,
        }
    }

    fn g2_tn_request(
        dims: (usize, usize, usize),
        route: PhysicalQualificationRoute,
        offset: Option<PhysicalQualificationOffset>,
    ) -> PhysicalQualificationRequest {
        match offset {
            Some(offset) => PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Tn,
                dims,
                route,
                offset,
            ),
            None => PhysicalQualificationRequest::contiguous(ResolvedGemmOp::Tn, dims, route),
        }
    }

    fn g2_emit_tn_holder_row(
        label: &str,
        qualified: &QualifiedPhysicalLaunch<'_>,
        guards: QualifiedGuardValidation,
    ) {
        let request = qualified.request;
        let node = qualified.evidence().nodes()[0];
        let contract = qualified.graph.nodes()[0].gemm_route().unwrap();
        let (mode, dtype, policy) = match request.route {
            PhysicalQualificationRoute::HalfPolicy {
                dtype, half_policy, ..
            } => ("auto", dtype, half_policy),
            PhysicalQualificationRoute::HalfForced { dtype, .. } => {
                ("forced", dtype, request.route.policy().half_triad_policy)
            }
            _ => unreachable!("G2 TN half row"),
        };
        let (m, k, n) = request.dims;
        let (lda, ldb, ldc) = node.strides;
        let (gx, gy, gz) = node.launch.grid_dim;
        let (bx, by, bz) = node.launch.block_dim;
        let (tm, tn) = node.tile.unwrap();
        // All string fields are closed enum names, CUDA symbols or local ASCII
        // fixture labels. Avoid an optional serde dependency in CUDA-only tests.
        println!(
            concat!(
                "{{\"schema\":\"G2TnHalfAutoV1\",\"event\":\"holder_gates\",",
                "\"case\":\"{}\",\"requested_mode\":\"{}\",\"requested_permission\":\"{:?}\",",
                "\"requested_route\":\"{:?}\",\"dtype\":\"{:?}\",\"op\":\"tn\",",
                "\"dims\":[{},{},{}],\"layout\":\"{:?}\",\"strides\":[{},{},{}],",
                "\"actual_symbol\":\"{}\",\"module\":\"{:?}\",\"tile\":[{},{}],",
                "\"bk\":{},\"stages\":{},\"grid\":[{},{},{}],\"block\":[{},{},{}],",
                "\"dynamic_shared_bytes\":{},\"numeric_contract\":\"{:?}\",\"ownership\":\"{:?}\",",
                "\"alpha\":1,\"beta\":1,\"bias\":false,\"eager_graph_equal\":true,",
                "\"execution_order\":\"eager_graph_graph_eager\",\"reset_nonzero_c_each_launch\":true,",
                "\"numeric_corpus\":\"three_term_dyadic_full_output_only\",",
                "\"guard_allocations\":{},\"guard_elements\":{},\"inputs_unchanged\":true,",
                "\"holder_gates\":\"pass\"}}"
            ),
            label,
            mode,
            policy,
            request.route,
            dtype,
            m,
            k,
            n,
            request.layout,
            lda,
            ldb,
            ldc,
            node.symbol,
            node.module_kind,
            tm,
            tn,
            contract.bk,
            contract.stages,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            node.launch.shared_mem_bytes,
            contract.numeric_contract,
            contract.ownership,
            guards.allocation_count(),
            guards.element_count()
        );
    }

    fn g2_run_tn_holder(
        ctx: &GpuCtx,
        label: &str,
        request: PhysicalQualificationRequest,
        stream_k: bool,
        expected_grid: u32,
    ) -> Result<Vec<u32>, String> {
        let original = ctx.gemm_route();
        let result = (|| {
            if matches!(request.route, PhysicalQualificationRoute::HalfPolicy { .. }) {
                let identity = request.timed_identity()?;
                if identity.planned_symbol.is_some() || identity.planned_tile.is_some() {
                    return Err("G2 AUTO request became a forced request".into());
                }
            }
            let mut qualified = qualify_physical_launch(ctx, request)?;
            qualified.validate_timed_request(ctx, request)?;
            let symbol = match (request.route.logical_dtype(), stream_k) {
                (PolicyDtype::Bf16, false) => "gemm_bi_tn_tc64_bf16",
                (PolicyDtype::F16, false) => "gemm_bi_tn_tc64_f16",
                (PolicyDtype::Bf16, true) => "gemm_bi_tn_tc64_streamk_bf16",
                (PolicyDtype::F16, true) => "gemm_bi_tn_tc64_streamk_f16",
                _ => return Err("G2 half fixture has an unexpected dtype".into()),
            };
            g2_check_half_tn_physical_evidence(
                &qualified,
                symbol,
                (64, 64),
                (64, 2),
                expected_grid,
                128,
                stream_k,
            )?;
            g2_check_seeded_tn_eager_graph(ctx, &mut qualified)?;
            let words = qualified_half_active_output(ctx, &qualified)?;
            let guards = qualified.validate_red_zones(ctx)?;
            g2_emit_tn_holder_row(label, &qualified, guards);
            Ok(words)
        })();
        // Also runs on a returned error; do not let an error mask leaked policy.
        assert_eq!(
            ctx.gemm_route(),
            original,
            "{label}: holder leaked lease state"
        );
        result
    }

    #[test]
    #[ignore = "requires CC8.9/142SM; actual half TN AUTO permission matrix and same-schedule bits"]
    fn g2_half_auto_tn_permission_matrix_sm89() {
        let ctx = g2_exact_ada_142_context();
        // These outcomes and tiled grid counts are hand-derived fixtures, not
        // calls to the selector. All have a Tile64 tiled fallback on this board.
        // Catch lost permission, over-broad Stream-K admission, boundary drift,
        // wrong dtype/symbol/ABI binding, and misaligned/tail operand addressing.
        let fixtures = [
            ("winner_batch_input", (10400, 384, 384), None, 36, true),
            ("winner_batch_out", (10400, 768, 384), None, 72, true),
            ("winner_prism_out", (4621, 768, 384), None, 72, true),
            ("winner_prism_input", (4621, 1024, 384), None, 96, true),
            ("winner_rect_tall", (4096, 512, 768), None, 96, true),
            ("winner_batch_in", (10400, 384, 1536), None, 144, true),
            ("fallback_filled_tail", (4621, 384, 1928), None, 186, false),
            ("fallback_shallow", (512, 384, 384), None, 36, false),
            ("fallback_little_work", (256, 512, 384), None, 48, false),
            // 142*32=4544 units. 36*125=4500 and 36*126=4536 BOTH decline.
            ("depth_m8000", (8000, 384, 384), None, 36, false),
            ("depth_m8001", (8001, 384, 384), None, 36, false),
            ("depth_m8064", (8064, 384, 384), None, 36, false),
            ("depth_m8065", (8065, 384, 384), None, 36, true),
            // 150*8 <=142*9; 160*8 >142*9. Both have ample reduction work.
            ("wave_150", (10400, 640, 960), None, 150, true),
            ("wave_160", (10400, 640, 1024), None, 160, false),
            // Tail in all three axes; one named operand moves by one element.
            (
                "tail_a1",
                (4621, 769, 385),
                Some(PhysicalQualificationOffset::A),
                91,
                true,
            ),
            (
                "tail_b1",
                (4621, 769, 385),
                Some(PhysicalQualificationOffset::B),
                91,
                true,
            ),
            (
                "tail_c1",
                (4621, 769, 385),
                Some(PhysicalQualificationOffset::Output),
                91,
                true,
            ),
        ];
        let policies = [
            HalfTriadPolicy::TiledParityV1,
            HalfTriadPolicy::AllowStreamKFixedOrderV1,
        ];
        let mut requests = Vec::new();
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for (_, dims, offset, _, admits) in fixtures {
                for half_policy in policies {
                    requests.push(g2_tn_request(
                        dims,
                        PhysicalQualificationRoute::HalfPolicy {
                            dtype,
                            tensor_cores: true,
                            half_policy,
                        },
                        offset,
                    ));
                }
                if admits {
                    requests.push(g2_tn_request(
                        dims,
                        PhysicalQualificationRoute::HalfForced {
                            dtype,
                            tile: TcTile::Tile64StreamK,
                        },
                        offset,
                    ));
                }
            }
        }
        presize_physical_qualification_suite(&ctx, &requests).expect("G2 TN matrix presize");
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for (label, dims, offset, tiled_grid, admits) in fixtures {
                let mut tiled_words = None;
                for half_policy in policies {
                    let stream_k =
                        admits && half_policy == HalfTriadPolicy::AllowStreamKFixedOrderV1;
                    let request = g2_tn_request(
                        dims,
                        PhysicalQualificationRoute::HalfPolicy {
                            dtype,
                            tensor_cores: true,
                            half_policy,
                        },
                        offset,
                    );
                    let words = g2_run_tn_holder(
                        &ctx,
                        label,
                        request,
                        stream_k,
                        if stream_k { 142 } else { tiled_grid },
                    )
                    .unwrap_or_else(|error| panic!("{label}/{dtype:?}/{half_policy:?}: {error}"));
                    if half_policy == HalfTriadPolicy::TiledParityV1 {
                        tiled_words = Some(words);
                    } else if stream_k {
                        let forced = g2_tn_request(
                            dims,
                            PhysicalQualificationRoute::HalfForced {
                                dtype,
                                tile: TcTile::Tile64StreamK,
                            },
                            offset,
                        );
                        let forced_words = g2_run_tn_holder(&ctx, label, forced, true, 142)
                            .unwrap_or_else(|error| panic!("{label}/{dtype:?}/forced: {error}"));
                        assert!(
                            words == forced_words,
                            "{label}/{dtype:?}: AUTO/forced Stream-K bits"
                        );
                    } else {
                        assert!(
                            Some(words) == tiled_words,
                            "{label}/{dtype:?}: permission changed same-tiled-schedule bits"
                        );
                    }
                }
                println!(
                    concat!(
                        "{{\"schema\":\"G2TnHalfAutoV1\",\"event\":\"same_schedule_comparison\",",
                        "\"case\":\"{}\",\"dtype\":\"{:?}\",\"auto_vs_forced_streamk\":{},",
                        "\"tiled_fallback_both_permissions\":{},\"bits_equal\":true}}"
                    ),
                    label, dtype, admits, !admits
                );
            }
        }
    }

    #[test]
    #[ignore = "requires CC8.9/142SM; actual AUTO Stream-K reuse across different legal scratch extents"]
    fn g2_half_auto_tn_scratch_reuse_sm89() {
        let ctx = g2_exact_ada_142_context();
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            let first = g2_tn_request(
                (10400, 384, 384),
                PhysicalQualificationRoute::HalfPolicy {
                    dtype,
                    tensor_cores: true,
                    half_policy: HalfTriadPolicy::AllowStreamKFixedOrderV1,
                },
                None,
            );
            let intervening = g2_tn_request(
                (129, 131, 100),
                PhysicalQualificationRoute::HalfForced {
                    dtype,
                    tile: TcTile::Tile64StreamK,
                },
                Some(PhysicalQualificationOffset::B),
            );
            presize_physical_qualification_suite(&ctx, &[first, intervening])
                .expect("G2 scratch reuse presize");
            let before = g2_run_tn_holder(&ctx, "reuse_first", first, true, 142).unwrap();
            // This accessor only retrieves the existing OnceLock buffers. Never
            // fill/reset partials or flags here: that would hide reuse defects.
            let workspace =
                super::super::launch::sm80_streamk_workspace(&ctx.stream, &ctx.kernels, 142)
                    .expect("existing stream-K workspace");
            assert_ne!(workspace.0, 0);
            assert_ne!(workspace.1, 0);
            // Six output tiles * three slabs = 18 CTAs, then return to 142.
            g2_run_tn_holder(&ctx, "reuse_intervening", intervening, true, 18).unwrap();
            assert_eq!(
                workspace,
                super::super::launch::sm80_streamk_workspace(&ctx.stream, &ctx.kernels, 18)
                    .unwrap()
            );
            let after = g2_run_tn_holder(&ctx, "reuse_return", first, true, 142).unwrap();
            assert!(
                before == after,
                "{dtype:?}: scratch reuse changed same-AUTO-schedule bits"
            );
            assert_eq!(
                workspace,
                super::super::launch::sm80_streamk_workspace(&ctx.stream, &ctx.kernels, 142)
                    .unwrap()
            );
            // Address stability plus numerical reuse is not a scratch-canary proof.
            println!(
                concat!(
                    "{{\"schema\":\"G2TnHalfAutoV1\",\"event\":\"scratch_reuse\",",
                    "\"dtype\":\"{:?}\",\"grid_sequence\":[142,18,142],\"workspace_addresses_stable\":true,",
                    "\"test_zeroed_scratch\":false,\"scratch_canary_coverage\":false,\"bits_equal\":true}}"
                ),
                dtype
            );
        }
    }

    fn g2_half_allocation_words(
        ctx: &GpuCtx,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<[Vec<u32>; 3], String> {
        // Intentionally bypass lease validation for READBACK ONLY: after a
        // rejected drifted launch, inspect all original data and red zones.
        let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
            return Err("G2 lease fixture requires half resources".into());
        };
        let c = resources.output.to_cpu(ctx)?;
        let mut a = vec![0.0; resources.a.len_elems()];
        let mut b = vec![0.0; resources.b.len_elems()];
        resources.a.download_f32(&ctx.stream, &mut a)?;
        resources.b.download_f32(&ctx.stream, &mut b)?;
        ctx.stream
            .synchronize()
            .map_err(|error| format!("G2 lease readback: {error:?}"))?;
        Ok([c, a, b].map(|values| values.into_iter().map(f32::to_bits).collect()))
    }

    #[test]
    #[ignore = "requires CC8.9/142SM; physical half lease rejects drift and restores all policy fields"]
    fn g2_half_tn_policy_lease_drift_restore_sm89() {
        let ctx = g2_exact_ada_142_context();
        for dtype in [WeightDtype::Bf16, WeightDtype::F16] {
            for original_half in [
                HalfTriadPolicy::TiledParityV1,
                HalfTriadPolicy::AllowStreamKFixedOrderV1,
            ] {
                for error_exit in [false, true] {
                    // Set every normalized field opposite to the requested
                    // lease. Even when the physical tile is unchanged, drift
                    // in explicit numeric permission must reject before work.
                    ctx.set_batch_invariant(false);
                    ctx.set_bi_gemm_family(BiGemmFamily::Fixed);
                    ctx.set_bi_tensor_cores(false);
                    ctx.set_fast_gemm(true);
                    ctx.set_f32_triad_policy(F32TriadPolicy::AllowDeterministicTf32V1);
                    ctx.set_half_triad_policy(original_half);
                    let original = ctx.gemm_route();
                    let frozen_tf32 = ctx.tf32();
                    let requested_half = g2_other_half_policy(original_half);
                    let request = g2_tn_request(
                        (256, 512, 384),
                        PhysicalQualificationRoute::HalfPolicy {
                            dtype,
                            tensor_cores: true,
                            half_policy: requested_half,
                        },
                        Some(PhysicalQualificationOffset::Output),
                    );
                    presize_physical_qualification_suite(&ctx, &[request]).unwrap();
                    let result = (|| -> Result<(), String> {
                        let mut qualified = qualify_physical_launch(&ctx, request)?;
                        assert!(ctx.batch_invariant());
                        assert_eq!(ctx.bi_gemm_family(), BiGemmFamily::Triad);
                        assert!(ctx.bi_tensor_cores());
                        assert!(!ctx.fast_gemm());
                        assert_eq!(ctx.f32_triad_policy(), F32TriadPolicy::ExactScalarFmaV1);
                        assert_eq!(ctx.half_triad_policy(), requested_half);
                        assert_eq!(ctx.tf32(), frozen_tf32);
                        g2_check_seeded_tn_eager_graph(&ctx, &mut qualified)?;
                        let symbol = if dtype == WeightDtype::Bf16 {
                            "gemm_bi_tn_tc64_bf16"
                        } else {
                            "gemm_bi_tn_tc64_f16"
                        };
                        g2_check_half_tn_physical_evidence(
                            &qualified,
                            symbol,
                            (64, 64),
                            (64, 2),
                            48,
                            128,
                            false,
                        )?;
                        let guarded = qualified.validate_red_zones(&ctx)?;
                        g2_emit_tn_holder_row("lease_before_drift", &qualified, guarded);
                        let before = g2_half_allocation_words(&ctx, &qualified)?;
                        let leased = ctx.gemm_route();
                        let conflicting = PhysicalQualificationRequest {
                            route: PhysicalQualificationRoute::HalfPolicy {
                                dtype,
                                tensor_cores: true,
                                half_policy: original_half,
                            },
                            ..request
                        };
                        let second_error = match qualify_physical_launch(&ctx, conflicting) {
                            Ok(second) => {
                                drop(second);
                                return Err("G2 second holder was accepted".into());
                            }
                            Err(error) => error,
                        };
                        assert!(
                            second_error
                                .contains("already has a live physical qualification holder")
                        );
                        assert_eq!(
                            ctx.gemm_route(),
                            leased,
                            "second holder changed the live lease"
                        );
                        assert!(
                            g2_half_allocation_words(&ctx, &qualified)? == before,
                            "second holder rejection enqueued visible work"
                        );

                        ctx.set_half_triad_policy(original_half);
                        let eager_error = qualified.measure_eager_window_ms(&ctx, 1).unwrap_err();
                        assert!(
                            eager_error.contains("physical qualification policy lease")
                                && eager_error.contains("GEMM route changed"),
                            "{eager_error}"
                        );
                        assert!(
                            g2_half_allocation_words(&ctx, &qualified)? == before,
                            "drifted eager changed nonzero TN output, inputs or guards"
                        );
                        let graph_error = qualified.measure_graph_window_ms(&ctx, 1).unwrap_err();
                        assert!(
                            graph_error.contains("physical qualification policy lease")
                                && graph_error.contains("GEMM route changed"),
                            "{graph_error}"
                        );
                        assert!(
                            g2_half_allocation_words(&ctx, &qualified)? == before,
                            "drifted graph changed nonzero TN output, inputs or guards"
                        );
                        if error_exit {
                            // Propagate the actual API error through holder drop.
                            return Err(graph_error);
                        }
                        ctx.set_half_triad_policy(requested_half);
                        qualified.validate_timed_request(&ctx, request)?;
                        g2_check_seeded_tn_eager_graph(&ctx, &mut qualified)?;
                        Ok(())
                    })();
                    if error_exit {
                        let error = result.expect_err("G2 actual drift error must propagate");
                        assert!(error.contains("physical qualification policy lease"));
                    } else {
                        result.unwrap();
                    }
                    assert_eq!(
                        ctx.gemm_route(),
                        original,
                        "drop/error leaked any lease field"
                    );
                    assert_eq!(ctx.tf32(), frozen_tf32);
                    // A new valid holder after drop proves the active token was
                    // released on both the normal and propagated-error paths.
                    let next = qualify_physical_launch(&ctx, request).unwrap();
                    next.validate_timed_request(&ctx, request).unwrap();
                    drop(next);
                    assert_eq!(
                        ctx.gemm_route(),
                        original,
                        "replacement holder leaked state"
                    );
                    println!(
                        concat!(
                            "{{\"schema\":\"G2TnHalfAutoV1\",\"event\":\"lease_guard\",",
                            "\"dtype\":\"{:?}\",\"original_permission\":\"{:?}\",\"requested_permission\":\"{:?}\",",
                            "\"propagated_error_exit\":{},\"second_holder_rejected\":true,",
                            "\"drift_eager_rejected\":true,\"drift_graph_rejected\":true,",
                            "\"nonzero_c_inputs_guards_unchanged\":true,\"all_lease_fields_restored\":true,",
                            "\"replacement_holder_accepted\":true}}"
                        ),
                        dtype, original_half, requested_half, error_exit
                    );
                }
            }
        }
    }

    fn seeded_half_values(storage: QualificationStorage, salt: usize) -> Vec<f32> {
        let mut values = storage.host_values();
        for row in 0..storage.rows {
            for column in 0..storage.width {
                let logical = row * storage.width + column;
                let numerator = ((logical * 13 + salt * 7) % 29) as f32 - 14.0;
                values[storage.offset + row * storage.stride + column] = numerator / 16.0;
            }
        }
        values
    }

    fn seed_qualified_half_inputs(
        ctx: &GpuCtx,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<(), String> {
        let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
            return Err("seeded offset fixture requires half resources".into());
        };
        resources
            .a
            .upload_f32(&ctx.stream, &seeded_half_values(resources.storage.a, 3))?;
        resources
            .b
            .upload_f32(&ctx.stream, &seeded_half_values(resources.storage.b, 11))
    }

    fn qualified_half_active_output(
        ctx: &GpuCtx,
        qualified: &QualifiedPhysicalLaunch<'_>,
    ) -> Result<Vec<u32>, String> {
        let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
            return Err("offset output fixture requires half resources".into());
        };
        let values = resources.output.to_cpu(ctx)?;
        let storage = resources.storage.output;
        let mut active = Vec::with_capacity(storage.rows * storage.width);
        for row in 0..storage.rows {
            let start = storage.offset + row * storage.stride;
            active.extend(
                values[start..start + storage.width]
                    .iter()
                    .map(|value| value.to_bits()),
            );
        }
        Ok(active)
    }

    fn run_seeded_offset_order(
        ctx: &GpuCtx,
        request: PhysicalQualificationRequest,
        graph_first: bool,
    ) -> Result<(Vec<u32>, Vec<u32>), String> {
        let mut qualified = qualify_physical_launch(ctx, request)?;
        seed_qualified_half_inputs(ctx, &qualified)?;
        let (eager, graph) = if graph_first {
            qualified.measure_graph_window_ms(ctx, 1)?;
            let graph = qualified_half_active_output(ctx, &qualified)?;
            qualified.measure_eager_window_ms(ctx, 1)?;
            let eager = qualified_half_active_output(ctx, &qualified)?;
            (eager, graph)
        } else {
            qualified.measure_eager_window_ms(ctx, 1)?;
            let eager = qualified_half_active_output(ctx, &qualified)?;
            qualified.measure_graph_window_ms(ctx, 1)?;
            let graph = qualified_half_active_output(ctx, &qualified)?;
            (eager, graph)
        };
        let guards = qualified.validate_red_zones(ctx)?;
        if guards.allocation_count() != 3 {
            return Err("offset fixture did not validate three guarded allocations".into());
        }
        Ok((eager, graph))
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn a_and_b_offsets_match_eager_and_graph_with_nonzero_operands_in_both_orders() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let cases = [
            (
                "policy A+1",
                PhysicalQualificationRequest::one_element_offset(
                    ResolvedGemmOp::Nn,
                    (129, 65, 129),
                    PhysicalQualificationRoute::HalfPolicy {
                        dtype: WeightDtype::Bf16,
                        tensor_cores: true,
                        half_policy: HalfTriadPolicy::TiledParityV1,
                    },
                    PhysicalQualificationOffset::A,
                ),
            ),
            (
                "forced B+1",
                PhysicalQualificationRequest::one_element_offset(
                    ResolvedGemmOp::Nn,
                    (129, 65, 129),
                    PhysicalQualificationRoute::HalfForced {
                        dtype: WeightDtype::Bf16,
                        tile: TcTile::Tile64,
                    },
                    PhysicalQualificationOffset::B,
                ),
            ),
        ];
        let requests = cases.map(|(_, request)| request);
        presize_physical_qualification_suite(&ctx, &requests).unwrap();
        for (label, request) in cases {
            let (ab_eager, ab_graph) = run_seeded_offset_order(&ctx, request, false).unwrap();
            let (ba_eager, ba_graph) = run_seeded_offset_order(&ctx, request, true).unwrap();
            assert!(ab_eager.iter().any(|word| *word != 0));
            assert!(ab_eager == ab_graph, "{label}: AB eager differs from graph");
            assert!(ab_eager == ba_eager, "{label}: eager differs across orders");
            assert!(ab_eager == ba_graph, "{label}: graph differs across orders");
        }
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn edge_layout_facade_owns_subviews_and_validates_red_zones() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let route = PhysicalQualificationRoute::HalfForced {
            dtype: WeightDtype::Bf16,
            tile: TcTile::Tile64,
        };
        let requests = [
            PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Nn,
                (129, 65, 129),
                route,
                PhysicalQualificationOffset::A,
            ),
            PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Tn,
                (65, 128, 128),
                route,
                PhysicalQualificationOffset::B,
            ),
            PhysicalQualificationRequest::one_element_offset(
                ResolvedGemmOp::Nt,
                (129, 129, 72),
                route,
                PhysicalQualificationOffset::Output,
            ),
            PhysicalQualificationRequest::direct_padded_nn(
                (129, 65, 129),
                WeightDtype::Bf16,
                TcTile::Tile64,
                PhysicalPaddedNnLayout::Lda72Ldb136Ldc131,
            ),
        ];
        presize_physical_qualification_suite(&ctx, &requests).unwrap();
        for request in requests {
            let shape = request.physical_shape();
            let expected_strides = (shape.lda, shape.ldb, shape.ldc);
            let mut qualified = qualify_physical_launch(&ctx, request).unwrap();
            assert!(
                qualified
                    .evidence()
                    .nodes()
                    .iter()
                    .filter(|node| node.kind == PhysicalLaunchKind::Gemm)
                    .all(|node| node.strides == expected_strides)
            );
            qualified.measure_eager_window_ms(&ctx, 1).unwrap();
            qualified.measure_graph_window_ms(&ctx, 1).unwrap();
            let guards = qualified.validate_red_zones(&ctx).unwrap();
            assert_eq!(guards.allocation_count(), 3);
            assert!(guards.element_count() >= 3 * QUALIFICATION_GUARD_ELEMENTS);
        }
    }

    #[test]
    #[ignore = "requires an SM80+ CUDA device and NVRTC"]
    fn output_offset_eager_and_graph_preserve_the_leading_red_zone() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).expect("CUDA device");
        let ctx = GpuCtx::new(&device).expect("CUDA context");
        let request = PhysicalQualificationRequest::one_element_offset(
            ResolvedGemmOp::Nt,
            (129, 129, 72),
            PhysicalQualificationRoute::HalfForced {
                dtype: WeightDtype::Bf16,
                tile: TcTile::Tile64,
            },
            PhysicalQualificationOffset::Output,
        );
        presize_physical_qualification_suite(&ctx, &[request]).unwrap();
        let mut qualified = qualify_physical_launch(&ctx, request).unwrap();
        let QualifiedPhysicalResources::Half(resources) = &qualified.resources else {
            panic!("half qualification resources");
        };
        assert_eq!(
            (
                resources.storage.output.rows,
                resources.storage.output.width,
                resources.storage.output.stride,
                resources.storage.output.offset,
                resources.storage.output.len,
            ),
            (129, 129, 129, 1, 16_674)
        );
        assert_eq!(
            resources.binding.0,
            resources.output.ptr() + WeightDtype::Bf16.size_bytes() as u64
        );
        assert_eq!(qualified.evidence().nodes().len(), 1);
        assert_eq!(
            qualified.evidence().nodes()[0].symbol,
            "gemm_bi_nt_tc64_bf16"
        );
        assert_eq!(qualified.evidence().nodes()[0].shape, (129, 129, 72));
        assert_eq!(qualified.evidence().nodes()[0].strides, (72, 72, 129));

        qualified.measure_eager_window_ms(&ctx, 1).unwrap();
        qualified
            .validate_red_zones(&ctx)
            .expect("eager preserves output offset red zone");
        qualified.measure_graph_window_ms(&ctx, 1).unwrap();
        qualified
            .validate_red_zones(&ctx)
            .expect("graph preserves output offset red zone");
    }
}
