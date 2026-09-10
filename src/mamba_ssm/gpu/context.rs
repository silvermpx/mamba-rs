//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::dtype::WeightDtype;
use super::gemm_bi_triad::{
    F32PreparedLaunchCache, Sm90aPreparedLaunchCache, Sm100PreparedLaunchCache,
    Sm120PreparedLaunchCache,
};
pub use super::gemm_mode::GemmMode;
use super::gemm_mode::{MathModeBackend, MathTransitionError, change_math_mode};
use super::kernel_identity::{
    ArtifactIdentity, BackendSet, CapturedGemmGraphPlan, CompilerIdentity, ModuleKind,
    PhysicalGemmBackend, PolicyDtype, PreparedGemmCaptureManifest, RecordedGemmTrace,
    ResolvedGemmLaunchSet, ResolvedGemmRoute, ResolvedNumericContract,
    build_resolved_gemm_launch_set,
};
use super::kernels::MambaKernels;
use crate::config::MambaConfig;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GPU_CTX_TOKEN: AtomicU64 = AtomicU64::new(1);

fn m1_mixed_graph_max_dim(dims: &super::forward::GpuMambaDims) -> usize {
    dims.d_model
        .max(2 * dims.d_inner)
        .max(dims.xdbl_dim)
        .max(dims.mamba_input_dim)
}

fn next_gpu_ctx_token() -> Result<u64, String> {
    NEXT_GPU_CTX_TOKEN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |token| {
            token.checked_add(1)
        })
        .map_err(|_| "GpuCtx instance token space exhausted".to_string())
}

struct GemmEnvValues {
    mode: Result<String, std::env::VarError>,
    batch_invariant: Result<String, std::env::VarError>,
    fast_gemm: Result<String, std::env::VarError>,
    tensor_cores: Result<String, std::env::VarError>,
    f32_policy: Result<String, std::env::VarError>,
    half_policy: Result<String, std::env::VarError>,
    family: Result<String, std::env::VarError>,
    arch_rung: Result<String, std::env::VarError>,
}

impl GemmEnvValues {
    fn read() -> Self {
        Self {
            mode: std::env::var("MAMBA_RS_GEMM_MODE"),
            batch_invariant: std::env::var("MAMBA_RS_BATCH_INVARIANT"),
            fast_gemm: std::env::var("MAMBA_RS_FAST_GEMM"),
            tensor_cores: std::env::var("MAMBA_RS_BI_TENSOR_CORES"),
            f32_policy: std::env::var("MAMBA_RS_BI_F32_POLICY"),
            half_policy: std::env::var("MAMBA_RS_BI_HALF_POLICY"),
            family: std::env::var("MAMBA_RS_BI_GEMM_FAMILY"),
            arch_rung: std::env::var("MAMBA_RS_ARCH_RUNG"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedGemmEnv {
    mode: GemmMode,
    tensor_cores: bool,
    family: BiGemmFamily,
    f32_policy: F32TriadPolicy,
    half_policy: HalfTriadPolicy,
}

fn env_value_is_present(value: &Result<String, std::env::VarError>) -> bool {
    !matches!(value, Err(std::env::VarError::NotPresent))
}

fn gemm_mode_from_results(
    mode: Result<String, std::env::VarError>,
    batch_invariant: Result<String, std::env::VarError>,
    fast_gemm: Result<String, std::env::VarError>,
) -> Result<GemmMode, String> {
    let canonical_present = env_value_is_present(&mode);
    let batch_invariant_present = env_value_is_present(&batch_invariant);
    let fast_gemm_present = env_value_is_present(&fast_gemm);
    if canonical_present && (batch_invariant_present || fast_gemm_present) {
        return Err(
            "MAMBA_RS_GEMM_MODE conflicts with deprecated MAMBA_RS_BATCH_INVARIANT and \
             MAMBA_RS_FAST_GEMM selectors; remove the old selector"
                .into(),
        );
    }
    if canonical_present {
        return match mode {
            Ok(value) => GemmMode::parse_env_value(&value),
            Err(std::env::VarError::NotUnicode(value)) => Err(format!(
                "MAMBA_RS_GEMM_MODE={value:?} is not valid Unicode \
                 (use deterministic, cublas-fast, or cublas-pedantic)"
            )),
            Err(std::env::VarError::NotPresent) => unreachable!("presence checked above"),
        };
    }

    let batch_invariant =
        optional_tier_flag_from_result("MAMBA_RS_BATCH_INVARIANT", batch_invariant)?;
    let fast_gemm = optional_tier_flag_from_result("MAMBA_RS_FAST_GEMM", fast_gemm)?;
    match (batch_invariant, fast_gemm) {
        (None, None) | (Some(true), None | Some(false)) => Ok(GemmMode::Deterministic),
        (Some(true), Some(true)) => Err(
            "MAMBA_RS_BATCH_INVARIANT=true conflicts with MAMBA_RS_FAST_GEMM=true; \
             use MAMBA_RS_GEMM_MODE to select one mode"
                .into(),
        ),
        (None | Some(false), Some(true)) => Ok(GemmMode::CublasFast),
        (None, Some(false)) | (Some(false), None | Some(false)) => Ok(GemmMode::CublasPedantic),
    }
}

fn resolve_gemm_env(
    values: GemmEnvValues,
    default_family: BiGemmFamily,
) -> Result<ResolvedGemmEnv, String> {
    validate_arch_rung_flag(values.arch_rung)?;
    let deterministic_controls = [
        ("MAMBA_RS_BI_TENSOR_CORES", &values.tensor_cores),
        ("MAMBA_RS_BI_F32_POLICY", &values.f32_policy),
        ("MAMBA_RS_BI_HALF_POLICY", &values.half_policy),
        ("MAMBA_RS_BI_GEMM_FAMILY", &values.family),
    ];
    let mode = gemm_mode_from_results(values.mode, values.batch_invariant, values.fast_gemm)?;
    if mode != GemmMode::Deterministic {
        let present: Vec<_> = deterministic_controls
            .into_iter()
            .filter_map(|(name, value)| env_value_is_present(value).then_some(name))
            .collect();
        if !present.is_empty() {
            return Err(format!(
                "deterministic GEMM controls {} cannot be set while the resolved GEMM mode is {}; \
                 remove them or select deterministic",
                present.join(", "),
                mode.as_str()
            ));
        }
        return Ok(ResolvedGemmEnv {
            mode,
            tensor_cores: true,
            family: default_family,
            f32_policy: F32TriadPolicy::ExactScalarFmaV1,
            half_policy: HalfTriadPolicy::TiledParityV1,
        });
    }

    let tensor_cores =
        optional_tier_flag_from_result("MAMBA_RS_BI_TENSOR_CORES", values.tensor_cores)?
            .unwrap_or(true);
    let f32_policy = f32_triad_policy_from_result(values.f32_policy)?;
    let half_policy = half_triad_policy_from_result(values.half_policy)?
        .unwrap_or(HalfTriadPolicy::TiledParityV1);
    let family = bi_gemm_family_from_result(values.family, default_family)?;
    validate_custom_policy(tensor_cores, half_policy)?;
    Ok(ResolvedGemmEnv {
        mode,
        tensor_cores,
        family,
        f32_policy,
        half_policy,
    })
}

/// Which batch-invariant GEMM family serves the forward while
/// [`GpuCtx::batch_invariant`] is on. Both are deterministic; they are
/// named for the STRUCTURE that produces their differing guarantee, not
/// for a role (either can serve a forward) and not for a tensor-core
/// tier (both instantiate on Tensor Cores for bf16/f16 and on CUDA cores
/// for f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum BiGemmFamily {
    /// The multi-tile dispatcher (`kernels/gemm_bi_triad/`) - the default.
    /// Carries the full triad (NN + TN + NT), so it is the only family
    /// that can serve a backward, and it is the fastest deterministic
    /// path. It picks a kernel by shape (ultra-thin M<32, narrow-N,
    /// GEMV, split-K M<=1024, Slim/Big above), so its invariance holds
    /// across every M INSIDE one bucket; crossing a boundary changes the
    /// reduction association deterministically.
    #[default]
    Triad,
    /// The standalone inference ladder (`kernels/gemm_bi_inference/`):
    /// forward-only NN, batch-invariant BY CONSTRUCTION. Its portable
    /// thin/64/128/wide rungs are bit-identical per output element and
    /// `SPLIT_K=1`; Hopper and datacenter Blackwell use separately
    /// qualified architecture rungs. Training still belongs to Triad.
    Inference,
}

/// `MAMBA_RS_ARCH_RUNG` knows one value, `off`; any other spelling would
/// leave the architecture rung running as if the flag were unset, so it is
/// refused here instead.
fn validate_arch_rung_flag(value: Result<String, std::env::VarError>) -> Result<(), String> {
    match value {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "off" => Ok(()),
            other => Err(format!(
                "MAMBA_RS_ARCH_RUNG={other:?} is not a recognized value; only off is \
                 (leave it unset to keep the rung on)"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(()),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "MAMBA_RS_ARCH_RUNG={value:?} is not valid Unicode; only off is recognized"
        )),
    }
}

fn bi_gemm_family_from_result(
    value: Result<String, std::env::VarError>,
    default_family: BiGemmFamily,
) -> Result<BiGemmFamily, String> {
    match value {
        Ok(value) => match value.trim() {
            "" => Ok(BiGemmFamily::Triad),
            value if value.eq_ignore_ascii_case("triad") => Ok(BiGemmFamily::Triad),
            value
                if value.eq_ignore_ascii_case("inference")
                    || value.eq_ignore_ascii_case("fixed") =>
            {
                Ok(BiGemmFamily::Inference)
            }
            value => Err(format!(
                "MAMBA_RS_BI_GEMM_FAMILY={value:?} is not a recognized family; \
                 only inference or triad are accepted"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(default_family),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "MAMBA_RS_BI_GEMM_FAMILY={value:?} is not valid Unicode; \
             only inference or triad are accepted"
        )),
    }
}

fn tier_flag_from_result(
    name: &str,
    value: Result<String, std::env::VarError>,
) -> Result<bool, String> {
    match value {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" | "" => Ok(false),
            other => Err(format!(
                "{name}={other:?} is not a recognized flag value \
                 (use 1/true/yes/on or 0/false/no/off)"
            )),
        },
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "{name}={value:?} is not valid Unicode \
             (use 1/true/yes/on or 0/false/no/off)"
        )),
    }
}

fn optional_tier_flag_from_result(
    name: &str,
    value: Result<String, std::env::VarError>,
) -> Result<Option<bool>, String> {
    if matches!(value, Err(std::env::VarError::NotPresent)) {
        Ok(None)
    } else {
        tier_flag_from_result(name, value).map(Some)
    }
}

/// Numeric policy for deterministic batch-invariant f32 GEMMs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum F32TriadPolicy {
    /// Preserve the scalar `__fmaf_rn` reduction contract.
    #[default]
    ExactScalarFmaV1 = 0,
    /// Permit frozen and qualified deterministic TF32 routes.
    ///
    /// This is permission, not a forced backend. Unsupported or unmeasured
    /// cells keep the exact scalar FMA contract.
    AllowDeterministicTf32V1 = 1,
}

impl F32TriadPolicy {
    /// Parse the strict public environment spelling for this policy.
    pub fn parse_env_value(value: &str) -> Result<Self, String> {
        match value.trim_ascii() {
            "exact" => Ok(Self::ExactScalarFmaV1),
            "tf32" => Ok(Self::AllowDeterministicTf32V1),
            _ => Err(format!(
                "MAMBA_RS_BI_F32_POLICY={value:?} is not a recognized f32 triad policy \
                 (use exact or tf32)"
            )),
        }
    }
}

fn f32_triad_policy_from_result(
    value: Result<String, std::env::VarError>,
) -> Result<F32TriadPolicy, String> {
    match value {
        Ok(value) => F32TriadPolicy::parse_env_value(&value),
        Err(std::env::VarError::NotPresent) => Ok(F32TriadPolicy::ExactScalarFmaV1),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "MAMBA_RS_BI_F32_POLICY={value:?} is not valid Unicode (use exact or tf32)"
        )),
    }
}

/// Numeric policy for batch-invariant half-precision (bf16/f16) Triad GEMMs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HalfTriadPolicy {
    /// Every automatic half route reproduces the forced portable tensor-core
    /// kernel bit for bit: one owner CTA per output tile, one reduction order.
    #[default]
    TiledParityV1 = 0,
    /// Permit the measured stream-K half routes: a persistent grid folds
    /// per-CTA partials in a fixed order that differs from the tiled
    /// reduction, while repeated eager and graph launches of the same route
    /// stay bit-identical.
    ///
    /// This is permission, not a forced schedule. Unmeasured shapes, and
    /// shapes whose tile grid already fills the device, keep the tiled
    /// contract.
    AllowStreamKFixedOrderV1 = 1,
}

impl HalfTriadPolicy {
    /// Parse the strict public environment spelling for this policy.
    pub fn parse_env_value(value: &str) -> Result<Self, String> {
        match value.trim_ascii() {
            "tiled" => Ok(Self::TiledParityV1),
            "streamk" => Ok(Self::AllowStreamKFixedOrderV1),
            _ => Err(format!(
                "MAMBA_RS_BI_HALF_POLICY={value:?} is not a recognized half triad policy \
                 (use tiled or streamk)"
            )),
        }
    }
}

/// The explicit half policy, `None` when the environment names none: the
/// tensor-core tier then runs its fastest deterministic route (stream-K
/// where the SM89 rule admits it), and `tiled` opts back into bit parity
/// with the tiled kernels.
fn half_triad_policy_from_result(
    value: Result<String, std::env::VarError>,
) -> Result<Option<HalfTriadPolicy>, String> {
    match value {
        Ok(value) => HalfTriadPolicy::parse_env_value(&value).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "MAMBA_RS_BI_HALF_POLICY={value:?} is not valid Unicode (use tiled or streamk)"
        )),
    }
}

fn validate_custom_policy(
    bi_tensor_cores: bool,
    half_policy: HalfTriadPolicy,
) -> Result<(), String> {
    if half_policy == HalfTriadPolicy::AllowStreamKFixedOrderV1 && !bi_tensor_cores {
        return Err(
            "MAMBA_RS_BI_HALF_POLICY=streamk requires MAMBA_RS_BI_TENSOR_CORES=1; \
             stream-K routes are in the deterministic tensor-core tier"
                .into(),
        );
    }
    Ok(())
}

pub use super::kernel_identity::{
    BackendSet as GemmBackendSet, GemmPolicy, GemmRouteIdentity, NumericContractSet,
};

/// Complete policy, compiler, artifact, and device identity pinned by graphs.
pub type GemmRoute = GemmRouteIdentity;

struct GemmRouteRecorder {
    context: GemmRouteIdentity,
    mode: GemmRouteRecorderMode,
    routes: Vec<ResolvedGemmRoute>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmRouteRecorderMode {
    /// Runs outside CUDA capture, so route storage and prepared launch caches
    /// may grow while the eager body resolves its exact physical inventory.
    GrowableEager,
    /// Storage was reserved before CUDA capture and may not grow inside it.
    FixedCapture { route_capacity: u32 },
}

pub(crate) struct GemmRouteRecordingGuard<'a> {
    ctx: &'a GpuCtx,
}

impl GemmRouteRecordingGuard<'_> {
    #[cfg(test)]
    pub(crate) fn finish(self) -> Result<CapturedGemmGraphPlan, String> {
        let recorder = self.ctx.take_gemm_route_recording()?;
        if recorder.routes.is_empty() {
            return Err("captured GEMM graph plan must not be empty".into());
        }
        recorder
            .context
            .ensure_current(self.ctx.gemm_route(), "GEMM graph capture")?;
        let launches = build_resolved_gemm_launch_set(&recorder.routes)?;
        let routes = recorder.routes.into_boxed_slice();
        Ok(CapturedGemmGraphPlan::new(
            recorder.context,
            launches,
            routes,
        ))
    }

    pub(crate) fn finish_trace(self) -> Result<RecordedGemmTrace, String> {
        let recorder = self.ctx.take_gemm_route_recording()?;
        if recorder.mode != GemmRouteRecorderMode::GrowableEager {
            return Err("fixed GEMM capture recording cannot produce an eager trace".into());
        }
        recorder
            .context
            .ensure_current(self.ctx.gemm_route(), "eager GEMM route recording")?;
        RecordedGemmTrace::from_routes(recorder.context, recorder.routes)
    }

    pub(crate) fn finish_against_manifest(
        self,
        manifest: &PreparedGemmCaptureManifest,
    ) -> Result<Option<CapturedGemmGraphPlan>, String> {
        let recorder = self.ctx.take_gemm_route_recording()?;
        let GemmRouteRecorderMode::FixedCapture { route_capacity } = recorder.mode else {
            return Err("growable eager GEMM recording cannot finish a CUDA capture".into());
        };
        let route_capacity =
            usize::try_from(route_capacity).expect("u32 GEMM route capacity always fits in usize");
        manifest.validate_capture_request(self.ctx.gemm_route(), route_capacity)?;
        recorder
            .context
            .ensure_current(self.ctx.gemm_route(), "GEMM graph capture")?;
        let launches = resolved_gemm_launches(&recorder.routes)?;
        manifest.validate_capture_result(recorder.context, launches)?;
        let Some(launches) = launches else {
            return Ok(None);
        };
        let routes = recorder.routes.into_boxed_slice();
        Ok(Some(CapturedGemmGraphPlan::new(
            recorder.context,
            launches,
            routes,
        )))
    }
}

fn resolved_gemm_launches(
    routes: &[ResolvedGemmRoute],
) -> Result<Option<ResolvedGemmLaunchSet>, String> {
    if routes.is_empty() {
        Ok(None)
    } else {
        build_resolved_gemm_launch_set(routes).map(Some)
    }
}

impl Drop for GemmRouteRecordingGuard<'_> {
    fn drop(&mut self) {
        self.ctx.clear_gemm_route_recording();
    }
}

/// GPU execution context — holds everything needed for kernel launches.
///
/// Created once at init, passed by reference to all GPU functions.
#[doc(hidden)]
pub struct GpuCtxResources {
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub kernels: Arc<MambaKernels>,
    pub blas: cudarc::cublas::CudaBlas,
    pub _blas_workspace: Arc<cudarc::driver::CudaSlice<u8>>,
    /// Reusable GPU byte staging buffer for f32→bf16/f16 activation downcast
    /// before mixed-precision GEMM. Grown lazily on first use.
    half_staging: RefCell<Option<cudarc::driver::CudaSlice<u8>>>,
    half_staging_ptr: RefCell<cudarc::driver::sys::CUdeviceptr>,
    half_staging_bytes: RefCell<usize>,
    bi_upcast_scratch: [RefCell<Option<super::buffers::GpuBuffer>>; 3],
}

/// GPU execution context — holds everything needed for kernel launches.
///
/// Created once at init, passed by reference to all GPU functions. Its GPU
/// resource core is immutable so a captured graph can retain the exact stream,
/// modules, cuBLAS workspace, and lazy scratch allocations it recorded.
///
/// ```compile_fail
/// # use mamba_rs::mamba_ssm::gpu::context::GpuCtx;
/// # fn replace_stream(ctx: &mut GpuCtx, stream: std::sync::Arc<cudarc::driver::CudaStream>) {
/// ctx.stream = stream;
/// # }
/// ```
pub struct GpuCtx {
    resources: Rc<GpuCtxResources>,
    gemm_route_recorder: RefCell<Option<GemmRouteRecorder>>,
    f32_prepared_launches: RefCell<F32PreparedLaunchCache>,
    sm120_prepared_launches: RefCell<Sm120PreparedLaunchCache>,
    sm100_prepared_launches: RefCell<Sm100PreparedLaunchCache>,
    sm90a_prepared_launches: RefCell<Sm90aPreparedLaunchCache>,
    pub(crate) fixed_tf32_maps: RefCell<super::gemm_bi_inference::FixedTf32MapCache>,
    pub(crate) fixed_postbias_maps: RefCell<super::gemm_bi_inference::FixedPostBiasMapCache>,
    pub(crate) fixed_half_maps: RefCell<super::gemm_bi_inference::FixedHalfMapCache>,
    /// Canonical GEMM route and vendor-math authority.
    gemm_mode: Cell<GemmMode>,
    /// Diagnostic retained after an unverified cuBLAS math rollback.
    gemm_unusable: RefCell<Option<String>>,
    /// Opt-in tensor-core tier for the batch-invariant typed GEMMs.
    /// SEPARATE numeric contract: mma.sync f32 accumulation
    /// differs from the scalar __fmaf_rn chain, so outputs do not bit-match
    /// the scalar triad — but the TC kernels are fully deterministic and
    /// batch-invariant for their admitted shapes. Effective only together
    /// with `batch_invariant`. Env: MAMBA_RS_BI_TENSOR_CORES.
    bi_tensor_cores: std::cell::Cell<bool>,
    /// Which deterministic family serves the forward while
    /// `batch_invariant` is on. Env: MAMBA_RS_BI_GEMM_FAMILY
    /// (`triad` | `fixed`). Part of the numeric route, so it rides
    /// [`GpuCtx::gemm_route`] into every capture identity.
    bi_gemm_family: std::cell::Cell<BiGemmFamily>,
    /// Explicit numeric policy for deterministic batch-invariant f32 GEMMs.
    f32_triad_policy: std::cell::Cell<F32TriadPolicy>,
    /// Explicit numeric policy for batch-invariant half-precision GEMMs.
    half_triad_policy: std::cell::Cell<HalfTriadPolicy>,
    /// The state capacity the kernels were compiled with — part of the
    /// numeric-route identity a bench stamp must carry.
    state_cap: usize,
    instance_token: u64,
    device_identity: super::kernel_identity::DeviceIdentity,
    device_caps: super::kernel_identity::DeviceCaps,
    policy_hash: super::kernel_identity::Sha256Digest,
    /// Number of CUDA graphs captured on this context. Mode and custom-policy
    /// setters warn after capture; replay compares the complete recorded route
    /// and rejects a different live configuration.
    graphs_captured: std::cell::Cell<u64>,
    /// Once a graph can observe the grow-only typed-GEMM scratch, its
    /// allocation addresses are immutable for the rest of this context.
    graph_scratch_frozen: std::cell::Cell<bool>,
}

impl std::ops::Deref for GpuCtx {
    type Target = GpuCtxResources;

    fn deref(&self) -> &Self::Target {
        &self.resources
    }
}

fn validate_multiprocessor_identity(
    device_multiprocessor_count: u32,
    kernel_multiprocessor_count: u32,
) -> Result<(), String> {
    if device_multiprocessor_count == 0 || kernel_multiprocessor_count == 0 {
        return Err("CUDA topology requires a nonzero multiprocessor count".into());
    }
    if device_multiprocessor_count != kernel_multiprocessor_count {
        return Err(format!(
            "CUDA topology changed while loading kernels: device identity has {device_multiprocessor_count} multiprocessors but loaded kernels observed {kernel_multiprocessor_count}"
        ));
    }
    Ok(())
}

struct CublasMathBackend<'a> {
    blas: &'a cudarc::cublas::CudaBlas,
}

impl MathModeBackend for CublasMathBackend<'_> {
    type Mode = cudarc::cublas::sys::cublasMath_t;

    fn query(&mut self) -> Result<Self::Mode, String> {
        let mut mode = Self::Mode::CUBLAS_DEFAULT_MATH;
        let status =
            unsafe { cudarc::cublas::sys::cublasGetMathMode(*self.blas.handle(), &mut mode) };
        if status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(mode)
        } else {
            Err(format!("cublasGetMathMode failed: {status:?}"))
        }
    }

    fn update(&mut self, mode: Self::Mode) -> Result<(), String> {
        let status = unsafe { cudarc::cublas::sys::cublasSetMathMode(*self.blas.handle(), mode) };
        if status == cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(format!("cublasSetMathMode({mode:?}) failed: {status:?}"))
        }
    }
}

impl GpuCtx {
    /// Create a GPU context, compile its kernels, and initialize cuBLAS.
    ///
    /// The context starts in [`GemmMode::Deterministic`] with the
    /// [`BiGemmFamily::Triad`] family, exact f32 policy, tiled half policy,
    /// and tensor-core permission enabled. This constructor ignores GEMM
    /// environment variables; use [`Self::new_from_env`] to read them.
    /// The kernel state capacity is 64.
    ///
    /// ```no_run
    /// # use mamba_rs::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
    /// # use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    /// # fn example(device: &GpuDevice) -> Result<(), String> {
    /// let ctx = GpuCtx::new(device)?;
    /// assert_eq!(ctx.gemm_mode(), GemmMode::Deterministic);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        Self::new_with_mode(device, GemmMode::default())
    }

    /// Create a context in an explicit GEMM mode with state capacity 64.
    ///
    /// `mode` controls custom-versus-cuBLAS dispatch, cuBLAS handle math, and
    /// context-aware GemmEx compute. Environment mode selectors are ignored.
    /// Construction returns an error if kernel compilation, resource setup, or
    /// the requested cuBLAS math configuration fails.
    ///
    /// ```no_run
    /// # use mamba_rs::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
    /// # use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    /// # fn example(device: &GpuDevice) -> Result<(), String> {
    /// let ctx = GpuCtx::new_with_mode(device, GemmMode::CublasPedantic)?;
    /// assert_eq!(ctx.gemm_mode(), GemmMode::CublasPedantic);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_mode(device: &GpuDevice, mode: GemmMode) -> Result<Self, String> {
        Self::new_with_state_cap_and_mode(device, 64, mode)
    }

    /// Create a context from the `MAMBA_RS_*` GEMM environment variables.
    ///
    /// `MAMBA_RS_GEMM_MODE` accepts the canonical names returned by
    /// [`GemmMode::as_str`]. Its presence conflicts with either legacy
    /// `MAMBA_RS_BATCH_INVARIANT` or `MAMBA_RS_FAST_GEMM`. Without the
    /// canonical variable, the legacy variables are resolved together:
    /// batch-invariant true selects Deterministic unless fast is also true;
    /// fast true with batch-invariant absent or false selects CublasFast; and
    /// an explicit false without a positive selector selects CublasPedantic.
    /// Their joint absence defaults to Deterministic, while joint true values
    /// are an error. Deterministic family, precision, and tensor-core variables
    /// are rejected by presence under a cuBLAS mode. All values are resolved
    /// and validated before GPU resources are constructed. Invalid,
    /// conflicting, or non-Unicode values return an error.
    pub fn new_from_env(device: &GpuDevice) -> Result<Self, String> {
        let config = resolve_gemm_env(GemmEnvValues::read(), BiGemmFamily::Triad)?;
        Self::new_with_state_cap_and_config(device, 64, config)
    }

    /// [`Self::new_from_env`] with an explicit kernel state capacity.
    ///
    /// The environment is resolved before construction. `state_cap` is passed
    /// to kernel compilation and is returned by [`Self::state_cap`].
    pub fn new_from_env_with_state_cap(
        device: &GpuDevice,
        state_cap: usize,
    ) -> Result<Self, String> {
        let config = resolve_gemm_env(GemmEnvValues::read(), BiGemmFamily::Triad)?;
        Self::new_with_state_cap_and_config(device, state_cap, config)
    }

    /// Create a GPU context whose kernels are compiled with the given
    /// state capacity in [`GemmMode::Deterministic`].
    ///
    /// This constructor ignores GEMM environment variables. See
    /// [`Self::new_with_state_cap_and_mode`] for an explicit alternative mode.
    pub fn new_with_state_cap(device: &GpuDevice, state_cap: usize) -> Result<Self, String> {
        Self::new_with_state_cap_and_mode(device, state_cap, GemmMode::default())
    }

    /// Create a GPU context with an explicit kernel state capacity and mode.
    ///
    /// `state_cap` controls the compiled SSM state capacity. `mode` controls
    /// GEMM dispatch and cuBLAS numeric settings. Environment mode selectors
    /// are ignored. Invalid resource or cuBLAS setup returns an error.
    pub fn new_with_state_cap_and_mode(
        device: &GpuDevice,
        state_cap: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_with_state_cap_and_config(
            device,
            state_cap,
            ResolvedGemmEnv {
                mode,
                tensor_cores: true,
                family: BiGemmFamily::Triad,
                f32_policy: F32TriadPolicy::ExactScalarFmaV1,
                half_policy: HalfTriadPolicy::TiledParityV1,
            },
        )
    }

    fn new_with_state_cap_and_config(
        device: &GpuDevice,
        state_cap: usize,
        config: ResolvedGemmEnv,
    ) -> Result<Self, String> {
        // Disable cudarc's per-slice CudaEvent tracking. Rationale: we
        // execute every op on a single ctx.stream throughout fwd / bwd /
        // optimizer, so the multi-stream synchronization events cudarc
        // would otherwise auto-record per `&CudaSlice` kernel arg
        // (driver/safe/launch.rs:100) only add overhead — and worse,
        // they emit cuStreamWaitEvent ops that reference work issued
        // BEFORE `cuStreamBeginCapture`, breaking CUDA Graph capture
        // with CUDA_ERROR_STREAM_CAPTURE_ISOLATION ("dependency created
        // on uncaptured work in another stream"). Inference graphs work
        // today only because they exclusively use `cached_ptr()` (raw
        // u64) which bypasses the slice-arg path. Backward + optimizer
        // hit the slice path, so disabling event tracking is the proper
        // fix that doesn't require rewriting every kernel call.
        //
        // Safety contract (per cudarc::CudaContext::disable_event_tracking):
        //   1. No slice freed while another stream uses it. ✓ (single stream)
        //   2. No slice used on another stream before alloc completes.
        //      ✓ (we sync on ctx.stream after every batch alloc)
        //   3. No concurrent writes from multiple streams. ✓ (single stream)
        // All three hold by construction since GpuCtx owns exactly ONE
        // CudaStream and every op routes through it.
        unsafe {
            device.context().disable_event_tracking();
        }
        let stream = device.fork_stream()?;
        let arch = device.nvrtc_target();
        let kernels = MambaKernels::compile_with_state_cap(device.context(), arch, state_cap)?;
        let device_identity = device.identity();
        validate_multiprocessor_identity(
            device_identity.multiprocessor_count,
            kernels.multiprocessor_count(),
        )?;
        // The splitk/transpose scratch buffers inside `kernels` were
        // alloc_zeros'd on the DEFAULT stream; `ctx.stream` is NON_BLOCKING
        // and never orders against it. Drain once here so first use on
        // ctx.stream can't race the init memset (same hazard class as the
        // legacy-stream memcpy fix in buffers.rs).
        device
            .default_stream()
            .synchronize()
            .map_err(|e| format!("default-stream drain after kernel compile: {e:?}"))?;
        validate_custom_policy(config.tensor_cores, config.half_policy)?;
        let (blas, ws) = device.create_cublas(&stream, config.mode)?;
        // Numeric routing defaults are nonambient. Explicit setters or the
        // `new_from_env*` constructors are the only ways to change them.
        let instance_token = next_gpu_ctx_token()?;
        let compiler = kernels.compiler_identity();
        let optin_shared_bytes = device
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(|error| format!("query opt-in shared memory: {error:?}"))?;
        let tensor_map_access = device
            .context()
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED,
            )
            .map(|value| value != 0)
            .unwrap_or(false);
        let device_caps = super::kernel_identity::DeviceCaps {
            compute_capability: device.compute_capability,
            nvrtc_version: compiler.nvrtc_version,
            accepted_target: kernels
                .specialized_compiler_identity()
                .map(|identity| identity.target),
            optin_shared_bytes: u32::try_from(optin_shared_bytes)
                .map_err(|_| format!("negative opt-in shared memory {optin_shared_bytes}"))?,
            tensor_map_access,
        };
        let kernels = Arc::new(kernels);
        Ok(Self {
            resources: Rc::new(GpuCtxResources {
                stream,
                kernels,
                blas,
                _blas_workspace: Arc::new(ws),
                half_staging: RefCell::new(None),
                half_staging_ptr: RefCell::new(0),
                half_staging_bytes: RefCell::new(0),
                bi_upcast_scratch: [RefCell::new(None), RefCell::new(None), RefCell::new(None)],
            }),
            gemm_route_recorder: RefCell::new(None),
            f32_prepared_launches: RefCell::new(F32PreparedLaunchCache::default()),
            sm120_prepared_launches: RefCell::new(Sm120PreparedLaunchCache::default()),
            sm100_prepared_launches: RefCell::new(Sm100PreparedLaunchCache::default()),
            sm90a_prepared_launches: RefCell::new(Sm90aPreparedLaunchCache::default()),
            fixed_tf32_maps: RefCell::new(super::gemm_bi_inference::FixedTf32MapCache::default()),
            fixed_postbias_maps: RefCell::new(
                super::gemm_bi_inference::FixedPostBiasMapCache::default(),
            ),
            fixed_half_maps: RefCell::new(super::gemm_bi_inference::FixedHalfMapCache::default()),
            gemm_mode: Cell::new(config.mode),
            gemm_unusable: RefCell::new(None),
            bi_tensor_cores: Cell::new(config.tensor_cores),
            bi_gemm_family: Cell::new(config.family),
            f32_triad_policy: Cell::new(config.f32_policy),
            half_triad_policy: Cell::new(config.half_policy),
            state_cap,
            instance_token,
            device_identity,
            device_caps,
            policy_hash: super::kernel_identity::gemm_dispatch_policy_digest(
                device_identity.multiprocessor_count,
            ),
            graphs_captured: Cell::new(0),
            graph_scratch_frozen: Cell::new(false),
        })
    }

    pub(crate) fn resource_anchor(&self) -> Rc<GpuCtxResources> {
        self.resources.clone()
    }

    pub(crate) fn instance_token(&self) -> u64 {
        self.instance_token
    }

    pub(crate) fn stream_token(&self) -> usize {
        Arc::as_ptr(&self.stream) as usize
    }

    /// Freeze graph-visible staging allocations immediately before capture.
    /// A failed capture deliberately leaves the context frozen: CUDA may have
    /// observed the addresses before returning the error.
    pub(crate) fn freeze_graph_scratch(&self) {
        self.graph_scratch_frozen.set(true);
    }

    /// Run `f` with the three grow-only f32 scratch buffers used by the
    /// batch-invariant typed-GEMM upcast fallback, sized to at least
    /// `elems = (a, b, c)` f32 elements each. Buffers persist across calls
    /// (grow-only) so steady-state training steps do not allocate.
    pub(crate) fn with_bi_upcast_scratch<R>(
        &self,
        elems: (usize, usize, usize),
        f: impl FnOnce(
            &mut super::buffers::GpuBuffer,
            &mut super::buffers::GpuBuffer,
            &mut super::buffers::GpuBuffer,
        ) -> Result<R, String>,
    ) -> Result<R, String> {
        let sizes = [elems.0, elems.1, elems.2];
        let needs_growth = self
            .bi_upcast_scratch
            .iter()
            .zip(&sizes)
            .any(|(cell, &need)| cell.borrow().as_ref().map_or(0, |b| b.len()) < need.max(1));
        if needs_growth && self.graph_scratch_frozen.get() {
            return Err(
                "batch-invariant upcast scratch cannot grow after CUDA graph capture; \
                 destroy the context or pre-size the largest shape before capture"
                    .into(),
            );
        }
        for (cell, &need) in self.bi_upcast_scratch.iter().zip(&sizes) {
            let mut slot = cell.borrow_mut();
            let have = slot.as_ref().map_or(0, |b| b.len());
            let need = need.max(1);
            if have < need {
                *slot = Some(super::buffers::GpuBuffer::zeros(&self.stream, need)?);
            }
        }
        let mut a = self.bi_upcast_scratch[0].borrow_mut();
        let mut b = self.bi_upcast_scratch[1].borrow_mut();
        let mut c = self.bi_upcast_scratch[2].borrow_mut();
        f(
            a.as_mut().expect("bi_upcast_scratch[0] sized above"),
            b.as_mut().expect("bi_upcast_scratch[1] sized above"),
            c.as_mut().expect("bi_upcast_scratch[2] sized above"),
        )
    }

    /// Current device pointers of the three `bi_upcast_scratch` slots
    /// (0 = unallocated). CUDA-Graph guard: capture snapshots these and
    /// replay asserts they have not moved — a lazy regrow after capture
    /// would leave the graph dereferencing freed memory.
    pub(crate) fn bi_upcast_scratch_ptrs(&self) -> [cudarc::driver::sys::CUdeviceptr; 3] {
        let p = |i: usize| {
            self.bi_upcast_scratch[i]
                .borrow()
                .as_ref()
                .map_or(0, |b| b.cached_ptr())
        };
        [p(0), p(1), p(2)]
    }

    pub(crate) fn ensure_graph_scratch_ptrs(
        &self,
        half_staging: cudarc::driver::sys::CUdeviceptr,
        bi_upcast: [cudarc::driver::sys::CUdeviceptr; 3],
        label: &str,
    ) -> Result<(), String> {
        if self.half_staging_ptr() != half_staging || self.bi_upcast_scratch_ptrs() != bi_upcast {
            return Err(format!(
                "{label}: graph-visible staging scratch changed since capture"
            ));
        }
        Ok(())
    }

    /// Pre-size the batch-invariant typed-GEMM upcast scratch for a mixed
    /// training step BEFORE CUDA Graph capture, so `with_bi_upcast_scratch`
    /// inside the captured body never grows (a lazy grow during capture
    /// fails the capture; one after capture frees pointers a previously
    /// captured graph still references). Sizing covers every step GEMM
    /// (in_proj / x_proj / dt_proj / out_proj fwd, dW, dX) at
    /// `m = batch·seq_len`. No-op for f32 or when the batch-invariant
    /// flag is off (the captured body then never touches this scratch).
    pub fn presize_bi_upcast_scratch_for_train(
        &self,
        cfg: &MambaConfig,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) || !self.batch_invariant() {
            return Ok(());
        }
        let m = batch * seq_len;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let xproj_out = cfg.dt_rank() + 2 * cfg.d_state;
        // Largest single GEMM operand dim and largest K×N weight across the
        // step's GEMMs; every slot request (m·k, k·n, m·n) is ≤ this bound.
        let max_dim = dm.max(2 * di).max(xproj_out);
        let max_kn = (dm * 2 * di)
            .max(di * xproj_out)
            .max(cfg.dt_rank() * di)
            .max(di * dm);
        let elems = (m * max_dim).max(max_kn);
        self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))
    }

    /// [`Self::presize_bi_upcast_scratch_for_train`] for a trainable
    /// NON-IDENTITY input projection: extends the sizing bound with the
    /// input_proj GEMM operands — the `m·input_dim` activation slot and the
    /// `input_dim·d_model` weight. Without this, an `input_dim` (e.g. a
    /// vision patch dim P²) exceeding `max(d_model, 2·d_inner, xproj_out)`
    /// makes `with_bi_upcast_scratch` grow INSIDE CUDA Graph capture
    /// (illegal alloc) or after it (freed-pointer replay). With
    /// `input_dim <= d_model` (incl. identity) the bound equals the base
    /// function's.
    pub fn presize_bi_upcast_scratch_for_train_with_input(
        &self,
        cfg: &MambaConfig,
        batch: usize,
        seq_len: usize,
        input_dim: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) || !self.batch_invariant() {
            return Ok(());
        }
        let m = batch * seq_len;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let xproj_out = cfg.dt_rank() + 2 * cfg.d_state;
        let max_dim = dm.max(2 * di).max(xproj_out).max(input_dim);
        let max_kn = (dm * 2 * di)
            .max(di * xproj_out)
            .max(cfg.dt_rank() * di)
            .max(di * dm)
            .max(input_dim * dm);
        let elems = (m * max_dim).max(max_kn);
        self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))
    }

    /// Mamba-3 twin of [`Self::presize_bi_upcast_scratch_for_train`] — the
    /// M3 step GEMMs are in_proj (`d_model → in_proj_out_dim`) and
    /// out_proj (`d_inner → d_model`).
    pub fn presize_bi_upcast_scratch_for_train_m3(
        &self,
        cfg: &crate::mamba3_siso::config::Mamba3Config,
        batch: usize,
        seq_len: usize,
        input_dim: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) || !self.batch_invariant() {
            return Ok(());
        }
        let m = batch * seq_len;
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ip = cfg.in_proj_out_dim();
        // input_dim covers the non-identity input projection's operands
        // (X[m, input_dim] @ W[input_dim, d_model]); with the identity
        // branch it equals d_model and changes nothing.
        let max_dim = dm.max(ip).max(di).max(input_dim);
        let max_kn = (dm * ip).max(di * dm).max(input_dim * dm);
        let elems = (m * max_dim).max(max_kn);
        self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))
    }

    pub(crate) fn presize_mixed_graph_scratch_m1(
        &self,
        dims: &super::forward::GpuMambaDims,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let max_dim = m1_mixed_graph_max_dim(dims);
        self.ensure_half_staging(dims.bt() * max_dim * dtype.size_bytes())?;
        if self.batch_invariant() {
            let max_kn = (dims.d_model * 2 * dims.d_inner)
                .max(dims.d_inner * dims.xdbl_dim)
                .max(dims.dt_rank * dims.d_inner)
                .max(dims.d_inner * dims.d_model)
                .max(dims.mamba_input_dim * dims.d_model);
            let elems = (dims.bt() * max_dim).max(max_kn);
            self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))?;
        }
        Ok(())
    }

    pub(crate) fn presize_mixed_graph_scratch_m3(
        &self,
        dims: &crate::mamba3_siso::gpu::state::GpuMamba3Dims,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let max_dim = dims
            .d_model
            .max(dims.d_inner)
            .max(dims.in_proj_dim)
            .max(dims.mamba_input_dim);
        self.ensure_half_staging(dims.bt() * max_dim * dtype.size_bytes())?;
        if self.batch_invariant() {
            let max_kn = (dims.d_model * dims.in_proj_dim)
                .max(dims.d_inner * dims.d_model)
                .max(dims.mamba_input_dim * dims.d_model);
            let elems = (dims.bt() * max_dim).max(max_kn);
            self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))?;
        }
        Ok(())
    }

    /// Change the context's GEMM mode and cuBLAS math setting together.
    ///
    /// The change is rejected while this context's stream is being captured or
    /// while a GEMM route recorder is active. The current handle math is read
    /// before mutation and restored if the requested cuBLAS update fails. The
    /// canonical mode is published only after a successful update. If rollback
    /// cannot be verified, the context becomes unusable and later supported
    /// GEMM, capture, and replay entry points return the stored diagnostic.
    /// Calling this with the current mode is a no-op on a usable context.
    ///
    /// ```no_run
    /// # use mamba_rs::mamba_ssm::gpu::context::{GemmMode, GpuCtx};
    /// # fn example(ctx: &GpuCtx) -> Result<(), String> {
    /// ctx.set_gemm_mode(GemmMode::CublasFast)?;
    /// assert_eq!(ctx.gemm_mode(), GemmMode::CublasFast);
    /// ctx.set_gemm_mode(GemmMode::Deterministic)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_gemm_mode(&self, mode: GemmMode) -> Result<(), String> {
        self.ensure_gemm_usable()?;
        if mode == self.gemm_mode.get() {
            return Ok(());
        }
        validate_custom_policy(self.bi_tensor_cores.get(), self.half_triad_policy.get())?;
        if self
            .gemm_route_recorder
            .try_borrow()
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())?
            .is_some()
        {
            return Err("cannot change GEMM mode while GEMM route recording is active".into());
        }
        let capture_status = self
            .stream
            .capture_status()
            .map_err(|error| format!("query CUDA stream capture state: {error:?}"))?;
        if capture_status
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(format!(
                "cannot change GEMM mode while CUDA stream capture state is {capture_status:?}"
            ));
        }

        let mut backend = CublasMathBackend { blas: &self.blas };
        match change_math_mode(&mut backend, mode.cublas_math()) {
            Ok(()) => {
                if self.graphs_captured.get() > 0 {
                    eprintln!(
                        "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
                    );
                }
                self.gemm_mode.set(mode);
                Ok(())
            }
            Err(MathTransitionError::Recoverable(error)) => Err(error),
            Err(MathTransitionError::Unusable(error)) => {
                let diagnostic = format!(
                    "GPU context is unusable after an unverified cuBLAS math rollback: {error}"
                );
                *self.gemm_unusable.borrow_mut() = Some(diagnostic.clone());
                Err(diagnostic)
            }
        }
    }

    /// Return the context's canonical GEMM mode.
    ///
    /// Custom family and precision settings are separate and may be dormant
    /// while a cuBLAS mode is selected; they are preserved across mode changes.
    pub fn gemm_mode(&self) -> GemmMode {
        self.gemm_mode.get()
    }

    /// Reject GEMM, capture, and replay work after an unverified math rollback.
    ///
    /// Context-aware routing layers call this before they select or enqueue a
    /// GEMM. The returned diagnostic is stable for the remaining context
    /// lifetime; creating a new context is the recovery path.
    pub(crate) fn ensure_gemm_usable(&self) -> Result<(), String> {
        match self.gemm_unusable.borrow().as_ref() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    pub(crate) fn ensure_vendor_gemm(&self, label: &str) -> Result<GemmMode, String> {
        self.ensure_gemm_usable()?;
        let mode = self.gemm_mode();
        if mode == GemmMode::Deterministic {
            Err(format!(
                "{label}: deterministic GEMM mode reached a cuBLAS dispatch boundary"
            ))
        } else {
            Ok(mode)
        }
    }

    /// Compatibility adapter for the pre-0.7 batch-invariant flag.
    ///
    /// New code should call [`Self::set_gemm_mode`]. `true` selects
    /// [`GemmMode::Deterministic`]. `false` changes Deterministic to
    /// [`GemmMode::CublasPedantic`] and leaves an existing cuBLAS mode
    /// unchanged. A failed transition panics because this legacy signature
    /// cannot return the error.
    #[deprecated(since = "0.7.0", note = "use GpuCtx::set_gemm_mode")]
    pub fn set_batch_invariant(&self, on: bool) {
        let mode = self.gemm_mode().legacy_batch_invariant(on);
        self.set_gemm_mode(mode)
            .unwrap_or_else(|error| panic!("set_batch_invariant migration failed: {error}"));
    }

    /// Return whether [`GemmMode::Deterministic`] is selected.
    pub fn batch_invariant(&self) -> bool {
        self.gemm_mode().batch_invariant()
    }

    /// Choose which custom family serves deterministic forward GEMMs.
    ///
    /// The value is dormant in either cuBLAS mode and is preserved across
    /// mode changes. See [`BiGemmFamily`] for the available custom routes.
    pub fn set_bi_gemm_family(&self, family: BiGemmFamily) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.bi_gemm_family.set(family);
    }

    /// The deterministic family currently selected.
    pub fn bi_gemm_family(&self) -> BiGemmFamily {
        self.bi_gemm_family.get()
    }

    /// Set permission for qualified deterministic Triad tensor-core GEMM routes.
    ///
    /// Fresh contexts set this to `true`. It affects dispatch only in
    /// [`GemmMode::Deterministic`]; cuBLAS modes leave the value stored but
    /// dormant. Setting it to `false` keeps Triad's custom scalar/fallback policy.
    /// Inference's native half ladder uses its own dispatch and is unaffected.
    /// This permission does not enable vendor TF32 and does not implicitly
    /// select the stream-K half policy.
    pub fn set_bi_tensor_cores(&self, on: bool) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.bi_tensor_cores.set(on);
    }

    /// Compatibility adapter for the pre-0.7 fast-GEMM flag.
    ///
    /// New code should call [`Self::set_gemm_mode`]. `true` selects
    /// [`GemmMode::CublasFast`]. `false` leaves Deterministic unchanged and
    /// selects [`GemmMode::CublasPedantic`] from either cuBLAS mode. A failed
    /// transition panics because this legacy signature cannot return the error.
    #[deprecated(since = "0.7.0", note = "use GpuCtx::set_gemm_mode")]
    pub fn set_fast_gemm(&self, on: bool) {
        let mode = self.gemm_mode().legacy_fast_gemm(on);
        self.set_gemm_mode(mode)
            .unwrap_or_else(|error| panic!("set_fast_gemm migration failed: {error}"));
    }

    /// Return whether [`GemmMode::CublasFast`] is selected.
    pub fn fast_gemm(&self) -> bool {
        self.gemm_mode().fast_gemm()
    }

    /// Record that a CUDA graph was captured on this context.
    pub(crate) fn note_graph_capture(&self) {
        self.graphs_captured.set(self.graphs_captured.get() + 1);
    }

    /// Compact view of the three boolean GEMM controls.
    ///
    /// Use [`Self::gemm_route`] when complete policy, artifact, compiler, and
    /// device identity is required.
    pub fn gemm_flags(&self) -> (bool, bool, bool) {
        (
            self.batch_invariant(),
            self.bi_tensor_cores.get(),
            self.fast_gemm(),
        )
    }

    pub(crate) fn begin_gemm_route_recording(
        &self,
        capacity: usize,
    ) -> Result<GemmRouteRecordingGuard<'_>, String> {
        self.ensure_gemm_usable()?;
        let mut active = self
            .gemm_route_recorder
            .try_borrow_mut()
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())?;
        if active.is_some() {
            return Err("nested GEMM route recording is not supported".into());
        }
        let route_capacity = u32::try_from(capacity)
            .map_err(|_| "GEMM route recording capacity exceeds u32::MAX".to_string())?;
        let mut routes = Vec::new();
        routes
            .try_reserve_exact(capacity)
            .map_err(|error| format!("reserve GEMM route recording capacity: {error}"))?;
        let recorder = GemmRouteRecorder {
            context: self.gemm_route(),
            mode: GemmRouteRecorderMode::FixedCapture { route_capacity },
            routes,
        };
        *active = Some(recorder);
        Ok(GemmRouteRecordingGuard { ctx: self })
    }

    fn begin_eager_gemm_route_recording(&self) -> Result<GemmRouteRecordingGuard<'_>, String> {
        self.ensure_gemm_usable()?;
        let mut active = self
            .gemm_route_recorder
            .try_borrow_mut()
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())?;
        if active.is_some() {
            return Err("nested GEMM route recording is not supported".into());
        }
        *active = Some(GemmRouteRecorder {
            context: self.gemm_route(),
            mode: GemmRouteRecorderMode::GrowableEager,
            routes: Vec::new(),
        });
        Ok(GemmRouteRecordingGuard { ctx: self })
    }

    /// Executes one ordinary eager body and records the exact ordered physical
    /// GEMM routes it launches.
    ///
    /// This diagnostic API runs outside CUDA Graph capture, so lazy launch
    /// preparation and cache allocation remain legal. The body really enqueues
    /// its kernels once; callers must not include this recording pass in timing
    /// samples or treat it as graph capture.
    pub fn record_eager_gemm_trace<F>(&self, body: F) -> Result<RecordedGemmTrace, String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        let recording = self.begin_eager_gemm_route_recording()?;
        body()?;
        recording.finish_trace()
    }

    /// Run one ordinary eager body and return its capture manifest.
    pub fn record_eager_gemm_manifest<F>(
        &self,
        body: F,
    ) -> Result<PreparedGemmCaptureManifest, String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        self.record_eager_gemm_trace(body)
            .map(|trace| trace.manifest())
    }

    pub(crate) fn with_f32_prepared_launches<T>(
        &self,
        access: impl FnOnce(&mut F32PreparedLaunchCache) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut launches = self
            .f32_prepared_launches
            .try_borrow_mut()
            .map_err(|_| "prepared f32 Triad cache is already borrowed".to_string())?;
        access(&mut launches)
    }

    pub(crate) fn with_sm120_prepared_launches<T>(
        &self,
        access: impl FnOnce(&mut Sm120PreparedLaunchCache) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut launches = self
            .sm120_prepared_launches
            .try_borrow_mut()
            .map_err(|_| "prepared SM120 TMA cache is already borrowed".to_string())?;
        access(&mut launches)
    }

    pub(crate) fn with_sm90a_prepared_launches<T>(
        &self,
        access: impl FnOnce(&mut Sm90aPreparedLaunchCache) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut launches = self
            .sm90a_prepared_launches
            .try_borrow_mut()
            .map_err(|_| "prepared SM90a WGMMA cache is already borrowed".to_string())?;
        access(&mut launches)
    }

    pub(crate) fn with_sm100_prepared_launches<T>(
        &self,
        access: impl FnOnce(&mut Sm100PreparedLaunchCache) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut launches = self
            .sm100_prepared_launches
            .try_borrow_mut()
            .map_err(|_| "prepared SM100 TCGEN cache is already borrowed".to_string())?;
        access(&mut launches)
    }

    /// Whether this context currently inventories pointer-bound GEMM routes.
    /// A conflicting borrow is an error, never an inactive-recorder verdict.
    pub(crate) fn gemm_route_recording_active(&self) -> Result<bool, String> {
        self.gemm_route_recorder
            .try_borrow()
            .map(|recorder| recorder.is_some())
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())
    }

    pub(crate) fn record_resolved_gemm_route(
        &self,
        route: ResolvedGemmRoute,
    ) -> Result<(), String> {
        let mut active = self
            .gemm_route_recorder
            .try_borrow_mut()
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())?;
        let Some(recorder) = active.as_mut() else {
            return Ok(());
        };
        if let GemmRouteRecorderMode::FixedCapture { route_capacity } = recorder.mode {
            let route_count = u32::try_from(recorder.routes.len())
                .expect("fixed GEMM route count is bounded by u32 capacity");
            if route_count >= route_capacity {
                return Err(format!(
                    "GEMM route recording exceeded its capacity {route_capacity}"
                ));
            }
        }
        recorder.routes.push(route);
        Ok(())
    }

    fn take_gemm_route_recording(&self) -> Result<GemmRouteRecorder, String> {
        self.gemm_route_recorder
            .try_borrow_mut()
            .map_err(|_| "GEMM route recorder is already borrowed".to_string())?
            .take()
            .ok_or_else(|| "GEMM route recorder is not active".to_string())
    }

    fn clear_gemm_route_recording(&self) {
        if let Ok(mut active) = self.gemm_route_recorder.try_borrow_mut() {
            *active = None;
        }
    }

    fn live_gemm_module_binding(
        &self,
        module_kind: ModuleKind,
    ) -> Option<(ArtifactIdentity, CompilerIdentity)> {
        let artifacts = self.kernels.artifact_set_identity();
        match module_kind {
            ModuleKind::Fixed => Some((artifacts.fixed, self.kernels.compiler_identity())),
            ModuleKind::TriadScalar => Some((
                artifacts.triad_scalar,
                self.kernels.triad_scalar_compiler_identity(),
            )),
            ModuleKind::TriadSm80 => Some((
                self.kernels.artifact_set_identity().triad_sm80,
                self.kernels.triad_sm80_compiler_identity(),
            )),
            ModuleKind::TriadSm89Finalist => self
                .kernels
                .artifact_set_identity()
                .specialized
                .filter(|artifact| artifact.module_kind == module_kind)
                .zip(self.kernels.triad_sm89_finalist_compiler_identity()),
            ModuleKind::TriadSm89Half => artifacts
                .sm89_half
                .zip(self.kernels.triad_sm89_half_compiler_identity()),
            ModuleKind::TriadSm89ExactF32 => artifacts
                .sm89_exact_f32
                .zip(self.kernels.triad_sm89_exact_f32_compiler_identity()),
            ModuleKind::TriadSm89ExactF32D128 => artifacts
                .sm89_exact_f32_d128
                .zip(self.kernels.triad_sm89_exact_f32_d128_compiler_identity()),
            ModuleKind::TriadSm89Tf32Joint => artifacts
                .sm89_tf32_joint
                .zip(self.kernels.triad_sm89_tf32_joint_compiler_identity()),
            ModuleKind::TriadSm90a | ModuleKind::TriadSm100 | ModuleKind::TriadSm120 => self
                .kernels
                .artifact_set_identity()
                .specialized
                .filter(|artifact| artifact.module_kind == module_kind)
                .zip(self.kernels.specialized_compiler_identity()),
            ModuleKind::Mamba3Combined => None,
        }
    }

    fn live_qualified_tf32_binding(
        &self,
        module_kind: ModuleKind,
    ) -> Option<super::gemm_bi_triad::Tf32QualifiedModule> {
        let availability = self.kernels.f32_triad_availability();
        match module_kind {
            ModuleKind::TriadSm80 => availability.portable,
            ModuleKind::TriadSm89Finalist => availability.finalist,
            ModuleKind::TriadSm89Half => None,
            ModuleKind::TriadSm89ExactF32 => None,
            ModuleKind::TriadSm89ExactF32D128 => None,
            ModuleKind::TriadSm89Tf32Joint => availability.joint,
            ModuleKind::TriadSm90a | ModuleKind::TriadSm100 | ModuleKind::TriadSm120 => {
                availability.specialized
            }
            _ => None,
        }
        .filter(|binding| binding.module_kind == module_kind)
    }

    pub(crate) fn validate_resolved_gemm_route(
        &self,
        route: &ResolvedGemmRoute,
        label: &str,
    ) -> Result<(), String> {
        let context = self.gemm_route();
        let inference_terminal = matches!(
            route.backend,
            PhysicalGemmBackend::InferenceScalarFmaV1
                | PhysicalGemmBackend::InferenceWmmaV1
                | PhysicalGemmBackend::InferenceMma16V1
                | PhysicalGemmBackend::InferenceSm90aWgmmaV1
                | PhysicalGemmBackend::InferenceSm100Tcgen05V1
                | PhysicalGemmBackend::InferenceMmaTf32RnaV1
                | PhysicalGemmBackend::InferenceSm120TmaFmaV1
                | PhysicalGemmBackend::InferenceSm120TmaMma16V1
                | PhysicalGemmBackend::InferenceSm120TmaMmaTf32RnaV1
                | PhysicalGemmBackend::FixedMatvecEightWarpV1
        ) || (route.backend
            == PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1
            && route.numeric_contract == ResolvedNumericContract::ScalarFmaPostDotBiasV1)
            || (context.policy.bi_gemm_family == BiGemmFamily::Inference
                && route.backend == PhysicalGemmBackend::MmaTf32RnaV1
                && route.symbol == "gemm_bi_nn_sm80_mma_tf32_v1_m128n128_bk32_s3");
        if inference_terminal {
            return self.validate_inference_terminal_route(route, label);
        }
        if context.policy.bi_gemm_family == BiGemmFamily::Inference
            && route.backend == PhysicalGemmBackend::Sm120TmaFmaExactV1
        {
            super::gemm_bi_inference::identity::validate_cached_bridge(route)?;
            if self.kernels.tf32_function(route.symbol).is_none() {
                return Err(format!(
                    "{label}: prepared exact-TMA bridge holder is not loaded"
                ));
            }
        }
        if route.backend == PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1
            && (route.numeric_contract != ResolvedNumericContract::ScalarFmaV1
                || context.policy.bi_gemm_family != BiGemmFamily::Triad
                || route.symbol != "gemm_bi_nn_fixed_sm89_f32_n64_copyplan_v1"
                || self
                    .kernels
                    .inference_terminal_function(route.symbol)
                    .is_none())
        {
            return Err(format!(
                "{label}: Fixed copy-plan backend/family/numeric binding changed"
            ));
        }
        if !context.policy.batch_invariant || !context.backend_set.contains(BackendSet::TRIAD) {
            return Err(format!(
                "{label}: captured Triad backend is unavailable under the live GEMM policy"
            ));
        }
        let required_contract = match route.numeric_contract {
            ResolvedNumericContract::ScalarFmaPostDotBiasV1
            | ResolvedNumericContract::WmmaF32PostDotBiasV1
            | ResolvedNumericContract::ScalarFmaEightWarpTreePostDotBiasV1 => {
                return Err(format!(
                    "{label}: Inference numeric contract has an incompatible backend"
                ));
            }
            ResolvedNumericContract::ScalarFmaV1
            | ResolvedNumericContract::ScalarFmaSplitKPartialV1
            | ResolvedNumericContract::ScalarFmaSplitKF32ReduceV1
            | ResolvedNumericContract::ScalarFmaTnNarrowSplitMPartialV1
            | ResolvedNumericContract::ScalarFmaTnNarrowSplitMF64ReduceV1
            | ResolvedNumericContract::ScalarFmaTnSplitMF64ReduceV1
            | ResolvedNumericContract::ScalarFmaTnSplitMPartialV1
            | ResolvedNumericContract::ScalarFmaFixedSplitFoldV1
            | ResolvedNumericContract::ZeroReductionEpilogueF32V1 => {
                NumericContractSet::TRIAD_SCALAR_FMA_V1
            }
            ResolvedNumericContract::MmaSyncF32V1
            | ResolvedNumericContract::WgmmaF32V1
            | ResolvedNumericContract::Tcgen05F32V1 => NumericContractSet::TRIAD_MMA_SYNC_V1,
            ResolvedNumericContract::MmaSyncF32StreamKFixedOrderV1 => {
                NumericContractSet::TRIAD_MMA_SYNC_STREAM_K_V1
            }
            ResolvedNumericContract::MmaTf32RnaV1
            | ResolvedNumericContract::MmaTf32PreRnaAV1
            | ResolvedNumericContract::MmaTf32AddHalfUlpV1
            | ResolvedNumericContract::Tf32RnaPreprocessV1
            | ResolvedNumericContract::Sm90aWgmmaTf32TmaV1
            | ResolvedNumericContract::Sm100Tcgen05Tf32TmaV1
            | ResolvedNumericContract::Sm120TmaMmaTf32RnaV1 => {
                NumericContractSet::TRIAD_DETERMINISTIC_TF32_V1
            }
            ResolvedNumericContract::MmaTf32RnaSplitK2V1
            | ResolvedNumericContract::MmaTf32RnaSplitK4V1
            | ResolvedNumericContract::MmaTf32RnaSplitK8V1
            | ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1 => {
                NumericContractSet::TRIAD_DETERMINISTIC_TF32_SPLIT_K_V1
            }
        };
        if !context.numeric_contracts.contains(required_contract) {
            return Err(format!(
                "{label}: captured Triad numeric contract is unavailable under the live GEMM policy"
            ));
        }
        let expected_module = expected_route_module(route.backend);
        if route.module_kind != expected_module || route.artifact.module_kind != expected_module {
            return Err(format!(
                "{label}: captured physical backend no longer matches its module binding"
            ));
        }
        let uses_qualified_tf32_module = matches!(
            route.numeric_contract,
            super::kernel_identity::ResolvedNumericContract::MmaTf32RnaV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32PreRnaAV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32AddHalfUlpV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK2V1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK4V1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK8V1
                | super::kernel_identity::ResolvedNumericContract::Sm90aWgmmaTf32TmaV1
                | super::kernel_identity::ResolvedNumericContract::Sm100Tcgen05Tf32TmaV1
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaV1
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1
                | super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32V1
        ) && route.module_kind != ModuleKind::TriadScalar
            || route.backend == super::kernel_identity::PhysicalGemmBackend::Sm120TmaFmaExactV1;
        if uses_qualified_tf32_module {
            let binding = self
                .live_qualified_tf32_binding(route.module_kind)
                .ok_or_else(|| format!("{label}: captured qualified TF32 module is not loaded"))?;
            if route.artifact != binding.artifact
                || route.compiler != binding.compiler
                || route.target != binding.target
                || route.device != binding.device
                || route.device_caps != binding.device_caps
            {
                return Err(format!(
                    "{label}: captured GEMM route no longer matches its qualified TF32 binding"
                ));
            }
        } else {
            let (artifact, compiler) = self
                .live_gemm_module_binding(route.module_kind)
                .ok_or_else(|| format!("{label}: captured GEMM module is not loaded"))?;
            if route.artifact != artifact
                || route.compiler != compiler
                || route.target != compiler.target
                || route.device != context.device
                || route.device_caps != context.device_caps
            {
                return Err(format!(
                    "{label}: captured GEMM route no longer matches its live module binding"
                ));
            }
        }
        if route.tuning_table_revision
            != expected_route_tuning_revision(route.backend, context.tuning_table_revision)
            || route.schedule_revision
                != expected_route_schedule_revision(route.backend, context.schedule_set_revision)
        {
            return Err(format!("{label}: captured GEMM route revision is stale"));
        }
        let tf32_numeric = matches!(
            route.numeric_contract,
            super::kernel_identity::ResolvedNumericContract::MmaTf32RnaV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32PreRnaAV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32AddHalfUlpV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK2V1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK4V1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK8V1
                | super::kernel_identity::ResolvedNumericContract::Sm90aWgmmaTf32TmaV1
                | super::kernel_identity::ResolvedNumericContract::Sm100Tcgen05Tf32TmaV1
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaV1
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1
        );
        if tf32_numeric
            && (route.dtype != PolicyDtype::F32
                || self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32V1)
        {
            return Err(format!(
                "{label}: captured deterministic TF32 route is disabled by the live policy"
            ));
        }
        if route.dtype == PolicyDtype::F32
            && !tf32_numeric
            && route.numeric_contract
                != super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32V1
            && !scalar_backend_supports_logical_f32(route.backend)
        {
            return Err(format!(
                "{label}: captured logical-f32 route has an incompatible physical backend"
            ));
        }
        if route.numeric_contract
            == super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32V1
            && (route.dtype != PolicyDtype::F32
                || route.instruction_family
                    != super::kernel_identity::ResolvedInstructionFamily::ScalarFma
                || route.instruction_shape
                    != super::kernel_identity::ResolvedInstructionShape { m: 1, n: 1, k: 1 }
                || route.operand_conversion
                    != super::kernel_identity::ResolvedOperandConversion::None
                || match route.op {
                    super::kernel_identity::ResolvedGemmOp::Nn => route.shape.1 != 0,
                    super::kernel_identity::ResolvedGemmOp::Tn => route.shape.0 != 0,
                    super::kernel_identity::ResolvedGemmOp::Nt => route.shape.2 != 0,
                })
        {
            return Err(format!(
                "{label}: captured zero-reduction epilogue identity is inconsistent"
            ));
        }
        if route.dtype != PolicyDtype::F32
            && route.backend != PhysicalGemmBackend::ScalarFmaV1
            && !self.bi_tensor_cores()
        {
            return Err(format!(
                "{label}: captured typed Tensor Core route is disabled by the live policy"
            ));
        }
        Ok(())
    }

    fn validate_inference_terminal_route(
        &self,
        route: &ResolvedGemmRoute,
        label: &str,
    ) -> Result<(), String> {
        let context = self.gemm_route();
        let spec = super::gemm_bi_inference::identity::terminal(route.symbol)
            .ok_or_else(|| format!("{label}: unknown Inference terminal"))?;
        spec.validate_route(route)?;
        let required = spec.required_contract(context.policy.bi_gemm_family)?;
        let backend = if route.module_kind == ModuleKind::Fixed {
            BackendSet::FIXED
        } else {
            BackendSet::TRIAD
        };
        if !context.policy.batch_invariant
            || !context.backend_set.contains(backend)
            || !context.numeric_contracts.contains(required)
            || (required == NumericContractSet::FIXED_DETERMINISTIC_TF32_V1
                && self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32V1)
        {
            return Err(format!(
                "{label}: Inference terminal is disabled by the live policy"
            ));
        }
        let (artifact, compiler) = self
            .live_gemm_module_binding(route.module_kind)
            .ok_or_else(|| format!("{label}: Inference terminal module is not loaded"))?;
        if route.artifact != artifact
            || route.compiler != compiler
            || route.target != compiler.target
            || route.device != context.device
            || route.device_caps != context.device_caps
            || self
                .kernels
                .inference_terminal_function(route.symbol)
                .is_none()
        {
            return Err(format!(
                "{label}: Inference terminal live function/module binding changed"
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_resolved_input_transform(
        &self,
        symbol: &'static str,
        transform: &super::kernel_identity::ResolvedInputTransform,
        label: &str,
    ) -> Result<(), String> {
        use super::kernel_identity::{ResolvedOperandConversion, ResolvedTransformOutputOwnership};
        if self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32V1
            || transform.numeric_contract != ResolvedNumericContract::Tf32RnaPreprocessV1
            || transform.operand_conversion != ResolvedOperandConversion::RegisterCvtRnaTf32F32V1
            || transform.output_ownership
                != ResolvedTransformOutputOwnership::PreparedScratchAllocationV1
            || symbol != super::gemm_bi_triad::TN_PRE_RNA_TRANSPOSE_SYMBOL
        {
            return Err(format!(
                "{label}: captured input transform contract is unavailable"
            ));
        }
        let binding = self
            .live_qualified_tf32_binding(ModuleKind::TriadSm89Tf32Joint)
            .ok_or_else(|| format!("{label}: captured input transform module is not loaded"))?;
        if transform.artifact != binding.artifact
            || transform.compiler != binding.compiler
            || transform.target != binding.target
            || transform.device != binding.device
            || transform.device_caps != binding.device_caps
            || transform.tuning_table_revision
                != super::gemm_bi_triad::SM89_TF32_JOINT_TUNING_REVISION
            || transform.schedule_revision != super::kernel_identity::SCHEDULE_REVISION
            || transform.resources_digest == [0; 32]
            || self
                .kernels
                .triad_sm89_tf32_joint_function(symbol)
                .is_none()
        {
            return Err(format!(
                "{label}: captured input transform no longer matches its live module or resources"
            ));
        }
        Ok(())
    }

    /// Return the GEMM policy used by eager launches and graph guards.
    ///
    /// The compatibility booleans are derived from [`Self::gemm_mode`]. The
    /// deterministic family, tensor-core permission, and f32/half policies are
    /// stored separately so vendor-mode round trips preserve them.
    pub fn gemm_policy(&self) -> GemmPolicy {
        let (bi, tc, fast) = self.gemm_flags();
        let family = self.bi_gemm_family.get();
        GemmPolicy {
            batch_invariant: bi,
            bi_tensor_cores: tc,
            fast_gemm: fast,
            cublas_tf32: self.tf32(),
            f32_triad_policy: self.f32_triad_policy.get(),
            half_triad_policy: self.half_triad_policy.get(),
            bi_gemm_family: family,
        }
    }

    /// Complete route identity used by eager launches and graph guards.
    pub fn gemm_route(&self) -> GemmRoute {
        let policy = self.gemm_policy();
        let (backend_set, numeric_contracts) =
            super::kernel_identity::route_backend_contract_sets(policy);
        GemmRouteIdentity {
            policy,
            backend_set,
            numeric_contracts,
            compiler: self.kernels.compiler_identity(),
            artifacts: self.kernels.artifact_set_identity(),
            policy_revision: super::kernel_identity::POLICY_REVISION,
            policy_hash: self.policy_hash,
            device: self.device_identity,
            device_caps: self.device_caps,
            tuning_table_revision: super::kernel_identity::TUNING_TABLE_REVISION,
            schedule_set_revision: super::kernel_identity::SCHEDULE_REVISION,
            state_capacity: u32::try_from(self.state_cap)
                .expect("validated state capacity fits in u32"),
        }
    }

    /// Return the requested deterministic tensor-core permission.
    ///
    /// The value affects dispatch only in [`GemmMode::Deterministic`]; use
    /// [`Self::gemm_policy`] for the complete effective configuration.
    pub fn bi_tensor_cores(&self) -> bool {
        self.bi_tensor_cores.get()
    }

    /// Select the numeric policy used by deterministic f32 Triad GEMMs.
    ///
    /// [`F32TriadPolicy::ExactScalarFmaV1`] is the default. The deterministic
    /// TF32 permission is independent of vendor TF32 and is dormant in cuBLAS
    /// modes.
    pub fn set_f32_triad_policy(&self, policy: F32TriadPolicy) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.f32_triad_policy.set(policy);
    }

    /// Return the stored deterministic f32 Triad policy.
    pub fn f32_triad_policy(&self) -> F32TriadPolicy {
        self.f32_triad_policy.get()
    }

    /// Select the numeric policy used by deterministic half-precision Triad GEMMs.
    ///
    /// [`HalfTriadPolicy::TiledParityV1`] is the default. Stream-K must be
    /// selected explicitly and requires tensor-core permission when a mode
    /// transition or environment configuration is validated.
    pub fn set_half_triad_policy(&self, policy: HalfTriadPolicy) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.half_triad_policy.set(policy);
    }

    /// Return the stored deterministic half-precision Triad policy.
    pub fn half_triad_policy(&self) -> HalfTriadPolicy {
        self.half_triad_policy.get()
    }

    /// Compatibility adapter that removes vendor TF32.
    ///
    /// New code should call [`Self::set_gemm_mode`]. In
    /// [`GemmMode::CublasFast`] this selects [`GemmMode::CublasPedantic`]. It
    /// leaves Deterministic and CublasPedantic unchanged. In particular it does
    /// not change [`F32TriadPolicy::AllowDeterministicTf32V1`], which controls
    /// custom deterministic TF32 rather than vendor TF32. A failed transition
    /// panics because this legacy signature cannot return the error.
    #[deprecated(since = "0.7.0", note = "use GpuCtx::set_gemm_mode")]
    pub fn disable_tf32(&self) {
        let mode = self.gemm_mode().legacy_disable_tf32();
        self.set_gemm_mode(mode)
            .unwrap_or_else(|error| panic!("disable_tf32 migration failed: {error}"));
    }

    /// Return whether vendor TF32 handle math is selected.
    ///
    /// This is true only in [`GemmMode::CublasFast`]. It is independent of
    /// [`Self::f32_triad_policy`]: custom deterministic TF32 permission does
    /// not enable vendor TF32, and vendor TF32 does not change the custom
    /// policy.
    pub fn tf32(&self) -> bool {
        self.gemm_mode().tf32()
    }

    /// The state capacity this context's kernels were compiled with.
    pub fn state_cap(&self) -> usize {
        self.state_cap
    }

    pub(in crate::mamba_ssm::gpu) fn compute_capability(&self) -> (u32, u32) {
        self.device_identity.compute_capability
    }

    /// Presize the batch-invariant GEMM scratch buffers, including Split-K
    /// partials and W-transpose staging, OUTSIDE any CUDA graph capture. cudarc's
    /// alloc is cuMemAllocAsync where the device has memory pools, so a
    /// first-use allocation on a CAPTURING stream becomes a graph memory
    /// node - legal, silent, and the OnceLock then caches a graph-owned
    /// VA that later eager launches dereference. Call before capturing
    /// any graph that may run batch-invariant GEMMs; a no-op when the
    /// buffers already exist or the flag is off.
    pub fn presize_bi_scratch(&self) -> Result<(), String> {
        if self.batch_invariant() {
            self.kernels.splitk_scratch_buf(&self.stream)?;
            self.kernels.transpose_scratch_buf(&self.stream)?;
        }
        Ok(())
    }

    /// Pre-size the half-precision staging buffer for a known engine
    /// config, batch, and dtype. Eliminates lazy-grow during the hot path
    /// — critical for CUDA Graph capture safety: if a captured graph baked
    /// a staging pointer and a later call grew the buffer, the freed
    /// allocation would be dereferenced on replay (CUDA_ERROR_ILLEGAL_ADDRESS
    /// or silent corruption). Sizes for the worst-case step-time GEMM
    /// operand (in_proj input = batch × d_model, the largest staging consumer
    /// in step_kernels). Idempotent — safe to call multiple times.
    pub fn presize_half_staging_for_step(
        &self,
        cfg: &MambaConfig,
        batch: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let max_in_elems = batch * dm.max(di).max(dt_rank);
        let bytes = max_in_elems * dtype.size_bytes();
        self.ensure_half_staging(bytes)
    }

    /// Presize the half-staging buffer for a Mamba-3 training step.
    /// Same rationale as [`Self::presize_half_staging_for_train`] but for
    /// the M3 weight set whose `in_proj_out_dim` differs from M1.
    pub fn presize_half_staging_for_train_m3(
        &self,
        cfg: &crate::mamba3_siso::config::Mamba3Config,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let ip = cfg.in_proj_out_dim();
        let max_dim = dm.max(di).max(ip);
        let bytes = batch * seq_len * max_dim * dtype.size_bytes();
        self.ensure_half_staging(bytes)
    }

    /// Presize the half-staging buffer for a training step (forward + backward).
    /// Uses (batch * seq_len) instead of just batch — training operates on the
    /// full sequence, not T=1. Critical for CUDA Graph capture: a lazy grow
    /// during the captured body would bake a freed pointer into the graph,
    /// causing CUDA_ERROR_ILLEGAL_ADDRESS on replay.
    pub fn presize_half_staging_for_train(
        &self,
        cfg: &MambaConfig,
        batch: usize,
        seq_len: usize,
        dtype: WeightDtype,
    ) -> Result<(), String> {
        if matches!(dtype, WeightDtype::F32) {
            return Ok(());
        }
        let dm = cfg.d_model;
        let di = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let max_dim = dm.max(di).max(dt_rank);
        let bytes = batch * seq_len * max_dim * dtype.size_bytes();
        self.ensure_half_staging(bytes)
    }

    /// Ensure the half-precision staging buffer is at least `bytes` in size.
    /// In the steady state this is a no-op when `presize_half_staging_for_step`
    /// was called at engine construction; the lazy grow path remains as a
    /// fallback for prefill (which runs outside any captured graph) or for
    /// callers that don't presize.
    pub fn ensure_half_staging(&self, bytes: usize) -> Result<(), String> {
        let mut cur = self.half_staging_bytes.borrow_mut();
        if *cur >= bytes {
            return Ok(());
        }
        if self.graph_scratch_frozen.get() {
            return Err(
                "half-precision staging cannot grow after CUDA graph capture; destroy the \
                 context or pre-size the largest shape before capture"
                    .into(),
            );
        }
        // Grow by at least the requested size, rounded up to a 4 KiB page —
        // no speculative doubling that wastes memory at the plateau (the old
        // `bytes.max(*cur * 2)` rule could leave us at 4× the actual need
        // after a few growths).
        let page = 4096;
        let new_size = bytes.div_ceil(page) * page;
        let buf = self
            .stream
            .alloc_zeros::<u8>(new_size)
            .map_err(|e| format!("half_staging alloc {new_size}B failed: {e:?}"))?;
        let ptr = {
            use cudarc::driver::DevicePtr;
            let (p, _g) = buf.device_ptr(&self.stream);
            p
        };
        *self.half_staging.borrow_mut() = Some(buf);
        *self.half_staging_ptr.borrow_mut() = ptr;
        *cur = new_size;
        Ok(())
    }

    pub fn half_staging_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        *self.half_staging_ptr.borrow()
    }
}

fn expected_route_schedule_revision(backend: PhysicalGemmBackend, generic: u16) -> u16 {
    if backend == PhysicalGemmBackend::Sm120TmaMma16V1 {
        super::gemm_bi_triad::SM120_SCHEDULE_REVISION
    } else {
        generic
    }
}

const fn expected_route_module(backend: PhysicalGemmBackend) -> ModuleKind {
    match backend {
        PhysicalGemmBackend::ScalarFmaV1
        | PhysicalGemmBackend::ScalarFmaSplitKPartialV1
        | PhysicalGemmBackend::ScalarFmaSplitKF32ReduceV1
        | PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartialV1
        | PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1 => ModuleKind::TriadScalar,
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1
        | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1 => {
            ModuleKind::TriadSm89ExactF32
        }
        PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89V1 => ModuleKind::TriadSm89ExactF32D128,
        PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1
        | PhysicalGemmBackend::InferenceScalarFmaV1
        | PhysicalGemmBackend::InferenceWmmaV1
        | PhysicalGemmBackend::InferenceMma16V1
        | PhysicalGemmBackend::InferenceSm90aWgmmaV1
        | PhysicalGemmBackend::InferenceSm100Tcgen05V1
        | PhysicalGemmBackend::InferenceMmaTf32RnaV1
        | PhysicalGemmBackend::InferenceSm120TmaFmaV1
        | PhysicalGemmBackend::InferenceSm120TmaMma16V1
        | PhysicalGemmBackend::InferenceSm120TmaMmaTf32RnaV1
        | PhysicalGemmBackend::FixedMatvecEightWarpV1 => ModuleKind::Fixed,
        PhysicalGemmBackend::Sm80Mma16V1
        | PhysicalGemmBackend::MmaTf32RnaV1
        | PhysicalGemmBackend::MmaTf32RnaSplitK2V1
        | PhysicalGemmBackend::MmaTf32RnaSplitK4V1
        | PhysicalGemmBackend::MmaTf32RnaSplitK8V1 => ModuleKind::TriadSm80,
        PhysicalGemmBackend::Sm89MmaTf32Compact8V1 => ModuleKind::TriadSm89Finalist,
        PhysicalGemmBackend::Sm89MmaTf32PreRnaV1
        | PhysicalGemmBackend::Sm89MmaTf32AddHalfV1
        | PhysicalGemmBackend::Sm89MmaTf32NtALdmatrixV1 => ModuleKind::TriadSm89Tf32Joint,
        PhysicalGemmBackend::Sm89Mma16HalfS3V1 | PhysicalGemmBackend::Sm89Mma16HalfS2V1 => {
            ModuleKind::TriadSm89Half
        }
        PhysicalGemmBackend::Sm90aWgmmaV1 | PhysicalGemmBackend::Sm90aWgmmaTf32TmaV1 => {
            ModuleKind::TriadSm90a
        }
        PhysicalGemmBackend::Sm100Tcgen05V1 | PhysicalGemmBackend::Sm100Tcgen05Tf32TmaV1 => {
            ModuleKind::TriadSm100
        }
        PhysicalGemmBackend::Sm120TmaMma16V1
        | PhysicalGemmBackend::Sm120TmaMmaTf32RnaV1
        | PhysicalGemmBackend::Sm120TmaMmaTf32RnaStreamKV1
        | PhysicalGemmBackend::Sm120TmaFmaExactV1 => ModuleKind::TriadSm120,
    }
}

fn expected_route_tuning_revision(backend: PhysicalGemmBackend, generic: u16) -> u16 {
    match backend {
        PhysicalGemmBackend::Sm89MmaTf32Compact8V1 => {
            super::gemm_bi_triad::SM89_FINALIST_TUNING_REVISION
        }
        PhysicalGemmBackend::Sm89MmaTf32PreRnaV1
        | PhysicalGemmBackend::Sm89MmaTf32AddHalfV1
        | PhysicalGemmBackend::Sm89MmaTf32NtALdmatrixV1 => {
            super::gemm_bi_triad::SM89_TF32_JOINT_TUNING_REVISION
        }
        PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1 => {
            super::kernel_identity::SM89_FIXED_COPYPLAN_ROUTE_REVISION
        }
        PhysicalGemmBackend::Sm89Mma16HalfS3V1 | PhysicalGemmBackend::Sm89Mma16HalfS2V1 => {
            super::kernel_identity::SM89_HALF_ROUTE_REVISION
        }
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1
        | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1 => {
            super::kernel_identity::SM89_EXACT_F32_TN_ROUTE_REVISION
        }
        PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89V1 => {
            super::kernel_identity::SM89_EXACT_F32_D128_ROUTE_REVISION
        }
        _ => generic,
    }
}

const fn scalar_backend_supports_logical_f32(backend: PhysicalGemmBackend) -> bool {
    matches!(
        backend,
        PhysicalGemmBackend::ScalarFmaV1
            | PhysicalGemmBackend::ScalarFmaSplitKPartialV1
            | PhysicalGemmBackend::ScalarFmaSplitKF32ReduceV1
            | PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartialV1
            | PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1
            | PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1
            | PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1
            | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1
            | PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89V1
            | PhysicalGemmBackend::Sm120TmaFmaExactV1
    )
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "needs a CUDA device"]
    fn inference_recorder_query_rejects_conflicting_borrow() {
        let device = crate::mamba_ssm::gpu::device::GpuDevice::new(0).unwrap();
        let ctx = super::GpuCtx::new(&device).unwrap();
        assert!(!ctx.gemm_route_recording_active().unwrap());
        let borrow = ctx.gemm_route_recorder.borrow_mut();
        assert!(ctx.gemm_route_recording_active().is_err());
        drop(borrow);
        ctx.record_eager_gemm_trace(|| {
            assert!(ctx.gemm_route_recording_active()?);
            Ok(())
        })
        .unwrap();
        assert!(!ctx.gemm_route_recording_active().unwrap());
    }

    use super::{
        BiGemmFamily, F32TriadPolicy, GemmEnvValues, GemmMode, HalfTriadPolicy,
        bi_gemm_family_from_result, expected_route_module, expected_route_schedule_revision,
        expected_route_tuning_revision, f32_triad_policy_from_result, m1_mixed_graph_max_dim,
        resolve_gemm_env, scalar_backend_supports_logical_f32, tier_flag_from_result,
        validate_multiprocessor_identity,
    };
    use crate::config::ScanMode;
    use crate::mamba_ssm::gpu::forward::GpuMambaDims;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalGemmBackend, SCHEDULE_REVISION,
    };
    #[cfg(unix)]
    use std::ffi::OsString;

    #[test]
    fn mixed_graph_scratch_covers_the_two_inner_projection_output() {
        let dims = GpuMambaDims {
            batch: 1,
            d_model: 64,
            d_inner: 129,
            d_state: 8,
            d_conv: 4,
            dt_rank: 4,
            xdbl_dim: 20,
            seq_len: 3,
            mamba_input_dim: 17,
            n_layers: 1,
            scan_mode: ScanMode::Sequential,
            rms_norm_eps: 1e-5,
        };
        assert_eq!(m1_mixed_graph_max_dim(&dims), 258);
    }

    #[test]
    fn device_and_loaded_kernel_multiprocessor_counts_must_match() {
        assert!(validate_multiprocessor_identity(142, 142).is_ok());
        for counts in [(0, 142), (142, 0), (108, 142)] {
            let error = validate_multiprocessor_identity(counts.0, counts.1)
                .expect_err("incoherent CUDA topology must be rejected");
            assert!(error.contains("multiprocessor"), "{error}");
        }
    }

    #[test]
    fn sm120_mma16_graph_routes_use_their_sealed_schedule_revision() {
        assert_eq!(
            expected_route_schedule_revision(
                PhysicalGemmBackend::Sm120TmaMma16V1,
                SCHEDULE_REVISION,
            ),
            super::super::gemm_bi_triad::SM120_SCHEDULE_REVISION
        );
        assert_eq!(
            expected_route_schedule_revision(
                PhysicalGemmBackend::Sm120TmaMmaTf32RnaV1,
                SCHEDULE_REVISION,
            ),
            SCHEDULE_REVISION
        );
    }

    #[test]
    fn sm89_finalist_routes_use_their_private_tuning_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        let finalist =
            expected_route_tuning_revision(PhysicalGemmBackend::Sm89MmaTf32Compact8V1, generic);
        assert_eq!(
            finalist,
            super::super::gemm_bi_triad::SM89_FINALIST_TUNING_REVISION
        );
        assert_ne!(finalist, 0);
        assert_ne!(finalist, 1);

        let portable = expected_route_tuning_revision(PhysicalGemmBackend::MmaTf32RnaV1, generic);
        assert_eq!(portable, 45);
        assert_ne!(portable, finalist);
    }

    #[test]
    fn sm89_tf32_joint_backends_use_the_joint_module_and_private_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        for backend in [
            PhysicalGemmBackend::Sm89MmaTf32PreRnaV1,
            PhysicalGemmBackend::Sm89MmaTf32AddHalfV1,
            PhysicalGemmBackend::Sm89MmaTf32NtALdmatrixV1,
        ] {
            assert_eq!(
                expected_route_module(backend),
                ModuleKind::TriadSm89Tf32Joint
            );
            assert_eq!(
                expected_route_tuning_revision(backend, generic),
                super::super::gemm_bi_triad::SM89_TF32_JOINT_TUNING_REVISION
            );
        }
        assert_eq!(generic, 45);
    }

    #[test]
    fn sm89_fixed_copyplan_routes_use_a_private_revision_without_moving_global_45() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        let copyplan = expected_route_tuning_revision(
            PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1,
            generic,
        );
        assert_eq!(
            copyplan,
            crate::mamba_ssm::gpu::kernel_identity::SM89_FIXED_COPYPLAN_ROUTE_REVISION
        );
        assert_ne!(copyplan, 0);
        assert_ne!(copyplan, 2);
        assert_eq!(
            expected_route_tuning_revision(PhysicalGemmBackend::ScalarFmaV1, generic),
            45
        );
        assert_eq!(
            expected_route_module(PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1),
            ModuleKind::Fixed
        );
        assert_eq!(
            expected_route_module(PhysicalGemmBackend::ScalarFmaV1),
            ModuleKind::TriadScalar
        );
    }

    #[test]
    fn sm89_half_routes_use_the_isolated_module_and_private_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        for backend in [
            PhysicalGemmBackend::Sm89Mma16HalfS3V1,
            PhysicalGemmBackend::Sm89Mma16HalfS2V1,
        ] {
            assert_eq!(
                expected_route_tuning_revision(backend, generic),
                crate::mamba_ssm::gpu::kernel_identity::SM89_HALF_ROUTE_REVISION
            );
            assert_eq!(expected_route_module(backend), ModuleKind::TriadSm89Half);
        }
        assert_eq!(generic, 45);
    }

    #[test]
    fn sm89_exact_f32_tn_routes_use_the_isolated_module_and_private_revision() {
        for backend in [
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1,
        ] {
            assert_eq!(
                expected_route_module(backend),
                ModuleKind::TriadSm89ExactF32
            );
            assert_eq!(
                expected_route_tuning_revision(backend, 45),
                crate::mamba_ssm::gpu::kernel_identity::SM89_EXACT_F32_TN_ROUTE_REVISION
            );
            assert!(scalar_backend_supports_logical_f32(backend));
        }
    }

    #[test]
    fn sm89_exact_f32_d128_routes_use_their_isolated_module_and_private_revision() {
        let backend = PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89V1;
        assert_eq!(
            expected_route_module(backend),
            ModuleKind::TriadSm89ExactF32D128
        );
        assert_eq!(
            expected_route_tuning_revision(backend, 45),
            crate::mamba_ssm::gpu::kernel_identity::SM89_EXACT_F32_D128_ROUTE_REVISION
        );
        assert!(scalar_backend_supports_logical_f32(backend));
    }

    #[test]
    fn logical_f32_accepts_only_scalar_triad_backends() {
        for backend in [
            PhysicalGemmBackend::ScalarFmaV1,
            PhysicalGemmBackend::ScalarFmaSplitKPartialV1,
            PhysicalGemmBackend::ScalarFmaSplitKF32ReduceV1,
            PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartialV1,
            PhysicalGemmBackend::ScalarFmaTnSplitMF64ReduceV1,
            PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlanV1,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFusedV1,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartialV1,
            PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89V1,
            PhysicalGemmBackend::Sm120TmaFmaExactV1,
        ] {
            assert!(scalar_backend_supports_logical_f32(backend), "{backend:?}");
        }
        for backend in [
            PhysicalGemmBackend::Sm80Mma16V1,
            PhysicalGemmBackend::MmaTf32RnaV1,
            PhysicalGemmBackend::Sm120TmaMma16V1,
        ] {
            assert!(!scalar_backend_supports_logical_f32(backend), "{backend:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn f32_triad_policy_environment_is_strict_and_defaults_to_exact() {
        use std::os::unix::ffi::OsStringExt as _;

        assert_eq!(
            f32_triad_policy_from_result(Err(std::env::VarError::NotPresent)).unwrap(),
            F32TriadPolicy::ExactScalarFmaV1
        );
        assert_eq!(
            f32_triad_policy_from_result(Ok("tf32".into())).unwrap(),
            F32TriadPolicy::AllowDeterministicTf32V1
        );
        let error = f32_triad_policy_from_result(Err(std::env::VarError::NotUnicode(
            OsString::from_vec(vec![b't', b'f', 0xff, b'3', b'2']),
        )))
        .expect_err("non-Unicode policy must fail");
        assert!(error.contains("MAMBA_RS_BI_F32_POLICY"), "{error}");
        assert!(error.contains("exact") && error.contains("tf32"), "{error}");
    }

    #[test]
    fn tier_flag_environment_is_strict_and_fail_closed() {
        assert!(!tier_flag_from_result("TEST_FLAG", Err(std::env::VarError::NotPresent)).unwrap());
        for value in ["1", " true ", "YES", "on"] {
            assert!(tier_flag_from_result("TEST_FLAG", Ok(value.into())).unwrap());
        }
        for value in ["0", " false ", "NO", "off", ""] {
            assert!(!tier_flag_from_result("TEST_FLAG", Ok(value.into())).unwrap());
        }
        let error = tier_flag_from_result("TEST_FLAG", Ok("enabled".into()))
            .expect_err("unknown flag values must fail");
        assert!(error.contains("TEST_FLAG"), "{error}");

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;

            let error = tier_flag_from_result(
                "TEST_FLAG",
                Err(std::env::VarError::NotUnicode(OsString::from_vec(vec![
                    b't', b'r', 0xff, b'u', b'e',
                ]))),
            )
            .expect_err("non-Unicode flag values must fail");
            assert!(error.contains("TEST_FLAG"), "{error}");
            assert!(error.contains("not valid Unicode"), "{error}");
        }
    }

    #[test]
    fn bi_gemm_family_environment_accepts_only_semantic_family_names() {
        assert_eq!(
            bi_gemm_family_from_result(
                Err(std::env::VarError::NotPresent),
                BiGemmFamily::Inference,
            )
            .unwrap(),
            BiGemmFamily::Inference
        );
        for value in ["", " \t\n", "triad", "  TrIaD\t"] {
            assert_eq!(
                bi_gemm_family_from_result(Ok(value.into()), BiGemmFamily::Inference).unwrap(),
                BiGemmFamily::Triad,
                "{value:?}"
            );
        }
        for value in ["fixed", "\nFiXeD ", "inference", " \nInFeReNcE\t"] {
            assert_eq!(
                bi_gemm_family_from_result(Ok(value.into()), BiGemmFamily::Triad).unwrap(),
                BiGemmFamily::Inference,
                "{value:?}"
            );
        }
        for value in ["gemm_bi", "batch_invariant", "warptile", "wmma", "other"] {
            let error = bi_gemm_family_from_result(Ok(value.into()), BiGemmFamily::Triad)
                .expect_err("unknown family names must fail");
            assert!(error.contains("MAMBA_RS_BI_GEMM_FAMILY"), "{error}");
            assert!(error.contains("only inference or triad"), "{error}");
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt as _;

            let error = bi_gemm_family_from_result(
                Err(std::env::VarError::NotUnicode(OsString::from_vec(vec![
                    b't', b'r', 0xff, b'i', b'a', b'd',
                ]))),
                BiGemmFamily::Triad,
            )
            .expect_err("non-Unicode family names must fail");
            assert!(error.contains("MAMBA_RS_BI_GEMM_FAMILY"), "{error}");
            assert!(error.contains("only inference or triad"), "{error}");
        }
    }

    fn absent_env() -> Result<String, std::env::VarError> {
        Err(std::env::VarError::NotPresent)
    }

    fn empty_gemm_env() -> GemmEnvValues {
        GemmEnvValues {
            mode: absent_env(),
            batch_invariant: absent_env(),
            fast_gemm: absent_env(),
            tensor_cores: absent_env(),
            f32_policy: absent_env(),
            half_policy: absent_env(),
            family: absent_env(),
            arch_rung: absent_env(),
        }
    }

    #[test]
    fn gemm_mode_environment_resolves_canonical_and_legacy_tables() {
        for (mode, expected) in [
            ("deterministic", GemmMode::Deterministic),
            ("cublas-fast", GemmMode::CublasFast),
            ("cublas-pedantic", GemmMode::CublasPedantic),
        ] {
            let mut values = empty_gemm_env();
            values.mode = Ok(format!(" \t{mode}\n"));
            assert_eq!(
                resolve_gemm_env(values, BiGemmFamily::Triad).unwrap().mode,
                expected
            );
        }

        for (batch_invariant, fast_gemm, expected) in [
            (None, None, GemmMode::Deterministic),
            (Some(true), None, GemmMode::Deterministic),
            (Some(true), Some(false), GemmMode::Deterministic),
            (Some(false), None, GemmMode::CublasPedantic),
            (Some(false), Some(false), GemmMode::CublasPedantic),
            (None, Some(false), GemmMode::CublasPedantic),
            (None, Some(true), GemmMode::CublasFast),
            (Some(false), Some(true), GemmMode::CublasFast),
        ] {
            let mut values = empty_gemm_env();
            values.batch_invariant = batch_invariant
                .map(|value| Ok(value.to_string()))
                .unwrap_or_else(absent_env);
            values.fast_gemm = fast_gemm
                .map(|value| Ok(value.to_string()))
                .unwrap_or_else(absent_env);
            assert_eq!(
                resolve_gemm_env(values, BiGemmFamily::Triad).unwrap().mode,
                expected,
                "BI={batch_invariant:?} FAST={fast_gemm:?}"
            );
        }
    }

    #[test]
    fn gemm_mode_environment_rejects_conflicts_and_vendor_custom_controls() {
        for legacy_name in ["batch", "fast"] {
            let mut values = empty_gemm_env();
            values.mode = Ok("deterministic".into());
            if legacy_name == "batch" {
                values.batch_invariant = Ok("false".into());
            } else {
                values.fast_gemm = Ok("false".into());
            }
            let error = resolve_gemm_env(values, BiGemmFamily::Triad).unwrap_err();
            assert!(error.contains("MAMBA_RS_GEMM_MODE"), "{error}");
            assert!(error.contains("MAMBA_RS_BATCH_INVARIANT"), "{error}");
            assert!(error.contains("MAMBA_RS_FAST_GEMM"), "{error}");
        }

        let mut values = empty_gemm_env();
        values.batch_invariant = Ok("true".into());
        values.fast_gemm = Ok("true".into());
        assert!(resolve_gemm_env(values, BiGemmFamily::Triad).is_err());

        for custom in ["tensor", "f32", "half", "family"] {
            let mut values = empty_gemm_env();
            values.mode = Ok("cublas-fast".into());
            match custom {
                "tensor" => values.tensor_cores = Ok("false".into()),
                "f32" => values.f32_policy = Ok("exact".into()),
                "half" => values.half_policy = Ok("tiled".into()),
                "family" => values.family = Ok("triad".into()),
                _ => unreachable!(),
            }
            let error = resolve_gemm_env(values, BiGemmFamily::Triad).unwrap_err();
            assert!(error.contains("deterministic GEMM controls"), "{error}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn gemm_mode_environment_rejects_non_unicode_canonical_value() {
        use std::os::unix::ffi::OsStringExt as _;

        let mut values = empty_gemm_env();
        values.mode = Err(std::env::VarError::NotUnicode(OsString::from_vec(vec![
            b'f', b'a', 0xff, b's', b't',
        ])));
        let error = resolve_gemm_env(values, BiGemmFamily::Triad).unwrap_err();
        assert!(error.contains("MAMBA_RS_GEMM_MODE"), "{error}");
        assert!(error.contains("not valid Unicode"), "{error}");
    }

    #[test]
    fn deterministic_environment_defaults_and_validates_custom_policy() {
        let resolved = resolve_gemm_env(empty_gemm_env(), BiGemmFamily::Inference).unwrap();
        assert_eq!(resolved.mode, GemmMode::Deterministic);
        assert!(resolved.tensor_cores);
        assert_eq!(resolved.family, BiGemmFamily::Inference);
        assert_eq!(resolved.f32_policy, F32TriadPolicy::ExactScalarFmaV1);
        assert_eq!(resolved.half_policy, HalfTriadPolicy::TiledParityV1);

        let mut values = empty_gemm_env();
        values.tensor_cores = Ok("false".into());
        values.half_policy = Ok("streamk".into());
        let error = resolve_gemm_env(values, BiGemmFamily::Triad).unwrap_err();
        assert!(error.contains("MAMBA_RS_BI_HALF_POLICY=streamk"), "{error}");
        assert!(error.contains("MAMBA_RS_BI_TENSOR_CORES=1"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn half_triad_policy_environment_is_strict_and_absent_means_unset() {
        use std::os::unix::ffi::OsStringExt as _;

        assert_eq!(
            super::half_triad_policy_from_result(Err(std::env::VarError::NotPresent)).unwrap(),
            None
        );
        assert_eq!(
            super::half_triad_policy_from_result(Ok(" tiled ".into())).unwrap(),
            Some(super::HalfTriadPolicy::TiledParityV1)
        );
        assert_eq!(
            super::half_triad_policy_from_result(Ok("streamk".into())).unwrap(),
            Some(super::HalfTriadPolicy::AllowStreamKFixedOrderV1)
        );
        for wrong in ["", "stream-k", "StreamK", "1", "on"] {
            let error = super::half_triad_policy_from_result(Ok(wrong.into()))
                .expect_err("unknown half policy spelling must fail");
            assert!(error.contains("MAMBA_RS_BI_HALF_POLICY"), "{error}");
            assert!(
                error.contains("tiled") && error.contains("streamk"),
                "{error}"
            );
        }
        let error = super::half_triad_policy_from_result(Err(std::env::VarError::NotUnicode(
            OsString::from_vec(vec![b's', b't', 0xff, b'k']),
        )))
        .expect_err("non-Unicode policy must fail");
        assert!(error.contains("MAMBA_RS_BI_HALF_POLICY"), "{error}");
        assert!(
            error.contains("tiled") && error.contains("streamk"),
            "{error}"
        );
    }

    #[test]
    fn arch_rung_flag_knows_only_off() {
        assert!(super::validate_arch_rung_flag(Err(std::env::VarError::NotPresent)).is_ok());
        for value in ["", "off", " OFF\n"] {
            assert!(
                super::validate_arch_rung_flag(Ok(value.into())).is_ok(),
                "{value:?}"
            );
        }
        for value in ["on", "1", "0", "disabled"] {
            let error = super::validate_arch_rung_flag(Ok(value.into())).unwrap_err();
            assert!(error.contains("MAMBA_RS_ARCH_RUNG"), "{error}");
        }
    }
}
