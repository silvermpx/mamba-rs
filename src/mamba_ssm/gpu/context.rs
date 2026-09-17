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
}

impl GemmEnvValues {
    fn read() -> Self {
        Self {
            mode: std::env::var("MAMBA_RS_GEMM_MODE"),
        }
    }
}

/// What a context serves and how its f32 products round. The constructor
/// that knows the role sets it once: a model context serves the Inference
/// family, a trainer or a plain context the Triad family, and the storage
/// precision the caller asked for decides whether f32 products stay exact
/// or take the deterministic TF32 kernels where one is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GemmRole {
    pub(crate) family: BiGemmFamily,
    pub(crate) f32_numeric: F32TriadPolicy,
}

impl GemmRole {
    pub(crate) fn triad(dtype: WeightDtype) -> Self {
        Self {
            family: BiGemmFamily::Triad,
            f32_numeric: dtype.f32_numeric(),
        }
    }

    pub(crate) fn inference(dtype: WeightDtype) -> Self {
        Self {
            family: BiGemmFamily::Inference,
            f32_numeric: dtype.f32_numeric(),
        }
    }
}

impl WeightDtype {
    /// The f32 numeric contract a storage precision asks for: `Tf32` permits
    /// the deterministic TF32 kernels, everything else keeps every f32
    /// product exact.
    pub(crate) fn f32_numeric(self) -> F32TriadPolicy {
        match self {
            Self::Tf32 => F32TriadPolicy::AllowDeterministicTf32,
            Self::F32 | Self::Bf16 | Self::F16 => F32TriadPolicy::ExactScalarFma,
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

fn explicit_gemm_config(mode: GemmMode, role: GemmRole) -> ResolvedGemmEnv {
    ResolvedGemmEnv {
        mode,
        family: role.family,
        tensor_cores: true,
        f32_policy: role.f32_numeric,
        half_policy: HalfTriadPolicy::AllowStreamKFixedOrder,
    }
}

/// The mode `MAMBA_RS_GEMM_MODE` names; unset means the default,
/// [`GemmMode::Deterministic`].
fn gemm_mode_from_env(mode: Result<String, std::env::VarError>) -> Result<GemmMode, String> {
    match mode {
        Ok(value) => GemmMode::parse_env_value(&value),
        Err(std::env::VarError::NotPresent) => Ok(GemmMode::Deterministic),
        Err(std::env::VarError::NotUnicode(value)) => Err(format!(
            "MAMBA_RS_GEMM_MODE={value:?} is not valid Unicode \
             (use deterministic, cublas-fast, or cublas-pedantic)"
        )),
    }
}

fn resolve_gemm_env(values: GemmEnvValues, role: GemmRole) -> Result<ResolvedGemmEnv, String> {
    Ok(explicit_gemm_config(gemm_mode_from_env(values.mode)?, role))
}

/// Which deterministic GEMM family serves the forward. Both are
/// deterministic; they are named for the STRUCTURE that produces their
/// differing guarantee, not for a tensor-core tier (both instantiate on
/// Tensor Cores for bf16/f16 and on CUDA cores for f32). The context's role
/// chooses: a model context serves Inference, a trainer Triad. Not a user
/// setting; the qualification instruments reach it through
/// [`GpuCtx::route_controls`].
#[doc(hidden)]
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

/// Numeric policy for deterministic f32 GEMMs, set by the storage precision
/// the context was built for ([`WeightDtype::Tf32`] permits TF32). Not a
/// user setting; the qualification instruments reach it through
/// [`GpuCtx::route_controls`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum F32TriadPolicy {
    /// Preserve the scalar `__fmaf_rn` reduction contract.
    #[default]
    ExactScalarFma = 0,
    /// Permit frozen and qualified deterministic TF32 routes.
    ///
    /// This is permission, not a forced backend. Unsupported or unmeasured
    /// cells keep the exact scalar FMA contract.
    AllowDeterministicTf32 = 1,
}

/// Numeric policy for deterministic half-precision (bf16/f16) Triad GEMMs.
/// Every context permits the measured stream-K weight-gradient routes; the
/// shape gates decide per call. Not a user setting; the qualification
/// instruments reach it through [`GpuCtx::route_controls`].
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HalfTriadPolicy {
    /// Every automatic half route reproduces the forced portable tensor-core
    /// kernel bit for bit: one owner CTA per output tile, one reduction order.
    #[default]
    TiledParity = 0,
    /// Permit the measured stream-K half routes: a persistent grid folds
    /// per-CTA partials in a fixed order that differs from the tiled
    /// reduction, while repeated eager and graph launches of the same route
    /// stay bit-identical.
    ///
    /// This is permission, not a forced schedule. Unmeasured shapes, and
    /// shapes whose tile grid already fills the device, keep the tiled
    /// contract.
    AllowStreamKFixedOrder = 1,
}

pub use super::kernel_identity::GemmRouteIdentity;
#[doc(hidden)]
pub use super::kernel_identity::{BackendSet as GemmBackendSet, GemmPolicy, NumericContractSet};

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
        CapturedGemmGraphPlan::new(recorder.context, launches, routes)
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
        CapturedGemmGraphPlan::new(recorder.context, launches, routes).map(Some)
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
    /// The tensor-core tier of the deterministic typed GEMMs, granted to
    /// every context; the shape and device gates decide per call. Its
    /// numeric contract is its own: mma.sync f32 accumulation differs from
    /// the scalar __fmaf_rn chain, so outputs do not bit-match the scalar
    /// triad, but the TC kernels are fully deterministic and
    /// batch-invariant for their admitted shapes. The qualification
    /// instruments turn it off to measure the scalar tier.
    bi_tensor_cores: std::cell::Cell<bool>,
    /// Which deterministic family serves the forward: the context's role
    /// (a model serves Inference, a trainer Triad). Part of the numeric
    /// route, so it rides [`GpuCtx::gemm_route`] into every capture identity.
    bi_gemm_family: std::cell::Cell<BiGemmFamily>,
    /// The f32 numeric contract the storage precision asked for.
    f32_triad_policy: std::cell::Cell<F32TriadPolicy>,
    /// The half-precision numeric contract: stream-K permitted.
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
    /// The context starts in [`GemmMode::Deterministic`] and serves the
    /// Triad kernels with exact f32 products, the role of a plain context.
    /// This constructor ignores `MAMBA_RS_GEMM_MODE`; use
    /// [`Self::new_from_env`] to read it. The kernel state capacity is 64.
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

    /// Create a context in the mode `MAMBA_RS_GEMM_MODE` names.
    ///
    /// The variable accepts the canonical names returned by
    /// [`GemmMode::as_str`] (`deterministic`, `cublas-fast`,
    /// `cublas-pedantic`); unset means [`GemmMode::Deterministic`]. It is
    /// the only variable the crate reads for its GEMMs. The value is
    /// resolved before GPU resources are constructed; an invalid or
    /// non-Unicode value returns an error.
    pub fn new_from_env(device: &GpuDevice) -> Result<Self, String> {
        Self::new_from_env_with_state_cap_and_role(device, 64, GemmRole::triad(WeightDtype::F32))
    }

    /// [`Self::new_from_env`] with an explicit kernel state capacity.
    ///
    /// The environment is resolved before construction. `state_cap` is passed
    /// to kernel compilation and is returned by [`Self::state_cap`].
    pub fn new_from_env_with_state_cap(
        device: &GpuDevice,
        state_cap: usize,
    ) -> Result<Self, String> {
        Self::new_from_env_with_state_cap_and_role(
            device,
            state_cap,
            GemmRole::triad(WeightDtype::F32),
        )
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
    /// GEMM dispatch and cuBLAS numeric settings. `MAMBA_RS_GEMM_MODE` is
    /// ignored. Invalid resource or cuBLAS setup returns an error.
    pub fn new_with_state_cap_and_mode(
        device: &GpuDevice,
        state_cap: usize,
        mode: GemmMode,
    ) -> Result<Self, String> {
        Self::new_with_state_cap_mode_and_role(
            device,
            state_cap,
            mode,
            GemmRole::triad(WeightDtype::F32),
        )
    }

    pub(crate) fn new_from_env_with_state_cap_and_role(
        device: &GpuDevice,
        state_cap: usize,
        role: GemmRole,
    ) -> Result<Self, String> {
        let config = resolve_gemm_env(GemmEnvValues::read(), role)?;
        Self::new_with_state_cap_and_config(device, state_cap, config)
    }

    pub(crate) fn new_with_state_cap_mode_and_role(
        device: &GpuDevice,
        state_cap: usize,
        mode: GemmMode,
        role: GemmRole,
    ) -> Result<Self, String> {
        Self::new_with_state_cap_and_config(device, state_cap, explicit_gemm_config(mode, role))
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
        // A stored stream-K permission is dormant while tensor cores are
        // off: the half tier then runs its scalar policy and never reaches
        // the stream-K routes, so the mode change has nothing to refuse. The
        // pair is still refused when it is requested from the environment or
        // the explicit configuration.
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

    /// The context's GEMM mode.
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

    #[cfg(test)]
    pub(crate) fn poison_gemm_for_test(&self) {
        *self.gemm_unusable.borrow_mut() = Some("test: GPU GEMM context is unusable".into());
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

    /// Whether [`GemmMode::Deterministic`] is selected.
    pub(crate) fn batch_invariant(&self) -> bool {
        self.gemm_mode().batch_invariant()
    }

    /// Choose which custom family serves deterministic forward GEMMs.
    ///
    /// The value is dormant in either cuBLAS mode and is preserved across
    /// mode changes. See [`BiGemmFamily`] for the available custom routes.
    pub(crate) fn set_bi_gemm_family(&self, family: BiGemmFamily) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.bi_gemm_family.set(family);
    }

    /// The deterministic family currently selected.
    pub(crate) fn bi_gemm_family(&self) -> BiGemmFamily {
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
    pub(crate) fn set_bi_tensor_cores(&self, on: bool) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.bi_tensor_cores.set(on);
    }

    /// Whether [`GemmMode::CublasFast`] is selected.
    pub(crate) fn fast_gemm(&self) -> bool {
        self.gemm_mode().fast_gemm()
    }

    /// Record that a CUDA graph was captured on this context.
    pub(crate) fn note_graph_capture(&self) {
        self.graphs_captured.set(self.graphs_captured.get() + 1);
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
        self.validate_resolved_gemm_route_in(&self.gemm_route(), route, label)
    }

    /// The same validation against a route identity the caller already
    /// holds, so a graph replay computes the live identity once for all of
    /// its routes.
    pub(crate) fn validate_resolved_gemm_route_in(
        &self,
        context: &GemmRouteIdentity,
        route: &ResolvedGemmRoute,
        label: &str,
    ) -> Result<(), String> {
        let context = *context;
        let inference_terminal = matches!(
            route.backend,
            PhysicalGemmBackend::InferenceScalarFma
                | PhysicalGemmBackend::InferenceWmma
                | PhysicalGemmBackend::InferenceMma16
                | PhysicalGemmBackend::InferenceSm90aWgmma
                | PhysicalGemmBackend::InferenceSm100Tcgen05
                | PhysicalGemmBackend::InferenceMmaTf32Rna
                | PhysicalGemmBackend::InferenceSm120TmaFma
                | PhysicalGemmBackend::InferenceSm120TmaMma16
                | PhysicalGemmBackend::InferenceSm120TmaMmaTf32Rna
                | PhysicalGemmBackend::FixedMatvecEightWarp
        ) || (route.backend
            == PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan
            && route.numeric_contract == ResolvedNumericContract::ScalarFmaPostDotBias)
            || (context.policy.bi_gemm_family == BiGemmFamily::Inference
                && route.backend == PhysicalGemmBackend::MmaTf32Rna
                && route.symbol == "nn_sm80_mma_tf32_m128n128_bk32_s3");
        if inference_terminal {
            return self.validate_inference_terminal_route(&context, route, label);
        }
        if context.policy.bi_gemm_family == BiGemmFamily::Inference
            && route.backend == PhysicalGemmBackend::Sm120TmaFmaExact
        {
            super::gemm_bi_inference::identity::validate_cached_bridge(route)?;
            if self.kernels.tf32_function(route.symbol).is_none() {
                return Err(format!(
                    "{label}: prepared exact-TMA bridge holder is not loaded"
                ));
            }
        }
        if route.backend == PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan
            && (route.numeric_contract != ResolvedNumericContract::ScalarFma
                || context.policy.bi_gemm_family != BiGemmFamily::Triad
                || route.symbol != "nn_sm89_f32_n64_copyplan"
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
            ResolvedNumericContract::ScalarFmaPostDotBias
            | ResolvedNumericContract::WmmaF32PostDotBias
            | ResolvedNumericContract::ScalarFmaEightWarpTreePostDotBias => {
                return Err(format!(
                    "{label}: Inference numeric contract has an incompatible backend"
                ));
            }
            ResolvedNumericContract::ScalarFma
            | ResolvedNumericContract::ScalarFmaSplitKPartial
            | ResolvedNumericContract::ScalarFmaSplitKF32Reduce
            | ResolvedNumericContract::ScalarFmaTnNarrowSplitMPartial
            | ResolvedNumericContract::ScalarFmaTnNarrowSplitMF64Reduce
            | ResolvedNumericContract::ScalarFmaTnSplitMF64Reduce
            | ResolvedNumericContract::ScalarFmaTnSplitMPartial
            | ResolvedNumericContract::ScalarFmaFixedSplitFold
            | ResolvedNumericContract::ZeroReductionEpilogueF32 => {
                NumericContractSet::TRIAD_SCALAR_FMA
            }
            ResolvedNumericContract::MmaSyncF32
            | ResolvedNumericContract::WgmmaF32
            | ResolvedNumericContract::Tcgen05F32 => NumericContractSet::TRIAD_MMA_SYNC,
            ResolvedNumericContract::MmaSyncF32StreamKFixedOrder => {
                NumericContractSet::TRIAD_MMA_SYNC_STREAM_K
            }
            ResolvedNumericContract::MmaTf32Rna
            | ResolvedNumericContract::MmaTf32PreRnaAV1
            | ResolvedNumericContract::MmaTf32AddHalfUlp
            | ResolvedNumericContract::Tf32RnaPreprocess
            | ResolvedNumericContract::Sm90aWgmmaTf32Tma
            | ResolvedNumericContract::Sm100Tcgen05Tf32Tma
            | ResolvedNumericContract::Sm120TmaMmaTf32Rna => {
                NumericContractSet::TRIAD_DETERMINISTIC_TF32
            }
            ResolvedNumericContract::MmaTf32RnaSplitK2
            | ResolvedNumericContract::MmaTf32RnaSplitK4
            | ResolvedNumericContract::MmaTf32RnaSplitK8
            | ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1 => {
                NumericContractSet::TRIAD_DETERMINISTIC_TF32_SPLIT_K
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
            super::kernel_identity::ResolvedNumericContract::MmaTf32Rna
                | super::kernel_identity::ResolvedNumericContract::MmaTf32PreRnaAV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32AddHalfUlp
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK2
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK4
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK8
                | super::kernel_identity::ResolvedNumericContract::Sm90aWgmmaTf32Tma
                | super::kernel_identity::ResolvedNumericContract::Sm100Tcgen05Tf32Tma
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32Rna
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1
                | super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32
        ) && route.module_kind != ModuleKind::TriadScalar
            || route.backend == super::kernel_identity::PhysicalGemmBackend::Sm120TmaFmaExact;
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
            super::kernel_identity::ResolvedNumericContract::MmaTf32Rna
                | super::kernel_identity::ResolvedNumericContract::MmaTf32PreRnaAV1
                | super::kernel_identity::ResolvedNumericContract::MmaTf32AddHalfUlp
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK2
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK4
                | super::kernel_identity::ResolvedNumericContract::MmaTf32RnaSplitK8
                | super::kernel_identity::ResolvedNumericContract::Sm90aWgmmaTf32Tma
                | super::kernel_identity::ResolvedNumericContract::Sm100Tcgen05Tf32Tma
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32Rna
                | super::kernel_identity::ResolvedNumericContract::Sm120TmaMmaTf32RnaStreamKV1
        );
        if tf32_numeric
            && (route.dtype != PolicyDtype::F32
                || self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32)
        {
            return Err(format!(
                "{label}: captured deterministic TF32 route is disabled by the live policy"
            ));
        }
        if route.dtype == PolicyDtype::F32
            && !tf32_numeric
            && route.numeric_contract
                != super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32
            && !scalar_backend_supports_logical_f32(route.backend)
        {
            return Err(format!(
                "{label}: captured logical-f32 route has an incompatible physical backend"
            ));
        }
        if route.numeric_contract
            == super::kernel_identity::ResolvedNumericContract::ZeroReductionEpilogueF32
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
            && route.backend != PhysicalGemmBackend::ScalarFma
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
        context: &GemmRouteIdentity,
        route: &ResolvedGemmRoute,
        label: &str,
    ) -> Result<(), String> {
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
            || (required == NumericContractSet::FIXED_DETERMINISTIC_TF32
                && self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32)
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
        if self.f32_triad_policy() != F32TriadPolicy::AllowDeterministicTf32
            || transform.numeric_contract != ResolvedNumericContract::Tf32RnaPreprocess
            || transform.operand_conversion != ResolvedOperandConversion::RegisterCvtRnaTf32F32
            || transform.output_ownership
                != ResolvedTransformOutputOwnership::PreparedScratchAllocation
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

    /// The GEMM policy used by eager launches and graph guards: the mode's
    /// booleans and the role's family and numeric contracts.
    pub(crate) fn gemm_policy(&self) -> GemmPolicy {
        GemmPolicy {
            batch_invariant: self.batch_invariant(),
            bi_tensor_cores: self.bi_tensor_cores.get(),
            fast_gemm: self.fast_gemm(),
            cublas_tf32: self.tf32(),
            f32_triad_policy: self.f32_triad_policy.get(),
            half_triad_policy: self.half_triad_policy.get(),
            bi_gemm_family: self.bi_gemm_family.get(),
        }
    }

    /// The complete route identity: mode, storage precision and role, the
    /// kernels selected, compiler and artifact identity, device identity.
    /// Eager launches record it and graph replays compare it; a program
    /// pins a numeric route by keeping this value.
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

    /// The deterministic tensor-core permission; every context grants it,
    /// the shape and device gates decide per call.
    pub(crate) fn bi_tensor_cores(&self) -> bool {
        self.bi_tensor_cores.get()
    }

    /// Select the numeric policy used by deterministic f32 Triad GEMMs.
    ///
    /// The constructor sets it from the storage precision
    /// ([`WeightDtype::Tf32`] permits TF32). The deterministic TF32
    /// permission is independent of vendor TF32 and is dormant in cuBLAS
    /// modes.
    pub(crate) fn set_f32_triad_policy(&self, policy: F32TriadPolicy) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.f32_triad_policy.set(policy);
    }

    /// The deterministic f32 Triad policy.
    pub(crate) fn f32_triad_policy(&self) -> F32TriadPolicy {
        self.f32_triad_policy.get()
    }

    /// The storage precision an f32 engine on this context was built for:
    /// `Tf32` when its products may take the deterministic TF32 kernels,
    /// `F32` otherwise.
    pub(crate) fn f32_storage_dtype(&self) -> WeightDtype {
        match self.f32_triad_policy.get() {
            F32TriadPolicy::AllowDeterministicTf32 => WeightDtype::Tf32,
            F32TriadPolicy::ExactScalarFma => WeightDtype::F32,
        }
    }

    /// Select the numeric policy used by deterministic half-precision Triad GEMMs.
    ///
    /// [`HalfTriadPolicy::AllowStreamKFixedOrder`] is the default. The
    /// stream-K routes live in the tensor-core tier, so the permission is
    /// dormant while tensor cores are off.
    pub(crate) fn set_half_triad_policy(&self, policy: HalfTriadPolicy) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.half_triad_policy.set(policy);
    }

    /// The deterministic half-precision Triad policy.
    pub(crate) fn half_triad_policy(&self) -> HalfTriadPolicy {
        self.half_triad_policy.get()
    }

    /// Whether vendor TF32 handle math is selected: true only in
    /// [`GemmMode::CublasFast`]. Independent of [`WeightDtype::Tf32`], the
    /// crate's own deterministic TF32.
    pub(crate) fn tf32(&self) -> bool {
        self.gemm_mode().tf32()
    }

    /// The route controls of this context, for the qualification
    /// instruments and the tests that pin one tier against another. Every
    /// value here is derived by the constructor from the mode, the storage
    /// precision and the context's role; a program has no reason to touch
    /// it, and it is not part of the documented API.
    #[doc(hidden)]
    pub fn route_controls(&self) -> RouteControls<'_> {
        RouteControls { ctx: self }
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
    if backend == PhysicalGemmBackend::Sm120TmaMma16 {
        super::gemm_bi_triad::SM120_SCHEDULE_REVISION
    } else {
        generic
    }
}

const fn expected_route_module(backend: PhysicalGemmBackend) -> ModuleKind {
    match backend {
        PhysicalGemmBackend::ScalarFma
        | PhysicalGemmBackend::ScalarFmaSplitKPartial
        | PhysicalGemmBackend::ScalarFmaSplitKF32Reduce
        | PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartial
        | PhysicalGemmBackend::ScalarFmaTnSplitMF64Reduce => ModuleKind::TriadScalar,
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFused
        | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartial => {
            ModuleKind::TriadSm89ExactF32
        }
        PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89 => ModuleKind::TriadSm89ExactF32D128,
        PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan
        | PhysicalGemmBackend::InferenceScalarFma
        | PhysicalGemmBackend::InferenceWmma
        | PhysicalGemmBackend::InferenceMma16
        | PhysicalGemmBackend::InferenceSm90aWgmma
        | PhysicalGemmBackend::InferenceSm100Tcgen05
        | PhysicalGemmBackend::InferenceMmaTf32Rna
        | PhysicalGemmBackend::InferenceSm120TmaFma
        | PhysicalGemmBackend::InferenceSm120TmaMma16
        | PhysicalGemmBackend::InferenceSm120TmaMmaTf32Rna
        | PhysicalGemmBackend::FixedMatvecEightWarp => ModuleKind::Fixed,
        PhysicalGemmBackend::Sm80Mma16
        | PhysicalGemmBackend::MmaTf32Rna
        | PhysicalGemmBackend::MmaTf32RnaSplitK2
        | PhysicalGemmBackend::MmaTf32RnaSplitK4
        | PhysicalGemmBackend::MmaTf32RnaSplitK8 => ModuleKind::TriadSm80,
        PhysicalGemmBackend::Sm89MmaTf32Compact8 => ModuleKind::TriadSm89Finalist,
        PhysicalGemmBackend::Sm89MmaTf32PreRna
        | PhysicalGemmBackend::Sm89MmaTf32AddHalf
        | PhysicalGemmBackend::Sm89MmaTf32NtALdmatrix => ModuleKind::TriadSm89Tf32Joint,
        PhysicalGemmBackend::Sm89Mma16HalfS3 | PhysicalGemmBackend::Sm89Mma16HalfS2 => {
            ModuleKind::TriadSm89Half
        }
        PhysicalGemmBackend::Sm90aWgmma | PhysicalGemmBackend::Sm90aWgmmaTf32Tma => {
            ModuleKind::TriadSm90a
        }
        PhysicalGemmBackend::Sm100Tcgen05 | PhysicalGemmBackend::Sm100Tcgen05Tf32Tma => {
            ModuleKind::TriadSm100
        }
        PhysicalGemmBackend::Sm120TmaMma16
        | PhysicalGemmBackend::Sm120TmaMmaTf32Rna
        | PhysicalGemmBackend::Sm120TmaMmaTf32RnaStreamKV1
        | PhysicalGemmBackend::Sm120TmaFmaExact => ModuleKind::TriadSm120,
    }
}

fn expected_route_tuning_revision(backend: PhysicalGemmBackend, generic: u16) -> u16 {
    match backend {
        PhysicalGemmBackend::Sm89MmaTf32Compact8 => {
            super::gemm_bi_triad::SM89_FINALIST_TUNING_REVISION
        }
        PhysicalGemmBackend::Sm89MmaTf32PreRna
        | PhysicalGemmBackend::Sm89MmaTf32AddHalf
        | PhysicalGemmBackend::Sm89MmaTf32NtALdmatrix => {
            super::gemm_bi_triad::SM89_TF32_JOINT_TUNING_REVISION
        }
        PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan => {
            super::kernel_identity::SM89_FIXED_COPYPLAN_ROUTE_REVISION
        }
        PhysicalGemmBackend::Sm89Mma16HalfS3 | PhysicalGemmBackend::Sm89Mma16HalfS2 => {
            super::kernel_identity::SM89_HALF_ROUTE_REVISION
        }
        PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFused
        | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartial => {
            super::kernel_identity::SM89_EXACT_F32_TN_ROUTE_REVISION
        }
        PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89 => {
            super::kernel_identity::SM89_EXACT_F32_D128_ROUTE_REVISION
        }
        _ => generic,
    }
}

const fn scalar_backend_supports_logical_f32(backend: PhysicalGemmBackend) -> bool {
    matches!(
        backend,
        PhysicalGemmBackend::ScalarFma
            | PhysicalGemmBackend::ScalarFmaSplitKPartial
            | PhysicalGemmBackend::ScalarFmaSplitKF32Reduce
            | PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartial
            | PhysicalGemmBackend::ScalarFmaTnSplitMF64Reduce
            | PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan
            | PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFused
            | PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartial
            | PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89
            | PhysicalGemmBackend::Sm120TmaFmaExact
    )
}

/// See [`GpuCtx::route_controls`].
#[doc(hidden)]
pub struct RouteControls<'a> {
    ctx: &'a GpuCtx,
}

#[doc(hidden)]
impl RouteControls<'_> {
    pub fn family(&self) -> BiGemmFamily {
        self.ctx.bi_gemm_family()
    }

    pub fn set_family(&self, family: BiGemmFamily) {
        self.ctx.set_bi_gemm_family(family);
    }

    pub fn tensor_cores(&self) -> bool {
        self.ctx.bi_tensor_cores()
    }

    pub fn set_tensor_cores(&self, on: bool) {
        self.ctx.set_bi_tensor_cores(on);
    }

    pub fn f32_policy(&self) -> F32TriadPolicy {
        self.ctx.f32_triad_policy()
    }

    pub fn set_f32_policy(&self, policy: F32TriadPolicy) {
        self.ctx.set_f32_triad_policy(policy);
    }

    pub fn half_policy(&self) -> HalfTriadPolicy {
        self.ctx.half_triad_policy()
    }

    pub fn set_half_policy(&self, policy: HalfTriadPolicy) {
        self.ctx.set_half_triad_policy(policy);
    }

    pub fn policy(&self) -> GemmPolicy {
        self.ctx.gemm_policy()
    }

    pub fn tf32(&self) -> bool {
        self.ctx.tf32()
    }

    pub fn batch_invariant(&self) -> bool {
        self.ctx.batch_invariant()
    }

    pub fn fast_gemm(&self) -> bool {
        self.ctx.fast_gemm()
    }
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
        BiGemmFamily, F32TriadPolicy, GemmEnvValues, GemmMode, GemmRole, HalfTriadPolicy,
        ResolvedGemmEnv, WeightDtype, expected_route_module, expected_route_schedule_revision,
        expected_route_tuning_revision, explicit_gemm_config, m1_mixed_graph_max_dim,
        resolve_gemm_env, scalar_backend_supports_logical_f32, validate_multiprocessor_identity,
    };
    use crate::config::ScanMode;
    use crate::mamba_ssm::gpu::forward::GpuMambaDims;
    use crate::mamba_ssm::gpu::kernel_identity::{
        ModuleKind, PhysicalGemmBackend, SCHEDULE_REVISION,
    };
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
            expected_route_schedule_revision(PhysicalGemmBackend::Sm120TmaMma16, SCHEDULE_REVISION,),
            super::super::gemm_bi_triad::SM120_SCHEDULE_REVISION
        );
        assert_eq!(
            expected_route_schedule_revision(
                PhysicalGemmBackend::Sm120TmaMmaTf32Rna,
                SCHEDULE_REVISION,
            ),
            SCHEDULE_REVISION
        );
    }

    #[test]
    fn sm89_finalist_routes_use_their_private_tuning_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        let finalist =
            expected_route_tuning_revision(PhysicalGemmBackend::Sm89MmaTf32Compact8, generic);
        assert_eq!(
            finalist,
            super::super::gemm_bi_triad::SM89_FINALIST_TUNING_REVISION
        );
        assert_ne!(finalist, 0);
        assert_ne!(finalist, 1);

        let portable = expected_route_tuning_revision(PhysicalGemmBackend::MmaTf32Rna, generic);
        assert_eq!(portable, 45);
        assert_ne!(portable, finalist);
    }

    #[test]
    fn sm89_tf32_joint_backends_use_the_joint_module_and_private_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        for backend in [
            PhysicalGemmBackend::Sm89MmaTf32PreRna,
            PhysicalGemmBackend::Sm89MmaTf32AddHalf,
            PhysicalGemmBackend::Sm89MmaTf32NtALdmatrix,
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
            PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan,
            generic,
        );
        assert_eq!(
            copyplan,
            crate::mamba_ssm::gpu::kernel_identity::SM89_FIXED_COPYPLAN_ROUTE_REVISION
        );
        assert_ne!(copyplan, 0);
        assert_ne!(copyplan, 2);
        assert_eq!(
            expected_route_tuning_revision(PhysicalGemmBackend::ScalarFma, generic),
            45
        );
        assert_eq!(
            expected_route_module(PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan),
            ModuleKind::Fixed
        );
        assert_eq!(
            expected_route_module(PhysicalGemmBackend::ScalarFma),
            ModuleKind::TriadScalar
        );
    }

    #[test]
    fn sm89_half_routes_use_the_isolated_module_and_private_revision() {
        let generic = super::super::gemm_bi_triad::F32_TF32_TUNING_REVISION;
        for backend in [
            PhysicalGemmBackend::Sm89Mma16HalfS3,
            PhysicalGemmBackend::Sm89Mma16HalfS2,
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
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFused,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartial,
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
        let backend = PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89;
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
            PhysicalGemmBackend::ScalarFma,
            PhysicalGemmBackend::ScalarFmaSplitKPartial,
            PhysicalGemmBackend::ScalarFmaSplitKF32Reduce,
            PhysicalGemmBackend::ScalarFmaTnNarrowSplitMPartial,
            PhysicalGemmBackend::ScalarFmaTnSplitMF64Reduce,
            PhysicalGemmBackend::ScalarFmaSm89FixedCopyPlan,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DualChunkFused,
            PhysicalGemmBackend::ScalarFmaSm89ExactF32DirectSplitMPartial,
            PhysicalGemmBackend::ScalarFmaTnDirectF64FoldSm89,
            PhysicalGemmBackend::Sm120TmaFmaExact,
        ] {
            assert!(scalar_backend_supports_logical_f32(backend), "{backend:?}");
        }
        for backend in [
            PhysicalGemmBackend::Sm80Mma16,
            PhysicalGemmBackend::MmaTf32Rna,
            PhysicalGemmBackend::Sm120TmaMma16,
        ] {
            assert!(!scalar_backend_supports_logical_f32(backend), "{backend:?}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn explicit_gemm_config_carries_mode_role_and_numeric_defaults() {
        for mode in [
            GemmMode::Deterministic,
            GemmMode::CublasFast,
            GemmMode::CublasPedantic,
        ] {
            for (dtype, f32_policy) in [
                (WeightDtype::F32, F32TriadPolicy::ExactScalarFma),
                (WeightDtype::Tf32, F32TriadPolicy::AllowDeterministicTf32),
                (WeightDtype::Bf16, F32TriadPolicy::ExactScalarFma),
                (WeightDtype::F16, F32TriadPolicy::ExactScalarFma),
            ] {
                for (role, family) in [
                    (GemmRole::inference(dtype), BiGemmFamily::Inference),
                    (GemmRole::triad(dtype), BiGemmFamily::Triad),
                ] {
                    assert_eq!(
                        explicit_gemm_config(mode, role),
                        ResolvedGemmEnv {
                            mode,
                            family,
                            tensor_cores: true,
                            f32_policy,
                            half_policy: HalfTriadPolicy::AllowStreamKFixedOrder,
                        }
                    );
                }
            }
        }
    }

    fn absent_env() -> Result<String, std::env::VarError> {
        Err(std::env::VarError::NotPresent)
    }

    fn empty_gemm_env() -> GemmEnvValues {
        GemmEnvValues { mode: absent_env() }
    }

    #[test]
    fn gemm_mode_environment_reads_one_variable() {
        for (value, expected) in [
            ("deterministic", GemmMode::Deterministic),
            ("cublas-fast", GemmMode::CublasFast),
            ("cublas-pedantic", GemmMode::CublasPedantic),
        ] {
            let values = GemmEnvValues {
                mode: Ok(value.into()),
            };
            let resolved = resolve_gemm_env(values, GemmRole::triad(WeightDtype::Tf32)).unwrap();
            assert_eq!(resolved.mode, expected);
            assert_eq!(resolved.family, BiGemmFamily::Triad);
            assert_eq!(resolved.f32_policy, F32TriadPolicy::AllowDeterministicTf32);
            assert!(resolved.tensor_cores);
            assert_eq!(
                resolved.half_policy,
                HalfTriadPolicy::AllowStreamKFixedOrder
            );
        }
        let resolved =
            resolve_gemm_env(empty_gemm_env(), GemmRole::inference(WeightDtype::F32)).unwrap();
        assert_eq!(resolved.mode, GemmMode::Deterministic);
        assert_eq!(resolved.family, BiGemmFamily::Inference);
        assert_eq!(resolved.f32_policy, F32TriadPolicy::ExactScalarFma);
        let error = resolve_gemm_env(
            GemmEnvValues {
                mode: Ok("fast".into()),
            },
            GemmRole::triad(WeightDtype::F32),
        )
        .unwrap_err();
        assert!(error.contains("MAMBA_RS_GEMM_MODE"), "{error}");
        let error = resolve_gemm_env(
            GemmEnvValues {
                mode: Err(std::env::VarError::NotUnicode(OsString::from("x"))),
            },
            GemmRole::triad(WeightDtype::F32),
        )
        .unwrap_err();
        assert!(error.contains("MAMBA_RS_GEMM_MODE"), "{error}");
    }
}
