//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::dtype::WeightDtype;
use super::kernels::MambaKernels;
use crate::config::MambaConfig;
use std::cell::RefCell;
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

/// Which batch-invariant GEMM family serves the forward while
/// [`GpuCtx::batch_invariant`] is on. Both are deterministic; they are
/// named for the STRUCTURE that produces their differing guarantee, not
/// for a role (either can serve a forward) and not for a tensor-core
/// tier (both instantiate on Tensor Cores for bf16/f16 and on CUDA cores
/// for f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum BiGemmFamily {
    /// The multi-tile dispatcher (kernels/gemm_bi_triad.cu) - the default.
    /// Carries the full triad (NN + TN + NT), so it is the only family
    /// that can serve a backward, and it is the fastest deterministic
    /// path. It picks a kernel by shape (ultra-thin M<32, narrow-N,
    /// GEMV, split-K M<=1024, Slim/Big above), so its invariance holds
    /// across every M INSIDE one bucket; crossing a boundary changes the
    /// reduction association deterministically.
    #[default]
    Triad,
    /// The single fixed tile (kernels/gemm_bi_fixed.cu):
    /// forward-only NN, batch-invariant BY CONSTRUCTION. One 64x64x32
    /// tile, SPLIT_K=1, and a K-reduction for `C[i,j]` that reads only
    /// `A[i,:]` and `B[:,j]` - there are no buckets to cross, so the
    /// output cannot depend on how many rows share the launch. bf16/f16
    /// instantiate on Tensor Cores; f32 runs the CUDA-core FMA tile (Ada
    /// Tensor Cores accept no f32 operands), the same hardware path
    /// cuBLAS takes for f32.
    Fixed,
}

pub use super::kernel_identity::{
    BackendSet as GemmBackendSet, GemmPolicy, GemmRouteIdentity, NumericContractSet,
};

/// Complete policy, compiler, artifact, and device identity pinned by graphs.
pub type GemmRoute = GemmRouteIdentity;

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
    /// Opt-in flag for the batch-invariant matvec path (`matvec_bi_*`).
    /// Default: `false` → cuBLAS gemv (faster, but M=1/M=N may differ at
    /// sub-ULP scale). Set via `set_batch_invariant(true)` or the
    /// `MAMBA_RS_BATCH_INVARIANT=1` environment variable when strict
    /// cross-batch bit-identity is required.
    batch_invariant: std::cell::Cell<bool>,
    /// Opt-in tensor-core tier for the batch-invariant typed GEMMs
    /// (stage 5). SEPARATE numeric contract: mma.sync f32 accumulation
    /// differs from the scalar __fmaf_rn chain, so outputs do not bit-match
    /// the scalar triad — but the TC kernels are fully deterministic and
    /// batch-invariant across all M. Effective only together with
    /// `batch_invariant`. Env: MAMBA_RS_BI_TENSOR_CORES.
    bi_tensor_cores: std::cell::Cell<bool>,
    /// Which deterministic family serves the forward while
    /// `batch_invariant` is on. Env: MAMBA_RS_BI_GEMM_FAMILY
    /// (`warptile` | `wmma`). Part of the numeric route, so it rides
    /// [`GpuCtx::gemm_route`] into every capture identity.
    bi_gemm_family: std::cell::Cell<BiGemmFamily>,
    /// Opt-in non-PEDANTIC cuBLAS compute for the typed (bf16/f16) GEMMs:
    /// `CUBLAS_COMPUTE_32F` lets cuBLAS pick BMMA/HMMA tensor-core kernels
    /// with f32 accumulate. SEPARATE numeric contract from the PEDANTIC
    /// default (different reduction trees; still deterministic for a fixed
    /// shape within a process). Ignored by the batch-invariant path, which
    /// never calls cuBLAS. Env: MAMBA_RS_FAST_GEMM.
    fast_gemm: std::cell::Cell<bool>,
    /// TF32 SGEMM math is enabled at cublas creation; parity tests clear
    /// it via [`Self::disable_tf32`]. Tracked so bench stamps can print
    /// the full four-bit numeric route.
    tf32: std::cell::Cell<bool>,
    /// The state capacity the kernels were compiled with — part of the
    /// numeric-route identity a bench stamp must carry.
    state_cap: usize,
    instance_token: u64,
    device_identity: super::kernel_identity::DeviceIdentity,
    policy_hash: super::kernel_identity::Sha256Digest,
    /// Number of CUDA graphs captured on this context: the tier
    /// setters warn when flipped after a capture — the captured kernels
    /// cannot follow, and the replay-time flag assert refuses to run.
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

impl GpuCtx {
    /// Create a GPU context: compile kernels, init cuBLAS with TF32.
    /// Kernels get the default state capacity of 64; models with a
    /// larger `d_state` use [`Self::new_with_state_cap`].
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
        Self::new_with_state_cap(device, 64)
    }

    /// Create a GPU context whose kernels are compiled with the given
    /// state capacity (see
    /// [`crate::mamba_ssm::gpu::kernels::state_capacity`]).
    pub fn new_with_state_cap(device: &GpuDevice, state_cap: usize) -> Result<Self, String> {
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
        // The splitk/transpose scratch buffers inside `kernels` were
        // alloc_zeros'd on the DEFAULT stream; `ctx.stream` is NON_BLOCKING
        // and never orders against it. Drain once here so first use on
        // ctx.stream can't race the init memset (same hazard class as the
        // legacy-stream memcpy fix in buffers.rs).
        device
            .default_stream()
            .synchronize()
            .map_err(|e| format!("default-stream drain after kernel compile: {e:?}"))?;
        let (blas, ws) = device.create_cublas(&stream)?;
        // Strict flag parsing: an unrecognized value must FAIL, not
        // silently mean off. "True" (Python str(True)), "ON", a stray
        // trailing space - all previously read as false, indistinguishable
        // from "not set", and a mis-set tier flag measures or serves a
        // numeric route nobody asked for.
        let tier_flag = |name: &str| -> Result<bool, String> {
            match std::env::var(name) {
                Err(_) => Ok(false),
                Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => Ok(true),
                    "0" | "false" | "no" | "off" | "" => Ok(false),
                    other => Err(format!(
                        "{name}={other:?} is not a recognized flag value \
                         (use 1/true/yes/on or 0/false/no/off)"
                    )),
                },
            }
        };
        let batch_invariant = tier_flag("MAMBA_RS_BATCH_INVARIANT")?;
        let bi_tensor_cores = tier_flag("MAMBA_RS_BI_TENSOR_CORES")?;
        let fast_gemm = tier_flag("MAMBA_RS_FAST_GEMM")?;
        // Same strict-parse law as the tier flags: an unrecognized value
        // fails rather than silently meaning the default family.
        let bi_gemm_family = match std::env::var("MAMBA_RS_BI_GEMM_FAMILY") {
            Err(_) => BiGemmFamily::Triad,
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "triad" | "sgemm_bi" => BiGemmFamily::Triad,
                "fixed" | "gemm_bi" => BiGemmFamily::Fixed,
                other => {
                    return Err(format!(
                        "MAMBA_RS_BI_GEMM_FAMILY={other:?} is not a recognized family \
                         (use fixed or triad)"
                    ));
                }
            },
        };
        // The TC tier flag is only read inside bi_sgemm_*_typed, which is
        // reachable only under batch_invariant() - TC alone is a silent
        // no-op that has already cost a day of follow-up readings.
        if bi_tensor_cores && !batch_invariant {
            return Err(
                "MAMBA_RS_BI_TENSOR_CORES=1 without MAMBA_RS_BATCH_INVARIANT=1 is a \
                 silent no-op: the tensor-core tier is reachable only under the \
                 batch-invariant dispatch. Set both or neither."
                    .to_string(),
            );
        }
        let instance_token = next_gpu_ctx_token()?;
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
            batch_invariant: std::cell::Cell::new(batch_invariant),
            bi_tensor_cores: std::cell::Cell::new(bi_tensor_cores),
            bi_gemm_family: std::cell::Cell::new(bi_gemm_family),
            fast_gemm: std::cell::Cell::new(fast_gemm),
            tf32: std::cell::Cell::new(true),
            state_cap,
            instance_token,
            device_identity: device.identity(),
            policy_hash: super::kernel_identity::legacy_sm80_policy_digest(),
            graphs_captured: std::cell::Cell::new(0),
            graph_scratch_frozen: std::cell::Cell::new(false),
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

    /// Enable or disable the batch-invariant matvec path.
    /// When `true`, dispatches to the custom `matvec_bi_*` kernel which
    /// produces bit-identical logits regardless of batch size. When `false`
    /// (default), uses cuBLAS gemv for maximum throughput.
    pub fn set_batch_invariant(&self, on: bool) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.batch_invariant.set(on);
    }

    /// Returns `true` if the batch-invariant matvec path is enabled.
    pub fn batch_invariant(&self) -> bool {
        self.batch_invariant.get()
    }

    /// Choose which deterministic GEMM family serves the forward while
    /// `batch_invariant` is on (see [`BiGemmFamily`]). No effect while
    /// `batch_invariant` is off - cuBLAS serves.
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

    /// Enable or disable the tensor-core tier of the batch-invariant typed
    /// GEMMs (stage 5). Different numeric contract than the scalar triad —
    /// deterministic and batch-invariant, but not bit-equal to it.
    pub fn set_bi_tensor_cores(&self, on: bool) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.bi_tensor_cores.set(on);
    }

    /// Enable or disable the non-PEDANTIC cuBLAS compute mode for typed
    /// GEMMs (see the `fast_gemm` field doc for the numeric contract).
    pub fn set_fast_gemm(&self, on: bool) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        self.fast_gemm.set(on);
    }

    /// Returns `true` if the non-PEDANTIC typed-GEMM compute is enabled.
    pub fn fast_gemm(&self) -> bool {
        self.fast_gemm.get()
    }

    /// Record that a CUDA graph was captured on this context.
    pub(crate) fn note_graph_capture(&self) {
        self.graphs_captured.set(self.graphs_captured.get() + 1);
    }

    /// The three runtime controls used by legacy callers.
    pub fn gemm_flags(&self) -> (bool, bool, bool) {
        (
            self.batch_invariant.get(),
            self.bi_tensor_cores.get(),
            self.fast_gemm.get(),
        )
    }

    /// Complete route identity used by eager launches and graph guards.
    pub fn gemm_route(&self) -> GemmRoute {
        let (bi, tc, fast) = self.gemm_flags();
        let family = self.bi_gemm_family.get();
        let policy = GemmPolicy {
            batch_invariant: bi,
            bi_tensor_cores: tc,
            fast_gemm: fast,
            tf32: self.tf32.get(),
            bi_gemm_family: family,
        };
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
            state_capacity: u32::try_from(self.state_cap)
                .expect("validated state capacity fits in u32"),
        }
    }

    /// Returns `true` if the tensor-core bi tier is enabled.
    pub fn bi_tensor_cores(&self) -> bool {
        self.bi_tensor_cores.get()
    }

    /// Disable TF32 Tensor Cores — use full f32 SGEMM for parity tests.
    pub fn disable_tf32(&self) {
        if self.graphs_captured.get() > 0 {
            eprintln!(
                "mamba-rs WARNING: GEMM route changed after graph capture; replay will reject it"
            );
        }
        let status = unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *self.blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            )
        };
        assert_eq!(
            status,
            cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            "cublasSetMathMode default math failed"
        );
        self.tf32.set(false);
    }

    /// TF32 SGEMM math state (true until [`Self::disable_tf32`]).
    pub fn tf32(&self) -> bool {
        self.tf32.get()
    }

    /// The state capacity this context's kernels were compiled with.
    pub fn state_cap(&self) -> usize {
        self.state_cap
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

#[cfg(test)]
mod tests {
    use super::m1_mixed_graph_max_dim;
    use crate::config::ScanMode;
    use crate::mamba_ssm::gpu::forward::GpuMambaDims;

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
}
