//! GPU context: bundles stream, compiled kernels, and cuBLAS handle.
//!
//! Reduces argument count for GPU functions from 3 (stream, kernels, blas)
//! to 1 (ctx). All GPU forward/backward/inference functions take `&GpuCtx`.

use super::device::GpuDevice;
use super::dtype::WeightDtype;
use super::kernels::MambaKernels;
use crate::config::MambaConfig;
use std::cell::RefCell;
use std::sync::Arc;

/// GPU execution context — holds everything needed for kernel launches.
///
/// Created once at init, passed by reference to all GPU functions.
pub struct GpuCtx {
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub kernels: MambaKernels,
    pub blas: cudarc::cublas::CudaBlas,
    pub _blas_workspace: cudarc::driver::CudaSlice<u8>,
    /// Reusable GPU byte staging buffer for f32→bf16/f16 activation downcast
    /// before mixed-precision GEMM. Grown lazily on first use.
    half_staging: RefCell<Option<cudarc::driver::CudaSlice<u8>>>,
    half_staging_ptr: RefCell<cudarc::driver::sys::CUdeviceptr>,
    half_staging_bytes: RefCell<usize>,
    /// Opt-in flag for the batch-invariant matvec path (`matvec_bi_*`).
    /// Default: `false` → cuBLAS gemv (faster, but M=1/M=N may differ at
    /// sub-ULP scale). Set via `set_batch_invariant(true)` or the
    /// `MAMBA_RS_BATCH_INVARIANT=1` environment variable when strict
    /// cross-batch bit-identity is required.
    batch_invariant: std::cell::Cell<bool>,
    /// Grow-only f32 scratch triple for the batch-invariant typed-GEMM
    /// upcast fallback: typed shapes without a native typed bucket run as
    /// "upcast inputs → f32 sgemm_bi → RNE downcast output", bit-identical
    /// to a native typed kernel by the stage-2 contract. Lazily grown on
    /// first hit; steady-state training steps reuse without allocation.
    bi_upcast_scratch: [RefCell<Option<super::buffers::GpuBuffer>>; 3],
}

impl GpuCtx {
    /// Create a GPU context: compile kernels, init cuBLAS with TF32.
    pub fn new(device: &GpuDevice) -> Result<Self, String> {
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
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let kernels = MambaKernels::compile(device.context(), arch)?;
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
        let batch_invariant = std::env::var("MAMBA_RS_BATCH_INVARIANT")
            .ok()
            .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"));
        Ok(Self {
            stream,
            kernels,
            blas,
            _blas_workspace: ws,
            half_staging: RefCell::new(None),
            half_staging_ptr: RefCell::new(0),
            half_staging_bytes: RefCell::new(0),
            batch_invariant: std::cell::Cell::new(batch_invariant),
            bi_upcast_scratch: [RefCell::new(None), RefCell::new(None), RefCell::new(None)],
        })
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

    /// Mamba-3 twin of [`Self::presize_bi_upcast_scratch_for_train`] — the
    /// M3 step GEMMs are in_proj (`d_model → in_proj_out_dim`) and
    /// out_proj (`d_inner → d_model`).
    pub fn presize_bi_upcast_scratch_for_train_m3(
        &self,
        cfg: &crate::mamba3_siso::config::Mamba3Config,
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
        let ip = cfg.in_proj_out_dim();
        let max_dim = dm.max(ip).max(di);
        let max_kn = (dm * ip).max(di * dm);
        let elems = (m * max_dim).max(max_kn);
        self.with_bi_upcast_scratch((elems, elems, elems), |_, _, _| Ok(()))
    }

    /// Enable or disable the batch-invariant matvec path.
    /// When `true`, dispatches to the custom `matvec_bi_*` kernel which
    /// produces bit-identical logits regardless of batch size. When `false`
    /// (default), uses cuBLAS gemv for maximum throughput.
    pub fn set_batch_invariant(&self, on: bool) {
        self.batch_invariant.set(on);
    }

    /// Returns `true` if the batch-invariant matvec path is enabled.
    pub fn batch_invariant(&self) -> bool {
        self.batch_invariant.get()
    }

    /// Disable TF32 Tensor Cores — use full f32 SGEMM for parity tests.
    pub fn disable_tf32(&self) {
        unsafe {
            cudarc::cublas::sys::cublasSetMathMode(
                *self.blas.handle(),
                cudarc::cublas::sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            );
        }
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
