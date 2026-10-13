//! The device half of the dynamic loss scaler: the overflow flag the
//! inf/nan scan sets, the scan and the unscale. The host state machine is
//! [`DynamicLossScaler`] in [`crate::optim::loss_scaler`].

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use crate::gpu::buffers::GpuBuffer;
use crate::gpu::context::GpuCtx;
use crate::gpu::kernels::MambaKernels;
use crate::gpu::launch::grid_1d_wide;
pub use crate::optim::loss_scaler::DynamicLossScaler;

/// Single-element device int that the GPU kernel atomicOr-s into when an
/// inf/nan grad is found. CPU reads it after the scan.
pub struct OverflowFlag {
    data: CudaSlice<i32>,
}

impl OverflowFlag {
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        let data = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| format!("OverflowFlag alloc: {:?}", e))?;
        Ok(Self { data })
    }

    /// Reset the flag to 0 before a backward pass.
    pub fn zero(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
        stream
            .memset_zeros(&mut self.data)
            .map_err(|e| format!("OverflowFlag zero: {:?}", e))
    }

    /// Read the flag from device → host. 0 = clean, nonzero = overflow seen.
    /// Synchronous (forces a stream sync via `clone_dtoh`).
    pub fn read(&self, stream: &Arc<CudaStream>) -> Result<i32, String> {
        let host = stream
            .clone_dtoh(&self.data)
            .map_err(|e| format!("OverflowFlag read: {:?}", e))?;
        Ok(host[0])
    }

    pub(crate) fn cuda_slice(&mut self) -> &mut CudaSlice<i32> {
        &mut self.data
    }

    /// Stable device pointer for snapshot/assertion in CUDA-Graph holders.
    /// Equivalent to `device_ptr(stream).0` but without creating a
    /// `SyncOnDrop` guard — safe under `disable_event_tracking`.
    pub fn stable_ptr(&self, stream: &Arc<CudaStream>) -> cudarc::driver::sys::CUdeviceptr {
        use cudarc::driver::DevicePtr;
        let (p, _g) = self.data.device_ptr(stream);
        p
    }
}

/// Scan one f32 grad buffer for inf/nan. Atomically OR-s a 1 into `flag`
/// if any element is non-finite. Caller must `flag.zero()` before the
/// FIRST call of a backward pass; subsequent calls on the same backward
/// (e.g. across multiple weight tensors) accumulate into the same flag.
pub fn check_inf_nan_gpu(
    ctx: &GpuCtx,
    kernels: &MambaKernels,
    flag: &mut OverflowFlag,
    grads: &GpuBuffer,
) -> Result<(), String> {
    if grads.is_empty() {
        return Ok(());
    }
    let n = grads.len() as i64;
    let cfg = grid_1d_wide(grads.len());
    let mut bld = ctx.stream.launch_builder(&kernels.check_inf_nan_f32);
    let flag_ptr = {
        use cudarc::driver::DevicePtr;
        let (p, _g) = flag.cuda_slice().device_ptr(&ctx.stream);
        p
    };
    let grad_ptr = grads.cached_ptr();
    bld.arg(&flag_ptr);
    bld.arg(&grad_ptr);
    bld.arg(&n);
    unsafe { bld.launch(cfg) }.map_err(|e| format!("check_inf_nan_f32: {:?}", e))?;
    Ok(())
}

/// 1-element device buffer holding `1/loss_scale`. CPU writes the value
/// here BEFORE each f16 graph replay; the captured `scale_grads_skip`
/// kernel reads it via a stable device pointer baked at capture time.
pub struct UnscaleFactor {
    buf: GpuBuffer,
}

impl UnscaleFactor {
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, String> {
        Ok(Self {
            buf: GpuBuffer::zeros(stream, 1)?,
        })
    }
    /// Async H2D of the next-step `1/scale`. Stream-ordered so a subsequent
    /// `cuGraphLaunch` on the same stream sees the new value.
    pub fn write(&mut self, stream: &Arc<CudaStream>, unscale: f32) -> Result<(), String> {
        self.buf.upload(stream, &[unscale])
    }
    pub fn ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.buf.cached_ptr()
    }
}

/// CUDA-Graph-capturable conditional unscale. Reads the
/// overflow flag and the unscale factor from device buffers; zeros grads
/// if the flag is set, otherwise multiplies by `1/loss_scale`.
pub fn scale_grads_skip_gpu(
    ctx: &GpuCtx,
    kernels: &MambaKernels,
    flag: &mut OverflowFlag,
    grads: &mut GpuBuffer,
    unscale: &UnscaleFactor,
) -> Result<(), String> {
    if grads.is_empty() {
        return Ok(());
    }
    let n = grads.len() as i64;
    let cfg = grid_1d_wide(grads.len());
    let mut bld = ctx.stream.launch_builder(&kernels.scale_grads_skip_f32);
    let flag_ptr = {
        use cudarc::driver::DevicePtr;
        let (p, _g) = flag.cuda_slice().device_ptr(&ctx.stream);
        p
    };
    let grad_ptr = grads.cached_ptr();
    let unscale_ptr = unscale.ptr();
    bld.arg(&grad_ptr);
    bld.arg(&flag_ptr);
    bld.arg(&unscale_ptr);
    bld.arg(&n);
    unsafe { bld.launch(cfg) }.map_err(|e| format!("scale_grads_skip_f32: {e:?}"))?;
    Ok(())
}

/// In-place multiply every element of `grads` by `scale`. Used both for
/// unscaling (scale = 1 / loss_scale) and grad clipping (scale = clip /
/// global_grad_norm).
pub fn scale_grads_gpu(
    ctx: &GpuCtx,
    kernels: &MambaKernels,
    grads: &mut GpuBuffer,
    scale: f32,
) -> Result<(), String> {
    if grads.is_empty() {
        return Ok(());
    }
    let n = grads.len() as i64;
    let cfg = grid_1d_wide(grads.len());
    let mut bld = ctx.stream.launch_builder(&kernels.scale_grads_f32);
    let grad_ptr = grads.cached_ptr();
    bld.arg(&grad_ptr);
    bld.arg(&scale);
    bld.arg(&n);
    unsafe { bld.launch(cfg) }.map_err(|e| format!("scale_grads_f32: {:?}", e))?;
    Ok(())
}
