//! The cuBLAS handles of a process, kept for its life.
//!
//! `cublasDestroy` synchronizes the whole device: the cuBLAS documentation
//! says it calls `cudaDeviceSynchronize`. That waits on every stream of the
//! device, and while any stream of the context captures a CUDA graph a
//! synchronize of the device is invalid and ends that capture, whichever
//! thread runs it. So a context never destroys its handle. It returns the
//! handle here when it drops, and the next context of the same CUDA context
//! takes it, bound to its own stream, math mode and workspace. A process
//! keeps as many handles as it held at once.

use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, MutexGuard};

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{cublasMath_t, cublasStatus_t};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

/// The workspace a handle runs with. Without one cuBLAS allocates its own
/// inside a captured GEMM. 32 MiB covers the Hopper kernels.
pub(crate) const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// Idle handles, each with the CUDA context it was created in.
static IDLE: Mutex<Vec<(usize, CudaBlas)>> = Mutex::new(Vec::new());

/// The idle list. Every update leaves it whole, so a lock poisoned by a
/// panic elsewhere has nothing to repair.
fn idle() -> MutexGuard<'static, Vec<(usize, CudaBlas)>> {
    IDLE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A cuBLAS handle bound to one stream, with a workspace of its own.
/// Dropping it waits for the stream and returns the handle to the idle list,
/// and the workspace is freed on its stream.
pub(crate) struct PooledBlas {
    blas: ManuallyDrop<CudaBlas>,
    context: usize,
    workspace: CudaSlice<u8>,
}

impl PooledBlas {
    /// An idle handle of `stream`'s context, or a new one, bound to `stream`
    /// with `math` and a new workspace.
    pub(crate) fn take(stream: &Arc<CudaStream>, math: cublasMath_t) -> Result<Self, String> {
        let context = stream.context().cu_ctx() as usize;
        let reused = {
            let mut list = idle();
            list.iter()
                .position(|(owner, _)| *owner == context)
                .map(|at| list.swap_remove(at).1)
        };
        let mut blas = match reused {
            Some(blas) => blas,
            None => CudaBlas::new(stream.clone())
                .map_err(|error| format!("cuBLAS init failed: {error:?}"))?,
        };
        match bind(&mut blas, stream, math) {
            Ok(workspace) => Ok(Self {
                blas: ManuallyDrop::new(blas),
                context,
                workspace,
            }),
            Err(error) => {
                idle().push((context, blas));
                Err(error)
            }
        }
    }

    pub(crate) fn blas(&self) -> &CudaBlas {
        &self.blas
    }
}

/// Bind `blas` to `stream`: the stream first, since setting it resets the
/// workspace, then a new workspace and the math mode.
fn bind(
    blas: &mut CudaBlas,
    stream: &Arc<CudaStream>,
    math: cublasMath_t,
) -> Result<CudaSlice<u8>, String> {
    unsafe { blas.set_stream(stream.clone()) }
        .map_err(|error| format!("cublasSetStream failed: {error:?}"))?;
    let workspace = stream
        .alloc_zeros::<u8>(WORKSPACE_BYTES)
        .map_err(|error| format!("cuBLAS workspace alloc failed: {error:?}"))?;
    let status = {
        let (ptr, _sync) = workspace.device_ptr(stream);
        unsafe {
            cudarc::cublas::sys::cublasSetWorkspace_v2(
                *blas.handle(),
                ptr as *mut std::ffi::c_void,
                WORKSPACE_BYTES,
            )
        }
    };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasSetWorkspace_v2 failed: {status:?}"));
    }
    let status = unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), math) };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasSetMathMode({math:?}) failed: {status:?}"));
    }
    Ok(workspace)
}

impl Drop for PooledBlas {
    fn drop(&mut self) {
        // The next owner binds the handle to its own stream at once, so the
        // GEMMs this one enqueued finish first. A stream of its own is not
        // the device: waiting on it ends no capture elsewhere.
        let _ = self.workspace.stream().synchronize();
        // SAFETY: the handle leaves the wrapper here and nowhere else.
        let blas = unsafe { ManuallyDrop::take(&mut self.blas) };
        idle().push((self.context, blas));
    }
}
