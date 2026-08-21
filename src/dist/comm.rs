//! NCCL communicator wrapper for the data-parallel world.
//!
//! Thin by design: this layer moves bytes and performs the one library
//! collective the opt-in `NcclSum` tier uses. The default `FixedOrder`
//! contract keeps its float math out of the library BY REFUSING to run
//! over this transport until its house reducer is wired (the emulated
//! world implements the fold today; the transport-backed tier lands
//! with the multi-GPU validation). Bound at the `result` level of the
//! binding — the higher-level safe wrapper panics in Drop on abort
//! errors, which is exactly wrong for a fail-fast world that aborts as
//! a matter of course.

use std::path::Path;
use std::time::{Duration, Instant};

use cudarc::nccl::{result, sys};

use super::error::DistError;

/// Minimum library version this crate accepts, encoded as
/// `major*10000 + minor*100 + patch` (the NCCL convention). Older
/// versions carry known collective hangs on modern architectures.
const MIN_NCCL_VERSION: i32 = 22600;

pub struct MambaComm {
    comm: sys::ncclComm_t,
    rank: usize,
    world: usize,
    /// Keeps the rank's CUDA primary context retained for the
    /// communicator's whole lifetime — NCCL binds to the context that is
    /// current at init, and dropping it would release the device out
    /// from under the communicator.
    _cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// The communicator is used from its owning thread only (the same
// single-stream discipline the GPU context imposes), but moving that
// ownership between threads is sound: NCCL communicators are not tied
// to a thread, only to the CUDA context this struct retains. The raw
// handle removes the auto-derived Send; restore it explicitly.
unsafe impl Send for MambaComm {}

// One communicator per process, used from the owning thread only — the
// same single-stream discipline the GPU context already imposes.

impl MambaComm {
    /// Query the loaded library version, refusing versions below the
    /// supported floor. Also serves as the "is the library present"
    /// preflight: a missing libnccl surfaces here, before any rank
    /// spawns.
    pub fn preflight_version() -> Result<i32, DistError> {
        let v = result::get_nccl_version()
            .map_err(|e| DistError::Transport(format!("NCCL version query: {e:?}")))?;
        if v < MIN_NCCL_VERSION {
            return Err(DistError::Transport(format!(
                "NCCL {v} is below the supported floor {MIN_NCCL_VERSION}"
            )));
        }
        Ok(v)
    }

    /// Rank 0 generates the unique id and publishes it at `path` via
    /// atomic rename; other ranks poll for it. The file lives in the
    /// job's rendezvous directory, so ranks of different jobs cannot
    /// cross-connect.
    pub fn exchange_unique_id(
        path: &Path,
        rank: usize,
        timeout: Duration,
    ) -> Result<sys::ncclUniqueId, DistError> {
        if rank == 0 {
            let id = result::get_uniqueid()
                .map_err(|e| DistError::Transport(format!("NCCL unique id: {e:?}")))?;
            let bytes: Vec<u8> = id.internal.iter().map(|&b| b as u8).collect();
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, &bytes)
                .map_err(|e| DistError::Rendezvous(format!("write {}: {e}", tmp.display())))?;
            std::fs::rename(&tmp, path)
                .map_err(|e| DistError::Rendezvous(format!("rename {}: {e}", path.display())))?;
            Ok(id)
        } else {
            let deadline = Instant::now() + timeout;
            loop {
                if let Ok(bytes) = std::fs::read(path)
                    && bytes.len() == 128
                {
                    let mut id = sys::ncclUniqueId { internal: [0; 128] };
                    for (dst, &src) in id.internal.iter_mut().zip(bytes.iter()) {
                        *dst = src as core::ffi::c_char;
                    }
                    return Ok(id);
                }
                if Instant::now() >= deadline {
                    return Err(DistError::Rendezvous(format!(
                        "rank {rank}: NCCL unique id did not appear at {} within {timeout:?}",
                        path.display()
                    )));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// Join the world. The caller must have made `device_ordinal` the
    /// current CUDA context before calling (constructing the GPU context
    /// does). Blocking init — a peer that never arrives leaves this
    /// call waiting, which the supervising process converts into a
    /// fail-fast kill; a nonblocking init with an in-process deadline is
    /// a planned hardening.
    pub fn init(
        unique_id: sys::ncclUniqueId,
        rank: usize,
        world: usize,
        cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    ) -> Result<Self, DistError> {
        let mut comm: sys::ncclComm_t = std::ptr::null_mut();
        unsafe { result::comm_init_rank(&mut comm, world as i32, unique_id, rank as i32) }
            .map_err(|e| {
                // The binding's error type is information-free; NCCL's
                // own last-error string names the real cause (duplicate
                // device, transport failure, version skew, ...).
                let detail = unsafe {
                    let p = sys::ncclGetLastError(std::ptr::null_mut());
                    if p.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                DistError::Transport(format!("NCCL init rank {rank}/{world}: {e:?} — {detail}"))
            })?;
        Ok(Self {
            comm,
            rank,
            world,
            _cuda_ctx: cuda_ctx,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world(&self) -> usize {
        self.world
    }

    /// In-place f32 sum across ranks — the `NcclSum` tier's collective.
    /// The mean scale stays with the caller (sum then multiply, exact
    /// for power-of-two worlds).
    ///
    /// # Safety contract (checked by the caller)
    /// `ptr` must be a device allocation of at least `count` f32s valid
    /// on `stream`.
    pub fn all_reduce_sum_f32(
        &self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        count: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        unsafe {
            result::all_reduce(
                ptr as *const core::ffi::c_void,
                ptr as *mut core::ffi::c_void,
                count,
                sys::ncclDataType_t::ncclFloat32,
                sys::ncclRedOp_t::ncclSum,
                self.comm,
                stream.cu_stream() as *mut _,
            )
        }
        .map_err(|e| DistError::Transport(format!("NCCL allreduce({count} f32): {e:?}")))?;
        Ok(())
    }

    /// Integer maximum across ranks — the exact, dtype-independent
    /// carrier for boolean flags (any-rank-overflowed and friends).
    pub fn all_reduce_max_i32(
        &self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        count: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        unsafe {
            result::all_reduce(
                ptr as *const core::ffi::c_void,
                ptr as *mut core::ffi::c_void,
                count,
                sys::ncclDataType_t::ncclInt32,
                sys::ncclRedOp_t::ncclMax,
                self.comm,
                stream.cu_stream() as *mut _,
            )
        }
        .map_err(|e| DistError::Transport(format!("NCCL allreduce max({count} i32): {e:?}")))?;
        Ok(())
    }

    /// Clean shutdown: finalize (collective) then destroy. Call on the
    /// orderly exit path; Drop only aborts.
    pub fn shutdown(mut self) -> Result<(), DistError> {
        let comm = std::mem::replace(&mut self.comm, std::ptr::null_mut());
        std::mem::forget(self);
        unsafe {
            if let Err(e) = result::comm_finalize(comm) {
                // Do not leak the handle on a failed finalize: tear it
                // down the abort way before reporting.
                let _ = result::comm_abort(comm);
                let _ = result::comm_destroy(comm);
                return Err(DistError::Transport(format!("NCCL finalize: {e:?}")));
            }
            if let Err(e) = result::comm_destroy(comm) {
                let _ = result::comm_abort(comm);
                return Err(DistError::Transport(format!("NCCL destroy: {e:?}")));
            }
        }
        Ok(())
    }
}

impl Drop for MambaComm {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            // Fail-fast path: abort tears the communicator down without
            // waiting for peers. Errors here are unreportable by
            // construction (we may be unwinding) and must never panic.
            let _ = unsafe { result::comm_abort(self.comm) };
        }
    }
}
