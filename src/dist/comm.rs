//! NCCL communicator wrapper for the data-parallel world.
//!
//! Thin by design: this layer moves bytes (send/recv/broadcast for the
//! fixed-order reducer's shard exchange) and performs the one library
//! collective the opt-in `NcclSum` tier uses. The default `FixedOrder`
//! contract keeps its float math out of the library entirely — its adds
//! happen in the `det_sum_ranks` kernel, and this layer only carries
//! the addends. Bound at the `result` level of the
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
    /// Set (from any thread) the moment `ncclCommAbort` runs on this
    /// handle. Abort FREES the communicator, so every later teardown
    /// path (`shutdown`, `Drop`) must become a no-op — a second abort
    /// or a destroy on the freed handle is a double free.
    aborted: std::sync::Arc<std::sync::atomic::AtomicBool>,
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

    /// Join the world with a hard deadline: the blocking library init
    /// runs on a helper thread bound to `cuda_ctx`, and a peer that
    /// never arrives turns into a rank error after `deadline` instead
    /// of an eternal wait (the supervisor then reaps the world; the
    /// blocked helper is reclaimed by process exit).
    pub fn init_with_deadline(
        unique_id: sys::ncclUniqueId,
        rank: usize,
        world: usize,
        cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
        deadline: Duration,
    ) -> Result<Self, DistError> {
        super::watchdog::run_with_deadline("nccl-init", deadline, move || {
            cuda_ctx
                .bind_to_thread()
                .map_err(|e| DistError::Transport(format!("bind CUDA ctx for init: {e:?}")))?;
            Self::init(unique_id, rank, world, cuda_ctx)
        })?
    }

    /// Arm the collective watchdog around `f`: if the window does not
    /// close within `deadline`, the communicator is ABORTED from the
    /// timer thread (NCCL's sanctioned cross-thread unblock), which
    /// converts a transport hang inside `f` into a loud error.
    pub(super) fn with_watchdog<R>(
        &self,
        name: &str,
        deadline: Duration,
        f: impl FnOnce() -> Result<R, DistError>,
    ) -> Result<R, DistError> {
        let comm_addr = self.comm as usize;
        let aborted = self.aborted.clone();
        let wd = super::watchdog::Watchdog::arm(name, deadline, move || {
            // Order matters: mark first, so a teardown racing the abort
            // can never see an unmarked-but-freed handle.
            aborted.store(true, std::sync::atomic::Ordering::Release);
            let _ = unsafe { result::comm_abort(comm_addr as sys::ncclComm_t) };
        })?;
        let out = f();
        if wd.disarm() {
            // The abort ran: whatever f() reported, the communicator is
            // gone and the window is failed.
            return Err(DistError::Transport(format!(
                "{name}: collective window exceeded {deadline:?} — communicator aborted"
            )));
        }
        out
    }

    /// Join the world. The caller must have made `device_ordinal` the
    /// current CUDA context before calling (constructing the GPU context
    /// does). Blocking, deadline-free — prefer
    /// [`Self::init_with_deadline`]; this stays public for callers that
    /// manage their own timeline.
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
            aborted: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

    /// Byte-movement primitive: send `count` f32s to `peer`. Carries no
    /// arithmetic — the fixed-order reducer moves addends with this and
    /// keeps every floating-point add in its own kernel.
    pub(super) fn send_f32(
        &self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        count: usize,
        peer: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        unsafe {
            result::send(
                ptr as *const core::ffi::c_void,
                count,
                sys::ncclDataType_t::ncclFloat32,
                peer as core::ffi::c_int,
                self.comm,
                stream.cu_stream() as *mut _,
            )
        }
        .map_err(|e| DistError::Transport(format!("NCCL send({count} f32 -> {peer}): {e:?}")))?;
        Ok(())
    }

    /// Byte-movement primitive: receive `count` f32s from `peer`.
    pub(super) fn recv_f32(
        &self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        count: usize,
        peer: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        unsafe {
            result::recv(
                ptr as *mut core::ffi::c_void,
                count,
                sys::ncclDataType_t::ncclFloat32,
                peer as core::ffi::c_int,
                self.comm,
                stream.cu_stream() as *mut _,
            )
        }
        .map_err(|e| DistError::Transport(format!("NCCL recv({count} f32 <- {peer}): {e:?}")))?;
        Ok(())
    }

    /// Byte-movement primitive: in-place broadcast of `count` f32s
    /// rooted at `root` (send and receive buffers coincide — NCCL only
    /// reads the buffer at the root).
    pub(super) fn broadcast_f32(
        &self,
        ptr: cudarc::driver::sys::CUdeviceptr,
        count: usize,
        root: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        unsafe {
            result::broadcast(
                ptr as *const core::ffi::c_void,
                ptr as *mut core::ffi::c_void,
                count,
                sys::ncclDataType_t::ncclFloat32,
                root as core::ffi::c_int,
                self.comm,
                stream.cu_stream() as *mut _,
            )
        }
        .map_err(|e| {
            DistError::Transport(format!("NCCL broadcast({count} f32, root {root}): {e:?}"))
        })?;
        Ok(())
    }

    /// Run `f` inside one NCCL group (aggregated launch): the p2p calls
    /// enqueued within are matched as a set, which is what makes the
    /// all-pairs exchange deadlock-free. The group is closed even when
    /// `f` errors — an unbalanced group_start poisons every later call.
    pub(super) fn group<R>(f: impl FnOnce() -> Result<R, DistError>) -> Result<R, DistError> {
        result::group_start()
            .map_err(|e| DistError::Transport(format!("NCCL group start: {e:?}")))?;
        let out = f();
        let end = result::group_end();
        let r = out?;
        end.map_err(|e| DistError::Transport(format!("NCCL group end: {e:?}")))?;
        Ok(r)
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
    /// orderly exit path; Drop only aborts. Nulling the handle first
    /// makes the subsequent Drop a no-op while the context Arc still
    /// releases normally.
    pub fn shutdown(mut self) -> Result<(), DistError> {
        let comm = std::mem::replace(&mut self.comm, std::ptr::null_mut());
        if self.aborted.load(std::sync::atomic::Ordering::Acquire) {
            // The watchdog already aborted (and thereby freed) the
            // communicator; there is nothing left to finalize.
            return Err(DistError::Transport(
                "communicator was aborted by the collective watchdog —                  teardown already complete"
                    .into(),
            ));
        }
        unsafe {
            if let Err(e) = result::comm_finalize(comm) {
                // Abort tears down and FREES the handle; no destroy may
                // follow it.
                let _ = result::comm_abort(comm);
                return Err(DistError::Transport(format!("NCCL finalize: {e:?}")));
            }
            if let Err(e) = result::comm_destroy(comm) {
                // The failed destroy already consumed the handle; a
                // follow-up abort would double-free it.
                return Err(DistError::Transport(format!("NCCL destroy: {e:?}")));
            }
        }
        Ok(())
    }
}

impl Drop for MambaComm {
    fn drop(&mut self) {
        if !self.comm.is_null() && !self.aborted.load(std::sync::atomic::Ordering::Acquire) {
            // Fail-fast path: abort tears the communicator down without
            // waiting for peers. Skipped entirely when the watchdog
            // already aborted — the handle is freed and a second abort
            // would be a double free. Errors here are unreportable by
            // construction (we may be unwinding) and must never panic.
            let _ = unsafe { result::comm_abort(self.comm) };
        }
    }
}
