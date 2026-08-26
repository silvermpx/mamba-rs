//! Transport-backed fixed-order reduction: the device half of the
//! `ReduceContract::FixedOrder` contract.
//!
//! The dataflow is the one `fold` documents: rank r owns one contiguous
//! shard of the arena; every peer's copy of that shard reaches r as
//! PURE BYTE MOVEMENT (NCCL send/recv — no library arithmetic); r folds
//! the W addends per element in strictly ascending source-rank order on
//! device (`det_sum_ranks`, one thread per element, a serial add
//! chain); the reduced shards travel back as byte movement again
//! (per-owner broadcast). The only floating-point additions on the
//! whole path happen inside the fold kernel, in program-text order — so
//! the reduced bits are independent of transport, delivery order,
//! topology, and library version, exactly as the contract promises.
//!
//! The kernel produces the SUM; the mean scale stays with the caller
//! (sum then multiply by `1/W`, exact for power-of-two worlds) — the
//! same split `all_reduce_grad_sum` already has for the `NcclSum` tier.
//!
//! [`LoopbackWorld`] is the single-GPU proof harness: all W arenas live
//! in one process, byte movement is device-to-device copy, and the fold
//! runs the SAME kernel with the SAME slot-by-source-rank layout — so
//! the numeric path is bit-identical to the multi-process one, and the
//! oracle tests pin it against the host reference fold.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;

use super::error::DistError;
use super::fold::shard_plan;
use crate::mamba_ssm::gpu::buffers::GpuBuffer;
use crate::mamba_ssm::gpu::device::GpuDevice;

/// One thread per output element; the W addends are summed in a serial
/// ascending-source-rank chain, so the association order is program
/// text, not schedule. `stacked` is `[world][len]`, slot index = source
/// logical rank.
const DET_SUM_RANKS_SRC: &str = r#"
extern "C" __global__ void det_sum_ranks(
    float* __restrict__ out,
    const float* __restrict__ stacked,
    int world,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    float acc = stacked[i];
    for (int r = 1; r < world; ++r) {
        acc += stacked[(size_t)r * (size_t)len + (size_t)i];
    }
    out[i] = acc;
}
"#;

/// Plain synchronous device allocation (`cuMemAlloc`) — context-scoped,
/// not stream-scoped, so scratch created here is valid on any stream of
/// the context with no ordering hazard. Deliberately NOT zero-filled:
/// a driver memset would ride the legacy NULL stream, which a
/// NON_BLOCKING consumer stream never orders against (the exact hazard
/// class gpu/context.rs and gpu/buffers.rs document), and the reducer
/// fully writes every slot before the fold reads any — initialization
/// would be a pure race liability. Freed on drop.
struct DeviceScratch {
    ptr: cudarc::driver::sys::CUdeviceptr,
    /// Keeps the owning context retained for the allocation's lifetime.
    _ctx: Arc<cudarc::driver::CudaContext>,
}

impl DeviceScratch {
    fn alloc(ctx: &Arc<cudarc::driver::CudaContext>, elems: usize) -> Result<Self, DistError> {
        // cuMemAlloc allocates in the calling thread's CURRENT context —
        // bind the handed one first so the scratch always lives where
        // the kernel and the copies run.
        ctx.bind_to_thread()
            .map_err(|e| DistError::Transport(format!("bind ctx for scratch alloc: {e:?}")))?;
        let bytes = elems.max(1) * std::mem::size_of::<f32>();
        let mut ptr: cudarc::driver::sys::CUdeviceptr = 0;
        unsafe {
            let r = cudarc::driver::sys::cuMemAlloc_v2(&mut ptr, bytes);
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(DistError::Transport(format!(
                    "reducer scratch alloc ({bytes} B): {r:?}"
                )));
            }
        }
        Ok(Self {
            ptr,
            _ctx: ctx.clone(),
        })
    }
}

impl Drop for DeviceScratch {
    fn drop(&mut self) {
        // A failed free leaks device memory but cannot corrupt anything.
        let _ = unsafe { cudarc::driver::sys::cuMemFree_v2(self.ptr) };
    }
}

/// The compiled fixed-order fold kernel, pinned to one device.
pub struct DetReduceKernel {
    func: cudarc::driver::CudaFunction,
    /// Holds the device's primary context retained for the module's
    /// whole lifetime — the loaded module lives in this context, and
    /// releasing it would tear the code out from under the function.
    ctx: Arc<cudarc::driver::CudaContext>,
}

impl DetReduceKernel {
    /// Compile the fold kernel for `ordinal`'s architecture.
    pub fn compile(ordinal: usize) -> Result<Self, DistError> {
        let device = GpuDevice::new(ordinal)
            .map_err(|e| DistError::Transport(format!("reducer device init: {e}")))?;
        let arch = GpuDevice::nvrtc_arch(device.compute_capability);
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(DET_SUM_RANKS_SRC, opts)
            .map_err(|e| DistError::Transport(format!("NVRTC det_sum_ranks: {e:?}")))?;
        let ctx = device.context().clone();
        let module = ctx
            .load_module(ptx)
            .map_err(|e| DistError::Transport(format!("det_sum_ranks module load: {e:?}")))?;
        let func = module
            .load_function("det_sum_ranks")
            .map_err(|e| DistError::Transport(format!("det_sum_ranks lookup: {e:?}")))?;
        Ok(Self { func, ctx })
    }

    /// Fold `world` stacked addends into `out` (`len` elements each),
    /// ascending slot order. Stream-ordered; no synchronization.
    ///
    /// # Safety
    /// `out_ptr` must address at least `len` f32s and `stacked_ptr` at
    /// least `world * len` f32s, both valid device allocations of the
    /// context this kernel was compiled for, and both valid for the
    /// whole stream-ordered lifetime of the launch. The regions must
    /// not overlap.
    pub unsafe fn launch(
        &self,
        out_ptr: cudarc::driver::sys::CUdeviceptr,
        stacked_ptr: cudarc::driver::sys::CUdeviceptr,
        world: usize,
        len: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        if len == 0 {
            return Ok(());
        }
        if len > i32::MAX as usize || world > i32::MAX as usize {
            return Err(DistError::Transport(format!(
                "det_sum_ranks: len {len} / world {world} exceed the i32 kernel ABI"
            )));
        }
        let world_i = world as i32;
        let len_i = len as i32;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((len as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&self.func);
        b.arg(&out_ptr);
        b.arg(&stacked_ptr);
        b.arg(&world_i);
        b.arg(&len_i);
        unsafe { b.launch(cfg) }
            .map_err(|e| DistError::Transport(format!("det_sum_ranks launch: {e:?}")))?;
        Ok(())
    }
}

/// Per-context state for the transport-backed fixed-order reduction:
/// the compiled kernel plus stacked receive scratch per arena length.
#[cfg(feature = "nccl")]
pub(super) struct FixedOrderState {
    kernel: DetReduceKernel,
    /// One scratch per distinct arena length: the gradient lane and the
    /// small host-head lane alternate lengths, and a keep-per-length
    /// cache means an allocation is NEVER freed while a caller could
    /// still hold work referencing it — no free-in-flight hazard by
    /// construction (the set of distinct lengths is tiny in practice).
    stacked: std::cell::RefCell<std::collections::HashMap<usize, DeviceScratch>>,
}

#[cfg(feature = "nccl")]
impl FixedOrderState {
    pub(super) fn compile(ordinal: usize) -> Result<Self, DistError> {
        Ok(Self {
            kernel: DetReduceKernel::compile(ordinal)?,
            stacked: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Stacked receive scratch for `world * my_len` elements, cached per
    /// arena length.
    fn stacked_ptr(
        &self,
        arena_len: usize,
        world: usize,
        my_len: usize,
    ) -> Result<cudarc::driver::sys::CUdeviceptr, DistError> {
        let mut map = self.stacked.borrow_mut();
        if let std::collections::hash_map::Entry::Vacant(slot) = map.entry(arena_len) {
            slot.insert(DeviceScratch::alloc(&self.kernel.ctx, world * my_len)?);
        }
        let Some(scratch) = map.get(&arena_len) else {
            return Err(DistError::Transport(
                "reducer stacked scratch missing after allocation".into(),
            ));
        };
        Ok(scratch.ptr)
    }
}

/// The production dataflow over a live communicator: exchange peer
/// copies of my shard (byte movement), fold ascending on device,
/// redistribute every owner's reduced shard (byte movement). In-place
/// on `arena`; stream-ordered on `stream`.
///
/// Cross-rank contract: every rank must present the SAME `arena.len()`
/// (guaranteed upstream by identical model shapes on every replica).
/// NCCL does not validate p2p size agreement, so a length mismatch is
/// a transport hang bounded only by the collective watchdog, not a
/// nameable error.
#[cfg(feature = "nccl")]
pub(super) fn reduce_sum_fixed_order_nccl(
    comm: &super::comm::MambaComm,
    state: &FixedOrderState,
    arena: &GpuBuffer,
    stream: &cudarc::driver::CudaStream,
) -> Result<(), DistError> {
    let world = comm.world();
    let me = comm.rank();
    if world <= 1 {
        return Ok(());
    }
    let n = arena.len();
    let plan = shard_plan(n, world);
    let my = plan[me];
    let f32_size = std::mem::size_of::<f32>() as u64;
    let arena_base = arena.cached_ptr();
    let stacked_base = state.stacked_ptr(n, world, my.len)?;

    // Exchange: my copy of shard p goes to owner p; peer copies of MY
    // shard land in `stacked` at slot = source rank. All calls in one
    // NCCL group — the aggregated p2p pattern is deadlock-free by
    // construction.
    super::comm::MambaComm::group(|| {
        for (p, shard) in plan.iter().enumerate() {
            if p == me {
                continue;
            }
            if shard.len > 0 {
                comm.send_f32(
                    arena_base + shard.start as u64 * f32_size,
                    shard.len,
                    p,
                    stream,
                )?;
            }
            if my.len > 0 {
                comm.recv_f32(
                    stacked_base + (p * my.len) as u64 * f32_size,
                    my.len,
                    p,
                    stream,
                )?;
            }
        }
        Ok(())
    })?;
    if my.len > 0 {
        // Own copy into its slot, then the ascending fold into my arena
        // shard (both stream-ordered after the receives).
        copy_d2d(
            stacked_base + (me * my.len) as u64 * f32_size,
            arena_base + my.start as u64 * f32_size,
            my.len,
            stream,
        )?;
        unsafe {
            state.kernel.launch(
                arena_base + my.start as u64 * f32_size,
                stacked_base,
                world,
                my.len,
                stream,
            )
        }?;
    }
    // Distribute: every owner's reduced shard reaches every rank as
    // byte movement (in-place broadcast rooted at the owner).
    super::comm::MambaComm::group(|| {
        for (o, shard) in plan.iter().enumerate() {
            if shard.len > 0 {
                comm.broadcast_f32(
                    arena_base + shard.start as u64 * f32_size,
                    shard.len,
                    o,
                    stream,
                )?;
            }
        }
        Ok(())
    })
}

/// Raw async device-to-device copy of `len` f32s (stream-ordered).
fn copy_d2d(
    dst: cudarc::driver::sys::CUdeviceptr,
    src: cudarc::driver::sys::CUdeviceptr,
    len: usize,
    stream: &cudarc::driver::CudaStream,
) -> Result<(), DistError> {
    if len == 0 {
        return Ok(());
    }
    let bytes = len * std::mem::size_of::<f32>();
    let r =
        unsafe { cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.cu_stream()) };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(DistError::Transport(format!("reducer D2D copy: {r:?}")));
    }
    Ok(())
}

/// Single-GPU proof harness for the fixed-order dataflow: all W rank
/// arenas live in one process, byte movement is device-to-device copy,
/// and the fold is the SAME kernel with the SAME slot-by-source-rank
/// layout the live path uses — the numeric route is identical, only the
/// byte mover differs (and byte movement has no numerics to get wrong).
pub struct LoopbackWorld {
    arenas: Vec<GpuBuffer>,
    /// Copy peer contributions in DESCENDING source order during the
    /// exchange. Slot placement is by source rank, so the fold input —
    /// and therefore the bits — must not change; the oracle test pins
    /// exactly that delivery-order immunity.
    pub reverse_delivery: bool,
}

impl LoopbackWorld {
    pub fn new(arenas: Vec<GpuBuffer>) -> Self {
        Self {
            arenas,
            reverse_delivery: false,
        }
    }

    pub fn arenas(&self) -> &[GpuBuffer] {
        &self.arenas
    }

    /// Run one full fixed-order reduction round across all ranks:
    /// per-rank exchange + fold, then the distribute phase. After the
    /// call every arena holds the identical ascending-rank sum.
    pub fn run_round(
        &mut self,
        kernel: &DetReduceKernel,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), DistError> {
        let world = self.arenas.len();
        if world <= 1 {
            return Ok(());
        }
        let n = self.arenas[0].len();
        for a in &self.arenas {
            if a.len() != n {
                return Err(DistError::Transport(
                    "loopback arenas must share one length".into(),
                ));
            }
        }
        let plan = shard_plan(n, world);
        let f32_size = std::mem::size_of::<f32>() as u64;

        // Phase A: each owner collects + folds its shard. A fold writes
        // only the owner's own shard region, and later exchanges read
        // only OTHER shard regions of that arena — so sequential
        // per-rank processing is exact.
        for (me, my) in plan.iter().enumerate() {
            if my.len == 0 {
                continue;
            }
            let stacked = DeviceScratch::alloc(&kernel.ctx, world * my.len)?;
            let order: Vec<usize> = if self.reverse_delivery {
                (0..world).rev().collect()
            } else {
                (0..world).collect()
            };
            // Enqueue copies + fold, then ALWAYS drain the stream before
            // `stacked` drops — including on the error paths: an async
            // copy already in flight would otherwise read freed memory.
            let enqueue = || -> Result<(), DistError> {
                for src in order {
                    copy_d2d(
                        stacked.ptr + (src * my.len) as u64 * f32_size,
                        self.arenas[src].cached_ptr() + my.start as u64 * f32_size,
                        my.len,
                        stream,
                    )?;
                }
                unsafe {
                    kernel.launch(
                        self.arenas[me].cached_ptr() + my.start as u64 * f32_size,
                        stacked.ptr,
                        world,
                        my.len,
                        stream,
                    )
                }
            };
            let enqueued = enqueue();
            let synced = stream
                .synchronize()
                .map_err(|e| DistError::Transport(format!("loopback sync: {e:?}")));
            enqueued?;
            synced?;
        }

        // Phase B: distribute every owner's reduced shard to all ranks.
        for (o, shard) in plan.iter().enumerate() {
            if shard.len == 0 {
                continue;
            }
            for r in 0..world {
                if r == o {
                    continue;
                }
                copy_d2d(
                    self.arenas[r].cached_ptr() + shard.start as u64 * f32_size,
                    self.arenas[o].cached_ptr() + shard.start as u64 * f32_size,
                    shard.len,
                    stream,
                )?;
            }
        }
        stream
            .synchronize()
            .map_err(|e| DistError::Transport(format!("loopback sync: {e:?}")))?;
        Ok(())
    }
}

/// Host reference for the transport dataflow's SUM (no mean scale):
/// per element, fold the W addends in ascending source-rank order.
/// The device kernel must reproduce these exact bits.
pub fn reduce_sum_reference(addends: &[&[f32]], out: &mut [f32]) {
    let world = addends.len();
    assert!(world > 0, "need at least one addend");
    for a in addends {
        assert_eq!(a.len(), out.len(), "addend length mismatch");
    }
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = addends[0][i];
        for a in &addends[1..] {
            acc += a[i];
        }
        *o = acc;
    }
}
