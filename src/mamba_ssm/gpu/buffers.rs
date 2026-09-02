//! GPU memory buffer wrapping `CudaSlice<f32>`.
//!
//! Drop-safe: CudaSlice deallocates on drop.
//! All GPU memory management goes through GpuBuffer to prevent leaks.

use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc, LazyLock, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

static NEXT_MANAGED_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);
static MANAGED_ALLOCATIONS: LazyLock<Mutex<ManagedAllocationRegistry>> =
    LazyLock::new(|| Mutex::new(ManagedAllocationRegistry::default()));

#[derive(Clone)]
pub(crate) struct ManagedAllocationEpochStamp {
    allocations: Box<[Arc<AtomicBool>]>,
}

impl ManagedAllocationEpochStamp {
    pub(crate) fn is_current(&self) -> bool {
        self.allocations
            .iter()
            .all(|alive| alive.load(Ordering::Acquire))
    }
}

pub(crate) struct ManagedAllocationRegistration {
    context_handle: usize,
    base: u64,
    id: u64,
    alive: Arc<AtomicBool>,
}

#[derive(Clone)]
struct ManagedAllocationRange {
    end: u64,
    id: u64,
    alive: Arc<AtomicBool>,
}

#[derive(Default)]
struct ManagedAllocationDomain {
    ranges: BTreeMap<u64, ManagedAllocationRange>,
}

#[derive(Default)]
struct ManagedAllocationRegistry {
    domains: HashMap<usize, ManagedAllocationDomain>,
}

fn advance_counter(counter: &AtomicU64, label: &str) -> Result<u64, String> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
            value.checked_add(1)
        })
        .map_err(|_| format!("{label} exhausted"))
}

pub(crate) fn register_managed_allocation_range(
    context_handle: usize,
    base: u64,
    bytes: u64,
) -> Result<ManagedAllocationRegistration, String> {
    if context_handle == 0 || base == 0 || bytes == 0 {
        return Err("managed CUDA allocation range must be non-empty".into());
    }
    let end = base
        .checked_add(bytes)
        .ok_or_else(|| "managed CUDA allocation range overflows u64".to_string())?;
    let id = advance_counter(&NEXT_MANAGED_ALLOCATION_ID, "managed CUDA allocation id")?;
    let mut registry = MANAGED_ALLOCATIONS
        .lock()
        .map_err(|_| "managed CUDA allocation registry is poisoned".to_string())?;
    let domain = registry.domains.entry(context_handle).or_default();
    if domain
        .ranges
        .range(..=base)
        .next_back()
        .is_some_and(|(_, range)| range.end > base)
        || domain
            .ranges
            .range(base..)
            .next()
            .is_some_and(|(&next, _)| next < end)
    {
        return Err("managed CUDA allocation ranges overlap".into());
    }
    let alive = Arc::new(AtomicBool::new(true));
    domain.ranges.insert(
        base,
        ManagedAllocationRange {
            end,
            id,
            alive: alive.clone(),
        },
    );
    Ok(ManagedAllocationRegistration {
        context_handle,
        base,
        id,
        alive,
    })
}

pub(crate) fn managed_allocation_epoch_for_ranges(
    context_handle: usize,
    ranges: &[(u64, u64)],
) -> Option<ManagedAllocationEpochStamp> {
    if ranges.is_empty() {
        return None;
    }
    let registry = MANAGED_ALLOCATIONS.lock().ok()?;
    let domain = registry.domains.get(&context_handle)?;
    let mut allocations = Vec::with_capacity(ranges.len());
    for &(pointer, bytes) in ranges {
        let end = pointer.checked_add(bytes)?;
        let (_, range) = domain.ranges.range(..=pointer).next_back()?;
        if bytes == 0 || end > range.end {
            return None;
        }
        if !allocations
            .iter()
            .any(|alive| Arc::ptr_eq(alive, &range.alive))
        {
            allocations.push(range.alive.clone());
        }
    }
    Some(ManagedAllocationEpochStamp {
        allocations: allocations.into_boxed_slice(),
    })
}

impl Drop for ManagedAllocationRegistration {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Release);
        let mut registry = MANAGED_ALLOCATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let remove_domain = if let Some(domain) = registry.domains.get_mut(&self.context_handle) {
            if domain
                .ranges
                .get(&self.base)
                .is_some_and(|range| range.id == self.id)
            {
                domain.ranges.remove(&self.base);
            }
            domain.ranges.is_empty()
        } else {
            false
        };
        if remove_domain {
            registry.domains.remove(&self.context_handle);
        }
    }
}

/// GPU memory buffer — the fundamental GPU data type.
///
/// Wraps `CudaSlice<f32>` with convenience methods for upload/download.
/// Analogous to `Vec<f32>` on CPU.
pub struct GpuBuffer {
    managed_registration: Option<ManagedAllocationRegistration>,
    data: cudarc::driver::CudaSlice<f32>,
    len: usize,
    /// Cached device pointer — stable for the lifetime of the allocation.
    /// Avoids `device_ptr()` which creates a SyncOnDrop guard that calls
    /// `cuStreamSynchronize` on drop — illegal during CUDA Graph capture.
    cached_ptr: cudarc::driver::sys::CUdeviceptr,
}

/// Mutable kernel argument for a `GpuBuffer`.
///
/// The wrapper keeps cudarc's stream synchronization behavior without exposing
/// the owning `CudaSlice` for replacement.
pub struct GpuBufferKernelArg<'a> {
    data: &'a mut cudarc::driver::CudaSlice<f32>,
}

unsafe impl<'launch, 'buffer: 'launch> cudarc::driver::PushKernelArg<GpuBufferKernelArg<'buffer>>
    for cudarc::driver::LaunchArgs<'launch>
{
    #[inline(always)]
    fn arg(&mut self, arg: GpuBufferKernelArg<'buffer>) -> &mut Self {
        <Self as cudarc::driver::PushKernelArg<&'buffer mut cudarc::driver::CudaSlice<f32>>>::arg(
            self, arg.data,
        )
    }
}

impl GpuBuffer {
    /// Allocate zeroed GPU memory.
    pub fn zeros(stream: &Arc<cudarc::driver::CudaStream>, len: usize) -> Result<Self, String> {
        let data = stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| format!("GPU alloc_zeros({}) failed: {:?}", len, e))?;
        let cached_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _guard) = data.device_ptr(stream);
            ptr
        };
        let managed_registration = if len == 0 {
            None
        } else {
            let bytes = u64::try_from(
                len.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| format!("GPU allocation size overflows usize: {len} floats"))?,
            )
            .map_err(|_| format!("GPU allocation size exceeds u64: {len} floats"))?;
            Some(register_managed_allocation_range(
                stream.context().cu_ctx() as usize,
                cached_ptr,
                bytes,
            )?)
        };
        Ok(Self {
            managed_registration,
            data,
            len,
            cached_ptr,
        })
    }

    /// Upload from CPU slice to GPU.
    pub fn from_cpu(stream: &Arc<cudarc::driver::CudaStream>, src: &[f32]) -> Result<Self, String> {
        let data = stream
            .clone_htod(src)
            .map_err(|e| format!("GPU upload({} floats) failed: {:?}", src.len(), e))?;
        let cached_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _guard) = data.device_ptr(stream);
            ptr
        };
        let managed_registration = if src.is_empty() {
            None
        } else {
            let bytes = u64::try_from(std::mem::size_of_val(src))
                .map_err(|_| format!("GPU upload size exceeds u64: {} floats", src.len()))?;
            Some(register_managed_allocation_range(
                stream.context().cu_ctx() as usize,
                cached_ptr,
                bytes,
            )?)
        };
        Ok(Self {
            managed_registration,
            len: src.len(),
            data,
            cached_ptr,
        })
    }

    /// Download GPU data to CPU Vec.
    pub fn to_cpu(&self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<Vec<f32>, String> {
        stream
            .clone_dtoh(&self.data)
            .map_err(|e| format!("GPU download({} floats) failed: {:?}", self.len, e))
    }

    /// Upload from CPU slice into existing GPU buffer (no realloc).
    /// Panics if src.len() != self.len.
    pub fn upload(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        src: &[f32],
    ) -> Result<(), String> {
        assert_eq!(
            src.len(),
            self.len,
            "upload size mismatch: src={} gpu={}",
            src.len(),
            self.len
        );
        stream
            .memcpy_htod(src, &mut self.data)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Download into existing CPU slice (no alloc).
    /// Panics if dst.len() != self.len.
    pub fn download(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dst: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(
            dst.len(),
            self.len,
            "download size mismatch: dst={} gpu={}",
            dst.len(),
            self.len
        );
        stream
            .memcpy_dtoh(&self.data, dst)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Fill with zeros (async on stream).
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        stream
            .memset_zeros(&mut self.data)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Device-to-device copy from another GpuBuffer.
    /// Panics if sizes don't match.
    pub fn copy_from(
        &mut self,
        src: &GpuBuffer,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), String> {
        assert_eq!(
            self.len, src.len,
            "D2D copy size mismatch: dst={} src={}",
            self.len, src.len
        );
        stream
            .memcpy_dtod(&src.data, &mut self.data)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Device-to-device copy using raw cached pointers (CUDA Graph safe).
    ///
    /// Unlike `copy_from`, this never calls `device_ptr()` or creates
    /// `SyncOnDrop` guards, so it's safe during CUDA Graph capture.
    pub fn copy_from_raw(
        &mut self,
        src: &GpuBuffer,
        stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), String> {
        assert_eq!(
            self.len, src.len,
            "D2D copy size mismatch: dst={} src={}",
            self.len, src.len
        );
        if self.len > 0 {
            let byte_count = self.len * std::mem::size_of::<f32>();
            let result = unsafe {
                cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    self.cached_ptr,
                    src.cached_ptr,
                    byte_count,
                    stream.cu_stream(),
                )
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "D2D copy_raw({} floats) failed: {:?}",
                    self.len, result
                ));
            }
        }
        Ok(())
    }

    /// Length in f32 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw CudaSlice reference for cuBLAS and kernel launches.
    pub fn inner(&self) -> &cudarc::driver::CudaSlice<f32> {
        &self.data
    }

    /// Mutable argument for kernel launches.
    pub fn inner_mut(&mut self) -> GpuBufferKernelArg<'_> {
        GpuBufferKernelArg {
            data: &mut self.data,
        }
    }

    /// Raw device pointer as u64 (no sync, CUDA Graph safe).
    ///
    /// Returns the cached pointer from allocation time. No `device_ptr()` call,
    /// no `SyncOnDrop` guard, no `cuStreamSynchronize`. Safe during graph capture.
    pub fn raw_ptr(
        &self,
        _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> cudarc::driver::sys::CUdeviceptr {
        self.cached_ptr
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Cached raw device pointer (stable for buffer lifetime, no sync).
    pub fn cached_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.cached_ptr
    }

    /// Raw device pointer at f32 element offset (no sync, CUDA Graph safe).
    ///
    /// Adds `offset * sizeof(f32)` to the cached base pointer.
    /// Used for per-layer sub-buffer access in kernel launches.
    ///
    /// # Panics
    /// Panics if `offset >= self.len`.
    pub fn raw_ptr_at(
        &self,
        _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        offset: usize,
    ) -> cudarc::driver::sys::CUdeviceptr {
        assert!(
            offset < self.len,
            "raw_ptr_at offset {} >= len {}",
            offset,
            self.len
        );
        let byte_off = (offset * std::mem::size_of::<f32>()) as u64;
        self.cached_ptr + byte_off
        // guard drops here, borrow ends — safe on single stream
    }

    /// Device pointer at f32 element offset, returned as reference for kernel builder.arg().
    ///
    /// Returns a boxed CUdeviceptr that lives long enough for the kernel launch.
    /// Use: `builder.arg(&buf.inner_at(offset))` or store in a local variable first.
    pub fn inner_at(&self, offset: usize) -> cudarc::driver::sys::CUdeviceptr {
        assert!(
            offset < self.len,
            "inner_at offset {} >= len {}",
            offset,
            self.len
        );
        self.cached_ptr + (offset * std::mem::size_of::<f32>()) as u64
    }

    /// Mutable device pointer at f32 element offset for kernel builder.arg().
    /// Same as inner_at — mutability is semantic (kernel will write to this address).
    pub fn inner_mut_at(&mut self, offset: usize) -> cudarc::driver::sys::CUdeviceptr {
        assert!(
            offset < self.len,
            "inner_mut_at offset {} >= len {}",
            offset,
            self.len
        );
        self.cached_ptr + (offset * std::mem::size_of::<f32>()) as u64
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        drop(self.managed_registration.take());
    }
}

/// Non-owning view into a contiguous GPU buffer (gradients or weights).
///
/// Stores a raw device pointer + length into a flat `GpuBuffer` backing store.
/// Zero-cost abstraction — no allocation, no sync, CUDA Graph safe.
/// Used by backward functions and optimizer to access individual tensors
/// within a single flat allocation (gradient buffer or weight buffer).
pub struct GradSlice {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
}

impl GradSlice {
    /// Construct a `GradSlice` from a raw device pointer + length. For
    /// internal use and parity tests; production code should obtain
    /// `GradSlice`s via [`GpuMambaGrads`] which manages the flat allocation.
    #[doc(hidden)]
    pub fn from_raw(ptr: cudarc::driver::sys::CUdeviceptr, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Raw device pointer (for kernel args and cuBLAS raw calls).
    pub fn ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.ptr
    }

    /// Length in f32 elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for kernel args (alias for ptr(), matches GpuBuffer API).
    pub fn raw_ptr(
        &self,
        _stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> cudarc::driver::sys::CUdeviceptr {
        self.ptr
    }

    /// Raw pointer as reference (for kernel builder.arg() which needs &u64).
    pub fn inner(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.ptr
    }

    /// Create a GradSlice from a base pointer and offset+len.
    pub fn from_offset(base: cudarc::driver::sys::CUdeviceptr, offset: usize, len: usize) -> Self {
        Self {
            ptr: base + (offset * std::mem::size_of::<f32>()) as u64,
            len,
        }
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Download this slice from GPU to a CPU Vec.
    ///
    /// Uses raw cuMemcpyDtoH on the slice's device pointer.
    /// Unlike GpuBuffer::to_cpu(), this works on non-owning views.
    ///
    /// Performs a full `cuCtxSynchronize` first — `cuMemcpyDtoH_v2` is
    /// host-synchronous but does NOT order against NON_BLOCKING streams,
    /// so pending kernels could otherwise still be writing the region.
    /// GradSlice copies happen at idle points; the device-wide sync is fine.
    pub fn to_cpu(&self) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; self.len];
        if self.len > 0 {
            cu_ctx_sync("GradSlice::to_cpu")?;
            let byte_count = self.len * std::mem::size_of::<f32>();
            let result = unsafe {
                cudarc::driver::sys::cuMemcpyDtoH_v2(
                    dst.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr,
                    byte_count,
                )
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "GradSlice::to_cpu({} floats) failed: {:?}",
                    self.len, result
                ));
            }
        }
        Ok(dst)
    }

    /// Upload CPU data into this slice's GPU memory region.
    ///
    /// Uses raw cuMemcpyHtoD on the slice's device pointer, bracketed by
    /// `cuCtxSynchronize`: the leading sync lets in-flight kernels finish
    /// reading the region; the trailing sync flushes the pageable-copy
    /// tail DMA before later kernel launches on NON_BLOCKING streams.
    /// Panics if src.len() != self.len.
    pub fn upload_from_cpu(&self, src: &[f32]) -> Result<(), String> {
        assert_eq!(
            src.len(),
            self.len,
            "GradSlice upload size mismatch: src={} slice={}",
            src.len(),
            self.len
        );
        if self.len > 0 {
            cu_ctx_sync("GradSlice::upload_from_cpu (pre)")?;
            let byte_count = self.len * std::mem::size_of::<f32>();
            let result = unsafe {
                cudarc::driver::sys::cuMemcpyHtoD_v2(
                    self.ptr,
                    src.as_ptr() as *const std::ffi::c_void,
                    byte_count,
                )
            };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(format!(
                    "GradSlice::upload_from_cpu({} floats) failed: {:?}",
                    self.len, result
                ));
            }
            cu_ctx_sync("GradSlice::upload_from_cpu (post)")?;
        }
        Ok(())
    }
}

/// Device-wide sync (`cuCtxSynchronize`) — waits for ALL streams including
/// NON_BLOCKING ones. Used to bracket legacy-stream memcpys that cannot
/// take a stream parameter without breaking their call sites.
fn cu_ctx_sync(what: &str) -> Result<(), String> {
    let r = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("{what}: cuCtxSynchronize failed: {r:?}"));
    }
    Ok(())
}

/// Type alias for non-owning weight views into flat weight buffers.
///
/// Same struct as GradSlice — just a (ptr, len) pair into a flat GpuBuffer.
/// The alias clarifies intent: GradSlice for gradient views, WeightSlice for weight views.
pub type WeightSlice = GradSlice;

// ---------------------------------------------------------------------------
// Mixed-precision weight storage (inference only).
//
// GpuByteBuffer: raw bytes for a single arena that can hold mixed dtypes.
// WeightSliceDyn: (ptr, len_elems, dtype) view into the arena.
// Used by GpuMambaMixedWeights for bf16/f16 inference weight storage.
// Training and grads stay f32 via GpuBuffer/GradSlice above (unchanged).
// ---------------------------------------------------------------------------

use super::dtype::WeightDtype;

/// Raw byte-backed GPU buffer — used for mixed-dtype weight arenas.
pub struct GpuByteBuffer {
    managed_registration: Option<ManagedAllocationRegistration>,
    data: cudarc::driver::CudaSlice<u8>,
    len_bytes: usize,
    cached_ptr: cudarc::driver::sys::CUdeviceptr,
}

impl GpuByteBuffer {
    pub fn zeros(
        stream: &Arc<cudarc::driver::CudaStream>,
        len_bytes: usize,
    ) -> Result<Self, String> {
        let data = stream
            .alloc_zeros::<u8>(len_bytes)
            .map_err(|e| format!("GPU alloc_zeros({len_bytes} bytes) failed: {e:?}"))?;
        let cached_ptr = {
            use cudarc::driver::DevicePtr;
            let (ptr, _g) = data.device_ptr(stream);
            ptr
        };
        let managed_registration = if len_bytes == 0 {
            None
        } else {
            Some(register_managed_allocation_range(
                stream.context().cu_ctx() as usize,
                cached_ptr,
                u64::try_from(len_bytes)
                    .map_err(|_| format!("GPU byte allocation size exceeds u64: {len_bytes}"))?,
            )?)
        };
        Ok(Self {
            managed_registration,
            data,
            len_bytes,
            cached_ptr,
        })
    }

    pub fn cached_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.cached_ptr
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    #[cfg(test)]
    fn replace_managed_allocation_generation_for_test(&mut self) -> Result<(), String> {
        if self.len_bytes == 0 {
            return Err("cannot replace an empty managed allocation generation".into());
        }
        let registration = self
            .managed_registration
            .take()
            .ok_or_else(|| "GPU byte buffer has no managed allocation registration".to_string())?;
        let context_handle = registration.context_handle;
        drop(registration);
        self.managed_registration = Some(register_managed_allocation_range(
            context_handle,
            self.cached_ptr,
            u64::try_from(self.len_bytes)
                .map_err(|_| format!("GPU byte allocation size exceeds u64: {}", self.len_bytes))?,
        )?);
        Ok(())
    }

    pub fn inner(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.data
    }

    /// Async memset to zero on the given stream.
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        stream
            .memset_zeros(&mut self.data)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Upload raw bytes into the whole buffer (exact-length contract).
    pub fn upload_bytes(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        bytes: &[u8],
    ) -> Result<(), String> {
        assert_eq!(bytes.len(), self.len_bytes, "upload_bytes size mismatch");
        stream
            .memcpy_htod(bytes, &mut self.data)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }

    /// Download the buffer contents as f64 values (the buffer must hold
    /// exactly `dst.len()` f64s). Used for the grad-clip norm partials.
    pub fn download_f64(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dst: &mut [f64],
    ) -> Result<(), String> {
        assert_eq!(
            std::mem::size_of_val(dst),
            self.len_bytes,
            "download_f64 size mismatch: dst={} f64s, gpu={} bytes",
            dst.len(),
            self.len_bytes
        );
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
        stream
            .memcpy_dtoh(&self.data, bytes)
            .map_err(|e| format!("GPU op failed: {:?}", e))
    }
}

impl Drop for GpuByteBuffer {
    fn drop(&mut self) {
        drop(self.managed_registration.take());
    }
}

/// Dtype-aware owning buffer — holds activation scratch in any dtype.
///
/// Used by GpuInferenceScratch to hold activations in f32/bf16/fp16 uniformly.
/// Exposes `.cached_ptr()` + `.len_elems()` so all existing kernel call-sites
/// work unchanged. `upload_f32` / `download_f32` do on-the-fly dtype conversion
/// for CPU <-> GPU transfers.
pub struct DtypedBuf {
    inner: GpuByteBuffer,
    n_elems: usize,
    dtype: WeightDtype,
}

impl DtypedBuf {
    pub fn zeros(
        stream: &Arc<cudarc::driver::CudaStream>,
        n_elems: usize,
        dtype: WeightDtype,
    ) -> Result<Self, String> {
        let inner = GpuByteBuffer::zeros(stream, n_elems * dtype.size_bytes())?;
        Ok(Self {
            inner,
            n_elems,
            dtype,
        })
    }

    pub fn cached_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.inner.cached_ptr()
    }

    pub fn len_elems(&self) -> usize {
        self.n_elems
    }

    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    pub fn size_bytes(&self) -> usize {
        self.n_elems * self.dtype.size_bytes()
    }

    #[cfg(test)]
    pub(crate) fn replace_managed_allocation_generation_for_test(&mut self) -> Result<(), String> {
        self.inner.replace_managed_allocation_generation_for_test()
    }

    /// Async memset to zero on the given stream. Works regardless of dtype
    /// (raw byte fill).
    pub fn zero(&mut self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<(), String> {
        self.inner.zero(stream)
    }

    /// Upload f32 data from CPU, converting to dtype on-the-fly.
    /// Stream-ordered on `stream` and synchronized before returning.
    pub fn upload_f32(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        src: &[f32],
    ) -> Result<(), String> {
        assert_eq!(src.len(), self.n_elems, "DtypedBuf upload size mismatch");
        let ptr = self.inner.cached_ptr();
        match self.dtype {
            WeightDtype::F32 => {
                let bytes: &[u8] = bytemuck::cast_slice(src);
                cu_memcpy_htod_raw(stream, ptr, bytes)
            }
            WeightDtype::Bf16 => {
                let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&buf);
                cu_memcpy_htod_raw(stream, ptr, bytes)
            }
            WeightDtype::F16 => {
                let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&buf);
                cu_memcpy_htod_raw(stream, ptr, bytes)
            }
        }
    }

    /// Download to f32, converting from dtype on-the-fly.
    /// Stream-ordered on `stream` and synchronized before returning.
    pub fn download_f32(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dst: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(dst.len(), self.n_elems, "DtypedBuf download size mismatch");
        let ptr = self.inner.cached_ptr();
        match self.dtype {
            WeightDtype::F32 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
                cu_memcpy_dtoh_raw(stream, ptr, bytes)
            }
            WeightDtype::Bf16 => {
                let mut buf = vec![half::bf16::ZERO; self.n_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(stream, ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
            WeightDtype::F16 => {
                let mut buf = vec![half::f16::ZERO; self.n_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(stream, ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
        }
    }
}

/// Stream-ordered HtoD copy + sync. The synchronous legacy-stream
/// `cuMemcpyHtoD_v2` may return while the tail DMA into device memory is
/// still in flight ("synchronous w.r.t. host" only covers the source
/// buffer), and `ctx.stream` is NON_BLOCKING — it does not serialize with
/// the legacy stream, so a kernel launched right after could read a
/// half-written buffer. Enqueue on the caller's stream instead, then sync
/// so the (possibly temporary) host buffer can be dropped.
pub(crate) fn cu_memcpy_htod_raw(
    stream: &Arc<cudarc::driver::CudaStream>,
    dst: cudarc::driver::sys::CUdeviceptr,
    bytes: &[u8],
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            dst,
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len(),
            stream.cu_stream(),
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyHtoDAsync: {r:?}"));
    }
    stream
        .synchronize()
        .map_err(|e| format!("cuMemcpyHtoDAsync sync: {e:?}"))
}

/// Stream-ordered DtoH copy + sync — see [`cu_memcpy_htod_raw`] for why the
/// legacy-stream synchronous copy is unsafe against a NON_BLOCKING stream.
pub(crate) fn cu_memcpy_dtoh_raw(
    stream: &Arc<cudarc::driver::CudaStream>,
    src: cudarc::driver::sys::CUdeviceptr,
    bytes: &mut [u8],
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            bytes.as_mut_ptr() as *mut std::ffi::c_void,
            src,
            bytes.len(),
            stream.cu_stream(),
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyDtoHAsync: {r:?}"));
    }
    stream
        .synchronize()
        .map_err(|e| format!("cuMemcpyDtoHAsync sync: {e:?}"))
}

/// Non-owning view into a dtype-tagged region of a `GpuByteBuffer`.
#[derive(Clone, Copy)]
pub struct WeightSliceDyn {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len_elems: usize,
    dtype: WeightDtype,
}

impl WeightSliceDyn {
    pub fn from_byte_offset(
        base: cudarc::driver::sys::CUdeviceptr,
        byte_offset: usize,
        len_elems: usize,
        dtype: WeightDtype,
    ) -> Self {
        Self {
            ptr: base + byte_offset as u64,
            len_elems,
            dtype,
        }
    }

    pub fn ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.ptr
    }

    pub fn len_elems(&self) -> usize {
        self.len_elems
    }

    pub fn dtype(&self) -> WeightDtype {
        self.dtype
    }

    pub fn size_bytes(&self) -> usize {
        self.len_elems * self.dtype.size_bytes()
    }

    /// Download contents to f32 CPU buffer, upcasting from typed dtype.
    /// Counterpart of [`Self::upload_from_cpu_f32`]; use for parity tests
    /// that compare typed device weights against f32 master copies.
    pub fn download_to_f32(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        dst: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(dst.len(), self.len_elems, "size mismatch");
        if self.len_elems == 0 {
            return Ok(());
        }
        match self.dtype {
            WeightDtype::F32 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
                cu_memcpy_dtoh_raw(stream, self.ptr, bytes)
            }
            WeightDtype::Bf16 => {
                let mut buf = vec![half::bf16::ZERO; self.len_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(stream, self.ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
            WeightDtype::F16 => {
                let mut buf = vec![half::f16::ZERO; self.len_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(stream, self.ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
        }
    }

    /// Upload f32 CPU data, downcasting to `dtype` on CPU side.
    pub fn upload_from_cpu_f32(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        src: &[f32],
    ) -> Result<(), String> {
        assert_eq!(src.len(), self.len_elems, "size mismatch");
        if self.len_elems == 0 {
            return Ok(());
        }
        match self.dtype {
            WeightDtype::F32 => self.upload_raw_bytes(stream, bytemuck::cast_slice(src)),
            WeightDtype::Bf16 => {
                let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
                self.upload_raw_bytes(stream, bytemuck::cast_slice(&buf))
            }
            WeightDtype::F16 => {
                let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
                self.upload_raw_bytes(stream, bytemuck::cast_slice(&buf))
            }
        }
    }

    /// Upload raw bytes matching this slice's dtype (no conversion).
    /// Caller must ensure `bytes.len() == self.size_bytes()`.
    pub fn upload_raw_bytes(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        bytes: &[u8],
    ) -> Result<(), String> {
        assert_eq!(bytes.len(), self.size_bytes(), "byte size mismatch");
        cu_memcpy_htod_raw(stream, self.ptr, bytes)
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuBuffer({} floats, {} KB)",
            self.len,
            self.size_bytes() / 1024
        )
    }
}

/// Page-locked (pinned) host staging buffer for the H2D/D2H hot path.
///
/// Allocated with `cuMemHostAlloc(flags = 0)` — CACHEABLE pinned memory.
/// Never use `CudaContext::alloc_pinned` for host-READ buffers: cudarc
/// 0.19 hardcodes `CU_MEMHOSTALLOC_WRITECOMBINED` there, and reading
/// write-combined memory from the CPU is a large regression (the
/// mean-pool / patchify consumers read every float).
///
/// Usage contract: expose the buffer as ordinary slices and pass them to
/// the normal [`GpuBuffer::upload`] / [`GpuBuffer::download`] — the CUDA
/// driver detects the page-locked pointer and takes the true-DMA path
/// (no pageable staging copy). THE CALLER OWNS THE SYNC: `download` is
/// an async enqueue (cuMemcpyDtoHAsync; the safe wrapper adds NO sync
/// for plain slices), so the host may read the destination only after
/// `stream.synchronize()`. Pageable destinations happen to block in
/// the driver; pinned ones do not — never lean on that difference.
/// Same calls, same bytes — the pin changes WHERE the copy engine
/// reads, never a value.
pub struct PinnedHostBuf {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: the allocation is plain process memory (page-locked, cacheable);
// the raw pointer is owned uniquely by this struct and freed exactly once.
unsafe impl Send for PinnedHostBuf {}
unsafe impl Sync for PinnedHostBuf {}

impl PinnedHostBuf {
    /// Allocate `len` f32s of zero-initialized pinned host memory.
    pub fn zeroed(len: usize) -> Result<Self, String> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("PinnedHostBuf: {len} floats overflows"))?;
        let ptr = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(|e| format!("cuMemHostAlloc({bytes} bytes): {e:?}"))?
            .cast::<f32>();
        let buf = Self { ptr, len };
        // malloc_host memory is uninitialized — zero it once at build.
        unsafe { std::ptr::write_bytes(buf.ptr, 0, len) };
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        // A failed free leaks the pages but cannot corrupt anything —
        // ignore (process teardown frees them regardless).
        let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GpuBuffer, GpuBufferKernelArg, GpuByteBuffer, MANAGED_ALLOCATIONS,
        managed_allocation_epoch_for_ranges, register_managed_allocation_range,
    };
    use crate::mamba_ssm::gpu::device::GpuDevice;

    #[test]
    fn managed_allocation_epoch_covers_subviews_until_owner_drop() {
        let registration = register_managed_allocation_range(0x1001, 0x20_0000, 4096).unwrap();
        let stamp = managed_allocation_epoch_for_ranges(0x1001, &[(0x20_0100, 512)])
            .expect("registered subview must have a managed epoch");

        assert!(stamp.is_current());
        drop(registration);
        assert!(!stamp.is_current());
        assert!(managed_allocation_epoch_for_ranges(0x1001, &[(0x20_0100, 512)]).is_none());
    }

    #[test]
    fn managed_allocation_epoch_rejects_aba_address_reuse() {
        let first = register_managed_allocation_range(0x1002, 0x30_0000, 4096).unwrap();
        let first_stamp = managed_allocation_epoch_for_ranges(0x1002, &[(0x30_0000, 4096)])
            .expect("first allocation must be registered");
        drop(first);
        let second = register_managed_allocation_range(0x1002, 0x30_0000, 4096).unwrap();
        let second_stamp = managed_allocation_epoch_for_ranges(0x1002, &[(0x30_0000, 4096)])
            .expect("reused address must be registered as a new allocation");

        assert!(!first_stamp.is_current());
        assert!(second_stamp.is_current());
        drop(second);
    }

    #[test]
    fn managed_allocation_stamp_ignores_unrelated_allocation_churn() {
        let tracked = register_managed_allocation_range(0x1005, 0x50_0000, 4096).unwrap();
        let stamp = managed_allocation_epoch_for_ranges(0x1005, &[(0x50_0100, 512)])
            .expect("tracked allocation must have a managed stamp");

        let unrelated = register_managed_allocation_range(0x1005, 0x60_0000, 4096).unwrap();
        assert!(stamp.is_current());
        drop(unrelated);
        assert!(stamp.is_current());

        drop(tracked);
        assert!(!stamp.is_current());
    }

    #[test]
    fn managed_allocation_registry_removes_empty_context_domains() {
        let context_handle = 0x1006;
        let first = register_managed_allocation_range(context_handle, 0x70_0000, 4096).unwrap();
        let second = register_managed_allocation_range(context_handle, 0x80_0000, 4096).unwrap();

        drop(first);
        assert!(
            MANAGED_ALLOCATIONS
                .lock()
                .unwrap()
                .domains
                .contains_key(&context_handle)
        );
        drop(second);
        assert!(
            !MANAGED_ALLOCATIONS
                .lock()
                .unwrap()
                .domains
                .contains_key(&context_handle)
        );
    }

    #[test]
    fn mutable_gpu_buffer_access_returns_only_a_kernel_argument() {
        fn require_kernel_arg_signature(
            _method: for<'a> fn(&'a mut GpuBuffer) -> GpuBufferKernelArg<'a>,
        ) {
        }

        require_kernel_arg_signature(GpuBuffer::inner_mut);
    }

    #[test]
    fn managed_allocation_epoch_requires_one_domain_and_complete_ranges() {
        let registration = register_managed_allocation_range(0x1003, 0x40_0000, 4096).unwrap();

        assert!(
            managed_allocation_epoch_for_ranges(0x1004, &[(0x40_0000, 64)]).is_none(),
            "another CUDA context must not inherit this allocation"
        );
        assert!(
            managed_allocation_epoch_for_ranges(0x1003, &[(0x40_0f00, 512)]).is_none(),
            "a range crossing the managed allocation end must not be trusted"
        );
        assert!(
            managed_allocation_epoch_for_ranges(0x1003, &[(u64::MAX - 7, 16)]).is_none(),
            "overflowing required ranges must not be trusted"
        );
        drop(registration);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn owning_gpu_buffers_register_and_unregister_their_ranges() {
        let device = GpuDevice::new(0).expect("open CUDA device");
        let stream = device.fork_stream().expect("create CUDA stream");
        let context_handle = stream.context().cu_ctx() as usize;

        let f32_buffer = GpuBuffer::zeros(&stream, 64).expect("allocate f32 buffer");
        let f32_stamp = managed_allocation_epoch_for_ranges(
            context_handle,
            &[(f32_buffer.cached_ptr(), f32_buffer.size_bytes() as u64)],
        )
        .expect("GpuBuffer allocation must be registered");
        drop(f32_buffer);
        assert!(!f32_stamp.is_current());

        let byte_buffer = GpuByteBuffer::zeros(&stream, 256).expect("allocate byte buffer");
        let byte_stamp = managed_allocation_epoch_for_ranges(
            context_handle,
            &[(byte_buffer.cached_ptr(), byte_buffer.len_bytes() as u64)],
        )
        .expect("GpuByteBuffer allocation must be registered");
        drop(byte_buffer);
        assert!(!byte_stamp.is_current());
    }
}
