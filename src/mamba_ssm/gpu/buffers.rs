//! GPU memory buffer wrapping CudaSlice<f32>.
//!
//! Drop-safe: CudaSlice deallocates on drop.
//! All GPU memory management goes through GpuBuffer to prevent leaks.

use std::sync::Arc;

/// GPU memory buffer — the fundamental GPU data type.
///
/// Wraps `CudaSlice<f32>` with convenience methods for upload/download.
/// Analogous to `Vec<f32>` on CPU.
pub struct GpuBuffer {
    data: cudarc::driver::CudaSlice<f32>,
    len: usize,
    /// Cached device pointer — stable for the lifetime of the allocation.
    /// Avoids `device_ptr()` which creates a SyncOnDrop guard that calls
    /// `cuStreamSynchronize` on drop — illegal during CUDA Graph capture.
    cached_ptr: cudarc::driver::sys::CUdeviceptr,
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
        Ok(Self {
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
        Ok(Self {
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

    /// Mutable raw CudaSlice reference for cuBLAS and kernel launches.
    pub fn inner_mut(&mut self) -> &mut cudarc::driver::CudaSlice<f32> {
        &mut self.data
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
    /// IMPORTANT: caller must `stream.synchronize()` before calling this
    /// if async operations (kernels, cuBLAS) are pending on a non-default stream.
    /// `cuMemcpyDtoH_v2` is host-synchronous but does NOT wait for non-default streams.
    pub fn to_cpu(&self) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; self.len];
        if self.len > 0 {
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
    /// Uses raw cuMemcpyHtoD on the slice's device pointer.
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
        }
        Ok(())
    }
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
        Ok(Self {
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

    pub fn inner(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.data
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

    /// Upload f32 data from CPU, converting to dtype on-the-fly.
    pub fn upload_f32(
        &self,
        _stream: &Arc<cudarc::driver::CudaStream>,
        src: &[f32],
    ) -> Result<(), String> {
        assert_eq!(src.len(), self.n_elems, "DtypedBuf upload size mismatch");
        let ptr = self.inner.cached_ptr();
        match self.dtype {
            WeightDtype::F32 => {
                let bytes: &[u8] = bytemuck::cast_slice(src);
                cu_memcpy_htod_raw(ptr, bytes)
            }
            WeightDtype::Bf16 => {
                let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&buf);
                cu_memcpy_htod_raw(ptr, bytes)
            }
            WeightDtype::F16 => {
                let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
                let bytes: &[u8] = bytemuck::cast_slice(&buf);
                cu_memcpy_htod_raw(ptr, bytes)
            }
        }
    }

    /// Download to f32, converting from dtype on-the-fly.
    pub fn download_f32(
        &self,
        _stream: &Arc<cudarc::driver::CudaStream>,
        dst: &mut [f32],
    ) -> Result<(), String> {
        assert_eq!(dst.len(), self.n_elems, "DtypedBuf download size mismatch");
        let ptr = self.inner.cached_ptr();
        match self.dtype {
            WeightDtype::F32 => {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
                cu_memcpy_dtoh_raw(ptr, bytes)
            }
            WeightDtype::Bf16 => {
                let mut buf = vec![half::bf16::ZERO; self.n_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
            WeightDtype::F16 => {
                let mut buf = vec![half::f16::ZERO; self.n_elems];
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                cu_memcpy_dtoh_raw(ptr, bytes)?;
                for (d, &v) in dst.iter_mut().zip(&buf) {
                    *d = v.to_f32();
                }
                Ok(())
            }
        }
    }
}

fn cu_memcpy_htod_raw(dst: cudarc::driver::sys::CUdeviceptr, bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyHtoD_v2(
            dst,
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len(),
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyHtoD: {r:?}"));
    }
    Ok(())
}

fn cu_memcpy_dtoh_raw(
    src: cudarc::driver::sys::CUdeviceptr,
    bytes: &mut [u8],
) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let r = unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            bytes.as_mut_ptr() as *mut std::ffi::c_void,
            src,
            bytes.len(),
        )
    };
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemcpyDtoH: {r:?}"));
    }
    Ok(())
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

    /// Upload f32 CPU data, downcasting to `dtype` on CPU side.
    pub fn upload_from_cpu_f32(&self, src: &[f32]) -> Result<(), String> {
        assert_eq!(src.len(), self.len_elems, "size mismatch");
        if self.len_elems == 0 {
            return Ok(());
        }
        match self.dtype {
            WeightDtype::F32 => self.upload_raw_bytes(bytemuck::cast_slice(src)),
            WeightDtype::Bf16 => {
                let buf: Vec<half::bf16> = src.iter().map(|&v| half::bf16::from_f32(v)).collect();
                self.upload_raw_bytes(bytemuck::cast_slice(&buf))
            }
            WeightDtype::F16 => {
                let buf: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
                self.upload_raw_bytes(bytemuck::cast_slice(&buf))
            }
        }
    }

    /// Upload raw bytes matching this slice's dtype (no conversion).
    /// Caller must ensure `bytes.len() == self.size_bytes()`.
    pub fn upload_raw_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        assert_eq!(bytes.len(), self.size_bytes(), "byte size mismatch");
        let result = unsafe {
            cudarc::driver::sys::cuMemcpyHtoD_v2(
                self.ptr,
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("WeightSliceDyn upload failed: {result:?}"));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    // Tests require CUDA device — run on GPU server only
    // cargo test --features cuda -- gpu
}
