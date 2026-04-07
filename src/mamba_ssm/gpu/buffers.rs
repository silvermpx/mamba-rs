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
