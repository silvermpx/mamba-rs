// Shared prelude for templated (multi-dtype) kernels.
//
// Pattern: every activation-touching kernel has 3 extern "C" instantiations:
//   NAME_f32, NAME_bf16, NAME_f16
// Suffix is chosen by Rust dispatch based on activation dtype.
//
// All math happens in f32; dtype conversion is upcast-on-load,
// downcast-on-store (single PTX cvt instruction each).
//
// Storage typing: T_IN / T_OUT are the activation dtype.
// Weights that must remain f32 (a_log/a_neg, D, norm weights, biases)
// are passed as `const float*` explicitly.

#ifndef _MAMBA_TYPED_PRELUDE_CUH
#define _MAMBA_TYPED_PRELUDE_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef LOG2E
#define LOG2E 1.4426950408889634f
#endif

// ---- Upcast helpers (load) ------------------------------------------------
__device__ __forceinline__ float to_f(float v)          { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v)         { return __half2float(v); }

// ---- Downcast helpers (store) --------------------------------------------
__device__ __forceinline__ float         from_f_f32(float v)  { return v; }
__device__ __forceinline__ __nv_bfloat16 from_f_bf16(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ __half        from_f_f16(float v)  { return __float2half_rn(v); }

// ---- Packed pair upcast (2 elements at once) ------------------------------
// Halves LDS instruction count for warp-uniform smem reads. Used in matvec
// inner loop where smem_a reads are broadcast to all lanes in a warp.
// Address must be 4-byte aligned (2 elements × sizeof(T_IO)).
__device__ __forceinline__ float2 pair_to_f2(const float* p) {
    return {p[0], p[1]};
}
__device__ __forceinline__ float2 pair_to_f2(const __nv_bfloat16* p) {
    __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p);
    return {__bfloat162float(__low2bfloat16(v)),
            __bfloat162float(__high2bfloat16(v))};
}
__device__ __forceinline__ float2 pair_to_f2(const __half* p) {
    __half2 v = *reinterpret_cast<const __half2*>(p);
    return {__half2float(__low2half(v)),
            __half2float(__high2half(v))};
}

#endif  // _MAMBA_TYPED_PRELUDE_CUH
