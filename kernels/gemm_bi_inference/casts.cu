// The dtype casts the inference GEMM routes launch around a GEMM (the
// input upcast and the output downcast), in the Inference module so the
// routes' launch census finds them beside the GEMMs; the training paths
// use them too.

#include "_typed_prelude.cuh"

// ---------------------------------------------------------------------------
// Dtype cast kernels — for mixed-precision inference weight upload.
// f32 -> bf16: used when HF checkpoint is f32 but user requested bf16 storage.
// f32 -> f16:  same, but for f16 storage (rare — bf16 preferred for Mamba).
// ---------------------------------------------------------------------------

extern "C" __global__ void cast_f32_to_bf16(
    __nv_bfloat16* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Round-to-nearest, matching the intermediate-value downcasts in all
    // typed kernels (from_f_bf16 in _typed_prelude.cuh). The default
    // `__float2bfloat16` is round-toward-zero, which adds a systematic
    // negative bias to every weight and compounds across GEMMs — visible
    // as degenerate greedy decoding on small models (e.g. mamba-130m).
    dst[i] = __float2bfloat16_rn(src[i]);
}

extern "C" __global__ void cast_f32_to_f16(
    __half* dst, const float* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Same rationale as cast_f32_to_bf16: match the _rn rounding mode used
    // by from_f_f16 in _typed_prelude.cuh.
    dst[i] = __float2half_rn(src[i]);
}

// Typed → f32 casts for the M3 mixed-precision backward,
// where some kernels (rmsnorm_bwd, m3_split_bwd's f32 inputs, etc.)
// are pure-f32 and need a typed staging buffer cast back to f32.
extern "C" __global__ void cast_bf16_to_f32(
    float* dst, const __nv_bfloat16* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

extern "C" __global__ void cast_f16_to_f32(
    float* dst, const __half* src, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}
