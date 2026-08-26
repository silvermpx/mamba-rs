__device__ __forceinline__ float4 ld_global_L2_128B(const float* p) {
    float4 v;
    asm("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}
__device__ __forceinline__ bool sgb_is_aligned_4(const void* ptr) {
    return ((unsigned long long)ptr & 3ULL) == 0;
}

__device__ __forceinline__ bool sgb_is_aligned_8(const void* ptr) {
    return ((unsigned long long)ptr & 7ULL) == 0;
}

__device__ __forceinline__ bool sgb_is_aligned_16(const void* ptr) {
    return ((unsigned long long)ptr & 15ULL) == 0;
}

