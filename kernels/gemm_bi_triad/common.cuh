// CUDA parameter bundles use only 4-byte scalar fields. Each bundle also
// proves standard layout, member widths, alignment, and total size; together
// those checks prove exact declaration-order offsets without relying on
// offsetof, which NVRTC's standalone environment does not provide.
static_assert(sizeof(float) == 4, "CUDA float width changed");
static_assert(sizeof(int) == 4, "CUDA int width changed");

__device__ __forceinline__ float4 ld_global_L2_128B(const float* p) {
    float4 v;
    asm("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}
__device__ __forceinline__ bool is_aligned_4(const void* ptr) {
    return ((unsigned long long)ptr & 3ULL) == 0;
}

__device__ __forceinline__ bool is_aligned_8(const void* ptr) {
    return ((unsigned long long)ptr & 7ULL) == 0;
}

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return ((unsigned long long)ptr & 15ULL) == 0;
}
