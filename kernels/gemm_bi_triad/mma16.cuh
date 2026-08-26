__device__ __forceinline__ int sgb_cp_async_valid_elems(
    bool row_valid, int extent, int start) {
    if (!row_valid || start >= extent) return 0;
    int remaining = extent - start;
    return remaining < 8 ? remaining : 8;
}

template <typename T>
__device__ __forceinline__ const T* sgb_cp_async_source(
    const T* base, long long valid_offset, int valid_bytes) {
    // PTX ignores the source for a zero-byte lane, but C++ still requires
    // the address expression itself to remain inside the allocation.
    return valid_bytes == 0 ? base : base + valid_offset;
}

__device__ __forceinline__ void sgb_cp_async_16_zfill(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

