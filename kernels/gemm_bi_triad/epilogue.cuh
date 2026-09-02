__device__ __forceinline__ void gemm_bi_store_pair_rne(
    __half* dst, float x, float y) {
    *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(x, y);
}

__device__ __forceinline__ void gemm_bi_store_pair_rne(
    __nv_bfloat16* dst, float x, float y) {
    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(x, y);
}

__device__ __forceinline__ void gemm_bi_accumulate_float2_or_scalar(
    float* dst, float x, float y, bool packed) {
    float current_x;
    float current_y;
    if (packed) {
        float2 current = *reinterpret_cast<const float2*>(dst);
        current_x = current.x;
        current_y = current.y;
    } else {
        current_x = dst[0];
        current_y = dst[1];
    }
    current_x += x;
    current_y += y;
    if (packed) {
        float2 result = {current_x, current_y};
        *reinterpret_cast<float2*>(dst) = result;
    } else {
        dst[0] = current_x;
        dst[1] = current_y;
    }
}

template <typename T>
__device__ __forceinline__ T* gemm_bi_output_start_if_valid(
    T* base, long long row_offset, int column, int extent) {
    if (column >= extent) return nullptr;
    return base + row_offset + column;
}

