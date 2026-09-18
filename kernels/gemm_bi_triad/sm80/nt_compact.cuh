// Shared-destination mapping for the Ada NT compact-layout finalist.
// Global source ownership is unchanged; both writers and readers use this
// bijection so each aligned four-float group remains contiguous.
__device__ __forceinline__ int nt_compact8_xor_k(
    int row_or_column, int logical_k) {
    return logical_k ^ ((row_or_column & 7) << 2);
}
