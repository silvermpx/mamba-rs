// Test-only shared-destination mapping for the Ada NT compact-layout discovery.
// The Rust harness composes this helper ahead of a count-checked transformation
// of the production SM80 direct template. Global source ownership is unchanged.
__device__ __forceinline__ int nt_test_compact_xor_k(
    int row_or_column, int logical_k) {
    return logical_k ^ ((row_or_column & 7) << 2);
}
