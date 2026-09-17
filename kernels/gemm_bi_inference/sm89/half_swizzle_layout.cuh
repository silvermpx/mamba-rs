// Production Fixed SM89 homogeneous-half swizzle layout. Shared by the CUDA twin and
// pure-host address/ldmatrix tests; no CUDA toolkit is needed by the latter.
#pragma once
#if defined(__CUDACC__)
#define SM89_FHS_HD __host__ __device__
#else
#define SM89_FHS_HD
#endif
namespace sm89_fixed_half_swizzle_layout {
constexpr int kStageElements = 8192;
constexpr int kOutputStride = 136;
constexpr int kSharedBytes = 69632;
static_assert(4 * kStageElements * 2 <= kSharedBytes, "S2 staging fits");
static_assert(128 * kOutputStride * 4 == kSharedBytes, "unchanged output scratch fits");
SM89_FHS_HD constexpr int a_index(int row, int k) {
    return row * 64 + (k ^ ((row & 7) * 8));
}
SM89_FHS_HD constexpr int b_index(int k, int column) {
    return k * 128 + (column ^ ((k & 7) * 8));
}
SM89_FHS_HD constexpr unsigned a_copy_offset(int thread, int slice) {
    return unsigned(2 * a_index((thread >> 3) + slice * 32, (thread & 7) * 8));
}
SM89_FHS_HD constexpr unsigned b_copy_offset(int thread, int slice) {
    return unsigned(2 * b_index((thread >> 4) + slice * 16, (thread & 15) * 8));
}
SM89_FHS_HD constexpr unsigned a_fragment_base(int warp_m, int atom, int lane) {
    return unsigned(2 * a_index(warp_m + atom * 16 + (lane & 15), (lane & 16) ? 8 : 0));
}
SM89_FHS_HD constexpr unsigned b_fragment_base(int warp_n, int atom, int lane) {
    return unsigned(2 * b_index(lane & 15, warp_n + atom * 8));
}
SM89_FHS_HD constexpr unsigned a_fragment_issue(unsigned base, int issue) {
    // XOR touches only the low seven byte bits; each A row starts at 128B.
    return base ^ unsigned(issue * 32);
}
SM89_FHS_HD constexpr unsigned b_fragment_issue(unsigned base, int issue) {
    // Advancing K by 16 leaves the low-three-row-bit permutation unchanged.
    return base + unsigned(issue * 16 * 128 * 2);
}
} // namespace sm89_fixed_half_swizzle_layout
#undef SM89_FHS_HD
