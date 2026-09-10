#ifndef GEMM_BI_TF32_NT_PADDED_LDMATRIX_CUH
#define GEMM_BI_TF32_NT_PADDED_LDMATRIX_CUH

struct GemmBiTf32NtPaddedLdmatrixAddress {
    int row;
    int reduction;
};

#if defined(__CUDACC__)
#define GEMM_BI_TF32_NT_PADDED_MAP_INLINE __host__ __device__ __forceinline__ constexpr
#else
#define GEMM_BI_TF32_NT_PADDED_MAP_INLINE constexpr
#endif

GEMM_BI_TF32_NT_PADDED_MAP_INLINE GemmBiTf32NtPaddedLdmatrixAddress
gemm_bi_tf32_nt_padded_a_address(int warp_m, int atom, int k8, int lane) {
    return {
        warp_m + atom * 16 + (lane & 15),
        k8 + ((lane >> 4) << 2),
    };
}

GEMM_BI_TF32_NT_PADDED_MAP_INLINE GemmBiTf32NtPaddedLdmatrixAddress
gemm_bi_tf32_nt_padded_b_address(int warp_n, int atom, int k8, int lane) {
    return {
        warp_n + atom * 8 + (lane & 7),
        k8 + (((lane >> 3) & 1) << 2),
    };
}

#undef GEMM_BI_TF32_NT_PADDED_MAP_INLINE

#if defined(__CUDACC__)

template <int BM, int BN, int Stages, int MAtoms, int NAtoms>
__device__ __forceinline__ void gemm_bi_tf32_nt_padded_ldmatrix_fragments(
    SgbTf32Storage<SgbTf32Nt, BM, BN, Stages>* storage,
    int stage, int warp_m, int warp_n, int k8,
    unsigned (&a_fragments)[MAtoms][4],
    unsigned (&b_fragments)[NAtoms][2]) {
    static_assert(BM == 128 && BN == 64 && Stages == 3,
                  "padded NT ldmatrix helper is only for the M128N64 BK32 S3 candidate");
    static_assert(MAtoms == 4 && NAtoms == 4,
                  "padded NT ldmatrix helper fragment shape changed");

    const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
    for (int atom = 0; atom < MAtoms; ++atom) {
        const auto source = gemm_bi_tf32_nt_padded_a_address(warp_m, atom, k8, lane);
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
            &gemm_bi_tf32_a_slot<SgbTf32Nt>(
                storage, stage, source.row, source.reduction)));
        unsigned raw0, raw1, raw2, raw3;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
            : "r"(address));
        a_fragments[atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
        a_fragments[atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
        a_fragments[atom][2] = gemm_bi_tf32_rna(__uint_as_float(raw2));
        a_fragments[atom][3] = gemm_bi_tf32_rna(__uint_as_float(raw3));
    }

#pragma unroll
    for (int atom = 0; atom < NAtoms; ++atom) {
        const auto source = gemm_bi_tf32_nt_padded_b_address(warp_n, atom, k8, lane);
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(
            &gemm_bi_tf32_b_slot<SgbTf32Nt>(
                storage, stage, source.reduction, source.row)));
        unsigned raw0, raw1;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(raw0), "=r"(raw1)
            : "r"(address));
        b_fragments[atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
        b_fragments[atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
    }
}

#endif  // defined(__CUDACC__)

#endif  // GEMM_BI_TF32_NT_PADDED_LDMATRIX_CUH
