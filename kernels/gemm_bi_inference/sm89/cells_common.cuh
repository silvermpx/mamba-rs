// Shared helpers of the Ada inference cells. Every cell kernel takes
// (C, A, B, bias, FixedSm89HalfParams) with alpha = 1 and beta = 0 on the
// production route; alpha and beta are honored anyway so the epilogues
// match the ladder kernels.
__device__ __forceinline__ unsigned sm89_cell_smem_addr(const void* pointer) {
    return (unsigned)__cvta_generic_to_shared(pointer);
}

// One 16-byte cp.async with a zero-fill length. A zero-length copy still
// needs an in-object source address, so callers pass the allocation base.
__device__ __forceinline__ void sm89_cell_cp_async_16(unsigned destination, const void* source, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(destination), "l"(source), "r"(bytes));
}

__device__ __forceinline__ void sm89_cell_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
__device__ __forceinline__ void sm89_cell_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

__device__ __forceinline__ void sm89_cell_ldmatrix_x4(unsigned address, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

__device__ __forceinline__ void sm89_cell_ldmatrix_x2_trans(unsigned address, unsigned (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(address));
}

__device__ __forceinline__ void sm89_cell_ldmatrix_x4_trans(unsigned address, unsigned (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}

template <typename T> struct Sm89CellHalf;
template <> struct Sm89CellHalf<__nv_bfloat16> {
    static __device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
    static __device__ __forceinline__ unsigned pack(float lo, float hi) {
        __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&pair);
    }
};
template <> struct Sm89CellHalf<__half> {
    static __device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    static __device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
    static __device__ __forceinline__ unsigned pack(float lo, float hi) {
        __half2 pair = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&pair);
    }
};

__device__ __forceinline__ void sm89_cell_mma_tf32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The incumbents convert operands with cvt.rna.tf32.f32 on the raw bits;
// the same instruction keeps every special value's behavior identical.
__device__ __forceinline__ unsigned sm89_cell_tf32(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}
