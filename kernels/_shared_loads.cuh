// Shared-memory float4 reads written in PTX.
//
// NVVM may split a float4 read of shared memory into a partly used vector
// read plus reloads of single lanes, and which lanes it reloads follows the
// order of its own address-keyed tables, so the same source compiled into
// different PTX from one process to the next. A read written in PTX leaves
// it nothing to choose; the values it delivers and the arithmetic that takes
// them stay the same. The address must lie in shared memory on 16 bytes.
__device__ __forceinline__ float4 lds_f32x4(const float* p) {
    float4 v;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
    return v;
}

// Four consecutive floats of a shared row that need not start on 16
// bytes, read through PTX in one statement: one address and the other three
// at constant offsets. A loop that reads its float4 operands through PTX
// reads the values it combines with them here, in the order it wants them
// issued; NVVM keeps that order, where a plain read after a PTX one would
// wait for its first use.
__device__ __forceinline__ void lds_f32_row4(const float* p, float (&v)[4]) {
    asm volatile("ld.shared.f32 %0, [%4];\n\t"
                 "ld.shared.f32 %1, [%4+4];\n\t"
                 "ld.shared.f32 %2, [%4+8];\n\t"
                 "ld.shared.f32 %3, [%4+12];\n"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}

// Sixteen floats of shared memory (16-byte aligned) as four float4, and
// sixteen consecutive floats of a row that need not be aligned, each in one
// PTX statement at constant offsets from one address: the shape an
// unrolled sixteen-index step reads, with no address arithmetic between.
__device__ __forceinline__ void lds_f32x16(const float* p, float4 (&v)[4]) {
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%16];\n\t"
                 "ld.shared.v4.f32 {%4,%5,%6,%7}, [%16+16];\n\t"
                 "ld.shared.v4.f32 {%8,%9,%10,%11}, [%16+32];\n\t"
                 "ld.shared.v4.f32 {%12,%13,%14,%15}, [%16+48];\n"
                 : "=f"(v[0].x), "=f"(v[0].y), "=f"(v[0].z), "=f"(v[0].w),
                   "=f"(v[1].x), "=f"(v[1].y), "=f"(v[1].z), "=f"(v[1].w),
                   "=f"(v[2].x), "=f"(v[2].y), "=f"(v[2].z), "=f"(v[2].w),
                   "=f"(v[3].x), "=f"(v[3].y), "=f"(v[3].z), "=f"(v[3].w)
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}

// Four floats a run-time stride apart (one from each of four rows of a
// tile), and four float4 so placed (each on 16 bytes), read through PTX in
// one statement in row order: a reduction over rows takes four rows' reads
// together ahead of their arithmetic, where one row's PTX read at a time
// would hold every later read of the loop behind it.
__device__ __forceinline__ void lds_f32_rows4(const float* p, int stride, float (&v)[4]) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    const unsigned s = (unsigned)stride * 4u;
    asm volatile("ld.shared.f32 %0, [%4];\n\t"
                 "ld.shared.f32 %1, [%5];\n\t"
                 "ld.shared.f32 %2, [%6];\n\t"
                 "ld.shared.f32 %3, [%7];\n"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3])
                 : "r"(a), "r"(a + s), "r"(a + 2u * s), "r"(a + 3u * s));
}

__device__ __forceinline__ void lds_f32x4_rows4(const float* p, int stride, float4 (&v)[4]) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    const unsigned s = (unsigned)stride * 4u;
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%16];\n\t"
                 "ld.shared.v4.f32 {%4,%5,%6,%7}, [%17];\n\t"
                 "ld.shared.v4.f32 {%8,%9,%10,%11}, [%18];\n\t"
                 "ld.shared.v4.f32 {%12,%13,%14,%15}, [%19];\n"
                 : "=f"(v[0].x), "=f"(v[0].y), "=f"(v[0].z), "=f"(v[0].w),
                   "=f"(v[1].x), "=f"(v[1].y), "=f"(v[1].z), "=f"(v[1].w),
                   "=f"(v[2].x), "=f"(v[2].y), "=f"(v[2].z), "=f"(v[2].w),
                   "=f"(v[3].x), "=f"(v[3].y), "=f"(v[3].z), "=f"(v[3].w)
                 : "r"(a), "r"(a + s), "r"(a + 2u * s), "r"(a + 3u * s));
}

__device__ __forceinline__ void lds_f32_row16(const float* p, float (&v)[16]) {
    asm volatile("ld.shared.f32 %0, [%16];\n\t"
                 "ld.shared.f32 %1, [%16+4];\n\t"
                 "ld.shared.f32 %2, [%16+8];\n\t"
                 "ld.shared.f32 %3, [%16+12];\n\t"
                 "ld.shared.f32 %4, [%16+16];\n\t"
                 "ld.shared.f32 %5, [%16+20];\n\t"
                 "ld.shared.f32 %6, [%16+24];\n\t"
                 "ld.shared.f32 %7, [%16+28];\n\t"
                 "ld.shared.f32 %8, [%16+32];\n\t"
                 "ld.shared.f32 %9, [%16+36];\n\t"
                 "ld.shared.f32 %10, [%16+40];\n\t"
                 "ld.shared.f32 %11, [%16+44];\n\t"
                 "ld.shared.f32 %12, [%16+48];\n\t"
                 "ld.shared.f32 %13, [%16+52];\n\t"
                 "ld.shared.f32 %14, [%16+56];\n\t"
                 "ld.shared.f32 %15, [%16+60];\n"
                 : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]),
                   "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]),
                   "=f"(v[8]), "=f"(v[9]), "=f"(v[10]), "=f"(v[11]),
                   "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15])
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
}
