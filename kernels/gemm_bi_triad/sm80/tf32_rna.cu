// The batch-invariant TF32 forward's wide and multistage NN tiles, one
// module of their own on every SM80-tier target so that no other module's
// identity moves with them. Every tile here walks an output's reduction the
// way the portable tiles do: the accumulator seeded with the bias, both
// operands rounded with cvt.rna.tf32.f32, ascending k8 steps over the
// reduction padded with zero-filled slabs to a whole number of 32-deep
// groups, then alpha and beta as the portable epilogue applies them. The
// tile a row count picks is therefore scheduling only and never moves a bit.
//
// The bodies are the portable tier's helpers, the wide NN template, the
// 128 x 128 tile and the multistage tiles, copied with their operand
// rounding set to cvt.rna. Every tile copies whole 16-byte chunks: both
// operand strides must be multiples of four floats and both bases 16-byte
// aligned.

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return ((unsigned long long)ptr & 15ULL) == 0;
}

__device__ __forceinline__ void cp_async_16_zfill_l2(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

struct Sm80Tf32KernelParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};

static_assert(sizeof(Sm80Tf32KernelParams) == 32, "TF32 parameter ABI drift");
static_assert(alignof(Sm80Tf32KernelParams) == 4, "TF32 parameter alignment drift");
static_assert(__is_standard_layout(Sm80Tf32KernelParams),
              "TF32 parameters must remain standard layout");
// Eight ordered 4-byte fields in 32 bytes leave no internal or tail padding.
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->alpha) == 4,
              "TF32 alpha size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->beta) == 4,
              "TF32 beta size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->m) == 4,
              "TF32 M size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->k) == 4,
              "TF32 K size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->n) == 4,
              "TF32 N size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->lda) == 4,
              "TF32 A stride size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->ldb) == 4,
              "TF32 B stride size changed");
static_assert(sizeof(((Sm80Tf32KernelParams*)0)->ldc) == 4,
              "TF32 output stride size changed");

enum SgbTf32Op { SgbTf32Nn, SgbTf32Tn, SgbTf32Nt };

__device__ __forceinline__ void tf32_mma_m16n8k8(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_rows(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Tn ? p.k : p.m;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_columns(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nt ? p.k : p.n;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_reduction(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nn ? p.k : (Op == SgbTf32Tn ? p.m : p.n);
}

template <SgbTf32Op Op>
__device__ __forceinline__ float tf32_epilogue(
    float accumulator, float old_output, const float* bias, int column,
    const Sm80Tf32KernelParams& params) {
    if constexpr (Op == SgbTf32Nn) {
        (void)bias;
        (void)column;
        float value = params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
        if (params.beta == 0.0f) return value;
        return __fmaf_rn(params.beta, old_output, value);
    } else if constexpr (Op == SgbTf32Tn) {
        (void)bias;
        (void)column;
        return __fmaf_rn(params.alpha, accumulator, old_output);
    } else {
        (void)old_output;
        (void)bias;
        (void)column;
        return params.alpha == 1.0f
            ? accumulator
            : __fmul_rn(params.alpha, accumulator);
    }
}

template <SgbTf32Op Op>
__device__ __forceinline__ void tf32_store(
    float* output, int row, int column, float accumulator,
    const float* bias, const Sm80Tf32KernelParams& params) {
    int rows = tf32_rows<Op>(params);
    int columns = tf32_columns<Op>(params);
    if (row >= rows || column >= columns) return;
    float* destination = output + (long long)row * params.ldc + column;
    float old_output = 0.0f;
    if constexpr (Op == SgbTf32Tn) {
        old_output = *destination;
    } else if constexpr (Op == SgbTf32Nn) {
        if (params.beta != 0.0f) old_output = *destination;
    }
    float value = tf32_epilogue<Op>(accumulator, old_output, bias, column, params);
    *destination = value;
}

template <SgbTf32Op Op, int BM, int BN>
__device__ __forceinline__ void tf32_zero_reduction_epilogue(
    float* output, const float* bias, const Sm80Tf32KernelParams& params) {
    (void)&tf32_epilogue<Op>;
    int rows = tf32_rows<Op>(params);
    int columns = tf32_columns<Op>(params);
    int tile_row = (int)blockIdx.x / ((columns + BN - 1) / BN) * BM;
    int tile_column = (int)blockIdx.x % ((columns + BN - 1) / BN) * BN;
    for (int linear = (int)threadIdx.x; linear < BM * BN; linear += (int)blockDim.x) {
        int row = tile_row + linear / BN;
        int column = tile_column + linear % BN;
        if (row < rows && column < columns) {
            float accumulator = Op == SgbTf32Nn && bias != nullptr ? bias[column] : 0.0f;
            tf32_store<Op>(output, row, column, accumulator, bias, params);
        }
    }
}

// ---------------------------------------------------------------------------
// The wide NN template: a row-major A plane fed by ldmatrix, B by scalar
// loads, Mode 1 rounding both operands with cvt.rna.
// ---------------------------------------------------------------------------

struct NnWideParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(NnWideParams) == 32, "NN parameter ABI");

__device__ __forceinline__ void nn_wide_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void nn_wide_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
__device__ __forceinline__ void nn_wide_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

__device__ __forceinline__ void nn_wide_ldmatrix_x4(
    unsigned (&r)[4], unsigned address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(address));
}

__device__ __forceinline__ unsigned nn_wide_lds(unsigned address) {
    unsigned value;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(value) : "r"(address));
    return value;
}

__device__ __forceinline__ void nn_wide_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// Half an ulp added in floating point from the operand's own exponent: normal
// values round as with cvt.rna, a NaN stays a NaN.
__device__ __forceinline__ unsigned nn_wide_add_half(unsigned bits) {
    return __float_as_uint(fmaf(__uint_as_float(bits & 0xff800000U), 1.0f / 2048.0f, __uint_as_float(bits)));
}

__device__ __forceinline__ unsigned nn_wide_rna(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

// A plane: [row][32 reduction] with the chunk index folded with the row.
__device__ __forceinline__ int nn_wide_a_slot(int row, int reduction) {
    return row * 32 + (reduction ^ ((row & 7) << 2));
}

// B plane: [32 reduction][BStride columns] with the chunk index folded with
// the reduction row, so the four rows a fragment reads land on distinct
// banks. A tile narrower than its storage stride simply leaves the last
// chunks of every row unused.
template <int BStride>
__device__ __forceinline__ int nn_wide_b_slot(int reduction, int column) {
    int chunk = (column >> 2) ^ ((reduction & 3) << 1);
    return reduction * BStride + chunk * 4 + (column & 3);
}

template <int Slices>
struct NnWidePlan {
    const float* source[Slices];
    unsigned destination[Slices];
    int bytes_cap[Slices];
};

template <int BM, int BN, int WarpsM, int WarpsN, int Stages>
struct NnWideConfig {
    static constexpr int Threads = 32 * WarpsM * WarpsN;
    static constexpr int WM = BM / WarpsM;
    static constexpr int WN = BN / WarpsN;
    static constexpr int MAtoms = WM / 16;
    static constexpr int NAtoms = WN / 8;
    static constexpr int BStride = (BN + 31) / 32 * 32;
    static constexpr int AChunks = BM * 8;
    static constexpr int BChunks = 32 * (BN / 4);
    static constexpr int ASlices = (AChunks + Threads - 1) / Threads;
    static constexpr int BSlices = (BChunks + Threads - 1) / Threads;
    static constexpr unsigned AStageBytes = (unsigned)BM * 128U;
    static constexpr unsigned BStageBytes = (unsigned)BStride * 128U;
    static constexpr unsigned SharedBytes = (AStageBytes + BStageBytes) * (unsigned)Stages;
    static_assert(WM % 16 == 0, "warp rows must be whole m16 atoms");
    static_assert(WN % 8 == 0, "warp columns must be whole n8 atoms");
    static_assert(BN % 8 == 0, "B rows must be whole n8 atoms");
    static_assert(Stages >= 2, "at least two stages");
};

template <int MAtoms, int NAtoms>
struct NnWideFragments {
    unsigned a[MAtoms][4];
    unsigned b[NAtoms][2];
};

// Rounding modes: 0 = half-ulp add on both operands (the retained Ada NN
// kernels), 1 = cvt.rna on both (the retained portable kernels), 2 = A
// already rounded by the retained transpose pass, cvt.rna on B (TN).
template <int Mode>
__device__ __forceinline__ unsigned nn_wide_round_a(unsigned bits) {
    if constexpr (Mode == 0) return nn_wide_add_half(bits);
    else if constexpr (Mode == 1) return nn_wide_rna(bits);
    else return bits;
}

template <int Mode>
__device__ __forceinline__ unsigned nn_wide_round_b(unsigned bits) {
    if constexpr (Mode == 0) return nn_wide_add_half(bits);
    else return nn_wide_rna(bits);
}

template <int Mode, int BStride, int MAtoms, int NAtoms>
__device__ __forceinline__ void nn_wide_load_fragments(
    unsigned a_address, const unsigned (&b_address)[NAtoms], unsigned b_stage,
    int step, NnWideFragments<MAtoms, NAtoms>& fragments) {
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
        unsigned raw[4];
        nn_wide_ldmatrix_x4(raw, a_address + (unsigned)m_atom * 2048U);
        fragments.a[m_atom][0] = nn_wide_round_a<Mode>(raw[0]);
        fragments.a[m_atom][1] = nn_wide_round_a<Mode>(raw[1]);
        fragments.a[m_atom][2] = nn_wide_round_a<Mode>(raw[2]);
        fragments.a[m_atom][3] = nn_wide_round_a<Mode>(raw[3]);
    }
    // Each n-atom keeps its own swizzled address of (reduction = thread,
    // column = warp_n + n_atom * 8 + group) in stage 0; the row four below
    // shares the swizzle, and so does every later k8 step.
    unsigned step_bytes = b_stage + (unsigned)step * 8U * (unsigned)BStride * 4U;
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        unsigned lo = nn_wide_lds(b_address[n_atom] + step_bytes);
        unsigned hi = nn_wide_lds(b_address[n_atom] + step_bytes + 4U * (unsigned)BStride * 4U);
        fragments.b[n_atom][0] = nn_wide_round_b<Mode>(lo);
        fragments.b[n_atom][1] = nn_wide_round_b<Mode>(hi);
    }
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void nn_wide_mma_step(
    const NnWideFragments<MAtoms, NAtoms>& fragments,
    float (&acc)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            nn_wide_mma(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

template <int Mode, int BM, int BN, int WarpsM, int WarpsN, int Stages>
__device__ __forceinline__ void nn_wide_kernel(
    float* output, const float* a, const float* b, const float* bias,
    NnWideParams params) {
    using Cfg = NnWideConfig<BM, BN, WarpsM, WarpsN, Stages>;
    constexpr bool TnPre = Mode == 2;
    constexpr int Threads = Cfg::Threads;
    constexpr int MAtoms = Cfg::MAtoms;
    constexpr int NAtoms = Cfg::NAtoms;
    constexpr int ASlices = Cfg::ASlices;
    constexpr int BSlices = Cfg::BSlices;
    constexpr int BStride = Cfg::BStride;
    constexpr int IssueSlots = ASlices > BSlices ? ASlices : BSlices;

    int column_tiles = (params.n + BN - 1) / BN;
    int tile_row = (int)blockIdx.x / column_tiles * BM;
    int tile_column = (int)blockIdx.x % column_tiles * BN;

    extern __shared__ __align__(16) unsigned char shared_bytes[];
    unsigned a_plane = (unsigned)__cvta_generic_to_shared(shared_bytes);
    unsigned b_plane = a_plane + Cfg::AStageBytes * (unsigned)Stages;

    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp / WarpsN) * Cfg::WM;
    int warp_n = (warp % WarpsN) * Cfg::WN;
    int group = lane >> 2;
    int thread = lane & 3;

    float acc[MAtoms][NAtoms][4];
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                float seed = 0.0f;
                if constexpr (!TnPre) {
                    int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                    int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                    if (row < params.m && column < params.n && bias != nullptr) {
                        seed = bias[column];
                    }
                }
                acc[m_atom][n_atom][element] = seed;
            }
        }
    }

    // Per-step A ldmatrix addresses in stage 0; the B scalar address of
    // step 0 (rows thread and thread+4 are 4 rows apart, a fixed byte offset).
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
    unsigned a_step_address[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        a_step_address[step] = a_plane + (unsigned)nn_wide_a_slot(a_row, step * 8 + a_k) * 4U;
    }
    unsigned b_address[NAtoms];
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        b_address[n_atom] = b_plane
            + (unsigned)nn_wide_b_slot<BStride>(thread, warp_n + n_atom * 8 + group) * 4U;
    }

    // Copy plans: A chunks are (row, 16-byte chunk of the 32 reduction
    // floats); B chunks are (reduction row, 16-byte chunk of the BN columns).
    NnWidePlan<ASlices> a_plan;
    int a_chunk_offset = ((int)threadIdx.x & 7) * 4;
#pragma unroll
    for (int slice = 0; slice < ASlices; ++slice) {
        int linear = (int)threadIdx.x + slice * Threads;
        int row = linear >> 3;
        bool in_tile = linear < Cfg::AChunks;
        int global_row = tile_row + row;
        bool valid = in_tile && global_row < params.m;
        a_plan.source[slice] = a + (long long)(valid ? global_row : 0) * params.lda + a_chunk_offset;
        a_plan.destination[slice] = in_tile ? a_plane + (unsigned)nn_wide_a_slot(row, a_chunk_offset) * 4U : a_plane;
        a_plan.bytes_cap[slice] = valid ? 16 : 0;
    }
    NnWidePlan<BSlices> b_plan;
    int b_row_offset[BSlices];
#pragma unroll
    for (int slice = 0; slice < BSlices; ++slice) {
        int linear = (int)threadIdx.x + slice * Threads;
        int k_row = linear / (BN / 4);
        int column = (linear % (BN / 4)) * 4;
        bool in_tile = linear < Cfg::BChunks;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        if (!in_tile) columns = 0;
        b_row_offset[slice] = k_row;
        b_plan.source[slice] = b + (long long)(in_tile ? k_row : 0) * params.ldb + (columns > 0 ? global_column : 0);
        b_plan.destination[slice] = in_tile ? b_plane + (unsigned)nn_wide_b_slot<BStride>(k_row, column) * 4U : b_plane;
        b_plan.bytes_cap[slice] = columns * 4;
    }
    long long b_tile_rows = 32LL * params.ldb;

    unsigned tile_count = ((unsigned)params.k + 31U) / 32U;

    auto issue = [&](int slot, unsigned write_a, unsigned write_b, int reduction_base) {
        if (slot < ASlices) {
            bool in_tile = (int)threadIdx.x + slot * Threads < Cfg::AChunks;
            if (in_tile) {
                int remaining = params.k - reduction_base - a_chunk_offset;
                remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
                int bytes = remaining * 4;
                bytes = bytes < a_plan.bytes_cap[slot] ? bytes : a_plan.bytes_cap[slot];
                nn_wide_copy_cg(a_plan.destination[slot] + write_a, a_plan.source[slot], bytes);
            }
        }
        if (slot < BSlices) {
            bool in_tile = (int)threadIdx.x + slot * Threads < Cfg::BChunks;
            if (in_tile) {
                int bytes = reduction_base + b_row_offset[slot] < params.k ? b_plan.bytes_cap[slot] : 0;
                nn_wide_copy_cg(b_plan.destination[slot] + write_b, b_plan.source[slot], bytes);
            }
        }
    };
    auto advance = [&]() {
#pragma unroll
        for (int slice = 0; slice < ASlices; ++slice) a_plan.source[slice] += 32;
#pragma unroll
        for (int slice = 0; slice < BSlices; ++slice) b_plan.source[slice] += b_tile_rows;
    };

#pragma unroll
    for (int stage = 0; stage < Stages - 1; ++stage) {
        if ((unsigned)stage < tile_count) {
#pragma unroll
            for (int slot = 0; slot < IssueSlots; ++slot) {
                issue(slot, Cfg::AStageBytes * (unsigned)stage, Cfg::BStageBytes * (unsigned)stage, stage * 32);
            }
            advance();
        }
        nn_wide_commit();
    }

    for (unsigned tile_base = 0; tile_base < tile_count; tile_base += (unsigned)Stages) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            unsigned tile = tile_base + (unsigned)stage;
            if (tile < tile_count) {
                nn_wide_wait<Stages - 2>();
                __syncthreads();
                unsigned next = tile + (unsigned)(Stages - 1);
                bool has_next = next < tile_count;
                int write_stage = (stage + Stages - 1) % Stages;
                unsigned write_a = Cfg::AStageBytes * (unsigned)write_stage;
                unsigned write_b = Cfg::BStageBytes * (unsigned)write_stage;
                unsigned read_a = Cfg::AStageBytes * (unsigned)stage;
                unsigned read_b = Cfg::BStageBytes * (unsigned)stage;

                NnWideFragments<MAtoms, NAtoms> fragments[2];
                nn_wide_load_fragments<Mode, BStride, MAtoms, NAtoms>(
                    a_step_address[0] + read_a, b_address, read_b, 0, fragments[0]);
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    if (has_next) {
#pragma unroll
                        for (int slot = step; slot < IssueSlots; slot += 4) {
                            issue(slot, write_a, write_b, (int)next * 32);
                        }
                    }
                    if (step < 3) {
                        nn_wide_load_fragments<Mode, BStride, MAtoms, NAtoms>(
                            a_step_address[step + 1] + read_a, b_address, read_b, step + 1,
                            fragments[(step + 1) & 1]);
                    }
                    nn_wide_mma_step<MAtoms, NAtoms>(fragments[step & 1], acc);
                }
                nn_wide_commit();
                if (has_next) advance();
            }
        }
    }
    nn_wide_wait<0>();

    bool paired = (params.ldc & 1) == 0
        && ((reinterpret_cast<unsigned long long>(output) & 7ULL) == 0ULL);
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int row = tile_row + warp_m + m_atom * 16 + group + half * 8;
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread;
                if (row >= params.m) continue;
                float v0 = acc[m_atom][n_atom][2 * half];
                float v1 = acc[m_atom][n_atom][2 * half + 1];
                float* destination = output + (long long)row * params.ldc + column;
                if constexpr (TnPre) {
                    if (paired && column + 1 < params.n) {
                        float2 old = *reinterpret_cast<const float2*>(destination);
                        *reinterpret_cast<float2*>(destination) = make_float2(
                            __fmaf_rn(params.alpha, v0, old.x), __fmaf_rn(params.alpha, v1, old.y));
                    } else {
                        if (column < params.n) destination[0] = __fmaf_rn(params.alpha, v0, destination[0]);
                        if (column + 1 < params.n) destination[1] = __fmaf_rn(params.alpha, v1, destination[1]);
                    }
                } else {
                    if (params.alpha != 1.0f) {
                        v0 = __fmul_rn(params.alpha, v0);
                        v1 = __fmul_rn(params.alpha, v1);
                    }
                    if (paired && column + 1 < params.n) {
                        if (params.beta != 0.0f) {
                            float2 old = *reinterpret_cast<const float2*>(destination);
                            v0 = __fmaf_rn(params.beta, old.x, v0);
                            v1 = __fmaf_rn(params.beta, old.y, v1);
                        }
                        *reinterpret_cast<float2*>(destination) = make_float2(v0, v1);
                    } else {
                        if (column < params.n) {
                            if (params.beta != 0.0f) v0 = __fmaf_rn(params.beta, destination[0], v0);
                            destination[0] = v0;
                        }
                        if (column + 1 < params.n) {
                            if (params.beta != 0.0f) v1 = __fmaf_rn(params.beta, destination[1], v1);
                            destination[1] = v1;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The 128 x 128 tile on eight computing warps.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int tf32w_a_index(int row, int k) {
    int chunk = (k >> 2) ^ (row & 7);
    return row * 32 + chunk * 4 + (k & 3);
}

__device__ __forceinline__ int tf32w_b_index(int k, int column) {
    int chunk = (column >> 2) ^ ((k & 3) << 1);
    return k * 128 + chunk * 4 + (column & 3);
}

// One quarter of a stage's copies: the A and B rows this thread owns in
// slice `slice` (two 16-byte cp.async per thread), so the eight copies of a
// stage interleave with the four k8 steps of the stage being computed.
// Each thread owns eight 16-byte copies per stage (four of A, four of B):
// their global pointers, shared destinations and the row / column bounds
// depend on the tile and the thread only, so they are computed once; a
// stage advances the pointers by its k slab and clamps the copy length by
// the remaining reduction (zero bytes zero-fill the slot; a zero-length
// copy keeps a valid address).
struct Tf32wCopyPlan {
    const float* a_source[4];
    const float* b_source[4];
    unsigned a_destination[4];
    unsigned b_destination[4];
    int a_k_offset;
    int b_k_row;
    bool a_row_valid[4];
    int b_column_bytes[4];
};

__device__ __forceinline__ void tf32w_copy_plan(
    float* a_stage0, float* b_stage0, const float* a, const float* b,
    const Sm80Tf32KernelParams& params, int tile_row, int tile_column, Tf32wCopyPlan& plan) {
    plan.a_k_offset = ((int)threadIdx.x & 7) * 4;
    plan.b_k_row = (int)threadIdx.x >> 5;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        int linear = (int)threadIdx.x + slice * 256;
        int row = linear >> 3;
        int global_row = tile_row + row;
        plan.a_row_valid[slice] = global_row < params.m;
        plan.a_source[slice] = a + (long long)(plan.a_row_valid[slice] ? global_row : 0) * params.lda + plan.a_k_offset;
        plan.a_destination[slice] = (unsigned)__cvta_generic_to_shared(a_stage0 + tf32w_a_index(row, plan.a_k_offset));
        int k_row = linear >> 5;
        int column = (linear & 31) * 4;
        int global_column = tile_column + column;
        int columns = params.n - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        plan.b_column_bytes[slice] = columns * 4;
        plan.b_source[slice] = b + (long long)k_row * params.ldb + (columns > 0 ? global_column : 0);
        plan.b_destination[slice] = (unsigned)__cvta_generic_to_shared(b_stage0 + tf32w_b_index(k_row, column));
    }
}

// One quarter of a stage's copies. `stage_bytes` is the stage's byte offset
// in the ring, `k_base` the stage's first k; the pointers already point at
// the stage (the caller advances them).
__device__ __forceinline__ void tf32w_stage_slice(
    const Tf32wCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction, int slice) {
    {
        int remaining = reduction - k_base - plan.a_k_offset;
        remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
        int bytes = plan.a_row_valid[slice] ? remaining * 4 : 0;
        cp_async_16_zfill_l2(plan.a_destination[slice] + stage_bytes, plan.a_source[slice], bytes);
    }
    {
        int bytes = k_base + plan.b_k_row + slice * 8 < reduction ? plan.b_column_bytes[slice] : 0;
        cp_async_16_zfill_l2(plan.b_destination[slice] + stage_bytes, plan.b_source[slice], bytes);
    }
}

// A whole 16-byte chunk, for a stage whose K slab lies inside the
// reduction of a launch whose column count is a multiple of four (no B
// chunk straddles a row end): no length to clamp, so the copy is the
// address alone. A row or column past the matrix reads the clamped address
// the plan keeps, and the outputs it feeds are never stored.
__device__ __forceinline__ void tf32w_copy_whole(unsigned shared_dst, const void* global_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_dst), "l"(global_src));
}

__device__ __forceinline__ void tf32w_stage_slice_whole(
    const Tf32wCopyPlan& plan, unsigned stage_bytes, int slice) {
    tf32w_copy_whole(plan.a_destination[slice] + stage_bytes, plan.a_source[slice]);
    tf32w_copy_whole(plan.b_destination[slice] + stage_bytes, plan.b_source[slice]);
}

__device__ __forceinline__ void tf32w_advance_plan(Tf32wCopyPlan& plan, long long b_rows) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        plan.a_source[slice] += 32;
        plan.b_source[slice] += b_rows;
    }
}

__device__ __forceinline__ void tf32w_stage_async(
    const Tf32wCopyPlan& plan, unsigned stage_bytes, int k_base, int reduction) {
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        tf32w_stage_slice(plan, stage_bytes, k_base, reduction, slice);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

struct Tf32wFragments {
    unsigned a[4][4];
    unsigned b[4][2];
};

// The shared-memory element offsets of this lane's fragments, computed once
// per thread: the swizzle of A depends on the k8 step through its chunk
// index, so the ldmatrix offsets are kept per (atom, step); the swizzle of B
// depends only on the lane's k row within the step, so one offset per
// (atom, half) serves every step with the step's row term added.
struct Tf32wFragmentOffsets {
    int a[4][4];
    int b[4][2];
};

__device__ __forceinline__ void tf32w_fragment_offsets(
    int warp_m, int warp_n, int group, int thread, int lane, Tf32wFragmentOffsets& offsets) {
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            offsets.a[m_atom][step] = tf32w_a_index(a_row + m_atom * 16, step * 8 + a_k);
        }
    }
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        int column = warp_n + n_atom * 8 + group;
        offsets.b[n_atom][0] = tf32w_b_index(thread, column);
        offsets.b[n_atom][1] = tf32w_b_index(thread + 4, column);
    }
}

// TF32 rounding with cvt.rna.tf32.f32: to nearest with ties away from zero,
// subnormals included, as every other tile of the batch-invariant policy and
// its CPU model round.
__device__ __forceinline__ unsigned tf32w_round(unsigned bits) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(__uint_as_float(bits)));
    return result;
}

// The A fragment of one 16 x 8 atom through ldmatrix.x4: lanes 0-7 address
// rows 0-7 at k8, 8-15 rows 8-15 at k8, 16-23 rows 0-7 at k8 + 4 and 24-31
// rows 8-15 at k8 + 4, so the four registers land as a0..a3 of the mma
// (row g / row g + 8 at k t, then the same at k t + 4). Every row address is
// a 16-byte chunk of the swizzled stage.
__device__ __forceinline__ void tf32w_load_fragments(
    const float* a_stage, const float* b_stage, int step,
    const Tf32wFragmentOffsets& offsets, Tf32wFragments& fragments) {
    unsigned a_base = (unsigned)__cvta_generic_to_shared(a_stage);
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
        unsigned address = a_base + (unsigned)offsets.a[m_atom][step] * 4U;
        unsigned raw0, raw1, raw2, raw3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                     : "r"(address));
        fragments.a[m_atom][0] = tf32w_round(raw0);
        fragments.a[m_atom][1] = tf32w_round(raw1);
        fragments.a[m_atom][2] = tf32w_round(raw2);
        fragments.a[m_atom][3] = tf32w_round(raw3);
    }
    const float* b_step = b_stage + step * 8 * 128;
#pragma unroll
    for (int n_atom = 0; n_atom < 4; ++n_atom) {
        fragments.b[n_atom][0] = tf32w_round(__float_as_uint(b_step[offsets.b[n_atom][0]]));
        fragments.b[n_atom][1] = tf32w_round(__float_as_uint(b_step[offsets.b[n_atom][1]]));
    }
}

__device__ __forceinline__ void tf32w_mma(const Tf32wFragments& fragments, float (&acc)[4][4][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
            tf32_mma_m16n8k8(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

// The stage one tile of the main loop reads and the stage it fills.
struct Tf32wTileStages {
    const float* a_read;
    const float* b_read;
    unsigned write_bytes;
    int fill_k;
    bool fills;
};

// The four k8 steps of one tile with the next stage's copies spread over
// them; `Whole` is the copy form of a stage whose slab lies inside the
// reduction.
template <bool Whole>
__device__ __forceinline__ void tf32w_tile_steps(
    const Tf32wCopyPlan& plan, const Tf32wFragmentOffsets& offsets,
    const Tf32wTileStages& stages, int reduction, Tf32wFragments (&fragments)[2],
    float (&acc)[4][4][4]) {
#pragma unroll
    for (int issue = 0; issue < 4; ++issue) {
        if (Whole) {
            tf32w_stage_slice_whole(plan, stages.write_bytes, issue);
        } else if (stages.fills) {
            tf32w_stage_slice(plan, stages.write_bytes, stages.fill_k, reduction, issue);
        }
        if (issue < 3) {
            tf32w_load_fragments(
                stages.a_read, stages.b_read, issue + 1, offsets, fragments[(issue + 1) & 1]);
        }
        tf32w_mma(fragments[issue & 1], acc);
    }
}

template <int Stages>
__device__ __forceinline__ void tf32w_nn_kernel(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    extern __shared__ __align__(16) unsigned char tf32w_shared[];
    float* a_stages = reinterpret_cast<float*>(tf32w_shared);
    float* b_stages = a_stages + Stages * 128 * 32;
    int column_tiles = (params.n + 127) / 128;
    int tile_row = (int)blockIdx.x / column_tiles * 128;
    int tile_column = (int)blockIdx.x % column_tiles * 128;
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = (warp >> 2) * 64;
    int warp_n = (warp & 3) * 32;
    int group = lane >> 2;
    int thread = lane & 3;
    float acc[4][4][4];
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                acc[m_atom][n_atom][element] =
                    row < params.m && column < params.n && bias != nullptr ? bias[column] : 0.0f;
            }
        }
    }
    Tf32wFragmentOffsets offsets;
    tf32w_fragment_offsets(warp_m, warp_n, group, thread, lane, offsets);
    Tf32wCopyPlan plan;
    tf32w_copy_plan(a_stages, b_stages, a, b, params, tile_row, tile_column, plan);
    long long b_slab_rows = 32LL * params.ldb;
    unsigned tile_count = (static_cast<unsigned>(params.k) + 31U) / 32U;
#pragma unroll
    for (unsigned tile = 0; tile < Stages - 1; ++tile) {
        if (tile < tile_count) {
            tf32w_stage_async(plan, tile * 128 * 32 * 4U, (int)(tile * 32U), params.k);
            tf32w_advance_plan(plan, b_slab_rows);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    bool whole_chunks = (params.n & 3) == 0;
    int read_stage = 0;
    for (unsigned tile = 0; tile < tile_count; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Stages - 2));
        // The barrier also retires every warp's reads of the stage the copies
        // below overwrite (the one computed in the previous iteration).
        __syncthreads();
        unsigned fill = tile + Stages - 1;
        int write_stage = read_stage == 0 ? Stages - 1 : read_stage - 1;
        Tf32wTileStages stages;
        stages.a_read = a_stages + read_stage * 128 * 32;
        stages.b_read = b_stages + read_stage * 32 * 128;
        stages.write_bytes = (unsigned)write_stage * 128U * 32U * 4U;
        stages.fill_k = (int)(fill * 32U);
        stages.fills = fill < tile_count;
        Tf32wFragments fragments[2];
        tf32w_load_fragments(stages.a_read, stages.b_read, 0, offsets, fragments[0]);
        if (whole_chunks && stages.fills && stages.fill_k + 32 <= params.k) {
            tf32w_tile_steps<true>(plan, offsets, stages, params.k, fragments, acc);
        } else {
            tf32w_tile_steps<false>(plan, offsets, stages, params.k, fragments, acc);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        if (stages.fills) tf32w_advance_plan(plan, b_slab_rows);
        if (++read_stage == Stages) read_stage = 0;
    }
    // Epilogue: the tile goes through shared memory (row stride 136 floats:
    // the fragment-shaped float2 writes land in two wavefronts) and out as
    // 16-byte row segments, one warp per 512-byte row, when the output is
    // 16-byte addressable and the tile's columns lie inside n; each element
    // sees the same alpha and beta arithmetic as the scalar store.
    __syncthreads();
    float* tile = reinterpret_cast<float*>(tf32w_shared);
    bool vector_rows = tile_column + 128 <= params.n && (params.ldc & 3) == 0
        && is_aligned_16(output);
    if (vector_rows) {
#pragma unroll
        for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
            for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    int row = warp_m + m_atom * 16 + group + half * 8;
                    int column = warp_n + n_atom * 8 + 2 * thread;
                    *reinterpret_cast<float2*>(tile + row * 136 + column) = make_float2(
                        acc[m_atom][n_atom][2 * half], acc[m_atom][n_atom][2 * half + 1]);
                }
            }
        }
        __syncthreads();
        bool scale = params.alpha != 1.0f;
        bool blend = params.beta != 0.0f;
#pragma unroll 4
        for (int linear = (int)threadIdx.x; linear < 128 * 32; linear += 256) {
            int row = linear >> 5;
            int chunk = (linear & 31) * 4;
            int global_row = tile_row + row;
            if (global_row >= params.m) continue;
            float4 value = *reinterpret_cast<const float4*>(tile + row * 136 + chunk);
            float* destination = output + (long long)global_row * params.ldc + tile_column + chunk;
            if (scale) {
                value.x = __fmul_rn(params.alpha, value.x);
                value.y = __fmul_rn(params.alpha, value.y);
                value.z = __fmul_rn(params.alpha, value.z);
                value.w = __fmul_rn(params.alpha, value.w);
            }
            if (blend) {
                float4 old = *reinterpret_cast<const float4*>(destination);
                value.x = __fmaf_rn(params.beta, old.x, value.x);
                value.y = __fmaf_rn(params.beta, old.y, value.y);
                value.z = __fmaf_rn(params.beta, old.z, value.z);
                value.w = __fmaf_rn(params.beta, old.w, value.w);
            }
            *reinterpret_cast<float4*>(destination) = value;
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < 4; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < 4; ++n_atom) {
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                int row = tile_row + warp_m + m_atom * 16 + group + (element >= 2 ? 8 : 0);
                int column = tile_column + warp_n + n_atom * 8 + 2 * thread + (element & 1);
                tf32_store<SgbTf32Nn>(output, row, column, acc[m_atom][n_atom][element], bias, params);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The multistage tiles: a cp.async ring of Stages slabs, the next k8 step's
// fragments loading under the current step's products.
// ---------------------------------------------------------------------------

extern __shared__ __align__(128) float tf32_ms_ring[];

template <SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
struct Tf32MsConfig {
    static constexpr int Threads = 32 * WarpsM * WarpsN;
    static constexpr int WM = BM / WarpsM;
    static constexpr int WN = BN / WarpsN;
    static constexpr int MAtoms = WM / 16;
    static constexpr int NAtoms = WN / 8;
    static constexpr int KSteps = BK / 8;
    static constexpr bool ACross = Op != SgbTf32Tn;
    static constexpr bool BCross = Op == SgbTf32Nt;
    // A transposed A operand's rows are dealt permuted (see the header).
    static constexpr bool APerm = !ACross;
    static constexpr int APlane = ACross ? BM * BK : BK * (BM + 8);
    static constexpr int BPlane = BCross ? BN * BK : BK * (BN + 8);
    static constexpr int StageFloats = APlane + BPlane;
    static constexpr int SharedBytes = Stages * StageFloats * 4;
    static constexpr int ACopies = BM * BK / 4 / Threads;
    static constexpr int BCopies = BN * BK / 4 / Threads;
    static_assert(BK == 16 || BK == 32, "a slab is 16 or 32 reduction columns");
    static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tiles are whole atoms");
    static_assert(!BCross || NAtoms % 2 == 0, "ldmatrix.x4 feeds two n8 atoms");
    static_assert(BM * BK / 4 % Threads == 0 && BN * BK / 4 % Threads == 0,
                  "every thread copies the same number of chunks");
    static_assert(Stages >= 2, "at least two stages");
    static_assert(SharedBytes <= 99 * 1024, "the ring fits one SM's opt-in shared memory");
};

// A [row][BK] plane: the chunk is XORed with the row's index among the
// eight rows whose chunk of the same column lands in one bank set, so the
// eight row addresses of one ldmatrix matrix hit eight distinct bank groups.
template <int BK>
__device__ __forceinline__ int tf32_ms_cross(int row, int chunk) {
    constexpr int RowsPerLine = 32 / BK;
    constexpr int Chunks = BK / 4;
    return row * BK + ((chunk ^ ((row / RowsPerLine) & (Chunks - 1))) << 2);
}

// A [BK][Columns + 8] plane: a fragment load reads reduction rows t (and
// t + 4) of four consecutive lane groups' columns; the eight-float pad moves
// each row by eight banks, so the loads of a half warp hit distinct banks.
template <int Columns>
__device__ __forceinline__ int tf32_ms_cong(int row, int column) {
    return row * (Columns + 8) + column;
}

__device__ __forceinline__ void tf32_ms_copy(unsigned shared_dst, const float* global_src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(bytes));
}

__device__ __forceinline__ void tf32_ms_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
__device__ __forceinline__ void tf32_ms_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

__device__ __forceinline__ void tf32_ms_ldmatrix_x4(unsigned (&r)[4], unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

__device__ __forceinline__ int tf32_ms_bytes(int remaining) {
    return remaining >= 4 ? 16 : (remaining > 0 ? remaining * 4 : 0);
}

// Every fragment word rounds with cvt.rna.tf32.f32, to nearest with ties
// away from zero, subnormals included.
__device__ __forceinline__ unsigned tf32_ms_round(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

// One operand's copies for one thread. Its chunks sit `pass` floats apart
// in global memory and `ShiftBytes` bytes apart in the slot, so the swizzle
// of the first one serves them all; the source moves `slab` floats from one
// slab to the next.
//
// A [rows][BK] operand (reduction contiguous): the thread keeps one 16-byte
// chunk column, its rows are `pass / ld` apart, and a row past the output
// edge is marked in `valid`; the reduction bound reads the chunk's index.
// A [BK][columns] operand (output contiguous): the thread keeps one chunk
// of columns, whose bytes inside the output are `column_bytes`, and its
// reduction rows are `pass / ld` apart from `k_row`.
template <bool Cross, int Copies, int ShiftBytes>
struct Tf32MsCopy {
    const float* source;
    long long pass;
    long long slab;
    unsigned shared;
    unsigned valid;
    int column_bytes;
    int k;
    int k_pass;

    // Full: the tile lies inside the output and the reduction is whole
    // 32-deep groups, so every chunk of every slab is whole and only `on`
    // decides; the per-chunk bounds would otherwise hold enough predicates
    // to serialize the fragment rounding behind one.
    template <bool Full>
    __device__ __forceinline__ void issue(unsigned slot_base, int reduction, bool on) const {
#pragma unroll
        for (int copy = 0; copy < Copies; ++copy) {
            int bytes;
            if constexpr (Full) {
                bytes = on ? 16 : 0;
            } else if constexpr (Cross) {
                bytes = on && ((valid >> copy) & 1U) ? tf32_ms_bytes(reduction - k) : 0;
            } else {
                bytes = on && k + copy * k_pass < reduction ? column_bytes : 0;
            }
            tf32_ms_copy(slot_base + shared + (unsigned)(copy * ShiftBytes), source + copy * pass, bytes);
        }
    }

    __device__ __forceinline__ void advance(int bk) {
        source += slab;
        k += bk;
    }
};

template <SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
struct Tf32MsTile {
    using Cfg = Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    static constexpr int ARowsPerPass = Cfg::ACross ? Cfg::Threads / (BK / 4) : Cfg::Threads / (BM / 4);
    static constexpr int BRowsPerPass = Cfg::BCross ? Cfg::Threads / (BK / 4) : Cfg::Threads / (BN / 4);
    static constexpr int AShift = ARowsPerPass * (Cfg::ACross ? BK : BM + 8) * 4;
    static constexpr int BShift = BRowsPerPass * (Cfg::BCross ? BK : BN + 8) * 4;
    static_assert(!Cfg::ACross || ARowsPerPass % (8 * (32 / BK)) == 0,
                  "every pass keeps the first chunk's swizzle");
    static_assert(!Cfg::BCross || BRowsPerPass % (8 * (32 / BK)) == 0,
                  "every pass keeps the first chunk's swizzle");

    Tf32MsCopy<Cfg::ACross, Cfg::ACopies, AShift> a_copy;
    Tf32MsCopy<Cfg::BCross, Cfg::BCopies, BShift> b_copy;
    int reduction;
    int tile_row;
    int tile_column;
    unsigned shared;

    // A [rows][BK] operand's plan: rows `row0 + copy * pass` of the tile.
    template <int Copies, int Shift, int RowsPerPass>
    __device__ __forceinline__ static Tf32MsCopy<true, Copies, Shift> cross_plan(
        const float* base, int ld, int edge, int tile_origin, int k_begin) {
        int chunk = (int)threadIdx.x % (BK / 4);
        int row = (int)threadIdx.x / (BK / 4);
        unsigned valid = 0;
#pragma unroll
        for (int copy = 0; copy < Copies; ++copy) {
            if (tile_origin + row + copy * RowsPerPass < edge) valid |= 1U << copy;
        }
        Tf32MsCopy<true, Copies, Shift> plan;
        plan.source = base + (long long)(tile_origin + row) * ld + k_begin + chunk * 4;
        plan.pass = (long long)RowsPerPass * ld;
        plan.slab = BK;
        plan.shared = (unsigned)tf32_ms_cross<BK>(row, chunk) * 4U;
        plan.valid = valid;
        plan.column_bytes = 0;
        plan.k = k_begin + chunk * 4;
        plan.k_pass = 0;
        return plan;
    }

    // A [BK][Columns] operand's plan: reduction rows `k_row + copy * pass`.
    template <int Copies, int Shift, int RowsPerPass, int Columns>
    __device__ __forceinline__ static Tf32MsCopy<false, Copies, Shift> cong_plan(
        const float* base, int ld, int edge, int tile_origin, int k_begin) {
        int column = (int)threadIdx.x % (Columns / 4) * 4;
        int k_row = (int)threadIdx.x / (Columns / 4);
        Tf32MsCopy<false, Copies, Shift> plan;
        plan.source = base + (long long)(k_begin + k_row) * ld + tile_origin + column;
        plan.pass = (long long)RowsPerPass * ld;
        plan.slab = (long long)BK * ld;
        plan.shared = (unsigned)tf32_ms_cong<Columns>(k_row, column) * 4U;
        plan.valid = 0;
        plan.column_bytes = tf32_ms_bytes(edge - (tile_origin + column));
        plan.k = k_begin + k_row;
        plan.k_pass = RowsPerPass;
        return plan;
    }

    __device__ __forceinline__ Tf32MsTile(
        const float* a, const float* b, int rows, int columns, int reduction_extent,
        int lda, int ldb, int row_origin, int column_origin, int k_begin)
        : reduction(reduction_extent), tile_row(row_origin), tile_column(column_origin),
          shared((unsigned)__cvta_generic_to_shared(tf32_ms_ring)) {
        if constexpr (Cfg::ACross) {
            a_copy = cross_plan<Cfg::ACopies, AShift, ARowsPerPass>(a, lda, rows, row_origin, k_begin);
        } else {
            a_copy = cong_plan<Cfg::ACopies, AShift, ARowsPerPass, BM>(a, lda, rows, row_origin, k_begin);
        }
        if constexpr (Cfg::BCross) {
            b_copy = cross_plan<Cfg::BCopies, BShift, BRowsPerPass>(b, ldb, columns, column_origin, k_begin);
        } else {
            b_copy = cong_plan<Cfg::BCopies, BShift, BRowsPerPass, BN>(b, ldb, columns, column_origin, k_begin);
        }
    }

    // Stage the plans' current slab into `slot` (all zero-filled when off)
    // and move the plans to the next slab.
    template <bool Full>
    __device__ __forceinline__ void stage_next(unsigned slot_offset, bool on) {
        unsigned slot_base = shared + slot_offset;
        a_copy.template issue<Full>(slot_base, reduction, on);
        b_copy.template issue<Full>(slot_base + (unsigned)Cfg::APlane * 4U, reduction, on);
        a_copy.advance(BK);
        b_copy.advance(BK);
    }
};

template <int MAtoms, int NAtoms>
struct Tf32MsFragments {
    unsigned a[MAtoms][4];
    unsigned b[NAtoms][2];
};

// The raw fragment words of k8 step `step` of the slab at `slot_offset`;
// tf32_ms_round_fragments rounds them just before their products.
template <SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
__device__ __forceinline__ void tf32_ms_fragments(
    Tf32MsFragments<Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>::MAtoms,
                    Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>::NAtoms>& f,
    unsigned shared, unsigned slot_offset, int step, int warp_m, int warp_n, int lane) {
    using Cfg = Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    unsigned a_plane = shared + slot_offset;
    unsigned b_plane = a_plane + (unsigned)Cfg::APlane * 4U;
    const float* ring = reinterpret_cast<const float*>(
        reinterpret_cast<const unsigned char*>(tf32_ms_ring) + slot_offset);
    int group = lane >> 2;
    int thread = lane & 3;
    if constexpr (Cfg::ACross) {
#pragma unroll
        for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + (lane & 15);
            int chunk = step * 2 + (lane >> 4);
            unsigned raw[4];
            tf32_ms_ldmatrix_x4(raw, a_plane + (unsigned)tf32_ms_cross<BK>(row, chunk) * 4U);
#pragma unroll
            for (int word = 0; word < 4; ++word) f.a[m_atom][word] = raw[word];
        }
    } else {
        // Rows 2g and 2g + 1 of an atom: its mma rows g and g + 8.
        int k = step * 8 + thread;
#pragma unroll
        for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + 2 * group;
            float2 low = *reinterpret_cast<const float2*>(ring + tf32_ms_cong<BM>(k, row));
            float2 high = *reinterpret_cast<const float2*>(ring + tf32_ms_cong<BM>(k + 4, row));
            f.a[m_atom][0] = __float_as_uint(low.x);
            f.a[m_atom][1] = __float_as_uint(low.y);
            f.a[m_atom][2] = __float_as_uint(high.x);
            f.a[m_atom][3] = __float_as_uint(high.y);
        }
    }
    if constexpr (Cfg::BCross) {
#pragma unroll
        for (int pair = 0; pair < Cfg::NAtoms / 2; ++pair) {
            int quad = lane >> 3;
            int row = warp_n + (pair * 2 + (quad >> 1)) * 8 + (lane & 7);
            int chunk = step * 2 + (quad & 1);
            unsigned raw[4];
            tf32_ms_ldmatrix_x4(raw, b_plane + (unsigned)tf32_ms_cross<BK>(row, chunk) * 4U);
            f.b[2 * pair][0] = raw[0];
            f.b[2 * pair][1] = raw[1];
            f.b[2 * pair + 1][0] = raw[2];
            f.b[2 * pair + 1][1] = raw[3];
        }
    } else {
        // A fragment's two words sit four reduction rows apart, so they
        // are two loads either way.
        const float* plane = ring + Cfg::APlane;
        int k = step * 8 + thread;
#pragma unroll
        for (int n_atom = 0; n_atom < Cfg::NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            f.b[n_atom][0] = __float_as_uint(plane[tf32_ms_cong<BN>(k, column)]);
            f.b[n_atom][1] = __float_as_uint(plane[tf32_ms_cong<BN>(k + 4, column)]);
        }
    }
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void tf32_ms_round_fragments(Tf32MsFragments<MAtoms, NAtoms>& f) {
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int word = 0; word < 4; ++word) f.a[m_atom][word] = tf32_ms_round(__uint_as_float(f.a[m_atom][word]));
    }
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
        for (int word = 0; word < 2; ++word) f.b[n_atom][word] = tf32_ms_round(__uint_as_float(f.b[n_atom][word]));
    }
}

// Walk `tiles` slabs into the accumulators, the plans starting at the
// first. The next step's fragments load while the current step's products
// issue, the first fragments of a slab load during the previous slab's last
// step, and a slab's copies issue with its first step, so the tensor cores
// keep work across the slab boundary's barrier. A step rounds its own
// fragments, loaded a step earlier, just before its products, so the
// rounding overlaps the previous step's products still in the tensor pipe
// instead of waiting on loads just issued.
template <bool Full, SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
__device__ __forceinline__ void tf32_ms_mainloop(
    Tf32MsTile<Op, BM, BN, BK, WarpsM, WarpsN, Stages>& tile, int tiles,
    int warp_m, int warp_n, int lane,
    float (&acc)[Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>::MAtoms]
                [Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>::NAtoms][4]) {
    using Cfg = Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    static_assert(Cfg::KSteps >= 2, "the slab barrier sits on the second-to-last step");
    constexpr unsigned SlotBytes = (unsigned)Cfg::StageFloats * 4U;
    constexpr unsigned RingBytes = SlotBytes * (unsigned)Stages;
#pragma unroll
    for (int slab = 0; slab < Stages - 1; ++slab) {
        tile.template stage_next<Full>((unsigned)slab * SlotBytes, slab < tiles);
        tf32_ms_commit();
    }
    tf32_ms_wait<Stages - 2>();
    __syncthreads();
    // Slot byte offsets, so every fragment and copy address is a constant
    // away from one register.
    unsigned read_slot = 0;
    unsigned write_slot = (unsigned)(Stages - 1) * SlotBytes;
    int fill = Stages - 1;
    Tf32MsFragments<Cfg::MAtoms, Cfg::NAtoms> f[2];
    tf32_ms_fragments<Op, BM, BN, BK, WarpsM, WarpsN, Stages>(
        f[0], tile.shared, read_slot, 0, warp_m, warp_n, lane);
    for (int t = 0; t < tiles; ++t) {
#pragma unroll
        for (int step = 0; step < Cfg::KSteps; ++step) {
            // After the barrier of the second-to-last step, read_slot is
            // already the next slab's; past the walk the load is unused.
            tf32_ms_fragments<Op, BM, BN, BK, WarpsM, WarpsN, Stages>(
                f[(step + 1) & 1], tile.shared, read_slot,
                step + 1 < Cfg::KSteps ? step + 1 : 0, warp_m, warp_n, lane);
            tf32_ms_round_fragments(f[step & 1]);
            if (step == 0) {
                tile.template stage_next<Full>(write_slot, fill < tiles);
                ++fill;
            }
#pragma unroll
            for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < Cfg::NAtoms; ++n_atom) {
                    tf32_mma_m16n8k8(acc[m_atom][n_atom], f[step & 1].a[m_atom], f[step & 1].b[n_atom]);
                }
            }
            if (step == Cfg::KSteps - 2) {
                tf32_ms_commit();
                tf32_ms_wait<Stages - 2>();
                __syncthreads();
                read_slot = read_slot + SlotBytes == RingBytes ? 0U : read_slot + SlotBytes;
                write_slot = write_slot + SlotBytes == RingBytes ? 0U : write_slot + SlotBytes;
            }
        }
    }
    tf32_ms_wait<0>();
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_ms_rows(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Tn ? p.k : p.m;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_ms_columns(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nt ? p.k : p.n;
}

template <SgbTf32Op Op>
__device__ __forceinline__ int tf32_ms_reduction(const Sm80Tf32KernelParams& p) {
    return Op == SgbTf32Nn ? p.k : (Op == SgbTf32Tn ? p.m : p.n);
}

// Where a lane's accumulators sit in the warp tile: row half `half` of
// atom `m_atom` (rows permuted for a transposed A, see the header), and
// each n8 atom's pair of consecutive columns.
template <SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
struct Tf32MsOut {
    using Cfg = Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    static constexpr int Vec = 2;
    static constexpr int Groups = Cfg::NAtoms;

    __device__ __forceinline__ static int row(int m_atom, int half, int group) {
        if constexpr (Cfg::APerm) {
            return m_atom * 16 + 2 * group + half;
        } else {
            return m_atom * 16 + group + 8 * half;
        }
    }

    __device__ __forceinline__ static int column(int n_atom, int thread) {
        return n_atom * 8 + 2 * thread;
    }

    __device__ __forceinline__ static float& value(
        float (&acc)[Cfg::MAtoms][Cfg::NAtoms][4], int m_atom, int half, int n_atom, int v) {
        return acc[m_atom][n_atom][2 * half + v];
    }
};

template <int Vec>
__device__ __forceinline__ void tf32_ms_load_vec(float (&v)[Vec], const float* source) {
    static_assert(Vec == 2, "pairs");
    float2 x = *reinterpret_cast<const float2*>(source);
    v[0] = x.x;
    v[1] = x.y;
}

template <int Vec>
__device__ __forceinline__ void tf32_ms_store_vec(float* destination, const float (&v)[Vec]) {
    static_assert(Vec == 2, "pairs");
    *reinterpret_cast<float2*>(destination) = make_float2(v[0], v[1]);
}

template <int Vec>
__device__ __forceinline__ void tf32_ms_store_vec_cg(float* destination, const float (&v)[Vec]) {
    static_assert(Vec == 2, "pairs");
    __stcg(reinterpret_cast<float2*>(destination), make_float2(v[0], v[1]));
}

// One output tile walking the whole reduction in one chain per output.
template <SgbTf32Op Op, int BM, int BN, int BK, int WarpsM, int WarpsN, int Stages>
__device__ __forceinline__ void tf32_ms_tile_kernel(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    static_assert(Op != SgbTf32Tn, "a weight gradient takes the split members");
    using Cfg = Tf32MsConfig<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    using Out = Tf32MsOut<Op, BM, BN, BK, WarpsM, WarpsN, Stages>;
    constexpr int Vec = Out::Vec;
    int rows = tf32_ms_rows<Op>(params);
    int columns = tf32_ms_columns<Op>(params);
    int reduction = tf32_ms_reduction<Op>(params);
    int row_tiles = (rows + BM - 1) / BM;
    Tf32MsTile<Op, BM, BN, BK, WarpsM, WarpsN, Stages> tile(
        a, b, rows, columns, reduction, params.lda, params.ldb,
        (int)blockIdx.x % row_tiles * BM, (int)blockIdx.x / row_tiles * BN, 0);
    int warp = (int)threadIdx.x >> 5;
    int lane = (int)threadIdx.x & 31;
    int warp_m = warp / WarpsN * Cfg::WM;
    int warp_n = warp % WarpsN * Cfg::WN;
    int group = lane >> 2;
    int thread = lane & 3;
    int row_base = tile.tile_row + warp_m;
    int column_base = tile.tile_column + warp_n;

    float acc[Cfg::MAtoms][Cfg::NAtoms][4];
#pragma unroll
    for (int column_group = 0; column_group < Out::Groups; ++column_group) {
#pragma unroll
        for (int v = 0; v < Vec; ++v) {
            int column = column_base + Out::column(column_group, thread) + v;
            float seed = 0.0f;
            if constexpr (Op == SgbTf32Nn) {
                if (bias != nullptr && column < columns) seed = bias[column];
            }
#pragma unroll
            for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
                Out::value(acc, m_atom, 0, column_group, v) = seed;
                Out::value(acc, m_atom, 1, column_group, v) = seed;
            }
        }
    }

    // Whole 32-deep groups, so the k8 steps match the 32-deep tiles'.
    int tiles = (reduction + 31) / 32 * (32 / BK);
    if (tile.tile_row + BM <= rows && tile.tile_column + BN <= columns && reduction % 32 == 0) {
        tf32_ms_mainloop<true>(tile, tiles, warp_m, warp_n, lane, acc);
    } else {
        tf32_ms_mainloop<false>(tile, tiles, warp_m, warp_n, lane, acc);
    }

    bool vectors = (params.ldc & (Vec - 1)) == 0
        && ((reinterpret_cast<unsigned long long>(output) & (unsigned long long)(4 * Vec - 1)) == 0ULL);
    bool reads = Op == SgbTf32Nn && params.beta != 0.0f;
    if (vectors && !reads && tile.tile_row + BM <= rows && tile.tile_column + BN <= columns) {
        // A whole tile with nothing to read back: every pair lands.
#pragma unroll
        for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                float* output_row = output
                    + (long long)(row_base + Out::row(m_atom, half, group)) * params.ldc + column_base;
#pragma unroll
                for (int n_atom = 0; n_atom < Cfg::NAtoms; ++n_atom) {
                    float v[Vec];
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        v[i] = Out::value(acc, m_atom, half, n_atom, i);
                        if (params.alpha != 1.0f) v[i] = __fmul_rn(params.alpha, v[i]);
                    }
                    tf32_ms_store_vec(output_row + Out::column(n_atom, thread), v);
                }
            }
        }
        return;
    }
#pragma unroll
    for (int m_atom = 0; m_atom < Cfg::MAtoms; ++m_atom) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = row_base + Out::row(m_atom, half, group);
            if (row >= rows) continue;
            float* output_row = output + (long long)row * params.ldc;
#pragma unroll
            for (int column_group = 0; column_group < Out::Groups; ++column_group) {
                int column = column_base + Out::column(column_group, thread);
                if (column >= columns) continue;
                bool whole = vectors && column + Vec <= columns;
                float v[Vec];
                float old[Vec];
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    v[i] = Out::value(acc, m_atom, half, column_group, i);
                    old[i] = 0.0f;
                }
                if (reads) {
                    if (whole) {
                        tf32_ms_load_vec(old, output_row + column);
                    } else {
#pragma unroll
                        for (int i = 0; i < Vec; ++i) {
                            if (column + i < columns) old[i] = output_row[column + i];
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < Vec; ++i) {
                    if (params.alpha != 1.0f) v[i] = __fmul_rn(params.alpha, v[i]);
                    if constexpr (Op == SgbTf32Nn) {
                        if (params.beta != 0.0f) v[i] = __fmaf_rn(params.beta, old[i], v[i]);
                    }
                }
                if (whole) {
                    tf32_ms_store_vec(output_row + column, v);
                } else {
#pragma unroll
                    for (int i = 0; i < Vec; ++i) {
                        if (column + i < columns) output_row[column + i] = v[i];
                    }
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm80_tf32_rna_m128n96_w2x4_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    NnWideParams params) {
    nn_wide_kernel<1, 128, 96, 2, 4, 3>(output, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm80_tf32_rna_m96n96_w2x4_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    NnWideParams params) {
    nn_wide_kernel<1, 96, 96, 2, 4, 3>(output, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(256, 1)
void nn_sm80_tf32_rna_m128n128_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    Sm80Tf32KernelParams params) {
    if (params.k == 0) {
        tf32_zero_reduction_epilogue<SgbTf32Nn, 128, 128>(output, bias, params);
        return;
    }
    tf32w_nn_kernel<3>(output, a, b, bias, params);
}

#define TF32_RNA_MS_TILE(NAME, OP, BM, BN, BK, WARPS_M, WARPS_N, STAGES, MIN_BLOCKS)      \
    extern "C" __global__ __launch_bounds__(32 * WARPS_M * WARPS_N, MIN_BLOCKS)           \
    void NAME(float* output, const float* a, const float* b, const float* bias,           \
              Sm80Tf32KernelParams params) {                                              \
        tf32_ms_tile_kernel<OP, BM, BN, BK, WARPS_M, WARPS_N, STAGES>(                     \
            output, a, b, bias, params);                                                  \
    }

TF32_RNA_MS_TILE(nn_sm80_tf32_rna_ms_m64n128_bk16_s6, SgbTf32Nn, 64, 128, 16, 2, 2, 6, 1)
TF32_RNA_MS_TILE(nn_sm80_tf32_rna_ms_m128n64_bk32_s3, SgbTf32Nn, 128, 64, 32, 2, 2, 3, 1)
TF32_RNA_MS_TILE(nn_sm80_tf32_rna_ms_m64n64_bk32_s3, SgbTf32Nn, 64, 64, 32, 2, 2, 3, 2)
// The input gradient's tiles: B is the weight read across its rows, the
// same ring and chain as the forward's.
TF32_RNA_MS_TILE(nt_sm80_tf32_rna_ms_m64n128_bk16_s6, SgbTf32Nt, 64, 128, 16, 2, 2, 6, 1)
TF32_RNA_MS_TILE(nt_sm80_tf32_rna_ms_m128n64_bk32_s3, SgbTf32Nt, 128, 64, 32, 2, 2, 3, 1)
TF32_RNA_MS_TILE(nt_sm80_tf32_rna_ms_m128n128_bk16_s5, SgbTf32Nt, 128, 128, 16, 2, 2, 5, 1)

#undef TF32_RNA_MS_TILE

static_assert(Tf32MsConfig<SgbTf32Nn, 64, 128, 16, 2, 2, 6>::SharedBytes == 76800, "NN ms 64x128 ring");
static_assert(Tf32MsConfig<SgbTf32Nn, 128, 64, 32, 2, 2, 3>::SharedBytes == 76800, "NN ms 128x64 ring");
static_assert(Tf32MsConfig<SgbTf32Nn, 64, 64, 32, 2, 2, 3>::SharedBytes == 52224, "NN ms 64x64 ring");
static_assert(Tf32MsConfig<SgbTf32Nt, 64, 128, 16, 2, 2, 6>::SharedBytes == 73728, "NT ms 64x128 ring");
static_assert(Tf32MsConfig<SgbTf32Nt, 128, 64, 32, 2, 2, 3>::SharedBytes == 73728, "NT ms 128x64 ring");
static_assert(Tf32MsConfig<SgbTf32Nt, 128, 128, 16, 2, 2, 5>::SharedBytes == 81920, "NT ms 128x128 ring");
