// Deterministic TF32 TN weight-gradient GEMM for Ada, straight from the
// saved input: C[m][n] = alpha * sum_k A[k][m] * B[k][n] + C[m][n].
//
// Both operands keep the reduction as their slow index, so neither can feed
// ldmatrix; both stage planes are [32 reduction][columns] with the chunk
// index folded with the reduction row, and every fragment word is a scalar
// shared load rounded with cvt.rna.tf32.f32, the same rounding the retained
// route applies (in its transpose pass for A, at the fragment for B). The
// transpose pass and its scratch are gone: the saved input is read once.
// The reduction runs in ascending k8 steps into one accumulator per output
// element, so the bits equal the retained route's.
//
// Parameters: alpha, beta (implicitly one), m (rows = columns of A),
// k (the reduction = rows of A and B), n, lda, ldb, ldc. Launch: one CTA per
// output tile, column tiles fastest; dynamic shared memory = Stages * (BM +
// BN) * 128 bytes.

struct TnWideParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(TnWideParams) == 32, "TN parameter ABI");

__device__ __forceinline__ void tn_wide_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void tn_wide_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
__device__ __forceinline__ void tn_wide_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

__device__ __forceinline__ unsigned tn_wide_lds(unsigned address) {
    unsigned value;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(value) : "r"(address));
    return value;
}

__device__ __forceinline__ void tn_wide_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ unsigned tn_wide_rna(unsigned bits) {
    float value = __uint_as_float(bits);
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}

// [32 reduction][Stride columns] with the chunk index folded with the
// reduction row, so the four rows a fragment reads land on distinct banks.
// The stride is the tile width rounded up to eight 16-byte chunks, which
// keeps the fold inside the row; a narrower tile leaves the tail unused.
template <int Stride>
__device__ __forceinline__ int tn_wide_slot(int reduction, int column) {
    int chunk = (column >> 2) ^ ((reduction & 3) << 1);
    return reduction * Stride + chunk * 4 + (column & 3);
}

template <int Width>
struct TnWideStride {
    static constexpr int value = (Width + 31) / 32 * 32;
    static constexpr unsigned RowBytes = (unsigned)value * 4U;
};

template <int Slices>
struct TnWidePlan {
    const float* source[Slices];
    unsigned destination[Slices];
    int bytes_cap[Slices];
    int row[Slices];
};

template <int Width, int Threads, int Slices>
__device__ __forceinline__ void tn_wide_plan(
    TnWidePlan<Slices>& plan, unsigned plane, const float* global, int leading,
    int tile_origin, int extent) {
    constexpr int Chunks = 32 * (Width / 4);
    constexpr int Stride = TnWideStride<Width>::value;
#pragma unroll
    for (int slice = 0; slice < Slices; ++slice) {
        int linear = (int)threadIdx.x + slice * Threads;
        int k_row = linear / (Width / 4);
        int column = (linear % (Width / 4)) * 4;
        bool in_tile = linear < Chunks;
        int global_column = tile_origin + column;
        int columns = extent - global_column;
        columns = columns < 0 ? 0 : (columns > 4 ? 4 : columns);
        if (!in_tile) columns = 0;
        plan.row[slice] = k_row;
        plan.source[slice] = global + (long long)(in_tile ? k_row : 0) * leading + (columns > 0 ? global_column : 0);
        plan.destination[slice] = in_tile ? plane + (unsigned)tn_wide_slot<Stride>(k_row, column) * 4U : plane;
        plan.bytes_cap[slice] = columns * 4;
    }
}

template <int BM, int BN, int WarpsM, int WarpsN, int Stages>
struct TnWideConfig {
    static constexpr int Threads = 32 * WarpsM * WarpsN;
    static constexpr int WM = BM / WarpsM;
    static constexpr int WN = BN / WarpsN;
    static constexpr int MAtoms = WM / 16;
    static constexpr int NAtoms = WN / 8;
    static constexpr int AStride = TnWideStride<BM>::value;
    static constexpr int BStride = TnWideStride<BN>::value;
    static constexpr int AChunks = 32 * (BM / 4);
    static constexpr int BChunks = 32 * (BN / 4);
    static constexpr int ASlices = (AChunks + Threads - 1) / Threads;
    static constexpr int BSlices = (BChunks + Threads - 1) / Threads;
    static constexpr unsigned AStageBytes = (unsigned)AStride * 128U;
    static constexpr unsigned BStageBytes = (unsigned)BStride * 128U;
    static constexpr unsigned SharedBytes = (AStageBytes + BStageBytes) * (unsigned)Stages;
    static_assert(WM % 16 == 0, "warp rows must be whole m16 atoms");
    static_assert(WN % 8 == 0, "warp columns must be whole n8 atoms");
    static_assert(BM % 4 == 0 && BN % 4 == 0, "tiles are whole 16-byte chunks");
    static_assert(Stages >= 2, "at least two stages");
};

template <int MAtoms, int NAtoms>
struct TnWideFragments {
    unsigned a[MAtoms][4];
    unsigned b[NAtoms][2];
};

// a_address[m][0] points at (reduction = thread, column = warp_m + 16 m +
// group), a_address[m][1] at column + 8; b_address[n] at (thread, warp_n +
// 8 n + group). Rows thread + 4 and later k8 steps keep the same swizzle.
template <int AStride, int BStride, int MAtoms, int NAtoms>
__device__ __forceinline__ void tn_wide_load_fragments(
    const unsigned (&a_address)[MAtoms][2], unsigned a_stage,
    const unsigned (&b_address)[NAtoms], unsigned b_stage, int step,
    TnWideFragments<MAtoms, NAtoms>& fragments) {
    unsigned a_step = a_stage + (unsigned)step * 8U * (unsigned)AStride * 4U;
    unsigned b_step = b_stage + (unsigned)step * 8U * (unsigned)BStride * 4U;
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
        unsigned r0 = tn_wide_lds(a_address[m_atom][0] + a_step);
        unsigned r1 = tn_wide_lds(a_address[m_atom][1] + a_step);
        unsigned r2 = tn_wide_lds(a_address[m_atom][0] + a_step + 4U * (unsigned)AStride * 4U);
        unsigned r3 = tn_wide_lds(a_address[m_atom][1] + a_step + 4U * (unsigned)AStride * 4U);
        fragments.a[m_atom][0] = tn_wide_rna(r0);
        fragments.a[m_atom][1] = tn_wide_rna(r1);
        fragments.a[m_atom][2] = tn_wide_rna(r2);
        fragments.a[m_atom][3] = tn_wide_rna(r3);
    }
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        unsigned lo = tn_wide_lds(b_address[n_atom] + b_step);
        unsigned hi = tn_wide_lds(b_address[n_atom] + b_step + 4U * (unsigned)BStride * 4U);
        fragments.b[n_atom][0] = tn_wide_rna(lo);
        fragments.b[n_atom][1] = tn_wide_rna(hi);
    }
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void tn_wide_mma_step(
    const TnWideFragments<MAtoms, NAtoms>& fragments,
    float (&acc)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            tn_wide_mma(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

template <int BM, int BN, int WarpsM, int WarpsN, int Stages>
__device__ __forceinline__ void tn_wide_kernel(
    float* output, const float* a, const float* b, TnWideParams params) {
    using Cfg = TnWideConfig<BM, BN, WarpsM, WarpsN, Stages>;
    constexpr int Threads = Cfg::Threads;
    constexpr int MAtoms = Cfg::MAtoms;
    constexpr int NAtoms = Cfg::NAtoms;
    constexpr int ASlices = Cfg::ASlices;
    constexpr int BSlices = Cfg::BSlices;
    constexpr int AStride = Cfg::AStride;
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
            acc[m_atom][n_atom][0] = 0.0f;
            acc[m_atom][n_atom][1] = 0.0f;
            acc[m_atom][n_atom][2] = 0.0f;
            acc[m_atom][n_atom][3] = 0.0f;
        }
    }

    unsigned a_address[MAtoms][2];
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
        int column = warp_m + m_atom * 16 + group;
        a_address[m_atom][0] = a_plane + (unsigned)tn_wide_slot<AStride>(thread, column) * 4U;
        a_address[m_atom][1] = a_plane + (unsigned)tn_wide_slot<AStride>(thread, column + 8) * 4U;
    }
    unsigned b_address[NAtoms];
#pragma unroll
    for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
        b_address[n_atom] = b_plane + (unsigned)tn_wide_slot<BStride>(thread, warp_n + n_atom * 8 + group) * 4U;
    }

    TnWidePlan<ASlices> a_plan;
    TnWidePlan<BSlices> b_plan;
    tn_wide_plan<BM, Threads, ASlices>(a_plan, a_plane, a, params.lda, tile_row, params.m);
    tn_wide_plan<BN, Threads, BSlices>(b_plan, b_plane, b, params.ldb, tile_column, params.n);
    long long a_tile_rows = 32LL * params.lda;
    long long b_tile_rows = 32LL * params.ldb;

    unsigned tile_count = ((unsigned)params.k + 31U) / 32U;

    auto issue = [&](int slot, unsigned write_a, unsigned write_b, int reduction_base) {
        if (slot < ASlices) {
            bool in_tile = (int)threadIdx.x + slot * Threads < Cfg::AChunks;
            if (in_tile) {
                int bytes = reduction_base + a_plan.row[slot] < params.k ? a_plan.bytes_cap[slot] : 0;
                tn_wide_copy_cg(a_plan.destination[slot] + write_a, a_plan.source[slot], bytes);
            }
        }
        if (slot < BSlices) {
            bool in_tile = (int)threadIdx.x + slot * Threads < Cfg::BChunks;
            if (in_tile) {
                int bytes = reduction_base + b_plan.row[slot] < params.k ? b_plan.bytes_cap[slot] : 0;
                tn_wide_copy_cg(b_plan.destination[slot] + write_b, b_plan.source[slot], bytes);
            }
        }
    };
    auto advance = [&]() {
#pragma unroll
        for (int slice = 0; slice < ASlices; ++slice) a_plan.source[slice] += a_tile_rows;
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
        tn_wide_commit();
    }

    for (unsigned tile_base = 0; tile_base < tile_count; tile_base += (unsigned)Stages) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            unsigned tile = tile_base + (unsigned)stage;
            if (tile < tile_count) {
                tn_wide_wait<Stages - 2>();
                __syncthreads();
                unsigned next = tile + (unsigned)(Stages - 1);
                bool has_next = next < tile_count;
                int write_stage = (stage + Stages - 1) % Stages;
                unsigned write_a = Cfg::AStageBytes * (unsigned)write_stage;
                unsigned write_b = Cfg::BStageBytes * (unsigned)write_stage;
                unsigned read_a = Cfg::AStageBytes * (unsigned)stage;
                unsigned read_b = Cfg::BStageBytes * (unsigned)stage;

                TnWideFragments<MAtoms, NAtoms> fragments[2];
                tn_wide_load_fragments<AStride, BStride, MAtoms, NAtoms>(
                    a_address, read_a, b_address, read_b, 0, fragments[0]);
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    if (has_next) {
#pragma unroll
                        for (int slot = step; slot < IssueSlots; slot += 4) {
                            issue(slot, write_a, write_b, (int)next * 32);
                        }
                    }
                    if (step < 3) {
                        tn_wide_load_fragments<AStride, BStride, MAtoms, NAtoms>(
                            a_address, read_a, b_address, read_b, step + 1, fragments[(step + 1) & 1]);
                    }
                    tn_wide_mma_step<MAtoms, NAtoms>(fragments[step & 1], acc);
                }
                tn_wide_commit();
                if (has_next) advance();
            }
        }
    }
    tn_wide_wait<0>();

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
                if (paired && column + 1 < params.n) {
                    float2 old = *reinterpret_cast<const float2*>(destination);
                    *reinterpret_cast<float2*>(destination) = make_float2(
                        __fmaf_rn(params.alpha, v0, old.x), __fmaf_rn(params.alpha, v1, old.y));
                } else {
                    if (column < params.n) destination[0] = __fmaf_rn(params.alpha, v0, destination[0]);
                    if (column + 1 < params.n) destination[1] = __fmaf_rn(params.alpha, v1, destination[1]);
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(384, 1)
void tn_sm89_tf32_m192n192_w3x4_bk32_s2(
    float* output, const float* a, const float* b, const float* bias,
    TnWideParams params) {
    (void)bias;
    tn_wide_kernel<192, 192, 3, 4, 2>(output, a, b, params);
}

