// Deterministic TF32 GEMM for Ada with a row-major A operand and a
// column-contiguous B operand, in two numeric modes that share one body:
//
//   NN forward:  C[m][n] = alpha * A[m][k] * B[k][n] (+ bias, + beta * C)
//                A and B fragment words get the half-ulp add the retained Ada
//                NN kernels apply before the tensor core truncates to tf32.
//   TN weight gradient on a pre-rounded transposed A:
//                C[m][n] = alpha * A[m][k] * B[k][n] + C[m][n]
//                A already holds cvt.rna tf32 values (the retained transpose
//                pass writes them); B is rounded with cvt.rna at the fragment.
//
// A fragments come from ldmatrix over a chunk-swizzled [row][32] plane; B
// fragments are scalar loads from a chunk-swizzled [32][BN] plane, since a
// 32-bit transpose is out of ldmatrix's reach. The reduction runs in
// ascending k8 steps into one accumulator per output element, so the bits
// equal the retained kernels' for every tile shape instantiated here.
//
// Parameters: alpha, beta, m, k (the reduction), n, lda, ldb, ldc. Launch:
// one CTA per output tile, column tiles fastest; dynamic shared memory =
// Stages * (BM + BN) * 128 bytes.

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

__device__ __forceinline__ unsigned nn_wide_add_half(unsigned bits) {
    return bits + 0x1000U;
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

extern "C" __global__ __launch_bounds__(384, 1)
void tn_sm89_tf32_pre_rna_m96n192_w3x4_bk32_s2(
    float* output, const float* a, const float* b, const float* bias,
    NnWideParams params) {
    nn_wide_kernel<2, 96, 192, 3, 4, 2>(output, a, b, bias, params);
}

extern "C" __global__ __launch_bounds__(256, 1)
void tn_sm89_tf32_pre_rna_m96n96_bk32_s3(
    float* output, const float* a, const float* b, const float* bias,
    NnWideParams params) {
    nn_wide_kernel<2, 96, 96, 2, 4, 3>(output, a, b, bias, params);
}

