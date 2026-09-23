// Deterministic TF32 NT GEMM for Ada: dX[m][k] = alpha * A[m][n] * B[k][n]^T.
//
// Both operands are reduction-contiguous, so every operand fragment comes
// from ldmatrix over a chunk-swizzled stage plane, and every fragment word
// gets the same half-ulp add the retained Ada NT kernel applies before the
// tensor core truncates to tf32. The reduction runs in ascending k8 steps
// into one accumulator per output element, so the bits equal the retained
// kernel's for every tile shape instantiated here; the tile shape only
// changes how the work is cut across the device.
//
// Parameters: alpha, beta (unused), m (rows), k (output columns), n (the
// reduction), lda, ldb, ldc. Launch: one CTA per output tile, column tiles
// fastest; dynamic shared memory = Stages * (BM + BN) * row bytes (128 for
// BK = 32, 112 for BK = 24).

struct NtEpiParams {
    float alpha;
    float beta;
    int m;
    int k;
    int n;
    int lda;
    int ldb;
    int ldc;
};
static_assert(sizeof(NtEpiParams) == 32, "NT parameter ABI");

__device__ __forceinline__ void nt_epi_copy_cg(
    unsigned shared_dst, const void* global_src, int valid_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :: "r"(shared_dst), "l"(global_src), "r"(valid_bytes));
}

__device__ __forceinline__ void nt_epi_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int Pending>
__device__ __forceinline__ void nt_epi_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending));
}

__device__ __forceinline__ void nt_epi_ldmatrix_x4(
    unsigned (&r)[4], unsigned address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(address));
}

__device__ __forceinline__ void nt_epi_ldmatrix_x2(
    unsigned (&r)[2], unsigned address) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(address));
}

__device__ __forceinline__ void nt_epi_mma(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// The tensor core reads the upper 19 bits of a tf32 operand. Half an ulp of
// the kept mantissa, added in floating point from the operand's own exponent
// before that truncation, rounds every normal value to nearest, ties away
// from zero, keeps a NaN a NaN and needs no predicate. The retained NT kernel
// rounds every word this way; the retained portable kernels round with
// cvt.rna.tf32.f32 instead, so both are offered and the one matching the
// production route of a cell is selected.
template <bool Rna>
__device__ __forceinline__ unsigned nt_epi_round(unsigned bits) {
    if constexpr (Rna) {
        float value = __uint_as_float(bits);
        unsigned result;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
        return result;
    } else {
        return __float_as_uint(fmaf(__uint_as_float(bits & 0xff800000U), 1.0f / 2048.0f, __uint_as_float(bits)));
    }
}

// Stage plane layout for one operand: BK reduction floats per row.
//   BK = 32: 32 floats per row, the 16-byte chunk index folded with the row
//            so eight consecutive rows never share a bank group.
//   BK = 24: rows padded to 28 floats (112 bytes); seven is odd, so eight
//            consecutive rows already land on eight distinct bank groups
//            without any folding.
template <int BK>
struct NtEpiLayout {
    static_assert(BK == 24 || BK == 32, "supported reduction tiles");
    static constexpr int Steps = BK / 8;
    static constexpr int ChunksPerRow = BK / 4;
    static constexpr int RowStride = BK == 32 ? 32 : 28;
    static constexpr unsigned RowBytes = (unsigned)RowStride * 4U;
    __device__ static __forceinline__ int slot(int row, int reduction) {
        if constexpr (BK == 32) {
            return row * 32 + (reduction ^ ((row & 7) << 2));
        } else {
            return row * RowStride + reduction;
        }
    }
};

template <int Rows, int Threads, int BK>
struct NtEpiOperandPlan {
    static constexpr int Chunks = Rows * NtEpiLayout<BK>::ChunksPerRow;
    static constexpr int Slices = (Chunks + Threads - 1) / Threads;
    const float* source[Slices];
    unsigned destination[Slices];
    int chunk_offset[Slices];
    bool valid[Slices];
};

template <int Rows, int Threads, int BK>
__device__ __forceinline__ void nt_epi_plan(
    NtEpiOperandPlan<Rows, Threads, BK>& plan, unsigned plane_base,
    const float* global, int leading, int tile_origin, int extent) {
    using Plan = NtEpiOperandPlan<Rows, Threads, BK>;
    constexpr int ChunksPerRow = NtEpiLayout<BK>::ChunksPerRow;
#pragma unroll
    for (int slice = 0; slice < Plan::Slices; ++slice) {
        int linear = (int)threadIdx.x + slice * Threads;
        int row = linear / ChunksPerRow;
        int chunk_offset = (linear % ChunksPerRow) * 4;
        bool in_tile = linear < Plan::Chunks;
        int global_row = tile_origin + row;
        bool row_valid = in_tile && global_row < extent;
        plan.valid[slice] = row_valid;
        plan.chunk_offset[slice] = chunk_offset;
        plan.source[slice] =
            global + (long long)(row_valid ? global_row : 0) * leading + chunk_offset;
        plan.destination[slice] = in_tile
            ? plane_base + (unsigned)NtEpiLayout<BK>::slot(row, chunk_offset) * 4U
            : plane_base;
    }
}

template <int Rows, int Threads, int BK>
__device__ __forceinline__ void nt_epi_issue_slice(
    const NtEpiOperandPlan<Rows, Threads, BK>& plan, int slice,
    unsigned stage_bytes, int reduction_base, int reduction) {
    using Plan = NtEpiOperandPlan<Rows, Threads, BK>;
    if (slice < Plan::Slices) {
        bool in_tile = (int)threadIdx.x + slice * Threads < Plan::Chunks;
        if (in_tile) {
            int remaining = reduction - reduction_base - plan.chunk_offset[slice];
            remaining = remaining < 0 ? 0 : (remaining > 4 ? 4 : remaining);
            nt_epi_copy_cg(
                plan.destination[slice] + stage_bytes, plan.source[slice],
                plan.valid[slice] ? remaining * 4 : 0);
        }
    }
}

template <int Rows, int Threads, int BK>
__device__ __forceinline__ void nt_epi_advance(
    NtEpiOperandPlan<Rows, Threads, BK>& plan) {
#pragma unroll
    for (int slice = 0; slice < NtEpiOperandPlan<Rows, Threads, BK>::Slices; ++slice) {
        plan.source[slice] += BK;
    }
}

template <int BM, int BN, int WarpsM, int WarpsN, int Stages, int BK>
struct NtEpiConfig {
    static constexpr int Threads = 32 * WarpsM * WarpsN;
    static constexpr int WM = BM / WarpsM;
    static constexpr int WN = BN / WarpsN;
    static constexpr int MAtoms = WM / 16;
    static constexpr int NAtoms = WN / 8;
    static constexpr int NPairs = NAtoms / 2;
    static constexpr bool NOdd = (NAtoms % 2) == 1;
    static constexpr unsigned AStageBytes = (unsigned)BM * NtEpiLayout<BK>::RowBytes;
    static constexpr unsigned BStageBytes = (unsigned)BN * NtEpiLayout<BK>::RowBytes;
    static constexpr unsigned StageBytes = AStageBytes + BStageBytes;
    static constexpr unsigned SharedBytes = StageBytes * (unsigned)Stages;
    static_assert(WM % 16 == 0, "warp rows must be whole m16 atoms");
    static_assert(WN % 8 == 0, "warp columns must be whole n8 atoms");
    static_assert(Stages >= 2, "at least two stages");
};

template <int MAtoms, int NAtoms>
struct NtEpiFragments {
    unsigned a[MAtoms][4];
    unsigned b[NAtoms][2];
};

// Loads the fragments of one k8 step. The per-step offsets are held per
// lane; atoms and atom pairs sit at a fixed sixteen-row stride.
// An odd trailing n-atom comes from an x2 ldmatrix: lanes 0-7 and 8-15
// address its eight rows at the two k chunks, the same rows the x4 form
// uses for its first pair, so the trailing atom sits at the pair stride.
template <int MAtoms, int NAtoms, int BK, bool Rna>
__device__ __forceinline__ void nt_epi_load_fragments(
    unsigned a_address, unsigned b_address,
    NtEpiFragments<MAtoms, NAtoms>& fragments) {
    constexpr int NPairs = NAtoms / 2;
    constexpr unsigned AtomBytes = 16U * NtEpiLayout<BK>::RowBytes;
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
        unsigned raw[4];
        nt_epi_ldmatrix_x4(raw, a_address + (unsigned)m_atom * AtomBytes);
        fragments.a[m_atom][0] = nt_epi_round<Rna>(raw[0]);
        fragments.a[m_atom][1] = nt_epi_round<Rna>(raw[1]);
        fragments.a[m_atom][2] = nt_epi_round<Rna>(raw[2]);
        fragments.a[m_atom][3] = nt_epi_round<Rna>(raw[3]);
    }
#pragma unroll
    for (int pair = 0; pair < NPairs; ++pair) {
        unsigned raw[4];
        nt_epi_ldmatrix_x4(raw, b_address + (unsigned)pair * AtomBytes);
        fragments.b[2 * pair][0] = nt_epi_round<Rna>(raw[0]);
        fragments.b[2 * pair][1] = nt_epi_round<Rna>(raw[1]);
        fragments.b[2 * pair + 1][0] = nt_epi_round<Rna>(raw[2]);
        fragments.b[2 * pair + 1][1] = nt_epi_round<Rna>(raw[3]);
    }
    if constexpr ((NAtoms % 2) == 1) {
        unsigned raw[2];
        nt_epi_ldmatrix_x2(raw, b_address + (unsigned)NPairs * AtomBytes);
        fragments.b[NAtoms - 1][0] = nt_epi_round<Rna>(raw[0]);
        fragments.b[NAtoms - 1][1] = nt_epi_round<Rna>(raw[1]);
    }
}

template <int MAtoms, int NAtoms>
__device__ __forceinline__ void nt_epi_mma_step(
    const NtEpiFragments<MAtoms, NAtoms>& fragments,
    float (&acc)[MAtoms][NAtoms][4]) {
#pragma unroll
    for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            nt_epi_mma(acc[m_atom][n_atom], fragments.a[m_atom], fragments.b[n_atom]);
        }
    }
}

template <int BM, int BN, int WarpsM, int WarpsN, int Stages, int BK, bool Rna>
__device__ __forceinline__ void nt_epi_kernel(
    float* output, const float* a, const float* b, NtEpiParams params) {
    using Cfg = NtEpiConfig<BM, BN, WarpsM, WarpsN, Stages, BK>;
    using Layout = NtEpiLayout<BK>;
    constexpr int Threads = Cfg::Threads;
    constexpr int MAtoms = Cfg::MAtoms;
    constexpr int NAtoms = Cfg::NAtoms;
    constexpr int Steps = Layout::Steps;

    int column_tiles = (params.k + BN - 1) / BN;
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

    // Per-step ldmatrix addresses in stage 0.
    int a_row = warp_m + (lane & 15);
    int a_k = (lane >> 4) << 2;
    int b_row = warp_n + (((lane >> 4) & 1) << 3) + (lane & 7);
    int b_k = ((lane >> 3) & 1) << 2;
    unsigned a_step_address[Steps];
    unsigned b_step_address[Steps];
#pragma unroll
    for (int step = 0; step < Steps; ++step) {
        a_step_address[step] = a_plane + (unsigned)Layout::slot(a_row, step * 8 + a_k) * 4U;
        b_step_address[step] = b_plane + (unsigned)Layout::slot(b_row, step * 8 + b_k) * 4U;
    }

    NtEpiOperandPlan<BM, Threads, BK> a_plan;
    NtEpiOperandPlan<BN, Threads, BK> b_plan;
    nt_epi_plan<BM, Threads, BK>(a_plan, a_plane, a, params.lda, tile_row, params.m);
    nt_epi_plan<BN, Threads, BK>(b_plan, b_plane, b, params.ldb, tile_column, params.k);

    constexpr int ASlices = NtEpiOperandPlan<BM, Threads, BK>::Slices;
    constexpr int BSlices = NtEpiOperandPlan<BN, Threads, BK>::Slices;
    constexpr int IssueSlots = ASlices > BSlices ? ASlices : BSlices;

    unsigned tile_count = ((unsigned)params.n + (unsigned)BK - 1U) / (unsigned)BK;

    // Prologue: the first Stages-1 tiles in flight.
#pragma unroll
    for (int stage = 0; stage < Stages - 1; ++stage) {
        if ((unsigned)stage < tile_count) {
#pragma unroll
            for (int slot = 0; slot < IssueSlots; ++slot) {
                nt_epi_issue_slice<BM, Threads, BK>(a_plan, slot, Cfg::AStageBytes * (unsigned)stage, stage * BK, params.n);
                nt_epi_issue_slice<BN, Threads, BK>(b_plan, slot, Cfg::BStageBytes * (unsigned)stage, stage * BK, params.n);
            }
            nt_epi_advance<BM, Threads, BK>(a_plan);
            nt_epi_advance<BN, Threads, BK>(b_plan);
        }
        nt_epi_commit();
    }

    // Main loop, unrolled by Stages so every stage offset is an immediate.
    for (unsigned tile_base = 0; tile_base < tile_count; tile_base += (unsigned)Stages) {
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            unsigned tile = tile_base + (unsigned)stage;
            if (tile < tile_count) {
                nt_epi_wait<Stages - 2>();
                __syncthreads();
                unsigned next = tile + (unsigned)(Stages - 1);
                bool has_next = next < tile_count;
                int write_stage = (stage + Stages - 1) % Stages;
                unsigned write_a = Cfg::AStageBytes * (unsigned)write_stage;
                unsigned write_b = Cfg::BStageBytes * (unsigned)write_stage;
                unsigned read_a = Cfg::AStageBytes * (unsigned)stage;
                unsigned read_b = Cfg::BStageBytes * (unsigned)stage;
                int next_base = (int)next * BK;

                NtEpiFragments<MAtoms, NAtoms> fragments[2];
                nt_epi_load_fragments<MAtoms, NAtoms, BK, Rna>(
                    a_step_address[0] + read_a, b_step_address[0] + read_b, fragments[0]);
#pragma unroll
                for (int step = 0; step < Steps; ++step) {
                    if (has_next) {
                        // Spread the copy issue over the k8 steps.
#pragma unroll
                        for (int slot = step; slot < IssueSlots; slot += Steps) {
                            nt_epi_issue_slice<BM, Threads, BK>(a_plan, slot, write_a, next_base, params.n);
                            nt_epi_issue_slice<BN, Threads, BK>(b_plan, slot, write_b, next_base, params.n);
                        }
                    }
                    if (step < Steps - 1) {
                        nt_epi_load_fragments<MAtoms, NAtoms, BK, Rna>(
                            a_step_address[step + 1] + read_a,
                            b_step_address[step + 1] + read_b,
                            fragments[(step + 1) & 1]);
                    }
                    nt_epi_mma_step<MAtoms, NAtoms>(fragments[step & 1], acc);
                }
                nt_epi_commit();
                if (has_next) {
                    nt_epi_advance<BM, Threads, BK>(a_plan);
                    nt_epi_advance<BN, Threads, BK>(b_plan);
                }
            }
        }
    }
    nt_epi_wait<0>();

    // Epilogue: the accumulators go back through the stage planes so the
    // global writes are whole float4 rows. Each lane holds two adjacent
    // columns of a row, which as a direct store would touch a 32-byte
    // piece of eight different rows per instruction; staging turns that
    // into one contiguous row per four lanes. The tile is written in row
    // chunks that fit the shared allocation.
    constexpr int RowPad = BN + 8;
    constexpr int ChunkRows = (int)(Cfg::SharedBytes / (unsigned)(RowPad * 4)) >= BM
        ? BM
        : ((int)(Cfg::SharedBytes / (unsigned)(RowPad * 4)) / 16) * 16;
    static_assert(ChunkRows >= 16, "the staged epilogue needs at least one atom row chunk");
    float* tile = reinterpret_cast<float*>(shared_bytes);
    bool row_vector = (params.ldc & 3) == 0
        && ((reinterpret_cast<unsigned long long>(output) & 15ULL) == 0ULL)
        && tile_column + BN <= params.k;
    if (row_vector) {
        for (int base = 0; base < BM; base += ChunkRows) {
            __syncthreads();
#pragma unroll
            for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
#pragma unroll
                for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        int row = warp_m + m_atom * 16 + group + half * 8;
                        int local = row - base;
                        if (local < 0 || local >= ChunkRows) continue;
                        int column = warp_n + n_atom * 8 + 2 * thread;
                        *reinterpret_cast<float2*>(tile + local * RowPad + column) =
                            make_float2(acc[m_atom][n_atom][2 * half],
                                        acc[m_atom][n_atom][2 * half + 1]);
                    }
                }
            }
            __syncthreads();
            for (int linear = (int)threadIdx.x; linear < ChunkRows * (BN / 4);
                 linear += Threads) {
                int local = linear / (BN / 4);
                int column = (linear % (BN / 4)) * 4;
                // The last chunk runs past the tile when BM is not a whole
                // number of chunks; those rows hold stale staged words.
                if (base + local >= BM) continue;
                int global_row = tile_row + base + local;
                if (global_row >= params.m) continue;
                float4 value = *reinterpret_cast<const float4*>(tile + local * RowPad + column);
                if (params.alpha != 1.0f) {
                    value.x = __fmul_rn(params.alpha, value.x);
                    value.y = __fmul_rn(params.alpha, value.y);
                    value.z = __fmul_rn(params.alpha, value.z);
                    value.w = __fmul_rn(params.alpha, value.w);
                }
                *reinterpret_cast<float4*>(
                    output + (long long)global_row * params.ldc + tile_column + column) = value;
            }
        }
        return;
    }
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
                if (params.alpha != 1.0f) {
                    v0 = __fmul_rn(params.alpha, v0);
                    v1 = __fmul_rn(params.alpha, v1);
                }
                float* destination = output + (long long)row * params.ldc + column;
                if (column < params.k) destination[0] = v0;
                if (column + 1 < params.k) destination[1] = v1;
            }
        }
    }
}

// Row-staged epilogue variants.

extern "C" __global__ __launch_bounds__(256, 1)
void nt_sm89_tf32_rowstage_m128n192_w2x4_bk32_s2(
    float* output, const float* a, const float* b, const float* bias,
    NtEpiParams params) {
    (void)bias;
    nt_epi_kernel<128, 192, 2, 4, 2, 32, false>(output, a, b, params);
}

