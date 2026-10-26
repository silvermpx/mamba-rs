// The batch-invariant exact forward's thin tiles: 16 rows x 32 columns per
// CTA, a lane per column and ROWS rows per thread. Every output is one
// chain, the chain of every exact tile and of a CPU twin: the bias seeds the
// accumulator (+0.0 without one), then one __fmaf_rn per input in ascending
// k. The other tiles fold the reduction in 16-deep slabs and zero-fill a
// ragged last one, whose zero products turn a -0.0 accumulator into +0.0;
// these tiles fold only the real inputs and end a ragged reduction with
// that one +0.0 add, so all of them store the same value, signed zeros
// included.
//
// The same tiles fold the chunk partials of the deep reductions: with
// `slab` below `k`, CTA z of the grid's third axis folds inputs
// [z * slab, z * slab + slab) into partial plane z, from +0.0 and with no
// bias - the planes `nn_splitk_slim_partial` writes, which `splitk_reduce`
// then sums in order.
//
// Every tile also serves the input gradient: with B_T the B operand is the
// forward's weight W stored [columns][k], so the stage element (k, column)
// is read from W[column][k]. Only the loader changes; the slabs, the chains
// and the planes are the forward's, so an input gradient folds its
// reduction exactly as a forward over the transposed weight would.
//
// A batch of one or a few rows has few outputs and long chains, and a
// chain cannot be cut. One row per thread puts as many warps on the machine
// as the batch has outputs. A six-stage cp.async ring keeps the next
// 64-deep slabs in flight - enough to cover an L2 round trip - and a thread
// reads its inputs a group of eight ahead of its FMAs, so a chain waits on
// its own FMA latency rather than on the loads. A warp whose rows all lie
// past the batch loads and waits at the barriers but folds nothing.

#define EXACT_THIN_BM 16
#define EXACT_THIN_BN 32
#define EXACT_THIN_BK 64
#define EXACT_THIN_STAGES 6
#define EXACT_THIN_GROUP 8
#define EXACT_THIN_A_STAGE (EXACT_THIN_BM * EXACT_THIN_BK)
#define EXACT_THIN_B_STAGE (EXACT_THIN_BK * EXACT_THIN_BN)

struct ExactThinParams {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    // The inputs one CTA folds: k for a whole chain, the slab length when
    // the grid's third axis cuts the reduction into partial planes.
    int slab;
    // The padding grain the +0.0 close stands for: a plane whose length is
    // not a multiple of it ends with one +0.0 add, the zero products of
    // the grain-deep slab the other kernels zero-fill (16 for a whole
    // chain, 32 for a split-K chunk); 0 never closes.
    int close;
};

static_assert(sizeof(ExactThinParams) == 32, "thin tile parameter ABI drift");
static_assert(alignof(ExactThinParams) == 4, "thin tile parameter alignment drift");
static_assert(EXACT_THIN_BN == 32, "a lane owns one column");
static_assert(EXACT_THIN_BK % EXACT_THIN_GROUP == 0, "whole groups per slab");
static_assert(EXACT_THIN_GROUP % 4 == 0, "a group is whole 16-byte row words");
static_assert(
    EXACT_THIN_STAGES * (EXACT_THIN_A_STAGE + EXACT_THIN_B_STAGE) * 4 == 73728,
    "the dynamic shared memory the loader opts the thin tile into");

// One 16-byte word of a slab row, or four 4-byte copies where the row
// stride, the base or an edge forbids the word. Words past an edge are
// zero-filled.
__device__ __forceinline__ void exact_thin_copy_word(
    unsigned dst, const float* row_base, int first, int limit, bool vector
) {
    if (vector && first + 3 < limit) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst), "l"(row_base + first));
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            bool valid = first + j < limit;
            const float* src = valid ? row_base + first + j : row_base;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(dst + (unsigned)(j * 4)), "l"(src), "r"(valid ? 4 : 0));
        }
    }
}

// One stage element of a transposed B: (k, column) from W[column][k], or a
// zero past an edge.
__device__ __forceinline__ void exact_copy_bt(
    unsigned dst, const float* B, int ldb, int g_k, int k_limit, int g_column, int n
) {
    const bool valid = g_k < k_limit && g_column < n;
    const float* src = valid ? B + (long long)g_column * ldb + g_k : B;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
}

template <int ROWS, bool B_T>
__device__ __forceinline__ void exact_thin_tile(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const ExactThinParams& params,
    float* smem
) {
    constexpr int THREADS = (EXACT_THIN_BM / ROWS) * 32;
    // A slabs are row-major, so a warp reads a row's next four inputs as
    // one 16-byte broadcast; B slabs are row-major, so a warp reads 32
    // consecutive columns.
    float* As = smem;
    float* Bs = smem + EXACT_THIN_STAGES * EXACT_THIN_A_STAGE;

    // Plane z folds its own slab of the reduction into its own plane.
    const int k_begin = blockIdx.z * params.slab;
    const int k_len = min(params.slab, params.k - k_begin);
    A += k_begin;
    B += B_T ? (long long)k_begin : (long long)k_begin * params.ldb;
    C += (long long)blockIdx.z * params.m * params.ldc;

    const int row_tile = blockIdx.y * EXACT_THIN_BM;
    const int column_tile = blockIdx.x * EXACT_THIN_BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_first = row_tile + warp * ROWS;
    const int column = column_tile + lane;
    const bool live = row_first < params.m;

    float acc[ROWS];
    {
        const float seed = (bias != nullptr && column < params.n) ? bias[column] : 0.0f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[r] = seed;
        }
    }

    const bool a_rows_vector = ((params.lda & 3) == 0) && is_aligned_16(A);
    const bool b_rows_vector = ((params.ldb & 3) == 0) && is_aligned_16(B);
    const int tiles = (k_len + EXACT_THIN_BK - 1) / EXACT_THIN_BK;

    // One slab: A [16 rows x 64 k], 256 words; B [64 k x 32 columns], 512
    // words. A row or a k past an edge copies zeros.
    auto issue = [&](int stage, int k0) {
        for (int word = threadIdx.x; word < EXACT_THIN_A_STAGE / 4; word += THREADS) {
            int r = word / (EXACT_THIN_BK / 4);
            int k4 = (word % (EXACT_THIN_BK / 4)) * 4;
            int g_row = row_tile + r;
            unsigned dst = (unsigned)__cvta_generic_to_shared(
                As + stage * EXACT_THIN_A_STAGE + r * EXACT_THIN_BK + k4);
            if (g_row < params.m) {
                exact_thin_copy_word(dst, A + (long long)g_row * params.lda, k0 + k4,
                                     k_len, a_rows_vector);
            } else {
                exact_thin_copy_word(dst, A, 0, 0, false);
            }
        }
        if (B_T) {
            // A lane per k down a column's row of W, so a warp reads 32
            // consecutive inputs of one column.
            for (int e = threadIdx.x; e < EXACT_THIN_B_STAGE; e += THREADS) {
                int kk = e % EXACT_THIN_BK;
                int c = e / EXACT_THIN_BK;
                unsigned dst = (unsigned)__cvta_generic_to_shared(
                    Bs + stage * EXACT_THIN_B_STAGE + kk * EXACT_THIN_BN + c);
                exact_copy_bt(dst, B, params.ldb, k0 + kk, k_len, column_tile + c, params.n);
            }
        } else {
            for (int word = threadIdx.x; word < EXACT_THIN_B_STAGE / 4; word += THREADS) {
                int kk = word / (EXACT_THIN_BN / 4);
                int c4 = (word % (EXACT_THIN_BN / 4)) * 4;
                int g_k = k0 + kk;
                unsigned dst = (unsigned)__cvta_generic_to_shared(
                    Bs + stage * EXACT_THIN_B_STAGE + kk * EXACT_THIN_BN + c4);
                if (g_k < k_len) {
                    exact_thin_copy_word(dst, B + (long long)g_k * params.ldb, column_tile + c4,
                                         params.n, b_rows_vector);
                } else {
                    exact_thin_copy_word(dst, B, 0, 0, false);
                }
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < EXACT_THIN_STAGES - 1; ++s) {
        if (s < tiles) {
            issue(s, s * EXACT_THIN_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }

    for (int t = 0; t < tiles; ++t) {
        // Slab t has landed once all but the newest STAGES - 2 groups are
        // done; the barrier also retires every thread's reads of the stage
        // the next issue overwrites, the one slab t - 1 folded from.
        asm volatile("cp.async.wait_group %0;\n" :: "n"(EXACT_THIN_STAGES - 2));
        __syncthreads();
        int next = t + EXACT_THIN_STAGES - 1;
        if (next < tiles) {
            issue(next % EXACT_THIN_STAGES, next * EXACT_THIN_BK);
        }
        asm volatile("cp.async.commit_group;\n");
        if (!live) {
            continue;
        }

        const int stage = t % EXACT_THIN_STAGES;
        const float* a_rows = As + stage * EXACT_THIN_A_STAGE + warp * ROWS * EXACT_THIN_BK;
        const float* b_column = Bs + stage * EXACT_THIN_B_STAGE + lane;
        const int depth = min(EXACT_THIN_BK, k_len - t * EXACT_THIN_BK);
        if (depth == EXACT_THIN_BK) {
            float a_now[ROWS][EXACT_THIN_GROUP];
            float b_now[EXACT_THIN_GROUP];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int i = 0; i < EXACT_THIN_GROUP; i += 4) {
                    float4 word = *reinterpret_cast<const float4*>(
                        a_rows + r * EXACT_THIN_BK + i);
                    a_now[r][i] = word.x;
                    a_now[r][i + 1] = word.y;
                    a_now[r][i + 2] = word.z;
                    a_now[r][i + 3] = word.w;
                }
            }
            #pragma unroll
            for (int i = 0; i < EXACT_THIN_GROUP; ++i) {
                b_now[i] = b_column[i * EXACT_THIN_BN];
            }
            #pragma unroll
            for (int g = 0; g < EXACT_THIN_BK / EXACT_THIN_GROUP; ++g) {
                float a_next[ROWS][EXACT_THIN_GROUP];
                float b_next[EXACT_THIN_GROUP];
                if (g + 1 < EXACT_THIN_BK / EXACT_THIN_GROUP) {
                    const int k1 = (g + 1) * EXACT_THIN_GROUP;
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        #pragma unroll
                        for (int i = 0; i < EXACT_THIN_GROUP; i += 4) {
                            float4 word = *reinterpret_cast<const float4*>(
                                a_rows + r * EXACT_THIN_BK + k1 + i);
                            a_next[r][i] = word.x;
                            a_next[r][i + 1] = word.y;
                            a_next[r][i + 2] = word.z;
                            a_next[r][i + 3] = word.w;
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < EXACT_THIN_GROUP; ++i) {
                        b_next[i] = b_column[(k1 + i) * EXACT_THIN_BN];
                    }
                }
                #pragma unroll
                for (int i = 0; i < EXACT_THIN_GROUP; ++i) {
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        acc[r] = __fmaf_rn(a_now[r][i], b_now[i], acc[r]);
                    }
                }
                if (g + 1 < EXACT_THIN_BK / EXACT_THIN_GROUP) {
                    #pragma unroll
                    for (int i = 0; i < EXACT_THIN_GROUP; ++i) {
                        #pragma unroll
                        for (int r = 0; r < ROWS; ++r) {
                            a_now[r][i] = a_next[r][i];
                        }
                        b_now[i] = b_next[i];
                    }
                }
            }
        } else {
            for (int kk = 0; kk < depth; ++kk) {
                const float b = b_column[kk * EXACT_THIN_BN];
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[r] = __fmaf_rn(a_rows[r * EXACT_THIN_BK + kk], b, acc[r]);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    if (live && column < params.n) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float value = acc[r];
            if (params.close > 0 && (k_len % params.close) != 0) {
                value = __fadd_rn(value, 0.0f);
            }
            if (row_first + r < params.m) {
                C[(long long)(row_first + r) * params.ldc + column] = value;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(512)
void nn_exact_thin_r1_m16n32_bk64_s6(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    extern __shared__ __align__(16) float exact_thin_smem[];
    exact_thin_tile<1, false>(C, A, B, bias, params, exact_thin_smem);
}

// The thin tile of an input gradient: dX = dY W^T, B the weight [n][k].
extern "C" __global__ __launch_bounds__(512)
void nt_exact_thin_r1_m16n32_bk64_s6(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    extern __shared__ __align__(16) float exact_thin_smem[];
    exact_thin_tile<1, true>(C, A, B, bias, params, exact_thin_smem);
}

// The chunk partial on a small tile: 32 rows x 64 columns per CTA on 128
// threads, four rows by four columns per thread, chunk z of the grid's
// third axis into partial plane z. Each output folds its chunk the way
// `nn_splitk_slim_partial` folds it - from +0.0, ascending k, 32-deep
// slabs whose ragged tail is zero-filled - so the two write the same
// planes bit for bit; this one spreads a thin batch over four times the
// CTAs of the 128-row slim tile. Two cp.async stages: the chunk's second
// slab loads while the first folds.
#define EXACT_CHUNK_BM 32
#define EXACT_CHUNK_BN 64
#define EXACT_CHUNK_BK 32
#define EXACT_CHUNK_TM 4
#define EXACT_CHUNK_TN 4
#define EXACT_CHUNK_A_ROW (EXACT_CHUNK_BM + 4)
#define EXACT_CHUNK_B_ROW (EXACT_CHUNK_BN + 4)

struct ExactChunkParams {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int chunk;
    int pad;
};

static_assert(sizeof(ExactChunkParams) == 32, "chunk partial parameter ABI drift");
static_assert(alignof(ExactChunkParams) == 4, "chunk partial parameter alignment drift");

template <bool B_T>
__device__ __forceinline__ void exact_chunk_partial_m32n64_tile(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const ExactChunkParams& params
) {
    const int M = params.m;
    const int N = params.n;
    const int K = params.k;
    const int lda = params.lda;
    const int ldb = params.ldb;
    const int chunk = params.chunk;
    // A slabs k-major (a thread's four rows side by side), B slabs row-major.
    __shared__ __align__(16) float As[2][EXACT_CHUNK_BK * EXACT_CHUNK_A_ROW];
    __shared__ __align__(16) float Bs[2][EXACT_CHUNK_BK * EXACT_CHUNK_B_ROW];

    const int z = blockIdx.z;
    const int k_begin = z * chunk;
    if (k_begin >= K) {
        return;
    }
    const int k_end = min(K, k_begin + chunk);
    const int num_pid_n = (N + EXACT_CHUNK_BN - 1) / EXACT_CHUNK_BN;
    const int pid_m = blockIdx.x / num_pid_n;
    const int pid_n = blockIdx.x % num_pid_n;
    const int row_tile = pid_m * EXACT_CHUNK_BM;
    const int column_tile = pid_n * EXACT_CHUNK_BN;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Four warps in a 2 x 2 grid of 16 x 32; a lane owns 4 x 4 outputs.
    const int row0 = (warp >> 1) * 16 + (lane >> 3) * EXACT_CHUNK_TM;
    const int col0 = (warp & 1) * 32 + (lane & 7) * EXACT_CHUNK_TN;

    const bool b_rows_vector = ((ldb & 3) == 0) && is_aligned_16(B);

    auto issue = [&](int stage, int k0) {
        // A [32 rows x 32 k]: 1024 words, eight per thread, row fastest so
        // the k-major stage takes them without a bank conflict.
        unsigned a_base = (unsigned)__cvta_generic_to_shared(&As[stage][0]);
        #pragma unroll
        for (int i = 0; i < (EXACT_CHUNK_BM * EXACT_CHUNK_BK) / 128; ++i) {
            int e = threadIdx.x + i * 128;
            int r = e % EXACT_CHUNK_BM;
            int kk = e / EXACT_CHUNK_BM;
            int g_row = row_tile + r;
            int g_k = k0 + kk;
            bool valid = g_row < M && g_k < k_end;
            const float* src = valid ? A + (long long)g_row * lda + g_k : A;
            unsigned dst = a_base + (unsigned)((kk * EXACT_CHUNK_A_ROW + r) * 4);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
        }
        // B [32 k x 64 columns]: 512 words of 16 bytes, four per thread, or
        // for a transposed B 2048 elements, a lane per k of a column's row.
        unsigned b_base = (unsigned)__cvta_generic_to_shared(&Bs[stage][0]);
        if (B_T) {
            #pragma unroll
            for (int i = 0; i < (EXACT_CHUNK_BK * EXACT_CHUNK_BN) / 128; ++i) {
                int e = threadIdx.x + i * 128;
                int kk = e % EXACT_CHUNK_BK;
                int c = e / EXACT_CHUNK_BK;
                unsigned dst = b_base + (unsigned)((kk * EXACT_CHUNK_B_ROW + c) * 4);
                exact_copy_bt(dst, B, ldb, k0 + kk, k_end, column_tile + c, N);
            }
            return;
        }
        #pragma unroll
        for (int i = 0; i < (EXACT_CHUNK_BK * EXACT_CHUNK_BN) / (4 * 128); ++i) {
            int e = threadIdx.x + i * 128;
            int kk = e / (EXACT_CHUNK_BN / 4);
            int c4 = (e % (EXACT_CHUNK_BN / 4)) * 4;
            int g_k = k0 + kk;
            int g_column = column_tile + c4;
            unsigned dst = b_base + (unsigned)((kk * EXACT_CHUNK_B_ROW + c4) * 4);
            if (b_rows_vector && g_k < k_end && g_column + 3 < N) {
                const float* src = B + (long long)g_k * ldb + g_column;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst), "l"(src));
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bool valid = g_k < k_end && g_column + j < N;
                    const float* src = valid ? B + (long long)g_k * ldb + g_column + j : B;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                                 :: "r"(dst + (unsigned)(j * 4)), "l"(src),
                                    "r"(valid ? 4 : 0));
                }
            }
        }
    };

    float acc[EXACT_CHUNK_TM][EXACT_CHUNK_TN];
    #pragma unroll
    for (int i = 0; i < EXACT_CHUNK_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < EXACT_CHUNK_TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int tiles = (k_end - k_begin + EXACT_CHUNK_BK - 1) / EXACT_CHUNK_BK;
    issue(0, k_begin);
    asm volatile("cp.async.commit_group;\n");
    for (int t = 0; t < tiles; ++t) {
        if (t + 1 < tiles) {
            issue((t + 1) & 1, k_begin + (t + 1) * EXACT_CHUNK_BK);
            asm volatile("cp.async.commit_group;\n");
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        const float* a_stage = &As[t & 1][row0];
        const float* b_stage = &Bs[t & 1][col0];
        #pragma unroll
        for (int kk = 0; kk < EXACT_CHUNK_BK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(a_stage + kk * EXACT_CHUNK_A_ROW);
            const float4 b = *reinterpret_cast<const float4*>(b_stage + kk * EXACT_CHUNK_B_ROW);
            const float av[4] = {a.x, a.y, a.z, a.w};
            const float bv[4] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < EXACT_CHUNK_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < EXACT_CHUNK_TN; ++j) {
                    acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    const long long plane_stride = (long long)M * N;
    float* plane = partial + (long long)z * plane_stride;
    #pragma unroll
    for (int i = 0; i < EXACT_CHUNK_TM; ++i) {
        const int g_row = row_tile + row0 + i;
        if (g_row >= M) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < EXACT_CHUNK_TN; ++j) {
            const int g_column = column_tile + col0 + j;
            if (g_column < N) {
                __stcg(plane + (long long)g_row * N + g_column, acc[i][j]);
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 4)
void exact_chunk_partial_m32n64(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    ExactChunkParams params
) {
    exact_chunk_partial_m32n64_tile<false>(partial, A, B, params);
}

// The chunk partial of an input gradient, B the weight [n][k].
extern "C" __global__ __launch_bounds__(128, 4)
void nt_exact_chunk_partial_m32n64(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    ExactChunkParams params
) {
    exact_chunk_partial_m32n64_tile<true>(partial, A, B, params);
}

#undef EXACT_CHUNK_B_ROW
#undef EXACT_CHUNK_A_ROW
#undef EXACT_CHUNK_TN
#undef EXACT_CHUNK_TM
#undef EXACT_CHUNK_BK
#undef EXACT_CHUNK_BN
#undef EXACT_CHUNK_BM

// The 16 x 16 tile of the thin batches: `nn_narrow_small`'s
// outputs on 64 threads, two by two per thread, the same 16-deep slabs
// zero-filled past the reduction - so the same chain and the same padding,
// bit for bit - with a four-stage cp.async ring in place of its
// load-then-wait slab, so the next slabs are in flight while one folds. A
// k-row of the A stage is 18 words: a warp's 16 k by 2 rows then land on 32
// distinct banks. It takes the thin tiles' parameters and folds partial
// planes the same way: plane z of the grid's third axis folds inputs
// [z * slab, z * slab + slab) from +0.0, and a plane whose length is not a
// multiple of `close` ends with one +0.0 add - redundant after the
// zero-filled slab of a whole chain, the 32-deep grain's tail on a chunk.
#define EXACT_C16_BM 16
#define EXACT_C16_BN 16
#define EXACT_C16_BK 16
#define EXACT_C16_STAGES 4
#define EXACT_C16_A_ROW (EXACT_C16_BM + 2)
#define EXACT_C16_B_ROW EXACT_C16_BN

template <bool B_T>
__device__ __forceinline__ void exact_m16n16_tile(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const ExactThinParams& params
) {
    __shared__ __align__(16) float As[EXACT_C16_STAGES][EXACT_C16_BK * EXACT_C16_A_ROW];
    __shared__ __align__(16) float Bs[EXACT_C16_STAGES][EXACT_C16_BK * EXACT_C16_B_ROW];

    const int k_begin = blockIdx.z * params.slab;
    const int k_len = min(params.slab, params.k - k_begin);
    A += k_begin;
    B += B_T ? (long long)k_begin : (long long)k_begin * params.ldb;
    C += (long long)blockIdx.z * params.m * params.ldc;

    const int row_tile = blockIdx.y * EXACT_C16_BM;
    const int column_tile = blockIdx.x * EXACT_C16_BN;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Two warps of 8 x 16; a lane owns 2 x 2.
    const int row0 = warp * 8 + (lane >> 3) * 2;
    const int col0 = (lane & 7) * 2;

    float acc[2][2];
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int g_column = column_tile + col0 + j;
        const float seed = (bias != nullptr && g_column < params.n) ? bias[g_column] : 0.0f;
        acc[0][j] = seed;
        acc[1][j] = seed;
    }

    const bool b_rows_vector = ((params.ldb & 3) == 0) && is_aligned_16(B);
    const int tiles = (k_len + EXACT_C16_BK - 1) / EXACT_C16_BK;

    auto issue = [&](int stage, int k0) {
        // A [16 rows x 16 k]: 256 words, four per thread, k-major stage; a
        // half-warp reads one row's 16 k, one 64-byte run.
        unsigned a_base = (unsigned)__cvta_generic_to_shared(&As[stage][0]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = lane & 15;
            const int r = warp * 8 + i * 2 + (lane >> 4);
            const int g_row = row_tile + r;
            const int g_k = k0 + kk;
            const bool valid = g_row < params.m && g_k < k_len;
            const float* src = valid ? A + (long long)g_row * params.lda + g_k : A;
            const unsigned dst = a_base + (unsigned)((kk * EXACT_C16_A_ROW + r) * 4);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
        }
        // B [16 k x 16 columns]: 64 words of 16 bytes, one per thread, or
        // for a transposed B 256 elements, four per thread, a half-warp per
        // column's 16 k.
        unsigned b_base = (unsigned)__cvta_generic_to_shared(&Bs[stage][0]);
        if (B_T) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int e = threadIdx.x + i * 64;
                const int kt = e & 15;
                const int c = e >> 4;
                exact_copy_bt(b_base + (unsigned)((kt * EXACT_C16_B_ROW + c) * 4), B,
                              params.ldb, k0 + kt, k_len, column_tile + c, params.n);
            }
            return;
        }
        const int kk = threadIdx.x >> 2;
        const int c4 = (threadIdx.x & 3) * 4;
        const int g_k = k0 + kk;
        const int g_column = column_tile + c4;
        const unsigned dst = b_base + (unsigned)((kk * EXACT_C16_B_ROW + c4) * 4);
        if (b_rows_vector && g_k < k_len && g_column + 3 < params.n) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst), "l"(B + (long long)g_k * params.ldb + g_column));
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool valid = g_k < k_len && g_column + j < params.n;
                const float* src = valid ? B + (long long)g_k * params.ldb + g_column + j : B;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                             :: "r"(dst + (unsigned)(j * 4)), "l"(src), "r"(valid ? 4 : 0));
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < EXACT_C16_STAGES - 1; ++s) {
        if (s < tiles) {
            issue(s, s * EXACT_C16_BK);
        }
        asm volatile("cp.async.commit_group;\n");
    }
    for (int t = 0; t < tiles; ++t) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(EXACT_C16_STAGES - 2));
        __syncthreads();
        const int next = t + EXACT_C16_STAGES - 1;
        if (next < tiles) {
            issue(next % EXACT_C16_STAGES, next * EXACT_C16_BK);
        }
        asm volatile("cp.async.commit_group;\n");
        const float* a_stage = &As[t % EXACT_C16_STAGES][row0];
        const float* b_stage = &Bs[t % EXACT_C16_STAGES][col0];
        #pragma unroll
        for (int kk = 0; kk < EXACT_C16_BK; ++kk) {
            const float2 a = *reinterpret_cast<const float2*>(a_stage + kk * EXACT_C16_A_ROW);
            const float2 b = *reinterpret_cast<const float2*>(b_stage + kk * EXACT_C16_B_ROW);
            acc[0][0] = __fmaf_rn(a.x, b.x, acc[0][0]);
            acc[0][1] = __fmaf_rn(a.x, b.y, acc[0][1]);
            acc[1][0] = __fmaf_rn(a.y, b.x, acc[1][0]);
            acc[1][1] = __fmaf_rn(a.y, b.y, acc[1][1]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    const bool close = params.close > 0 && (k_len % params.close) != 0;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int g_row = row_tile + row0 + i;
        if (g_row >= params.m) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int g_column = column_tile + col0 + j;
            if (g_column < params.n) {
                const float value = close ? __fadd_rn(acc[i][j], 0.0f) : acc[i][j];
                C[(long long)g_row * params.ldc + g_column] = value;
            }
        }
    }
}

// The whole chain of an output, seeded with its bias.
extern "C" __global__ __launch_bounds__(64)
void nn_exact_m16n16_bk16_s4(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_m16n16_tile<false>(C, A, B, bias, params);
}

// The whole chain of an input-gradient output, B the weight [n][k].
extern "C" __global__ __launch_bounds__(64)
void nt_exact_m16n16_bk16_s4(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_m16n16_tile<true>(C, A, B, bias, params);
}

// The same tile over a deep layer's chunks, one partial plane per chunk:
// a symbol of its own, since its output is a split-K partial rather than
// an output tile.
extern "C" __global__ __launch_bounds__(64)
void exact_chunk_partial_m16n16_bk16_s4(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_m16n16_tile<false>(partial, A, B, bias, params);
}

// The 16 x 16 chunk partial of an input gradient, B the weight [n][k].
extern "C" __global__ __launch_bounds__(64)
void nt_exact_chunk_partial_m16n16_bk16_s4(
    float* __restrict__ partial,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_m16n16_tile<true>(partial, A, B, bias, params);
}

// The 16 x 16 chunk partials with the planes folded by the CTA that finishes
// a tile last: every chunk stores its plane and counts in, and the last one
// to arrive reloads the tile's planes and adds them in ascending chunk order,
// then scales by alpha and adds the bias, the order `splitk_reduce` adds
// them in, so which CTA arrives last never moves a bit. The tile's counter
// wraps back to zero on the last arrival, so the next launch finds it clear,
// and no second launch is needed.
template <bool B_T>
__device__ __forceinline__ void exact_chunk_fused_m16n16(
    float* __restrict__ out,
    float* __restrict__ partial,
    unsigned* __restrict__ counters,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const ExactThinParams& params,
    float alpha
) {
    exact_m16n16_tile<B_T>(partial, A, B, nullptr, params);
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        const unsigned slot = blockIdx.y * gridDim.x + blockIdx.x;
        __threadfence();
        last = atomicInc(counters + slot, gridDim.z - 1U) == gridDim.z - 1U;
        if (last) {
            __threadfence();
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    // A thread folds four outputs of the tile, a quarter warp's run of 16
    // columns at a time. All four planes' loads of a chunk issue before its
    // additions, and the chunks unroll so the next loads overlap the adds:
    // a serial load-then-add per plane would wait one L2 round trip per
    // chunk in the tail of the launch. Each output still adds its planes in
    // ascending chunk order.
    constexpr int PerThread = EXACT_C16_BM * EXACT_C16_BN / 64;
    const long long plane = (long long)params.m * params.ldc;
    const int row_tile = blockIdx.y * EXACT_C16_BM;
    const int column_tile = blockIdx.x * EXACT_C16_BN;
    long long index[PerThread];
    bool valid[PerThread];
    float sum[PerThread];
    #pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int e = threadIdx.x + i * 64;
        const int g_row = row_tile + e / EXACT_C16_BN;
        const int g_column = column_tile + e % EXACT_C16_BN;
        valid[i] = g_row < params.m && g_column < params.n;
        index[i] = (long long)g_row * params.ldc + g_column;
        sum[i] = valid[i] ? __ldcg(partial + index[i]) : 0.0f;
    }
    #pragma unroll 4
    for (unsigned z = 1; z < gridDim.z; ++z) {
        const float* source = partial + (long long)z * plane;
        float value[PerThread];
        #pragma unroll
        for (int i = 0; i < PerThread; ++i) {
            value[i] = valid[i] ? __ldcg(source + index[i]) : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < PerThread; ++i) {
            sum[i] += value[i];
        }
    }
    #pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        if (!valid[i]) {
            continue;
        }
        float result = sum[i] * alpha;
        if (bias != nullptr) {
            result += bias[column_tile + (threadIdx.x + i * 64) % EXACT_C16_BN];
        }
        out[index[i]] = result;
    }
}

extern "C" __global__ __launch_bounds__(64)
void exact_chunk_fused_m16n16_bk16_s4(
    float* __restrict__ out,
    float* __restrict__ partial,
    unsigned* __restrict__ counters,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params,
    float alpha
) {
    exact_chunk_fused_m16n16<false>(out, partial, counters, A, B, bias, params, alpha);
}

extern "C" __global__ __launch_bounds__(64)
void nt_exact_chunk_fused_m16n16_bk16_s4(
    float* __restrict__ out,
    float* __restrict__ partial,
    unsigned* __restrict__ counters,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params,
    float alpha
) {
    exact_chunk_fused_m16n16<true>(out, partial, counters, A, B, bias, params, alpha);
}

// The 32 x 64 tile of a whole exact chain over operands read along the
// reduction: 128 threads, four rows by four columns per thread, 16-deep
// slabs zero-filled past the reduction (the 16 x 16 tile's slabs, so the
// same chain and the same padding, bit for bit), a four-stage cp.async ring
// of 16-byte words. A is staged [row][k], as it lies in memory; B is staged
// [column][k] for an input gradient (the weight [n][k]) and [k][column] for
// a forward. A thread reads a row's or a column's four consecutive k as one
// 16-byte word and folds them in ascending k, so a slab costs eight 16-byte
// reads per 64 products and no copy is split into 4-byte pieces.
#define EXACT_KM_BM 32
#define EXACT_KM_BN 64
#define EXACT_KM_BK 16
#define EXACT_KM_STAGES 4
#define EXACT_KM_ROW (EXACT_KM_BK + 4)
#define EXACT_KM_B_NN_ROW (EXACT_KM_BN + 4)

template <bool B_T>
struct ExactKmStage {
    float a[EXACT_KM_BM * EXACT_KM_ROW];
    float b[B_T ? EXACT_KM_BN * EXACT_KM_ROW : EXACT_KM_BK * EXACT_KM_B_NN_ROW];
};

// One 16-byte word, or four 4-byte copies where the stride or the base
// forbids the word; elements past `limit` (or a row past the edge) are
// zero-filled.
__device__ __forceinline__ void exact_km_copy(
    unsigned dst, const float* src_row, int first, int limit, bool row_valid, bool vector
) {
    if (vector) {
        int bytes = row_valid ? max(0, min(4, limit - first)) * 4 : 0;
        const float* src = bytes > 0 ? src_row + first : src_row;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(dst), "l"(src), "r"(bytes));
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            bool valid = row_valid && first + j < limit;
            const float* src = valid ? src_row + first + j : src_row;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(dst + (unsigned)(j * 4)), "l"(src), "r"(valid ? 4 : 0));
        }
    }
}

template <bool B_T>
__device__ __forceinline__ void exact_km_m32n64_tile(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const ExactThinParams& params
) {
    __shared__ __align__(16) ExactKmStage<B_T> stages[EXACT_KM_STAGES];

    const int k_begin = blockIdx.z * params.slab;
    const int k_len = min(params.slab, params.k - k_begin);
    A += k_begin;
    B += B_T ? (long long)k_begin : (long long)k_begin * params.ldb;
    C += (long long)blockIdx.z * params.m * params.ldc;

    const int row_tile = blockIdx.y * EXACT_KM_BM;
    const int column_tile = blockIdx.x * EXACT_KM_BN;
    const int t = threadIdx.x;
    // Rows rg + 8i; columns cg + 16j over a transposed B, so a quarter warp
    // reads eight consecutive staged columns, and cg * 4 + j over a forward's
    // B, so a thread reads four consecutive columns of a staged k row.
    const int rg = t >> 4;
    const int cg = t & 15;

    float acc[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int g_column = column_tile + (B_T ? cg + 16 * j : cg * 4 + j);
        const float seed = (bias != nullptr && g_column < params.n) ? bias[g_column] : 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[i][j] = seed;
    }

    const bool a_vector = ((params.lda & 3) == 0) && is_aligned_16(A);
    const bool b_vector = ((params.ldb & 3) == 0) && is_aligned_16(B);
    const int tiles = (k_len + EXACT_KM_BK - 1) / EXACT_KM_BK;

    auto issue = [&](int stage, int k0) {
        ExactKmStage<B_T>& s = stages[stage];
        // A [32 rows x 16 k]: 128 words, one per thread.
        {
            const int r = t >> 2;
            const int w = (t & 3) * 4;
            const int g_row = row_tile + r;
            const unsigned dst = (unsigned)__cvta_generic_to_shared(&s.a[r * EXACT_KM_ROW + w]);
            exact_km_copy(dst, A + (long long)min(g_row, params.m - 1) * params.lda, k0 + w, k_len,
                          g_row < params.m, a_vector);
        }
        if (B_T) {
            // B [64 columns x 16 k]: 256 words, two per thread.
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int e = t + q * 128;
                const int c = e >> 2;
                const int w = (e & 3) * 4;
                const int g_column = column_tile + c;
                const unsigned dst = (unsigned)__cvta_generic_to_shared(&s.b[c * EXACT_KM_ROW + w]);
                exact_km_copy(dst, B + (long long)min(g_column, params.n - 1) * params.ldb, k0 + w,
                              k_len, g_column < params.n, b_vector);
            }
        } else {
            // B [16 k x 64 columns]: 256 words, two per thread.
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int e = t + q * 128;
                const int kk = e >> 4;
                const int w = (e & 15) * 4;
                const int g_k = k0 + kk;
                const unsigned dst =
                    (unsigned)__cvta_generic_to_shared(&s.b[kk * EXACT_KM_B_NN_ROW + w]);
                exact_km_copy(dst, B + (long long)min(g_k, k_len - 1) * params.ldb, column_tile + w,
                              params.n, g_k < k_len, b_vector);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < EXACT_KM_STAGES - 1; ++s) {
        if (s < tiles) issue(s, s * EXACT_KM_BK);
        asm volatile("cp.async.commit_group;\n");
    }
    for (int tile = 0; tile < tiles; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" :: "n"(EXACT_KM_STAGES - 2));
        __syncthreads();
        const int next = tile + EXACT_KM_STAGES - 1;
        if (next < tiles) issue(next % EXACT_KM_STAGES, next * EXACT_KM_BK);
        asm volatile("cp.async.commit_group;\n");
        const ExactKmStage<B_T>& s = stages[tile % EXACT_KM_STAGES];
        #pragma unroll
        for (int g = 0; g < EXACT_KM_BK / 4; ++g) {
            float4 a[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = *reinterpret_cast<const float4*>(&s.a[(rg + 8 * i) * EXACT_KM_ROW + 4 * g]);
            }
            if (B_T) {
                float4 b[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    b[j] = *reinterpret_cast<const float4*>(&s.b[(cg + 16 * j) * EXACT_KM_ROW + 4 * g]);
                }
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float av = kk == 0 ? a[i].x : kk == 1 ? a[i].y : kk == 2 ? a[i].z : a[i].w;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const float bv = kk == 0 ? b[j].x : kk == 1 ? b[j].y : kk == 2 ? b[j].z : b[j].w;
                            acc[i][j] = __fmaf_rn(av, bv, acc[i][j]);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const float4 b = *reinterpret_cast<const float4*>(
                        &s.b[(4 * g + kk) * EXACT_KM_B_NN_ROW + cg * 4]);
                    const float bv[4] = {b.x, b.y, b.z, b.w};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float av = kk == 0 ? a[i].x : kk == 1 ? a[i].y : kk == 2 ? a[i].z : a[i].w;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(av, bv[j], acc[i][j]);
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n");

    const bool close = params.close > 0 && (k_len % params.close) != 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int g_row = row_tile + rg + 8 * i;
        if (g_row >= params.m) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int g_column = column_tile + (B_T ? cg + 16 * j : cg * 4 + j);
            if (g_column < params.n) {
                const float value = close ? __fadd_rn(acc[i][j], 0.0f) : acc[i][j];
                C[(long long)g_row * params.ldc + g_column] = value;
            }
        }
    }
}

// The 32 x 64 reduction-major chain tile of a forward.
extern "C" __global__ __launch_bounds__(128)
void nn_exact_km_m32n64_bk16_s4(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_km_m32n64_tile<false>(C, A, B, bias, params);
}

// The same tile of an input gradient, B the weight [n][k].
extern "C" __global__ __launch_bounds__(128)
void nt_exact_km_m32n64_bk16_s4(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    ExactThinParams params
) {
    exact_km_m32n64_tile<true>(C, A, B, bias, params);
}

#undef EXACT_KM_B_NN_ROW
#undef EXACT_KM_ROW
#undef EXACT_KM_STAGES
#undef EXACT_KM_BK
#undef EXACT_KM_BN
#undef EXACT_KM_BM

#undef EXACT_C16_B_ROW
#undef EXACT_C16_A_ROW
#undef EXACT_C16_STAGES
#undef EXACT_C16_BK
#undef EXACT_C16_BN
#undef EXACT_C16_BM

#undef EXACT_THIN_B_STAGE
#undef EXACT_THIN_A_STAGE
#undef EXACT_THIN_GROUP
#undef EXACT_THIN_STAGES
#undef EXACT_THIN_BK
#undef EXACT_THIN_BN
#undef EXACT_THIN_BM
