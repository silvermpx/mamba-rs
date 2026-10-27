// The 64 x 64 tile over a cut layer's forward, staged in 32-deep slabs by
// 16-byte copies: 128 threads, each eight rows by four columns. Each piece of
// `piece_tiles` 16-deep tiles is a chain from +0.0 over whole 32-deep groups,
// added in order to the sum a thread keeps in registers, the bias after the
// pieces - the bits of the 128 x 128, 128 x 64 and 64 x 64 pieces tiles and
// of the chunk partials. Its 32 accumulators and their kept sum leave room
// for three CTAs a multiprocessor where the 64-accumulator tiles hold two,
// and its copies move a slab in four 16-byte requests a thread where those
// tiles transpose A one float at a time.
#define EXACT_NN_M64N64_BK32_BM 64
#define EXACT_NN_M64N64_BK32_BN 64
#define EXACT_NN_M64N64_BK32_BK 32
#define EXACT_NN_M64N64_BK32_THREADS 128
#define EXACT_NN_M64N64_BK32_GROUP_M 8

// One slab of A [64 x 32] and B [32 x 64] into `buffer` by 16-byte copies of
// any reduction and alignment: a short copy zero-fills the rest of its 16
// bytes, and a row or column past the edge copies nothing and fills zeros.
#define EXACT_NN_M64N64_BK32_STAGE_ASYNC(BUFFER, K_TILE)                                     \
    do {                                                                                     \
        unsigned _as = (unsigned)__cvta_generic_to_shared(&smem_a[(BUFFER)][0]);             \
        unsigned _bs = (unsigned)__cvta_generic_to_shared(&smem_b[(BUFFER)][0]);             \
        for (int _i = threadIdx.x; _i < EXACT_NN_M64N64_BK32_BM * (EXACT_NN_M64N64_BK32_BK / 4); \
             _i += EXACT_NN_M64N64_BK32_THREADS) {                                           \
            int _r = _i / (EXACT_NN_M64N64_BK32_BK / 4);                                     \
            int _c = (_i % (EXACT_NN_M64N64_BK32_BK / 4)) * 4;                               \
            int _gr = row0 + _r;                                                             \
            int _gc = (K_TILE) + _c;                                                         \
            int _valid = (_gr < M) ? (K - _gc) : 0;                                          \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);                  \
            unsigned _dst = _as + (unsigned)((_r * EXACT_NN_M64N64_BK32_BK + _c) * 4);       \
            const void* _src = _bytes > 0 ? (const void*)&A[(long long)_gr * lda + _gc]      \
                                          : (const void*)A;                                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"                  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));                              \
        }                                                                                    \
        for (int _i = threadIdx.x; _i < EXACT_NN_M64N64_BK32_BK * (EXACT_NN_M64N64_BK32_BN / 4); \
             _i += EXACT_NN_M64N64_BK32_THREADS) {                                           \
            int _r = _i / (EXACT_NN_M64N64_BK32_BN / 4);                                     \
            int _c = (_i % (EXACT_NN_M64N64_BK32_BN / 4)) * 4;                               \
            int _gr = (K_TILE) + _r;                                                         \
            int _gc = col0 + _c;                                                             \
            int _valid = (_gr < K) ? (N - _gc) : 0;                                          \
            int _bytes = _valid >= 4 ? 16 : (_valid > 0 ? _valid * 4 : 0);                  \
            unsigned _dst = _bs + (unsigned)((_r * EXACT_NN_M64N64_BK32_BN + _c) * 4);       \
            const void* _src = _bytes > 0 ? (const void*)&B[(long long)_gr * ldb + _gc]      \
                                          : (const void*)B;                                  \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"                  \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));                              \
        }                                                                                    \
        asm volatile("cp.async.commit_group;\n");                                           \
    } while (0)

// The same slab by plain loads, for operands the 16-byte copies cannot read.
#define EXACT_NN_M64N64_BK32_STAGE_SCALAR(BUFFER, K_TILE)                                    \
    do {                                                                                     \
        for (int _i = threadIdx.x; _i < EXACT_NN_M64N64_BK32_BM * EXACT_NN_M64N64_BK32_BK;  \
             _i += EXACT_NN_M64N64_BK32_THREADS) {                                           \
            int _r = _i / EXACT_NN_M64N64_BK32_BK;                                           \
            int _c = _i % EXACT_NN_M64N64_BK32_BK;                                           \
            int _gr = row0 + _r;                                                             \
            int _gc = (K_TILE) + _c;                                                         \
            smem_a[(BUFFER)][_i] = (_gr < M && _gc < K) ? A[(long long)_gr * lda + _gc] : 0.0f; \
        }                                                                                    \
        for (int _i = threadIdx.x; _i < EXACT_NN_M64N64_BK32_BK * EXACT_NN_M64N64_BK32_BN;  \
             _i += EXACT_NN_M64N64_BK32_THREADS) {                                           \
            int _r = _i / EXACT_NN_M64N64_BK32_BN;                                           \
            int _c = _i % EXACT_NN_M64N64_BK32_BN;                                           \
            int _gr = (K_TILE) + _r;                                                         \
            int _gc = col0 + _c;                                                             \
            smem_b[(BUFFER)][_i] = (_gr < K && _gc < N) ? B[(long long)_gr * ldb + _gc] : 0.0f; \
        }                                                                                    \
    } while (0)

// The bias follows the pieces unscaled, so a bias comes with alpha 1.
extern "C" __global__ __launch_bounds__(EXACT_NN_M64N64_BK32_THREADS, 3)
void nn_m64n64_bk32_pieces_keep(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float alpha, float beta,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    int piece_tiles
) {
    assert(alpha == 1.0f || bias == nullptr);
    __align__(16) __shared__ float smem_a[2][EXACT_NN_M64N64_BK32_BM * EXACT_NN_M64N64_BK32_BK];
    __align__(16) __shared__ float smem_b[2][EXACT_NN_M64N64_BK32_BK * EXACT_NN_M64N64_BK32_BN];

    int num_pid_m = (M + EXACT_NN_M64N64_BK32_BM - 1) / EXACT_NN_M64N64_BK32_BM;
    int num_pid_n = (N + EXACT_NN_M64N64_BK32_BN - 1) / EXACT_NN_M64N64_BK32_BN;
    int num_pid_in_group = EXACT_NN_M64N64_BK32_GROUP_M * num_pid_n;
    int group_id = blockIdx.x / num_pid_in_group;
    int first_pid_m = group_id * EXACT_NN_M64N64_BK32_GROUP_M;
    int group_size_m = min(num_pid_m - first_pid_m, EXACT_NN_M64N64_BK32_GROUP_M);
    int pid_m = first_pid_m + ((blockIdx.x % num_pid_in_group) % group_size_m);
    int pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;
    int row0 = pid_m * EXACT_NN_M64N64_BK32_BM;
    int col0 = pid_n * EXACT_NN_M64N64_BK32_BN;
    int tx = threadIdx.x & 15;
    int ty = threadIdx.x >> 4;

    // A cut layer starts every piece from +0.0; the finished pieces' sum
    // starts at -0.0, the one value whose sum with any chain is that chain,
    // so a piece end adds without asking whether it closes the first piece.
    float acc[8][4];
    float kept[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
            kept[i][j] = -0.0f;
        }
    }

    // Whole 32-deep groups, as the chunk partials walk them; a piece is an
    // even number of 16-deep tiles, so its end falls on a slab's end.
    int num_k_tiles = (K + EXACT_NN_M64N64_BK32_BK - 1) / EXACT_NN_M64N64_BK32_BK;
    int piece_slabs = piece_tiles / 2;
    int fold_slab = piece_slabs;
    bool async_stage = is_aligned_16(A) && is_aligned_16(B) && (lda & 3) == 0 && (ldb & 3) == 0;
    if (num_k_tiles > 0) {
        if (async_stage) {
            EXACT_NN_M64N64_BK32_STAGE_ASYNC(0, 0);
        } else {
            EXACT_NN_M64N64_BK32_STAGE_SCALAR(0, 0);
        }
    }
    int read_buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (async_stage) {
            asm volatile("cp.async.wait_group 0;\n");
        }
        // Both the copies' visibility across warps and the previous slab's
        // consumption finish before a buffer is read or refilled.
        __syncthreads();
        if (kt + 1 < num_k_tiles) {
            int next_k = (kt + 1) * EXACT_NN_M64N64_BK32_BK;
            if (async_stage) {
                EXACT_NN_M64N64_BK32_STAGE_ASYNC(read_buf ^ 1, next_k);
            } else {
                EXACT_NN_M64N64_BK32_STAGE_SCALAR(read_buf ^ 1, next_k);
            }
        }
#pragma unroll
        for (int kk = 0; kk < EXACT_NN_M64N64_BK32_BK; kk++) {
            float a_reg[8];
            float b_reg[4];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                a_reg[i] = smem_a[read_buf][(ty * 8 + i) * EXACT_NN_M64N64_BK32_BK + kk];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                b_reg[j] = smem_b[read_buf][kk * EXACT_NN_M64N64_BK32_BN + tx * 4 + j];
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[i][j] = __fmaf_rn(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        if (kt + 1 == fold_slab && fold_slab < num_k_tiles) {
            // The piece's chain is added to the kept sum in order, and the
            // chain restarts from +0.0.
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    kept[i][j] = __fadd_rn(kept[i][j], acc[i][j]);
                    acc[i][j] = 0.0f;
                }
            }
            fold_slab += piece_slabs;
        }
        read_buf ^= 1;
    }

    // The last piece meets the earlier pieces' sum, then the bias: the order
    // the chunk fold's reduction adds the planes and the bias in.
    int row_base = row0 + ty * 8;
    int col_base = col0 + tx * 4;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = row_base + i;
        if (r >= M) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = col_base + j;
            if (col >= N) {
                continue;
            }
            float value = alpha * __fadd_rn(kept[i][j], acc[i][j]);
            if (bias != nullptr) {
                value = __fadd_rn(value, bias[col]);
            }
            if (beta != 0.0f) {
                value += beta * C[(long long)r * ldc + col];
            }
            C[(long long)r * ldc + col] = value;
        }
    }
}

#undef EXACT_NN_M64N64_BK32_STAGE_SCALAR
#undef EXACT_NN_M64N64_BK32_STAGE_ASYNC
#undef EXACT_NN_M64N64_BK32_GROUP_M
#undef EXACT_NN_M64N64_BK32_THREADS
#undef EXACT_NN_M64N64_BK32_BK
#undef EXACT_NN_M64N64_BK32_BN
#undef EXACT_NN_M64N64_BK32_BM
