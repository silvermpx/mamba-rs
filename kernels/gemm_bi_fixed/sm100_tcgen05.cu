// ============================================================================
// Datacenter Blackwell (sm_100 family) inference rung: NN forward through
// tcgen05.mma with f32 accumulation in Tensor Memory.
// ============================================================================
// One 128-thread CTA owns a 128x128 output tile and walks K in ascending
// 64-wide slabs, each slab issued as four ascending tcgen05.mma K16 steps
// into the same TMEM accumulator. The K-slab order, the tail zero-fill
// and the single RNE downcast at the store mirror the mma.sync ladder's
// contract; whether the bits EQUAL that ladder on the same inputs is a
// hardware question (the fifth-generation tensor core sums a different
// internal block width), answered by the forced-entry census on a real
// CC 10.x part before any dispatch cell may route here.
//
// The accumulator does not live in registers: tcgen05 accumulates into
// Tensor Memory, a per-SM 512-column x 128-lane f32 array. The CTA
// allocates 128 columns once (warp-issued, power-of-two, explicitly
// deallocated), seeds them with bias through tcgen05.st when bias is
// present (the first MMA then runs with enable-input-d, preserving the
// bias-before-reduction placement of every other rung), and reads them
// back warp-by-warp with tcgen05.ld: warp w owns TMEM lanes 32w..32w+31,
// so each output element has exactly one store owner and the epilogue
// needs no reduction, no shuffle and no atomics.
//
// Operands stage through cp.async into shared memory laid out in the
// 128-byte swizzle the tcgen05 matrix descriptors declare: the 16-byte
// chunk at logical column c of row r lives at physical chunk c ^ (r & 7).
// A is K-major (a 128-byte row per output row), B is N-major in two
// complete 64-column halves 8192 bytes apart, which is the canonical
// tiling the descriptor's leading-offset field encodes. cp.async writes
// land on the generic proxy while tcgen05 reads through the async proxy,
// so a fence.proxy.async plus the tcgen05 thread-sync fences make the
// staged bytes visible before each slab is consumed. Hardware TMA is the
// natural transport upgrade for this rung and can replace the staging
// loops at qualification time without moving a single output bit: both
// transports deliver the same bytes to the same swizzled addresses.
//
// Stage reuse is gated by the tensor core itself: each slab's MMAs are
// followed by tcgen05.commit, which arrives on an mbarrier only when the
// tracked MMA work no longer reads the stage; every thread waits on that
// barrier's phase before the buffer is refilled.
//
// Everything below compiles only for the CC 10.x family targets
// (compute_100a/100f/103a/103f): tcgen05 exists on datacenter Blackwell
// alone (Hopper has wgmma, consumer Blackwell kept mma.sync), and the
// loader looks these symbols up only when the device resolves there.
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1030)

// Shared-memory matrix descriptor for tcgen05: 14-bit address fields in
// 16-byte units, version 1, the 128-byte-swizzle layout tag in bits
// 61-63. Distinct from the wgmma descriptor (version and layout encode
// differently); never share builders between the two families.
static __device__ __forceinline__ unsigned long long
sm100_desc(const void* smem_ptr, unsigned lbo16, unsigned sbo16) {
    unsigned long long addr =
        (unsigned long long)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (addr >> 4) & 0x3FFFULL;
    d |= ((unsigned long long)(lbo16 & 0x3FFF)) << 16;
    d |= ((unsigned long long)(sbo16 & 0x3FFF)) << 32;
    /* Version sits ABOVE the stride field (bits 46-47), not in the
     * low half - a version bit misplaced at bit 14 presents a
     * Hopper-format descriptor to a tensor core expecting version 1. */
    d |= 1ULL << 46; // descriptor version 1
    d |= 2ULL << 61; // 128-byte swizzle
    return d;
}

#define SM100_BM 128
#define SM100_BN 128
#define SM100_BK 64
#define SM100_STAGES 2
// Row strides in half-precision elements: pad-free, the swizzle owns
// bank-conflict freedom, and the descriptors assume contiguous rows.
#define SM100_LDA SM100_BK
#define SM100_LDB 64
#define SM100_A_BYTES (SM100_BM * SM100_LDA * 2)
#define SM100_STAGE_BYTES (SM100_A_BYTES + SM100_BK * SM100_BN * 2)

// Instruction descriptors are fixed constants encoding f32 D, the A/B
// type, both operands' major mode, M=128 and N=128 for the NN op.
#define SM100_IDESC_F16 0x08210010u
#define SM100_IDESC_BF16 0x08210490u

// Stage one A(128x64) + B(64x128) tile pair into the swizzled layout.
// Every 16-byte chunk lands at (row, chunk ^ (row & 7)); out-of-range
// rows and K/N tails zero-fill so the mma sees exact zeros, matching
// the mma.sync ladder's tail rule. The source pointer is formed only
// when bytes remain in the object. B's two 64-column halves live 8192
// bytes apart, the separation its descriptor declares.
#define SM100_STAGE_ASYNC(buf, bkIdx)                                         \
    do {                                                                      \
        unsigned _as = sm_sbase + (unsigned)((buf) * SM100_STAGE_BYTES);      \
        unsigned _bs = _as + (unsigned)SM100_A_BYTES;                         \
        for (int _i = threadIdx.x; _i < SM100_BM * (SM100_BK / 8);            \
             _i += blockDim.x) {                                              \
            int _m = _i / (SM100_BK / 8);                                     \
            int _c = _i % (SM100_BK / 8);                                     \
            int _gr = pid_m * SM100_BM + _m;                                  \
            int _gc = (bkIdx) + _c * 8;                                       \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as +                                             \
                (unsigned)((_m * SM100_LDA + ((_c ^ (_m & 7)) * 8)) * 2);     \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                            \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < 2 * SM100_BK * (64 / 8);              \
             _i += blockDim.x) {                                              \
            int _h = _i / (SM100_BK * 8);                                     \
            int _r = (_i / 8) % SM100_BK;                                     \
            int _c = _i % 8;                                                  \
            int _gk = (bkIdx) + _r;                                           \
            int _gn = pid_n * SM100_BN + _h * 64 + _c * 8;                    \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs + (unsigned)(_h * 8192) +                     \
                (unsigned)((_r * SM100_LDB + ((_c ^ (_r & 7)) * 8)) * 2);     \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                            \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_SM100(SUFFIX, T_ACT, FROM_F, IDESC)                 \
extern "C" __global__ __launch_bounds__(128, 1)                               \
void gemm_bi_nn_sm100_tcgen_c4_##SUFFIX(                                      \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    extern __shared__ __align__(1024) unsigned char sm100_dynsmem[];          \
    __shared__ __align__(8) unsigned long long empty_bar;                     \
    __shared__ unsigned tmem_base;                                            \
    unsigned sm_sbase = (unsigned)__cvta_generic_to_shared(sm100_dynsmem);    \
    int num_pid_n = (N + SM100_BN - 1) / SM100_BN;                            \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    int tid = threadIdx.x;                                                    \
    int warp = tid >> 5;                                                      \
    int lane = tid & 31;                                                      \
    /* TMEM allocation and the barrier are warp-zero duties: alloc is   */    \
    /* warp-issued, init is one thread plus the required init fence.    */    \
    if (warp == 0) {                                                          \
        unsigned _tb = (unsigned)__cvta_generic_to_shared(&tmem_base);        \
        asm volatile(                                                         \
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "        \
            "[%0], %1;" :: "r"(_tb), "r"(128u));                              \
        if (lane == 0) {                                                      \
            unsigned _mb = (unsigned)__cvta_generic_to_shared(&empty_bar);    \
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"             \
                         :: "r"(_mb));                                        \
            asm volatile("fence.mbarrier_init.release.cluster;");             \
        }                                                                     \
    }                                                                         \
    __syncthreads();                                                          \
    unsigned taddr = tmem_base;                                               \
    unsigned mbar = (unsigned)__cvta_generic_to_shared(&empty_bar);           \
    int has_bias = bias != nullptr ? 1 : 0;                                   \
    if (has_bias) {                                                           \
        /* Seed the accumulator with bias at its output column, exactly */    \
        /* like every other rung (alpha must be 1.0 with bias). Each    */    \
        /* warp stores its own 32 TMEM lanes, eight columns at a time;  */    \
        /* tail columns store zero.                                     */    \
        unsigned wrow = taddr + ((unsigned)(warp * 32) << 16);                \
        for (int c0 = 0; c0 < SM100_BN; c0 += 8) {                            \
            float bv[8];                                                      \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 8; j++) {                                     \
                int col = pid_n * SM100_BN + c0 + j;                          \
                bv[j] = col < N ? bias[col] : 0.0f;                           \
            }                                                                 \
            asm volatile(                                                     \
                "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], "                \
                "{%1,%2,%3,%4,%5,%6,%7,%8};"                                  \
                :: "r"(wrow + (unsigned)c0),                                  \
                   "f"(bv[0]), "f"(bv[1]), "f"(bv[2]), "f"(bv[3]),            \
                   "f"(bv[4]), "f"(bv[5]), "f"(bv[6]), "f"(bv[7]));           \
        }                                                                     \
        asm volatile("tcgen05.wait::st.sync.aligned;");                       \
        asm volatile("tcgen05.fence::before_thread_sync;");                   \
    }                                                                         \
    __syncthreads();                                                          \
    int num_k_tiles = (K + SM100_BK - 1) / SM100_BK;                          \
    SM100_STAGE_ASYNC(0, 0);                                                  \
    asm volatile("cp.async.commit_group;\n");                                 \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        asm volatile("cp.async.wait_group 0;\n");                             \
        __syncthreads();                                                      \
        /* Staged bytes arrived on the generic proxy; make them visible */    \
        /* to the async proxy the tensor core reads through.            */    \
        asm volatile("fence.proxy.async.shared::cta;\n");                     \
        int rd = kt & (SM100_STAGES - 1);                                     \
        /* The next stage's copies are issued BEFORE this slab's MMAs   */    \
        /* so they overlap them. WAR-safe: buffer (kt+1)&1 was consumed */    \
        /* by the MMAs of iteration kt-1, whose commit barrier every    */    \
        /* thread has already waited on.                                */    \
        if (kt + 1 < num_k_tiles) {                                           \
            SM100_STAGE_ASYNC((kt + 1) & (SM100_STAGES - 1),                  \
                              (kt + 1) * SM100_BK);                           \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
        if (tid == 0) {                                                       \
            const unsigned char* stage =                                      \
                sm100_dynsmem + rd * SM100_STAGE_BYTES;                       \
            unsigned long long a_base = sm100_desc(stage, 1, 64);             \
            unsigned long long b_base =                                       \
                sm100_desc(stage + SM100_A_BYTES, 512, 64);                   \
            asm volatile("tcgen05.fence::after_thread_sync;");                \
            _Pragma("unroll")                                                 \
            for (int ks = 0; ks < SM100_BK / 16; ks++) {                      \
                unsigned long long da =                                       \
                    a_base + (unsigned long long)(ks * 2);                    \
                unsigned long long db =                                       \
                    b_base + (unsigned long long)(ks * 128);                  \
                unsigned acc_d =                                              \
                    (kt > 0 || ks > 0 || has_bias) ? 1u : 0u;                 \
                asm volatile(                                                 \
                    "{.reg .pred p;\n\t"                                      \
                    "setp.ne.b32 p, %4, 0;\n\t"                               \
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "   \
                    "{%5, %6, %7, %8}, p;}\n\t"                               \
                    :: "r"(taddr), "l"(da), "l"(db), "r"(IDESC),              \
                       "r"(acc_d), "r"(0u), "r"(0u), "r"(0u), "r"(0u));       \
            }                                                                 \
            asm volatile(                                                     \
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one"           \
                ".shared::cluster.b64 [%0];" :: "r"(mbar));                   \
        }                                                                     \
        /* The commit arrives only when the tensor core no longer reads */    \
        /* the stage; every thread holds here before refilling it.      */    \
        asm volatile(                                                         \
            "{\n\t.reg .pred p;\n"                                            \
            "WAIT_%=:\n\t"                                                    \
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"       \
            "@!p bra WAIT_%=;\n\t}"                                           \
            :: "r"(mbar), "r"((unsigned)(kt & 1)));                           \
        __syncthreads();                                                      \
    }                                                                         \
    /* Epilogue: warp w reads TMEM lanes 32w..32w+31 (its own quarter), */    \
    /* eight columns per load; alpha through an explicit unfused        */    \
    /* multiply (target-independent bits), paired store on aligned even */    \
    /* destinations, scalar RNE tail otherwise - the mma.sync rules.    */    \
    asm volatile("tcgen05.fence::after_thread_sync;");                        \
    int row = pid_m * SM100_BM + warp * 32 + lane;                            \
    unsigned wrow = taddr + ((unsigned)(warp * 32) << 16);                    \
    for (int c0 = 0; c0 < SM100_BN; c0 += 8) {                                \
        float r[8];                                                           \
        asm volatile(                                                         \
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "                          \
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"                                \
            : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]),                 \
              "=f"(r[4]), "=f"(r[5]), "=f"(r[6]), "=f"(r[7])                  \
            : "r"(wrow + (unsigned)c0));                                      \
        asm volatile("tcgen05.wait::ld.sync.aligned;");                       \
        /* The ld above is warp-collective; only the stores may diverge. */   \
        if (row < M) {                                                        \
            _Pragma("unroll")                                                 \
            for (int j = 0; j < 8; j += 2) {                                  \
                int col = pid_n * SM100_BN + c0 + j;                          \
                if (col >= N) continue;                                       \
                float v0 = __fmul_rn(alpha, r[j]);                            \
                float v1 = __fmul_rn(alpha, r[j + 1]);                        \
                T_ACT* dst = &C[(long long)row * ldc + col];                  \
                bool packed = beta == 0.0f && (ldc & 1) == 0 &&               \
                              col + 1 < N &&                                  \
                              ((reinterpret_cast<unsigned long long>(dst)     \
                                & 3u) == 0u);                                 \
                if (packed) {                                                 \
                    gbf_store_pair_rne(dst, v0, v1);                          \
                } else {                                                      \
                    for (int e = 0; e < 2; e++) {                             \
                        int gc = col + e;                                     \
                        if (gc >= N) continue;                                \
                        float val = e ? v1 : v0;                              \
                        if (beta != 0.0f)                                     \
                            val = __fmaf_rn(                                  \
                                beta,                                         \
                                to_f(C[(long long)row * ldc + gc]), val);     \
                        C[(long long)row * ldc + gc] = FROM_F(val);           \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    /* All TMEM traffic is finished; release the allocation the same    */    \
    /* warp made. Relinquish before dealloc is the mandated order.      */    \
    __syncthreads();                                                          \
    if (warp == 0) {                                                          \
        asm volatile(                                                         \
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");    \
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"  \
                     :: "r"(taddr), "r"(128u));                               \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_SM100(bf16, __nv_bfloat16, from_f_bf16, SM100_IDESC_BF16)
DEFINE_GEMM_BI_NN_SM100(f16,  __half,        from_f_f16,  SM100_IDESC_F16)

#undef SM100_BM
#undef SM100_BN
#undef SM100_BK
#undef SM100_STAGES
#undef SM100_LDA
#undef SM100_LDB
#undef SM100_A_BYTES
#undef SM100_STAGE_BYTES
#undef SM100_IDESC_F16
#undef SM100_IDESC_BF16
#undef SM100_STAGE_ASYNC
#undef DEFINE_GEMM_BI_NN_SM100

#endif
