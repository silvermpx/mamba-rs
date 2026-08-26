// ============================================================================
// Hopper (sm_90a) inference rung: NN forward through wgmma.mma_async.
// ============================================================================
// One warpgroup (128 threads) owns a 64x128 output tile and walks K in
// ascending 64-wide slabs, each slab issued as four ascending
// wgmma.mma_async.m64n128k16 steps into the same f32 accumulators. The
// K-slab order, the tail zero-fill and the single RNE downcast at the
// store mirror the sm_89 ladder's contract. Bias does NOT: it joins in
// the epilogue (after alpha, before beta) rather than pre-seeding the
// accumulators, because the first wgmma group then runs with
// scale-d = 0 and ptxas stops serializing it against a register init
// chain. This is the rung's own numeric contract either way: whether
// its bits EQUAL the mma.sync ladder on the same inputs is a hardware
// question (Hopper's tensor core sums sixteen products per internal
// block where Ada sums eight), answered by the forced-entry census on
// a real sm_90a part before any dispatch cell may route here.
//
// Operands stage through cp.async into shared memory laid out in the
// 128-byte swizzle the wgmma descriptors declare: the 16-byte chunk at
// logical column c of row r lives at physical chunk c ^ (r & 7). The
// swizzle only relocates bytes, never values. cp.async writes land on
// the generic proxy while wgmma reads through the async proxy, so a
// fence.proxy.async makes the staged bytes visible before each slab is
// consumed.
//
// Everything below compiles only for sm_90a: wgmma exists on Hopper
// alone (consumer Blackwell dropped it for mma.sync, datacenter
// Blackwell replaced it with tcgen05), and the loader looks these
// symbols up only when the device resolves to sm_90a.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900

// Shared-memory matrix descriptor: 14-bit fields in 16-byte units, the
// 128-byte-swizzle layout tag in bits 62-63.
static __device__ __forceinline__ unsigned long long
sm90_desc(const void* smem_ptr, unsigned lbo16, unsigned sbo16) {
    unsigned long long addr =
        (unsigned long long)__cvta_generic_to_shared(smem_ptr);
    unsigned long long d = 0;
    d |= (addr >> 4) & 0x3FFFULL;
    d |= ((unsigned long long)(lbo16 & 0x3FFF)) << 16;
    d |= ((unsigned long long)(sbo16 & 0x3FFF)) << 32;
    d |= 1ULL << 62; // 128-byte swizzle
    return d;
}

#define SM90_BM 64
#define SM90_BN 128
#define SM90_BK 64
#define SM90_STAGES 2
// Row strides in elements (half-precision): pad-free, the swizzle owns
// bank-conflict freedom, and the descriptors assume contiguous rows.
#define SM90_LDA SM90_BK
#define SM90_LDB SM90_BN

// Stage one A(64x64) + B(64x128) tile pair into the swizzled layout.
// Every 16-byte chunk lands at (row, chunk ^ (row & 7)); out-of-range
// rows and K/N tails zero-fill so the mma sees exact zeros, matching
// the sm_89 ladder's tail rule. The source pointer is formed only when
// bytes remain in the object.
#define SM90_STAGE_ASYNC(buf, bkIdx)                                          \
    do {                                                                      \
        unsigned _as = As_sbase +                                             \
                       (unsigned)((buf) * SM90_BM * SM90_LDA * 2);            \
        unsigned _bs = Bs_sbase +                                             \
                       (unsigned)((buf) * SM90_BK * SM90_LDB * 2);            \
        for (int _i = threadIdx.x; _i < SM90_BM * (SM90_BK / 8);              \
             _i += blockDim.x) {                                              \
            int _m = _i / (SM90_BK / 8);                                      \
            int _c = _i % (SM90_BK / 8);                                      \
            int _k = _c * 8;                                                  \
            int _gr = pid_m * SM90_BM + _m;                                   \
            int _gc = (bkIdx) + _k;                                           \
            int _valid = (_gr < M) ? (K - _gc) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _as +                                             \
                (unsigned)((_m * SM90_LDA + ((_c ^ (_m & 7)) * 8)) * 2);      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&A[(long long)_gr * lda + _gc]                 \
                : (const void*)A;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
        for (int _i = threadIdx.x; _i < SM90_BK * (SM90_BN / 8);              \
             _i += blockDim.x) {                                              \
            int _k = _i / (SM90_BN / 8);                                      \
            int _c = _i % (SM90_BN / 8);                                      \
            int _n = _c * 8;                                                  \
            int _gk = (bkIdx) + _k;                                           \
            int _gn = pid_n * SM90_BN + _n;                                   \
            int _valid = (_gk < K) ? (N - _gn) : 0;                           \
            int _bytes = _valid >= 8 ? 16 : (_valid > 0 ? _valid * 2 : 0);    \
            unsigned _dst = _bs +                                             \
                (unsigned)((_k * SM90_LDB + ((_c ^ (_k & 7)) * 8)) * 2);      \
            const void* _src = (_bytes > 0)                                   \
                ? (const void*)&B[(long long)_gk * ldb + _gn]                 \
                : (const void*)B;                                             \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"    \
                         :: "r"(_dst), "l"(_src), "r"(_bytes));               \
        }                                                                     \
    } while (0)

#define DEFINE_GEMM_BI_NN_SM90(SUFFIX, T_ACT, FROM_F, WG_T)                   \
extern "C" __global__ __launch_bounds__(128, 1)                               \
void gemm_bi_nn_sm90a_wgmma_wg1_##SUFFIX(                                     \
    T_ACT* __restrict__ C,                                                    \
    const T_ACT* __restrict__ A,                                              \
    const T_ACT* __restrict__ B,                                              \
    const float* __restrict__ bias,                                           \
    float alpha, float beta,                                                  \
    int M, int N, int K,                                                      \
    int lda, int ldb, int ldc                                                 \
) {                                                                           \
    extern __shared__ __align__(1024) unsigned char sm90_dynsmem[];           \
    T_ACT* As = reinterpret_cast<T_ACT*>(sm90_dynsmem);                       \
    T_ACT* Bs = reinterpret_cast<T_ACT*>(                                     \
        sm90_dynsmem + SM90_STAGES * SM90_BM * SM90_LDA * sizeof(T_ACT));     \
    int num_pid_n = (N + SM90_BN - 1) / SM90_BN;                              \
    int pid_m = blockIdx.x / num_pid_n;                                       \
    int pid_n = blockIdx.x % num_pid_n;                                       \
    unsigned As_sbase = (unsigned)__cvta_generic_to_shared(As);               \
    unsigned Bs_sbase = (unsigned)__cvta_generic_to_shared(Bs);               \
    int wg_tid = threadIdx.x;                                                 \
    /* Accumulators: m64n128 spreads 64 f32 per thread over the group. */     \
    float acc[64];                                                            \
    int q = wg_tid & 3;                                                       \
    int row8 = (wg_tid >> 2) & 7;                                             \
    int warp = wg_tid >> 5;                                                   \
    /* No accumulator pre-seed: the first wgmma group runs with          */   \
    /* scale-d = 0, which zeroes D regardless of register contents and   */   \
    /* frees ptxas from serializing the first group against an init      */   \
    /* chain. Bias joins in the epilogue - after alpha, before beta -    */   \
    /* which is this rung's own contract (the census decides its family  */   \
    /* membership either way; Hopper's wider internal reduce already     */   \
    /* makes bit-equality with the mma.sync ladder a hardware question). */   \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < 64; r++) {                                            \
        acc[r] = 0.0f;                                                        \
    }                                                                         \
    int num_k_tiles = (K + SM90_BK - 1) / SM90_BK;                            \
    SM90_STAGE_ASYNC(0, 0);                                                   \
    asm volatile("cp.async.commit_group;\n");                                 \
    for (int kt = 0; kt < num_k_tiles; kt++) {                                \
        asm volatile("cp.async.wait_group 0;\n");                             \
        __syncthreads();                                                      \
        /* Staged bytes arrived on the generic proxy; make them visible */    \
        /* to the async proxy the wgmma reads through.                  */    \
        asm volatile("fence.proxy.async.shared::cta;\n");                     \
        int rd = kt & (SM90_STAGES - 1);                                      \
        unsigned long long a_base = sm90_desc(                                \
            As + rd * SM90_BM * SM90_LDA, 1, 64);                             \
        unsigned long long b_base = sm90_desc(                                \
            Bs + rd * SM90_BK * SM90_LDB, 512, 64);                           \
        asm volatile("wgmma.fence.sync.aligned;\n");                          \
        _Pragma("unroll")                                                     \
        for (int ks = 0; ks < SM90_BK / 16; ks++) {                           \
            unsigned long long da = a_base + (unsigned long long)(ks * 2);    \
            unsigned long long db = b_base + (unsigned long long)(ks * 128);  \
            unsigned scale_d = (kt == 0 && ks == 0) ? 0u : 1u;                \
            asm volatile(                                                     \
                "{.reg .pred p;\n\t"                                          \
                "setp.ne.b32 p, %66, 0;\n\t"                                  \
                "wgmma.mma_async.sync.aligned.m64n128k16.f32." WG_T "." WG_T  \
                " {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"    \
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,"    \
                "%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,"    \
                "%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,"    \
                "%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 1;}\n"       \
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),     \
                  "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]),     \
                  "+f"(acc[8]), "+f"(acc[9]), "+f"(acc[10]), "+f"(acc[11]),   \
                  "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15]), \
                  "+f"(acc[16]), "+f"(acc[17]), "+f"(acc[18]), "+f"(acc[19]), \
                  "+f"(acc[20]), "+f"(acc[21]), "+f"(acc[22]), "+f"(acc[23]), \
                  "+f"(acc[24]), "+f"(acc[25]), "+f"(acc[26]), "+f"(acc[27]), \
                  "+f"(acc[28]), "+f"(acc[29]), "+f"(acc[30]), "+f"(acc[31]), \
                  "+f"(acc[32]), "+f"(acc[33]), "+f"(acc[34]), "+f"(acc[35]), \
                  "+f"(acc[36]), "+f"(acc[37]), "+f"(acc[38]), "+f"(acc[39]), \
                  "+f"(acc[40]), "+f"(acc[41]), "+f"(acc[42]), "+f"(acc[43]), \
                  "+f"(acc[44]), "+f"(acc[45]), "+f"(acc[46]), "+f"(acc[47]), \
                  "+f"(acc[48]), "+f"(acc[49]), "+f"(acc[50]), "+f"(acc[51]), \
                  "+f"(acc[52]), "+f"(acc[53]), "+f"(acc[54]), "+f"(acc[55]), \
                  "+f"(acc[56]), "+f"(acc[57]), "+f"(acc[58]), "+f"(acc[59]), \
                  "+f"(acc[60]), "+f"(acc[61]), "+f"(acc[62]), "+f"(acc[63])  \
                : "l"(da), "l"(db), "r"(scale_d));                            \
        }                                                                     \
        asm volatile("wgmma.commit_group.sync.aligned;\n");                   \
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");                   \
        __syncthreads();                                                      \
        if (kt + 1 < num_k_tiles) {                                           \
            SM90_STAGE_ASYNC((kt + 1) & (SM90_STAGES - 1),                    \
                             (kt + 1) * SM90_BK);                             \
        }                                                                     \
        asm volatile("cp.async.commit_group;\n");                            \
    }                                                                         \
    /* Epilogue: alpha through an explicit unfused multiply (target-  */      \
    /* independent bits), paired store on aligned even destinations,  */      \
    /* scalar RNE tail otherwise - the same rules as the sm_89 tiles. */      \
    _Pragma("unroll")                                                         \
    for (int r = 0; r < 64; r += 2) {                                         \
        int pair_row = (r >> 1) & 1;                                          \
        int n_group = r >> 2;                                                 \
        int row = pid_m * SM90_BM + row8 + 16 * warp + 8 * pair_row;          \
        int col = pid_n * SM90_BN + 2 * q + 8 * n_group;                      \
        if (row >= M) continue;                                               \
        float v0 = __fmul_rn(alpha, acc[r]);                                  \
        float v1 = __fmul_rn(alpha, acc[r + 1]);                              \
        if (bias != nullptr) {                                                \
            if (col < N) v0 = __fadd_rn(v0, bias[col]);                       \
            if (col + 1 < N) v1 = __fadd_rn(v1, bias[col + 1]);               \
        }                                                                     \
        T_ACT* dst = (col < N) ? &C[(long long)row * ldc + col] : (T_ACT*)0;  \
        bool packed = beta == 0.0f && (ldc & 1) == 0 && col + 1 < N &&        \
                      ((reinterpret_cast<unsigned long long>(dst) & 3u)       \
                       == 0u);                                                \
        if (packed) {                                                         \
            gbf_store_pair_rne(dst, v0, v1);                                  \
        } else {                                                              \
            for (int e = 0; e < 2; e++) {                                     \
                int gc = col + e;                                             \
                if (gc >= N) continue;                                        \
                float val = e ? v1 : v0;                                      \
                if (beta != 0.0f)                                             \
                    val = __fmaf_rn(beta, to_f(C[(long long)row * ldc + gc]), val);         \
                C[(long long)row * ldc + gc] = FROM_F(val);                   \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

DEFINE_GEMM_BI_NN_SM90(bf16, __nv_bfloat16, from_f_bf16, "bf16")
DEFINE_GEMM_BI_NN_SM90(f16,  __half,        from_f_f16,  "f16")

#undef SM90_BM
#undef SM90_BN
#undef SM90_BK
#undef SM90_STAGES
#undef SM90_LDA
#undef SM90_LDB
#undef SM90_STAGE_ASYNC
#undef DEFINE_GEMM_BI_NN_SM90

#endif
