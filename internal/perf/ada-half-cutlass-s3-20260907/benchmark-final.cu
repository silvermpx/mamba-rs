// Standalone force-only Ada half CUTLASS-style S3 schedule experiment.
// From the repository root (CUDA 13.2 toolkit + cuBLAS on Ada):
// nvcc -I. -Ikernels -O3 -std=c++17 -arch=sm_89 -lineinfo -Xptxas=-v \
//   internal/perf/ada-half-cutlass-s3-20260907/benchmark.cu -lcublas \
//   -o /tmp/ada-half-cutlass-s3-bench
// CLI: binary bf16|f16 M K N bias(0|1)
//   [--warmup 128] [--graph-ops 20] [--windows 21]
// Use --windows 0 for the full correctness corpus without timed samples.
// No fast-math, register cap, source substitution, external kernels, or NVRTC.
// Includes the actual current production Swizzle body and an evidence-only
// source twin whose aligned mainloop schedule is the sole behavioral change.
// The include-root -Ikernels is required by common.cuh's typed-prelude include.
//
// Timed contract: alpha=1, beta=0, no bias, and one RNE half downcast. The
// cuBLAS operation is native half GEMM with
// COMPUTE_32F / GEMM_DEFAULT_TENSOR_OP / DEFAULT_MATH. This requests native
// half Tensor Core eligibility; actual vendor instruction selection still
// requires profiling. It does NOT promise bit identity or batch invariance.
// The independent reference expands stored half inputs exactly into F32,
// seeds unrounded F32 bias, and uses F32-input COMPUTE_32F_PEDANTIC.
//
// Every custom arm must match production bytes before/repeatedly after eager
// and graph launches, across batch prefixes and true A/C row subviews. Gates
// include K0 with null A/B, M/N/K tails, odd/padded strides, independent 2-byte
// A/B/C misalignment, nontrivial alpha/beta, and exceptional half/bias values.
// Numeric tolerance is ONLY used against the independent reference, never to
// admit custom/custom differences. Exceptional corpus uses bit/guard gates,
// not a meaningless finite error norm. Guards cover allocations, row padding,
// and rows outside the current view; all input bytes must remain unchanged.
// This is a bounded corpus, not proof for all inputs; run compute-sanitizer
// separately to detect reads that guards cannot observe.
//
// All timed samples use identical CUDA events enclosing ONE 20-op graph.
// Three comparisons use true A B B A windows; comparison order reverses on
// alternating windows. Timing records are buffered until post-timing gates
// pass. A nonzero exit or missing complete record invalidates the whole run.
// Runtime resources are reported, not claimed to equal NVRTC production code
// generation; retain nvcc/PTXAS output and compare production resources.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernels/gemm_bi_fixed/common.cuh"
#include "kernels/gemm_bi_fixed/sm89_half_swizzle.cu"
#include "internal/perf/ada-half-cutlass-s3-20260907/candidate.cu"

namespace half_swizzle_bench {
constexpr int CustomArms = 2, Arms = 3;
// Each arm uses its own REQUIRED launch and opt-in shared-memory extent.
constexpr int SharedBytes[CustomArms] = {69632, 98304};
constexpr int Threads[CustomArms] = {256, 256};
constexpr int TileM[CustomArms] = {128, 128};
constexpr int TileN[CustomArms] = {128, 128};
constexpr size_t GuardElements = 64;
constexpr const char* Names[Arms] = {
    "production_swizzle", "candidate_cutlass_s3", "cublas_native_half_tc"};
constexpr int Comparisons[][2] = {
    {0, 1}, {2, 0}, {2, 1}};
constexpr int ComparisonCount = sizeof(Comparisons) / sizeof(Comparisons[0]);

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
int integer(const char* text, int low, int high) {
    if (!text || !*text || *text == '+' || *text == '-')
        throw std::runtime_error("expected unsigned decimal integer");
    for (const char* p = text; *p; ++p)
        if (*p < '0' || *p > '9') throw std::runtime_error("invalid integer suffix");
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (errno || *end || value < low || value > high)
        throw std::runtime_error("integer outside permitted range");
    return int(value);
}
size_t extent(int rows, int stride) {
    if (rows < 0 || stride <= 0 || size_t(rows) >
        (std::numeric_limits<size_t>::max() / 8 - 2 * GuardElements - 8) / size_t(stride))
        throw std::runtime_error("matrix extent overflow");
    return size_t(rows) * size_t(stride);
}
void dimensions(int m, int k, int n) {
    if (m <= 0 || n <= 0 || k < 0 || m > INT_MAX - 256 ||
        n > INT_MAX - 256 || k > INT_MAX - 256 ||
        ((int64_t(m) + 127) / 128) * ((int64_t(n) + 127) / 128) > INT_MAX ||
        int64_t(m) * n > int64_t(INT_MAX) * 256)
        throw std::runtime_error("invalid or overflowing GEMM dimensions");
    extent(m, std::max(k, 1)); extent(k, n); extent(m, n);
}
template<class T> T raw_half(uint16_t bits) {
    static_assert(sizeof(T) == 2, "half storage");
    T value;
    std::memcpy(&value, &bits, 2);
    return value;
}
float raw_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
template<class T> struct Type;
template<> struct Type<__half> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16F;
    static constexpr const char* name = "f16";
    static constexpr double tolerance = 0.0025;
    static __host__ __device__ __half from(float x) { return __float2half_rn(x); }
    static __host__ __device__ float to(__half x) { return __half2float(x); }
    static uint16_t special(int i) {
        constexpr uint16_t values[] = {0x0000, 0x8000, 0x0001, 0x8001,
            0x03ff, 0x83ff, 0x7c00, 0xfc00, 0x7e11, 0xfe21, 0x3c00, 0xbc00};
        return values[i % 12];
    }
};
template<> struct Type<__nv_bfloat16> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16BF;
    static constexpr const char* name = "bf16";
    static constexpr double tolerance = 0.01;
    static __host__ __device__ __nv_bfloat16 from(float x) { return __float2bfloat16_rn(x); }
    static __host__ __device__ float to(__nv_bfloat16 x) { return __bfloat162float(x); }
    static uint16_t special(int i) {
        constexpr uint16_t values[] = {0x0000, 0x8000, 0x0001, 0x8001,
            0x007f, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffd1, 0x3f80, 0xbf80};
        return values[i % 12];
    }
};
template<class T> struct Buffer {
    T* p = nullptr;
    size_t size;
    explicit Buffer(size_t count) : size(count) {
        if (std::max(count, size_t(1)) > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("allocation overflow");
        check(cudaMalloc(&p, std::max(count, size_t(1)) * sizeof(T)));
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& values) {
        if (values.size() != size) throw std::runtime_error("upload extent");
        if (size) check(cudaMemcpy(p, values.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> values(size);
        if (size) check(cudaMemcpy(values.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
};
template<class T> struct Matrix {
    const int rows, cols, stride;
    const size_t origin;
    Buffer<T> allocation;
    Matrix(int r, int c, int ld, int offset = 0)
        : rows(r), cols(c), stride(ld), origin(GuardElements + offset),
          allocation(origin + extent(r, ld) + GuardElements) {
        if (c < 0 || ld < std::max(c, 1) || offset < 0 || offset > 7)
            throw std::runtime_error("matrix layout");
    }
    T* data() const { return allocation.p + origin; }
    std::vector<T> blank() const {
        std::vector<T> values(allocation.size);
        std::memset(values.data(), 0xff, values.size() * sizeof(T));
        return values;
    }
    void poison(cudaStream_t stream) {
        check(cudaMemsetAsync(allocation.p, 0xff, allocation.size * sizeof(T), stream));
    }
    std::vector<T> read_view(int first, int count, const std::string& label) const {
        if (first < 0 || count <= 0 || first + count > rows)
            throw std::runtime_error("view bounds");
        auto all = allocation.read();
        std::vector<T> result(size_t(count) * cols);
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(all.data());
        auto guard = [&](size_t begin, size_t end) {
            auto bad = std::find_if(bytes + begin * sizeof(T), bytes + end * sizeof(T),
                [](unsigned char byte) { return byte != 0xff; });
            if (bad != bytes + end * sizeof(T))
                throw std::runtime_error(label + ": redzone/padding/inactive-row write at element " +
                    std::to_string(size_t(bad - bytes) / sizeof(T)));
        };
        guard(0, origin + size_t(first) * stride);
        for (int row = 0; row < count; ++row) {
            size_t at = origin + size_t(first + row) * stride;
            std::memcpy(result.data() + size_t(row) * cols, all.data() + at, size_t(cols) * sizeof(T));
            guard(at + cols, at + stride);
        }
        guard(origin + size_t(first + count) * stride, all.size());
        return result;
    }
};
template<class T> void bit_gate(const std::vector<T>& got, const std::vector<T>& expected,
                                const std::string& label) {
    if (got.size() != expected.size()) throw std::runtime_error(label + ": bit extent");
    if (!got.empty() && std::memcmp(got.data(), expected.data(), got.size() * sizeof(T))) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &expected[i], sizeof(T))) {
                uint64_t a = 0, b = 0;
                std::memcpy(&a, &got[i], sizeof(T)); std::memcpy(&b, &expected[i], sizeof(T));
                std::fprintf(stderr, "%s: first mismatch index=%zu got=0x%llx expected=0x%llx\n",
                    label.c_str(), i, (unsigned long long)a, (unsigned long long)b);
                break;
            }
        }
        throw std::runtime_error(label + ": BIT PARITY FAILURE");
    }
}
template<class T> double numeric_gate(const std::vector<T>& got, const std::vector<float>& ref,
                                     const std::string& label) {
    if (got.size() != ref.size() || got.empty()) throw std::runtime_error(label + ": numeric extent");
    double norm = 1.0, error = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double value = Type<T>::to(got[i]);
        if (!std::isfinite(value) || !std::isfinite(ref[i]))
            throw std::runtime_error(label + ": nonfinite finite-corpus result");
        norm = std::max(norm, std::abs(double(ref[i])));
        error = std::max(error, std::abs(value - ref[i]));
    }
    double normalized = error / norm;
    if (normalized > Type<T>::tolerance)
        throw std::runtime_error(label + ": PEDANTIC normalized error " + std::to_string(normalized));
    return normalized;
}
float seed_value(uint64_t index, uint64_t salt) {
    uint64_t x = index + salt + UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return float(int(x % 8191) - 4095) / 8192.0f;
}

template<class T> __global__ void reset_old(T* c, const T* old, int m, int n, int ldc) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(m) * n) c[(i / n) * ldc + i % n] = old[(i / n) * ldc + i % n];
}
template<class T> __global__ void vendor_seed(T* c, const float* bias, int m, int n, int ldc) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(m) * n)
        c[(i / n) * ldc + i % n] = Type<T>::from(bias ? bias[i % n] : 0.0f);
}
template<class T> __global__ void reference_seed(float* c, const T* old, const float* bias,
                                                float alpha, float beta, int m, int n, int old_ld) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(m) * n) {
        float value = __fmul_rn(alpha, bias ? bias[i % n] : 0.0f);
        if (beta != 0.0f) value = __fmaf_rn(beta, Type<T>::to(old[(i / n) * old_ld + i % n]), value);
        c[i] = value;
    }
}

template<class T> std::array<const void*, CustomArms> kernels();
template<> std::array<const void*, CustomArms> kernels<__half>() {
    return {reinterpret_cast<const void*>(gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_f16),
        reinterpret_cast<const void*>(ada_half_cutlass_s3_f16)};
}
template<> std::array<const void*, CustomArms> kernels<__nv_bfloat16>() {
    return {reinterpret_cast<const void*>(gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16),
        reinterpret_cast<const void*>(ada_half_cutlass_s3_bf16)};
}
template<class T> std::array<const char*, CustomArms> symbols();
template<> std::array<const char*, CustomArms> symbols<__half>() {
    return {"gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_f16",
        "ada_half_cutlass_s3_f16"};
}
template<> std::array<const char*, CustomArms> symbols<__nv_bfloat16>() {
    return {"gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_bf16",
        "ada_half_cutlass_s3_bf16"};
}
template<class T> void resources() {
    auto functions = kernels<T>();
    std::array<cudaFuncAttributes, CustomArms> attributes{};
    std::array<int, CustomArms> active_blocks{};
    for (int arm = 0; arm < CustomArms; ++arm) {
        auto function = functions[arm];
        check(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes[arm]));
        check(cudaFuncSetAttribute(function, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared));
        auto& attrs = attributes[arm];
        check(cudaFuncGetAttributes(&attrs, function));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks[arm], function, Threads[arm], SharedBytes[arm]));
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"resources\",\"dtype\":\"%s\",\"arm\":\"%s\",\"threads\":%d,\"dynamic_shared_bytes\":%d,\"registers\":%d,\"static_shared_bytes\":%zu,\"local_bytes\":%zu,\"max_threads_per_block\":%d,\"active_blocks_per_sm\":%d,\"binary_version\":%d,\"ptx_version\":%d}\n",
            Type<T>::name, Names[arm], Threads[arm], SharedBytes[arm], attrs.numRegs, attrs.sharedSizeBytes,
            attrs.localSizeBytes, attrs.maxThreadsPerBlock, active_blocks[arm],
            attrs.binaryVersion, attrs.ptxVersion);
        if (active_blocks[arm] != 1 || attrs.maxThreadsPerBlock < Threads[arm]
            || attrs.localSizeBytes != 0 || attrs.sharedSizeBytes != 0)
            throw std::runtime_error(std::string(Names[arm]) + ": occupancy/thread/local-memory gate");
    }
    if (attributes[1].numRegs > 255)
        throw std::runtime_error("candidate_cutlass_s3: architectural register gate");
}

struct Layout { int a_pad = 0, b_pad = 0, c_pad = 0, a_offset = 0, b_offset = 0, c_offset = 0; };
struct Options { int warmup = 128, graph_ops = 20, windows = 21; };
template<class T> struct Case {
    const int m, k, n, lda, ldb, ldc;
    const bool biased, exceptional, vendor_enabled;
    const float alpha, beta;
    const Options options;
    Matrix<T> a, b, old;
    Matrix<float> bias, reference;
    Buffer<float> a32, b32;
    std::array<std::unique_ptr<Matrix<T>>, Arms> out;
    std::vector<T> a_host, b_host, old_host, golden;
    std::vector<float> bias_host, ref_values;
    cudaStream_t stream = nullptr;
    cublasHandle_t fast = nullptr, pedantic = nullptr;
    cudaGraph_t graphs[Arms]{};
    cudaGraphExec_t execs[Arms]{};
    int first = 0, rows;
    std::array<double, Arms> errors{};

    Case(int M, int K, int N, bool use_bias, Layout layout, Options opts,
         bool use_vendor = false, float scale = 1.0f, float accumulate = 0.0f,
         bool special = false)
        : m(M), k(K), n(N), lda(std::max(K, 1) + layout.a_pad),
          ldb(N + layout.b_pad), ldc(N + layout.c_pad), biased(use_bias),
          exceptional(special), vendor_enabled(use_vendor), alpha(scale), beta(accumulate), options(opts),
          a(M, K, lda, layout.a_offset), b(K, N, ldb, layout.b_offset),
          old(M, N, ldc, layout.c_offset), bias(1, N, N), reference(M, N, N),
          a32(extent(M, std::max(K, 1))), b32(extent(K, N)), rows(M) {
        dimensions(m, k, n);
        if ((biased && alpha != 1.0f) || (vendor_enabled && (alpha != 1.0f || beta != 0.0f || exceptional)))
            throw std::runtime_error("unsupported arithmetic contract");
        for (auto& output : out) output.reset(new Matrix<T>(m, n, ldc, layout.c_offset));
        a_host = a.blank(); b_host = b.blank(); old_host = old.blank(); bias_host = bias.blank();
        std::vector<float> fa(a32.size, 0.0f), fb(b32.size, 0.0f);
        for (int row = 0; row < m; ++row) for (int col = 0; col < k; ++col) {
            uint64_t i = uint64_t(row) * k + col;
            T value = Type<T>::from(seed_value(i, UINT64_C(0x0adaa001)));
            if (exceptional && col < 12) value = raw_half<T>(Type<T>::special((col + row) % 12));
            a_host[a.origin + size_t(row) * lda + col] = value;
            fa[size_t(row) * std::max(k, 1) + col] = Type<T>::to(value);
        }
        for (int row = 0; row < k; ++row) for (int col = 0; col < n; ++col) {
            uint64_t i = uint64_t(row) * n + col;
            T value = Type<T>::from(seed_value(i, UINT64_C(0x0adab001)));
            if (exceptional && row < 12) value = raw_half<T>(Type<T>::special((row + col + 3) % 12));
            b_host[b.origin + size_t(row) * ldb + col] = value;
            fb[size_t(row) * n + col] = Type<T>::to(value);
        }
        for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col)
            old_host[old.origin + size_t(row) * ldc + col] = Type<T>::from(seed_value(uint64_t(row) * n + col, UINT64_C(0x0adac001)));
        constexpr uint32_t special_bias[] = {0, 0x80000000, 1, 0x80000001,
            0x7f800000, 0xff800000, 0x7fc01234, 0xffc04321};
        for (int col = 0; col < n; ++col) {
            // Deliberately not generally representable in half/bf16 storage.
            float value = seed_value(col, UINT64_C(0x0adab1a5)) * 0.173f;
            if (exceptional && col < 8) value = raw_float(special_bias[col]);
            bias_host[bias.origin + col] = value;
        }
        a.allocation.upload(a_host); b.allocation.upload(b_host); old.allocation.upload(old_host);
        bias.allocation.upload(bias_host); a32.upload(fa); b32.upload(fb);
        check(cudaDeviceSynchronize());
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blas_check(cublasCreate(&fast)); blas_check(cublasCreate(&pedantic));
        for (auto handle : {fast, pedantic}) {
            blas_check(cublasSetStream(handle, stream));
            blas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
            blas_check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
        }
        blas_check(cublasSetMathMode(fast, CUBLAS_DEFAULT_MATH));
        blas_check(cublasSetMathMode(pedantic, CUBLAS_PEDANTIC_MATH));
    }
    ~Case() {
        if (stream) cudaStreamSynchronize(stream);
        destroy_graphs();
        if (fast) cublasDestroy(fast);
        if (pedantic) cublasDestroy(pedantic);
        if (stream) cudaStreamDestroy(stream);
    }
    int arm_count() const { return vendor_enabled ? Arms : CustomArms; }
    void destroy_graphs() {
        for (int arm = 0; arm < Arms; ++arm) {
            if (execs[arm]) cudaGraphExecDestroy(execs[arm]);
            if (graphs[arm]) cudaGraphDestroy(graphs[arm]);
            execs[arm] = nullptr; graphs[arm] = nullptr;
        }
    }
    void poison() { for (auto& output : out) output->poison(stream); }
    void inputs_unchanged(const std::string& label) const {
        bit_gate(a.allocation.read(), a_host, label + ":A immutable+guards");
        bit_gate(b.allocation.read(), b_host, label + ":B immutable+guards");
        bit_gate(old.allocation.read(), old_host, label + ":old immutable+guards");
        bit_gate(bias.allocation.read(), bias_host, label + ":bias immutable+guards");
    }
    void reference_launch() {
        reference.poison(stream);
        unsigned blocks = unsigned((size_t(m) * n + 255) / 256);
        reference_seed<<<blocks, 256, 0, stream>>>(reference.data(), old.data(), biased ? bias.data() : nullptr,
            alpha, beta, m, n, ldc);
        check(cudaGetLastError());
        const float one = 1.0f;
        if (k) blas_check(cublasGemmEx(pedantic, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, &alpha, b32.p, CUDA_R_32F, n, a32.p, CUDA_R_32F, k,
            &one, reference.data(), CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    }
    void launch(int arm) {
        T* c = out[arm]->data() + size_t(first) * ldc;
        const T* a_ptr = k ? a.data() + size_t(first) * lda : nullptr;
        const T* b_ptr = k ? b.data() : nullptr;
        const T* old_ptr = old.data() + size_t(first) * ldc;
        const float* bias_ptr = biased ? bias.data() : nullptr;
        unsigned blocks = unsigned((size_t(rows) * n + 255) / 256);
        if (arm < CustomArms) {
            // Each beta!=0 operation starts from IDENTICAL Cold, including in
            // repeat graphs. This reset is only present in untimed edge gates.
            if (beta != 0.0f) reset_old<<<blocks, 256, 0, stream>>>(c, old_ptr, rows, n, ldc);
            FixedSm89HalfSwizzleParams params{
                alpha, beta, rows, n, k, lda, ldb, ldc};
            static_assert(sizeof(params) == sizeof(AdaHalfCutlassS3Params));
            void* args[] = {&c, &a_ptr, &b_ptr, &bias_ptr, &params};
            auto function = kernels<T>()[arm];
            unsigned grid = unsigned(((int64_t(rows) + TileM[arm] - 1) / TileM[arm])
                * ((int64_t(n) + TileN[arm] - 1) / TileN[arm]));
            check(cudaLaunchKernel(function, dim3(grid),
                dim3(Threads[arm]), args, SharedBytes[arm], stream));
        } else {
            if (!vendor_enabled) throw std::runtime_error("vendor not admitted for this case");
            if (biased || !k) vendor_seed<<<blocks, 256, 0, stream>>>(c, bias_ptr, rows, n, ldc);
            const float one = 1.0f, zero = 0.0f;
            if (k) blas_check(cublasGemmEx(fast, CUBLAS_OP_N, CUBLAS_OP_N,
                n, rows, k, &one, b_ptr, Type<T>::cuda_type, ldb, a_ptr, Type<T>::cuda_type, lda,
                biased ? &one : &zero, c, Type<T>::cuda_type, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        check(cudaGetLastError());
    }
    void make_graphs() {
        if (execs[0]) return;
        for (int arm = 0; arm < arm_count(); ++arm) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            for (int i = 0; i < options.graph_ops; ++i) launch(arm);
            check(cudaStreamEndCapture(stream, &graphs[arm]));
            check(cudaGraphInstantiate(&execs[arm], graphs[arm], nullptr, nullptr, 0));
        }
    }
    void graph_census(const std::string& label) {
        for (int arm = 0; arm < CustomArms; ++arm) {
            size_t count = 0;
            check(cudaGraphGetNodes(graphs[arm], nullptr, &count));
            std::vector<cudaGraphNode_t> nodes(count);
            check(cudaGraphGetNodes(graphs[arm], nodes.data(), &count));
            if (count != size_t(options.graph_ops))
                throw std::runtime_error(label + ": unexpected custom graph node count");
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType type{};
                check(cudaGraphNodeGetType(node, &type));
                if (type != cudaGraphNodeTypeKernel)
                    throw std::runtime_error(label + ": non-kernel custom graph node");
                cudaKernelNodeParams params{};
                check(cudaGraphKernelNodeGetParams(node, &params));
                if (!params.kernelParams) throw std::runtime_error(label + ": missing captured kernelParams");
                T* captured_c;
                const T* captured_a;
                const T* captured_b;
                const float* captured_bias;
                FixedSm89HalfSwizzleParams captured_scalars;
                std::memcpy(&captured_c, params.kernelParams[0], sizeof(captured_c));
                std::memcpy(&captured_a, params.kernelParams[1], sizeof(captured_a));
                std::memcpy(&captured_b, params.kernelParams[2], sizeof(captured_b));
                std::memcpy(&captured_bias, params.kernelParams[3], sizeof(captured_bias));
                std::memcpy(&captured_scalars, params.kernelParams[4], sizeof(captured_scalars));
                FixedSm89HalfSwizzleParams expected_scalars{alpha,beta,rows,n,k,lda,ldb,ldc};
                if (captured_c != out[arm]->data() + size_t(first) * ldc ||
                    captured_a != (k ? a.data() + size_t(first) * lda : nullptr) ||
                    captured_b != (k ? b.data() : nullptr) ||
                    captured_bias != (biased ? bias.data() : nullptr) ||
                    std::memcmp(&captured_scalars, &expected_scalars, sizeof(captured_scalars)) != 0)
                    throw std::runtime_error(label + ": captured kernel argument mismatch");
                unsigned expected_grid = unsigned(((int64_t(rows) + TileM[arm] - 1) / TileM[arm])
                    * ((int64_t(n) + TileN[arm] - 1) / TileN[arm]));
                if (params.func != const_cast<void*>(kernels<T>()[arm])
                    || params.gridDim.x != expected_grid || params.gridDim.y != 1
                    || params.gridDim.z != 1 || params.blockDim.x != unsigned(Threads[arm])
                    || params.blockDim.y != 1 || params.blockDim.z != 1
                    || params.sharedMemBytes != unsigned(SharedBytes[arm]))
                    throw std::runtime_error(label + ": custom graph physical identity mismatch");
            }
            std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"graph_identity\",\"dtype\":\"%s\",\"label\":\"%s\",\"arm\":\"%s\",\"physical_symbol\":\"%s\",\"kernel_nodes\":%zu,\"grid_x\":%u,\"block_x\":%d,\"shared_bytes\":%d,\"source_abi_args\":5,\"captured_arguments_checked\":true,\"exact_function_pointer\":true,\"stable_replay_identity\":true}\n",
                Type<T>::name, label.c_str(), Names[arm], symbols<T>()[arm], count,
                unsigned(((int64_t(rows) + TileM[arm] - 1) / TileM[arm])
                    * ((int64_t(n) + TileN[arm] - 1) / TileN[arm])),
                Threads[arm], SharedBytes[arm]);
        }
    }
    template<class V> std::vector<V> slice(const std::vector<V>& full) const {
        return std::vector<V>(full.begin() + size_t(first) * n, full.begin() + size_t(first + rows) * n);
    }
    void output_gate(const std::string& label) {
        auto wanted = slice(golden);
        std::vector<float> ref;
        if (!exceptional) ref = slice(ref_values);
        for (int arm = 0; arm < arm_count(); ++arm) {
            auto got = out[arm]->read_view(first, rows, label + ":" + Names[arm]);
            if (arm < CustomArms) bit_gate(got, wanted, label + ":" + Names[arm] + " versus production-full");
            if (!exceptional)
                errors[arm] = std::max(errors[arm], numeric_gate(got, ref, label + ":" + Names[arm]));
        }
    }
    void poison_graph_replay(const std::string& label) {
        // Invert every expected half bit: even exceptional NaNs must differ.
        // Padding/redzones/inactive rows retain the ordinary 0xff guard.
        check(cudaStreamSynchronize(stream));
        auto wanted = slice(golden);
        std::vector<T> poisoned(wanted.size());
        for (size_t i = 0; i < wanted.size(); ++i) {
            uint16_t bits;
            std::memcpy(&bits, &wanted[i], sizeof(bits));
            bits ^= 0xffff;
            std::memcpy(&poisoned[i], &bits, sizeof(bits));
        }
        for (int arm = 0; arm < arm_count(); ++arm) {
            auto host = out[arm]->blank();
            for (int row = 0; row < rows; ++row)
                std::memcpy(host.data() + out[arm]->origin + size_t(first + row) * ldc,
                    poisoned.data() + size_t(row) * n, size_t(n) * sizeof(T));
            out[arm]->allocation.upload(host);
            auto got = out[arm]->read_view(first, rows, label + ":poison-guards");
            bit_gate(got, poisoned, label + ":poison-upload");
            for (size_t i = 0; i < got.size(); ++i)
                if (std::memcmp(&got[i], &wanted[i], sizeof(T)) == 0)
                    throw std::runtime_error(label + ": output not poisoned");
        }
    }
    void validate(const std::string& label, bool establish = false) {
        poison();
        if (establish && !exceptional) reference_launch();
        for (int arm = 0; arm < arm_count(); ++arm) launch(arm);
        check(cudaStreamSynchronize(stream));
        if (establish) {
            if (first || rows != m) throw std::runtime_error("golden must cover full matrix");
            golden = out[0]->read_view(0, m, label + ":production-golden");
            if (!exceptional) ref_values = reference.read_view(0, m, label + ":PEDANTIC");
        }
        output_gate(label + ":eager");
        for (int repeat = 0; repeat < 2; ++repeat) {
            poison();
            for (int arm = 0; arm < arm_count(); ++arm) launch(arm);
            check(cudaStreamSynchronize(stream));
            output_gate(label + ":eager-repeat");
        }
        make_graphs();
        if (label == "requested") graph_census(label);
        for (int repeat = 0; repeat < 2; ++repeat) {
            poison_graph_replay(label);
            // Negative test only: the output gate must catch a no-op graph.
            for (int arm = 0; arm < arm_count(); ++arm)
                if (arm != 1 || !std::getenv("ADA_S3_TEST_SKIP_GRAPH"))
                    check(cudaGraphLaunch(execs[arm], stream));
            check(cudaStreamSynchronize(stream));
            output_gate(label + ":graph-repeat");
        }
        inputs_unchanged(label);
        if (!exceptional) reference.read_view(0, m, label + ":reference-guards");
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"validation\",\"label\":\"%s\",\"dtype\":\"%s\",\"full_m\":%d,\"first_row\":%d,\"m\":%d,\"k\":%d,\"n\":%d,\"lda\":%d,\"ldb\":%d,\"ldc\":%d,\"a_alignment_mod16\":%zu,\"b_alignment_mod16\":%zu,\"c_alignment_mod16\":%zu,\"bias\":%s,\"alpha\":%.9g,\"beta\":%.9g,\"corpus\":\"%s\",\"numeric_reference\":%s,\"vendor_checked\":%s,\"custom_exact_bits\":true,\"eager_repeat\":true,\"graph_repeat\":true,\"graph_poisoned_each_replay\":true,\"graph_poison_differs_every_output\":true,\"guards\":true,\"inputs_unchanged\":true}\n",
            label.c_str(), Type<T>::name, m, first, rows, k, n, lda, ldb, ldc,
            (reinterpret_cast<uintptr_t>(a.data() + size_t(first) * lda) & 15),
            (reinterpret_cast<uintptr_t>(b.data()) & 15),
            (reinterpret_cast<uintptr_t>(out[0]->data() + size_t(first) * ldc) & 15),
            biased ? "true" : "false", alpha, beta, exceptional ? "exceptional_v1" : "signed_half_nonrepresentable_bias_v1",
            exceptional ? "false" : "true", vendor_enabled ? "true" : "false");
    }
    void view(int start, int count, const std::string& label) {
        if (start < 0 || count <= 0 || start > m - count) throw std::runtime_error("invalid view");
        check(cudaStreamSynchronize(stream));
        destroy_graphs(); first = start; rows = count;
        validate(label);
    }
    void batch_gates() {
        std::vector<int> sizes = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
            255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, m - 1};
        std::sort(sizes.begin(), sizes.end());
        sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
        for (int count : sizes) if (count > 0 && count < m) view(0, count, "batch-prefix");
        for (int start : {1, 17, 127, m - 1})
            if (start > 0 && start < m) view(start, std::min(129, m - start), "row-subview");
        view(0, m, "restore-full-after-views");
    }
};

template<class T> void edge_gates(Options options) {
    for (bool biased : {false, true}) {
        for (int k : {0, 1, 15, 16, 17, 63, 64, 65}) {
            Layout layout;
            // Rounded physical strides exercise aligned cp.async zero-fill
            // even when logical K/N are not multiples of eight.
            layout.a_pad = ((std::max(k, 1) + 7) / 8) * 8 - std::max(k, 1);
            layout.b_pad = 136 - 131; layout.c_pad = 136 - 131;
            Case<T> test(129, k, 131, biased, layout, options);
            test.validate("aligned-tail-k-ladder", true);
        }
        const Layout layouts[] = {
            {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1},
            {1, 1, 1, 0, 0, 0}, {3, 5, 7, 1, 1, 1}};
        for (const auto& layout : layouts) {
            Case<T> test(129, 128, 256, biased, layout, options);
            test.validate("independent-misalignment-and-odd-strides", true);
            test.view(1, 128, "unaligned-row-subview");
        }
        Case<T> batch(257, 65, 129, biased, {}, options);
        batch.validate("tail-batch-base", true);
        batch.batch_gates();
        for (int k : {0, 64, 65}) {
            Case<T> beta(129, k, 131, biased, {1, 1, 1, 1, 1, 1}, options,
                         false, biased ? 1.0f : -0.75f, 0.5f);
            beta.validate("nontrivial-alpha-beta", true);
            beta.view(1, 128, "nontrivial-alpha-beta-row-subview");
        }
        for (int k : {0, 64, 65}) {
            Case<T> special(129, k, 256, biased, {}, options, false, 1.0f, 0.0f, true);
            special.validate("exceptional", true);
            special.view(1, 128, "exceptional-row-subview");
        }
        for (int k : {64, 128, 192, 256}) {
            Case<T> short_ring(129, k, 256, biased, {}, options);
            short_ring.validate("aligned-short-s3-prologue-drain", true);
            short_ring.view(1, 128, "aligned-short-s3-prologue-drain-view");
        }
        // Aligned 3+ BK64 stages exercise S3 wraparound with both full vector
        // output columns and an 8-column scalar tail. Views retain the same
        // physical rows/strides and prove slab starts do not alter the chain.
        for (int k : {192, 193, 256, 384, 768}) {
            Layout layout;
            layout.a_pad = ((k + 7) / 8) * 8 - k;
            Case<T> staged(385, k, 264, biased, layout, options);
            staged.validate("aligned-three-plus-stage-base", true);
            for (int count : {1, 16, 64, 128, 129, 256})
                staged.view(0, count, "aligned-three-plus-stage-prefix");
            for (int first : {1, 17, 127, 128, 129, 256})
                staged.view(first, std::min(129, 385 - first), "aligned-three-plus-stage-slab");
        }
        Case<T> special_staged(257, 192, 256, biased, {}, options, false, 1.0f, 0.0f, true);
        special_staged.validate("aligned-three-stage-exceptional", true);
        special_staged.view(17, 129, "aligned-three-stage-exceptional-slab");
        // Explicit non-unit alpha while vector epilogue is eligible.
        if (!biased) {
            Case<T> scaled(129, 128, 256, false, {}, options, false, -0.75f);
            scaled.validate("vector-output-nonunit-alpha", true);
        }
    }
}

double median(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("empty median");
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return values.size() & 1 ? values[mid] : (values[mid - 1] + values[mid]) * 0.5;
}
double p95(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("empty p95");
    std::sort(values.begin(), values.end());
    size_t index = (95 * values.size() + 99) / 100 - 1;
    return values[index];
}
struct Sample { int window, order, comparison, position, arm; double us; };
template<class T> void benchmark(int m, int k, int n, bool biased, Options options) {
    resources<T>();
    std::fflush(stdout);
    edge_gates<T>(options);
    Case<T> test(m, k, n, biased, {}, options, true);
    test.validate("requested", true);
    test.batch_gates();
    if (options.windows == 0) {
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"correctness_complete\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"all_gates_passed\":true,\"timed_samples\":0}\n",
            Type<T>::name, m, k, n, biased ? "true" : "false");
        return;
    }
    for (int warm = 0; warm < options.warmup; ++warm)
        for (int arm = 0; arm < Arms; ++arm) test.launch(arm);
    check(cudaStreamSynchronize(test.stream));
    // Warm graph replay as well; no readback or stdout between this and timing.
    for (int arm = 0; arm < Arms; ++arm) check(cudaGraphLaunch(test.execs[arm], test.stream));
    check(cudaStreamSynchronize(test.stream));
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::vector<Sample> samples;
    samples.reserve(size_t(options.windows) * ComparisonCount * 4);
    std::array<std::vector<double>, Arms> times;
    std::array<std::vector<double>, ComparisonCount> ratios;
    auto measure = [&](int arm) {
        check(cudaEventRecord(start, test.stream));
        check(cudaGraphLaunch(test.execs[arm], test.stream));
        check(cudaEventRecord(stop, test.stream));
        check(cudaEventSynchronize(stop));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop));
        double us = double(ms) * 1000.0 / options.graph_ops;
        if (!std::isfinite(us) || us <= 0.0) throw std::runtime_error("invalid CUDA event sample");
        return us;
    };
    for (int window = 0; window < options.windows; ++window) {
        for (int order = 0; order < ComparisonCount; ++order) {
            int comparison = window & 1 ? ComparisonCount - 1 - order : order;
            int a = Comparisons[comparison][0], b = Comparisons[comparison][1];
            bool baab = (window & 1) != 0;
            int pair_order[4] = {baab ? b : a, baab ? a : b,
                                 baab ? a : b, baab ? b : a};
            double values[4];
            for (int pos = 0; pos < 4; ++pos) {
                int arm = pair_order[pos];
                values[pos] = measure(arm);
                samples.push_back({window, order, comparison, pos, arm, values[pos]});
                times[arm].push_back(values[pos]);
            }
            ratios[comparison].push_back(baab
                ? (values[0] + values[3]) / (values[1] + values[2])
                : (values[1] + values[2]) / (values[0] + values[3]));
        }
    }
    check(cudaStreamSynchronize(test.stream));
    test.output_gate("post-timing");
    test.inputs_unchanged("post-timing");
    test.reference.read_view(0, m, "post-timing:reference-guards");
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    for (const auto& s : samples) {
        int a = Comparisons[s.comparison][0], b = Comparisons[s.comparison][1];
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"sample\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"window\":%d,\"comparison_order\":%d,\"pair_order\":\"%s\",\"comparison\":%d,\"a\":\"%s\",\"b\":\"%s\",\"position\":%d,\"arm\":\"%s\",\"us_per_op\":%.9f}\n",
            Type<T>::name, m, k, n, biased ? "true" : "false", s.window, s.order,
            (s.window & 1) ? "BAAB" : "ABBA", s.comparison, Names[a], Names[b],
            s.position, Names[s.arm], s.us);
    }
    for (int c = 0; c < ComparisonCount; ++c) {
        for (int window = 0; window < options.windows; ++window)
            std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"pair\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"comparison\":%d,\"window\":%d,\"pair_order\":\"%s\",\"a\":\"%s\",\"b\":\"%s\",\"b_over_a\":%.9f}\n",
                Type<T>::name, m, k, n, biased ? "true" : "false", c, window,
                (window & 1) ? "BAAB" : "ABBA", Names[Comparisons[c][0]],
                Names[Comparisons[c][1]], ratios[c][window]);
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"comparison_summary\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"comparison\":%d,\"a\":\"%s\",\"b\":\"%s\",\"abba_windows\":%d,\"paired_b_over_a_p50\":%.9f,\"paired_b_over_a_p95\":%.9f}\n",
            Type<T>::name, m, k, n, biased ? "true" : "false", c,
            Names[Comparisons[c][0]], Names[Comparisons[c][1]], options.windows,
            median(ratios[c]), p95(ratios[c]));
    }
    for (int arm = 0; arm < Arms; ++arm)
        std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"arm_summary\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"arm\":\"%s\",\"us_per_op_p50\":%.9f,\"samples\":%zu,\"max_observed_normalized_reference_error\":%.9g}\n",
            Type<T>::name, m, k, n, biased ? "true" : "false", Names[arm],
            median(times[arm]), times[arm].size(), test.errors[arm]);
    std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"complete\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"raw_samples\":%zu,\"all_gates_passed\":true}\n",
        Type<T>::name, m, k, n, biased ? "true" : "false", samples.size());
}
} // namespace half_swizzle_bench

int main(int argc, char** argv) try {
    using namespace half_swizzle_bench;
    if (argc < 6 || (argc - 6) % 2)
        throw std::runtime_error("usage: binary bf16|f16 M K N bias(0|1) [--warmup N] [--graph-ops N] [--windows N]");
    std::string dtype = argv[1];
    if (dtype != "bf16" && dtype != "f16") throw std::runtime_error("dtype must be bf16 or f16");
    int m = integer(argv[2], 1, INT_MAX - 256), k = integer(argv[3], 0, INT_MAX - 256);
    int n = integer(argv[4], 1, INT_MAX - 256);
    bool biased = integer(argv[5], 0, 1) != 0;
    if (biased) throw std::runtime_error("candidate is B0 bias-free only");
    dimensions(m, k, n);
    Options options;
    bool seen_warmup = false, seen_graph = false, seen_windows = false;
    for (int i = 6; i < argc; i += 2) {
        std::string option = argv[i];
        bool* seen = nullptr;
        if (option == "--warmup") { seen = &seen_warmup; options.warmup = integer(argv[i + 1], 1, 100000); }
        else if (option == "--graph-ops") { seen = &seen_graph; options.graph_ops = integer(argv[i + 1], 1, 10000); }
        else if (option == "--windows") { seen = &seen_windows; options.windows = integer(argv[i + 1], 0, 10001); }
        else throw std::runtime_error("unknown option: " + option);
        if (*seen) throw std::runtime_error("duplicate option: " + option);
        *seen = true;
    }
    check(cudaSetDevice(0));
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0));
    if (device.major != 8 || device.minor != 9) throw std::runtime_error("this OWN experiment requires CC8.9");
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    std::printf("{\"schema\":\"AdaHalfCutlassS3V1\",\"record\":\"configuration\",\"dtype\":\"%s\",\"m\":%d,\"k\":%d,\"n\":%d,\"bias\":%s,\"alpha\":1,\"beta\":0,\"cc\":89,\"sm_count\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"warmup_eager_ops_per_arm\":%d,\"graph_ops\":%d,\"abba_windows_per_comparison\":%d,\"comparisons\":%d,\"timing\":\"cuda_events_graph\",\"timed_corpus\":\"signed_half_v1\",\"vendor_compute\":\"CUBLAS_COMPUTE_32F\",\"vendor_algorithm\":\"CUBLAS_GEMM_DEFAULT_TENSOR_OP\",\"vendor_math\":\"CUBLAS_DEFAULT_MATH\",\"vendor_bias_broadcast_timed\":false,\"vendor_bias_rounds_to_half\":false,\"reference_compute\":\"CUBLAS_COMPUTE_32F_PEDANTIC\",\"reference_input_and_output\":\"F32\",\"reference_bias\":\"none\",\"numeric_norm\":\"max_abs_error/max(1,reference_inf_norm)\",\"guard_elements_per_side\":%zu}\n",
        dtype.c_str(), m, k, n, biased ? "true" : "false", device.multiProcessorCount, runtime, driver,
        options.warmup, options.graph_ops, options.windows, ComparisonCount,
        GuardElements);
    if (dtype == "bf16") benchmark<__nv_bfloat16>(m, k, n, biased, options);
    else benchmark<__half>(m, k, n, biased, options);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAILED (discard run unless complete record exists and exit=0): %s\n", error.what());
    return 1;
}
