// c++ -std=c++17 -O2 tests/cuda/gemm_bi_fixed_sm89_swizzle_layout.cpp -o /tmp/gemm_bi_fixed_sm89_swizzle_layout
// Pure-host exhaustive proof over the production constexpr mapping helpers.
#include "../../kernels/gemm_bi_inference/sm80/half_swizzle_layout.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
namespace l = sm89_fixed_half_swizzle_layout;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
int main() try {
    std::array<int, 4 * l::kStageElements> shared{};
    shared.fill(-1);
    size_t chunks = 0, fragment_halfs = 0, bank_groups = 0;
    for (int stage = 0; stage < 2; ++stage) {
        int a_stage = stage * l::kStageElements;
        int b_stage = (2 + stage) * l::kStageElements;
        for (int thread = 0; thread < 256; ++thread) for (int slice = 0; slice < 4; ++slice) {
            int ar = (thread >> 3) + slice * 32, ak = (thread & 7) * 8;
            int bk = (thread >> 4) + slice * 16, bc = (thread & 15) * 8;
            unsigned ad = l::a_copy_offset(thread, slice), bd = l::b_copy_offset(thread, slice);
            require(ad % 16 == 0 && bd % 16 == 0, "cp.async 16B destination alignment");
            require(ad / 2 + 7 < l::kStageElements && bd / 2 + 7 < l::kStageElements,
                "copy chunk escapes its stage");
            for (int e = 0; e < 8; ++e) {
                require(l::a_index(ar, ak + e) == int(ad / 2) + e,
                    "A scalar and cp.async chunk mappings disagree");
                require(l::b_index(bk, bc + e) == int(bd / 2) + e,
                    "B scalar and cp.async chunk mappings disagree");
                int ai = a_stage + int(ad / 2) + e, bi = b_stage + int(bd / 2) + e;
                require(shared[ai] == -1 && shared[bi] == -1, "staging mapping collision");
                shared[ai] = stage * 100000 + ar * 64 + ak + e;
                shared[bi] = stage * 100000 + 20000 + bk * 128 + bc + e;
            }
            chunks += 2;
        }
    }
    for (int value : shared) require(value != -1, "staging mapping leaves holes");
    for (int stage = 0; stage < 2; ++stage) for (int warp = 0; warp < 8; ++warp)
    for (int atom = 0; atom < 4; ++atom) for (int issue = 0; issue < 4; ++issue) {
        int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
        std::array<unsigned, 32> a{}, b{};
        for (int lane = 0; lane < 32; ++lane) {
            a[lane] = l::a_fragment_issue(l::a_fragment_base(wm, atom, lane), issue);
            b[lane] = l::b_fragment_issue(l::b_fragment_base(wn, atom, lane), issue);
            require(a[lane] % 16 == 0 && b[lane] % 16 == 0, "ldmatrix row not naturally aligned");
        }
        for (int lane = 0; lane < 32; ++lane) for (int reg = 0; reg < 4; ++reg)
        for (int e = 0; e < 2; ++e) {
            unsigned index = a[reg * 8 + lane / 4] / 2 + (lane % 4) * 2 + e;
            require(index < l::kStageElements, "A fragment escapes stage");
            int row = wm + atom * 16 + (reg % 2) * 8 + lane / 4;
            int k = issue * 16 + (reg / 2) * 8 + (lane % 4) * 2 + e;
            require(shared[stage * l::kStageElements + index] == stage * 100000 + row * 64 + k,
                "ldmatrix.x4 A fragment changed logical value or lane");
            ++fragment_halfs;
        }
        for (int lane = 0; lane < 32; ++lane) for (int reg = 0; reg < 2; ++reg)
        for (int e = 0; e < 2; ++e) {
            unsigned index = b[reg * 8 + (lane % 4) * 2 + e] / 2 + lane / 4;
            require(index < l::kStageElements, "B fragment escapes stage");
            int k = issue * 16 + reg * 8 + (lane % 4) * 2 + e;
            int column = wn + atom * 8 + lane / 4;
            require(shared[(2 + stage) * l::kStageElements + index] ==
                stage * 100000 + 20000 + k * 128 + column,
                "ldmatrix.x2.trans B fragment changed logical value or lane");
            ++fragment_halfs;
        }
        for (int operand = 0; operand < 2; ++operand)
        for (int matrix = 0; matrix < (operand == 0 ? 4 : 2); ++matrix) {
            std::array<bool, 32> banks{};
            for (int row = 0; row < 8; ++row) for (int word = 0; word < 4; ++word) {
                unsigned address = (operand == 0 ? a : b)[matrix * 8 + row] + word * 4;
                unsigned bank = (address / 4) % 32;
                require(!banks[bank], "ldmatrix row group has a shared-bank conflict");
                banks[bank] = true;
            }
            ++bank_groups;
        }
    }
    std::printf("{\"schema\":\"Sm89HalfSwizzleProductionHostV1\",\"all_passed\":true,\"copied_chunks\":%zu,\"checked_fragment_halfs\":%zu,\"conflict_free_ldmatrix_groups\":%zu,\"staging_bytes\":65536,\"dynamic_shared_bytes\":%d}\n",
        chunks, fragment_halfs, bank_groups, l::kSharedBytes);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HOST MAPPING FAILURE: %s\n", error.what());
    return 1;
}
