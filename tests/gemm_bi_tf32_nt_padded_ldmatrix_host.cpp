#include "gemm_bi_tf32_nt_padded_ldmatrix.cuh"

#include <cstdlib>
#include <iostream>

namespace {

constexpr int kStride = 36;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void padded_ldmatrix_address_mapping_matches_scalar_fragments() {
    constexpr int k8_offsets[] = {0, 8, 16, 24};
    for (int warp = 0; warp < 4; ++warp) {
        const int warp_m = (warp >> 1) * 64;
        const int warp_n = (warp & 1) * 32;
        for (int k8 : k8_offsets) {
            for (int lane = 0; lane < 32; ++lane) {
                const int group = lane >> 2;
                const int thread = lane & 3;
                for (int atom = 0; atom < 4; ++atom) {
                    for (int reg = 0; reg < 4; ++reg) {
                        const int address_lane = reg * 8 + group;
                        const auto source = gemm_bi_tf32_nt_padded_a_address(
                            warp_m, atom, k8, address_lane);
                        const int want_row = warp_m + atom * 16 + group
                            + ((reg & 1) != 0 ? 8 : 0);
                        const int want_reduction = k8 + thread
                            + (reg >= 2 ? 4 : 0);
                        require(source.row == want_row,
                                "A ldmatrix register row differs from scalar fragment");
                        require(source.reduction + thread == want_reduction,
                                "A ldmatrix register K differs from scalar fragment");
                    }
                    for (int reg = 0; reg < 2; ++reg) {
                        const int address_lane = reg * 8 + group;
                        const auto source = gemm_bi_tf32_nt_padded_b_address(
                            warp_n, atom, k8, address_lane);
                        const int want_row = warp_n + atom * 8 + group;
                        const int want_reduction = k8 + thread + reg * 4;
                        require(source.row == want_row,
                                "B ldmatrix register row differs from scalar fragment");
                        require(source.reduction + thread == want_reduction,
                                "B ldmatrix register K differs from scalar fragment");
                    }
                }
            }
        }
    }
}

void padded_ldmatrix_addresses_are_aligned_and_in_bounds() {
    constexpr int k8_offsets[] = {0, 8, 16, 24};
    for (int warp = 0; warp < 4; ++warp) {
        const int warp_m = (warp >> 1) * 64;
        const int warp_n = (warp & 1) * 32;
        for (int k8 : k8_offsets) {
            for (int lane = 0; lane < 32; ++lane) {
                for (int atom = 0; atom < 4; ++atom) {
                    const auto a = gemm_bi_tf32_nt_padded_a_address(
                        warp_m, atom, k8, lane);
                    require(a.row >= 0 && a.row < 128,
                            "A row address leaves the M128 stage");
                    require(a.reduction >= 0 && a.reduction + 3 < 32,
                            "A row chunk reaches the stride-36 padding");
                    require(((a.row * kStride + a.reduction) * 4) % 16 == 0,
                            "A row chunk is not naturally 16-byte aligned");

                    const auto b = gemm_bi_tf32_nt_padded_b_address(
                        warp_n, atom, k8, lane);
                    require(b.row >= 0 && b.row < 64,
                            "B row address leaves the N64 stage");
                    require(b.reduction >= 0 && b.reduction + 3 < 32,
                            "B row chunk reaches the stride-36 padding");
                    require(((b.row * kStride + b.reduction) * 4) % 16 == 0,
                            "B row chunk is not naturally 16-byte aligned");
                }
            }
        }
    }
}

void padded_ldmatrix_x2_upper_lanes_repeat_valid_addresses() {
    constexpr int k8_offsets[] = {0, 8, 16, 24};
    for (int warp = 0; warp < 4; ++warp) {
        const int warp_n = (warp & 1) * 32;
        for (int atom = 0; atom < 4; ++atom) {
            for (int k8 : k8_offsets) {
                for (int lane = 16; lane < 32; ++lane) {
                    const auto upper = gemm_bi_tf32_nt_padded_b_address(
                        warp_n, atom, k8, lane);
                    const auto lower = gemm_bi_tf32_nt_padded_b_address(
                        warp_n, atom, k8, lane - 16);
                    require(upper.row == lower.row && upper.reduction == lower.reduction,
                            "B x2 upper lane does not repeat its lower-lane address");
                }
            }
        }
    }
}

}  // namespace

int main() {
    padded_ldmatrix_address_mapping_matches_scalar_fragments();
    padded_ldmatrix_addresses_are_aligned_and_in_bounds();
    padded_ldmatrix_x2_upper_lanes_repeat_valid_addresses();
    std::cout << "PASS: padded NT ldmatrix mapping, bounds, alignment, and x2 replication\n";
}
