pub const SYMBOL: &str = "gemm_bi_nt_test_compact_a_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s2";
const PARENT_SYMBOL: &str = "gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";
const EXPECTED_PARENT_FNV64: u64 = 0x2ded_0f1a_133e_5949;

const A_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            int row = warp_m + m_atom * 16 + group;
            a_fragments[m_atom][0] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));
            a_fragments[m_atom][1] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));
            a_fragments[m_atom][2] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));
            a_fragments[m_atom][3] =
                gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));
        }
"#;

const A_LDMATRIX_LOAD: &str = r#"#pragma unroll
        for (int m_atom = 0; m_atom < MAtoms; ++m_atom) {
            if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {
                int lane = (int)threadIdx.x & 31;
                int row = warp_m + m_atom * 16 + (lane & 15);
                int reduction = k8 + ((lane >> 4) << 2);
                unsigned address = (unsigned)__cvta_generic_to_shared(
                    &gemm_bi_tf32_a_slot<Op>(storage, stage, row, reduction));
                unsigned raw0, raw1, raw2, raw3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                    : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3)
                    : "r"(address));
                a_fragments[m_atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
                a_fragments[m_atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
                a_fragments[m_atom][2] = gemm_bi_tf32_rna(__uint_as_float(raw2));
                a_fragments[m_atom][3] = gemm_bi_tf32_rna(__uint_as_float(raw3));
            } else {
                int row = warp_m + m_atom * 16 + group;
                a_fragments[m_atom][0] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread));
                a_fragments[m_atom][1] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread));
                a_fragments[m_atom][2] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row, k8 + thread + 4));
                a_fragments[m_atom][3] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_a_slot<Op>(storage, stage, row + 8, k8 + thread + 4));
            }
        }
"#;

pub const fn compact_a_index(row: usize, reduction: usize) -> usize {
    row * 32 + (reduction ^ ((row & 7) << 2))
}

pub const fn ldmatrix_a_address(
    warp_m: usize,
    atom: usize,
    k8: usize,
    lane: usize,
) -> (usize, usize) {
    (warp_m + atom * 16 + (lane & 15), k8 + ((lane >> 4) << 2))
}

pub const fn scalar_a_coordinate(
    warp_m: usize,
    atom: usize,
    k8: usize,
    lane: usize,
    register: usize,
) -> (usize, usize) {
    let group = lane >> 2;
    let thread = lane & 3;
    (
        warp_m + atom * 16 + group + if register & 1 != 0 { 8 } else { 0 },
        k8 + thread + if register >= 2 { 4 } else { 0 },
    )
}

pub fn candidate_source(compact_parent: &str) -> Result<String, String> {
    let digest = fnv64(compact_parent.as_bytes());
    if digest != EXPECTED_PARENT_FNV64 {
        return Err(format!(
            "compact A-only ldmatrix parent digest changed: expected {EXPECTED_PARENT_FNV64:#018x}, observed {digest:#018x}"
        ));
    }
    transform_source(compact_parent)
}

fn transform_source(compact_parent: &str) -> Result<String, String> {
    let mut source = compact_parent.to_owned();
    replace_exact(&mut source, A_LOAD, A_LDMATRIX_LOAD, 1, "A fragment load")?;
    replace_exact(&mut source, PARENT_SYMBOL, SYMBOL, 2, "target symbol")?;
    Ok(source)
}

fn replace_exact(
    source: &mut String,
    from: &str,
    to: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let count = source.matches(from).count();
    if count != expected {
        return Err(format!(
            "compact A-only ldmatrix {label} count changed: expected {expected}, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x4_register_coordinates_equal_current_scalar_a_fragments() {
        for warp in 0..8 {
            let warp_m = (warp >> 1) * 32;
            for atom in 0..2 {
                for k8 in [0, 8, 16, 24] {
                    for lane in 0..32 {
                        let group = lane >> 2;
                        let thread = lane & 3;
                        for register in 0..4 {
                            let address_lane = register * 8 + group;
                            let (row, reduction) =
                                ldmatrix_a_address(warp_m, atom, k8, address_lane);
                            assert_eq!(
                                (row, reduction + thread),
                                scalar_a_coordinate(warp_m, atom, k8, lane, register)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_addresses_are_aligned_in_bounds_and_match_n96_index() {
        for row in 0..128 {
            for reduction in 0..32 {
                let chunk = (reduction >> 2) ^ (row & 7);
                let n96_index = row * 32 + chunk * 4 + (reduction & 3);
                assert_eq!(compact_a_index(row, reduction), n96_index);
            }
        }
        for warp in 0..8 {
            let warp_m = (warp >> 1) * 32;
            for atom in 0..2 {
                for k8 in [0, 8, 16, 24] {
                    for lane in 0..32 {
                        let (row, reduction) = ldmatrix_a_address(warp_m, atom, k8, lane);
                        assert!(row < 128);
                        assert!(reduction + 3 < 32);
                        assert_eq!(compact_a_index(row, reduction) * 4 % 16, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn source_adapter_changes_only_the_target_a_load_primitive() {
        let parent = format!(
            "prefix\n{A_LOAD}GEMM_BI_TF32_DEFINE_KERNEL({PARENT_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)\nTF32_ASSERT_KERNEL_SIGNATURE({PARENT_SYMBOL});\nsuffix\n"
        );
        let candidate = transform_source(&parent).unwrap();
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert!(candidate.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"));
        let restored = candidate
            .replacen(A_LDMATRIX_LOAD, A_LOAD, 1)
            .replace(SYMBOL, PARENT_SYMBOL);
        assert_eq!(restored, parent);
    }

    #[test]
    fn source_adapter_rejects_missing_or_duplicated_anchors() {
        let parent = format!("{A_LOAD}{PARENT_SYMBOL}{PARENT_SYMBOL}");
        assert!(transform_source(&parent.replacen(A_LOAD, "", 1)).is_err());
        assert!(transform_source(&format!("{parent}{A_LOAD}")).is_err());
        assert!(transform_source(&parent.replacen(PARENT_SYMBOL, "missing", 1)).is_err());
        assert!(transform_source(&format!("{parent}{PARENT_SYMBOL}")).is_err());
    }
}
