pub const SYMBOL: &str = "gemm_bi_nt_test_compact_ab_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s2";
const PARENT_SYMBOL: &str = "gemm_bi_nt_test_compact_a_ldmatrix_sm80_mma_tf32_v1_m128n64_bk32_s2";
const EXPECTED_PARENT_FNV64: u64 = 0xa526_e1e9_d56a_ed87;

const B_LOAD: &str = r#"#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            int column = warp_n + n_atom * 8 + group;
            b_fragments[n_atom][0] =
                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread, column));
            b_fragments[n_atom][1] =
                gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread + 4, column));
        }
"#;

const B_LDMATRIX_LOAD: &str = r#"#pragma unroll
        for (int n_atom = 0; n_atom < NAtoms; ++n_atom) {
            if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2) {
                int lane = (int)threadIdx.x & 31;
                int column = warp_n + n_atom * 8 + (lane & 7);
                int reduction = k8 + (((lane >> 3) & 1) << 2);
                unsigned address = (unsigned)__cvta_generic_to_shared(
                    &gemm_bi_tf32_b_slot<Op>(storage, stage, reduction, column));
                unsigned raw0, raw1;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                    : "=r"(raw0), "=r"(raw1)
                    : "r"(address));
                b_fragments[n_atom][0] = gemm_bi_tf32_rna(__uint_as_float(raw0));
                b_fragments[n_atom][1] = gemm_bi_tf32_rna(__uint_as_float(raw1));
            } else {
                int column = warp_n + n_atom * 8 + group;
                b_fragments[n_atom][0] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread, column));
                b_fragments[n_atom][1] =
                    gemm_bi_tf32_rna(gemm_bi_tf32_b_slot<Op>(storage, stage, k8 + thread + 4, column));
            }
        }
"#;

pub const fn compact_index(column: usize, reduction: usize) -> usize {
    column * 32 + (reduction ^ ((column & 7) << 2))
}

pub const fn ldmatrix_b_address(
    warp_n: usize,
    atom: usize,
    k8: usize,
    lane: usize,
) -> (usize, usize) {
    (
        warp_n + atom * 8 + (lane & 7),
        k8 + (((lane >> 3) & 1) << 2),
    )
}

pub const fn scalar_b_coordinate(
    warp_n: usize,
    atom: usize,
    k8: usize,
    lane: usize,
    register: usize,
) -> (usize, usize) {
    (
        warp_n + atom * 8 + (lane >> 2),
        k8 + (lane & 3) + register * 4,
    )
}

pub fn candidate_source(a_ldmatrix_parent: &str) -> Result<String, String> {
    let digest = fnv64(a_ldmatrix_parent.as_bytes());
    if digest != EXPECTED_PARENT_FNV64 {
        return Err(format!(
            "compact A+B ldmatrix parent digest changed: expected {EXPECTED_PARENT_FNV64:#018x}, observed {digest:#018x}"
        ));
    }
    transform_source(a_ldmatrix_parent)
}

pub fn restore_parent(candidate: &str) -> Result<String, String> {
    let mut source = candidate.to_owned();
    replace_exact(
        &mut source,
        B_LDMATRIX_LOAD,
        B_LOAD,
        1,
        "restored B fragment load",
    )?;
    replace_exact(
        &mut source,
        SYMBOL,
        PARENT_SYMBOL,
        2,
        "restored target symbol",
    )?;
    Ok(source)
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn transform_source(a_ldmatrix_parent: &str) -> Result<String, String> {
    let mut source = a_ldmatrix_parent.to_owned();
    replace_exact(&mut source, B_LOAD, B_LDMATRIX_LOAD, 1, "B fragment load")?;
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
            "compact A+B ldmatrix {label} count changed: expected {expected}, observed {count}"
        ));
    }
    *source = source.replace(from, to);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x2_register_coordinates_equal_current_scalar_b_fragments() {
        for warp in 0..8 {
            let warp_n = (warp & 1) * 32;
            for atom in 0..4 {
                for k8 in [0, 8, 16, 24] {
                    for lane in 0..32 {
                        for register in 0..2 {
                            let address_lane = register * 8 + (lane >> 2);
                            let (column, reduction) =
                                ldmatrix_b_address(warp_n, atom, k8, address_lane);
                            assert_eq!(
                                (column, reduction + (lane & 3)),
                                scalar_b_coordinate(warp_n, atom, k8, lane, register)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_b_addresses_are_aligned_and_in_bounds() {
        for warp in 0..8 {
            let warp_n = (warp & 1) * 32;
            for atom in 0..4 {
                for k8 in [0, 8, 16, 24] {
                    for lane in 0..32 {
                        let (column, reduction) = ldmatrix_b_address(warp_n, atom, k8, lane);
                        assert!(column < 64);
                        assert!(reduction + 3 < 32);
                        assert_eq!(compact_index(column, reduction) * 4 % 16, 0);
                        for word in 0..4 {
                            assert_eq!(
                                compact_index(column, reduction) + word,
                                compact_index(column, reduction + word)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn source_adapter_changes_only_target_b_load_and_symbol() {
        let a_load = "A-only ldmatrix load and RNA/K8/MMA body\n";
        let parent = format!(
            "prefix\n{a_load}{B_LOAD}GEMM_BI_TF32_DEFINE_KERNEL({PARENT_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)\nTF32_ASSERT_KERNEL_SIGNATURE({PARENT_SYMBOL});\nsuffix\n"
        );
        let candidate = transform_source(&parent).unwrap();
        assert_eq!(candidate.matches(SYMBOL).count(), 2);
        assert!(candidate.contains(a_load));
        assert!(candidate.contains("ldmatrix.sync.aligned.m8n8.x2.shared.b16"));
        assert!(candidate.contains("gemm_bi_tf32_rna(__uint_as_float(raw0))"));
        assert!(candidate.contains("gemm_bi_tf32_rna(__uint_as_float(raw1))"));
        let restored = restore_parent(&candidate).unwrap();
        assert_eq!(restored, parent);
    }

    #[test]
    fn source_adapter_rejects_missing_or_duplicated_anchors() {
        let parent = format!("{B_LOAD}{PARENT_SYMBOL}{PARENT_SYMBOL}");
        assert!(transform_source(&parent.replacen(B_LOAD, "", 1)).is_err());
        assert!(transform_source(&format!("{parent}{B_LOAD}")).is_err());
        assert!(transform_source(&parent.replacen(PARENT_SYMBOL, "missing", 1)).is_err());
        assert!(transform_source(&format!("{parent}{PARENT_SYMBOL}")).is_err());
    }

    #[test]
    fn candidate_source_rejects_a_parent_with_matching_seams_but_changed_body() {
        let changed_parent = format!(
            "unexpected-prefix\n{B_LOAD}GEMM_BI_TF32_DEFINE_KERNEL({PARENT_SYMBOL}, SgbTf32Nt, 128, 64, 2, 256, 1)\nTF32_ASSERT_KERNEL_SIGNATURE({PARENT_SYMBOL});\nunexpected-suffix\n"
        );
        assert!(candidate_source(&changed_parent).is_err());
    }
}
