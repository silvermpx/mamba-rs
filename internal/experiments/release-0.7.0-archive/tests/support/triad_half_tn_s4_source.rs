pub const BF16_SYMBOL: &str = "gemm_bi_tn_test_tc64_bk32_s4_bf16";
pub const F16_SYMBOL: &str = "gemm_bi_tn_test_tc64_bk32_s4_f16";

const START: &str = "#define GEMM_BI_TN_RECT_BM 128";
const END: &str = "#undef GEMM_BI_TN_RECT_BM";

fn require_exact(source: &str, needle: &str, expected: usize, label: &str) -> Result<(), String> {
    let observed = source.matches(needle).count();
    if observed != expected {
        return Err(format!(
            "{label} source anchor {needle:?}: expected {expected}, observed {observed}"
        ));
    }
    Ok(())
}

fn replace_exact(
    source: &mut String,
    before: &str,
    after: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    require_exact(source, before, expected, label)?;
    *source = source.replace(before, after);
    Ok(())
}

/// Derives an isolated half-TN M64xN64/BK32/S4 candidate section from the
/// production rectangular kernel. The caller supplies the common CUDA
/// preamble (types, conversion helpers, and cp.async helpers) when compiling.
pub fn compose_source(production: &str) -> Result<String, String> {
    require_exact(production, START, 1, "candidate section start")?;
    require_exact(production, END, 1, "candidate section end")?;
    let start = production.find(START).unwrap();
    let end = production[start..].find(END).unwrap() + start + END.len();
    let mut source = production[start..end].to_owned();

    for (label, before, after) in [
        (
            "M64 tile",
            "#define GEMM_BI_TN_RECT_BM 128",
            "#define GEMM_BI_TN_RECT_BM 64",
        ),
        (
            "four shared stages",
            "#define GEMM_BI_TN_RECT_STAGES 3",
            "#define GEMM_BI_TN_RECT_STAGES 4",
        ),
        (
            "four compute warps",
            "#define GEMM_BI_TN_RECT_THREADS 256",
            "#define GEMM_BI_TN_RECT_THREADS 128",
        ),
        (
            "launch bounds",
            "__launch_bounds__(256, 2)",
            "__launch_bounds__(128, 1)",
        ),
        (
            "candidate export",
            "void gemm_bi_tn_tc128x64_##SUFFIX",
            "void gemm_bi_tn_test_tc64_bk32_s4_##SUFFIX",
        ),
        (
            "three-group prefill",
            "for (int p = 0; p < 2; p++)",
            "for (int p = 0; p < 3; p++)",
        ),
        (
            "two pending groups",
            "cp.async.wait_group 1",
            "cp.async.wait_group 2",
        ),
        (
            "four-stage refill distance",
            "int next = kt + 2;",
            "int next = kt + 3;",
        ),
    ] {
        replace_exact(&mut source, before, after, 1, label)?;
    }

    require_exact(
        &source,
        "int macro64_tiles = (M_red - 1) / 64 + 1;",
        1,
        "padded macro64 schedule",
    )?;
    require_exact(
        &source,
        "int k32_tiles = 2 * macro64_tiles;",
        1,
        "paired BK32 schedule",
    )?;
    require_exact(
        &source,
        "GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(bf16, __nv_bfloat16, from_f_bf16, \"bf16\")",
        1,
        "BF16 emission",
    )?;
    require_exact(
        &source,
        "GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(f16, __half, from_f_f16, \"f16\")",
        1,
        "F16 emission",
    )?;
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCTION: &str = include_str!("../../kernels/gemm_bi_triad/sm80.cu");

    fn old_k16_sequence(reduction: usize) -> Vec<usize> {
        (0..reduction.div_ceil(64))
            .flat_map(|tile| (0..4).map(move |step| tile * 64 + step * 16))
            .collect()
    }

    fn new_k16_sequence(reduction: usize) -> Vec<usize> {
        let macro64_tiles = reduction.div_ceil(64);
        (0..2 * macro64_tiles)
            .flat_map(|tile| (0..2).map(move |step| tile * 32 + step * 16))
            .collect()
    }

    #[test]
    fn four_warps_own_every_m64n64_output_once() {
        let mut owners = vec![0_u8; 64 * 64];
        for warp in 0..4 {
            let warp_m = (warp >> 1) * 32;
            let warp_n = (warp & 1) * 32;
            for row in warp_m..warp_m + 32 {
                for column in warp_n..warp_n + 32 {
                    owners[row * 64 + column] += 1;
                }
            }
        }
        assert!(owners.into_iter().all(|count| count == 1));

        let source = compose_source(PRODUCTION).unwrap();
        assert!(source.contains("int warpM = (warp >> 1) * 32;"));
        assert!(source.contains("int warpN = (warp & 1) * 32;"));
        assert!(source.contains("float acc[2][4][4];"));
    }

    #[test]
    fn four_stage_ring_keeps_current_and_pending_slots_distinct() {
        for tiles in 1usize..=131 {
            for current in 0..tiles {
                let read = current % 4;
                let pending = [current + 1, current + 2]
                    .into_iter()
                    .filter(|&tile| tile < tiles)
                    .map(|tile| tile % 4)
                    .collect::<Vec<_>>();
                let next = current + 3;
                if next < tiles {
                    let write = next % 4;
                    assert_ne!(write, read, "tiles={tiles} current={current}");
                    assert!(!pending.contains(&write), "tiles={tiles} current={current}");
                }
            }
        }

        let source = compose_source(PRODUCTION).unwrap();
        assert!(source.contains("for (int p = 0; p < 3; p++)"));
        assert!(source.contains("cp.async.wait_group 2"));
        assert!(source.contains("int next = kt + 3;"));
        assert!(source.contains("kt % GEMM_BI_TN_RECT_STAGES"));
    }

    #[test]
    fn bk32_s4_keeps_tc64_padding_and_k16_association() {
        for reduction in [
            0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2_048, 4_621,
        ] {
            assert_eq!(new_k16_sequence(reduction), old_k16_sequence(reduction));
        }

        let source = compose_source(PRODUCTION).unwrap();
        assert!(source.contains("#define GEMM_BI_TN_RECT_BM 64"));
        assert!(source.contains("#define GEMM_BI_TN_RECT_BN 64"));
        assert!(source.contains("#define GEMM_BI_TN_RECT_BK 32"));
        assert!(source.contains("#define GEMM_BI_TN_RECT_STAGES 4"));
        assert!(source.contains("int macro64_tiles = (M_red - 1) / 64 + 1;"));
        assert!(source.contains("int k32_tiles = 2 * macro64_tiles;"));
        assert!(source.contains("mma.sync.aligned.m16n8k16.row.col.f32."));
        assert!(source.contains("C[(long long)gr * N + gc] += alpha * acc[fm][fn][e];"));
    }

    #[test]
    fn adapter_emits_only_the_two_candidate_exports_and_fails_closed() {
        let source = compose_source(PRODUCTION).unwrap();
        assert_eq!(source.matches("extern \"C\" __global__").count(), 1);
        assert!(source.contains("void gemm_bi_tn_test_tc64_bk32_s4_##SUFFIX"));
        assert_eq!(
            source
                .matches("GEMM_BI_DEFINE_GEMM_BI_TN_TC128X64(")
                .count(),
            3
        );
        assert!(!source.contains("gemm_bi_tn_tc128x64_##SUFFIX"));
        assert!(!source.contains("gemm_bi_nn_"));
        assert!(!source.contains("gemm_bi_nt_"));

        let missing = PRODUCTION.replacen("#define GEMM_BI_TN_RECT_BM 128", "", 1);
        assert!(compose_source(&missing).is_err());
        let duplicate = format!("{PRODUCTION}\n#define GEMM_BI_TN_RECT_BM 128\n");
        assert!(compose_source(&duplicate).is_err());
    }
}
