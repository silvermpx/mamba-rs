pub const M16N16_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m16n16_f64fold_v1";
pub const M16N16_T128_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m16n16_t128_f64fold_v1";
pub const M8N32_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m8n32_f64fold_v1";
pub const M8N16_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m8n16_f64fold_v1";

const ORIGINAL_NAMESPACE: &str = "namespace GemmBiTnUnderfillDirect {";
const ADAPTED_NAMESPACE: &str = "namespace GemmBiTnAdaD128Direct {";
const USING_BOUNDARY: &str = "using M32N32 = Kernel<32, 32, 128, 0, false, false>;";

pub fn compose_warp_source(original: &str) -> Result<String, String> {
    let mut source = compose_source(original)?;
    const END: &str = "} // namespace GemmBiTnAdaD128Direct";
    replace_once(
        &mut source,
        END,
        &format!(
            r#"using M16N16T128 = Kernel<16, 16, 128, 0, false, true>;
static_assert(M16N16T128::OutputsPerThread == 2, "thread ownership");
static_assert(M16N16T128::SharedBytes == 4096, "shared bytes");
static_assert(M16N16T128::RowTiles * M16N16T128::ColumnTiles == 256, "grid");
{END}"#
        ),
        "namespace end",
    )?;
    let mut wrapper = export(M16N16_T128_SYMBOL, "M16N16T128");
    replace_once(
        &mut wrapper,
        "__launch_bounds__(64, 4)",
        "__launch_bounds__(128, 4)",
        "warp launch bounds",
    )?;
    source.push_str(&wrapper);
    Ok(source)
}

pub fn compose_source(original: &str) -> Result<String, String> {
    require_once(original, ORIGINAL_NAMESPACE, "source namespace")?;
    require_once(original, USING_BOUNDARY, "first using alias")?;
    let boundary = original
        .find(USING_BOUNDARY)
        .ok_or_else(|| "first using alias disappeared".to_string())?;
    let mut source = original[..boundary].to_owned();
    for (label, from, to) in [
        ("namespace", ORIGINAL_NAMESPACE, ADAPTED_NAMESPACE),
        (
            "M reduction",
            "constexpr int MRed = 256;",
            "constexpr int MRed = 1024;",
        ),
        (
            "K output",
            "constexpr int KOut = 512;",
            "constexpr int KOut = 128;",
        ),
        (
            "N output",
            "constexpr int N = 384;",
            "constexpr int N = 512;",
        ),
        (
            "chunk count assertion",
            "static_assert(Chunks == 16, \"the Split-M reduction tree changed\");",
            "static_assert(Chunks == 64, \"the Split-M reduction tree changed\");",
        ),
    ] {
        replace_once(&mut source, from, to, label)?;
    }

    source.push_str(
        r#"using M16N16 = Kernel<16, 16, 64, 0, false, true>;
using M8N32 = Kernel<8, 32, 64, 0, false, true>;

static_assert(M16N16::OutputsPerThread == 4,
              "M16N16 thread ownership changed");
static_assert(M16N16::SharedBytes == 4096,
              "M16N16 shared-memory contract changed");
static_assert(M16N16::Stage == 512,
              "M16N16 stage extent changed");
static_assert(M16N16::RowTiles * M16N16::ColumnTiles == 256,
              "M16N16 flat-grid contract changed");
static_assert(M8N32::OutputsPerThread == 4,
              "M8N32 thread ownership changed");
static_assert(M8N32::SharedBytes == 5120,
              "M8N32 shared-memory contract changed");
static_assert(M8N32::Stage == 640,
              "M8N32 stage extent changed");
static_assert(M8N32::RowTiles * M8N32::ColumnTiles == 256,
              "M8N32 flat-grid contract changed");

} // namespace GemmBiTnAdaD128Direct

"#,
    );
    source.push_str(&export(M16N16_SYMBOL, "M16N16"));
    source.push_str(&export(M8N32_SYMBOL, "M8N32"));
    Ok(source)
}

pub fn compose_wave_source(original: &str) -> Result<String, String> {
    const NAMESPACE_END: &str = "} // namespace GemmBiTnAdaD128Direct\n";
    const M8N16_CONTRACT: &str = concat!(
        "using M8N16 = Kernel<8, 16, 64, 0, false, true>;\n",
        "static_assert(M8N16::OutputsPerThread == 2,\n",
        "              \"M8N16 thread ownership changed\");\n",
        "static_assert(M8N16::SharedBytes == 3072,\n",
        "              \"M8N16 shared-memory contract changed\");\n",
        "static_assert(M8N16::Stage == 384,\n",
        "              \"M8N16 stage extent changed\");\n",
        "static_assert(M8N16::RowTiles * M8N16::ColumnTiles == 512,\n",
        "              \"M8N16 flat-grid contract changed\");\n\n"
    );
    let mut source = compose_source(original)?;
    replace_once(
        &mut source,
        NAMESPACE_END,
        &format!("{M8N16_CONTRACT}{NAMESPACE_END}"),
        "adapted namespace end",
    )?;
    source.push_str(&export(M8N16_SYMBOL, "M8N16"));
    Ok(source)
}

fn export(symbol: &str, alias: &str) -> String {
    format!(
        r#"extern "C" __global__ __launch_bounds__(64, 4)
void {symbol}(
    float* output,
    const float* a,
    const float* b,
    float alpha,
    int m,
    int k,
    int n
) {{
    GemmBiTnAdaD128Direct::{alias}::run(output, a, b, alpha, m, k, n);
}}

"#,
    )
}

fn require_once(source: &str, needle: &str, label: &str) -> Result<(), String> {
    let count = source.matches(needle).count();
    if count != 1 {
        return Err(format!(
            "{label} count changed: expected 1, observed {count}"
        ));
    }
    Ok(())
}

fn replace_once(source: &mut String, from: &str, to: &str, label: &str) -> Result<(), String> {
    require_once(source, from, label)?;
    *source = source.replacen(from, to, 1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGINAL: &str = include_str!("../gemm_bi_scalar_tn_underfill_direct_experiment.cu");

    #[test]
    fn more_warps_preserves_tile_reuse_and_doubles_compute_threads() {
        let source = compose_warp_source(ORIGINAL).unwrap();
        assert!(source.contains("using M16N16T128 = Kernel<16, 16, 128, 0, false, true>;"));
        assert!(source.contains("M16N16T128::OutputsPerThread == 2"));
        assert!(source.contains("M16N16T128::SharedBytes == 4096"));
        assert!(source.contains("M16N16T128::RowTiles * M16N16T128::ColumnTiles == 256"));
        assert!(source.contains(&format!(
            "__launch_bounds__(128, 4)\nvoid {M16N16_T128_SYMBOL}("
        )));
        assert_eq!(source.matches("partial[owned] = __fmaf_rn(").count(), 1);
        assert_eq!(
            source
                .matches("sums[owned] = __dadd_rn(sums[owned], value);")
                .count(),
            1
        );
    }

    #[test]
    fn adapter_changes_only_the_exact_d128_contract_and_two_exports() {
        let source = compose_source(ORIGINAL).unwrap();
        assert!(source.contains("namespace GemmBiTnAdaD128Direct {"));
        assert!(source.contains("constexpr int MRed = 1024;"));
        assert!(source.contains("constexpr int KOut = 128;"));
        assert!(source.contains("constexpr int N = 512;"));
        assert!(source.contains("static_assert(Chunks == 64"));
        assert_eq!(source.matches("extern \"C\" __global__").count(), 2);
        assert_eq!(source.matches(M16N16_SYMBOL).count(), 1);
        assert_eq!(source.matches(M8N32_SYMBOL).count(), 1);
        assert!(!source.contains("gemm_bi_tn_underfill_"));
    }

    #[test]
    fn adapter_preserves_the_ascending_f32_then_f64_rounding_sequence() {
        let source = compose_source(ORIGINAL).unwrap();
        for required in [
            "float partial[OutputsPerThread] = {0.0f};",
            "partial[owned] = __fmaf_rn(",
            "if (chunk == 0)",
            "sums[owned] = __dadd_rn(sums[owned], value);",
            "double scaled = __dmul_rn((double)alpha, sums[owned]);",
            "output[index] = __fadd_rn(",
        ] {
            assert!(source.contains(required), "missing {required:?}");
        }
        assert!(
            source
                .find("for (int chunk = 0; chunk < Chunks; ++chunk)")
                .unwrap()
                < source.find("__dadd_rn(sums[owned], value)").unwrap()
        );
    }

    #[test]
    fn adapter_fails_closed_when_an_exact_boundary_drifts() {
        for changed in [
            ORIGINAL.replacen("constexpr int MRed = 256;", "constexpr int MRed = 257;", 1),
            ORIGINAL.replacen("using M32N32 =", "using Drifted =", 1),
            format!("{ORIGINAL}\n{ORIGINAL}"),
        ] {
            assert!(compose_source(&changed).is_err());
        }
    }

    #[test]
    fn wave_adapter_preserves_the_original_adapter_and_adds_only_m8n16() {
        let original = compose_source(ORIGINAL).unwrap();
        let wave = compose_wave_source(ORIGINAL).unwrap();
        assert_eq!(wave.matches(M16N16_SYMBOL).count(), 1);
        assert_eq!(wave.matches(M8N32_SYMBOL).count(), 1);
        assert_eq!(wave.matches(M8N16_SYMBOL).count(), 1);
        assert_eq!(wave.matches("extern \"C\" __global__").count(), 3);
        assert!(wave.contains("using M8N16 = Kernel<8, 16, 64, 0, false, true>;"));
        assert!(wave.contains("M8N16::OutputsPerThread == 2"));
        assert!(wave.contains("M8N16::SharedBytes == 3072"));
        assert!(wave.contains("M8N16::Stage == 384"));
        assert!(wave.contains("M8N16::RowTiles * M8N16::ColumnTiles == 512"));
        assert_eq!(
            wave.replacen(
                concat!(
                    "using M8N16 = Kernel<8, 16, 64, 0, false, true>;\n",
                    "static_assert(M8N16::OutputsPerThread == 2,\n",
                    "              \"M8N16 thread ownership changed\");\n",
                    "static_assert(M8N16::SharedBytes == 3072,\n",
                    "              \"M8N16 shared-memory contract changed\");\n",
                    "static_assert(M8N16::Stage == 384,\n",
                    "              \"M8N16 stage extent changed\");\n",
                    "static_assert(M8N16::RowTiles * M8N16::ColumnTiles == 512,\n",
                    "              \"M8N16 flat-grid contract changed\");\n\n"
                ),
                "",
                1,
            )
            .replacen(&export(M8N16_SYMBOL, "M8N16"), "", 1),
            original,
        );
    }
}
