pub const M16N16_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m16n16_f64fold_v1";
pub const M16N16_T128_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m16n16_t128_f64fold_v1";
pub const M8N32_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m8n32_f64fold_v1";
pub const M8N16_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m8n16_f64fold_v1";
pub const OUT_M16N16_SYMBOL: &str = "gemm_bi_tn_ada_d128out_direct_m16n16_f64fold_v1";
pub const OUT_M8N16_SYMBOL: &str = "gemm_bi_tn_ada_d128out_direct_m8n16_f64fold_v1";
pub const IN_FOLDPIPE_SYMBOL: &str = "gemm_bi_tn_ada_d128_direct_m16n16_foldpipe_v1";
pub const OUT_FOLDPIPE_SYMBOL: &str = "gemm_bi_tn_ada_d128out_direct_m8n16_foldpipe_v1";

const ORIGINAL_NAMESPACE: &str = "namespace GemmBiTnUnderfillDirect {";
const ADAPTED_NAMESPACE: &str = "namespace GemmBiTnAdaD128Direct {";
const FOLDPIPE_NAMESPACE: &str = "namespace GemmBiTnAdaD128FoldPipe {";
const USING_BOUNDARY: &str = "using M32N32 = Kernel<32, 32, 128, 0, false, false>;";
const CHUNK_LOOP_HEADER: &str = "for (int chunk = 0; chunk < Chunks; ++chunk)";

pub fn compose_foldpipe_source(original: &str) -> Result<String, String> {
    let mut source = compose_source(original)?;
    source.push_str(&compose_foldpipe_kernel(
        original,
        128,
        512,
        "M16N16FoldPipe",
        "Kernel<16, 16, 64, 0, false, true>",
        4,
        4096,
        512,
        256,
        IN_FOLDPIPE_SYMBOL,
    )?);
    Ok(source)
}

pub fn compose_out_foldpipe_source(original: &str) -> Result<String, String> {
    let mut source = compose_out_source(original)?;
    source.push_str(&compose_foldpipe_kernel(
        original,
        256,
        128,
        "M8N16FoldPipe",
        "Kernel<8, 16, 64, 0, false, true>",
        2,
        3072,
        384,
        256,
        OUT_FOLDPIPE_SYMBOL,
    )?);
    Ok(source)
}

#[allow(clippy::too_many_arguments)]
fn compose_foldpipe_kernel(
    original: &str,
    k_out: usize,
    n: usize,
    alias: &str,
    kernel: &str,
    outputs_per_thread: usize,
    shared_bytes: usize,
    stage: usize,
    grid: usize,
    symbol: &str,
) -> Result<String, String> {
    require_once(original, ORIGINAL_NAMESPACE, "source namespace")?;
    require_once(original, USING_BOUNDARY, "first using alias")?;
    require_once(original, CHUNK_LOOP_HEADER, "chunk loop header")?;
    let start = original
        .find(ORIGINAL_NAMESPACE)
        .ok_or_else(|| "source namespace disappeared".to_string())?;
    let boundary = original
        .find(USING_BOUNDARY)
        .ok_or_else(|| "first using alias disappeared".to_string())?;
    let mut source = original[start..boundary].to_owned();
    for (label, from, to) in [
        ("namespace", ORIGINAL_NAMESPACE, FOLDPIPE_NAMESPACE),
        (
            "M reduction",
            "constexpr int MRed = 256;",
            "constexpr int MRed = 1024;",
        ),
        (
            "K output",
            "constexpr int KOut = 512;",
            &format!("constexpr int KOut = {k_out};"),
        ),
        (
            "N output",
            "constexpr int N = 384;",
            &format!("constexpr int N = {n};"),
        ),
        (
            "chunk count assertion",
            "static_assert(Chunks == 16, \"the Split-M reduction tree changed\");",
            "static_assert(Chunks == 64, \"the Split-M reduction tree changed\");",
        ),
    ] {
        replace_once(&mut source, from, to, label)?;
    }

    const SCALAR_CHUNK_BODY: &str = r#"            const float* a_tile = shared + read_stage * Stage;
            const float* b_tile = a_tile + AStage;
            float partial[OutputsPerThread] = {0.0f};

            // Each chunk starts from positive zero, exactly like the old
            // Split-M partial kernel.
            #pragma unroll
            for (int reduction = 0; reduction < BK; ++reduction) {
                float a_value = a_tile[reduction * AStride + local_row];
                #pragma unroll
                for (int owned = 0; owned < OutputsPerThread; ++owned) {
                    partial[owned] = __fmaf_rn(
                        a_value,
                        b_tile[reduction * TileN + local_column_base + owned],
                        partial[owned]);
                }
            }

            // Chunk zero seeds the double accumulator directly. Later chunks
            // follow the reducer's ascending fc=1..15 add order.
            #pragma unroll
            for (int owned = 0; owned < OutputsPerThread; ++owned) {
                double value = (double)partial[owned];
                if (chunk == 0) {
                    sums[owned] = value;
                } else {
                    sums[owned] = __dadd_rn(sums[owned], value);
                }
            }
            read_stage ^= 1;"#;
    const PIPELINED_CHUNK_BODY: &str = r#"            const float* a_tile = shared + read_stage * Stage;
            const float* b_tile = a_tile + AStage;
            // Fold the preceding chunk before computing the current chunk.
            // Chunk zero still seeds the double accumulator directly, and
            // every later chunk retains the original ascending add order.
            if (chunk > 0) {
                #pragma unroll
                for (int owned = 0; owned < OutputsPerThread; ++owned) {
                    double value = (double)previous[owned];
                    if (chunk == 1) {
                        sums[owned] = value;
                    } else {
                        sums[owned] = __dadd_rn(sums[owned], value);
                    }
                }
            }

            #pragma unroll
            for (int owned = 0; owned < OutputsPerThread; ++owned) {
                current[owned] = 0.0f;
            }

            // Each chunk starts from positive zero, exactly like the old
            // Split-M partial kernel.
            #pragma unroll
            for (int reduction = 0; reduction < BK; ++reduction) {
                float a_value = a_tile[reduction * AStride + local_row];
                #pragma unroll
                for (int owned = 0; owned < OutputsPerThread; ++owned) {
                    current[owned] = __fmaf_rn(
                        a_value,
                        b_tile[reduction * TileN + local_column_base + owned],
                        current[owned]);
                }
            }

            #pragma unroll
            for (int owned = 0; owned < OutputsPerThread; ++owned) {
                previous[owned] = current[owned];
            }
            read_stage ^= 1;"#;
    replace_once(
        &mut source,
        "        int read_stage = 0;",
        concat!(
            "        int read_stage = 0;\n",
            "        float previous[OutputsPerThread];\n",
            "        float current[OutputsPerThread];"
        ),
        "partial register banks",
    )?;
    replace_once(
        &mut source,
        SCALAR_CHUNK_BODY,
        PIPELINED_CHUNK_BODY,
        "chunk compute and fold body",
    )?;
    const AFTER_CHUNK_LOOP: &str = r#"

        #pragma unroll
        for (int owned = 0; owned < OutputsPerThread; ++owned) {
            int row = output_row_base + local_row;"#;
    let final_fold = r#"

        #pragma unroll
        for (int owned = 0; owned < OutputsPerThread; ++owned) {
            double value = (double)previous[owned];
            sums[owned] = __dadd_rn(sums[owned], value);
        }

        #pragma unroll
        for (int owned = 0; owned < OutputsPerThread; ++owned) {
            int row = output_row_base + local_row;"#;
    replace_once(
        &mut source,
        AFTER_CHUNK_LOOP,
        final_fold,
        "final chunk drain",
    )?;

    source.push_str(&format!(
        r#"using {alias} = {kernel};
static_assert({alias}::OutputsPerThread == {outputs_per_thread},
              "fold-pipeline thread ownership changed");
static_assert({alias}::SharedBytes == {shared_bytes},
              "fold-pipeline shared-memory contract changed");
static_assert({alias}::Stage == {stage},
              "fold-pipeline stage extent changed");
static_assert({alias}::RowTiles * {alias}::ColumnTiles == {grid},
              "fold-pipeline flat-grid contract changed");

}} // namespace GemmBiTnAdaD128FoldPipe

"#,
    ));
    source.push_str(&export_in_namespace(
        FOLDPIPE_NAMESPACE
            .strip_prefix("namespace ")
            .and_then(|name| name.strip_suffix(" {"))
            .ok_or_else(|| "fold-pipeline namespace is malformed".to_string())?,
        symbol,
        alias,
    ));
    Ok(source)
}

pub fn compose_out_source(original: &str) -> Result<String, String> {
    let mut source = compose_wave_source(original)?;
    for (from, to) in [
        ("constexpr int KOut = 128;", "constexpr int KOut = 256;"),
        ("constexpr int N = 512;", "constexpr int N = 128;"),
        (
            "M16N16::RowTiles * M16N16::ColumnTiles == 256",
            "M16N16::RowTiles * M16N16::ColumnTiles == 128",
        ),
        (
            "M8N32::RowTiles * M8N32::ColumnTiles == 256",
            "M8N32::RowTiles * M8N32::ColumnTiles == 128",
        ),
        (
            "M8N16::RowTiles * M8N16::ColumnTiles == 512",
            "M8N16::RowTiles * M8N16::ColumnTiles == 256",
        ),
        (M16N16_SYMBOL, OUT_M16N16_SYMBOL),
        (M8N16_SYMBOL, OUT_M8N16_SYMBOL),
        (
            M8N32_SYMBOL,
            "gemm_bi_tn_ada_d128out_direct_m8n32_f64fold_v1",
        ),
    ] {
        replace_once(&mut source, from, to, "d128-out specialization")?;
    }
    Ok(source)
}

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
    export_in_namespace("GemmBiTnAdaD128Direct", symbol, alias)
}

fn export_in_namespace(namespace: &str, symbol: &str, alias: &str) -> String {
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
    {namespace}::{alias}::run(output, a, b, alpha, m, k, n);
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
    fn out_projection_changes_geometry_but_not_splitm64_arithmetic() {
        let source = compose_out_source(ORIGINAL).unwrap();
        assert!(source.contains("constexpr int MRed = 1024;"));
        assert!(source.contains("constexpr int KOut = 256;"));
        assert!(source.contains("constexpr int N = 128;"));
        assert!(source.contains("static_assert(Chunks == 64"));
        assert!(source.contains("M16N16::RowTiles * M16N16::ColumnTiles == 128"));
        assert!(source.contains("M8N16::RowTiles * M8N16::ColumnTiles == 256"));
        assert!(source.contains(OUT_M16N16_SYMBOL));
        assert!(source.contains(OUT_M8N16_SYMBOL));
        assert!(!source.contains(M16N16_SYMBOL));
        assert!(!source.contains(M8N16_SYMBOL));
        assert_eq!(source.matches("partial[owned] = __fmaf_rn(").count(), 1);
        assert_eq!(
            source
                .matches("sums[owned] = __dadd_rn(sums[owned], value);")
                .count(),
            1
        );
        for (tm, tn, threads) in [(16, 16, 64), (8, 16, 64)] {
            assert_eq!((256 / tm) * (128 / tn), if tm == 16 { 128 } else { 256 });
            let per_thread = tm * tn / threads;
            let column_groups = tn / per_thread;
            let mut owners = vec![0; tm * tn];
            for thread in 0..threads {
                for owned in 0..per_thread {
                    owners[(thread / column_groups) * tn
                        + (thread % column_groups) * per_thread
                        + owned] += 1;
                }
            }
            assert!(owners.iter().all(|count| *count == 1));
        }
    }

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

    #[test]
    fn fold_pipeline_folds_each_chunk_once_in_ascending_order() {
        let mut computed = Vec::new();
        let mut folded = Vec::new();
        for chunk in 0usize..64 {
            if chunk > 0 {
                folded.push(chunk - 1);
            }
            computed.push((chunk, (0usize..16).collect::<Vec<_>>()));
        }
        folded.push(63);
        assert_eq!(folded, (0usize..64).collect::<Vec<_>>());
        assert!(
            computed
                .iter()
                .all(|(_, reductions)| reductions == &(0usize..16).collect::<Vec<_>>())
        );

        let source = compose_foldpipe_source(ORIGINAL).unwrap();
        let pipe = source
            .rfind(FOLDPIPE_NAMESPACE)
            .map(|start| &source[start..])
            .unwrap();
        assert!(source.contains("float previous[OutputsPerThread];"));
        assert!(source.contains("float current[OutputsPerThread];"));
        assert!(!source.contains("partial[2][OutputsPerThread]"));
        assert!(!source.contains("current_bank"));
        assert!(!source.contains("previous_bank"));
        assert_eq!(source.matches("current[owned] = 0.0f;").count(), 1);
        assert_eq!(source.matches("current[owned] = __fmaf_rn(").count(), 1);
        assert_eq!(
            source.matches("previous[owned] = current[owned];").count(),
            1
        );
        assert!(source.contains("for (int reduction = 0; reduction < BK; ++reduction)"));
        let fold_previous = pipe
            .find("double value = (double)previous[owned];")
            .unwrap();
        let seed_chunk_zero = pipe.find("if (chunk == 1)").unwrap();
        let zero_current = pipe.find("current[owned] = 0.0f;").unwrap();
        let fma_current = pipe.find("current[owned] = __fmaf_rn(").unwrap();
        let rotate = pipe.find("previous[owned] = current[owned];").unwrap();
        let final_fold = pipe
            .rfind("double value = (double)previous[owned];")
            .unwrap();
        assert!(fold_previous < seed_chunk_zero);
        assert!(seed_chunk_zero < zero_current);
        assert!(zero_current < fma_current);
        assert!(fma_current < rotate);
        assert!(rotate < final_fold);
    }

    #[test]
    fn fold_pipeline_adds_distinct_in_and_out_exports_without_removing_controls() {
        let retained_input = compose_source(ORIGINAL).unwrap();
        let input = compose_foldpipe_source(ORIGINAL).unwrap();
        assert!(input.starts_with(&retained_input));
        assert_eq!(input.matches(M16N16_SYMBOL).count(), 1);
        assert_eq!(input.matches(M8N32_SYMBOL).count(), 1);
        assert_eq!(input.matches(IN_FOLDPIPE_SYMBOL).count(), 1);
        assert!(!input.contains(OUT_FOLDPIPE_SYMBOL));

        let retained_output = compose_out_source(ORIGINAL).unwrap();
        let output = compose_out_foldpipe_source(ORIGINAL).unwrap();
        assert!(output.starts_with(&retained_output));
        assert_eq!(output.matches(OUT_M16N16_SYMBOL).count(), 1);
        assert_eq!(output.matches(OUT_M8N16_SYMBOL).count(), 1);
        assert_eq!(output.matches(OUT_FOLDPIPE_SYMBOL).count(), 1);
        assert!(!output.contains(IN_FOLDPIPE_SYMBOL));
    }

    #[test]
    fn fold_pipeline_fails_closed_when_the_chunk_loop_drifts() {
        for changed in [
            ORIGINAL.replacen(
                "for (int chunk = 0; chunk < Chunks; ++chunk)",
                "for (int chunk = 1; chunk < Chunks; ++chunk)",
                1,
            ),
            ORIGINAL.replacen(
                "partial[owned] = __fmaf_rn(",
                "partial[owned] = __fmaf_rz(",
                1,
            ),
            ORIGINAL.replacen(
                "sums[owned] = __dadd_rn(sums[owned], value);",
                "sums[owned] += value;",
                1,
            ),
        ] {
            assert!(compose_foldpipe_source(&changed).is_err());
            assert!(compose_out_foldpipe_source(&changed).is_err());
        }
    }
}
