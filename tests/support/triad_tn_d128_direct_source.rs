pub const M16N16_SYMBOL: &str = "tn_ada_d128_direct_m16n16_f64fold";
pub const M8N32_SYMBOL: &str = "tn_ada_d128_direct_m8n32_f64fold";
pub const M8N16_SYMBOL: &str = "tn_ada_d128_direct_m8n16_f64fold";
pub const OUT_M16N16_SYMBOL: &str = "tn_ada_d128out_direct_m16n16_f64fold";
pub const OUT_M8N16_SYMBOL: &str = "tn_ada_d128out_direct_m8n16_f64fold";

const ORIGINAL_NAMESPACE: &str = "namespace GemmBiTnUnderfillDirect {";
const ADAPTED_NAMESPACE: &str = "namespace GemmBiTnAdaD128Direct {";
const USING_BOUNDARY: &str = "using M32N32 = Kernel<32, 32, 128, 0, false, false>;";

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
        (M8N32_SYMBOL, "tn_ada_d128out_direct_m8n32_f64fold"),
    ] {
        replace_once(&mut source, from, to, "d128-out specialization")?;
    }
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
