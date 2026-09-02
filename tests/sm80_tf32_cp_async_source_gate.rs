use std::{fs, path::Path};

fn sm80_source() -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/gemm_bi_triad/sm80.cu"))
        .expect("read SM80 triad source")
}

#[test]
fn partial_tf32_vectors_use_scalar_width_cp_async_at_all_alignments() {
    let source = sm80_source();
    let start = source
        .find("void gemm_bi_tf32_cp_async_zfill(")
        .expect("TF32 cp.async selector");
    let end = source[start..]
        .find("template <SgbTf32Op Op, int BM, int BN, int Stages,")
        .map(|offset| start + offset)
        .expect("TF32 staging template after cp.async selector");
    let selector = &source[start..end];

    assert!(
        selector.contains("if constexpr (!Narrow)"),
        "wide row-stride routes must keep their proven 16-byte fast path"
    );
    let wide = selector
        .split("} else if (valid_bytes == 16 && gemm_bi_is_aligned_16(global_src))")
        .next()
        .expect("wide TF32 copy selector branch");
    assert!(
        wide.contains("if (valid_bytes == 16)"),
        "wide row strides do not prove that a partial logical tail has 16 readable bytes"
    );
    assert!(
        wide.contains("gemm_bi_tf32_cp_async_4x4_zfill("),
        "wide row-stride partial tails must use independently zero-filled 4-byte copies"
    );
    assert!(
        selector.contains("else if (valid_bytes == 16 && gemm_bi_is_aligned_16(global_src))"),
        "narrow row-stride routes may issue a 16-byte transaction only for a full aligned vector"
    );
    assert!(
        !selector.contains("else if (gemm_bi_is_aligned_16(global_src))"),
        "source alignment alone does not prove that a partial tail has 16 readable bytes"
    );
    assert!(
        selector.contains("gemm_bi_tf32_cp_async_4x4_zfill("),
        "partial TF32 vectors must decompose into independently zero-filled 4-byte copies"
    );
}
