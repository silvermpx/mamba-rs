const HALF_F32OUT_S3_SOURCE: &str =
    include_str!("../../../../kernels/gemm_bi_inference/sm89_half_f32out_s3.cu");
const F32_M128N64_TAIL_COPYPLAN_SOURCE: &str =
    include_str!("../../../../kernels/gemm_bi_inference/sm89_f32_m128n64_tail_copyplan.cu");
const RETAINED_EXPORT_DEFINITIONS: [&str; 3] = [
    "void gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_bf16(",
    "void gemm_bi_nn_inference_sm89_tc128_f32out_s3_v1_f16(",
    "void gemm_bi_nn_inference_sm89_f32_m128n64_tail_copyplan_v1(",
];

pub(crate) fn compiler_supported(
    device_cc: Option<(i32, i32)>,
    target: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> bool {
    device_cc == Some((8, 9))
        && target == "sm_89"
        && matches!(state_cap, 16 | 64)
        && matches!(nvrtc, (12, 8) | (13, 0) | (13, 2))
}

pub(crate) fn compose_fixed_source(
    mut composed: String,
    device_cc: Option<(i32, i32)>,
    target: &str,
    state_cap: usize,
    nvrtc: (i32, i32),
) -> Result<String, String> {
    if !compiler_supported(device_cc, target, state_cap, nvrtc) {
        return Ok(composed);
    }
    for definition in RETAINED_EXPORT_DEFINITIONS {
        if composed.contains(definition) {
            return Err(format!(
                "retained Inference source already defines {definition}"
            ));
        }
    }

    append_fragment(
        &mut composed,
        "kernels/gemm_bi_inference/sm89_half_f32out_s3.cu",
        HALF_F32OUT_S3_SOURCE,
    );
    append_fragment(
        &mut composed,
        "kernels/gemm_bi_inference/sm89_f32_m128n64_tail_copyplan.cu",
        F32_M128N64_TAIL_COPYPLAN_SOURCE,
    );
    Ok(composed)
}

fn append_fragment(composed: &mut String, logical_path: &str, fragment: &str) {
    composed.push_str("\n#line 1 \"");
    composed.push_str(logical_path);
    composed.push_str("\"\n");
    composed.push_str(fragment);
    if !fragment.ends_with('\n') {
        composed.push('\n');
    }
}
