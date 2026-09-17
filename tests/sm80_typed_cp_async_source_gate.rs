use std::{fs, path::Path};

fn sm80_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels/gemm_bi_triad/sm80/mma.cu"),
    )
    .expect("read SM80 triad source")
}

#[test]
fn typed_nn_fast_staging_requires_complete_half_vectors() {
    let source = sm80_source();
    for schedule in ["TC128", "TC64", "TC16"] {
        let marker = format!("#define GEMM_BI_DEFINE_GEMM_BI_NN_{schedule}");
        let start = source
            .find(&marker)
            .unwrap_or_else(|| panic!("missing typed NN {schedule} macro"));
        let predicate_start = source[start..]
            .find("bool fast_stage =")
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("missing typed NN {schedule} fast-stage predicate"));
        let predicate_end = source[predicate_start..]
            .find("float acc")
            .map(|offset| predicate_start + offset)
            .unwrap_or_else(|| panic!("missing typed NN {schedule} accumulator after predicate"));
        let predicate = &source[predicate_start..predicate_end];

        assert!(
            predicate.contains("((K & 7) == 0)"),
            "typed NN {schedule} must not issue a partial 16-byte A transaction"
        );
        assert!(
            predicate.contains("((N & 7) == 0)"),
            "typed NN {schedule} must not issue a partial 16-byte B transaction"
        );
    }
}
