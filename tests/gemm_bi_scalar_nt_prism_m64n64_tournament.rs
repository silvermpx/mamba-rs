const TRANSPOSE_32X16: &str = "gemm_bi_transpose_f32_32x16_d768_v1";
const TRANSPOSE_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar_nt_d768_transpose.cu");
const M64_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu");
const PRODUCTION_PRISM_M64_SYMBOL: &str = "gemm_bi_nn_prism_m64n64_bk16_s2_v1";
const PRISM_SPECIALIZED_M64_SYMBOL: &str =
    "gemm_bi_nn_prism_m64n64_bk16_s2_scalar_direct_experiment_v1";
const PRISM_SPECIALIZED_VECTOR_M64_SYMBOL: &str = "gemm_bi_nn_prism_m64n64_bk16_s2_vector_v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct M64ParameterGeometry {
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrismGeometry {
    logical_nt_m_kout_nred: (usize, usize, usize),
    m64_params: M64ParameterGeometry,
    scratch_elements: usize,
    transpose_grid: (u32, u32, u32),
    m64_grid: (u32, u32, u32),
}

const fn candidate_geometry() -> PrismGeometry {
    PrismGeometry {
        logical_nt_m_kout_nred: (4_621, 384, 1_928),
        m64_params: M64ParameterGeometry {
            m: 4_621,
            n: 384,
            k: 1_928,
            lda: 1_928,
            ldb: 384,
            ldc: 384,
        },
        scratch_elements: 384 * 1_928,
        transpose_grid: (1_928_u32.div_ceil(32), 384_u32.div_ceil(32), 1),
        m64_grid: (4_621_u32.div_ceil(64) * 384_u32.div_ceil(64), 1, 1),
    }
}

fn replace_once(source: String, from: &str, to: &str) -> Result<String, String> {
    let count = source.matches(from).count();
    if count != 1 {
        return Err(format!(
            "specialized source expected one {from:?}, found {count}"
        ));
    }
    Ok(source.replacen(from, to, 1))
}

fn replace_between_once(
    source: String,
    start: &str,
    end: &str,
    replacement: &str,
) -> Result<String, String> {
    let start_count = source.matches(start).count();
    let end_count = source.matches(end).count();
    if start_count != 1 || end_count != 1 {
        return Err(format!(
            "specialized source range is ambiguous: start={start_count} end={end_count}"
        ));
    }
    let start_index = source.find(start).expect("counted start") + start.len();
    let end_index = source[start_index..]
        .find(end)
        .map(|offset| start_index + offset)
        .ok_or_else(|| "specialized source range is reversed".to_owned())?;
    let mut result = String::with_capacity(source.len() + replacement.len());
    result.push_str(&source[..start_index]);
    result.push_str(replacement);
    result.push_str(&source[end_index..]);
    Ok(result)
}

fn generic_m64_source() -> Result<&'static str, String> {
    M64_SOURCE
        .split_once("// Exact-F32 NN schedule specialized for the prism backward cell.")
        .map(|(generic, _)| generic)
        .ok_or_else(|| "production M64 source lost the prism specialization boundary".to_owned())
}

fn production_prism_m64_source() -> Result<&'static str, String> {
    M64_SOURCE
        .split_once("// Exact-F32 NN schedule specialized for the prism backward cell.")
        .map(|(_, prism)| prism)
        .ok_or_else(|| "production M64 source lost the prism specialization body".to_owned())
}

fn prism_specialized_m64_source() -> Result<String, String> {
    let mut source = generic_m64_source()?.to_owned();
    let params_name = "SgbNnM64N64Params";
    let params_count = source.matches(params_name).count();
    if params_count != 5 {
        return Err(format!(
            "specialized source expected five parameter type uses, found {params_count}"
        ));
    }
    source = source.replace(params_name, "SgbNnPrismScalarDirectM64N64Params");
    source = replace_once(
        source,
        "gemm_bi_nn_m64n64_bk16_s2_v1",
        PRISM_SPECIALIZED_M64_SYMBOL,
    )?;
    source = replace_once(
        source,
        "const float* __restrict__ bias,",
        "const float* __restrict__,",
    )?;
    source = replace_once(
        source,
        "    assert(params.alpha == 1.0f || bias == nullptr);\n",
        "",
    )?;
    source = replace_between_once(
        source,
        "    float threadResults[EXACT_NN_M64N64_TM * EXACT_NN_M64N64_TN];\n\n",
        "    float* C_warp = C",
        r#"    #pragma unroll
    for (int result = 0;
         result < EXACT_NN_M64N64_TM * EXACT_NN_M64N64_TN;
         ++result) {
        threadResults[result] = 0.0f;
    }

"#,
    )?;
    source = replace_between_once(
        source,
        "    #undef ISSUE_EXACT_NN_M64N64_TILE\n\n",
        "\n}\n\n#undef EXACT_NN_M64N64_B_ROW_STRIDE",
        r#"    #pragma unroll
    for (int result_row = 0; result_row < EXACT_NN_M64N64_TM; ++result_row) {
        int global_row = pid_m * EXACT_NN_M64N64_BM
            + warp_row * EXACT_NN_M64N64_WM
            + thread_row * EXACT_NN_M64N64_TM + result_row;
        if (global_row >= params.m) {
            continue;
        }
        #pragma unroll
        for (int result_column = 0;
             result_column < EXACT_NN_M64N64_TN;
             result_column += 4) {
            int idx = result_row * EXACT_NN_M64N64_TN + result_column;
            float* destination = &C_warp[
                (thread_row * EXACT_NN_M64N64_TM + result_row) * params.ldc
                + thread_column * EXACT_NN_M64N64_TN + result_column];
            destination[0] = threadResults[idx + 0];
            destination[1] = threadResults[idx + 1];
            destination[2] = threadResults[idx + 2];
            destination[3] = threadResults[idx + 3];
        }
    }
"#,
    )?;
    Ok(source)
}

fn prism_specialized_vector_m64_source() -> Result<String, String> {
    let mut source = prism_specialized_m64_source()?;
    source = source.replace(
        "SgbNnPrismScalarDirectM64N64Params",
        "SgbNnPrismVectorM64N64Params",
    );
    source = replace_once(
        source,
        PRISM_SPECIALIZED_M64_SYMBOL,
        PRISM_SPECIALIZED_VECTOR_M64_SYMBOL,
    )?;
    source = replace_once(
        source,
        r#"            float* destination = &C_warp[
                (thread_row * EXACT_NN_M64N64_TM + result_row) * params.ldc
                + thread_column * EXACT_NN_M64N64_TN + result_column];
            destination[0] = threadResults[idx + 0];
            destination[1] = threadResults[idx + 1];
            destination[2] = threadResults[idx + 2];
            destination[3] = threadResults[idx + 3];"#,
        r#"            float* destination = &C_warp[
                (thread_row * EXACT_NN_M64N64_TM + result_row) * params.ldc
                + thread_column * EXACT_NN_M64N64_TN + result_column];
            float4 output = {
                threadResults[idx + 0],
                threadResults[idx + 1],
                threadResults[idx + 2],
                threadResults[idx + 3]
            };
            reinterpret_cast<float4*>(destination)[0] = output;"#,
    )?;
    Ok(source)
}

fn exact_m64_compute(cuda: &str) -> Result<&str, String> {
    cuda.split_once("    float* C_warp = C")
        .ok_or_else(|| "M64 source lost compute start".to_owned())?
        .1
        .split_once("    #undef ISSUE_EXACT_NN_M64N64_TILE")
        .map(|(compute, _)| compute)
        .ok_or_else(|| "M64 source lost compute end".to_owned())
}

fn validate_prism_specialized_m64_source(source: &str) -> Result<(), String> {
    let marker = format!("void {PRISM_SPECIALIZED_M64_SYMBOL}(");
    if source.matches(&marker).count() != 1 {
        return Err("specialized symbol count changed".into());
    }
    let (_, signature) = source
        .split_once(&marker)
        .ok_or_else(|| "specialized signature is missing".to_owned())?;
    let (parameters, _) = signature
        .split_once(") {")
        .ok_or_else(|| "specialized parameter list is missing".to_owned())?;
    if parameters.matches(',').count() >= 7 {
        return Err("specialized ABI exceeds seven arguments".into());
    }
    for forbidden in [
        "atomic",
        "mma.sync",
        "params.alpha *",
        "params.beta !=",
        "bias != nullptr",
        "SgbNnM64N64Params",
        "float2 output",
        "float4 output",
    ] {
        if source.contains(forbidden) {
            return Err(format!("specialized source contains {forbidden}"));
        }
    }
    for required in [
        "#define EXACT_NN_M64N64_BK 16",
        "#define EXACT_NN_M64N64_GROUP_M 16",
        "SgbNnPrismScalarDirectM64N64Params params",
        "for (int dot_index = 0; dot_index < EXACT_NN_M64N64_BK; ++dot_index)",
        "threadResults[idx] = __fmaf_rn(",
        "destination[0] = threadResults[idx + 0];",
        "destination[3] = threadResults[idx + 3];",
    ] {
        if !source.contains(required) {
            return Err(format!("specialized source lost {required}"));
        }
    }
    if exact_m64_compute(source)? != exact_m64_compute(generic_m64_source()?)? {
        return Err("specialized loader or ascending compute body changed".into());
    }
    Ok(())
}

fn validate_prism_specialized_vector_m64_source(source: &str) -> Result<(), String> {
    let marker = format!("void {PRISM_SPECIALIZED_VECTOR_M64_SYMBOL}(");
    if source.matches(&marker).count() != 1 {
        return Err("specialized vector symbol count changed".into());
    }
    let (_, signature) = source
        .split_once(&marker)
        .ok_or_else(|| "specialized vector signature is missing".to_owned())?;
    let (parameters, _) = signature
        .split_once(") {")
        .ok_or_else(|| "specialized vector parameter list is missing".to_owned())?;
    if parameters.matches(',').count() >= 7 {
        return Err("specialized vector ABI exceeds seven arguments".into());
    }
    for forbidden in [
        "atomic",
        "mma.sync",
        "params.alpha *",
        "params.beta !=",
        "bias != nullptr",
        "SgbNnM64N64Params",
        "SgbNnPrismM64N64Params",
        "SgbNnPrismScalarDirectM64N64Params",
        "destination[0] = threadResults",
    ] {
        if source.contains(forbidden) {
            return Err(format!("specialized vector source contains {forbidden}"));
        }
    }
    for required in [
        "#define EXACT_NN_M64N64_BK 16",
        "#define EXACT_NN_M64N64_GROUP_M 16",
        "SgbNnPrismVectorM64N64Params params",
        "for (int dot_index = 0; dot_index < EXACT_NN_M64N64_BK; ++dot_index)",
        "threadResults[idx] = __fmaf_rn(",
        "float4 output = {",
        "reinterpret_cast<float4*>(destination)[0] = output;",
    ] {
        if !source.contains(required) {
            return Err(format!("specialized vector source lost {required}"));
        }
    }
    if exact_m64_compute(source)? != exact_m64_compute(generic_m64_source()?)? {
        return Err("specialized vector loader or ascending compute body changed".into());
    }
    Ok(())
}

fn validate_production_prism_m64_source(source: &str) -> Result<(), String> {
    let marker = format!("void {PRODUCTION_PRISM_M64_SYMBOL}(");
    if source.matches(&marker).count() != 1 {
        return Err("production prism symbol count changed".into());
    }
    let (_, signature) = source
        .split_once(&marker)
        .ok_or_else(|| "production prism signature is missing".to_owned())?;
    let (parameters, _) = signature
        .split_once(") {")
        .ok_or_else(|| "production prism parameter list is missing".to_owned())?;
    if parameters.matches(',').count() >= 7 {
        return Err("production prism ABI exceeds seven arguments".into());
    }
    for forbidden in [
        "atomic",
        "mma.sync",
        "params.alpha *",
        "params.beta !=",
        "bias != nullptr",
        "destination[0] = threadResults",
    ] {
        if source.contains(forbidden) {
            return Err(format!("production prism source contains {forbidden}"));
        }
    }
    for required in [
        "#define EXACT_NN_M64N64_BK 16",
        "#define EXACT_NN_M64N64_GROUP_M 16",
        "SgbNnPrismM64N64Params params",
        "threadResults[idx] = __fmaf_rn(",
        "float4 output = {",
        "reinterpret_cast<float4*>(destination)[0] = output;",
    ] {
        if !source.contains(required) {
            return Err(format!("production prism source lost {required}"));
        }
    }
    if exact_m64_compute(source)? != exact_m64_compute(generic_m64_source()?)? {
        return Err("production prism loader or ascending compute body changed".into());
    }
    Ok(())
}

#[test]
fn prism_candidate_uses_exact_production_sources_and_bounded_abis() {
    for (source, symbol) in [
        (TRANSPOSE_SOURCE, TRANSPOSE_32X16),
        (M64_SOURCE, "gemm_bi_nn_m64n64_bk16_s2_v1"),
    ] {
        let marker = format!("void {symbol}(");
        assert_eq!(source.matches(&marker).count(), 1);
        let (_, signature) = source.split_once(&marker).expect("candidate signature");
        let (parameters, _) = signature
            .split_once(") {")
            .expect("candidate parameter list");
        assert!(parameters.matches(',').count() < 7);
        for forbidden in ["atomic", "mma.sync"] {
            assert!(!source.contains(forbidden), "{symbol} contains {forbidden}");
        }
    }
    for required in [
        "__shared__ float tile[32][33]",
        "transpose_f32_32x16_d768_body",
        "__launch_bounds__(512, 2)",
    ] {
        assert!(
            TRANSPOSE_SOURCE.contains(required),
            "candidate lost {required}"
        );
    }
    assert!(M64_SOURCE.contains("SgbNnM64N64Params params"));
    assert!(
        M64_SOURCE.contains(
            "int num_k_tiles = (params.k + EXACT_NN_M64N64_BK - 1) / EXACT_NN_M64N64_BK;"
        )
    );
    assert!(M64_SOURCE.contains("for (int tile = 0; tile < num_k_tiles; ++tile)"));
    assert!(M64_SOURCE.contains("for (int dot_index = 0;"));
    assert!(M64_SOURCE.contains("threadResults[idx] = __fmaf_rn("));
}

#[test]
fn prism_specialized_m64_changes_only_the_exact_epilogue_contract() {
    let source = prism_specialized_m64_source().expect("specialized CUDA source");
    validate_prism_specialized_m64_source(&source).expect("specialized source contract");
}

#[test]
fn prism_specialized_m64_identity_rejects_contract_mutations() {
    let source = prism_specialized_m64_source().expect("specialized CUDA source");
    assert!(validate_prism_specialized_m64_source(&source).is_ok());
    for mutated in [
        source.replacen("__fmaf_rn(", "__fadd_rn(", 1),
        source.replacen(
            "destination[1] = threadResults[idx + 1];",
            "destination[1] = params.alpha * threadResults[idx + 1];",
            1,
        ),
        source.replacen(
            "destination[0] = threadResults[idx + 0];",
            "atomicExch(reinterpret_cast<unsigned*>(destination), 0);",
            1,
        ),
        source.replacen(PRISM_SPECIALIZED_M64_SYMBOL, "wrong_symbol", 1),
    ] {
        assert!(validate_prism_specialized_m64_source(&mutated).is_err());
    }
}

#[test]
fn prism_specialized_vector_m64_keeps_compute_and_hardwires_float4_epilogue() {
    let source = prism_specialized_vector_m64_source().expect("vector CUDA source");
    validate_prism_specialized_vector_m64_source(&source).expect("vector source contract");
}

#[test]
fn production_prism_m64_is_the_single_hardwired_vector_body() {
    let source = production_prism_m64_source().expect("production prism CUDA source");
    validate_production_prism_m64_source(source).expect("production prism source contract");
    assert_eq!(M64_SOURCE.matches(PRODUCTION_PRISM_M64_SYMBOL).count(), 1);
}

#[test]
fn prism_specialized_vector_m64_identity_rejects_contract_mutations() {
    let source = prism_specialized_vector_m64_source().expect("vector CUDA source");
    assert!(validate_prism_specialized_vector_m64_source(&source).is_ok());
    for mutated in [
        source.replacen("__fmaf_rn(", "__fadd_rn(", 1),
        source.replacen("float4 output = {", "float2 output = {", 1),
        source.replacen(
            "reinterpret_cast<float4*>(destination)[0] = output;",
            "destination[0] = output.x;",
            1,
        ),
        source.replacen(PRISM_SPECIALIZED_VECTOR_M64_SYMBOL, "wrong_symbol", 1),
    ] {
        assert!(validate_prism_specialized_vector_m64_source(&mutated).is_err());
    }
}

#[test]
fn prism_candidate_geometry_is_exact_and_covers_the_tail() {
    assert_eq!(
        candidate_geometry(),
        PrismGeometry {
            logical_nt_m_kout_nred: (4_621, 384, 1_928),
            m64_params: M64ParameterGeometry {
                m: 4_621,
                n: 384,
                k: 1_928,
                lda: 1_928,
                ldb: 384,
                ldc: 384,
            },
            scratch_elements: 740_352,
            transpose_grid: (61, 12, 1),
            m64_grid: (438, 1, 1),
        }
    );
    assert_eq!(candidate_geometry().m64_params.k % 16, 8);
    assert!(candidate_geometry().scratch_elements * size_of::<f32>() <= 16 * 1024 * 1024);
}

#[cfg(feature = "cuda")]
mod cuda_tournament {
    use std::ffi::{CStr, c_int, c_void};
    use std::sync::Arc;

    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::blas::gpu_gemm_bi_backward_dx_raw;
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::context::{BiGemmFamily, F32TriadPolicy, GpuCtx};
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::mamba_ssm::gpu::gemm_bi_triad::{
        PhysicalQualificationRequest, PhysicalQualificationRoute, qualify_physical_launch,
    };
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;
    use mamba_rs::mamba_ssm::gpu::kernel_identity::{
        ArtifactIdentity, ArtifactKind, COMPILER_REVISION, COMPOSER_REVISION, CompilerIdentity,
        CudaTarget, ModuleKind, NUMERIC_ABI_REVISION, ResolvedGemmOp, SCHEDULE_REVISION,
    };
    use sha2::{Digest as _, Sha256};

    use super::{
        PRISM_SPECIALIZED_M64_SYMBOL, PRISM_SPECIALIZED_VECTOR_M64_SYMBOL,
        PRODUCTION_PRISM_M64_SYMBOL, TRANSPOSE_32X16, TRANSPOSE_SOURCE, candidate_geometry,
        prism_specialized_m64_source, prism_specialized_vector_m64_source,
    };

    const DIMS: (usize, usize, usize) = (4_621, 384, 1_928);
    const M64_SYMBOL: &str = "gemm_bi_nn_m64n64_bk16_s2_v1";
    const GENERIC_NT_SYMBOL: &str = "gemm_bi_nt";
    const GUARD: usize = 64;
    const INPUT_GUARD: u32 = 0x7fc0_a768;
    const OUTPUT_GUARD: u32 = 0x7fc0_c768;
    const M64_SHARED: usize = 17_408;
    const NT_SHARED: usize = 33_376;
    const TARGET_WINDOW_US: f64 = 10_000.0;
    const TRANSPOSE_STATIC_SHARED: usize = 32 * 33 * size_of::<f32>();
    const MIN_WINDOWS: usize = 101;
    const SCALAR_MIN_P05_SPEEDUP: f64 = 1.20;
    const SCALAR_MIN_P50_SPEEDUP: f64 = 1.25;
    const SPECIALIZED_MIN_P05_SPEEDUP: f64 = 1.005;
    const SPECIALIZED_MIN_P50_SPEEDUP: f64 = 1.01;
    const PEDANTIC_MIN_P05_SPEEDUP: f64 = 0.90;
    const PEDANTIC_MIN_P50_SPEEDUP: f64 = 0.92;
    const PARITY_MIN_P05: f64 = 0.97;
    const PARITY_MIN_P50: f64 = 0.985;
    const PARITY_MAX_P50: f64 = 1.015;
    const PARITY_MAX_P95: f64 = 1.03;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Arm {
        Production,
        GenericNt,
        Candidate,
        Specialized,
        SpecializedVector,
        CublasPedantic,
        CublasFast,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CublasTranspose {
        None,
        Transpose,
    }

    impl CublasTranspose {
        fn cuda(self) -> cudarc::cublas::sys::cublasOperation_t {
            use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};

            match self {
                Self::None => CUBLAS_OP_N,
                Self::Transpose => CUBLAS_OP_T,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct CublasNtGeometry {
        logical_op: ResolvedGemmOp,
        trans_a: CublasTranspose,
        trans_b: CublasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        lda: c_int,
        ldb: c_int,
        ldc: c_int,
        output_rows: usize,
        output_columns: usize,
        output_elements: usize,
    }

    fn cublas_nt_geometry(
        logical_m_kout_nred: (usize, usize, usize),
    ) -> Result<CublasNtGeometry, String> {
        let (m, k_out, n_reduction) = logical_m_kout_nred;
        let as_int = |value: usize, name: &str| {
            c_int::try_from(value)
                .map_err(|_| format!("prism cuBLAS {name} exceeds c_int: {value}"))
        };
        let output_elements = m
            .checked_mul(k_out)
            .ok_or_else(|| format!("prism cuBLAS output size overflows: {m} * {k_out}"))?;
        Ok(CublasNtGeometry {
            logical_op: ResolvedGemmOp::Nt,
            trans_a: CublasTranspose::Transpose,
            trans_b: CublasTranspose::None,
            m: as_int(k_out, "physical M")?,
            n: as_int(m, "physical N")?,
            k: as_int(n_reduction, "physical K")?,
            lda: as_int(n_reduction, "lda")?,
            ldb: as_int(n_reduction, "ldb")?,
            ldc: as_int(k_out, "ldc")?,
            output_rows: m,
            output_columns: k_out,
            output_elements,
        })
    }

    fn validate_cublas_nt_geometry(
        logical_m_kout_nred: (usize, usize, usize),
        actual: CublasNtGeometry,
    ) -> Result<(), String> {
        let expected = cublas_nt_geometry(logical_m_kout_nred)?;
        if actual != expected {
            return Err(format!(
                "cuBLAS NT geometry changed: actual={actual:?} expected={expected:?}"
            ));
        }
        Ok(())
    }

    impl Arm {
        const fn name(self) -> &'static str {
            match self {
                Self::Production => "production_nt_prism_vector_tag30",
                Self::GenericNt => "generic_nt",
                Self::Candidate => "transpose32x16_plus_m64n64",
                Self::Specialized => "transpose32x16_plus_prism_specialized_m64n64",
                Self::SpecializedVector => "transpose32x16_plus_prism_specialized_vector_m64n64",
                Self::CublasPedantic => "cublas_pedantic",
                Self::CublasFast => "cublas_fast_tf32",
            }
        }

        const fn is_candidate(self) -> bool {
            matches!(
                self,
                Self::Candidate | Self::Specialized | Self::SpecializedVector
            )
        }

        const fn m64_symbol(self) -> &'static str {
            match self {
                Self::Candidate => M64_SYMBOL,
                Self::Specialized => PRISM_SPECIALIZED_M64_SYMBOL,
                Self::SpecializedVector => PRISM_SPECIALIZED_VECTOR_M64_SYMBOL,
                _ => panic!("non-candidate arm has no M64 symbol"),
            }
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct NnParams {
        alpha: f32,
        beta: f32,
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for NnParams {}

    struct Kernel {
        function: CudaFunction,
        config: LaunchConfig,
        symbol: &'static str,
    }

    struct Runtime {
        _device: GpuDevice,
        ctx: GpuCtx,
        _module: Arc<CudaModule>,
        m64: Kernel,
        specialized_m64: Kernel,
        specialized_vector_m64: Kernel,
        generic_nt: Kernel,
        transpose16: Kernel,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CandidateNodeIdentity {
        symbol: String,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        arguments_digest: [u8; 32],
    }

    fn validate_candidate_identity(
        nodes: &[CandidateNodeIdentity],
        m64_symbol: &str,
    ) -> Result<(), String> {
        if nodes.len() != 2 {
            return Err(format!(
                "candidate must contain two ordered nodes, found {}",
                nodes.len()
            ));
        }
        if nodes[0].symbol != TRANSPOSE_32X16 || nodes[1].symbol != m64_symbol {
            return Err(format!("candidate node order changed: {nodes:?}"));
        }
        let expected_configs = [
            ((61, 12, 1), (32, 16, 1), 0),
            ((438, 1, 1), (128, 1, 1), M64_SHARED as u32),
        ];
        for (node, (grid, block, shared)) in nodes.iter().zip(expected_configs) {
            if (node.grid_dim, node.block_dim, node.shared_mem_bytes) != (grid, block, shared) {
                return Err(format!("{} launch config changed: {node:?}", node.symbol));
            }
        }
        for node in nodes {
            if node.arguments_digest == [0; 32] {
                return Err(format!("{} has a zero argument digest", node.symbol));
            }
        }
        if nodes[0].arguments_digest == nodes[1].arguments_digest {
            return Err("candidate node argument digests collided".into());
        }
        Ok(())
    }

    fn validate_qualified_environment(
        compute_capability: (u32, u32),
        multiprocessor_count: u32,
        nvrtc_target: &str,
        compiler: CompilerIdentity,
        artifact: ArtifactIdentity,
    ) -> Result<(), String> {
        if compute_capability != (12, 0)
            || multiprocessor_count != 170
            || nvrtc_target != "compute_120"
            || compiler.target.as_str() != "compute_120"
            || compiler.nvrtc_version != (13, 2)
            || !compiler.nvrtc_library_known
            || compiler.source_digest == [0; 32]
            || compiler.invocation_digest == [0; 32]
            || compiler.header_manifest_digest == [0; 32]
            || compiler.nvrtc_library_domain == [0; 32]
            || compiler.output_kind != ArtifactKind::Ptx
            || compiler.composer_revision != COMPOSER_REVISION
            || compiler.compiler_revision != COMPILER_REVISION
            || compiler.numeric_abi_revision != NUMERIC_ABI_REVISION
            || compiler.schedule_revision != SCHEDULE_REVISION
            || artifact.module_kind != ModuleKind::TriadScalar
            || artifact.artifact_kind != compiler.output_kind
            || artifact.compile_key != compiler.invocation_digest
            || artifact.compile_key == [0; 32]
            || artifact.artifact_digest == [0; 32]
        {
            return Err(format!(
                "prism tournament requires the qualified compute_120 NVRTC 13.2 TriadScalar artifact domain: cc={compute_capability:?} sms={multiprocessor_count} target={nvrtc_target} compiler={compiler:?} artifact={artifact:?}"
            ));
        }
        Ok(())
    }

    struct GuardedBuffer {
        buffer: GpuBuffer,
        expected: Vec<f32>,
        offset: usize,
        len: usize,
        guard: u32,
    }

    impl GuardedBuffer {
        fn new(stream: &Arc<CudaStream>, values: Vec<f32>, guard: u32) -> Result<Self, String> {
            let offset = GUARD;
            let len = values.len();
            let mut expected = vec![f32::from_bits(guard); offset + len + GUARD];
            expected[offset..offset + len].copy_from_slice(&values);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &expected)?,
                expected,
                offset,
                len,
                guard,
            })
        }

        fn ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, self.offset)
        }

        fn active_bits(&self, stream: &Arc<CudaStream>, label: &str) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            for (index, value) in values[..self.offset]
                .iter()
                .chain(&values[self.offset + self.len..])
                .enumerate()
            {
                if value.to_bits() != self.guard {
                    return Err(format!("{label} red zone changed at {index}"));
                }
            }
            Ok(values[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }

        fn validate_unchanged(&self, stream: &Arc<CudaStream>, label: &str) -> Result<(), String> {
            let actual = self.buffer.to_cpu(stream)?;
            if actual
                .iter()
                .zip(&self.expected)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
            {
                return Err(format!("{label} or its red zones changed"));
            }
            Ok(())
        }

        fn expected_active_bits(&self) -> Vec<u32> {
            self.expected[self.offset..self.offset + self.len]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        }
    }

    struct Fixture {
        a: GuardedBuffer,
        production_a: GpuBuffer,
        production_a_expected: Vec<f32>,
        b: GuardedBuffer,
        scratch: GuardedBuffer,
        generic_output: GuardedBuffer,
        candidate_output: GuardedBuffer,
        production_output: GpuBuffer,
        cublas_output: GpuBuffer,
        params: NnParams,
    }

    fn compose_source() -> String {
        let specialized =
            prism_specialized_m64_source().expect("build exact prism specialized source");
        let specialized_vector = prism_specialized_vector_m64_source()
            .expect("build exact prism specialized vector source");
        [
            include_str!("../kernels/_typed_prelude.cuh"),
            include_str!("../kernels/gemm_bi_triad/contract.cuh"),
            include_str!("../kernels/gemm_bi_triad/common.cuh"),
            include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
            include_str!("../kernels/gemm_bi_triad/scalar.cu"),
            include_str!("../kernels/gemm_bi_triad/scalar_nn_m64n64.cu"),
            TRANSPOSE_SOURCE,
            specialized.as_str(),
            specialized_vector.as_str(),
        ]
        .iter()
        .map(|source| {
            source
                .lines()
                .filter(|line| !line.trim().starts_with("#include \"_typed_prelude.cuh\""))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n")
    }

    #[test]
    fn composed_prism_source_defines_each_parameter_type_once() {
        let source = compose_source();
        for name in [
            "SgbNnM64N64Params",
            "SgbNnPrismM64N64Params",
            "SgbNnPrismScalarDirectM64N64Params",
            "SgbNnPrismVectorM64N64Params",
        ] {
            assert_eq!(
                source.matches(&format!("struct {name} {{")).count(),
                1,
                "composed CUDA parameter type {name}"
            );
        }
    }

    fn load_kernel(
        module: &Arc<CudaModule>,
        symbol: &'static str,
        config: LaunchConfig,
        dynamic_shared: usize,
    ) -> Result<Kernel, String> {
        let function = module
            .load_function(symbol)
            .map_err(|error| format!("load {symbol}: {error:?}"))?;
        if dynamic_shared > 0 {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    dynamic_shared as i32,
                )
                .map_err(|error| format!("set {symbol} shared memory: {error:?}"))?;
        }
        Ok(Kernel {
            function,
            config,
            symbol,
        })
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        let identity = device.identity();
        if identity.compute_capability != (12, 0) || identity.multiprocessor_count != 170 {
            return Err(format!(
                "prism tournament requires CC12.0/170 SM, found {:?}/{} SM",
                identity.compute_capability, identity.multiprocessor_count
            ));
        }
        let ctx = GpuCtx::new(&device)?;
        ctx.set_batch_invariant(true);
        ctx.set_bi_gemm_family(BiGemmFamily::Triad);
        ctx.set_f32_triad_policy(F32TriadPolicy::ExactScalarFmaV1);
        validate_qualified_environment(
            identity.compute_capability,
            identity.multiprocessor_count,
            device.nvrtc_target(),
            ctx.kernels.triad_scalar_compiler_identity(),
            ctx.kernels.artifact_set_identity().triad_scalar,
        )?;
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "-DGEMM_BI_GROUP_M=16".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
            .map_err(|error| format!("compile prism tournament: {error:?}"))?;
        let ptx_source = ptx.to_src();
        let candidate_artifact_digest: [u8; 32] = Sha256::digest(ptx_source.as_bytes()).into();
        if candidate_artifact_digest == [0; 32] {
            return Err("candidate PTX artifact digest is zero".into());
        }
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_source))
            .map_err(|error| format!("load prism tournament module: {error:?}"))?;
        let geometry = candidate_geometry();
        let m64 = load_kernel(
            &module,
            M64_SYMBOL,
            LaunchConfig {
                grid_dim: geometry.m64_grid,
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
            },
            M64_SHARED,
        )?;
        let specialized_m64 = load_kernel(
            &module,
            PRISM_SPECIALIZED_M64_SYMBOL,
            LaunchConfig {
                grid_dim: geometry.m64_grid,
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
            },
            M64_SHARED,
        )?;
        let specialized_vector_m64 = load_kernel(
            &module,
            PRISM_SPECIALIZED_VECTOR_M64_SYMBOL,
            LaunchConfig {
                grid_dim: geometry.m64_grid,
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
            },
            M64_SHARED,
        )?;
        let generic_nt = load_kernel(
            &module,
            GENERIC_NT_SYMBOL,
            LaunchConfig {
                grid_dim: (111, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: NT_SHARED as u32,
            },
            NT_SHARED,
        )?;
        let transpose16 = load_kernel(
            &module,
            TRANSPOSE_32X16,
            LaunchConfig {
                grid_dim: geometry.transpose_grid,
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
            },
            0,
        )?;
        Ok(Runtime {
            _device: device,
            ctx,
            _module: module,
            m64,
            specialized_m64,
            specialized_vector_m64,
            generic_nt,
            transpose16,
        })
    }

    fn values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 4093 == 0 {
                    -0.0
                } else {
                    let sign = ((state >> 63) as u32) << 31;
                    let exponent = (123 + ((state >> 29) as u32 % 8)) << 23;
                    let mantissa = (state as u32 & 0x007f_ffff) | 1;
                    f32::from_bits(sign | exponent | mantissa)
                }
            })
            .collect()
    }

    fn new_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let a_expected = values(m * n, 0xa768_1001);
        let b_values = values(k_out * n, 0xb768_1002);
        new_fixture_with_values(runtime, a_expected, b_values)
    }

    fn new_fixture_with_values(
        runtime: &Runtime,
        a_values: Vec<f32>,
        b_values: Vec<f32>,
    ) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        if a_values.len() != m * n || b_values.len() != k_out * n {
            return Err("prism fixture dimensions do not match inputs".into());
        }
        let m64 = candidate_geometry().m64_params;
        let output = vec![0.0; m * k_out];
        Ok(Fixture {
            a: GuardedBuffer::new(&runtime.ctx.stream, a_values.clone(), INPUT_GUARD)?,
            production_a: GpuBuffer::from_cpu(&runtime.ctx.stream, &a_values)?,
            production_a_expected: a_values,
            b: GuardedBuffer::new(&runtime.ctx.stream, b_values, INPUT_GUARD)?,
            scratch: GuardedBuffer::new(&runtime.ctx.stream, vec![0.0; k_out * n], OUTPUT_GUARD)?,
            generic_output: GuardedBuffer::new(&runtime.ctx.stream, output.clone(), OUTPUT_GUARD)?,
            candidate_output: GuardedBuffer::new(
                &runtime.ctx.stream,
                output.clone(),
                OUTPUT_GUARD,
            )?,
            production_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &output)?,
            cublas_output: GpuBuffer::from_cpu(&runtime.ctx.stream, &output)?,
            params: NnParams {
                alpha: 1.0,
                beta: 0.0,
                m: m64.m as i32,
                n: m64.n as i32,
                k: m64.k as i32,
                lda: m64.lda as i32,
                ldb: m64.ldb as i32,
                ldc: m64.ldc as i32,
            },
        })
    }

    fn launch_cublas_nt(
        runtime: &Runtime,
        output: u64,
        a: u64,
        b: u64,
        logical_m_kout_nred: (usize, usize, usize),
        mode: Arm,
    ) -> Result<(), String> {
        use cudarc::cublas::sys::cublasComputeType_t::{
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_PEDANTIC,
        };

        let compute = match mode {
            Arm::CublasPedantic => CUBLAS_COMPUTE_32F_PEDANTIC,
            Arm::CublasFast => CUBLAS_COMPUTE_32F_FAST_TF32,
            _ => return Err(format!("{mode:?} is not a cuBLAS denominator")),
        };
        let geometry = cublas_nt_geometry(logical_m_kout_nred)?;
        validate_cublas_nt_geometry(logical_m_kout_nred, geometry)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *runtime.ctx.blas.handle(),
                geometry.trans_a.cuda(),
                geometry.trans_b.cuda(),
                geometry.m,
                geometry.n,
                geometry.k,
                (&alpha as *const f32).cast::<c_void>(),
                b as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.lda,
                a as *const c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldb,
                (&beta as *const f32).cast::<c_void>(),
                output as *mut c_void,
                WeightDtype::F32.cuda_data_type(),
                geometry.ldc,
                compute,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(|error| format!("{} launch failed: {error:?}", mode.name()))?;
        }
        Ok(())
    }

    fn launch_cublas(runtime: &Runtime, fixture: &Fixture, mode: Arm) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        launch_cublas_nt(
            runtime,
            fixture.cublas_output.raw_ptr_at(stream, 0),
            fixture.a.ptr(stream),
            fixture.b.ptr(stream),
            DIMS,
            mode,
        )
    }

    fn validate_cublas_orientation(runtime: &Runtime) -> Result<(), String> {
        const LOGICAL: (usize, usize, usize) = (3, 2, 4);
        let geometry = cublas_nt_geometry(LOGICAL)?;
        let expected = [14.0_f32, 34.0, 33.0, 57.0, 65.0, 115.0];
        if geometry.output_elements != expected.len()
            || (geometry.output_rows, geometry.output_columns) != (3, 2)
        {
            return Err(format!(
                "cuBLAS orientation output shape changed: geometry={geometry:?} expected=3x2"
            ));
        }
        let a = GuardedBuffer::new(
            &runtime.ctx.stream,
            vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            INPUT_GUARD,
        )?;
        let b = GuardedBuffer::new(
            &runtime.ctx.stream,
            vec![7.0, 11.0, 13.0, 0.0, 17.0, 19.0, 23.0, 0.0],
            INPUT_GUARD,
        )?;
        let output = GuardedBuffer::new(
            &runtime.ctx.stream,
            vec![0.0; geometry.output_elements],
            OUTPUT_GUARD,
        )?;
        let expected_bits = expected.map(f32::to_bits);
        for mode in [Arm::CublasPedantic, Arm::CublasFast] {
            for repeat in 0..3 {
                launch_cublas_nt(
                    runtime,
                    output.ptr(&runtime.ctx.stream),
                    a.ptr(&runtime.ctx.stream),
                    b.ptr(&runtime.ctx.stream),
                    LOGICAL,
                    mode,
                )?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{} orientation sync: {error:?}", mode.name()))?;
                let actual = output.active_bits(&runtime.ctx.stream, mode.name())?;
                if actual.as_slice() != expected_bits {
                    return Err(format!(
                        "{} orientation repeat {repeat} changed: actual={actual:08x?} expected={expected_bits:08x?}",
                        mode.name()
                    ));
                }
                a.validate_unchanged(&runtime.ctx.stream, "cuBLAS orientation A")?;
                b.validate_unchanged(&runtime.ctx.stream, "cuBLAS orientation B")?;
            }
        }
        Ok(())
    }

    fn launch_arm(runtime: &Runtime, fixture: &mut Fixture, arm: Arm) -> Result<(), String> {
        let stream = &runtime.ctx.stream;
        let a = fixture.a.ptr(stream);
        let b = fixture.b.ptr(stream);
        match arm {
            Arm::Production => gpu_gemm_bi_backward_dx_raw(
                &runtime.ctx,
                &mut fixture.production_output,
                &fixture.production_a,
                b,
                DIMS.0,
                DIMS.1,
                DIMS.2,
            ),
            Arm::GenericNt => {
                let output = fixture.generic_output.ptr(stream);
                let alpha = 1.0_f32;
                let (m, k_out, n) = (DIMS.0 as i32, DIMS.1 as i32, DIMS.2 as i32);
                let mut builder = stream.launch_builder(&runtime.generic_nt.function);
                builder.arg(&output);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&alpha);
                builder.arg(&m);
                builder.arg(&n);
                builder.arg(&k_out);
                unsafe { builder.launch(runtime.generic_nt.config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch {}: {error:?}", runtime.generic_nt.symbol))
            }
            Arm::Candidate | Arm::Specialized | Arm::SpecializedVector => {
                let transpose = &runtime.transpose16;
                let m64 = match arm {
                    Arm::Candidate => &runtime.m64,
                    Arm::Specialized => &runtime.specialized_m64,
                    Arm::SpecializedVector => &runtime.specialized_vector_m64,
                    _ => unreachable!(),
                };
                let scratch = fixture.scratch.ptr(stream);
                let rows = DIMS.1 as i32;
                let cols = DIMS.2 as i32;
                let mut transpose_builder = stream.launch_builder(&transpose.function);
                transpose_builder.arg(&scratch);
                transpose_builder.arg(&b);
                transpose_builder.arg(&rows);
                transpose_builder.arg(&cols);
                unsafe { transpose_builder.launch(transpose.config) }
                    .map_err(|error| format!("launch {}: {error:?}", transpose.symbol))?;
                let output = fixture.candidate_output.ptr(stream);
                let bias = 0_u64;
                let mut m64_builder = stream.launch_builder(&m64.function);
                m64_builder.arg(&output);
                m64_builder.arg(&a);
                m64_builder.arg(&scratch);
                m64_builder.arg(&bias);
                m64_builder.arg(&fixture.params);
                unsafe { m64_builder.launch(m64.config) }
                    .map(|_| ())
                    .map_err(|error| format!("launch {}: {error:?}", m64.symbol))
            }
            Arm::CublasPedantic | Arm::CublasFast => launch_cublas(runtime, fixture, arm),
        }
    }

    fn capture_arm(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<CudaGraph, String> {
        unsafe { capture_into_graph(&runtime.ctx.stream, || launch_arm(runtime, fixture, arm)) }
    }

    fn digest_arguments(arguments: &[&[u8]]) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"nt-prism-candidate-arguments.v1");
        for argument in arguments {
            digest.update((argument.len() as u64).to_le_bytes());
            digest.update(argument);
        }
        digest.finalize().into()
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
    }

    fn eager_candidate_identity(
        runtime: &Runtime,
        fixture: &Fixture,
        arm: Arm,
    ) -> Vec<CandidateNodeIdentity> {
        let scratch = fixture.scratch.ptr(&runtime.ctx.stream);
        let b = fixture.b.ptr(&runtime.ctx.stream);
        let rows = DIMS.1 as i32;
        let cols = DIMS.2 as i32;
        let output = fixture.candidate_output.ptr(&runtime.ctx.stream);
        let a = fixture.a.ptr(&runtime.ctx.stream);
        let bias = 0_u64;
        let transpose = &runtime.transpose16;
        let m64 = match arm {
            Arm::Candidate => &runtime.m64,
            Arm::Specialized => &runtime.specialized_m64,
            Arm::SpecializedVector => &runtime.specialized_vector_m64,
            _ => panic!("eager candidate identity requires a candidate arm"),
        };
        vec![
            CandidateNodeIdentity {
                symbol: transpose.symbol.into(),
                grid_dim: transpose.config.grid_dim,
                block_dim: transpose.config.block_dim,
                shared_mem_bytes: transpose.config.shared_mem_bytes,
                arguments_digest: digest_arguments(&[
                    bytes_of(&scratch),
                    bytes_of(&b),
                    bytes_of(&rows),
                    bytes_of(&cols),
                ]),
            },
            CandidateNodeIdentity {
                symbol: m64.symbol.into(),
                grid_dim: m64.config.grid_dim,
                block_dim: m64.config.block_dim,
                shared_mem_bytes: m64.config.shared_mem_bytes,
                arguments_digest: digest_arguments(&[
                    bytes_of(&output),
                    bytes_of(&a),
                    bytes_of(&scratch),
                    bytes_of(&bias),
                    bytes_of(&fixture.params),
                ]),
            },
        ]
    }

    fn cuda_ok(result: sys::CUresult, label: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{label}: {result:?}"))
        }
    }

    unsafe fn graph_node_identity(node: sys::CUgraphNode) -> Result<CandidateNodeIdentity, String> {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        cuda_ok(
            unsafe { sys::cuGraphNodeGetType(node, &mut kind) },
            "graph node type",
        )?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            return Err(format!("candidate graph contains non-kernel node {kind:?}"));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "graph kernel params",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "kernel name",
        )?;
        if name.is_null() {
            return Err("graph kernel name is null".into());
        }
        let symbol = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|error| format!("graph kernel name is not UTF-8: {error}"))?
            .to_owned();
        let argument_sizes: &[usize] = match symbol.as_str() {
            TRANSPOSE_32X16 => &[8, 8, 4, 4],
            M64_SYMBOL | PRISM_SPECIALIZED_M64_SYMBOL | PRISM_SPECIALIZED_VECTOR_M64_SYMBOL => {
                &[8, 8, 8, 8, size_of::<NnParams>()]
            }
            _ => return Err(format!("unexpected candidate graph symbol {symbol}")),
        };
        if params.kernelParams.is_null() {
            return Err(format!("{symbol} graph kernelParams is null"));
        }
        let mut arguments = Vec::with_capacity(argument_sizes.len());
        for (index, size) in argument_sizes.iter().copied().enumerate() {
            let pointer = unsafe { *params.kernelParams.add(index) };
            if pointer.is_null() {
                return Err(format!("{symbol} argument {index} is null"));
            }
            arguments.push(unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), size) });
        }
        Ok(CandidateNodeIdentity {
            symbol,
            grid_dim: (params.gridDimX, params.gridDimY, params.gridDimZ),
            block_dim: (params.blockDimX, params.blockDimY, params.blockDimZ),
            shared_mem_bytes: params.sharedMemBytes,
            arguments_digest: digest_arguments(&arguments),
        })
    }

    fn graph_candidate_identity(
        graph: &CudaGraph,
        m64_symbol: &str,
    ) -> Result<Vec<CandidateNodeIdentity>, String> {
        let raw = graph.cu_graph();
        let mut node_count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(raw, std::ptr::null_mut(), &mut node_count) },
            "graph node count",
        )?;
        if node_count != 2 {
            return Err(format!(
                "candidate graph has {node_count} nodes, expected two"
            ));
        }
        let mut edge_count = 0_usize;
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut edge_count,
                )
            },
            "graph edge count",
        )?;
        if edge_count != 1 {
            return Err(format!(
                "candidate graph has {edge_count} edges, expected one"
            ));
        }
        let mut from = [std::ptr::null_mut(); 1];
        let mut to = [std::ptr::null_mut(); 1];
        let mut edge_data: [sys::CUgraphEdgeData; 1] = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe {
                sys::cuGraphGetEdges_v2(
                    raw,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    edge_data.as_mut_ptr(),
                    &mut edge_count,
                )
            },
            "graph edges",
        )?;
        if edge_data[0].from_port != 0
            || edge_data[0].to_port != 0
            || edge_data[0].type_
                != sys::CUgraphDependencyType::CU_GRAPH_DEPENDENCY_TYPE_DEFAULT as u8
            || edge_data[0].reserved != [0; 5]
        {
            return Err("candidate graph edge descriptor is not the exact CUDA default".into());
        }
        let ordered = vec![unsafe { graph_node_identity(from[0]) }?, unsafe {
            graph_node_identity(to[0])
        }?];
        validate_candidate_identity(&ordered, m64_symbol)?;
        Ok(ordered)
    }

    fn bits(runtime: &Runtime, fixture: &Fixture, arm: Arm) -> Result<Vec<u32>, String> {
        match arm {
            Arm::Production => Ok(fixture
                .production_output
                .to_cpu(&runtime.ctx.stream)?
                .iter()
                .map(|value| value.to_bits())
                .collect()),
            Arm::GenericNt => fixture
                .generic_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            candidate if candidate.is_candidate() => fixture
                .candidate_output
                .active_bits(&runtime.ctx.stream, arm.name()),
            _ => unreachable!(),
        }
    }

    fn validate_scratch_transpose(runtime: &Runtime, fixture: &Fixture) -> Result<(), String> {
        let (_, rows, cols) = DIMS;
        let input = fixture.b.expected_active_bits();
        let scratch = fixture
            .scratch
            .active_bits(&runtime.ctx.stream, "transpose scratch")?;
        for row in 0..rows {
            for col in 0..cols {
                let expected = input[row * cols + col];
                let actual = scratch[col * rows + row];
                if actual != expected {
                    return Err(format!(
                        "transpose scratch differs at input ({row},{col}): actual=0x{actual:08x} expected=0x{expected:08x}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn exceptional_fixture(runtime: &Runtime) -> Result<Fixture, String> {
        let (m, k_out, n) = DIMS;
        let mut a = vec![0.0; m * n];
        for row in 0..m {
            a[row * n] = 1.0;
        }
        let mut b = vec![0.0; k_out * n];
        for (column, bits) in [
            0x8000_0000,
            0x0000_0001,
            0x7f80_0000,
            0x7fc0_1234,
            0xff80_0000,
        ]
        .into_iter()
        .enumerate()
        {
            b[column * n] = f32::from_bits(bits);
        }
        new_fixture_with_values(runtime, a, b)
    }

    fn validate_exceptional_oracle(bits: &[u32]) -> Result<(), String> {
        let (_, k_out, _) = DIMS;
        for row in [0, DIMS.0 - 1] {
            let base = row * k_out;
            if bits[base] != 0x0000_0000
                || bits[base + 1] != 0x0000_0001
                || bits[base + 2] != 0x7f80_0000
                || !f32::from_bits(bits[base + 3]).is_nan()
                || bits[base + 4] != 0xff80_0000
            {
                return Err(format!(
                    "isolated exceptional outputs changed at row {row}: {:08x?}",
                    &bits[base..base + 5]
                ));
            }
        }
        Ok(())
    }

    struct ResourceContract {
        threads: u32,
        static_shared: usize,
        dynamic_shared: usize,
        register_cap: i32,
        minimum_occupancy: u32,
    }

    fn check_resources(kernel: &Kernel, contract: ResourceContract) -> Result<(), String> {
        let threads = kernel.config.block_dim.0 * kernel.config.block_dim.1;
        let registers = kernel
            .function
            .num_regs()
            .map_err(|error| format!("regs: {error:?}"))?;
        let local = kernel
            .function
            .local_size_bytes()
            .map_err(|error| format!("local: {error:?}"))?;
        let static_shared = kernel
            .function
            .shared_size_bytes()
            .map_err(|error| format!("static shared: {error:?}"))?;
        let occupancy = kernel
            .function
            .occupancy_max_active_blocks_per_multiprocessor(
                threads,
                kernel.config.shared_mem_bytes as usize,
                None,
            )
            .map_err(|error| format!("occupancy: {error:?}"))?;
        eprintln!(
            "nt_prism resource symbol={} threads={} registers={} local_bytes={} static_shared_bytes={} dynamic_shared_bytes={} active_blocks={}",
            kernel.symbol,
            threads,
            registers,
            local,
            static_shared,
            kernel.config.shared_mem_bytes,
            occupancy
        );
        if threads != contract.threads
            || static_shared as usize != contract.static_shared
            || kernel.config.shared_mem_bytes as usize != contract.dynamic_shared
            || local != 0
            || registers > contract.register_cap
            || occupancy < contract.minimum_occupancy
        {
            return Err(format!(
                "{} resource contract failed: expected threads={} static={} dynamic={} registers<={} occupancy>={}",
                kernel.symbol,
                contract.threads,
                contract.static_shared,
                contract.dynamic_shared,
                contract.register_cap,
                contract.minimum_occupancy
            ));
        }
        Ok(())
    }

    fn validate_physical_production(runtime: &Runtime) -> Result<(), String> {
        let request = PhysicalQualificationRequest::contiguous(
            ResolvedGemmOp::Nt,
            DIMS,
            PhysicalQualificationRoute::F32Policy(F32TriadPolicy::ExactScalarFmaV1),
        );
        let qualified = qualify_physical_launch(&runtime.ctx, request)?;
        eprintln!(
            "nt_prism production nodes={} digest={:02x?}",
            qualified.evidence().launch_count(),
            qualified.evidence().launch_digest()
        );
        for (index, node) in qualified.evidence().nodes().iter().enumerate() {
            eprintln!(
                "nt_prism production node={} symbol={} grid={:?} block={:?} args={:02x?}",
                index,
                node.symbol,
                node.launch.grid_dim,
                node.launch.block_dim,
                node.launch.arguments_digest
            );
        }
        let evidence = qualified.evidence();
        if !evidence.eager_graph_equal() {
            return Err("production eager/graph physical nodes differ".into());
        }
        let nodes = evidence.nodes();
        if nodes.len() != 2 || evidence.launch_count() != 2 {
            return Err(format!(
                "production prism route must contain two ordered nodes, found {}",
                nodes.len()
            ));
        }
        if nodes.iter().any(|node| {
            node.module_kind != ModuleKind::TriadScalar
                || node.logical_op != ResolvedGemmOp::Nt
                || node.shape != DIMS
                || node.strides != (1_928, 1_928, 384)
                || node.launch.arguments_digest == [0; 32]
        }) || nodes[0].symbol != TRANSPOSE_32X16
            || nodes[0].tile != Some((32, 32))
            || (
                nodes[0].launch.grid_dim,
                nodes[0].launch.block_dim,
                nodes[0].launch.shared_mem_bytes,
            ) != ((61, 12, 1), (32, 16, 1), 0)
            || nodes[1].symbol != PRODUCTION_PRISM_M64_SYMBOL
            || nodes[1].tile != Some((64, 64))
            || (
                nodes[1].launch.grid_dim,
                nodes[1].launch.block_dim,
                nodes[1].launch.shared_mem_bytes,
            ) != ((438, 1, 1), (128, 1, 1), M64_SHARED as u32)
            || nodes[0].launch.arguments_digest == nodes[1].launch.arguments_digest
            || evidence.launch_digest() == [0; 32]
        {
            return Err(format!(
                "production prism physical identity changed: {nodes:?}"
            ));
        }
        qualified.validate_red_zones(&runtime.ctx)?;
        Ok(())
    }

    #[test]
    fn candidate_identity_gate_rejects_count_order_zero_and_collisions() {
        let transpose = CandidateNodeIdentity {
            symbol: TRANSPOSE_32X16.into(),
            grid_dim: (61, 12, 1),
            block_dim: (32, 16, 1),
            shared_mem_bytes: 0,
            arguments_digest: [1; 32],
        };
        let m64 = CandidateNodeIdentity {
            symbol: M64_SYMBOL.into(),
            grid_dim: (438, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: M64_SHARED as u32,
            arguments_digest: [2; 32],
        };
        assert!(validate_candidate_identity(&[transpose.clone(), m64.clone()], M64_SYMBOL).is_ok());
        assert!(validate_candidate_identity(std::slice::from_ref(&transpose), M64_SYMBOL).is_err());
        assert!(
            validate_candidate_identity(&[m64.clone(), transpose.clone()], M64_SYMBOL).is_err()
        );
        let mut zero = transpose.clone();
        zero.arguments_digest = [0; 32];
        assert!(validate_candidate_identity(&[zero, m64.clone()], M64_SYMBOL).is_err());
        let mut collision = m64;
        collision.arguments_digest = transpose.arguments_digest;
        assert!(validate_candidate_identity(&[transpose, collision], M64_SYMBOL).is_err());

        let mutations: [fn(&mut [CandidateNodeIdentity; 2]); 6] = [
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].grid_dim.0 += 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].block_dim.1 -= 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[0].shared_mem_bytes += 4,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].grid_dim.0 += 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].block_dim.0 -= 1,
            |nodes: &mut [CandidateNodeIdentity; 2]| nodes[1].shared_mem_bytes -= 4,
        ];
        for mutate in mutations {
            let mut nodes = [
                CandidateNodeIdentity {
                    symbol: TRANSPOSE_32X16.into(),
                    grid_dim: (61, 12, 1),
                    block_dim: (32, 16, 1),
                    shared_mem_bytes: 0,
                    arguments_digest: [1; 32],
                },
                CandidateNodeIdentity {
                    symbol: M64_SYMBOL.into(),
                    grid_dim: (438, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: M64_SHARED as u32,
                    arguments_digest: [2; 32],
                },
            ];
            mutate(&mut nodes);
            assert!(validate_candidate_identity(&nodes, M64_SYMBOL).is_err());
        }
    }

    #[test]
    fn specialized_arm_has_a_distinct_exact_two_node_identity() {
        assert!(Arm::Specialized.is_candidate());
        assert_eq!(Arm::Candidate.m64_symbol(), M64_SYMBOL);
        assert_eq!(Arm::Specialized.m64_symbol(), PRISM_SPECIALIZED_M64_SYMBOL);
        assert_ne!(Arm::Specialized.m64_symbol(), Arm::Candidate.m64_symbol());
        let nodes = [
            CandidateNodeIdentity {
                symbol: TRANSPOSE_32X16.into(),
                grid_dim: (61, 12, 1),
                block_dim: (32, 16, 1),
                shared_mem_bytes: 0,
                arguments_digest: [1; 32],
            },
            CandidateNodeIdentity {
                symbol: PRISM_SPECIALIZED_M64_SYMBOL.into(),
                grid_dim: (438, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: M64_SHARED as u32,
                arguments_digest: [2; 32],
            },
        ];
        assert!(validate_candidate_identity(&nodes, PRISM_SPECIALIZED_M64_SYMBOL).is_ok());
        assert!(validate_candidate_identity(&nodes, M64_SYMBOL).is_err());

        assert!(Arm::SpecializedVector.is_candidate());
        assert_eq!(
            Arm::SpecializedVector.m64_symbol(),
            PRISM_SPECIALIZED_VECTOR_M64_SYMBOL
        );
        assert_ne!(
            Arm::SpecializedVector.m64_symbol(),
            Arm::Specialized.m64_symbol()
        );
        let mut vector_nodes = nodes;
        vector_nodes[1].symbol = PRISM_SPECIALIZED_VECTOR_M64_SYMBOL.into();
        assert!(
            validate_candidate_identity(&vector_nodes, PRISM_SPECIALIZED_VECTOR_M64_SYMBOL).is_ok()
        );
        assert!(validate_candidate_identity(&vector_nodes, PRISM_SPECIALIZED_M64_SYMBOL).is_err());
    }

    #[test]
    fn performance_gate_enforces_each_denominator_contract_at_its_boundary() {
        assert!(
            validate_performance_gate(
                Arm::Production,
                Arm::Specialized,
                "ABBA",
                MIN_WINDOWS,
                SPECIALIZED_MIN_P05_SPEEDUP,
                SPECIALIZED_MIN_P50_SPEEDUP,
                1.02,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::Production,
                Arm::SpecializedVector,
                "BAAB",
                MIN_WINDOWS,
                0.99,
                1.0,
                1.01,
            )
            .is_ok()
        );
        for (p05, p50) in [
            (
                SPECIALIZED_MIN_P05_SPEEDUP - 0.001,
                SPECIALIZED_MIN_P50_SPEEDUP,
            ),
            (
                SPECIALIZED_MIN_P05_SPEEDUP,
                SPECIALIZED_MIN_P50_SPEEDUP - 0.001,
            ),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::Production,
                    Arm::Specialized,
                    "BAAB",
                    MIN_WINDOWS,
                    p05,
                    p50,
                    1.02,
                )
                .is_err()
            );
        }

        assert!(
            validate_performance_gate(
                Arm::GenericNt,
                Arm::Specialized,
                "ABBA",
                MIN_WINDOWS,
                SCALAR_MIN_P05_SPEEDUP,
                SCALAR_MIN_P50_SPEEDUP,
                1.30,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::GenericNt,
                Arm::SpecializedVector,
                "BAAB",
                MIN_WINDOWS,
                SCALAR_MIN_P05_SPEEDUP,
                SCALAR_MIN_P50_SPEEDUP,
                1.30,
            )
            .is_ok()
        );
        for (p05, p50) in [
            (SCALAR_MIN_P05_SPEEDUP - 0.001, SCALAR_MIN_P50_SPEEDUP),
            (SCALAR_MIN_P05_SPEEDUP, SCALAR_MIN_P50_SPEEDUP - 0.001),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::GenericNt,
                    Arm::Specialized,
                    "BAAB",
                    MIN_WINDOWS,
                    p05,
                    p50,
                    1.30,
                )
                .is_err()
            );
        }

        assert!(
            validate_performance_gate(
                Arm::CublasPedantic,
                Arm::Specialized,
                "ABBA",
                MIN_WINDOWS,
                PEDANTIC_MIN_P05_SPEEDUP,
                PEDANTIC_MIN_P50_SPEEDUP,
                0.95,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::CublasPedantic,
                Arm::SpecializedVector,
                "BAAB",
                MIN_WINDOWS,
                PEDANTIC_MIN_P05_SPEEDUP,
                PEDANTIC_MIN_P50_SPEEDUP,
                0.95,
            )
            .is_ok()
        );
        for (p05, p50) in [
            (PEDANTIC_MIN_P05_SPEEDUP - 0.001, PEDANTIC_MIN_P50_SPEEDUP),
            (PEDANTIC_MIN_P05_SPEEDUP, PEDANTIC_MIN_P50_SPEEDUP - 0.001),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::CublasPedantic,
                    Arm::Specialized,
                    "BAAB",
                    MIN_WINDOWS,
                    p05,
                    p50,
                    0.95,
                )
                .is_err()
            );
        }
        assert!(
            validate_performance_gate(
                Arm::CublasFast,
                Arm::Specialized,
                "ABBA",
                MIN_WINDOWS,
                0.001,
                0.002,
                0.003,
            )
            .is_ok()
        );
        assert!(
            validate_performance_gate(
                Arm::CublasFast,
                Arm::SpecializedVector,
                "BAAB",
                MIN_WINDOWS,
                0.001,
                0.002,
                0.003,
            )
            .is_ok()
        );

        assert!(
            validate_performance_gate(
                Arm::GenericNt,
                Arm::Production,
                "ABBA",
                MIN_WINDOWS,
                SCALAR_MIN_P05_SPEEDUP,
                SCALAR_MIN_P50_SPEEDUP,
                1.30,
            )
            .is_ok()
        );
        for (p05, p50) in [
            (SCALAR_MIN_P05_SPEEDUP - 0.001, SCALAR_MIN_P50_SPEEDUP),
            (SCALAR_MIN_P05_SPEEDUP, SCALAR_MIN_P50_SPEEDUP - 0.001),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::GenericNt,
                    Arm::Production,
                    "ABBA",
                    MIN_WINDOWS,
                    p05,
                    p50,
                    1.30,
                )
                .is_err()
            );
        }

        for candidate in [Arm::Candidate, Arm::SpecializedVector] {
            assert!(
                validate_performance_gate(
                    Arm::Production,
                    candidate,
                    "ABBA",
                    MIN_WINDOWS,
                    0.99,
                    1.0,
                    1.01,
                )
                .is_ok()
            );
        }
        for (p05, p50, p95) in [
            (PARITY_MIN_P05 - 0.001, 1.0, 1.01),
            (0.99, PARITY_MIN_P50 - 0.001, 1.01),
            (0.99, PARITY_MAX_P50 + 0.001, 1.01),
            (0.99, 1.0, PARITY_MAX_P95 + 0.001),
            (1.20, 1.25, 1.30),
        ] {
            for candidate in [Arm::Candidate, Arm::SpecializedVector] {
                assert!(
                    validate_performance_gate(
                        Arm::Production,
                        candidate,
                        "ABBA",
                        MIN_WINDOWS,
                        p05,
                        p50,
                        p95,
                    )
                    .is_err()
                );
            }
        }

        assert!(
            validate_performance_gate(
                Arm::CublasPedantic,
                Arm::Production,
                "BAAB",
                MIN_WINDOWS,
                0.90,
                0.92,
                0.95,
            )
            .is_ok()
        );
        for (p05, p50) in [(0.899_999, 0.92), (0.90, 0.919_999)] {
            assert!(
                validate_performance_gate(
                    Arm::CublasPedantic,
                    Arm::Production,
                    "BAAB",
                    MIN_WINDOWS,
                    p05,
                    p50,
                    0.95,
                )
                .is_err()
            );
        }
        assert!(
            validate_performance_gate(
                Arm::CublasFast,
                Arm::Production,
                "ABBA",
                MIN_WINDOWS,
                0.001,
                0.002,
                0.003,
            )
            .is_ok()
        );

        for (order, windows, p05, p50, p95) in [
            ("AABB", MIN_WINDOWS, 1.0, 1.02, 1.04),
            ("ABBA", MIN_WINDOWS - 1, 1.0, 1.02, 1.04),
            ("ABBA", MIN_WINDOWS, f64::NAN, 1.02, 1.04),
            ("ABBA", MIN_WINDOWS, 1.0, f64::INFINITY, 1.04),
            ("ABBA", MIN_WINDOWS, 1.0, 1.02, 0.0),
        ] {
            assert!(
                validate_performance_gate(
                    Arm::GenericNt,
                    Arm::Production,
                    order,
                    windows,
                    p05,
                    p50,
                    p95,
                )
                .is_err()
            );
        }
        assert!(
            validate_performance_gate(
                Arm::Candidate,
                Arm::GenericNt,
                "ABBA",
                MIN_WINDOWS,
                1.0,
                1.0,
                1.0,
            )
            .is_err()
        );
    }

    #[test]
    fn paired_timing_uses_one_nonzero_iteration_count_for_both_arms() {
        assert_eq!(paired_iteration_count(7, 11).unwrap(), 11);
        assert_eq!(paired_iteration_count(13, 5).unwrap(), 13);
        assert!(paired_iteration_count(0, 5).is_err());
        assert!(paired_iteration_count(5, 0).is_err());
    }

    #[test]
    fn official_timing_rejects_debug_builds() {
        assert!(validate_release_build(false).is_ok());
        assert!(validate_release_build(true).is_err());
    }

    #[test]
    fn window_count_parser_defaults_only_when_absent_and_rejects_bad_values() {
        use std::env::VarError;
        use std::ffi::OsString;

        assert_eq!(parse_windows_env(Err(VarError::NotPresent)).unwrap(), 101);
        assert_eq!(parse_windows_env(Ok("101".into())).unwrap(), 101);
        assert_eq!(parse_windows_env(Ok("151".into())).unwrap(), 151);
        for value in ["", "not-a-number", "100"] {
            assert!(parse_windows_env(Ok(value.into())).is_err());
        }
        assert!(parse_windows_env(Err(VarError::NotUnicode(OsString::from("invalid")))).is_err());
    }

    #[test]
    fn percentile_and_paired_sample_reject_invalid_measurements() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 0.50).unwrap(), 2.0);
        assert!(percentile(&[], 0.50).is_err());
        for samples in [vec![0.0], vec![-1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(percentile(&samples, 0.50).is_err());
        }
        for fraction in [0.0, 1.01, f64::NAN, f64::INFINITY] {
            assert!(percentile(&[1.0], fraction).is_err());
        }

        assert_eq!(paired_sample(2.0, 4.0, 1.0, 3.0).unwrap(), (3.0, 2.0, 1.5));
        for sample in [
            (0.0, 4.0, 1.0, 3.0),
            (f64::NAN, 4.0, 1.0, 3.0),
            (2.0, 4.0, -1.0, 3.0),
            (f64::MAX, f64::MAX, f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        ] {
            assert!(paired_sample(sample.0, sample.1, sample.2, sample.3).is_err());
        }
    }

    #[test]
    fn performance_failure_aggregation_preserves_every_reported_pair() {
        assert!(finish_performance_failures("orders", Vec::new()).is_ok());
        let error = finish_performance_failures(
            "pairs",
            vec!["production failed".into(), "pedantic failed".into()],
        )
        .unwrap_err();
        assert!(error.contains("production failed"));
        assert!(error.contains("pedantic failed"));
    }

    #[test]
    fn cublas_nt_geometry_rejects_op_dimension_stride_and_output_mutations() {
        let logical = (3, 2, 4);
        let expected = CublasNtGeometry {
            logical_op: ResolvedGemmOp::Nt,
            trans_a: CublasTranspose::Transpose,
            trans_b: CublasTranspose::None,
            m: 2,
            n: 3,
            k: 4,
            lda: 4,
            ldb: 4,
            ldc: 2,
            output_rows: 3,
            output_columns: 2,
            output_elements: 6,
        };
        assert_eq!(cublas_nt_geometry(logical).unwrap(), expected);
        assert!(validate_cublas_nt_geometry(logical, expected).is_ok());

        let mutations: [fn(&mut CublasNtGeometry); 12] = [
            |geometry| geometry.logical_op = ResolvedGemmOp::Nn,
            |geometry| geometry.trans_a = CublasTranspose::None,
            |geometry| geometry.trans_b = CublasTranspose::Transpose,
            |geometry| geometry.m += 1,
            |geometry| geometry.n += 1,
            |geometry| geometry.k += 1,
            |geometry| geometry.lda += 1,
            |geometry| geometry.ldb += 1,
            |geometry| geometry.ldc += 1,
            |geometry| geometry.output_rows += 1,
            |geometry| geometry.output_columns += 1,
            |geometry| geometry.output_elements += 1,
        ];
        for mutate in mutations {
            let mut mutated = expected;
            mutate(&mut mutated);
            assert!(validate_cublas_nt_geometry(logical, mutated).is_err());
        }
    }

    #[test]
    fn qualified_environment_rejects_scalar_compiler_and_artifact_mismatches() {
        let compiler = CompilerIdentity {
            source_digest: [1; 32],
            invocation_digest: [2; 32],
            header_manifest_digest: [3; 32],
            target: CudaTarget::new("compute_120").unwrap(),
            nvrtc_version: (13, 2),
            nvrtc_library_domain: [4; 32],
            nvrtc_library_known: true,
            output_kind: ArtifactKind::Ptx,
            composer_revision: COMPOSER_REVISION,
            compiler_revision: COMPILER_REVISION,
            numeric_abi_revision: NUMERIC_ABI_REVISION,
            schedule_revision: SCHEDULE_REVISION,
        };
        let artifact = ArtifactIdentity {
            module_kind: ModuleKind::TriadScalar,
            artifact_kind: ArtifactKind::Ptx,
            compile_key: compiler.invocation_digest,
            artifact_digest: [5; 32],
        };
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, artifact).is_ok()
        );
        assert!(
            validate_qualified_environment((8, 9), 128, "compute_89", compiler, artifact).is_err()
        );
        let mut wrong_compiler = compiler;
        wrong_compiler.nvrtc_version = (13, 1);
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", wrong_compiler, artifact)
                .is_err()
        );
        let mut unknown_library = compiler;
        unknown_library.nvrtc_library_known = false;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", unknown_library, artifact)
                .is_err()
        );
        let compiler_mutations: [fn(&mut CompilerIdentity); 8] = [
            |identity: &mut CompilerIdentity| identity.source_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.invocation_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.header_manifest_digest = [0; 32],
            |identity: &mut CompilerIdentity| identity.nvrtc_library_domain = [0; 32],
            |identity: &mut CompilerIdentity| identity.composer_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.compiler_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.numeric_abi_revision ^= 1,
            |identity: &mut CompilerIdentity| identity.schedule_revision ^= 1,
        ];
        for mutate in compiler_mutations {
            let mut mutated = compiler;
            mutate(&mut mutated);
            assert!(
                validate_qualified_environment((12, 0), 170, "compute_120", mutated, artifact)
                    .is_err()
            );
        }
        let mut wrong_artifact = artifact;
        wrong_artifact.compile_key = [9; 32];
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.artifact_kind = ArtifactKind::Cubin;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.module_kind = ModuleKind::TriadSm80;
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
        wrong_artifact = artifact;
        wrong_artifact.artifact_digest = [0; 32];
        assert!(
            validate_qualified_environment((12, 0), 170, "compute_120", compiler, wrong_artifact)
                .is_err()
        );
    }

    fn parsed_registers(resource_block: &str) -> usize {
        let (_, tail) = resource_block
            .split_once("Used ")
            .expect("ptxas register line");
        tail.split_whitespace()
            .next()
            .expect("ptxas register count")
            .parse()
            .expect("numeric ptxas register count")
    }

    #[test]
    #[ignore = "requires CUDA NVRTC and ptxas but launches no GPU work"]
    fn prism_sources_compile_with_bounded_sm89_and_sm120_resources() {
        for (arch, sm, contracts) in [
            (
                "compute_120",
                "sm_120",
                [
                    (GENERIC_NT_SYMBOL, 123, 0),
                    (M64_SYMBOL, 101, 0),
                    (PRODUCTION_PRISM_M64_SYMBOL, 101, 0),
                    (PRISM_SPECIALIZED_M64_SYMBOL, 101, 0),
                    (PRISM_SPECIALIZED_VECTOR_M64_SYMBOL, 101, 0),
                    (TRANSPOSE_32X16, 24, TRANSPOSE_STATIC_SHARED),
                ],
            ),
            (
                "compute_89",
                "sm_89",
                [
                    (GENERIC_NT_SYMBOL, 128, 0),
                    (M64_SYMBOL, 120, 0),
                    (PRODUCTION_PRISM_M64_SYMBOL, 123, 0),
                    (PRISM_SPECIALIZED_M64_SYMBOL, 120, 0),
                    (PRISM_SPECIALIZED_VECTOR_M64_SYMBOL, 123, 0),
                    (TRANSPOSE_32X16, 18, TRANSPOSE_STATIC_SHARED),
                ],
            ),
        ] {
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some(arch),
                options: vec![
                    "--fmad=true".to_owned(),
                    "--extra-device-vectorization".to_owned(),
                    "-DNDEBUG".to_owned(),
                    "-DGEMM_BI_GROUP_M=16".to_owned(),
                ],
                include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(compose_source(), options)
                .unwrap_or_else(|error| panic!("{arch} compile failed: {error:?}"))
                .to_src();
            assert!(ptx.contains("fma.rn.f32"));
            let stem = format!("mamba-rs-prism-{}-{sm}", std::process::id());
            let ptx_path = std::env::temp_dir().join(format!("{stem}.ptx"));
            let cubin_path = std::env::temp_dir().join(format!("{stem}.cubin"));
            std::fs::write(&ptx_path, &ptx).expect("write temporary PTX");
            let ptxas = std::env::var("CUDA_HOME")
                .map(|root| std::path::PathBuf::from(root).join("bin/ptxas"))
                .unwrap_or_else(|_| "ptxas".into());
            let output = std::process::Command::new(ptxas)
                .arg("--verbose")
                .arg(format!("--gpu-name={sm}"))
                .arg(&ptx_path)
                .arg("--output-file")
                .arg(&cubin_path)
                .output()
                .unwrap_or_else(|error| panic!("launch ptxas for {sm}: {error}"));
            let log = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.status.success(), "ptxas {sm} failed:\n{log}");
            for (symbol, register_cap, static_shared) in contracts {
                let marker = format!("Compiling entry function '{symbol}'");
                let block = log
                    .split_once(&marker)
                    .unwrap_or_else(|| panic!("{sm} resource log lost {symbol}"))
                    .1
                    .split("Compiling entry function '")
                    .next()
                    .expect("resource block");
                assert!(
                    block
                        .contains("0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"),
                    "{sm} {symbol} local/spill regression:\n{block}"
                );
                let registers = parsed_registers(block);
                eprintln!(
                    "nt_prism_compile_resource sm={sm} symbol={symbol} registers={registers} static_shared={static_shared} stack=0 spill_stores=0 spill_loads=0"
                );
                assert!(
                    registers <= register_cap,
                    "{sm} {symbol} exceeds {register_cap} registers:\n{block}"
                );
                if static_shared == 0 {
                    assert!(!block.contains("bytes smem"), "{sm} {symbol}:\n{block}");
                } else {
                    assert!(
                        block.contains(&format!("{static_shared} bytes smem")),
                        "{sm} {symbol}:\n{block}"
                    );
                }
            }
            let _ = std::fs::remove_file(ptx_path);
            let _ = std::fs::remove_file(cubin_path);
        }
    }

    #[test]
    #[ignore = "requires an exclusive CC12.0/170-SM GPU"]
    fn prism_transpose_candidates_are_exact_and_graph_stable() -> Result<(), String> {
        let runtime = new_runtime()?;
        validate_physical_production(&runtime)?;
        validate_cublas_orientation(&runtime)?;
        check_resources(
            &runtime.generic_nt,
            ResourceContract {
                threads: 256,
                static_shared: 0,
                dynamic_shared: NT_SHARED,
                register_cap: 123,
                minimum_occupancy: 2,
            },
        )?;
        check_resources(
            &runtime.m64,
            ResourceContract {
                threads: 128,
                static_shared: 0,
                dynamic_shared: M64_SHARED,
                register_cap: 103,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.specialized_m64,
            ResourceContract {
                threads: 128,
                static_shared: 0,
                dynamic_shared: M64_SHARED,
                register_cap: 103,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.specialized_vector_m64,
            ResourceContract {
                threads: 128,
                static_shared: 0,
                dynamic_shared: M64_SHARED,
                register_cap: 103,
                minimum_occupancy: 4,
            },
        )?;
        check_resources(
            &runtime.transpose16,
            ResourceContract {
                threads: 512,
                static_shared: TRANSPOSE_STATIC_SHARED,
                dynamic_shared: 0,
                register_cap: 28,
                minimum_occupancy: 2,
            },
        )?;
        let mut fixture = new_fixture(&runtime)?;
        launch_arm(&runtime, &mut fixture, Arm::GenericNt)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("generic sync: {error:?}"))?;
        let exact = bits(&runtime, &fixture, Arm::GenericNt)?;

        for arm in [Arm::Candidate, Arm::Specialized, Arm::SpecializedVector] {
            let m64 = match arm {
                Arm::Candidate => &runtime.m64,
                Arm::Specialized => &runtime.specialized_m64,
                Arm::SpecializedVector => &runtime.specialized_vector_m64,
                _ => unreachable!(),
            };
            eprintln!(
                "candidate route arm={} nodes=[{},{}]",
                arm.name(),
                runtime.transpose16.symbol,
                m64.symbol
            );
            for repeat in 0..3 {
                launch_arm(&runtime, &mut fixture, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} eager: {error:?}"))?;
                if bits(&runtime, &fixture, arm)? != exact {
                    return Err(format!(
                        "{arm:?} eager repeat {repeat} differs from ascending generic NT"
                    ));
                }
                validate_scratch_transpose(&runtime, &fixture)?;
            }
            let graph = capture_arm(&runtime, &mut fixture, arm)?;
            let eager_identity = eager_candidate_identity(&runtime, &fixture, arm);
            validate_candidate_identity(&eager_identity, arm.m64_symbol())?;
            let graph_identity = graph_candidate_identity(&graph, arm.m64_symbol())?;
            if eager_identity != graph_identity {
                return Err(format!(
                    "candidate eager/graph physical identity differs: eager={eager_identity:?} graph={graph_identity:?}"
                ));
            }
            eprintln!("candidate eager/graph identity={graph_identity:?}");
            for repeat in 0..3 {
                graph
                    .launch()
                    .map_err(|error| format!("{arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} graph sync: {error:?}"))?;
                if bits(&runtime, &fixture, arm)? != exact {
                    return Err(format!("{arm:?} graph repeat {repeat} differs from eager"));
                }
                validate_scratch_transpose(&runtime, &fixture)?;
            }
        }

        launch_arm(&runtime, &mut fixture, Arm::Production)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("production sync: {error:?}"))?;
        let production = bits(&runtime, &fixture, Arm::Production)?;
        if production != exact {
            return Err("production output differs from ascending generic NT".into());
        }
        for arm in [Arm::GenericNt, Arm::Production] {
            let reference = if arm == Arm::GenericNt {
                &exact
            } else {
                &production
            };
            let graph = capture_arm(&runtime, &mut fixture, arm)?;
            for repeat in 0..3 {
                launch_arm(&runtime, &mut fixture, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} repeat: {error:?}"))?;
                if &bits(&runtime, &fixture, arm)? != reference {
                    return Err(format!("{arm:?} eager repeat {repeat} changed"));
                }
                graph
                    .launch()
                    .map_err(|error| format!("{arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("{arm:?} graph sync: {error:?}"))?;
                if &bits(&runtime, &fixture, arm)? != reference {
                    return Err(format!("{arm:?} graph repeat {repeat} changed"));
                }
            }
        }
        fixture.a.validate_unchanged(&runtime.ctx.stream, "A")?;
        fixture.b.validate_unchanged(&runtime.ctx.stream, "B")?;
        let production_a = fixture.production_a.to_cpu(&runtime.ctx.stream)?;
        if production_a
            .iter()
            .zip(&fixture.production_a_expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("production A changed".into());
        }

        let mut exceptional = exceptional_fixture(&runtime)?;
        launch_arm(&runtime, &mut exceptional, Arm::GenericNt)?;
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("exceptional generic sync: {error:?}"))?;
        let exceptional_exact = bits(&runtime, &exceptional, Arm::GenericNt)?;
        validate_exceptional_oracle(&exceptional_exact)?;
        for arm in [
            Arm::Production,
            Arm::GenericNt,
            Arm::Candidate,
            Arm::Specialized,
            Arm::SpecializedVector,
        ] {
            launch_arm(&runtime, &mut exceptional, arm)?;
            runtime
                .ctx
                .stream
                .synchronize()
                .map_err(|error| format!("exceptional {arm:?} warmup sync: {error:?}"))?;
            if bits(&runtime, &exceptional, arm)? != exceptional_exact {
                return Err(format!(
                    "exceptional {arm:?} eager warmup differs from generic NT"
                ));
            }
            if arm.is_candidate() {
                validate_scratch_transpose(&runtime, &exceptional)?;
            }
            let graph = capture_arm(&runtime, &mut exceptional, arm)?;
            let mut repeated: Option<Vec<u32>> = None;
            for repeat in 0..3 {
                launch_arm(&runtime, &mut exceptional, arm)?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("exceptional {arm:?} eager sync: {error:?}"))?;
                let eager = bits(&runtime, &exceptional, arm)?;
                if eager != exceptional_exact {
                    return Err(format!(
                        "exceptional {arm:?} eager repeat {repeat} differs from generic NT"
                    ));
                }
                if arm.is_candidate() {
                    validate_scratch_transpose(&runtime, &exceptional)?;
                }
                graph
                    .launch()
                    .map_err(|error| format!("exceptional {arm:?} graph: {error:?}"))?;
                runtime
                    .ctx
                    .stream
                    .synchronize()
                    .map_err(|error| format!("exceptional {arm:?} graph sync: {error:?}"))?;
                let graph_bits = bits(&runtime, &exceptional, arm)?;
                if graph_bits != eager {
                    return Err(format!(
                        "exceptional {arm:?} graph repeat {repeat} differs from eager"
                    ));
                }
                if repeated.as_ref().is_some_and(|prior| prior != &eager) {
                    return Err(format!("exceptional {arm:?} repeated bits changed"));
                }
                repeated.get_or_insert(eager);
                if arm.is_candidate() {
                    validate_scratch_transpose(&runtime, &exceptional)?;
                }
            }
        }
        exceptional
            .a
            .validate_unchanged(&runtime.ctx.stream, "exceptional A")?;
        exceptional
            .b
            .validate_unchanged(&runtime.ctx.stream, "exceptional B")?;
        let exceptional_production_a = exceptional.production_a.to_cpu(&runtime.ctx.stream)?;
        if exceptional_production_a
            .iter()
            .zip(&exceptional.production_a_expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            return Err("exceptional production A changed".into());
        }
        Ok(())
    }

    fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
        if values.is_empty() {
            return Err("percentile requires at least one sample".into());
        }
        if !fraction.is_finite() || !(0.0 < fraction && fraction <= 1.0) {
            return Err(format!("percentile fraction is invalid: {fraction}"));
        }
        if values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("percentile samples must be finite and positive".into());
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let index = ((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);
        Ok(sorted[index])
    }

    fn paired_sample(
        baseline_first: f64,
        baseline_second: f64,
        candidate_first: f64,
        candidate_second: f64,
    ) -> Result<(f64, f64, f64), String> {
        if [
            baseline_first,
            baseline_second,
            candidate_first,
            candidate_second,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("paired timing inputs must be finite and positive".into());
        }
        let baseline_us = 0.5 * baseline_first + 0.5 * baseline_second;
        let candidate_us = 0.5 * candidate_first + 0.5 * candidate_second;
        let speedup = baseline_us / candidate_us;
        if [baseline_us, candidate_us, speedup]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(format!(
                "paired timing result is invalid: baseline={baseline_us} candidate={candidate_us} speedup={speedup}"
            ));
        }
        Ok((baseline_us, candidate_us, speedup))
    }

    fn measure(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("timing requires at least one iteration".into());
        }
        let start = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("start: {error:?}"))?;
        for _ in 0..iterations {
            launch_arm(runtime, fixture, arm)?;
        }
        let end = runtime
            .ctx
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("end: {error:?}"))?;
        let elapsed_us = f64::from(
            start
                .elapsed_ms(&end)
                .map_err(|error| format!("elapsed: {error:?}"))?,
        ) * 1_000.0
            / iterations as f64;
        if !elapsed_us.is_finite() || elapsed_us <= 0.0 {
            return Err(format!(
                "{} produced an invalid elapsed time {elapsed_us}",
                arm.name()
            ));
        }
        Ok(elapsed_us)
    }

    fn calibrated_iterations(
        runtime: &Runtime,
        fixture: &mut Fixture,
        arm: Arm,
    ) -> Result<usize, String> {
        let pilot = measure(runtime, fixture, arm, 3)?;
        if !pilot.is_finite() || pilot <= 0.0 {
            return Err(format!("{} calibration is invalid: {pilot}", arm.name()));
        }
        let iterations = (TARGET_WINDOW_US / pilot).round().clamp(3.0, 500.0) as usize;
        if iterations == 0 {
            return Err(format!("{} calibrated to zero iterations", arm.name()));
        }
        Ok(iterations)
    }

    fn paired_iteration_count(
        baseline_iterations: usize,
        candidate_iterations: usize,
    ) -> Result<usize, String> {
        if baseline_iterations == 0 || candidate_iterations == 0 {
            return Err("paired timing requires nonzero calibrated iterations".into());
        }
        Ok(baseline_iterations.max(candidate_iterations))
    }

    fn validate_release_build(debug_assertions: bool) -> Result<(), String> {
        if debug_assertions {
            return Err("official timing requires a release build".into());
        }
        Ok(())
    }

    fn validate_performance_gate(
        baseline: Arm,
        candidate: Arm,
        order: &str,
        windows: usize,
        speedup_p05: f64,
        speedup_p50: f64,
        speedup_p95: f64,
    ) -> Result<(), String> {
        if !matches!(order, "ABBA" | "BAAB") {
            return Err(format!("unsupported timing order {order}"));
        }
        if windows < MIN_WINDOWS {
            return Err(format!(
                "timing has {windows} windows, requires at least {MIN_WINDOWS}"
            ));
        }
        if [speedup_p05, speedup_p50, speedup_p95]
            .into_iter()
            .any(|ratio| !ratio.is_finite() || ratio <= 0.0)
        {
            return Err(format!(
                "non-finite or non-positive speedup in {order}: p05={speedup_p05} p50={speedup_p50} p95={speedup_p95}"
            ));
        }
        match (baseline, candidate) {
            (Arm::GenericNt, Arm::Production | Arm::Specialized | Arm::SpecializedVector) => {
                if speedup_p05 < SCALAR_MIN_P05_SPEEDUP || speedup_p50 < SCALAR_MIN_P50_SPEEDUP {
                    return Err(format!(
                        "exact scalar speedup failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9}; required p05>={SCALAR_MIN_P05_SPEEDUP:.2} p50>={SCALAR_MIN_P50_SPEEDUP:.2}"
                    ));
                }
            }
            (Arm::Production, Arm::Specialized) => {
                if speedup_p05 < SPECIALIZED_MIN_P05_SPEEDUP
                    || speedup_p50 < SPECIALIZED_MIN_P50_SPEEDUP
                {
                    return Err(format!(
                        "specialized production speedup failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9}; required p05>={SPECIALIZED_MIN_P05_SPEEDUP:.3} p50>={SPECIALIZED_MIN_P50_SPEEDUP:.2}"
                    ));
                }
            }
            (Arm::Production, Arm::Candidate | Arm::SpecializedVector) => {
                if speedup_p05 < PARITY_MIN_P05
                    || !(PARITY_MIN_P50..=PARITY_MAX_P50).contains(&speedup_p50)
                    || speedup_p95 > PARITY_MAX_P95
                {
                    return Err(format!(
                        "production/direct candidate parity failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9} p95={speedup_p95:.9}; required p05>={PARITY_MIN_P05:.3}, {PARITY_MIN_P50:.3}<=p50<={PARITY_MAX_P50:.3}, p95<={PARITY_MAX_P95:.3}"
                    ));
                }
            }
            (Arm::CublasPedantic, Arm::Production | Arm::Specialized | Arm::SpecializedVector) => {
                if speedup_p05 < PEDANTIC_MIN_P05_SPEEDUP || speedup_p50 < PEDANTIC_MIN_P50_SPEEDUP
                {
                    return Err(format!(
                        "exact scalar pedantic guard failed in {order}: p05={speedup_p05:.9} p50={speedup_p50:.9}; required p05>={PEDANTIC_MIN_P05_SPEEDUP:.2} p50>={PEDANTIC_MIN_P50_SPEEDUP:.2}"
                    ));
                }
            }
            (Arm::CublasFast, Arm::Production | Arm::Specialized | Arm::SpecializedVector) => {}
            _ => {
                return Err(format!(
                    "unsupported prism performance pair: {} -> {}",
                    baseline.name(),
                    candidate.name()
                ));
            }
        }
        Ok(())
    }

    fn parse_windows_env(value: Result<String, std::env::VarError>) -> Result<usize, String> {
        let windows = match value {
            Ok(raw) => raw
                .parse::<usize>()
                .map_err(|error| format!("invalid MAMBA_RS_NT_PRISM_WINDOWS={raw:?}: {error}"))?,
            Err(std::env::VarError::NotPresent) => MIN_WINDOWS,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("MAMBA_RS_NT_PRISM_WINDOWS is not valid Unicode".into());
            }
        };
        if windows < MIN_WINDOWS {
            return Err(format!(
                "timing has {windows} windows, requires at least {MIN_WINDOWS}"
            ));
        }
        Ok(windows)
    }

    fn finish_performance_failures(scope: &str, failures: Vec<String>) -> Result<(), String> {
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "prism performance {scope} failed:\n{}",
                failures.join("\n")
            ))
        }
    }

    fn paired(
        runtime: &Runtime,
        fixture: &mut Fixture,
        baseline: Arm,
        candidate: Arm,
        windows: usize,
    ) -> Result<(), String> {
        if windows < MIN_WINDOWS {
            return Err(format!(
                "timing has {windows} windows, requires at least {MIN_WINDOWS}"
            ));
        }
        let baseline_iterations = calibrated_iterations(runtime, fixture, baseline)?;
        let candidate_iterations = calibrated_iterations(runtime, fixture, candidate)?;
        let iterations = paired_iteration_count(baseline_iterations, candidate_iterations)?;
        let mut order_failures = Vec::new();
        for (order, baseline_first) in [("ABBA", true), ("BAAB", false)] {
            let mut baseline_samples = Vec::with_capacity(windows);
            let mut candidate_samples = Vec::with_capacity(windows);
            let mut ratios = Vec::with_capacity(windows);
            for _ in 0..windows {
                let (b0, b1, c0, c1) = if baseline_first {
                    let b0 = measure(runtime, fixture, baseline, iterations)?;
                    let c0 = measure(runtime, fixture, candidate, iterations)?;
                    let c1 = measure(runtime, fixture, candidate, iterations)?;
                    let b1 = measure(runtime, fixture, baseline, iterations)?;
                    (b0, b1, c0, c1)
                } else {
                    let c0 = measure(runtime, fixture, candidate, iterations)?;
                    let b0 = measure(runtime, fixture, baseline, iterations)?;
                    let b1 = measure(runtime, fixture, baseline, iterations)?;
                    let c1 = measure(runtime, fixture, candidate, iterations)?;
                    (b0, b1, c0, c1)
                };
                let (baseline_us, candidate_us, speedup) = paired_sample(b0, b1, c0, c1)?;
                baseline_samples.push(baseline_us);
                candidate_samples.push(candidate_us);
                ratios.push(speedup);
            }
            let baseline_p50 = percentile(&baseline_samples, 0.50)?;
            let candidate_p50 = percentile(&candidate_samples, 0.50)?;
            let speedup_p05 = percentile(&ratios, 0.05)?;
            let speedup_p50 = percentile(&ratios, 0.50)?;
            let speedup_p95 = percentile(&ratios, 0.95)?;
            eprintln!(
                "nt_prism baseline={} candidate={} order={} windows={} iterations={} baseline_us_p50={:.6} candidate_us_p50={:.6} speedup_p05={:.9} speedup_p50={:.9} speedup_p95={:.9}",
                baseline.name(),
                candidate.name(),
                order,
                windows,
                iterations,
                baseline_p50,
                candidate_p50,
                speedup_p05,
                speedup_p50,
                speedup_p95,
            );
            if let Err(error) = validate_performance_gate(
                baseline,
                candidate,
                order,
                windows,
                speedup_p05,
                speedup_p50,
                speedup_p95,
            ) {
                order_failures.push(format!("{order}: {error}"));
            }
        }
        finish_performance_failures(
            &format!("orders for {} -> {}", baseline.name(), candidate.name()),
            order_failures,
        )
    }

    #[test]
    #[ignore = "requires an exclusive quiet CC12.0/170-SM GPU"]
    fn prism_transpose_tournament_abba_baab() -> Result<(), String> {
        validate_release_build(cfg!(debug_assertions))?;
        let runtime = new_runtime()?;
        let mut fixture = new_fixture(&runtime)?;
        for arm in [
            Arm::Production,
            Arm::GenericNt,
            Arm::Candidate,
            Arm::Specialized,
            Arm::SpecializedVector,
            Arm::CublasPedantic,
            Arm::CublasFast,
        ] {
            for _ in 0..5 {
                launch_arm(&runtime, &mut fixture, arm)?;
            }
        }
        runtime
            .ctx
            .stream
            .synchronize()
            .map_err(|error| format!("warmup: {error:?}"))?;
        let windows = parse_windows_env(std::env::var("MAMBA_RS_NT_PRISM_WINDOWS"))?;
        let mut pair_failures = Vec::new();
        for (baseline, candidate) in [
            (Arm::GenericNt, Arm::Production),
            (Arm::Production, Arm::SpecializedVector),
            (Arm::CublasPedantic, Arm::Production),
            (Arm::CublasFast, Arm::Production),
        ] {
            if let Err(error) = paired(&runtime, &mut fixture, baseline, candidate, windows) {
                pair_failures.push(format!(
                    "{} -> {}: {error}",
                    baseline.name(),
                    candidate.name()
                ));
            }
        }
        finish_performance_failures("pairs", pair_failures)
    }
}
