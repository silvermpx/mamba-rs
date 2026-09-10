#[cfg(feature = "cuda")]
mod common;

const EXTENSION_SOURCE: &str = include_str!("sm120_tf32_pair_store_experiment.cu");
const TEST_SOURCE: &str = include_str!("sm120_tf32_pair_store_experiment.rs");
const PRODUCTION_SM120_SOURCE: &str = include_str!("../kernels/gemm_bi_triad/sm120.cu");
#[cfg(feature = "cuda")]
const PTX_OUTPUT_ENV: &str = "MAMBA_RS_SM120_TF32_PAIR_PTX";

const CANDIDATE_SYMBOLS: [&str; 18] = [
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_pair_v1",
    "gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_w16_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_w16_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2_exp_a3d_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_exp_pair_producer_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s4_exp_a3d_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk64_s2_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m96n64_bk32_s3_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m96n64_bk32_s4_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n96_bk32_s3_exp_pair_v1",
    "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n96_bk32_s4_exp_pair_v1",
];

fn percentile(values: &[f64], fraction: f64) -> Result<f64, String> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || !(0.0..=1.0).contains(&fraction)
    {
        return Err("percentile requires positive samples and a unit fraction".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Ok(sorted[index])
}

fn speedup_order_passes(speedups: &[f64], min_median: f64, min_p05: f64) -> bool {
    percentile(speedups, 0.50).is_ok_and(|median| median >= min_median)
        && percentile(speedups, 0.05).is_ok_and(|p05| p05 > min_p05)
}

#[test]
fn extension_is_test_only_and_not_a_production_override() {
    for symbol in CANDIDATE_SYMBOLS {
        assert!(
            EXTENSION_SOURCE.contains(symbol),
            "missing test-only candidate {symbol}"
        );
    }
    for required in [
        "struct Sm120Tf32Pair",
        "reinterpret_cast<float2*>",
        "full_tile",
        "params.ldc & 1",
        "sm120_tf32_store<Op>",
        "sm120_tf32_issue_stage<Op, M, N, Stages>",
        "sm120_tf32_issue_stage_w16_exp_v1",
        "float accumulator[4][4]",
        "constexpr int warps = 8",
        "cp.async.bulk.tensor.3d.shared::cta.global.tile",
        "sm120_tf32_tn_load_a3d_exp_v1",
        "sm120_tf32_entry<Sm120Tn, 128, 64, 4>",
        "sm120_tf32_entry<Sm120Tn, 64, 128, 4>",
        "constexpr int compute_warps",
        "if constexpr (ProducerWarp)",
    ] {
        assert!(
            EXTENSION_SOURCE.contains(required),
            "extension lost required pair-store contract {required}"
        );
    }
    for forbidden in [
        "gemm_bi_nn_sm120_tma_mma_tf32",
        "atomic",
        "split_k",
        "TnNarrow",
    ] {
        assert!(
            !EXTENSION_SOURCE.contains(forbidden),
            "test-only extension contains forbidden token {forbidden}"
        );
    }
}

#[test]
fn experiment_defines_no_more_than_five_kernel_parameters() {
    let entry = "extern \"C\" __global__";
    assert_eq!(
        EXTENSION_SOURCE.matches(entry).count(),
        CANDIDATE_SYMBOLS.len()
    );
    assert!(EXTENSION_SOURCE.contains("void* output"));
    assert!(EXTENSION_SOURCE.contains("CUtensorMap a_map"));
    assert!(EXTENSION_SOURCE.contains("CUtensorMap b_map"));
    assert!(EXTENSION_SOURCE.contains("const float* bias"));
    assert!(EXTENSION_SOURCE.contains("Sm120KernelParams params"));
}

#[cfg(feature = "cuda")]
fn compose_cuda_source() -> String {
    [
        include_str!("../kernels/_typed_prelude.cuh"),
        include_str!("../kernels/gemm_bi_triad/contract.cuh"),
        include_str!("../kernels/gemm_bi_triad/common.cuh"),
        include_str!("../kernels/gemm_bi_triad/epilogue.cuh"),
        PRODUCTION_SM120_SOURCE,
        EXTENSION_SOURCE,
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
fn experiment_symbols_are_absent_from_production_registration() {
    for source in [
        PRODUCTION_SM120_SOURCE,
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/contract.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/dispatch.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/launch.rs"),
        include_str!("../src/mamba_ssm/gpu/gemm_bi_triad/modules.rs"),
    ] {
        assert!(
            !source.contains("_exp_pair_v1")
                && !source.contains("_exp_w16")
                && !source.contains("_exp_a3d")
                && !source.contains("_bk32_s4_exp")
                && !source.contains("_producer_v1"),
            "test-only candidate leaked into production"
        );
    }
}

#[test]
fn runtime_harness_freezes_required_gates() {
    let runtime = TEST_SOURCE
        .rsplit_once("mod cuda_experiment {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA experiment module");
    for required in [
        "#[ignore = \"requires an exclusive SM120 GPU\"]",
        "local_size_bytes",
        "num_regs",
        "occupancy_max_active_blocks_per_multiprocessor",
        "capture_into_graph",
        "Path::Eager",
        "Path::Graph",
        "candidate eager bits differ from incumbent",
        "candidate graph bits differ from incumbent",
        "post-timing candidate bits differ from incumbent",
        "MAMBA_RS_SM120_TF32_PAIR_WINDOWS",
        "MAMBA_RS_SM120_TF32_PAIR_ITERATIONS",
        "MAMBA_RS_SM120_TF32_PAIR_MIN_MEDIAN_SPEEDUP",
        "MAMBA_RS_SM120_TF32_PAIR_MIN_P05_SPEEDUP",
        "MAMBA_RS_SM120_TF32_PAIR_PERFORMANCE",
        "MAMBA_RS_SM120_TF32_PAIR_CASE_FILTER",
        "TARGET_WINDOW_MS",
        "CALIBRATION_PROBE_LAUNCHES",
        "MAX_WINDOW_LAUNCHES",
        "calibrate_iterations",
        "warmups: env_usize(\"MAMBA_RS_SM120_TF32_PAIR_WARMUPS\", 128)",
        "QuietGpu",
        "require_pre_context",
        "require_cohort",
        "verify_post_cohort",
        "ab_speedups",
        "ba_speedups",
    ] {
        assert!(
            runtime.contains(required),
            "runtime harness lost required gate {required}"
        );
    }
    assert!(TEST_SOURCE.contains("const PTX_OUTPUT_ENV: &str = \"MAMBA_RS_SM120_TF32_PAIR_PTX\""));
    assert!(TEST_SOURCE.contains("std::env::var_os(PTX_OUTPUT_ENV)"));
}

#[test]
fn target_cohorts_use_production_s4_and_filter_before_gpu_work() {
    let runtime = TEST_SOURCE
        .rsplit_once("mod cuda_experiment {")
        .map(|(_, runtime)| runtime)
        .expect("CUDA experiment module");
    let route = runtime
        .split_once("const TN_M64N128_BK64_S2_VS_PRODUCTION_S4: Route = Route {")
        .map(|(_, route)| route)
        .expect("dedicated BK64-versus-production route")
        .split_once("};")
        .map(|(route, _)| route)
        .expect("dedicated route terminator");
    assert!(route.contains("incumbent: \"gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair\""));
    assert!(route.contains("..TN_M64N128_BK64_S2_PAIR"));
    assert_eq!(
        runtime
            .matches("id: \"tn_target_m64n128_bk64_s2_vs_prod_s4_")
            .count(),
        3
    );
    for (id, dims) in [
        (
            "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x1536x1536",
            "dims: (2048, 1536, 1536)",
        ),
        (
            "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x768x3072",
            "dims: (2048, 768, 3072)",
        ),
        (
            "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x1536x768",
            "dims: (2048, 1536, 768)",
        ),
    ] {
        let case = runtime
            .split_once(&format!("id: \"{id}\""))
            .map(|(_, case)| case)
            .unwrap_or_else(|| panic!("target cohort lost {id}"))
            .split_once("},")
            .map(|(case, _)| case)
            .expect("target case terminator");
        for required in [
            "op: Op::Tn",
            dims,
            "alpha: 1.0",
            "route: TN_M64N128_BK64_S2_VS_PRODUCTION_S4",
            "performance: true",
        ] {
            assert!(case.contains(required), "{id} lost {required}");
        }
    }

    let producer_route = runtime
        .split_once("const TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4: Route = Route {")
        .map(|(_, route)| route)
        .expect("dedicated producer-versus-production route")
        .split_once("};")
        .map(|(route, _)| route)
        .expect("dedicated producer route terminator");
    assert!(
        producer_route
            .contains("incumbent: \"gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair\"")
    );
    assert!(producer_route.contains("candidate: CANDIDATE_SYMBOLS[11]"));
    assert!(producer_route.contains("..TN_M64N128_S4_PAIR_PRODUCER"));
    assert_eq!(
        runtime
            .matches("id: \"tn_target_producer_vs_prod_s4_")
            .count(),
        3
    );
    for (id, dims) in [
        (
            "tn_target_producer_vs_prod_s4_2048x1536x1536",
            "dims: (2048, 1536, 1536)",
        ),
        (
            "tn_target_producer_vs_prod_s4_2048x768x3072",
            "dims: (2048, 768, 3072)",
        ),
        (
            "tn_target_producer_vs_prod_s4_2048x1536x768",
            "dims: (2048, 1536, 768)",
        ),
    ] {
        let case = runtime
            .split_once(&format!("id: \"{id}\""))
            .map(|(_, case)| case)
            .unwrap_or_else(|| panic!("producer target cohort lost {id}"))
            .split_once("},")
            .map(|(case, _)| case)
            .expect("producer target case terminator");
        for required in [
            "op: Op::Tn",
            dims,
            "alpha: 1.0",
            "route: TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4",
            "performance: true",
        ] {
            assert!(case.contains(required), "{id} lost {required}");
        }
    }

    let entry = runtime
        .rsplit_once("fn sm120_tf32_pair_store_experiment()")
        .map(|(_, entry)| entry)
        .expect("runtime experiment entry");
    let selection = entry
        .find("let selected_cases = CASES")
        .expect("case selection");
    let empty_guard = entry
        .find("if selected_cases.is_empty()")
        .expect("empty case selection must fail closed");
    assert!(entry.contains("SM120 TF32 pair-store case filter matched no cases"));
    let quiet_gpu = entry.find("QuietGpu::").expect("quiet GPU preflight");
    let runtime_load = entry.find("new_runtime()").expect("CUDA runtime load");
    let case_loop = entry
        .find("for case in selected_cases")
        .expect("filtered runtime case loop");
    let kernel_load = entry.find("load_kernel_pair").expect("kernel pair load");
    assert!(
        selection < empty_guard
            && empty_guard < quiet_gpu
            && quiet_gpu < runtime_load
            && runtime_load < case_loop
            && case_loop < kernel_load
    );
}

#[test]
fn pair_store_admission_requires_each_order_to_pass() {
    assert!(speedup_order_passes(&[1.02, 1.03, 1.04], 1.01, 1.0));
    assert!(!speedup_order_passes(&[0.99, 1.05, 1.06], 1.01, 1.0));
    assert!(!speedup_order_passes(&[1.0, 1.0, 1.0], 1.01, 1.0));
}

#[cfg(feature = "cuda")]
#[test]
fn appended_extension_compiles_for_compute_120() {
    let options = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_120"),
        options: vec![
            "--fmad=true".to_owned(),
            "--extra-device-vectorization".to_owned(),
            "-DNDEBUG".to_owned(),
            "--frandom-seed=12053201".to_owned(),
        ],
        include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
        ..Default::default()
    };
    let image = cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
        .expect("test-only SM120 TF32 pair-store extension must compile");
    let ptx = image.to_src();
    if let Some(path) = std::env::var_os(PTX_OUTPUT_ENV).filter(|path| !path.is_empty()) {
        use std::io::Write as _;

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create test-only SM120 TF32 PTX artifact");
        file.write_all(ptx.as_bytes())
            .expect("write test-only SM120 TF32 PTX artifact");
        file.sync_all()
            .expect("sync test-only SM120 TF32 PTX artifact");
    }
    for symbol in CANDIDATE_SYMBOLS {
        assert!(ptx.contains(symbol), "compiled PTX omitted {symbol}");
    }
}

#[cfg(feature = "cuda")]
mod cuda_experiment {
    use std::sync::Arc;

    use crate::common::gpu_quiet::QuietGpu;
    use cudarc::driver::{
        CudaFunction, CudaGraph, CudaModule, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
        sys,
    };
    use mamba_rs::mamba_ssm::gpu::buffers::GpuBuffer;
    use mamba_rs::mamba_ssm::gpu::device::GpuDevice;
    use mamba_rs::mamba_ssm::gpu::graph_capture::capture_into_graph;

    use super::{CANDIDATE_SYMBOLS, compose_cuda_source, percentile, speedup_order_passes};

    const GUARD_ELEMENTS: usize = 32;
    const GUARD_BITS: u32 = 0x7fc1_2053;
    const CORRECTNESS_REPEATS: usize = 3;
    const TARGET_WINDOW_MS: f64 = 5.0;
    const CALIBRATION_PROBE_LAUNCHES: usize = 16;
    const MAX_WINDOW_LAUNCHES: usize = 1_000_000;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Op {
        Tn,
        Nt,
    }

    #[derive(Clone, Copy, Debug)]
    struct Route {
        incumbent: &'static str,
        candidate: &'static str,
        tile: (usize, usize),
        incumbent_stages: usize,
        candidate_stages: usize,
        incumbent_threads: u32,
        candidate_threads: u32,
        candidate_max_registers: i32,
        candidate_min_blocks: u32,
        candidate_a_rank3: bool,
        candidate_reduction_tile: usize,
        candidate_tile: Option<(usize, usize)>,
    }

    impl Route {
        const fn candidate_tile(self) -> (usize, usize) {
            match self.candidate_tile {
                Some(tile) => tile,
                None => self.tile,
            }
        }

        const fn dynamic_shared_bytes(
            tile: (usize, usize),
            stages: usize,
            reduction_tile: usize,
        ) -> u32 {
            (128 + stages * (tile.0 + tile.1) * reduction_tile * 4) as u32
        }

        const fn incumbent_shared_bytes(self) -> u32 {
            Self::dynamic_shared_bytes(self.tile, self.incumbent_stages, 32)
        }

        const fn candidate_shared_bytes(self) -> u32 {
            Self::dynamic_shared_bytes(
                self.candidate_tile(),
                self.candidate_stages,
                self.candidate_reduction_tile,
            )
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Case {
        id: &'static str,
        op: Op,
        dims: (usize, usize, usize),
        alpha: f32,
        route: Route,
        performance: bool,
    }

    const TN_M128N64_S2: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s2",
        candidate: CANDIDATE_SYMBOLS[0],
        tile: (128, 64),
        incumbent_stages: 2,
        candidate_stages: 2,
        incumbent_threads: 256,
        candidate_threads: 256,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };
    const TN_M128N64_S3: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3",
        candidate: CANDIDATE_SYMBOLS[1],
        tile: (128, 64),
        incumbent_stages: 3,
        candidate_stages: 3,
        incumbent_threads: 256,
        candidate_threads: 256,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };
    const TN_M64N64_S2: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        candidate: CANDIDATE_SYMBOLS[2],
        tile: (64, 64),
        incumbent_stages: 2,
        candidate_stages: 2,
        incumbent_threads: 128,
        candidate_threads: 128,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };
    const NT_M64N128_S2: Route = Route {
        incumbent: "gemm_bi_nt_sm120_tma_mma_tf32_v1_m64n128_bk32_s2",
        candidate: CANDIDATE_SYMBOLS[3],
        tile: (64, 128),
        incumbent_stages: 2,
        candidate_stages: 2,
        incumbent_threads: 256,
        candidate_threads: 256,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };

    const TN_M64N64_S2_W16: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        candidate: CANDIDATE_SYMBOLS[4],
        tile: (64, 64),
        incumbent_stages: 2,
        candidate_stages: 2,
        incumbent_threads: 128,
        candidate_threads: 256,
        candidate_max_registers: 80,
        candidate_min_blocks: 2,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };
    const TN_M64N64_S2_W16_PAIR: Route = Route {
        candidate: CANDIDATE_SYMBOLS[5],
        ..TN_M64N64_S2_W16
    };

    const TN_M64N64_S2_A3D: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n64_bk32_s2",
        candidate: CANDIDATE_SYMBOLS[6],
        tile: (64, 64),
        incumbent_stages: 2,
        candidate_stages: 2,
        incumbent_threads: 128,
        candidate_threads: 128,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: true,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };

    const TN_M128N64_S4: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m128n64_bk32_s3",
        candidate: CANDIDATE_SYMBOLS[7],
        tile: (128, 64),
        incumbent_stages: 3,
        candidate_stages: 4,
        incumbent_threads: 256,
        candidate_threads: 256,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };
    const TN_M64N128_S4: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3",
        candidate: CANDIDATE_SYMBOLS[8],
        tile: (64, 128),
        incumbent_stages: 3,
        candidate_stages: 4,
        incumbent_threads: 256,
        candidate_threads: 256,
        candidate_max_registers: 128,
        candidate_min_blocks: 1,
        candidate_a_rank3: false,
        candidate_reduction_tile: 32,
        candidate_tile: None,
    };

    const TN_M128N64_S4_PAIR: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[7],
        candidate: CANDIDATE_SYMBOLS[9],
        incumbent_stages: 4,
        ..TN_M128N64_S4
    };
    const TN_M64N128_S4_PAIR: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[8],
        candidate: CANDIDATE_SYMBOLS[10],
        incumbent_stages: 4,
        ..TN_M64N128_S4
    };
    const TN_M64N128_S4_PAIR_FINAL: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s3",
        candidate: CANDIDATE_SYMBOLS[10],
        incumbent_stages: 3,
        candidate_stages: 4,
        ..TN_M64N128_S4
    };
    const TN_M64N128_S4_PRODUCTION_PAIR_VS_EXPERIMENT: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair",
        candidate: CANDIDATE_SYMBOLS[10],
        incumbent_stages: 4,
        candidate_stages: 4,
        ..TN_M64N128_S4
    };
    const TN_M64N128_S4_PAIR_PRODUCER: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[10],
        candidate: CANDIDATE_SYMBOLS[11],
        incumbent_stages: 4,
        candidate_stages: 4,
        incumbent_threads: 256,
        candidate_threads: 288,
        ..TN_M64N128_S4
    };
    const TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair",
        candidate: CANDIDATE_SYMBOLS[11],
        ..TN_M64N128_S4_PAIR_PRODUCER
    };
    const TN_M128N64_S4_A3D: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[7],
        candidate: CANDIDATE_SYMBOLS[12],
        incumbent_stages: 4,
        candidate_stages: 4,
        candidate_a_rank3: true,
        ..TN_M128N64_S4
    };

    const TN_M64N128_BK64_S2_PAIR: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[10],
        candidate: CANDIDATE_SYMBOLS[13],
        incumbent_stages: 4,
        candidate_stages: 2,
        candidate_reduction_tile: 64,
        ..TN_M64N128_S4
    };
    const TN_M64N128_BK64_S2_VS_PRODUCTION_S4: Route = Route {
        incumbent: "gemm_bi_tn_sm120_tma_mma_tf32_v1_m64n128_bk32_s4_pair",
        ..TN_M64N128_BK64_S2_PAIR
    };

    const TN_M96N64_S3_PAIR: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[10],
        candidate: CANDIDATE_SYMBOLS[14],
        incumbent_stages: 4,
        candidate_stages: 3,
        candidate_threads: 192,
        candidate_tile: Some((96, 64)),
        ..TN_M64N128_S4
    };
    const TN_M96N64_S4_PAIR: Route = Route {
        candidate: CANDIDATE_SYMBOLS[15],
        candidate_stages: 4,
        ..TN_M96N64_S3_PAIR
    };
    const TN_M64N96_S3_PAIR: Route = Route {
        incumbent: CANDIDATE_SYMBOLS[10],
        candidate: CANDIDATE_SYMBOLS[16],
        incumbent_stages: 4,
        candidate_stages: 3,
        candidate_threads: 192,
        candidate_tile: Some((64, 96)),
        ..TN_M64N128_S4
    };
    const TN_M64N96_S4_PAIR: Route = Route {
        candidate: CANDIDATE_SYMBOLS[17],
        candidate_stages: 4,
        ..TN_M64N96_S3_PAIR
    };

    const CASES: [Case; 38] = [
        Case {
            id: "tn_large",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 0.75,
            route: TN_M128N64_S2,
            performance: true,
        },
        Case {
            id: "tn_large_deep",
            op: Op::Tn,
            dims: (4096, 3072, 1536),
            alpha: 1.25,
            route: TN_M64N64_S2,
            performance: true,
        },
        Case {
            id: "tn_d768_out_proj",
            op: Op::Tn,
            dims: (2048, 1536, 768),
            alpha: 0.75,
            route: TN_M128N64_S3,
            performance: true,
        },
        Case {
            id: "nt_large_deep",
            op: Op::Nt,
            dims: (4096, 3072, 1536),
            alpha: 0.75,
            route: NT_M64N128_S2,
            performance: true,
        },
        Case {
            id: "tn_scalar_tail",
            op: Op::Tn,
            dims: (96, 132, 68),
            alpha: 1.25,
            route: TN_M128N64_S2,
            performance: false,
        },
        Case {
            id: "nt_scalar_tail",
            op: Op::Nt,
            dims: (68, 132, 96),
            alpha: 1.25,
            route: NT_M64N128_S2,
            performance: false,
        },
        Case {
            id: "tn_prism_w16",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N64_S2_W16,
            performance: true,
        },
        Case {
            id: "tn_prism_w16_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N64_S2_W16_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_aligned_w16",
            op: Op::Tn,
            dims: (4608, 384, 1920),
            alpha: 1.0,
            route: TN_M64N64_S2_W16,
            performance: true,
        },
        Case {
            id: "tn_prism_aligned_w16_pair",
            op: Op::Tn,
            dims: (4608, 384, 1920),
            alpha: 1.0,
            route: TN_M64N64_S2_W16_PAIR,
            performance: true,
        },
        Case {
            id: "tn_large_w16",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M64N64_S2_W16,
            performance: true,
        },
        Case {
            id: "tn_nontrivial_alpha_w16_pair",
            op: Op::Tn,
            dims: (96, 132, 68),
            alpha: 1.25,
            route: TN_M64N64_S2_W16_PAIR,
            performance: false,
        },
        Case {
            id: "tn_prism_a3d",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N64_S2_A3D,
            performance: true,
        },
        Case {
            id: "tn_prism_aligned_a3d",
            op: Op::Tn,
            dims: (4608, 384, 1920),
            alpha: 1.0,
            route: TN_M64N64_S2_A3D,
            performance: true,
        },
        Case {
            id: "tn_large_a3d",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M64N64_S2_A3D,
            performance: true,
        },
        Case {
            id: "tn_prism_m128n64_s4",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M128N64_S4,
            performance: true,
        },
        Case {
            id: "tn_prism_m64n128_s4",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_S4,
            performance: true,
        },
        Case {
            id: "tn_large_m128n64_s4",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M128N64_S4,
            performance: true,
        },
        Case {
            id: "tn_large_m64n128_s4",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M64N128_S4,
            performance: true,
        },
        Case {
            id: "tn_prism_m128n64_s4_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M128N64_S4_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_m64n128_s4_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_S4_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_m64n128_s4_pair_producer",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_S4_PAIR_PRODUCER,
            performance: true,
        },
        Case {
            id: "tn_prism_m128n64_s4_a3d",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M128N64_S4_A3D,
            performance: true,
        },
        Case {
            id: "tn_large_m128n64_s4_a3d",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M128N64_S4_A3D,
            performance: true,
        },
        Case {
            id: "tn_prism_m64n128_bk64_s2_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_BK64_S2_PAIR,
            performance: true,
        },
        Case {
            id: "tn_large_m64n128_bk64_s2_pair",
            op: Op::Tn,
            dims: (2048, 3072, 768),
            alpha: 1.0,
            route: TN_M64N128_BK64_S2_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_irregular_m96n64_s3_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M96N64_S3_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_irregular_m96n64_s4_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M96N64_S4_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_irregular_m64n96_s3_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N96_S3_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_irregular_m64n96_s4_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N96_S4_PAIR,
            performance: true,
        },
        Case {
            id: "tn_prism_final_m64n128_s4_pair",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_S4_PAIR_FINAL,
            performance: true,
        },
        Case {
            id: "tn_prism_production_s4_pair_vs_experiment",
            op: Op::Tn,
            dims: (4621, 384, 1928),
            alpha: 1.0,
            route: TN_M64N128_S4_PRODUCTION_PAIR_VS_EXPERIMENT,
            performance: true,
        },
        Case {
            id: "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x1536x1536",
            op: Op::Tn,
            dims: (2048, 1536, 1536),
            alpha: 1.0,
            route: TN_M64N128_BK64_S2_VS_PRODUCTION_S4,
            performance: true,
        },
        Case {
            id: "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x768x3072",
            op: Op::Tn,
            dims: (2048, 768, 3072),
            alpha: 1.0,
            route: TN_M64N128_BK64_S2_VS_PRODUCTION_S4,
            performance: true,
        },
        Case {
            id: "tn_target_m64n128_bk64_s2_vs_prod_s4_2048x1536x768",
            op: Op::Tn,
            dims: (2048, 1536, 768),
            alpha: 1.0,
            route: TN_M64N128_BK64_S2_VS_PRODUCTION_S4,
            performance: true,
        },
        Case {
            id: "tn_target_producer_vs_prod_s4_2048x1536x1536",
            op: Op::Tn,
            dims: (2048, 1536, 1536),
            alpha: 1.0,
            route: TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4,
            performance: true,
        },
        Case {
            id: "tn_target_producer_vs_prod_s4_2048x768x3072",
            op: Op::Tn,
            dims: (2048, 768, 3072),
            alpha: 1.0,
            route: TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4,
            performance: true,
        },
        Case {
            id: "tn_target_producer_vs_prod_s4_2048x1536x768",
            op: Op::Tn,
            dims: (2048, 1536, 768),
            alpha: 1.0,
            route: TN_M64N128_S4_PRODUCER_VS_PRODUCTION_S4,
            performance: true,
        },
    ];

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct KernelParams {
        a_x: i32,
        a_y: i32,
        b_x: i32,
        b_y: i32,
        alpha: f32,
        beta: f32,
        m: i32,
        k: i32,
        n: i32,
        ldc: i32,
    }

    unsafe impl DeviceRepr for KernelParams {}

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct TensorMap(sys::CUtensorMap);

    unsafe impl DeviceRepr for TensorMap {}

    #[derive(Clone, Copy)]
    struct TensorMaps {
        a: TensorMap,
        b: TensorMap,
    }

    #[derive(Clone, Copy)]
    struct CandidateMaps {
        incumbent: TensorMaps,
        candidate: TensorMaps,
    }

    struct GuardedOutput {
        buffer: GpuBuffer,
        initial: Vec<f32>,
        active_len: usize,
    }

    impl GuardedOutput {
        fn new(stream: &Arc<CudaStream>, active: Vec<f32>) -> Result<Self, String> {
            let active_len = active.len();
            let mut initial = vec![f32::from_bits(GUARD_BITS); active_len + 2 * GUARD_ELEMENTS];
            initial[GUARD_ELEMENTS..GUARD_ELEMENTS + active_len].copy_from_slice(&active);
            Ok(Self {
                buffer: GpuBuffer::from_cpu(stream, &initial)?,
                initial,
                active_len,
            })
        }

        fn reset(&mut self, stream: &Arc<CudaStream>) -> Result<(), String> {
            self.buffer.upload(stream, &self.initial)
        }

        fn active_ptr(&self, stream: &Arc<CudaStream>) -> u64 {
            self.buffer.raw_ptr_at(stream, GUARD_ELEMENTS)
        }

        fn bits(&self, stream: &Arc<CudaStream>) -> Result<Vec<u32>, String> {
            let values = self.buffer.to_cpu(stream)?;
            for (index, value) in values[..GUARD_ELEMENTS]
                .iter()
                .chain(&values[GUARD_ELEMENTS + self.active_len..])
                .enumerate()
            {
                if value.to_bits() != GUARD_BITS {
                    return Err(format!("output red zone changed at guard element {index}"));
                }
            }
            Ok(values[GUARD_ELEMENTS..GUARD_ELEMENTS + self.active_len]
                .iter()
                .map(|value| value.to_bits())
                .collect())
        }
    }

    struct Fixture {
        _a: GpuBuffer,
        _b: GpuBuffer,
        incumbent: GuardedOutput,
        candidate: GuardedOutput,
        maps: CandidateMaps,
        params: KernelParams,
    }

    struct Runtime {
        _device: GpuDevice,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    struct KernelPair {
        incumbent: CudaFunction,
        candidate: CudaFunction,
        incumbent_config: LaunchConfig,
        candidate_config: LaunchConfig,
    }

    #[derive(Clone, Copy, Debug)]
    enum Arm {
        Incumbent,
        Candidate,
    }

    #[derive(Clone, Copy, Debug)]
    enum Path {
        Eager,
        Graph,
    }

    struct CapturedPair {
        incumbent: CudaGraph,
        candidate: CudaGraph,
    }

    struct Measurement<'a> {
        runtime: &'a Runtime,
        kernels: &'a KernelPair,
        fixture: &'a Fixture,
        graphs: &'a CapturedPair,
        path: Path,
    }

    #[derive(Clone, Copy, Debug)]
    struct PairedTiming {
        incumbent_us: f64,
        candidate_us: f64,
    }

    impl PairedTiming {
        fn speedup(self) -> f64 {
            self.incumbent_us / self.candidate_us
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct PerfConfig {
        performance: bool,
        windows_per_order: usize,
        iterations: Option<usize>,
        warmups: usize,
        min_median_speedup: f64,
        min_p05_speedup: f64,
    }

    impl PerfConfig {
        fn from_env() -> Result<Self, String> {
            Ok(Self {
                performance: env_bool("MAMBA_RS_SM120_TF32_PAIR_PERFORMANCE", true)?,
                windows_per_order: env_usize("MAMBA_RS_SM120_TF32_PAIR_WINDOWS", 101)?,
                iterations: env_optional_usize("MAMBA_RS_SM120_TF32_PAIR_ITERATIONS")?,
                warmups: env_usize("MAMBA_RS_SM120_TF32_PAIR_WARMUPS", 128)?,
                min_median_speedup: env_f64("MAMBA_RS_SM120_TF32_PAIR_MIN_MEDIAN_SPEEDUP", 1.01)?,
                min_p05_speedup: env_f64("MAMBA_RS_SM120_TF32_PAIR_MIN_P05_SPEEDUP", 1.0)?,
            })
        }
    }

    fn env_bool(name: &str, default: bool) -> Result<bool, String> {
        match std::env::var(name) {
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!("{name} must equal 0 or 1, got {value:?}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(format!("read {name}: {error}")),
        }
    }

    fn env_usize(name: &str, default: usize) -> Result<usize, String> {
        let value = std::env::var(name)
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|error| format!("parse {name}: {error}"))?
            .unwrap_or(default);
        if value == 0 {
            Err(format!("{name} must be positive"))
        } else {
            Ok(value)
        }
    }

    fn env_optional_usize(name: &str) -> Result<Option<usize>, String> {
        match std::env::var(name) {
            Ok(value) => {
                let value = value
                    .parse::<usize>()
                    .map_err(|error| format!("parse {name}: {error}"))?;
                if value == 0 {
                    Err(format!("{name} must be positive"))
                } else {
                    Ok(Some(value))
                }
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(format!("read {name}: {error}")),
        }
    }

    fn env_f64(name: &str, default: f64) -> Result<f64, String> {
        let value = std::env::var(name)
            .ok()
            .map(|value| value.parse::<f64>())
            .transpose()
            .map_err(|error| format!("parse {name}: {error}"))?
            .unwrap_or(default);
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(format!("{name} must be positive and finite"))
        }
    }

    fn compile_experiment_ptx() -> Result<String, String> {
        let options = cudarc::nvrtc::CompileOptions {
            arch: Some("compute_120"),
            options: vec![
                "--fmad=true".to_owned(),
                "--extra-device-vectorization".to_owned(),
                "-DNDEBUG".to_owned(),
                "--frandom-seed=12053201".to_owned(),
            ],
            include_paths: mamba_rs::mamba_ssm::gpu::kernels::cuda_include_paths(),
            ..Default::default()
        };
        cudarc::nvrtc::compile_ptx_with_opts(compose_cuda_source(), options)
            .map(|image| image.to_src())
            .map_err(|error| format!("compile test-only SM120 TF32 extension: {error:?}"))
    }

    fn new_runtime() -> Result<Runtime, String> {
        let device = GpuDevice::new(0)?;
        if device.compute_capability != (12, 0) {
            return Err(format!(
                "pair-store experiment requires SM120, found {:?}",
                device.compute_capability
            ));
        }
        let stream = device.fork_stream()?;
        let module = device
            .context()
            .load_module(cudarc::nvrtc::Ptx::from_src(compile_experiment_ptx()?))
            .map_err(|error| format!("load test-only SM120 TF32 module: {error:?}"))?;
        Ok(Runtime {
            _device: device,
            stream,
            module,
        })
    }

    fn seeded_values(len: usize, salt: u64) -> Vec<f32> {
        let mut state = salt;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let signed = ((state >> 32) as u32 % 2049) as i32 - 1024;
                signed as f32 / 4096.0
            })
            .collect()
    }

    fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, String> {
        left.checked_mul(right)
            .ok_or_else(|| format!("{label} extent overflows usize"))
    }

    fn tensor_map(
        pointer: u64,
        width: usize,
        rows: usize,
        stride: usize,
        box_dimensions: [u32; 2],
    ) -> Result<TensorMap, String> {
        let dimensions = [
            u64::try_from(width).map_err(|_| "tensor-map width exceeds u64")?,
            u64::try_from(rows).map_err(|_| "tensor-map rows exceed u64")?,
        ];
        let outer_stride = [u64::try_from(
            stride
                .checked_mul(4)
                .ok_or("tensor-map byte stride overflows usize")?,
        )
        .map_err(|_| "tensor-map byte stride exceeds u64")?];
        let element_strides = [1_u32, 1_u32];
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let result = unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                2,
                pointer as usize as *mut std::ffi::c_void,
                dimensions.as_ptr(),
                outer_stride.as_ptr(),
                box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("encode test tensor map: {result:?}"));
        }
        Ok(TensorMap(unsafe { raw.assume_init() }))
    }

    fn rank3_plane_tensor_map(
        pointer: u64,
        width: usize,
        rows: usize,
        tile_width: usize,
    ) -> Result<TensorMap, String> {
        if !width.is_multiple_of(32) || !tile_width.is_multiple_of(32) {
            return Err("rank-3 plane tensor map requires 32-element alignment".into());
        }
        let dimensions = [
            32_u64,
            u64::try_from(width / 32).map_err(|_| "rank-3 plane count exceeds u64")?,
            u64::try_from(rows).map_err(|_| "rank-3 row count exceeds u64")?,
        ];
        let row_bytes = width
            .checked_mul(4)
            .ok_or("rank-3 row byte stride overflows usize")?;
        let global_strides = [
            128_u64,
            u64::try_from(row_bytes).map_err(|_| "rank-3 row stride exceeds u64")?,
        ];
        let box_dimensions = [
            32_u32,
            u32::try_from(tile_width / 32).map_err(|_| "rank-3 tile planes exceed u32")?,
            32_u32,
        ];
        let element_strides = [1_u32, 1_u32, 1_u32];
        let mut raw = std::mem::MaybeUninit::<sys::CUtensorMap>::zeroed();
        let result = unsafe {
            sys::cuTensorMapEncodeTiled(
                raw.as_mut_ptr(),
                sys::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
                3,
                pointer as usize as *mut std::ffi::c_void,
                dimensions.as_ptr(),
                global_strides.as_ptr(),
                box_dimensions.as_ptr(),
                element_strides.as_ptr(),
                sys::CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
                sys::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                sys::CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
                sys::CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("encode rank-3 plane tensor map: {result:?}"));
        }
        Ok(TensorMap(unsafe { raw.assume_init() }))
    }

    fn new_fixture(runtime: &Runtime, case: Case) -> Result<Fixture, String> {
        let (m, k, n) = case.dims;
        let (a_len, b_len, output_len, ldc) = match case.op {
            Op::Tn => (
                checked_product(m, k, "TN A")?,
                checked_product(m, n, "TN B")?,
                checked_product(k, n, "TN output")?,
                n,
            ),
            Op::Nt => (
                checked_product(m, n, "NT A")?,
                checked_product(k, n, "NT B")?,
                checked_product(m, k, "NT output")?,
                k,
            ),
        };
        let a = GpuBuffer::from_cpu(&runtime.stream, &seeded_values(a_len, 0xa120_0001))?;
        let b = GpuBuffer::from_cpu(&runtime.stream, &seeded_values(b_len, 0xb120_0002))?;
        let initial = seeded_values(output_len, 0xc120_0003);
        let incumbent = GuardedOutput::new(&runtime.stream, initial.clone())?;
        let candidate = GuardedOutput::new(&runtime.stream, initial)?;
        let (a_width, a_rows, a_box, b_width, b_rows, b_box) = match case.op {
            Op::Tn => (k, m, [32, 32], n, m, [32, 32]),
            Op::Nt => (
                n,
                m,
                [32, case.route.tile.0 as u32],
                n,
                k,
                [32, case.route.tile.1 as u32],
            ),
        };
        let incumbent_maps = TensorMaps {
            a: tensor_map(a.raw_ptr(&runtime.stream), a_width, a_rows, a_width, a_box)?,
            b: tensor_map(b.raw_ptr(&runtime.stream), b_width, b_rows, b_width, b_box)?,
        };
        let candidate_maps = if case.route.candidate_a_rank3 {
            if case.op != Op::Tn {
                return Err("rank-3 A map is restricted to TN experiments".into());
            }
            TensorMaps {
                a: rank3_plane_tensor_map(
                    a.raw_ptr(&runtime.stream),
                    a_width,
                    a_rows,
                    case.route.candidate_tile().0,
                )?,
                b: incumbent_maps.b,
            }
        } else if case.route.candidate_reduction_tile != 32 {
            let reduction_box = u32::try_from(case.route.candidate_reduction_tile)
                .map_err(|_| "candidate reduction tile exceeds u32")?;
            TensorMaps {
                a: tensor_map(
                    a.raw_ptr(&runtime.stream),
                    a_width,
                    a_rows,
                    a_width,
                    [32, reduction_box],
                )?,
                b: tensor_map(
                    b.raw_ptr(&runtime.stream),
                    b_width,
                    b_rows,
                    b_width,
                    [32, reduction_box],
                )?,
            }
        } else if case.op == Op::Nt && case.route.candidate_tile() != case.route.tile {
            let candidate_tile = case.route.candidate_tile();
            TensorMaps {
                a: tensor_map(
                    a.raw_ptr(&runtime.stream),
                    a_width,
                    a_rows,
                    a_width,
                    [32, candidate_tile.0 as u32],
                )?,
                b: tensor_map(
                    b.raw_ptr(&runtime.stream),
                    b_width,
                    b_rows,
                    b_width,
                    [32, candidate_tile.1 as u32],
                )?,
            }
        } else {
            incumbent_maps
        };
        let params = KernelParams {
            a_x: 0,
            a_y: 0,
            b_x: 0,
            b_y: 0,
            alpha: case.alpha,
            beta: 1.0,
            m: i32::try_from(m).map_err(|_| "M exceeds i32")?,
            k: i32::try_from(k).map_err(|_| "K exceeds i32")?,
            n: i32::try_from(n).map_err(|_| "N exceeds i32")?,
            ldc: i32::try_from(ldc).map_err(|_| "ldc exceeds i32")?,
        };
        Ok(Fixture {
            _a: a,
            _b: b,
            incumbent,
            candidate,
            maps: CandidateMaps {
                incumbent: incumbent_maps,
                candidate: candidate_maps,
            },
            params,
        })
    }

    fn load_kernel_pair(runtime: &Runtime, case: Case) -> Result<KernelPair, String> {
        let incumbent = runtime
            .module
            .load_function(case.route.incumbent)
            .map_err(|error| format!("load {}: {error:?}", case.route.incumbent))?;
        let candidate = runtime
            .module
            .load_function(case.route.candidate)
            .map_err(|error| format!("load {}: {error:?}", case.route.candidate))?;
        let incumbent_shared = case.route.incumbent_shared_bytes();
        let candidate_shared = case.route.candidate_shared_bytes();
        for (label, function, shared) in [
            (case.route.incumbent, &incumbent, incumbent_shared),
            (case.route.candidate, &candidate, candidate_shared),
        ] {
            function
                .set_attribute(
                    sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared as i32,
                )
                .map_err(|error| format!("set dynamic shared memory for {label}: {error:?}"))?;
        }
        let (m, k, n) = case.dims;
        let (rows, columns) = match case.op {
            Op::Tn => (k, n),
            Op::Nt => (m, k),
        };
        let incumbent_grid = rows.div_ceil(case.route.tile.0) * columns.div_ceil(case.route.tile.1);
        let candidate_tile = case.route.candidate_tile();
        let candidate_grid = rows.div_ceil(candidate_tile.0) * columns.div_ceil(candidate_tile.1);
        let incumbent_grid_dim = (
            u32::try_from(incumbent_grid).map_err(|_| "incumbent grid exceeds u32")?,
            1,
            1,
        );
        let candidate_grid_dim = (
            u32::try_from(candidate_grid).map_err(|_| "candidate grid exceeds u32")?,
            1,
            1,
        );
        Ok(KernelPair {
            incumbent,
            candidate,
            incumbent_config: LaunchConfig {
                grid_dim: incumbent_grid_dim,
                block_dim: (case.route.incumbent_threads, 1, 1),
                shared_mem_bytes: incumbent_shared,
            },
            candidate_config: LaunchConfig {
                grid_dim: candidate_grid_dim,
                block_dim: (case.route.candidate_threads, 1, 1),
                shared_mem_bytes: candidate_shared,
            },
        })
    }

    fn function(kernels: &KernelPair, arm: Arm) -> &CudaFunction {
        match arm {
            Arm::Incumbent => &kernels.incumbent,
            Arm::Candidate => &kernels.candidate,
        }
    }

    fn output(fixture: &Fixture, arm: Arm) -> &GuardedOutput {
        match arm {
            Arm::Incumbent => &fixture.incumbent,
            Arm::Candidate => &fixture.candidate,
        }
    }

    fn output_mut(fixture: &mut Fixture, arm: Arm) -> &mut GuardedOutput {
        match arm {
            Arm::Incumbent => &mut fixture.incumbent,
            Arm::Candidate => &mut fixture.candidate,
        }
    }

    fn maps(fixture: &Fixture, arm: Arm) -> &TensorMaps {
        match arm {
            Arm::Incumbent => &fixture.maps.incumbent,
            Arm::Candidate => &fixture.maps.candidate,
        }
    }

    fn launch(
        runtime: &Runtime,
        kernels: &KernelPair,
        fixture: &Fixture,
        arm: Arm,
    ) -> Result<(), String> {
        let output = output(fixture, arm).active_ptr(&runtime.stream);
        let bias = 0_u64;
        let maps = maps(fixture, arm);
        let mut builder = runtime.stream.launch_builder(function(kernels, arm));
        builder.arg(&output);
        builder.arg(&maps.a);
        builder.arg(&maps.b);
        builder.arg(&bias);
        builder.arg(&fixture.params);
        let config = match arm {
            Arm::Incumbent => kernels.incumbent_config,
            Arm::Candidate => kernels.candidate_config,
        };
        unsafe { builder.launch(config) }
            .map(|_| ())
            .map_err(|error| format!("launch {arm:?}: {error:?}"))
    }

    fn capture_pair(
        runtime: &Runtime,
        kernels: &KernelPair,
        fixture: &Fixture,
    ) -> Result<CapturedPair, String> {
        let incumbent = unsafe {
            capture_into_graph(&runtime.stream, || {
                launch(runtime, kernels, fixture, Arm::Incumbent)
            })
        }?;
        let candidate = unsafe {
            capture_into_graph(&runtime.stream, || {
                launch(runtime, kernels, fixture, Arm::Candidate)
            })
        }?;
        Ok(CapturedPair {
            incumbent,
            candidate,
        })
    }

    fn graph(graphs: &CapturedPair, arm: Arm) -> &CudaGraph {
        match arm {
            Arm::Incumbent => &graphs.incumbent,
            Arm::Candidate => &graphs.candidate,
        }
    }

    fn resource_gate(case: Case, kernels: &KernelPair) -> Result<(), String> {
        let mut facts = Vec::new();
        for (label, function, threads, shared) in [
            (
                case.route.incumbent,
                &kernels.incumbent,
                case.route.incumbent_threads,
                case.route.incumbent_shared_bytes() as usize,
            ),
            (
                case.route.candidate,
                &kernels.candidate,
                case.route.candidate_threads,
                case.route.candidate_shared_bytes() as usize,
            ),
        ] {
            let local = function
                .local_size_bytes()
                .map_err(|error| format!("query {label} local bytes: {error:?}"))?;
            let registers = function
                .num_regs()
                .map_err(|error| format!("query {label} registers: {error:?}"))?;
            let occupancy = function
                .occupancy_max_active_blocks_per_multiprocessor(threads, shared, None)
                .map_err(|error| format!("query {label} occupancy: {error:?}"))?;
            if local != 0 {
                return Err(format!("{label} uses {local} local bytes"));
            }
            let register_limit = if label == case.route.candidate {
                case.route.candidate_max_registers
            } else {
                128
            };
            if registers > register_limit {
                return Err(format!(
                    "{label} uses {registers} registers, above {register_limit}"
                ));
            }
            if occupancy == 0 {
                return Err(format!("{label} has zero occupancy"));
            }
            facts.push((label, registers, occupancy, threads));
        }
        if facts[1].2 < case.route.candidate_min_blocks {
            return Err(format!(
                "{} occupancy {} is below required {} blocks",
                facts[1].0, facts[1].2, case.route.candidate_min_blocks
            ));
        }
        let incumbent_warps = facts[0].2 * facts[0].3 / 32;
        let candidate_warps = facts[1].2 * facts[1].3 / 32;
        if case.route.candidate_tile() == case.route.tile && candidate_warps < incumbent_warps {
            return Err(format!(
                "{} active warps {candidate_warps} are below incumbent {incumbent_warps}",
                facts[1].0
            ));
        }
        eprintln!(
            concat!(
                "{} resources incumbent_regs={} candidate_regs={} ",
                "blocks={}/{} active_warps={}/{}"
            ),
            case.id,
            facts[0].1,
            facts[1].1,
            facts[0].2,
            facts[1].2,
            incumbent_warps,
            candidate_warps,
        );
        Ok(())
    }

    fn run_path(
        runtime: &Runtime,
        kernels: &KernelPair,
        fixture: &Fixture,
        graphs: &CapturedPair,
        arm: Arm,
        path: Path,
    ) -> Result<(), String> {
        match path {
            Path::Eager => launch(runtime, kernels, fixture, arm),
            Path::Graph => graph(graphs, arm)
                .launch()
                .map_err(|error| format!("launch {arm:?} graph: {error:?}")),
        }
    }

    fn execute_and_read(
        runtime: &Runtime,
        kernels: &KernelPair,
        fixture: &mut Fixture,
        graphs: &CapturedPair,
        arm: Arm,
        path: Path,
    ) -> Result<Vec<u32>, String> {
        output_mut(fixture, arm).reset(&runtime.stream)?;
        run_path(runtime, kernels, fixture, graphs, arm, path)?;
        runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize {arm:?} {path:?}: {error:?}"))?;
        output(fixture, arm).bits(&runtime.stream)
    }

    fn correctness_gate(
        runtime: &Runtime,
        kernels: &KernelPair,
        fixture: &mut Fixture,
        graphs: &CapturedPair,
        case: Case,
    ) -> Result<(), String> {
        let mut reference = None;
        for repeat in 0..CORRECTNESS_REPEATS {
            let incumbent_eager = execute_and_read(
                runtime,
                kernels,
                fixture,
                graphs,
                Arm::Incumbent,
                Path::Eager,
            )?;
            let candidate_eager = execute_and_read(
                runtime,
                kernels,
                fixture,
                graphs,
                Arm::Candidate,
                Path::Eager,
            )?;
            if candidate_eager != incumbent_eager {
                return Err(format!(
                    "{} candidate eager bits differ from incumbent at repeat {repeat}",
                    case.id
                ));
            }
            let incumbent_graph = execute_and_read(
                runtime,
                kernels,
                fixture,
                graphs,
                Arm::Incumbent,
                Path::Graph,
            )?;
            let candidate_graph = execute_and_read(
                runtime,
                kernels,
                fixture,
                graphs,
                Arm::Candidate,
                Path::Graph,
            )?;
            if candidate_graph != incumbent_graph {
                return Err(format!(
                    "{} candidate graph bits differ from incumbent at repeat {repeat}",
                    case.id
                ));
            }
            if incumbent_eager != incumbent_graph {
                return Err(format!(
                    "{} incumbent eager and graph bits differ at repeat {repeat}",
                    case.id
                ));
            }
            if reference
                .as_ref()
                .is_some_and(|expected| expected != &incumbent_eager)
            {
                return Err(format!("{} repeated output bits changed", case.id));
            }
            reference.get_or_insert(incumbent_eager);
        }
        Ok(())
    }

    fn cuda_ok(result: sys::CUresult, operation: &str) -> Result<(), String> {
        if result == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{operation}: {result:?}"))
        }
    }

    fn graph_symbol(graph: &CudaGraph) -> Result<String, String> {
        let mut count = 0_usize;
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), std::ptr::null_mut(), &mut count) },
            "query graph node count",
        )?;
        if count != 1 {
            return Err(format!("experiment graph has {count} nodes, expected one"));
        }
        let mut node = std::ptr::null_mut();
        cuda_ok(
            unsafe { sys::cuGraphGetNodes(graph.cu_graph(), &mut node, &mut count) },
            "query graph node",
        )?;
        let mut params = unsafe { std::mem::zeroed() };
        cuda_ok(
            unsafe { sys::cuGraphKernelNodeGetParams_v2(node, &mut params) },
            "query graph kernel parameters",
        )?;
        let mut name = std::ptr::null();
        cuda_ok(
            unsafe { sys::cuFuncGetName(&mut name, params.func) },
            "query graph function name",
        )?;
        if name.is_null() {
            return Err("graph function name is null".into());
        }
        unsafe { std::ffi::CStr::from_ptr(name) }
            .to_str()
            .map(str::to_owned)
            .map_err(|error| format!("graph function name is not UTF-8: {error}"))
    }

    fn topology_gate(case: Case, graphs: &CapturedPair) -> Result<(), String> {
        let incumbent = graph_symbol(&graphs.incumbent)?;
        let candidate = graph_symbol(&graphs.candidate)?;
        if incumbent != case.route.incumbent || candidate != case.route.candidate {
            return Err(format!(
                "{} captured unexpected symbols incumbent={incumbent} candidate={candidate}",
                case.id
            ));
        }
        Ok(())
    }

    fn launch_measurement(measurement: &Measurement<'_>, arm: Arm) -> Result<(), String> {
        run_path(
            measurement.runtime,
            measurement.kernels,
            measurement.fixture,
            measurement.graphs,
            arm,
            measurement.path,
        )
    }

    fn measure_window(
        measurement: &Measurement<'_>,
        arm: Arm,
        iterations: usize,
    ) -> Result<f64, String> {
        if iterations == 0 {
            return Err("pair-store timing iterations must be positive".into());
        }
        let start = measurement
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {arm:?} start event: {error:?}"))?;
        for _ in 0..iterations {
            launch_measurement(measurement, arm)?;
        }
        let end = measurement
            .runtime
            .stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| format!("record {arm:?} end event: {error:?}"))?;
        let elapsed_us = start
            .elapsed_ms(&end)
            .map(f64::from)
            .map_err(|error| format!("measure {arm:?} events: {error:?}"))?
            * 1000.0
            / iterations as f64;
        if elapsed_us.is_finite() && elapsed_us > 0.0 {
            Ok(elapsed_us)
        } else {
            Err(format!("{arm:?} timing was not positive and finite"))
        }
    }

    fn calibrate_iterations(measurement: &Measurement<'_>, arm: Arm) -> Result<usize, String> {
        let per_launch_us = measure_window(measurement, arm, CALIBRATION_PROBE_LAUNCHES)?;
        Ok(
            ((TARGET_WINDOW_MS * 1000.0 / per_launch_us).ceil() as usize)
                .clamp(1, MAX_WINDOW_LAUNCHES),
        )
    }

    fn measure_pair(
        measurement: &Measurement<'_>,
        first: Arm,
        iterations: usize,
    ) -> Result<PairedTiming, String> {
        let second = match first {
            Arm::Incumbent => Arm::Candidate,
            Arm::Candidate => Arm::Incumbent,
        };
        let first_us = measure_window(measurement, first, iterations)?;
        let second_us = measure_window(measurement, second, iterations)?;
        Ok(match first {
            Arm::Incumbent => PairedTiming {
                incumbent_us: first_us,
                candidate_us: second_us,
            },
            Arm::Candidate => PairedTiming {
                incumbent_us: second_us,
                candidate_us: first_us,
            },
        })
    }

    fn summarize_order(samples: &[PairedTiming]) -> Result<(f64, f64, f64, f64), String> {
        let incumbent = samples
            .iter()
            .map(|sample| sample.incumbent_us)
            .collect::<Vec<_>>();
        let candidate = samples
            .iter()
            .map(|sample| sample.candidate_us)
            .collect::<Vec<_>>();
        let speedups = samples
            .iter()
            .copied()
            .map(PairedTiming::speedup)
            .collect::<Vec<_>>();
        Ok((
            percentile(&incumbent, 0.50)?,
            percentile(&candidate, 0.50)?,
            percentile(&speedups, 0.50)?,
            percentile(&speedups, 0.05)?,
        ))
    }

    fn paired_perf_screen(
        measurement: &Measurement<'_>,
        config: PerfConfig,
        case: Case,
        quiet: &QuietGpu,
    ) -> Result<(), String> {
        let label = format!("pair-store/{}/{:?}", case.id, measurement.path);
        quiet.require_cohort(&format!("{label}/pre-warmup"))?;
        for _ in 0..config.warmups {
            launch_measurement(measurement, Arm::Incumbent)?;
            launch_measurement(measurement, Arm::Candidate)?;
        }
        measurement
            .runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize performance warmups: {error:?}"))?;
        quiet.require_cohort(&format!("{label}/pre-calibration"))?;
        let iterations = match config.iterations {
            Some(iterations) => iterations,
            None => calibrate_iterations(measurement, Arm::Incumbent)?
                .max(calibrate_iterations(measurement, Arm::Candidate)?),
        };
        quiet.require_cohort(&format!("{label}/pre-timed"))?;
        let mut ab_samples = Vec::with_capacity(config.windows_per_order);
        let mut ba_samples = Vec::with_capacity(config.windows_per_order);
        for window in 0..config.windows_per_order {
            ab_samples.push(measure_pair(measurement, Arm::Incumbent, iterations)?);
            ba_samples.push(measure_pair(measurement, Arm::Candidate, iterations)?);
            if window + 1 == config.windows_per_order.div_ceil(2) {
                measurement
                    .runtime
                    .stream
                    .synchronize()
                    .map_err(|error| format!("synchronize performance midpoint: {error:?}"))?;
                quiet.require_cohort(&format!("{label}/mid-timed"))?;
            }
        }
        measurement
            .runtime
            .stream
            .synchronize()
            .map_err(|error| format!("synchronize performance postflight: {error:?}"))?;
        quiet.verify_post_cohort(&format!("{label}/post-timed"))?;
        let incumbent_bits =
            output(measurement.fixture, Arm::Incumbent).bits(&measurement.runtime.stream)?;
        let candidate_bits =
            output(measurement.fixture, Arm::Candidate).bits(&measurement.runtime.stream)?;
        if candidate_bits != incumbent_bits {
            return Err(format!(
                "{} {:?} post-timing candidate bits differ from incumbent",
                case.id, measurement.path
            ));
        }

        let ab_speedups = ab_samples
            .iter()
            .copied()
            .map(PairedTiming::speedup)
            .collect::<Vec<_>>();
        let ba_speedups = ba_samples
            .iter()
            .copied()
            .map(PairedTiming::speedup)
            .collect::<Vec<_>>();
        let (ab_incumbent, ab_candidate, ab_median, ab_p05) = summarize_order(&ab_samples)?;
        let (ba_incumbent, ba_candidate, ba_median, ba_p05) = summarize_order(&ba_samples)?;
        eprintln!(
            concat!(
                "{} {:?} samples_per_order={} iterations={} ",
                "ab_incumbent_us={ab_incumbent:.6} ab_candidate_us={ab_candidate:.6} ",
                "ab_median_speedup={ab_median:.6} ab_p05_speedup={ab_p05:.6} ",
                "ba_incumbent_us={ba_incumbent:.6} ba_candidate_us={ba_candidate:.6} ",
                "ba_median_speedup={ba_median:.6} ba_p05_speedup={ba_p05:.6}"
            ),
            case.id,
            measurement.path,
            config.windows_per_order,
            iterations,
            ab_incumbent = ab_incumbent,
            ab_candidate = ab_candidate,
            ab_median = ab_median,
            ab_p05 = ab_p05,
            ba_incumbent = ba_incumbent,
            ba_candidate = ba_candidate,
            ba_median = ba_median,
            ba_p05 = ba_p05,
        );
        if !speedup_order_passes(
            &ab_speedups,
            config.min_median_speedup,
            config.min_p05_speedup,
        ) || !speedup_order_passes(
            &ba_speedups,
            config.min_median_speedup,
            config.min_p05_speedup,
        ) {
            return Err(format!(
                concat!(
                    "{} {:?} pair-store screen failed: ",
                    "AB median={ab_median:.6} p05={ab_p05:.6}; ",
                    "BA median={ba_median:.6} p05={ba_p05:.6}"
                ),
                case.id,
                measurement.path,
                ab_median = ab_median,
                ab_p05 = ab_p05,
                ba_median = ba_median,
                ba_p05 = ba_p05,
            ));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive SM120 GPU"]
    fn sm120_tf32_pair_store_experiment() -> Result<(), String> {
        let config = PerfConfig::from_env()?;
        let case_filter = std::env::var("MAMBA_RS_SM120_TF32_PAIR_CASE_FILTER")
            .ok()
            .filter(|filter| !filter.is_empty());
        let selected_cases = CASES
            .into_iter()
            .filter(|case| {
                case_filter
                    .as_ref()
                    .is_none_or(|filter| case.id.contains(filter))
            })
            .collect::<Vec<_>>();
        if selected_cases.is_empty() {
            return Err("SM120 TF32 pair-store case filter matched no cases".into());
        }
        let quiet = QuietGpu::for_cuda_ordinal(0)?;
        quiet.require_pre_context("pair-store/pre-context")?;
        let runtime = new_runtime()?;
        for case in selected_cases {
            let kernels = load_kernel_pair(&runtime, case)?;
            resource_gate(case, &kernels)?;
            let mut fixture = new_fixture(&runtime, case)?;
            let graphs = capture_pair(&runtime, &kernels, &fixture)?;
            topology_gate(case, &graphs)?;
            correctness_gate(&runtime, &kernels, &mut fixture, &graphs, case)?;
            if case.performance && config.performance {
                for path in [Path::Eager, Path::Graph] {
                    paired_perf_screen(
                        &Measurement {
                            runtime: &runtime,
                            kernels: &kernels,
                            fixture: &fixture,
                            graphs: &graphs,
                            path,
                        },
                        config,
                        case,
                        &quiet,
                    )?;
                }
            }
            runtime
                .stream
                .synchronize()
                .map_err(|error| format!("synchronize before graph drop: {error:?}"))?;
            drop(graphs);
        }
        Ok(())
    }
}
